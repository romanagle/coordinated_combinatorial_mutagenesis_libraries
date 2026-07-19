"""matched_2x5_panel.py

2 × 5 figure:
  Row 1 (scatter): ŷ vs y for each matched (surrogate, GT) pair — main sequence only.
  Row 2 (violin):  Spearman ρ across 10 sequences for the same matched pair only.

Matched pairs (columns 1–5):
  1. Additive surrogate      × Additive GT
  2. Pairwise surrogate      × Additive+Pairwise GT
  3. Additive GE surrogate   × Nonlinear Additive GT
  4. Pairwise GE surrogate   × Nonlinear Additive+Pairwise GT
  5. Pairwise GE surrogate   × RB Surrogate GT   (re-run + saved)

Saved outputs
  <out_dir>/matched_2x5_panel.png
  <violin_data_dir>/rb_rho_results.json   (RB Surrogate GT ρ for 10 sequences)

Usage:
    cd /home/nagle/final_version
    python scripts/matched_2x5_panel.py \\
        --experiment  RNCMPT00111 \\
        --seq         AAAAAAACCCCCAAAAAAUCGGCUGGACCGGGAAAAAAAAA \\
        --violin_data outputs/postabrcms/vts1high/spearman_violin_data \\
        --out_dir     outputs/postabrcms/vts1high \\
        --seed        42
"""

import argparse
import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from scipy.stats import spearmanr, pearsonr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '/home/nagle/final_version/squid-nn')
sys.path.append('/home/nagle/final_version/squid-manuscript/squid')
sys.path.append('/home/nagle/final_version/squid-nn/squid')
sys.path.append('/home/nagle/final_version/residualbind')

import squid.surrogate_zoo
from seq_utils import rna_to_one_hot, remove_padding
from ground_truth import (
    init_additive_noWT, init_pairwise_potts_optionA, init_sigmoid_nonlin,
    compute_gt_scores_for_library_potts, additive_affinity_noWT,
    uniformize_by_histogram,
)
import helper
from residualbind import ResidualBind

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

NUCS   = ['A', 'C', 'G', 'U']
_NUCS  = np.array(list("ACGU"))

SURROGATE_CONFIGS = {
    "additive": {
        "gpmap": "additive", "linearity": "linear",
        "regression_type": "GE", "noise": "Gaussian",
        "noise_order": 0, "reg_strength": 12,
    },
    "pairwise": {
        "gpmap": "pairwise", "linearity": "linear",
        "regression_type": "GE", "noise": "Gaussian",
        "noise_order": 0, "reg_strength": 0.1,
    },
    "additive_GE": {
        "gpmap": "additive", "linearity": "nonlinear",
        "regression_type": "GE", "noise": "SkewedT",
        "noise_order": 2, "reg_strength": 12, "hidden_nodes": 50,
    },
    "pairwise_GE": {
        "gpmap": "pairwise", "linearity": "nonlinear",
        "regression_type": "GE", "noise": "SkewedT",
        "noise_order": 2, "reg_strength": 0.1, "hidden_nodes": 50,
    },
}

RB_SGT_CFG = {
    "gpmap": "pairwise", "linearity": "nonlinear",
    "regression_type": "GE", "noise": "Gaussian",
    "noise_order": 2, "reg_strength": 0.005, "hidden_nodes": 10,
}

# (surrogate_cfg, analytic_gt_key_or_"rb", col title, GT label for violin y-axis)
MATCHED_PAIRS = [
    ("additive",    "additive",                 "Additive\nSurrogate × GT",              "Additive GT"),
    ("pairwise",    "additive_pairwise",        "Pairwise\nSurrogate × GT",              "Additive+Pair GT"),
    ("additive_GE", "nonlin_additive",          "Additive GE\nSurrogate × GT",           "Nonlin Additive GT"),
    ("pairwise_GE", "nonlin_additive_pairwise", "Pairwise GE\nSurrogate × GT",           "Nonlin Add+Pair GT"),
    ("pairwise_GE", "rb",                       "Pairwise GE\nSurrogate × RB Surr. GT",  "RB Surrogate GT"),
]

BLUE   = "#4C72B0"
ORANGE = "#DD8452"
VIOLIN_W = 0.42

# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _to_str(X):
    return np.array(["".join(_NUCS[np.argmax(x, axis=1)]) for x in X], dtype=object)


def _predict_chunked(model, x_str, chunk=1_000):
    parts = []
    for s in range(0, len(x_str), chunk):
        parts.append(np.asarray(model.x_to_yhat(x_str[s:s + chunk]), dtype=float).ravel())
    return np.concatenate(parts)


def generate_library(wt_onehot, n_seqs, exact_mut_count, rng):
    L, wt_idx = wt_onehot.shape[0], np.argmax(wt_onehot, axis=1)
    parts, CHUNK = [], 50_000
    for start in range(0, n_seqs, CHUNK):
        nc       = min(CHUNK, n_seqs - start)
        pos      = np.argpartition(rng.random((nc, L)), exact_mut_count, axis=1)[:, :exact_mut_count]
        X        = np.tile(wt_onehot[None], (nc, 1, 1)).astype(np.float32)
        n_idx    = np.repeat(np.arange(nc), exact_mut_count)
        p_idx    = pos.ravel()
        rand_nuc = rng.integers(0, 3, size=nc * exact_mut_count)
        new_nucs = np.where(rand_nuc >= wt_idx[p_idx], rand_nuc + 1, rand_nuc)
        X[n_idx, p_idx, :]        = 0
        X[n_idx, p_idx, new_nucs] = 1
        parts.append(X)
    return np.concatenate(parts, axis=0)


def train_mavenn(X, y, cfg, label="", verbose=0):
    N        = X.shape[0]
    bsz      = max(32, min(N // 150, 2048))
    lr       = 5e-4 * min(1.0, (20_000 / N) ** 0.5)
    is_nlp   = cfg["gpmap"] == "pairwise" and cfg["linearity"] == "nonlinear"
    epochs   = 1000 if is_nlp else 500
    patience = 50   if is_nlp else 25
    wrapper  = squid.surrogate_zoo.SurrogateMAVENN(
        X.shape, num_tasks=1,
        gpmap=cfg["gpmap"], regression_type=cfg["regression_type"],
        linearity=cfg["linearity"], noise=cfg["noise"],
        noise_order=cfg["noise_order"], reg_strength=cfg["reg_strength"],
        hidden_nodes=cfg.get("hidden_nodes", 50),
        alphabet=NUCS, deduplicate=True, gpu=True,
    )
    model, _, test_df = wrapper.train(
        X, y, learning_rate=lr, epochs=epochs, batch_size=bsz,
        early_stopping=True, patience=patience, restore_best_weights=True,
        save_dir=None, verbose=verbose,
    )
    print(f"    [{label}]  lr={lr:.2e}  bsz={bsz}  ep={epochs}", flush=True)
    return model, test_df


def _rho_r2(yhat, ytrue):
    m = np.isfinite(yhat) & np.isfinite(ytrue)
    if m.sum() < 3:
        return np.nan, np.nan
    rho = float(spearmanr(yhat[m], ytrue[m])[0])
    r   = float(pearsonr( yhat[m], ytrue[m])[0])
    return rho, r**2


def _scatter_data(model, test_df, X_eval, y_eval):
    """Return (yhat_rand, y_rand, yhat_eval, y_eval_arr)."""
    cols  = list(test_df.columns)
    x_col = "x" if "x" in cols else "X"
    y_col = "y" if "y" in cols else next(c for c in cols if c.startswith("y"))
    yhat_rand = _predict_chunked(model, np.asarray(test_df[x_col]))
    y_rand    = np.asarray(test_df[y_col], dtype=float).ravel()
    yhat_eval = _predict_chunked(model, _to_str(X_eval))
    return yhat_rand, y_rand, yhat_eval, np.asarray(y_eval, dtype=float).ravel()


def _test_df_cols(test_df):
    cols  = list(test_df.columns)
    x_col = "x" if "x" in cols else "X"
    y_col = "y" if "y" in cols else next(c for c in cols if c.startswith("y"))
    return x_col, y_col


def pad_to_41(X):
    N, L, A = X.shape
    if L == 41:
        return X
    pad = 41 - L
    return np.concatenate([
        np.zeros((N, pad // 2,       A), dtype=np.float32),
        X,
        np.zeros((N, pad - pad // 2, A), dtype=np.float32),
    ], axis=1)


def rb_score(X, residbind, batch=512):
    X_pad = pad_to_41(X)
    parts = []
    for i in range(0, len(X_pad), batch):
        parts.append(np.asarray(residbind.predict(X_pad[i:i + batch]), dtype=float).ravel())
    return np.concatenate(parts)


# ---------------------------------------------------------------------------
# Scatter data — main sequence (top row)
# ---------------------------------------------------------------------------

def scatter_analytic(wt_oh, exact_mut, seed, lib_size, n_pool, target_n):
    """Score one library with all 4 analytic GTs; build eval per GT.
    Returns { gt_key: (X_lib, y_lib, X_eval, y_eval) }
    """
    rng = np.random.default_rng(seed)
    W_mut, mut_map, b0 = init_additive_noWT(rng, wt_oh, sigma=0.5, l1_w=0.1, bias=0.0)
    edges, J = init_pairwise_potts_optionA(
        rng, wt_oh, p_edge=0.70, df=5.0, lambda_J=2.0, p_rescue=0.10, wt_rowcol_zero=True,
    )
    rng_ref = np.random.default_rng(seed + 1)
    X_ref   = generate_library(wt_oh, 200_000, exact_mut, rng_ref)
    s_add   = additive_affinity_noWT(X_ref, W_mut, mut_map, b=b0).reshape(-1)
    nk      = init_sigmoid_nonlin(s_add)
    del X_ref, s_add

    rng_lib = np.random.default_rng(seed + 200)
    X_lib   = generate_library(wt_oh, lib_size, exact_mut, rng_lib)
    scores  = compute_gt_scores_for_library_potts(
        X_lib, W_mut=W_mut, mut_map=mut_map, b0=b0,
        nonlin_name="sigmoid", nonlin_kwargs=nk, edges=edges, J=J,
    )

    rng_pool = np.random.default_rng(seed + 99_999)
    X_pool   = generate_library(wt_oh, n_pool, exact_mut, rng_pool)
    pool_sc  = compute_gt_scores_for_library_potts(
        X_pool, W_mut=W_mut, mut_map=mut_map, b0=b0,
        nonlin_name="sigmoid", nonlin_kwargs=nk, edges=edges, J=J,
    )

    result = {}
    for gt_key in scores:
        y_pool_k = pool_sc[gt_key].astype(float)
        y_uni, keep = uniformize_by_histogram(
            y_pool_k, X=None, n_bins=200, clip_hi=98, target_n=target_n, seed=seed,
        )
        result[gt_key] = (X_lib, scores[gt_key].astype(float), X_pool[keep], y_uni.astype(float))
    return result


def scatter_rb(wt_oh, exact_mut, seed, residbind, lib_size, n_pool, target_n):
    """Train RB Surrogate GT; score library with it; build eval.
    Returns (rb_sgt_model, X_lib2, y_lib2, X_eval, y_eval).
    """
    # Stage 1: train RB Surrogate GT
    rng1  = np.random.default_rng(seed)
    X_trn = generate_library(wt_oh, lib_size, exact_mut, rng1)
    y_trn = rb_score(X_trn, residbind).reshape(-1, 1)
    print(f"  [RB-scatter] RB scores: [{y_trn.min():.4f}, {y_trn.max():.4f}]", flush=True)
    rb_sgt, _ = train_mavenn(X_trn, y_trn, RB_SGT_CFG, label="RB_SGT", verbose=1)

    # Stage 2: score second library with RB Surrogate GT
    rng2  = np.random.default_rng(seed + 500)
    X_lib = generate_library(wt_oh, lib_size, exact_mut, rng2)
    y_lib = _predict_chunked(rb_sgt, _to_str(X_lib))

    rng3   = np.random.default_rng(seed + 500 + 99_999)
    X_pool = generate_library(wt_oh, n_pool, exact_mut, rng3)
    y_pool = _predict_chunked(rb_sgt, _to_str(X_pool))
    y_uni, keep = uniformize_by_histogram(
        y_pool, X=None, n_bins=200, clip_hi=98, target_n=target_n, seed=seed,
    )
    return rb_sgt, X_lib, y_lib, X_pool[keep], y_uni.astype(float)


# ---------------------------------------------------------------------------
# RB Surrogate GT violin — all 10 saved sequences (bottom row, col 5)
# ---------------------------------------------------------------------------

def run_rb_violin(save_dir, residbind, seed, out_json):
    """For each seq_NNN/ sub-folder, train RB Surrogate GT + 4 surrogates.
    Returns accum_rb = {cfg_name: {"rand": [...], "eval": [...]}}
    Saves to out_json (checkpoints after each sequence).
    """
    # Resume from checkpoint if it exists
    if os.path.exists(out_json):
        with open(out_json) as f:
            accum_rb = json.load(f)
        # convert any lists to plain lists (JSON round-trip safe)
        print(f"  [RB-violin] Resuming from {out_json}", flush=True)
    else:
        accum_rb = {cfg: {"rand": [], "eval": []} for cfg in SURROGATE_CONFIGS}

    n_done = len(accum_rb[list(SURROGATE_CONFIGS.keys())[0]]["rand"])

    seq_dirs = sorted(
        d for d in [os.path.join(save_dir, f"seq_{i:03d}") for i in range(100)]
        if os.path.isdir(d)
    )
    print(f"  [RB-violin] {len(seq_dirs)} sequences found, {n_done} already done", flush=True)

    for s_idx, seq_dir in enumerate(seq_dirs):
        if s_idx < n_done:
            continue
        wt_oh   = np.load(os.path.join(seq_dir, "sequence_onehot.npy")).astype(np.float32)
        seq_str = "".join(_NUCS[np.argmax(wt_oh, axis=1)])
        L       = wt_oh.shape[0]
        em      = min(4, max(1, L // 2))
        seed_s  = seed + s_idx * 100_000

        print(f"\n  [{s_idx+1}/{len(seq_dirs)}] {seq_str}  L={L}  em={em}", flush=True)

        # Stage 1: RB Surrogate GT for this sequence
        rng1  = np.random.default_rng(seed_s)
        X_trn = generate_library(wt_oh, 20_000, em, rng1)
        y_trn = rb_score(X_trn, residbind).reshape(-1, 1)
        rb_sgt, _ = train_mavenn(X_trn, y_trn, RB_SGT_CFG, label="RB_SGT", verbose=0)

        # Stage 2: score 20k library + eval pool with RB Surrogate GT
        rng2  = np.random.default_rng(seed_s + 500)
        X_lib = generate_library(wt_oh, 20_000, em, rng2)
        y_lib = _predict_chunked(rb_sgt, _to_str(X_lib)).reshape(-1, 1)

        rng3   = np.random.default_rng(seed_s + 500 + 99_999)
        X_pool = generate_library(wt_oh, 50_000, em, rng3)
        y_pool = _predict_chunked(rb_sgt, _to_str(X_pool))
        y_uni, keep = uniformize_by_histogram(
            y_pool, X=None, n_bins=200, clip_hi=98, target_n=5_000, seed=seed_s,
        )
        X_eval = X_pool[keep]
        y_eval = y_uni.astype(float)

        # Train all 4 surrogates
        for cfg_name, cfg in SURROGATE_CONFIGS.items():
            model, test_df = train_mavenn(X_lib, y_lib, cfg, label=cfg_name, verbose=0)
            x_col, y_col = _test_df_cols(test_df)
            yhat_r = _predict_chunked(model, np.asarray(test_df[x_col]))
            y_r    = np.asarray(test_df[y_col], dtype=float).ravel()
            yhat_e = _predict_chunked(model, _to_str(X_eval))
            m_r    = np.isfinite(yhat_r) & np.isfinite(y_r)
            m_e    = np.isfinite(yhat_e) & np.isfinite(y_eval)
            rho_r  = float(spearmanr(yhat_r[m_r], y_r[m_r])[0])   if m_r.sum() >= 3 else float("nan")
            rho_e  = float(spearmanr(yhat_e[m_e], y_eval[m_e])[0]) if m_e.sum() >= 3 else float("nan")
            accum_rb[cfg_name]["rand"].append(rho_r)
            accum_rb[cfg_name]["eval"].append(rho_e)
            print(f"    {cfg_name:<14}  ρ_rand={rho_r:+.4f}  ρ_eval={rho_e:+.4f}", flush=True)

        # Checkpoint after each sequence
        with open(out_json, "w") as f:
            json.dump(accum_rb, f, indent=2)
        print(f"  [checkpoint] saved → {out_json}", flush=True)

    return accum_rb


# ---------------------------------------------------------------------------
# Draw one violin cell (single matched config)
# ---------------------------------------------------------------------------

def _draw_violin_cell(ax, rho_rand, rho_eval, title):
    """Two violins (blue=rand, orange=eval) for a single surrogate config."""
    for pos, data, color in [(0.0, rho_rand, BLUE), (0.55, rho_eval, ORANGE)]:
        data = np.asarray(data, dtype=float)
        data = data[np.isfinite(data)]
        if data.size == 0:
            continue
        if data.size >= 3:
            parts = ax.violinplot(data, positions=[pos], widths=VIOLIN_W,
                                  showmedians=False, showextrema=False)
            for pc in parts["bodies"]:
                pc.set_facecolor(color); pc.set_alpha(0.65)
        rng_j  = np.random.default_rng(42)
        jitter = rng_j.uniform(-0.08, 0.08, size=data.size)
        ax.scatter(pos + jitter, data, color=color, s=20, zorder=7, alpha=0.8, linewidths=0)
        if data.size >= 2:
            mean = np.nanmean(data)
            sem  = np.nanstd(data, ddof=1) / np.sqrt(data.size)
            ax.errorbar([pos], [mean], yerr=[[sem], [sem]], fmt="o", color="black",
                        markersize=5, linewidth=2.0, capsize=4, capthick=2.0, zorder=10)

    ax.set_xticks([0.275])
    ax.set_xticklabels([""], fontsize=8)
    ax.set_ylabel("Spearman ρ", fontsize=9)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(-0.35, 0.9)
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.axhline(0, linewidth=0.7, color="gray", linestyle="--", alpha=0.4)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq",         default="AAAAAAACCCCCAAAAAAUCGGCUGGACCGGGAAAAAAAAA")
    parser.add_argument("--experiment",  default="RNCMPT00111")
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--lib_size",    type=int, default=20_000)
    parser.add_argument("--n_pool",      type=int, default=50_000)
    parser.add_argument("--target_n",    type=int, default=5_000)
    parser.add_argument("--violin_data",
                        default="outputs/postabrcms/vts1high/spearman_violin_data")
    parser.add_argument("--rho_json",
                        default="outputs/postabrcms/vts1high/spearman_violin_data/rho_results.json")
    parser.add_argument("--out_dir",     default="outputs/postabrcms/vts1high")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    seq      = args.seq.upper().replace("T", "U")
    oh       = rna_to_one_hot(seq)
    wt_oh, _ = remove_padding(oh)
    L        = wt_oh.shape[0]
    exact_mut = min(4, max(1, L // 2))
    print(f"[init] L={L}  exact_mut={exact_mut}  seed={args.seed}", flush=True)

    rb_rho_json = os.path.join(args.violin_data, "rb_rho_results.json")

    # ── Load ResidualBind ─────────────────────────────────────────────────
    print("\n[0] Loading ResidualBind…", flush=True)
    data_path   = os.path.expanduser("~/residualbind/data/RNAcompete_2013/rnacompete2013.h5")
    save_path   = "/home/nagle/residualbind/weights/log_norm_seq"
    rbp_index   = helper.find_experiment_index(data_path, args.experiment)
    train, _, _ = helper.load_rnacompete_data(
        data_path, ss_type="seq", normalization="log_norm", rbp_index=rbp_index,
    )
    input_shape  = list(train["inputs"].shape)[1:]
    weights_path = os.path.join(save_path, args.experiment + "_weights.hdf5")
    residbind    = ResidualBind(input_shape, 1, weights_path)
    residbind.load_weights()
    print(f"  ResidualBind loaded ({args.experiment})", flush=True)

    # ── Scatter data (top row) — main sequence ───────────────────────────
    print("\n[1] Analytic GT scatter data (main sequence)…", flush=True)
    analytic_data = scatter_analytic(wt_oh, exact_mut, args.seed,
                                     args.lib_size, args.n_pool, args.target_n)

    print("\n[2] RB Surrogate GT scatter data (main sequence)…", flush=True)
    rb_sgt_main, X_lib_rb, y_lib_rb, X_eval_rb, y_eval_rb = scatter_rb(
        wt_oh, exact_mut, args.seed, residbind, args.lib_size, args.n_pool, args.target_n,
    )
    rb_scores_path = os.path.join(args.violin_data, "rb_y_lib.npy")
    np.save(rb_scores_path, y_lib_rb.astype(np.float32))
    print(f"  [save] RB Surrogate GT training scores → {rb_scores_path}", flush=True)

    # ── Violin data (bottom row) ─────────────────────────────────────────
    print("\n[3] Loading analytic violin ρ values…", flush=True)
    with open(args.rho_json) as f:
        rho_analytic = json.load(f)

    print("\n[4] Running RB Surrogate GT violin (10 sequences)…", flush=True)
    accum_rb = run_rb_violin(args.violin_data, residbind, args.seed, rb_rho_json)

    # ── Train matched surrogates for scatter (top row) ───────────────────
    print("\n[5] Training matched surrogates for scatter plots…", flush=True)
    scatter_results = []   # list of (yhat_rand, y_rand, yhat_eval, y_eval)

    for cfg_name, gt_key, col_title, _ in MATCHED_PAIRS:
        print(f"  {cfg_name} × {gt_key}", flush=True)
        if gt_key == "rb":
            X_lib_use = X_lib_rb
            y_use     = y_lib_rb.reshape(-1, 1)
            X_eval_use = X_eval_rb
            y_eval_use = y_eval_rb
        else:
            X_lib_use, y_use_flat, X_eval_use, y_eval_use = analytic_data[gt_key]
            y_use = y_use_flat.reshape(-1, 1)

        model, test_df = train_mavenn(
            X_lib_use, y_use, SURROGATE_CONFIGS[cfg_name],
            label=f"{cfg_name}×{gt_key}", verbose=1,
        )
        yhat_rand, y_rand, yhat_eval, y_eval_arr = _scatter_data(
            model, test_df, X_eval_use, y_eval_use,
        )
        scatter_results.append((yhat_rand, y_rand, yhat_eval, y_eval_arr))

    # ── Compose 2×5 figure ───────────────────────────────────────────────
    print("\n[6] Drawing 2×5 panel…", flush=True)

    fig = plt.figure(figsize=(30, 11))
    gs  = GridSpec(2, 5, figure=fig,
                   height_ratios=[3, 2], hspace=0.55, wspace=0.38,
                   left=0.05, right=0.97, top=0.91, bottom=0.06)

    for col, (cfg_name, gt_key, col_title, violin_label) in enumerate(MATCHED_PAIRS):

        # ── Top: scatter ─────────────────────────────────────────────────
        ax_sc = fig.add_subplot(gs[0, col])
        yhat_rand, y_rand, yhat_eval, y_eval_arr = scatter_results[col]

        all_v = np.concatenate([yhat_rand, y_rand, yhat_eval, y_eval_arr])
        all_v = all_v[np.isfinite(all_v)]
        lo, hi = np.percentile(all_v, [0.5, 99.5])
        pad    = (hi - lo) * 0.05
        lim    = [lo - pad, hi + pad]

        ax_sc.scatter(y_rand,     yhat_rand, s=1, alpha=0.07, color=BLUE,
                      rasterized=True)
        ax_sc.scatter(y_eval_arr, yhat_eval, s=1, alpha=0.07, color=ORANGE,
                      rasterized=True)
        ax_sc.plot(lim, lim, "k--", linewidth=0.8, zorder=10)
        ax_sc.set_xlim(lim); ax_sc.set_ylim(lim)
        ax_sc.set_xlabel("y", fontsize=8)
        ax_sc.set_ylabel("ŷ", fontsize=8)
        ax_sc.tick_params(labelsize=6, pad=2)
        ax_sc.set_title(col_title, fontsize=9, fontweight="bold", pad=5)

        rho_r, r2_r = _rho_r2(yhat_rand, y_rand)   # order unchanged — rho is symmetric
        rho_e, r2_e = _rho_r2(yhat_eval, y_eval_arr)
        ax_sc.text(0.04, 0.30,
                   f"rand  ρ={rho_r:.3f}  R²={r2_r:.3f}\n"
                   f"eval   ρ={rho_e:.3f}  R²={r2_e:.3f}",
                   transform=ax_sc.transAxes, fontsize=6, va="top",
                   bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85,
                             ec="#cccccc", linewidth=0.6))

        # ── Bottom: violin ───────────────────────────────────────────────
        ax_vl = fig.add_subplot(gs[1, col])

        if gt_key == "rb":
            rho_rand = accum_rb[cfg_name]["rand"]
            rho_eval = accum_rb[cfg_name]["eval"]
        else:
            rho_rand = rho_analytic[gt_key][cfg_name]["rand"]
            rho_eval = rho_analytic[gt_key][cfg_name]["eval"]

        _draw_violin_cell(ax_vl, rho_rand, rho_eval, violin_label)

    # Shared legend — placed below the suptitle, top-right
    fig.legend(
        handles=[
            Patch(facecolor=BLUE,   alpha=0.65, label="Random (MAVE test split)"),
            Patch(facecolor=ORANGE, alpha=0.65, label="Eval (curated library)"),
            plt.Line2D([0], [0], color="black", marker="o", linewidth=2,
                       markersize=5, label="Mean ± SEM"),
        ],
        loc="upper right", fontsize=8, bbox_to_anchor=(0.99, 0.98),
        framealpha=0.85, edgecolor="#cccccc",
    )

    fig.suptitle(
        f"{args.experiment}  —  Matched Surrogate × GT Panel\n"
        "Top: y vs ŷ scatter (main sequence)    |    Bottom: Spearman ρ across 10 sequences",
        fontsize=11, y=0.975,
    )

    out_path = os.path.join(args.out_dir, "matched_2x5_panel.png")
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[done] figure → {out_path}", flush=True)
    print(f"[done] RB ρ values → {rb_rho_json}", flush=True)


if __name__ == "__main__":
    main()
