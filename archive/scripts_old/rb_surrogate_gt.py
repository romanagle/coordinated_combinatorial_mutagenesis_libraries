"""rb_surrogate_gt.py

ResidualBind Surrogate GT pipeline:

  Stage 1: Generate 20 k library → ResidualBind scores → train pairwise_GE
           MAVENN surrogate = SurrogateGT_RB (biologically grounded GT).

  Stage 2: Generate another 20 k library → score with SurrogateGT_RB →
           train all 4 surrogate configs → record Spearman ρ on random split
           and uniformised eval library.

  Stage 3a (Panel): Replace the bottom row of the existing noisy panel with
           the RB surrogate GT y-vs-ŷ row (main sequence only).

  Stage 3b (Violin): For each of the N saved sequences in --save_dir, run
           stages 1-2 to collect ρ values, then append a new subplot to the
           existing Spearman ρ violin figure.

Usage:
    cd /home/nagle/final_version
    python scripts/rb_surrogate_gt.py \\
        --seq         AAAAAAACCCCCAAAAAAUCGGCUGGACCGGGAAAAAAAAA \\
        --experiment  RNCMPT00111 \\
        --save_dir    outputs/postabrcms/vts1high/spearman_violin_data \\
        --panel_in    outputs/surrogate_gt_pipeline/noisy_panel_v2/panel_noise0.05.png \\
        --violin_in   outputs/postabrcms/vts1high/spearman_violin_run0.png \\
        --seed        42
"""

import argparse
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '/home/nagle/final_version/squid-nn')
sys.path.append('/home/nagle/final_version/squid-manuscript/squid')
sys.path.append('/home/nagle/final_version/squid-nn/squid')
sys.path.append('/home/nagle/final_version/residualbind')

import squid.surrogate_zoo
from seq_utils import rna_to_one_hot, remove_padding
from ground_truth import uniformize_by_histogram
import helper
from residualbind import ResidualBind

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUCS    = ['A', 'C', 'G', 'U']
_NUCS_A = np.array(list("ACGU"))

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

# Nonlinear pairwise GE — matches paper's overparameterised surrogate GT config
SURROGATE_GT_RB_CFG = {
    "gpmap": "pairwise", "linearity": "nonlinear",
    "regression_type": "GE", "noise": "Gaussian",
    "noise_order": 2, "reg_strength": 0.005, "hidden_nodes": 10,
}

CFG_LABELS = {
    "additive":    "Additive",
    "pairwise":    "Pairwise",
    "additive_GE": "Add GE",
    "pairwise_GE": "Pair GE",
}

BLUE   = "#4C72B0"
ORANGE = "#DD8452"


# ---------------------------------------------------------------------------
# Library / scoring utilities
# ---------------------------------------------------------------------------

def _to_str(X):
    return np.array(["".join(_NUCS_A[np.argmax(x, axis=1)]) for x in X], dtype=object)


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


def pad_to_41(X_lib):
    """Zero-pad library sequences to ResidualBind's required 41 nt."""
    N, L, A = X_lib.shape
    if L == 41:
        return X_lib
    pad = 41 - L
    beg = np.zeros((N, pad // 2,          A), dtype=np.float32)
    end = np.zeros((N, pad - pad // 2,    A), dtype=np.float32)
    return np.concatenate([beg, X_lib, end], axis=1)


def rb_score_library(X_lib, residbind, batch_size=512):
    """Score a library with ResidualBind (auto-pads to 41 nt, returns flat array)."""
    X_padded = pad_to_41(X_lib)
    parts = []
    for i in range(0, len(X_padded), batch_size):
        pred = residbind.predict(X_padded[i:i + batch_size])
        parts.append(np.asarray(pred, dtype=float).ravel())
    return np.concatenate(parts)


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
    print(f"    [{label}]  lr={lr:.2e}  bsz={bsz}  ep={epochs}")
    return model, test_df


# ---------------------------------------------------------------------------
# Stage 1: Train SurrogateGT_RB from ResidualBind
# ---------------------------------------------------------------------------

def stage1_train_rb_surrogate(wt_onehot, residbind, exact_mut_count, seed,
                               n_train=20_000, verbose=1):
    """Generate 20 k library → RB scores → train pairwise_GE SurrogateGT_RB."""
    print(f"  [Stage 1] Generating {n_train:,}-seq library → ResidualBind …")
    rng   = np.random.default_rng(seed)
    X_lib = generate_library(wt_onehot, n_train, exact_mut_count, rng)
    y_lib = rb_score_library(X_lib, residbind)
    print(f"  [Stage 1] RB scores: [{y_lib.min():.4f}, {y_lib.max():.4f}]  "
          f"std={y_lib.std():.4f}")
    print(f"  [Stage 1] Training SurrogateGT_RB (nonlinear pairwise GE) …")
    model, test_df = train_mavenn(X_lib, y_lib.reshape(-1, 1),
                                  SURROGATE_GT_RB_CFG, label="SurrogateGT_RB",
                                  verbose=verbose)
    print("  [Stage 1] SurrogateGT_RB trained.")
    return model, X_lib, y_lib


# ---------------------------------------------------------------------------
# Stage 2: Evaluate surrogate recovery from SurrogateGT_RB
# ---------------------------------------------------------------------------

def stage2_eval_recovery(rb_sgt_model, wt_onehot, exact_mut_count, seed,
                          n_train=20_000, n_pool=50_000, target_n=5_000):
    """Generate another 20 k → score with SurrogateGT_RB → train 4 surrogates."""
    print(f"  [Stage 2] Generating {n_train:,}-seq library → SurrogateGT_RB …")
    rng_lib = np.random.default_rng(seed + 500)
    X_lib   = generate_library(wt_onehot, n_train, exact_mut_count, rng_lib)
    y_lib   = _predict_chunked(rb_sgt_model, _to_str(X_lib)).reshape(-1, 1)
    print(f"  [Stage 2] GT scores: [{y_lib.min():.4f}, {y_lib.max():.4f}]")

    print(f"  [Stage 2] Building eval library from {n_pool:,}-seq pool …")
    rng_pool = np.random.default_rng(seed + 500 + 99_999)
    X_pool   = generate_library(wt_onehot, n_pool, exact_mut_count, rng_pool)
    y_pool   = _predict_chunked(rb_sgt_model, _to_str(X_pool))
    y_uni, keep_idx = uniformize_by_histogram(
        y_pool, X=None, n_bins=200, clip_hi=98, target_n=target_n, seed=seed,
    )
    X_eval = X_pool[keep_idx]
    y_eval = y_uni.astype(float)
    del X_pool, y_pool
    print(f"  [Stage 2] Eval library: {len(y_eval):,} seqs  "
          f"[{y_eval.min():.4f}, {y_eval.max():.4f}]")

    row = {}
    for cfg_name, cfg in SURROGATE_CONFIGS.items():
        print(f"  [Stage 2] training {cfg_name} …")
        model, test_df = train_mavenn(X_lib, y_lib, cfg, label=cfg_name)
        row[cfg_name] = {
            "model": model, "test_df": test_df,
            "X_eval": X_eval, "y_eval": y_eval,
            "y_rand_all": y_lib.ravel(),
        }
    return row


# ---------------------------------------------------------------------------
# Stage 3a helpers: panel row
# ---------------------------------------------------------------------------

def _row_scatter_stats(model, test_df, X_eval, y_eval):
    """Return (yh_rs, y_rs, yh_ev, rho_r, rho_e, r2r, r2e) for one cell."""
    cols  = list(test_df.columns)
    x_col = "x" if "x" in cols else "X"
    y_col = "y" if "y" in cols else next(c2 for c2 in cols if c2.startswith("y"))
    x_rs  = np.asarray(test_df[x_col])
    y_rs  = np.asarray(test_df[y_col], dtype=float).ravel()
    yh_rs = _predict_chunked(model, x_rs)
    yh_ev = _predict_chunked(model, _to_str(X_eval))
    m_r   = np.isfinite(y_rs)   & np.isfinite(yh_rs)
    m_e   = np.isfinite(y_eval) & np.isfinite(yh_ev)
    rho_r = float(spearmanr(yh_rs[m_r], y_rs[m_r])[0])   if m_r.sum() >= 3 else np.nan
    rho_e = float(spearmanr(yh_ev[m_e], y_eval[m_e])[0]) if m_e.sum() >= 3 else np.nan
    r_r   = float(pearsonr( yh_rs[m_r], y_rs[m_r])[0])   if m_r.sum() >= 3 else np.nan
    r_e   = float(pearsonr( yh_ev[m_e], y_eval[m_e])[0]) if m_e.sum() >= 3 else np.nan
    r2r   = r_r ** 2 if np.isfinite(r_r) else np.nan
    r2e   = r_e ** 2 if np.isfinite(r_e) else np.nan
    return yh_rs, y_rs, yh_ev, rho_r, rho_e, r2r, r2e


def plot_rb_row_strip(row_results, out_path):
    """Plot 1×4 y-vs-ŷ strip matching noisy_panel.py cell style."""
    n_cols = len(SURROGATE_CONFIGS)
    fig, axes = plt.subplots(1, n_cols, figsize=(4.2 * n_cols, 4.0))

    row_lo, row_hi = np.inf, -np.inf
    cells = []
    for cfg_name in SURROGATE_CONFIGS:
        entry  = row_results[cfg_name]
        y_eval = entry["y_eval"].ravel()
        yh_rs, y_rs, yh_ev, rho_r, rho_e, r2r, r2e = _row_scatter_stats(
            entry["model"], entry["test_df"], entry["X_eval"], y_eval,
        )
        m_r = np.isfinite(y_rs)   & np.isfinite(yh_rs)
        m_e = np.isfinite(y_eval) & np.isfinite(yh_ev)
        all_v = np.concatenate([yh_rs[m_r], yh_ev[m_e], y_rs[m_r], y_eval[m_e]])
        if all_v.size:
            row_lo = min(row_lo, float(all_v.min()))
            row_hi = max(row_hi, float(all_v.max()))
        cells.append((cfg_name, yh_rs, y_rs, yh_ev, y_eval, rho_r, rho_e, r2r, r2e))

    for c, (cfg_name, yh_rs, y_rs, yh_ev, y_eval,
            rho_r, rho_e, r2r, r2e) in enumerate(cells):
        ax = axes[c]
        ax.scatter(yh_rs, y_rs,   s=1, alpha=0.08, color="C0", rasterized=True,
                   label=f"random  ρ={rho_r:.2f}  R²={r2r:.2f}")
        ax.scatter(yh_ev, y_eval, s=1, alpha=0.08, color="C1", rasterized=True,
                   label=f"eval     ρ={rho_e:.2f}  R²={r2e:.2f}")
        ax.set_xlabel("ŷ", fontsize=8)
        ax.set_ylabel("y", fontsize=8)
        ax.tick_params(labelsize=6)
        ax.set_title(f"{CFG_LABELS[cfg_name]} / RB Surrogate GT", fontsize=7.5)
        h, l = ax.get_legend_handles_labels()
        ax.legend(h, l, markerscale=6, fontsize=5.5, loc="upper left", framealpha=0.6)
        if np.isfinite(row_lo) and np.isfinite(row_hi) and row_lo < row_hi:
            pad = (row_hi - row_lo) * 0.05
            lm  = [row_lo - pad, row_hi + pad]
            ax.set_xlim(lm); ax.set_ylim(lm)
            ax.plot(lm, lm, "k--", linewidth=0.8, zorder=10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] RB row strip → {out_path}")


def replace_panel_bottom_row(panel_in, row_strip_path, panel_out):
    """Crop top 4/5 of panel_in; stitch row_strip as the new bottom row."""
    from PIL import Image
    top_img = Image.open(panel_in).convert("RGB")
    bot_img = Image.open(row_strip_path).convert("RGB")
    W, H    = top_img.size

    crop_h   = int(round(H * 4 / 5))
    top_crop = top_img.crop((0, 0, W, crop_h))

    new_h    = int(round(bot_img.height * W / bot_img.width))
    bot_rs   = bot_img.resize((W, new_h), Image.LANCZOS)

    combined = Image.new("RGB", (W, crop_h + new_h), (255, 255, 255))
    combined.paste(top_crop, (0, 0))
    combined.paste(bot_rs,   (0, crop_h))
    combined.save(panel_out, dpi=(150, 150))
    print(f"  [stitch] updated panel → {panel_out}")


# ---------------------------------------------------------------------------
# Stage 3b helpers: violin
# ---------------------------------------------------------------------------

def _rho_pair(model, test_df, X_eval, y_eval):
    """Return (rho_rand, rho_eval) for a trained model."""
    cols  = list(test_df.columns)
    x_col = "x" if "x" in cols else "X"
    y_col = "y" if "y" in cols else next(c2 for c2 in cols if c2.startswith("y"))
    x_rs  = np.asarray(test_df[x_col])
    y_rs  = np.asarray(test_df[y_col], dtype=float).ravel()
    yh_rs = _predict_chunked(model, x_rs)
    yh_ev = _predict_chunked(model, _to_str(X_eval))
    m_r   = np.isfinite(y_rs)   & np.isfinite(yh_rs)
    m_e   = np.isfinite(y_eval) & np.isfinite(yh_ev)
    rho_r = float(spearmanr(yh_rs[m_r], y_rs[m_r])[0]) if m_r.sum() >= 3 else np.nan
    rho_e = float(spearmanr(yh_ev[m_e], y_eval[m_e])[0]) if m_e.sum() >= 3 else np.nan
    return rho_r, rho_e


def plot_rb_violin_strip(accum_rb, out_path, experiment="", n_seqs=10):
    """Single violin subplot for RB Surrogate GT, matching spearman_violin.py style."""
    cfg_names = list(SURROGATE_CONFIGS.keys())
    VIOLIN_W  = 0.42

    # Width matches the existing 2×2 violin (14 in); height is half a row (~4.5 in)
    fig, ax = plt.subplots(1, 1, figsize=(14.0, 4.5))
    positions, data_lists, colors = [], [], []
    xtick_pos, xtick_labs = [], []

    x_pos = 0.0
    for cfg_name in cfg_names:
        rho_rnd = np.asarray(accum_rb[cfg_name]["rand"], dtype=float)
        rho_cur = np.asarray(accum_rb[cfg_name]["eval"], dtype=float)
        rho_rnd = rho_rnd[np.isfinite(rho_rnd)]
        rho_cur = rho_cur[np.isfinite(rho_cur)]
        positions.extend([x_pos, x_pos + 0.55])
        data_lists.extend([rho_rnd, rho_cur])
        colors.extend([BLUE, ORANGE])
        xtick_pos.append(x_pos + 0.275)
        xtick_labs.append(CFG_LABELS.get(cfg_name, cfg_name))
        x_pos += 1.4

    for pos, data, color in zip(positions, data_lists, colors):
        if data.size == 0:
            continue
        if data.size >= 3:
            parts = ax.violinplot(data, positions=[pos], widths=VIOLIN_W,
                                  showmedians=False, showextrema=False)
            for pc in parts["bodies"]:
                pc.set_facecolor(color)
                pc.set_alpha(0.65)
        rng_j  = np.random.default_rng(42)
        jitter = rng_j.uniform(-0.08, 0.08, size=data.size)
        ax.scatter(np.full(data.size, pos) + jitter, data,
                   color=color, s=25, zorder=7, alpha=0.8, linewidths=0)
        if data.size >= 2:
            mean = np.nanmean(data)
            sem  = np.nanstd(data, ddof=1) / np.sqrt(data.size)
            ax.errorbar([pos], [mean], yerr=[[sem], [sem]], fmt="o",
                        color="black", markersize=5, linewidth=2.0,
                        capsize=5, capthick=2.0, zorder=10)
        elif data.size == 1:
            ax.scatter([pos], data, color="black", s=40, zorder=10)

    ax.set_xticks(xtick_pos)
    ax.set_xticklabels(xtick_labs, fontsize=9)
    ax.set_ylabel("Spearman ρ", fontsize=10)
    ax.set_title("ResidualBind Surrogate GT", fontsize=11, fontweight="bold")
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0, linewidth=0.8, color="gray", linestyle="--", alpha=0.4)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    if xtick_pos:
        ax.set_xlim(-0.5, x_pos - 0.15)

    from matplotlib.patches import Patch
    fig.legend(
        handles=[
            Patch(facecolor=BLUE,   alpha=0.65, label="Random (MAVE test split)"),
            Patch(facecolor=ORANGE, alpha=0.65, label="Eval (curated library)"),
            plt.Line2D([0], [0], color="black", marker="o", linewidth=2,
                       markersize=5, label="Mean ± SEM"),
        ],
        loc="upper right", fontsize=9, bbox_to_anchor=(1.0, 1.0),
    )
    fig.suptitle(
        f"{experiment} — RB Surrogate GT Spearman ρ"
        f"  (N={n_seqs} sequences, L=41)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] RB violin strip → {out_path}")


def append_violin_strip(violin_in, strip_path, violin_out):
    """Stitch RB violin strip below the existing violin figure."""
    from PIL import Image
    top = Image.open(violin_in).convert("RGB")
    bot = Image.open(strip_path).convert("RGB")
    W   = top.width
    new_h = int(round(bot.height * W / bot.width))
    bot   = bot.resize((W, new_h), Image.LANCZOS)
    combined = Image.new("RGB", (W, top.height + new_h), (255, 255, 255))
    combined.paste(top, (0, 0))
    combined.paste(bot, (0, top.height))
    combined.save(violin_out, dpi=(150, 150))
    print(f"  [stitch] updated violin → {violin_out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq",
                        default="AAAAAAACCCCCAAAAAAUCGGCUGGACCGGGAAAAAAAAA")
    parser.add_argument("--experiment", default="RNCMPT00111")
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--save_dir",
                        default="outputs/postabrcms/vts1high/spearman_violin_data",
                        help="Dir with seq_00X/ sub-folders from spearman_violin.py")
    parser.add_argument("--panel_in",
                        default="outputs/surrogate_gt_pipeline/noisy_panel_v2/panel_noise0.05.png")
    parser.add_argument("--violin_in",
                        default="outputs/postabrcms/vts1high/spearman_violin_run0.png")
    args = parser.parse_args()

    tmp_dir = os.path.join(os.path.dirname(os.path.abspath(args.panel_in)), "_rb_tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    # ── Load ResidualBind ─────────────────────────────────────────────────
    data_path    = os.path.expanduser(
        "~/residualbind/data/RNAcompete_2013/rnacompete2013.h5"
    )
    save_path    = "/home/nagle/residualbind/weights/log_norm_seq"
    rbp_index    = helper.find_experiment_index(data_path, args.experiment)
    train, _, _  = helper.load_rnacompete_data(
        data_path, ss_type="seq", normalization="log_norm", rbp_index=rbp_index,
    )
    input_shape  = list(train["inputs"].shape)[1:]
    weights_path = os.path.join(save_path, args.experiment + "_weights.hdf5")
    residbind    = ResidualBind(input_shape, 1, weights_path)
    residbind.load_weights()
    print(f"ResidualBind loaded ({args.experiment})")

    # ── Main sequence (used for panel) ────────────────────────────────────
    seq       = args.seq.upper().replace("T", "U")
    wt_main, _ = remove_padding(rna_to_one_hot(seq))
    L_main    = wt_main.shape[0]
    exact_mut = min(4, max(1, L_main // 2))
    print(f"Main seq: {seq}  L={L_main}  exact_mut={exact_mut}")

    # ══════════════════════════════════════════════════════════════════════
    # Panel (main sequence only)
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print("  Panel — Stage 1: Train SurrogateGT_RB")
    print(f"{'='*65}")
    rb_sgt_main, _, _ = stage1_train_rb_surrogate(
        wt_main, residbind, exact_mut, args.seed, verbose=1,
    )

    print(f"\n{'='*65}")
    print("  Panel — Stage 2: Eval surrogate recovery")
    print(f"{'='*65}")
    row_results = stage2_eval_recovery(
        rb_sgt_main, wt_main, exact_mut, args.seed,
    )

    print(f"\n{'='*65}")
    print("  Panel — Stage 3a: Replace bottom row")
    print(f"{'='*65}")
    row_strip = os.path.join(tmp_dir, "rb_row_strip.png")
    plot_rb_row_strip(row_results, row_strip)
    replace_panel_bottom_row(args.panel_in, row_strip, args.panel_in)

    # ══════════════════════════════════════════════════════════════════════
    # Violin (all saved sequences)
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print("  Violin — loading saved sequences")
    print(f"{'='*65}")

    seq_dirs = sorted(
        d for d in [
            os.path.join(args.save_dir, f"seq_{i:03d}") for i in range(100)
        ]
        if os.path.isdir(d)
    )
    print(f"  Found {len(seq_dirs)} saved sequences.")

    accum_rb = {cfg: {"rand": [], "eval": []} for cfg in SURROGATE_CONFIGS}

    for s_idx, seq_dir in enumerate(seq_dirs):
        wt_oh   = np.load(os.path.join(seq_dir, "sequence_onehot.npy")).astype(np.float32)
        seq_str = "".join(_NUCS_A[np.argmax(wt_oh, axis=1)])
        L       = wt_oh.shape[0]
        em      = min(4, max(1, L // 2))
        seed_s  = args.seed + s_idx * 100_000

        print(f"\n  [{s_idx+1}/{len(seq_dirs)}] {seq_str}  L={L}  exact_mut={em}")

        rb_sgt_s, _, _ = stage1_train_rb_surrogate(
            wt_oh, residbind, em, seed_s, verbose=0,
        )
        row_s = stage2_eval_recovery(
            rb_sgt_s, wt_oh, em, seed_s,
        )

        for cfg_name in SURROGATE_CONFIGS:
            rho_r, rho_e = _rho_pair(
                row_s[cfg_name]["model"],
                row_s[cfg_name]["test_df"],
                row_s[cfg_name]["X_eval"],
                row_s[cfg_name]["y_eval"],
            )
            accum_rb[cfg_name]["rand"].append(rho_r)
            accum_rb[cfg_name]["eval"].append(rho_e)
            print(f"    {cfg_name:<14}  ρ_rand={rho_r:+.4f}  ρ_eval={rho_e:+.4f}")

    print(f"\n{'='*65}")
    print("  Violin — Stage 3b: Append RB violin strip")
    print(f"{'='*65}")
    strip_path = os.path.join(tmp_dir, "rb_violin_strip.png")
    plot_rb_violin_strip(
        accum_rb, strip_path,
        experiment=args.experiment, n_seqs=len(seq_dirs),
    )
    append_violin_strip(args.violin_in, strip_path, args.violin_in)

    print(f"\n{'='*65}")
    print("  Done.")
    print(f"  Panel  → {args.panel_in}")
    print(f"  Violin → {args.violin_in}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
