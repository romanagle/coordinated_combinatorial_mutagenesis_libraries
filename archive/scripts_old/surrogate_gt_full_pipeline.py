"""surrogate_gt_full_pipeline.py

Full pipeline — runs sequentially, no subprocesses, safe to leave unattended.

Stage 1  — Train SurrogateGT
    Nonlinear additive+pairwise MAVENN surrogate fitted on 20k sequences
    scored by the synthetic nonlin_additive_pairwise GT (seed=42).
    The trained surrogate becomes the new oracle for all later stages.

Stage 2  — Build eval library
    Score a large pool with SurrogateGT; uniformise to a flat distribution.
    This eval library is reused across all library-size sweeps.

Stage 3  — Panel row (y-vs-ŷ for all 4 surrogate types vs SurrogateGT)
    Train each of the 4 surrogate configs on a 20k library scored by
    SurrogateGT; produce scatter plots annotated with ρ and R².
    Stitch as a new 5th row onto the existing 16-panel PNG.
    Also plot overlaid random-vs-eval distribution.

Stage 4  — Library-size sweep
    For each N in [200, 2_000, 20_000, 200_000, 2_000_000]:
      generate N-sequence library → score with SurrogateGT →
      train all 4 surrogate configs → record Spearman ρ (random + eval).
    Plot ρ vs N.

Usage:
    python scripts/surrogate_gt_full_pipeline.py \\
        --existing_panel outputs/postabrcms/vts1high/y_vs_yhat_16panel_run0.png \\
        --out_dir        outputs/surrogate_gt_pipeline
"""

import argparse
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr

sys.path.insert(0, os.path.dirname(__file__))
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

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

NUCS     = ['A', 'C', 'G', 'U']
_NUCS_A  = np.array(list("ACGU"))

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

CFG_LABELS = {
    "additive":    "Additive",
    "pairwise":    "Pairwise",
    "additive_GE": "Additive GE",
    "pairwise_GE": "Pairwise GE",
}

CFG_COLORS = {
    "additive":    "#1f77b4",
    "pairwise":    "#ff7f0e",
    "additive_GE": "#2ca02c",
    "pairwise_GE": "#d62728",
}

SURROGATE_GT_CFG = {
    "gpmap": "pairwise", "linearity": "nonlinear",
    "regression_type": "GE", "noise": "Gaussian",
    "noise_order": 2, "reg_strength": 0.005, "hidden_nodes": 10,
}

LIBRARY_SIZES = [200, 2_000, 20_000, 200_000, 2_000_000]


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _to_str(X):
    return np.array(["".join(_NUCS_A[np.argmax(x, axis=1)]) for x in X], dtype=object)


def _predict_chunked(model, x_str, chunk=1_000):
    """Small chunk size to avoid GPU OOM on pairwise models (L×A×L×A tensor)."""
    parts = []
    for s in range(0, len(x_str), chunk):
        parts.append(np.asarray(model.x_to_yhat(x_str[s:s + chunk]), dtype=float).ravel())
    return np.concatenate(parts)


def generate_library(wt_onehot, n_seqs, exact_mut_count, rng):
    L      = wt_onehot.shape[0]
    wt_idx = np.argmax(wt_onehot, axis=1)
    parts, CHUNK = [], 50_000
    for start in range(0, n_seqs, CHUNK):
        nc       = min(CHUNK, n_seqs - start)
        noise    = rng.random((nc, L))
        pos      = np.argpartition(noise, exact_mut_count, axis=1)[:, :exact_mut_count]
        X        = np.tile(wt_onehot[None], (nc, 1, 1)).astype(np.float32)
        n_idx    = np.repeat(np.arange(nc), exact_mut_count)
        p_idx    = pos.ravel()
        wt_at    = wt_idx[p_idx]
        rand_nuc = rng.integers(0, 3, size=nc * exact_mut_count)
        new_nucs = np.where(rand_nuc >= wt_at, rand_nuc + 1, rand_nuc)
        X[n_idx, p_idx, :]        = 0
        X[n_idx, p_idx, new_nucs] = 1
        parts.append(X)
    return np.concatenate(parts, axis=0)


def train_mavenn(X, y, cfg, label="", verbose=0):
    N        = X.shape[0]
    bsz      = max(32, min(N // 150, 2048))
    lr       = 5e-4 * min(1.0, (20_000 / N) ** 0.5)
    is_nlp   = (cfg["gpmap"] == "pairwise" and cfg["linearity"] == "nonlinear")
    epochs   = 1000 if is_nlp else 500
    patience = 50   if is_nlp else 25

    wrapper = squid.surrogate_zoo.SurrogateMAVENN(
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
    return model, test_df, wrapper


def spearman_on_testdf(model, test_df):
    cols  = list(test_df.columns)
    x_col = "x" if "x" in cols else "X"
    y_col = "y" if "y" in cols else next(c for c in cols if c.startswith("y"))
    x_str = np.asarray(test_df[x_col])
    y_t   = np.asarray(test_df[y_col], dtype=float).ravel()
    y_h   = np.asarray(model.x_to_yhat(x_str), dtype=float).ravel()
    m     = np.isfinite(y_t) & np.isfinite(y_h)
    return float(spearmanr(y_t[m], y_h[m])[0]) if m.sum() >= 3 else np.nan


# ---------------------------------------------------------------------------
# Stage 1 — Train SurrogateGT
# ---------------------------------------------------------------------------

def stage1_train_surrogate_gt(wt_onehot, exact_mut_count, seed):
    print(f"\n{'='*65}")
    print("  Stage 1: Train SurrogateGT")
    print(f"{'='*65}")

    rng = np.random.default_rng(seed)
    W_mut, mut_map, b0 = init_additive_noWT(rng, wt_onehot, sigma=0.5, l1_w=0.1, bias=0.0)
    edges, J = init_pairwise_potts_optionA(
        rng, wt_onehot,
        p_edge=0.70, df=5.0, lambda_J=2.0, p_rescue=0.10, wt_rowcol_zero=True,
    )

    rng_ref = np.random.default_rng(seed + 1)
    X_ref   = generate_library(wt_onehot, 200_000, exact_mut_count, rng_ref)
    s_add   = additive_affinity_noWT(X_ref, W_mut, mut_map, b=b0).reshape(-1)
    nk      = init_sigmoid_nonlin(s_add)
    del X_ref, s_add
    print(f"  sigmoid ref_std = {nk['_norm_std']:.4f}")

    rng_lib = np.random.default_rng(seed + 2)
    X_lib   = generate_library(wt_onehot, 20_000, exact_mut_count, rng_lib)
    gt_sc   = compute_gt_scores_for_library_potts(
        X_lib, W_mut=W_mut, mut_map=mut_map, b0=b0,
        nonlin_name="sigmoid", nonlin_kwargs=nk, edges=edges, J=J,
    )
    y = gt_sc["nonlin_additive_pairwise"].astype(float).reshape(-1, 1)
    print(f"  training library: N={len(y):,}  "
          f"y range=[{y.min():.3f}, {y.max():.3f}]")

    sgt_model, _, _ = train_mavenn(X_lib, y, SURROGATE_GT_CFG,
                                   label="SurrogateGT", verbose=1)
    print("  SurrogateGT trained.")
    return sgt_model


# ---------------------------------------------------------------------------
# Stage 2 — Build eval library from SurrogateGT
# ---------------------------------------------------------------------------

def stage2_build_eval_library(sgt_model, wt_onehot, exact_mut_count, seed,
                               n_pool=50_000, target_n=5_000):
    print(f"\n{'='*65}")
    print("  Stage 2: Build SurrogateGT eval library")
    print(f"{'='*65}")

    rng_pool = np.random.default_rng(seed + 99_999)
    X_pool   = generate_library(wt_onehot, n_pool, exact_mut_count, rng_pool)
    print(f"  scoring pool of {n_pool:,} …")
    y_pool   = _predict_chunked(sgt_model, _to_str(X_pool))

    y_uni, keep_idx = uniformize_by_histogram(
        y_pool, X=None, n_bins=200, clip_hi=98,
        target_n=target_n, seed=seed,
    )
    X_eval = X_pool[keep_idx]
    y_eval = y_uni.astype(float)
    print(f"  eval library: {len(y_eval):,} seqs  "
          f"range=[{y_eval.min():.4f}, {y_eval.max():.4f}]")
    del X_pool, y_pool
    return X_eval, y_eval


# ---------------------------------------------------------------------------
# Stage 3 — Panel row
# ---------------------------------------------------------------------------

def stage3_panel_row(sgt_model, wt_onehot, exact_mut_count, seed,
                     X_eval, y_eval, out_dir, existing_panel):
    print(f"\n{'='*65}")
    print("  Stage 3: Panel row (y-vs-ŷ + distribution)")
    print(f"{'='*65}")

    # 3a. Generate 20k panel library
    rng_panel = np.random.default_rng(seed + 200)
    X_panel   = generate_library(wt_onehot, 20_000, exact_mut_count, rng_panel)
    y_panel   = _predict_chunked(sgt_model, _to_str(X_panel)).reshape(-1, 1)
    print(f"  panel library: N={len(y_panel):,}  "
          f"range=[{y_panel.min():.4f}, {y_panel.max():.4f}]")

    # 3b. Train all 4 surrogate configs
    results = {}
    for cfg_name, cfg in SURROGATE_CONFIGS.items():
        print(f"\n  training {cfg_name} …")
        model, test_df, _ = train_mavenn(X_panel, y_panel, cfg, label=cfg_name)
        results[cfg_name] = {"model": model, "test_df": test_df,
                              "X_eval": X_eval, "y_eval": y_eval}

    # 3c. y-vs-ŷ strip
    new_row_path = os.path.join(out_dir, "surrogate_gt_row.png")
    _plot_row(results, new_row_path)

    # 3d. Stitch onto existing 16-panel
    stitched_path = os.path.join(out_dir, "y_vs_yhat_20panel.png")
    _stitch(existing_panel, new_row_path, stitched_path)

    # 3e. Distribution plot
    dist_path = os.path.join(out_dir, "surrogate_gt_random_vs_eval.png")
    _plot_distribution(y_panel.ravel(), y_eval, dist_path)


def _plot_row(results, out_path):
    cfg_names = list(SURROGATE_CONFIGS.keys())
    fig, axes = plt.subplots(1, len(cfg_names), figsize=(4.2 * len(cfg_names), 4.0))
    row_lo, row_hi = np.inf, -np.inf
    panel_data = []

    for c, cfg_name in enumerate(cfg_names):
        entry      = results[cfg_name]
        model      = entry["model"]
        df         = entry["test_df"]
        X_eval     = entry["X_eval"]
        y_eval     = entry["y_eval"].ravel()
        cols       = list(df.columns)
        x_col      = "x" if "x" in cols else "X"
        y_col      = "y" if "y" in cols else next(c2 for c2 in cols if c2.startswith("y"))
        X_rs       = np.asarray(df[x_col])
        y_rs       = np.asarray(df[y_col], dtype=float).ravel()
        yh_rs      = _predict_chunked(model, X_rs)
        yh_ev      = _predict_chunked(model, _to_str(X_eval))
        m_r        = np.isfinite(y_rs) & np.isfinite(yh_rs)
        m_e        = np.isfinite(y_eval) & np.isfinite(yh_ev)
        rho_r      = float(spearmanr(yh_rs[m_r], y_rs[m_r])[0])  if m_r.sum() >= 3 else np.nan
        rho_e      = float(spearmanr(yh_ev[m_e], y_eval[m_e])[0]) if m_e.sum() >= 3 else np.nan
        r_r        = float(pearsonr( yh_rs[m_r], y_rs[m_r])[0])  if m_r.sum() >= 3 else np.nan
        r_e        = float(pearsonr( yh_ev[m_e], y_eval[m_e])[0]) if m_e.sum() >= 3 else np.nan
        r2_r, r2_e = r_r**2 if np.isfinite(r_r) else np.nan, r_e**2 if np.isfinite(r_e) else np.nan
        all_v = np.concatenate([yh_rs[m_r], yh_ev[m_e], y_rs[m_r], y_eval[m_e]])
        if all_v.size:
            row_lo = min(row_lo, float(all_v.min()))
            row_hi = max(row_hi, float(all_v.max()))
        panel_data.append((c, cfg_name, yh_rs, y_rs, yh_ev, y_eval,
                           rho_r, rho_e, r_r, r_e, r2_r, r2_e))

    for (c, cfg_name, yh_rs, y_rs, yh_ev, y_eval,
         rho_r, rho_e, r_r, r_e, r2_r, r2_e) in panel_data:
        ax = axes[c]
        ax.scatter(yh_rs, y_rs, s=1, alpha=0.08, color="C0", rasterized=True,
                   label=f"random   ρ={rho_r:.2f}  r={r_r:.2f}  R²={r2_r:.2f}")
        ax.scatter(yh_ev, y_eval, s=1, alpha=0.08, color="C1", rasterized=True,
                   label=f"eval      ρ={rho_e:.2f}  r={r_e:.2f}  R²={r2_e:.2f}")
        ax.set_xlabel("ŷ", fontsize=8)
        ax.set_ylabel("y", fontsize=8)
        ax.tick_params(labelsize=6)
        ax.set_title(f"{CFG_LABELS[cfg_name]} / Surrogate GT", fontsize=7.5)
        h, l = ax.get_legend_handles_labels()
        ax.legend(h, l, markerscale=6, fontsize=6, loc="upper left", framealpha=0.6)
        if np.isfinite(row_lo) and np.isfinite(row_hi) and row_lo < row_hi:
            pad = (row_hi - row_lo) * 0.05
            lm  = [row_lo - pad, row_hi + pad]
            ax.set_xlim(lm); ax.set_ylim(lm)
            ax.plot(lm, lm, "k--", linewidth=0.8, zorder=10)

    fig.suptitle("y vs ŷ — Surrogate GT row\n"
                 "blue = random MAVE test split    orange = uniform eval library",
                 fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] surrogate-GT row → {out_path}")


def _stitch(existing_path, new_row_path, out_path):
    try:
        from PIL import Image
        top = Image.open(existing_path).convert("RGB")
        bot = Image.open(new_row_path).convert("RGB")
        if top.width != bot.width:
            new_h = int(bot.height * top.width / bot.width)
            bot   = bot.resize((top.width, new_h), Image.LANCZOS)
        combined = Image.new("RGB", (top.width, top.height + bot.height), (255, 255, 255))
        combined.paste(top, (0, 0))
        combined.paste(bot, (0, top.height))
        combined.save(out_path, dpi=(150, 150))
        print(f"  [stitch] 20-panel → {out_path}")
    except Exception as e:
        print(f"  [warn] stitch failed ({e}); skipping")


def _plot_distribution(y_rand, y_eval, out_path):
    y_rand = y_rand[np.isfinite(y_rand)]
    y_eval = y_eval[np.isfinite(y_eval)]
    lo, hi = np.percentile(np.concatenate([y_rand, y_eval]), [0.5, 99.5])
    bins   = np.linspace(lo, hi, 121)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.hist(y_rand, bins=bins, density=True, alpha=0.45,
            label=f"Random library (N={len(y_rand):,})")
    ax.hist(y_eval, bins=bins, density=True, alpha=0.45,
            label=f"Eval library (N={len(y_eval):,})")
    ax.set_title("Surrogate GT — random vs eval distributions")
    ax.set_xlabel("Surrogate GT score")
    ax.set_ylabel("Density")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] distribution → {out_path}")


# ---------------------------------------------------------------------------
# Stage 4 — Library-size sweep
# ---------------------------------------------------------------------------

def stage4_lib_size_sweep(sgt_model, wt_onehot, exact_mut_count, seed,
                           X_eval, y_eval, out_dir, library_sizes):
    print(f"\n{'='*65}")
    print("  Stage 4: Library-size sweep")
    print(f"{'='*65}")

    # results[cfg_name] = {"random": [], "eval": []}
    results = {cfg: {"random": [], "eval": []} for cfg in SURROGATE_CONFIGS}

    for N in library_sizes:
        print(f"\n  --- N = {N:,} ---")
        rng_lib = np.random.default_rng(seed + N)
        X_lib   = generate_library(wt_onehot, N, exact_mut_count, rng_lib)
        y_lib   = _predict_chunked(sgt_model, _to_str(X_lib)).reshape(-1, 1)
        print(f"  scored {N:,} seqs  y=[{y_lib.min():.3f}, {y_lib.max():.3f}]")

        for cfg_name, cfg in SURROGATE_CONFIGS.items():
            print(f"    {cfg_name:<14} … ", end="", flush=True)
            try:
                model, test_df, _ = train_mavenn(X_lib, y_lib, cfg, label="")
                rho_rand = spearman_on_testdf(model, test_df)
                # eval
                yh_ev = _predict_chunked(model, _to_str(X_eval))
                m_e   = np.isfinite(y_eval) & np.isfinite(yh_ev)
                rho_ev = float(spearmanr(y_eval[m_e], yh_ev[m_e])[0]) \
                         if m_e.sum() >= 3 else np.nan
            except Exception as e:
                print(f"FAILED ({e})")
                rho_rand = rho_ev = np.nan
            print(f"rho_random={rho_rand:+.3f}   rho_eval={rho_ev:+.3f}")
            results[cfg_name]["random"].append(rho_rand)
            results[cfg_name]["eval"].append(rho_ev)

        del X_lib, y_lib

    _plot_lib_size(results, library_sizes, out_dir)
    return results


def _plot_lib_size(results, library_sizes, out_dir):
    sizes = np.array(library_sizes)

    for split in ("random", "eval"):
        fig, ax = plt.subplots(figsize=(7, 5))
        for cfg_name, color in CFG_COLORS.items():
            rhos = np.array(results[cfg_name][split], dtype=float)
            ax.semilogx(sizes, rhos, marker="o", color=color,
                        linewidth=1.8, markersize=5, label=CFG_LABELS[cfg_name])
        ax.axhline(1.0, color="black", linestyle=":", linewidth=0.8, alpha=0.5)
        ax.set_xlabel("Library size", fontsize=11)
        ax.set_ylabel("Spearman ρ", fontsize=11)
        ax.set_title(f"Surrogate GT — library-size sweep ({split} split)", fontsize=11)
        ax.set_ylim(-0.1, 1.05)
        ax.set_xticks(sizes)
        ax.set_xticklabels([f"{s:,}" for s in sizes], rotation=35,
                            ha="right", fontsize=8)
        ax.legend(fontsize=9)
        ax.grid(True, which="both", alpha=0.25)
        fig.tight_layout()
        path = os.path.join(out_dir, f"lib_size_sweep_{split}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  [plot] lib-size {split} → {path}")

    # Combined (both splits, dashed = eval)
    fig, ax = plt.subplots(figsize=(7, 5))
    for cfg_name, color in CFG_COLORS.items():
        rhos_r = np.array(results[cfg_name]["random"], dtype=float)
        rhos_e = np.array(results[cfg_name]["eval"],   dtype=float)
        ax.semilogx(sizes, rhos_r, marker="o", color=color,
                    linewidth=1.8, markersize=5, linestyle="-",
                    label=f"{CFG_LABELS[cfg_name]} (random)")
        ax.semilogx(sizes, rhos_e, marker="s", color=color,
                    linewidth=1.8, markersize=5, linestyle="--",
                    label=f"{CFG_LABELS[cfg_name]} (eval)")
    ax.axhline(1.0, color="black", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Library size", fontsize=11)
    ax.set_ylabel("Spearman ρ", fontsize=11)
    ax.set_title("Surrogate GT — library-size sweep (combined)", fontsize=11)
    ax.set_ylim(-0.1, 1.05)
    ax.set_xticks(sizes)
    ax.set_xticklabels([f"{s:,}" for s in sizes], rotation=35, ha="right", fontsize=8)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    path = os.path.join(out_dir, "lib_size_sweep_combined.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] lib-size combined → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq",
                        default="AAAAAAACCCCCAAAAAAUCGGCUGGACCGGGAAAAAAAAA")
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--existing_panel",
                        default="outputs/postabrcms/vts1high/y_vs_yhat_16panel_run0.png")
    parser.add_argument("--out_dir",
                        default="outputs/surrogate_gt_pipeline")
    parser.add_argument("--max_lib_size", type=int, default=2_000_000,
                        help="Skip library sizes above this (e.g. 200000 for a quick run)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    seq           = args.seq.upper().replace("T", "U")
    oh_seq        = rna_to_one_hot(seq)
    wt_onehot, _  = remove_padding(oh_seq)
    L             = wt_onehot.shape[0]
    exact_mut     = min(4, max(1, L // 2))
    lib_sizes     = [N for N in LIBRARY_SIZES if N <= args.max_lib_size]

    print(f"Sequence: {seq}")
    print(f"L={L}  exact_mut={exact_mut}  library sizes={lib_sizes}")

    # Stage 1
    sgt_model = stage1_train_surrogate_gt(wt_onehot, exact_mut, args.seed)

    # Stage 2
    X_eval, y_eval = stage2_build_eval_library(
        sgt_model, wt_onehot, exact_mut, args.seed,
    )

    # Stage 3
    stage3_panel_row(
        sgt_model, wt_onehot, exact_mut, args.seed,
        X_eval, y_eval, args.out_dir, args.existing_panel,
    )

    # Stage 4
    stage4_lib_size_sweep(
        sgt_model, wt_onehot, exact_mut, args.seed,
        X_eval, y_eval, args.out_dir, lib_sizes,
    )

    print(f"\n{'='*65}")
    print("  All stages complete.")
    print(f"  Outputs in: {args.out_dir}/")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
