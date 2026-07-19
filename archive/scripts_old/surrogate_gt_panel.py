"""surrogate_gt_panel.py

1.  Re-train the SurrogateGT (nonlinear add+pairwise MAVENN surrogate
    fitted on the nonlin_add_pair synthetic GT scores) for the main sequence.

2.  Use the trained SurrogateGT as a new GT oracle:
      - generate a fresh 20k random library, score with SurrogateGT
      - build a uniformised eval library from those scores

3.  Train all 4 surrogate configs on the SurrogateGT-scored random library.

4.  Produce a 1-row × 4-col y-vs-ŷ strip (one panel per surrogate config)
    and stitch it as a 5th row onto the existing 16-panel PNG.

5.  Produce the overlaid random-vs-eval distribution plot.

Usage:
    python scripts/surrogate_gt_panel.py \\
        --existing_panel outputs/postabrcms/vts1high/y_vs_yhat_16panel_run0.png \\
        --out_dir        outputs/postabrcms/vts1high/surrogate_gt_row
"""

import argparse
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import spearmanr, pearsonr

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, '/home/nagle/final_version/squid-nn')
sys.path.append('/home/nagle/final_version/squid-manuscript/squid')
sys.path.append('/home/nagle/final_version/squid-nn/squid')
sys.path.append('/home/nagle/final_version/residualbind')

import squid.surrogate_zoo
from seq_utils import rna_to_one_hot, remove_padding
from ground_truth import (
    init_additive_noWT,
    init_pairwise_potts_optionA,
    init_sigmoid_nonlin,
    compute_gt_scores_for_library_potts,
    additive_affinity_noWT,
    uniformize_by_histogram,
)

# ---------------------------------------------------------------------------
# Constants — must match designed_seq.py
# ---------------------------------------------------------------------------

NUCS = ['A', 'C', 'G', 'U']
_NUCS_ARR = np.array(list("ACGU"))

# Surrogate configs matching designed_seq.py
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

# SurrogateGT config (nonlin add+pair, matching train_surrogate_gt.py)
SURROGATE_GT_CFG = {
    "gpmap": "pairwise", "linearity": "nonlinear",
    "regression_type": "GE", "noise": "Gaussian",
    "noise_order": 2, "reg_strength": 0.005, "hidden_nodes": 10,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_str(X):
    return np.array(["".join(_NUCS_ARR[np.argmax(x, axis=1)]) for x in X], dtype=object)


def generate_library(wt_onehot, n_seqs, exact_mut_count, rng):
    L      = wt_onehot.shape[0]
    wt_idx = np.argmax(wt_onehot, axis=1)
    parts  = []
    CHUNK  = 50_000
    for start in range(0, n_seqs, CHUNK):
        nc    = min(CHUNK, n_seqs - start)
        noise = rng.random((nc, L))
        pos   = np.argpartition(noise, exact_mut_count, axis=1)[:, :exact_mut_count]
        X     = np.tile(wt_onehot[None], (nc, 1, 1)).astype(np.float32)
        n_idx = np.repeat(np.arange(nc), exact_mut_count)
        p_idx = pos.ravel()
        wt_at    = wt_idx[p_idx]
        rand_nuc = rng.integers(0, 3, size=nc * exact_mut_count)
        new_nucs = np.where(rand_nuc >= wt_at, rand_nuc + 1, rand_nuc)
        X[n_idx, p_idx, :]        = 0
        X[n_idx, p_idx, new_nucs] = 1
        parts.append(X)
    return np.concatenate(parts, axis=0)


def _predict_chunked(model, x_str, chunk=10_000):
    parts = []
    for s in range(0, len(x_str), chunk):
        parts.append(np.asarray(model.x_to_yhat(x_str[s:s+chunk]), dtype=float).ravel())
    return np.concatenate(parts)


def train_mavenn(X, y, cfg, label=""):
    N   = X.shape[0]
    bsz = max(32, min(N // 150, 2048))
    lr  = 5e-4 * min(1.0, (20_000 / N) ** 0.5)
    is_nl_pair = (cfg["gpmap"] == "pairwise" and cfg["linearity"] == "nonlinear")
    epochs  = 1000 if is_nl_pair else 500
    patience = 50  if is_nl_pair else 25

    wrapper = squid.surrogate_zoo.SurrogateMAVENN(
        X.shape, num_tasks=1,
        gpmap=cfg["gpmap"], regression_type=cfg["regression_type"],
        linearity=cfg["linearity"], noise=cfg["noise"],
        noise_order=cfg["noise_order"], reg_strength=cfg["reg_strength"],
        hidden_nodes=cfg.get("hidden_nodes", 50),
        alphabet=NUCS, deduplicate=True, gpu=True,
    )
    model, mave_df, test_df = wrapper.train(
        X, y, learning_rate=lr, epochs=epochs, batch_size=bsz,
        early_stopping=True, patience=patience, restore_best_weights=True,
        save_dir=None, verbose=0,
    )
    print(f"  [{label}] trained  (lr={lr:.2e} bsz={bsz} ep={epochs})")
    return model, test_df


# ---------------------------------------------------------------------------
# Step 1 — train SurrogateGT
# ---------------------------------------------------------------------------

def build_surrogate_gt(wt_onehot, exact_mut_count, seed):
    """Return a trained MAVENN model representing the SurrogateGT."""
    rng = np.random.default_rng(seed)
    W_mut, mut_map, b0 = init_additive_noWT(rng, wt_onehot, sigma=0.5, l1_w=0.1, bias=0.0)
    edges, J = init_pairwise_potts_optionA(
        rng, wt_onehot,
        p_edge=0.70, df=5.0, lambda_J=2.0, p_rescue=0.10, wt_rowcol_zero=True,
    )
    rng_ref = np.random.default_rng(seed + 1)
    X_ref   = generate_library(wt_onehot, 200_000, exact_mut_count, rng_ref)
    s_add   = additive_affinity_noWT(X_ref, W_mut, mut_map, b=b0).reshape(-1)
    nonlin_kwargs = init_sigmoid_nonlin(s_add)
    del X_ref, s_add
    print(f"[sgt] sigmoid ref_std={nonlin_kwargs['_norm_std']:.4f}")

    rng_lib = np.random.default_rng(seed + 2)
    X_lib   = generate_library(wt_onehot, 20_000, exact_mut_count, rng_lib)
    gt_sc   = compute_gt_scores_for_library_potts(
        X_lib, W_mut=W_mut, mut_map=mut_map, b0=b0,
        nonlin_name="sigmoid", nonlin_kwargs=nonlin_kwargs, edges=edges, J=J,
    )
    y = gt_sc["nonlin_additive_pairwise"].astype(float).reshape(-1, 1)
    print(f"[sgt] training SurrogateGT on nonlin_add_pair scores "
          f"range=[{y.min():.3f}, {y.max():.3f}]")
    model, _ = train_mavenn(X_lib, y, SURROGATE_GT_CFG, label="SurrogateGT")
    return model


# ---------------------------------------------------------------------------
# Step 2 — score library with SurrogateGT + build eval library
# ---------------------------------------------------------------------------

def score_with_sgt(sgt_model, wt_onehot, exact_mut_count, seed,
                   n_random=20_000, n_pool=300_000, target_n=5_000):
    """Generate random library, score with SurrogateGT, build eval library."""
    L = wt_onehot.shape[0]

    # Random training library
    rng_rand = np.random.default_rng(seed)
    X_rand   = generate_library(wt_onehot, n_random, exact_mut_count, rng_rand)
    y_rand   = _predict_chunked(sgt_model, _to_str(X_rand)).reshape(-1, 1)
    print(f"[sgt score] random y range: [{y_rand.min():.4f}, {y_rand.max():.4f}]")

    # Eval pool
    rng_pool = np.random.default_rng(seed + 99_999)
    X_pool   = generate_library(wt_onehot, n_pool, exact_mut_count, rng_pool)
    y_pool   = _predict_chunked(sgt_model, _to_str(X_pool)).reshape(-1)

    y_uni, keep_idx = uniformize_by_histogram(
        y_pool, X=None, n_bins=200, clip_hi=98, target_n=target_n, seed=seed,
    )
    X_eval = X_pool[keep_idx]
    y_eval = y_uni.astype(float)
    print(f"[sgt score] eval library: {len(y_eval):,} seqs  "
          f"range=[{y_eval.min():.4f}, {y_eval.max():.4f}]")

    return X_rand, y_rand, X_eval, y_eval


# ---------------------------------------------------------------------------
# Step 3 — train all 4 surrogates on SurrogateGT-scored library
# ---------------------------------------------------------------------------

def train_all_surrogates(X_rand, y_rand, X_eval, y_eval):
    results = {}
    for cfg_name, cfg in SURROGATE_CONFIGS.items():
        print(f"\n[surr] training {cfg_name} …")
        model, test_df = train_mavenn(X_rand, y_rand, cfg, label=cfg_name)
        results[cfg_name] = {
            "model":   model,
            "test_df": test_df,
            "X_eval":  X_eval,
            "y_eval":  y_eval,
        }
    return results


# ---------------------------------------------------------------------------
# Step 4 — 1-row × 4-col y-vs-ŷ strip
# ---------------------------------------------------------------------------

def plot_new_row(results, out_path):
    cfg_names = list(SURROGATE_CONFIGS.keys())
    n_cols    = len(cfg_names)

    fig, axes = plt.subplots(1, n_cols, figsize=(4.2 * n_cols, 4.0))
    row_lo, row_hi = np.inf, -np.inf

    panel_data = []
    for c, cfg_name in enumerate(cfg_names):
        entry  = results[cfg_name]
        model  = entry["model"]
        df     = entry["test_df"]
        X_eval = entry["X_eval"]
        y_eval = entry["y_eval"].ravel()

        cols  = list(df.columns)
        x_col = "x" if "x" in cols else "X"
        y_col = "y" if "y" in cols else next(c2 for c2 in cols if c2.startswith("y"))

        X_rand_str   = np.asarray(df[x_col])
        y_rand_arr   = np.asarray(df[y_col], dtype=float).ravel()
        yhat_rand    = _predict_chunked(model, X_rand_str)
        yhat_eval    = _predict_chunked(model, _to_str(X_eval))

        m_r = np.isfinite(y_rand_arr) & np.isfinite(yhat_rand)
        m_e = np.isfinite(y_eval)     & np.isfinite(yhat_eval)

        rho_r = spearmanr(yhat_rand[m_r], y_rand_arr[m_r])[0] if m_r.sum() >= 3 else np.nan
        rho_e = spearmanr(yhat_eval[m_e], y_eval[m_e])[0]     if m_e.sum() >= 3 else np.nan
        r_r   = pearsonr( yhat_rand[m_r], y_rand_arr[m_r])[0] if m_r.sum() >= 3 else np.nan
        r_e   = pearsonr( yhat_eval[m_e], y_eval[m_e])[0]     if m_e.sum() >= 3 else np.nan
        r2_r  = r_r**2 if np.isfinite(r_r) else np.nan
        r2_e  = r_e**2 if np.isfinite(r_e) else np.nan

        all_vals = np.concatenate([yhat_rand[m_r], yhat_eval[m_e],
                                   y_rand_arr[m_r], y_eval[m_e]])
        if all_vals.size:
            row_lo = min(row_lo, float(all_vals.min()))
            row_hi = max(row_hi, float(all_vals.max()))

        panel_data.append((c, cfg_name,
                           yhat_rand, y_rand_arr, yhat_eval, y_eval,
                           rho_r, rho_e, r_r, r_e, r2_r, r2_e))

    for (c, cfg_name,
         yhat_rand, y_rand_arr, yhat_eval, y_eval,
         rho_r, rho_e, r_r, r_e, r2_r, r2_e) in panel_data:
        ax = axes[c]
        ax.scatter(yhat_rand, y_rand_arr, s=1, alpha=0.08, color="C0", rasterized=True,
                   label=f"random   ρ={rho_r:.2f}  r={r_r:.2f}  R²={r2_r:.2f}")
        ax.scatter(yhat_eval, y_eval,     s=1, alpha=0.08, color="C1", rasterized=True,
                   label=f"eval      ρ={rho_e:.2f}  r={r_e:.2f}  R²={r2_e:.2f}")
        ax.set_xlabel("ŷ", fontsize=8)
        ax.set_ylabel("y", fontsize=8)
        ax.tick_params(labelsize=6)
        ax.set_title(f"{CFG_LABELS[cfg_name]} / Surrogate GT", fontsize=7.5)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, markerscale=6, fontsize=6,
                  loc="upper left", framealpha=0.6)

        if np.isfinite(row_lo) and np.isfinite(row_hi) and row_lo < row_hi:
            pad  = (row_hi - row_lo) * 0.05
            lims = [row_lo - pad, row_hi + pad]
            ax.set_xlim(lims); ax.set_ylim(lims)
            ax.plot(lims, lims, "k--", linewidth=0.8, zorder=10)

    fig.suptitle(
        "y vs ŷ — Surrogate GT row\n"
        "blue = random MAVE test split    orange = uniform eval library",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] surrogate-GT row → {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Step 5 — stitch the new row onto the existing 16-panel
# ---------------------------------------------------------------------------

def stitch_panels(existing_path, new_row_path, out_path):
    from PIL import Image
    top = Image.open(existing_path).convert("RGB")
    bot = Image.open(new_row_path).convert("RGB")

    # Rescale new row to match existing width
    if top.width != bot.width:
        new_h = int(bot.height * top.width / bot.width)
        bot   = bot.resize((top.width, new_h), Image.LANCZOS)

    combined = Image.new("RGB", (top.width, top.height + bot.height), (255, 255, 255))
    combined.paste(top, (0, 0))
    combined.paste(bot, (0, top.height))
    combined.save(out_path, dpi=(150, 150))
    print(f"[stitch] combined panel → {out_path}")


# ---------------------------------------------------------------------------
# Step 6 — random vs eval distribution plot
# ---------------------------------------------------------------------------

def plot_distribution(y_rand, y_eval, out_path):
    y_rand = np.asarray(y_rand, dtype=float).ravel()
    y_eval = np.asarray(y_eval, dtype=float).ravel()
    y_rand = y_rand[np.isfinite(y_rand)]
    y_eval = y_eval[np.isfinite(y_eval)]

    all_v = np.concatenate([y_rand, y_eval])
    lo, hi = np.percentile(all_v, [0.5, 99.5])
    if lo == hi:
        lo -= 1e-6; hi += 1e-6
    bins = np.linspace(lo, hi, 121)

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
    print(f"[plot] distribution → {out_path}")


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
                        default="outputs/postabrcms/vts1high/surrogate_gt_row")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    seq = args.seq.upper().replace("T", "U")

    oh_seq               = rna_to_one_hot(seq)
    wt_onehot, _         = remove_padding(oh_seq)
    L                    = wt_onehot.shape[0]
    exact_mut_count      = min(4, max(1, L // 2))
    print(f"[init] L={L}  exact_mut_count={exact_mut_count}")

    # 1. Train SurrogateGT
    print("\n=== Step 1: Train SurrogateGT ===")
    sgt_model = build_surrogate_gt(wt_onehot, exact_mut_count, seed=args.seed)

    # 2. Score new library with SurrogateGT
    print("\n=== Step 2: Score library with SurrogateGT ===")
    X_rand, y_rand, X_eval, y_eval = score_with_sgt(
        sgt_model, wt_onehot, exact_mut_count, seed=args.seed + 100,
    )

    # 3. Train all 4 surrogates
    print("\n=== Step 3: Train 4 surrogate configs ===")
    results = train_all_surrogates(X_rand, y_rand, X_eval, y_eval)

    # 4. Plot new row
    print("\n=== Step 4: Plot new row ===")
    new_row_path = os.path.join(args.out_dir, "surrogate_gt_row.png")
    plot_new_row(results, new_row_path)

    # 5. Stitch
    print("\n=== Step 5: Stitch panels ===")
    stitched_path = os.path.join(args.out_dir, "y_vs_yhat_20panel.png")
    stitch_panels(args.existing_panel, new_row_path, stitched_path)

    # 6. Distribution
    print("\n=== Step 6: Distribution plot ===")
    dist_path = os.path.join(args.out_dir, "surrogate_gt_random_vs_eval.png")
    plot_distribution(y_rand, y_eval, dist_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
