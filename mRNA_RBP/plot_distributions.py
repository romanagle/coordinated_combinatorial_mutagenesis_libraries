"""
mRNA_RBP/plot_distributions.py

9 plots — one per (mutation_rate × library_size) combination.
Each plot overlays two distributions for a single GT score type:
  - Random library     (lib_{N}.npz pooled across instances)
  - Uniform eval       (eval_uniform_{gt_key}.npz pooled across instances,
                        upsampled from the 2M pool so the distribution is flat)

Output:
  <OUT_BASE>/plots/<GT_KEY>/mut{pct:02d}_lib{N}.png
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_HERE, "..", "scripts"))
sys.path.insert(0, os.path.join(_HERE, "..", "residualbind"))

from mRNA_RBP.generate_libraries import (
    OUT_BASE, N_INSTANCES, MUT_RATES_PCT, LIB_SIZES, GT_KEYS,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

GT_KEY  = "nonlin_additive_pairwise"   # score type to plot; change as needed
BINS    = 60
ALPHA   = 0.5
COLOR_RANDOM = "#4C9BE8"   # blue   — random library
COLOR_EVAL   = "#F5A623"   # orange — uniform eval

GT_TITLE = {
    "additive":                 "Additive GT",
    "additive_pairwise":        "Add+Pair GT",
    "nonlin_additive":          "Nonlin ∘ Additive GT",
    "nonlin_additive_pairwise": "Nonlin ∘ Add+Pair GT",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_scores(path: str, key: str) -> np.ndarray:
    d = np.load(path)
    return d[f"scores_{key}"].astype(np.float64)


def pool_instances(file_fn, key: str) -> np.ndarray:
    """Concatenate `key` scores across all instances using file_fn(k)."""
    parts = []
    for k in range(N_INSTANCES):
        path = file_fn(k)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Missing: {path}\nRun generate_libraries.py first."
            )
        parts.append(load_scores(path, key))
    return np.concatenate(parts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_plots(gt_key: str = GT_KEY):
    plots_dir = os.path.join(OUT_BASE, "plots", gt_key)
    os.makedirs(plots_dir, exist_ok=True)

    print(f"GT score: {gt_key}")
    print(f"Instances pooled: {N_INSTANCES}\n")

    for pct in MUT_RATES_PCT:
        for n_lib in LIB_SIZES:
            rand_scores = pool_instances(
                lambda k, p=pct, n=n_lib: os.path.join(
                    OUT_BASE, f"instance_{k:02d}", f"mut{p:02d}", f"lib_{n}.npz"
                ),
                gt_key,
            )
            eval_scores = pool_instances(
                lambda k, p=pct, n=n_lib: os.path.join(
                    OUT_BASE, f"instance_{k:02d}", f"mut{p:02d}",
                    f"eval_uniform_{gt_key}_lib{n}.npz",
                ),
                gt_key,
            )

            lo = min(rand_scores.min(), eval_scores.min())
            hi = max(rand_scores.max(), eval_scores.max())
            pad = (hi - lo) * 0.02
            bins = np.linspace(lo - pad, hi + pad, BINS + 1)

            fig, ax = plt.subplots(figsize=(8, 4.5))

            ax.hist(rand_scores, bins=bins, density=True, alpha=ALPHA,
                    color=COLOR_RANDOM,
                    label=f"Random library  (N={len(rand_scores):,})")
            ax.hist(eval_scores, bins=bins, density=True, alpha=ALPHA,
                    color=COLOR_EVAL,
                    label=f"Uniform eval library  (N={len(eval_scores):,})")

            ax.set_xlabel("GT score")
            ax.set_ylabel("Density")
            ax.legend(framealpha=0.9)
            ax.grid(True, linestyle="--", alpha=0.4)
            plt.tight_layout()

            fname = f"mut{pct:02d}_lib{n_lib}.png"
            out_path = os.path.join(plots_dir, fname)
            plt.savefig(out_path, dpi=150)
            plt.savefig(os.path.join(plots_dir, f"mut{pct:02d}_lib{n_lib}.svg"))
            plt.close(fig)
            print(f"  mut{pct:02d}%  lib{n_lib:>6,}  →  {out_path} / .svg")

    print("\nDone.")


if __name__ == "__main__":
    run_plots()
