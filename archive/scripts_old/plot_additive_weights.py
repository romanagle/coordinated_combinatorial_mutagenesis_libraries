"""plot_additive_weights.py

Visualise additive ground-truth weights as a 4×L heat map.

Colour scheme:
  - Red   → high positive weight
  - Blue  → high negative weight
  - White → zero
  - Yellow box outline → wildtype nucleotide at each position

Usage:
    python scripts/plot_additive_weights.py --seq AUGCCUAGAAGUGUGUGAUCGCAUU
    python scripts/plot_additive_weights.py --seq AUGCCUAGAA... --seed 42 --out weights.png
"""

import argparse
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, '/home/nagle/final_version/squid-nn')
sys.path.append('/home/nagle/final_version/squid-manuscript/squid')
sys.path.append('/home/nagle/final_version/residualbind')

from seq_utils import rna_to_one_hot, remove_padding
from ground_truth import init_additive_noWT

NUCS = ['A', 'C', 'G', 'U']


def build_full_weight_matrix(W_mut, mut_map, wt_onehot):
    """Reconstruct a (4, L) weight matrix from the noWT parameterisation.

    WT nucleotide at each position gets weight 0.
    Non-WT nucleotides get their W_mut values placed at the correct row.

    Returns:
        W_full : (4, L) float array
        wt_idx : (L,)  int array — which nucleotide (0-3) is WT at each position
    """
    L = wt_onehot.shape[0]
    wt_idx = np.argmax(wt_onehot, axis=1)   # (L,)
    W_full = np.zeros((4, L), dtype=float)
    for i in range(L):
        for k in range(3):
            nuc = mut_map[i, k]          # nucleotide index (0-3)
            W_full[nuc, i] = W_mut[i, k]
    return W_full, wt_idx


def plot_weight_heatmap(W_full, wt_idx, seq, out_path):
    """Plot a 4×L heatmap with WT positions outlined in yellow."""
    L = W_full.shape[1]

    # Symmetric colour scale
    vmax = np.abs(W_full).max()
    vmax = vmax if vmax > 0 else 1.0

    # Width scales with sequence length; cap so it doesn't get absurdly wide
    fig_w = max(8, min(L * 0.35, 30))
    fig, ax = plt.subplots(figsize=(fig_w, 3.2))

    im = ax.imshow(
        W_full,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-vmax, vmax=vmax,
        interpolation="nearest",
    )

    # Yellow outline on WT cell
    for i, wt in enumerate(wt_idx):
        rect = patches.Rectangle(
            (i - 0.5, wt - 0.5), 1, 1,
            linewidth=2.0, edgecolor="gold", facecolor="yellow", alpha=0.55,
            zorder=3,
        )
        ax.add_patch(rect)

    ax.set_yticks(range(4))
    ax.set_yticklabels(NUCS, fontsize=11)

    # x-axis: show position number + WT nucleotide every few positions
    step = max(1, L // 20)
    xtick_pos   = list(range(0, L, step))
    xtick_labels = [f"{i}\n{seq[i]}" for i in xtick_pos]
    ax.set_xticks(xtick_pos)
    ax.set_xticklabels(xtick_labels, fontsize=7)

    ax.set_xlabel("Position", fontsize=11)
    ax.set_ylabel("Nucleotide", fontsize=11)
    ax.set_title(
        f"Additive weights  (L={L})   yellow = wildtype",
        fontsize=11,
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label("Weight", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved → {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="4×L heatmap of additive ground-truth weights"
    )
    parser.add_argument("--seq",  required=True, help="RNA sequence (A/C/G/U)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sigma", type=float, default=0.5,
                        help="Std dev for weight initialisation (default 0.5)")
    parser.add_argument("--out",  type=str, default=None,
                        help="Output PNG path (default: additive_weights_<L>.png)")
    args = parser.parse_args()

    seq = args.seq.upper().replace("T", "U")
    oh  = rna_to_one_hot(seq)
    wt_onehot, _ = remove_padding(oh)
    L = wt_onehot.shape[0]

    rng = np.random.default_rng(args.seed)
    W_mut, mut_map, _ = init_additive_noWT(rng, wt_onehot, sigma=args.sigma)

    W_full, wt_idx = build_full_weight_matrix(W_mut, mut_map, wt_onehot)

    out_path = args.out or f"additive_weights_L{L}.png"
    plot_weight_heatmap(W_full, wt_idx, seq, out_path)


if __name__ == "__main__":
    main()
