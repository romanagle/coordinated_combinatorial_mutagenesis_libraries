"""PROTOTYPE / THROWAWAY: three paired distribution layouts from real caches."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
CACHES = {
    "VTS1": ROOT / "mRNA_RBP/outputs/ground_truth_collections/deepSQUID VTS1/libraries_used_for_figures/random_region_score_cache/deepsquid_vts1_natural_random_library_scores.npz",
    "HuR": ROOT / "mRNA_RBP/outputs/ground_truth_collections/deepSQUID HuR/libraries_used_for_figures/random_region_score_cache/deepsquid_hur_natural_random_library_scores.npz",
}
RATES = ((5, "5%"), (10, "10%"), (25, "25%"))
COLORS = {"stem": "#3B6FB6", "motif": "#D97A32"}
NUC = {"A": 0, "C": 1, "G": 2, "U": 3}


def load():
    data = {}
    for name, path in CACHES.items():
        z = np.load(path)
        wt = np.array([NUC[c] for c in str(z["wt_seq"].item())])
        stem = np.unique(z["stem_pairs"].astype(int))
        motif = z["motif_positions"].astype(int)
        rates = {}
        for pct, label in RATES:
            nids = z[f"rand{pct:02d}_nids"]
            score = z[f"rand{pct:02d}_delta_scores"].astype(float)
            stem_hit = (nids[:, stem] != wt[stem]).any(axis=1)
            motif_hit = (nids[:, motif] != wt[motif]).any(axis=1)
            rates[label] = {
                "all": score,
                "stem": score[stem_hit & ~motif_hit],
                "motif": score[motif_hit],
            }
        data[name] = rates
    return data


def style(ax):
    ax.axvline(0, color="#222222", ls=":", lw=1.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#E6E6E6", lw=.7)


def variant_a(data):
    """Two-column ridgelines; category separation without overlapping fills."""
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.2), sharex=True, sharey=True)
    bins = np.linspace(-3, 2.1, 75)
    rows = [(r, c) for r in ("5%", "10%", "25%") for c in ("motif", "stem")]
    for ax, name in zip(axes, ("VTS1", "HuR")):
        for idx, (rate, cat) in enumerate(rows):
            vals = data[name][rate][cat]
            hist, edges = np.histogram(vals, bins=bins, density=True)
            x = (edges[:-1] + edges[1:]) / 2
            hist = hist / max(hist.max(), 1e-9) * .72
            y = len(rows) - 1 - idx
            ax.fill_between(x, y, y + hist, color=COLORS[cat], alpha=.68, lw=0)
            ax.plot(x, y + hist, color=COLORS[cat], lw=1)
        ax.set_title(name, fontweight="bold")
        ax.set_yticks(range(len(rows)), [f"{r}  {c.title()}" for r, c in rows[::-1]])
        ax.set_xlabel("deepSQUID score relative to WT")
        style(ax)
    fig.suptitle("A — Paired ridgelines (region classes separated)", fontsize=13)
    fig.text(.5, .015, "Dotted: WT = 0  •  ‘Neither’ omitted  •  VTS1 motif/stem position sets overlap", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .045, 1, .94))
    fig.savefig(OUT / "variant_a_paired_ridgelines.png", dpi=180)
    plt.close(fig)


def variant_b(data):
    """ECDF grid for the complete library; comparison prioritizes skew/shape."""
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharex=True, sharey=True)
    colors = {"VTS1": "#6953A6", "HuR": "#2A9D8F"}
    for ax, (_, rate) in zip(axes, RATES):
        for name in ("VTS1", "HuR"):
            x = np.sort(data[name][rate]["all"])
            y = np.arange(1, len(x) + 1) / len(x)
            ax.plot(x, y, lw=2.2, color=colors[name], label=name)
        ax.set_title(f"{rate} mutation rate")
        ax.set_xlabel("score relative to WT")
        style(ax)
    axes[0].set_ylabel("Cumulative fraction")
    axes[-1].legend(frameon=False)
    fig.suptitle("B — Complete-library cumulative distributions", fontsize=13)
    fig.text(.5, .015, "Each curve includes every random-library sequence; dotted line is WT = 0", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, .92))
    fig.savefig(OUT / "variant_b_complete_library_ecdf.png", dpi=180)
    plt.close(fig)


def variant_c(data):
    """Compact quantile bands: distribution summary without choosing means."""
    fig, ax = plt.subplots(figsize=(9.8, 4.8))
    colors = {"VTS1": "#6953A6", "HuR": "#2A9D8F"}
    positions, labels = [], []
    y = 0
    for _, rate in RATES:
        for name in ("VTS1", "HuR"):
            vals = data[name][rate]["all"]
            q05, q25, q50, q75, q95 = np.quantile(vals, [.05, .25, .5, .75, .95])
            ax.plot([q05, q95], [y, y], color=colors[name], lw=2, alpha=.45)
            ax.plot([q25, q75], [y, y], color=colors[name], lw=10, solid_capstyle="butt")
            ax.scatter(q50, y, s=35, color="white", edgecolor=colors[name], lw=1.7, zorder=3)
            positions.append(y); labels.append(f"{rate}   {name}")
            y += 1
        y += .45
    ax.set_yticks(positions, labels)
    ax.set_xlabel("deepSQUID score relative to WT")
    ax.axvline(0, color="#222", ls=":", lw=1.2)
    ax.grid(axis="x", color="#E6E6E6", lw=.7)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.invert_yaxis()
    ax.set_title("C — Complete-library quantile intervals", fontsize=13)
    fig.text(.5, .02, "Thick: middle 50%  •  Thin: 5th–95th percentile  •  Circle: median  •  Dotted: WT", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .06, 1, 1))
    fig.savefig(OUT / "variant_c_quantile_intervals.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    d = load()
    variant_a(d)
    variant_b(d)
    variant_c(d)
    print("Wrote three THROWAWAY prototype PNGs to", OUT)
