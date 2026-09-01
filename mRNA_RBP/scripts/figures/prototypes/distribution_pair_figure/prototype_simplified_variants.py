"""THROWAWAY: three simplified paired-distribution figure variants.

Question: what is the smallest view that still communicates motif-versus-stem
activity distributions across the VTS1 and HuR landscapes?
"""

from pathlib import Path
import importlib.util

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parents[5] / "mRNA_RBP" / "prototypes" / "distribution_pair_figure"
SOURCE = Path(__file__).resolve().parent / "prototype_distribution_pair.py"
COLORS = {"stem": "#3B6FB6", "motif": "#D97A32"}
NAMES = {"VTS1": "VTS1", "HuR": "HuR"}


def load_data():
    spec = importlib.util.spec_from_file_location("paired_source", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load()


def limits(data, rates):
    groups = [
        data[name][rate][cat]
        for name in ("VTS1", "HuR")
        for rate in rates
        for cat in ("motif", "stem")
    ]
    lo = min(np.percentile(x, 0.2) for x in groups if len(x))
    hi = max(np.percentile(x, 99.8) for x in groups if len(x))
    return lo, hi


def ridge(ax, values, y, bins, color):
    hist, edges = np.histogram(values, bins=bins, density=True)
    x = (edges[:-1] + edges[1:]) / 2
    hist = hist / max(hist.max(), 1e-9) * 0.7
    ax.fill_between(x, y, y + hist, color=color, alpha=0.68, lw=0)
    ax.plot(x, y + hist, color=color, lw=1.2)


def finish_ridge_axis(ax):
    ax.axvline(0, color="#222", ls=":", lw=1.2)
    ax.grid(axis="x", color="#E6E6E6", lw=0.7)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("ResidualBind score relative to WT")


def variant_a(data):
    """One representative mutation rate; four total distributions."""
    rate = "10%"
    lo, hi = limits(data, (rate,))
    bins = np.linspace(lo, hi, 70)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5), sharex=True, sharey=True)
    for ax, name in zip(axes, ("VTS1", "HuR")):
        ridge(ax, data[name][rate]["motif"], 1, bins, COLORS["motif"])
        ridge(ax, data[name][rate]["stem"], 0, bins, COLORS["stem"])
        ax.set_yticks([0, 1], ["Stem hit only", "Motif hit only"])
        ax.set_title(NAMES[name], fontweight="bold")
        finish_ridge_axis(ax)
    fig.suptitle("Region-conditioned activity at 10% mutation rate", fontsize=13)
    fig.text(0.5, 0.01, "Both/neither omitted  •  Dotted line: WT = 0", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 0.92))
    fig.savefig(HERE / "simplified_a_10pct_focus.png", dpi=180)
    plt.close(fig)


def variant_b(data):
    """Mutation-rate endpoints; removes the redundant middle condition."""
    rates = ("5%", "25%")
    lo, hi = limits(data, rates)
    bins = np.linspace(lo, hi, 70)
    rows = [(rate, cat) for rate in rates for cat in ("motif", "stem")]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.3), sharex=True, sharey=True)
    for ax, name in zip(axes, ("VTS1", "HuR")):
        for idx, (rate, cat) in enumerate(rows):
            ridge(ax, data[name][rate][cat], len(rows) - 1 - idx, bins, COLORS[cat])
        ax.set_yticks(
            range(len(rows)),
            [f"{rate}  {cat.title()}" for rate, cat in rows[::-1]],
        )
        ax.set_title(NAMES[name], fontweight="bold")
        finish_ridge_axis(ax)
    fig.suptitle("Low- versus high-mutation libraries", fontsize=13)
    fig.text(0.5, 0.01, "Both/neither omitted  •  Dotted line: WT = 0", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 0.92))
    fig.savefig(HERE / "simplified_b_rate_endpoints.png", dpi=180)
    plt.close(fig)


def variant_c(data):
    """All rates retained as compact median and interquartile summaries."""
    rates = ("5%", "10%", "25%")
    lo, hi = limits(data, rates)
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.1), sharex=True, sharey=True)
    y_positions = np.arange(len(rates))[::-1]
    offsets = {"motif": 0.13, "stem": -0.13}
    for ax, name in zip(axes, ("VTS1", "HuR")):
        for cat in ("motif", "stem"):
            for y, rate in zip(y_positions, rates):
                q25, q50, q75 = np.quantile(data[name][rate][cat], [0.25, 0.5, 0.75])
                yy = y + offsets[cat]
                ax.plot([q25, q75], [yy, yy], color=COLORS[cat], lw=5, solid_capstyle="butt")
                ax.scatter(q50, yy, color="white", edgecolor=COLORS[cat], lw=1.5, s=28, zorder=3)
        ax.axvline(0, color="#222", ls=":", lw=1.2)
        ax.set_xlim(lo, hi)
        ax.set_yticks(y_positions, rates)
        ax.set_xlabel("ResidualBind score relative to WT")
        ax.set_title(NAMES[name], fontweight="bold")
        ax.grid(axis="x", color="#E6E6E6", lw=0.7)
        ax.spines[["top", "right", "left"]].set_visible(False)
    handles = [
        plt.Line2D([0], [0], color=COLORS["motif"], lw=5, label="Motif hit only"),
        plt.Line2D([0], [0], color=COLORS["stem"], lw=5, label="Stem hit only"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False)
    fig.suptitle("Region-conditioned activity summaries", fontsize=13)
    fig.text(0.5, 0.08, "Bars: interquartile range  •  Circles: median  •  Dotted line: WT = 0", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.15, 1, 0.92))
    fig.savefig(HERE / "simplified_c_quantile_summary.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    data = load_data()
    variant_a(data)
    variant_b(data)
    variant_c(data)
    print("Wrote three THROWAWAY simplified variants and review page to", HERE)
