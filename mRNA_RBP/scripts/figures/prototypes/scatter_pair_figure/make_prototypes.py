"""THROWAWAY PROTOTYPE: matched deepSQUID HuR/VTS1 prediction scatters."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[5]
OUT = Path(__file__).resolve().parents[5] / "mRNA_RBP" / "prototypes" / "scatter_pair_figure"
COLLECTIONS = ROOT / "mRNA_RBP/outputs/ground_truth_collections"
SOURCES = {
    "HuR": COLLECTIONS / "deepSQUID HuR/libraries_used_for_figures/scatter_by_mutcount_predictions_high.npz",
    "VTS1": COLLECTIONS / "deepSQUID VTS1/libraries_used_for_figures/scatter_by_mutcount_predictions_high.npz",
}
ORDER = ("HuR", "VTS1")
LIMITS = (-3.05, 2.05)


def load():
    data = {}
    for name, path in SOURCES.items():
        with np.load(path) as cache:
            data[name] = {
                "y": cache["y_activity_10"],
                "yhat": cache["yhat_activity_10"],
                "mutcount": cache["rate_labels_10"],
                "rho": float(cache["rho_activity_10"]),
            }
    return data


def base_axes(title):
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.45), sharex=True, sharey=True)
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.99)
    for ax, name in zip(axes, ORDER):
        ax.plot(LIMITS, LIMITS, color="#2f3437", lw=1.3, ls="--", zorder=1)
        ax.set(xlim=LIMITS, ylim=LIMITS, aspect="equal", title=name)
        ax.grid(color="#d9dddf", lw=0.55, alpha=0.7)
        ax.set_axisbelow(True)
        ax.set_xlabel("Oracle score")
    axes[0].set_ylabel("Surrogate prediction")
    return fig, axes


def finish(fig, filename, bottom=0.13):
    fig.text(
        0.5,
        0.025,
        "High-WT landscape · 10% mutation training library (n = 20,000) · activity-balanced evaluation",
        ha="center",
        fontsize=8.5,
        color="#555b5f",
    )
    fig.subplots_adjust(left=0.09, right=0.98, top=0.86, bottom=bottom, wspace=0.14)
    fig.savefig(OUT / filename, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def variant_a(data):
    fig, axes = base_axes("Matched prediction agreement")
    rng = np.random.default_rng(42)
    for ax, name in zip(axes, ORDER):
        d = data[name]
        take = rng.choice(len(d["y"]), size=min(8000, len(d["y"])), replace=False)
        ax.scatter(d["y"][take], d["yhat"][take], s=7, c="#2878a8", alpha=0.20, linewidths=0)
        ax.text(
            0.05, 0.94, rf"Spearman $\rho$ = {d['rho']:.3f}", transform=ax.transAxes,
            ha="left", va="top", fontsize=11, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#b9c0c4", alpha=0.92),
        )
    finish(fig, "variant_a_clean_pair.png")


def variant_b(data):
    fig, axes = base_axes("Matched prediction agreement")
    colors = {3: "#5e3c99", 5: "#3288bd", 7: "#66c2a5", 15: "#f46d43"}
    rng = np.random.default_rng(42)
    for ax, name in zip(axes, ORDER):
        d = data[name]
        for count in sorted(colors):
            indices = np.flatnonzero(d["mutcount"] == count)
            indices = rng.choice(indices, min(2000, len(indices)), replace=False)
            ax.scatter(d["y"][indices], d["yhat"][indices], s=7, color=colors[count], alpha=0.22, linewidths=0)
        ax.text(0.05, 0.94, rf"$\rho$ = {d['rho']:.3f}", transform=ax.transAxes,
                va="top", fontsize=11, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.9))
    handles = [Line2D([], [], marker="o", ls="", color=colors[k], label=str(k), markersize=5) for k in colors]
    fig.legend(handles=handles, title="Mutations per sequence", loc="upper center",
               bbox_to_anchor=(0.5, 0.91), ncol=4, frameon=False, fontsize=8)
    for ax in axes:
        ax.set_title(ax.get_title(), pad=35)
    finish(fig, "variant_b_mutcount_context.png")


def variant_c(data):
    fig, axes = base_axes("Prediction density under a matched evaluation")
    for ax, name in zip(axes, ORDER):
        d = data[name]
        ax.hexbin(d["y"], d["yhat"], gridsize=55, extent=(*LIMITS, *LIMITS),
                  mincnt=1, bins="log", cmap="Blues", linewidths=0, zorder=2)
        ax.plot(LIMITS, LIMITS, color="#202426", lw=1.4, ls="--", zorder=3)
        ax.set_title(f"{name}\n" + rf"Spearman $\rho$ = {d['rho']:.3f}", fontsize=11, fontweight="bold")
    fig.text(0.975, 0.49, "Darker hexagons = more sequences (log count)", rotation=90,
             va="center", ha="right", fontsize=8.5, color="#555b5f")
    finish(fig, "variant_c_density_pair.png")


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})
    data = load()
    variant_a(data)
    variant_b(data)
    variant_c(data)
    print("Wrote variant_a_clean_pair.png, variant_b_mutcount_context.png, variant_c_density_pair.png")


if __name__ == "__main__":
    main()
