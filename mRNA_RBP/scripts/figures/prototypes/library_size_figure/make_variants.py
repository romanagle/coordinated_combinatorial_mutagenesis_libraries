#!/usr/bin/env python3
"""THROWAWAY PROTOTYPE: generate three central library-size figure variants."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize


HERE = Path(__file__).resolve().parents[5] / "mRNA_RBP" / "prototypes" / "library_size_figure"
ROOT = HERE.parents[1]
OUT = HERE / "outputs"
SIZES = [200, 2_000, 20_000]
SYSTEMS = [
    (
        "Synthetic GT",
        ROOT / "outputs/ground_truth_collections/Synthetic GT/libraries_used_for_figures/lib_size_spearman_results_type3.json",
    ),
    (
        "VTS1",
        ROOT / "outputs/ground_truth_collections/deepSQUID VTS1/libraries_used_for_figures/lib_size_spearman_results_high.json",
    ),
    (
        "HuR",
        ROOT / "outputs/ground_truth_collections/deepSQUID HuR/libraries_used_for_figures/lib_size_spearman_results_high.json",
    ),
]
TRIPTYCH_ORDER = ["HuR", "Synthetic GT", "VTS1"]
COLORS = {"Random holdout": "#3166A8", "Uniform evaluation": "#E27A3F"}


def load_data():
    data = {}
    for name, path in SYSTEMS:
        cache = json.loads(path.read_text())
        oracle_block = next(iter(cache["surrogate"].values()))
        model = oracle_block["nonlinear additive + pairwise"]["10"]
        data[name] = {
            "Random holdout": [np.asarray(model[str(n)]["rand"], float) for n in SIZES],
            "Uniform evaluation": [np.asarray(model[str(n)]["type2"], float) for n in SIZES],
        }
    return data


def base_style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "figure.facecolor": "white",
        }
    )


def variant_a(data):
    """Classic small multiples: trends and uncertainty are primary."""
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.7), sharex=True, sharey=True)
    x = np.arange(3)
    for ax, system in zip(axes, TRIPTYCH_ORDER):
        for label, values in data[system].items():
            means = np.array([v.mean() for v in values])
            sds = np.array([v.std(ddof=1) if len(v) > 1 else np.nan for v in values])
            ax.plot(x, means, color=COLORS[label], marker="o", lw=2.2, ms=6, label=label)
            if np.isfinite(sds).any():
                ax.fill_between(x, means - sds, means + sds, color=COLORS[label], alpha=0.13, linewidth=0)
        ax.set_title(system)
        ax.set_xticks(x, ["200", "2K", "20K"])
        ax.set_ylim(0, 1.04)
        ax.grid(axis="y", color="#D8D8D8", lw=0.7, alpha=0.75)
        ax.set_xlabel("Training library size")
    axes[0].set_ylabel("Spearman ρ")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle("Evaluation regime changes the apparent recovery of model behavior", y=1.10, fontsize=13, fontweight="bold")
    fig.text(0.5, -0.02, "Nonlinear additive + pairwise surrogate · 10% mutation rate", ha="center", color="#555555")
    fig.tight_layout()
    fig.savefig(OUT / "variant_a_triptych.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def variant_b(data):
    """Paired slope display: the random–uniform gap is primary."""
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 4.5), sharey=True)
    for ax, size in zip(axes, SIZES):
        endpoints = []
        for row, (system, _) in enumerate(SYSTEMS):
            rand = data[system]["Random holdout"][SIZES.index(size)].mean()
            unif = data[system]["Uniform evaluation"][SIZES.index(size)].mean()
            ax.plot([0, 1], [rand, unif], color="#A8A8A8", lw=2, zorder=1)
            ax.scatter(0, rand, s=60, color=COLORS["Random holdout"], zorder=2)
            ax.scatter(1, unif, s=60, color=COLORS["Uniform evaluation"], zorder=2)
            endpoints.append((system, rand, unif))
        # Labels are offset in screen space where endpoints coincide; marks stay exact.
        for side, value_index, color, ha, dx in [(0, 1, COLORS["Random holdout"], "right", -9),
                                                  (1, 2, COLORS["Uniform evaluation"], "left", 9)]:
            ranked = sorted(endpoints, key=lambda item: item[value_index])
            for rank, item in enumerate(ranked):
                dy = (rank - 1) * 11 if max(x[value_index] for x in ranked) - min(x[value_index] for x in ranked) < 0.06 else 0
                ax.annotate(f"{item[value_index]:.2f}", (side, item[value_index]), xytext=(dx, dy),
                            textcoords="offset points", ha=ha, va="center", color=color, fontsize=9)
        ax.set_title(f"{size:,} sequences")
        ax.set_xlim(-0.35, 1.35)
        ax.set_xticks([0, 1], ["Random\nholdout", "Uniform\nevaluation"])
        ax.set_ylim(0, 1.04)
        ax.grid(axis="y", color="#E0E0E0", lw=0.7)
    # System identity is encoded by one slope per system; label at midpoint.
    for ax, size in zip(axes, SIZES):
        idx = SIZES.index(size)
        midpoints = []
        for system, _ in SYSTEMS:
            r = data[system]["Random holdout"][idx].mean()
            u = data[system]["Uniform evaluation"][idx].mean()
            midpoints.append((system, (r + u) / 2))
        ranked = sorted(midpoints, key=lambda item: item[1])
        for rank, (system, midpoint) in enumerate(ranked):
            dy = (rank - 1) * 12 if ranked[-1][1] - ranked[0][1] < 0.08 else 0
            ax.annotate(system, (0.5, midpoint), xytext=(0, dy), textcoords="offset points",
                        ha="center", va="center", fontsize=8,
                        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1})
    axes[0].set_ylabel("Spearman ρ")
    fig.suptitle("The evaluation gap at each training-library size", fontsize=13, fontweight="bold", y=1.01)
    fig.text(0.5, 0.01, "Nonlinear additive + pairwise surrogate · 10% mutation rate", ha="center", color="#555555")
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(OUT / "variant_b_evaluation_gaps.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def variant_c(data):
    """Heatmap table: compact lookup and across-system comparison are primary."""
    rows = []
    row_labels = []
    for system, _ in SYSTEMS:
        for regime in ["Random holdout", "Uniform evaluation"]:
            rows.append([v.mean() for v in data[system][regime]])
            row_labels.append(f"{system}  |  {regime}")
    matrix = np.asarray(rows)
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    im = ax.imshow(matrix, cmap="viridis", norm=Normalize(0, 1), aspect="auto")
    ax.set_xticks(range(3), ["200", "2K", "20K"])
    ax.set_yticks(range(6), row_labels)
    ax.set_xlabel("Training library size")
    ax.set_title("Spearman ρ across systems and evaluation regimes", fontsize=13, pad=16)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            color = "white" if matrix[i, j] < 0.55 else "#111111"
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color=color, fontweight="bold")
    for y in [1.5, 3.5]:
        ax.axhline(y, color="white", lw=4)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    cbar.set_label("Spearman ρ")
    fig.text(0.5, 0.01, "Nonlinear additive + pairwise surrogate · 10% mutation rate", ha="center", color="#555555")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT / "variant_c_heatmap.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def variant_d(data):
    """Triptych with the 20K random-minus-uniform Spearman gap annotated."""
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.7), sharex=True, sharey=True)
    x = np.arange(3)
    for ax, system in zip(axes, TRIPTYCH_ORDER):
        means_by_label = {}
        for label, values in data[system].items():
            means = np.array([v.mean() for v in values])
            means_by_label[label] = means
            sds = np.array([v.std(ddof=1) if len(v) > 1 else np.nan for v in values])
            ax.plot(x, means, color=COLORS[label], marker="o", lw=2.2, ms=6, label=label)
            if np.isfinite(sds).any():
                ax.fill_between(x, means - sds, means + sds, color=COLORS[label], alpha=0.13, linewidth=0)
        delta = means_by_label["Random holdout"][-1] - means_by_label["Uniform evaluation"][-1]
        ax.annotate(
            f"20K Δρ = {delta:.2f}",
            xy=(2, means_by_label["Uniform evaluation"][-1]),
            xytext=(-8, -24),
            textcoords="offset points",
            ha="right",
            va="top",
            fontsize=9,
            fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "#BBBBBB", "boxstyle": "round,pad=0.25"},
        )
        ax.set_title(system)
        ax.set_xticks(x, ["200", "2K", "20K"])
        ax.set_ylim(0, 1.04)
        ax.grid(axis="y", color="#D8D8D8", lw=0.7, alpha=0.75)
        ax.set_xlabel("Training library size")
    axes[0].set_ylabel("Spearman ρ")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle("Evaluation regime changes the apparent recovery of model behavior", y=1.10, fontsize=13, fontweight="bold")
    fig.text(0.5, -0.02, "Δρ = random holdout − uniform evaluation · nonlinear additive + pairwise · 10% mutation rate", ha="center", color="#555555")
    fig.tight_layout()
    fig.savefig(OUT / "variant_d_triptych_annotated_delta.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def variant_e(data):
    """Compact table of the 20K means and random-minus-uniform gap."""
    rows = []
    for system in TRIPTYCH_ORDER:
        random_mean = data[system]["Random holdout"][-1].mean()
        uniform_mean = data[system]["Uniform evaluation"][-1].mean()
        rows.append([system, f"{random_mean:.2f}", f"{uniform_mean:.2f}", f"{random_mean - uniform_mean:.2f}"])
    fig, ax = plt.subplots(figsize=(7.2, 2.5))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=["Landscape", "Random holdout ρ", "Uniform evaluation ρ", "Δρ"],
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.55)
    for col in range(4):
        table[(0, col)].set_facecolor("#EAEAEA")
        table[(0, col)].set_text_props(fontweight="bold")
    ax.set_title("Evaluation gap at 20K training variants", fontsize=13, fontweight="bold", pad=14)
    fig.text(0.5, 0.03, "Δρ = random holdout − uniform evaluation · means across initializations", ha="center", color="#555555")
    fig.tight_layout()
    fig.savefig(OUT / "variant_e_20k_delta_table.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def variant_f(data):
    """Biological application only: HuR negative control and VTS1."""
    biological_order = ["HuR", "VTS1"]
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.8), sharex=True, sharey=True)
    x = np.arange(3)
    for ax, system in zip(axes, biological_order):
        for label, values in data[system].items():
            means = np.array([v.mean() for v in values])
            sds = np.array([v.std(ddof=1) if len(v) > 1 else np.nan for v in values])
            ax.plot(x, means, color=COLORS[label], marker="o", lw=2.2, ms=6, label=label)
            if np.isfinite(sds).any():
                ax.fill_between(x, means - sds, means + sds, color=COLORS[label], alpha=0.13, linewidth=0)
        ax.set_title(system)
        ax.set_xticks(x, ["200", "2K", "20K"])
        ax.set_ylim(0, 1.04)
        ax.grid(axis="y", color="#D8D8D8", lw=0.7, alpha=0.75)
        ax.set_xlabel("Training library size")
    axes[0].set_ylabel("Spearman ρ")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle("Biological landscapes recreate evaluation-dependent accuracy inflation", y=1.10, fontsize=13, fontweight="bold")
    fig.text(0.5, -0.02, "Nonlinear additive + pairwise surrogate · 10% mutation rate", ha="center", color="#555555")
    fig.tight_layout()
    fig.savefig(OUT / "variant_f_biological_pair.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def variant_g(data):
    """Biological pair with the uniform-evaluation saturated SSM baseline."""
    biological_order = ["HuR", "VTS1"]
    path_by_system = {name: path for name, path in SYSTEMS}
    ssm_baselines = {}
    for system in biological_order:
        cache = json.loads(path_by_system[system].read_text())
        instance_zero = cache["saturated"]["0"]
        oracle_block = next(iter(instance_zero.values()))
        ssm_baselines[system] = float(oracle_block["type2"])

    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.8), sharex=True, sharey=True)
    x = np.arange(3)
    for ax, system in zip(axes, biological_order):
        for label, values in data[system].items():
            means = np.array([v.mean() for v in values])
            sds = np.array([v.std(ddof=1) if len(v) > 1 else np.nan for v in values])
            ax.plot(x, means, color=COLORS[label], marker="o", lw=2.2, ms=6, label=label)
            if np.isfinite(sds).any():
                ax.fill_between(x, means - sds, means + sds, color=COLORS[label], alpha=0.13, linewidth=0)
        baseline = ssm_baselines[system]
        ax.axhline(baseline, color="#39845A", linestyle="--", linewidth=2.0, label="SSM baseline")
        ax.text(0.03, baseline + 0.025, f"SSM ρ = {baseline:.2f}", color="#286A46", fontsize=9, fontweight="bold")
        ax.set_title(system)
        ax.set_xticks(x, ["200", "2K", "20K"])
        ax.set_ylim(0, 1.04)
        ax.grid(axis="y", color="#D8D8D8", lw=0.7, alpha=0.75)
        ax.set_xlabel("Training library size")
    axes[0].set_ylabel("Spearman ρ")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle("Biological landscapes recreate evaluation-dependent accuracy inflation", y=1.10, fontsize=13, fontweight="bold")
    fig.text(0.5, -0.02, "SSM baseline uses uniform evaluation · nonlinear additive + pairwise surrogate at 10% mutation rate", ha="center", color="#555555")
    fig.tight_layout()
    fig.savefig(OUT / "variant_g_biological_pair_with_ssm.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    base_style()
    data = load_data()
    variant_a(data)
    variant_b(data)
    variant_c(data)
    variant_d(data)
    variant_e(data)
    variant_f(data)
    variant_g(data)
    print("Wrote:")
    for path in sorted(OUT.glob("variant_*.png")):
        print(path)


if __name__ == "__main__":
    main()
