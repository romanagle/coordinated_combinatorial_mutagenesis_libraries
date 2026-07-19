"""
Plot a direct Type3 comparison between the synthetic mRNA/RBP ground truth and
the ResidualBind ensemble oracle.

This script intentionally consumes the values from the completed Type3 runs
instead of retraining the surrogate models. Re-run bar_surrogate_models_type3.py
first if those values need to be refreshed.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(_HERE, "outputs", "notebook_plots")

COLOR_SYNTHETIC = "#4C72B0"
COLOR_RESIDUALBIND = "#C44E52"

LIB_LABELS = [
    ("random", "Random holdout"),
    ("type2", "Activity-balanced"),
    ("pairwise", "Pairwise eval"),
    ("type3", "Exhaustive eval"),
]

MODEL_LABELS = [
    "SSM",
    "Additive",
    "Additive\n+Pairwise",
    "NL Additive\nonly",
    "NL Additive\n+Pairwise",
]

SYNTHETIC = {
    "SSM": {"random": 0.941, "type2": 0.945, "pairwise": 0.000, "type3": 0.978},
    "Additive": {"random": 0.946, "type2": 0.457, "pairwise": -0.486, "type3": 0.938},
    "Additive\n+Pairwise": {"random": 0.970, "type2": 0.233, "pairwise": 0.544, "type3": 0.892},
    "NL Additive\nonly": {"random": 0.941, "type2": 0.503, "pairwise": -0.554, "type3": 0.964},
    "NL Additive\n+Pairwise": {"random": 0.988, "type2": 0.448, "pairwise": 0.704, "type3": 0.961},
}

RESIDUALBIND = {
    "SSM": {"random": 0.883, "type2": 0.861, "pairwise": 0.990, "type3": 0.966},
    "Additive": {"random": 0.886, "type2": 0.853, "pairwise": 0.931, "type3": 0.943},
    "Additive\n+Pairwise": {"random": 0.937, "type2": 0.884, "pairwise": 0.833, "type3": 0.916},
    "NL Additive\nonly": {"random": 0.884, "type2": 0.838, "pairwise": 0.953, "type3": 0.965},
    "NL Additive\n+Pairwise": {"random": 0.966, "type2": 0.851, "pairwise": 0.862, "type3": 0.964},
}

SYNTHETIC_RMSE = {
    "SSM": {"random": 0.128, "type2": 0.084, "pairwise": 0.319, "type3": 0.052},
    "Additive": {"random": 0.100, "type2": 0.194, "pairwise": 0.349, "type3": 0.261},
    "Additive\n+Pairwise": {"random": 0.060, "type2": 0.326, "pairwise": 0.568, "type3": 0.434},
    "NL Additive\nonly": {"random": 0.082, "type2": 0.190, "pairwise": 0.355, "type3": 0.253},
    "NL Additive\n+Pairwise": {"random": 0.042, "type2": 0.155, "pairwise": 0.273, "type3": 0.228},
}

RESIDUALBIND_RMSE = {
    "SSM": {"random": 0.153, "type2": 0.172, "pairwise": 0.019, "type3": 0.062},
    "Additive": {"random": 0.141, "type2": 0.165, "pairwise": 0.055, "type3": 0.076},
    "Additive\n+Pairwise": {"random": 0.088, "type2": 0.131, "pairwise": 0.089, "type3": 0.087},
    "NL Additive\nonly": {"random": 0.145, "type2": 0.174, "pairwise": 0.098, "type3": 0.106},
    "NL Additive\n+Pairwise": {"random": 0.083, "type2": 0.106, "pairwise": 0.054, "type3": 0.050},
}


def _annotate(ax, bars):
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.018 if h >= 0 else h - 0.04,
            f"{h:.2f}",
            ha="center",
            va="bottom" if h >= 0 else "top",
            fontsize=8,
        )


def _save_focused(values_synthetic=SYNTHETIC, values_residualbind=RESIDUALBIND,
                  ylabel="Spearman rho",
                  title="Nonlinear additive+pairwise surrogate by oracle",
                  out_stem="type3_oracle_comparison_nonlinear_additive_pairwise",
                  ylim=(-0.12, 1.12), save_svg=False):
    model = "NL Additive\n+Pairwise"
    x = np.arange(len(LIB_LABELS))
    width = 0.34
    synthetic_vals = [values_synthetic[model][key] for key, _ in LIB_LABELS]
    residualbind_vals = [values_residualbind[model][key] for key, _ in LIB_LABELS]

    fig, ax = plt.subplots(figsize=(7.5, 4.7))
    fig.subplots_adjust(bottom=0.22)
    bars_synth = ax.bar(
        x - width / 2, synthetic_vals, width,
        color=COLOR_SYNTHETIC, label="Synthetic ground truth",
    )
    bars_resbind = ax.bar(
        x + width / 2, residualbind_vals, width,
        color=COLOR_RESIDUALBIND, label="ResidualBind ensemble oracle",
    )
    _annotate(ax, bars_synth)
    _annotate(ax, bars_resbind)

    ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in LIB_LABELS], fontsize=9)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_ylim(*ylim)
    ax.set_title(title, fontsize=12, pad=8)
    ax.legend(fontsize=9, framealpha=0.9)

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, out_stem + ".png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(out)
    if save_svg:
        svg_out = out.replace(".png", ".svg")
        fig.savefig(svg_out, bbox_inches="tight")
        print(svg_out)
    plt.close(fig)


def _save_all_models(values_synthetic=SYNTHETIC, values_residualbind=RESIDUALBIND,
                     ylabel="Spearman rho",
                     out_stem="type3_oracle_comparison_all_models",
                     ylim=(-0.62, 1.12), save_svg=False):
    x = np.arange(len(MODEL_LABELS))
    width = 0.18
    offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]
    colors = ["#8C8C8C", "#4C72B0", "#DD8452", "#55A868"]

    fig, axes = plt.subplots(2, 1, figsize=(11.5, 7.5), sharex=True, sharey=True)
    for ax, title, values in [
        (axes[0], "Synthetic ground truth", values_synthetic),
        (axes[1], "ResidualBind ensemble oracle", values_residualbind),
    ]:
        for offset, color, (key, label) in zip(offsets, colors, LIB_LABELS):
            ys = [values[model][key] for model in MODEL_LABELS]
            ax.bar(x + offset, ys, width, color=color, label=label)
        ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_ylim(*ylim)
        ax.set_title(title, fontsize=11, pad=6)
        ax.tick_params(axis="y", labelsize=9)

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(MODEL_LABELS, fontsize=10)
    axes[0].legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.08), fontsize=9)

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, out_stem + ".png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(out)
    if save_svg:
        svg_out = out.replace(".png", ".svg")
        fig.savefig(svg_out, bbox_inches="tight")
        print(svg_out)
    plt.close(fig)


def main(save_svg=False):
    _save_focused(save_svg=save_svg)
    _save_all_models(save_svg=save_svg)
    _save_focused(
        SYNTHETIC_RMSE,
        RESIDUALBIND_RMSE,
        ylabel="RMSE",
        title="Nonlinear additive+pairwise surrogate RMSE by oracle",
        out_stem="type3_oracle_comparison_nonlinear_additive_pairwise_rmse",
        ylim=(0.0, 0.31),
        save_svg=save_svg,
    )
    _save_all_models(
        SYNTHETIC_RMSE,
        RESIDUALBIND_RMSE,
        ylabel="RMSE",
        out_stem="type3_oracle_comparison_all_models_rmse",
        ylim=(0.0, 0.62),
        save_svg=save_svg,
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--svg", action="store_true",
                        help="Also save an .svg copy of each figure")
    args = parser.parse_args()
    main(save_svg=args.svg)
