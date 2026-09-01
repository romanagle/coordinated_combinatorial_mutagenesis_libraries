"""THROWAWAY PROTOTYPE: three HuR/VTS1 failure-decomposition bar layouts."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata, spearmanr


HERE = Path(__file__).resolve().parents[5] / "mRNA_RBP" / "prototypes" / "activity_balanced_failure_bars"
ROOT = HERE.parents[1]
OUT = HERE / "outputs"
MUT_COUNTS = np.array([3, 5, 7, 15])
TRAIN_RATES = (5, 10, 25)
COLORS = {5: "#4C72B0", 10: "#DD8452", 25: "#55A868"}
CACHES = {
    "HuR": ROOT / "runs/deepsquid/hur/high/scatter_by_mutcount_predictions.npz",
    "VTS1": ROOT / "runs/deepsquid/vts1/high/scatter_by_mutcount_predictions.npz",
}


def load_rhos():
    values = {}
    sample_sizes = {}
    for landscape, path in CACHES.items():
        values[landscape] = {}
        with np.load(path) as data:
            for rate in TRAIN_RATES:
                y = data[f"y_activity_{rate}"]
                yhat = data[f"yhat_activity_{rate}"]
                labels = data[f"rate_labels_{rate}"]
                values[landscape][rate] = np.array([
                    spearmanr(y[labels == count], yhat[labels == count]).statistic
                    for count in MUT_COUNTS
                ])
                sample_sizes[landscape] = np.array([
                    np.sum(labels == count) for count in MUT_COUNTS
                ])
    return values, sample_sizes


def load_rank_rmse():
    """Global percentile-rank RMSE split by mutation-count stratum."""
    values = {}
    for landscape, path in CACHES.items():
        values[landscape] = {}
        with np.load(path) as data:
            for rate in TRAIN_RATES:
                y = data[f"y_activity_{rate}"]
                yhat = data[f"yhat_activity_{rate}"]
                labels = data[f"rate_labels_{rate}"]
                n = len(y)
                true_pct = (rankdata(y, method="average") - 1) / (n - 1)
                pred_pct = (rankdata(yhat, method="average") - 1) / (n - 1)
                residual = pred_pct - true_pct
                values[landscape][rate] = np.array([
                    np.sqrt(np.mean(residual[labels == count] ** 2))
                    for count in MUT_COUNTS
                ])
    return values


def load_signed_rank_bias():
    """Mean predicted-minus-true global percentile rank by stratum."""
    values = {}
    for landscape, path in CACHES.items():
        values[landscape] = {}
        with np.load(path) as data:
            for rate in TRAIN_RATES:
                y = data[f"y_activity_{rate}"]
                yhat = data[f"yhat_activity_{rate}"]
                labels = data[f"rate_labels_{rate}"]
                n = len(y)
                true_pct = (rankdata(y, method="average") - 1) / (n - 1)
                pred_pct = (rankdata(yhat, method="average") - 1) / (n - 1)
                residual = pred_pct - true_pct
                values[landscape][rate] = np.array([
                    np.mean(residual[labels == count]) for count in MUT_COUNTS
                ])
    return values


def finish(fig, filename):
    fig.suptitle(
        "Where random-trained nonlinear additive + pairwise surrogates fail",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


def grouped_bars(values):
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 7.4), sharex=True, sharey=True)
    x = np.arange(len(MUT_COUNTS))
    width = 0.24
    for ax, landscape in zip(axes, ("HuR", "VTS1")):
        for offset, rate in zip((-1, 0, 1), TRAIN_RATES):
            ax.bar(x + offset * width, values[landscape][rate], width,
                   color=COLORS[rate], label=f"{rate}% random training")
        ax.set_title(landscape, loc="left", fontweight="bold")
        ax.set_ylabel("Spearman ρ")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.2)
    axes[0].legend(ncol=3, frameon=False, loc="lower left", fontsize=9)
    axes[1].set_xticks(x, MUT_COUNTS)
    axes[1].set_xlabel("Mutations per activity-balanced sequence")
    finish(fig, "variant_a_grouped_training_rates.png")


def mean_bars_with_points(values, sample_sizes):
    fig, axes = plt.subplots(2, 1, figsize=(7.4, 7.4), sharex=True, sharey=True)
    x = np.arange(len(MUT_COUNTS))
    for ax, landscape in zip(axes, ("HuR", "VTS1")):
        matrix = np.vstack([values[landscape][rate] for rate in TRAIN_RATES])
        means = matrix.mean(axis=0)
        ax.bar(x, means, width=0.66, color="#9AA0A6", alpha=0.78,
               label="Mean across training rates")
        for rate in TRAIN_RATES:
            ax.scatter(x, values[landscape][rate], s=35, color=COLORS[rate],
                       zorder=3, label=f"{rate}% training")
        ax.set_title(landscape, loc="left", fontweight="bold")
        ax.set_ylabel("Mean Spearman ρ")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.2)
        for i, n in enumerate(sample_sizes[landscape]):
            ax.text(i, 0.035, f"n={n:,}", ha="center", fontsize=8, rotation=90)
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, ncol=2, frameon=False, loc="lower left", fontsize=8)
    axes[1].set_xticks(x, MUT_COUNTS)
    axes[1].set_xlabel("Mutations per activity-balanced sequence")
    finish(fig, "variant_b_mean_with_training_points.png")


def horizontal_rows(values):
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 7.2), sharex=True)
    y = np.arange(len(MUT_COUNTS))
    for ax, landscape in zip(axes, ("HuR", "VTS1")):
        matrix = np.vstack([values[landscape][rate] for rate in TRAIN_RATES])
        means = matrix.mean(axis=0)
        lo, hi = matrix.min(axis=0), matrix.max(axis=0)
        ax.barh(y, means, color="#6C8EBF", alpha=0.85)
        ax.errorbar(means, y, xerr=np.vstack([means - lo, hi - means]), fmt="none",
                    ecolor="#202124", capsize=4, lw=1.4,
                    label="Range across training rates")
        ax.set_yticks(y, [f"{count} mutations" for count in MUT_COUNTS])
        ax.invert_yaxis()
        ax.set_title(landscape, loc="left", fontweight="bold")
        ax.grid(axis="x", alpha=0.2)
    axes[0].legend(frameon=False, loc="lower right", fontsize=9)
    axes[1].set_xlabel("Mean Spearman ρ across random-training mutation rates")
    axes[1].set_xlim(0, 1.05)
    finish(fig, "variant_c_horizontal_rows.png")


def rank_rmse_bars(values):
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 7.4), sharex=True, sharey=True)
    x = np.arange(len(MUT_COUNTS))
    width = 0.24
    for ax, landscape in zip(axes, ("HuR", "VTS1")):
        for offset, rate in zip((-1, 0, 1), TRAIN_RATES):
            heights = values[landscape][rate]
            bars = ax.bar(x + offset * width, heights, width,
                          color=COLORS[rate], label=f"{rate}% random training")
            ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=7, rotation=90)
        ax.set_title(landscape, loc="left", fontweight="bold")
        ax.set_ylabel("Percentile-rank RMSE")
        ax.set_ylim(0, 0.58)
        ax.grid(axis="y", alpha=0.2)
    axes[0].legend(ncol=3, frameon=False, loc="upper left", fontsize=9)
    axes[1].set_xticks(x, MUT_COUNTS)
    axes[1].set_xlabel("Mutations per activity-balanced sequence")
    fig.suptitle(
        "Which sequences drive global ranking error?",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / "variant_d_global_percentile_rank_rmse.png",
                dpi=180, bbox_inches="tight")
    plt.close(fig)


def rank_rmse_10pct_variants(values):
    landscapes = ("HuR", "VTS1")
    color = COLORS[10]

    # E — conventional bars.
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.8), sharex=True, sharey=True)
    for ax, landscape in zip(axes, landscapes):
        bars = ax.bar(MUT_COUNTS, values[landscape][10], width=[1.3, 1.3, 1.3, 2.2],
                      color=color, alpha=0.9)
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
        ax.set_title(landscape, loc="left", fontweight="bold")
        ax.set_ylabel("Percentile-rank RMSE")
        ax.set_ylim(0, 0.56)
        ax.grid(axis="y", alpha=0.2)
    axes[1].set_xticks(MUT_COUNTS)
    axes[1].set_xlabel("Mutations per activity-balanced sequence")
    fig.suptitle("Global ranking error — 10% random training", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / "variant_e_10pct_bars.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def combined_10pct_variants(values):
    landscapes = ("HuR", "VTS1")
    color = COLORS[10]
    x = np.arange(len(MUT_COUNTS))
    hur = values["HuR"][10]
    vts1 = values["VTS1"][10]
    landscape_colors = {"HuR": "#4C72B0", "VTS1": "#C44E52"}

    # H — two profiles overlaid on one categorical axis.
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for name, y in (("HuR", hur), ("VTS1", vts1)):
        ax.plot(x, y, marker="o", ms=8, lw=2.5,
                color=landscape_colors[name], label=name)
        for xi, value in zip(x, y):
            ax.annotate(f"{value:.3f}", (xi, value), xytext=(0, 8),
                        textcoords="offset points", ha="center", fontsize=8)
    ax.set_xticks(x, MUT_COUNTS)
    ax.set_xlabel("Mutations per activity-balanced sequence")
    ax.set_ylabel("Percentile-rank RMSE")
    ax.set_ylim(0, 0.56)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)
    ax.set_title("Global ranking error by mutation count\n10% random training")
    fig.tight_layout()
    fig.savefig(OUT / "variant_h_overlay_profiles.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def paired_magnitude_and_direction(rank_rmse, signed_bias):
    """L — preferred paired bars plus a distinct directional-bias panel."""
    values = rank_rmse
    landscapes = ("HuR", "VTS1")
    color = COLORS[10]
    hur = rank_rmse["HuR"][10]
    vts1 = rank_rmse["VTS1"][10]
    landscape_colors = {"HuR": "#4C72B0", "VTS1": "#C44E52"}
    x = np.arange(len(MUT_COUNTS))
    width = 0.36
    colors = {"HuR": "#4C72B0", "VTS1": "#C44E52"}
    fig, axes = plt.subplots(2, 1, figsize=(7.8, 7.0), sharex=True)

    ax = axes[0]
    h = ax.bar(x - width / 2, rank_rmse["HuR"][10], width,
               color=colors["HuR"], label="HuR")
    v = ax.bar(x + width / 2, rank_rmse["VTS1"][10], width,
               color=colors["VTS1"], label="VTS1")
    ax.bar_label(h, fmt="%.3f", padding=3, fontsize=8)
    ax.bar_label(v, fmt="%.3f", padding=3, fontsize=8)
    ax.set_ylabel("Percentile-rank RMSE")
    ax.set_ylim(0, 0.56)
    ax.set_title("Magnitude of ranking error", loc="left", fontweight="bold")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)

    ax = axes[1]
    h = ax.bar(x - width / 2, signed_bias["HuR"][10], width, color=colors["HuR"])
    v = ax.bar(x + width / 2, signed_bias["VTS1"][10], width, color=colors["VTS1"])
    ax.axhline(0, color="#202124", lw=1)
    ax.bar_label(h, fmt="%+.3f", padding=3, fontsize=8)
    ax.bar_label(v, fmt="%+.3f", padding=3, fontsize=8)
    ax.set_ylabel("Mean signed percentile-rank error")
    ax.set_title("Direction of systematic rank bias", loc="left", fontweight="bold")
    ax.text(0.01, 0.97, "over-ranked ↑", transform=ax.transAxes,
            ha="left", va="top", fontsize=9, color="#555555")
    ax.text(0.99, 0.03, "under-ranked ↓", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=9, color="#555555")
    ax.set_ylim(-0.24, 0.25)
    ax.grid(axis="y", alpha=0.2)
    ax.set_xticks(x, MUT_COUNTS)
    ax.set_xlabel("Mutations per activity-balanced sequence")

    fig.suptitle("Global ranking error · 10% random training", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / "variant_l_paired_rmse_and_signed_bias.png",
                dpi=180, bbox_inches="tight")
    plt.close(fig)

    # I — dumbbells foreground the landscape difference at each order.
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    y = np.arange(len(MUT_COUNTS))
    for yi, h, v in zip(y, hur, vts1):
        ax.plot([h, v], [yi, yi], color="#B8BDC3", lw=3, zorder=1)
    ax.scatter(hur, y, s=85, color=landscape_colors["HuR"], label="HuR", zorder=2)
    ax.scatter(vts1, y, s=85, color=landscape_colors["VTS1"], label="VTS1", zorder=2)
    for yi, h, v in zip(y, hur, vts1):
        ax.annotate(f"{h:.3f}", (h, yi), xytext=(-7, 8), textcoords="offset points",
                    ha="right", fontsize=8, color=landscape_colors["HuR"])
        ax.annotate(f"{v:.3f}", (v, yi), xytext=(7, 8), textcoords="offset points",
                    ha="left", fontsize=8, color=landscape_colors["VTS1"])
    ax.set_yticks(y, [f"{m} mutations" for m in MUT_COUNTS])
    ax.invert_yaxis()
    ax.set_xlim(0, 0.56)
    ax.set_xlabel("Percentile-rank RMSE")
    ax.grid(axis="x", alpha=0.2)
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("HuR–VTS1 ranking-error gap\n10% random training")
    fig.tight_layout()
    fig.savefig(OUT / "variant_i_dumbbell.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # J — familiar paired bars, one compact panel.
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    width = 0.36
    b1 = ax.bar(x - width / 2, hur, width, color=landscape_colors["HuR"], label="HuR")
    b2 = ax.bar(x + width / 2, vts1, width, color=landscape_colors["VTS1"], label="VTS1")
    ax.bar_label(b1, fmt="%.3f", padding=3, fontsize=8, rotation=90)
    ax.bar_label(b2, fmt="%.3f", padding=3, fontsize=8, rotation=90)
    ax.set_xticks(x, MUT_COUNTS)
    ax.set_xlabel("Mutations per activity-balanced sequence")
    ax.set_ylabel("Percentile-rank RMSE")
    ax.set_ylim(0, 0.56)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)
    ax.set_title("Global ranking error by mutation count\n10% random training")
    fig.tight_layout()
    fig.savefig(OUT / "variant_j_paired_bars.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # K — compact matrix for pattern recognition rather than trajectory.
    matrix = np.vstack([hur, vts1])
    fig, ax = plt.subplots(figsize=(7.6, 2.7))
    image = ax.imshow(matrix, cmap="YlOrRd", vmin=0, vmax=0.52, aspect="auto")
    for row in range(2):
        for col in range(4):
            value = matrix[row, col]
            ax.text(col, row, f"{value:.3f}", ha="center", va="center",
                    color="white" if value > 0.30 else "#202124", fontweight="bold")
    ax.set_xticks(x, MUT_COUNTS)
    ax.set_yticks([0, 1], ["HuR", "VTS1"])
    ax.set_xlabel("Mutations per activity-balanced sequence")
    ax.set_title("Percentile-rank RMSE · 10% random training")
    fig.colorbar(image, ax=ax, label="Rank RMSE", fraction=0.035, pad=0.03)
    fig.tight_layout()
    fig.savefig(OUT / "variant_k_heatmap.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # F — lollipops emphasize values without visually weighting bar area.
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.8), sharex=True, sharey=True)
    for ax, landscape in zip(axes, landscapes):
        y = values[landscape][10]
        ax.vlines(MUT_COUNTS, 0, y, color="#B8BDC3", lw=3)
        ax.scatter(MUT_COUNTS, y, s=95, color=color, zorder=3)
        for x, value in zip(MUT_COUNTS, y):
            ax.annotate(f"{value:.3f}", (x, value), xytext=(0, 8),
                        textcoords="offset points", ha="center", fontsize=9)
        ax.set_title(landscape, loc="left", fontweight="bold")
        ax.set_ylabel("Percentile-rank RMSE")
        ax.set_ylim(0, 0.56)
        ax.grid(axis="y", alpha=0.2)
    axes[1].set_xticks(MUT_COUNTS)
    axes[1].set_xlabel("Mutations per activity-balanced sequence")
    fig.suptitle("Global ranking error — 10% random training", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / "variant_f_10pct_lollipops.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # G — connected profile treats mutation count as ordered and foregrounds shape.
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.8), sharex=True, sharey=True)
    for ax, landscape in zip(axes, landscapes):
        y = values[landscape][10]
        ax.plot(MUT_COUNTS, y, color=color, lw=2.5, marker="o", ms=8)
        ax.fill_between(MUT_COUNTS, 0, y, color=color, alpha=0.10)
        for x, value in zip(MUT_COUNTS, y):
            ax.annotate(f"{value:.3f}", (x, value), xytext=(0, 8),
                        textcoords="offset points", ha="center", fontsize=9)
        ax.set_title(landscape, loc="left", fontweight="bold")
        ax.set_ylabel("Percentile-rank RMSE")
        ax.set_ylim(0, 0.56)
        ax.grid(axis="y", alpha=0.2)
    axes[1].set_xticks(MUT_COUNTS)
    axes[1].set_xlabel("Mutations per activity-balanced sequence")
    fig.suptitle("Global ranking error depends on mutation count", fontsize=13)
    fig.text(0.5, 0.925, "Nonlinear additive + pairwise surrogate · 10% random training",
             ha="center", fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(OUT / "variant_g_10pct_connected_profile.png",
                dpi=180, bbox_inches="tight")
    plt.close(fig)


def load_standardized_score_residuals():
    values = {}
    for landscape, path in CACHES.items():
        with np.load(path) as data:
            y = data["y_activity_10"].astype(float)
            yhat = data["yhat_activity_10"].astype(float)
            labels = data["rate_labels_10"].astype(int)
        residual = (yhat - y) / np.std(y)
        values[landscape] = {
            count: residual[labels == count] for count in MUT_COUNTS
        }
    return values


def standardized_residual_variants(values):
    x = np.arange(len(MUT_COUNTS))
    colors = {"HuR": "#9BB8D3", "VTS1": "#E7A3A8"}
    median_colors = {"HuR": "#264F78", "VTS1": "#8E2F3A"}

    # M — distributions expose direction, variance, skew, and mixed errors.
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    offsets = {"HuR": -0.18, "VTS1": 0.18}
    for landscape in ("HuR", "VTS1"):
        data = [values[landscape][count] for count in MUT_COUNTS]
        positions = x + offsets[landscape]
        bp = ax.boxplot(data, positions=positions, widths=0.28, patch_artist=True,
                        showfliers=False, whis=(5, 95), manage_ticks=False,
                        medianprops={"color": "white", "linewidth": 1.5})
        for box in bp["boxes"]:
            box.set_facecolor(colors[landscape])
            box.set_alpha(0.85)
        for part in ("whiskers", "caps"):
            for artist in bp[part]:
                artist.set_color(colors[landscape])
        ax.plot([], [], color=colors[landscape], lw=8, label=landscape)
    ax.axhline(0, color="#202124", lw=1)
    ax.set_xticks(x, MUT_COUNTS)
    ax.set_xlabel("Mutations per activity-balanced sequence")
    ax.set_ylabel("Standardized score residual  (ŷ − y) / SD(y)")
    ax.set_title("Direction and spread of prediction error · 10% random training")
    ax.text(0.01, 0.98, "overpredicted ↑", transform=ax.transAxes,
            ha="left", va="top", fontsize=9, color="#555555")
    ax.text(0.01, 0.02, "underpredicted ↓", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=9, color="#555555")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "variant_m_standardized_residual_boxplots.png",
                dpi=180, bbox_inches="tight")
    plt.close(fig)

    # O — paired violins reveal density and multimodality around zero.
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    for landscape in ("HuR", "VTS1"):
        data = [values[landscape][count] for count in MUT_COUNTS]
        positions = x + offsets[landscape]
        violins = ax.violinplot(data, positions=positions, widths=0.32,
                                showmeans=False, showmedians=False,
                                showextrema=False, points=200)
        for body in violins["bodies"]:
            body.set_facecolor(colors[landscape])
            body.set_edgecolor(colors[landscape])
            body.set_alpha(0.72)
        medians = [np.median(group) for group in data]
        for position, median in zip(positions, medians):
            ax.hlines(median, position - 0.10, position + 0.10,
                      color=median_colors[landscape], linewidth=2.4, zorder=3)
        ax.plot([], [], color=colors[landscape], lw=8, alpha=0.72, label=landscape)
    ax.axhline(0, color="#202124", lw=1)
    ax.set_xticks(x, MUT_COUNTS)
    ax.set_xlabel("Mutations per activity-balanced sequence")
    ax.set_ylabel("Standardized score residual  (ŷ − y) / SD(y)")
    ax.set_title("Direction and distribution of prediction error · 10% random training")
    ax.text(0.01, 0.98, "overpredicted ↑", transform=ax.transAxes,
            ha="left", va="top", fontsize=9, color="#555555")
    ax.text(0.01, 0.02, "underpredicted ↓", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=9, color="#555555")
    ax.annotate("10% training library\n≈ 4 mutations",
                xy=(0.5, 0.0), xytext=(0.5, 0.92),
                ha="center", va="bottom", fontsize=9, color="#555555",
                arrowprops={"arrowstyle": "-|>", "color": "#555555", "lw": 1.3})
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "variant_o_standardized_residual_violins.png",
                dpi=180, bbox_inches="tight")
    plt.close(fig)

    # N — a lighter summary if full distributions are visually too dense.
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    for landscape, offset in offsets.items():
        means = np.array([np.mean(values[landscape][count]) for count in MUT_COUNTS])
        stds = np.array([np.std(values[landscape][count]) for count in MUT_COUNTS])
        ax.errorbar(x + offset, means, yerr=stds, fmt="o", ms=7, capsize=4,
                    lw=1.7, color=colors[landscape], label=landscape)
    ax.axhline(0, color="#202124", lw=1)
    ax.set_xticks(x, MUT_COUNTS)
    ax.set_xlabel("Mutations per activity-balanced sequence")
    ax.set_ylabel("Mean standardized score residual ± SD")
    ax.set_title("Prediction bias and variance · 10% random training")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "variant_n_standardized_residual_mean_sd.png",
                dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    values, sample_sizes = load_rhos()
    grouped_bars(values)
    mean_bars_with_points(values, sample_sizes)
    horizontal_rows(values)
    rank_rmse = load_rank_rmse()
    rank_rmse_bars(rank_rmse)
    rank_rmse_10pct_variants(rank_rmse)
    combined_10pct_variants(rank_rmse)
    paired_magnitude_and_direction(rank_rmse, load_signed_rank_bias())
    standardized_residual_variants(load_standardized_score_residuals())
    print(f"Wrote fifteen throwaway prototypes to {OUT}")


if __name__ == "__main__":
    main()
