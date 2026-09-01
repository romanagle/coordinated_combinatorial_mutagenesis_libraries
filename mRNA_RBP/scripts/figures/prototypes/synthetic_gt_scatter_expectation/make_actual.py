#!/usr/bin/env python3
"""Render actual matched Synthetic GT control holdout predictions."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parents[5] / "mRNA_RBP" / "prototypes" / "synthetic_gt_scatter_expectation"
PREDICTIONS = HERE / "outputs" / "synthetic_gt_control_predictions.npz"
OUTPUT = HERE / "outputs" / "synthetic_gt_scatter_actual.png"
DISPLAY_SEED = 20260809
RANDOM_COLOR = "#D94B45"
ACTIVITY_COLOR = "#3776B6"
PANELS = [
    ("positive", "Positive control: structured Synthetic GT"),
    ("negative", "Negative control: motif-only Synthetic GT"),
]


def sample_pair(random_y, random_yhat, activity_y, activity_yhat, seed):
    n = min(4_000, len(random_y), len(activity_y))
    rng = np.random.default_rng(seed)
    random_take = rng.choice(len(random_y), n, replace=False)
    activity_take = rng.choice(len(activity_y), n, replace=False)
    return (
        random_y[random_take],
        random_yhat[random_take],
        activity_y[activity_take],
        activity_yhat[activity_take],
    )


def main():
    if not PREDICTIONS.is_file():
        raise FileNotFoundError(
            f"Missing frozen predictions: {PREDICTIONS}. "
            "Run generate_actual_predictions.py first."
        )
    data = np.load(PREDICTIONS)

    all_values = []
    panel_data = []
    for index, (key, title) in enumerate(PANELS):
        y_random = data[f"{key}_y_rand_10"].astype(float)
        yhat_random = data[f"{key}_yhat_rand_10"].astype(float)
        y_activity = data[f"{key}_y_activity_10"].astype(float)
        yhat_activity = data[f"{key}_yhat_activity_10"].astype(float)
        shown = sample_pair(
            y_random,
            yhat_random,
            y_activity,
            yhat_activity,
            DISPLAY_SEED + index,
        )
        panel_data.append(
            (
                title,
                shown,
                float(data[f"{key}_rho_rand_10"]),
                float(data[f"{key}_rho_activity_10"]),
            )
        )
        all_values.extend((y_random, yhat_random, y_activity, yhat_activity))

    combined = np.concatenate(all_values)
    low = float(np.nanmin(combined))
    high = max(0.0, float(np.nanmax(combined)))
    pad = 0.035 * (high - low)
    limits = (low - pad, high + pad)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.6), sharex=True, sharey=True)
    for ax, (title, shown, rho_random, rho_activity) in zip(axes, panel_data):
        random_y, random_yhat, activity_y, activity_yhat = shown
        ax.scatter(
            activity_y,
            activity_yhat,
            s=7,
            alpha=0.18,
            color=ACTIVITY_COLOR,
            edgecolors="none",
            rasterized=True,
            label="Activity-balanced evaluation",
        )
        ax.scatter(
            random_y,
            random_yhat,
            s=7,
            alpha=0.24,
            color=RANDOM_COLOR,
            edgecolors="none",
            rasterized=True,
            label="Random-library holdout",
        )
        ax.plot(limits, limits, linestyle="--", color="#444444", linewidth=1.2)
        ax.set_xlim(limits)
        ax.set_ylim(limits)
        ax.set_aspect("equal", adjustable="box")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(
            f"{title}\nrandom ρ = {rho_random:.3f}  |  "
            f"activity-balanced ρ = {rho_activity:.3f}",
            fontsize=11,
            fontweight="bold",
            pad=12,
        )
        ax.set_xlabel("Ground-truth activity", fontsize=10)
    axes[0].set_ylabel("Predicted activity", fontsize=10)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles[::-1],
        labels[::-1],
        loc="upper center",
        ncol=2,
        frameon=False,
        markerscale=3.0,
        bbox_to_anchor=(0.5, 0.91),
    )
    fig.suptitle("Actual holdout behavior", fontsize=16, fontweight="bold", y=0.985)
    negative_n = len(data["negative_y_activity_10"])
    activity_note = (
        "standard 3/5/7/15-mutant activity-balanced evaluation "
        f"(20K target; negative retained {negative_n:,} after uniformization)"
    )
    fig.text(
        0.5,
        0.018,
        "Instance 00; both controls: fixed 10% (4 mutations), 20K training libraries; "
        + activity_note + "; "
        "equal deterministic display samples; Spearman statistics use all predictions",
        ha="center",
        fontsize=8.2,
        color="#555555",
    )
    fig.tight_layout(rect=(0.02, 0.055, 1, 0.84), w_pad=2.4)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
