"""PROTOTYPE ONLY: corrected predictor-by-evaluation-library figure variants."""

from pathlib import Path
import json

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np


HERE = Path(__file__).resolve().parent
COLLECTIONS = HERE.parents[1] / "outputs" / "ground_truth_collections"

LANDSCAPES = {
    "Synthetic GT": COLLECTIONS / "Synthetic GT/libraries_used_for_figures/lib_size_spearman_results_type3.json",
    "VTS1": COLLECTIONS / "deepSQUID VTS1/libraries_used_for_figures/lib_size_spearman_results_high.json",
    "HuR": COLLECTIONS / "deepSQUID HuR/libraries_used_for_figures/lib_size_spearman_results_high.json",
}
EVALS = ["Random\nholdout", "Activity-\nbalanced", "Targeted\npairwise", "Saturated\nsingles + doubles"]
KEYS = ["rand", "type2", "pairwise", "type3"]
PREDICTORS = ["SSM additive\nbaseline", "Random-trained\nnonlinear add. + pairwise"]
COLORS = {"SSM": "#3E72AE", "Random": "#C86442"}


def _mean(value):
    return float(np.nanmean(np.asarray(value, dtype=float)))


def load_data():
    """Return real Spearman values; do not manufacture unavailable comparisons."""
    out = {}
    for landscape, path in LANDSCAPES.items():
        results = json.loads(path.read_text())
        oracle = next(iter(results["surrogate"]))
        trained = results["surrogate"][oracle]["nonlinear additive + pairwise"]["10"]["20000"]

        ssm_runs = []
        for run in results["saturated"].values():
            if landscape == "Synthetic GT":
                scores = run["additive"]
            else:
                scores = run[oracle]
            ssm_runs.append(scores)

        ssm = [np.nan]
        for key in KEYS[1:]:
            ssm.append(_mean([run[key] for run in ssm_runs]))
        random_trained = [_mean(trained[key]) for key in KEYS]
        out[landscape] = np.asarray([ssm, random_trained])
    return out


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def footer(fig, saturated=True):
    caveat = " • saturated values provisional pending removal of 4 triple mutants" if saturated else ""
    fig.text(
        .5, .012,
        "PROTOTYPE • Spearman ρ • high-WT landscapes • 10% random training, n=20,000" + caveat,
        ha="center", fontsize=7.5, color="#7A473D",
    )


def variant_a(data):
    """Small-multiple matrices make training and evaluation axes explicit."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), facecolor="#FCFBF8")
    fig.suptitle("Training source and evaluation library answer different questions", x=.055, ha="left", fontsize=17, weight="bold")
    fig.text(.055, .88, "Rows are predictors; columns are held-out evaluation sets. Saturated add. + pairwise is never treated as training data.", fontsize=9.5)
    norm = Normalize(0, 1)
    for ax, (landscape, values) in zip(axes, data.items()):
        ax.imshow(values, cmap="YlGnBu", norm=norm, aspect="auto")
        for r in range(2):
            for c in range(4):
                value = values[r, c]
                label = "not run" if np.isnan(value) else f"{value:.2f}"
                ax.text(c, r, label, ha="center", va="center", color="white" if value > .58 else "#25313B", weight="bold", fontsize=10)
        ax.set_title(landscape, weight="bold", pad=10)
        ax.set_xticks(range(4), EVALS)
        ax.tick_params(axis="x", length=0, labelsize=8)
        ax.set_yticks(range(2), PREDICTORS if ax is axes[0] else [])
        ax.tick_params(axis="y", length=0)
        for spine in ax.spines.values(): spine.set_visible(False)
    fig.subplots_adjust(left=.18, right=.98, bottom=.19, top=.77, wspace=.20)
    footer(fig)
    fig.savefig(HERE / "variant_a_role_cards.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def variant_b(data):
    """Evaluation-library view emphasizes inflation within the random-trained model."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.2), sharey=True, facecolor="white")
    fig.suptitle("The same random-trained model looks different depending on where it is tested", x=.055, ha="left", fontsize=17, weight="bold")
    fig.text(.055, .895, "Targeted pairwise currently probes double mutants only in annotated base-pairing regions.", fontsize=9.5)
    palette = ["#C86442", "#5E879D", "#8C72A8"]
    for ax, (landscape, values) in zip(axes, data.items()):
        vals = values[1, :3]
        y = np.arange(3)
        ax.barh(y, vals, color=palette, height=.62)
        for yy, value in zip(y, vals):
            ax.text(min(value + .015, 1.01), yy, f"{value:.2f}", va="center", fontsize=9, weight="bold")
        ax.set_title(landscape, weight="bold")
        ax.set_xlim(0, 1.04)
        ax.axvline(1, color="#AAA", lw=.8)
        ax.grid(axis="x", alpha=.16)
        ax.set_xlabel("Spearman ρ")
        ax.invert_yaxis()
    axes[0].set_yticks(range(3), [x.replace("\n", " ") for x in EVALS[:3]])
    fig.text(.055, .075, "Predictor: nonlinear additive + pairwise surrogate trained on a 20K random library", fontsize=9, weight="bold", color=COLORS["Random"])
    fig.subplots_adjust(left=.18, right=.98, bottom=.18, top=.79, wspace=.18)
    footer(fig, saturated=False)
    fig.savefig(HERE / "variant_b_mutation_order.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def variant_c(data):
    """Role-first schematic plus a compact comparison strip."""
    fig = plt.figure(figsize=(13.5, 7.2), facecolor="#FAFAF7")
    fig.suptitle("Separate what builds the predictor from what challenges it", x=.055, ha="left", fontsize=17, weight="bold")

    ax = fig.add_axes([.05, .49, .90, .36]); ax.axis("off")
    ax.text(.01, .98, "PREDICTORS / TRAINING SOURCES", fontsize=9, color="#555", weight="bold", va="top")
    cards = [
        (.02, .18, .22, .58, COLORS["SSM"], "SSM additive baseline", "Single substitutions\nLower bound for additive-only learning"),
        (.29, .18, .25, .58, COLORS["Random"], "Random-trained surrogate", "20K variants at 10% mutation\nNonlinear additive + pairwise model"),
    ]
    for x, y, w, h, color, title, body in cards:
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=.015",fc="white",ec=color,lw=2))
        ax.text(x+.025,y+h-.14,title,weight="bold",fontsize=10,color=color)
        ax.text(x+.025,y+.16,body,fontsize=8.5,linespacing=1.5)
    ax.annotate("evaluated on", xy=(.61,.47), xytext=(.55,.47), arrowprops=dict(arrowstyle="->",lw=1.5,color="#666"), va="center", fontsize=9, color="#555")
    eval_x = [.67, .76, .85, .94]
    eval_titles = ["Random", "Activity-\nbalanced", "Targeted\npairwise", "Saturated"]
    eval_sub = ["holdout", "uniform activity", "base-pair regions", "singles + doubles"]
    for x, title, sub in zip(eval_x, eval_titles, eval_sub):
        ax.add_patch(FancyBboxPatch((x-.04,.20),.08,.54,boxstyle="round,pad=.01",fc="#EEF1EE",ec="#9AA59D",lw=1))
        ax.text(x,.57,title,ha="center",va="center",fontsize=7.7,weight="bold")
        ax.text(x,.30,sub,ha="center",va="center",fontsize=6.7,color="#555",rotation=90 if len(sub)>15 else 0)

    ax2 = fig.add_axes([.10, .10, .82, .29]); ax2.axis("off")
    ax2.text(0, 1.04, "RECOVERY SUMMARY", fontsize=9, color="#555", weight="bold")
    cell_w, row_h = .058, .18
    for li, (landscape, values) in enumerate(data.items()):
        base_x = .16 + li*.29
        ax2.text(base_x+.085, .88, landscape, ha="center", weight="bold", fontsize=9)
        for r, rowname in enumerate(["SSM", "Random"]):
            y=.58-r*.25
            if li==0: ax2.text(.08,y,rowname,ha="right",va="center",fontsize=8,color=COLORS[rowname])
            for c, value in enumerate(values[r]):
                x=base_x+c*cell_w
                fc="#E6E4DF" if np.isnan(value) else plt.cm.YlGnBu(value)
                ax2.add_patch(Rectangle((x,y-row_h/2),cell_w-.005,row_h,fc=fc,ec="white"))
                ax2.text(x+(cell_w-.005)/2,y,"—" if np.isnan(value) else f"{value:.2f}",ha="center",va="center",fontsize=6.5,color="white" if value>.6 else "#222",weight="bold")
        for c, label in enumerate(["R", "A", "T", "S"]): ax2.text(base_x+c*cell_w+.026,.05,label,ha="center",fontsize=7,color="#666")
    ax2.text(.53, -.08, "R random holdout   A activity-balanced   T targeted pairwise   S saturated singles+doubles", ha="center", fontsize=7.5, color="#555")
    footer(fig)
    fig.savefig(HERE / "variant_c_coverage_matrix.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    style()
    data = load_data()
    variant_a(data)
    variant_b(data)
    variant_c(data)
    for path in sorted(HERE.glob("variant_*.png")):
        print(path)
