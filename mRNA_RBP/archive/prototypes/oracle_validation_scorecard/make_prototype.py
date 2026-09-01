"""Throwaway prototype: bar-free redesign of deepsquid_vs_real_oracle_heldout_test_bar.png.

Not wired into the pipeline. VTS1 value/n reused from the archived old-sequence
run (archive/inactive_runs/vts1_high_old_sequence_20260831) purely to test the
layout — it is flagged as stale in the plot itself and must be regenerated
under the current VTS1 sequence before this design is used for real.
"""
import matplotlib.pyplot as plt

points = [
    {"label": "VTS1\n(high-activity WT)", "rho": 0.966, "n": 39_991, "stale": True},
    {"label": "HuR\n(high-activity WT)", "rho": 0.973, "n": 40_358, "stale": False},
]

fig, ax = plt.subplots(figsize=(5.5, 3.2))

xs = list(range(len(points)))
for x, p in zip(xs, points):
    color = "#9a9a9a" if p["stale"] else "#4C72B0"
    ax.scatter([x], [p["rho"]], s=140, color=color, zorder=3)
    ax.annotate(f"{p['rho']:.3f}", (x, p["rho"]), textcoords="offset points",
                xytext=(0, 10), ha="center", fontsize=12, fontweight="bold")
    n_label = f"n={p['n']:,}" + (" (old seq., pending\nregeneration)" if p["stale"] else "")
    ax.annotate(n_label, (x, p["rho"]), textcoords="offset points",
                xytext=(0, -22), ha="center", fontsize=8.5, color="#555555")

ax.set_xlim(-0.6, len(points) - 0.4)
ax.set_ylim(0.90, 1.0)
ax.set_xticks(xs)
ax.set_xticklabels([p["label"] for p in points], fontsize=10)
ax.set_ylabel("Spearman ρ vs. experimental values\n(held-out test)")
ax.set_title("Deep-squid accuracy: held-out test performance")
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
ax.spines["left"].set_bounds(0.90, 1.0)
ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.6)

fig.tight_layout()
out = "/home/nagle/final_version/mRNA_RBP/prototypes/oracle_validation_scorecard/outputs/deepsquid_vs_real_oracle_heldout_dotplot.png"
fig.savefig(out, dpi=200)
print(out)
