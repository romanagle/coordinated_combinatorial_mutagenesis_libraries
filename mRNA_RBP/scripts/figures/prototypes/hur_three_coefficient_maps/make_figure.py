"""Make the HuR version of the three-way additive/pairwise comparison."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))

from mRNA_RBP.prototypes.vts1_three_coefficient_maps import make_figure as figure


HERE = Path(__file__).resolve().parents[5] / "mRNA_RBP" / "prototypes" / "hur_three_coefficient_maps"
figure.RBP = "HuR"
figure.SCORE_KEY = "deepsquid_hur"
figure.SQUID = ROOT / (
    "mRNA_RBP/outputs/ground_truth_collections/ResidualBind oracle HuR/"
    "libraries_used_for_figures/surrogate_coefs_high/"
    "coefs_k00_mut10_lib20000_nonlinear_additive_p_pairwise_hur_residualbind.npz"
)
figure.DEEPSQUID = ROOT / (
    "mRNA_RBP/outputs/ground_truth_collections/deepSQUID HuR/"
    "libraries_used_for_figures/surrogate_coefs_high/"
    "coefs_k00_mut10_lib20000_nonlinear_additive_p_pairwise_deepsquid_hur.npz"
)
figure.INSTANCE = ROOT / "mRNA_RBP/runs/deepsquid/hur/high/instance_00"
figure.HERE = HERE
figure.OUT = HERE / "hur_additive_pairwise_coefficient_maps_10pct_20k.png"
figure.OUT_PDF = HERE / "hur_additive_pairwise_coefficient_maps_10pct_20k.pdf"


if __name__ == "__main__":
    figure.main()
