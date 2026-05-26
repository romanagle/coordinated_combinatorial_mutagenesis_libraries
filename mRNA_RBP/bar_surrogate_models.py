"""
mRNA_RBP/bar_surrogate_models.py

Grouped bar chart: Spearman ρ for each surrogate model × eval library.

Models (x-axis groups):
  SSM | Additive | Additive+Pairwise | NL Additive only

Eval libraries (bar pair per group):
  type2 eval library (blue) | pairwise eval library (orange)

Surrogates use SurrogateMAVENN (SQUID/MAVE-NN), same configs as lib_size_spearman.py.

Run with the squid conda environment:
  /home/nagle/miniconda3/envs/squid/bin/python mRNA_RBP/bar_surrogate_models.py

Output: mRNA_RBP/outputs/notebook_plots/bar_surrogate_models.png
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "squid-nn"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "squid-manuscript", "squid"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import tensorflow as tf

import squid.surrogate_zoo
from mRNA_RBP.generate_libraries import SEQ, STEM_PAIRS, MOTIF_POSITIONS, OUT_BASE, STEM_SIGMA
from mRNA_RBP.gt_init import MrnaRbpGroundTruth
from mRNA_RBP.evaluate import make_pairwise_dataset, make_ssm_deltas, predict_ssm

GT_KEY   = "nonlin_additive_pairwise"
INSTANCE = 0
MUT_PCT  = 10
LIB_SIZE = 20_000
NUCS     = ["A", "C", "G", "U"]
_NUCS_A  = np.array(list("ACGU"))

COLOR_TYPE2    = "#4C72B0"
COLOR_PAIRWISE = "#DD8452"
BAR_WIDTH      = 0.30

SURROGATE_CONFIGS = {
    "Additive": {
        "gpmap": "additive", "linearity": "linear",
        "regression_type": "GE", "noise": "Gaussian",
        "noise_order": 0, "reg_strength": 0.01,
    },
    "Additive\n+Pairwise": {
        "gpmap": "pairwise", "linearity": "linear",
        "regression_type": "GE", "noise": "Gaussian",
        "noise_order": 0, "reg_strength": 0.1,
    },
    "NL Additive\nonly": {
        "gpmap": "additive", "linearity": "nonlinear",
        "regression_type": "GE", "noise": "SkewedT",
        "noise_order": 2, "reg_strength": 0.01, "hidden_nodes": 50,
    },
}


def nuc_ids_to_str(nuc_ids: np.ndarray) -> np.ndarray:
    return np.array(["".join(_NUCS_A[row]) for row in nuc_ids], dtype=object)


def predict_chunked(model, x_str, chunk: int = 500) -> np.ndarray:
    parts = []
    for s in range(0, len(x_str), chunk):
        parts.append(np.asarray(model.x_to_yhat(x_str[s:s + chunk]), dtype=float).ravel())
    return np.concatenate(parts)


def _rho(y_true, y_hat):
    mask = np.isfinite(y_true) & np.isfinite(y_hat)
    if mask.sum() < 3:
        return float("nan")
    return float(spearmanr(y_true[mask], y_hat[mask])[0])


def train_surrogate(X: np.ndarray, y: np.ndarray, cfg: dict):
    N   = X.shape[0]
    bsz = max(32, min(N // 150, 2048))
    lr  = 5e-4 * min(1.0, (20_000 / N) ** 0.5)
    is_nl = cfg["linearity"] == "nonlinear"
    epochs  = 1000 if is_nl else 500
    patience = 50  if is_nl else 25

    wrapper = squid.surrogate_zoo.SurrogateMAVENN(
        X.shape, num_tasks=1,
        gpmap=cfg["gpmap"],
        regression_type=cfg["regression_type"],
        linearity=cfg["linearity"],
        noise=cfg["noise"],
        noise_order=cfg["noise_order"],
        reg_strength=cfg["reg_strength"],
        hidden_nodes=cfg.get("hidden_nodes", 50),
        alphabet=NUCS, deduplicate=True, gpu=True,
    )
    model, _, test_df = wrapper.train(
        X, y,
        learning_rate=lr, epochs=epochs, batch_size=bsz,
        early_stopping=True, patience=patience,
        restore_best_weights=True, save_dir=None, verbose=0,
    )
    return wrapper, model


def main():
    inst_dir   = os.path.join(OUT_BASE, f"instance_{INSTANCE:02d}")
    lib_path   = os.path.join(inst_dir, f"mut{MUT_PCT:02d}", f"lib_{LIB_SIZE}.npz")
    type2_path = os.path.join(inst_dir, "type2.npz")

    gt    = MrnaRbpGroundTruth(SEQ, STEM_PAIRS, MOTIF_POSITIONS,
                                seed=INSTANCE, stem_sigma=STEM_SIGMA)
    wt_oh = gt.wt_one_hot()

    # Training library
    d_lib   = np.load(lib_path)
    nuc_ids = d_lib["nuc_ids"]
    X_train = np.eye(4, dtype=np.float32)[nuc_ids]
    y_train = gt.score_all(X_train)[GT_KEY].reshape(-1, 1).astype(np.float32)

    # Eval: type2
    d_t2   = np.load(type2_path)
    X_t2   = np.eye(4, dtype=np.float32)[d_t2["nuc_ids"].astype(int)]
    y_t2   = gt.score_all(X_t2)[GT_KEY].astype(float)
    str_t2 = nuc_ids_to_str(d_t2["nuc_ids"])

    # Eval: pairwise structured lib
    x_pw, y_pw, pair_labels, nuc_combos = make_pairwise_dataset(wt_oh, gt)
    str_pw = nuc_ids_to_str(np.argmax(x_pw, axis=2).astype(np.uint8))

    # SSM baseline
    delta_W = make_ssm_deltas(wt_oh, gt)
    ssm_t2  = _rho(y_t2, predict_ssm(X_t2, delta_W))
    ssm_pw  = _rho(y_pw, predict_ssm(x_pw, delta_W))
    print(f"SSM      type2={ssm_t2:.3f}  pairwise={ssm_pw:.3f}")

    means_t2 = [ssm_t2]
    means_pw = [ssm_pw]

    for label, cfg in SURROGATE_CONFIGS.items():
        print(f"\nTraining {label.replace(chr(10), ' ')} …")
        wrapper, model = train_surrogate(X_train, y_train, cfg)

        rho_t2 = _rho(y_t2, predict_chunked(model, str_t2))
        rho_pw = _rho(y_pw, predict_chunked(model, str_pw))
        means_t2.append(rho_t2)
        means_pw.append(rho_pw)
        print(f"  type2={rho_t2:.3f}  pairwise={rho_pw:.3f}")
        tf.keras.backend.clear_session()

    # ---------------------------------------------------------------------------
    # Plot
    # ---------------------------------------------------------------------------
    labels = ["SSM"] + list(SURROGATE_CONFIGS.keys())
    x      = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(9, 4.5))
    fig.subplots_adjust(bottom=0.18)

    bars_t2 = ax.bar(x - BAR_WIDTH / 2, means_t2, BAR_WIDTH,
                     color=COLOR_TYPE2,    label="Type 2 eval library")
    bars_pw = ax.bar(x + BAR_WIDTH / 2, means_pw, BAR_WIDTH,
                     color=COLOR_PAIRWISE, label="Pairwise eval library")

    for bar in list(bars_t2) + list(bars_pw):
        h = bar.get_height()
        if np.isfinite(h):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    h + 0.01 if h >= 0 else h - 0.03,
                    f"{h:.2f}",
                    ha="center", va="bottom" if h >= 0 else "top", fontsize=7.5)

    ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Spearman ρ", fontsize=11)
    ax.set_ylim(-0.25, 1.12)
    ax.legend(fontsize=9, framealpha=0.85)
    ax.tick_params(axis="y", labelsize=9)

    out = os.path.join(os.path.dirname(__file__), "outputs",
                       "notebook_plots", "bar_surrogate_models.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(out.replace(".png", ".svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved → {out} / bar_surrogate_models.svg")


if __name__ == "__main__":
    main()
