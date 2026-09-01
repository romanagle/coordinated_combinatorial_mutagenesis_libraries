"""mRNA_RBP/scripts/experiments/negctrl_variant_differential.py

Hypothesis differential for candidate negative-control synthetic
ground-truth (GT) designs (see the explaining-patterns skill). Compares
several structurally distinct negative-control variants -- including the
currently registered `mrna_negative_control` design -- side by side:

  1. Additive (alpha) and pairwise (beta) coefficient maps for each variant.
  2. The 10% mutation-rate / 20,000-sequence random library's Spearman rho
     (surrogate-vs-GT on MAVE-NN's internal random holdout split), using the
     same "nonlinear additive + pairwise" surrogate config and training
     routine as lib_size_spearman.py.

One-off exploratory script (mRNA_RBP prototypes workflow) -- not part of
the registered oracle pipeline. Results feed a differential note under
research-notebook/Artifacts/random_lib_mutagenesis/.

Usage:
    python mRNA_RBP/scripts/experiments/negctrl_variant_differential.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import tensorflow as tf

_HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_HERE, "..", "external", "squid-nn"))
sys.path.insert(0, os.path.join(_HERE, "..", "external", "squid-manuscript", "squid"))

from mRNA_RBP.src.ground_truth import (
    soft_threshold, additive_affinity_noWT, pairwise_potts_energy,
)
from mRNA_RBP.src.gt_init import init_wc_pairwise_sparse, MRNA_RBP_DEFAULTS
from mRNA_RBP.src.seq_utils import rna_to_one_hot
from mRNA_RBP.src.sequence_configs import MSI1_SEQ, MSI1_MOTIF_POSITIONS
from mRNA_RBP.src import viz
from mRNA_RBP.scripts.pipeline.lib_size_spearman import (
    SURROGATE_CONFIGS, train_surrogate, predict_chunked, _rho, _generate_pool,
)

OUT_DIR = os.path.join(_HERE, "prototypes", "negctrl_variant_differential", "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

SEED     = 0        # matches the "instance 00" convention used elsewhere in the pipeline
MUT_PCT  = 10
LIB_SIZE = 20_000

DIST_MUT_PCTS = [5, 10, 25]   # mutation rates shown in the activity-distribution grid


class NegControlVariant:
    """Configurable negative-control GT for the hypothesis differential.

    Mirrors gt_init.MrnaNegativeControlGroundTruth's interface (alpha, beta,
    edges, wt_one_hot, wt_activity, score_all) with extra toggles -- motif
    concentration, pairwise leakage, nonlinearity -- so structurally distinct
    negative-control designs can be compared side by side without touching
    the registered oracle (gt_init.py).
    """

    def __init__(self, seq, motif_positions, *, name, use_motif=True,
                 motif_sigma=3.0, bg_sigma=0.10, l1_w=MRNA_RBP_DEFAULTS["l1_w"],
                 pairwise_sigma=0.0, pairwise_p_edge=0.0,
                 pairwise_l1=MRNA_RBP_DEFAULTS["l1_P"],
                 use_nonlin=True, a=1.0, b=1.0, c=1.5, d=0.0, seed=SEED):
        self.name = name
        self.seq = seq
        self._wt_oh = rna_to_one_hot(seq)
        L = self._wt_oh.shape[0]
        wt_idx = np.argmax(self._wt_oh, axis=1)
        rng = np.random.default_rng(seed)

        motif_set = set(motif_positions) if use_motif else set()
        self._mut_map = np.zeros((L, 3), dtype=int)
        self._W_mut = np.zeros((L, 3), dtype=np.float32)
        for pos in range(L):
            non_wt = [n for n in range(4) if n != wt_idx[pos]]
            self._mut_map[pos] = non_wt
            sigma = motif_sigma if pos in motif_set else bg_sigma
            raw = -np.abs(rng.normal(0.0, sigma, size=3).astype(np.float32))
            self._W_mut[pos] = soft_threshold(raw, l1_w)

        if pairwise_sigma > 0 and pairwise_p_edge > 0:
            edges, J = init_wc_pairwise_sparse(
                rng, self._wt_oh, seq, p_edge=pairwise_p_edge,
                sigma_P=pairwise_sigma, l1_P=pairwise_l1, edge_seed=seed,
            )
        else:
            edges = np.empty((0, 2), dtype=np.int32)
            J = np.zeros((L, L, 4, 4), dtype=np.float32)
        self._edges = edges
        self._J = J
        self.stem_pairs = [tuple(int(v) for v in e) for e in edges.tolist()]
        self.motif_positions = list(motif_positions) if use_motif else []

        self._use_nonlin = bool(use_nonlin)
        self._a, self._b, self._c, self._d = float(a), float(b), float(c), float(d)
        self._wt_raw = float(a) / (1.0 + np.exp(-float(c))) + float(d)
        self.score_key = "nonlin_additive_pairwise" if self._use_nonlin else "additive_pairwise"

    @property
    def alpha(self):
        L = self._wt_oh.shape[0]
        alpha = np.zeros((L, 4), dtype=np.float32)
        for pos in range(L):
            for k in range(3):
                alpha[pos, int(self._mut_map[pos, k])] = self._W_mut[pos, k]
        return alpha

    @property
    def beta(self):
        return {(int(i), int(j)): self._J[int(i), int(j)] for (i, j) in self._edges.tolist()}

    @property
    def edges(self):
        return self._edges

    def wt_one_hot(self):
        return self._wt_oh.copy()

    def wt_activity(self):
        return 0.0

    def _apply_nonlin(self, s):
        if not self._use_nonlin:
            return s
        raw = self._a / (1.0 + np.exp(-(self._b * s + self._c))) + self._d
        return raw - self._wt_raw

    def score_all(self, x):
        x = np.asarray(x, dtype=np.float32)
        s_add = additive_affinity_noWT(x, self._W_mut, self._mut_map, b=0.0).reshape(-1)
        s_pair = (
            pairwise_potts_energy(x, self._edges, self._J, b=0.0).reshape(-1)
            if len(self._edges) else np.zeros_like(s_add)
        )
        return {
            "additive":                 s_add,
            "additive_pairwise":        s_add + s_pair,
            "nonlin_additive":          self._apply_nonlin(s_add),
            "nonlin_additive_pairwise": self._apply_nonlin(s_add + s_pair),
        }

    def __call__(self, x):
        return self.score_all(x)[self.score_key]

    def __repr__(self):
        return (f"NegControlVariant({self.name}, motif={len(self.motif_positions)}, "
                f"pairwise_edges={len(self.stem_pairs)}, nonlin={self._use_nonlin})")


# ---------------------------------------------------------------------------
# The differential: five candidate negative-control designs spanning
# domain / sampling-coverage / statistical-artifact hypotheses about what
# makes a valid negative control.
# ---------------------------------------------------------------------------

VARIANT_SPECS = [
    dict(name="V0_motif_only_current_design",
         use_motif=True,  motif_sigma=3.0, bg_sigma=0.10,
         pairwise_sigma=0.0, pairwise_p_edge=0.0, use_nonlin=True),
    dict(name="V1_null_no_motif",
         use_motif=False, motif_sigma=3.0, bg_sigma=0.10,
         pairwise_sigma=0.0, pairwise_p_edge=0.0, use_nonlin=True),
    dict(name="V2_motif_plus_leaky_pairwise",
         use_motif=True,  motif_sigma=3.0, bg_sigma=0.10,
         pairwise_sigma=0.3, pairwise_p_edge=0.15, use_nonlin=True),
    dict(name="V3_motif_linear_no_nonlin",
         use_motif=True,  motif_sigma=3.0, bg_sigma=0.10,
         pairwise_sigma=0.0, pairwise_p_edge=0.0, use_nonlin=False),
    dict(name="V4_diffuse_moderate_background",
         use_motif=False, motif_sigma=3.0, bg_sigma=0.5,
         pairwise_sigma=0.0, pairwise_p_edge=0.0, use_nonlin=True),
]

VARIANT_RATIONALE = {
    "V0_motif_only_current_design":
        "Domain: currently registered mrna_negative_control design -- HuR-like single "
        "privileged motif, near-neutral background, no pairwise. If-true: a concentrated "
        "motif with weak background still recovers near-ceiling on random holdout. "
        "If-false: motif concentration alone depresses rho_rand.",
    "V1_null_no_motif":
        "Sampling/coverage: same weak background scale as V0 everywhere, no privileged "
        "region at all. If-true: rho_rand matches V0 (signal concentration doesn't matter "
        "for random-holdout recoverability). If-false: removing the motif changes rho_rand, "
        "implicating concentration rather than mere presence of structure.",
    "V2_motif_plus_leaky_pairwise":
        "Domain (mechanism boundary): V0 plus weak, WC-compatible-but-not-stem-restricted "
        "pairwise edges at low scale/density -- recreates the original (failed) unstructured "
        "negative control's leaky topology layered on a motif-only background. If-true: even "
        "weak non-structural pairwise coupling depresses rho_rand relative to V0. If-false: "
        "a pairwise-capable surrogate absorbs it and rho_rand stays high.",
    "V3_motif_linear_no_nonlin":
        "Statistical artifact/methodological: V0 without the sigmoid squashing. If-true: "
        "removing the nonlinearity changes rho_rand relative to V0, implicating the "
        "nonlinearity itself. If-false: rho_rand is unchanged, ruling out the nonlinearity "
        "as a random-holdout recoverability factor (as opposed to activity-balanced "
        "mutation-count extrapolation, where it was already implicated separately).",
    "V4_diffuse_moderate_background":
        "Sampling/coverage: no privileged region, background scale matches the structured "
        "GT's own bg_sigma=0.5 (denser/stronger than V1). If-true: rho_rand tracks total "
        "additive signal magnitude, not concentration. If-false: rho_rand matches V0/V1 "
        "regardless of background scale.",
}


def build_variants():
    return [NegControlVariant(MSI1_SEQ, MSI1_MOTIF_POSITIONS, seed=SEED, **spec)
            for spec in VARIANT_SPECS]


# ---------------------------------------------------------------------------
# Coefficient maps
# ---------------------------------------------------------------------------

def make_coefficient_figures(variants):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    L = len(MSI1_SEQ)

    # Individual full-detail figures via the shared plotting code (viz.py).
    for v in variants:
        fig = viz.plot_coefficients(
            v.alpha, v.beta, L,
            stem_pairs=v.stem_pairs, motif_positions=v.motif_positions,
            title=v.name,
        )
        fig.savefig(os.path.join(OUT_DIR, f"coef_{v.name}.png"), dpi=130, bbox_inches="tight")
        plt.close(fig)

    # Composite grid: one row per variant, additive (left) + pairwise summary (right).
    n = len(variants)
    fig, axes = plt.subplots(n, 2, figsize=(11, 2.6 * n),
                              gridspec_kw={"width_ratios": [3, 1], "hspace": 0.6, "wspace": 0.25})
    NUCS = ["A", "U", "G", "C"]
    for row, v in enumerate(variants):
        ax_add, ax_pair = axes[row, 0], axes[row, 1]
        alpha = v.alpha
        disp = alpha.T[[0, 3, 2, 1], :]
        vmax_add = max(float(np.abs(disp).max()), 1e-6)
        im = ax_add.imshow(disp, aspect="auto", cmap="RdBu", vmin=-vmax_add, vmax=vmax_add,
                            origin="lower", interpolation="nearest")
        for pos in v.motif_positions:
            ax_add.axvspan(pos - 0.5, pos + 0.5, color="#AED6F1", alpha=0.5, zorder=0)
        ax_add.set_yticks(range(4)); ax_add.set_yticklabels(NUCS, fontsize=7)
        ax_add.set_xlabel("Position", fontsize=8)
        ax_add.set_title(f"{v.name}  (score_key={v.score_key})", fontsize=9, loc="left")
        fig.colorbar(im, ax=ax_add, fraction=0.02, pad=0.01)

        beta_pos = np.zeros((L, L), dtype=np.float32)
        for (i, j), M in v.beta.items():
            flat = np.asarray(M).ravel()
            beta_pos[i, j] = flat[np.argmax(np.abs(flat))] if flat.size else 0.0
        if v.beta:
            vmax_p = max(float(np.abs(beta_pos).max()), 1e-6)
            im2 = ax_pair.imshow(beta_pos, cmap="RdBu", vmin=-vmax_p, vmax=vmax_p,
                                  origin="upper", interpolation="nearest")
            fig.colorbar(im2, ax=ax_pair, fraction=0.046, pad=0.04)
            ax_pair.set_title(f"pairwise ({len(v.stem_pairs)} edges)", fontsize=8)
        else:
            ax_pair.text(0.5, 0.5, "no pairwise\n(edges = 0)", ha="center", va="center", fontsize=8)
            ax_pair.set_xticks([]); ax_pair.set_yticks([])
        ax_pair.set_xlabel("j", fontsize=7); ax_pair.set_ylabel("i", fontsize=7)

    fig.suptitle("Negative-control variant differential -- additive + pairwise coefficients",
                 y=1.0, fontsize=12)
    out_path = os.path.join(OUT_DIR, "negctrl_variant_differential_coefficients.png")
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Activity-score distributions: variants (rows) x mutation rates (cols)
# ---------------------------------------------------------------------------

def make_activity_distribution_grid(variants, mut_pcts=DIST_MUT_PCTS, lib_size=LIB_SIZE):
    """5 (variants) x 3 (mutation rates) grid of random-library GT score
    histograms -- no surrogate training, GT scoring only."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_rows, n_cols = len(variants), len(mut_pcts)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.6 * n_cols, 2.4 * n_rows),
                              sharex=False, gridspec_kw={"hspace": 0.45, "wspace": 0.25})

    for row, v in enumerate(variants):
        wt_oh = v.wt_one_hot()
        L = wt_oh.shape[0]
        for col, pct in enumerate(mut_pcts):
            mut_count = max(1, round(pct * L / 100))
            rng = np.random.default_rng(SEED * 10_000 + pct * 100 + lib_size)
            nuc_ids = _generate_pool(wt_oh, lib_size, mut_count, rng)
            X = np.eye(4, dtype=np.float32)[nuc_ids]
            y = v.score_all(X)[v.score_key]

            ax = axes[row, col]
            ax.hist(y, bins=60, color="#4C72B0", alpha=0.75,
                    edgecolor="white", linewidth=0.3)
            ax.axvline(0.0, color="black", linewidth=1.2, linestyle="--", label="WT")
            ax.tick_params(labelsize=7)
            if row == 0:
                ax.set_title(f"{pct}% mutation rate", fontsize=10)
            if col == 0:
                ax.set_ylabel(v.name, fontsize=8)
            if row == n_rows - 1:
                ax.set_xlabel("Activity score", fontsize=8)
            print(f"  [dist] {v.name}  mut{pct}%  n={len(y)}  "
                  f"mean={float(np.mean(y)):+.3f}  std={float(np.std(y)):.3f}")

    fig.suptitle(f"Negative-control variant activity distributions "
                 f"(random libraries, lib_size={lib_size:,})", y=1.0, fontsize=12)
    out_path = os.path.join(OUT_DIR, "negctrl_variant_activity_distributions_5x3.png")
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# 10% mutation-rate / 20,000-sequence random-holdout Spearman rho
# ---------------------------------------------------------------------------

def run_spearman(variants):
    results = {}
    for v in variants:
        print(f"\n{'=' * 70}\n  {v.name}  (score_key={v.score_key})\n{'=' * 70}")
        wt_oh = v.wt_one_hot()
        L = wt_oh.shape[0]
        mut_count = max(1, round(MUT_PCT * L / 100))
        rng = np.random.default_rng(SEED * 10_000 + MUT_PCT * 100 + LIB_SIZE)
        nuc_ids = _generate_pool(wt_oh, LIB_SIZE, mut_count, rng)
        X = np.eye(4, dtype=np.float32)[nuc_ids]
        y = v.score_all(X)[v.score_key].reshape(-1, 1)

        cfg = SURROGATE_CONFIGS["nonlinear additive + pairwise"]
        try:
            wrapper, model, test_df = train_surrogate(X, y, cfg)
        except Exception as e:
            print(f"  [FAIL] {e}")
            tf.keras.backend.clear_session()
            results[v.name] = {"rho_rand": None, "n_train": int(len(X)), "error": str(e)}
            continue

        x_col = "x" if "x" in test_df.columns else "X"
        y_col = "y" if "y" in test_df.columns else next(c for c in test_df.columns if c.startswith("y"))
        y_test = np.asarray(test_df[y_col], dtype=float).ravel()
        yhat_test = predict_chunked(model, np.asarray(test_df[x_col]))
        rho_rand = _rho(y_test, yhat_test)
        tf.keras.backend.clear_session()

        print(f"  n_train={len(X)}  rho_rand(mut{MUT_PCT}%, lib{LIB_SIZE})={rho_rand:+.4f}")
        results[v.name] = {"rho_rand": rho_rand, "n_train": int(len(X))}
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_coef", action="store_true",
                        help="Skip the coefficient-map figures")
    parser.add_argument("--skip_spearman", action="store_true",
                        help="Skip surrogate training / rho_rand (fast: only distributions/coefs)")
    parser.add_argument("--skip_dist", action="store_true",
                        help="Skip the variants x mutation-rate activity-distribution grid")
    args = parser.parse_args()

    variants = build_variants()

    if not args.skip_coef:
        coef_fig = make_coefficient_figures(variants)
        print(f"[coef] composite figure -> {coef_fig}")

    if not args.skip_dist:
        dist_fig = make_activity_distribution_grid(variants)
        print(f"[dist] composite figure -> {dist_fig}")

    if args.skip_spearman:
        print("[skip] surrogate training / rho_rand")
        return

    results = run_spearman(variants)

    out = {
        "seed": SEED, "mut_pct": MUT_PCT, "lib_size": LIB_SIZE,
        "surrogate_config": "nonlinear additive + pairwise",
        "variants": {},
    }
    for spec in VARIANT_SPECS:
        name = spec["name"]
        out["variants"][name] = {
            **results[name],
            "rationale": VARIANT_RATIONALE[name],
            "spec": {k: v for k, v in spec.items() if k != "name"},
        }
    out_json = os.path.join(OUT_DIR, "negctrl_variant_differential_results.json")
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[done] results -> {out_json}")

    print(f"\n{'Variant':40s}  {'rho_rand':>10s}")
    for spec in VARIANT_SPECS:
        name = spec["name"]
        r = results[name]["rho_rand"]
        print(f"{name:40s}  {r:+.4f}" if r is not None else f"{name:40s}  FAILED")


if __name__ == "__main__":
    main()
