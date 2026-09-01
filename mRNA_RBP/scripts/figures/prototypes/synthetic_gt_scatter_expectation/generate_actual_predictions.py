#!/usr/bin/env python3
"""Freeze matched mixed-count predictions for the Synthetic GT controls."""

from pathlib import Path
import sys

import numpy as np
import tensorflow as tf
from scipy.stats import pearsonr, spearmanr


HERE = Path(__file__).resolve().parents[5] / "mRNA_RBP" / "prototypes" / "synthetic_gt_scatter_expectation"
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "external" / "squid-nn"))
sys.path.insert(0, str(REPO / "external" / "squid-manuscript" / "squid"))

from mRNA_RBP.scripts.pipeline.generate_libraries import sample_unique_mutants
from mRNA_RBP.scripts.pipeline.generate_scatter_by_mutcount_predictions import run_one
from mRNA_RBP.src.ground_truth import uniformize_by_histogram
from mRNA_RBP.scripts.pipeline.lib_size_spearman import (
    SURROGATE_CONFIGS,
    nuc_ids_to_str,
    predict_chunked,
    train_surrogate,
)
from mRNA_RBP.src.oracles import build_oracle, sequence_config_for_oracle


OUTPUT = HERE / "outputs" / "synthetic_gt_control_predictions.npz"
SCORE_KEY = "nonlin_additive_pairwise"
MODEL_NAME = "nonlinear additive + pairwise"
SEED = 20260809
MUTATION_COUNTS = (3, 5, 7, 15)
CANDIDATES_PER_COUNT = 50_000
ACTIVITY_TARGET = 20_000


def make_oracle(name):
    seq, stem_pairs, motif_positions = sequence_config_for_oracle(name)
    return build_oracle(
        name,
        seq=seq,
        stem_pairs=stem_pairs,
        motif_positions=motif_positions,
        seed=0,
        stem_sigma=3.0,
    )


def candidate_pool(wt, rng):
    ids = []
    labels = []
    for count in MUTATION_COUNTS:
        block = sample_unique_mutants(wt, count, CANDIDATES_PER_COUNT, rng)
        ids.append(block)
        labels.append(np.full(len(block), count, dtype=np.uint8))
    ids = np.concatenate(ids)
    labels = np.concatenate(labels)
    _, keep = np.unique(ids, axis=0, return_index=True)
    keep.sort()
    return ids[keep], labels[keep]


def activity_balanced(oracle, ids, labels, seed):
    scores = oracle.score_all(np.eye(4, dtype=np.float32)[ids])[SCORE_KEY].astype(float)
    _, keep = uniformize_by_histogram(
        scores,
        X=None,
        n_bins=200,
        clip_lo=1,
        clip_hi=99,
        target_n=ACTIVITY_TARGET,
        seed=seed,
    )
    return ids[keep], labels[keep], scores[keep]


def fixed_10pct_training_pool(oracle, eval_ids, rng):
    eval_rows = {row.tobytes() for row in eval_ids}
    candidates = sample_unique_mutants(oracle.wt_one_hot(), 4, 40_000, rng)
    keep = np.array([row.tobytes() not in eval_rows for row in candidates])
    ids = candidates[keep][:20_000]
    if len(ids) != 20_000:
        raise RuntimeError(f"Only {len(ids)} non-evaluation four-mutant training rows")
    ids = ids[rng.permutation(len(ids))]
    scores = oracle.score_all(np.eye(4, dtype=np.float32)[ids])[SCORE_KEY].astype(float)
    return ids, scores


def metrics(y, yhat):
    return {
        "rho": float(spearmanr(y, yhat)[0]),
        "r2": float(pearsonr(y, yhat)[0] ** 2),
        "rmse": float(np.sqrt(np.mean((y - yhat) ** 2))),
    }


def run_control(name, oracle, candidate_ids, candidate_labels, offset):
    eval_ids, eval_labels, y_activity = activity_balanced(
        oracle, candidate_ids, candidate_labels, SEED + 100 + offset
    )
    train_ids, train_scores = fixed_10pct_training_pool(
        oracle, eval_ids, np.random.default_rng(SEED + 200 + offset)
    )
    cfg = SURROGATE_CONFIGS[MODEL_NAME]
    print(f"training {name} fixed-10% surrogate (N={len(train_ids):,})")
    _, model, test_df = train_surrogate(
        np.eye(4, dtype=np.float32)[train_ids], train_scores.reshape(-1, 1), cfg
    )
    x_col = "x" if "x" in test_df.columns else "X"
    y_col = "y" if "y" in test_df.columns else next(
        col for col in test_df.columns if col.startswith("y")
    )
    y_random = np.asarray(test_df[y_col], dtype=float).ravel()
    yhat_random = predict_chunked(model, np.asarray(test_df[x_col]))
    yhat_activity = predict_chunked(model, nuc_ids_to_str(eval_ids))

    if spearmanr(y_random, yhat_random)[0] < 0:
        yhat_random = -yhat_random
        yhat_activity = -yhat_activity

    random_metrics = metrics(y_random, yhat_random)
    activity_metrics = metrics(y_activity, yhat_activity)
    tf.keras.backend.clear_session()
    print(
        f"  random rho={random_metrics['rho']:.3f}; "
        f"activity-balanced rho={activity_metrics['rho']:.3f}"
    )
    return {
        "y_rand_10": y_random,
        "yhat_rand_10": yhat_random,
        "y_activity_10": y_activity,
        "yhat_activity_10": yhat_activity,
        "rate_labels_10": eval_labels,
        "rho_rand_10": np.asarray(random_metrics["rho"]),
        "rho_activity_10": np.asarray(activity_metrics["rho"]),
        "r2_rand_10": np.asarray(random_metrics["r2"]),
        "r2_activity_10": np.asarray(activity_metrics["r2"]),
        "rmse_rand_10": np.asarray(random_metrics["rmse"]),
        "rmse_activity_10": np.asarray(activity_metrics["rmse"]),
    }


def main():
    positive = make_oracle("mrna")
    negative = make_oracle("negative_control")
    if len(negative.edges):
        raise AssertionError("Negative control must contain no pairwise edges")
    if not negative.motif_positions:
        raise AssertionError("Negative control must retain a privileged motif")

    candidate_ids, candidate_labels = candidate_pool(
        negative.wt_one_hot(), np.random.default_rng(SEED)
    )
    payload = {
        "mutation_rate": np.asarray(10, dtype=np.int16),
        "training_mutation_count": np.asarray(4, dtype=np.int16),
        "library_size": np.asarray(20_000, dtype=np.int32),
        "seed": np.asarray(SEED, dtype=np.int32),
        "model_name": np.asarray(MODEL_NAME),
        "negative_control": np.asarray(
            "motif additive; near-neutral background (sigma=0.10); no pairwise"
        ),
    }
    # The experiment changes only the negative control. Preserve the frozen
    # positive predictions exactly when regenerating this artifact.
    if not OUTPUT.is_file():
        raise FileNotFoundError(
            f"Missing {OUTPUT}; cannot preserve the existing positive control"
        )
    frozen = np.load(OUTPUT)
    for key in frozen.files:
        if key.startswith("positive_"):
            payload[key] = frozen[key]

    np.random.seed(SEED + 1)
    tf.random.set_seed(SEED + 1)
    negative_result = run_control(
        "negative", negative, candidate_ids, candidate_labels, 1
    )
    for key, value in negative_result.items():
        payload[f"negative_{key}"] = value

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(OUTPUT), **payload)
    print("Saved actual control predictions -> {}".format(OUTPUT))


if __name__ == "__main__":
    main()
