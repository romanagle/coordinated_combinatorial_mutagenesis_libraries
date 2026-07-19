import argparse
import sys
sys.path.insert(0, '/home/nagle/final_version/squid-nn')
import squid
import tensorflow as tf
import os
import numpy as np
import pandas as pd
from tensorflow import keras
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
import mavenn
from scipy.stats import spearmanr, pearsonr
from typing import Optional
from collections import defaultdict
from matplotlib.patches import Rectangle

sys.path.append('/home/nagle/final_version/squid-manuscript/squid')
sys.path.append('/home/nagle/final_version/squid-nn/squid')
sys.path.append('/home/nagle/final_version/residualbind')

from residualbind import ResidualBind
import helper, explain, dinuc_shuffle
from prediction import paired_positions, predict_ss

# Local modules
from seq_utils import rna_to_one_hot, remove_padding, onehot_to_seq, eval_library, write_eval_library_txt
from ground_truth import (
    additive_affinity_noWT, pairwise_adjacent_noWT, pairwise_potts_energy,
    apply_global_nonlin, init_additive_noWT, init_pairwise_adjacent_noWT,
    init_pairwise_potts_optionA, compute_gt_scores_for_library_potts,
    uniformize_by_histogram,
)
from stats_utils import spearman_on_testdf
from epistasis import ism_double_tensor, max_with_sign, epistasis_tensor, epistasis_map
from plotting import (
    plot_y_vs_yhat, plot_random_vs_eval_per_gt, plot_random_library_distributions_three_nonlins,
    plot_epistasis_maxpooled_like, plot_epistasis_map, plot_raw_pairs_as_LxL_cells_with_4x4_inside,
    plot_gt_weight_dists_per_function, spearman_bars_per_surrogate, spearmanavg_by_cfg,
    plot_kde_grid_surrogate_vs_gt, plot_coef_dists_per_gt, plot_16_y_vs_yhat_grid,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

normalization = 'log_norm'   # 'log_norm' or 'clip_norm'
ss_type = 'seq'              # 'seq', 'pu', or 'struct'
NUCS = ['A', 'C', 'G', 'U']

SURROGATE_CONFIGS = {
    # no-GE (linear)
    "additive": {
        "gpmap": "additive", "linearity": "linear",
        "regression_type": "GE", "noise": "Gaussian",
        "noise_order": 0, "reg_strength": 12,
    },
    "pairwise": {
        "gpmap": "pairwise", "linearity": "linear",
        "regression_type": "GE", "noise": "Gaussian",
        "noise_order": 0, "reg_strength": 0.1,
    },
    # GE (nonlinear) — 50 hidden nodes matches paper's overparameterised GE nonlinearity;
    # noise_order=2 → ge_heteroskedasticity_order=2 (quadratic noise variance in phi)
    "additive_GE": {
        "gpmap": "additive", "linearity": "nonlinear",
        "regression_type": "GE", "noise": "SkewedT",
        "noise_order": 2, "reg_strength": 12, "hidden_nodes": 50,
    },
    "pairwise_GE": {
        "gpmap": "pairwise", "linearity": "nonlinear",
        "regression_type": "GE", "noise": "SkewedT",
        "noise_order": 2, "reg_strength": 0.1, "hidden_nodes": 50,
    },
}


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Run coordinated combinatorial mutagenesis using ResidualBind and RNACompete dataset"
)
parser.add_argument("--seq", type=str)
parser.add_argument("--activity", type=str, help='TESTING low vs high activity')
parser.add_argument("--mut_rate", type=int, default=4,
                    help='Integer value for how many positions you want to mutate in a given sequence')
parser.add_argument("--lib_size", type=int, default=20000,
                    help='How many mutagenized sequences you want in your library')
parser.add_argument("--subdir", type=str, help='Subdirectory inside outputs')
parser.add_argument("--experiment", type=str, help='ex: RNCMPT00111 or RNCMPT00042')
parser.add_argument("--numrounds", type=int, help='Number of repeated runs')

args = parser.parse_args()


# ---------------------------------------------------------------------------
# Paths and model loading
# ---------------------------------------------------------------------------

data_path = Path.home() / 'residualbind' / 'data' / 'RNAcompete_2013' / 'rnacompete2013.h5'
results_path = helper.make_directory('/home/nagle/residualbind/results', 'rnacompete_2013')
save_path = '/home/nagle/residualbind/weights/log_norm_seq'
plot_path = helper.make_directory(save_path, 'FINAL')

experiment = args.experiment
rbp_index  = helper.find_experiment_index(data_path, experiment)

train, valid, test = helper.load_rnacompete_data(
    data_path, ss_type='seq', normalization=normalization, rbp_index=rbp_index
)

input_shape  = list(train['inputs'].shape)[1:]
num_class    = 1
weights_path = os.path.join(save_path, experiment + '_weights.hdf5')

residbind = ResidualBind(input_shape, num_class, weights_path)
residbind.load_weights()
print('Analyzing: ' + experiment)

# ---------------------------------------------------------------------------
# CLI args → local variables
# ---------------------------------------------------------------------------

input_seq = args.seq
num_muts  = args.mut_rate
lib_size  = args.lib_size
activity  = args.activity
subdir    = args.subdir
mut_rate  = num_muts / 41
numrounds = args.numrounds

if input_seq is None:
    raise ValueError("Please provide --seq")
elif len(input_seq) > 41:
    raise ValueError("Sequence needs to be smaller than 41 nucleotides.")
elif num_muts > len(input_seq):
    raise ValueError("You can't mutate more positions than the length of your input sequence.")
elif not all(base in "AUGC" for base in input_seq):
    print("Invalid RNA sequence, only stick to letters A, U, G, C.")

plot_path = helper.make_directory(save_path, 'FINAL')


# ---------------------------------------------------------------------------
# Global sequence state
# ---------------------------------------------------------------------------

oh_seq         = rna_to_one_hot(input_seq)
no_padding_seq, _ = remove_padding(oh_seq)
print(no_padding_seq.shape)

padding_amt = 41 - oh_seq.shape[0]   # residualbind requires 41 nt
seq_length  = no_padding_seq.shape[0]
mut_window  = [0, seq_length]

GLOBAL_MASK   = np.tril(np.ones((seq_length, seq_length), dtype=bool), -1)
GLOBAL_MASK_T = GLOBAL_MASK.T


# ---------------------------------------------------------------------------
# Plotting helpers that use global paths
# ---------------------------------------------------------------------------

def plot_fig(listofscores, iteration, path=None):
    if path is None:
        path = (f"/home/nagle/final_version/outputs/{subdir}/{activity}"
                f"/dist_of_preds/1214/iter_{iteration}_pred_binding_affinity_distribution.png")
    print(len(listofscores))
    plt.figure(figsize=(8, 5))
    plt.hist(listofscores, density=True, bins=100, edgecolor='black', alpha=0.7)
    plt.title("Binding Affinity Score Prediction Distribution after Random Mutagenesis")
    plt.xlabel("Binding Affinity score")
    plt.ylabel("Density")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)


def y_yhat_fig(y, y_hat, iteration):
    y     = np.asarray(y,     dtype=float).ravel()
    y_hat = np.asarray(y_hat, dtype=float).ravel()
    mask  = ~np.isnan(y) & ~np.isnan(y_hat)
    y, y_hat = y[mask], y_hat[mask]
    if y.size == 0 or y_hat.size == 0 or y.size != y_hat.size:
        raise ValueError(f"Bad inputs: y size={y.size}, y_hat size={y_hat.size}")

    r, _ = pearsonr(y, y_hat)
    plt.figure(figsize=(8, 5))
    plt.scatter(y, y_hat, alpha=0.7)
    plt.xlabel("y (True Values)"); plt.ylabel("ŷ (Predicted Values)")
    plt.title("y vs ŷ Scatter Plot")
    min_val = min(y.min(), y_hat.min()); max_val = max(y.max(), y_hat.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="Ideal Fit")
    plt.text(0.05, 0.95, f"Pearson r = {r:.3f}", transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top')
    plt.legend()
    plt.savefig(
        f"/home/nagle/final_version/outputs/{subdir}/{activity}"
        f"/dist_of_preds/1214/iter_{iteration}_y_yhat.png", dpi=300
    )
    plt.close()


# ---------------------------------------------------------------------------
# Model prediction
# ---------------------------------------------------------------------------

def predictor(input_library, iteration, plot=True, true_preds=None):
    """Add padding, run residbind, optionally plot."""
    print(f"Here is the size of the input library: {input_library.shape}")
    beg_padding = np.zeros((input_library.shape[0], padding_amt // 2, 4))
    end_padding = np.zeros((input_library.shape[0], padding_amt // 2 + (padding_amt % 2), 4))
    seqs_with_padding = np.concatenate([beg_padding, input_library, end_padding], axis=1)
    print(f"Here is the size of the input library: {seqs_with_padding.shape}")
    prediction = residbind.predict(seqs_with_padding)

    if plot:
        plot_fig(prediction, iteration)
    if true_preds is not None and not true_preds.empty:
        y_yhat_fig(prediction, true_preds, iteration)

    return seqs_with_padding, prediction


def predictgivensequence(inputseq):
    oh      = rna_to_one_hot(inputseq)
    no_pad, _ = remove_padding(oh)
    pad_amt = 41 - oh.shape[0]
    _, pred = predictor(np.expand_dims(no_pad, axis=0), "test1", plot=False)
    return pred


# ---------------------------------------------------------------------------
# Library generation
# ---------------------------------------------------------------------------

def generateLibrary(paired_position_list, input_seq_str, mode):
    pred_generator = squid.predictor.CustomPredictor(
        pred_fun=residbind.predict, reduce_fun="name", batch_size=512
    )
    padded_indices = [idx for idx, i in enumerate(no_padding_seq) if np.array_equal(i, np.zeros(4))]
    rate = 4

    if mode == "random":
        mut_generator = squid.mutagenizer.RandomMutagenesis(mut_rate=rate, uniform=False)
    else:
        mut_generator = squid.mutagenizer.CustomMutagenesis(mut_rate=rate)

    oh          = rna_to_one_hot(input_seq_str)
    no_pad, _   = remove_padding(oh)
    s_len       = no_pad.shape[0]
    mut_win     = [0, s_len]
    pad_amt     = 41 - oh.shape[0]

    mave = squid.mave.InSilicoMAVE(
        mut_generator, pred_generator, s_len,
        mut_window=mut_win, paired_position_list=paired_position_list
    )
    x_mut, y_mut = mave.generate(no_pad, padding_amt=pad_amt, num_sim=lib_size)
    return x_mut, y_mut


# ---------------------------------------------------------------------------
# In-silico saturated mutagenesis
# ---------------------------------------------------------------------------

def ism(input_seq_oh, dot_bracket, outputdir):
    """Single-site ISM: scores a 4×L matrix of substitutions.

    input_seq_oh : (L,4) one-hot
    dot_bracket  : str of length L
    """
    seq_length_local = input_seq_oh.shape[0]
    score_matrix = np.empty((4, seq_length_local))
    mut_oh = {
        0: np.array([1, 0, 0, 0]), 1: np.array([0, 1, 0, 0]),
        2: np.array([0, 0, 1, 0]), 3: np.array([0, 0, 0, 1]),
    }

    for j in range(seq_length_local):
        for k in range(4):
            new_seq = np.concatenate([
                input_seq_oh[0:j],
                np.expand_dims(mut_oh[k], axis=0),
                input_seq_oh[j + 1:]
            ])
            _, prediction = predictor(np.expand_dims(new_seq, axis=0), None, plot=False)
            score_matrix[k][j] = prediction.item()

    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(score_matrix, aspect='auto', cmap='viridis')
    ax.set_yticks([0, 1, 2, 3]); ax.set_yticklabels(['A', 'C', 'G', 'U'])
    ax.set_xticks(np.arange(seq_length_local)); ax.set_xticklabels(list(dot_bracket))
    ax.set_xlabel("Dot-bracket position"); ax.set_ylabel("Nucleotide")
    ax.set_title("4×L ISM Heatmap")
    plt.colorbar(im, ax=ax, label="Binding Affinity Score")

    for j in range(seq_length_local):
        wt_idx = np.argmax(input_seq_oh[j])
        ax.add_patch(Rectangle(
            (j - 0.5, wt_idx - 0.5), 1, 1, linewidth=1.5, edgecolor='red', facecolor='none'
        ))

    plt.tight_layout()
    plt.savefig(f"{outputdir}/heatmap_output_with_boxes_middle.png", dpi=300)
    return score_matrix


# ---------------------------------------------------------------------------
# Epistasis-map helper (uses global paths)
# ---------------------------------------------------------------------------

def ep_map(iteration, finalmin=None, finalmax=None):
    weights = np.load(
        f'/home/nagle/final_version/outputs/{subdir}/{activity}'
        f'/pairwise_weights/1214/weights_epoch_{iteration}.npy'
    )
    L = weights.shape[0]
    max_pooled_arr = np.empty((L, L))
    for i in range(L):
        for j in range(L):
            intermediate = [weights[i][k][j][l] for k in range(4) for l in range(4)]
            max_pooled_arr[i][j] = max(intermediate, key=abs)

    mask        = np.tril(np.ones_like(max_pooled_arr, dtype=bool), -1)
    flipped_arr = max_pooled_arr.T
    flipped_mask = mask.T
    min_val, max_val = np.nanmin(flipped_arr), np.nanmax(flipped_arr)

    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(flipped_arr, mask=flipped_mask, cmap='seismic',
                     square=True, center=0, vmin=min_val, vmax=max_val)
    ax.xaxis.tick_bottom(); ax.yaxis.tick_left()

    threshold = 0.6
    for i in range(max_pooled_arr.shape[0]):
        for j in range(max_pooled_arr.shape[1]):
            if np.abs(max_pooled_arr[i, j]) > threshold:
                rect = Rectangle((i, j), 1, 1, fill=False, edgecolor='black', linewidth=2)
                if np.abs(i - j) > 7:
                    rect = Rectangle((i, j), 1, 1, fill=False, edgecolor='yellow', linewidth=4)
                ax.add_patch(rect)

    plt.tight_layout()
    plt.savefig(
        f"/home/nagle/final_version/outputs/{subdir}/{activity}/new_ep_maps/1214/NEWepmap{iteration}.png"
    )
    return flipped_arr, flipped_mask


# ---------------------------------------------------------------------------
# ISM evolution (bottom-percentile selection)
# ---------------------------------------------------------------------------

def ism_evol(input_seqs):
    """3-mutation ISM step: exhaustively try each single mutation, keep top."""
    n_seqs = input_seqs.shape[0]
    print(f"Number of sequences predicted: {n_seqs * 41 * 3}\n")

    mut_oh = {
        0: np.array([1, 0, 0, 0]), 1: np.array([0, 1, 0, 0]),
        2: np.array([0, 0, 1, 0]), 3: np.array([0, 0, 0, 1]),
    }
    mut_library = []
    for i in range(input_seqs.shape[0]):
        mut_seqs = []
        for j in range(input_seqs.shape[1]):
            curr_idx = list(mut_oh.keys()).index(
                [k for k, v in mut_oh.items() if np.array_equal(v, input_seqs[i][j])][0]
            )
            for k in range(1, 4):
                next_index   = (curr_idx + k) % 4
                inserted_mut = np.array(mut_oh[next_index]).reshape((1, 4))
                new_seq      = np.concatenate([input_seqs[i][0:j], inserted_mut, input_seqs[i][j + 1:]])
                mut_seqs.append(new_seq)
        mut_library.extend(mut_seqs)

    new_lib = np.array(mut_library)
    seqs_with_predictions = []
    seqs_with_padding, prediction = predictor(new_lib, None, plot=False)
    for seqs, scores in zip(seqs_with_padding, prediction):
        seqs_with_predictions.append((remove_padding(seqs)[0], scores.item()))

    toppreds = sorted(seqs_with_predictions, key=lambda k: k[1], reverse=True)[:n_seqs]
    return toppreds


def bottom_percentile_iter(input_library, iteration):
    seqs_with_padding, prediction = predictor(input_library, iteration, plot=False)
    percentile          = 20
    bottompercentile    = np.percentile(prediction, percentile)
    bottompercentileseqs, bottompercentilescores = [], []
    otherseqs, otherscores = [], []

    for seq, score in zip(input_library, prediction):
        if score <= bottompercentile:
            bottompercentileseqs.append(seq); bottompercentilescores.append(score)
        else:
            otherseqs.append(seq); otherscores.append(score.item())

    print(f"\nLength of the bottom {percentile}th percentile seqs: {len(bottompercentileseqs)}")
    print(f"\nLength of the other {100 - percentile}th percentile seqs: {len(otherseqs)}\n")

    no_pad_bottomseqs = [remove_padding(i)[0] for i in bottompercentileseqs]
    new_lib_list      = ism_evol(np.array(no_pad_bottomseqs))
    newpercentilescores = [k[1] for k in new_lib_list]
    newpercentileseqs   = [k[0] for k in new_lib_list]

    print(f"Average of the bottom {percentile}th percentile scores before ISM: "
          f"{np.mean(bottompercentilescores)}")
    print(f"Average of the bottom {percentile}th percentile scores after ISM: "
          f"{np.mean(newpercentilescores)}")

    otherscores.extend(newpercentilescores)
    plot_fig(np.array(otherscores), iteration)

    return (np.concatenate([np.array(otherseqs), np.array(newpercentileseqs)], axis=0),
            np.array(otherscores))


# ---------------------------------------------------------------------------
# Secondary-structure-guided mutagenesis
# ---------------------------------------------------------------------------

def mutagenesis_pipeline(input, output_dir, num_mut_seqs):
    from ink import deep_sea
    if input.ndim != 2:
        input = np.squeeze(input)
    cleaned_seq = "".join(onehot_to_seq(np.expand_dims(input, axis=0)))

    print("\nStep 2: New library generated from secondary structure prior.\n")
    predict_ss(cleaned_seq, output_dir)

    seq_file_name     = f"{cleaned_seq}.txt"
    file_absolute_path = os.path.join(output_dir, seq_file_name)

    if os.path.isfile(file_absolute_path):
        with open(file_absolute_path, 'r') as f:
            content             = f.read()
            paired_position_list = paired_positions(content)

    print(f"Here are the positions that are paired: {paired_position_list}")
    mutated_seq, mut_index = deep_sea(
        no_padding_seq, num_mut_seqs, 4, NUCS, 'uniform', paired_position_list, []
    )
    return mutated_seq


# ---------------------------------------------------------------------------
# Surrogate modeling
# ---------------------------------------------------------------------------

def surrogate(iteration, x_lib, y, *, cfg_name, X_test=None, y_test=None):
    cfg = SURROGATE_CONFIGS[cfg_name]

    surrogate_wrapper = squid.surrogate_zoo.SurrogateMAVENN(
        x_lib.shape,
        num_tasks=y.shape[1] if y.ndim == 2 else 1,
        gpmap=cfg["gpmap"],
        regression_type=cfg["regression_type"],
        linearity=cfg["linearity"],
        noise=cfg["noise"],
        noise_order=cfg["noise_order"],
        reg_strength=cfg["reg_strength"],
        hidden_nodes=cfg.get("hidden_nodes", 50),
        alphabet=NUCS,
        deduplicate=True,
        gpu=True,
    )

    mavenn_model, mave_df, test_df = surrogate_wrapper.train(
        x_lib, y,
        learning_rate=5e-4, epochs=500, batch_size=100,
        early_stopping=True, patience=25, restore_best_weights=True,
        save_dir=None, verbose=1,
    )

    base = f"/home/nagle/final_version/outputs/{subdir}/{activity}/surrogates/{cfg_name}"
    os.makedirs(base, exist_ok=True)

    fig, preds, g_truth = squid.impress.plot_y_vs_yhat(mavenn_model, mave_df)
    fig.savefig(f"{base}/y_y_hat_train_iter{iteration}.png", dpi=300)
    plt.close(fig)

    try:
        rho_random = spearman_on_testdf(mavenn_model, test_df)
    except Exception as e:
        print(f"[warn] rho_random failed ({cfg_name}): {e}")
        rho_random = np.nan

    rho_curated = np.nan
    if X_test is not None and y_test is not None:
        def onehot_to_str(x):
            nucs = np.array(list("ACGU"))
            return "".join(nucs[np.argmax(x, axis=1)])

        X_eval_str = np.array([onehot_to_str(x) for x in X_test], dtype=object)
        eval_df = pd.DataFrame({"x": X_eval_str, "y": y_test.ravel()})
        fig2, preds2, gtruth2, rho_curated = plot_y_vs_yhat(mavenn_model, eval_df)
        fig2.savefig(f"{base}/y_y_hat_curated_iter{iteration}.png", dpi=300)
        plt.close(fig2)

    params = surrogate_wrapper.get_params(gauge="consensus")

    add_fig = squid.impress.plot_additive_logo(params[1], view_window=None, alphabet=NUCS)
    add_fig.savefig(f"{base}/additive_iter{iteration}.png", dpi=300)
    plt.close(add_fig)
    np.save(f"{base}/additive_weights_iter{iteration}.npy", params[1])

    if cfg["gpmap"] == "pairwise":
        pair_fig = squid.impress.plot_pairwise_matrix(params[2], view_window=None, alphabet=NUCS)
        pair_fig.savefig(f"{base}/pairwise_iter{iteration}.png", dpi=300)
        plt.close(pair_fig)
        np.save(f"{base}/pairwise_weights_iter{iteration}.npy", params[2])

    plot_fig(preds, f"{iteration}_{cfg_name}", path=f"{base}/pred_dist_iter{iteration}.png")

    return {
        "cfg": cfg_name,
        "rho_curated": float(rho_curated) if rho_curated is not None else np.nan,
        "rho_random":  float(rho_random)  if rho_random  is not None else np.nan,
        "test_df":     test_df,
        "mavenn_model": mavenn_model,
        "params":      params,
    }


# ---------------------------------------------------------------------------
# Eval library builder (depends on plot_fig for optional plots)
# ---------------------------------------------------------------------------

def make_4_gt_eval_libraries(
    *, x_base, input_seq_str, gt_params, nonlin_name, nonlin_kwargs, out_dir,
    n_pool=2_000_000, target_n=20000, n_bins=200, clip_hi=98, seed=0,
    mut_rate=None,
):
    """Generate 4 uniformly-distributed GT eval libraries.

    Samples `n_pool` sequences (scored cheaply via the analytic GT model),
    then downsamples each GT key to exactly `target_n` sequences with a
    uniform score distribution.

    When `mut_rate` is provided, pool sequences are generated by applying
    independent per-position Bernoulli mutations at that rate from the WT
    sequence (matching the x_mut training distribution).  When None, fully
    random sequences are used instead.

    Scoring is done in two passes so that energy normalisation uses the
    pool-level std (not a per-chunk std), which guarantees the nonlinearity
    sees consistently scaled inputs across all pool sequences.

      Pass 1: collect raw additive (s_add) and combined (s_addpair) energies
              in 100K chunks — peak memory ~16 MB for the two float32 arrays.
      Post:   compute pool std for each energy stream.
      Pass 2: apply global normalisation + nonlinearity in another round of
              chunks, storing the four GT scores.
    """
    W_mut   = gt_params["W_mut"]
    mut_map = gt_params["mut_map"]
    b0      = gt_params["b"]
    edges   = gt_params["edges"]
    J       = gt_params["J"]

    gt_keys = ["additive", "additive_pairwise", "nonlin_additive", "nonlin_additive_pairwise"]
    _, L, A = x_base.shape

    # ── Generate n_pool sequences as uint8 nucleotide indices
    rng_pool = np.random.default_rng(seed + 99_999)
    if mut_rate is not None:
        # Mutate from WT at the same per-position rate used for x_mut so that
        # the pool score distribution matches the training library.
        nuc_to_idx = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
        wt_nuc_ids = np.array([nuc_to_idx[c] for c in input_seq_str[:L]], dtype=np.uint8)
        mut_mask    = rng_pool.random(size=(n_pool, L)) < mut_rate          # (n_pool, L)
        mut_offsets = rng_pool.integers(1, 4, size=(n_pool, L), dtype=np.uint8)  # 1,2,3
        nuc_ids = np.where(
            mut_mask,
            (wt_nuc_ids[None, :].astype(np.uint16) + mut_offsets) % A,
            wt_nuc_ids[None, :],
        ).astype(np.uint8)
        print(f"[GT eval] pool generated via mut_rate={mut_rate:.4f} from WT")
    else:
        nuc_ids = rng_pool.integers(0, A, size=(n_pool, L), dtype=np.uint8)
        print("[GT eval] pool generated fully random (uniform over all nucleotides)")

    chunk_size = 100_000

    # ── Pass 1: collect raw energies for global std computation
    s_add_all     = np.empty(n_pool, dtype=np.float32)
    s_addpair_all = np.empty(n_pool, dtype=np.float32)
    for start in range(0, n_pool, chunk_size):
        end     = min(start + chunk_size, n_pool)
        X_chunk = np.eye(A, dtype=np.float32)[nuc_ids[start:end]]
        sa  = additive_affinity_noWT(X_chunk, W_mut, mut_map, b=b0).reshape(-1)
        sp  = pairwise_potts_energy(X_chunk, edges, J, b=0.0).reshape(-1)
        s_add_all[start:end]     = sa
        s_addpair_all[start:end] = sa + sp

    # Pool-level mean and std (used consistently across all chunks so that
    # z-scoring is global, not per-chunk).
    mean_add    = float(np.mean(s_add_all))
    mean_addpair = float(np.mean(s_addpair_all))
    std_add     = float(np.std(s_add_all))     + 1e-8
    std_addpair = float(np.std(s_addpair_all)) + 1e-8
    print(f"[GT eval] pool mean_add={mean_add:.4f}  mean_addpair={mean_addpair:.4f}")
    print(f"[GT eval] pool std_add={std_add:.4f}  std_addpair={std_addpair:.4f}")

    # ── Pass 2: apply normalization + nonlinearity in chunks
    pool_scores = {k: np.empty(n_pool, dtype=np.float32) for k in gt_keys}
    for start in range(0, n_pool, chunk_size):
        end = min(start + chunk_size, n_pool)
        sa  = s_add_all[start:end].astype(float)
        sap = s_addpair_all[start:end].astype(float)
        pool_scores["additive"][start:end]          = sa
        pool_scores["additive_pairwise"][start:end] = sap
        if nonlin_name == "sigmoid":
            # Both are normalised by std_add only — NOT by std_addpair.
            # Dividing sap by its own std_addpair would cancel out the pairwise
            # penalties, collapsing both distributions onto the same scale.
            # Using std_add preserves the extra magnitude of pairwise interactions
            # so nonlin_additive_pairwise sits further in the negative tail.
            pool_scores["nonlin_additive"][start:end]          = apply_global_nonlin(
                sa / std_add, nonlin_name, nonlin_kwargs,
            ).reshape(-1)
            pool_scores["nonlin_additive_pairwise"][start:end] = apply_global_nonlin(
                sap / std_add, nonlin_name, nonlin_kwargs,
            ).reshape(-1)
        else:
            pool_scores["nonlin_additive"][start:end]          = apply_global_nonlin(
                (sa - mean_add) / std_add, nonlin_name, nonlin_kwargs,
            ).reshape(-1)
            pool_scores["nonlin_additive_pairwise"][start:end] = apply_global_nonlin(
                (sap - mean_add) / std_add, nonlin_name, nonlin_kwargs,
            ).reshape(-1)

    # Anchor nonlin scores so WT (s=0) maps to nonlin output = 0.
    # For sigmoid (scale-only): WT z = 0/std = 0.
    # For tanh-family (z-score): WT z = (0 - mean) / std.
    if nonlin_name == "sigmoid":
        wt_z_add = wt_z_addpair = 0.0
    else:
        wt_z_add     = (0.0 - mean_add)     / std_add
        wt_z_addpair = (0.0 - mean_addpair) / std_addpair
    wt_nl_add     = float(apply_global_nonlin(np.array([[wt_z_add]]),     nonlin_name, nonlin_kwargs))
    wt_nl_addpair = float(apply_global_nonlin(np.array([[wt_z_addpair]]), nonlin_name, nonlin_kwargs))
    pool_scores["nonlin_additive"]           -= wt_nl_add
    pool_scores["nonlin_additive_pairwise"]  -= wt_nl_addpair

    # Free the raw energy arrays now that we have pool_scores
    del s_add_all, s_addpair_all

    os.makedirs(out_dir, exist_ok=True)

    npz_path = os.path.join(out_dir, "pool_2M.npz")
    np.savez_compressed(
        npz_path,
        nuc_ids=nuc_ids,
        **{f"scores_{k}": pool_scores[k] for k in gt_keys},
    )
    print(f"[GT eval] saved 2M pool → {npz_path}")

    eval_libs = {}

    for k in gt_keys:
        yk = pool_scores[k].astype(float)
        # uniformize_by_histogram returns (y_uni, keep_indices) when X=None
        y_uni, keep_idx = uniformize_by_histogram(
            yk, X=None, n_bins=n_bins, clip_hi=clip_hi,
            target_n=target_n, seed=seed + hash(k) % 10_000,
        )
        # reconstruct one-hot only for the selected sequences
        X_uni = np.eye(A, dtype=np.float32)[nuc_ids[keep_idx]]

        txt_path = os.path.join(out_dir, f"evallibrary_{k}.txt")
        write_eval_library_txt(txt_path, X_uni, y_uni)

        try:
            plot_fig(y_uni, k, path=os.path.join(out_dir, f"uniformevallibrary_{k}.png"))
        except Exception as e:
            print(f"[warn] plotting failed for {k}: {e}")

        eval_libs[k] = {
            "X_eval": X_uni,
            "y_eval": y_uni.astype(float),
            "path":   txt_path,
        }
        print(f"[GT eval] {k}: wrote {X_uni.shape[0]} seqs -> {txt_path}")

    return eval_libs


# ---------------------------------------------------------------------------
# Main round functions
# ---------------------------------------------------------------------------

def random_mut_library_round0(rho_storage, coef_store, run_idx, epmapmin, epmapmax):
    """Generate a random library, build GT eval sets, train all surrogate configs."""

    pred_generator = squid.predictor.CustomPredictor(
        pred_fun=residbind.predict, reduce_fun="name", batch_size=512
    )
    mut_generator = squid.mutagenizer.RandomMutagenesis(mut_rate=mut_rate, uniform=False)
    mave = squid.mave.InSilicoMAVE(mut_generator, pred_generator, seq_length, mut_window=mut_window)

    print("Step 1: Generate randomly initialized library.")
    x_mut, y_mut = mave.generate(no_padding_seq, padding_amt=padding_amt, num_sim=int(lib_size))

    gt_bundle = plot_random_library_distributions_three_nonlins(
        x_mut,
        input_seq=input_seq,
        rng=np.random.default_rng(run_idx),
        save_dir=f"/home/nagle/final_version/outputs/{subdir}/{activity}/dist_of_preds",
    )

    nonlin_family  = "sigmoid"
    nonlin_kwargs  = gt_bundle["nonlin_kwargs"]["sigmoid"]

    eval_outdir = f"/home/nagle/final_version/{experiment}_graphs/gt_eval_libs/run_{run_idx}"
    eval_libs = make_4_gt_eval_libraries(
        x_base=x_mut,
        input_seq_str=input_seq,
        gt_params=gt_bundle["params"],
        nonlin_name=nonlin_family,
        nonlin_kwargs=nonlin_kwargs,
        out_dir=eval_outdir,
        n_pool=2_000_000,
        target_n=20000,
        n_bins=200,
        clip_hi=98,
        seed=run_idx,
        mut_rate=mut_rate,
    )

    gt_params = gt_bundle["params"]
    W_mut     = gt_params["W_mut"]
    mut_map   = gt_params["mut_map"]
    b0        = gt_params["b"]
    edges     = gt_params["edges"]
    J         = gt_params["J"]

    gt_keys_local = ["additive", "additive_pairwise", "nonlin_additive", "nonlin_additive_pairwise"]
    nonlin_family_for_plots = "sigmoid"

    scores_random = compute_gt_scores_for_library_potts(
        x_mut, W_mut=W_mut, mut_map=mut_map, b0=b0,
        nonlin_name=nonlin_family_for_plots, nonlin_kwargs=nonlin_kwargs, edges=edges, J=J,
    )

    scores_eval_for_plot = {}
    for k in gt_keys_local:
        Xk = eval_libs[k]["X_eval"]
        sk = compute_gt_scores_for_library_potts(
            Xk, W_mut=W_mut, mut_map=mut_map, b0=b0,
            nonlin_name=nonlin_family_for_plots, nonlin_kwargs=nonlin_kwargs, edges=edges, J=J,
        )
        scores_eval_for_plot[k] = sk[k]

    scores_eval = {k: scores_eval_for_plot[k] for k in gt_keys_local}

    plot_random_vs_eval_per_gt(
        scores_random, scores_eval,
        nonlin_label="GT",
        save_dir=f"/home/nagle/final_version/outputs/{subdir}/{activity}"
                 f"/random_vs_eval_gt_dists/run_{run_idx}",
        bins=120,
    )

    gt_w_outdir = (f"/home/nagle/final_version/outputs/{subdir}/{activity}"
                   f"/gt_weight_dists/run_{run_idx}")
    plot_gt_weight_dists_per_function(
        W_mut=W_mut, edges=edges, J=J, gt_keys=gt_keys_local, outdir=gt_w_outdir,
        bw_scale=1.25, clip=(0.5, 99.5), drop_pairwise_zeros=True,
    )
    plot_gt_weight_dists_per_function(
        W_mut=W_mut, edges=edges, J=J, gt_keys=gt_keys_local, outdir=gt_w_outdir,
        bw_scale=1.25, clip=(0.5, 99.5), drop_pairwise_zeros=True, peak_normalize=True,
    )

    train_nonlin_family = "sigmoid"
    gt   = gt_bundle["scores"][train_nonlin_family]
    gt_y = {k: np.asarray(gt[k], dtype=float).reshape(-1) for k in gt_keys_local}

    step = "0"
    rho_storage.setdefault(step, {})
    for cfg_name in SURROGATE_CONFIGS:
        rho_storage[step].setdefault(cfg_name, {
            "curated": {k: [] for k in gt_keys_local},
            "random":  {k: [] for k in gt_keys_local},
        })
    for cfg_name in SURROGATE_CONFIGS:
        coef_store.setdefault(cfg_name, {
            "additive": {k: [] for k in gt_keys_local},
            "pairwise": {k: [] for k in gt_keys_local},
        })

    for k, arr in gt_y.items():
        a = np.asarray(arr).reshape(-1)
        print(f"[GT {k}] N={a.size} finite%={np.mean(np.isfinite(a)):.3f} "
              f"nan={np.isnan(a).sum()} inf={np.isinf(a).sum()} "
              f"std={np.nanstd(a):.6f} min={np.nanmin(a):.3f} max={np.nanmax(a):.3f}")

    grid_results = {gt_key: {} for gt_key in gt_keys_local}

    for gt_key in gt_keys_local:
        y_true = np.asarray(gt_y[gt_key], dtype=float).reshape(-1, 1)
        m      = np.isfinite(y_true[:, 0])
        x_use, y_use = x_mut[m], y_true[m]

        if y_use.shape[0] < 50 or np.nanstd(y_use) == 0:
            print(f"[skip] gt_key={gt_key}: N={y_use.shape[0]} std={np.nanstd(y_use)}")
            for cfg_name in SURROGATE_CONFIGS:
                rho_storage[step][cfg_name]["curated"][gt_key].append(np.nan)
                rho_storage[step][cfg_name]["random"][gt_key].append(np.nan)
            continue

        X_eval_gt = eval_libs[gt_key]["X_eval"]
        y_eval_gt = eval_libs[gt_key]["y_eval"]

        for cfg_name in SURROGATE_CONFIGS:
            out = surrogate(
                "0", x_use, y_use,
                cfg_name=cfg_name, X_test=X_eval_gt, y_test=y_eval_gt,
            )
            rho_storage[step][cfg_name]["curated"][gt_key].append(out["rho_curated"])
            rho_storage[step][cfg_name]["random"][gt_key].append(out["rho_random"])

            grid_results[gt_key][cfg_name] = {
                "model":   out["mavenn_model"],
                "test_df": out["test_df"],
                "X_eval":  X_eval_gt,
                "y_eval":  y_eval_gt,
            }

            params = out.get("params")
            if params is not None and len(params) > 1:
                W_add = np.asarray(params[1], dtype=float).ravel()
                W_add = W_add[np.isfinite(W_add)]
                coef_store[cfg_name]["additive"][gt_key].append(W_add)

                if SURROGATE_CONFIGS[cfg_name]["gpmap"] == "pairwise" and len(params) > 2:
                    W_pair = np.asarray(params[2], dtype=float).ravel()
                    W_pair = W_pair[np.isfinite(W_pair)]
                    coef_store[cfg_name]["pairwise"][gt_key].append(W_pair)

    plot_16_y_vs_yhat_grid(
        grid_results,
        gt_keys=gt_keys_local,
        cfg_names=list(SURROGATE_CONFIGS.keys()),
        out_path=(f"/home/nagle/final_version/outputs/{subdir}/{activity}"
                  f"/y_vs_yhat_16panel_run{run_idx}.png"),
        iteration=0,
    )

    print(f"\nRun {run_idx} step 0 Spearman (curated vs random):")
    for cfg_name in SURROGATE_CONFIGS:
        cur = {k: np.nanmean(rho_storage[step][cfg_name]["curated"][k][-1:]) for k in gt_keys_local}
        rnd = {k: np.nanmean(rho_storage[step][cfg_name]["random"][k][-1:])  for k in gt_keys_local}
        print(f"  [{cfg_name}] curated: { {k: f'{v:.3f}' for k, v in cur.items()} }")
        print(f"  [{cfg_name}] random : { {k: f'{v:.3f}' for k, v in rnd.items()} }")

    out_seq_path = (f"/home/nagle/final_version/outputs/{subdir}/{activity}"
                    f"/sequences_random_mut{run_idx}.txt")
    os.makedirs(os.path.dirname(out_seq_path), exist_ok=True)
    with open(out_seq_path, "w") as f:
        for i in x_mut:
            seq = "".join(onehot_to_seq(np.expand_dims(remove_padding(i)[0], axis=0)))
            f.write(seq + "\n")

    return mave, epmapmin, epmapmax


def ssmutagenesis_roundpostss(mave, rho_storage, epmapmin, epmapmax):
    x_mutnew, y_mutnew = mave.generate(
        no_padding_seq, padding_amt=padding_amt, num_sim=int(lib_size / 4)
    )
    candidate_lib = mutagenesis_pipeline(
        no_padding_seq, '/home/nagle/final_version/outputs/ss_preds', int(lib_size * 3 / 4)
    )
    seqs_with_padding, prediction = predictor(candidate_lib, "post_ss", plot=False)
    new_lib   = np.concatenate([x_mutnew, seqs_with_padding], axis=0)
    new_lib_y = np.concatenate([y_mutnew, prediction], axis=0)

    with open(f"/home/nagle/final_version/outputs/{subdir}/{activity}/sequences_ss_mut.txt", "w") as f:
        for i in new_lib:
            seq = "".join(onehot_to_seq(
                np.expand_dims(np.expand_dims(remove_padding(i), axis=0)[0][0], axis=0)
            ))
            f.write(seq + "\n")

    top5, tempmin, tempmax, _, rho = surrogate(
        "post_ss", new_lib, new_lib_y, 12, 14, X_test=X_eval, y_test=y_eval
    )
    rho_storage["post_ss"].append(rho)
    if tempmin <= epmapmin: epmapmin = tempmin
    if tempmax >= epmapmax: epmapmax = tempmax
    return top5, epmapmin, epmapmax


def ism_round1_3(onehot_df, x_mut, epmapmin, epmapmax):
    candidate_lib = x_mut
    for i in range(1, 4):
        print(f"Iteration {i}")
        candidate_lib, candidate_lib_scores = bottom_percentile_iter(candidate_lib, i)
        seqs_with_padding, prediction = predictor(candidate_lib, i, plot=False)

        with open(f"/home/nagle/final_version/outputs/{subdir}/{activity}"
                  f"/sequences{i}_ss_mut.txt", "w") as f:
            for j in seqs_with_padding:
                seq = "".join(onehot_to_seq(
                    np.expand_dims(np.expand_dims(remove_padding(j), axis=0)[0][0], axis=0)
                ))
                f.write(seq + "\n")

        top5, tempmin, tempmax, _ = surrogate(
            i, candidate_lib, candidate_lib_scores, 12, 14, X_test=X_eval, y_test=y_eval
        )
        if tempmin <= epmapmin: epmapmin = tempmin
        if tempmax >= epmapmax: epmapmax = tempmax
    return top5, epmapmin, epmapmax


def roundoptimization(numrounds_local, onehot_df, rho_storage, top5, epmapmin, epmapmax):
    for i in range(1, numrounds_local):
        x_mut, y_mut = generateLibrary(top5, input_seq, "custom")

        with open(f"/home/nagle/final_version/outputs/{subdir}/{activity}"
                  f"/sequences{i}_ss_mut.txt", "w") as f:
            for j in x_mut:
                seq = "".join(onehot_to_seq(
                    np.expand_dims(np.expand_dims(remove_padding(j), axis=0)[0][0], axis=0)
                ))
                f.write(seq + "\n")

        top5, tempmin, tempmax, _, rho = surrogate(
            i, x_mut, y_mut, 12, 14, X_test=X_eval, y_test=y_eval
        )
        rho_storage[i].append(rho)
        if tempmin <= epmapmin: epmapmin = tempmin
        if tempmax >= epmapmax: epmapmax = tempmax
    return top5, epmapmin, epmapmax


def ism_plots(output_dir=None):
    single_mut_matrix = ism(
        no_padding_seq, "..................((((.....))))..........", output_dir
    )  # NOTE: update dot-bracket as needed

    delta_pairs, raw_pairs, F0, wt_idx = ism_double_tensor(
        no_padding_seq, predictor, batch_size=4096,
        skip_wt=True, include_diag=False, return_raw=True,
    )

    output_dir_tensors = (f'/home/nagle/final_version/outputs/{subdir}/{activity}'
                          f'/round_tensors_{experiment}')
    max_sign_e_tensor = epistasis_tensor(
        raw_pairs, single_mut_matrix, F0, GLOBAL_MASK_T, output_dir_tensors
    )

    plot_epistasis_maxpooled_like(max_sign_e_tensor)
    plot_epistasis_map(max_sign_e_tensor, output_dir)
    plot_raw_pairs_as_LxL_cells_with_4x4_inside(raw_pairs, wt_idx, output_dir)

    print("raw min/max:", np.nanmin(raw_pairs), np.nanmax(raw_pairs))
    print("E   min/max:", np.nanmin(max_sign_e_tensor), np.nanmax(max_sign_e_tensor))


# ---------------------------------------------------------------------------
# Curated evaluation library (loaded once at startup)
# ---------------------------------------------------------------------------

eval_lib_df = eval_library(f'/home/nagle/final_version/{experiment}_graphs/evallibrary.txt')
X_eval      = np.stack(eval_lib_df["X"].values, axis=0).astype(np.float32)
X_eval_local = X_eval
y_eval      = eval_lib_df["y"].values.astype(float)

onehot_df = pd.DataFrame({
    "X_seq": eval_lib_df["X"].apply(lambda s: "".join(onehot_to_seq(np.expand_dims(s, axis=0)))),
    "y":     eval_lib_df["y"],
})


# ---------------------------------------------------------------------------
# Main run loop
# ---------------------------------------------------------------------------

STEPS = ["0", "post_ss"] + [str(i) for i in range(1, 10)]

gt_keys = ["additive", "additive_pairwise", "nonlin_additive", "nonlin_additive_pairwise"]
step    = "0"

rho_storage = {
    step: {
        cfg_name: {
            "curated": {k: [] for k in gt_keys},
            "random":  {k: [] for k in gt_keys},
        }
        for cfg_name in SURROGATE_CONFIGS
    }
}

coef_store = {
    cfg_name: {
        "additive": defaultdict(list),
        "pairwise": defaultdict(list),
    }
    for cfg_name in SURROGATE_CONFIGS
}

for run_idx in range(numrounds):
    print(f"\n=== Run {run_idx + 1}\n")
    epmapmin = epmapmax = 0

    mave, epmapmin, epmapmax = random_mut_library_round0(
        rho_storage=rho_storage,
        coef_store=coef_store,
        run_idx=run_idx,
        epmapmin=epmapmin,
        epmapmax=epmapmax,
    )

    # Optional stages (uncomment to enable):
    # top5, epmapmin, epmapmax = ssmutagenesis_roundpostss(mave, rho_storage, epmapmin, epmapmax)
    # top5, epmapmin, epmapmax = roundoptimization(10, onehot_df, rho_storage, top5, epmapmin, epmapmax)

    ep_results = {}
    for j in [0]:
        flippedarr, flippedmask = ep_map(j, epmapmin, epmapmax)
        ep_results[j] = {"arr": flippedarr, "mask": flippedmask}
        ep_results[j]["arr"][ep_results[j]["mask"]] = 0
        np.save(
            f"/home/nagle/final_version/outputs/{subdir}/{activity}"
            f"/round_tensors_{experiment}/epmap_{j}_{run_idx}_{experiment}_masked.npy",
            ep_results[j]["arr"],
        )


# ---------------------------------------------------------------------------
# Post-run summary plots
# ---------------------------------------------------------------------------

outdir     = f"/home/nagle/final_version/outputs/{subdir}/{activity}/spearman_bars"
cfg_names  = list(SURROGATE_CONFIGS.keys())

spearman_bars_per_surrogate(
    rho_storage,
    output_dir=outdir,
    experiment=experiment,
    step="0",
    gt_keys=gt_keys,
    cfg_names=cfg_names,
    show_sem=True,
)

out_png_raw = f"/home/nagle/final_version/outputs/{subdir}/{activity}/coef_kde_grid_raw.png"
out_png_log = f"/home/nagle/final_version/outputs/{subdir}/{activity}/coef_kde_grid_logabs.png"
out_png_raw_pn = f"/home/nagle/final_version/outputs/{subdir}/{activity}/coef_kde_grid_raw_peaknorm.png"
out_png_log_pn = f"/home/nagle/final_version/outputs/{subdir}/{activity}/coef_kde_grid_logabs_peaknorm.png"

plot_kde_grid_surrogate_vs_gt(
    coef_store, SURROGATE_CONFIGS,
    experiment=experiment, gt_keys=gt_keys, cfg_names=cfg_names,
    out_png=out_png_raw, log_version=False,
)
plot_kde_grid_surrogate_vs_gt(
    coef_store, SURROGATE_CONFIGS,
    experiment=experiment, gt_keys=gt_keys, cfg_names=cfg_names,
    out_png=out_png_log, log_version=True,
)
plot_kde_grid_surrogate_vs_gt(
    coef_store, SURROGATE_CONFIGS,
    experiment=experiment, gt_keys=gt_keys, cfg_names=cfg_names,
    out_png=out_png_raw_pn, log_version=False, peak_normalize=True,
)
plot_kde_grid_surrogate_vs_gt(
    coef_store, SURROGATE_CONFIGS,
    experiment=experiment, gt_keys=gt_keys, cfg_names=cfg_names,
    out_png=out_png_log_pn, log_version=True, peak_normalize=True,
)


'''
ism_plots(f'/home/nagle/final_version/{experiment}_graphs')
'''

'''
JUNKYARD

ism_plots(f'/home/nagle/final_version/{experiment}_graphs')
print(predictgivensequence(input_seq))
----------------------------------------------------------------------------------------------------------------------------------------
outpath = "/home/nagle/final_version/rbp_targets_scoreoutput.csv"
score = predictgivensequence(input_seq)[0][0]

write_header = not os.path.exists(outpath)

with open(outpath, "a") as f:
    if write_header:
        f.write("rbp,sequence,score\n")
    f.write(f"{experiment},{input_seq},{score}\n")
------------------------------------------------------------------------------------------------------------------------------------
score = predictgivensequence(input_seq)[0][0]
print(score)
'''
