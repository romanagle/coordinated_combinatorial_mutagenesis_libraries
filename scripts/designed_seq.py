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
from scipy.stats import spearmanr
sys.path.append('/home/nagle/final_version/squid-manuscript/squid')
from ink import deep_sea
sys.path.append('/home/nagle/final_version/squid-nn/squid')
from matplotlib.patches import Rectangle
sys.path.append('/home/nagle/final_version/residualbind')    
from residualbind import ResidualBind
import helper, explain, dinuc_shuffle
import matplotlib.patches as patches
from prediction import paired_positions, predict_ss
from scipy.stats import pearsonr
from typing import Optional
from collections import defaultdict
from scipy.stats import gaussian_kde

normalization = 'log_norm'   # 'log_norm' or 'clip_norm'
ss_type = 'seq'     # 'seq', 'pu', or 'struct'
NUCS = ['A','C','G','U']


data_path = Path.home() / 'residualbind'/ 'data'/'RNAcompete_2013'/'rnacompete2013.h5'
results_path = helper.make_directory('/home/nagle/residualbind/results', 'rnacompete_2013')
save_path = '/home/nagle/residualbind/weights/log_norm_seq'

plot_path = helper.make_directory(save_path, 'FINAL')

parser = argparse.ArgumentParser(description="Run coordinated combinatorial mutagenesis using ResidualBind and RNACompete datset")
parser.add_argument("--seq", type=str)
parser.add_argument("--activity", type=str, help='TESTING low vs high activity')
parser.add_argument("--mut_rate", type=int, default=4, help='Integer value for how many positions you want to mutate in a given sequence')
parser.add_argument("--lib_size", type=int, default=20000, help='How many mutagenized sequences you want in your library')
parser.add_argument("--subdir", type=str, help='Subdirectory inside outputs')
parser.add_argument("--experiment", type=str, help='ex: RNCMPT00111 or RNCMPT00042')
parser.add_argument("--numrounds", type=int, help='Number of repeated runs')


args = parser.parse_args()

SURROGATE_CONFIGS = {
    # no-GE (linear)
    "additive": {
        "gpmap": "additive",
        "linearity": "linear",
        "regression_type": "GE",
        "noise": "Gaussian",
        "noise_order": 0,
        "reg_strength": 12,
    },
    "pairwise": {
        "gpmap": "pairwise",
        "linearity": "linear",
        "regression_type": "GE",
        "noise": "Gaussian",
        "noise_order": 0,
        "reg_strength": 12,
    },

    # GE (nonlinear)
    "additive_GE": {
        "gpmap": "additive",
        "linearity": "nonlinear",
        "regression_type": "GE",
        "noise": "SkewedT",
        "noise_order": 14,
        "reg_strength": 12,
    },
    "pairwise_GE": {
        "gpmap": "pairwise",
        "linearity": "nonlinear",
        "regression_type": "GE",
        "noise": "SkewedT",
        "noise_order": 14,
        "reg_strength": 12,
    },
}


#experiment = 'RNCMPT00111' #VTS1
#experiment = 'RNCMPT00042' #Nab2
experiment = args.experiment
rbp_index = helper.find_experiment_index(data_path, experiment)

# load rbp dataset
train, valid, test = helper.load_rnacompete_data(data_path,
                                                     ss_type='seq',
                                                     normalization=normalization,
                                                     rbp_index=rbp_index)


# load residualbind model
input_shape = list(train['inputs'].shape)[1:]
num_class = 1
weights_path = os.path.join(save_path, experiment + '_weights.hdf5')

residbind = ResidualBind(input_shape, num_class, weights_path)
residbind.load_weights()

print('Analyzing: '+ experiment)

input_seq = args.seq
num_muts = args.mut_rate
lib_size = args.lib_size
activity = args.activity
subdir = args.subdir
mut_rate = (num_muts/41)
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


def rna_to_one_hot(rna):
    mapping = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
    chars = [chr(c) for c in dinuc_shuffle.string_to_char_array(rna)]
    indices = [mapping[char] for char in chars]
    return np.eye(4)[indices]

def remove_padding(seq):
    #seq: sequence to mutate, dimensions: (41, 4)
    no_pad = []
    for idx, nuc in enumerate(seq):
        if np.array_equal(nuc, np.zeros(4)):
            continue
        else:
            no_pad.append(nuc)
    return np.stack(no_pad), (len(seq) - np.stack(no_pad).shape[0])

def onehot_to_seq(seq):
  #seq: 3D array dimensions: (1, n, 4)
  # n <= 41, depending on padding amount
    letters_seq = []
    for i in range(seq.shape[0]):
        for j in range(seq.shape[1]):
            if np.array_equal(seq[i,j], [1., 0., 0., 0.]):
                letters_seq.append("A")
            elif np.array_equal(seq[i,j], [0., 1., 0., 0.]):
                letters_seq.append("C")
            elif np.array_equal(seq[i, j], [0., 0., 1., 0.]):
                letters_seq.append("G")
            elif np.array_equal(seq[i,j], [0., 0., 0., 1.]):
                letters_seq.append("U")
            else:
                print(f'Uneven padding at position {j}')
                break
    return letters_seq

def plot_fig(listofscores, iteration, path = None):
    if path == None:
        path = f"/home/nagle/final_version/outputs/{subdir}/{activity}/dist_of_preds/1214/iter_{iteration}_pred_binding_affinity_distribution.png"
    print(len(listofscores))
    plt.figure(figsize=(8, 5))
    plt.hist(listofscores, density = True, bins=100, edgecolor='black', alpha=0.7)
    plt.title(f"Binding Affinity Score Prediction Distribution after Random Mutagenesis")
    plt.xlabel("Binding Affinity score")
    #plt.xlim(-2,2)
    #plt.axvline(scores[0], color='green', linestyle='dashed', linewidth=2, label=f'target sequence')
    #plt.axvline(scores[1], color='orange', linestyle='dashed', linewidth=2, label=f'natural')
    #plt.axvline(scores[2], color='orange', linestyle='dashed', linewidth=2, label=f'3')
    plt.ylabel("Density")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)

def y_yhat_fig(y, y_hat, iteration):
    y = np.asarray(y, dtype=float).ravel()
    y_hat = np.asarray(y_hat, dtype=float).ravel()

    # Pairwise drop NaNs (if any)
    mask = ~np.isnan(y) & ~np.isnan(y_hat)
    y, y_hat = y[mask], y_hat[mask]

    # Safety check
    if y.size == 0 or y_hat.size == 0 or y.size != y_hat.size:
        raise ValueError(f"Bad inputs: y size={y.size}, y_hat size={y_hat.size}")

    r, p = pearsonr(y, y_hat)
    r2 = r**2

    plt.figure(figsize=(8, 5))
    plt.scatter(y, y_hat, alpha=0.7)
    plt.xlabel("y (True Values)")
    plt.ylabel("ŷ (Predicted Values)")
    plt.title("y vs ŷ Scatter Plot")

    # Add diagonal reference line
    min_val = min(y.min(), y_hat.min())
    max_val = max(y.max(), y_hat.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="Ideal Fit")

    # Add correlation coefficient text
    plt.text(0.05, 0.95, f"Pearson r = {r:.3f}", transform=plt.gca().transAxes,
            fontsize=12, verticalalignment='top')

    plt.legend()
    plt.savefig(f"/home/nagle/final_version/outputs/{subdir}/{activity}/dist_of_preds/1214/iter_{iteration}_y_yhat.png", dpi=300)
    plt.close()
def mutagenesis_pipeline(input, output_dir, num_mut_seqs):

    if input.ndim != 2:
        input = np.squeeze(input)
        
    cleaned_seq = "".join(onehot_to_seq(np.expand_dims(input, axis=0)))

    print("\nStep 2: New library generated from secondary structure prior.\n") 
    
    predict_ss(cleaned_seq, output_dir)

    seq_file_name = f"{cleaned_seq}.txt"

    file_absolute_path = os.path.join(output_dir, seq_file_name)
    
    if os.path.isfile(file_absolute_path):
        with open(file_absolute_path, 'r') as f:
            content = f.read()
            paired_position_list = paired_positions(content)

    #4 mutations is 10% of sequence length
    print(f"Here are the positions that are paired: {paired_position_list}")
    mutated_seq, mut_index = deep_sea(no_padding_seq, num_mut_seqs, 4, NUCS, 'uniform', paired_position_list, []) #deep_sea outputs a 3D array, not a 2d


    return mutated_seq

def bottom_percentile_iter(input_library, iteration):

    seqs_with_padding, prediction = predictor(input_library, iteration, plot=False)

    percentile = 20
    bottompercentile = np.percentile(prediction, percentile)
    bottompercentileseqs = []
    bottompercentilescores = []
    otherseqs = []
    otherscores = []
    for seq, score in zip(input_library, prediction):
        if score <= bottompercentile:
            bottompercentileseqs.append(seq)
            bottompercentilescores.append(score)
        else:
            otherseqs.append(seq)
            otherscores.append(score.item())

    print(f"\nLength of the bottom {percentile}th percentile seqs: {len(bottompercentileseqs)}")
    print(f"\nLength of the other {100-percentile}th percentile seqs: {len(otherseqs)}\n")

    no_pad_bottomseqs = []
    for i in bottompercentileseqs:
        no_padding_seq, _ = remove_padding(i)
        no_pad_bottomseqs.append(no_padding_seq)

    #new library dict
    new_lib_list = ism_evol(np.array(no_pad_bottomseqs))
    newpercentilescores = [k[1] for k in new_lib_list]
    newpercentileseqs = [k[0] for k in new_lib_list]
    print(f"Average of the bottom {percentile}th percentile scores before ISM: {np.mean(bottompercentilescores)}")
    print(f"Average of the bottom {percentile}th percentile scores after ISM: {np.mean(newpercentilescores)}")

    otherscores.extend(newpercentilescores)

    plot_fig(np.array(otherscores), iteration)

    return np.concatenate([np.array(otherseqs), np.array(newpercentileseqs)], axis=0), np.array(otherscores)

def ism_evol(input_seqs):
    #dimensions ex: (20, n, 4)
    lib_size = input_seqs.shape[0]
    print(f"Number of sequences predicted: {lib_size * 41 * 3}\n")

    mut_library = []
    mut_oh = {0: np.array([1, 0, 0, 0]),
          1: np.array([0, 1, 0, 0]),
          2: np.array([0, 0, 1, 0]),
          3: np.array([0, 0, 0, 1])}
    for i in range(input_seqs.shape[0]):
        mut_seqs = []
        for j in range(input_seqs.shape[1]):
            curr_nuc_oh = [k for k, v in mut_oh.items() if np.array_equal(v, input_seqs[i][j])]
            curr_idx = list(mut_oh.keys()).index(curr_nuc_oh[0])
            for k in range(1,4):
                next_index = (curr_idx + k) % 4
                inserted_mut = np.array(mut_oh[next_index]).reshape((1,4))
                new_seq = np.concatenate([input_seqs[i][0:j], inserted_mut, input_seqs[i][j+1:]])
                mut_seqs.append(new_seq)
                #print("".join(onehot_to_seq(np.expand_dims(new_seq, axis=0))))
        mut_library.extend(mut_seqs)

    new_lib = np.array(mut_library)

    seqs_with_predictions = []

    seqs_with_padding, prediction = predictor(new_lib, None, plot=False)
    for seqs, scores in zip(seqs_with_padding, prediction):
        seqs_with_predictions.append((remove_padding(seqs)[0], scores.item()))
    
    toppreds = sorted(seqs_with_predictions, key=lambda k: k[1], reverse=True)[:lib_size]

    return toppreds

def generateLibrary(paired_position_list, input_seq, mode):

    pred_generator = squid.predictor.CustomPredictor(pred_fun=residbind.predict, reduce_fun = "name",
                                                  batch_size=512
                                                  )

    padded_indices = []
    for idx, i in enumerate(input_seq):
        if np.array_equal(i, np.zeros(4)):
            padded_indices.append(idx)

    mut_rate = 4#round(num_muts*len(input_seq), 1)

    if mode == "random":
        mut_generator = squid.mutagenizer.RandomMutagenesis(mut_rate=mut_rate, uniform=False)
    else:
        mut_generator = squid.mutagenizer.CustomMutagenesis(mut_rate=mut_rate)


    oh_seq = rna_to_one_hot(input_seq)
    no_padding_seq, _ = remove_padding(oh_seq) #needs to be in one-hot encoded form


    seq_length = no_padding_seq.shape[0]
    mut_window = [0, seq_length]
    padding_amt = 41 - oh_seq.shape[0] #residualbind requires that the sequence is 41 nucleotides long

    mave = squid.mave.InSilicoMAVE(mut_generator, pred_generator, seq_length, mut_window=mut_window, paired_position_list=paired_position_list)

    x_mut, y_mut = mave.generate(no_padding_seq, padding_amt=padding_amt, num_sim=lib_size)

    return x_mut, y_mut

def plot_y_vs_yhat(model, X_test, y_test, save_dir=None):
    """Function for visualizing comparison of MAVE values and MAVE-NN predictions.

    Parameters
    ----------
    model : mavenn.src.model.Model
        MAVE-NN model object.
    mave_df : pandas.core.frame.DataFrame
        Dataframe containing MAVE training splits, y floats, and x strings  (shape : (N,3))
    save_dir : str
        Directory for saving figures to file.

    Returns
    -------
    matplotlib.pyplot.Figure
    """

    # plot mavenn y versus yhat
    fig, ax = plt.subplots(1,1,figsize=[5,5])
    yhat_test = model.x_to_yhat(X_test) #compute yhat on test data
    rho, _ = spearmanr(yhat_test.ravel(), y_test)   
    ax.scatter(yhat_test, y_test, color='C0', s=1, alpha=.1,
               label='test data')
    xlim = [min(yhat_test), max(yhat_test)]
    ax.plot(xlim, xlim, '--', color='k', label='diagonal', zorder=100)
    ax.set_xlabel('model prediction ($\hat{y}$)')
    ax.set_ylabel('measurement ($y$)')
    ax.set_title(f"Standard metric of model performance:\nSpearman correlation = {rho:.3f}")
    ax.legend(loc='upper left')
    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir,'mavenn_measure_yhat.png'), facecolor='w', dpi=200)
        plt.close()
    #else:
        #plt.show()
    return fig, yhat_test, y_test, rho

from scipy.stats import spearmanr


def surrogate(
    iteration,
    x_lib,
    y,
    *,
    cfg_name: str,
    X_test=None,
    y_test=None
):
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
        alphabet=NUCS,
        deduplicate=True,
        gpu=True
    )

    # train
    mavenn_model, mave_df, test_df = surrogate_wrapper.train(
        x_lib, y,
        learning_rate=5e-4,
        epochs=500,
        batch_size=100,
        early_stopping=True,
        patience=25,
        restore_best_weights=True,
        save_dir=None,
        verbose=1
    )

    # where to save outputs for this config
    base = f"/home/nagle/final_version/outputs/{subdir}/{activity}/surrogates/{cfg_name}"
    os.makedirs(base, exist_ok=True)

    # (1) training scatter
    fig, preds, g_truth = squid.impress.plot_y_vs_yhat(mavenn_model, mave_df)
    fig.savefig(f"{base}/y_y_hat_train_iter{iteration}.png", dpi=300)
    plt.close(fig)

    # (2) rho on random test_df (no retrain)
    try:
        rho_random = spearman_on_testdf(mavenn_model, test_df)
    except Exception as e:
        print(f"[warn] rho_random failed ({cfg_name}): {e}")
        rho_random = np.nan

    # (3) rho on curated eval set (optional)
    rho_curated = np.nan
    if X_test is not None and y_test is not None:
        def onehot_to_str(x):
            nucs = np.array(list("ACGU"))
            return "".join(nucs[np.argmax(x, axis=1)])

        X_eval_str = np.array([onehot_to_str(x) for x in X_test], dtype=object)
        fig2, preds2, gtruth2, rho_curated = plot_y_vs_yhat(mavenn_model, X_eval_str, y_test)
        fig2.savefig(f"{base}/y_y_hat_curated_iter{iteration}.png", dpi=300)
        plt.close(fig2)

    # (4) save params if available for this gpmap
    params = surrogate_wrapper.get_params(gauge="consensus")

    # additive logo always exists (params[1])
    add_fig = squid.impress.plot_additive_logo(params[1], view_window=None, alphabet=NUCS)
    add_fig.savefig(f"{base}/additive_iter{iteration}.png", dpi=300)
    plt.close(add_fig)
    np.save(f"{base}/additive_weights_iter{iteration}.npy", params[1])

    # pairwise only if pairwise gpmap
    if cfg["gpmap"] == "pairwise":
        pair_fig = squid.impress.plot_pairwise_matrix(params[2], view_window=None, alphabet=NUCS)
        pair_fig.savefig(f"{base}/pairwise_iter{iteration}.png", dpi=300)
        plt.close(pair_fig)
        np.save(f"{base}/pairwise_weights_iter{iteration}.npy", params[2])

    # your old plotting of preds distribution
    plot_fig(preds, f"{iteration}_{cfg_name}", path=f"{base}/pred_dist_iter{iteration}.png")

    return {
        "cfg": cfg_name,
        "rho_curated": float(rho_curated) if rho_curated is not None else np.nan,
        "rho_random": float(rho_random) if rho_random is not None else np.nan,
        "test_df": test_df,
        "mavenn_model": mavenn_model,
        "params": params,   # keep in memory if needed
    }
def compute_gt_scores_for_library(
    X: np.ndarray,                 # (N,L,4)
    *,
    W_mut: np.ndarray,             # (L,3)
    P_mut: np.ndarray,             # (L-1,3,3)
    edges: np.ndarray,
    J: np.ndarray,
    mut_map: np.ndarray,           # (L,3)
    b0: float,
    nonlin_name: str,
    nonlin_kwargs: dict,
):
    """
    Returns dict with 4 GT keys, each -> (N,) float array
    """
    s_add = additive_affinity_noWT(X, W_mut, mut_map, b=b0)            # (N,1)
    #s_pair = pairwise_adjacent_noWT(X, P_mut, mut_map, b=0.0)          # (N,1)
    s_pair = pairwise_potts_energy(X, edges, J, b=0.0)
    s_addpair = s_add + s_pair                                         # (N,1)

    y_add = s_add.reshape(-1)
    y_addpair = s_addpair.reshape(-1)
    y_nonlin_add = apply_global_nonlin(s_add, nonlin_name, nonlin_kwargs).reshape(-1)
    y_nonlin_addpair = apply_global_nonlin(s_addpair, nonlin_name, nonlin_kwargs).reshape(-1)

    return {
        "additive": y_add,
        "additive_pairwise": y_addpair,
        "nonlin_additive": y_nonlin_add,
        "nonlin_additive_pairwise": y_nonlin_addpair,
    }

def compute_gt_scores_for_library_potts(
    X: np.ndarray,                 # (N,L,4)
    *,
    W_mut: np.ndarray,             # (L,3)
    mut_map: np.ndarray,           # (L,3)
    b0: float,
    nonlin_name: str,
    nonlin_kwargs: dict,

    # pairwise options (choose ONE)
    P_mut: np.ndarray = None,      # (L-1,3,3)  adjacent noWT
    edges: np.ndarray = None,      # (M,2)      potts
    J: np.ndarray = None,          # (L,L,4,4)  potts
):
    """
    Returns dict with 4 GT keys, each -> (N,) float array
    """

    s_add = additive_affinity_noWT(X, W_mut, mut_map, b=b0)  # (N,1)

    # choose pairwise implementation
    if edges is not None and J is not None:
        s_pair = pairwise_potts_energy(X, edges, J, b=0.0)   # (N,1)
    elif P_mut is not None:
        s_pair = pairwise_adjacent_noWT(X, P_mut, mut_map, b=0.0)  # (N,1)
    else:
        # no pairwise term
        s_pair = np.zeros_like(s_add)

    s_addpair = s_add + s_pair  # (N,1)

    y_add = s_add.reshape(-1)
    y_addpair = s_addpair.reshape(-1)
    y_nonlin_add = apply_global_nonlin(s_add, nonlin_name, nonlin_kwargs).reshape(-1)
    y_nonlin_addpair = apply_global_nonlin(s_addpair, nonlin_name, nonlin_kwargs).reshape(-1)

    return {
        "additive": y_add,
        "additive_pairwise": y_addpair,
        "nonlin_additive": y_nonlin_add,
        "nonlin_additive_pairwise": y_nonlin_addpair,
    }
def plot_random_vs_eval_per_gt(
    scores_random: dict,
    scores_eval: dict,
    *,
    nonlin_label: str,
    save_dir: str,
    bins: int = 120,
    density: bool = True,
    alpha_random: float = 0.45,
    alpha_eval: float = 0.45,
):
    """
    Makes 4 separate plots: one per GT key.
    Each plot overlays random-lib dist vs eval-lib dist.
    """
    os.makedirs(save_dir, exist_ok=True)

    gt_keys = ["additive", "additive_pairwise", "nonlin_additive", "nonlin_additive_pairwise"]

    for k in gt_keys:
        r = np.asarray(scores_random[k], dtype=float).reshape(-1)
        e = np.asarray(scores_eval[k], dtype=float).reshape(-1)

        # finite mask (avoid all-NaN axis)
        r = r[np.isfinite(r)]
        e = e[np.isfinite(e)]

        fig, ax = plt.subplots(figsize=(8.5, 5))

        if r.size == 0 and e.size == 0:
            ax.text(0.5, 0.5, f"{k}: no finite values", ha="center", va="center", transform=ax.transAxes)
            out = os.path.join(save_dir, f"{nonlin_label}_{k}_random_vs_eval.png")
            plt.tight_layout()
            plt.savefig(out, dpi=300)
            plt.close(fig)
            continue

        # shared bins (stable overlay)
        all_vals = np.concatenate([r, e]) if (r.size and e.size) else (r if r.size else e)
        lo, hi = np.percentile(all_vals, [0.5, 99.5]) if all_vals.size > 5 else (np.min(all_vals), np.max(all_vals))
        if lo == hi:
            lo -= 1e-6
            hi += 1e-6
        bin_edges = np.linspace(lo, hi, bins + 1)

        if r.size:
            ax.hist(r, bins=bin_edges, density=density, alpha=alpha_random, label=f"Random library (N={len(r)})")
        if e.size:
            ax.hist(e, bins=bin_edges, density=density, alpha=alpha_eval, label=f"Eval library (N={len(e)})")

        ax.set_title(f"{k} — {nonlin_label}: random vs eval distributions")
        ax.set_xlabel("Ground-truth score (WT-referenced)")
        ax.set_ylabel("Density" if density else "Count")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend()

        plt.tight_layout()
        out = os.path.join(save_dir, f"{nonlin_label}_{k}_random_vs_eval.png")
        plt.savefig(out, dpi=300)
        plt.close(fig)

def spearman_on_testdf(model, test_df):
    cols = list(test_df.columns)

    # sequences
    if "x" in cols:
        X = np.asarray(test_df["x"])
    elif "X" in cols:
        X = np.asarray(test_df["X"])
    else:
        raise KeyError(f"no sequence col in test_df. cols={cols}")

    # y true
    y_col = "y" if "y" in cols else (next((c for c in cols if c.startswith("y")), None))
    if y_col is None:
        raise KeyError(f"no y col in test_df. cols={cols}")
    y = np.asarray(test_df[y_col], dtype=float).reshape(-1)

    # predict
    yhat = np.asarray(model.x_to_yhat(X), dtype=float).reshape(-1)

    m = np.isfinite(y) & np.isfinite(yhat)
    if m.sum() < 3 or np.nanstd(y[m]) == 0 or np.nanstd(yhat[m]) == 0:
        return np.nan

    rho, _ = spearmanr(yhat[m], y[m])
    return float(rho)

def ep_map(iteration, finalmin=None, finalmax=None):
    weights = np.load(f'/home/nagle/final_version/outputs/{subdir}/{activity}/pairwise_weights/1214/weights_epoch_{iteration}.npy')

    seq_length = weights.shape[0]
    max_pooled_arr = np.empty((seq_length,seq_length))
    for i in range(seq_length):
        for j in range(seq_length):
            intermediate = []
            for k in range(4):
                for l in range(4):
                    intermediate.append(weights[i][k][j][l])
            max_pooled_arr[i][j] = max(intermediate, key=abs)

    mask = np.tril(np.ones_like(max_pooled_arr, dtype=bool), -1)

    flipped_arr = max_pooled_arr.T
    flipped_mask = mask.T
    min_val, max_val = np.nanmin(flipped_arr), np.nanmax(flipped_arr)

    # Plot
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(flipped_arr, mask=flipped_mask, cmap='seismic', square=True, center = 0, vmin=min_val, vmax=max_val)

    # Move ticks
    ax.xaxis.tick_bottom()        # Move x-axis to top
    ax.yaxis.tick_left()      # Move y-axis to right

    threshold=0.6
    impt_pos = {}
    # Add rectangle borders around cells above threshold
    for i in range(max_pooled_arr.shape[0]):
        for j in range(max_pooled_arr.shape[1]):
            if np.abs(max_pooled_arr[i, j]) > threshold:
                rect = Rectangle((i, j), 1, 1, fill=False, edgecolor='black', linewidth=2)
                if (np.abs(i-j) > 7):
                    impt_pos[(i,j)] = max_pooled_arr[i, j]
                    rect = Rectangle((i, j), 1, 1, fill=False, edgecolor='yellow', linewidth=4)
                ax.add_patch(rect)

    plt.tight_layout()
    plt.savefig(f"/home/nagle/final_version/outputs/{subdir}/{activity}/new_ep_maps/1214/NEWepmap{iteration}.png")
    return flipped_arr, flipped_mask

def predictor(input_library, iteration, plot=True, true_preds = None):
    #input_library has no padding
    print(f"Here is the size of the input library: {input_library.shape}")

    beg_padding = np.zeros((input_library.shape[0], padding_amt//2, 4))
    end_padding = np.zeros((input_library.shape[0], padding_amt//2 + (padding_amt % 2), 4))

    seqs_with_padding = np.concatenate([beg_padding, input_library, end_padding], axis=1)
    print(f"Here is the size of the input library: {seqs_with_padding.shape}")
    prediction = residbind.predict(seqs_with_padding)

    if plot:
        plot_fig(prediction, iteration)
    if true_preds is not None and not true_preds.empty:
        y_yhat_fig(prediction, true_preds, iteration)

    return seqs_with_padding, prediction

def ism(input_seq, dot_bracket,outputdir):
    """
    input_seq : np.ndarray
        One-hot encoded sequence (L, 4).
    dot_bracket : str
        Dot-bracket notation string of length L.
    """
    seq_length = input_seq.shape[0]
    score_matrix = np.empty((4, seq_length))

    mut_oh = {
        0: np.array([1, 0, 0, 0]),
        1: np.array([0, 1, 0, 0]),
        2: np.array([0, 0, 1, 0]),
        3: np.array([0, 0, 0, 1])
    }

    for j in range(seq_length):
        for k in range(4):
            new_seq = np.concatenate([
                input_seq[0:j],
                np.expand_dims(mut_oh[k], axis=0),
                input_seq[j+1:]
            ])
            seqs_with_padding, prediction = predictor(
                np.expand_dims(new_seq, axis=0), None, plot=False
            )
            score_matrix[k][j] = prediction.item()

    # ---- Heatmap plotting ----
    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(score_matrix, aspect='auto', cmap='viridis')

    # Y-axis: nucleotides
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(['A', 'C', 'G', 'U'])

    # X-axis: dot-bracket notation
    ax.set_xticks(np.arange(seq_length))
    ax.set_xticklabels(list(dot_bracket))
    ax.set_xlabel("Dot-bracket position")
    ax.set_ylabel("Nucleotide")
    ax.set_title("4×L ISM Heatmap")

    plt.colorbar(im, ax=ax, label="Binding Affinity Score")

    # ---- Draw boxes around original nucleotides ----
    for j in range(seq_length):
        wt_idx = np.argmax(input_seq[j])  # row of wild-type nucleotide
        rect = Rectangle(
            (j - 0.5, wt_idx - 0.5),  # bottom-left corner
            1, 1,                     # width, height of box
            linewidth=1.5,
            edgecolor='red',
            facecolor='none'
        )
        ax.add_patch(rect)

    plt.tight_layout()
    plt.savefig(f"{outputdir}/heatmap_output_with_boxes_middle.png", dpi=300)

    return score_matrix

def eval_library(path_to_library=None):
    seqs, scores = [], []
    with open(path_to_library, "r") as f:
        for line in f:
            if not line.strip():
                continue
            seq, score = line.strip().split(",")
            seqs.append(rna_to_one_hot(seq))
            scores.append(float(score))
    # stack into arrays
    x_mut = np.stack(seqs, axis=0)   # shape (N, L, 4)
    y_mut = np.array(scores, dtype=np.float32)  # shape (N,)
    seq_list = [x_mut[i] for i in range(x_mut.shape[0])]
    return pd.DataFrame({"X": seq_list, "y": y_mut})

def _batch_predictor(predictor, X, batch_size=4096):
    """X: (N,L,4) -> (N,) scores; predictor returns (aux, scores)."""
    N = X.shape[0]
    out = np.empty(N, dtype=np.float32)
    i = 0
    while i < N:
        j = min(N, i + batch_size)
        _, s = predictor(X[i:j], None, plot=False)
        out[i:j] = np.asarray(s).reshape(-1).astype(np.float32)
        i = j
    return out

def ism_double_tensor(input_seq, predictor, batch_size=4096,
                      skip_wt=False, include_diag=False, return_raw=False):
    """
    Build full double-mutant Δ tensor (no single mutants).
    Args
      input_seq: (L,4) one-hot over A,C,G,U
      predictor: your model; called in batches
      skip_wt:  if True, cells where a==WT_i or b==WT_j are set to NaN (9 combos per pair)
      include_diag: include i==j pairs (usually False)
      return_raw: also return the raw score tensor (not just deltas)
    Returns
      delta_pairs: (4,4,L,L) with Δ = score(mut) - score(WT)
      raw_pairs (optional): (4,4,L,L) raw scores for each (a,b,i,j)
      baseline: float score(WT)
      wt_idx: (L,) int WT base per position
    """
    x = np.asarray(input_seq, dtype=np.float32)
    assert x.ndim == 2 and x.shape[1] == 4, "Expecting (L,4) one-hot"
    L = x.shape[0]
    wt_idx = x.argmax(axis=1)

    # baseline once
    _, wt_score = predictor(x[None, ...], None, plot=False)
    baseline = float(np.asarray(wt_score).reshape(-1)[0])

    # prepare outputs
    raw_pairs = np.full((4, 4, L, L), np.nan, dtype=np.float32)
    delta_pairs = np.full_like(raw_pairs, np.nan)

    # build and score in chunks
    combos, meta = [], []   # meta holds (a,b,i,j)
    max_chunk = max(batch_size, 4096)

    for i in range(L):
        for j in range(L):
            if (i == j) and not include_diag:
                continue
            for a in range(4):
                if skip_wt and a == wt_idx[i]:
                    continue
                for b in range(4):
                    if skip_wt and b == wt_idx[j]:
                        continue
                    X = x.copy()
                    X[i, :] = 0.0; X[i, a] = 1.0
                    X[j, :] = 0.0; X[j, b] = 1.0
                    combos.append(X)
                    meta.append((a, b, i, j))
                    if len(combos) >= max_chunk:
                        S = _batch_predictor(predictor, np.stack(combos), batch_size)
                        for val, (aa, bb, ii, jj) in zip(S, meta):
                            raw_pairs[aa, bb, ii, jj] = float(val)
                        combos.clear(); meta.clear()
    if combos:
        S = _batch_predictor(predictor, np.stack(combos), batch_size)
        for val, (aa, bb, ii, jj) in zip(S, meta):
            raw_pairs[aa, bb, ii, jj] = float(val)

    # convert to deltas
    mask = ~np.isnan(raw_pairs)
    delta_pairs[mask] = raw_pairs[mask] - baseline

    return (delta_pairs, raw_pairs, baseline, wt_idx) if return_raw else (delta_pairs, baseline, wt_idx)

def plot_double_slice_one_nuc(delta_pairs, a, b, title=None, out_png=None):
    """
    Plot a (L×L) heatmap for the chosen nucleotide pair: i→a, j→b.
    a,b in {0:A,1:C,2:G,3:U}
    """
    L = delta_pairs.shape[-1]
    mat = delta_pairs[a, b, :, :]
    plt.figure(figsize=(6.2, 5.6))
    im = plt.imshow(mat, origin="lower", interpolation="none")
    plt.xlabel(f"j (mutate to {NUCS[b]})")
    plt.ylabel(f"i (mutate to {NUCS[a]})")
    plt.title(title or f"Double ISM Δ score (i→{NUCS[a]}, j→{NUCS[b]})")
    cbar = plt.colorbar(im); cbar.set_label("Δ score")
    plt.tight_layout()
    if out_png:
        plt.savefig(out_png, dpi=300)
    plt.show()

def plot_raw_pairs_as_LxL_cells_with_4x4_inside(
    raw_pairs,            # (4,4,L,L)
    wt_idx,               # (L,)
    outputdir,
    nucs=("A","C","G","U"),
    gap=1,
    step_pos=5,
    show_wt_pixel=True,
    figsize=(12, 12),

    # scaling for RAW scores
    scale="minmax",           # "percentile", "minmax", or "fixed"
    clip_percentiles=(2, 98),
    vmin_fixed=None,
    vmax_fixed=None,

    # appearance
    box_outline_color="black",
    box_outline_lw=1.2,
    inner_grid_alpha=0.20,
    annotate_mini=False,
    mini_fontsize=4,
    mini_alpha=0.9,
    mini_which="both",             # "row", "col", "both"
    cbar_label="Raw score (double mutant)",
):
    assert raw_pairs.ndim == 4 and raw_pairs.shape[:2] == (4, 4), \
        f"Expected (4,4,L,L), got {raw_pairs.shape}"
    _, _, L0, L1 = raw_pairs.shape
    assert L0 == L1, f"Expected square LxL, got {L0}x{L1}"
    L = L0
    assert len(wt_idx) == L, f"wt_idx length {len(wt_idx)} must match L={L}"

    cell = 4
    stride = cell + gap
    H = L * cell + (L - 1) * gap
    W = L * cell + (L - 1) * gap

    big = np.full((H, W), np.nan, dtype=float)

    # Fill big image: each (i,j) gets a 4x4 block = raw_pairs[:,:,i,j]
    for i in range(L):
        r0 = i * stride
        for j in range(L):
            c0 = j * stride
            big[r0:r0+cell, c0:c0+cell] = raw_pairs[:, :, i, j]

    finite = big[np.isfinite(big)]
    if finite.size == 0:
        vmin, vmax = 0.0, 1.0
    else:
        if scale == "percentile":
            lo, hi = clip_percentiles
            vmin = np.percentile(finite, lo)
            vmax = np.percentile(finite, hi)
        elif scale == "minmax":
            vmin = float(np.min(finite))
            vmax = float(np.max(finite))
        elif scale == "fixed":
            if vmin_fixed is None or vmax_fixed is None:
                raise ValueError("scale='fixed' requires vmin_fixed and vmax_fixed")
            vmin, vmax = float(vmin_fixed), float(vmax_fixed)
        else:
            raise ValueError("scale must be 'percentile', 'minmax', or 'fixed'")

        if vmin == vmax:
            vmin, vmax = vmin - 1e-6, vmax + 1e-6

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(big, origin="lower", interpolation="none", vmin=vmin, vmax=vmax, aspect="equal")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label(cbar_label)

    # Outer ticks at position-cell centers
    centers = np.arange(L) * stride + (cell - 1) / 2.0
    tick_pos = np.arange(0, L, step_pos) if (step_pos and step_pos > 0) else np.arange(L)

    ax.set_xticks(centers[tick_pos])
    ax.set_xticklabels([str(k) for k in tick_pos])
    ax.set_yticks(centers[tick_pos])
    ax.set_yticklabels([str(k) for k in tick_pos])

    ax.set_xlabel("Position j")
    ax.set_ylabel("Position i")
    ax.set_title(f"Raw double-mutant scores: {L}×{L} position cells, each cell is 4×4 nucleotides")

    # Subtle inner lines separating 1x1 mini squares (within each 4x4)
    for i in range(L):
        base_y = i * stride
        for a in range(1, cell):
            ax.axhline(base_y + a - 0.5, linewidth=0.3, alpha=inner_grid_alpha)
    for j in range(L):
        base_x = j * stride
        for b in range(1, cell):
            ax.axvline(base_x + b - 0.5, linewidth=0.3, alpha=inner_grid_alpha)

    # Dark outline around each 4x4 position-cell (i,j)
    for i in range(L):
        y0 = i * stride - 0.5
        for j in range(L):
            x0 = j * stride - 0.5
            rect = Rectangle(
                (x0, y0),
                cell, cell,
                linewidth=box_outline_lw,
                edgecolor=box_outline_color,
                facecolor="none",
            )
            ax.add_patch(rect)

    # Annotate each mini square with letters in bottom-right corner
    if annotate_mini:
        for i in range(L):
            base_y = i * stride
            for j in range(L):
                base_x = j * stride
                for a in range(4):
                    for b in range(4):
                        if mini_which == "row":
                            txt = nucs[a]
                        elif mini_which == "col":
                            txt = nucs[b]
                        elif mini_which == "both":
                            txt = f"{nucs[a]}{nucs[b]}"
                        else:
                            raise ValueError('mini_which must be "row", "col", or "both"')

                        ax.text(
                            base_x + b + 0.45,     # bottom-right of the 1x1 mini square
                            base_y + a - 0.45,
                            txt,
                            fontsize=mini_fontsize,
                            ha="right",
                            va="bottom",
                            alpha=mini_alpha,
                        )

    # Outline WT entry (a_wt,b_wt) inside each (i,j) cell (still useful for reference)
    if show_wt_pixel:
        for i in range(L):
            a_wt = int(wt_idx[i])
            for j in range(L):
                b_wt = int(wt_idx[j])
                r = i * stride + a_wt
                c = j * stride + b_wt
                rect = Rectangle(
                    (c - 0.5, r - 0.5),
                    1.0, 1.0,
                    linewidth=0.45,
                    edgecolor="black",
                    facecolor="none",
                    alpha=0.9
                )
                ax.add_patch(rect)

    fig.tight_layout()
    fig.savefig(f"{outputdir}/raw_pairs_nested.png", dpi=300, bbox_inches="tight")
    plt.show()
    
#------------------------------------------------------------------------------------------------------

oh_seq = rna_to_one_hot(input_seq)
no_padding_seq, _ = remove_padding(oh_seq) #needs to be in one-hot encoded form
print(no_padding_seq.shape)

padding_amt = 41 - oh_seq.shape[0] #residualbind requires that the sequence is 41 nucleotides long

seq_length = no_padding_seq.shape[0]
mut_window = [0, seq_length]

GLOBAL_MASK = np.tril(np.ones((seq_length, seq_length), dtype=bool), -1)
GLOBAL_MASK_T = GLOBAL_MASK.T

def max_with_sign(e):
    """
    e: (A, B, L, L)
    returns: (L, L) with the signed value e[a*, b*, i, j]
             where (a*,b*) = argmax over |e[:, :, i, j]|
    """
    A, B, L, _ = e.shape
    abs_e = np.abs(e)

    flat = abs_e.reshape(A * B, L, L)              # (A*B, L, L)
    all_nan = np.all(np.isnan(flat), axis=0)       # (L, L) mask of all-NaN positions

    # Replace NaNs with -inf so argmax works without errors
    safe = np.where(np.isnan(flat), -np.inf, flat) # (A*B, L, L)
    idx = np.argmax(safe, axis=0)                  # (L, L) indices in [0, A*B)

    # Map flat index back to (a,b)
    a_idx, b_idx = np.divmod(idx, B)               # each (L, L)

    # Build (i,j) grids
    I = np.arange(L)[:, None]                      # (L, 1)
    J = np.arange(L)[None, :]                      # (1, L)

    # Gather signed values
    out = e[a_idx, b_idx, I, J]                    # (L, L)

    # Put NaNs back where all entries were NaN
    out[all_nan] = np.nan
    return out

def epistasis_tensor(raw_pairs, singles_raw, F0, mask, outputdir):
    """
    raw_pairs:  (4,4,L,L)  -> F(x_ij^(a,b))
    singles_raw:(4,L)      -> F(x_i^(a))
    returns:    (4,4,L,L)  -> e_{ij}^{(a,b)}
    """
    A0, A1, L0, L1 = raw_pairs.shape
    assert A0==4 and A1==4 and L0==L1==singles_raw.shape[1]
    L = L0
    # broadcast: e[a,b,i,j] = F_ij[a,b,i,j] - F_i[a,i] - F_j[b,j] + F0
    e = raw_pairs.copy()
    e -= singles_raw[:, None, :, None]         # subtract F(x_i^(a))
    e -= singles_raw[None, :, None, :]         # subtract F(x_j^(b))
    e += F0

    maxwithsigntensor = max_with_sign(e)
    maxwithsigntensorcopy = maxwithsigntensor.copy()
    maxwithsigntensorcopy[mask] = 0
    maxwithsigntensorcopy = np.nan_to_num(maxwithsigntensorcopy, nan=0.0)
    np.save(f"{outputdir}/epistasis_{experiment}_masked.npy", maxwithsigntensorcopy)
    #np.savetxt(f"{outputdir}/max_sign_e_tensor_values.csv", maxwithsigntensorcopy, delimiter=",")
    return maxwithsigntensor

def epistasis_map(e_tensor, wt_idx, agg="maxabs", skip_wt=True):
    """
    e_tensor: (4,4,L,L)  epistasis for (a,b,i,j)
    wt_idx:   (L,)       WT base index per position
    agg: 'maxabs' | 'mean' | 'max' | 'min'
    """
    A, _, L, _ = e_tensor.shape
    mask = np.ones_like(e_tensor, dtype=bool)   # (4,4,L,L)

    if skip_wt:
        # a != WT_i  -> (4,L) -> (4,1,L,1)
        ai_ok = (np.arange(A)[:, None] != wt_idx[None, :])[:, None, :, None]
        # b != WT_j  -> (L,4) -> transpose to (4,L) -> (1,4,1,L)
        bj_ok = (np.arange(A)[None, :] != wt_idx[:, None]).T[None, :, None, :]
        mask &= ai_ok
        mask &= bj_ok

    e_masked = np.where(mask, e_tensor, np.nan)

    if   agg == "maxabs": E = np.nanmax(np.abs(e_masked), axis=(0,1))
    elif agg == "mean":   E = np.nanmean(e_masked,          axis=(0,1))
    elif agg == "max":    E = np.nanmax(e_masked,           axis=(0,1))
    elif agg == "min":    E = np.nanmin(e_masked,           axis=(0,1))
    else: raise ValueError("agg must be 'maxabs' | 'mean' | 'max' | 'min'")

    np.fill_diagonal(E, 0.0)
    E = 0.5*(E + E.T)
    return E

def plot_epistasis_maxpooled_like(e_tensor):
    # Max-pool by absolute value but KEEP SIGN of the argmax-ab combo
    # Flatten (a,b) -> 16
    max_pooled_arr = e_tensor.copy()
    '''

    max_pooled_arr = np.empty((seq_length,seq_length))
    for i in range(seq_length):
        for j in range(seq_length):
            intermediate = []
            for k in range(4):
                for l in range(4):
                    intermediate.append(e_use[k][l][i][j])
            max_pooled_arr[i][j] = max(intermediate, key=abs)'''

    mask = np.tril(np.ones_like(max_pooled_arr, dtype=bool), -1)

    flipped_arr = max_pooled_arr.T
    flipped_mask = mask.T

    min_val, max_val = np.nanmin(flipped_arr), np.nanmax(flipped_arr)

    # Plot
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(flipped_arr, mask=flipped_mask, cmap='seismic', center = 0, square=True, vmin=min_val, vmax=max_val)

    # Move ticks
    ax.xaxis.tick_bottom()        # Move x-axis to top
    ax.yaxis.tick_left()      # Move y-axis to right

    threshold=0.6 * max_val
    impt_pos = {}
    # Add rectangle borders around cells above threshold
    for i in range(max_pooled_arr.shape[0]):
        for j in range(max_pooled_arr.shape[1]):
            if not flipped_mask[i,j]:
                continue
            if np.abs(max_pooled_arr[i, j]) > threshold:
                rect = Rectangle((i, j), 1, 1, fill=False, edgecolor='black', linewidth=2)
                if (np.abs(i-j) > 7):
                    impt_pos[(i,j)] = max_pooled_arr[i, j]
                    rect = Rectangle((i, j), 1, 1, fill=False, edgecolor='yellow', linewidth=4)
                ax.add_patch(rect)

    plt.tight_layout()
    plt.savefig(f"epmaptest.png")

def plot_epistasis_map(E, outputdir):
    plt.figure(figsize=(6.5, 5.8))
    im = plt.imshow(E, origin="lower", interpolation="none", aspect="auto")
    plt.xlabel("j (position)"); plt.ylabel("i (position)")
    plt.title("Pairwise epistasis (max |ε| over bases)")
    cbar = plt.colorbar(im); cbar.set_label("epistasis")
    plt.tight_layout()
    plt.savefig(f"{outputdir}/epistasis_map.png", dpi=300)
    plt.show()

#Generate curated evaluation library, experiment specific

eval_lib_df = eval_library(f'/home/nagle/final_version/{experiment}_graphs/evallibrary.txt')
X_eval = np.stack(eval_lib_df["X"].values, axis=0).astype(np.float32)  # (N,L,4)
print(eval_lib_df["X"].iloc[0].shape)

# curated eval set (one-hot)
# use the one-hot eval arrays you already built globally
X_eval_local = X_eval
y_eval = eval_lib_df["y"].values.astype(float)                         # (N,)

# (optional) keep string df only for printing
onehot_df = pd.DataFrame({
    "X_seq": eval_lib_df["X"].apply(lambda s: "".join(onehot_to_seq(np.expand_dims(s, axis=0)))),
    "y": eval_lib_df["y"]
})
#print(onehot_df.head())

def _softmax(x: np.ndarray, axis: int = -1, temperature: float = 1.0) -> np.ndarray:
    t = max(float(temperature), 1e-12)
    z = x / t
    z = z - np.max(z, axis=axis, keepdims=True)
    ez = np.exp(z)
    return ez / np.sum(ez, axis=axis, keepdims=True)

def _mixture_of_tanh(
    s: np.ndarray,
    alphas: np.ndarray,          # (K,)
    betas: np.ndarray,           # (K,)
    mix_logits: np.ndarray,      # (K,)
    temperature: float = 1.0,
) -> np.ndarray:
    """
    g(s) = sum_k softmax(mix_logits/temperature)_k * tanh(alpha_k * s + beta_k)
    Returns: (N,1)
    """
    s = np.asarray(s, dtype=float)
    if s.ndim == 1:
        s = s.reshape(-1, 1)         # <-- critical fix (N,1)
    elif s.ndim == 2 and s.shape[1] != 1:
        # allow (N,1) only
        raise ValueError(f"s must be (N,) or (N,1); got {s.shape}")

    alphas = np.asarray(alphas, dtype=float).reshape(1, -1)  # (1,K)
    betas  = np.asarray(betas,  dtype=float).reshape(1, -1)  # (1,K)
    mix_logits = np.asarray(mix_logits, dtype=float).reshape(-1)  # (K,)

    pi = _softmax(mix_logits, axis=-1, temperature=temperature).reshape(1, -1)  # (1,K)

    comps = np.tanh(s * alphas + betas)      # (N,K)
    out = (comps * pi).sum(axis=-1, keepdims=True)  # (N,1)
    return out

def global_nonlinearity_additive_affinity(
    x_mut: np.ndarray,
    w: np.ndarray,
    b: float = 0.0,
    *,
    nonlin: str = "tanh",
    out_scale: float = 1.0,
    out_bias: float = 0.0,
    tanh_alpha: float = 1.0,
    tanh_beta: float = 0.0,
    gate_logits: Optional[np.ndarray] = None,   # <-- default None
    gate_temperature: float = 1.0,
    mix_alphas: Optional[np.ndarray] = None,    # <-- default None
    mix_betas: Optional[np.ndarray] = None,     # <-- default None
    mix_logits: Optional[np.ndarray] = None,    # <-- default None
    mix_temperature: float = 1.0,
) -> np.ndarray:
    s = additive_affinity(x_mut, w, b=b)  # (N,1)

    if nonlin == "tanh":
        y = np.tanh(float(tanh_alpha) * s + float(tanh_beta))

    elif nonlin == "softmax_gate":
        if gate_logits is None:
            gate_logits = np.array([0.0, 0.0], dtype=float)
        gate_logits = np.asarray(gate_logits, dtype=float).reshape(-1)
        if gate_logits.shape[0] != 2:
            raise ValueError("gate_logits must have shape (2,) for softmax_gate.")
        pi = _softmax(gate_logits, axis=-1, temperature=gate_temperature)
        y = pi[0] * s + pi[1] * np.tanh(float(tanh_alpha) * s + float(tanh_beta))

    elif nonlin == "mix_tanh":
        if mix_alphas is None or mix_betas is None or mix_logits is None:
            raise ValueError("mix_alphas, mix_betas, mix_logits are required for nonlin='mix_tanh'.")
        y = _mixture_of_tanh(
            s,
            np.asarray(mix_alphas, dtype=float).reshape(-1),
            np.asarray(mix_betas,  dtype=float).reshape(-1),
            np.asarray(mix_logits, dtype=float).reshape(-1),
            temperature=mix_temperature,
        )
    else:
        raise ValueError("nonlin must be one of: 'tanh', 'softmax_gate', 'mix_tanh'.")

    return (float(out_scale) * y + float(out_bias)).reshape(-1, 1)
def global_nonlinearity_additive_pairwise_affinity(
    x_mut: np.ndarray,
    w: np.ndarray,
    P: np.ndarray,
    b: float = 0.0,
    *,
    nonlin: str = "tanh",
    out_scale: float = 1.0,
    out_bias: float = 0.0,
    tanh_alpha: float = 1.0,
    tanh_beta: float = 0.0,
    gate_logits: Optional[np.ndarray] = None,   # <-- default None
    gate_temperature: float = 1.0,
    mix_alphas: Optional[np.ndarray] = None,    # <-- default None
    mix_betas: Optional[np.ndarray] = None,     # <-- default None
    mix_logits: Optional[np.ndarray] = None,    # <-- default None
    mix_temperature: float = 1.0,
) -> np.ndarray:
    s = additive_pairwise_affinity(x_mut, w, P, b=b)  # (N,1)

    if nonlin == "tanh":
        y = np.tanh(float(tanh_alpha) * s + float(tanh_beta))

    elif nonlin == "softmax_gate":
        if gate_logits is None:
            gate_logits = np.array([0.0, 0.0], dtype=float)
        gate_logits = np.asarray(gate_logits, dtype=float).reshape(-1)
        if gate_logits.shape[0] != 2:
            raise ValueError("gate_logits must have shape (2,) for softmax_gate.")
        pi = _softmax(gate_logits, axis=-1, temperature=gate_temperature)
        y = pi[0] * s + pi[1] * np.tanh(float(tanh_alpha) * s + float(tanh_beta))

    elif nonlin == "mix_tanh":
        if mix_alphas is None or mix_betas is None or mix_logits is None:
            raise ValueError("mix_alphas, mix_betas, mix_logits are required for nonlin='mix_tanh'.")
        y = _mixture_of_tanh(
            s,
            np.asarray(mix_alphas, dtype=float).reshape(-1),
            np.asarray(mix_betas,  dtype=float).reshape(-1),
            np.asarray(mix_logits, dtype=float).reshape(-1),
            temperature=mix_temperature,
        )
    else:
        raise ValueError("nonlin must be one of: 'tanh', 'softmax_gate', 'mix_tanh'.")

    return (float(out_scale) * y + float(out_bias)).reshape(-1, 1)


def soft_threshold(x: np.ndarray, lam: float) -> np.ndarray:
    """Prox operator for L1: argmin_z 0.5||z-x||^2 + lam||z||_1"""
    return np.sign(x) * np.maximum(np.abs(x) - lam, 0.0)

def init_additive_noWT(rng, wt_onehot, sigma=0.5, l1_w=0.0, bias=0.0):
    """
    wt_onehot: (L,4) wildtype sequence
    returns:
        W_mut: (L,3) gaussian weights for NON-WT nucleotides
        mut_index_map: (L,3) which nuc index each column refers to
    """
    L = wt_onehot.shape[0]

    wt_idx = np.argmax(wt_onehot, axis=1)  # (L,)

    mut_index_map = np.zeros((L,3), dtype=int)

    for i in range(L):
        non_wt = [n for n in range(4) if n != wt_idx[i]]
        mut_index_map[i] = non_wt

    W_mut = rng.normal(0.0, sigma, size=(L,3)).astype(np.float32)

    if l1_w > 0:
        W_mut = soft_threshold(W_mut, l1_w)

    return W_mut, mut_index_map, float(bias)


def additive_affinity_noWT(x_mut, W_mut, mut_index_map, b=0.0):
    """
    x_mut: (N,L,4)
    W_mut: (L,3)
    mut_index_map: (L,3)
    returns: (N,1)
    """
    N, L, _ = x_mut.shape

    s = np.zeros(N, dtype=np.float32)

    for k in range(3):
        nuc_idx = mut_index_map[:, k]      # (L,)
        # gather x_mut[:,:,nuc_idx] position-wise
        # advanced indexing trick:
        Xk = x_mut[np.arange(N)[:,None], np.arange(L)[None,:], nuc_idx[None,:]]
        s += np.einsum("nl,l->n", Xk, W_mut[:,k])

    return (s + b).reshape(-1,1)

def pairwise_adjacent_noWT(x_mut, P_mut, mut_map, b=0.0):
    """
    x_mut: (N,L,4)
    P_mut: (L-1,3,3) weights for adjacent positions (i,i+1), excluding WT at each position
    mut_map: (L,3) nucleotide index per non-WT column
    returns: (N,1)
    """
    N, L, _ = x_mut.shape
    s = np.zeros(N, dtype=np.float32)

    for i in range(L - 1):
        a_idx = mut_map[i]      # (3,)
        b_idx = mut_map[i + 1]  # (3,)

        Xi = x_mut[:, i, a_idx]       # (N,3)
        Xj = x_mut[:, i + 1, b_idx]   # (N,3)

        # contribution: sum_{a,b} Xi[a]*Xj[b]*P_mut[i,a,b]
        s += np.einsum("na,nb,ab->n", Xi, Xj, P_mut[i])

    return (s + b).reshape(-1,1)

def apply_global_nonlin(s, nonlin_name, nonlin_kwargs):
    s = np.asarray(s, dtype=float)
    if s.ndim == 1:
        s = s.reshape(-1, 1)

    out_scale = float(nonlin_kwargs.get("out_scale", 1.0))
    out_bias  = float(nonlin_kwargs.get("out_bias", 0.0))

    if nonlin_name == "tanh":
        alpha = float(nonlin_kwargs.get("tanh_alpha", 1.0))
        beta  = float(nonlin_kwargs.get("tanh_beta", 0.0))
        y = np.tanh(alpha * s + beta)

    elif nonlin_name == "softmax_gate":
        gate_logits = np.asarray(nonlin_kwargs.get("gate_logits", np.array([0.0, 0.0])), dtype=float).reshape(-1)
        T = float(nonlin_kwargs.get("gate_temperature", 1.0))
        pi = _softmax(gate_logits, axis=-1, temperature=T)
        alpha = float(nonlin_kwargs.get("tanh_alpha", 1.0))
        beta  = float(nonlin_kwargs.get("tanh_beta", 0.0))
        y = pi[0] * s + pi[1] * np.tanh(alpha * s + beta)

    elif nonlin_name == "mix_tanh":
        y = _mixture_of_tanh(
            s,
            np.asarray(nonlin_kwargs["mix_alphas"], dtype=float).reshape(-1),
            np.asarray(nonlin_kwargs["mix_betas"],  dtype=float).reshape(-1),
            np.asarray(nonlin_kwargs["mix_logits"], dtype=float).reshape(-1),
            temperature=float(nonlin_kwargs.get("mix_temperature", 1.0)),
        )
    else:
        raise ValueError("nonlin_name must be 'tanh', 'softmax_gate', or 'mix_tanh'")

    return (out_scale * y + out_bias).reshape(-1, 1)

def uniformize_by_histogram(scores, X, n_bins=20000, clip_hi=98, seed=42):
    """
    Same spirit as your old version: equalize counts per bin.
    Differences vs your old version:
      - uses n_bins ~ O(100-500) by default (more stable than 20000)
      - uses [0, clip_hi] percentile range, like you did
      - samples per_bin = min nonzero bin count
    Returns: scores_uniform (N,), X_uniform (N,L,4)
    """
    rng = np.random.default_rng(seed)
    scores = np.asarray(scores, dtype=float).reshape(-1)
    X = np.asarray(X)

    finite = np.isfinite(scores)
    scores = scores[finite]
    X = X[finite]

    lo = np.nanpercentile(scores, 0.0)
    hi = np.nanpercentile(scores, clip_hi)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        raise ValueError("Scores have degenerate range; can't uniformize.")

    edges = np.linspace(lo, hi, int(n_bins) + 1)

    bin_ids = np.digitize(scores, edges, right=False) - 1
    bin_ids = np.clip(bin_ids, 0, int(n_bins) - 1)

    counts = np.bincount(bin_ids, minlength=int(n_bins))
    nonzero = counts[counts > 0]
    if nonzero.size == 0:
        raise ValueError("No non-empty bins; can't uniformize.")
    per_bin = int(nonzero.min())

    keep = []
    for b in range(int(n_bins)):
        idx = np.where(bin_ids == b)[0]
        if idx.size == 0:
            continue
        if idx.size <= per_bin:
            keep.append(idx)
        else:
            keep.append(rng.choice(idx, size=per_bin, replace=False))

    keep = np.concatenate(keep) if keep else np.array([], dtype=int)
    return scores[keep], X[keep]


def write_eval_library_txt(path, X, y):
    """
    Writes: seq,y per line. X is (N,L,4) unpadded one-hot.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for Xi, yi in zip(X, y):
            seq = "".join(onehot_to_seq(Xi[None, ...]))
            f.write(f"{seq},{float(yi)}\n")


def make_4_gt_eval_libraries(
    *,
    x_base,                 # (N,L,4) base random library you generated
    input_seq_str,          # WT string (for init functions)
    gt_params,              # gt_bundle["params"]
    nonlin_name,            # "tanh" (or whichever you chose)
    nonlin_kwargs,          # dict for apply_global_nonlin
    out_dir,                # where to write
    n_bins=20000,             # histogram bins for uniformize
    clip_hi=98,
    seed=0
):
    """
    Returns:
      eval_libs: dict gt_key -> dict with X_eval, y_eval, txt_path
    """

    W_mut   = gt_params["W_mut"]
    mut_map = gt_params["mut_map"]
    b0      = gt_params["b"]
    edges   = gt_params["edges"]
    J       = gt_params["J"]

    # score the SAME base library under each GT function
    scores = compute_gt_scores_for_library_potts(
        x_base,
        W_mut=W_mut,
        mut_map=mut_map,
        b0=b0,
        nonlin_name=nonlin_name,
        nonlin_kwargs=nonlin_kwargs,
        edges=edges,
        J=J,
    )
    gt_keys = ["additive", "additive_pairwise", "nonlin_additive", "nonlin_additive_pairwise"]

    eval_libs = {}

    for k in gt_keys:
        yk = np.asarray(scores[k], dtype=float).reshape(-1)

        # uniformize by THIS GT score
        y_uni, X_uni = uniformize_by_histogram(
            yk, x_base,
            n_bins=n_bins,
            clip_hi=clip_hi,
            seed=seed + hash(k) % 10_000
        )

        # write file
        txt_path = os.path.join(out_dir, f"evallibrary_{k}.txt")
        write_eval_library_txt(txt_path, X_uni, y_uni)

        # optional plots (pre/post)
        try:
            plot_fig(yk, os.path.join(out_dir, f"evallibrarypreuniform_{k}.png"))
            plot_fig(y_uni, os.path.join(out_dir, f"uniformevallibrary_{k}.png"))
        except Exception as e:
            print(f"[warn] plotting failed for {k}: {e}")

        eval_libs[k] = {
            "X_eval": X_uni.astype(np.float32),
            "y_eval": y_uni.astype(float),
            "path": txt_path,
        }

        print(f"[GT eval] {k}: wrote {X_uni.shape[0]} seqs -> {txt_path}")

    return eval_libs

def _finite_1d(x):
    x = np.asarray(x).reshape(-1)
    return x[np.isfinite(x)]

def _flatten_runs(arr_list):
    """arr_list is list of 1D arrays over runs -> one long 1D vector"""
    if not arr_list:
        return np.array([], dtype=float)
    chunks = []
    for a in arr_list:
        if a is None:
            continue
        aa = _finite_1d(a)
        if aa.size:
            chunks.append(aa)
    return np.concatenate(chunks) if chunks else np.array([], dtype=float)

def _flatten_list_of_arrays(list_of_arrays):
    if not list_of_arrays:
        return np.array([], dtype=float)
    chunks = []
    for a in list_of_arrays:
        if a is None:
            continue
        aa = _finite_1d(a)
        if aa.size:
            chunks.append(aa)
    return np.concatenate(chunks) if chunks else np.array([], dtype=float)

def _fd_bins(x, clamp=(25, 80)):
    x = _finite_1d(x)
    n = x.size
    if n < 2:
        return clamp[0]
    q25, q75 = np.percentile(x, [25, 75])
    iqr = q75 - q25
    if iqr <= 0:
        return clamp[0]
    h = 2.0 * iqr * (n ** (-1/3))
    if h <= 0:
        return clamp[0]
    nb = int(np.ceil((x.max() - x.min()) / h))
    return max(clamp[0], min(clamp[1], nb))

def _robust_z(x, eps=1e-9):
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    s = 1.4826 * mad
    if not np.isfinite(s) or s < eps:
        s = np.nanstd(x)
    if not np.isfinite(s) or s < eps:
        s = 1.0
    return (x - med) / s

def _transform(x, mode):
    x = _finite_1d(x)
    if mode is None:
        return x
    if mode == "logabs":
        return np.log10(np.abs(x) + 1e-8)
    if mode == "robust_z":
        return _robust_z(x)
    raise ValueError("transform must be None | 'logabs' | 'robust_z'")

def plot_overlay_hist_kde(series_dict, *, title, out_png,
                          transform="logabs",
                          bins="fd",
                          density=True,
                          kde=True,
                          kde_bw_scale=1.4,
                          rug=True):
    os.makedirs(os.path.dirname(out_png), exist_ok=True)

    data = {lab: _transform(x, transform) for lab, x in series_dict.items()}
    pooled = np.concatenate([v for v in data.values() if v.size], axis=0) if any(v.size for v in data.values()) else np.array([])
    if pooled.size == 0:
        print(f"[warn] no data: {title}")
        return

    nb = _fd_bins(pooled) if bins == "fd" else int(bins)

    lo, hi = np.percentile(pooled, [0.5, 99.5])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo, hi = pooled.min(), pooled.max()
        if lo == hi:
            lo -= 1e-6; hi += 1e-6
    edges = np.linspace(lo, hi, nb + 1)
    grid = np.linspace(lo, hi, 400)

    plt.figure(figsize=(9, 5))

    for lab, x in data.items():
        if x.size == 0:
            continue

        plt.hist(
            x, bins=edges, density=density,
            histtype="step", linewidth=2.0,
            alpha=0.95,
            label=f"{lab} (N={x.size})",
        )

        if kde and x.size >= 5 and np.nanstd(x) > 0:
            kde_obj = gaussian_kde(x)
            kde_obj.set_bandwidth(bw_method=kde_obj.factor * float(kde_bw_scale))
            y = kde_obj(grid)
            plt.plot(grid, y, linewidth=2.0)

        if rug:
            ymin, ymax = plt.gca().get_ylim()
            rug_y = ymin + 0.02 * (ymax - ymin)
            plt.plot(x, np.full_like(x, rug_y), "|", markersize=10, alpha=0.25)

    xlabel = "coefficient"
    if transform == "logabs":
        xlabel = "log10(|coef| + 1e-8)"
    elif transform == "robust_z":
        xlabel = "robust z-score (median/MAD)"

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Density" if density else "Count")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

def plot_coef_dists_per_gt_overlay_across_cfgs(
    coef_store,
    outdir,
    experiment,
    gt_keys,
    cfg_names,
    which="additive",          # "additive" or "pairwise"
    transform="logabs",        # None | "logabs" | "robust_z"
):
    """
    Overlays ALL configs on one plot per GT key.
    This is what you want when scales differ.
    """
    os.makedirs(outdir, exist_ok=True)

    for gt_key in gt_keys:
        series = {}
        for cfg in cfg_names:
            try:
                arrs = coef_store[cfg][which][gt_key]  # list of arrays
            except KeyError:
                arrs = []
            series[cfg] = _flatten_list_of_arrays(arrs)

        plot_overlay_hist_kde(
            series,
            title=f"{experiment} — {gt_key} — {which} coef dist (overlay)",
            out_png=os.path.join(outdir, f"{experiment}_{gt_key}_{which}_overlay_histkde.png"),
            transform=transform,
            bins="fd",
            density=True,
            kde=True,
            kde_bw_scale=1.4,
            rug=True,
        )

def init_pairwise_adjacent_noWT(rng, wt_onehot, mut_map, sigma_P=0.2, l1_P=0.0):
    """
    returns P_mut: (L-1,3,3) for adjacent pairs, excluding WT at each position
    """
    L = wt_onehot.shape[0]
    P_mut = rng.normal(0.0, sigma_P, size=(L-1, 3, 3)).astype(np.float32)
    if l1_P > 0:
        P_mut = soft_threshold(P_mut, l1_P)
    return P_mut

def plot_random_library_distributions_three_nonlins(
    x_mut: np.ndarray,
    input_seq: str,
    *,
    rng=None,
    sigma_w: float = 0.5,
    sigma_P: float = 0.2,
    l1_w: float = 0.1,
    l1_P: float = 0.05,
    bias: float = 0.0,
    bins: int = 120,
    density: bool = True,
    save_dir: str = None,
    return_scores: bool = True,
):
    if rng is None:
        rng = np.random.default_rng(0)

    wt_oh = rna_to_one_hot(input_seq)

    # noWT additive params
    W_mut, mut_map, b0 = init_additive_noWT(
        rng, wt_oh, sigma=sigma_w, l1_w=l1_w, bias=bias
    )
    # noWT adjacent pairwise params
    P_mut = init_pairwise_adjacent_noWT(rng, wt_oh, mut_map, sigma_P=sigma_P, l1_P=l1_P)
    edges, J = init_pairwise_potts_optionA(
    rng, wt_oh,
    p_edge=0.30,
    df=5.0,
    lambda_J=0.5,
    p_rescue=0.10,
    wt_rowcol_zero=True,
    )

    def make_bins(arrs):
        all_y = np.concatenate([np.asarray(a).ravel() for a in arrs])
        lo, hi = np.percentile(all_y, [0.5, 99.5])
        return np.linspace(lo, hi, bins + 1)

    all_scores = {}
    wt_scores = {}

    def plot_one(nonlin_name, nonlin_kwargs):
        # base scores (WT gauge)
        s_add = additive_affinity_noWT(x_mut, W_mut, mut_map, b0)              # (N,1)
        #s_pair = pairwise_adjacent_noWT(x_mut, P_mut, mut_map, b=0.0)          # (N,1)
        s_pair = pairwise_potts_energy(x_mut, edges, J, b=0.0)
        s_addpair = s_add + s_pair

        y_add = s_add.reshape(-1)
        y_addpair = s_addpair.reshape(-1)
        y_nonlin_add = apply_global_nonlin(s_add, nonlin_name, nonlin_kwargs).reshape(-1)
        y_nonlin_addpair = apply_global_nonlin(s_addpair, nonlin_name, nonlin_kwargs).reshape(-1)

        # WT scores are 0 in this gauge
        wt0 = np.array([0.0])
        wt_add = 0.0
        wt_addpair = 0.0
        wt_nonlin_add = float(apply_global_nonlin(wt0, nonlin_name, nonlin_kwargs).reshape(-1)[0])
        wt_nonlin_addpair = wt_nonlin_add

        all_scores[nonlin_name] = {
            "additive": y_add,
            "additive_pairwise": y_addpair,
            "nonlin_additive": y_nonlin_add,
            "nonlin_additive_pairwise": y_nonlin_addpair,
        }
        wt_scores[nonlin_name] = {
            "additive": wt_add,
            "additive_pairwise": wt_addpair,
            "nonlin_additive": wt_nonlin_add,
            "nonlin_additive_pairwise": wt_nonlin_addpair,
        }

        bin_edges = make_bins([y_add, y_addpair, y_nonlin_add, y_nonlin_addpair])

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.hist(y_add, bins=bin_edges, density=density, alpha=0.3, label="Additive (noWT)")
        ax.hist(y_addpair, bins=bin_edges, density=density, alpha=0.3, label="Additive+Pairwise (noWT)")
        ax.hist(y_nonlin_add, bins=bin_edges, density=density, alpha=0.3, label=f"{nonlin_name} ∘ Additive")
        ax.hist(y_nonlin_addpair, bins=bin_edges, density=density, alpha=0.3, label=f"{nonlin_name} ∘ (Additive+Pairwise)")

        for v in [wt_add, wt_addpair, wt_nonlin_add, wt_nonlin_addpair]:
            ax.axvline(v, linestyle="--")

        ax.set_title("Random distribution per ground truth function")
        ax.set_xlabel("Ground-truth score (WT-referenced)")
        ax.set_ylabel("Density" if density else "Count")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()
        plt.tight_layout()

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(f"{save_dir}/random_dist_{nonlin_name}.png", dpi=300)
        plt.close(fig)

    # three nonlinear families
    plot_one("tanh", dict(tanh_alpha=0.5, tanh_beta=0.0, out_scale=1.0, out_bias=0.0))

    plot_one("mix_tanh", dict(
        mix_alphas=np.array([0.5, 1.5, 3.0]),
        mix_betas=np.array([0.0, 0.0, 0.0]),
        mix_logits=np.array([0.0, 0.0, 0.0]),
        mix_temperature=1.0
    ))

    plot_one("softmax_gate", dict(
        gate_logits=np.array([1.0, -1.0]),
        gate_temperature=1.0,
        tanh_alpha=0.5,
        tanh_beta=0.0
    ))

    if return_scores:
        return {
            #"params": {"W_mut": W_mut, "P_mut": P_mut, "b": b0, "mut_map": mut_map},
            "params": {"W_mut": W_mut, "edges": edges, "J": J, "b": b0, "mut_map": mut_map},
            "scores": all_scores,
            "wt_scores": wt_scores,
        }

def spearmanavg_by_cfg(rho_storage, output_dir, experiment, step="0", gt_keys=None):
    """
    Supports BOTH:
      1) new-style:
         rho_storage[step][cfg]["curated"][gt_key] -> list
         rho_storage[step][cfg]["random"][gt_key]  -> list

      2) old-style:
         rho_storage[step]["curated"][gt_key] -> list
         rho_storage[step]["random"][gt_key]  -> list
    """

    print("\n==============================")
    print("AVERAGE SPEARMAN COEFF VALUES")
    print("==============================")

    step_dict = rho_storage.get(step, {})

    # --------
    # Case A: old-style present
    # --------
    if "curated" in step_dict and "random" in step_dict and isinstance(step_dict["curated"], dict):
        if gt_keys is None:
            gt_keys = list(step_dict["curated"].keys())
        print("\n--- (old-style) ---")
        for k in gt_keys:
            cur = np.asarray(step_dict["curated"].get(k, []), dtype=float)
            rnd = np.asarray(step_dict["random"].get(k, []), dtype=float)
            cur_mean = np.nanmean(cur) if cur.size else np.nan
            rnd_mean = np.nanmean(rnd) if rnd.size else np.nan
            print(f"{k}: curated={cur_mean:.4f}   random={rnd_mean:.4f}")

    # --------
    # Case B: new-style configs (skip "curated"/"random")
    # --------
    cfg_names = []
    for name, val in step_dict.items():
        if name in ("curated", "random"):
            continue
        if isinstance(val, dict) and ("curated" in val) and ("random" in val):
            cfg_names.append(name)

    if not cfg_names:
        print("\n[warn] No cfg-style entries found under rho_storage[step].")
        return

    # infer gt_keys from first cfg if needed
    if gt_keys is None:
        cfg0 = cfg_names[0]
        gt_keys = list(step_dict[cfg0]["curated"].keys())

    for cfg_name in cfg_names:
        cur_dict = step_dict[cfg_name].get("curated", {})
        rnd_dict = step_dict[cfg_name].get("random", {})

        print(f"\n--- {cfg_name} ---")
        for k in gt_keys:
            cur = np.asarray(cur_dict.get(k, []), dtype=float)
            rnd = np.asarray(rnd_dict.get(k, []), dtype=float)

            cur_mean = np.nanmean(cur) if cur.size else np.nan
            rnd_mean = np.nanmean(rnd) if rnd.size else np.nan
            print(f"{k}: curated={cur_mean:.4f}   random={rnd_mean:.4f}")
            
def random_mut_library_round0(rho_storage, coef_store, run_idx, epmapmin, epmapmax):
    """
    Runs 4 surrogate configs on the SAME (x_mut, y_true) for each GT key.
    Also saves:
      - 4 plots: random-lib vs eval-lib distributions per GT key
      - GT weight distributions (W_mut vs J) per GT key
    """

    # -------------------------
    # Build random library
    # -------------------------
    pred_generator = squid.predictor.CustomPredictor(
        pred_fun=residbind.predict,
        reduce_fun="name",
        batch_size=512
    )
    mut_generator = squid.mutagenizer.RandomMutagenesis(mut_rate=mut_rate, uniform=False)
    mave = squid.mave.InSilicoMAVE(mut_generator, pred_generator, seq_length, mut_window=mut_window)

    print("Step 1: Generate randomly initialized library.")
    x_mut, y_mut = mave.generate(no_padding_seq, padding_amt=padding_amt, num_sim=int(lib_size))

    # -------------------------
    # Ground-truth synthetic params + (optional) distribution plots you already had
    # -------------------------
    gt_bundle = plot_random_library_distributions_three_nonlins(
        x_mut,
        input_seq=input_seq,
        rng=np.random.default_rng(run_idx),
        save_dir=f"/home/nagle/final_version/outputs/{subdir}/{activity}/dist_of_preds"
    )
    nonlin_family = "tanh"
    nonlin_kwargs = dict(tanh_alpha=0.5, tanh_beta=0.0, out_scale=1.0, out_bias=0.0)

    eval_outdir = f"/home/nagle/final_version/{experiment}_graphs/gt_eval_libs/run_{run_idx}"
    eval_libs = make_4_gt_eval_libraries(
        x_base=x_mut,                 # use SAME base random lib
        input_seq_str=input_seq,
        gt_params=gt_bundle["params"],# SAME GT params
        nonlin_name=nonlin_family,
        nonlin_kwargs=nonlin_kwargs,
        out_dir=eval_outdir,
        n_bins=20000,                   # <<<<< do NOT use 20000; this is the stable version
        clip_hi=98,
        seed=run_idx,
    )

    # -------------------------
    # Extract GT params (these define the GT functions)
    # -------------------------
    gt_params = gt_bundle["params"]
    W_mut   = gt_params["W_mut"]
    mut_map = gt_params["mut_map"]
    b0      = gt_params["b"]
    edges   = gt_params["edges"]
    J       = gt_params["J"]

    gt_keys_local = ["additive", "additive_pairwise", "nonlin_additive", "nonlin_additive_pairwise"]

    # -------------------------
    # Plot: RANDOM vs EVAL score distributions per GT key (4 plots)
    # IMPORTANT: compute scores for BOTH libraries using SAME GT params
    # -------------------------
    nonlin_family_for_plots = "tanh"
    nonlin_kwargs = dict(tanh_alpha=0.5, tanh_beta=0.0, out_scale=1.0, out_bias=0.0)

    scores_random = compute_gt_scores_for_library_potts(
        x_mut,
        W_mut=W_mut,
        mut_map=mut_map,
        b0=b0,
        nonlin_name=nonlin_family_for_plots,
        nonlin_kwargs=nonlin_kwargs,
        edges=edges,
        J=J,
    )

    # Build eval scores dict: each GT key gets scored on *its own* eval library
    scores_eval_for_plot = {}
    for k in gt_keys_local:
        Xk = eval_libs[k]["X_eval"]
        sk = compute_gt_scores_for_library_potts(
            Xk,
            W_mut=W_mut,
            mut_map=mut_map,
            b0=b0,
            nonlin_name=nonlin_family_for_plots,
            nonlin_kwargs=nonlin_kwargs,
            edges=edges,
            J=J,
        )
        scores_eval_for_plot[k] = sk[k]   # ONLY the matching GT key distribution

    # Convert into the dict shape plot_random_vs_eval_per_gt expects
    scores_eval = {k: scores_eval_for_plot[k] for k in gt_keys_local}

    plot_random_vs_eval_per_gt(
        scores_random,
        scores_eval,
        nonlin_label="GT",
        save_dir=f"/home/nagle/final_version/outputs/{subdir}/{activity}/random_vs_eval_gt_dists/run_{run_idx}",
        bins=120,
    )

    # -------------------------
    # Plot: GT WEIGHT distributions (W_mut vs J) per GT key
    # -------------------------
    gt_w_outdir = f"/home/nagle/final_version/outputs/{subdir}/{activity}/gt_weight_dists/run_{run_idx}"
    plot_gt_weight_dists_per_function(
        W_mut=W_mut,
        edges=edges,
        J=J,
        gt_keys=gt_keys_local,
        outdir=gt_w_outdir,
        bw_scale=1.25,
        clip=(0.5, 99.5),
        drop_pairwise_zeros=True,
    )

    # -------------------------
    # Choose which GT family you TRAIN surrogates on
    # (this is the only place where "tanh" matters for y_true)
    # -------------------------
    train_nonlin_family = "tanh"
    gt = gt_bundle["scores"][train_nonlin_family]
    gt_y = {k: np.asarray(gt[k], dtype=float).reshape(-1) for k in gt_keys_local}

    # -------------------------
    # init storage
    # -------------------------
    step = "0"
    rho_storage.setdefault(step, {})

    for cfg_name in SURROGATE_CONFIGS.keys():
        rho_storage[step].setdefault(cfg_name, {
            "curated": {k: [] for k in gt_keys_local},
            "random":  {k: [] for k in gt_keys_local},
        })

    for cfg_name in SURROGATE_CONFIGS.keys():
        coef_store.setdefault(cfg_name, {
            "additive": {k: [] for k in gt_keys_local},
            "pairwise": {k: [] for k in gt_keys_local},
        })

    # -------------------------
    # quick diagnostics on GT vectors
    # -------------------------
    for k, arr in gt_y.items():
        a = np.asarray(arr).reshape(-1)
        print(
            f"[GT {k}] N={a.size} finite%={np.mean(np.isfinite(a)):.3f} "
            f"nan={np.isnan(a).sum()} inf={np.isinf(a).sum()} "
            f"std={np.nanstd(a):.6f} min={np.nanmin(a):.3f} max={np.nanmax(a):.3f}"
        )

    # -------------------------
    # Run ALL 4 surrogate configs for each GT y_true
    # -------------------------
    for gt_key in gt_keys_local:
        y_true = np.asarray(gt_y[gt_key], dtype=float).reshape(-1, 1)

        m = np.isfinite(y_true[:, 0])
        x_use = x_mut[m]
        y_use = y_true[m]

        if y_use.shape[0] < 50 or np.nanstd(y_use) == 0:
            print(f"[skip] gt_key={gt_key}: N={y_use.shape[0]} std={np.nanstd(y_use)}")
            for cfg_name in SURROGATE_CONFIGS.keys():
                rho_storage[step][cfg_name]["curated"][gt_key].append(np.nan)
                rho_storage[step][cfg_name]["random"][gt_key].append(np.nan)
            continue

        X_eval_gt = eval_libs[gt_key]["X_eval"]
        y_eval_gt = eval_libs[gt_key]["y_eval"]

        for cfg_name in SURROGATE_CONFIGS.keys():
            out = surrogate(
                iteration="0",
                x_lib=x_use,
                y=y_use,
                cfg_name=cfg_name,
                X_test=X_eval_gt,
                y_test=y_eval_gt,
            )

            rho_storage[step][cfg_name]["curated"][gt_key].append(out["rho_curated"])
            rho_storage[step][cfg_name]["random"][gt_key].append(out["rho_random"])

            # store coefficients
            params = out.get("params", None)
            if params is not None and len(params) > 1:
                W_add = np.asarray(params[1], dtype=float).ravel()
                W_add = W_add[np.isfinite(W_add)]
                coef_store[cfg_name]["additive"][gt_key].append(W_add)

                if (SURROGATE_CONFIGS[cfg_name]["gpmap"] == "pairwise") and len(params) > 2:
                    W_pair = np.asarray(params[2], dtype=float).ravel()
                    W_pair = W_pair[np.isfinite(W_pair)]
                    coef_store[cfg_name]["pairwise"][gt_key].append(W_pair)

    # -------------------------
    # print summary for this run
    # -------------------------
    print(f"\nRun {run_idx} step 0 Spearman (curated vs random):")
    for cfg_name in SURROGATE_CONFIGS.keys():
        cur = {k: np.nanmean(rho_storage[step][cfg_name]["curated"][k][-1:]) for k in gt_keys_local}
        rnd = {k: np.nanmean(rho_storage[step][cfg_name]["random"][k][-1:])  for k in gt_keys_local}
        print(f"  [{cfg_name}] curated: { {k: f'{v:.3f}' for k, v in cur.items()} }")
        print(f"  [{cfg_name}] random : { {k: f'{v:.3f}' for k, v in rnd.items()} }")

    # -------------------------
    # write sequences
    # -------------------------
    out_seq_path = f"/home/nagle/final_version/outputs/{subdir}/{activity}/sequences_random_mut{run_idx}.txt"
    os.makedirs(os.path.dirname(out_seq_path), exist_ok=True)
    with open(out_seq_path, "w") as f:
        for i in x_mut:
            seq = "".join(onehot_to_seq(np.expand_dims(remove_padding(i)[0], axis=0)))
            f.write(seq + "\n")

    return mave, epmapmin, epmapmax
def init_pairwise_potts_optionA(
    rng,
    wt_onehot: np.ndarray,        # (L,4)
    p_edge: float = 0.9,         # fraction of interacting position pairs
    df: float = 2.0,              # Student-t degrees of freedom
    lambda_J: float = 3,       # coupling scale
    p_rescue: float = 0.10,       # fraction of entries allowed to be negative (rescue)
    wt_rowcol_zero: bool = True,  # epistasis only when both sites deviate from WT
):
    """
    Returns:
      edges: (M,2) int array of interacting pairs (i<j)
      J:     dense tensor (L,L,4,4) float32, only meaningful for (i<j) in edges; others are 0.
    """
    wt_onehot = np.asarray(wt_onehot, dtype=np.float32)
    assert wt_onehot.ndim == 2 and wt_onehot.shape[1] == 4
    L = wt_onehot.shape[0]
    wt_idx = np.argmax(wt_onehot, axis=1).astype(int)  # (L,)

    # sample interacting pairs
    edges = []
    for i in range(L):
        for j in range(i + 1, L):
            if rng.random() < p_edge:
                edges.append((i, j))
    edges = np.array(edges, dtype=np.int32)

    # dense tensor for simplicity
    J = np.zeros((L, L, 4, 4), dtype=np.float32)

    # fill each selected (i,j) with heavy-tailed entries
    for (i, j) in edges:
        M = np.abs(rng.standard_t(df, size=(4, 4))).astype(np.float32) * float(lambda_J)

        # rare rescue entries
        if p_rescue > 0:
            mask = (rng.random((4, 4)) < p_rescue)
            M[mask] *= -1.0

        if wt_rowcol_zero:
            wi = int(wt_idx[i])
            wj = int(wt_idx[j])
            M[wi, :] = 0.0
            M[:, wj] = 0.0
            M[wi, wj] = 0.0

        J[i, j, :, :] = M  # only store i<j

    return edges, J
def pairwise_potts_energy(
    x_mut: np.ndarray,     # (N,L,4) one-hot
    edges: np.ndarray,     # (M,2)
    J: np.ndarray,         # (L,L,4,4)
    b: float = 0.0,
) -> np.ndarray:
    """
    Computes pairwise Potts energy term:
      s_pair(n) = sum_{(i,j) in edges} J[i,j, x_i, x_j]
    Returns (N,1).
    """
    x_mut = np.asarray(x_mut, dtype=np.float32)
    assert x_mut.ndim == 3 and x_mut.shape[2] == 4
    N, L, _ = x_mut.shape

    # convert one-hot to indices (N,L)
    x_idx = np.argmax(x_mut, axis=2).astype(np.int32)

    s = np.zeros(N, dtype=np.float32)

    # loop over edges (M ~ 10-60 is fine)
    for (i, j) in edges:
        ai = x_idx[:, i]  # (N,)
        bj = x_idx[:, j]  # (N,)
        s += J[i, j, ai, bj]

    return (s + float(b)).reshape(-1, 1)

def ssmutagenesis_roundpostss(mave, rho_storage, epmapmin, epmapmax):
    #iteration 0: candidate lib is ss prediction
    x_mutnew, y_mutnew = mave.generate(no_padding_seq, padding_amt=padding_amt, num_sim=int(lib_size/4))
    candidate_lib = mutagenesis_pipeline(no_padding_seq, '/home/nagle/final_version/outputs/ss_preds', int(lib_size*3/4))

    seqs_with_padding, prediction = predictor(candidate_lib, "post_ss", plot=False)

    new_lib = np.concatenate([x_mutnew, seqs_with_padding], axis=0)
    new_lib_y = np.concatenate([y_mutnew, prediction], axis=0)

    with open(f"/home/nagle/final_version/outputs/{subdir}/{activity}/sequences_ss_mut.txt", "w") as f:
        for i in new_lib:
            seq = "".join(
                onehot_to_seq(
                    np.expand_dims(
                        np.expand_dims(remove_padding(i), axis=0)[0][0],
                        axis=0
                    )
                )
            )
            f.write(seq + "\n")

    top5, tempmin, tempmax, test_df_ignore, rho = surrogate("post_ss", new_lib, new_lib_y, 12,14, X_test=X_eval, y_test=y_eval)
    rho_storage["post_ss"].append(rho)
    print(f"this is the tempmin {tempmin}")
    print(f"this is the tempmax {tempmax}")
    if tempmin <= epmapmin:
        print(f"min is now {tempmin}")
        epmapmin = tempmin
    if tempmax >= epmapmax:
        print(f"max is now {tempmax}")
        epmapmax = tempmax
    return top5, epmapmin, epmapmax

def ism_round1_3(onehot_df, x_mut, epmapmin, epmapmax):
    #iterations 1-4: bottom percentile
    candidate_lib = x_mut
    for i in range(1,4):
        print(f"Iteration {i}")
        candidate_lib, candidate_lib_scores = bottom_percentile_iter(candidate_lib, i)

        seqs_with_padding, prediction = predictor(candidate_lib, i, plot=False)

        with open(f"/home/nagle/final_version/outputs/{subdir}/{activity}/sequences{i}_ss_mut.txt", "w") as f:
            for j in seqs_with_padding:
                seq = "".join(
                    onehot_to_seq(
                        np.expand_dims(
                            np.expand_dims(remove_padding(j), axis=0)[0][0],
                            axis=0
                        )
                    )
                )
                f.write(seq + "\n")

        top5, tempmin, tempmax, _ = surrogate(i, candidate_lib, candidate_lib_scores, 12,14, X_test=X_eval, y_test=y_eval)

        print(f"this is the tempmin {tempmin}")
        print(f"this is the tempmax {tempmax}")
        if tempmin <= epmapmin:
            print(f"min is now {tempmin}")
            epmapmin = tempmin
        if tempmax >= epmapmax:
            print(f"max is now {tempmax}")
            epmapmax = tempmax
    return top5, epmapmin, epmapmax

def roundoptimization(numrounds, onehot_df, rho_storage, top5, epmapmin, epmapmax):
    for i in range(1,numrounds):
        x_mut, y_mut = generateLibrary(top5, input_seq, "custom")

        with open(f"/home/nagle/final_version/outputs/{subdir}/{activity}/sequences{i}_ss_mut.txt", "w") as f:
            for j in x_mut:
                seq = "".join(
                    onehot_to_seq(
                        np.expand_dims(
                            np.expand_dims(remove_padding(j), axis=0)[0][0],
                            axis=0
                        )
                    )
                )
                f.write(seq + "\n")


        top5, tempmin, tempmax, test_df_ignore, rho = surrogate(i, x_mut, y_mut, 12, 14, X_test=X_eval, y_test=y_eval)
        rho_storage[i].append(rho)

        print(f"this is the tempmin {tempmin}")
        print(f"this is the tempmax {tempmax}")
        if tempmin <= epmapmin:
            print(f"min is now {tempmin}")
            epmapmin = tempmin
        if tempmax >= epmapmax:
            print(f"max is now {tempmax}")
            epmapmax = tempmax
    return top5, epmapmin, epmapmax

def ism_plots(output_dir=None):
    #generating ISM graphs
    #ism vs secondary structure heatmap
    #you need to input the given secondary structure
    single_mut_matrix = ism(no_padding_seq, "..................((((.....))))..........", output_dir) #NEED TO CHANGE

    delta_pairs, raw_pairs, F0, wt_idx = ism_double_tensor(
        no_padding_seq, predictor, batch_size=4096,
        skip_wt=True, include_diag=False, return_raw=True
    )

    output_dir_tensors = f'/home/nagle/final_version/outputs/{subdir}/{activity}/round_tensors_{experiment}'
    max_sign_e_tensor = epistasis_tensor(raw_pairs, single_mut_matrix, F0, GLOBAL_MASK_T, output_dir_tensors)   # (L,L)

    plot_epistasis_maxpooled_like(max_sign_e_tensor)

    plot_epistasis_map(max_sign_e_tensor, output_dir)

    # raw_pairs from ism_double_tensor(..., return_raw=True)
    plot_raw_pairs_as_LxL_cells_with_4x4_inside(raw_pairs, wt_idx, output_dir)

    print("raw min/max:", np.nanmin(raw_pairs), np.nanmax(raw_pairs))
    print("E   min/max:", np.nanmin(max_sign_e_tensor), np.nanmax(max_sign_e_tensor))
def _as_list_of_arrays(x):
    """Return [] or [np.array] or list of arrays."""
    if x is None:
        return []
    if isinstance(x, list):
        return [np.asarray(a).ravel() for a in x if a is not None and np.asarray(a).size]
    # if someone accidentally passed a single array
    if isinstance(x, np.ndarray):
        return [x.ravel()] if x.size else []
    # if someone accidentally passed a dict/defaultdict
    return []

def _logabs(x, eps=1e-8):
    x = _finite_1d(x)
    return np.log10(np.abs(x) + eps)

def _kde_line(x, grid, bw_scale=1.25):
    kde = gaussian_kde(x)
    kde.set_bandwidth(bw_method=kde.factor * float(bw_scale))
    return kde(grid)

def extract_active_potts_J(J, edges, drop_zeros=True):
    """
    J: (L,L,4,4), edges: (M,2)
    returns: 1D vector of active coupling entries from those edges.
    """
    vals = []
    for (i, j) in edges:
        block = np.asarray(J[i, j, :, :], dtype=float).reshape(-1)
        if drop_zeros:
            block = block[block != 0.0]
        if block.size:
            vals.append(block)
    if not vals:
        return np.array([], dtype=float)
    return _finite_1d(np.concatenate(vals))

def plot_gt_weight_dists_per_function(
    *, W_mut, edges, J,
    gt_keys,
    outdir,
    bw_scale=1.25,
    clip=(0.5, 99.5),
    drop_pairwise_zeros=True,
):
    """
    Creates distributions of *ground-truth weights* for each GT function:
      - additive weights: W_mut
      - pairwise weights: active Potts J on edges
    Saves per gt_key:
      - raw overlay KDE
      - logabs overlay KDE
    """
    os.makedirs(outdir, exist_ok=True)

    w_add = _finite_1d(W_mut)
    w_pair = extract_active_potts_J(J, edges, drop_zeros=drop_pairwise_zeros)

    # which GTs actually include pairwise weights
    needs_pair = set(["additive_pairwise", "nonlin_additive_pairwise"])

    for gt in gt_keys:
        for log_version in [False, True]:

            a = _logabs(w_add) if log_version else w_add
            p = _logabs(w_pair) if log_version else w_pair

            # if this GT doesn't include pairwise, we just plot additive
            include_pair = (gt in needs_pair)

            pooled = a.copy()
            if include_pair and p.size:
                pooled = np.concatenate([pooled, p])

            if pooled.size < 5:
                print(f"[skip] {gt} (log={log_version}) — not enough points")
                continue

            lo, hi = np.percentile(pooled, clip)
            if lo == hi:
                lo -= 1e-6; hi += 1e-6
            grid = np.linspace(lo, hi, 600)

            plt.figure(figsize=(8.5, 5))

            # additive KDE
            if a.size >= 5 and np.nanstd(a) > 0:
                plt.plot(grid, _kde_line(a, grid, bw_scale=bw_scale),
                         linewidth=2.3, label=f"GT additive weights (N={a.size})")

            # pairwise KDE (only if GT includes pairwise)
            if include_pair and p.size >= 5 and np.nanstd(p) > 0:
                plt.plot(grid, _kde_line(p, grid, bw_scale=bw_scale),
                         linewidth=2.3, linestyle="--",
                         label=f"GT pairwise weights (N={p.size})")

            xlabel = "weight"
            if log_version:
                xlabel = "log10(|weight| + 1e-8)"

            plt.title(f"Ground-truth weight distributions — {gt}" + (" (logabs)" if log_version else ""))
            plt.xlabel(xlabel)
            plt.ylabel("density")
            plt.grid(True, linestyle="--", alpha=0.35)
            plt.legend()
            plt.tight_layout()

            suffix = "logabs" if log_version else "raw"
            outpng = os.path.join(outdir, f"GT_weight_dist_{gt}_{suffix}.png")
            plt.savefig(outpng, dpi=300)
            plt.close()
            print(f"[saved] {outpng}")

def _kde_curve(x, grid, bw_scale=1.3):
    kde = gaussian_kde(x)
    kde.set_bandwidth(bw_method=kde.factor * float(bw_scale))
    return kde(grid)

def plot_add_vs_pair_kde_for_cfg(
    coef_store,
    cfg_name,
    gt_key,
    *,
    experiment,
    outdir,
    bw_scale=1.3,
    clip=(0.5, 99.5),
):
    """
    Makes TWO plots for this cfg+gt_key:
      1) KDE(additive) vs KDE(pairwise)
      2) KDE(logabs additive) vs KDE(logabs pairwise)
    """
    os.makedirs(outdir, exist_ok=True)

    add_list  = coef_store.get(cfg_name, {}).get("additive", {}).get(gt_key, [])
    pair_list = coef_store.get(cfg_name, {}).get("pairwise", {}).get(gt_key, [])

    x_add  = _flatten_runs(add_list)
    x_pair = _flatten_runs(pair_list)

    if x_add.size < 5 or x_pair.size < 5:
        print(f"[skip] {cfg_name} {gt_key}: need >=5 points each (add={x_add.size}, pair={x_pair.size})")
        return

    # ---------- RAW ----------
    pooled = np.concatenate([x_add, x_pair])
    lo, hi = np.percentile(pooled, clip)
    if lo == hi:
        lo -= 1e-6; hi += 1e-6
    grid = np.linspace(lo, hi, 600)

    out_png = os.path.join(outdir, f"{experiment}_{cfg_name}_{gt_key}_add_vs_pair_KDE.png")

    plt.figure(figsize=(8.5, 5))
    plt.plot(grid, _kde_curve(x_add,  grid, bw_scale=bw_scale), linewidth=2.2, label=f"additive (N={x_add.size})")
    plt.plot(grid, _kde_curve(x_pair, grid, bw_scale=bw_scale), linewidth=2.2, label=f"pairwise (N={x_pair.size})")
    plt.title(f"{experiment} — {cfg_name} — {gt_key} — coef KDE (add vs pair)")
    plt.xlabel("coefficient value")
    plt.ylabel("density")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"[saved] {out_png}")

    # ---------- LOGABS ----------
    x_add_l  = _logabs(x_add)
    x_pair_l = _logabs(x_pair)

    pooledl = np.concatenate([x_add_l, x_pair_l])
    lo, hi = np.percentile(pooledl, clip)
    if lo == hi:
        lo -= 1e-6; hi += 1e-6
    gridl = np.linspace(lo, hi, 600)

    out_png_log = os.path.join(outdir, f"{experiment}_{cfg_name}_{gt_key}_add_vs_pair_KDE_logabs.png")

    plt.figure(figsize=(8.5, 5))
    plt.plot(gridl, _kde_curve(x_add_l,  gridl, bw_scale=bw_scale), linewidth=2.2, label=f"additive (N={x_add.size})")
    plt.plot(gridl, _kde_curve(x_pair_l, gridl, bw_scale=bw_scale), linewidth=2.2, label=f"pairwise (N={x_pair.size})")
    plt.title(f"{experiment} — {cfg_name} — {gt_key} — coef KDE (log10|coef|)")
    plt.xlabel("log10(|coef| + 1e-8)")
    plt.ylabel("density")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png_log, dpi=300)
    plt.close()
    print(f"[saved] {out_png_log}")

def get_coef_vec(coef_store, cfg, which, gt_key):
    arr_list = coef_store.get(cfg, {}).get(which, {}).get(gt_key, [])
    if not arr_list:
        return np.array([], dtype=float)
    x = np.concatenate([np.asarray(a).reshape(-1) for a in arr_list if a is not None], axis=0)
    return x[np.isfinite(x)]

def _transform(x, mode):
    if mode is None:
        return x
    if mode == "logabs":
        return np.log10(np.abs(x) + 1e-8)
    if mode == "robust_z":
        med = np.nanmedian(x)
        mad = np.nanmedian(np.abs(x - med))
        s = 1.4826 * mad if mad > 0 else np.nanstd(x)
        if not np.isfinite(s) or s == 0:
            s = 1.0
        return (x - med) / s
    raise ValueError("transform must be None | logabs | robust_z")

def spearman_bars_per_surrogate(
    rho_storage,
    *,
    output_dir,
    experiment,
    step="0",
    gt_keys=None,
    cfg_names=None,
    show_sem=True,        # error bars across runs
):
    """
    Saves 4 plots (one per cfg). Each plot has 8 bars:
      for each gt_key: [curated, random]
    """

    os.makedirs(output_dir, exist_ok=True)

    step_dict = rho_storage.get(step, {})
    if not step_dict:
        raise KeyError(f"No rho_storage found for step={step}. Keys: {list(rho_storage.keys())}")

    # infer cfg_names
    if cfg_names is None:
        cfg_names = [
            c for c in step_dict.keys()
            if isinstance(step_dict.get(c), dict) and "curated" in step_dict[c] and "random" in step_dict[c]
        ]

    if not cfg_names:
        raise ValueError(f"No cfg entries found under rho_storage[{step}]")

    # infer gt_keys
    if gt_keys is None:
        gt_keys = list(step_dict[cfg_names[0]]["curated"].keys())

    for cfg in cfg_names:
        cur = step_dict[cfg]["curated"]
        rnd = step_dict[cfg]["random"]

        # build bar data: (curated, random) per gt_key
        means = []
        sems = []
        labels = []

        for k in gt_keys:
            a = np.asarray(cur.get(k, []), dtype=float)
            b = np.asarray(rnd.get(k, []), dtype=float)

            # means
            a_mean = np.nanmean(a) if a.size else np.nan
            b_mean = np.nanmean(b) if b.size else np.nan

            # sems
            if show_sem:
                a_sem = (np.nanstd(a, ddof=1) / np.sqrt(np.sum(np.isfinite(a)))) if np.sum(np.isfinite(a)) > 1 else 0.0
                b_sem = (np.nanstd(b, ddof=1) / np.sqrt(np.sum(np.isfinite(b)))) if np.sum(np.isfinite(b)) > 1 else 0.0
            else:
                a_sem, b_sem = 0.0, 0.0

            means.extend([a_mean, b_mean])
            sems.extend([a_sem, b_sem])
            labels.extend([f"{k}\ncurated", f"{k}\nrandom"])

        x = np.arange(len(labels))

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(x, means, yerr=sems, capsize=4)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_ylabel("Spearman correlation")
        ax.set_title(f"{experiment} — Spearman (step {step}) — {cfg}\n(curated eval vs random test_df)")

        ax.axhline(0, linewidth=1)
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)

        plt.tight_layout()
        outpath = os.path.join(output_dir, f"{experiment}_spearman8bars_step{step}_{cfg}.png")
        plt.savefig(outpath, dpi=250)
        plt.close(fig)
        print(f"[saved] {outpath}")

# ---------- helpers ----------
def _finite_flat(arr_list):
    if not arr_list:
        return np.array([])
    chunks = []
    for a in arr_list:
        if a is None:
            continue
        x = np.asarray(a).reshape(-1)
        x = x[np.isfinite(x)]
        if x.size:
            chunks.append(x)
    return np.concatenate(chunks) if chunks else np.array([])

# ---------- MAIN GRID PLOT ----------
def plot_kde_grid_surrogate_vs_gt(
    coef_store,
    SURROGATE_CONFIGS,
    *,
    experiment,
    gt_keys,
    cfg_names,
    out_png,
    log_version=False,
    bw_scale=1.3,
):
    """
    Makes ONE 4x4 grid:

        rows = surrogate models
        cols = ground truths

    additive gpmap -> 1 KDE line
    pairwise gpmap -> 2 KDE lines
    """

    n_rows = len(cfg_names)
    n_cols = len(gt_keys)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.8 * n_cols, 3.2 * n_rows),
        sharex=False,
        sharey=False,
    )

    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)
    if n_cols == 1:
        axes = np.expand_dims(axes, axis=1)

    # -------------------------------------------------
    # LOOP GRID
    # -------------------------------------------------
    for r, cfg_name in enumerate(cfg_names):

        gpmap = SURROGATE_CONFIGS[cfg_name]["gpmap"]

        for c, gt_key in enumerate(gt_keys):

            ax = axes[r, c]

            add_list = coef_store.get(cfg_name, {}).get("additive", {}).get(gt_key, [])
            pair_list = coef_store.get(cfg_name, {}).get("pairwise", {}).get(gt_key, [])

            x_add = _finite_flat(add_list)
            x_pair = _finite_flat(pair_list) if gpmap == "pairwise" else np.array([])

            if log_version:
                x_add = _logabs(x_add)
                if gpmap == "pairwise":
                    x_pair = _logabs(x_pair)

            # decide pooled grid
            pooled = x_add
            if gpmap == "pairwise" and x_pair.size:
                pooled = np.concatenate([x_add, x_pair]) if x_add.size else x_pair

            if pooled.size < 5:
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
                continue

            lo, hi = np.percentile(pooled, [0.5, 99.5])
            if lo == hi:
                lo -= 1e-6
                hi += 1e-6

            grid = np.linspace(lo, hi, 500)

            # -------- additive line --------
            if x_add.size >= 5 and np.nanstd(x_add) > 0:
                y = _kde_curve(x_add, grid, bw_scale=bw_scale)
                ax.plot(grid, y, linewidth=2.0, label="additive")

            # -------- pairwise line --------
            if gpmap == "pairwise" and x_pair.size >= 5 and np.nanstd(x_pair) > 0:
                y = _kde_curve(x_pair, grid, bw_scale=bw_scale)
                ax.plot(grid, y, linewidth=2.0, linestyle="--", label="pairwise")

            # titles
            if r == 0:
                ax.set_title(gt_key, fontsize=11)

            if c == 0:
                ax.set_ylabel(cfg_name, fontsize=11)

            ax.grid(True, linestyle="--", alpha=0.35)

    # axis labels
    xlabel = "coefficient"
    if log_version:
        xlabel = "log10(|coef| + 1e-8)"

    fig.supxlabel(xlabel)
    fig.supylabel("density")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")

    fig.suptitle(f"{experiment} — surrogate × ground truth KDE grid", fontsize=13)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"[saved] {out_png}")

def plot_kde_coef_dists(
    coef_store,
    outdir,
    experiment,
    gt_keys,
    cfg_names,
    which="additive",        # or "pairwise"
    transform="logabs",      # None | "logabs" | "robust_z"
    bw_scale=1.4             # smoothing factor (>1 smoother)
):
    """
    Makes ONE KDE overlay per GT key across all surrogate configs.
    """

    os.makedirs(outdir, exist_ok=True)

    for gt_key in gt_keys:

        # collect data across configs
        series = {}
        for cfg in cfg_names:
            arr_list = coef_store.get(cfg, {}).get(which, {}).get(gt_key, [])
            series[cfg] = _finite_flat(arr_list)

        # pooled grid range
        pooled = np.concatenate([v for v in series.values() if v.size], axis=0) \
                 if any(v.size for v in series.values()) else np.array([])

        if pooled.size == 0:
            print(f"[skip] {gt_key} ({which}) — no data")
            continue

        pooled = _transform(pooled, transform)

        lo, hi = np.percentile(pooled, [0.5, 99.5])
        if lo == hi:
            lo -= 1e-6
            hi += 1e-6

        grid = np.linspace(lo, hi, 500)

        # ----------------------------
        # Plot
        # ----------------------------
        plt.figure(figsize=(9,5))

        for cfg, raw in series.items():
            if raw.size < 5:
                continue

            x = _transform(raw, transform)

            if np.nanstd(x) == 0:
                continue

            kde = gaussian_kde(x)
            kde.set_bandwidth(bw_method=kde.factor * bw_scale)
            y = kde(grid)

            plt.plot(grid, y, linewidth=2.2, label=f"{cfg} (N={x.size})")

        xlabel = "coefficient"
        if transform == "logabs":
            xlabel = "log10(|coef| + 1e-8)"
        elif transform == "robust_z":
            xlabel = "robust z-score (median/MAD)"

        plt.title(f"{experiment} — {gt_key} — {which} KDE")
        plt.xlabel(xlabel)
        plt.ylabel("Density")
        plt.grid(True, linestyle="--", alpha=0.35)
        plt.legend()
        plt.tight_layout()

        out = os.path.join(outdir, f"{experiment}_{gt_key}_{which}_KDE.png")
        plt.savefig(out, dpi=300)
        plt.close()
        print(f"[saved] {out}")

def plot_coef_dists_per_gt(coef_store, outdir, experiment, gt_keys, bins=120):
    """
    coef_store[cfg]["additive"][gt_key] -> [flat arrays over runs]
    coef_store[cfg]["pairwise"][gt_key] -> [flat arrays over runs]
    """
    os.makedirs(outdir, exist_ok=True)

    for cfg_name in coef_store.keys():
        for gt_key in gt_keys:
            add_list  = _as_list_of_arrays(coef_store[cfg_name]["additive"].get(gt_key, []))
            pair_list = _as_list_of_arrays(coef_store[cfg_name]["pairwise"].get(gt_key, []))

            add  = np.concatenate(add_list)  if len(add_list)  else np.array([])
            pair = np.concatenate(pair_list) if len(pair_list) else np.array([])

            # if totally empty, skip to avoid meaningless plots
            if add.size == 0 and pair.size == 0:
                print(f"[skip] {cfg_name} {gt_key}: no coefficients stored")
                continue

            fig, ax = plt.subplots(figsize=(9, 5))

            # shared bins for overlay
            allv = np.concatenate([add, pair]) if (add.size and pair.size) else (add if add.size else pair)
            lo, hi = np.percentile(allv, [1, 99]) if allv.size > 10 else (allv.min(), allv.max())
            if lo == hi:
                lo -= 1e-6; hi += 1e-6
            bin_edges = np.linspace(lo, hi, bins + 1)

            if add.size:
                ax.hist(add, bins=bin_edges, density=True, alpha=0.45, label=f"additive (n={add.size})")
            if pair.size:
                ax.hist(pair, bins=bin_edges, density=True, alpha=0.45, label=f"pairwise (n={pair.size})")

            ax.set_title(f"{experiment} — {cfg_name} — {gt_key} coefficient distributions")
            ax.set_xlabel("Coefficient value")
            ax.set_ylabel("Density")
            ax.grid(True, linestyle="--", alpha=0.35)
            ax.legend()

            plt.tight_layout()
            out = f"{outdir}/coef_dist_{experiment}_{cfg_name}_{gt_key}.png"
            plt.savefig(out, dpi=250)
            plt.close(fig)
            print(f"[saved] {out}")

def predictgivensequence(inputseq):
    oh_seq = rna_to_one_hot(inputseq)
    no_padding_seq, _ = remove_padding(oh_seq) #needs to be in one-hot encoded form

    padding_amt = 41 - oh_seq.shape[0] #residualbind requires that the sequence is 41 nucleotides long

    _, pred = predictor(np.expand_dims(no_padding_seq, axis=0), "test1", plot=False)
    return pred


STEPS = ["0", "post_ss"] + [str(i) for i in range(1, 10)]

gt_keys = ["additive", "additive_pairwise", "nonlin_additive", "nonlin_additive_pairwise"]
step = "0"

rho_storage = {
    step: {
        cfg_name: {
            "curated": {k: [] for k in gt_keys},
            "random":  {k: [] for k in gt_keys},
        }
        for cfg_name in SURROGATE_CONFIGS.keys()
    }
}


coef_store = {
    cfg_name: {
        "additive": defaultdict(list),  # coef_store[cfg]["additive"][gt_key] -> [1D arrays over runs]
        "pairwise": defaultdict(list),  # coef_store[cfg]["pairwise"][gt_key] -> [1D arrays over runs]
    }
    for cfg_name in SURROGATE_CONFIGS.keys()
}

for run_idx in range(numrounds):
    print(f"\n=== Run {run_idx+1}\n")

    # Reset per-run values
    epmapmin = 0
    epmapmax = 0
    
    mave, epmapmin, epmapmax = random_mut_library_round0(
        rho_storage=rho_storage,
        coef_store=coef_store, 
        run_idx=run_idx,
        epmapmin=epmapmin,
        epmapmax=epmapmax
    )

    #top5, epmapmin, epmapmax = ssmutagenesis_roundpostss(mave, rho_storage, epmapmin, epmapmax)

    #top5, epmapmin, epmapmax = roundoptimization(10, onehot_df, rho_storage, top5, epmapmin, epmapmax)

    ep_results = {}

    for j in [0]:#, 'post_ss']:#, 1, 2,3, 4, 5, 6, 7, 8, 9]:
            flippedarr, flippedmask = ep_map(j, epmapmin, epmapmax)
            ep_results[j] = {
                "arr": flippedarr,
                "mask": flippedmask,
            }
            ep_results[j]["arr"][ep_results[j]["mask"]] = 0
            np.save(f"/home/nagle/final_version/outputs/{subdir}/{activity}/round_tensors_{experiment}/epmap_{j}_{run_idx}_{experiment}_masked.npy", ep_results[j]["arr"])
            #np.savetxt(f"epmap_{j}_masked.csv", ep_results[j]["arr"], delimiter=",")

outdir = f"/home/nagle/final_version/outputs/{subdir}/{activity}/spearman_bars"
cfg_names = list(SURROGATE_CONFIGS.keys())
gt_keys = ["additive", "additive_pairwise", "nonlin_additive", "nonlin_additive_pairwise"]

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

plot_kde_grid_surrogate_vs_gt(
    coef_store,
    SURROGATE_CONFIGS,
    experiment=experiment,
    gt_keys=gt_keys,
    cfg_names=list(SURROGATE_CONFIGS.keys()),
    out_png=out_png_raw,
    log_version=False,
)

plot_kde_grid_surrogate_vs_gt(
    coef_store,
    SURROGATE_CONFIGS,
    experiment=experiment,
    gt_keys=gt_keys,
    cfg_names=list(SURROGATE_CONFIGS.keys()),
    out_png=out_png_log,
    log_version=True,
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

