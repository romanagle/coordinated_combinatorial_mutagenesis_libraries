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
import matplotlib.pyplot as plt
 
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

def surrogate(iteration, x_lib, y, reg_strength, noise_order, X_test=None, y_test=None):

    gpmap='pairwise'

    surrogate_model = squid.surrogate_zoo.SurrogateMAVENN(x_lib.shape, num_tasks=y.shape[0],
                                                      gpmap=gpmap, regression_type='GE',
                                                      linearity='nonlinear', noise='SkewedT',
                                                      noise_order=noise_order, reg_strength=reg_strength,
                                                      alphabet=NUCS, deduplicate=True,
                                                   gpu=True)
    
    # train surrogate model
    surrogate, mave_df, test_df = surrogate_model.train(x_lib, y, learning_rate=5e-4, epochs=500, batch_size=100,
                                                early_stopping=True, patience=25, restore_best_weights=True,
                                                save_dir=None, verbose=1)
    
    print(test_df)

        # retrieve model parameters
    params = surrogate_model.get_params(gauge='consensus')

    fig, preds, g_truth = squid.impress.plot_y_vs_yhat(surrogate, mave_df)
    fig.savefig(f"/home/nagle/final_version/outputs/{subdir}/{activity}/reconstruction_loss/1214/y_y_hat{iteration}.png", dpi=300)

    if X_test is not None and y_test is not None:
        fig, preds, g_truth, rho = plot_y_vs_yhat(surrogate, X_test, y_test)
        fig.savefig(f"/home/nagle/final_version/outputs/{subdir}/{activity}/y_y_hatrandomlib{iteration}.png", dpi=300)



    plot_fig(preds, iteration)

    additive = squid.impress.plot_additive_logo(params[1], view_window=None, alphabet=NUCS)
    additive.savefig(f"/home/nagle/final_version/outputs/{subdir}/{activity}/additivemaps/1214/additivemap{iteration}.png", dpi=300)
    np.save(f"/home/nagle/final_version/outputs/{subdir}/{activity}/additive_weights/1214/weights_epoch_{iteration}.npy", params[1])

    pairwise = squid.impress.plot_pairwise_matrix(params[2], view_window=None, alphabet=NUCS)
    pairwise.savefig(f"/home/nagle/final_version/outputs/{subdir}/{activity}/squid_ep_maps/1214/SQUIDepmap{iteration}.png", dpi=300)
    np.save(f"/home/nagle/final_version/outputs/{subdir}/{activity}/pairwise_weights/1214/weights_epoch_{iteration}.npy", params[2])

    seq_length = params[2].shape[0]
    max_pooled_arr = np.empty((seq_length,seq_length))
    for i in range(seq_length):
        for j in range(seq_length):
            intermediate = []
            for k in range(4):
                for l in range(4):
                    intermediate.append(params[2][i][k][j][l])
            max_pooled_arr[i][j] = max(intermediate, key=abs)

    threshold=0.1
    impt_pos = {}
    # Add rectangle borders around cells above threshold
    for i in range(max_pooled_arr.shape[0]):
        for j in range(max_pooled_arr.shape[1]):
            if np.abs(max_pooled_arr[i, j]) > threshold:
                if (np.abs(i-j) > 7):
                    impt_pos[(i,j)] = max_pooled_arr[i, j]
    sorted_by_weight = list(sorted(impt_pos.items(), key=lambda item: abs(item[1]), reverse=True))[:5]
    maxweight = list(sorted(impt_pos.items(), key=lambda item: item[1], reverse=True))[0][1]
    minweight = list(sorted(impt_pos.items(), key=lambda item: item[1]))[0][1]
    selected_post = [k for k, v in sorted_by_weight]

    plt.tight_layout()
    plt.savefig(f"/home/nagle/final_version/outputs/{subdir}/{activity}/new_ep_maps/1214/NEWepmap{iteration}.png")

    return selected_post, minweight, maxweight, test_df, rho
    #return None, None, None, None

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
    scale="percentile",           # "percentile", "minmax", or "fixed"
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
print(eval_lib_df["X"].iloc[0].shape)

eval_lib_df["X_seq"] = eval_lib_df["X"].apply(lambda s: "".join(onehot_to_seq(np.expand_dims(s, axis=0))))
onehot_df = pd.DataFrame({"X": list(eval_lib_df["X_seq"]), "y": eval_lib_df["y"]})
#print(onehot_df.head())


def random_mut_library_round0(rho_storage, run_idx, epmapmin, epmapmax):
    # Load data from given path
    pred_generator = squid.predictor.CustomPredictor(
        pred_fun=residbind.predict,
        reduce_fun="name",
        batch_size=512
    )
    mut_generator = squid.mutagenizer.RandomMutagenesis(mut_rate=mut_rate, uniform=False)
    mave = squid.mave.InSilicoMAVE(
        mut_generator, pred_generator, seq_length, mut_window=mut_window
    )

    print("Step 1: Generate randomly initialized library.")
    x_mut, y_mut = mave.generate(no_padding_seq, padding_amt=padding_amt, num_sim=int(lib_size))



    with open(f"/home/nagle/final_version/outputs/{subdir}/{activity}/sequences_random_mut{run_idx}.txt", "w") as f:
        for i in x_mut:
            seq = "".join(
                onehot_to_seq(
                    np.expand_dims(
                        np.expand_dims(remove_padding(i), axis=0)[0][0],
                        axis=0
                    )
                )
            )
            f.write(seq + "\n")

    plot_fig(y_mut, f"0_{run_idx}")

    top5, tempmin, tempmax, test_df_ignore, rho = surrogate(f"0", x_mut, y_mut, 12, 14, X_test=onehot_df["X"], y_test=onehot_df["y"])
    rho_storage["0"].append(rho)

    print(f"this is the tempmin {tempmin}")
    print(f"this is the tempmax {tempmax}")

    if tempmin <= epmapmin:
        print(f"min is now {tempmin}")
        epmapmin = tempmin
    if tempmax >= epmapmax:
        print(f"max is now {tempmax}")
        epmapmax = tempmax
    return mave, top5, epmapmin, epmapmax


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

    top5, tempmin, tempmax, test_df_ignore, rho = surrogate("post_ss", new_lib, new_lib_y, 12,14, X_test=onehot_df["X"], y_test=onehot_df["y"])
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

        top5, tempmin, tempmax, _ = surrogate(i, candidate_lib, candidate_lib_scores, 12,14, X_test=onehot_df["X"], y_test=onehot_df["y"])

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


        top5, tempmin, tempmax, test_df_ignore, rho = surrogate(i, x_mut, y_mut, 12, 14, X_test=onehot_df["X"], y_test=onehot_df["y"])
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


def spearmanavg(rho_storage, output_dir):
    print("\n" + "="*30)
    print(f"AVERAGE SPEARMAN COEFF VALUES ACROSS {numrounds} RUNS")
    print("="*30)

    for step in ["0", "post_ss"] + list(range(1, 10)):
        avg_rho = np.mean(rho_storage[step])
        print(f"Step {step}: {avg_rho:.4f}")

    print("="*30 + "\n")

    steps = ["0", "post_ss"] + list(range(1, 10))
    step_labels = [str(s) for s in steps]
    x = np.arange(len(step_labels))

    # Compute mean + error
    avg_rhos = [np.mean(rho_storage[step]) for step in steps]
    sem_rhos = [
        np.std(rho_storage[step], ddof=1) / np.sqrt(len(rho_storage[step]))
        for step in steps
    ]

    plt.figure()
    plt.errorbar(
        x,
        avg_rhos,
        yerr=sem_rhos,
        marker="o",
        capsize=4
    )
    plt.xticks(x, step_labels)
    plt.xlabel("Step")
    plt.ylabel("Average Spearman correlation")
    plt.title(f"Average Spearman correlation across {numrounds} runs")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/avg_spearman_across_steps_{experiment}.png", dpi=200)
    plt.close()

def ism_plots(output_dir=None):
    #generating ISM graphs
    #ism vs secondary structure heatmap
    #you need to input the given secondary structure
    single_mut_matrix = ism(no_padding_seq, "..................((((.....))))..........", output_dir)

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

def predictgivensequence(inputseq):
    oh_seq = rna_to_one_hot(inputseq)
    no_padding_seq, _ = remove_padding(oh_seq) #needs to be in one-hot encoded form

    padding_amt = 41 - oh_seq.shape[0] #residualbind requires that the sequence is 41 nucleotides long

    _, pred = predictor(np.expand_dims(no_padding_seq, axis=0), "test1", plot=False)
    return pred

'''
# Initialize rho storage
rho_storage = {step: [] for step in ["0", "post_ss"] + list(range(1, 10))}

for run_idx in range(numrounds):
    print(f"\n=== Run {run_idx+1}\n")

    # Reset per-run values
    epmapmin = 0
    epmapmax = 0
    
    mave, top5, epmapmin, epmapmax = random_mut_library_round0(rho_storage, run_idx, epmapmin, epmapmax)

    top5, epmapmin, epmapmax = ssmutagenesis_roundpostss(mave, rho_storage, epmapmin, epmapmax)

    top5, epmapmin, epmapmax = roundoptimization(10, onehot_df, rho_storage, top5, epmapmin, epmapmax)

    ep_results = {}

    for j in [0, 'post_ss', 1, 2,3, 4, 5, 6, 7, 8, 9]:
            flippedarr, flippedmask = ep_map(j, epmapmin, epmapmax)
            ep_results[j] = {
                "arr": flippedarr,
                "mask": flippedmask,
            }
            ep_results[j]["arr"][ep_results[j]["mask"]] = 0
            np.save(f"/home/nagle/final_version/outputs/{subdir}/{activity}/round_tensors_{experiment}/epmap_{j}_{run_idx}_{experiment}_masked.npy", ep_results[j]["arr"])
            #np.savetxt(f"epmap_{j}_masked.csv", ep_results[j]["arr"], delimiter=",")

spearmanavg(rho_storage,f'/home/nagle/final_version/outputs/{subdir}/{activity}')

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