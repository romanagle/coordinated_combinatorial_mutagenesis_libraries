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
parser.add_argument("--experiment", type=str, help='ex: RNCMPT00111 or RNCMPT00042')
parser.add_argument("--seq", type=str)
parser.add_argument("--mut_rate", type=int, default=4, help='Integer value for how many positions you want to mutate in a given sequence')
parser.add_argument("--lib_size", type=int, default=20000, help='How many mutagenized sequences you want in your library')


args = parser.parse_args()


experiment = args.experiment
input_seq = args.seq
num_muts = args.mut_rate
lib_size = args.lib_size
mut_rate = (num_muts/41)

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

def plot_fig(listofscores, path):
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


oh_seq = rna_to_one_hot(input_seq)
no_padding_seq, _ = remove_padding(oh_seq) #needs to be in one-hot encoded form

padding_amt = 41 - oh_seq.shape[0] #residualbind requires that the sequence is 41 nucleotides long

seq_length = no_padding_seq.shape[0]
mut_window = [0, seq_length]


#------------------------------------------------------------------------------------------------------

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

def quantile_edges(scores, q=50):
    # Compute q-quantile edges; drop duplicates when ties collapse bins
    scores = np.asarray(scores).ravel()
    quantiles = np.linspace(0, 1, q + 1)
    edges = np.quantile(scores, quantiles, interpolation="linear")
    # Drop duplicate edges to avoid empty-width bins
    edges = np.unique(edges)
    return edges

def uniformize_by_histogram(scores,
                            x_mut,
                            n_bins=50):
    rng = np.random.default_rng(42)     # e.g., 0.1 to cap any single value at 10%
    scores = np.asarray(scores).ravel()
    lo, hi = np.nanpercentile(scores, 0.0), np.nanpercentile(scores, 98)
    edges = np.linspace(lo, hi, n_bins + 1)

    # 2) Assign bin ids (0..n_bins-1)
    bin_ids = np.digitize(scores, edges, right=False) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    counts = np.bincount(bin_ids, minlength=n_bins)
    per_bin = counts[counts > 0].min()

    # 4) Sample equal number per bin
    keep_idx = []
    for b in range(n_bins):
        idx = np.nonzero(bin_ids == b)[0]
        if len(idx) == 0:
            continue
        if len(idx) <= per_bin:
            keep_idx.extend(idx)
        else:
            keep_idx.extend(rng.choice(idx, size=per_bin, replace=False))

    keep_idx = np.array(keep_idx)
    scores_uniform = scores[keep_idx]

    # Store in DataFrame
    df = pd.DataFrame({
        "score": scores_uniform,
        "bin": bin_ids[keep_idx]
    })
    newx_mut = x_mut[keep_idx]
    df["X"] = [newx_mut[i] for i in range(newx_mut.shape[0])]

    return np.array(df["score"]), np.array(df["X"])


plot_fig(y_mut, f"/home/nagle/final_version/{experiment}_graphs/evallibrarypreuniform.png")
scores_uniform, x_uniform = uniformize_by_histogram(np.squeeze(y_mut),x_mut,n_bins=20000)
plot_fig(scores_uniform, f"/home/nagle/final_version/{experiment}_graphs/uniformevallibrary.png")

print(f"number of sequences: {len(scores_uniform)}")
with open(f"/home/nagle/final_version/{experiment}_graphs/evallibrary.txt", "w") as f:
    for i, j in zip(x_uniform, scores_uniform):
        seq = "".join(
            onehot_to_seq(
                np.expand_dims(
                    np.expand_dims(remove_padding(i), axis=0)[0][0],
                    axis=0
                )
            )
        )
        f.write(seq + "," + str(j) + "\n")