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
sys.path.append('/home/nagle/final_version/squid-manuscript/squid')
from ink import deep_sea
sys.path.append('/home/nagle/final_version/squid-nn/squid')
from matplotlib.patches import Rectangle
sys.path.append('/home/nagle/final_version/residualbind')    
from residualbind import ResidualBind
import helper, explain, dinuc_shuffle
from prediction import paired_positions, predict_ss
from tqdm import tqdm

normalization = 'log_norm'   # 'log_norm' or 'clip_norm'
ss_type = 'seq'     # 'seq', 'pu', or 'struct'
alphabet = ['A','C','G','U']

data_path = Path.home() / 'residualbind'/ 'data'/'RNAcompete_2013'/'rnacompete2013.h5'
results_path = helper.make_directory('/home/nagle/residualbind/results', 'rnacompete_2013')
save_path = '/home/nagle/residualbind/weights/log_norm_seq'

plot_path = helper.make_directory(save_path, 'FINAL')

experiment = 'RNCMPT00111' #VTS1
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

parser = argparse.ArgumentParser(description="Run coordinated combinatorial mutagenesis using ResidualBind and RBPCompete datset")
parser.add_argument("--seq", type=str)
parser.add_argument("--activity", type=str, help='TESTING low vs high activity')
parser.add_argument("--mut_rate", type=int, default=4, help='Integer value for how many positions you want to mutate in a given sequence')
parser.add_argument("--lib_size", type=int, default=20000, help='How many mutagenized sequences you want in your library')


args = parser.parse_args()

input_seq = args.seq
num_muts = args.mut_rate
lib_size = args.lib_size
activity = args.activity

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

def plot_fig(listofscores, iteration, scores=[]):
    print(len(listofscores))
    plt.figure(figsize=(8, 5))
    plt.hist(listofscores, density = True, bins=100, edgecolor='black', alpha=0.7)
    plt.title(f"Binding Affinity Score Distribution of Residualbind predictions after {iteration} round(s) of mut")
    plt.xlabel("Binding Affinity score")
    #plt.xlim(-2,2)
    #plt.axvline(scores[0], color='red', linestyle='dashed', linewidth=2, label=f'1')
    #plt.axvline(scores[1], color='green', linestyle='dashed', linewidth=2, label=f'2')
    #plt.axvline(scores[2], color='orange', linestyle='dashed', linewidth=2, label=f'3')
    plt.ylabel("Density")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"/home/nagle/final_version/output/{activity}_ism_graphs/dist_of_preds/iter_{iteration}_pred_binding_affinity_distribution.png", dpi=300)




def ep_map(iteration, x_lib, y, reg_strength):

    gpmap='pairwise'

    surrogate_model = squid.surrogate_zoo.SurrogateMAVENN(x_lib.shape, num_tasks=y.shape[0],
                                                      gpmap=gpmap, regression_type='GE',
                                                      linearity='nonlinear', noise='SkewedT',
                                                      noise_order=2, reg_strength=reg_strength,
                                                      alphabet=alphabet, deduplicate=True,
                                                   gpu=True)
    
    # train surrogate model
    surrogate, mave_df = surrogate_model.train(x_lib, y, learning_rate=5e-4, epochs=500, batch_size=100,
                                                early_stopping=True, patience=25, restore_best_weights=True,
                                                save_dir=None, verbose=1)

        # retrieve model parameters
    params = surrogate_model.get_params(gauge='consensus')

    fig, preds, g_truth = squid.impress.plot_y_vs_yhat(surrogate, mave_df)

    plot_fig(preds, iteration)


    pairwise = squid.impress.plot_pairwise_matrix(params[2], view_window=None, alphabet=alphabet)
    pairwise.savefig(f"/home/nagle/final_version/output/{activity}_ism_graphs/squid_ep_maps/SQUIDepmap{iteration}.png", dpi=300)

    seq_length = params[2].shape[0]
    max_pooled_arr = np.empty((seq_length,seq_length))
    for i in range(seq_length):
        for j in range(seq_length):
            intermediate = []
            for k in range(4):
                for l in range(4):
                    intermediate.append(params[2][i][k][j][l])
            max_pooled_arr[i][j] = max(intermediate, key=abs)

    mask = np.tril(np.ones_like(max_pooled_arr, dtype=bool), -1)

    flipped_arr = max_pooled_arr.T
    flipped_mask = mask.T

    # Plot
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(flipped_arr, mask=flipped_mask, cmap='seismic', vmin=-1, vmax=1, square=True)

    # Move ticks
    ax.xaxis.tick_bottom()        # Move x-axis to top
    ax.yaxis.tick_left()      # Move y-axis to right

    threshold=0.30
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
    sorted_by_weight = list(sorted(impt_pos.items(), key=lambda item: abs(item[1]), reverse=True))[:5]
    print(sorted_by_weight)
    selected_post = [k for k, v in sorted_by_weight]

    plt.tight_layout()
    plt.savefig(f"/home/nagle/final_version/output/{activity}_ism_graphs/new_ep_maps/NEWepmap{iteration}.png")

    return selected_post

def predictor(input_library, iteration, plot=True):
    #input_library has no padding
    print(f"Here is the size of the input library: {input_library.shape}")

    beg_padding = np.zeros((input_library.shape[0], padding_amt//2, 4))
    end_padding = np.zeros((input_library.shape[0], padding_amt//2 + (padding_amt % 2), 4))

    seqs_with_padding = np.concatenate([beg_padding, input_library, end_padding], axis=1)
    print(f"Here is the size of the input library: {seqs_with_padding.shape}")
    prediction = residbind.predict(seqs_with_padding)

    if plot:
        plot_fig(prediction, iteration)

    return seqs_with_padding, prediction



oh_seq = rna_to_one_hot(input_seq)
no_padding_seq, _ = remove_padding(oh_seq) #needs to be in one-hot encoded form

padding_amt = 41 - oh_seq.shape[0] #residualbind requires that the sequence is 41 nucleotides long

pred_generator = squid.predictor.CustomPredictor(pred_fun=residbind.predict, reduce_fun = "name",
                                                  batch_size=512
                                                  )

mut_rate = 4#round(num_muts*len(input_seq), 1)

mut_generator = squid.mutagenizer.RandomMutagenesis(mut_rate=mut_rate, uniform=False)



seq_length = no_padding_seq.shape[0]
mut_window = [0, seq_length]

mave = squid.mave.InSilicoMAVE(mut_generator, pred_generator, seq_length, mut_window=mut_window, paired_position_list=[])

x_mut, y_mut = mave.generate(no_padding_seq, padding_amt=padding_amt, num_sim=lib_size)

reg_strength = [0.1, 0.4, 0.6, 0.8, 1, 1.2, 1.5, 2, 4]

for i, reg in enumerate(reg_strength):
    ep_map(i, x_mut, y_mut, reg)

#------------------------------------------------------------------------------------------------------

'''
input_seqagc = "AGCUCGAUCGAUCGUGCUGGAAAACGAUCGAUCGAGCUU"
oh_seqagc = rna_to_one_hot(input_seqagc)
no_padding_seqagc, _ = remove_padding(oh_seqagc) #needs to be in one-hot encoded form

padding_amt = 41 - oh_seqagc.shape[0] #residualbind requires that the sequence is 41 nucleotides long

_, predagc = predictor(np.expand_dims(no_padding_seqagc, axis=0), "test1", plot=False)
print(f"this is the prediction forl the highest act seq: {predagc}")


input_seqaaa = "AAAAAGCGCUUUUUAGCUGGAAAAUAAAAAGCGCUUUUU"
oh_seqaaa = rna_to_one_hot(input_seqaaa)
no_padding_seqaaa, _ = remove_padding(oh_seqaaa) #needs to be in one-hot encoded form


padding_amt = 41 - oh_seqaaa.shape[0] #residualbind requires that the sequence is 41 nucleotides long

_, predaaa = predictor(np.expand_dims(no_padding_seqaaa, axis=0), "test1", plot=False)

input_seqaag = "AAGCGCUUUUUAGCUGGAAAAAAAAAAUAAAAAGCGCUUU"
oh_seqaag = rna_to_one_hot(input_seqaag)
no_padding_seqaag, _ = remove_padding(oh_seqaag) #needs to be in one-hot encoded form


padding_amt = 41 - oh_seqaag.shape[0] #residualbind requires that the sequence is 41 nucleotides long

_, predaag = predictor(np.expand_dims(no_padding_seqaag, axis=0), "test1", plot=False)

plot_fig(test["targets"], "test", [predaag, predaaa, predagc])

'''
'''
pred_generator = squid.predictor.CustomPredictor(pred_fun=residbind.predict, reduce_fun = "name",
                                                  batch_size=512
                                                  )
# set up mutagenizer class for in silico MAVE

mut_rate = round(num_muts/len(input_seq), 1)
mut_generator = squid.mutagenizer.RandomMutagenesis(mut_rate=mut_rate, uniform=False)
# generate in silico MAVE
mave = squid.mave.InSilicoMAVE(mut_generator, pred_generator, seq_length, mut_window=mut_window)
 
print("\nStep 1: Generate randomly initialized library.\n") 
x_mut, y_mut = mave.generate(no_padding_seq, padding_amt=padding_amt, num_sim=lib_size)

plot_fig(y_mut, 0)
ep_map(0, x_mut, y_mut)


#iteration -1: candidate lib is ss prediction
candidate_lib = mutagenesis_pipeline(no_padding_seq, '/home/nagle/final_version/output/ss_preds', lib_size)
seqs_with_padding, prediction = predictor(candidate_lib, "post_ss", plot=False)
top5 = ep_map("post_ss", seqs_with_padding, prediction)

#iterations 1-4: bottom percentile

for i in range(1,4):
    print(f"Iteration {i}")
    candidate_lib, candidate_lib_scores = bottom_percentile_iter(x_mut, i)
    top5 = ep_map(i, candidate_lib, candidate_lib_scores)
for i in range(4,15):
    x_mut, y_mut = generateLibrary(top5, input_seq, "custom")
    top5 = ep_map(i, x_mut, y_mut)
'''
'''
#iterations 1-4: bottom percentile
for i in range(1,4):
    print(f"Iteration {i}")
    candidate_lib, candidate_lib_scores = bottom_percentile_iter(candidate_lib, i)
top5 = ep_map(4, candidate_lib, candidate_lib_scores)
for i in range(5,15):
    x_mut, y_mut = generateLibrary(top5, input_seq, "custom")
    top5 = ep_map(i, x_mut, y_mut)
'''