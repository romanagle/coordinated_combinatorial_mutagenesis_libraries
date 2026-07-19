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
parser.add_argument("--leftofstem", type=str)
parser.add_argument("--stemside1", type=str)
parser.add_argument("--middle", type=str)
parser.add_argument("--stemside2", type=str)
parser.add_argument("--rightofstem", type=str, default="")
parser.add_argument("--activity", type=str, help='TESTING low vs high activity')
parser.add_argument("--mut_rate", type=float, default=2, help='Integer value for how many positions you want to mutate in a given sequence')
parser.add_argument("--lib_size", type=int, default=20000, help='How many mutagenized sequences you want in your library')


args = parser.parse_args()

left_of_stem = args.leftofstem
stem_side_1 = args.stemside1
middle = args.middle
stem_side_2 = args.stemside2
right_of_stem = args.rightofstem
num_muts = args.mut_rate
lib_size = args.lib_size
activity = args.activity

plot_path = helper.make_directory(save_path, 'FINAL')



def ep_map(iteration, finalmin, finalmax):
    weights = np.load(f'/home/nagle/final_version/output/{activity}_ism_graphs/pairwise_weights/1214/weights_epoch_{iteration}.npy')

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

    # Plot
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(flipped_arr, mask=flipped_mask, cmap='seismic', square=True, vmin=finalmin, vmax=finalmax)

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
    plt.savefig(f"/home/nagle/final_version/output/{activity}_ism_graphs/new_ep_maps/1214/NEWepmap{iteration}.png")

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
    #print(seq)
  #seq: 3D array dimensions: (1, n, 4)
  # n <= 41, depending on padding amount
    letters_seq = []
    for i in range(seq.shape[0]):
        for j in range(seq.shape[1]):
            #print(seq[i,j])
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
    #print(len(listofscores))
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
    plt.savefig(f"/home/nagle/final_version/output/{activity}_ism_graphs/dist_of_preds/1214/iter_{iteration}_pred_binding_affinity_distribution.png", dpi=300)

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
    mutated_seq, mut_index = deep_sea(no_padding_seq, num_mut_seqs, 4, alphabet, 'uniform', paired_position_list, []) #deep_sea outputs a 3D array, not a 2d


    return mutated_seq

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
def surrogate(iteration, x_lib, y, reg_strength, noise_order):

    gpmap='pairwise'

    surrogate_model = squid.surrogate_zoo.SurrogateMAVENN(x_lib.shape, num_tasks=y.shape[0],
                                                      gpmap=gpmap, regression_type='GE',
                                                      linearity='nonlinear', noise='SkewedT',
                                                      noise_order=noise_order, reg_strength=reg_strength,
                                                      alphabet=alphabet, deduplicate=True,
                                                   gpu=True)
    
    # train surrogate model
    surrogate, mave_df = surrogate_model.train(x_lib, y, learning_rate=5e-4, epochs=500, batch_size=100,
                                                early_stopping=True, patience=25, restore_best_weights=True,
                                                save_dir=None, verbose=1)

        # retrieve model parameters
    params = surrogate_model.get_params(gauge='consensus')

    fig, preds, g_truth = squid.impress.plot_y_vs_yhat(surrogate, mave_df)
    fig.savefig(f"/home/nagle/final_version/output/{activity}_ism_graphs/reconstruction_loss/1214/y_y_hat{iteration}.png", dpi=300)

    plot_fig(preds, iteration)

    additive = squid.impress.plot_additive_logo(params[1], view_window=None, alphabet=alphabet)
    additive.savefig(f"/home/nagle/final_version/output/{activity}_ism_graphs/additivemaps/1214/additivemap{iteration}.png", dpi=300)
    np.save(f"/home/nagle/final_version/output/{activity}_ism_graphs/additive_weights/1214/weights_epoch_{iteration}.npy", params[1])

    pairwise = squid.impress.plot_pairwise_matrix(params[2], view_window=None, alphabet=alphabet)
    pairwise.savefig(f"/home/nagle/final_version/output/{activity}_ism_graphs/squid_ep_maps/1214/SQUIDepmap{iteration}.png", dpi=300)
    np.save(f"/home/nagle/final_version/output/{activity}_ism_graphs/pairwise_weights/1214/weights_epoch_{iteration}.npy", params[2])

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
    plt.savefig(f"/home/nagle/final_version/output/{activity}_ism_graphs/new_ep_maps/1214/NEWepmap{iteration}.png")

    return selected_post, minweight, maxweight
def predictor(input_library, iteration, plot=True):
    #input_library has no padding
    #print(f"Here is the size of the input library before: {input_library.shape}")

    beg_padding = np.zeros((input_library.shape[0], (41 - input_library.shape[1])//2, 4))
    end_padding = np.zeros((input_library.shape[0], (41 - input_library.shape[1])//2 + ((41 - input_library.shape[1]) % 2), 4))

    seqs_with_padding = np.concatenate([beg_padding, input_library, end_padding], axis=1)
    #print(f"Here is the size of the input library after: {seqs_with_padding.shape}")
    prediction = residbind.predict(seqs_with_padding)

    if plot:
        plot_fig(prediction, iteration)

    return seqs_with_padding, prediction
epmapmin = 0
epmapmax=0
oh_seq_left_of_stem = rna_to_one_hot(left_of_stem)
oh_seq_stem_side_1 = rna_to_one_hot(stem_side_1)
oh_middle = rna_to_one_hot(middle)
oh_seq_stem_side_2 = rna_to_one_hot(stem_side_2)
#oh_right_of_stem = rna_to_one_hot(right_of_stem)

no_padding_seq_left_of_stem, _ = remove_padding(oh_seq_left_of_stem)
no_padding_seq_stem_side_1, _  = remove_padding(oh_seq_stem_side_1)
no_padding_seq_middle, _  = remove_padding(oh_middle)
no_padding_seq_stem_side_2, _  = remove_padding(oh_seq_stem_side_2)
#no_padding_seq_right_of_stem, _  = remove_padding(oh_right_of_stem)


padding_amt_left_of_stem = 41 - oh_seq_left_of_stem.shape[0] #residualbind requires that the sequence is 41 nucleotides long
padding_amt_stem_side_1 = 41 - oh_seq_stem_side_1.shape[0]
padding_amt_middle = 41 - oh_middle.shape[0]
padding_amt_stem_side_2 = 41 - oh_seq_stem_side_2.shape[0]
#padding_amt_right_of_stem = 41 - oh_right_of_stem.shape[0]


seq_length_left_of_stem = no_padding_seq_left_of_stem.shape[0]
seq_length_stem_of_side_1 = no_padding_seq_stem_side_1.shape[0]
seq_length_middle = no_padding_seq_middle.shape[0]
seq_length_stem_of_side_2 = no_padding_seq_stem_side_2.shape[0]
#seq_length_right_of_stem = no_padding_seq_right_of_stem.shape[0]
mut_window_stem_side_1 = [0, seq_length_stem_of_side_1]
#print(mut_window_stem_side_1)
mut_window_stem_side_2 = [0, seq_length_stem_of_side_2]


pred_generator = squid.predictor.CustomPredictor(pred_fun=residbind.predict, reduce_fun = "name",
                                                  batch_size=512
                                                  )
# set up mutagenizer class for in silico MAVE

mut_rate = num_muts
mut_generator = squid.mutagenizer.RandomMutagenesis(mut_rate=mut_rate, uniform=False)
# generate in silico MAVE
#mave_l = squid.mave.InSilicoMAVE(mut_generator, pred_generator, seq_length_l, mut_window=mut_window_l)
mave_l = squid.mave.InSilicoMAVE(mut_generator, pred_generator, seq_length_stem_of_side_1, mut_window=mut_window_stem_side_1)
mave_r = squid.mave.InSilicoMAVE(mut_generator, pred_generator, seq_length_stem_of_side_2, mut_window=mut_window_stem_side_2)
#mave_r = squid.mave.InSilicoMAVE(mut_generator, pred_generator, seq_length_l, mut_window=mut_window_r)
x_mut_l, y_mut = mave_l.generate(no_padding_seq_stem_side_1, padding_amt=padding_amt_stem_side_1, num_sim=lib_size)
for i in range(x_mut_l.shape[0]):
    slice_2d = x_mut_l[i, :, :]
    print("".join(onehot_to_seq(np.expand_dims(slice_2d, axis=0))))

#x_mut_m, y_mut = mave_m.generate(no_padding_seq_motif, padding_amt=padding_amt_m, num_sim=lib_size)
x_mut_r, y_mut = mave_r.generate(no_padding_seq_stem_side_2, padding_amt=padding_amt_stem_side_2, num_sim=lib_size)


#x_mut = np.concatenate([np.broadcast_to(no_padding_seq_left_of_stem, (lib_size, seq_length_left_of_stem, 4)), x_mut_l, np.broadcast_to(no_padding_seq_middle, (lib_size, seq_length_middle, 4)), x_mut_r, np.broadcast_to(no_padding_seq_right_of_stem, (lib_size, seq_length_right_of_stem, 4))], axis=1)
x_mut = np.concatenate([np.broadcast_to(no_padding_seq_left_of_stem, (lib_size, seq_length_left_of_stem, 4)), x_mut_l, np.broadcast_to(no_padding_seq_middle, (lib_size, seq_length_middle, 4)), x_mut_r], axis=1)
seqs_with_padding, prediction = predictor(x_mut, "original", plot=False)
with open("/home/nagle/final_version/output/mut_stem_dists/libraries/seqs_mut_1.txt", "w") as f:
    for i in seqs_with_padding:
        seq = "".join(
            onehot_to_seq(
                np.expand_dims(
                    np.expand_dims(remove_padding(i), axis=0)[0][0],
                    axis=0
                )
            )
        )
        f.write(seq + "\n")



# plot and save normalized histogram
plt.figure(figsize=(6, 4))
plt.hist(prediction, edgecolor='black', density=True, bins=100)


input_seq = left_of_stem + stem_side_1 + middle + stem_side_2 + right_of_stem
print(f"this is the input seq {input_seq}")
oh_seq = rna_to_one_hot(input_seq)
no_padding_seq, _ = remove_padding(oh_seq) #needs to be in one-hot encoded form

padding_amt = 41 - oh_seq.shape[0] #residualbind requires that the sequence is 41 nucleotides long

_, pred = predictor(np.expand_dims(no_padding_seq, axis=0), "test1", plot=False)

pred = float(pred[0][0])


top5, tempmin, tempmax = surrogate("post_ss", seqs_with_padding, prediction, 12,14)
print(f"this is the tempmin {tempmin}")
print(f"this is the tempmax {tempmax}")
if tempmin <= epmapmin:
    print(f"min is now {tempmin}")
    epmapmin = tempmin
if tempmax >= epmapmax:
    print(f"max is now {tempmax}")
    epmapmax = tempmax



for i in range(1,15):
    x_mut, y_mut = generateLibrary(top5, input_seq, "custom")
    top5, tempmin, tempmax = surrogate(i, x_mut, y_mut, 12, 14)

    print(f"this is the tempmin {tempmin}")
    print(f"this is the tempmax {tempmax}")
    if tempmin <= epmapmin:
        print(f"min is now {tempmin}")
        epmapmin = tempmin
    if tempmax >= epmapmax:
        print(f"max is now {tempmax}")
        epmapmax = tempmax

for i in range(0,15):
    ep_map(i, epmapmin, epmapmax)

# Overlay a vertical line at the prediction value
plt.axvline(pred, color='red', linestyle='dashed', linewidth=2, label=f'Pred: {pred:.3f}')


plt.title(f"Normalized Histogram of Predictions of {input_seq}")
plt.xlabel("Prediction")
plt.ylabel("Density")

# save the figure
plt.savefig("/home/nagle/final_version/output/mut_stem_dists/mutate_stem_1.png")