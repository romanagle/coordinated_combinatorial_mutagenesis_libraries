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
parser.add_argument("--position1", type=int, help='first position')
parser.add_argument("--position2", type=int, help='second position')

args = parser.parse_args()

input_seq = args.seq
pos1 = args.position1
pos2 = args.position2


def rna_to_one_hot(rna):
    mapping = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
    chars = [chr(c) for c in dinuc_shuffle.string_to_char_array(rna)]
    indices = [mapping[char] for char in chars]
    return np.eye(4)[indices]

oh_seq = rna_to_one_hot(input_seq)
input_seq = oh_seq

#no_padding_seq, _ = remove_padding(oh_seq) #needs to be in one-hot encoded form

padding_amt = 41 - input_seq.shape[0] #residualbind requires that the sequence is 41 nucleotides long

seq_length = input_seq.shape[0]
mut_window = [0, seq_length]


def saturated_mut(position):
    mut_oh = {0: np.array([1, 0, 0, 0]),
            1: np.array([0, 1, 0, 0]),
            2: np.array([0, 0, 1, 0]),
            3: np.array([0, 0, 0, 1])}
    nucs = {'A': np.array([1, 0, 0, 0]),
            'C': np.array([0, 1, 0, 0]),
            'G': np.array([0, 0, 1, 0]),
            'U': np.array([0, 0, 0, 1])}

    mut_seqs = []
    curr_nuc_oh = [k for k, v in mut_oh.items() if np.array_equal(v, input_seq[position])]
    curr_idx = list(mut_oh.keys()).index(curr_nuc_oh[0])
    for k in range(1,4):
        next_index = (curr_idx + k) % 4
        inserted_mut = np.array(mut_oh[next_index]).reshape((1,4))
        letter = [k for k, v in nucs.items() if np.array_equal(v, mut_oh[next_index])][0]
        new_seq = np.concatenate([input_seq[0:position], inserted_mut, input_seq[position+1:]])
        mut_seqs.append((letter, new_seq))
        #print("".join(onehot_to_seq(np.expand_dims(new_seq, axis=0))))

    new_lib = np.array(mut_seqs)
    return new_lib



def predictor(input_library):
    #input_library has no padding
    justlib = np.array([k[1] for k in input_library])
    justletters = np.array([k[0] for k in input_library])
    print(f"Here is the size of the input library: {justlib.shape}")

    beg_padding = np.zeros((justlib.shape[0], padding_amt//2, 4))
    end_padding = np.zeros((justlib.shape[0], padding_amt//2 + (padding_amt % 2), 4))

    seqs_with_padding = np.concatenate([beg_padding, justlib, end_padding], axis=1)
    prediction = residbind.predict(seqs_with_padding)

    output = []
    for letters, prediction in zip(justletters, prediction):
        output.append((letters, prediction))

    return output

def original_prediction(input_library):
    beg_padding = np.zeros((input_library.shape[0], padding_amt//2, 4))
    end_padding = np.zeros((input_library.shape[0], padding_amt//2 + (padding_amt % 2), 4))

    seqs_with_padding = np.concatenate([beg_padding, input_library, end_padding], axis=1)
    prediction = residbind.predict(seqs_with_padding)

    return prediction

alphabet=['A', 'G', 'C', 'U']

def double(pos1, pos2):
    mut_oh = {0: np.array([1, 0, 0, 0]),
            1: np.array([0, 1, 0, 0]),
            2: np.array([0, 0, 1, 0]),
            3: np.array([0, 0, 0, 1])}
    nucs = {'A': np.array([1, 0, 0, 0]),
            'C': np.array([0, 1, 0, 0]),
            'G': np.array([0, 0, 1, 0]),
            'U': np.array([0, 0, 0, 1])}

    mut_seqs = []
    curr_nuc_oh_1 = [k for k, v in mut_oh.items() if np.array_equal(v, input_seq[pos1])]
    curr_idx_1 = list(mut_oh.keys()).index(curr_nuc_oh_1[0])
    curr_nuc_oh_2 = [k for k, v in mut_oh.items() if np.array_equal(v, input_seq[pos2])]
    curr_idx_2 = list(mut_oh.keys()).index(curr_nuc_oh_2[0])
    for k in range(1,4):
        for l in range(1,4):
            next_index_1 = (curr_idx_1 + k) % 4
            next_index_2 = (curr_idx_2 + l) % 4
            inserted_mut_1 = np.array(mut_oh[next_index_1]).reshape((1,4))
            inserted_mut_2 = np.array(mut_oh[next_index_2]).reshape((1,4))
            letter_1 = [k for k, v in nucs.items() if np.array_equal(v, mut_oh[next_index_1])][0]
            letter_2 = [k for k, v in nucs.items() if np.array_equal(v, mut_oh[next_index_2])][0]
            new_seq = np.concatenate([input_seq[0:pos1], inserted_mut_1, input_seq[pos1+1:pos2], inserted_mut_2, input_seq[pos2+1:]])
            mut_seqs.append(((letter_1, letter_2), new_seq))
            #print("".join(onehot_to_seq(np.expand_dims(new_seq, axis=0))))

    new_lib = np.array(mut_seqs)
    return new_lib

def tuple_selection(paired_list, x):
    for a, b in paired_list:
        if x == a:
            return b
        elif x == b:
            return a



predictionoriginal = original_prediction(np.expand_dims(input_seq, axis=0))
print(predictionoriginal)
pos1lib = saturated_mut(pos1)
output1 = predictor(pos1lib)
print(output1)
pos2lib = saturated_mut(pos2)
output2 = predictor(pos2lib)
print(output2)

one_hots = double(pos1, pos2)
prediction3 = predictor(one_hots)

print(prediction3)



