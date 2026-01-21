import sys
sys.path.insert(0, '/home/nagle/final_version/squid-nn')
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

def predictors(input_library, iteration, plot=True):

    padding_amt = 0
    #input_library has no padding
    print(f"Here is the size of the input library: {input_library.shape}")

    beg_padding = np.zeros((input_library.shape[0], padding_amt//2, 4))
    end_padding = np.zeros((input_library.shape[0], padding_amt//2 + (padding_amt % 2), 4))

    seqs_with_padding = np.concatenate([beg_padding, input_library, end_padding], axis=1)
    prediction = residbind.predict(seqs_with_padding)

    return seqs_with_padding, prediction

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
                continue
    return letters_seq



# load residualbind model
input_shape = list(train['inputs'].shape)[1:]
num_class = 1
weights_path = os.path.join(save_path, experiment + '_weights.hdf5')

residbind = ResidualBind(input_shape, num_class, weights_path)
residbind.load_weights()

print('Analyzing: '+ experiment)
# evaluate model

np.random.seed(32)
x_sampled = test["inputs"]
y_sampled = test["targets"]

preds = []
seqs, prediction = predictors(x_sampled, "any", plot=False)

for seq, pred in zip(seqs, prediction):
    preds.append((seq, pred.item()))

highest_seqs = [seq for seq, pred in sorted(preds, key= lambda x: x[1], reverse=True)[:5]]
lowest_seqs = [seq for seq, pred in sorted(preds, key= lambda x: x[1])[:5]]

print("\nhighest activity sequences:\n")
for i in highest_seqs:
    cleaned_seq = "".join(onehot_to_seq(np.expand_dims(i, axis=0)))
    print(cleaned_seq)

print("\nlowest activity sequences:\n")
for i in lowest_seqs:
    cleaned_seq = "".join(onehot_to_seq(np.expand_dims(i, axis=0)))
    print(cleaned_seq)
