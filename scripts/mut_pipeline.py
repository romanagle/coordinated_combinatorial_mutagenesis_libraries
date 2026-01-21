import argparse
import sys
sys.path.insert(0, '/home/nagle/final_version/squid-nn')
import squid
import tensorflow as tf  
import os
import numpy as np
import pandas as pd
import logomaker
from tensorflow import keras
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
#from comb_mut import onehot_to_seq
from scipy.stats import skew
from sklearn.preprocessing import StandardScaler
import mavenn
sys.path.append('/home/nagle/final_version/squid-manuscript/squid')
from ink import deep_sea
sys.path.append('/home/nagle/final_version/squid-nn/squid')
from matplotlib.patches import Rectangle
sys.path.append('/home/nagle/final_version/residualbind')    
from residualbind import ResidualBind
import helper, explain, dinuc_shuffle

parser = argparse.ArgumentParser(description="Run coordinated combinatorial mutagenesis using ResidualBind and RBPCompete datset")
parser.add_argument("--seq", type=str)
parser.add_argument("--mut_rate", type=int, default=4, help='Integer value for how many positions you want to mutate in a given sequence')
parser.add_argument("--lib_size", type=int, default=20000, help='How many mutagenized sequences you want in your library')

args = parser.parse_args()

input_seq = args.seq
mut_rate = args.mut_rate
lib_size = args.lib_size

if input_seq is None:
    raise ValueError("Please provide --seq")
elif len(input_seq) > 41:
    raise ValueError("Sequence needs to be smaller than 41 nucleotides.")
elif mut_rate > len(input_seq):
    raise ValueError("You can't mutate more positions than the length of your input sequence.")

normalization = 'log_norm'   # 'log_norm' or 'clip_norm'
ss_type = 'seq'     # 'seq', 'pu', or 'struct'
alphabet = ['A','C','G','U']
num_sims = lib_size

data_path = Path.home() / 'residualbind'/ 'data'/'RNAcompete_2013'/'rnacompete2013.h5'
results_path = helper.make_directory('/home/nagle/residualbind/results', 'rnacompete_2013')
save_path = '/home/nagle/residualbind/weights/log_norm_seq'

plot_path = helper.make_directory(save_path, 'FINAL')


experiment = 'RNCMPT00111' #VTS1
rbp_index = helper.find_experiment_index(data_path, experiment)
print('Analyzing: '+ experiment)

# load rbp dataset
train, valid, test = helper.load_rnacompete_data(data_path,
                                                     ss_type='seq',
                                                     normalization=normalization,
                                                     rbp_index=rbp_index)

#Adapted from: https://github.com/koo-lab/residualbind/blob/master/Figure3_VTS1_analysis.ipynb

# load residualbind model
input_shape = list(train['inputs'].shape)[1:]
num_class = 1
weights_path = os.path.join(save_path, experiment + '_weights.hdf5')
#model = tf.keras.models.load_model('ResidualBind32_ReLU_single_model.h5', custom_objects={"GELU": GELU})
model = ResidualBind(input_shape, num_class, weights_path)
model.load_weights()


def rna_to_one_hot(rna):
        return np.identity(4)[
            np.unique(dinuc_shuffle.string_to_char_array(rna), return_inverse=True)[1]
        ]
#print(rna_to_one_hot(input_seq).shape)


def remove_padding(seq):
    #seq: sequence to mutate, dimensions: (41, 4)
    no_pad = []
    for idx, nuc in enumerate(seq):
        if np.array_equal(nuc, np.zeros(4)):
            continue
        else:
            no_pad.append(nuc)
    return np.stack(no_pad), (len(seq) - np.stack(no_pad).shape[0])

pred_generator = squid.predictor.CustomPredictor(pred_fun=model.predict, reduce_fun = "name",
                                                  batch_size=512
                                                  )
# set up mutagenizer class for in silico MAVE
mut_generator = squid.mutagenizer.RandomMutagenesis(mut_rate=0.1, uniform=False)

outputs = []
    
oh_seq = rna_to_one_hot(input_seq)
no_padding_seq, _ = remove_padding(oh_seq) #needs to be in one-hot encoded form

padding_amt = 41 - oh_seq.shape[0] #residualbind requires that the sequence is 41 nucleotides long


seq_length = no_padding_seq.shape[0]
mut_window = [0, seq_length]


# generate in silico MAVE
mave = squid.mave.InSilicoMAVE(mut_generator, pred_generator, seq_length, mut_window=mut_window)
    
x_mut, y_mut = mave.generate(no_padding_seq, padding_amt=padding_amt, num_sim=num_sims)


# choose surrogate model type
gpmap = 'pairwise'
    
# MAVE-NN model with GE nonlinearity
    
surrogate_model = squid.surrogate_zoo.SurrogateMAVENN(x_mut.shape, num_tasks=y_mut.shape[1],
                                                      gpmap=gpmap, regression_type='GE',
                                                      linearity='nonlinear', noise='SkewedT',
                                                      noise_order=2, reg_strength=0.4,
                                                      alphabet=alphabet, deduplicate=True,
                                                   gpu=True)

# train surrogate model
surrogate, mave_df = surrogate_model.train(x_mut, y_mut, learning_rate=5e-4, epochs=500, batch_size=100,
                                               early_stopping=True, patience=25, restore_best_weights=True,
                                               save_dir=None, verbose=1)
    
# retrieve model parameters
params = surrogate_model.get_params(gauge='consensus')

outputs.append((surrogate_model, surrogate, mave_df, params))

