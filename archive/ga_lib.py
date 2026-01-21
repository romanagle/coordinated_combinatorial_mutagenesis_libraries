import sys
sys.path.insert(0, '/home/nagle/final_version/squid-nn')
import tensorflow as tf  
import os
import numpy as np
import pandas as pd
from tensorflow import keras
from pathlib import Path
import seaborn as sns
sys.path.append('/home/nagle/final_version/residualbind')    
from residualbind import ResidualBind
import helper, dinuc_shuffle
from tqdm import tqdm
from geneticalgorithm import geneticalgorithm as ga


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



def f(X):
    '''
    X = (1,41) array consisting of 0, 1, 2, 3 for ACGU respectively
    '''
    nucs = []
    for row in X:
        if row == 0:
            nucs.append('A')
        elif row == 1:
            nucs.append('C')
        elif row == 2:
            nucs.append('G')
        else:
            nucs.append('U')
    seq = "".join(nucs)

    def rna_to_one_hot(rna):
        oh = np.identity(4)[
            np.unique(dinuc_shuffle.string_to_char_array(rna), return_inverse=True)[1]
        ]
        return np.expand_dims(oh, axis=0)

    oh_input = rna_to_one_hot(seq)

    prediction = residbind.predict(oh_input)
    return -1 * prediction[0][0]

#print(predict(np.array([[0]]*41)))


varbound=np.array([[0,3]]*41)

algorithm_param = {'max_num_iteration': 300,\
                   'population_size':100,\
                   'mutation_probability':0.1,\
                   'elit_ratio': 0.01,\
                   'crossover_probability': 0.5,\
                   'parents_portion': 0.3,\
                   'crossover_type':'uniform',\
                   'max_iteration_without_improv':None}

model=ga(function=f,\
            dimension=41,\
            variable_type='int',\
            variable_boundaries=varbound,\
            algorithm_parameters=algorithm_param)

model.run()



num_seq = model.output_dict["variable"]

aff_score = -1 * model.output_dict["function"]
print(aff_score)
print(num_seq.shape)

def num_to_seq(arr):
    new_nucs = []
    for i in arr:
        if i == 0.0:
            new_nucs.append('A')
        elif i == 1.0:
            new_nucs.append('C')
        elif i == 2.0:
            new_nucs.append('G')
        else:
            new_nucs.append('U')
    seq = "".join(new_nucs)
    return seq

print(num_to_seq(num_seq))