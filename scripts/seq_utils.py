"""
seq_utils.py – Sequence encoding / decoding utilities.
"""

import sys
sys.path.append('/home/nagle/final_version/squid-manuscript/squid')
import numpy as np
import pandas as pd
from ink import deep_sea           # noqa: F401 – re-exported for convenience
from dinuc_shuffle import string_to_char_array


NUCS = ['A', 'C', 'G', 'U']


def rna_to_one_hot(rna):
    mapping = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
    chars = [chr(c) for c in string_to_char_array(rna)]
    indices = [mapping[char] for char in chars]
    return np.eye(4)[indices]


def remove_padding(seq):
    """seq: (41, 4) one-hot; returns (unpadded_seq, n_removed)."""
    no_pad = [nuc for nuc in seq if not np.array_equal(nuc, np.zeros(4))]
    no_pad = np.stack(no_pad)
    return no_pad, len(seq) - no_pad.shape[0]


def onehot_to_seq(seq):
    """seq: (1, n, 4) or (n, 4) one-hot → list of nucleotide characters."""
    letters_seq = []
    for i in range(seq.shape[0]):
        for j in range(seq.shape[1]):
            if   np.array_equal(seq[i, j], [1., 0., 0., 0.]): letters_seq.append("A")
            elif np.array_equal(seq[i, j], [0., 1., 0., 0.]): letters_seq.append("C")
            elif np.array_equal(seq[i, j], [0., 0., 1., 0.]): letters_seq.append("G")
            elif np.array_equal(seq[i, j], [0., 0., 0., 1.]): letters_seq.append("U")
            else:
                print(f'Uneven padding at position {j}')
                break
    return letters_seq


def eval_library(path_to_library=None):
    """Load a seq,score text file and return a DataFrame with X (one-hot) and y columns."""
    seqs, scores = [], []
    with open(path_to_library, "r") as f:
        for line in f:
            if not line.strip():
                continue
            seq, score = line.strip().split(",")
            seqs.append(rna_to_one_hot(seq))
            scores.append(float(score))
    x_mut = np.stack(seqs, axis=0)              # (N, L, 4)
    y_mut = np.array(scores, dtype=np.float32)   # (N,)
    seq_list = [x_mut[i] for i in range(x_mut.shape[0])]
    return pd.DataFrame({"X": seq_list, "y": y_mut})


def write_eval_library_txt(path, X, y):
    """Write (N,L,4) unpadded one-hot + y scores to a seq,y text file."""
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for Xi, yi in zip(X, y):
            seq = "".join(onehot_to_seq(Xi[None, ...]))
            f.write(f"{seq},{float(yi)}\n")
