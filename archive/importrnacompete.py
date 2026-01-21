import h5py
import numpy as np
import pandas as pd

exp_name = "RNCMPT00111"
h5_path = "/home/nagle/residualbind/data/RNAcompete_2013/rnacompete2013.h5"

NUCS = ["A", "C", "G", "U"]
L = 41  # sequence length

with h5py.File(h5_path, "r") as f:
    # Decode experiment names
    experiment_names = [e.decode("utf-8") for e in f["experiment"][()]]
    idx = experiment_names.index(exp_name)
    print("Found experiment index:", idx)

    def save_split(split):
        X = f[f"X_{split}"][()]      # (N, 4, 41)
        Y = f[f"Y_{split}"][:, idx]  # (N,)

        # Reorder X to (N, 41, 4)
        X = np.transpose(X, (0, 2, 1))

        # Flatten last two dims → (N, 41*4)
        N = X.shape[0]
        X_flat = X.reshape(N, -1)

        # Build column names for X
        feature_cols = [f"pos{p}_{n}" for p in range(L) for n in NUCS]

        # Create dataframe
        df = pd.DataFrame(X_flat, columns=feature_cols)
        df[exp_name] = Y

        outname = f"{exp_name}_{split}.csv"
        df.to_csv(outname, index=False)
        print(f"Wrote {outname}")

    for split in ["train", "valid", "test"]:
        save_split(split)