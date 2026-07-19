"""
group1_intron/train_oracle.py

Train XGBoost oracle on Group I intron P1ex exhaustive fitness data.
Data: Pincus et al. 2021 (PRJNA636762) — 65,536 sequences (all 4^8 combos)
      8 variable positions, fitness = log2(spliced/unselected) at 30C and 37C

Input:  ../archive/utr/data/thermosensor_fitness_labeled.csv
Output: group1_intron/oracle/xgb_oracle_30C.pkl
        group1_intron/oracle/xgb_oracle_37C.pkl
        group1_intron/oracle/oracle_stats.json

Run:
    /home/nagle/miniconda3/envs/toehold_gpu/bin/python group1_intron/train_oracle.py
"""
from __future__ import annotations
import json, os, pickle
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.model_selection import train_test_split
import xgboost as xgb

_HERE    = os.path.dirname(os.path.abspath(__file__))
DATA_CSV = os.path.join(_HERE, "..", "archive", "utr", "data",
                        "thermosensor_fitness_labeled.csv")
OUT_DIR  = os.path.join(_HERE, "oracle")
os.makedirs(OUT_DIR, exist_ok=True)

BASES = "ACGT"
BASE_IDX = {b: i for i, b in enumerate(BASES)}

WT_GENOTYPE = "TTTTACCG"   # n_mut=0, fitness_30C=1.813

def one_hot(seq: str) -> np.ndarray:
    """8-position sequence → (32,) one-hot vector."""
    x = np.zeros(len(seq) * 4, dtype=np.float32)
    for i, b in enumerate(seq):
        if b in BASE_IDX:
            x[i * 4 + BASE_IDX[b]] = 1.0
    return x


def main():
    print("Loading data...", flush=True)
    df = pd.read_csv(DATA_CSV)
    print(f"  {len(df):,} sequences", flush=True)

    X_all = np.stack([one_hot(g) for g in df["genotype"]])
    stats = {}

    for temp in ("30C", "37C"):
        col  = f"fitness_{temp}"
        y_all = df[col].values.astype(np.float32)

        # drop NaN rows
        ok  = ~np.isnan(y_all)
        X   = X_all[ok]
        y   = y_all[ok]
        print(f"  {temp}: {ok.sum():,} / {len(ok):,} sequences with valid fitness", flush=True)

        idx = np.arange(len(y))
        idx_tr, idx_tmp = train_test_split(idx, test_size=0.2, random_state=42)
        idx_va, idx_te  = train_test_split(idx_tmp, test_size=0.5, random_state=42)

        X_tr, y_tr = X[idx_tr], y[idx_tr]
        X_va, y_va = X[idx_va], y[idx_va]
        X_te, y_te = X[idx_te], y[idx_te]

        print(f"\nTraining XGBoost ({temp})...", flush=True)
        model = xgb.XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            early_stopping_rounds=20,
            eval_metric="rmse",
        )
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=50)

        for split_name, Xs, ys in [("val", X_va, y_va), ("test", X_te, y_te)]:
            pred  = model.predict(Xs)
            rho   = float(spearmanr(ys, pred).correlation)
            r     = float(pearsonr(ys, pred)[0])
            print(f"  {split_name:5s}  ρ={rho:.4f}  r={r:.4f}  n={len(ys):,}", flush=True)
            stats[f"{temp}_{split_name}_spearman"] = round(rho, 4)
            stats[f"{temp}_{split_name}_pearson"]  = round(r,   4)

        # WT score
        wt_pred = float(model.predict(one_hot(WT_GENOTYPE).reshape(1, -1))[0])
        print(f"  WT pred = {wt_pred:.4f}  (true = 1.813)", flush=True)
        stats[f"{temp}_wt_pred"] = round(wt_pred, 4)

        out_path = os.path.join(OUT_DIR, f"xgb_oracle_{temp}.pkl")
        with open(out_path, "wb") as f:
            pickle.dump(model, f)
        print(f"  Saved → {out_path}", flush=True)

    stats["wt_genotype"] = WT_GENOTYPE
    stats_path = os.path.join(OUT_DIR, "oracle_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nSaved → {stats_path}", flush=True)
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == "__main__":
    main()
