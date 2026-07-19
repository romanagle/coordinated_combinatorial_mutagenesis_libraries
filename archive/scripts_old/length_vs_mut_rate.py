"""length_vs_mut_rate.py

Sweeps sequence length × mutation rate and records Spearman ρ
on the random held-out MAVE test split.

  X-axis : sequence length  (5 levels, 25 → 200 by default)
  Y-axis : Spearman ρ  (y vs ŷ, random MAVE test split)
  Lines  : 3 per-position mutation rates — 10%, 25%, 50%

GT       : nonlinear additive + pairwise (Potts)
Surrogate: pairwise, nonlinear GE (Gaussian noise)

Fixed across conditions: library size (default 20,000).
Results averaged over --n_seqs random sequences per cell; ±1 SEM ribbon shown.

Usage:
    python scripts/length_vs_mut_rate.py \\
        --lib_size 20000 \\
        --n_seqs 5 \\
        --seed 0 \\
        --out_path outputs/length_vs_mut_rate.png

Quick smoke-test:
    python scripts/length_vs_mut_rate.py \\
        --lib_size 5000 --n_seqs 1 --lengths 25 100 --max_epochs 200 \\
        --out_path outputs/length_vs_mut_rate_test.png
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '/home/nagle/final_version/squid-nn')
sys.path.insert(0, '/home/nagle/final_version/squid-manuscript/squid')
sys.path.insert(0, '/home/nagle/final_version/squid-nn/squid')
sys.path.insert(0, '/home/nagle/final_version/residualbind')

# Enable GPU memory growth to prevent illegal-address crashes when TF models
# are trained sequentially in the same process.
import tensorflow as tf
for _gpu in tf.config.list_physical_devices('GPU'):
    try:
        tf.config.experimental.set_memory_growth(_gpu, True)
    except RuntimeError:
        pass

from ground_truth import (
    init_additive_noWT,
    init_pairwise_potts_optionA,
    init_sigmoid_nonlin,
    additive_affinity_noWT,
    pairwise_potts_energy,
    compute_gt_scores_for_library_potts,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUCS      = ['A', 'C', 'G', 'U']
_NUCS_ARR = np.array(list("ACGU"), dtype='U1')

DEFAULT_LENGTHS = [25, 62, 100, 150, 200]
MUT_RATES       = [0.10, 0.25, 0.50]
MUT_LABELS      = {0.10: "10%", 0.25: "25%", 0.50: "50%"}
MUT_COLORS      = {0.10: "#1f77b4", 0.25: "#ff7f0e", 0.50: "#2ca02c"}

# Matches the nonlinear additive+pairwise config from library_size_experiment.py
SURROGATE_CFG = dict(
    gpmap="pairwise", linearity="nonlinear",
    regression_type="GE", noise="Gaussian",
    noise_order=2, reg_strength=0.005, hidden_nodes=10,
)

# ---------------------------------------------------------------------------
# Sequence / library helpers
# ---------------------------------------------------------------------------

def _random_rna_onehot(L, rng):
    nuc_ids = rng.integers(0, 4, size=(L,), dtype=np.uint8)
    return np.eye(4, dtype=np.float32)[nuc_ids]


def _mutate_library(wt_oh, n_seqs, mut_rate, rng):
    """Bernoulli per-position mutations to a uniformly different nucleotide."""
    L, A    = wt_oh.shape
    wt_ids  = np.argmax(wt_oh, axis=1)
    mask    = rng.random(size=(n_seqs, L)) < mut_rate
    offsets = rng.integers(1, 4, size=(n_seqs, L), dtype=np.uint8)
    nuc_ids = np.where(
        mask, (wt_ids[None, :] + offsets) % A, wt_ids[None, :],
    ).astype(np.uint8)
    return np.eye(A, dtype=np.float32)[nuc_ids]   # (n_seqs, L, 4)


def _onehot_to_str_batch(X):
    return np.array(
        ["".join(_NUCS_ARR[np.argmax(x, axis=1)]) for x in X], dtype=object,
    )

# ---------------------------------------------------------------------------
# Surrogate training + evaluation
# ---------------------------------------------------------------------------

def _fit_and_eval(X, y, max_epochs):
    """Train surrogate and return Spearman ρ on random MAVE test split.

    Intended to run inside a spawned subprocess so the CUDA context is fully
    torn down between consecutive model runs (avoids BFC fragmentation crash).
    """
    import squid.surrogate_zoo

    N   = X.shape[0]
    bsz = max(32, min(N // 150, 2048))
    lr  = 5e-4 * min(1.0, (20_000 / N) ** 0.5)

    wrapper = squid.surrogate_zoo.SurrogateMAVENN(
        X.shape, num_tasks=1,
        gpmap=SURROGATE_CFG["gpmap"],
        regression_type=SURROGATE_CFG["regression_type"],
        linearity=SURROGATE_CFG["linearity"],
        noise=SURROGATE_CFG["noise"],
        noise_order=SURROGATE_CFG["noise_order"],
        reg_strength=SURROGATE_CFG["reg_strength"],
        hidden_nodes=SURROGATE_CFG["hidden_nodes"],
        alphabet=NUCS, deduplicate=True, gpu=True,
    )

    model, _, test_df = wrapper.train(
        X, y.reshape(-1, 1),
        learning_rate=lr, epochs=max_epochs, batch_size=bsz,
        early_stopping=True, patience=50, restore_best_weights=True,
        save_dir=None, verbose=0,
    )

    rho = np.nan
    try:
        cols  = list(test_df.columns)
        x_col = "x" if "x" in cols else next(c for c in cols if c in ("x", "X"))
        y_col = "y" if "y" in cols else next(c for c in cols if c.startswith("y"))
        X_str  = np.asarray(test_df[x_col])
        y_true = np.asarray(test_df[y_col], dtype=float).ravel()
        y_hat  = np.asarray(model.x_to_yhat(X_str), dtype=float).ravel()
        m      = np.isfinite(y_true) & np.isfinite(y_hat)
        if m.sum() >= 3:
            rho = float(spearmanr(y_true[m], y_hat[m])[0])
    except Exception as e:
        print(f"        [warn] rho computation failed: {e}", flush=True)

    return rho


def _safe_fit_and_eval(X, y, max_epochs):
    """Run surrogate training in a fresh subprocess via _lvm_worker.py.

    Using a separate script (not multiprocessing.spawn) lets us set
    TF_ENABLE_GPU_GARBAGE_COLLECTION=0 before tensorflow is ever imported,
    which prevents the BFC allocator use-after-free CUDA crash.
    """
    import subprocess
    import tempfile

    worker = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '_lvm_worker.py')

    # Prepend the conda env's lib dir so scipy's C extensions find the right
    # libstdc++ — without this the subprocess falls back to the system
    # libstdc++.so.6 which is missing GLIBCXX_3.4.29.
    conda_lib = os.path.join(os.path.dirname(sys.executable), '..', 'lib')
    conda_lib = os.path.normpath(conda_lib)
    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = conda_lib + ':' + env.get('LD_LIBRARY_PATH', '')

    with tempfile.TemporaryDirectory() as tmpdir:
        np.save(os.path.join(tmpdir, 'X.npy'), X)
        np.save(os.path.join(tmpdir, 'y.npy'), y)

        result = subprocess.run(
            [sys.executable, worker, tmpdir, str(max_epochs)],
            capture_output=True, text=True, env=env,
        )

    if result.returncode == 0:
        try:
            return float(result.stdout.strip().split('\n')[-1])
        except Exception as e:
            print(f"        [warn] couldn't parse rho from worker stdout: {e}",
                  flush=True)
    else:
        tail = result.stderr[-400:] if result.stderr else ""
        print(f"        [warn] worker exited {result.returncode}: ...{tail}",
              flush=True)
    return np.nan

# ---------------------------------------------------------------------------
# Single cell experiment
# ---------------------------------------------------------------------------

def run_cell(L, mut_rate, lib_size, max_epochs, master_rng):
    """One (length, mut_rate) run on a single fresh random sequence."""
    wt_oh = _random_rna_onehot(L, master_rng)

    rng_gt = np.random.default_rng(int(master_rng.integers(0, 2**31)))

    W_mut, mut_map, b0 = init_additive_noWT(
        rng_gt, wt_oh, sigma=0.5, l1_w=0.1, bias=0.0,
    )
    edges, J = init_pairwise_potts_optionA(
        rng_gt, wt_oh, p_edge=0.70, df=5.0, lambda_J=2.0,
        p_rescue=0.10, wt_rowcol_zero=True,
    )

    # Calibrate sigmoid nonlin on combined (additive + pairwise) scores so the
    # nonlinearity is scaled to the actual energy range at this mutation rate.
    rng_ref = np.random.default_rng(int(master_rng.integers(0, 2**31)))
    X_ref   = _mutate_library(wt_oh, min(lib_size, 20_000), mut_rate, rng_ref)
    s_add_ref  = additive_affinity_noWT(X_ref, W_mut, mut_map, b=b0).reshape(-1)
    s_pair_ref = pairwise_potts_energy(X_ref, edges, J, b=0.0).reshape(-1)
    nonlin_kwargs = init_sigmoid_nonlin(s_add_ref + s_pair_ref)
    del X_ref, s_add_ref, s_pair_ref

    # Training library
    rng_lib = np.random.default_rng(int(master_rng.integers(0, 2**31)))
    X = _mutate_library(wt_oh, lib_size, mut_rate, rng_lib)

    # GT scores — use nonlin_additive_pairwise key
    gt_scores = compute_gt_scores_for_library_potts(
        X,
        W_mut=W_mut, mut_map=mut_map, b0=b0,
        nonlin_name="sigmoid", nonlin_kwargs=nonlin_kwargs,
        edges=edges, J=J,
    )
    y = np.asarray(gt_scores["nonlin_additive_pairwise"], dtype=float).ravel()

    m = np.isfinite(y)
    X, y = X[m], y[m]
    if y.size < 50 or np.std(y) < 1e-6:
        print(f"        [skip] insufficient variance (n={y.size}, std={np.std(y):.2e})", flush=True)
        return np.nan
    print(f"        y: n={y.size}  range=[{y.min():.3f}, {y.max():.3f}]  std={np.std(y):.4f}", flush=True)

    rho = _safe_fit_and_eval(X, y, max_epochs)
    return rho

# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lib_size",   type=int, default=20_000)
    parser.add_argument("--n_seqs",     type=int, default=5,
                        help="Random sequences per (length, mut_rate) cell")
    parser.add_argument("--seed",       type=int, default=0)
    parser.add_argument("--max_epochs", type=int, default=1000,
                        help="Max surrogate training epochs (reduce for quick tests)")
    parser.add_argument("--out_path",   default="outputs/length_vs_mut_rate.png")
    parser.add_argument("--lengths",    type=int, nargs="+",
                        default=DEFAULT_LENGTHS)
    args = parser.parse_args()

    lengths  = sorted(args.lengths)
    n_seqs   = args.n_seqs
    lib_size = args.lib_size
    rng      = np.random.default_rng(args.seed)

    print(f"GT        : nonlinear additive + pairwise (Potts)")
    print(f"surrogate : pairwise nonlinear GE (Gaussian)")
    print(f"lengths   : {lengths}")
    print(f"mut_rates : {[f'{r:.0%}' for r in MUT_RATES]}")
    print(f"lib_size  : {lib_size:,}")
    print(f"n_seqs    : {n_seqs}")
    print(f"max_epochs: {args.max_epochs}")
    print(f"seed      : {args.seed}")
    total = len(lengths) * len(MUT_RATES) * n_seqs
    print(f"total runs: {total}\n")

    # results[mr][li] = list of rho values
    results = {mr: [[] for _ in lengths] for mr in MUT_RATES}

    done = 0
    for li, L in enumerate(lengths):
        for mr in MUT_RATES:
            for si in range(n_seqs):
                done += 1
                print(
                    f"[{done:3d}/{total}]  L={L:3d}  mut={mr:.0%}  "
                    f"seq {si+1}/{n_seqs}  ...",
                    flush=True,
                )
                rho = run_cell(L, mr, lib_size, args.max_epochs, rng)
                results[mr][li].append(rho)
                print(f"         rho = {rho:.4f}", flush=True)

    # Aggregate mean ± SEM
    means = {mr: np.full(len(lengths), np.nan) for mr in MUT_RATES}
    sems  = {mr: np.full(len(lengths), 0.0)    for mr in MUT_RATES}
    for mr in MUT_RATES:
        for li in range(len(lengths)):
            vals  = np.array(results[mr][li], dtype=float)
            valid = vals[np.isfinite(vals)]
            if valid.size > 0:
                means[mr][li] = np.mean(valid)
            if valid.size > 1:
                sems[mr][li]  = np.std(valid, ddof=1) / np.sqrt(valid.size)

    # Plot
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.array(lengths)
    for mr in MUT_RATES:
        mu  = means[mr]
        sem = sems[mr]
        ax.plot(x, mu, marker="o", linewidth=2, markersize=7,
                color=MUT_COLORS[mr], label=f"Mut rate {MUT_LABELS[mr]}")
        ax.fill_between(x, mu - sem, mu + sem,
                        color=MUT_COLORS[mr], alpha=0.18)

    ax.set_xlabel("Sequence length (nt)", fontsize=12)
    ax.set_ylabel("Spearman ρ  (random test split)", fontsize=12)
    ax.set_title(
        "Surrogate accuracy vs. sequence length\n"
        f"GT & surrogate: nonlinear additive+pairwise   "
        f"lib={lib_size:,}   N={n_seqs}/condition",
        fontsize=10,
    )
    ax.set_xticks(lengths)
    ax.set_xticklabels([str(L) for L in lengths])
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
    fig.savefig(args.out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[saved] plot → {args.out_path}")

    # Save CSV
    rows = []
    for mr in MUT_RATES:
        for li, L in enumerate(lengths):
            for si, rho in enumerate(results[mr][li]):
                rows.append({"length": L, "mut_rate": mr, "seq_idx": si, "rho": rho})
    df = pd.DataFrame(rows)
    csv_path = args.out_path.replace(".png", ".csv")
    df.to_csv(csv_path, index=False)
    print(f"[saved] data → {csv_path}")


if __name__ == "__main__":
    main()
