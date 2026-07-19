"""Subprocess worker for length_vs_mut_rate.py.

Must be run as a script so TF_ENABLE_GPU_GARBAGE_COLLECTION is set
before tensorflow is imported — multiprocessing spawn cannot do this
because module-level imports run before any user code.

Usage: python _lvm_worker.py <tmpdir> <max_epochs>
Prints Spearman rho as the last line of stdout.
"""
import os, sys
import numpy as np

# Decide GPU vs CPU before importing TF.
# L=200+ triggers a TF BFC allocator use-after-free crash on GPU; run on CPU
# for those lengths. Smaller lengths are fast enough that GPU is fine.
_tmpdir = sys.argv[1] if len(sys.argv) > 1 else ''
_L = 0
if _tmpdir:
    try:
        _X_hdr = np.load(os.path.join(_tmpdir, 'X.npy'), mmap_mode='r')
        _L = _X_hdr.shape[1]
    except Exception:
        pass

if _L >= 175:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''   # CPU-only for large L
else:
    os.environ['TF_ENABLE_GPU_GARBAGE_COLLECTION'] = '0'

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
sys.path.insert(0, '/home/nagle/final_version/squid-nn')
sys.path.insert(0, '/home/nagle/final_version/squid-manuscript/squid')
sys.path.insert(0, '/home/nagle/final_version/squid-nn/squid')
sys.path.insert(0, '/home/nagle/final_version/residualbind')

import tensorflow as tf
for _gpu in tf.config.list_physical_devices('GPU'):
    try:
        tf.config.experimental.set_memory_growth(_gpu, True)
    except RuntimeError:
        pass

import squid.surrogate_zoo
from scipy.stats import spearmanr

NUCS = ['A', 'C', 'G', 'U']
SURROGATE_CFG = dict(
    gpmap="pairwise", linearity="nonlinear",
    regression_type="GE", noise="Gaussian",
    noise_order=2, reg_strength=0.005, hidden_nodes=10,
)


def main():
    tmpdir     = sys.argv[1]
    max_epochs = int(sys.argv[2])

    X = np.load(os.path.join(tmpdir, 'X.npy'))
    y = np.load(os.path.join(tmpdir, 'y.npy'))

    N   = X.shape[0]
    L   = X.shape[1]
    # Keep per-step GPU memory (∝ batch × L²) below ~500 MB.
    # At L=200: 16 × 200² × 4² × 4 bytes ≈ 40 MB — safely within 16 GB.
    max_bsz = max(16, int(5e6 / (L * L)))
    bsz = max(16, min(N // 150, max_bsz))
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
        alphabet=NUCS, deduplicate=True, gpu=(_L < 175),
    )

    model, _, test_df = wrapper.train(
        X, y.reshape(-1, 1),
        learning_rate=lr, epochs=max_epochs, batch_size=bsz,
        early_stopping=True, patience=50, restore_best_weights=True,
        save_dir=None, verbose=0,
    )

    rho = float('nan')
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
        print(f"[warn] rho failed: {e}", file=sys.stderr)

    print(rho)


if __name__ == '__main__':
    main()
