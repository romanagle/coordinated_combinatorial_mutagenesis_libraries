import numpy as np
from typing import Optional, Dict, List, Any
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("--experiment", type=str, required=True,
                    help="Subdirectory inside finalgraphabrcms (used in epistasis_{experiment}_masked.npy)")
parser.add_argument("--subdir", type=str, help='Subdirectory inside outputs')
parser.add_argument("--activity", type=str, help='TESTING low vs high activity')
parser.add_argument("--numrounds", type=int, help='Number of repeated runs')
args = parser.parse_args()

experiment = args.experiment
subdir = args.subdir
activity = args.activity
numrounds = args.numrounds

listofrounds = [0, "post_ss", 1, 2,3, 4, 5, 6, 7, 8, 9]

ArrayLike = np.ndarray

def _to_numpy(x: ArrayLike) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)

def compare_maps_relative(
    ep_map: ArrayLike,
    other_map: ArrayLike,
    keep_mask: Optional[np.ndarray] = None,
    bins: int = 60,
    subsample: Optional[int] = None,
) -> Dict[str, float]:
    a = _to_numpy(ep_map).reshape(-1)
    b = _to_numpy(other_map).reshape(-1)

    if keep_mask is not None:
        m = keep_mask.reshape(-1).astype(bool)
        a = a[m]
        b = b[m]

    if a.size != b.size:
        raise ValueError(f"Mismatch after masking: {a.size} vs {b.size}")

    if subsample is not None and a.size > subsample:
        idx = np.random.permutation(a.size)[:subsample]
        a = a[idx]
        b = b[idx]

    eps = 1e-12
    a_abs, b_abs = np.abs(a), np.abs(b)

    a_rms = np.maximum(np.sqrt(np.mean(a_abs**2)), eps)
    b_rms = np.maximum(np.sqrt(np.mean(b_abs**2)), eps)
    a_rel = a_abs / a_rms
    b_rel = b_abs / b_rms

    cosine_rel_abs = np.dot(a_rel, b_rel) / ((np.linalg.norm(a_rel) + eps) * (np.linalg.norm(b_rel) + eps))

    def rankdata(x: np.ndarray) -> np.ndarray:
        n = x.size
        idx = np.argsort(x)
        ranks = np.empty_like(idx, dtype=np.float32)
        xs = x[idx]
        simple_ranks = np.arange(1, n + 1, dtype=np.float32)

        i = 0
        while i < n:
            j = i + 1
            while j < n and xs[j] == xs[i]:
                j += 1
            if j - i > 1:
                avg_rank = (i + 1 + j) / 2.0
                simple_ranks[i:j] = avg_rank
            i = j

        ranks[idx] = simple_ranks
        return ranks

    if a_abs.size >= 2:
        ra = rankdata(a_abs)
        rb = rankdata(b_abs)
        ra -= np.mean(ra)
        rb -= np.mean(rb)
        spearman_abs = np.dot(ra, rb) / ((np.linalg.norm(ra) * np.linalg.norm(rb)) + eps)
    else:
        spearman_abs = 0.0

    la = np.log(a_rel + eps)
    lb = np.log(b_rel + eps)
    mn = float(min(np.min(la), np.min(lb)))
    mx = float(max(np.max(la), np.max(lb)))

    if mn == mx:
        loghist_L1 = 0.0
    else:
        ha, _ = np.histogram(la, bins=bins, range=(mn, mx))
        hb, _ = np.histogram(lb, bins=bins, range=(mn, mx))
        ha = ha / (np.sum(ha) + eps)
        hb = hb / (np.sum(hb) + eps)
        loghist_L1 = np.sum(np.abs(ha - hb))

    sign_agreement = np.mean((np.sign(a) == np.sign(b)).astype(float))

    return {
        "n_compared": float(a.size),
        "cosine_rel_abs": float(cosine_rel_abs),
        "spearman_abs": float(spearman_abs),
        "loghist_L1": float(loghist_L1),
        "sign_agreement": float(sign_agreement),
        "scale_ratio_rms_abs": float(b_rms / a_rms),
    }

# ---- Load tensors ----
groundtruthtensor = np.load(f"/home/nagle/final_version/outputs/{subdir}/{activity}/round_tensors_{experiment}/epistasis_{experiment}_masked.npy")

# ---- Compute scores per round ----
scores_by_round: List[Dict[str, float]] = []
round_labels: List[str] = []


for r in listofrounds:
    # collect per-run scores for this round
    per_run_scores = defaultdict(list)

    for run_idx in range(numrounds):
        tensor = np.load(f"/home/nagle/final_version/outputs/{subdir}/{activity}/round_tensors_{experiment}/epmap_{r}_{run_idx}_{experiment}_masked.npy")
        scores = compare_maps_relative(tensor, groundtruthtensor)

        for k, v in scores.items():
            per_run_scores[k].append(v)

    # average across runs
    avg_scores = {
        k: float(np.mean(v))
        for k, v in per_run_scores.items()
    }

    scores_by_round.append(avg_scores)
    round_labels.append(str(r))

    print(f"Round {r} (avg over {numrounds} runs): {avg_scores}")


# ---- Prepare x-axis ----
x = np.arange(len(round_labels))

# metrics to plot (drop n_compared unless you want it too)
metric_names = [k for k in scores_by_round[0].keys() if k != "n_compared"]

# ---- One line graph per metric (AVG across numround runs) ----
for m in metric_names:
    y_mean = np.array([d[m] for d in scores_by_round], dtype=float)

    plt.figure()
    plt.plot(x, y_mean, marker="o")
    plt.xticks(x, round_labels, rotation=0)
    plt.xlabel("Round")
    plt.ylabel(f"{m} (mean over {numrounds} runs)")
    plt.title(f"{m} (mean over {numrounds} runs)")
    plt.tight_layout()
    plt.savefig(f"/home/nagle/final_version/outputs/{subdir}/{activity}/{m}_{experiment}_mean.png", dpi=200)
    plt.close()

# Which direction is better for each metric
metric_direction = {
    "cosine_rel_abs": "max",
    "spearman_abs": "max",
    "sign_agreement": "max",
    "loghist_L1": "min",
    "scale_ratio_rms_abs": "min",
}

# ---- Direction-aware normalized plot (AVG across numrounds runs) ----
plt.figure()

for m in metric_names:
    y = np.array([d[m] for d in scores_by_round], dtype=float)

    if np.allclose(y.max(), y.min()):
        y_norm = np.ones_like(y)  # all rounds equally good
    else:
        if metric_direction[m] == "max":
            y_norm = (y - y.min()) / (y.max() - y.min())
        elif metric_direction[m] == "min":
            y_norm = (y.max() - y) / (y.max() - y.min())
        else:
            raise ValueError(f"Unknown direction for metric {m}")

    plt.plot(x, y_norm, marker="o", label=m)

plt.xticks(x, round_labels)
plt.xlabel("Round")
plt.ylabel(f"Normalized score (1 = best; mean over {numrounds} runs)")
plt.title(f"Direction-aware normalized scores (mean over {numrounds} runs)")
plt.legend()
plt.tight_layout()
plt.savefig(f'/home/nagle/final_version/outputs/{subdir}/{activity}/all_metrics_normalized_directional_mean_{experiment}.png', dpi=200)
plt.close()