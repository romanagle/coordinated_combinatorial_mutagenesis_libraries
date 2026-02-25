"""
plotting.py – Visualization functions.

All functions accept explicit file paths; none read global variables.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import seaborn as sns
from scipy.stats import spearmanr, gaussian_kde

from stats_utils import (
    _finite_1d, _flatten_runs, _flatten_list_of_arrays, _finite_flat,
    _as_list_of_arrays, _fd_bins, _transform, _logabs,
    _kde_line, _kde_curve, extract_active_potts_J,
)
from ground_truth import (
    additive_affinity_noWT, pairwise_potts_energy, apply_global_nonlin,
    init_additive_noWT, init_pairwise_adjacent_noWT, init_pairwise_potts_optionA,
    init_tanh_nonlin,
    init_mix_tanh_nonlin,
)
from seq_utils import rna_to_one_hot

NUCS = ['A', 'C', 'G', 'U']


# ---------------------------------------------------------------------------
# Surrogate model diagnostics
# ---------------------------------------------------------------------------

def plot_y_vs_yhat(model, mave_df, save_dir=None):
    """Scatter of MAVE-NN y vs ŷ on the training set.

    Returns: (fig, preds, ground_truth, rho)
    """
    X_test  = np.asarray(mave_df["x"])
    y_test  = np.asarray(mave_df["y"], dtype=float)
    yhat_test = model.x_to_yhat(X_test)
    rho, _  = spearmanr(yhat_test.ravel(), y_test)

    fig, ax = plt.subplots(1, 1, figsize=[5, 5])
    ax.scatter(yhat_test, y_test, color='C0', s=1, alpha=.1, label='test data')
    xlim = [min(yhat_test), max(yhat_test)]
    ax.plot(xlim, xlim, '--', color='k', label='diagonal', zorder=100)
    ax.set_xlabel('model prediction ($\\hat{y}$)')
    ax.set_ylabel('measurement ($y$)')
    ax.set_title(f"Standard metric of model performance:\nSpearman correlation = {rho:.3f}")
    ax.legend(loc='upper left')
    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, 'mavenn_measure_yhat.png'), facecolor='w', dpi=200)
        plt.close()
    return fig, yhat_test, y_test, rho


# ---------------------------------------------------------------------------
# Distribution / histogram plots
# ---------------------------------------------------------------------------

def plot_random_vs_eval_per_gt(
    scores_random: dict,
    scores_eval: dict,
    *,
    nonlin_label: str,
    save_dir: str,
    bins: int = 120,
    density: bool = True,
    alpha_random: float = 0.45,
    alpha_eval: float = 0.45,
):
    """4 separate histograms, one per GT key, overlaying random vs eval."""
    os.makedirs(save_dir, exist_ok=True)
    gt_keys = ["additive", "additive_pairwise", "nonlin_additive", "nonlin_additive_pairwise"]

    for k in gt_keys:
        r = np.asarray(scores_random[k], dtype=float).reshape(-1)
        e = np.asarray(scores_eval[k],   dtype=float).reshape(-1)
        r, e = r[np.isfinite(r)], e[np.isfinite(e)]

        fig, ax = plt.subplots(figsize=(8.5, 5))

        if r.size == 0 and e.size == 0:
            ax.text(0.5, 0.5, f"{k}: no finite values", ha="center", va="center",
                    transform=ax.transAxes)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"{nonlin_label}_{k}_random_vs_eval.png"), dpi=300)
            plt.close(fig)
            continue

        all_vals = np.concatenate([r, e]) if (r.size and e.size) else (r if r.size else e)
        lo, hi   = (np.percentile(all_vals, [0.5, 99.5]) if all_vals.size > 5
                    else (np.min(all_vals), np.max(all_vals)))
        if lo == hi:
            lo -= 1e-6; hi += 1e-6
        bin_edges = np.linspace(lo, hi, bins + 1)

        if r.size:
            ax.hist(r, bins=bin_edges, density=density, alpha=alpha_random,
                    label=f"Random library (N={len(r)})")
        if e.size:
            ax.hist(e, bins=bin_edges, density=density, alpha=alpha_eval,
                    label=f"Eval library (N={len(e)})")

        ax.set_title(f"{k} — {nonlin_label}: random vs eval distributions")
        ax.set_xlabel("Ground-truth score (WT-referenced)")
        ax.set_ylabel("Density" if density else "Count")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{nonlin_label}_{k}_random_vs_eval.png"), dpi=300)
        plt.close(fig)


def plot_overlay_hist_kde(series_dict, *, title, out_png,
                          transform="logabs",
                          bins="fd",
                          density=True,
                          kde=True,
                          kde_bw_scale=1.4,
                          rug=True):
    os.makedirs(os.path.dirname(out_png), exist_ok=True)

    data   = {lab: _transform(_finite_1d(x), transform) for lab, x in series_dict.items()}
    pooled = (np.concatenate([v for v in data.values() if v.size], axis=0)
              if any(v.size for v in data.values()) else np.array([]))
    if pooled.size == 0:
        print(f"[warn] no data: {title}")
        return

    nb = _fd_bins(pooled) if bins == "fd" else int(bins)

    lo, hi = np.percentile(pooled, [0.5, 99.5])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo, hi = pooled.min(), pooled.max()
        if lo == hi:
            lo -= 1e-6; hi += 1e-6
    edges = np.linspace(lo, hi, nb + 1)
    grid  = np.linspace(lo, hi, 400)

    plt.figure(figsize=(9, 5))
    for lab, x in data.items():
        if x.size == 0:
            continue
        plt.hist(x, bins=edges, density=density, histtype="step", linewidth=2.0,
                 alpha=0.95, label=f"{lab} (N={x.size})")
        if kde and x.size >= 5 and np.nanstd(x) > 0:
            kde_obj = gaussian_kde(x)
            kde_obj.set_bandwidth(bw_method=kde_obj.factor * float(kde_bw_scale))
            plt.plot(grid, kde_obj(grid), linewidth=2.0)
        if rug:
            ymin, ymax = plt.gca().get_ylim()
            rug_y = ymin + 0.02 * (ymax - ymin)
            plt.plot(x, np.full_like(x, rug_y), "|", markersize=10, alpha=0.25)

    xlabel = {"logabs": "log10(|coef| + 1e-8)", "robust_z": "robust z-score (median/MAD)"}.get(
        transform, "coefficient"
    )
    plt.title(title); plt.xlabel(xlabel)
    plt.ylabel("Density" if density else "Count")
    plt.grid(True, linestyle="--", alpha=0.35); plt.legend(); plt.tight_layout()
    plt.savefig(out_png, dpi=300); plt.close()


def plot_coef_dists_per_gt_overlay_across_cfgs(
    coef_store,
    outdir,
    experiment,
    gt_keys,
    cfg_names,
    which="additive",
    transform="logabs",
):
    """Overlay all configs on one plot per GT key."""
    os.makedirs(outdir, exist_ok=True)
    for gt_key in gt_keys:
        series = {}
        for cfg in cfg_names:
            try:
                arrs = coef_store[cfg][which][gt_key]
            except KeyError:
                arrs = []
            series[cfg] = _flatten_list_of_arrays(arrs)

        plot_overlay_hist_kde(
            series,
            title=f"{experiment} — {gt_key} — {which} coef dist (overlay)",
            out_png=os.path.join(outdir, f"{experiment}_{gt_key}_{which}_overlay_histkde.png"),
            transform=transform, bins="fd", density=True, kde=True, kde_bw_scale=1.4, rug=True,
        )


def plot_random_library_distributions_three_nonlins(
    x_mut: np.ndarray,
    input_seq: str,
    *,
    rng=None,
    sigma_w: float = 0.5,
    sigma_P: float = 0.2,
    l1_w: float = 0.1,
    l1_P: float = 0.05,
    bias: float = 0.0,
    bins: int = 120,
    density: bool = True,
    save_dir: str = None,
    return_scores: bool = True,
):
    if rng is None:
        rng = np.random.default_rng(0)

    wt_oh = rna_to_one_hot(input_seq)
    W_mut, mut_map, b0 = init_additive_noWT(rng, wt_oh, sigma=sigma_w, l1_w=l1_w, bias=bias)
    P_mut = init_pairwise_adjacent_noWT(rng, wt_oh, mut_map, sigma_P=sigma_P, l1_P=l1_P)
    edges, J = init_pairwise_potts_optionA(
        rng, wt_oh, p_edge=0.30, df=5.0, lambda_J=0.5, p_rescue=0.10, wt_rowcol_zero=True,
    )

    def make_bins(arrs):
        all_y = np.concatenate([np.asarray(a).ravel() for a in arrs])
        lo, hi = np.percentile(all_y, [0.5, 99.5])
        return np.linspace(lo, hi, bins + 1)

    all_scores = {}
    wt_scores  = {}

    def plot_one(nonlin_name, nonlin_kwargs):
        s_add     = additive_affinity_noWT(x_mut, W_mut, mut_map, b0)
        s_pair    = pairwise_potts_energy(x_mut, edges, J, b=0.0)
        s_addpair = s_add + s_pair

        y_add            = s_add.reshape(-1)
        y_addpair        = s_addpair.reshape(-1)
        # Normalise each energy stream to unit std so the nonlinearity
        # parameters are scale-invariant (prevents pairwise saturation).
        s_add_n     = s_add     / (np.std(s_add)     + 1e-8)
        s_addpair_n = s_addpair / (np.std(s_addpair) + 1e-8)
        y_nonlin_add     = apply_global_nonlin(s_add_n,     nonlin_name, nonlin_kwargs).reshape(-1)
        y_nonlin_addpair = apply_global_nonlin(s_addpair_n, nonlin_name, nonlin_kwargs).reshape(-1)

        wt0              = np.array([0.0])
        wt_nonlin_add    = float(apply_global_nonlin(wt0, nonlin_name, nonlin_kwargs).reshape(-1)[0])

        all_scores[nonlin_name] = {
            "additive": y_add, "additive_pairwise": y_addpair,
            "nonlin_additive": y_nonlin_add, "nonlin_additive_pairwise": y_nonlin_addpair,
        }
        wt_scores[nonlin_name] = {
            "additive": 0.0, "additive_pairwise": 0.0,
            "nonlin_additive": wt_nonlin_add, "nonlin_additive_pairwise": wt_nonlin_add,
        }

        bin_edges = make_bins([y_add, y_addpair, y_nonlin_add, y_nonlin_addpair])
        fig, ax   = plt.subplots(figsize=(9, 5))
        ax.hist(y_add,            bins=bin_edges, density=density, alpha=0.3, label="Additive (noWT)")
        ax.hist(y_addpair,        bins=bin_edges, density=density, alpha=0.3, label="Additive+Pairwise (noWT)")
        ax.hist(y_nonlin_add,     bins=bin_edges, density=density, alpha=0.3, label=f"{nonlin_name} ∘ Additive")
        ax.hist(y_nonlin_addpair, bins=bin_edges, density=density, alpha=0.3,
                label=f"{nonlin_name} ∘ (Additive+Pairwise)")

        for v in wt_scores[nonlin_name].values():
            ax.axvline(v, linestyle="--")

        ax.set_title("Random distribution per ground truth function")
        ax.set_xlabel("Ground-truth score (WT-referenced)")
        ax.set_ylabel("Density" if density else "Count")
        ax.grid(True, linestyle="--", alpha=0.4); ax.legend(); plt.tight_layout()

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(f"{save_dir}/random_dist_{nonlin_name}.png", dpi=300)
        plt.close(fig)

    mix_tanh_kwargs = init_mix_tanh_nonlin(rng, n_components=50)
    tanh_kwargs     = init_tanh_nonlin(rng)
    plot_one("mix_tanh", mix_tanh_kwargs)
    plot_one("tanh", tanh_kwargs)
    plot_one("softmax_gate", dict(
        gate_logits=np.array([1.0, -1.0]), gate_temperature=1.0, tanh_alpha=0.5, tanh_beta=0.0,
    ))

    if return_scores:
        return {
            "params": {"W_mut": W_mut, "edges": edges, "J": J, "b": b0, "mut_map": mut_map},
            "scores": all_scores,
            "wt_scores": wt_scores,
            "nonlin_kwargs": {"mix_tanh": mix_tanh_kwargs, "tanh": tanh_kwargs},
        }


# ---------------------------------------------------------------------------
# Spearman summary plots
# ---------------------------------------------------------------------------

def spearmanavg_by_cfg(rho_storage, output_dir, experiment, step="0", gt_keys=None):
    """Print average Spearman values for all configs at a given step."""
    print("\n==============================")
    print("AVERAGE SPEARMAN COEFF VALUES")
    print("==============================")

    step_dict = rho_storage.get(step, {})

    # old-style
    if "curated" in step_dict and "random" in step_dict and isinstance(step_dict["curated"], dict):
        if gt_keys is None:
            gt_keys = list(step_dict["curated"].keys())
        print("\n--- (old-style) ---")
        for k in gt_keys:
            cur = np.asarray(step_dict["curated"].get(k, []), dtype=float)
            rnd = np.asarray(step_dict["random"].get(k,   []), dtype=float)
            print(f"{k}: curated={np.nanmean(cur) if cur.size else np.nan:.4f}   "
                  f"random={np.nanmean(rnd) if rnd.size else np.nan:.4f}")

    # new-style
    cfg_names = [
        name for name, val in step_dict.items()
        if name not in ("curated", "random") and isinstance(val, dict)
        and "curated" in val and "random" in val
    ]
    if not cfg_names:
        print("\n[warn] No cfg-style entries found under rho_storage[step].")
        return

    if gt_keys is None:
        gt_keys = list(step_dict[cfg_names[0]]["curated"].keys())

    for cfg_name in cfg_names:
        cur_dict = step_dict[cfg_name].get("curated", {})
        rnd_dict = step_dict[cfg_name].get("random",  {})
        print(f"\n--- {cfg_name} ---")
        for k in gt_keys:
            cur = np.asarray(cur_dict.get(k, []), dtype=float)
            rnd = np.asarray(rnd_dict.get(k, []), dtype=float)
            print(f"{k}: curated={np.nanmean(cur) if cur.size else np.nan:.4f}   "
                  f"random={np.nanmean(rnd) if rnd.size else np.nan:.4f}")


def spearman_bars_per_surrogate(
    rho_storage,
    *,
    output_dir,
    experiment,
    step="0",
    gt_keys=None,
    cfg_names=None,
    show_sem=True,
):
    """Save one bar-plot per surrogate config with 8 bars (curated/random × 4 GT keys)."""
    os.makedirs(output_dir, exist_ok=True)

    step_dict = rho_storage.get(step, {})
    if not step_dict:
        raise KeyError(f"No rho_storage found for step={step}. Keys: {list(rho_storage.keys())}")

    if cfg_names is None:
        cfg_names = [
            c for c in step_dict.keys()
            if isinstance(step_dict.get(c), dict)
            and "curated" in step_dict[c] and "random" in step_dict[c]
        ]
    if not cfg_names:
        raise ValueError(f"No cfg entries found under rho_storage[{step}]")
    if gt_keys is None:
        gt_keys = list(step_dict[cfg_names[0]]["curated"].keys())

    for cfg in cfg_names:
        cur = step_dict[cfg]["curated"]
        rnd = step_dict[cfg]["random"]

        means, sems, labels = [], [], []
        for k in gt_keys:
            a = np.asarray(cur.get(k, []), dtype=float)
            b = np.asarray(rnd.get(k, []), dtype=float)
            a_mean = np.nanmean(a) if a.size else np.nan
            b_mean = np.nanmean(b) if b.size else np.nan
            if show_sem:
                a_sem = (np.nanstd(a, ddof=1) / np.sqrt(np.sum(np.isfinite(a)))
                         if np.sum(np.isfinite(a)) > 1 else 0.0)
                b_sem = (np.nanstd(b, ddof=1) / np.sqrt(np.sum(np.isfinite(b)))
                         if np.sum(np.isfinite(b)) > 1 else 0.0)
            else:
                a_sem = b_sem = 0.0
            means.extend([a_mean, b_mean])
            sems.extend([a_sem, b_sem])
            labels.extend([f"{k}\ncurated", f"{k}\nrandom"])

        x = np.arange(len(labels))
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(x, means, yerr=sems, capsize=4)
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_ylabel("Spearman correlation")
        ax.set_title(f"{experiment} — Spearman (step {step}) — {cfg}\n"
                     f"(curated eval vs random test_df)")
        ax.axhline(0, linewidth=1)
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)
        plt.tight_layout()
        outpath = os.path.join(output_dir, f"{experiment}_spearman8bars_step{step}_{cfg}.png")
        plt.savefig(outpath, dpi=250); plt.close(fig)
        print(f"[saved] {outpath}")


# ---------------------------------------------------------------------------
# Epistasis / ISM plots
# ---------------------------------------------------------------------------

def plot_epistasis_maxpooled_like(e_tensor):
    """Max-pool epistasis tensor and display as heatmap."""
    max_pooled_arr = e_tensor.copy()
    mask = np.tril(np.ones_like(max_pooled_arr, dtype=bool), -1)

    flipped_arr  = max_pooled_arr.T
    flipped_mask = mask.T
    min_val, max_val = np.nanmin(flipped_arr), np.nanmax(flipped_arr)

    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(flipped_arr, mask=flipped_mask, cmap='seismic',
                     center=0, square=True, vmin=min_val, vmax=max_val)
    ax.xaxis.tick_bottom(); ax.yaxis.tick_left()

    threshold = 0.6 * max_val
    for i in range(max_pooled_arr.shape[0]):
        for j in range(max_pooled_arr.shape[1]):
            if not flipped_mask[i, j]:
                continue
            if np.abs(max_pooled_arr[i, j]) > threshold:
                rect = Rectangle((i, j), 1, 1, fill=False, edgecolor='black', linewidth=2)
                if np.abs(i - j) > 7:
                    rect = Rectangle((i, j), 1, 1, fill=False, edgecolor='yellow', linewidth=4)
                ax.add_patch(rect)

    plt.tight_layout()
    plt.savefig("epmaptest.png")


def plot_epistasis_map(E, outputdir):
    plt.figure(figsize=(6.5, 5.8))
    im = plt.imshow(E, origin="lower", interpolation="none", aspect="auto")
    plt.xlabel("j (position)"); plt.ylabel("i (position)")
    plt.title("Pairwise epistasis (max |ε| over bases)")
    cbar = plt.colorbar(im); cbar.set_label("epistasis")
    plt.tight_layout()
    plt.savefig(f"{outputdir}/epistasis_map.png", dpi=300)
    plt.show()


def plot_double_slice_one_nuc(delta_pairs, a, b, title=None, out_png=None):
    """Plot (L×L) Δ-score heatmap for the chosen nucleotide pair i→a, j→b."""
    mat = delta_pairs[a, b, :, :]
    plt.figure(figsize=(6.2, 5.6))
    im = plt.imshow(mat, origin="lower", interpolation="none")
    plt.xlabel(f"j (mutate to {NUCS[b]})")
    plt.ylabel(f"i (mutate to {NUCS[a]})")
    plt.title(title or f"Double ISM Δ score (i→{NUCS[a]}, j→{NUCS[b]})")
    cbar = plt.colorbar(im); cbar.set_label("Δ score")
    plt.tight_layout()
    if out_png:
        plt.savefig(out_png, dpi=300)
    plt.show()


def plot_raw_pairs_as_LxL_cells_with_4x4_inside(
    raw_pairs,
    wt_idx,
    outputdir,
    nucs=("A", "C", "G", "U"),
    gap=1,
    step_pos=5,
    show_wt_pixel=True,
    figsize=(12, 12),
    scale="minmax",
    clip_percentiles=(2, 98),
    vmin_fixed=None,
    vmax_fixed=None,
    box_outline_color="black",
    box_outline_lw=1.2,
    inner_grid_alpha=0.20,
    annotate_mini=False,
    mini_fontsize=4,
    mini_alpha=0.9,
    mini_which="both",
    cbar_label="Raw score (double mutant)",
):
    assert raw_pairs.ndim == 4 and raw_pairs.shape[:2] == (4, 4), \
        f"Expected (4,4,L,L), got {raw_pairs.shape}"
    _, _, L0, L1 = raw_pairs.shape
    assert L0 == L1, f"Expected square LxL, got {L0}x{L1}"
    L = L0
    assert len(wt_idx) == L

    cell   = 4
    stride = cell + gap
    H = W  = L * cell + (L - 1) * gap

    big = np.full((H, W), np.nan, dtype=float)
    for i in range(L):
        r0 = i * stride
        for j in range(L):
            c0 = j * stride
            big[r0:r0 + cell, c0:c0 + cell] = raw_pairs[:, :, i, j]

    finite = big[np.isfinite(big)]
    if finite.size == 0:
        vmin, vmax = 0.0, 1.0
    else:
        if scale == "percentile":
            lo, hi = clip_percentiles
            vmin = np.percentile(finite, lo); vmax = np.percentile(finite, hi)
        elif scale == "minmax":
            vmin = float(np.min(finite)); vmax = float(np.max(finite))
        elif scale == "fixed":
            if vmin_fixed is None or vmax_fixed is None:
                raise ValueError("scale='fixed' requires vmin_fixed and vmax_fixed")
            vmin, vmax = float(vmin_fixed), float(vmax_fixed)
        else:
            raise ValueError("scale must be 'percentile', 'minmax', or 'fixed'")
        if vmin == vmax:
            vmin, vmax = vmin - 1e-6, vmax + 1e-6

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(big, origin="lower", interpolation="none", vmin=vmin, vmax=vmax, aspect="equal")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label(cbar_label)

    centers  = np.arange(L) * stride + (cell - 1) / 2.0
    tick_pos = np.arange(0, L, step_pos) if (step_pos and step_pos > 0) else np.arange(L)
    ax.set_xticks(centers[tick_pos]); ax.set_xticklabels([str(k) for k in tick_pos])
    ax.set_yticks(centers[tick_pos]); ax.set_yticklabels([str(k) for k in tick_pos])
    ax.set_xlabel("Position j"); ax.set_ylabel("Position i")
    ax.set_title(f"Raw double-mutant scores: {L}×{L} position cells, each cell is 4×4 nucleotides")

    for i in range(L):
        base_y = i * stride
        for a in range(1, cell):
            ax.axhline(base_y + a - 0.5, linewidth=0.3, alpha=inner_grid_alpha)
    for j in range(L):
        base_x = j * stride
        for b in range(1, cell):
            ax.axvline(base_x + b - 0.5, linewidth=0.3, alpha=inner_grid_alpha)

    for i in range(L):
        y0 = i * stride - 0.5
        for j in range(L):
            x0 = j * stride - 0.5
            ax.add_patch(Rectangle((x0, y0), cell, cell,
                                   linewidth=box_outline_lw, edgecolor=box_outline_color,
                                   facecolor="none"))

    if annotate_mini:
        for i in range(L):
            base_y = i * stride
            for j in range(L):
                base_x = j * stride
                for a in range(4):
                    for b in range(4):
                        txt = (nucs[a] if mini_which == "row" else
                               nucs[b] if mini_which == "col" else f"{nucs[a]}{nucs[b]}")
                        ax.text(base_x + b + 0.45, base_y + a - 0.45, txt,
                                fontsize=mini_fontsize, ha="right", va="bottom", alpha=mini_alpha)

    if show_wt_pixel:
        for i in range(L):
            a_wt = int(wt_idx[i])
            for j in range(L):
                b_wt = int(wt_idx[j])
                r = i * stride + a_wt; c = j * stride + b_wt
                ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1.0, 1.0,
                                       linewidth=0.45, edgecolor="black",
                                       facecolor="none", alpha=0.9))

    fig.tight_layout()
    fig.savefig(f"{outputdir}/raw_pairs_nested.png", dpi=300, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# GT weight distribution plots
# ---------------------------------------------------------------------------

def plot_gt_weight_dists_per_function(
    *, W_mut, edges, J, gt_keys, outdir, bw_scale=1.25, clip=(0.5, 99.5),
    drop_pairwise_zeros=True,
):
    """KDE plots of GT weights (additive + pairwise) per ground-truth function."""
    os.makedirs(outdir, exist_ok=True)
    w_add  = _finite_1d(W_mut)
    w_pair = extract_active_potts_J(J, edges, drop_zeros=drop_pairwise_zeros)
    needs_pair = {"additive_pairwise", "nonlin_additive_pairwise"}

    for gt in gt_keys:
        for log_version in [False, True]:
            a = _logabs(w_add)  if log_version else w_add
            p = _logabs(w_pair) if log_version else w_pair
            include_pair = (gt in needs_pair)

            pooled = a.copy()
            if include_pair and p.size:
                pooled = np.concatenate([pooled, p])
            if pooled.size < 5:
                print(f"[skip] {gt} (log={log_version}) — not enough points")
                continue

            lo, hi = np.percentile(pooled, clip)
            if lo == hi:
                lo -= 1e-6; hi += 1e-6
            grid = np.linspace(lo, hi, 600)

            plt.figure(figsize=(8.5, 5))
            if a.size >= 5 and np.nanstd(a) > 0:
                plt.plot(grid, _kde_line(a, grid, bw_scale=bw_scale),
                         linewidth=2.3, label=f"GT additive weights (N={a.size})")
            if include_pair and p.size >= 5 and np.nanstd(p) > 0:
                plt.plot(grid, _kde_line(p, grid, bw_scale=bw_scale),
                         linewidth=2.3, linestyle="--",
                         label=f"GT pairwise weights (N={p.size})")

            xlabel = "log10(|weight| + 1e-8)" if log_version else "weight"
            plt.title(f"Ground-truth weight distributions — {gt}" +
                      (" (logabs)" if log_version else ""))
            plt.xlabel(xlabel); plt.ylabel("density")
            plt.grid(True, linestyle="--", alpha=0.35); plt.legend(); plt.tight_layout()

            suffix  = "logabs" if log_version else "raw"
            outpng  = os.path.join(outdir, f"GT_weight_dist_{gt}_{suffix}.png")
            plt.savefig(outpng, dpi=300); plt.close()
            print(f"[saved] {outpng}")


# ---------------------------------------------------------------------------
# Surrogate coefficient plots
# ---------------------------------------------------------------------------

def plot_add_vs_pair_kde_for_cfg(
    coef_store, cfg_name, gt_key, *, experiment, outdir, bw_scale=1.3, clip=(0.5, 99.5),
):
    """Two KDE plots (raw + logabs) comparing additive vs pairwise coefficients."""
    os.makedirs(outdir, exist_ok=True)

    add_list  = coef_store.get(cfg_name, {}).get("additive", {}).get(gt_key, [])
    pair_list = coef_store.get(cfg_name, {}).get("pairwise", {}).get(gt_key, [])
    x_add     = _flatten_runs(add_list)
    x_pair    = _flatten_runs(pair_list)

    if x_add.size < 5 or x_pair.size < 5:
        print(f"[skip] {cfg_name} {gt_key}: need >=5 points each "
              f"(add={x_add.size}, pair={x_pair.size})")
        return

    for log_version, suffix in [(False, "KDE"), (True, "KDE_logabs")]:
        xa = _logabs(x_add)  if log_version else x_add
        xp = _logabs(x_pair) if log_version else x_pair

        pooled = np.concatenate([xa, xp])
        lo, hi = np.percentile(pooled, clip)
        if lo == hi:
            lo -= 1e-6; hi += 1e-6
        grid = np.linspace(lo, hi, 600)

        out_png = os.path.join(outdir, f"{experiment}_{cfg_name}_{gt_key}_add_vs_pair_{suffix}.png")
        plt.figure(figsize=(8.5, 5))
        plt.plot(grid, _kde_curve(xa, grid, bw_scale=bw_scale), linewidth=2.2,
                 label=f"additive (N={x_add.size})")
        plt.plot(grid, _kde_curve(xp, grid, bw_scale=bw_scale), linewidth=2.2,
                 label=f"pairwise (N={x_pair.size})")
        plt.title(f"{experiment} — {cfg_name} — {gt_key} — coef " +
                  ("KDE (add vs pair)" if not log_version else "KDE (log10|coef|)"))
        plt.xlabel("coefficient value" if not log_version else "log10(|coef| + 1e-8)")
        plt.ylabel("density")
        plt.grid(True, linestyle="--", alpha=0.35); plt.legend(); plt.tight_layout()
        plt.savefig(out_png, dpi=300); plt.close()
        print(f"[saved] {out_png}")


def plot_kde_grid_surrogate_vs_gt(
    coef_store,
    SURROGATE_CONFIGS,
    *,
    experiment,
    gt_keys,
    cfg_names,
    out_png,
    log_version=False,
    bw_scale=1.3,
):
    """4×4 grid: rows = surrogate configs, cols = GT keys."""
    n_rows, n_cols = len(cfg_names), len(gt_keys)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.8 * n_cols, 3.2 * n_rows),
                             sharex=False, sharey=False)
    if n_rows == 1: axes = np.expand_dims(axes, axis=0)
    if n_cols == 1: axes = np.expand_dims(axes, axis=1)

    for r, cfg_name in enumerate(cfg_names):
        gpmap = SURROGATE_CONFIGS[cfg_name]["gpmap"]
        for c, gt_key in enumerate(gt_keys):
            ax = axes[r, c]

            add_list  = coef_store.get(cfg_name, {}).get("additive", {}).get(gt_key, [])
            pair_list = coef_store.get(cfg_name, {}).get("pairwise", {}).get(gt_key, [])
            x_add  = _finite_flat(add_list)
            x_pair = _finite_flat(pair_list) if gpmap == "pairwise" else np.array([])

            if log_version:
                x_add  = _logabs(x_add)
                if gpmap == "pairwise": x_pair = _logabs(x_pair)

            pooled = x_add
            if gpmap == "pairwise" and x_pair.size:
                pooled = np.concatenate([x_add, x_pair]) if x_add.size else x_pair

            if pooled.size < 5:
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
                continue

            lo, hi = np.percentile(pooled, [0.5, 99.5])
            if lo == hi: lo -= 1e-6; hi += 1e-6
            grid = np.linspace(lo, hi, 500)

            if x_add.size >= 5 and np.nanstd(x_add) > 0:
                ax.plot(grid, _kde_curve(x_add, grid, bw_scale=bw_scale),
                        linewidth=2.0, label="additive")
            if gpmap == "pairwise" and x_pair.size >= 5 and np.nanstd(x_pair) > 0:
                ax.plot(grid, _kde_curve(x_pair, grid, bw_scale=bw_scale),
                        linewidth=2.0, linestyle="--", label="pairwise")

            if r == 0: ax.set_title(gt_key, fontsize=11)
            if c == 0: ax.set_ylabel(cfg_name, fontsize=11)
            ax.grid(True, linestyle="--", alpha=0.35)

    xlabel = "log10(|coef| + 1e-8)" if log_version else "coefficient"
    fig.supxlabel(xlabel); fig.supylabel("density")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")

    fig.suptitle(f"{experiment} — surrogate × ground truth KDE grid", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=300); plt.close()
    print(f"[saved] {out_png}")


def plot_kde_coef_dists(
    coef_store, outdir, experiment, gt_keys, cfg_names,
    which="additive", transform="logabs", bw_scale=1.4,
):
    """One KDE overlay per GT key across all surrogate configs."""
    os.makedirs(outdir, exist_ok=True)

    for gt_key in gt_keys:
        series = {cfg: _finite_flat(coef_store.get(cfg, {}).get(which, {}).get(gt_key, []))
                  for cfg in cfg_names}
        pooled = (np.concatenate([v for v in series.values() if v.size], axis=0)
                  if any(v.size for v in series.values()) else np.array([]))

        if pooled.size == 0:
            print(f"[skip] {gt_key} ({which}) — no data"); continue

        pooled = _transform(pooled, transform)
        lo, hi = np.percentile(pooled, [0.5, 99.5])
        if lo == hi: lo -= 1e-6; hi += 1e-6
        grid = np.linspace(lo, hi, 500)

        plt.figure(figsize=(9, 5))
        for cfg, raw in series.items():
            if raw.size < 5: continue
            x = _transform(raw, transform)
            if np.nanstd(x) == 0: continue
            kde = gaussian_kde(x)
            kde.set_bandwidth(bw_method=kde.factor * bw_scale)
            plt.plot(grid, kde(grid), linewidth=2.2, label=f"{cfg} (N={x.size})")

        xlabel = {"logabs": "log10(|coef| + 1e-8)", "robust_z": "robust z-score (median/MAD)"
                  }.get(transform, "coefficient")
        plt.title(f"{experiment} — {gt_key} — {which} KDE")
        plt.xlabel(xlabel); plt.ylabel("Density")
        plt.grid(True, linestyle="--", alpha=0.35); plt.legend(); plt.tight_layout()
        out = os.path.join(outdir, f"{experiment}_{gt_key}_{which}_KDE.png")
        plt.savefig(out, dpi=300); plt.close()
        print(f"[saved] {out}")


def plot_coef_dists_per_gt(coef_store, outdir, experiment, gt_keys, bins=120):
    """Histogram overlay of additive vs pairwise coefficients per cfg × GT key."""
    os.makedirs(outdir, exist_ok=True)

    for cfg_name in coef_store.keys():
        for gt_key in gt_keys:
            add_list  = _as_list_of_arrays(coef_store[cfg_name]["additive"].get(gt_key, []))
            pair_list = _as_list_of_arrays(coef_store[cfg_name]["pairwise"].get(gt_key, []))
            add  = np.concatenate(add_list)  if len(add_list)  else np.array([])
            pair = np.concatenate(pair_list) if len(pair_list) else np.array([])

            if add.size == 0 and pair.size == 0:
                print(f"[skip] {cfg_name} {gt_key}: no coefficients stored"); continue

            fig, ax = plt.subplots(figsize=(9, 5))
            allv = np.concatenate([add, pair]) if (add.size and pair.size) else (add if add.size else pair)
            lo, hi = (np.percentile(allv, [1, 99]) if allv.size > 10 else (allv.min(), allv.max()))
            if lo == hi: lo -= 1e-6; hi += 1e-6
            bin_edges = np.linspace(lo, hi, bins + 1)

            if add.size:  ax.hist(add,  bins=bin_edges, density=True, alpha=0.45, label=f"additive (n={add.size})")
            if pair.size: ax.hist(pair, bins=bin_edges, density=True, alpha=0.45, label=f"pairwise (n={pair.size})")

            ax.set_title(f"{experiment} — {cfg_name} — {gt_key} coefficient distributions")
            ax.set_xlabel("Coefficient value"); ax.set_ylabel("Density")
            ax.grid(True, linestyle="--", alpha=0.35); ax.legend(); plt.tight_layout()
            out = f"{outdir}/coef_dist_{experiment}_{cfg_name}_{gt_key}.png"
            plt.savefig(out, dpi=250); plt.close(fig)
            print(f"[saved] {out}")
