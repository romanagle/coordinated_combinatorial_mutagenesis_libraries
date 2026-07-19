"""
Oracle-scored library distribution figures for MSI1 and VTS1 ResidualBind ensembles.

Generates 4 output figures (all → outputs/notebook_plots/):
  rand_lib_dist_msi1_oracle.png    — 3 panels: 5%, 10%, 25% random libs, MSI1 oracle
  rand_lib_dist_vts1_oracle.png    — 3 panels: 5%, 10%, 25% random libs, VTS1 oracle
  library_distributions_msi1_oracle.png — activity-balanced / Pairwise / Type3, MSI1 oracle
  library_distributions_vts1_oracle.png — activity-balanced / Pairwise / Type3, VTS1 oracle

MSI1 oracle: ResidualBind ensemble at rbp/results/msi1/
VTS1 oracle: ResidualBind ensemble at rbp/results/vts1/

Oracle scoring runs in toehold_gpu env (torch). Results are cached so subsequent
runs are instant without GPU.
"""

import re
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "rbp" / "src"))
sys.path.insert(0, str(_REPO))

from mRNA_RBP.sequence_configs import (
    MSI1_SEQ,
    MSI1_STEM_PAIRS,
    VTS1_SEQ,
    VTS1_STEM_PAIRS,
    generate_pairwise_nuc_ids,
    generate_type3_nuc_ids,
    sample_exact_mutants,
)
from mRNA_RBP.generate_libraries import (
    ACTIVITY_BALANCED_CANDIDATE_N,
    ACTIVITY_BALANCED_MUT_COUNTS,
    ACTIVITY_BALANCED_TARGET_N,
)

# ── RBP configs ───────────────────────────────────────────────────────────────

MSI1_CKPT       = _REPO / "rbp" / "results" / "msi1"
MSI1_CACHE      = _REPO / "mRNA_RBP" / "outputs" / "notebook_plots" / "cache_msi1_lib_scores.npz"

VTS1_CKPT       = _REPO / "rbp" / "results" / "vts1"
VTS1_CACHE      = _REPO / "mRNA_RBP" / "outputs" / "notebook_plots" / "cache_vts1_lib_scores.npz"

# Pre-scored files
MSI1_RB_BASE = (_REPO / "mRNA_RBP" / "outputs_vts1_residualbind"
                / "outputs_residualbind_ensemble" / "instance_00")
VTS1_LIB_BASE = _REPO / "mRNA_RBP" / "outputs_vts1_residualbind" / "instance_00"
SYNTH_INST    = _REPO / "mRNA_RBP" / "outputs" / "instance_00"

OUT_DIR = _REPO / "mRNA_RBP" / "outputs" / "notebook_plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BINS  = 60
ALPHA = 0.85
C_PAIR = "#4C72B0"
C_RAND = "#55A868"
_NUC_IDX = {"A": 0, "C": 1, "G": 2, "U": 3}


# ── sequence helpers ──────────────────────────────────────────────────────────

def to_onehot(seq: str) -> np.ndarray:
    oh = np.zeros((len(seq), 4), dtype=np.float32)
    for i, c in enumerate(seq.upper()):
        oh[i, _NUC_IDX[c]] = 1.0
    return oh


def nuc_ids_to_onehot(nuc_ids: np.ndarray) -> np.ndarray:
    return np.eye(4, dtype=np.float32)[nuc_ids]


def generate_random_lib(wt_seq: str, n: int, mut_rate: float, seed: int) -> np.ndarray:
    """Generate n random sequences with ~mut_rate fraction mutated per position."""
    rng = np.random.default_rng(seed)
    L = len(wt_seq)
    wt_ids = np.array([_NUC_IDX[c] for c in wt_seq.upper()], dtype=np.int32)
    nuc_ids = np.tile(wt_ids, (n, 1))
    mask = rng.random((n, L)) < mut_rate
    alt_ids = rng.integers(0, 3, size=(n, L))
    # shift alts to avoid WT allele
    for pos in range(L):
        wt_a = wt_ids[pos]
        alt_col = alt_ids[:, pos]
        alt_col[alt_col >= wt_a] += 1
    nuc_ids[mask] = alt_ids[mask]
    return nuc_ids


def generate_pairwise_lib(wt_seq: str, stem_pairs: list) -> np.ndarray:
    """All 16 nuc combos for each stem pair (other positions at WT)."""
    return generate_pairwise_nuc_ids(wt_seq, stem_pairs).astype(np.int32)


def _uniformize_indices(scores, n_bins=200, clip_lo=1, clip_hi=99, target_n=None, seed=42):
    """Return (scores_kept, orig_indices) that give a uniform score histogram."""
    rng = np.random.default_rng(seed)
    scores = np.asarray(scores, dtype=float).ravel()
    finite = np.isfinite(scores)
    orig_idx = np.where(finite)[0]
    s = scores[orig_idx]
    lo = np.nanpercentile(s, clip_lo)
    hi = np.nanpercentile(s, clip_hi)
    in_range = (s >= lo) & (s <= hi)
    range_pos = np.where(in_range)[0]
    s_r = s[range_pos]
    n_bins_eff = min(n_bins, len(s_r))
    edges = np.linspace(lo, hi, n_bins_eff + 1)
    bin_ids = np.clip(np.digitize(s_r, edges) - 1, 0, n_bins_eff - 1)
    counts = np.bincount(bin_ids, minlength=n_bins_eff)
    per_bin = int(counts[counts > 0].min())
    if target_n is not None:
        n_nonempty = int(np.sum(counts > 0))
        per_bin = min(per_bin, max(1, target_n // max(n_nonempty, 1)))
    keep_r = []
    for b in range(n_bins_eff):
        idx = np.where(bin_ids == b)[0]
        if idx.size == 0:
            continue
        keep_r.append(idx if idx.size <= per_bin
                      else rng.choice(idx, size=per_bin, replace=False))
    keep_r = np.concatenate(keep_r) if keep_r else np.array([], dtype=int)
    keep = range_pos[keep_r]
    return s[keep], orig_idx[keep]


def generate_type3_lib(wt_seq: str, n_3mut: int = 4, seed: int = 42) -> tuple:
    """All 1-mutant and 2-mutant sequences (exhaustive) plus n_3mut random 3-mutants."""
    nids, labels = generate_type3_nuc_ids(wt_seq, n_3mut=n_3mut, seed=seed)
    return nids.astype(np.int32), labels


# ── oracle ────────────────────────────────────────────────────────────────────

class RBEnsemble:
    def __init__(self, seq: str, ckpt_dir: Path, batch_size: int = 4096):
        import torch
        from rbp_bench.models import load_ensemble
        self._wt_oh = to_onehot(seq)
        paths = sorted(
            ckpt_dir.glob("member*.pt"),
            key=lambda p: int(re.search(r"member(\d+)", p.name).group(1)),
        )
        if not paths:
            raise FileNotFoundError(f"No member*.pt in {ckpt_dir}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sample = torch.zeros((2, self._wt_oh.shape[0], 4), dtype=torch.float32)
        model = load_ensemble(paths, "residualbind", {}, device, sample)
        model.eval()
        self._model = model
        self._device = device
        self._torch  = torch
        self.batch_size = batch_size
        with torch.no_grad():
            wt_t = torch.from_numpy(self._wt_oh[None]).to(device)
            self._wt_score = float(model(wt_t).detach().cpu().numpy()[0])
        print(f"  {len(paths)} members, device={device}, wt_score={self._wt_score:.4f}")

    def score(self, x: np.ndarray) -> np.ndarray:
        torch = self._torch
        outs = []
        with torch.no_grad():
            for s in range(0, len(x), self.batch_size):
                xb = torch.from_numpy(x[s:s+self.batch_size]).to(self._device)
                outs.append(self._model(xb).detach().cpu().numpy().reshape(-1))
        return (np.concatenate(outs) - self._wt_score).astype(np.float32)


# ── cache helpers ─────────────────────────────────────────────────────────────

def load_cache(path: Path) -> dict:
    if path.exists():
        d = np.load(path, allow_pickle=True)
        return {k: d[k] for k in d.files}
    return {}


def save_cache(path: Path, cache: dict):
    np.savez(path, **cache)


# ── score with oracle (loads from cache if available) ─────────────────────────

def get_scores(cache: dict, key: str, nuc_ids: np.ndarray, oracle) -> np.ndarray:
    if key in cache:
        return cache[key].astype(np.float64)
    print(f"  scoring {key} (n={len(nuc_ids)}) …")
    oh = nuc_ids_to_onehot(nuc_ids)
    scores = oracle.score(oh)
    cache[key] = scores
    return scores.astype(np.float64)


# ── plot helpers ──────────────────────────────────────────────────────────────

def make_bins(scores, n=BINS, xlim=None):
    if xlim is not None:
        lo, hi = xlim
    else:
        lo, hi = np.percentile(scores, [0.2, 99.8])
        pad = (hi - lo) * 0.03
        lo -= pad; hi += pad
    return np.linspace(lo, hi, n + 1)


def shared_xlim(all_scores_list):
    lo = min(np.percentile(s, 0.2) for s in all_scores_list)
    hi = max(np.percentile(s, 99.8) for s in all_scores_list)
    pad = (hi - lo) * 0.03
    return (lo - pad, hi + pad)


def _stacked_hist(ax, scores, labels, mut_range, colors, title, xlabel, xlim=None):
    all_bins = make_bins(scores, xlim=xlim)
    bottoms  = np.zeros(len(all_bins) - 1)
    widths   = np.diff(all_bins)
    total    = len(scores)
    for m in mut_range:
        mask = labels == m
        if mask.sum() == 0:
            continue
        counts, _ = np.histogram(scores[mask], bins=all_bins)
        density = counts / (total * widths)
        ax.bar(all_bins[:-1], density, width=widths, bottom=bottoms,
               color=colors[m], align="edge", alpha=0.92,
               label=f"{m}-mut (n={mask.sum():,})")
        bottoms += density
    ax.axvline(np.mean(scores), color="k", ls="--", lw=1,
               label=f"mean={np.mean(scores):.3f}")
    if xlim is not None:
        ax.set_xlim(xlim)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    handles, lbls = ax.get_legend_handles_labels()
    mean_h = handles.pop(-1); mean_l = lbls.pop(-1)
    ax.legend(handles + [mean_h], lbls + [mean_l],
              fontsize=8, framealpha=0.85, loc="upper left", handlelength=1.2)


def _plain_hist(ax, scores, title, xlabel, color=C_RAND, xlim=None, stat="mean"):
    bins = make_bins(scores, xlim=xlim)
    ax.hist(scores, bins=bins, density=True, color=color, alpha=ALPHA)
    stat_val = np.median(scores) if stat == "median" else np.mean(scores)
    ax.axvline(stat_val, color="k", ls="--", lw=1,
               label=f"{stat}={stat_val:.3f}")
    if xlim is not None:
        ax.set_xlim(xlim)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.legend(fontsize=8, framealpha=0.8)


# ── build data ────────────────────────────────────────────────────────────────

def build_msi1_data() -> dict:
    """Load pre-scored MSI1 oracle libs; score type2/pairwise/type3 if not cached."""
    cache = load_cache(MSI1_CACHE)
    changed = False

    # Random libs — pre-scored, just load
    data = {}
    for mut_dir, label in [("mut05", "rand05"), ("mut10", "rand10"), ("mut25", "rand25")]:
        path = MSI1_RB_BASE / mut_dir / "lib_20000.npz"
        d = np.load(path)
        data[label] = {
            "scores": d["scores_residualbind_ensemble"].astype(np.float64),
            "n": d["nuc_ids"].shape[0],
        }

    t2_path = SYNTH_INST / "activity_balanced.npz"
    if not t2_path.exists():
        t2_path = SYNTH_INST / "type2.npz"
    d_t2 = np.load(t2_path)
    t2_ids = d_t2["nuc_ids"]
    if "type2_nids" not in cache or not np.array_equal(cache["type2_nids"], t2_ids):
        cache.pop("type2", None)
        cache["type2_nids"] = t2_ids.astype(np.int32)
        changed = True

    needs_oracle = any(k not in cache for k in ("type2", "pairwise", "type3"))
    oracle = None
    if needs_oracle:
        print("\n=== Loading MSI1 oracle ===")
        oracle = RBEnsemble(MSI1_SEQ, MSI1_CKPT)

    need_t2_score = "type2" not in cache
    scores_t2 = get_scores(cache, "type2", d_t2["nuc_ids"], oracle) if oracle or need_t2_score else cache["type2"].astype(np.float64)
    if need_t2_score:
        cache["type2"] = scores_t2.astype(np.float32); changed = True
    labels_t2 = d_t2["rate_labels"].astype(int)
    data["type2"] = {"scores": scores_t2, "labels": labels_t2}

    d_pw = np.load(SYNTH_INST / "pairwise_lib.npz")
    scores_pw = get_scores(cache, "pairwise", d_pw["nuc_ids"], oracle) if oracle or "pairwise" not in cache else cache["pairwise"].astype(np.float64)
    if "pairwise" not in cache:
        cache["pairwise"] = scores_pw.astype(np.float32); changed = True
    data["pairwise"] = {"scores": scores_pw}

    t3_path = SYNTH_INST / "type3.npz"
    if t3_path.exists():
        d_t3 = np.load(t3_path)
        scores_t3 = get_scores(cache, "type3", d_t3["nuc_ids"], oracle) if oracle or "type3" not in cache else cache["type3"].astype(np.float64)
        if "type3" not in cache:
            cache["type3"] = scores_t3.astype(np.float32); changed = True
        labels_t3 = d_t3["rate_labels"].astype(np.uint8)
        data["type3"] = {"scores": scores_t3, "labels": labels_t3}

    if oracle is not None:
        save_cache(MSI1_CACHE, cache)
        print(f"  Cached → {MSI1_CACHE}")

    return data


def build_vts1_data() -> dict:
    """Load pre-scored VTS1 rand10; generate/score other libs if not cached."""
    cache = load_cache(VTS1_CACHE)
    if "wt_seq" not in cache or str(cache["wt_seq"].item()) != VTS1_SEQ:
        cache = {"wt_seq": np.array(VTS1_SEQ)}
    changed = False

    data = {}

    # rand10 — pre-scored
    d_r10 = np.load(VTS1_LIB_BASE / "vts1_mut10" / "lib_20000.npz")
    data["rand10"] = {"scores": d_r10["scores_vts1_residualbind"].astype(np.float64), "n": 20000}

    recipe = np.array([
        ACTIVITY_BALANCED_TARGET_N,
        ACTIVITY_BALANCED_CANDIDATE_N,
        *ACTIVITY_BALANCED_MUT_COUNTS,
    ], dtype=np.int32)
    stale_activity_balanced = (
        "activity_balanced_recipe" not in cache
        or not np.array_equal(cache["activity_balanced_recipe"], recipe)
    )
    needs_oracle = stale_activity_balanced or any(
        k not in cache for k in ("rand05", "rand25", "type2", "type2_labels",
                                 "pairwise", "type3", "type3_labels")
    )
    oracle = None
    if needs_oracle:
        print("\n=== Loading VTS1 oracle ===")
        oracle = RBEnsemble(VTS1_SEQ, VTS1_CKPT)

    def _score_or_load(key, nuc_ids):
        nonlocal changed
        if key in cache:
            return cache[key].astype(np.float64)
        oh = nuc_ids_to_onehot(nuc_ids)
        print(f"  scoring {key} (n={len(nuc_ids)}) …")
        s = oracle.score(oh).astype(np.float32)
        cache[key] = s; changed = True
        return s.astype(np.float64)

    # rand05 / rand25 — generate on-the-fly
    for mut_rate, label, seed in [(0.05, "rand05", 101), (0.25, "rand25", 102)]:
        nids_key = f"{label}_nids"
        if nids_key not in cache:
            nids = generate_random_lib(VTS1_SEQ, 20000, mut_rate, seed).astype(np.int32)
            cache[nids_key] = nids; changed = True
        else:
            nids = cache[nids_key].astype(np.int32)
        scores = _score_or_load(label, nids)
        data[label] = {"scores": scores, "n": len(nids)}

    # Activity-balanced — 200K candidates across 3/5/7/15 mutations, then
    # globally uniformized to 20K by oracle activity.
    t2_nids_key = "type2_nids"
    if stale_activity_balanced or t2_nids_key not in cache:
        per_count, rem = divmod(ACTIVITY_BALANCED_CANDIDATE_N,
                                len(ACTIVITY_BALANCED_MUT_COUNTS))
        all_nids, all_labels = [], []
        for i, k in enumerate(ACTIVITY_BALANCED_MUT_COUNTS):
            target_n = per_count + (1 if i < rem else 0)
            pool_nids = sample_exact_mutants(
                VTS1_SEQ, k, target_n, seed=20_000 + k
            ).astype(np.int32)
            all_nids.append(pool_nids)
            all_labels.append(np.full(len(pool_nids), k, dtype=np.int32))
            print(f"  activity-balanced {k}-mut candidates: {len(pool_nids):,} unique seqs")

        candidate_nids = np.concatenate(all_nids, axis=0).astype(np.int32)
        candidate_labels = np.concatenate(all_labels, axis=0)
        print(f"  scoring activity-balanced candidates (n={len(candidate_nids):,}) …")
        candidate_scores = oracle.score(nuc_ids_to_onehot(candidate_nids)).astype(np.float64)

        _, keep = _uniformize_indices(candidate_scores, n_bins=200,
                                      clip_lo=1, clip_hi=99,
                                      target_n=ACTIVITY_BALANCED_TARGET_N,
                                      seed=20_600)
        nids_t2   = candidate_nids[keep].astype(np.int32)
        labels_t2 = candidate_labels[keep]
        scores_t2 = candidate_scores[keep]

        vals, cnts = np.unique(labels_t2, return_counts=True)
        print(f"  activity-balanced total n={len(nids_t2):,}: " +
              "  ".join(f"{m}mut:{c}" for m, c in zip(vals, cnts)))

        cache["type2_nids"]   = nids_t2.astype(np.int32)
        cache["type2_labels"] = labels_t2
        cache["type2"]        = scores_t2.astype(np.float32)
        cache["activity_balanced_recipe"] = recipe
        cache["wt_seq"]       = np.array(VTS1_SEQ)
        changed = True
    else:
        nids_t2   = cache["type2_nids"].astype(np.int32)
        labels_t2 = cache["type2_labels"].astype(np.int32)
        scores_t2 = cache["type2"].astype(np.float64)

    data["type2"] = {"scores": scores_t2, "labels": labels_t2}

    # pairwise — all 16 combos per VTS1 stem pair
    pw_nids_key = "pairwise_nids"
    if pw_nids_key not in cache:
        nids_pw = generate_pairwise_lib(VTS1_SEQ, VTS1_STEM_PAIRS).astype(np.int32)
        cache[pw_nids_key] = nids_pw; changed = True
    else:
        nids_pw = cache[pw_nids_key].astype(np.int32)
    scores_pw = _score_or_load("pairwise", nids_pw)
    data["pairwise"] = {"scores": scores_pw}

    # type3 — exhaustive 1-mut and 2-mut
    t3_nids_key = "type3_nids"
    if t3_nids_key not in cache:
        nids_t3, labels_t3 = generate_type3_lib(VTS1_SEQ)
        cache[t3_nids_key] = nids_t3.astype(np.int32)
        cache["type3_labels"] = labels_t3.astype(np.int32); changed = True
    else:
        nids_t3  = cache[t3_nids_key].astype(np.int32)
        labels_t3 = cache["type3_labels"].astype(np.int32)
    scores_t3 = _score_or_load("type3", nids_t3)
    data["type3"] = {"scores": scores_t3, "labels": labels_t3}

    if changed:
        save_cache(VTS1_CACHE, cache)
        print(f"  Cached → {VTS1_CACHE}")

    return data


# ── figures ───────────────────────────────────────────────────────────────────

def plot_rand_lib_dist(data: dict, rbp_label: str, out_path: Path, xlim=None):
    xlabel = f"Oracle score (Δ activity, {rbp_label})"
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    fig.subplots_adjust(wspace=0.35)
    for ax, (key, pct_label) in zip(axes, [("rand05", "5%"), ("rand10", "10%"), ("rand25", "25%")]):
        d = data[key]
        scores, n = d["scores"], d["n"]
        _plain_hist(ax, scores,
                    title=f"Random lib ({pct_label} mut, n={n:,})",
                    xlabel=xlabel, xlim=xlim, stat="median")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


def plot_library_distributions(data: dict, rbp_label: str, out_path: Path, xlim=None):
    xlabel = f"Oracle score (Δ activity, {rbp_label})"
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    fig.subplots_adjust(wspace=0.35)

    # Panel 1 — activity-balanced (stacked by mut count)
    t2 = data["type2"]
    scores_t2, labels_t2 = t2["scores"], t2["labels"]
    mut_range_t2 = sorted(set(labels_t2.tolist()))
    _cmap_t2 = cm.get_cmap("plasma_r", len(mut_range_t2) + 2)
    colors_t2 = {m: _cmap_t2(i + 1) for i, m in enumerate(mut_range_t2)}
    _stacked_hist(axes[0], scores_t2, labels_t2, mut_range_t2, colors_t2,
                  title=f"Activity-balanced library  (n={len(scores_t2):,})", xlabel=xlabel, xlim=xlim)

    # Panel 2 — Pairwise
    scores_pw = data["pairwise"]["scores"]
    _plain_hist(axes[1], scores_pw,
                title=f"Pairwise library  (n={len(scores_pw):,})",
                xlabel=xlabel, color=C_PAIR, xlim=xlim)

    # Panel 3 — Type3 exhaustive (stacked by mut count)
    t3 = data.get("type3")
    if t3 is not None:
        scores_t3, labels_t3 = t3["scores"], t3["labels"]
        mut_range_t3 = sorted(set(labels_t3.tolist()))
        _cmap_t3 = cm.get_cmap("viridis", len(mut_range_t3) + 2)
        colors_t3 = {m: _cmap_t3(i + 1) for i, m in enumerate(mut_range_t3)}
        _stacked_hist(axes[2], scores_t3, labels_t3, mut_range_t3, colors_t3,
                      title=f"Exhaustive library  (n={len(scores_t3):,})", xlabel=xlabel, xlim=xlim)
    else:
        axes[2].set_visible(False)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("Building MSI1 data …")
    msi1 = build_msi1_data()
    print("Building VTS1 data …")
    vts1 = build_vts1_data()

    # Shared x-axis for rand_lib_dist (across all rand panels of both RBPs)
    rand_all = [msi1[k]["scores"] for k in ("rand05", "rand10", "rand25")]
    rand_all += [vts1[k]["scores"] for k in ("rand05", "rand10", "rand25")]
    xlim_rand = shared_xlim(rand_all)

    # Shared x-axis for library_distributions (type2, pairwise, type3 from both)
    lib_all = [msi1["type2"]["scores"], msi1["pairwise"]["scores"],
               vts1["type2"]["scores"], vts1["pairwise"]["scores"]]
    if "type3" in msi1:
        lib_all.append(msi1["type3"]["scores"])
    if "type3" in vts1:
        lib_all.append(vts1["type3"]["scores"])
    xlim_lib = shared_xlim(lib_all)

    print(f"\nShared xlim rand: {xlim_rand}")
    print(f"Shared xlim lib:  {xlim_lib}")

    plot_rand_lib_dist(msi1, "MSI1",
        OUT_DIR / "rand_lib_dist_msi1_oracle.png", xlim=xlim_rand)
    plot_rand_lib_dist(vts1, "VTS1",
        OUT_DIR / "rand_lib_dist_vts1_oracle.png", xlim=xlim_rand)
    plot_library_distributions(msi1, "MSI1",
        OUT_DIR / "library_distributions_msi1_oracle.png", xlim=xlim_lib)
    plot_library_distributions(vts1, "VTS1",
        OUT_DIR / "library_distributions_vts1_oracle.png", xlim=xlim_lib)
    print("\nDone.")


if __name__ == "__main__":
    main()
