"""
mRNA_RBP/gt_init.py

Case-specific ground-truth parameter initialization for mRNA-RBP experiments.

Biological motivation:
  - mRNA sequences bound by RBPs typically lack strong secondary structure at
    the binding site, so pairwise interactions are sparse (low p_edge) and most
    selected coupling entries are driven to zero by high L1 regularization.
  - Only Watson-Crick compatible pairs (G-C, C-G, A-U, U-A) are considered as
    potential edges — non-compatible positions cannot form stable base pairs.
  - The additive signal (which positions/nucleotides matter) is relatively dense;
    low L1 keeps most additive weights non-zero.
  - Binding affinity maps to a sigmoidal readout: a few critical mutations are
    highly deleterious while the majority of mutations are moderately tolerated.

Ground-truth model:
  score = sigmoid( energy / std )  −  sigmoid(0)

  Additive:  Gaussian noWT weights, low L1 sparsity
  Pairwise:  Gaussian noWT couplings on WC-compatible stem edges only,
             sparse selection (low p_edge), high L1 sparsity.
             Stems are consecutive nested WC pairs — they appear as vertical
             columns in the rotated-triangle contact map.
  Nonlin:    sigmoid (calibrated so WT maps to 0, mutations are ≤ 0)
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "residualbind"))

from ground_truth import (
    init_additive_noWT,
    init_sigmoid_nonlin,
    soft_threshold,
    additive_affinity_noWT,
    pairwise_potts_energy,
    apply_global_nonlin,
)
from seq_utils import rna_to_one_hot


# ---------------------------------------------------------------------------
# Case-specific defaults
# ---------------------------------------------------------------------------

MRNA_RBP_DEFAULTS = dict(
    sigma_w  = 0.5,   # additive weight scale
    l1_w     = 0.03,  # LOW additive L1: most positions contribute
    sigma_P  = 0.30,  # pairwise coupling scale
    l1_P     = 0.40,  # HIGH pairwise L1: drives most coupling entries to zero
    p_edge   = 0.15,  # LOW edge density: fraction of WC-compatible pairs selected
    bias     = 0.0,
)

_WC = {('G', 'C'), ('C', 'G'), ('A', 'U'), ('U', 'A')}


# ---------------------------------------------------------------------------
# Watson-Crick compatible pair utilities
# ---------------------------------------------------------------------------

def get_wc_compatible_pairs(seq: str) -> list:
    """Return all (i, j) pairs with i < j that are Watson-Crick compatible."""
    L = len(seq)
    return [(i, j) for i in range(L) for j in range(i + 1, L)
            if (seq[i], seq[j]) in _WC]


def init_wc_pairwise_sparse(
    rng,
    wt_onehot: np.ndarray,
    seq: str,
    p_edge: float = MRNA_RBP_DEFAULTS["p_edge"],
    sigma_P: float = MRNA_RBP_DEFAULTS["sigma_P"],
    l1_P: float = MRNA_RBP_DEFAULTS["l1_P"],
    wt_rowcol_zero: bool = True,
    edge_seed: int = 0,
) -> tuple:
    """Sparse Gaussian pairwise model restricted to Watson-Crick compatible edges.

    A fraction p_edge of all WC-compatible (G-C, C-G, A-U, U-A) pairs are
    selected deterministically given edge_seed.  Coupling weights are sampled
    as -|N(0, sigma_P)| then L1-soft-thresholded; high l1_P drives most
    entries within each edge to exactly zero.

    Passing different edge_seed values produces different (non-identical) edge
    subsets, so 10 instances of the same model have different pairing topologies.

    Returns (edges, J) in the same format as init_pairwise_potts_optionA.
    """
    wt_onehot = np.asarray(wt_onehot, dtype=np.float32)
    assert wt_onehot.ndim == 2 and wt_onehot.shape[1] == 4
    L = wt_onehot.shape[0]
    wt_idx = np.argmax(wt_onehot, axis=1).astype(int)

    compatible = get_wc_compatible_pairs(seq)
    n_select = max(1, round(p_edge * len(compatible)))

    rng_edges = np.random.default_rng(edge_seed)
    chosen = rng_edges.choice(len(compatible), size=n_select, replace=False)
    chosen.sort()
    edges = np.array([compatible[k] for k in chosen], dtype=np.int32)

    J = np.zeros((L, L, 4, 4), dtype=np.float32)
    for (i, j) in edges:
        M = -np.abs(rng.normal(0.0, sigma_P, (4, 4))).astype(np.float32)
        M = soft_threshold(M, l1_P).astype(np.float32)
        if wt_rowcol_zero:
            wi, wj = int(wt_idx[i]), int(wt_idx[j])
            M[wi, :] = 0.0
            M[:, wj] = 0.0
            M[wi, wj] = 0.0
        J[i, j] = M

    return edges, J


# ---------------------------------------------------------------------------
# Stem-based pairwise initializer
# ---------------------------------------------------------------------------

def _find_stems(seq: str, min_len: int = 2) -> list:
    """Return all maximal WC stems of length >= min_len as lists of (i, j) pairs.

    A stem rooted at outer pair (i, j) extends inward:
      (i, j), (i+1, j-1), ..., (i+k-1, j-k+1)
    where every pair is WC-compatible and non-overlapping (i+n < j-n).

    Only outermost pairs are considered roots — if (i-1, j+1) is also WC,
    (i, j) is an inner pair of a larger stem and is skipped as a root.
    Pairs assigned to a stem are not reused.

    In the rotated-triangle contact map these appear as vertical columns of
    cells at constant x = (i+j)/2.
    """
    L = len(seq)
    wc_set = {(i, j) for i in range(L) for j in range(i + 1, L)
              if (seq[i], seq[j]) in _WC}

    used: set = set()
    stems: list = []

    for i in range(L):
        for j in range(L - 1, i + 2 * min_len - 2, -1):
            if (i, j) not in wc_set or (i, j) in used:
                continue
            # skip if (i,j) is the inner pair of a larger stem
            if i > 0 and j < L - 1 and (i - 1, j + 1) in wc_set:
                continue
            # extend inward
            pairs = []
            n = 0
            while i + n < j - n and (i + n, j - n) in wc_set:
                pairs.append((i + n, j - n))
                used.add((i + n, j - n))
                n += 1
            if len(pairs) >= min_len:
                stems.append(pairs)

    return stems


def init_stem_pairwise(
    rng,
    wt_onehot: np.ndarray,
    seq: str,
    p_edge: float = MRNA_RBP_DEFAULTS["p_edge"],
    sigma_P: float = MRNA_RBP_DEFAULTS["sigma_P"],
    l1_P: float = MRNA_RBP_DEFAULTS["l1_P"],
    wt_rowcol_zero: bool = True,
    edge_seed: int = 0,
    min_stem_len: int = 2,
) -> tuple:
    """Sparse Gaussian pairwise model using consecutive WC stem structure.

    Selects whole stems (consecutive nested WC pairs) at random until the
    total pair count reaches p_edge * len(all_wc_pairs) — the same target
    as init_wc_pairwise_sparse.  Because stems are never split mid-way, the
    final count may exceed the target by at most one stem's length.

    Falls back to init_wc_pairwise_sparse when no valid stems are found.

    Returns (edges, J) in the same format as init_pairwise_potts_optionA.
    """
    wt_onehot = np.asarray(wt_onehot, dtype=np.float32)
    assert wt_onehot.ndim == 2 and wt_onehot.shape[1] == 4
    L = wt_onehot.shape[0]
    wt_idx = np.argmax(wt_onehot, axis=1).astype(int)

    n_target = max(1, round(p_edge * len(get_wc_compatible_pairs(seq))))
    stems = _find_stems(seq, min_len=min_stem_len)

    if not stems:
        return init_wc_pairwise_sparse(
            rng, wt_onehot, seq, p_edge=p_edge, sigma_P=sigma_P,
            l1_P=l1_P, wt_rowcol_zero=wt_rowcol_zero, edge_seed=edge_seed,
        )

    rng_edges = np.random.default_rng(edge_seed)
    order = rng_edges.permutation(len(stems))

    selected: list = []
    for idx in order:
        selected.extend(stems[idx])
        if len(selected) >= n_target:
            break

    edges = np.array(selected, dtype=np.int32)

    J = np.zeros((L, L, 4, 4), dtype=np.float32)
    for (i, j) in edges:
        M = -np.abs(rng.normal(0.0, sigma_P, (4, 4))).astype(np.float32)
        M = soft_threshold(M, l1_P).astype(np.float32)
        if wt_rowcol_zero:
            wi, wj = int(wt_idx[i]), int(wt_idx[j])
            M[wi, :] = 0.0
            M[:, wj] = 0.0
            M[wi, wj] = 0.0
        J[i, j] = M

    return edges, J


# ---------------------------------------------------------------------------
# Gaussian pairwise (non-WC constrained, original version)
# ---------------------------------------------------------------------------

def init_gaussian_pairwise_sparse(
    rng,
    wt_onehot: np.ndarray,
    p_edge: float = MRNA_RBP_DEFAULTS["p_edge"],
    sigma_P: float = MRNA_RBP_DEFAULTS["sigma_P"],
    l1_P: float = MRNA_RBP_DEFAULTS["l1_P"],
    wt_rowcol_zero: bool = True,
) -> tuple:
    """Sparse Gaussian pairwise over all possible pairs (no WC constraint).

    Edges are selected deterministically (evenly spaced across all L*(L-1)/2
    pairs).  Coupling weights are sampled as -|N(0, sigma_P)| then
    L1-soft-thresholded.

    Returns (edges, J) in the same format as init_pairwise_potts_optionA.
    """
    wt_onehot = np.asarray(wt_onehot, dtype=np.float32)
    assert wt_onehot.ndim == 2 and wt_onehot.shape[1] == 4
    L = wt_onehot.shape[0]
    wt_idx = np.argmax(wt_onehot, axis=1).astype(int)

    all_pairs = [(i, j) for i in range(L) for j in range(i + 1, L)]
    n_pairs = max(1, round(p_edge * len(all_pairs)))
    idx = np.round(np.linspace(0, len(all_pairs) - 1, n_pairs)).astype(int)
    edges = np.array([all_pairs[k] for k in idx], dtype=np.int32)

    J = np.zeros((L, L, 4, 4), dtype=np.float32)
    for (i, j) in edges:
        M = -np.abs(rng.normal(0.0, sigma_P, (4, 4))).astype(np.float32)
        M = soft_threshold(M, l1_P).astype(np.float32)
        if wt_rowcol_zero:
            wi, wj = int(wt_idx[i]), int(wt_idx[j])
            M[wi, :] = 0.0
            M[:, wj] = 0.0
            M[wi, wj] = 0.0
        J[i, j] = M

    return edges, J


# ---------------------------------------------------------------------------
# Case-specific bundle initializer
# ---------------------------------------------------------------------------

def init_mRNA_RBP_gt(
    rng,
    input_seq: str,
    x_calib: np.ndarray,
    *,
    sigma_w: float = MRNA_RBP_DEFAULTS["sigma_w"],
    l1_w: float    = MRNA_RBP_DEFAULTS["l1_w"],
    sigma_P: float = MRNA_RBP_DEFAULTS["sigma_P"],
    l1_P: float    = MRNA_RBP_DEFAULTS["l1_P"],
    p_edge: float  = MRNA_RBP_DEFAULTS["p_edge"],
    bias: float    = MRNA_RBP_DEFAULTS["bias"],
    edge_seed: int = 0,
) -> dict:
    """Initialise the mRNA-RBP ground-truth model and return a gt_bundle.

    Parameters
    ----------
    rng        : numpy Generator for sampling additive weights and coupling
                 magnitudes (does NOT control edge topology — use edge_seed)
    input_seq  : wildtype RNA sequence string
    x_calib    : (N, L, 4) one-hot calibration library used to fit sigmoid params
    edge_seed  : integer seed controlling which WC-compatible pairs are selected;
                 different values yield different pairing topologies

    Returns
    -------
    dict with keys:
      "params"        – {"W_mut", "edges", "J", "b", "mut_map"}
      "scores"        – {"sigmoid": {4 GT score arrays over x_calib}}
      "wt_scores"     – {"sigmoid": {4 GT scores at WT = 0.0 each}}
      "nonlin_kwargs" – {"sigmoid": calibrated sigmoid parameter dict}
    """
    wt_oh = rna_to_one_hot(input_seq)

    # Additive weights: Gaussian, low L1
    W_mut, mut_map, b0 = init_additive_noWT(
        rng, wt_oh, sigma=sigma_w, l1_w=l1_w, bias=bias
    )

    # Stem-structured pairwise: consecutive WC pairs (vertical columns in triangle plot)
    edges, J = init_stem_pairwise(
        rng, wt_oh, input_seq, p_edge=p_edge, sigma_P=sigma_P, l1_P=l1_P,
        edge_seed=edge_seed,
    )

    # Raw energy streams over calibration library
    s_add  = additive_affinity_noWT(x_calib, W_mut, mut_map, b=b0)
    s_pair = pairwise_potts_energy(x_calib, edges, J, b=0.0)
    s_addpair = s_add + s_pair

    # Sigmoid calibrated on additive energies; store addpair std separately
    sigmoid_kwargs = init_sigmoid_nonlin(s_add)
    sigmoid_kwargs["_norm_std_addpair"] = float(np.std(s_addpair)) + 1e-8

    std_add     = sigmoid_kwargs["_norm_std"]
    std_addpair = sigmoid_kwargs["_norm_std_addpair"]

    # Anchor: subtract sigmoid(0) so WT → 0 and all mutations ≤ 0
    wt_nl = float(apply_global_nonlin(np.array([[0.0]]), "sigmoid", sigmoid_kwargs))

    y_add     = s_add.reshape(-1)
    y_addpair = s_addpair.reshape(-1)
    y_nonlin_add = (
        apply_global_nonlin(s_add / std_add, "sigmoid", sigmoid_kwargs).reshape(-1) - wt_nl
    )
    y_nonlin_addpair = (
        apply_global_nonlin(s_addpair / std_addpair, "sigmoid", sigmoid_kwargs).reshape(-1) - wt_nl
    )

    scores = {
        "additive":                 y_add,
        "additive_pairwise":        y_addpair,
        "nonlin_additive":          y_nonlin_add,
        "nonlin_additive_pairwise": y_nonlin_addpair,
    }

    return {
        "params":        {"W_mut": W_mut, "edges": edges, "J": J, "b": b0, "mut_map": mut_map},
        "scores":        {"sigmoid": scores},
        "wt_scores":     {"sigmoid": {k: 0.0 for k in scores}},
        "nonlin_kwargs": {"sigmoid": sigmoid_kwargs},
    }


# ---------------------------------------------------------------------------
# Calibration pool helper (avoids circular import with generate_libraries)
# ---------------------------------------------------------------------------

def _sample_pool(wt_onehot: np.ndarray, n: int, mut_count: int, rng) -> np.ndarray:
    """Return (n, L) uint8 nuc_id array; each row is WT with exactly mut_count mutations."""
    L      = wt_onehot.shape[0]
    wt_idx = np.argmax(wt_onehot, axis=1).astype(np.uint8)
    ids    = np.tile(wt_idx[None, :], (n, 1))
    noise  = rng.random((n, L))
    pos    = np.argpartition(noise, mut_count, axis=1)[:, :mut_count]
    wt_at  = wt_idx[pos.ravel()]
    rand_nuc = rng.integers(0, 3, size=n * mut_count, dtype=np.uint8)
    new_nucs = np.where(rand_nuc >= wt_at, rand_nuc + 1, rand_nuc).astype(np.uint8)
    n_idx  = np.repeat(np.arange(n), mut_count)
    ids[n_idx, pos.ravel()] = new_nucs
    return ids


# ---------------------------------------------------------------------------
# Class-based interface (mirrors demo_additive_pairwise GroundTruth)
# ---------------------------------------------------------------------------

class MrnaRbpGroundTruth:
    """
    Manually-specified ground truth model for mRNA-RBP binding (AUGC alphabet).

    Mirrors the GroundTruth interface in demo_additive_pairwise (3).ipynb, but
    uses mRNA_RBP-style weight distributions and a hand-tuned sigmoid.

    Additive weights (all negative, noWT parameterisation):
      motif positions   – -HalfNormal(0, motif_sigma) + L1
      stem positions    – -HalfNormal(0, stem_additive_sigma) + L1
      background        – -HalfNormal(0, bg_sigma) + L1

    Pairwise weights (stem pairs only):
      WC-compatible entries (anti-diagonal in ACGU) and WT row/col → zero.
      Non-compensatory entries → -|N(0, stem_sigma)| + L1 soft-threshold.

    Nonlinearity (hand-tuned, human-readable):
      y = a · σ(b · latent + c) + d   anchored so WT → 0, mutations ≤ 0
      latent(WT) = 0 by noWT construction, so  wt_raw = a·σ(c)+d

    Sigmoid parameter intuition (same as demo notebook):

      | Param         | Controls                          | Rule of thumb                         |
      |---------------|-----------------------------------|---------------------------------------|
      | c             | WT operating point on sigmoid     | positive → WT in upper (active) tail  |
      | b             | sensitivity to latent energy      | higher → each broken pair costs more  |
      | a             | dynamic range                     | full swing from d to a+d              |
      | d             | activity floor                    | minimum for fully broken sequence     |
      | motif_sigma   | scale of motif additive penalty   | higher → motif mutations more costly  |
      | stem_sigma    | magnitude of stem coupling        | higher → stem pairs are more fragile  |

    Reference scenario (mRNA stem-fragile, biological default):
      motif_sigma=3.0, bg_sigma=0.5, stem_sigma=3.0, a=1.0, b=1.0, c=1.5, d=0.0

    Stem pair count for typical mRNA (~40–50% base-paired):
      L=41 → aim for 8–10 declared stem pairs (16–20 nt paired)

    Interface:
        gt.alpha          – (L, 4) additive weight matrix; WT nucleotide per position = 0
        gt.beta           – {(i, j): (4, 4) array} for each declared stem pair
        gt.edges          – (E, 2) stem pair indices
        gt.wt_one_hot()   – (L, 4) one-hot WT sequence
        gt.wt_activity()  – 0.0  (WT anchored to zero by construction)
        gt(x)             – (N,) nonlin_additive_pairwise scores for (N, L, 4) input
        gt.score_all(x)   – dict of all four GT score types (all ≤ 0, WT = 0)
    """

    def __init__(
        self,
        seq: str,
        stem_pairs: list,               # [(i, j), ...] — explicitly declared
        motif_positions: list,          # [i, ...] — explicitly declared
        *,
        # Additive: motif positions (-HalfNormal scale + L1)
        motif_sigma: float    = 3.0,
        motif_l1: float       = MRNA_RBP_DEFAULTS["l1_w"],      # 0.03
        # Additive: background positions (mRNA_RBP defaults)
        bg_sigma: float       = MRNA_RBP_DEFAULTS["sigma_w"],   # 0.5
        bg_l1: float          = MRNA_RBP_DEFAULTS["l1_w"],      # 0.03
        # Additive: stem positions (larger than bg — single break is costly)
        stem_additive_sigma: float = 0.0,
        # Pairwise: declared stem pairs (low L1 keeps blocks non-sparse)
        stem_sigma: float     = 3.0,
        stem_l1: float        = 0.05,
        # Sigmoid: y = a·σ(b·latent + c) + d,  anchored so WT = 0
        a: float              = 1.0,
        b: float              = 1.0,
        c: float              = 1.5,
        d: float              = 0.0,
        seed: int             = 42,
    ):
        self.seq             = seq
        self.stem_pairs      = list(stem_pairs)
        self.motif_positions = list(motif_positions)
        self._wt_oh          = rna_to_one_hot(seq)
        L                    = self._wt_oh.shape[0]
        wt_idx               = np.argmax(self._wt_oh, axis=1)

        # Sigmoid constants (stored for repr and _apply_sigmoid)
        self._a, self._b, self._c, self._d = float(a), float(b), float(c), float(d)
        # WT raw output: latent(WT)=0, so wt_raw = a·σ(c)+d
        self._wt_raw = float(a) / (1.0 + np.exp(-float(c))) + float(d)

        rng = np.random.default_rng(seed)

        # ── Additive weights ─────────────────────────────────────────────
        # noWT parameterisation: for each position, store 3 weights for
        # non-WT nucleotides; WT nucleotide implicitly has weight 0.
        mut_map   = np.zeros((L, 3), dtype=int)
        W_mut     = np.zeros((L, 3), dtype=np.float32)
        motif_set = set(motif_positions)
        stem_pos_set = {p for pair in stem_pairs for p in pair}

        for pos in range(L):
            non_wt = [n for n in range(4) if n != wt_idx[pos]]
            mut_map[pos] = non_wt
            if pos in motif_set:
                raw = -np.abs(rng.normal(0.0, motif_sigma, size=3).astype(np.float32))
                W_mut[pos] = soft_threshold(raw, motif_l1)
            elif pos in stem_pos_set:
                raw = -np.abs(rng.normal(0.0, stem_additive_sigma, size=3).astype(np.float32))
                W_mut[pos] = soft_threshold(raw, bg_l1)
            else:
                # Small negative: -|N(0, bg_sigma)|
                raw = -np.abs(rng.normal(0.0, bg_sigma, size=3).astype(np.float32))
                W_mut[pos] = soft_threshold(raw, bg_l1)

        # ── Pairwise at stem positions: non-compensatory penalty only ───────
        # For each declared stem pair (i,j), sample -|N(0,stem_sigma)| then
        # zero out: (a) WT row/col (noWT), (b) WC-compatible entries (anti-
        # diagonal in ACGU order) so compensatory double mutations have no
        # pairwise effect. Non-compensatory double mutations get a negative
        # coupling that further decreases the score.
        # Non-stem positions have J=0 (pairwise only in the stem region).
        _WC_IDX = {(0, 3), (1, 2), (2, 1), (3, 0)}  # ACGU: AU,CG,GC,UA

        if stem_pairs:
            edges = np.array(stem_pairs, dtype=np.int32)
        else:
            edges = np.empty((0, 2), dtype=np.int32)
        J = np.zeros((L, L, 4, 4), dtype=np.float32)
        for (i, j) in stem_pairs:
            M = -np.abs(rng.normal(0.0, stem_sigma, (4, 4))).astype(np.float32)
            M = soft_threshold(M, stem_l1)
            wi, wj = int(wt_idx[i]), int(wt_idx[j])
            M[wi, :] = 0.0
            M[:, wj] = 0.0
            M[wi, wj] = 0.0
            for (a, b) in _WC_IDX:
                M[a, b] = 0.0
            J[i, j] = M

        self._W_mut   = W_mut
        self._mut_map = mut_map
        self._edges   = edges
        self._J       = J

    # ── Sigmoid ──────────────────────────────────────────────────────────

    def _apply_sigmoid(self, s: np.ndarray) -> np.ndarray:
        """y = a·σ(b·s + c) + d  anchored so output at s=0 (WT) equals 0."""
        raw = self._a / (1.0 + np.exp(-(self._b * s + self._c))) + self._d
        return raw - self._wt_raw

    # ── Public interface ──────────────────────────────────────────────────

    @property
    def alpha(self) -> np.ndarray:
        """(L, 4) additive weight matrix; WT nucleotide per position = 0."""
        L     = self._wt_oh.shape[0]
        alpha = np.zeros((L, 4), dtype=np.float32)
        for pos in range(L):
            for k in range(3):
                alpha[pos, int(self._mut_map[pos, k])] = self._W_mut[pos, k]
        return alpha

    @property
    def beta(self) -> dict:
        """Dict {(i, j): (4, 4) array} for each declared stem pair.
        WC-compatible and WT-row/col entries are zero; non-compensatory entries
        are negative (drawn from -|N(0, stem_sigma)|)."""
        return {(int(i), int(j)): self._J[int(i), int(j)] for (i, j) in self.stem_pairs}

    @property
    def edges(self) -> np.ndarray:
        """(E, 2) int32 array of declared stem pair indices."""
        return self._edges

    def wt_one_hot(self) -> np.ndarray:
        return self._wt_oh.copy()

    def wt_activity(self) -> float:
        return 0.0

    def score_all(self, x: np.ndarray) -> dict:
        """Score (N, L, 4) one-hot sequences. All scores ≤ 0 with WT = 0."""
        x      = np.asarray(x, dtype=np.float32)
        s_add  = additive_affinity_noWT(x, self._W_mut, self._mut_map, b=0.0).reshape(-1)
        s_pair = (
            pairwise_potts_energy(x, self._edges, self._J, b=0.0).reshape(-1)
            if len(self._edges) else np.zeros_like(s_add)
        )
        return {
            "additive":                 s_add,
            "additive_pairwise":        s_add + s_pair,
            "nonlin_additive":          self._apply_sigmoid(s_add),
            "nonlin_additive_pairwise": self._apply_sigmoid(s_add + s_pair),
        }

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Score (N, L, 4) one-hot sequences. Returns (N,) nonlin_additive_pairwise."""
        return self.score_all(x)["nonlin_additive_pairwise"]

    def __repr__(self) -> str:
        return (
            f"MrnaRbpGroundTruth(L={len(self.seq)}, "
            f"stem_pairs={len(self.stem_pairs)}, motif={len(self.motif_positions)}, "
            f"sigmoid=(a={self._a}, b={self._b}, c={self._c}, d={self._d}), "
            f"wt_activity={self.wt_activity():.3f})"
        )
