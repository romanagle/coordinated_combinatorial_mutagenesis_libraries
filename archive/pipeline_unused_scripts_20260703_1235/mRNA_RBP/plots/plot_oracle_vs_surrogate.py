"""Two-column figure per RBP: oracle SSM+epistasis (left) vs surrogate α+β (right).

Left column (ResidualBind ensemble oracle):
  Top: 4×L single-mutant Δ-activity heatmap
  Bottom: 4L×4L epistasis = double_mut − ssm_i − ssm_j  (upper triangle)

Right column (nonlinear additive+pairwise MAVE-NN surrogate):
  Top: 4×L additive coefficient α
  Bottom: L×L pairwise coupling β (abs-max with sign per position pair, upper triangle)

Surrogate trained on 20k sequences at 10% mut rate, labeled by ResidualBind oracle.

Oracle SSM/DM scoring results are cached to avoid re-running on every plot call.

Outputs:
  mRNA_RBP/outputs/notebook_plots/oracle_vs_surrogate_msi1.png
  mRNA_RBP/outputs/notebook_plots/oracle_vs_surrogate_vts1.png

Run with toehold_gpu env (torch required for oracle; surrogate loading is numpy-only).
"""

import re
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from matplotlib.colors import TwoSlopeNorm

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "rbp" / "src"))

from coef_map_shared import right_panel_pairwise_vmax

_NUC_IDX   = {"A": 0, "C": 1, "G": 2, "U": 3}
NUCS_AUGC  = ["A", "U", "G", "C"]

_MOTIF_COLOR = '#AED6F1'   # soft blue
_BOX_COLOR   = '#1E8449'   # dark green

# ── RBP configs ──────────────────────────────────────────────────────────────

MSI1_SEQ             = "AAAAAAAAGCGCAUGCUUGCAUGGCAUGCGCAAAAAAAAAA"
MSI1_STEM_PAIRS      = [(8,30),(9,29),(10,28),(11,27),(12,26),(13,25),(14,24),(15,23)]
MSI1_MOTIF_POSITIONS = [17, 18, 19, 20, 21]
MSI1_CKPT            = _REPO / "rbp" / "results" / "msi1"
MSI1_COEF            = (_REPO / "mRNA_RBP"
                         / "outputs_vts1_residualbind"
                         / "outputs_residualbind_ensemble"
                         / "surrogate_coefs"
                         / "coefs_k00_mut10_lib20000_nonlinear_additive_p_pairwise_residualbind_ensemble.npz")
MSI1_CACHE           = _REPO / "mRNA_RBP" / "outputs" / "notebook_plots" / "cache_msi1_oracle.npz"

VTS1_SEQ             = "AAGGACACUAAGUACAGGUUGCUGGCACAGGGCGCUCAUAA"
VTS1_STEM_PAIRS      = [(16,21),(15,22),(14,23),(25,31),(24,32)]
VTS1_MOTIF_POSITIONS = [20, 21, 22, 23, 24]
VTS1_CKPT            = _REPO / "rbp" / "results" / "vts1"
VTS1_COEF            = (_REPO / "mRNA_RBP"
                         / "outputs_vts1_residualbind"
                         / "surrogate_coefs"
                         / "coefs_vts1_mut10_lib20000_nonlinear_additive_p_pairwise_vts1_residualbind.npz")
VTS1_CACHE           = _REPO / "mRNA_RBP" / "outputs" / "notebook_plots" / "cache_vts1_oracle.npz"

OUT_DIR = _REPO / "mRNA_RBP" / "outputs" / "notebook_plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── sequence helpers ──────────────────────────────────────────────────────────

def to_onehot(seq: str) -> np.ndarray:
    oh = np.zeros((len(seq), 4), dtype=np.float32)
    for i, c in enumerate(seq.upper()):
        oh[i, _NUC_IDX[c]] = 1.0
    return oh


def make_all_single_muts(seq: str) -> np.ndarray:
    wt = to_onehot(seq)
    L = len(seq)
    xs = []
    for i in range(L):
        for a in range(4):
            x = wt.copy(); x[i] = 0.0; x[i, a] = 1.0
            xs.append(x)
    return np.stack(xs)


def make_all_double_muts(seq: str) -> np.ndarray:
    wt = to_onehot(seq)
    L = len(seq)
    xs = []
    for i in range(L):
        for a in range(4):
            for j in range(L):
                for b in range(4):
                    x = wt.copy()
                    x[i] = 0.0; x[i, a] = 1.0
                    x[j] = 0.0; x[j, b] = 1.0
                    xs.append(x)
    return np.stack(xs)


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
        self._torch = torch
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
                xb = torch.from_numpy(x[s:s + self.batch_size]).to(self._device)
                outs.append(self._model(xb).detach().cpu().numpy().reshape(-1))
        return np.concatenate(outs).astype(np.float32) - self._wt_score


def get_oracle_scores(label: str, seq: str, ckpt_dir: Path, cache_path: Path):
    L = len(seq)
    if cache_path.exists():
        print(f"  Loading cached oracle scores for {label} …")
        d = np.load(cache_path)
        return d["ssm"].reshape(L, 4), d["dm"].reshape(L, 4, L, 4)

    print(f"\n=== Scoring {label} oracle (L={L}) ===")
    oracle = RBEnsemble(seq, ckpt_dir)
    print(f"  scoring {L*4} single mutants …")
    ssm = oracle.score(make_all_single_muts(seq)).reshape(L, 4)
    print(f"  scoring {L*4*L*4} double mutants …")
    dm = oracle.score(make_all_double_muts(seq)).reshape(L, 4, L, 4)
    np.savez(cache_path, ssm=ssm, dm=dm)
    print(f"  Cached → {cache_path}")
    return ssm, dm


# ── surrogate coef loader ─────────────────────────────────────────────────────

def load_surrogate(coef_path: Path):
    d = np.load(coef_path)
    y_std = float(d["y_std"]) if "y_std" in d else 1.0
    alpha = d["alpha"] * y_std          # (L, 4)
    J = d["J"] * y_std                  # (L, 4, L, 4) -- full pairwise gpmap,
                                         # not restricted to declared stem pairs
    return alpha, J


# ── region annotation helpers ─────────────────────────────────────────────────

def make_dot_bracket(L: int, stem_pairs: list) -> str:
    db = ['.'] * L
    for i, j in stem_pairs:
        db[i] = '('
        db[j] = ')'
    return ''.join(db)


def _contiguous_spans(pos_set):
    if not pos_set:
        return []
    s = sorted(pos_set)
    spans, start, end = [], s[0], s[0]
    for p in s[1:]:
        if p == end + 1:
            end = p
        else:
            spans.append((start, end))
            start = end = p
    spans.append((start, end))
    return spans


def _add_region_boxes(ax, stem_pairs, motif_positions, L):
    """Blue boxes for motif, green boxes for stem on a 4×L SSM/alpha panel."""
    stem_set = set(p for pair in stem_pairs for p in pair)
    mot_set  = set(motif_positions or [])
    for start, end in _contiguous_spans(mot_set):
        ax.add_patch(Rectangle(
            (start - 0.5, -0.5), end - start + 1, 4,
            linewidth=3.5, edgecolor=_MOTIF_COLOR, facecolor='none', zorder=4,
        ))
    for start, end in _contiguous_spans(stem_set):
        ax.add_patch(Rectangle(
            (start - 0.5, -0.5), end - start + 1, 4,
            linewidth=3.5, edgecolor=_BOX_COLOR, facecolor='none', zorder=4,
        ))


def _add_stem_boxes_LxL(ax, stem_pairs):
    """Green 1×1 boxes on declared stem positions in an L×L panel."""
    for (i, j) in stem_pairs:
        ax.add_patch(Rectangle(
            (j - 0.5, i - 0.5), 1, 1,
            linewidth=1.8, edgecolor=_BOX_COLOR, facecolor='none', zorder=5,
        ))


# ── figure builder ────────────────────────────────────────────────────────────

def make_figure(label, seq, stem_pairs, motif_positions, ssm, dm, alpha, J):
    L    = len(seq)
    wt_oh = to_onehot(seq)
    step = max(1, L // 15)

    # ── layout constants ──────────────────────────────────────────────────────
    c4     = max(0.07, min(0.14, 3.5 / L))  # oracle: cell size for 4L axis
    c1     = 4 * c4                           # = 1 position block width

    lm     = 1.05   # left margin (y-labels)
    cbp    = 0.15   # colorbar pad from data
    cbw    = 0.12   # colorbar width
    cgap   = 0.90   # gap between the two data columns (after first cbar)
    rm     = 0.55   # right margin
    tm     = 0.40   # top (room for dot-bracket + title)
    bm     = 0.50   # bottom
    gap    = 0.70   # vertical gap between strip and square panel

    top_h  = max(4 * c4, 1.10)   # strip height
    bot_h  = L * c1               # square panel height (same for both columns)
    data_w = L * c1               # data width (same for both columns)

    # x origins of each column
    x0_ora = lm
    x0_sur = lm + data_w + cbp + cbw + cgap

    total_w = x0_sur + data_w + cbp + cbw + rm
    total_h = tm + top_h + gap + bot_h + bm

    fig = plt.figure(figsize=(total_w, total_h))
    cmap    = plt.cm.RdBu
    cmap_na = cmap.copy()
    cmap_na.set_bad("white")
    cmap_gray = cmap.copy()
    cmap_gray.set_bad("lightgray")

    # ── column axes helper ────────────────────────────────────────────────────
    def _add_column_axes(x0):
        ax_top = fig.add_axes([
            x0 / total_w,
            (bm + bot_h + gap) / total_h,
            data_w / total_w,
            top_h / total_h,
        ])
        ax_bot = fig.add_axes([
            x0 / total_w,
            bm / total_h,
            data_w / total_w,
            bot_h / total_h,
        ])
        cax = fig.add_axes([
            (x0 + data_w + cbp) / total_w,
            bm / total_h,
            cbw / total_w,
            (top_h + gap + bot_h) / total_h,
        ])
        return ax_top, ax_bot, cax

    ax_ssm, ax_epi, cax_ora = _add_column_axes(x0_ora)
    ax_alp, ax_bet, cax_sur = _add_column_axes(x0_sur)

    # ═════════════════════════════════════════════════════════════════════════
    # LEFT COLUMN — oracle
    # ═════════════════════════════════════════════════════════════════════════

    # compute epistasis
    epi = dm - ssm[:, :, None, None] - ssm[None, None, :, :]   # (L,4,L,4)

    # Collapse each (i,j) 4×4 nucleotide block down to its abs-max entry
    # (sign preserved) -- same L×L convention used for the surrogate β panel.
    epi_blocks = epi.transpose(0, 2, 1, 3).reshape(L, L, 16)   # (L,L,16)
    epi_argmax = np.argmax(np.abs(epi_blocks), axis=-1)
    epi_LxL    = np.take_along_axis(epi_blocks, epi_argmax[..., None], axis=-1)[..., 0]

    non_wt_mask = ~wt_oh.astype(bool)
    ssm_vmax    = float(np.abs(ssm[non_wt_mask]).max())
    epi_iu      = epi_LxL[np.triu_indices(L, k=1)]
    epi_vmax    = float(np.nanpercentile(np.abs(epi_iu), 99))
    ora_vmax    = max(ssm_vmax, epi_vmax)
    ora_norm    = TwoSlopeNorm(vmin=-ora_vmax, vcenter=0.0, vmax=ora_vmax)

    # SSM strip (AUGC row order, WT=gray)
    ssm_disp = ssm.T[[0, 3, 2, 1], :]
    wt_disp  = wt_oh.T[[0, 3, 2, 1], :].astype(bool)
    ssm_masked = np.where(wt_disp, np.nan, ssm_disp)

    ax_ssm.imshow(ssm_masked, aspect="auto", cmap=cmap_gray, norm=ora_norm,
                  origin="lower", interpolation="nearest")
    _add_region_boxes(ax_ssm, stem_pairs, motif_positions, L)

    ax_ssm.set_xticks(range(0, L, step)); ax_ssm.set_xticklabels([])
    ax_ssm.tick_params(axis="x", length=0)
    ax_ssm.set_yticks(range(4)); ax_ssm.set_yticklabels(NUCS_AUGC, fontsize=11)
    ax_ssm.set_ylabel("Nucleotide", fontsize=11)
    ax_ssm.set_title(f"{label}  –  ResidualBind oracle", fontsize=13, pad=20)

    # dot-bracket above SSM
    db = make_dot_bracket(L, stem_pairs)
    _DB_COLOR = {'(': _BOX_COLOR, ')': _BOX_COLOR, '.': '#AAAAAA'}
    for i, char in enumerate(db):
        ax_ssm.text(i, 3.65, char, ha='center', va='bottom',
                    fontsize=8, fontfamily='monospace',
                    fontweight='bold' if char != '.' else 'normal',
                    color=_DB_COLOR[char], clip_on=False,
                    transform=ax_ssm.transData)

    motif_p = mpatches.Patch(edgecolor=_MOTIF_COLOR, facecolor='none', linewidth=3.5, label='Motif')
    stem_p  = mpatches.Patch(edgecolor=_BOX_COLOR,   facecolor='none', linewidth=3.5, label='Stem')
    ax_ssm.legend(handles=[motif_p, stem_p], loc='upper right', fontsize=9, framealpha=0.85)

    # Epistasis L×L, abs-max per (i,j) block (upper triangle)
    epi_mat = epi_LxL.copy().astype(float)
    epi_mat[np.tril_indices(L, k=-1)] = np.nan

    ax_epi.imshow(epi_mat, aspect="equal", cmap=cmap_na, norm=ora_norm,
                  origin="upper", interpolation="nearest")
    ax_epi.plot([-0.5, L-0.5], [-0.5, L-0.5], color="#AAAAAA", lw=0.8, zorder=6)
    for k in range(L + 1):
        ax_epi.axvline(k - 0.5, color="#CCCCCC", lw=0.35, zorder=3)
        ax_epi.axhline(k - 0.5, color="#CCCCCC", lw=0.35, zorder=3)
    _add_stem_boxes_LxL(ax_epi, stem_pairs)

    tick_pos  = list(range(0, L, step))
    tick_lbls = [str(p) for p in tick_pos]
    ax_epi.xaxis.tick_top(); ax_epi.xaxis.set_label_position("top")
    ax_epi.set_xticks(tick_pos); ax_epi.set_xticklabels(tick_lbls, fontsize=9)
    ax_epi.set_xlabel("Position j", fontsize=11)
    ax_epi.yaxis.tick_right(); ax_epi.yaxis.set_label_position("right")
    ax_epi.set_yticks(tick_pos); ax_epi.set_yticklabels(tick_lbls, fontsize=9)
    ax_epi.set_ylabel("Position i", fontsize=11)
    ax_epi.set_xlim(-0.5, L - 0.5); ax_epi.set_ylim(L - 0.5, -0.5)

    sm_ora = plt.cm.ScalarMappable(cmap=cmap, norm=ora_norm)
    sm_ora.set_array([])
    cb_ora = fig.colorbar(sm_ora, cax=cax_ora)
    cb_ora.set_label("Δ activity", fontsize=11)
    cb_ora.ax.tick_params(labelsize=9)

    # ═════════════════════════════════════════════════════════════════════════
    # RIGHT COLUMN — surrogate
    # ═════════════════════════════════════════════════════════════════════════

    # alpha strip (AUGC order, WT=gray)
    alp_disp   = alpha.T[[0, 3, 2, 1], :]
    alp_masked = np.where(wt_disp, np.nan, alp_disp)

    # beta L×L, abs-max per (i,j) block over the full pairwise gpmap (not just
    # declared stem pairs) -- same convention as the oracle epistasis panel.
    J_blocks  = J.transpose(0, 2, 1, 3).reshape(L, L, 16)
    J_argmax  = np.argmax(np.abs(J_blocks), axis=-1)
    beta_LxL  = np.take_along_axis(J_blocks, J_argmax[..., None], axis=-1)[..., 0]
    tri_mask = np.tril(np.ones((L, L), dtype=bool))
    beta_masked = np.where(tri_mask, np.nan, beta_LxL)

    # Pairwise (beta) scale is fixed per RBP so this "right panel" pairwise
    # map renders on the same scale as its counterpart in
    # coefficients_groundtruth_vs_surrogate_<rbp>.png (see coef_map_shared.py)
    # -- MSI1 and VTS1 do not need to match each other, just themselves.
    sur_vmax     = right_panel_pairwise_vmax(label)
    sur_norm     = TwoSlopeNorm(vmin=-sur_vmax, vcenter=0.0, vmax=sur_vmax)

    ax_alp.imshow(alp_masked, aspect="auto", cmap=cmap_gray, norm=sur_norm,
                  origin="lower", interpolation="nearest")
    _add_region_boxes(ax_alp, stem_pairs, motif_positions, L)

    ax_alp.set_xticks(range(0, L, step)); ax_alp.set_xticklabels([])
    ax_alp.tick_params(axis="x", length=0)
    ax_alp.set_yticks(range(4)); ax_alp.set_yticklabels(NUCS_AUGC, fontsize=11)
    ax_alp.set_ylabel("Nucleotide", fontsize=11)
    ax_alp.set_title(f"{label}  –  MAVE-NN surrogate (nl+pairwise, 20k, 10%)", fontsize=13, pad=20)

    # dot-bracket above alpha (same sequence, same stem_pairs)
    for i, char in enumerate(db):
        ax_alp.text(i, 3.65, char, ha='center', va='bottom',
                    fontsize=8, fontfamily='monospace',
                    fontweight='bold' if char != '.' else 'normal',
                    color=_DB_COLOR[char], clip_on=False,
                    transform=ax_alp.transData)

    ax_alp.legend(handles=[motif_p, stem_p], loc='upper right', fontsize=9, framealpha=0.85)

    # beta L×L
    ax_bet.imshow(beta_masked, aspect="equal", cmap=cmap_na, norm=sur_norm,
                  origin="upper", interpolation="nearest")
    ax_bet.plot([-0.5, L-0.5], [-0.5, L-0.5], color="#AAAAAA", lw=0.8, zorder=6)
    for k in range(L + 1):
        ax_bet.axvline(k - 0.5, color="#CCCCCC", lw=0.35, zorder=3)
        ax_bet.axhline(k - 0.5, color="#CCCCCC", lw=0.35, zorder=3)
    _add_stem_boxes_LxL(ax_bet, stem_pairs)

    tick_pos  = list(range(0, L, step))
    ax_bet.xaxis.tick_top(); ax_bet.xaxis.set_label_position("top")
    ax_bet.set_xticks(tick_pos); ax_bet.set_xticklabels(tick_lbls, fontsize=9)
    ax_bet.set_xlabel("Position j", fontsize=11)
    ax_bet.yaxis.tick_right(); ax_bet.yaxis.set_label_position("right")
    ax_bet.set_yticks(tick_pos); ax_bet.set_yticklabels(tick_lbls, fontsize=9)
    ax_bet.set_ylabel("Position i", fontsize=11)
    ax_bet.set_xlim(-0.5, L - 0.5); ax_bet.set_ylim(L - 0.5, -0.5)

    sm_sur = plt.cm.ScalarMappable(cmap=cmap, norm=sur_norm)
    sm_sur.set_array([])
    cb_sur = fig.colorbar(sm_sur, cax=cax_sur)
    cb_sur.set_label("Coefficient weight", fontsize=11)
    cb_sur.ax.tick_params(labelsize=9)

    return fig


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    configs = [
        ("MSI1", MSI1_SEQ, MSI1_STEM_PAIRS, MSI1_MOTIF_POSITIONS,
         MSI1_CKPT, MSI1_COEF, MSI1_CACHE),
        ("VTS1", VTS1_SEQ, VTS1_STEM_PAIRS, VTS1_MOTIF_POSITIONS,
         VTS1_CKPT, VTS1_COEF, VTS1_CACHE),
    ]

    for label, seq, stem_pairs, motif_positions, ckpt, coef_path, cache in configs:
        ssm, dm  = get_oracle_scores(label, seq, ckpt, cache)
        alpha, J = load_surrogate(coef_path)

        fig = make_figure(label, seq, stem_pairs, motif_positions,
                          ssm, dm, alpha, J)
        out = OUT_DIR / f"oracle_vs_surrogate_{label.lower()}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved → {out}")
        plt.close(fig)


if __name__ == "__main__":
    main()
