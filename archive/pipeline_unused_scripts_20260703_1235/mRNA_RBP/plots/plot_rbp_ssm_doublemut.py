"""Single-mutant (4×L) + double-mutant (4L×4L upper triangle) delta-activity
heatmaps for MSI1 and VTS1 ResidualBind ensembles, side by side.

Layout mirrors viz.plot_coefficients_4Lx4L: inch-based fixed margins keep
the pairwise panel square and the additive strip proportional.

Usage (toehold_gpu env):
    python mRNA_RBP/plots/plot_rbp_ssm_doublemut.py
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

_NUC_IDX = {"A": 0, "C": 1, "G": 2, "U": 3}
NUCS_AUGC = ["A", "U", "G", "C"]   # display row order (matches viz.py)

MSI1_SEQ             = "AAAAAAAAGCGCAUGCUUGCAUGGCAUGCGCAAAAAAAAAA"
MSI1_STEM_PAIRS      = [(8,30),(9,29),(10,28),(11,27),(12,26),(13,25),(14,24),(15,23)]
MSI1_MOTIF_POSITIONS = [17, 18, 19, 20, 21]
MSI1_CKPT            = _REPO / "rbp" / "results" / "msi1"

VTS1_SEQ             = "AAGGACACUAAGUACAGGUUGCUGGCACAGGGCGCUCAUAA"
VTS1_STEM_PAIRS      = [(16,21),(15,22),(14,23),(25,31),(24,32)]
VTS1_MOTIF_POSITIONS = [20, 21, 22, 23, 24]
VTS1_CKPT            = _REPO / "rbp" / "results" / "vts1"

_MOTIF_COLOR = '#AED6F1'   # soft blue — motif positions
_BOX_COLOR   = '#1E8449'   # dark green — stem boxes

OUT_DIR = _REPO / "mRNA_RBP" / "outputs" / "notebook_plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── helpers ───────────────────────────────────────────────────────────────────

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
    return np.stack(xs)   # (L*4, L, 4)


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
    return np.stack(xs)   # (L*4*L*4, L, 4)


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


# ── scoring ───────────────────────────────────────────────────────────────────

def run_rbp(label, seq, ckpt_dir):
    L = len(seq)
    print(f"\n=== {label} (L={L}) ===")
    oracle = RBEnsemble(seq, ckpt_dir)

    print(f"  scoring {L*4} single mutants …")
    ssm = oracle.score(make_all_single_muts(seq)).reshape(L, 4)

    print(f"  scoring {L*4*L*4} double mutants …")
    dm = oracle.score(make_all_double_muts(seq)).reshape(L, 4, L, 4)

    return ssm, dm


# ── layout constants (mirrors viz.py plot_coefficients_4Lx4L) ─────────────────

def _cell(L):
    return max(0.07, min(0.14, 3.5 / L))    # c4: inches per cell in 4L axis


# ── single-figure two-column plot ─────────────────────────────────────────────

def make_figure(results):
    """results: [(label, seq, stem_pairs, motif_positions, ssm (L,4), dm (L,4,L,4)), ...]"""
    n_cols = len(results)
    L = len(results[0][1])

    c4   = _cell(L)
    c1   = 4 * c4           # one position block width in inches
    lm   = 1.05             # left margin (y-labels)
    rm   = 0.40             # right margin
    tm   = 0.40             # top margin (extra room for dot-bracket text above SSM)
    bm   = 0.50             # bottom margin
    gap  = 0.70             # vertical gap between SSM strip and pairwise
    cgap = 0.80             # horizontal gap between the two RBP columns
    cbw  = 0.12             # colorbar width
    cbp  = 0.50             # colorbar left pad from data edge

    data_w = L * c1         # width of each data column (= 4L * c4)
    bot_h  = L * c1         # height of pairwise panel (square: 4L × 4L)
    top_h  = max(4 * c4, 1.10)   # SSM strip: at least 1.1" so it's visible

    col_w  = data_w + cbp + cbw          # total width per RBP column
    total_w = lm + col_w + cgap + col_w + rm
    total_h = tm + top_h + gap + bot_h + bm

    fig = plt.figure(figsize=(total_w, total_h))
    cmap = plt.cm.RdBu

    for ci, (label, seq, stem_pairs, motif_positions, ssm, dm) in enumerate(results):
        # x0 of data area for this column
        x0 = lm + ci * (col_w + cgap)

        # ── axes in figure-fraction coords ──────────────────────────────────
        ax_ssm = fig.add_axes([
            x0 / total_w,
            (bm + bot_h + gap) / total_h,
            data_w / total_w,
            top_h / total_h,
        ])
        ax_dm = fig.add_axes([
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

        # ── epistasis = double-mutant − additive expectation ─────────────────
        # epi[i,a,j,b] = dm[i,a,j,b] − ssm[i,a] − ssm[j,b]
        epi = dm - ssm[:, :, None, None] - ssm[None, None, :, :]   # (L,4,L,4)

        # ── scale: SSM and epistasis share one symmetric colormap ────────────
        wt_oh = to_onehot(seq)
        non_wt_mask = ~wt_oh.astype(bool)
        ssm_vmax = float(np.abs(ssm[non_wt_mask]).max())
        epi_flat = epi.reshape(L * 4, L * 4)
        epi_vmax = float(np.nanpercentile(np.abs(epi_flat), 99))
        vmax = max(ssm_vmax, epi_vmax)
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

        # ── SSM strip (4 × L), AUGC row order matching viz.py ───────────────
        # ssm is (L, 4) in ACGU order; reorder rows to AUGC = indices [0,3,2,1]
        ssm_disp = ssm.T[[0, 3, 2, 1], :]       # (4, L)
        wt_mask_disp = wt_oh.T[[0, 3, 2, 1], :].astype(bool)
        ssm_masked = np.where(wt_mask_disp, np.nan, ssm_disp)
        cmap_ssm = cmap.copy(); cmap_ssm.set_bad("lightgray")
        ax_ssm.imshow(ssm_masked, aspect="auto", cmap=cmap_ssm, norm=norm,
                      origin="lower", interpolation="nearest")

        # ── stem/motif region boxes on SSM strip ────────────────────────────
        stem_set = set(p for pair in stem_pairs for p in pair)
        mot_set  = set(motif_positions or [])
        for start, end in _contiguous_spans(mot_set):
            ax_ssm.add_patch(Rectangle(
                (start - 0.5, -0.5), end - start + 1, 4,
                linewidth=3.5, edgecolor=_MOTIF_COLOR, facecolor='none', zorder=4,
            ))
        for start, end in _contiguous_spans(stem_set):
            ax_ssm.add_patch(Rectangle(
                (start - 0.5, -0.5), end - start + 1, 4,
                linewidth=3.5, edgecolor=_BOX_COLOR, facecolor='none', zorder=4,
            ))

        step = max(1, L // 15)
        ax_ssm.set_xticks(range(0, L, step))
        ax_ssm.set_xticklabels([])           # pairwise panel below carries position labels
        ax_ssm.set_yticks(range(4))
        ax_ssm.set_yticklabels(NUCS_AUGC, fontsize=11)
        ax_ssm.tick_params(axis="x", length=0)
        if ci == 0:
            ax_ssm.set_ylabel("Nucleotide", fontsize=11)
        ax_ssm.set_title(f"{label}", fontsize=13, pad=20)

        motif_p = mpatches.Patch(edgecolor=_MOTIF_COLOR, facecolor='none', linewidth=3.5, label='Motif')
        stem_p  = mpatches.Patch(edgecolor=_BOX_COLOR,   facecolor='none', linewidth=3.5, label='Stem')
        ax_ssm.legend(handles=[motif_p, stem_p], loc='upper right',
                      fontsize=9, framealpha=0.85)

        # ── dot-bracket annotation above SSM strip ───────────────────────────
        db = make_dot_bracket(L, stem_pairs)
        _DB_COLOR = {'(': '#1E8449', ')': '#1E8449', '.': '#AAAAAA'}
        for i, char in enumerate(db):
            ax_ssm.text(
                i, 3.65, char,
                ha='center', va='bottom',
                fontsize=8, fontfamily='monospace',
                fontweight='bold' if char != '.' else 'normal',
                color=_DB_COLOR[char],
                clip_on=False,
                transform=ax_ssm.transData,
            )

        # ── epistasis upper triangle (4L × 4L) ──────────────────────────────
        mat = epi_flat.copy().astype(float)
        mat[np.tril_indices(L * 4, k=-1)] = np.nan
        cmap_dm = cmap.copy(); cmap_dm.set_bad("white")
        ax_dm.imshow(mat, aspect="equal", cmap=cmap_dm, norm=norm,
                     origin="upper", interpolation="nearest")

        # diagonal guide line
        ax_dm.plot([-0.5, L * 4 - 0.5], [-0.5, L * 4 - 0.5],
                   color="#AAAAAA", lw=0.8, zorder=6)

        # green 4×4 boxes on declared stem-pair blocks
        for (i, j) in stem_pairs:
            ax_dm.add_patch(Rectangle(
                (4 * j - 0.5, 4 * i - 0.5), 4, 4,
                linewidth=1.8, edgecolor=_BOX_COLOR, facecolor='none', zorder=5,
            ))

        # block grid lines
        for k in range(L + 1):
            ax_dm.axvline(4 * k - 0.5, color="#CCCCCC", lw=0.35, zorder=3)
            ax_dm.axhline(4 * k - 0.5, color="#CCCCCC", lw=0.35, zorder=3)

        # position ticks at block centers
        tick_pos  = list(range(0, L, step))
        tick_cell = [4 * p + 1.5 for p in tick_pos]
        ax_dm.xaxis.tick_top()
        ax_dm.xaxis.set_label_position("top")
        ax_dm.set_xticks(tick_cell)
        ax_dm.set_xticklabels([str(p) for p in tick_pos], fontsize=9)
        ax_dm.set_xlabel("Position j", fontsize=11)
        ax_dm.yaxis.tick_right()
        ax_dm.yaxis.set_label_position("right")
        ax_dm.set_yticks(tick_cell)
        ax_dm.set_yticklabels([str(p) for p in tick_pos], fontsize=9)
        ax_dm.set_ylabel("Position i", fontsize=11)
        ax_dm.set_xlim(-0.5, L * 4 - 0.5)
        ax_dm.set_ylim(L * 4 - 0.5, -0.5)

        # ── colorbar (spans both panels) ──────────────────────────────────
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cb = fig.colorbar(sm, cax=cax)
        cb.set_label("Δ activity", fontsize=11)
        cb.ax.tick_params(labelsize=9)

    return fig


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    configs = [
        ("MSI1", MSI1_SEQ, MSI1_STEM_PAIRS, MSI1_MOTIF_POSITIONS, MSI1_CKPT),
        ("VTS1", VTS1_SEQ, VTS1_STEM_PAIRS, VTS1_MOTIF_POSITIONS, VTS1_CKPT),
    ]

    results = []
    for label, seq, stem_pairs, motif_positions, ckpt in configs:
        ssm, dm = run_rbp(label, seq, ckpt)
        results.append((label, seq, stem_pairs, motif_positions, ssm, dm))

    fig = make_figure(results)
    out_path = OUT_DIR / "rbp_ssm_doublemut.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
