"""
mRNA_RBP/viz.py

Visualization utilities for MrnaRbpGroundTruth coefficients and libraries,
mirroring the src.visualize interface from demo_additive_pairwise (3).ipynb.

Usage
-----
    from mRNA_RBP import viz
    import matplotlib.pyplot as plt

    fig = viz.plot_coefficients(
        gt.alpha, gt.beta, L,
        stem_pairs=gt.stem_pairs,
        motif_positions=gt.motif_positions,
        title='Ground truth coefficients',
    )
    plt.show()
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from matplotlib.colors import TwoSlopeNorm

NUCS = ['A', 'U', 'G', 'C']

# Region shading colours (match demo notebook: blue=motif, green=stem)
_MOTIF_COLOR = '#AED6F1'   # soft blue  — motif positions
_STEM_COLOR  = '#A9DFBF'   # soft green — stem positions
_BOX_COLOR   = '#1E8449'   # dark green rectangle outline for pairwise blocks
_GRAY_COLOR  = '#D3D3D3'   # light gray for masked lower triangle


# ---------------------------------------------------------------------------
# plot_coefficients
# ---------------------------------------------------------------------------

def plot_coefficients(
    alpha,
    beta,
    L: int,
    *,
    stem_pairs=None,
    motif_positions=None,
    title: str = 'Coefficients',
    figsize=None,
    vmax: float = None,
) -> plt.Figure:
    """
    Two-panel coefficient figure matching demo_additive_pairwise notebook style.

    Top panel  — (4 × L) additive weight matrix α
        Blue column shading  = motif positions
        Green column shading = stem positions
        Diverging RdBu colormap centered at 0

    Bottom panel — (4L × 4L) pairwise weight matrix assembled from β
        Upper triangle only (lower triangle grayed out).
        Green rectangles outline declared stem-pair blocks.
        Shared colorbar labeled 'Coefficient weight'.

    Parameters
    ----------
    alpha           : (L, 4) additive weight matrix  (gt.alpha)
    beta            : {(i, j): (4, 4) array}          (gt.beta)
    L               : sequence length
    stem_pairs      : list of (i, j) pairs — shaded green + boxed in pairwise panel
    motif_positions : list of position indices — shaded blue in additive panel
    title           : overall figure suptitle
    figsize         : (width, height) in inches; auto-sized if None
    vmax            : symmetric colormap range [-vmax, +vmax]
    """
    alpha    = np.asarray(alpha, dtype=np.float32)       # (L, 4)
    stem_set = set(p for pair in (stem_pairs or []) for p in pair)
    mot_set  = set(motif_positions or [])

    # ── Assemble L × L pairwise matrix (abs-max with sign per position pair) ──
    beta_pos = np.zeros((L, L), dtype=np.float32)
    for (i, j), M in (beta or {}).items():
        M = np.asarray(M)
        flat = M.ravel()
        idx = np.argmax(np.abs(flat))
        beta_pos[i, j] = flat[idx]

    # Mask lower triangle (i >= j)
    tri_mask = np.tril(np.ones((L, L), dtype=bool))
    beta_masked = np.where(tri_mask, np.nan, beta_pos)

    if vmax is None:
        vals = beta_pos[~tri_mask] if np.any(~tri_mask) else np.array([0.0])
        vmax = max(float(np.abs(vals).max()), 1e-6)

    # Fixed margins (inches); both panels share identical x0 + width by construction
    c      = max(0.18, min(0.28, 8.0 / L))   # cell size in inches
    lm     = 1.05   # left  (ax0 y-labels + tick labels)
    # right: ax1 tick marks + tick labels + ylabel + gap + colorbar + cbar tick labels
    rm     = 1.20
    tm     = 0.15   # top margin
    bm     = 0.50   # bottom (ax1 x-label no longer here; ax0 bottom label)
    gap    = 0.80   # vertical gap (ax0 bottom ticks + ax1 top ticks both fit)

    data_w  = L * c
    bot_h   = L * c
    top_h   = 4 * c
    total_w = lm + data_w + rm
    total_h = tm + top_h + gap + bot_h + bm

    fig = plt.figure(figsize=(total_w, total_h))

    ax0 = fig.add_axes([lm / total_w,
                        (bm + bot_h + gap) / total_h,
                        data_w / total_w,
                        top_h / total_h])
    ax1 = fig.add_axes([lm / total_w,
                        bm / total_h,
                        data_w / total_w,
                        bot_h / total_h])
    # Colorbar sits right of ax1's y-tick labels and ylabel
    cax = fig.add_axes([(lm + data_w + 0.70) / total_w,
                        bm / total_h,
                        0.12 / total_w,
                        (top_h + gap + bot_h) / total_h])

    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = plt.cm.RdBu

    # ── Top: α ───────────────────────────────────────────────────────────
    # alpha.T shape (4, L): rows = AUGC, cols = positions
    disp = alpha.T[[0, 3, 2, 1], :]   # reorder to AUGC row order (A=0,U=3,G=2,C=1 in ACGU)

    im0 = ax0.imshow(
        disp, aspect='auto', cmap=cmap, norm=norm,
        origin='lower', interpolation='nearest', zorder=1,
    )

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

    for start, end in _contiguous_spans(mot_set):
        ax0.add_patch(Rectangle(
            (start - 0.5, -0.5), end - start + 1, 4,
            linewidth=3.5, edgecolor=_MOTIF_COLOR, facecolor='none', zorder=4,
        ))
    for start, end in _contiguous_spans(stem_set):
        ax0.add_patch(Rectangle(
            (start - 0.5, -0.5), end - start + 1, 4,
            linewidth=3.5, edgecolor=_BOX_COLOR, facecolor='none', zorder=4,
        ))

    step = max(1, L // 15)
    ax0.set_xticks(range(0, L, step))
    ax0.set_xticklabels([str(i) for i in range(0, L, step)], fontsize=8)
    ax0.set_yticks(range(4))
    ax0.set_yticklabels(NUCS, fontsize=9)
    ax0.set_xlabel('Position', fontsize=9)
    ax0.set_ylabel('Nucleotide', fontsize=9)

    motif_p = mpatches.Patch(edgecolor=_MOTIF_COLOR, facecolor='none', linewidth=3.5, label='Motif')
    stem_p  = mpatches.Patch(edgecolor=_BOX_COLOR,  facecolor='none', linewidth=3.5, label='Stem')
    ax0.legend(handles=[motif_p, stem_p], loc='upper right',
               fontsize=8, framealpha=0.85)

    # ── Bottom: β (upper triangle only, L×L) ─────────────────────────────
    cmap_pair = cmap.copy()
    cmap_pair.set_bad(color='white')
    ax1.set_facecolor('white')

    im1 = ax1.imshow(
        beta_masked, aspect='auto', cmap=cmap_pair, norm=norm,
        origin='upper', interpolation='nearest',
    )

    # Diagonal boundary line
    ax1.plot([-0.5, L - 0.5], [-0.5, L - 0.5],
             color='#AAAAAA', linewidth=0.8, zorder=6)

    # Green squares on declared stem pairs
    for (i, j) in (stem_pairs or []):
        ax1.add_patch(Rectangle(
            (j - 0.5, i - 0.5), 1, 1,
            linewidth=1.8, edgecolor=_BOX_COLOR,
            facecolor='none', zorder=4,
        ))

    step = max(1, L // 15)
    tick_p = range(0, L, step)
    ax1.xaxis.tick_top()
    ax1.xaxis.set_label_position('top')
    ax1.set_xticks(list(tick_p))
    ax1.set_xticklabels([str(p) for p in tick_p], fontsize=8)
    ax1.set_xlabel('Position j', fontsize=9)
    ax1.yaxis.tick_right()
    ax1.yaxis.set_label_position('right')
    ax1.set_yticks(list(tick_p))
    ax1.set_yticklabels([str(p) for p in tick_p], fontsize=8)
    ax1.set_ylabel('Position i', fontsize=9)
    # ── Shared colorbar ───────────────────────────────────────────────────
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label('Coefficient weight', fontsize=9)
    cb.ax.tick_params(labelsize=8)

    return fig


# ---------------------------------------------------------------------------
# plot_coefficients_4Lx4L  — full 4L×4L pairwise panel
# ---------------------------------------------------------------------------

def plot_coefficients_4Lx4L(
    alpha,
    beta,
    L: int,
    *,
    stem_pairs=None,
    motif_positions=None,
    title: str = 'Coefficients',
    figsize=None,
    vmax: float = None,
) -> plt.Figure:
    """
    Two-panel coefficient figure with a full (4L × 4L) pairwise weight matrix.

    Top panel  — (4 × L) additive weight matrix α  (identical to plot_coefficients)
    Bottom panel — (4L × 4L) pairwise weight matrix with the full 4×4 nucleotide
                   coupling blocks at each position pair.  Upper triangle only.
                   Nucleotide ordering within each block: A, C, G, U.
                   Block separators (thin gray lines) mark position boundaries.

    Parameters
    ----------
    alpha           : (L, 4) additive weight matrix  (gt.alpha)
    beta            : {(i, j): (4, 4) array}          (gt.beta)
    L               : sequence length
    stem_pairs      : list of (i, j) — green box outlines on pairwise panel
    motif_positions : list of position indices — blue shading in additive panel
    title           : overall figure suptitle
    figsize         : (width, height) in inches; auto-sized if None
    vmax            : symmetric colormap range [-vmax, +vmax]
    """
    alpha    = np.asarray(alpha, dtype=np.float32)       # (L, 4)
    stem_set = set(p for pair in (stem_pairs or []) for p in pair)
    mot_set  = set(motif_positions or [])

    # ── Assemble 4L × 4L pairwise matrix ─────────────────────────────────
    beta_4L = np.full((4 * L, 4 * L), np.nan, dtype=np.float32)
    for (i, j), M in (beta or {}).items():
        M = np.asarray(M, dtype=np.float32)
        if i < j:                              # upper triangle only
            beta_4L[4 * i: 4 * i + 4, 4 * j: 4 * j + 4] = M

    if vmax is None:
        finite = beta_4L[np.isfinite(beta_4L)]
        vmax = max(float(np.abs(finite).max()) if len(finite) else 0.0, 1e-6)

    # ── Figure layout (fixed margins in inches) ───────────────────────────
    c4    = max(0.045, min(0.08, 2.0 / L))   # cell size for 4L axis
    c1    = 4 * c4                            # block size = one position width
    lm    = 1.05
    rm    = 1.20
    tm    = 0.15
    bm    = 0.50
    gap   = 0.80

    data_w  = L * c1   # 4L cells wide = L position blocks × c1
    bot_h   = L * c1   # square 4L×4L panel
    top_h   = 4 * c4   # 4 nucleotide rows, each c4 tall

    total_w = lm + data_w + rm
    total_h = tm + top_h + gap + bot_h + bm

    fig = plt.figure(figsize=(total_w, total_h))

    ax0 = fig.add_axes([lm / total_w,
                        (bm + bot_h + gap) / total_h,
                        data_w / total_w,
                        top_h / total_h])
    ax1 = fig.add_axes([lm / total_w,
                        bm / total_h,
                        data_w / total_w,
                        bot_h / total_h])
    cax = fig.add_axes([(lm + data_w + 0.70) / total_w,
                        bm / total_h,
                        0.12 / total_w,
                        (top_h + gap + bot_h) / total_h])

    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = plt.cm.RdBu

    # ── Top: α (identical to plot_coefficients) ───────────────────────────
    disp = alpha.T[[0, 3, 2, 1], :]   # AUGC row order

    ax0.imshow(disp, aspect='auto', cmap=cmap, norm=norm,
               origin='lower', interpolation='nearest', zorder=1)

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

    for start, end in _contiguous_spans(mot_set):
        ax0.add_patch(Rectangle(
            (start - 0.5, -0.5), end - start + 1, 4,
            linewidth=3.5, edgecolor=_MOTIF_COLOR, facecolor='none', zorder=4,
        ))
    for start, end in _contiguous_spans(stem_set):
        ax0.add_patch(Rectangle(
            (start - 0.5, -0.5), end - start + 1, 4,
            linewidth=3.5, edgecolor=_BOX_COLOR, facecolor='none', zorder=4,
        ))

    step = max(1, L // 15)
    ax0.set_xticks(range(0, L, step))
    ax0.set_xticklabels([str(i) for i in range(0, L, step)], fontsize=8)
    ax0.set_yticks(range(4))
    ax0.set_yticklabels(NUCS, fontsize=9)
    ax0.set_xlabel('Position', fontsize=9)
    ax0.set_ylabel('Nucleotide', fontsize=9)

    motif_p = mpatches.Patch(edgecolor=_MOTIF_COLOR, facecolor='none', linewidth=3.5, label='Motif')
    stem_p  = mpatches.Patch(edgecolor=_BOX_COLOR,  facecolor='none', linewidth=3.5, label='Stem')
    ax0.legend(handles=[motif_p, stem_p], loc='upper right', fontsize=8, framealpha=0.85)

    # ── Bottom: β full 4L×4L ─────────────────────────────────────────────
    cmap_pair = cmap.copy()
    cmap_pair.set_bad(color='white')
    ax1.set_facecolor('white')

    ax1.imshow(beta_4L, aspect='equal', cmap=cmap_pair, norm=norm,
               origin='upper', interpolation='nearest')

    # Block separator grid lines at every 4 cells (position boundaries)
    grid_kw = dict(color='#CCCCCC', linewidth=0.35, zorder=3)
    for k in range(L + 1):
        ax1.axhline(4 * k - 0.5, **grid_kw)
        ax1.axvline(4 * k - 0.5, **grid_kw)

    # Green rectangles around stem-pair blocks (4×4 cells each)
    for (i, j) in (stem_pairs or []):
        ax1.add_patch(Rectangle(
            (4 * j - 0.5, 4 * i - 0.5), 4, 4,
            linewidth=1.8, edgecolor=_BOX_COLOR, facecolor='none', zorder=4,
        ))

    # Diagonal block boundary line
    ax1.plot([-0.5, 4 * L - 0.5], [-0.5, 4 * L - 0.5],
             color='#AAAAAA', linewidth=0.8, zorder=6)

    # Position ticks: center of each 4-cell block
    tick_pos  = range(0, L, max(1, L // 15))
    tick_cell = [4 * p + 1.5 for p in tick_pos]   # center of block
    ax1.xaxis.tick_top()
    ax1.xaxis.set_label_position('top')
    ax1.set_xticks(tick_cell)
    ax1.set_xticklabels([str(p) for p in tick_pos], fontsize=8)
    ax1.set_xlabel('Position j', fontsize=9)
    ax1.yaxis.tick_right()
    ax1.yaxis.set_label_position('right')
    ax1.set_yticks(tick_cell)
    ax1.set_yticklabels([str(p) for p in tick_pos], fontsize=8)
    ax1.set_ylabel('Position i', fontsize=9)
    ax1.set_xlim(-0.5, 4 * L - 0.5)
    ax1.set_ylim(4 * L - 0.5, -0.5)

    # ── Shared colorbar ───────────────────────────────────────────────────
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label('Coefficient weight', fontsize=9)
    cb.ax.tick_params(labelsize=8)

    return fig


# ---------------------------------------------------------------------------
# plot_coefficients_stem_compact  — i-rows only, full-width x-axis
# ---------------------------------------------------------------------------

def plot_coefficients_stem_compact(
    alpha,
    beta,
    L: int,
    *,
    stem_pairs=None,
    motif_positions=None,
    title: str = 'Coefficients',
    vmax: float = None,
) -> plt.Figure:
    """
    Compact two-panel figure: full 4×L additive panel on top, and a stem-only
    pairwise panel below that shares the same x-axis.

    Top panel  — (4 × L) additive weight matrix α  (identical to plot_coefficients)

    Bottom panel — rows: i-side stem positions only (n_i × 4 nucleotide rows)
                   cols: full 4L-wide axis, matching the top panel
                   Each stem pair (i, j) places its 4×4 β block at the correct
                   column group j, so the block aligns visually with position j
                   in the additive panel above.

    Nucleotide ordering within blocks: A, C, G, U (ACGU, index 0–3).

    Parameters
    ----------
    alpha           : (L, 4) additive weight matrix
    beta            : {(i, j): (4, 4) array}
    L               : sequence length
    stem_pairs      : list of (i, j) tuples
    motif_positions : list of position indices — blue shading in additive panel
    vmax            : symmetric colormap range [-vmax, +vmax]
    """
    alpha     = np.asarray(alpha, dtype=np.float32)
    stem_pairs = list(stem_pairs or [])
    mot_set    = set(motif_positions or [])

    # i-side positions in sorted order → each gets 4 rows in the bottom panel
    i_positions = sorted({i for i, j in stem_pairs})
    n_i         = len(i_positions)
    i_row_map   = {pos: r for r, pos in enumerate(i_positions)}

    # ── Assemble sparse (4*n_i) × (4L) matrix ────────────────────────────
    mat = np.full((4 * n_i, 4 * L), np.nan, dtype=np.float32)
    for (i, j), M in (beta or {}).items():
        if i not in i_row_map:
            continue
        row0 = 4 * i_row_map[i]
        col0 = 4 * j
        mat[row0: row0 + 4, col0: col0 + 4] = np.asarray(M, dtype=np.float32)

    if vmax is None:
        finite = mat[np.isfinite(mat)]
        vmax = max(float(np.abs(finite).max()) if len(finite) else 0.0, 1e-6)

    # ── Layout (inches) ───────────────────────────────────────────────────
    c4    = max(0.045, min(0.08, 2.0 / L))
    c1    = 4 * c4
    lm    = 1.05
    rm    = 1.20
    tm    = 0.15
    bm    = 0.50
    gap   = 0.80

    data_w  = L * c1
    top_h   = 4 * c4            # additive panel: 4 nuc rows
    bot_h   = n_i * c1          # pairwise panel: n_i position blocks × c1
    total_w = lm + data_w + rm
    total_h = tm + top_h + gap + bot_h + bm

    fig = plt.figure(figsize=(total_w, total_h))

    ax0 = fig.add_axes([lm / total_w,
                        (bm + bot_h + gap) / total_h,
                        data_w / total_w,
                        top_h / total_h])
    ax1 = fig.add_axes([lm / total_w,
                        bm / total_h,
                        data_w / total_w,
                        bot_h / total_h])
    cax = fig.add_axes([(lm + data_w + 0.70) / total_w,
                        bm / total_h,
                        0.12 / total_w,
                        (top_h + gap + bot_h) / total_h])

    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = plt.cm.RdBu

    # ── Top: α ───────────────────────────────────────────────────────────
    disp = alpha.T[[0, 3, 2, 1], :]   # AUGC display order
    ax0.imshow(disp, aspect='auto', cmap=cmap, norm=norm,
               origin='lower', interpolation='nearest', zorder=1)

    for pos in range(L):
        if pos in mot_set:
            ax0.axvspan(pos - 0.5, pos + 0.5, color=_MOTIF_COLOR, alpha=0.45, zorder=2)
        if any(pos == i or pos == j for i, j in stem_pairs):
            ax0.axvspan(pos - 0.5, pos + 0.5, color=_STEM_COLOR,  alpha=0.45, zorder=2)

    step = max(1, L // 15)
    ax0.set_xticks(range(0, L, step))
    ax0.set_xticklabels([str(p) for p in range(0, L, step)], fontsize=8)
    ax0.set_yticks(range(4))
    ax0.set_yticklabels(NUCS, fontsize=9)
    ax0.set_xlabel('Position', fontsize=9)
    ax0.set_ylabel('Nucleotide', fontsize=9)

    motif_p = mpatches.Patch(color=_MOTIF_COLOR, alpha=0.9, label='Motif')
    stem_p  = mpatches.Patch(color=_STEM_COLOR,  alpha=0.9, label='Stem')
    ax0.legend(handles=[motif_p, stem_p], loc='upper right', fontsize=8, framealpha=0.85)

    # ── Bottom: stem pairwise (i-rows, full-width cols) ───────────────────
    cmap_pair = cmap.copy()
    cmap_pair.set_bad(color='white')
    ax1.set_facecolor('white')

    ax1.imshow(mat, aspect='auto', cmap=cmap_pair, norm=norm,
               origin='upper', interpolation='nearest',
               extent=[-0.5, 4 * L - 0.5, n_i * 4 - 0.5, -0.5])

    # Vertical block separators (every 4 cells = one position)
    grid_kw = dict(color='#CCCCCC', linewidth=0.35, zorder=3)
    for k in range(L + 1):
        ax1.axvline(4 * k - 0.5, **grid_kw)
    # Horizontal separators at i-position boundaries
    for r in range(n_i + 1):
        ax1.axhline(4 * r - 0.5, **grid_kw)

    # Green rectangles around stem-pair blocks
    for (i, j) in stem_pairs:
        if i not in i_row_map:
            continue
        r = i_row_map[i]
        ax1.add_patch(Rectangle(
            (4 * j - 0.5, 4 * r - 0.5), 4, 4,
            linewidth=1.8, edgecolor=_BOX_COLOR, facecolor='none', zorder=4,
        ))

    # x-axis ticks: center of each position block (aligned with top panel)
    tick_pos  = list(range(0, L, step))
    tick_cell = [4 * p + 1.5 for p in tick_pos]
    ax1.xaxis.tick_top()
    ax1.xaxis.set_label_position('top')
    ax1.set_xticks(tick_cell)
    ax1.set_xticklabels([str(p) for p in tick_pos], fontsize=8)
    ax1.set_xlabel('Position j', fontsize=9)

    # y-axis ticks: center of each i-position block, labeled by position index
    ax1.yaxis.tick_right()
    ax1.yaxis.set_label_position('right')
    ax1.set_yticks([4 * r + 1.5 for r in range(n_i)])
    ax1.set_yticklabels([str(p) for p in i_positions], fontsize=8)
    ax1.set_ylabel('Position i', fontsize=9)

    ax1.set_xlim(-0.5, 4 * L - 0.5)
    ax1.set_ylim(n_i * 4 - 0.5, -0.5)

    # ── Shared colorbar ───────────────────────────────────────────────────
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label('Coefficient weight', fontsize=9)
    cb.ax.tick_params(labelsize=8)

    return fig


# ---------------------------------------------------------------------------
# plot_library  (mirrors src.library.plot_library from notebook)
# ---------------------------------------------------------------------------

def plot_stem_blocks(gt_beta, sur_beta, stem_pairs, *, figsize=None) -> plt.Figure:
    """Side-by-side 4×4 coupling matrices for GT vs surrogate, one column per stem pair.

    Parameters
    ----------
    gt_beta    : {(i, j): (4, 4)} GT coupling matrices  (gt.beta)
    sur_beta   : {(i, j): (4, 4)} surrogate coupling matrices
    stem_pairs : list of (i, j)
    """
    NUCS_LABEL = ['A', 'C', 'G', 'U']
    n = len(stem_pairs)
    if figsize is None:
        figsize = (max(8, n * 2.8), 5.5)

    fig, axes = plt.subplots(2, n, figsize=figsize,
                             gridspec_kw={'hspace': 0.4, 'wspace': 0.25})
    if n == 1:
        axes = axes.reshape(2, 1)

    all_vals = []
    for pair in stem_pairs:
        if pair in gt_beta:
            all_vals.extend(np.asarray(gt_beta[pair]).ravel())
        if pair in sur_beta:
            all_vals.extend(np.asarray(sur_beta[pair]).ravel())
    vmax = max(float(np.abs(all_vals).max()) if all_vals else 1.0, 1e-6)
    norm = plt.Normalize(vmin=-vmax, vmax=vmax)
    cmap = plt.cm.RdBu

    for col, (i, j) in enumerate(stem_pairs):
        gt_M  = np.asarray(gt_beta.get((i, j),  np.zeros((4, 4))), dtype=float)
        sur_M = np.asarray(sur_beta.get((i, j), np.zeros((4, 4))), dtype=float)

        for row, (M, label) in enumerate([(gt_M, 'GT'), (sur_M, 'Surrogate')]):
            ax = axes[row, col]
            ax.imshow(M, cmap=cmap, norm=norm, aspect='equal',
                      origin='upper', interpolation='nearest')
            ax.set_xticks(range(4)); ax.set_xticklabels(NUCS_LABEL, fontsize=7)
            ax.set_yticks(range(4)); ax.set_yticklabels(NUCS_LABEL, fontsize=7)
            if col == 0:
                ax.set_ylabel(f'{label}\nnuc i', fontsize=8)
            if row == 0:
                pass
            if row == 1:
                ax.set_xlabel('nuc j', fontsize=7)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.015, pad=0.02,
                 label='Coupling weight')
    return fig


def plot_gt_vs_surrogate(gt, sur_alpha, sur_beta, L, *,
                         stem_pairs=None, motif_positions=None) -> tuple:
    """Two coefficient figures: GT (left) and surrogate (right).

    Returns (fig_gt, fig_sur) — two separate figures so they can be saved
    independently or displayed side-by-side.

    Parameters
    ----------
    gt              : MrnaRbpGroundTruth instance
    sur_alpha       : (L, 4) surrogate additive weights (from MAVE-NN params[1])
    sur_beta        : {(i,j): (4,4)} surrogate pairwise weights
    L               : sequence length
    stem_pairs      : passed to both plots; defaults to gt.stem_pairs
    motif_positions : defaults to gt.motif_positions
    """
    sp = stem_pairs      if stem_pairs      is not None else gt.stem_pairs
    mp = motif_positions if motif_positions is not None else gt.motif_positions

    fig_gt  = plot_coefficients(gt.alpha, gt.beta,  L,
                                stem_pairs=sp, motif_positions=mp,
                                title='Ground truth coefficients')
    fig_sur = plot_coefficients(sur_alpha,  sur_beta, L,
                                stem_pairs=sp, motif_positions=mp,
                                title='Surrogate (learned) coefficients')
    return fig_gt, fig_sur


def plot_additive_eval(y_gt, y_sur, k_labels, *, y_ssm=None,
                       title='Additive evaluation') -> plt.Figure:
    """Scatter GT vs surrogate activity for k-mutant non-stem sequences.

    One subplot per unique k value.  y_ssm (SSM baseline) is overlaid when
    provided.

    Parameters
    ----------
    y_gt      : (N,) GT activity scores
    y_sur     : (N,) surrogate predicted scores
    k_labels  : (N,) int  mutation count per sequence
    y_ssm     : (N,) optional SSM baseline
    """
    ks    = sorted(set(k_labels))
    ncols = len(ks)
    fig, axes = plt.subplots(1, ncols, figsize=(4.5 * ncols, 4),
                             sharey=True, gridspec_kw={'wspace': 0.08})
    if ncols == 1:
        axes = [axes]

    for ax, k in zip(axes, ks):
        mask = k_labels == k
        yg, ys = y_gt[mask], y_sur[mask]
        ax.scatter(yg, ys, s=10, alpha=0.55, color='#4C72B0', label='Surrogate')
        if y_ssm is not None:
            ax.scatter(yg, y_ssm[mask], s=10, alpha=0.45, color='#DD8452',
                       marker='^', label='SSM baseline')
        lo = min(yg.min(), ys.min())
        hi = max(yg.max(), ys.max())
        ax.plot([lo, hi], [lo, hi], 'k--', lw=0.9, label='y = ŷ')
        r2 = float(np.corrcoef(yg, ys)[0, 1] ** 2)
        ax.set_xlabel('GT activity', fontsize=9)
        if ax is axes[0]:
            ax.set_ylabel('Predicted activity', fontsize=9)
        ax.legend(fontsize=7, loc='upper left')

    fig.tight_layout()
    return fig


def plot_pairwise_eval(y_gt, y_sur, pair_labels, nuc_combos, stem_pairs, *,
                       y_ssm=None, title='Pairwise evaluation',
                       max_pairs=None, show_title=True,
                       sur_label='Surrogate', ssm_label='SSM baseline',
                       per_col_norm=False, wt_nuc_ids=None) -> plt.Figure:
    """4×4 activity heatmaps for each stem pair: GT, surrogate (and SSM baseline).

    Each heatmap shows activity for all 16 nucleotide combinations at the stem
    pair (other positions fixed at WT).

    Parameters
    ----------
    y_gt         : (N,) GT activity scores   (from make_pairwise_dataset)
    y_sur        : (N,) surrogate scores
    pair_labels  : (N,) int  index into stem_pairs
    nuc_combos   : list of (nuc_i, nuc_j) tuples
    stem_pairs   : list of (i, j)
    y_ssm        : (N,) optional SSM baseline
    max_pairs    : int or None — only plot the first max_pairs rows
    show_title   : bool — whether to display the suptitle
    sur_label    : str — column header for the surrogate column
    ssm_label    : str — column header for the SSM baseline column
    per_col_norm : bool — normalize each column independently (stretches contrast)
    wt_nuc_ids   : (L,) int array of WT nucleotide indices — draws a black box
                   around the WT cell in every heatmap when provided
    """
    import matplotlib.patches as mpatches
    NUCS_LABEL = ['A', 'C', 'G', 'U']
    n_pairs = len(stem_pairs)
    if max_pairs is not None:
        n_pairs = min(n_pairs, max_pairs)

    ncols = 3 if y_ssm is not None else 2
    fig, axes = plt.subplots(n_pairs, ncols,
                             figsize=(ncols * 3.2, n_pairs * 3.0),
                             gridspec_kw={'hspace': 0.5, 'wspace': 0.3})
    if n_pairs == 1:
        axes = axes.reshape(1, ncols)

    col_titles = ['GT', sur_label] + ([ssm_label] if y_ssm is not None else [])
    col_data   = [y_gt, y_sur] + ([y_ssm] if y_ssm is not None else [])
    cmap = plt.cm.Blues_r

    if per_col_norm:
        col_vabs = [max(float(np.abs(yd).max()), 1e-6) for yd in col_data]
        col_norms = [plt.Normalize(vmin=-1, vmax=0)] * ncols
    else:
        all_y = np.concatenate([np.asarray(yd).ravel() for yd in col_data])
        vabs  = max(float(np.abs(all_y).max()), 1e-6)
        shared_norm = plt.Normalize(vmin=-vabs, vmax=0)
        col_norms = [shared_norm] * ncols

    nuc_combos = list(nuc_combos)
    for pidx in range(n_pairs):
        mask = pair_labels == pidx
        combos_p = [nuc_combos[i] for i in np.where(mask)[0]]
        yg_p   = y_gt[mask]
        ys_p   = y_sur[mask]
        yssm_p = y_ssm[mask] if y_ssm is not None else None

        def _to_mat(vals):
            M = np.zeros((4, 4))
            for (ni, nj), v in zip(combos_p, vals):
                M[ni, nj] = v
            return M

        per_pair_data = [yg_p, ys_p] + ([yssm_p] if yssm_p is not None else [])
        if per_col_norm:
            mats = [_to_mat(vals / col_vabs[c]) for c, vals in enumerate(per_pair_data)]
        else:
            mats = [_to_mat(d) for d in per_pair_data]

        si, sj = stem_pairs[pidx]
        for col, M in enumerate(mats):
            ax = axes[pidx, col]
            ax.imshow(M, cmap=cmap, norm=col_norms[col], aspect='equal',
                      origin='upper', interpolation='nearest')
            ax.set_xticks(range(4)); ax.set_xticklabels(NUCS_LABEL, fontsize=7)
            ax.set_yticks(range(4)); ax.set_yticklabels(NUCS_LABEL, fontsize=7)
            if pidx == 0:
                ax.set_title(col_titles[col], fontsize=8, pad=4)
            if col == 0:
                ax.set_ylabel(f'Pair ({si},{sj})\nnuc i', fontsize=8)
            if pidx == n_pairs - 1:
                ax.set_xlabel('nuc j', fontsize=7)
            if wt_nuc_ids is not None:
                wt_row = int(wt_nuc_ids[si])
                wt_col = int(wt_nuc_ids[sj])
                rect = mpatches.Rectangle(
                    (wt_col - 0.5, wt_row - 0.5), 1, 1,
                    linewidth=2, edgecolor='black', facecolor='none'
                )
                ax.add_patch(rect)

    if per_col_norm:
        # One colorbar per column, placed below the column's axes
        for col in range(ncols):
            col_axes = axes[:, col].tolist()
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=col_norms[col])
            sm.set_array([])
            fig.colorbar(sm, ax=col_axes, fraction=0.046, pad=0.08,
                         orientation='horizontal', label='Activity score')
    else:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=col_norms[0])
        sm.set_array([])
        fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.015, pad=0.04,
                     label='Activity score')

    return fig


def plot_library(y_mut, *, wt_y=0.0, bins=60, title='Library activity distribution') -> plt.Figure:
    """
    Histogram of library activity scores with a WT reference line.

    Parameters
    ----------
    y_mut : (N,) array of library GT scores (nonlin_additive_pairwise)
    wt_y  : WT activity (0.0 by convention)
    bins  : histogram bins
    title : axes title
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(y_mut, bins=bins, color='#4C72B0', alpha=0.75, edgecolor='white', linewidth=0.4)
    ax.axvline(wt_y, color='black', linewidth=1.6, linestyle='--', label=f'WT ({wt_y:.3f})')
    ax.set_xlabel('Activity score', fontsize=10)
    ax.set_ylabel('Count', fontsize=10)
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig
