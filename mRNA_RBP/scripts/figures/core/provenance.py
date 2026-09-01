"""Small provenance labels for generated figures."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional


def _existing_paths(paths: Optional[Iterable[object]]) -> list[Path]:
    if paths is None:
        return []
    out = []
    for item in paths:
        if item is None:
            continue
        path = Path(item)
        if path.exists():
            out.append(path)
    return out


def _infer_status(paths: list[Path]) -> str:
    env_status = os.environ.get("FIGURE_LIBRARY_STATUS")
    if env_status:
        return env_status.strip()
    path_text = " ".join(str(p).lower() for p in paths)
    if "cached" in path_text:
        return "cached"
    if "fresh" in path_text or paths:
        return "fresh"
    return "unknown"


def _infer_made(paths: list[Path]) -> str:
    env_made = os.environ.get("FIGURE_LIBRARY_MADE")
    if env_made:
        return env_made.strip()
    if not paths:
        return "unknown"
    newest = max(p.stat().st_mtime for p in paths)
    return datetime.fromtimestamp(newest).strftime("%Y-%m-%d %H:%M")


def stamp_figure(
    fig,
    *,
    library_status: Optional[str] = None,
    library_made: Optional[str] = None,
    source_paths: Optional[Iterable[object]] = None,
) -> str:
    """Stamp a top-right library provenance label and return the label text."""
    paths = _existing_paths(source_paths)
    status = (library_status or _infer_status(paths)).strip()
    made = (library_made or _infer_made(paths)).strip()
    label = f"Library: {status} | made {made}"
    fig.text(
        0.995,
        0.995,
        label,
        ha="right",
        va="top",
        fontsize=7,
        color="#333333",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#BBBBBB", alpha=0.82, lw=0.6),
    )
    return label
