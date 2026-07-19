"""Join per-model ``summary.json`` files into one comparison table (test Pearson per RBP)."""

from __future__ import annotations

import json
from pathlib import Path


def load_summaries(results_root: Path, metric: str = "test_pearson") -> dict[str, dict[str, float]]:
    """Discover ``<results_root>/<label>/summary.json`` and return ``{label: {rbp: <metric>}}``.

    The label is the sub-directory name (e.g. ``residualbind``, ``agft_modeA``), so the same model
    run with different settings stays distinguishable. ``metric`` is the per-RBP key to pull
    (``test_pearson`` for final results, ``best_val_pearson`` during exploration); missing keys
    become NaN.
    """
    out: dict[str, dict[str, float]] = {}
    for summary in sorted(results_root.glob("*/summary.json")):
        label = summary.parent.name
        rows = json.loads(summary.read_text())
        out[label] = {r["rbp"]: r.get(metric, float("nan")) for r in rows}
    return out


def format_table(summaries: dict[str, dict[str, float]]) -> str:
    """Render a Markdown table: rows = RBP, columns = run label, plus a mean row."""
    if not summaries:
        return "(no summary.json files found)"
    labels = list(summaries)
    rbps: list[str] = []
    for col in summaries.values():
        for rbp in col:
            if rbp not in rbps:
                rbps.append(rbp)

    def cell(v: float) -> str:
        return "—" if v != v else f"{v:.4f}"  # NaN check

    header = "| RBP | " + " | ".join(labels) + " |"
    sep = "|" + "---|" * (len(labels) + 1)
    lines = [header, sep]
    for rbp in rbps:
        lines.append("| " + rbp + " | "
                     + " | ".join(cell(summaries[l].get(rbp, float("nan"))) for l in labels) + " |")

    def mean(col: dict[str, float]) -> float:
        vals = [v for v in col.values() if v == v]
        return sum(vals) / len(vals) if vals else float("nan")

    lines.append("| **mean** | "
                 + " | ".join(f"**{cell(mean(summaries[l]))}**" for l in labels) + " |")
    return "\n".join(lines)
