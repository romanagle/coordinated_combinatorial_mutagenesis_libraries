#!/usr/bin/env python
"""Tabulate test Pearson across model runs (each a ``<label>/summary.json`` under a results root).

    python scripts/compare.py --results_dir results
    python scripts/compare.py --results_dir results --csv results/comparison.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from rbp_bench.report import format_table, load_summaries


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results_dir", required=True, help="Root holding <label>/summary.json dirs")
    ap.add_argument("--metric", default="test_pearson",
                    choices=["test_pearson", "best_val_pearson"],
                    help="Metric to tabulate (use best_val_pearson during exploration)")
    ap.add_argument("--csv", default=None, help="Optional path to also write the table as CSV")
    args = ap.parse_args()

    summaries = load_summaries(Path(args.results_dir), metric=args.metric)
    print(f"# metric: {args.metric}\n")
    print(format_table(summaries))

    if args.csv and summaries:
        labels = list(summaries)
        rbps: list[str] = []
        for col in summaries.values():
            for rbp in col:
                if rbp not in rbps:
                    rbps.append(rbp)
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["rbp", *labels])
            for rbp in rbps:
                w.writerow([rbp, *(summaries[l].get(rbp, "") for l in labels)])
        print(f"\nWrote {args.csv}", flush=True)


if __name__ == "__main__":
    main()
