#!/usr/bin/env python
"""Bar plot comparing single members vs the ensemble, per RBP, for a split.

Each RBP shows a 'single' bar (height = mean of the N members, with a dot per member) and an
'ensemble' bar (averaged-prediction, no dots). Reads the ensemble run's ``summary.json``.

    python scripts/plot_ensemble.py --results_dir results --ensemble residualbind_ens10 --split test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from rbp_bench.plotting import single_vs_ensemble_bar  # noqa: E402

# split -> (per-member key, ensemble key, axis label)
SPLIT_KEYS = {
    "test": ("member_test_pearsons", "test_pearson", "test Pearson r"),
    "valid": ("member_val_pearsons", "best_val_pearson", "best val Pearson r"),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--ensemble", default="residualbind_ens10",
                    help="Ensemble run label under results_dir")
    ap.add_argument("--split", choices=list(SPLIT_KEYS), default="test")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    member_key, ens_key, metric_label = SPLIT_KEYS[args.split]
    summary_path = Path(args.results_dir) / args.ensemble / "summary.json"
    rows = json.loads(summary_path.read_text())

    per_rbp = {}
    for r in rows:
        if member_key not in r or ens_key not in r:
            ap.error(f"{summary_path} lacks '{member_key}'/'{ens_key}' — run eval_test for --split {args.split}")
        per_rbp[r["rbp"]] = {"members": r[member_key], "ensemble": r[ens_key]}

    fig, _ = single_vs_ensemble_bar(per_rbp, metric_label=metric_label, split=args.split)

    out_dir = Path(args.output) if args.output else Path(args.results_dir) / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.split}_single_vs_ensemble.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
