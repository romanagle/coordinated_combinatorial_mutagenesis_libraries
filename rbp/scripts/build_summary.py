#!/usr/bin/env python
"""Assemble a run's ``summary.json`` from its per-RBP ``<RBP>/metrics.json`` files.

Useful when a run was split across processes/GPUs (each with --no_summary), or to rebuild a summary
after a partial/interrupted run.

    python scripts/build_summary.py --results_dir results/residualbind_ens10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from rbp_bench.data import RNACOMPETE_2009_RBPS


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results_dir", required=True, help="A run dir holding <RBP>/metrics.json")
    args = ap.parse_args()

    run_dir = Path(args.results_dir)
    by_rbp = {}
    for mj in run_dir.glob("*/metrics.json"):
        d = json.loads(mj.read_text())
        by_rbp[d["rbp"]] = d
    if not by_rbp:
        ap.error(f"No <RBP>/metrics.json found under {run_dir}")

    # Canonical RBP order first, then any extras.
    order = [r for r in RNACOMPETE_2009_RBPS if r in by_rbp]
    order += [r for r in by_rbp if r not in order]
    summary = [by_rbp[r] for r in order]

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote {run_dir / 'summary.json'} with {len(summary)} RBPs: {', '.join(order)}")


if __name__ == "__main__":
    main()
