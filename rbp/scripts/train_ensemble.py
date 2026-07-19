#!/usr/bin/env python
"""Train an N-member ensemble of a base model on RNAcompete, per RBP.

    python scripts/train_ensemble.py --config configs/residualbind_ens10.json \
        --h5_path /home/shared/data/rnacompete/rnacompete2009.h5 \
        --all_rbps --output_dir results/residualbind_ens10

The base model is ``config["model"]`` (override with ``--base_model``); ensemble size/seed come from
``config["ensemble"]`` or ``--n_members``/``--base_seed``. Output plugs into compare.py / plot.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from rbp_bench.data import RNACOMPETE_2009_RBPS
from rbp_bench.ensemble_train import run_ensemble


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--h5_path", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--base_model", default=None, help="Override config['model']")
    ap.add_argument("--n_members", type=int, default=None, help="Ensemble size (default 10)")
    ap.add_argument("--base_seed", type=int, default=None, help="Member seeds = base_seed + i")
    ap.add_argument("--rbp", default=None, help="Single RBP (default: all 9 with --all_rbps)")
    ap.add_argument("--rbps", nargs="+", default=None,
                    help="Explicit subset of RBPs (for splitting across GPUs)")
    ap.add_argument("--all_rbps", action="store_true")
    ap.add_argument("--no_summary", action="store_true",
                    help="Skip writing summary.json (use when several processes share an output "
                         "dir; assemble it later with scripts/build_summary.py)")
    ap.add_argument("--eval_test", action="store_true",
                    help="Also evaluate the ensemble on the held-out test split")
    args = ap.parse_args()

    config = json.loads(Path(args.config).read_text())
    base_model = args.base_model or config["model"]
    ens = config.get("ensemble", {})
    n_members = args.n_members if args.n_members is not None else ens.get("n_members", 10)
    base_seed = args.base_seed if args.base_seed is not None else ens.get("base_seed", 43)

    torch.manual_seed(base_seed)
    np.random.seed(base_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.rbps:
        rbps = args.rbps
    elif args.all_rbps:
        rbps = list(RNACOMPETE_2009_RBPS)
    elif args.rbp:
        rbps = [args.rbp]
    else:
        ap.error("Provide --rbp NAME, --rbps A B C, or --all_rbps")

    run_ensemble(base_model, rbps, config, args.h5_path, device, Path(args.output_dir),
                 n_members=n_members, base_seed=base_seed, eval_test=args.eval_test,
                 write_summary=not args.no_summary)


if __name__ == "__main__":
    main()
