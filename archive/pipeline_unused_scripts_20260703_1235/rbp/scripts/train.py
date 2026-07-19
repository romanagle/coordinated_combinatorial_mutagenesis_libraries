#!/usr/bin/env python
"""Single entry point to train any registered model on RNAcompete, per RBP.

    python scripts/train.py --config configs/residualbind.json \
        --h5_path /home/shared/data/rnacompete/rnacompete2009.h5 --all_rbps \
        --output_dir results/residualbind

    python scripts/train.py --config configs/alphagenome_modeB.json \
        --h5_path ... --pretrained_weights /home/shared/models/.../model_all_folds.safetensors \
        --all_rbps --output_dir results/agft_modeB

The model is selected by ``"model"`` in the config (override with ``--model``). ``--pretrained_weights``
overrides ``model_args.pretrained_weights`` (handy for machine-specific paths).
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
from rbp_bench.trainer import run_benchmark


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--h5_path", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--model", default=None, help="Override config['model']")
    ap.add_argument("--pretrained_weights", default=None,
                    help="Override model_args.pretrained_weights (alphagenome_ft)")
    ap.add_argument("--rbp", default=None, help="Single RBP (default: all 9 with --all_rbps)")
    ap.add_argument("--all_rbps", action="store_true")
    ap.add_argument("--eval_test", action="store_true",
                    help="Also evaluate the held-out test split (off by default to avoid "
                         "overfitting to test during exploration; validation is always reported)")
    args = ap.parse_args()

    config = json.loads(Path(args.config).read_text())
    model_name = args.model or config["model"]
    if args.pretrained_weights:
        config.setdefault("model_args", {})["pretrained_weights"] = args.pretrained_weights

    seed = config.get("seed", 43)
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.all_rbps:
        rbps = list(RNACOMPETE_2009_RBPS)
    elif args.rbp:
        rbps = [args.rbp]
    else:
        ap.error("Provide --rbp NAME or --all_rbps")

    run_benchmark(model_name, rbps, config, args.h5_path, device, Path(args.output_dir),
                  eval_test=args.eval_test)


if __name__ == "__main__":
    main()
