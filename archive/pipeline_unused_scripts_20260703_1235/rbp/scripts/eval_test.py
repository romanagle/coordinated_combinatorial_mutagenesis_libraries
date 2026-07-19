#!/usr/bin/env python
"""Evaluate a finished run on the held-out TEST split by loading its saved checkpoints.

No retraining: loads each RBP's ``best.pt`` (or ensemble ``member*.pt``), evaluates on test, and
writes ``test_pearson``/``test_mse`` back into ``<RBP>/metrics.json`` and the run's ``summary.json``.
For an ensemble run it also records per-member test scores and the averaged-prediction test score.

    # single-model run
    python scripts/eval_test.py --config configs/residualbind.json \
        --results_dir results/residualbind --h5_path $H5

    # AlphaGenome run (needs the pretrained weights to rebuild the backbone)
    python scripts/eval_test.py --config configs/alphagenome_bin8.json \
        --results_dir results/agft_bin8 --h5_path $H5 --pretrained_weights $W

    # ensemble run
    python scripts/eval_test.py --config configs/residualbind_ens10.json \
        --results_dir results/residualbind_ens10 --h5_path $H5
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from rbp_bench.data import RNACOMPETE_2009_RBPS, build_loaders  # noqa: E402
from rbp_bench.loading import load_model  # noqa: E402
from rbp_bench.models import EnsembleModel  # noqa: E402
from rbp_bench.trainer import evaluate  # noqa: E402


def _member_paths(rbp_dir: Path) -> list[Path]:
    paths = list(rbp_dir.glob("member*.pt"))
    return sorted(paths, key=lambda p: int(re.search(r"member(\d+)", p.name).group(1)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--results_dir", required=True, help="A run dir holding <RBP>/ checkpoints")
    ap.add_argument("--h5_path", required=True)
    ap.add_argument("--pretrained_weights", default=None, help="Required for alphagenome_ft runs")
    args = ap.parse_args()

    config = json.loads(Path(args.config).read_text())
    base_model = config["model"]
    model_args = dict(config.get("model_args", {}))
    if args.pretrained_weights:
        model_args["pretrained_weights"] = args.pretrained_weights
    data = config["data"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(args.results_dir)

    rbp_dirs = sorted([d for d in run_dir.iterdir() if d.is_dir() and (d / "metrics.json").exists()])
    if not rbp_dirs:
        ap.error(f"No <RBP>/metrics.json under {run_dir}")

    for rbp_dir in rbp_dirs:
        rbp = rbp_dir.name
        _, _, test_loader, _ = build_loaders(
            args.h5_path, rbp,
            sequence_length=data.get("sequence_length"),
            normalization=data.get("normalization", "log_norm"),
            batch_size=data.get("batch_size", 100), num_workers=data.get("num_workers", 4),
        )
        sample = next(iter(test_loader))[0]
        metrics = json.loads((rbp_dir / "metrics.json").read_text())

        members = _member_paths(rbp_dir)
        if members:  # ensemble: per-member + averaged-prediction test
            loaded = [load_model(torch.load(p, map_location=device, weights_only=False),
                                 base_model, model_args, device, sample) for p in members]
            metrics["member_test_pearsons"] = [evaluate(m, test_loader, device)[1] for m in loaded]
            mse, pear = evaluate(EnsembleModel(loaded).to(device), test_loader, device)
        else:  # single model
            ckpt = torch.load(rbp_dir / "best.pt", map_location=device, weights_only=False)
            model = load_model(ckpt, base_model, model_args, device, sample)
            mse, pear = evaluate(model, test_loader, device)

        metrics["test_mse"], metrics["test_pearson"] = mse, pear
        (rbp_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        print(f"  {rbp:>6s}: test pearson={pear:.4f}", flush=True)

    # Rebuild summary.json from the updated per-RBP metrics.
    by_rbp = {json.loads((d / "metrics.json").read_text())["rbp"]:
              json.loads((d / "metrics.json").read_text()) for d in rbp_dirs}
    order = [r for r in RNACOMPETE_2009_RBPS if r in by_rbp] + \
            [r for r in by_rbp if r not in RNACOMPETE_2009_RBPS]
    (run_dir / "summary.json").write_text(json.dumps([by_rbp[r] for r in order], indent=2))
    print(f"Updated {run_dir / 'summary.json'} with test metrics", flush=True)


if __name__ == "__main__":
    main()
