#!/usr/bin/env python
"""Aggregate the 10 independently-trained VTS1 (RNCMPT00111) members into an
ensemble, evaluate it on the held-out test split, and write summary.json in
the same shape as results/msi1/summary.json."""

import json
import sys
from pathlib import Path

import numpy as np
import torch

_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rbp_bench.models import get_builder
from rbp_bench.models.ensemble import EnsembleModel
from rbp_bench.trainer import evaluate
from train_rncmpt2013_member import build_loaders

RBP = "RNCMPT00111"
OUT = Path(__file__).resolve().parent.parent / "results" / "vts1"
H5 = "/home/nagle/residualbind/data/RNAcompete_2013/rnacompete2013.h5"


def _load_member_old_torch(path, device, sample_input):
    # this env's torch (1.12.1, used for GPU training) predates the
    # `weights_only` kwarg that rbp_bench.models.load_ensemble passes --
    # load_ensemble itself is fine under the newer torch in rbp/.venv
    # (used by ResidualBindEnsembleOracle at inference time).
    ckpt = torch.load(path, map_location=device)
    m = get_builder("residualbind")({}, device)
    with torch.no_grad():
        m(sample_input.to(device))
    m.load_state_dict(ckpt["model_state_dict"])
    m.eval()
    return m


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    _, _, test_loader = build_loaders(H5, RBP)

    member_paths = [OUT / f"member{i}.pt" for i in range(10)]
    x0, _ = next(iter(test_loader))
    members = [_load_member_old_torch(p, device, x0) for p in member_paths]
    ensemble = EnsembleModel(members).to(device)
    _, ens_test = evaluate(ensemble, test_loader, device)
    print(f"ensemble test pearson = {ens_test:.4f}")

    single_json = json.loads((OUT / "single.json").read_text())
    member_jsons = [json.loads((OUT / f"member{i}.json").read_text()) for i in range(10)]
    member_val = [m["val_pearson"] for m in member_jsons]
    member_test = [m["test_pearson"] for m in member_jsons]

    results = {
        "rbp": RBP,
        "single": {"val_pearson": single_json["val_pearson"], "test_pearson": single_json["test_pearson"]},
        "ensemble": {
            "n_members": 10,
            "test_pearson": float(ens_test),
            "mean_member_val_pearson": float(np.mean(member_val)),
            "mean_member_test_pearson": float(np.mean(member_test)),
            "member_val_pearsons": member_val,
            "member_test_pearsons": member_test,
        },
    }
    (OUT / "summary.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
