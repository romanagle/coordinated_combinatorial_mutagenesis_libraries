#!/usr/bin/env python
"""Train single ResidualBind and 10-member ensemble on MSI1 (RNCMPT00176) from rnacompete2013.h5.

The 2013 h5 stores all 244 RBPs in a flat layout:
    /X_{split}  (N, 4, L)   float32  -- 4-channel one-hot, channels A,C,G,U
    /Y_{split}  (N, 244)    float32  -- binding intensities; column order = /experiment
    /experiment (244,)      bytes    -- RBP IDs

We extract the RNCMPT00176 column, drop NaN probes, apply log_norm normalisation
(fit on train), then train with the same hypers as residualbind_ens10.json.

Usage:
    python scripts/train_msi1.py \
        --h5_path /home/nagle/residualbind/data/RNAcompete_2013/rnacompete2013.h5 \
        --output_dir results/msi1 [--n_members 10] [--base_seed 43]
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from rbp_bench.data import normalize_targets
from rbp_bench.metrics import pearson
from rbp_bench.models import get_builder
from rbp_bench.models.ensemble import EnsembleModel
from rbp_bench.trainer import evaluate, run_stage

RBP = "RNCMPT00176"


class MSI1Dataset(Dataset):
    """Single-RBP slice of rnacompete2013.h5 (flat multi-RBP layout).

    Reads the RNCMPT00176 column, drops NaN probes, and normalises targets.
    X shape in the file: (N, 4, L) — transposed to (N, L, 4) to match BenchModel.
    """

    def __init__(self, h5_path: str, split: str, rbp_col: int,
                 norm_params: list[float] | None = None) -> None:
        with h5py.File(h5_path, "r") as f:
            x = np.array(f[f"X_{split}"], dtype=np.float32)   # (N, 4, L)
            y = np.array(f[f"Y_{split}"], dtype=np.float32)[:, rbp_col]  # (N,)

        keep = ~np.isnan(y)
        x, y = x[keep], y[keep]
        x = np.transpose(x, (0, 2, 1))  # (N, L, 4)

        y_norm, self.norm_params = normalize_targets(y, "log_norm", norm_params)
        self.inputs = x
        self.targets = y_norm.astype(np.float32)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int):
        return torch.from_numpy(self.inputs[idx]), torch.tensor(self.targets[idx])


def build_msi1_loaders(h5_path: str, batch_size: int = 100, num_workers: int = 4):
    with h5py.File(h5_path, "r") as f:
        exps = [e.decode() for e in f["experiment"][:]]
    rbp_col = exps.index(RBP)

    train_ds = MSI1Dataset(h5_path, "train", rbp_col)
    valid_ds = MSI1Dataset(h5_path, "valid", rbp_col, train_ds.norm_params)
    test_ds  = MSI1Dataset(h5_path, "test",  rbp_col, train_ds.norm_params)

    kw = dict(num_workers=num_workers, pin_memory=True)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=False, **kw),
        DataLoader(valid_ds, batch_size=batch_size, shuffle=False, **kw),
        DataLoader(test_ds,  batch_size=batch_size, shuffle=False, **kw),
    )


STAGES = [
    {"trainable": "all", "lr": 1e-3, "weight_decay": 0.0,
     "num_epochs": 300, "patience": 20, "lr_decay": 0.3, "decay_patience": 7}
]


def _train_one(train_loader, valid_loader, device, seed: int, label: str):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    model = get_builder("residualbind")({}, device)
    x0, _ = next(iter(train_loader))
    with torch.no_grad():
        model(x0.to(device))

    best_val = -np.inf
    for stage in STAGES:
        best_val = max(best_val, run_stage(model, train_loader, valid_loader, device, stage,
                                           log_prefix=label))
    return model, float(best_val)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--h5_path",
                    default="/home/nagle/residualbind/data/RNAcompete_2013/rnacompete2013.h5")
    ap.add_argument("--output_dir", default="results/msi1")
    ap.add_argument("--n_members", type=int, default=10)
    ap.add_argument("--base_seed", type=int, default=43)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_loader, valid_loader, test_loader = build_msi1_loaders(args.h5_path)
    print(f"train={len(train_loader.dataset)}  valid={len(valid_loader.dataset)}"
          f"  test={len(test_loader.dataset)}", flush=True)

    # ── Single model (seed = base_seed) ──────────────────────────────────────
    print("\n=== single model ===", flush=True)
    single_model, single_val = _train_one(
        train_loader, valid_loader, device, args.base_seed, "single")
    _, single_test = evaluate(single_model, test_loader, device)
    payload = single_model.checkpoint_payload("all")
    payload.update({"model": "residualbind", "rbp": RBP, "seed": args.base_seed})
    torch.save(payload, out / "single.pt")
    print(f"  best val pearson = {single_val:.4f}", flush=True)
    print(f"  test  pearson    = {single_test:.4f}", flush=True)

    # ── Ensemble ──────────────────────────────────────────────────────────────
    print(f"\n=== {args.n_members}-member ensemble ===", flush=True)
    members, member_vals = [], []
    for i in range(args.n_members):
        seed = args.base_seed + i
        model, best_val = _train_one(train_loader, valid_loader, device, seed, f"member{i}")
        member_vals.append(best_val)
        payload = model.checkpoint_payload("all")
        payload.update({"model": "residualbind", "rbp": RBP, "member": i, "seed": seed})
        torch.save(payload, out / f"member{i}.pt")
        members.append(model)
        print(f"  member {i:2d} (seed {seed}): best val pearson = {best_val:.4f}", flush=True)

    ensemble = EnsembleModel(members).to(device)
    _, ens_val  = evaluate(ensemble, valid_loader, device)
    _, ens_test = evaluate(ensemble, test_loader, device)
    print(f"  ensemble val  pearson = {ens_val:.4f}  (mean member {np.mean(member_vals):.4f})",
          flush=True)
    print(f"  ensemble test pearson = {ens_test:.4f}", flush=True)

    # ── Summary ───────────────────────────────────────────────────────────────
    results = {
        "rbp": RBP,
        "single": {"val_pearson": single_val, "test_pearson": single_test},
        "ensemble": {
            "n_members": args.n_members,
            "val_pearson": ens_val,
            "test_pearson": ens_test,
            "mean_member_val_pearson": float(np.mean(member_vals)),
            "member_val_pearsons": member_vals,
        },
    }
    (out / "summary.json").write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out}/summary.json", flush=True)

    print("\n=== FINAL RESULTS ===")
    print(f"  single ResidualBind  — test Pearson r = {single_test:.4f}")
    print(f"  {args.n_members}-member ensemble    — test Pearson r = {ens_test:.4f}")


if __name__ == "__main__":
    main()
