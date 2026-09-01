#!/usr/bin/env python
"""Train ONE ResidualBind model (single or one ensemble member) on an arbitrary
RNCMPT-2013 column from rnacompete2013.h5. Generalizes train_msi1.py (which is
hardcoded to RNCMPT00176/MSI1) to any --rbp_id, and trains one model per
process so members can run in parallel across GPUs instead of sequentially
in one process.

Usage (run once per member, e.g. across 8 GPUs in parallel):
    python scripts/train_rncmpt2013_member.py --rbp_id RNCMPT00111 \
        --which single --device cuda:0 --output_dir results/vts1
    python scripts/train_rncmpt2013_member.py --rbp_id RNCMPT00111 \
        --which 0 --device cuda:1 --output_dir results/vts1
    ... --which 1 --device cuda:2 ...  (up to --which 9)

Writes:
    {output_dir}/{single.pt | memberN.pt}
    {output_dir}/{single.json | memberN.json}   -- {seed, val_pearson, test_pearson}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from rbp_bench.data import normalize_targets
from rbp_bench.models import get_builder
from rbp_bench.trainer import evaluate, run_stage

BASE_SEED = 43

STAGES = [
    {"trainable": "all", "lr": 1e-3, "weight_decay": 0.0,
     "num_epochs": 300, "patience": 20, "lr_decay": 0.3, "decay_patience": 7}
]


class RNCMPT2013Dataset(Dataset):
    """Single-RBP slice of rnacompete2013.h5 (flat multi-RBP layout, see train_msi1.py)."""

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


def build_loaders(h5_path: str, rbp_id: str, batch_size: int = 100, num_workers: int = 4):
    with h5py.File(h5_path, "r") as f:
        exps = [e.decode() for e in f["experiment"][:]]
    rbp_col = exps.index(rbp_id)

    train_ds = RNCMPT2013Dataset(h5_path, "train", rbp_col)
    valid_ds = RNCMPT2013Dataset(h5_path, "valid", rbp_col, train_ds.norm_params)
    test_ds  = RNCMPT2013Dataset(h5_path, "test",  rbp_col, train_ds.norm_params)

    kw = dict(num_workers=num_workers, pin_memory=True)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=False, **kw),
        DataLoader(valid_ds, batch_size=batch_size, shuffle=False, **kw),
        DataLoader(test_ds,  batch_size=batch_size, shuffle=False, **kw),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rbp_id", required=True, help="e.g. RNCMPT00111 (VTS1), RNCMPT00176 (MSI1)")
    ap.add_argument("--h5_path",
                    default="/home/nagle/residualbind/data/RNAcompete_2013/rnacompete2013.h5")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--which", required=True,
                    help="'single' or a member index (0..n_members-1)")
    ap.add_argument("--base_seed", type=int, default=BASE_SEED)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    is_single = args.which == "single"
    seed = args.base_seed if is_single else args.base_seed + int(args.which)
    label = "single" if is_single else f"member{args.which}"

    device = torch.device(args.device)
    print(f"[{label}] device={device}  rbp={args.rbp_id}  seed={seed}", flush=True)

    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    train_loader, valid_loader, test_loader = build_loaders(args.h5_path, args.rbp_id)
    print(f"[{label}] train={len(train_loader.dataset)}  valid={len(valid_loader.dataset)}"
          f"  test={len(test_loader.dataset)}", flush=True)

    model = get_builder("residualbind")({}, device)
    x0, _ = next(iter(train_loader))
    with torch.no_grad():
        model(x0.to(device))

    best_val = -np.inf
    for stage in STAGES:
        best_val = max(best_val, run_stage(model, train_loader, valid_loader, device, stage,
                                           log_prefix=f"{args.rbp_id}:{label}"))
    _, test_pearson = evaluate(model, test_loader, device)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = model.checkpoint_payload("all")
    payload.update({"model": "residualbind", "rbp": args.rbp_id, "seed": seed})
    if not is_single:
        payload["member"] = int(args.which)
    torch.save(payload, out / f"{label}.pt")

    result = {"seed": seed, "val_pearson": float(best_val), "test_pearson": float(test_pearson)}
    (out / f"{label}.json").write_text(json.dumps(result, indent=2))

    print(f"[{label}] best val pearson = {best_val:.4f}  test pearson = {test_pearson:.4f}",
          flush=True)


if __name__ == "__main__":
    main()
