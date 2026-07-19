"""Train an N-member ensemble of a base model, per RBP, and evaluate the averaged prediction.

Each member is trained independently with its own seed (``base_seed + i``) -> diverse init and data
order. Members are saved as ``{rbp}/member{i}.pt``; the ensemble's averaged-prediction metrics go in
``{rbp}/metrics.json`` and the per-RBP ``summary.json`` (same shape as single-model runs, so it
shows up as a column in ``compare.py`` / ``plot_performance.py``).

Reuses the single-model primitives (``run_stage``, ``evaluate``) unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .data import build_loaders
from .models import EnsembleModel, get_builder
from .trainer import evaluate, run_stage


def _train_member(base_model: str, config: dict, train_loader, valid_loader, device, seed: int):
    """Train one member to its best-by-val state; return ``(model, best_val, last_trainable)``."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    model = get_builder(base_model)(config.get("model_args", {}), device)
    x0, _ = next(iter(train_loader))
    with torch.no_grad():
        model(x0.to(device))  # materialise lazy params

    best_val, last_trainable = -np.inf, "all"
    for stage in config["stages"]:
        last_trainable = stage.get("trainable", "all")
        best_val = max(best_val, run_stage(model, train_loader, valid_loader, device, stage,
                                            log_prefix=f"seed{seed}:{last_trainable}"))
    return model, float(best_val), last_trainable


def train_ensemble_rbp(base_model: str, rbp: str, config: dict, h5_path: str, device,
                       output_dir: Path, *, n_members: int = 10, base_seed: int = 43,
                       eval_test: bool = False, label: str | None = None) -> dict:
    """Train ``n_members`` members for one RBP and evaluate the mean-prediction ensemble."""
    label = label or f"{base_model}_ens{n_members}"
    data = config["data"]
    print(f"\n=== {label} / {rbp} ({n_members} members) ===", flush=True)
    train_loader, valid_loader, test_loader, _ = build_loaders(
        h5_path, rbp,
        sequence_length=data.get("sequence_length"),
        normalization=data.get("normalization", "log_norm"),
        batch_size=data.get("batch_size", 100),
        num_workers=data.get("num_workers", 4),
        pin_memory=data.get("pin_memory", True),
    )

    out_dir = output_dir / rbp
    out_dir.mkdir(parents=True, exist_ok=True)

    members, member_vals = [], []
    for i in range(n_members):
        seed = base_seed + i
        model, best_val, last_trainable = _train_member(
            base_model, config, train_loader, valid_loader, device, seed)
        member_vals.append(best_val)
        payload = model.checkpoint_payload(last_trainable)
        payload.update({"model": base_model, "rbp": rbp, "member": i, "seed": seed})
        torch.save(payload, out_dir / f"member{i}.pt")
        members.append(model)
        print(f"  member {i:2d} (seed {seed}): best val pearson={best_val:.4f}", flush=True)

    ensemble = EnsembleModel(members).to(device)
    _, ens_val = evaluate(ensemble, valid_loader, device)
    mean_member = float(np.mean(member_vals))
    print(f"  [ensemble] val pearson={ens_val:.4f}  (mean member {mean_member:.4f})", flush=True)

    result = {"rbp": rbp, "model": label, "n_members": n_members,
              "best_val_pearson": ens_val, "mean_member_val_pearson": mean_member,
              "member_val_pearsons": member_vals}
    if eval_test:
        _, ens_test = evaluate(ensemble, test_loader, device)
        result["test_pearson"] = ens_test
        print(f"  [ensemble] test pearson={ens_test:.4f}", flush=True)

    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    return result


def run_ensemble(base_model: str, rbps: list[str], config: dict, h5_path: str, device,
                 output_dir: Path, *, n_members: int = 10, base_seed: int = 43,
                 eval_test: bool = False, write_summary: bool = True) -> list[dict]:
    """Train the ensemble across ``rbps``; write ``summary.json`` unless ``write_summary`` is False.

    Set ``write_summary=False`` when several processes each train a subset of RBPs into the same
    output dir (e.g. one GPU per subset); assemble the summary afterwards with ``build_summary``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    label = output_dir.name
    results = [train_ensemble_rbp(base_model, rbp, config, h5_path, device, output_dir,
                                  n_members=n_members, base_seed=base_seed,
                                  eval_test=eval_test, label=label) for rbp in rbps]
    if write_summary:
        (output_dir / "summary.json").write_text(json.dumps(results, indent=2))

    metric = "test_pearson" if eval_test else "best_val_pearson"
    desc = "test Pearson" if eval_test else "ensemble val Pearson"
    print(f"\n=== {label} summary ({desc}) ===", flush=True)
    for r in results:
        print(f"  {r['rbp']:>6s}: {r[metric]:.4f}  (mean member {r['mean_member_val_pearson']:.4f})",
              flush=True)
    vals = [r[metric] for r in results if not np.isnan(r[metric])]
    if vals:
        print(f"  {'mean':>6s}: {np.mean(vals):.4f}", flush=True)
    return results
