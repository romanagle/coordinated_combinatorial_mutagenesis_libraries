"""Model-agnostic training harness for the RNAcompete benchmark.

A run is described by a config dict (see ``configs/*.json``)::

    {
      "model": "residualbind",            # registry key
      "data":  {"sequence_length": null, "normalization": "log_norm",
                "batch_size": 100, "num_workers": 4},
      "model_args": {...},                # passed to the model builder
      "stages": [                          # trained in order; later stages continue from the best
        {"trainable": "all", "lr": 1e-3, "weight_decay": 0.0,
         "num_epochs": 300, "patience": 20,
         "lr_decay": 0.3, "decay_patience": 7}   # lr_decay/decay_patience optional
      ],
      "seed": 43
    }

Every model is a ``BenchModel`` with ``forward(x:(B,L,4)) -> (B,)`` and a ``set_stage(name)`` that
configures which params train. The harness handles loaders, lazy-param materialisation, the
per-stage early-stopping/LR-decay loop, evaluation, checkpointing, and the comparable
``summary.json``. Adding a model needs no trainer changes.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .data import build_loaders
from .metrics import pearson
from .models import get_builder


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[float, float]:
    """Return ``(mse, pearson_r)`` over ``loader``."""
    model.eval()
    preds, targets, losses = [], [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        p = model(x)
        losses.append(F.mse_loss(p.float(), y.float()).item() * len(y))
        preds.append(p.detach().cpu().numpy())
        targets.append(y.detach().cpu().numpy())
    n = len(loader.dataset)
    return sum(losses) / max(n, 1), pearson(np.concatenate(targets), np.concatenate(preds))


def run_stage(model, train_loader, valid_loader, device, stage: dict, log_prefix: str) -> float:
    """Train one stage with early stopping on val Pearson and optional plateau LR decay.

    Loads the best-by-val state back into ``model`` at the end (so the next stage continues from it).
    Returns the best validation Pearson.
    """
    model.set_stage(stage["trainable"])
    params = [p for p in model.parameters() if p.requires_grad]
    lr = float(stage["lr"])
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=float(stage.get("weight_decay", 0.0)))

    lr_decay = stage.get("lr_decay")              # None -> constant LR
    decay_patience = int(stage.get("decay_patience", 0))
    patience = int(stage["patience"])

    best = -np.inf
    best_state = copy.deepcopy(model.state_dict())
    stale = decay_stale = 0

    for epoch in range(int(stage["num_epochs"])):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            F.mse_loss(model(x).float(), y.float()).backward()
            optimizer.step()

        _, val_pearson = evaluate(model, valid_loader, device)
        improved = val_pearson > best
        print(f"  [{log_prefix}] epoch {epoch + 1:3d}  val_pearson={val_pearson:.4f}"
              f"{' *' if improved else ''}", flush=True)

        if improved:
            best, best_state = val_pearson, copy.deepcopy(model.state_dict())
            stale = decay_stale = 0
        else:
            stale += 1
            decay_stale += 1
            if lr_decay and decay_patience and decay_stale >= decay_patience:
                lr = max(lr * float(lr_decay), 1e-6)
                for g in optimizer.param_groups:
                    g["lr"] = lr
                decay_stale = 0
                print(f"  [{log_prefix}] decay LR -> {lr:.2e}", flush=True)
            if stale >= patience:
                print(f"  [{log_prefix}] early stop at epoch {epoch + 1}", flush=True)
                break

    model.load_state_dict(best_state)
    return best


def train_rbp(model_name: str, rbp: str, config: dict, h5_path: str,
              device, output_dir: Path, *, eval_test: bool = False) -> dict:
    """Train one model on one RBP.

    Trains each stage in order (each warm-starts from the previous stage's best-by-val state) and,
    per stage, saves a checkpoint ``{rbp}/stage{i}_{trainable}.pt`` and records that stage's
    end-of-training metrics (best validation Pearson always; test when ``eval_test``). ``best.pt`` is
    the best-by-val stage. The held-out test split is only touched when ``eval_test`` is True --
    keep it off during exploration to avoid overfitting to test.
    """
    data = config["data"]
    print(f"\n=== {model_name} / {rbp} ===", flush=True)
    train_loader, valid_loader, test_loader, _ = build_loaders(
        h5_path, rbp,
        sequence_length=data.get("sequence_length"),
        normalization=data.get("normalization", "log_norm"),
        batch_size=data.get("batch_size", 100),
        num_workers=data.get("num_workers", 4),
        pin_memory=data.get("pin_memory", True),
    )

    model = get_builder(model_name)(config.get("model_args", {}), device)
    # Materialise any lazy params before building optimizers.
    x0, _ = next(iter(train_loader))
    with torch.no_grad():
        model(x0.to(device))

    out_dir = output_dir / rbp
    out_dir.mkdir(parents=True, exist_ok=True)

    stage_metrics: list[dict] = []
    best_val, best_stage = -np.inf, 1
    for i, stage in enumerate(config["stages"], start=1):
        trainable = stage.get("trainable", f"stage{i}")
        stage_best = run_stage(model, train_loader, valid_loader, device, stage, log_prefix=trainable)

        # model now holds this stage's best-by-val state -> checkpoint + report it.
        ckpt = out_dir / f"stage{i}_{trainable}.pt"
        payload = model.checkpoint_payload(trainable)
        payload.update({"model": model_name, "rbp": rbp, "stage": i, "trainable": trainable})
        torch.save(payload, ckpt)

        entry = {"stage": i, "trainable": trainable, "best_val_pearson": float(stage_best)}
        msg = f"  [stage{i}:{trainable}] best val pearson={stage_best:.4f}"
        if eval_test:
            test_mse, test_pearson = evaluate(model, test_loader, device)
            entry["test_mse"], entry["test_pearson"] = test_mse, test_pearson
            msg += f"  | test pearson={test_pearson:.4f}"
        print(msg, flush=True)
        stage_metrics.append(entry)
        if stage_best > best_val:
            best_val, best_stage = stage_best, i

    # best.pt -> symlink to the best-by-val stage's checkpoint (no duplicate file).
    best_name = f"stage{best_stage}_{stage_metrics[best_stage - 1]['trainable']}.pt"
    best_link = out_dir / "best.pt"
    if best_link.exists() or best_link.is_symlink():
        best_link.unlink()
    best_link.symlink_to(best_name)

    result = {"rbp": rbp, "model": model_name, "best_val_pearson": float(best_val),
              "best_stage": best_stage, "stages": stage_metrics}
    if eval_test:  # top-level test = best-by-val stage's test, for the comparison table
        best_entry = stage_metrics[best_stage - 1]
        result["test_mse"] = best_entry["test_mse"]
        result["test_pearson"] = best_entry["test_pearson"]
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    return result


def run_benchmark(model_name: str, rbps: list[str], config: dict, h5_path: str,
                  device, output_dir: Path, *, eval_test: bool = False) -> list[dict]:
    """Train one model across ``rbps`` and write a top-level ``summary.json``.

    Reports best validation Pearson; also reports test Pearson when ``eval_test`` is set.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    results = [train_rbp(model_name, rbp, config, h5_path, device, output_dir, eval_test=eval_test)
               for rbp in rbps]
    (output_dir / "summary.json").write_text(json.dumps(results, indent=2))

    metric = "test_pearson" if eval_test else "best_val_pearson"
    label = "test Pearson" if eval_test else "best val Pearson"
    print(f"\n=== {model_name} summary ({label}) ===", flush=True)
    for r in results:
        print(f"  {r['rbp']:>6s}: {r[metric]:.4f}", flush=True)
    vals = [r[metric] for r in results if not np.isnan(r[metric])]
    if vals:
        print(f"  {'mean':>6s}: {np.mean(vals):.4f}", flush=True)
    return results
