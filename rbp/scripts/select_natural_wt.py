"""
Select high/low natural WT probes for a given RNAcompete-2013 RBP, for the
random-distribution plot / ground-truth WT sequence. Generalizes
select_msi1_natural_wt.py (archived) to any --rbp_id/--motif.

High WT criterion:
  highest measured exactly-one-<motif> probe with activity <= mean + 2 s.d.
  over the finite train/valid/test measurement distribution.

Low WT criterion:
  lowest measured exactly-one-<motif> probe with activity >= mean - 2 s.d.
  over the same distribution.

Usage:
    python rbp/scripts/select_natural_wt.py --rbp_id RNCMPT00112 --motif AUUUA --name hur
    python rbp/scripts/select_natural_wt.py --rbp_id RNCMPT00047 --motif ACUAAC --name qki
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[2]
H5_PATH = Path("/home/nagle/residualbind/data/RNAcompete_2013/rnacompete2013.h5")
NUCS = np.array(list("ACGU"))


def _decode(row: np.ndarray) -> str:
    x = row.T
    idx = x.argmax(axis=1)
    has_base = x.sum(axis=1) > 0.5
    return "".join(np.where(has_base, NUCS[idx], "N").tolist())


def _motif_starts(seq: str, motif: str) -> list[int]:
    seq = seq.replace("N", "A")
    return [i for i in range(len(seq) - len(motif) + 1) if seq[i:i + len(motif)] == motif]


def _load_archive(rbp_id: str):
    ys, seqs, meta = [], [], []
    with h5py.File(H5_PATH, "r") as f:
        experiments = [e.decode() for e in f["experiment"][:]]
        rbp_col = experiments.index(rbp_id)
        for split in ("train", "valid", "test"):
            x = f[f"X_{split}"]
            y = f[f"Y_{split}"][:, rbp_col]
            for i, value in enumerate(y):
                if np.isfinite(value):
                    ys.append(float(value))
                    seqs.append(_decode(x[i]))
                    meta.append((split, i))
    return np.array(ys, dtype=float), seqs, meta


def _encode_with_n(seq: str) -> np.ndarray:
    x = np.zeros((len(seq), 4), dtype=np.float32)
    for i, c in enumerate(seq):
        if c in "ACGU":
            x[i, "ACGU".index(c)] = 1.0
        elif c != "N":
            raise ValueError(f"Unsupported base {c!r}")
    return x


def _raw_score(seq: str, checkpoint_dir: Path) -> float | None:
    if not checkpoint_dir.exists():
        return None
    sys.path.insert(0, str(REPO / "rbp" / "src"))
    import torch
    from rbp_bench.models import load_ensemble

    paths = sorted(checkpoint_dir.glob("member*.pt"),
                    key=lambda p: int(p.stem.replace("member", "")))
    if not paths:
        return None
    device = torch.device("cpu")
    sample = torch.zeros((2, len(seq), 4), dtype=torch.float32)
    model = load_ensemble(paths, "residualbind", {}, device, sample)
    model.eval()
    with torch.no_grad():
        y = model(torch.from_numpy(_encode_with_n(seq)[None])).detach().cpu().numpy().reshape(-1)[0]
    return float(y)


def _print_choice(name: str, idx: int, ys: np.ndarray, seqs: list[str], meta,
                  mean: float, sd: float, motif: str, checkpoint_dir: Path) -> None:
    padded = seqs[idx]
    a_padded = padded.replace("N", "A")
    split, local_idx = meta[idx]
    print(f"\n{name}")
    print(f"selected split={split} local_idx={local_idx}")
    print(f"measured={ys[idx]:.6f} z={(ys[idx] - mean) / sd:.6f}")
    print(f"archived_padded={padded}")
    print(f"a_padded={a_padded}")
    print(f"motif={motif} starts={_motif_starts(a_padded, motif)}")
    raw = _raw_score(padded, checkpoint_dir)
    if raw is not None:
        print(f"raw_score_archived_N_padding={raw:.6f}")
        raw_a = _raw_score(a_padded, checkpoint_dir)
        print(f"raw_score_A_padding={raw_a:.6f}")
    else:
        print(f"[no ensemble checkpoints found yet at {checkpoint_dir}; skipping raw-score check]")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rbp_id", required=True, help="e.g. RNCMPT00112 (HuR)")
    ap.add_argument("--motif", required=True, help="e.g. AUUUA (HuR ARE core)")
    ap.add_argument("--name", required=True, help="short label, e.g. hur")
    args = ap.parse_args()

    checkpoint_dir = REPO / "rbp" / "results" / args.name

    ys, seqs, meta = _load_archive(args.rbp_id)
    mean = float(ys.mean())
    sd = float(ys.std(ddof=0))
    lower = mean - 2 * sd
    upper = mean + 2 * sd
    one_motif = np.array([i for i, s in enumerate(seqs) if len(_motif_starts(s, args.motif)) == 1], dtype=int)
    high_eligible = one_motif[ys[one_motif] <= upper]
    low_eligible = one_motif[ys[one_motif] >= lower]
    high_idx = int(high_eligible[np.argmax(ys[high_eligible])])
    low_idx = int(low_eligible[np.argmin(ys[low_eligible])])

    print(f"rbp_id={args.rbp_id} motif={args.motif}")
    print(f"n={len(ys)}")
    print(f"mean={mean:.6f} sd={sd:.6f} mean-2sd={lower:.6f} mean+2sd={upper:.6f}")
    print(f"exactly_one_{args.motif}={len(one_motif)}")
    _print_choice("high", high_idx, ys, seqs, meta, mean, sd, args.motif, checkpoint_dir)
    _print_choice("low", low_idx, ys, seqs, meta, mean, sd, args.motif, checkpoint_dir)


if __name__ == "__main__":
    main()
