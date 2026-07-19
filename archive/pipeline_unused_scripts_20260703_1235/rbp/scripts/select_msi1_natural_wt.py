"""
Select MSI1 natural WT probes for the random distribution plot.

The source data is RNAcompete 2013 RNCMPT00176/MSI1. We use natural probes
containing exactly one UAG motif so the motif-hit region is unambiguous.

High WT criterion:
  highest measured exactly-one-UAG probe with activity <= mean + 2 s.d. over
  the finite train/valid/test RNCMPT00176 measurement distribution.

Low WT criterion:
  lowest measured exactly-one-UAG probe with activity >= mean - 2 s.d. over the
  same distribution. In practice the finite minimum is already above this lower
  cutoff for MSI1.
"""

from __future__ import annotations

from pathlib import Path
import sys

import h5py
import numpy as np


REPO = Path(__file__).resolve().parents[2]
H5_PATH = Path("/home/nagle/residualbind/data/RNAcompete_2013/rnacompete2013.h5")
RBP = "RNCMPT00176"
MOTIF = "UAG"
NUCS = np.array(list("ACGU"))

SELECTED = {
    "high": "AAAAGAUAGAGCGACGUGUUGGUGUAAAGGAGGUCCUGAAA",
    "low": "AAAAGACGAACCAACACACAAUGUCCGGGGAGACUAGUAAA",
}


def _decode(row: np.ndarray) -> str:
    x = row.T
    idx = x.argmax(axis=1)
    has_base = x.sum(axis=1) > 0.5
    return "".join(np.where(has_base, NUCS[idx], "N").tolist())


def _motif_starts(seq: str) -> list[int]:
    seq = seq.replace("N", "A")
    return [i for i in range(len(seq) - len(MOTIF) + 1) if seq[i:i + len(MOTIF)] == MOTIF]


def _load_archive():
    ys, seqs, meta = [], [], []
    with h5py.File(H5_PATH, "r") as f:
        experiments = [e.decode() for e in f["experiment"][:]]
        rbp_col = experiments.index(RBP)
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


def _raw_msi1_score(seq: str) -> float:
    sys.path.insert(0, str(REPO / "rbp" / "src"))
    import torch
    from rbp_bench.models import load_ensemble

    paths = sorted(
        (REPO / "rbp" / "results" / "msi1").glob("member*.pt"),
        key=lambda p: int(p.stem.replace("member", "")),
    )
    device = torch.device("cpu")
    sample = torch.zeros((2, len(seq), 4), dtype=torch.float32)
    model = load_ensemble(paths, "residualbind", {}, device, sample)
    model.eval()
    with torch.no_grad():
        y = model(torch.from_numpy(_encode_with_n(seq)[None])).detach().cpu().numpy().reshape(-1)[0]
    return float(y)


def _print_choice(name: str, idx: int, ys: np.ndarray, seqs: list[str], meta: list[tuple[str, int]],
                  mean: float, sd: float) -> None:
    padded = seqs[idx]
    a_padded = padded.replace("N", "A")
    split, local_idx = meta[idx]
    print(f"\n{name}")
    print(f"selected split={split} local_idx={local_idx}")
    print(f"measured={ys[idx]:.6f} z={(ys[idx] - mean) / sd:.6f}")
    print(f"archived_padded={padded}")
    print(f"a_padded={a_padded}")
    print(f"motif={MOTIF} starts={_motif_starts(a_padded)}")
    print(f"matches_config={a_padded == SELECTED[name]}")
    print(f"raw_msi1_score_archived_N_padding={_raw_msi1_score(padded):.6f}")
    print(f"raw_msi1_score_A_padding={_raw_msi1_score(a_padded):.6f}")


def main() -> None:
    ys, seqs, meta = _load_archive()
    mean = float(ys.mean())
    sd = float(ys.std(ddof=0))
    lower = mean - 2 * sd
    upper = mean + 2 * sd
    one_motif = np.array([i for i, s in enumerate(seqs) if len(_motif_starts(s)) == 1], dtype=int)
    high_eligible = one_motif[ys[one_motif] <= upper]
    low_eligible = one_motif[ys[one_motif] >= lower]
    high_idx = int(high_eligible[np.argmax(ys[high_eligible])])
    low_idx = int(low_eligible[np.argmin(ys[low_eligible])])

    print(f"n={len(ys)}")
    print(f"mean={mean:.6f} sd={sd:.6f} mean-2sd={lower:.6f} mean+2sd={upper:.6f}")
    print(f"exactly_one_{MOTIF}={len(one_motif)}")
    _print_choice("high", high_idx, ys, seqs, meta, mean, sd)
    _print_choice("low", low_idx, ys, seqs, meta, mean, sd)


if __name__ == "__main__":
    main()
