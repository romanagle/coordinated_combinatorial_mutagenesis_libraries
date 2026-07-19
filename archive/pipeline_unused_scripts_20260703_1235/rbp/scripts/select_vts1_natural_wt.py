"""
Select the VTS1 natural WT probe used by downstream ResidualBind experiments.

Criterion:
  highest measured RNCMPT00111 natural RNAcompete probe with activity <= mean + 2 s.d.
  over the finite train/valid/test measurement distribution.

The selected archived probe contains RNAcompete padding Ns:
  NAGGACACUAAGUACAGGUUGCUGGCACAGGGCGCUCAUNN

For downstream mutation-library code, those padding Ns are represented as A:
  AAGGACACUAAGUACAGGUUGCUGGCACAGGGCGCUCAUAA
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[2]
DATA_DIR = REPO / "archive" / "RNCMPT00111_graphs"
NUCS = np.array(list("ACGU"))
SELECTED_SEQ = "AAGGACACUAAGUACAGGUUGCUGGCACAGGGCGCUCAUAA"


def _load_archive():
    ys, xs, meta = [], [], []
    for split in ("train", "valid", "test"):
        df = pd.read_csv(DATA_DIR / f"RNCMPT00111_{split}.csv")
        feature_cols = [c for c in df.columns if c.startswith("pos")]
        xs.append(df[feature_cols].to_numpy(dtype=np.float32).reshape(len(df), 41, 4))
        ys.append(df["RNCMPT00111"].to_numpy(dtype=float))
        meta.extend((split, i) for i in range(len(df)))
    return np.concatenate(xs), np.concatenate(ys), meta


def _decode(row: np.ndarray) -> str:
    idx = row.argmax(axis=1)
    has_base = row.sum(axis=1) > 0.5
    return "".join(np.where(has_base, NUCS[idx], "N").tolist())


def _encode_with_n(seq: str) -> np.ndarray:
    x = np.zeros((len(seq), 4), dtype=np.float32)
    for i, c in enumerate(seq):
        if c in "ACGU":
            x[i, "ACGU".index(c)] = 1.0
        elif c != "N":
            raise ValueError(f"Unsupported base {c!r}")
    return x


def _raw_vts1_score(seq: str) -> float:
    sys.path.insert(0, str(REPO / "rbp" / "src"))
    import torch
    from rbp_bench.models import load_ensemble

    paths = sorted(
        (REPO / "rbp" / "results" / "vts1").glob("member*.pt"),
        key=lambda p: int(p.stem.replace("member", "")),
    )
    device = torch.device("cpu")
    sample = torch.zeros((2, 41, 4), dtype=torch.float32)
    model = load_ensemble(paths, "residualbind", {}, device, sample)
    model.eval()
    x = _encode_with_n(seq)[None]
    with torch.no_grad():
        y = model(torch.from_numpy(x)).detach().cpu().numpy().reshape(-1)[0]
    return float(y)


def main():
    x, y, meta = _load_archive()
    finite = np.isfinite(y)
    mu = float(y[finite].mean())
    sd = float(y[finite].std(ddof=0))
    cutoff = mu + 2 * sd
    eligible = np.where(finite & (y <= cutoff))[0]
    selected_idx = int(eligible[np.argmax(y[eligible])])
    padded = _decode(x[selected_idx])
    a_padded = padded.replace("N", "A")
    split, local_idx = meta[selected_idx]

    print(f"n={len(y)} finite={int(finite.sum())}")
    print(f"mean={mu:.6f} sd={sd:.6f} mean+2sd={cutoff:.6f}")
    print(f"selected split={split} local_idx={local_idx}")
    print(f"measured={y[selected_idx]:.6f} z={(y[selected_idx] - mu) / sd:.6f}")
    print(f"archived_padded={padded}")
    print(f"a_padded={a_padded}")
    print(f"matches_config={a_padded == SELECTED_SEQ}")
    print(f"raw_vts1_score_archived_N_padding={_raw_vts1_score(padded):.6f}")
    print(f"raw_vts1_score_A_padding={_raw_vts1_score(a_padded):.6f}")


if __name__ == "__main__":
    main()
