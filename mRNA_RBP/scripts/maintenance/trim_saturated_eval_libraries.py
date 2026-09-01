"""Remove higher-order variants from existing saturated evaluation caches.

New pipeline runs generate only single and double mutants.  This small
migration keeps already-scored ``type3.npz`` files usable without rescoring
their sequences through an oracle.
"""

import argparse
import os
import tempfile

import numpy as np


def trim_library(path: str) -> tuple:
    """Keep mutation-order labels 1 and 2, preserving every NPZ field."""
    with np.load(path) as data:
        if "rate_labels" not in data.files:
            raise KeyError(f"{path}: missing rate_labels")
        labels = data["rate_labels"]
        keep = labels <= 2
        n_before = len(labels)
        payload = {
            key: (data[key][keep] if data[key].ndim > 0 and data[key].shape[0] == n_before
                  else data[key])
            for key in data.files
        }

    if not np.any(labels > 2):
        return n_before, n_before

    directory = os.path.dirname(os.path.abspath(path))
    fd, tmp_path = tempfile.mkstemp(prefix=".type3.", suffix=".npz", dir=directory)
    os.close(fd)
    try:
        np.savez_compressed(tmp_path, **payload)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
    return n_before, int(keep.sum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", help="Existing type3.npz files")
    args = parser.parse_args()
    for path in args.paths:
        before, after = trim_library(path)
        print(f"{path}: {before:,} -> {after:,} sequences")


if __name__ == "__main__":
    main()
