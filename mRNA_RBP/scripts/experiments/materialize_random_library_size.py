"""Deterministically materialize an additional random-library subsample size."""

import argparse
import os

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_base", required=True)
    parser.add_argument("--n_instances", type=int, required=True)
    parser.add_argument("--library_size", type=int, required=True)
    parser.add_argument("--mut_rates", nargs="+", type=int, default=[5, 10, 25])
    args = parser.parse_args()

    for k in range(args.n_instances):
        for rate in args.mut_rates:
            mut_dir = os.path.join(args.out_base, f"instance_{k:02d}", f"mut{rate:02d}")
            pool_path = os.path.join(mut_dir, "pool_2M.npz")
            out_path = os.path.join(mut_dir, f"lib_{args.library_size}.npz")
            if os.path.isfile(out_path):
                print(f"[skip] {out_path}")
                continue
            pool = np.load(pool_path)
            if args.library_size > len(pool["nuc_ids"]):
                raise ValueError(f"requested {args.library_size} from pool of {len(pool['nuc_ids'])}")
            rng = np.random.default_rng(k * 100_000 + rate * 1_000 + args.library_size)
            idx = rng.choice(len(pool["nuc_ids"]), args.library_size, replace=False)
            arrays = {key: pool[key][idx] if len(pool[key].shape) and pool[key].shape[0] == len(pool["nuc_ids"])
                      else pool[key] for key in pool.files}
            np.savez_compressed(out_path, **arrays)
            print(f"[write] {out_path} n={args.library_size:,}")


if __name__ == "__main__":
    main()
