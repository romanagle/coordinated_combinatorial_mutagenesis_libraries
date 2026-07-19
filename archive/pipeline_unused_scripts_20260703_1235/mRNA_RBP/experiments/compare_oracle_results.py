"""Compare mRNA ground-truth and ResidualBind-oracle downstream results.

Reads the nested JSON written by ``lib_size_spearman.py`` for each oracle and
writes a side-by-side CSV plus a grouped bar plot for one evaluation metric.
"""

import argparse
import csv
import json
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _mean_sem(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("nan"), float("nan"), 0
    sem = float(np.std(arr, ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
    return float(np.mean(arr)), sem, int(len(arr))


def _flatten(path, oracle_label, metric):
    with open(path) as f:
        cache = json.load(f)
    rows = []
    for gt_key, by_cfg in cache.get("surrogate", {}).items():
        for cfg_name, by_pct in by_cfg.items():
            for pct, by_lib in by_pct.items():
                for n_lib, entry in by_lib.items():
                    if metric not in entry:
                        continue
                    mean, sem, n = _mean_sem(entry[metric])
                    rows.append({
                        "oracle": oracle_label,
                        "gt_key": gt_key,
                        "surrogate": cfg_name,
                        "mut_pct": int(pct),
                        "lib_size": int(n_lib),
                        "metric": metric,
                        "mean": mean,
                        "sem": sem,
                        "n": n,
                    })
    return rows


def _write_csv(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fields = ["oracle", "gt_key", "surrogate", "mut_pct", "lib_size",
              "metric", "mean", "sem", "n"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _plot(path, rows, mut_pct, lib_size, save_svg=False):
    sub = [r for r in rows if r["mut_pct"] == mut_pct and r["lib_size"] == lib_size]
    if not sub:
        return False
    surrogates = []
    for r in sub:
        if r["surrogate"] not in surrogates:
            surrogates.append(r["surrogate"])
    oracles = []
    for r in sub:
        if r["oracle"] not in oracles:
            oracles.append(r["oracle"])

    x = np.arange(len(surrogates))
    width = 0.36
    fig, ax = plt.subplots(figsize=(max(8, 1.8 * len(surrogates)), 4.5))
    colors = {"MrnaRbpGroundTruth": "#4C72B0", "ResidualBind ensemble": "#DD8452"}

    for oi, oracle in enumerate(oracles):
        vals, errs = [], []
        for s in surrogates:
            matches = [r for r in sub if r["oracle"] == oracle and r["surrogate"] == s]
            vals.append(matches[0]["mean"] if matches else float("nan"))
            errs.append(matches[0]["sem"] if matches else 0.0)
        offset = (oi - (len(oracles) - 1) / 2.0) * width
        ax.bar(x + offset, vals, width, yerr=errs, capsize=3,
               label=oracle, color=colors.get(oracle))

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(surrogates, rotation=25, ha="right")
    ax.set_ylabel("Spearman rho")
    metric = sub[0]["metric"]
    ax.set_title(f"{metric}: mut {mut_pct}% / lib {lib_size:,}")
    ax.legend(framealpha=0.9)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    if save_svg:
        fig.savefig(path.replace(".png", ".svg"))
    plt.close(fig)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mrna_json",
                        default=os.path.join(_HERE, "outputs", "lib_size_spearman_results.json"))
    parser.add_argument("--residualbind_json",
                        default=os.path.join(_HERE, "outputs_residualbind_ensemble",
                                             "lib_size_spearman_results.json"))
    parser.add_argument("--metric", default="type2",
                        choices=["rand", "type1_200", "type1_2000", "type1_20000",
                                 "type2", "pairwise"])
    parser.add_argument("--mut_pct", type=int, default=10)
    parser.add_argument("--lib_size", type=int, default=20000)
    parser.add_argument("--out_dir",
                        default=os.path.join(_HERE, "outputs", "oracle_comparison"))
    parser.add_argument("--svg", action="store_true",
                        help="Also save an .svg copy of each figure")
    args = parser.parse_args()

    rows = []
    rows.extend(_flatten(args.mrna_json, "MrnaRbpGroundTruth", args.metric))
    rows.extend(_flatten(args.residualbind_json, "ResidualBind ensemble", args.metric))

    csv_path = os.path.join(args.out_dir, f"oracle_comparison_{args.metric}.csv")
    png_path = os.path.join(
        args.out_dir,
        f"oracle_comparison_{args.metric}_mut{args.mut_pct:02d}_lib{args.lib_size}.png",
    )
    _write_csv(csv_path, rows)
    plotted = _plot(png_path, rows, args.mut_pct, args.lib_size, save_svg=args.svg)
    print(f"Saved {csv_path}")
    if plotted:
        print(f"Saved {png_path}" + (" / .svg" if args.svg else ""))
    else:
        print("No rows matched the requested mut_pct/lib_size for plotting")


if __name__ == "__main__":
    main()
