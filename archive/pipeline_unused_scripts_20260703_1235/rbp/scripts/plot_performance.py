#!/usr/bin/env python
"""Plot model performance per RBP for a given split, from ``results/<label>/summary.json``.

    # validation (best_val_pearson) heatmap over all discovered models
    python scripts/plot_performance.py --results_dir results --split valid

    # test heatmap, only specific models in this order
    python scripts/plot_performance.py --results_dir results --split test \
        --models residualbind agft_bin4 agft_bin8

Models are auto-discovered from ``results/<label>/`` (a new model just appears), so this scales as
more methods are added.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from rbp_bench.plotting import KINDS, pretty_label  # noqa: E402
from rbp_bench.report import load_summaries  # noqa: E402

# Split -> (summary.json key, axis label).
SPLIT_METRIC = {
    "valid": ("best_val_pearson", "best val Pearson r"),
    "test": ("test_pearson", "test Pearson r"),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results_dir", required=True, help="Root holding <label>/summary.json dirs")
    ap.add_argument("--split", choices=list(SPLIT_METRIC), default="valid")
    ap.add_argument("--kind", choices=list(KINDS), default="heatmap")
    ap.add_argument("--models", nargs="*", default=None,
                    help="Subset of model dir names, in the order to display them "
                         "(default: all discovered, sorted by mean)")
    ap.add_argument("--labels", nargs="*", default=None,
                    help="Display names matching --models (default: auto-prettified)")
    ap.add_argument("--output", default=None,
                    help="Output dir (default: <results_dir>/figures)")
    ap.add_argument("--fmt", default="png", help="Image format (png, pdf, svg, ...)")
    ap.add_argument("--cmap", default="viridis")
    args = ap.parse_args()

    metric, metric_label = SPLIT_METRIC[args.split]
    summaries = load_summaries(Path(args.results_dir), metric=metric)
    if not summaries:
        ap.error(f"No summary.json found under {args.results_dir}")

    # Determine model order: explicit --models is preserved (not re-sorted); else all, by mean.
    explicit = args.models is not None
    if explicit:
        missing = [m for m in args.models if m not in summaries]
        if missing:
            print(f"warning: no results for {missing}", file=sys.stderr)
        order = [m for m in args.models if m in summaries]
        if not order:
            ap.error("None of the requested --models were found")
    else:
        order = list(summaries)

    # Display labels: explicit --labels, else auto-prettified (e.g. agft_bin8 -> "AGFT 8bp").
    if args.labels:
        if not explicit or len(args.labels) != len(args.models):
            ap.error("--labels must match --models one-to-one")
        names = {m: lbl for m, lbl in zip(args.models, args.labels)}
    else:
        names = {m: pretty_label(m) for m in order}
    display = {names[m]: summaries[m] for m in order}

    fig, _ = KINDS[args.kind](display, metric_label=metric_label, split=args.split,
                              cmap=args.cmap, sort_models=not explicit)

    out_dir = Path(args.output) if args.output else Path(args.results_dir) / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.split}_{args.kind}.{args.fmt}"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
