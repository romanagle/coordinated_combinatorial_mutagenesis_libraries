#!/usr/bin/env python
"""Collect and optionally regenerate the GT/surrogate analysis figure set.

The plotting scripts in mRNA_RBP historically write into several fixed output
locations. This wrapper gives the analysis a single reproducible entry point:
it records the ground-truth inputs, optionally reruns supported plot scripts,
and stages the final figures into one directory with a manifest.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


REPO = Path(__file__).resolve().parents[1]
DEFAULT_MRNA_PLOT_DIR = REPO / "mRNA_RBP" / "outputs" / "notebook_plots"
DEFAULT_COLLECTION_DIR = REPO / "mRNA_RBP" / "outputs" / "ground_truth_collections"

COLLECTION_PRESETS = {
    "synthetic_gt": {
        "ground_truth": "synthetic",
        "collection_dir": DEFAULT_COLLECTION_DIR / "Synthetic GT",
        "default_out_dir": REPO / "mRNA_RBP" / "outputs" / "reproducible_pipeline" / "synthetic_gt",
        "sequence": "AAAAAAAAGCGCAUGCUUGCAUGGCAUGCGCAAAAAAAAAA",
        "structure": "........((((((((.......))))))))..........",
        "motif_positions": "17-21",
    },
    "residualbind_vts1": {
        "ground_truth": "residualbind",
        "collection_dir": DEFAULT_COLLECTION_DIR / "ResidualBind oracle VTS1",
        "default_out_dir": REPO / "mRNA_RBP" / "outputs" / "reproducible_pipeline" / "residualbind_vts1",
        "sequence": "AAGGACACUAAGUACAGGUUGCUGGCACAGGGCGCUCAUAA",
        "structure": "..............(((....)))((.....))........",
        "motif_positions": "20-24",
    },
}

VTS1_ARTIFACT_COMMANDS = [
    [sys.executable, "mRNA_RBP/generate_residualbind_vts1_scatter_predictions.py", "--force"],
]

VTS1_PLOT_COMMANDS = [
    [sys.executable, "mRNA_RBP/plots/plot_residualbind_vts1_collection_figures.py"],
    [sys.executable, "mRNA_RBP/plots/plot_residualbind_vts1_result_figures.py"],
    [sys.executable, "mRNA_RBP/plots/plot_residualbind_vts1_rand_region_distributions.py"],
    [sys.executable, "mRNA_RBP/plots/plot_residualbind_vts1_scatter_by_mutcount.py"],
    [
        sys.executable,
        "mRNA_RBP/plots/plot_cross_mutrate.py",
        "--results_json",
        "mRNA_RBP/outputs/ground_truth_collections/ResidualBind oracle VTS1/libraries_used_for_figures/cross_mutrate_results.json",
        "--gt_key",
        "vts1_residualbind",
        "--out_prefix",
        "vts1_",
        "--out_base",
        "mRNA_RBP/outputs/ground_truth_collections/ResidualBind oracle VTS1/cached outputs/fresh_pipeline_workspace_20260704",
    ],
]

VTS1_CROSS_MUTRATE_PLOTS = [
    "vts1_cross_mutrate_heatmap.png",
    "vts1_cross_mutrate_libsize_additive.png",
    "vts1_cross_mutrate_libsize_additive_p_pairwise.png",
    "vts1_cross_mutrate_libsize_nonlinear_additive.png",
    "vts1_cross_mutrate_libsize_nonlinear_additive_p_pairwise.png",
]

DEFAULT_FIGURES = {
    "coefficient_analysis": (
        DEFAULT_MRNA_PLOT_DIR
        / "coefficients_map"
        / "coefficients_oracle_vs_surrogate_msi1.png"
    ),
    "evaluation_library_distributions": DEFAULT_MRNA_PLOT_DIR / "library_distributions.png",
    "scatter_by_mutcount": DEFAULT_MRNA_PLOT_DIR / "scatter_by_mutcount.png",
    "rho_vs_libsize_type3": DEFAULT_MRNA_PLOT_DIR / "rho_vs_libsize_type3.png",
    "model_comparison_type3": DEFAULT_MRNA_PLOT_DIR / "model_comparison_bar_type3.png",
    "cross_mutrate_heatmap": DEFAULT_MRNA_PLOT_DIR / "synthetic_gt_cross_mutrate_heatmap.png",
}

DEFAULT_RANDOM_LIBRARY_FIGURES = {
    "residualbind": [
        DEFAULT_COLLECTION_DIR
        / "ResidualBind oracle MSI1"
        / "figures"
        / "rand_lib_dist_msi1_oracle_region_classes.png",
        DEFAULT_COLLECTION_DIR
        / "ResidualBind oracle MSI1"
        / "figures"
        / "rand_lib_dist_msi1_oracle_region_classes_low_wt.png",
        DEFAULT_COLLECTION_DIR
        / "ResidualBind oracle VTS1"
        / "figures"
        / "rand_lib_dist_vts1_oracle_region_classes.png",
        DEFAULT_COLLECTION_DIR
        / "ResidualBind oracle VTS1"
        / "figures"
        / "rand_lib_dist_vts1_oracle_region_classes_low_wt.png",
    ],
    "deepsquid": [
        DEFAULT_COLLECTION_DIR
        / "deepSQUID MSI1"
        / "figures"
        / "rand_lib_dist_msi1_deepsquid.png",
        DEFAULT_COLLECTION_DIR
        / "deepSQUID VTS1"
        / "figures"
        / "rand_lib_dist_vts1_deepsquid.png",
    ],
}

SUPPORTED_REGEN_COMMANDS = [
    [
        sys.executable,
        "mRNA_RBP/lib_size_spearman.py",
        "--out_json",
        "mRNA_RBP/outputs/lib_size_spearman_results_type3.json",
        "--recompute_saturated",
        "--saturated_only",
        "--gt_keys",
        "additive",
        "additive_pairwise",
        "nonlin_additive",
        "nonlin_additive_pairwise",
    ],
    [sys.executable, "mRNA_RBP/plots/plot_library_distributions.py"],
    [sys.executable, "mRNA_RBP/plots/plot_synthetic_rand_region_distributions.py"],
    [sys.executable, "mRNA_RBP/plots/plot_scatter_by_mutcount.py"],
    [sys.executable, "mRNA_RBP/plots/plot_rho_vs_libsize_type3.py"],
    [sys.executable, "mRNA_RBP/plots/bar_surrogate_models_type3.py"],
    [
        sys.executable,
        "mRNA_RBP/plots/plot_cross_mutrate.py",
        "--out_prefix",
        "synthetic_gt_",
        "--out_base",
        "mRNA_RBP/outputs",
    ],
]


def parse_dot_bracket(dot_bracket: str) -> List[Tuple[int, int]]:
    """Return zero-based stem pairs from dot-bracket structure."""
    bracket_pairs = {"(": ")", "[": "]", "{": "}", "<": ">"}
    closing = {v: k for k, v in bracket_pairs.items()}
    stacks = {opener: [] for opener in bracket_pairs}
    pairs = []  # type: List[Tuple[int, int]]

    for idx, char in enumerate(dot_bracket.strip()):
        if char in bracket_pairs:
            stacks[char].append(idx)
        elif char in closing:
            opener = closing[char]
            if not stacks[opener]:
                raise ValueError(f"Unmatched closing bracket {char!r} at position {idx}")
            pairs.append((stacks[opener].pop(), idx))
        elif char == ".":
            continue
        else:
            raise ValueError(f"Unsupported dot-bracket character {char!r} at position {idx}")

    unclosed = [(opener, vals) for opener, vals in stacks.items() if vals]
    if unclosed:
        opener, vals = unclosed[0]
        raise ValueError(f"Unmatched opening bracket {opener!r} at position {vals[-1]}")
    return sorted(pairs)


def parse_positions(text: Optional[str]) -> List[int]:
    if not text:
        return []
    out = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = [int(x) for x in part.split("-", 1)]
            out.extend(range(start, end + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def validate_inputs(sequence: str, structure: str) -> Tuple[str, str, List[Tuple[int, int]]]:
    seq = sequence.strip().upper().replace("T", "U")
    if not seq:
        raise ValueError("--sequence cannot be empty")
    bad = sorted(set(seq) - set("ACGU"))
    if bad:
        raise ValueError(f"--sequence contains non-RNA bases: {bad}")
    db = structure.strip()
    if len(seq) != len(db):
        raise ValueError(
            f"--sequence length ({len(seq)}) must match --structure length ({len(db)})"
        )
    return seq, db, parse_dot_bracket(db)


def copy_file(src: Path, dst: Path, required: bool) -> dict:
    entry = {
        "source": str(src),
        "destination": str(dst),
        "exists": src.exists(),
        "copied": False,
    }
    if not src.exists():
        if required:
            raise FileNotFoundError(src)
        return entry
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    entry["copied"] = True
    return entry


def run_commands(commands: List[List[str]], env: Dict[str, str]) -> List[dict]:
    results = []
    for cmd in commands:
        print("[run]", " ".join(cmd), flush=True)
        completed = subprocess.run(cmd, cwd=REPO, env=env, check=False)
        results.append({"cmd": cmd, "returncode": completed.returncode})
        if completed.returncode != 0:
            raise RuntimeError(f"Command failed with return code {completed.returncode}: {cmd}")
    return results


def copy_vts1_cross_mutrate_outputs() -> List[dict]:
    copied = []
    src_dir = REPO / "mRNA_RBP" / "outputs" / "notebook_plots"
    dst_dir = (
        DEFAULT_COLLECTION_DIR
        / "ResidualBind oracle VTS1"
        / "figures"
    )
    for name in VTS1_CROSS_MUTRATE_PLOTS:
        copied.append(copy_file(src_dir / name, dst_dir / name, required=True))
    return copied


def build_subprocess_env() -> Dict[str, str]:
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(REPO))
    env.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
    env.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")

    squid_lib = "/home/nagle/miniconda3/envs/squid/lib"
    ld_parts = [squid_lib]
    for part in os.environ.get("LD_LIBRARY_PATH", "").split(":"):
        if part and part != squid_lib:
            ld_parts.append(part)
    if "/usr/local/cuda-11.2/lib64" not in ld_parts:
        ld_parts.append("/usr/local/cuda-11.2/lib64")
    env["LD_LIBRARY_PATH"] = ":".join(ld_parts)
    return env


def stage_outputs(args, out_dir: Path) -> list[dict]:
    staged = []
    figure_paths = dict(DEFAULT_FIGURES)
    if args.coefficient_figure:
        figure_paths["coefficient_analysis"] = Path(args.coefficient_figure)
    if args.mrna_plot_dir:
        mrna_dir = Path(args.mrna_plot_dir)
        for key, src in list(figure_paths.items()):
            if key == "coefficient_analysis" and args.coefficient_figure:
                continue
            if src.parent == DEFAULT_MRNA_PLOT_DIR:
                figure_paths[key] = mrna_dir / src.name
            elif DEFAULT_MRNA_PLOT_DIR in src.parents:
                figure_paths[key] = mrna_dir / src.relative_to(DEFAULT_MRNA_PLOT_DIR)

    for key, src in figure_paths.items():
        staged.append(copy_file(Path(src), out_dir / f"{key}.png", required=args.require_all))

    if args.ground_truth == "synthetic":
        random_sources = [
            Path(args.mrna_plot_dir) / "rand_lib_dist_synthetic_high_wt.png",
            Path(args.mrna_plot_dir) / "rand_lib_dist_synthetic_low_wt.png",
        ]
    else:
        if args.random_library_dir:
            random_dir = Path(args.random_library_dir)
            random_sources = sorted(random_dir.glob(args.random_library_glob))
        else:
            random_sources = DEFAULT_RANDOM_LIBRARY_FIGURES[args.ground_truth]
    if args.require_all and not random_sources:
        raise FileNotFoundError("No random-library figures found")
    random_out = out_dir / "random_library"
    for src in random_sources:
        staged.append(copy_file(src, random_out / src.name, required=args.require_all))
    return staged


def stage_collection_figures(collection_dir: Path, out_dir: Path, require_all: bool) -> list[dict]:
    figures_dir = collection_dir / "figures"
    if not figures_dir.exists():
        if require_all:
            raise FileNotFoundError(figures_dir)
        return []
    staged = []
    for src in sorted(figures_dir.glob("*.png")):
        staged.append(copy_file(src, out_dir / "figures" / src.name, required=require_all))
    return staged


def apply_collection_preset(args) -> dict:
    preset = COLLECTION_PRESETS.get(args.collection or "")
    if not preset:
        return {}
    if not args.ground_truth:
        args.ground_truth = preset["ground_truth"]
    if not args.sequence:
        args.sequence = preset["sequence"]
    if not args.structure:
        args.structure = preset["structure"]
    if not args.motif_positions:
        args.motif_positions = preset["motif_positions"]
    if not args.out_dir:
        args.out_dir = str(preset["default_out_dir"])
    return preset


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reproduce or collect the GT/surrogate figure pipeline outputs."
    )
    parser.add_argument(
        "--collection",
        choices=sorted(COLLECTION_PRESETS),
        default="",
        help=(
            "Named figure collection preset. This fills sequence/structure/motif "
            "defaults and stages the collection's active figures."
        ),
    )
    parser.add_argument(
        "--ground-truth",
        default="",
        choices=["synthetic", "residualbind", "deepsquid"],
        help="Ground-truth source used for the staged analysis.",
    )
    parser.add_argument("--sequence", default="", help="RNA sequence, A/C/G/U.")
    parser.add_argument(
        "--structure",
        default="",
        help="RNAFold dot-bracket secondary structure for the sequence.",
    )
    parser.add_argument(
        "--motif-positions",
        default="",
        help="Optional zero-based motif positions, e.g. '17-21' or '6,7,8'.",
    )
    parser.add_argument("--out-dir", default="", help="Single directory for staged outputs.")
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Rerun currently supported plotting scripts before staging outputs.",
    )
    parser.add_argument(
        "--regenerate-artifacts",
        action="store_true",
        help=(
            "Run artifact/training steps before plotting. For ResidualBind VTS1 this "
            "generates scatter_by_mutcount_predictions.npz once."
        ),
    )
    parser.add_argument(
        "--regenerate-plots",
        action="store_true",
        help="Run plotting-only steps before staging outputs.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable for --regenerate. Use the squid env for MAVE-NN plots.",
    )
    parser.add_argument(
        "--mrna-plot-dir",
        default=str(DEFAULT_MRNA_PLOT_DIR),
        help="Directory containing mRNA_RBP notebook plot outputs.",
    )
    parser.add_argument(
        "--random-library-dir",
        dest="random_library_dir",
        default="",
        help=(
            "Optional override directory containing random-library plot outputs. "
            "By default non-synthetic figures are staged from ground_truth_collections/."
        ),
    )
    parser.add_argument(
        "--random-library-glob",
        default="rand_lib_dist*.png",
        help="Glob of random-library figures to stage from --random-library-dir.",
    )
    parser.add_argument(
        "--coefficient-figure",
        default="",
        help="Override coefficient-analysis source PNG.",
    )
    parser.add_argument(
        "--require-all",
        action="store_true",
        help="Fail if any canonical output is missing.",
    )
    args = parser.parse_args()

    preset = apply_collection_preset(args)
    if not args.ground_truth:
        parser.error("--ground-truth is required unless --collection is provided")
    if not args.sequence:
        parser.error("--sequence is required unless --collection is provided")
    if not args.structure:
        parser.error("--structure is required unless --collection is provided")
    if not args.out_dir:
        parser.error("--out-dir is required unless --collection is provided")

    sequence, structure, stem_pairs = validate_inputs(args.sequence, args.structure)
    motif_positions = parse_positions(args.motif_positions)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    commands = []
    run_results = []
    copy_results = []
    if args.regenerate_artifacts:
        if args.collection != "residualbind_vts1":
            raise ValueError("--regenerate-artifacts is currently implemented for --collection residualbind_vts1")
        commands = [[args.python] + cmd[1:] for cmd in VTS1_ARTIFACT_COMMANDS]
        run_results.extend(run_commands(commands, build_subprocess_env()))

    if args.regenerate_plots:
        if args.collection != "residualbind_vts1":
            raise ValueError("--regenerate-plots is currently implemented for --collection residualbind_vts1")
        commands = [[args.python] + cmd[1:] for cmd in VTS1_PLOT_COMMANDS]
        run_results.extend(run_commands(commands, build_subprocess_env()))
        copy_results.extend(copy_vts1_cross_mutrate_outputs())

    if args.regenerate:
        if args.ground_truth != "synthetic":
            raise ValueError(
                "--regenerate currently supports the synthetic cache/schema. "
                "For residualbind/deepsquid, generate the score caches first and stage them here."
            )
        commands = [[args.python] + cmd[1:] for cmd in SUPPORTED_REGEN_COMMANDS]
        run_results.extend(run_commands(commands, build_subprocess_env()))

    if preset:
        staged = stage_collection_figures(Path(preset["collection_dir"]), out_dir, args.require_all)
    else:
        staged = stage_outputs(args, out_dir)
    manifest = {
        "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "repo": str(REPO),
        "inputs": {
            "collection": args.collection,
            "ground_truth": args.ground_truth,
            "sequence": sequence,
            "structure": structure,
            "stem_pairs_zero_based": stem_pairs,
            "motif_positions_zero_based": motif_positions,
        },
        "regenerate": bool(args.regenerate),
        "regenerate_artifacts": bool(args.regenerate_artifacts),
        "regenerate_plots": bool(args.regenerate_plots),
        "commands": run_results,
        "post_plot_copies": copy_results,
        "staged_outputs": staged,
        "notes": [
            "Random-library plots are staged under random_library/ because each GT can have multiple WT/oracle variants.",
            "Named collections stage active figures from their ground_truth_collections/<collection>/figures directory.",
            "Artifact-generation steps run before plotting steps; plotting scripts should consume frozen libraries/results/prediction artifacts.",
        ],
    }
    manifest_path = out_dir / "pipeline_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Saved manifest -> {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
