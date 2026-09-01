#!/usr/bin/env python
"""
mRNA_RBP/scripts/pipeline/build_ground_truth_collection.py

Single standardized entry point: given a registered oracle, generates every
library the pipeline needs, trains every surrogate needed, produces every
figure, and assembles the final curated collection directory (matching the
"Synthetic GT" layout: figures/<category>/... + libraries_used_for_figures/
+ manifest.json + README.md) -- one command, no manual copy/curation step,
no per-RBP bespoke scripts.

For oracles with a high/low natural-probe WT variant (VTS1, HuR, QKI -- see
oracles.oracle_uses_wt_activity), both contexts are run automatically and
their figures are suffixed _high/_low. Oracles with one fixed WT (mrna,
MSI1) run once, no suffix.

Two conda envs are used because they are genuinely required on this machine
(see run_gt_pipeline.sh for why): toehold_gpu (working torch+CUDA, scores
real ResidualBind oracles) and squid (working tensorflow/mavenn, trains
GE/MAVE-NN surrogates and does all plotting).

Usage:
    python mRNA_RBP/scripts/pipeline/build_ground_truth_collection.py --oracle hur --collection_name "ResidualBind oracle HuR"
    python mRNA_RBP/scripts/pipeline/build_ground_truth_collection.py --oracle vts1 --collection_name "ResidualBind oracle VTS1"
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from mRNA_RBP.src.oracles import (
    MRNA_ORACLE, MRNA_NEGATIVE_CONTROL_ORACLE, TWISTER_ORACLE, SURROGATE_ORACLE,
    DEEPSQUID_HUR_ORACLE, DEEPSQUID_VTS1_ORACLE,
    ORACLE_SHORT_NAME, default_output_base,
    normalize_oracle_name, oracle_uses_wt_activity, primary_gt_key,
)

# Synthetic GTs with no natural-probe/region-class concept (no real stem or
# motif region to bucket a random library by) -- region-classes stage is
# meaningless for these, same as MRNA_ORACLE.
_NO_REGION_CLASSES_ORACLES = (MRNA_ORACLE, MRNA_NEGATIVE_CONTROL_ORACLE)

# Oracles whose build_oracle() constructs a SurrogateMAVENNOracle (needs
# mavenn/tensorflow) rather than a live ResidualBind ensemble (needs torch)
# -- generate_libraries.py must run under the squid env for these, not
# toehold_gpu (which has no mavenn installed).
_MAVENN_BACKED_ORACLES = (SURROGATE_ORACLE, TWISTER_ORACLE, DEEPSQUID_HUR_ORACLE,
                          DEEPSQUID_VTS1_ORACLE)
# Deep squid treated as its own oracle has no third distillation layer of its
# own -- skip that stage for both.
_SKIP_VARIED_MUTRATE_LAYER = (DEEPSQUID_HUR_ORACLE, DEEPSQUID_VTS1_ORACLE)
# Both deep-squid oracles' region-classes figures borrow their raw-score
# anchor from the real ResidualBind oracle they were distilled from (see
# generate_random_region_score_cache.py) -- neither is skipped.
_SKIP_REGION_CLASSES = ()
# Oracles that are themselves a deep-squid surrogate get a "deepSQUID <NAME>"
# collection name (matching the deepSQUID MSI1/Twister naming convention)
# instead of the default "ResidualBind oracle <NAME>".
_DEEPSQUID_ORACLES = (DEEPSQUID_HUR_ORACLE, DEEPSQUID_VTS1_ORACLE)

TORCH_PY = "/home/nagle/miniconda3/envs/toehold_gpu/bin/python3"
SQUID_PY = "/home/nagle/miniconda3/envs/squid/bin/python3.7"
SQUID_LD_LIBRARY_PATH = "/home/nagle/miniconda3/envs/squid/lib:/usr/local/cuda-11.2/lib64"

COLLECTION_ROOT = REPO / "mRNA_RBP" / "outputs" / "ground_truth_collections"

CATEGORY_MAP = {
    # source filename (as produced by the plot scripts) -> figure category
    "library_distributions.png": "library_distributions",
    "activity_balanced_lib_dist.png": "library_distributions",
    "type3_lib_dist.png": "library_distributions",
    "rand_lib_dist.png": "library_distributions",
    "pairwise_library_distribution.png": "library_distributions",
    "coefficients_gt.png": "coefficients",
    "coefficients_surrogate.png": "coefficients",
    "rho_vs_libsize_type3.png": "library_size_sweep",
    "model_comparison_bar_type3.png": "model_comparison",
    "scatter_by_mutcount.png": "mutation_rate_sweep",
}


def log(msg: str) -> None:
    print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def run(python: str, script: str, args: list, env_extra: dict | None = None) -> None:
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
    env.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")
    if env_extra:
        env.update(env_extra)
    cmd = [python, script] + [str(a) for a in args]
    log("RUN " + " ".join(cmd))
    completed = subprocess.run(cmd, cwd=str(REPO), env=env)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed (exit {completed.returncode}): {' '.join(cmd)}")


def squid_env() -> dict:
    return {"LD_LIBRARY_PATH": f"{SQUID_LD_LIBRARY_PATH}:{os.environ.get('LD_LIBRARY_PATH', '')}"}


def _cli_oracle_arg(oracle_name: str) -> str:
    """Short CLI-facing oracle name (the choices=[] lists in each script use
    these short aliases, not the canonical internal identity string)."""
    return "mrna" if oracle_name == MRNA_ORACLE else ORACLE_SHORT_NAME.get(oracle_name, oracle_name)


def run_data_stages(oracle_name: str, wt_activity: str, n_instances: int) -> None:
    common = ["--oracle", _cli_oracle_arg(oracle_name), "--wt_activity", wt_activity]

    gen_lib_py, gen_lib_env = (
        (SQUID_PY, squid_env()) if oracle_name in _MAVENN_BACKED_ORACLES
        else (TORCH_PY, None)
    )
    run(gen_lib_py, "mRNA_RBP/scripts/pipeline/generate_libraries.py", common + ["--n_instances", n_instances],
        env_extra=gen_lib_env)
    if oracle_name not in _SKIP_VARIED_MUTRATE_LAYER:
        run(TORCH_PY, "mRNA_RBP/scripts/pipeline/generate_varied_mutrate_library.py", common)

    run(SQUID_PY, "mRNA_RBP/scripts/pipeline/lib_size_spearman.py", common + ["--n_instances", n_instances],
        env_extra=squid_env())
    if oracle_name not in _SKIP_VARIED_MUTRATE_LAYER:
        run(SQUID_PY, "mRNA_RBP/scripts/pipeline/train_surrogate_varied_mutrate.py", common, env_extra=squid_env())
    run(SQUID_PY, "mRNA_RBP/scripts/pipeline/cross_mutrate_eval.py", common + ["--n_instances", n_instances],
        env_extra=squid_env())
    run(SQUID_PY, "mRNA_RBP/scripts/pipeline/generate_scatter_by_mutcount_predictions.py", common + ["--force"],
        env_extra=squid_env())
    run(SQUID_PY, "mRNA_RBP/scripts/pipeline/generate_model_comparison_score_cache.py", common,
        env_extra=squid_env())
    if oracle_name not in _NO_REGION_CLASSES_ORACLES and oracle_name not in _SKIP_REGION_CLASSES:
        # Needs the torch env now -- computing wt_raw_score for the
        # raw-score companion figure requires building the live
        # ResidualBind ensemble (real oracles, or the real oracle a
        # deep-squid surrogate was distilled from). squid's torch install
        # can't import (see module docstring in oracles.py).
        run(TORCH_PY, "mRNA_RBP/scripts/pipeline/generate_random_region_score_cache.py", common)


def run_data_stages_twister(n_instances: int) -> None:
    """Twister has no live black-box oracle to distill from -- the real
    Kobori & Yokobayashi (2016) dataset only covers mutation orders 0/1/2
    (WT + all singles + all doubles), so there's nothing to query for a
    varied-mutrate (3-10 mutation) pool the way generate_varied_mutrate_
    library.py does for MSI1/VTS1/HuR/QKI. Deep squid is instead trained
    directly on the real measurements (parse_twister_data.py ->
    train_twister_deepsquid.py), which together replace that
    generate_varied_mutrate_library.py / train_surrogate_varied_mutrate.py
    pair. Everything here needs mavenn/tensorflow (SQUID_PY) -- there's no
    torch-side real oracle to score, so TORCH_PY is never used for Twister.
    """
    run(SQUID_PY, "mRNA_RBP/parse_twister_data.py", [], env_extra=squid_env())
    run(SQUID_PY, "mRNA_RBP/train_twister_deepsquid.py", [], env_extra=squid_env())

    common = ["--oracle", "twister_ribozyme"]
    run(SQUID_PY, "mRNA_RBP/scripts/pipeline/generate_libraries.py", common + ["--n_instances", n_instances],
        env_extra=squid_env())
    run(SQUID_PY, "mRNA_RBP/scripts/pipeline/lib_size_spearman.py", common + ["--n_instances", n_instances],
        env_extra=squid_env())
    run(SQUID_PY, "mRNA_RBP/scripts/pipeline/cross_mutrate_eval.py", common + ["--n_instances", n_instances],
        env_extra=squid_env())
    run(SQUID_PY, "mRNA_RBP/scripts/pipeline/generate_scatter_by_mutcount_predictions.py", common + ["--force"],
        env_extra=squid_env())
    run(SQUID_PY, "mRNA_RBP/scripts/pipeline/generate_model_comparison_score_cache.py", common,
        env_extra=squid_env())


def run_plot_stages(oracle_name: str, wt_activity: str, out_base: Path, stage_dir: Path,
                    oracle_label: str) -> None:
    stage_dir.mkdir(parents=True, exist_ok=True)
    score_key = primary_gt_key(oracle_name)
    common = ["--oracle", _cli_oracle_arg(oracle_name), "--wt_activity", wt_activity, "--out_dir", stage_dir]

    run(SQUID_PY, "mRNA_RBP/scripts/figures/core/plot_library_distributions.py", common, env_extra=squid_env())
    run(SQUID_PY, "mRNA_RBP/scripts/figures/core/plot_coefficients.py", common, env_extra=squid_env())
    run(SQUID_PY, "mRNA_RBP/scripts/figures/core/plot_scatter_by_mutcount.py", common, env_extra=squid_env())
    run(SQUID_PY, "mRNA_RBP/scripts/figures/core/plot_rho_vs_libsize_type3.py",
        ["--json_path", out_base / "lib_size_spearman_results.json",
         "--gt_key", score_key, "--out_dir", stage_dir],
        env_extra=squid_env())
    run(SQUID_PY, "mRNA_RBP/scripts/figures/core/bar_surrogate_models_type3.py",
        ["--score_cache", out_base / "model_comparison_score_cache.npz",
         "--oracle_label", oracle_label, "--out_dir", stage_dir],
        env_extra=squid_env())
    cross_json = out_base / "cross_mutrate_results.json"
    if cross_json.exists():
        run(SQUID_PY, "mRNA_RBP/scripts/figures/core/plot_cross_mutrate.py",
            ["--results_json", cross_json, "--gt_key", score_key,
             "--out_base", out_base, "--out_dir", stage_dir,
             "--out_prefix", f"{ORACLE_SHORT_NAME.get(oracle_name, oracle_name)}_"],
            env_extra=squid_env())


def stage_figures(stage_dir: Path, collection_dir: Path, suffix: str, manifest: list) -> None:
    fig_root = collection_dir / "figures"
    for src in sorted(stage_dir.glob("*.png")):
        if "cross_mutrate_libsize" in src.name:
            # Per-surrogate-config generalization panels are a diagnostic
            # byproduct of plot_cross_mutrate.py, not a curated figure --
            # only the summary heatmap gets promoted (matches the existing
            # Synthetic GT / MSI1 collections).
            continue
        category = CATEGORY_MAP.get(src.name)
        if category is None:
            if "cross_mutrate" in src.name:
                category = "mutation_rate_sweep"
            elif "coefficients" in src.name:
                category = "coefficients"
            else:
                category = "misc"
        stem, ext = os.path.splitext(src.name)
        dst_name = f"{stem}{suffix}{ext}"
        dst = fig_root / category / dst_name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        manifest.append({"source": str(src), "destination": str(dst)})
    coef_map_dir = stage_dir / "coefficients_map"
    if coef_map_dir.is_dir():
        for src in sorted(coef_map_dir.glob("*.png")):
            dst = fig_root / "coefficients" / src.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            manifest.append({"source": str(src), "destination": str(dst)})


def stage_coefficient_similarity(stage_dir: Path, collection_dir: Path, suffix: str,
                                 manifest: list, similarity_summary: dict) -> None:
    """Copy plot_coefficients.py's coefficients_similarity_*.json (mean
    cosine similarity, additive + pairwise weights, GT/oracle vs surrogate)
    into libraries_used_for_figures/ and fold its numbers into
    similarity_summary for the README."""
    lib_dir = collection_dir / "libraries_used_for_figures"
    for src in sorted(stage_dir.glob("coefficients_similarity_*.json")):
        dst = lib_dir / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        manifest.append({"source": str(src), "destination": str(dst)})
        data = json.loads(src.read_text())
        label = src.stem[len("coefficients_similarity_"):] + suffix
        similarity_summary[label] = {
            "mean_cosine_similarity_additive": data.get("mean_cosine_similarity_additive"),
            "mean_cosine_similarity_pairwise": data.get("mean_cosine_similarity_pairwise"),
            "sign_correction": data.get("sign_correction"),
            "mean_cosine_similarity_additive_sign_corrected":
                data.get("mean_cosine_similarity_additive_sign_corrected"),
            "mean_cosine_similarity_pairwise_sign_corrected":
                data.get("mean_cosine_similarity_pairwise_sign_corrected"),
        }


def copy_supporting_artifacts(out_base: Path, collection_dir: Path, suffix: str, manifest: list) -> None:
    """Copy the small/medium result artifacts (not the multi-GB raw pools)."""
    lib_dir = collection_dir / "libraries_used_for_figures"
    small_names = [
        "lib_size_spearman_results.json",
        "cross_mutrate_results.json",
        "scatter_by_mutcount_predictions.npz",
        "model_comparison_score_cache.npz",
    ]
    for name in small_names:
        src = out_base / name
        if src.exists():
            dst = lib_dir / f"{os.path.splitext(name)[0]}{suffix}{os.path.splitext(name)[1]}"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            manifest.append({"source": str(src), "destination": str(dst)})

    inst_src = out_base / "instance_00"
    if inst_src.is_dir():
        inst_dst = lib_dir / f"instance_00{suffix}"
        for name in ("gt_params.npz", "ssm.npz", "activity_balanced.npz", "type2.npz",
                    "pairwise_lib.npz", "type3.npz", "wt_seq.txt"):
            src = inst_src / name
            if src.exists():
                dst = inst_dst / name
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                manifest.append({"source": str(src), "destination": str(dst)})
        for mut_dir in sorted(inst_src.glob("mut*")):
            lib20k = mut_dir / "lib_20000.npz"
            if lib20k.exists():
                dst = inst_dst / mut_dir.name / "lib_20000.npz"
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(lib20k, dst)
                manifest.append({"source": str(lib20k), "destination": str(dst)})

    surrogate_coefs = out_base / "surrogate_coefs"
    if surrogate_coefs.is_dir():
        dst_dir = lib_dir / f"surrogate_coefs{suffix}"
        dst_dir.mkdir(parents=True, exist_ok=True)
        for src in surrogate_coefs.glob("*.npz"):
            dst = dst_dir / src.name
            shutil.copy2(src, dst)
            manifest.append({"source": str(src), "destination": str(dst)})


def run_region_classes_stage(oracle_name: str, collection_dir: Path, manifest: list) -> None:
    """Random-library-by-mutated-region figure(s), run once per oracle (not
    per wt_activity) since the high/low panels share one x-axis and need
    both out_base dirs loaded together -- see
    plot_residualbind_vts1_rand_region_distributions.py. Not applicable to
    Twister either -- there's no natural-probe/region-class concept for a
    ribozyme construct with one fixed doped region."""
    if (oracle_name in _NO_REGION_CLASSES_ORACLES or oracle_name == TWISTER_ORACLE
            or oracle_name in _SKIP_REGION_CLASSES):
        return
    short = ORACLE_SHORT_NAME.get(oracle_name, oracle_name)
    fig_dir = collection_dir / "figures" / "library_distributions"
    fig_dir.mkdir(parents=True, exist_ok=True)
    run(SQUID_PY, "mRNA_RBP/scripts/figures/core/plot_residualbind_vts1_rand_region_distributions.py",
        ["--oracle", _cli_oracle_arg(oracle_name), "--wt", "both", "--out_dir", fig_dir],
        env_extra=squid_env())
    for src in sorted(fig_dir.glob(f"rand_lib_dist_{short}_oracle_region_classes*.png")):
        manifest.append({"source": str(src), "destination": str(src)})

    cache_dst_dir = collection_dir / "libraries_used_for_figures" / "random_region_score_cache"
    for kind, wt_flag in (("high", "high"), ("low", "low")):
        if not oracle_uses_wt_activity(oracle_name) and kind == "low":
            continue
        out_base = Path(default_output_base(str(REPO / "mRNA_RBP"), oracle_name, wt_flag))
        cache_suffix = "" if kind == "high" else "_low_wt"
        src = out_base / f"{short}_natural_random_library_scores{cache_suffix}.npz"
        if src.exists():
            dst = cache_dst_dir / src.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            manifest.append({"source": str(src), "destination": str(dst)})


def write_readme_and_manifest(collection_dir: Path, manifest: list, oracle_name: str,
                              similarity_summary: dict | None = None) -> None:
    n_figures = len(list((collection_dir / "figures").rglob("*.png")))
    n_artifacts = len(manifest) - n_figures
    similarity_summary = similarity_summary or {}
    (collection_dir / "manifest.json").write_text(json.dumps({
        "label": collection_dir.name,
        "key": oracle_name,
        "created": dt.datetime.now(dt.timezone.utc).isoformat(),
        "note": "Built by mRNA_RBP/scripts/pipeline/build_ground_truth_collection.py -- one standardized command, "
                "no manual curation step.",
        "coefficient_similarity": similarity_summary,
        "entries": manifest,
    }, indent=2))

    similarity_lines = ""
    if similarity_summary:
        rows = "\n".join(
            f"| `{label}` | {v['mean_cosine_similarity_additive']:.4f} | "
            f"{v['mean_cosine_similarity_pairwise']:.4f} | "
            f"{v['mean_cosine_similarity_additive_sign_corrected']:.4f} | "
            f"{v['mean_cosine_similarity_pairwise_sign_corrected']:.4f} | "
            f"{'flipped' if v.get('sign_correction', 1) < 0 else '—'} |"
            for label, v in sorted(similarity_summary.items())
        )
        similarity_lines = (
            "\n## Coefficient similarity (GT/oracle vs surrogate)\n\n"
            "Mean cosine similarity between the additive (alpha) and pairwise (beta) weight "
            "matrices, at instance 0 / mut_rate 10% / lib_size 20000 (see "
            "`libraries_used_for_figures/coefficients_similarity_*.json`). Cosine similarity is "
            "scale-invariant but not sign-invariant, and MAVE-NN's GE nonlinearity has a real "
            "gauge freedom `(alpha, J, b) -> (-alpha, -J, -b)` that reproduces identical "
            "predictions -- the *sign-corrected* columns apply the single global sign (to both "
            "alpha and beta jointly) that maximizes agreement, so a surrogate that matched up to "
            "this legitimate flip isn't scored as if it learned the opposite direction.\n\n"
            "| condition | cos sim (additive) | cos sim (pairwise) | "
            "sign-corrected (additive) | sign-corrected (pairwise) | sign flipped? |\n"
            "|---|---|---|---|---|---|\n"
            f"{rows}\n"
        )

    (collection_dir / "README.md").write_text(
        f"# {collection_dir.name}\n\n"
        "Figures are in `figures/`; copied cache/model/weight artifacts are in "
        "`libraries_used_for_figures/`. Built end-to-end by "
        "`mRNA_RBP/scripts/pipeline/build_ground_truth_collection.py --oracle "
        f"{oracle_name}`. The manifest records every file included.\n\n"
        f"- Figures: `{n_figures}`\n"
        f"- Cached artifacts: `{n_artifacts}`\n"
        f"{similarity_lines}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", required=True)
    parser.add_argument("--collection_name", default=None,
                        help="Defaults to a name derived from the oracle")
    parser.add_argument("--n_instances", type=int, default=1)
    args = parser.parse_args()

    oracle_name = normalize_oracle_name(args.oracle)
    short = ORACLE_SHORT_NAME.get(oracle_name, oracle_name)
    if args.collection_name:
        collection_name = args.collection_name
    elif oracle_name == MRNA_ORACLE:
        collection_name = "Synthetic GT"
    elif oracle_name == MRNA_NEGATIVE_CONTROL_ORACLE:
        collection_name = "Synthetic GT Negative Control"
    elif oracle_name == TWISTER_ORACLE:
        # No live black-box oracle counterpart to pair with (unlike
        # MSI1/VTS1/HuR/QKI, which each get a "ResidualBind oracle <NAME>"
        # collection alongside their "deepSQUID <NAME>" one) -- deep squid
        # trained on the real data *is* the only oracle here.
        collection_name = "deepSQUID Twister"
    elif oracle_name in _DEEPSQUID_ORACLES:
        # short is e.g. "deepsquid_vts1" -- take the part after "deepsquid_".
        collection_name = f"deepSQUID {short.split('_', 1)[-1].upper()}"
    else:
        collection_name = f"ResidualBind oracle {short.upper()}"
    collection_dir = COLLECTION_ROOT / collection_name
    collection_dir.mkdir(parents=True, exist_ok=True)

    wt_activities = ["high", "low"] if oracle_uses_wt_activity(oracle_name) else ["high"]
    multi = len(wt_activities) > 1

    manifest: list = []
    similarity_summary: dict = {}
    for wt in wt_activities:
        log(f"=== oracle={oracle_name}  wt_activity={wt} ===")
        if oracle_name == TWISTER_ORACLE:
            run_data_stages_twister(args.n_instances)
        else:
            run_data_stages(oracle_name, wt, args.n_instances)

        out_base = Path(default_output_base(str(REPO / "mRNA_RBP"), oracle_name, wt))
        stage_dir = out_base / "_figure_staging"
        run_plot_stages(oracle_name, wt, out_base, stage_dir, oracle_label=collection_name)

        suffix = f"_{wt}" if multi else ""
        stage_figures(stage_dir, collection_dir, suffix, manifest)
        stage_coefficient_similarity(stage_dir, collection_dir, suffix, manifest, similarity_summary)
        copy_supporting_artifacts(out_base, collection_dir, suffix, manifest)

    run_region_classes_stage(oracle_name, collection_dir, manifest)

    write_readme_and_manifest(collection_dir, manifest, oracle_name, similarity_summary)
    log(f"=== DONE -> {collection_dir} ===")


if __name__ == "__main__":
    main()
