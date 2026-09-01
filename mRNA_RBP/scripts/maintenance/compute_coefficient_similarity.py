#!/usr/bin/env python
"""
mRNA_RBP/scripts/maintenance/compute_coefficient_similarity.py

Backfill mean cosine similarity (additive alpha + pairwise beta weight
matrices, GT/oracle view vs surrogate) for already-built collections under
outputs/ground_truth_collections/, without touching their existing frozen
PNG figures. Computes from artifacts each collection already has cached in
its own libraries_used_for_figures/ (gt_params.npz + ssm.npz +
pairwise_lib.npz + surrogate_coefs*/), at the same default condition
plot_coefficients.py uses (instance 0, mut_rate 10%, lib_size 20000,
nonlinear additive+pairwise surrogate) -- no live oracle, no retraining.

For every collection this discovers each instance_00[_<suffix>] present,
reads its oracle identity straight out of gt_params.npz (or treats the
"Synthetic GT" collection as the analytic MrnaRbpGroundTruth), and calls the
exact same view-construction / metric code plot_coefficients.py uses, so the
numbers are identical to what a fresh run of that script would report.

Writes, per collection:
  <collection>/coefficient_similarity.json   -- one entry per condition found
  <collection>/manifest.json                 -- adds/overwrites a
                                                  "coefficient_similarity" key
  <collection>/README.md                     -- adds/replaces a
                                                  "## Coefficient similarity"
                                                  section

Usage:
    python mRNA_RBP/scripts/maintenance/compute_coefficient_similarity.py
    python mRNA_RBP/scripts/maintenance/compute_coefficient_similarity.py --collection "ResidualBind oracle VTS1"
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from mRNA_RBP.src.oracles import MRNA_ORACLE, ORACLE_SHORT_NAME, normalize_oracle_name, sequence_config_for_oracle
from mRNA_RBP.scripts.figures.core.plot_coefficients import build_gt_and_surrogate_views
import mRNA_RBP.src.coef_metrics as coef_metrics

COLLECTION_ROOT = REPO / "mRNA_RBP" / "outputs" / "ground_truth_collections"
INSTANCE = 0
MUT_RATE = 10
LIB_SIZE = 20000


def _condition_label(oracle_name: str, wt_activity: str) -> str:
    short = "synthetic" if oracle_name == MRNA_ORACLE else ORACLE_SHORT_NAME.get(oracle_name, oracle_name)
    return f"{short}_{wt_activity}" if wt_activity != "default" else short


def _find_conditions(collection_dir: Path):
    """Yield (suffix, oracle_name, wt_activity, inst_dir, coef_dir) for every
    instance_00[_<suffix>] this collection has cached."""
    lib_dir = collection_dir / "libraries_used_for_figures"
    if collection_dir.name == "Synthetic GT":
        inst_dir = lib_dir / "instance_00"
        if inst_dir.is_dir():
            yield "", MRNA_ORACLE, "default", inst_dir, lib_dir / "surrogate_coefs"
        return

    for inst_dir in sorted(lib_dir.glob("instance_00*")):
        if not inst_dir.is_dir():
            continue
        suffix = inst_dir.name[len("instance_00"):]  # "", "_high", "_low"
        gt_params_path = inst_dir / "gt_params.npz"
        if not gt_params_path.is_file():
            print(f"  [skip] {inst_dir} -- no gt_params.npz cached, oracle identity unknown")
            continue
        d = np.load(gt_params_path)
        oracle_field = str(d["oracle"][0])
        wt_activity = str(d["wt_activity"][0])
        try:
            oracle_name = normalize_oracle_name(oracle_field)
        except ValueError:
            print(f"  [skip] {inst_dir} -- unrecognized oracle {oracle_field!r}")
            continue
        coef_dir = lib_dir / f"surrogate_coefs{suffix}"
        yield suffix, oracle_name, wt_activity, inst_dir, coef_dir


def compute_for_collection(collection_dir: Path) -> dict:
    print(f"=== {collection_dir.name} ===")
    similarity_summary: dict = {}
    for suffix, oracle_name, wt_activity, inst_dir, coef_dir in _find_conditions(collection_dir):
        score_key_glob = f"coefs_k{INSTANCE:02d}_mut{MUT_RATE:02d}_lib{LIB_SIZE}_nonlinear_additive_p_pairwise_*.npz"
        matches = sorted(coef_dir.glob(score_key_glob)) if coef_dir.is_dir() else []
        if not matches:
            print(f"  [skip] {inst_dir} -- no matching surrogate coefs in {coef_dir}")
            continue
        coef_path = matches[0]

        if oracle_name == MRNA_ORACLE:
            seq, stem_pairs, motif_positions = sequence_config_for_oracle(oracle_name)
        else:
            d = np.load(inst_dir / "gt_params.npz")
            stem_pairs = [tuple(int(v) for v in row) for row in d["edges"]]
            seq, motif_positions = None, None

        gt, sur_alpha, sur_beta = build_gt_and_surrogate_views(
            oracle_name, INSTANCE, seq, stem_pairs, motif_positions, str(inst_dir), str(coef_path),
        )
        similarity = coef_metrics.compute_coefficient_similarity(
            gt.alpha, sur_alpha, gt.beta, sur_beta, stem_pairs,
        )
        label = _condition_label(oracle_name, wt_activity)
        result = {
            "oracle": oracle_name, "wt_activity": wt_activity,
            "instance": INSTANCE, "mut_rate": MUT_RATE, "lib_size": LIB_SIZE,
            "coef_path": str(coef_path.relative_to(REPO)),
            **similarity,
        }
        similarity_summary[label] = result
        flip_note = " (sign-flipped)" if similarity["sign_correction"] < 0 else ""
        print(f"  {label}: additive={similarity['mean_cosine_similarity_additive']:.4f}  "
              f"pairwise={similarity['mean_cosine_similarity_pairwise']:.4f}"
              f"  |  sign-corrected: additive={similarity['mean_cosine_similarity_additive_sign_corrected']:.4f}"
              f"  pairwise={similarity['mean_cosine_similarity_pairwise_sign_corrected']:.4f}{flip_note}")

    if not similarity_summary:
        print("  (nothing computable -- no cached gt_params/ssm/pairwise_lib + matching surrogate coefs)")
        return similarity_summary

    (collection_dir / "coefficient_similarity.json").write_text(
        json.dumps(similarity_summary, indent=2)
    )
    _update_manifest(collection_dir, similarity_summary)
    _update_readme(collection_dir, similarity_summary)
    return similarity_summary


def _update_manifest(collection_dir: Path, similarity_summary: dict) -> None:
    manifest_path = collection_dir / "manifest.json"
    if not manifest_path.is_file():
        return
    data = json.loads(manifest_path.read_text())
    data["coefficient_similarity"] = {
        label: {
            "mean_cosine_similarity_additive": v["mean_cosine_similarity_additive"],
            "mean_cosine_similarity_pairwise": v["mean_cosine_similarity_pairwise"],
            "sign_correction": v["sign_correction"],
            "mean_cosine_similarity_additive_sign_corrected":
                v["mean_cosine_similarity_additive_sign_corrected"],
            "mean_cosine_similarity_pairwise_sign_corrected":
                v["mean_cosine_similarity_pairwise_sign_corrected"],
        }
        for label, v in similarity_summary.items()
    }
    manifest_path.write_text(json.dumps(data, indent=2))


_SECTION_RE = re.compile(r"\n## Coefficient similarity\b.*?(?=\n## |\Z)", re.DOTALL)


def _update_readme(collection_dir: Path, similarity_summary: dict) -> None:
    readme_path = collection_dir / "README.md"
    if not readme_path.is_file():
        return
    text = readme_path.read_text()
    text = _SECTION_RE.sub("", text)

    rows = "\n".join(
        f"| `{label}` | {v['mean_cosine_similarity_additive']:.4f} | "
        f"{v['mean_cosine_similarity_pairwise']:.4f} | "
        f"{v['mean_cosine_similarity_additive_sign_corrected']:.4f} | "
        f"{v['mean_cosine_similarity_pairwise_sign_corrected']:.4f} | "
        f"{'flipped' if v.get('sign_correction', 1) < 0 else '—'} |"
        for label, v in sorted(similarity_summary.items())
    )
    section = (
        "\n## Coefficient similarity (GT/oracle vs surrogate)\n\n"
        "Mean cosine similarity between the additive (alpha) and pairwise (beta) weight "
        "matrices, at instance 0 / mut_rate 10% / lib_size 20000, computed from cached "
        "artifacts by `mRNA_RBP/scripts/maintenance/compute_coefficient_similarity.py` (see "
        "`coefficient_similarity.json`). Cosine similarity is scale-invariant but not "
        "sign-invariant, and MAVE-NN's GE nonlinearity has a real gauge freedom "
        "`(alpha, J, b) -> (-alpha, -J, -b)` that reproduces identical predictions -- the "
        "*sign-corrected* columns apply the single global sign (to both alpha and beta "
        "jointly) that maximizes agreement, so a surrogate that matched up to this "
        "legitimate flip isn't scored as if it learned the opposite direction.\n\n"
        "| condition | cos sim (additive) | cos sim (pairwise) | "
        "sign-corrected (additive) | sign-corrected (pairwise) | sign flipped? |\n"
        "|---|---|---|---|---|---|\n"
        f"{rows}\n"
    )
    readme_path.write_text(text.rstrip("\n") + "\n" + section)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", default=None,
                        help="Name of one collection dir under ground_truth_collections/ "
                             "(default: all, skipping anything with 'legacy' in the name)")
    args = parser.parse_args()

    if args.collection:
        targets = [COLLECTION_ROOT / args.collection]
    else:
        targets = sorted(p for p in COLLECTION_ROOT.iterdir()
                         if p.is_dir() and "legacy" not in p.name.lower())

    for collection_dir in targets:
        compute_for_collection(collection_dir)


if __name__ == "__main__":
    main()
