#!/usr/bin/env python3
"""THROWAWAY PROTOTYPE: isolate the GT half of the paired coefficient map."""

from pathlib import Path

from PIL import Image


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = (
    ROOT
    / "outputs/ground_truth_collections/Synthetic GT/figures/coefficients"
    / "coefficients_groundtruth_vs_surrogate_synthetic.png"
)
OUTPUT = HERE / "outputs/synthetic_gt_coefficients_only.png"


def main():
    image = Image.open(SOURCE)
    midpoint = image.width // 2
    ground_truth_panel = image.crop((0, 0, midpoint, image.height))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    ground_truth_panel.save(OUTPUT)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
