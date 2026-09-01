# Reusable scientific code

This directory contains importable analysis modules, not executable workflows.

- `oracles.py`: oracle definitions, routing, model loading, and output-root selection.
- `sequence_configs.py`: biological sequences, structures, and motif coordinates.
- `gt_init.py` and `ground_truth.py`: synthetic ground-truth construction.
- `evaluate.py` and `coef_metrics.py`: evaluation and coefficient comparison utilities.
- `seq_utils.py`: sequence encoding helpers.
- `viz.py`: shared visualization utilities.

Executable entry points belong in `../scripts/`.
