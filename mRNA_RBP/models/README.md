# Canonical models

This directory contains fitted model weights required by runtime oracle code.

`residualbind/<target>/<wt>/varied_mutrate_nonlin_pairwise/` contains the retained MAVE-NN model, metadata, and training table for each HuR and VTS1 WT state. `oracles.py` loads deepSQUID HuR and VTS1 models from these paths.

Training runs remain under `../runs/`; models should be promoted here deliberately after retraining rather than loaded from an incidental run directory.
