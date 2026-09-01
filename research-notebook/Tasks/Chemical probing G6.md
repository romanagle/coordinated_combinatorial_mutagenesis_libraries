# [[Chemical probing G6]]

## Now

- [ ] On the cluster, `git pull` inside `G6_chemical_probing` and submit `slurm/run_reactivity_g6.sbatch` against `datasets/Deigan_2009/input_sequences_table.tsv` to fold `deigan_16s`/`deigan_23s`.
- [ ] Compare RNAfold, ShapeKnots, and G6 SHAPE effects on the 24-RNA benchmark.
- [ ] Retrieve and validate the completed Human RNAmap IRES outputs.
- [ ] Clean up the `G6_chemical_probing` codebase for a public GitHub release to send to PI.

## Next

- [ ] Run G6 CYK/ViennaRNA scoring on the exported RMDB dataset, separately for SHAPE and DMS.
- [ ] Stratify RMDB scoring results by reference source, RNA family, and sequence length.

## Waiting

- [ ]

## Ideas

-

## Completed recently

- Reorganized `G6_chemical_probing` into per-dataset `datasets/<Name>/` directories and pushed the reorg, `prepare_shapeknots.py`, and `run_shapeknots_shape_benchmark.py` to `origin/main`.
- Completed the remaining ShapeKnots with/without-SHAPE benchmark folds and plots (24/24).
- Completed the 24-RNA ViennaRNA comparison with and without native Deigan SHAPE pseudoenergies.
- Restricted `plot_g6.py`/`run_reactivity_g6_pipeline.py` to F1-only plotting; pushed to `G6_chemical_probing` (`f55d407`, `72c4d73`).
- Independently re-derived the 24 ShapeKnots RNAs from the Hajdin et al. 2013 primary source and reproduced the first ShapeKnots run's result.
- Submitted the resumable 71-record Human RNAmap IRES run on Cannon; run completed and produced IRES plots.
- Built R-scape 2.6.4 natively on Cannon (Rocky Linux x86_64), resolving the native-build blocker for the IRES pipeline.
- Published the consolidated Human RNAmap G6 workflow to `romanagle/G6_chemical_probing` (commit `82b1146`).
- Removed the default-fixtures/soft-fail exclusions pattern; pipeline now fails fast on bad records.
