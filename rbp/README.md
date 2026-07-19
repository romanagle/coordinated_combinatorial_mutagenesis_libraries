# rbp-benchmark

Model **RNAcompete** RBP-binding and **benchmark multiple methods** under one pipeline. Currently:
**ResidualBind** (PyTorch port) and **AlphaGenome encoder fine-tuning** (modes A/B). Adding a method
is a single new module + config — no changes to the training harness.

Self-contained suite that does **not** modify the vendored `residualbind/` or `alphagenome-encoder-ft/`
repos; it consumes `alphagenome_encoder_ft` / `alphagenome_pytorch` as installed libraries from the
in-repo env `.venv` (Python 3.13, torch cu126, 4× A100). Run with `.venv/bin/python`.

## Layout

```
src/rbp_bench/
  data.py        RNACompeteDataset, build_loaders, normalize_targets   (model-agnostic)
  metrics.py     pearson                                               (model-agnostic)
  trainer.py     ONE harness: evaluate, per-stage early-stop/LR-decay, run_benchmark, summary.json
  report.py      join results/*/summary.json into a comparison table
  models/
    base.py          BenchModel: forward(x:(B,L,4))->(B,), set_stage(name)
    residualbind.py  ResidualBindTorch + build()
    alphagenome_ft.py AlphaGenome encoder + RBPHead + encode_features + build()
    __init__.py      REGISTRY: name -> build()
configs/  residualbind.json, alphagenome_modeA.json, alphagenome_modeB.json
scripts/  train.py (--config/--model), compare.py
tests/
```

## Design

- **One `BenchModel` interface.** Every model is `forward(x:(B,L,4)) -> (B,)` with a `set_stage(name)`
  that picks which params train. ResidualBind has one stage (`all`); AlphaGenome-FT has two
  (`head` frozen-encoder, then `all`). The trainer optimises whatever has `requires_grad`.
- **Config-driven, model-agnostic trainer.** Data, stages, LR schedule, early stopping, checkpoints
  and `summary.json` all live in `trainer.py`; a config names the model + its `model_args` + stages.
- **Registry.** `scripts/train.py --config X` looks up `config["model"]` in `models.REGISTRY`.
  Add a model: drop `models/foo.py` with `build(model_args, device)`, register it, add `configs/foo.json`.

## Data

Uses the same `rnacompete2009.h5` as ResidualBind (nested per RBP; `X` is `(N, 9, 39)` = 4 nucleotide
one-hot + 5 RNAplfold structure channels — the pipeline keeps the first 4 = `ss_type="seq"`). Targets
are `log_norm`-normalised (params fit on train, reused on val/test). RBPs: VTS1, Fusip, HuR, PTB,
RBM4, SF2, SLM2, U1A, YB1.

## AlphaGenome-FT: the resolution sweep

The encoder downsamples 128× via SAME-padding ceil pooling, so a ~39 nt probe is accepted directly.
`model_args.feature_stage` selects **which downsampling stage's embeddings to use** — the number is
bp-per-bin. Lower = finer (more within-probe positions, fewer channels); `null` = the final 128 bp
trunk (whole probe → one vector). All configs fix `sequence_length=48`, two-stage (`head`→`all`),
`flatten` pooling, and `log_norm` (matching ResidualBind), so **resolution is the only variable**:

| Config | `feature_stage` | positions @ L=48 | channels |
|--------|-----------------|------------------|----------|
| `alphagenome_trunk.json` | `null` | 1   | 1536 |
| `alphagenome_bin16.json` | `16`   | 3   | 1280 |
| `alphagenome_bin8.json`  | `8`    | 6   | 1152 |
| `alphagenome_bin4.json`  | `4`    | 12  | 1024 |

Zero-padding positions are `N` to the encoder (all-zero one-hot), so the probe sits in the middle
with short `N` flanks. Finer stages keep within-probe position, which RBP motif prediction should
benefit from — the hypothesis under test.

## Usage

```bash
cd rbp_benchmark
PY=.venv/bin/python
H5=/home/shared/data/rnacompete/rnacompete2009.h5
W=/home/shared/models/alphagenome_pytorch/model_all_folds.safetensors

# baseline
$PY scripts/train.py --config configs/residualbind.json --h5_path $H5 --all_rbps \
    --output_dir results/residualbind

# AlphaGenome-FT resolution sweep (weights path is machine-specific -> pass on CLI)
for r in trunk bin16 bin8 bin4; do
  $PY scripts/train.py --config configs/alphagenome_$r.json --h5_path $H5 --pretrained_weights $W \
      --all_rbps --output_dir results/agft_$r
done

# comparison table (one column per results/<label>)
$PY scripts/compare.py --results_dir results --csv results/comparison.csv          # test Pearson
$PY scripts/compare.py --results_dir results --metric best_val_pearson             # during exploration

# ensemble of N independently-seeded models (robust predictions) -> shows up as its own column
$PY scripts/train_ensemble.py --config configs/residualbind_ens10.json --h5_path $H5 \
    --all_rbps --output_dir results/residualbind_ens10
bash scripts/run_ensemble_sweep.sh         # same, parallelised across 4 GPUs (+ build_summary + plot)

# performance figure (RBP x model heatmap) for a split -> results/figures/<split>_heatmap.png
$PY scripts/plot_performance.py --results_dir results --split valid                # best_val_pearson
$PY scripts/plot_performance.py --results_dir results --split test                 # test_pearson
$PY scripts/plot_performance.py --results_dir results --split valid --models residualbind agft_bin4

# tests (synthetic h5, no weights)
$PY -m pytest tests/ -q
```

Each run writes, per RBP, one checkpoint **per stage** plus `metrics.json`, and a top-level
`results/<label>/summary.json`. Checkpoints are minimal (mirroring `alphagenome_encoder_ft`'s save
modes — the unused transformer/decoder are never saved):

- AlphaGenome-FT `head` stage → `stage1_head.pt` = `{head_state_dict}` (~28 MB; encoder == the
  pretrained weights, kept in eval mode so it's exact).
- AlphaGenome-FT `all` stage → `stage2_all.pt` = `{encoder_state_dict, head_state_dict}` (~370 MB).
- ResidualBind → full `state_dict` (tiny).
- `best.pt` is a symlink to the best-by-val stage; `metrics.json` reports each stage's
  `best_val_pearson` (and `test_pearson` when `--eval_test`), plus top-level `best_stage`.

**Best validation vs. test.** Every run records the best validation Pearson (`best_val_pearson`)
across stages. The held-out **test split is only evaluated with `--eval_test`** — leave it off
during exploration so you don't overfit to test; `test_pearson`/`test_mse` are added to the JSON
only when the flag is set. `compare.py` defaults to `test_pearson`; pass
`--metric best_val_pearson` to tabulate exploration runs.

## Modeling note

ResidualBind is a strong, fast, purpose-built baseline. AlphaGenome is a **DNA** model trained on
long genomic tracks, so 39 nt **RNA** probes are far OOD and mode A collapses each probe to one
vector. Mode B (finer intermediate) + stage-2 unfreezing is the most promising lever; if it still
trails ResidualBind, an **RNA** backbone (RNA-FM / RiNALMo / UTR-LM) would be a more apt transfer
source — a natural next model to add to the registry.
