# Human RNAmap G6 plotting workflow

This bundle reproduces three figures comparing G6 structure prediction with and
without Human RNAmap DMS guidance:

1. A per-hit F1 scatterplot, colored by ViennaRNA MFE F1.
2. An F1 scatterplot averaged by Rfam family, also colored by ViennaRNA MFE F1.
3. A two-panel summary binned by baseline F1: counts above and below the
   equality line on top, and violin plots of delta F1 on the bottom.

The example outputs were generated from 2,502 complete Human RNAmap/Rfam hits
in 228 families. Two records were excluded.

## Included files

```text
humanrnamap_g6_workflow/
├── README.md
├── scripts/
│   ├── aggregate_plot_g6.py
│   └── plot_humanrnamap_g6_binned_delta.py
└── examples/
    └── humanrnamap_g6/
```

`aggregate_plot_g6.py` reads the Human RNAmap manifest and one comparison CSV
per hit. It writes the per-hit and family-level CSV files and scatterplots.

`plot_humanrnamap_g6_binned_delta.py` reads the per-hit F1 CSV produced by the
first script. It writes a binned summary CSV and the stacked count/violin PNG.

## Requirements

- Python 3.9 or newer
- NumPy
- Matplotlib

For example:

```bash
python3 -m pip install numpy matplotlib
```

## Required input layout

The aggregation script needs:

- `manifest.tsv`: tab-separated input metadata with at least `name`,
  `accession`, `chrom`, `length`, and `observed_dms` columns.
- A results directory containing one subdirectory per manifest `name`.
- In each result subdirectory, a `<name>.g6_comparison.csv` file.
- Optionally, `preparation_exclusions.csv` for hits excluded before folding.

Each comparison CSV must have rows named `tornado_g6_noprobe`,
`python_g6_cyk_dms`, and `viennarna_mfe`, with `f1`, `sensitivity`, and `ppv`
columns expressed as percentages from 0 to 100.

```text
inputs/
├── manifest.tsv
├── preparation_exclusions.csv
└── results/
    ├── hit_name_1/
    │   └── hit_name_1.g6_comparison.csv
    └── hit_name_2/
        └── hit_name_2.g6_comparison.csv
```

## Generate the plots

Run these commands from the unpacked bundle directory, substituting the input
paths for the local copies of the Human RNAmap data:

```bash
python3 scripts/aggregate_plot_g6.py \
  --manifest /path/to/manifest.tsv \
  --results-root /path/to/results \
  --preparation-exclusions /path/to/preparation_exclusions.csv \
  --outdir plots/humanrnamap_g6 \
  --metric f1

python3 scripts/plot_humanrnamap_g6_binned_delta.py \
  --input plots/humanrnamap_g6/humanrnamap_g6_f1.csv \
  --output-png plots/humanrnamap_g6/humanrnamap_g6_f1_binned_delta.png \
  --output-csv plots/humanrnamap_g6/humanrnamap_g6_f1_binned_delta.csv
```

The first command generates both scatterplot variants. The second generates
the stacked two-panel figure.

## Outputs

- `humanrnamap_g6_f1_scatter.{png,pdf}`: one point per complete hit. The x-axis
  is TornadoFold G6 without DMS, the y-axis is G6 with Human RNAmap DMS, and
  color is ViennaRNA MFE F1 against the same Rfam reference.
- `humanrnamap_g6_f1_by_family_scatter.{png,pdf}`: family means; point size
  indicates the number of sequences in the family.
- `humanrnamap_g6_f1_binned_delta.png`: top panel shows counts above and below
  the equality line; bottom panel shows the delta-F1 distribution in baseline
  F1 bins.
- `humanrnamap_g6_f1.csv`: per-hit values used by the plots.
- `humanrnamap_g6_f1_by_family.csv`: family-level summary.
- `humanrnamap_g6_f1_binned_delta.csv`: counts and distribution summaries for
  each baseline-F1 bin.
- `humanrnamap_g6_exclusions.csv`: records that could not be aggregated.

The example per-hit CSV records absolute cluster paths in `comparison_csv` for
provenance. Those paths are not used by the binned plotting script and do not
need to exist on another computer.

## Original cluster command

The included examples were regenerated in the project environment with:

```bash
MPLCONFIGDIR=/tmp/rivaslab-mpl-cache \
  .conda_envs/rfam-infernal/bin/python \
  scripts/tornado_crosscheck/aggregate_plot_g6.py \
  --results-root humanrnamap/tornado_g6_results \
  --outdir plots/humanrnamap_g6 \
  --metric f1

MPLCONFIGDIR=/tmp/rivaslab-mpl-cache \
  .conda_envs/rfam-infernal/bin/python \
  scripts/tornado_crosscheck/plot_humanrnamap_g6_binned_delta.py
```

`MPLCONFIGDIR` only redirects Matplotlib's cache to a writable temporary
directory; it does not affect the data or figures.
