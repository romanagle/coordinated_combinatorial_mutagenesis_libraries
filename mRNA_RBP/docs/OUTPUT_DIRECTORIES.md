# Results, runs, and models

## Where to look

| Question | Canonical location |
|---|---|
| Which evidence supports the manuscript? | [`outputs/ground_truth_collections/`](../outputs/ground_truth_collections/) |
| Where did an active computation run? | [`runs/`](../runs/) |
| Which fitted models do the oracles load? | [`models/`](../models/) |
| Where are inactive low-WT runs retained? | [`archive/inactive_runs/low_wt/`](../archive/inactive_runs/low_wt/) |

## Directory contract

- `outputs/` is for results selected and staged for interpretation. The canonical manuscript evidence is `outputs/ground_truth_collections/`.
- `runs/` is generated working state: libraries, caches, predictions, parameter sweeps, surrogate coefficients, and staging files.
- `models/` contains retained fitted weights that runtime code depends on. Models are not buried inside run directories.
- `archive/inactive_runs/` contains inactive material preserved for recovery. It is outside normal navigation and is not an evidence source.

There are no longer any root-level `outputs_*` directories. That historical naming convention encoded oracle and WT identity but obscured whether a directory contained final evidence or intermediate computation.

## Active run layout

```text
runs/
├── deepsquid/
│   ├── hur/high/
│   └── vts1/high/
├── residualbind/
│   ├── hur/high/
│   └── vts1/high/
└── synthetic_negative_control/
```

Future biological runs are routed to `runs/<oracle-family>/<target>/<wt-activity>/` by `src.oracles.default_output_base()`.

## Legacy material still inside `outputs/`

Files outside `outputs/ground_truth_collections/` are older synthetic-GT working artifacts. They remain a known migration item because the synthetic pipeline still defaults to `outputs/`. Do not treat them as curated manuscript evidence.
