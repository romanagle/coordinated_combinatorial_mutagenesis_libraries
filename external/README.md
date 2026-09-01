# External repositories

Third-party repositories retained for reference or as project dependencies.

| Repository | Role | Active pipeline dependency |
| --- | --- | --- |
| `rbp_benchmark/` | Upstream generic RBP benchmarking suite | No |
| `residualbind/` | Upstream ResidualBind implementation and data utilities | Yes |
| `squid-nn/` | SQUID/MAVE-NN surrogate implementation | Yes |
| `squid-manuscript/` | SQUID manuscript utilities used by the active pipeline | Yes |

Project-owned oracle training remains in `../rbp/`.

The active pipeline resolves these dependencies through `external/`; preserve their local modifications when updating upstream clones.
