# Executable scripts

Every active Python or shell command for this project lives here. Reusable scientific code lives separately in `../src/`.

## Layout

- `pipeline/`: library generation, training, evaluation, cache generation, and collection assembly.
- `maintenance/`: result repair and migration utilities.
- `figures/core/`: reusable manuscript figure builders.
- `figures/prototypes/<figure>/`: generators for assets retained in `../../prototypes/<figure>/`.
- `experiments/`: exploratory analyses without a settled narrative role.

Prototype scripts deliberately write back to the corresponding `mRNA_RBP/prototypes/<figure>/` asset directory; moving the code does not move or duplicate its figures and data.

Run commands from the repository root, for example:

```bash
python mRNA_RBP/scripts/pipeline/build_ground_truth_collection.py --help
python mRNA_RBP/scripts/figures/prototypes/library_size_figure/make_variants.py
```
