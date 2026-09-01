# Protein combinatorial-mutagenesis datasets for the random-library pipeline

## Question

- Which protein datasets contain variants with more than pairwise mutations, a quantitative sequence-to-function measurement, and downloadable sequence-level data suitable for fitting a surrogate?
- Preference: a sharply constrained or rugged landscape that can support the manuscript's caution that random-library measurements may conceal biological complexity.

## Recommendation

- **Use the linked Somermeyer et al. GFP collection first.** It is unusually close to the desired experiment: random multi-mutant libraries, direct fluorescence, four related biological landscapes, processed amino-acid genotype-to-brightness tables, and contrasting sharp versus flat fitness peaks.
- **Use AAV2 capsid production as the strongest non-GFP alternative.** It is much larger and strongly multi-mutant, but its primary phenotype is packaging viability/enrichment rather than a continuous molecular activity.
- **Keep yeast His3 as the strongest rugged-landscape alternative.** It has extensive higher-order epistasis and quantitative growth, but its library design uses combinations of naturally occurring ortholog states within separate segments rather than unbiased random mutagenesis.
- **Treat GB1 as a compact validation benchmark, not the principal biological replication.** It is combinatorially complete and quantitative, but only four sites vary.

## Ranked candidates

### 1. Somermeyer et al. 2022: four orthologous GFP fitness peaks — best overall fit

- **Scale and mutation order:** 35,500 amacGFP, 26,165 cgreGFP, 32,260 ppluGFP2, and 51,715 avGFP protein genotypes; the new libraries average 3–4 amino-acid substitutions per genotype, and avGFP averages 3.93. [Primary paper](https://doi.org/10.7554/eLife.75842)
- **Function:** measured fluorescence/brightness is a direct native protein phenotype and a quantitative surrogate target.
- **Constraint signal:** the authors explicitly distinguish sharp, epistatic peaks (avGFP and cgreGFP) from flatter, more mutation-tolerant peaks (amacGFP and ppluGFP2); most genotypes are near-wild-type or nonfunctional, consistent with a threshold-like landscape. [Primary paper](https://doi.org/10.7554/eLife.75842)
- **Downloadability:** the official repository provides amino-acid genotype-to-brightness CSVs, wild-type sequences, barcode-level/raw measurement material, PDB structures, and analysis code. Mutation labels use `AiB` with zero-based positions. [Official repository data guide](https://github.com/aequorea238/Orthologous_GFP_Fitness_Peaks/tree/master/data)
- **Pipeline fit:** high. Each protein can become its own landscape; mutation strings can be expanded against the supplied wild-type FASTA; brightness can be the regression target; Hamming distance supplies mutation count.
- **Important caveat:** these are four orthologs of one protein family, not four unrelated biological mechanisms. The phenotype is also strongly bimodal/thresholded, so classification and rank-based evaluation should accompany ordinary regression.
- **Redundancy note:** the linked collection already incorporates the Sarkisyan avGFP landscape after reformatting; do not count Sarkisyan 2016 as a fifth independent dataset. [Repository data guide](https://github.com/aequorea238/Orthologous_GFP_Fitness_Peaks/tree/master/data#final_datasets)

### 2. Bryant et al. 2021: AAV2 capsid diversification — best non-GFP fit

- **Scale and mutation order:** 201,426 assayed designed variants in a 28-amino-acid capsid segment; many viable designs have 12–29 mutations, and 110,689 were viable. [Primary paper](https://www.nature.com/articles/s41587-020-00793-4)
- **Function:** capsid production/packaging selection score, which is a biologically clear sequence-to-function relationship and was explicitly modeled with neural networks.
- **Constraint signal:** capsid assembly and genome packaging impose strong structural constraints, while the dataset spans far beyond pairwise substitutions.
- **Downloadability:** the authors provide a processed zipped CSV with selection scores, processing/synthesis code, model-training links, and raw sequencing under NCBI BioProject PRJNA673640. [Official repository](https://github.com/churchlab/Deep_diversification_AAV) and [processed data](https://github.com/churchlab/Deep_diversification_AAV/tree/main/Data)
- **Pipeline fit:** high after restricting representation to the 28-residue mutated window or reconstructing the full VP sequence.
- **Important caveats:** the score is enrichment/viability rather than a clean continuous molecular activity; the full collection mixes library-generation regimes and ML-designed sequences, so splits must preserve provenance and avoid training/test leakage caused by the design cycle.

### 3. Pokusaeva et al. 2019: yeast His3 ortholog-state combinations — best rugged higher-order landscape

- **Mutation order:** combinatorial libraries within 12 protein segments contain variants carrying multiple ortholog-derived amino-acid states; the study reports that the effects of 85% of states depend on background, 67% of sites show sign epistasis, and 46% show reciprocal sign epistasis. [Primary paper and official record](https://doi.org/10.1371/journal.pgen.1008079)
- **Function:** yeast growth complementation is a quantitative organismal proxy for His3 enzyme function.
- **Constraint signal:** exceptionally strong; it was designed to expose inaccessible paths and multi-site interactions in a structurally constrained enzyme.
- **Downloadability:** article supplements and the official ISTA record provide the alignment and segment-library/sequencing summaries. [Official ISTA record](https://research-explorer.ista.ac.at/record/6419)
- **Pipeline fit:** medium. ProteinGym's `HIS7_YEAST_Pokusaeva_2019` representation is convenient, but provenance should ultimately point back to the primary files.
- **Important caveats:** variants come from restricted alphabets of natural ortholog states, and each segment is a different local combinatorial design. It therefore does not reproduce an unconstrained random amino-acid library and may need per-segment modeling or segment-aware covariates.

### 4. Wu et al. 2016: complete four-site GB1 binding landscape — useful compact benchmark

- **Scale and mutation order:** all `20^4 = 160,000` combinations were designed at four sites; fitness was recovered for roughly 149,000 variants. This is unequivocally beyond pairwise and includes third- and fourth-order interactions. [Primary paper](https://doi.org/10.7554/eLife.16965)
- **Function:** quantitative IgG-Fc binding enrichment.
- **Constraint signal:** rugged, with reciprocal sign epistasis and inaccessible direct adaptive paths.
- **Downloadability:** the full landscape is included in the paper's supplementary data. [Official eLife figures/data page](https://elifesciences.org/articles/16965v1/figures)
- **Pipeline fit:** very high technically: four-residue sequence plus scalar fitness is almost trivial to ingest.
- **Important caveat:** only four preselected contact sites vary, so it tests combinatorial order well but says little about whole-protein random-library size or distributed structural constraint.

### 5. Sarkisyan et al. 2016: avGFP random-mutant landscape — useful only if the linked collection is not adopted wholesale

- **Scale and mutation order:** tens of thousands of full-length GFP genotypes, averaging 3.7 mutations and reaching up to 15 missense mutations. [Primary paper](https://doi.org/10.1038/nature17995)
- **Function and constraint:** quantitative fluorescence on a narrow peak shaped by negative epistasis and an apparent threshold.
- **Downloadability:** processed genotype/barcode/fluorescence tables and raw sequencing links are on the authors' Figshare deposit. [Official dataset](https://figshare.com/articles/dataset/Local_fitness_landscape_of_the_green_fluorescent_protein/3102154)
- **Pipeline fit:** high, but the Somermeyer repository already supplies a harmonized amino-acid version alongside three additional GFPs.

## Integration sketch

- Define one record as `landscape_id`, `wild_type_sequence`, `mutant_sequence`, `mutation_count`, `measurement`, `replicate_count`, and `library_provenance`.
- For Somermeyer GFP, reconstruct full amino-acid sequences from the supplied wild type plus mutation labels and retain synonymous replicate counts as an uncertainty/quality field.
- Analyze each GFP ortholog separately first; pooling them without an ortholog identifier would conflate different coordinate identities and different peak shapes.
- Preserve raw brightness as the primary target, but add a preregistered functional/nonfunctional threshold and rank-based metrics because intermediate fluorescence is scarce.
- For AAV, keep experimental-library and ML-designed variants in separate provenance groups; random variant-level splitting would overstate generalization.
- For His3, keep segment identity explicit and avoid treating missing cross-segment combinations as sampled negatives.

## Bottom line

- The forwarded paper is not merely a lead to another dataset: **its own released four-GFP dataset is probably the best immediate protein addition**.
- It supports both sides of the desired message within a harmonized assay: two landscapes appear sharply constrained and epistatic, whereas two are comparatively robust.
- A defensible multi-biological-dataset package would be **four GFP orthologs + AAV2 + His3**, with GB1 used as a compact combinatorial control.
