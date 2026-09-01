# SMN1 5′ splice-site dataset assessment

## Bottom line

- **Verdict — strong fit:** the SMN1 dataset satisfies the core requirements for a higher-order combinatorial sequence-to-function landscape and is easier to integrate than the GFP data because it is RNA, short, and nearly exhaustive.
- **Best use in the manuscript:** a compact, highly constrained RNA landscape showing that a familiar random-library workflow can encounter strong global nonlinearity and higher-order epistasis even across only eight variable nucleotides.
- **Important qualification:** this is a targeted, almost exhaustive combinatorial library rather than a sparse random library around wild type, and its constraint comes from splice-site recognition and thresholding rather than a folded RNA structure.

## What SMN1 is

- The assayed element is the **5′ splice site of exon 7 of the human survival of motor neuron 1 (`SMN1`) gene**, placed in an SMN1 minigene and measured in human cells.
- The original assay varied a nine-nucleotide splice-site motif of the form `NNN/GYNNNN`, where the slash is the exon–intron boundary, `+1 G` is fixed, and `+2 Y` is either `U` or `C`.
- The later modeling paper drops the invariant `+1 G` and represents each genotype as eight RNA letters at positions `−3, −2, −1, +2, +3, +4, +5, +6`.
- The canonical/reference eight-letter genotype used by the modeling repository is **`CAGUAAGU`**; restoring fixed `+1 G` gives the full local splice-site sequence `CAG/GUAAGU`.

Sources: [2025 preprint, Figure 2 and SMN1 Results](https://repository.cshl.edu/41945/1/10.1101.2025.08.15.670613.pdf), [official analysis settings with reference sequence and positions](https://github.com/cmarti/epik_analyses/blob/master/scripts/settings.py), [original 2018 assay paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC6179149/).

## Library design and mutation order

- The design is **fully combinatorial across the targeted element**, not limited to single or pairwise mutants.
- The valid designed space contains `4^7 × 2 = 32,768` splice sites because seven positions take four nucleotides and `+2` takes `U` or `C`.
- The processed modeling table contains **30,732 unique measured genotypes**, or about **93.8% of the valid designed space**.
- Relative to reference `CAGUAAGU`, measured variants span every mutation order from zero through eight:
  - distance 0: 1
  - distance 1: 22
  - distance 2: 209
  - distance 3: 1,100
  - distance 4: 3,604
  - distance 5: 7,492
  - distance 6: 9,557
  - distance 7: 6,743
  - distance 8: 2,004
- Therefore it decisively meets the “more than pairwise” requirement and provides abundant genotypes at orders 3–8.
- The library was made by focused randomization of the complete splice-site element, with randomized barcodes used to associate constructs with RNA measurements; it was designed for exhaustive coverage rather than sampled from a controlled per-site mutation rate around the SMN1 reference.

Sources: [original assay design and coverage](https://pmc.ncbi.nlm.nih.gov/articles/PMC6179149/), [official processed SMN1 table](https://github.com/cmarti/epik_analyses/blob/master/data/smn1.csv), [official complete eight-letter sequence list](https://github.com/cmarti/epik_analyses/blob/master/data/smn1.seqs.txt). Mutation-order counts above were calculated directly from the official processed table against the repository reference sequence.

## Exact phenotype

- The biological phenotype is **exon inclusion / splice-site activity**.
- The original massively parallel splicing assay estimates **Percent Spliced In (PSI)** from the abundance of each barcode in total RNA versus exon-inclusion RNA.
- The 2025 paper describes the target as the proportion of correctly spliced transcripts, or PSI.
- In the downloadable modeling table, the response column is `y`, with sequence-specific uncertainty `y_std`.
- The processed response is normalized so that reference `CAGUAAGU` has `y = 100`; its `y_std` is approximately `4.449`.
- Values in the processed table range from `0` to about `259.84`, so the downloadable `y` should be treated operationally as the authors’ **processed, reference-normalized splice-activity/PSI estimate**, not as a literal probability constrained to 0–100.
- The modeling paper explains that `y` was derived from one to seven replicate enrichment-ratio measurements using a bias-corrected geometric mean when all replicates were positive and otherwise a median, with a sequence-specific variance estimate.

Sources: [2025 preprint Methods, SMN1 processing](https://repository.cshl.edu/41945/1/10.1101.2025.08.15.670613.pdf), [official processed table](https://github.com/cmarti/epik_analyses/blob/master/data/smn1.csv), [original assay paper and PSI quantification](https://pmc.ncbi.nlm.nih.gov/articles/PMC6179149/).

## Data representation and downloads

- **Ready-to-model table:** [`data/smn1.csv`](https://github.com/cmarti/epik_analyses/blob/master/data/smn1.csv)
  - `seq` — complete eight-letter variable genotype, using RNA `U`.
  - `y` — processed quantitative splicing activity.
  - `y_std` — sequence-specific standard deviation/uncertainty.
- **Complete enumerated sequence space:** [`data/smn1.seqs.txt`](https://github.com/cmarti/epik_analyses/blob/master/data/smn1.seqs.txt)
  - Contains all `4^8 = 65,536` generic eight-letter strings; only the 32,768 strings with `+2 ∈ {C,U}` are valid members of the original GU/GC library.
- **Reference genotype and position definitions:** [`scripts/settings.py`](https://github.com/cmarti/epik_analyses/blob/master/scripts/settings.py)
- **Earlier official source table:** [`vcregression/data/Smn1/smn1data.csv`](https://github.com/davidmccandlish/vcregression/blob/master/vcregression/data/Smn1/smn1data.csv)
- **Original multi-context processed measurements:** [`vcregression_paper/Smn1/data/psi_9nt.txt`](https://github.com/davidmccandlish/vcregression/blob/master/vcregression_paper/Smn1/data/psi_9nt.txt)
  - Includes BRCA2, IKBKAP, and SMN1 measurements and their standard errors.
- **Original replicate-level ratios:** [`vcregression_paper/Smn1/data/ratios_9nt_ss_all.txt`](https://github.com/davidmccandlish/vcregression/blob/master/vcregression_paper/Smn1/data/ratios_9nt_ss_all.txt)

## Wild type and genotype reconstruction

- The full variable reference genotype is explicitly supplied as `CAGUAAGU`; unlike the GFP mutation-string tables, no reconstruction from a separate FASTA is needed.
- Each row already contains the entire eight-nucleotide genotype rather than a list of substitutions.
- The fixed `+1 G` is omitted from the machine-learning representation because it never varies.
- For mutation-rate analyses, Hamming distance can be calculated directly between `seq` and `CAGUAAGU`.

## Landscape constraint and structure

- The landscape is **strongly constrained and threshold-like**: activity is approximately a steep sigmoidal transformation of affinity between the 5′ splice site and U1 snRNA.
- Mutating critical nucleotides, such as the reference `U` at `+2`, commonly renders the splice site inactive; later mutations then have little additional phenotypic effect.
- An additive model performs poorly (`R²` around `0.15`), a pairwise model reaches about `0.45`, and models capable of higher-order epistasis exceed `0.8` with dense training data.
- The paper reports that pairs separated by as many as five sites can be more phenotypically correlated than pairs separated by one site, showing that identity and location of mutations matter, not only mutation count.
- The authors characterize the coarse landscape as single-peaked with strong global nonlinearity plus mutation-specific interactions.
- This supports a claim about a **highly constrained sequence-function landscape**, but not specifically a folded-structure constraint: the mechanism is spliceosome/U1 recognition and nonlinear exon-inclusion response.

Source: [2025 preprint, SMN1 Results and Discussion](https://repository.cshl.edu/41945/1/10.1101.2025.08.15.670613.pdf).

## Fit to the surrogate pipeline

- **Direct sequence-to-function pairs:** yes.
- **Quantitative target:** yes; use `y`, while retaining `y_std` for filtering, weighting, or uncertainty-aware evaluation.
- **Higher-order combinatorial variation:** yes, through eight mutated positions.
- **Measured wild type/reference:** yes; `CAGUAAGU`, `y = 100`.
- **Random-library subsampling:** yes; a sampled training library can be drawn from the measured table and held-out measured genotypes can be used as ground truth.
- **Mutation-rate experiments:** yes, by sampling genotypes according to Hamming distance from the reference, though these would be retrospective synthetic subsamples of an exhaustive library rather than the physical design used in the experiment.
- **Library-size experiments:** especially clean because the large measured pool can remain fixed while training size varies.
- **Activity-balanced evaluation:** feasible directly from observed `y` values without first fitting an oracle.
- **Surrogate comparison:** especially informative because published results already establish large separations among additive, pairwise, global-epistasis, and higher-order models.
- **Main limitation:** the input length is only eight variable nucleotides, so it tests combinatorial complexity very well but not the long-sequence representation problem faced by full regulatory regions or proteins.
- **Generalization opportunity:** the original experiment measured the same complete splice-site space in SMN1, BRCA2, and IKBKAP minigene contexts. Those can potentially serve as three context-specific landscapes, but preprocessing should be harmonized from `psi_9nt.txt` rather than assuming the ready-made `smn1.csv` pipeline applies unchanged.

## Comparison with the Somermeyer GFP dataset

| Property | SMN1 splice site | Somermeyer GFPs |
|---|---|---|
| Molecule | RNA splice-site sequence | Protein |
| Variable representation | Entire 8-nt genotype | Amino-acid substitutions plus WT FASTA |
| Landscape coverage | Nearly exhaustive compact space | Sparse sample from enormous protein spaces |
| Mutation order | 0–8, with thousands at orders 3–8 | Typically several substitutions; higher-order but sparse |
| Phenotype | Quantitative splicing/exon inclusion | Quantitative cellular fluorescence brightness |
| Constraint | U1 recognition plus sharp splicing threshold | Folding, maturation, stability, and fluorescence constraints |
| Pipeline advantage | Direct measured evaluation over most of the landscape | Longer, realistic protein sequence and multiple orthologs |
| Pipeline disadvantage | Only 8 variable sites; not a folded RNA structure | Arbitrary new sequences lack measured ground truth |
| Biological replication | Same splice-site library in three minigene contexts is available upstream | Four GFP ortholog landscapes in one protein family |

## Recommended role

- Use SMN1 as the **cleanest real-data analogue of a compact but unexpectedly nonlinear and higher-order RNA landscape**.
- Use GFP as the complementary **longer, sparse, structurally constrained protein example**.
- Together they make a stronger methods argument than either alone: SMN1 isolates combinatorial landscape complexity with near-complete measurement, while GFP tests whether the warning persists in realistic sparse protein libraries.
