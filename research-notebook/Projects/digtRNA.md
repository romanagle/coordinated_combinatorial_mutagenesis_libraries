
## Objective

- Characterize tRNA identity (ID) elements across isotypes, including divergence between Bacteria and Archaea.

## Current status

- FiLM-conditioned GNN identity-element analysis is now operational for clade-specific screens.
- A prototype clade screen ranks held-out clades by the change in isotype logit contrast when the clade's hierarchical FiLM label is omitted.
- Sensitive screening, logo-divergence joins, and nested child-clade follow-up are being used together to separate true clade-specific identity elements from broad FiLM calibration effects.
- RNA-FM/RiNALMo masked embedding structure remains useful background context, but the active analysis has shifted to GNN/FiLM attribution.

## Active blockers

- GPU access is needed for missing clade ISM jobs; the current CPU session had no CUDA available.
- Some legacy `.sprinzl.json` sidecars can be stale or frame-mismatched and must be verified against the raw saliency CSV before interpreting a logo.
- Fast FiLM omission scores alone do not prove motif divergence; they can reflect clade-level calibration/logit effects.

## Important decisions

- Track digtRNA as a distinct research project.
- Use held-out sequences only for clade screening: training-set sequences are excluded, while held-out test sequences and sequences left out of train/test may be included in downstream embedding-scale analyses.
- Run clade screens per isotype using the logit contrast `logit[target isotype] - mean(logit[other isotypes])`.
- Treat the FiLM omission screen as a candidate generator, then require clade-vs-domain ISM/logo divergence before calling a clade-specific identity element.
- For large divergent clades such as Patescibacteriota, run nested child-clade analysis to test whether the parent signal is localized to smaller lineages.
- Compare localized child-clade identity-element logos against the relevant domain background, not just against the parent clade.

## Durable research memory

- RNA-FM and RiNALMo masked embeddings both cluster strongly by isotype in high-dimensional cosine space.
- Broad type-II/long tRNA separation is present in both FMs and tracks Leu/Ser/SeC/Tyr length structure.
- RNA-FM same-isotype kNN fraction was 0.656 versus 0.076 random in a 12k sample.
- RiNALMo same-isotype kNN fraction was 0.730 versus 0.073 random in a 12k sample.
- Phylogeny contributes local structure but is weaker than isotype for the observed FM separation.
- RNA-FM short-isotype GMM did not support a simple two-cluster model by BIC.
- RNA-FM GMM separated Thr/fMet/His/Ala/Val-rich components from Glu/Cys/Gly/Gln-rich components.
- Prototype FiLM clade screen lives at `/data/roma/dsthesis/scripts/prototypes/film_clade_screen_prototype.py`.
- Conservative screen output: `/data/roma/dsthesis/figures/film_clade_screen_prototype.csv`.
- Sensitive screen output: `/data/roma/dsthesis/figures/film_clade_screen_sensitive_prototype.csv`.
- Sensitive screen produced 762 eligible clade/isotype hits with support thresholds `min_hidden=5`, `min_eligible=20`, `shrink_k=10`; high-scoring non-domain hits included Arg/Bacilli, His/Patescibacteriota, Arg/Patescibacteriota, Thr/Bacilli, Pro/Patescibacteriota, Arg/Nitrosopumilaceae, Ser/Burkholderiaceae, Glu/Verrucomicrobiota, and Thr/SCGC-AAA011-G17.
- Joined screen/logo divergence output: `/data/roma/dsthesis/figures/film_clade_screen_sensitive_with_logo_divergence.csv`.
- Patescibacteriota hits split into likely motif-divergent cases and calibration-like cases. His, Arg, Cys, and Trp showed high divergence from Bacteria, while Pro/Lys/Met/Ser were more similar to domain-level logos despite high FiLM screen scores.
- Nested Patescibacteriota analysis output: `/data/roma/dsthesis/figures/patescibacteriota_nested_screen_analysis.csv`.
- Patescibacteriota nested candidates: Minisyncoccia, Microgenomatia, UBA9973, Saccharimonadia, and UBA6257 are the main child clades to validate by child-vs-Bacteria logos.
- Decision log for the Patescibacteriota follow-up: `/data/roma/dsthesis/figures/clade_hier_IES_patescibacteriota_nested/DECISIONS.md`.
- Arg/Nitrosopumilaceae had a stale or mismatched sidecar that suppressed the A20 call in the rendered IE logo. The sidecar was backed up, rebuilt from the CSV modal Arg frame, and the Arg/Nitrosopumilaceae-vs-Archaea logo was rerendered.
- Frequency and IE logos answer different questions: position 20 is nearly always A in both Arg/Nitrosopumilaceae and Arg/Archaea, but the IE delta for A20 is much smaller in Nitrosopumilaceae than in Archaea.
- Isotype-wide frequency logo comparisons for Arg/Nitrosopumilaceae vs Archaea were generated for Sprinzl positions 20, 72, 73, 14, 15, 16, and 48 under `/data/roma/dsthesis/figures/freq_logos/pos_comparisons/`.

## Open research questions

- What are the most divergent ID elements per isotype, within Bacteria and within Archaea?
- What structural features (e.g., variable loop size/composition) carry identity information beyond position-dependent elements?
- Are there clade-specific ID signatures for isotype pairs sharing an aaRS (tRNA-Gln/Glu, tRNA-Asn/Asp)?
- How should an ID element be concretely/operationally defined?
- Which sensitive-screen hits remain after requiring both a large FiLM omission effect and strong clade-vs-domain ISM/logo divergence?
- Do Patescibacteriota parent-level hits mostly resolve to Minisyncoccia, Microgenomatia, UBA9973, Saccharimonadia, or UBA6257?
- How many legacy sidecar files need frame-verification before the existing clade IE figures can be trusted?

## Next milestone

- Run the queued GPU ISM scripts for sensitive clade hits and Patescibacteriota nested child clades, then curate the high-confidence clade-specific identity-element set.

## Future directions

- Scale embedding/screening to the full v99 sequence set after the held-out/test-set workflow is stable.
