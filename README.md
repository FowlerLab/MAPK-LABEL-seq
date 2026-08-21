# MAPK LABEL-seq — a pathway-scale variant effect atlas

Functional characterisation of variant effects across 17 core MAPK pathway
proteins, measured with
[LABEL-seq](https://pmc.ncbi.nlm.nih.gov/articles/PMC11785348/).

## Authors and contributions

- **Sriram Pendyala** — design, data, and analysis.
- **Jessica Simon** — the LABEL-seq method
  ([Simon, Fowler & Maly, *Nature Methods* 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC11785348/))
  and experiments, data, and the original scoring and annotation analysis this pipeline was rebuilt from.
- **Claude** (Anthropic, Opus 5) — pipeline, analysis scripts and documentation,
  written with Sriram.

Fowler and Maly Labs, Department of Genome Sciences, University of Washington.

## What is measured

Every variant is assayed in up to three ways, each read out by sequencing a
barcode library:

| assay | readout | question |
|---|---|---|
| **abundance** | Flag / HaloTag | how much protein is there |
| **activity** | pERK reporter ratio | what does it do to pathway output |
| **interaction** | Strep / Flag | does it still bind the partner (KRAS–LZTR1) |

Abundance is also measured under HSP90 inhibition (pimitespib), which is what
makes the chaperone-buffering analysis possible.

**Proteins.** RTKs EGFR, ERBB2, MET, RET · GTPases KRAS, MRAS · adaptor GRB2 ·
GEFs SOS1, SOS2 · RAFs ARAF, BRAF, CRAF · MEKs MEK1, MEK2 · scaffolds KSR1,
KSR2 · phosphatase SHP2.

**Scale.** 531,797 variant effects over 168,896 distinct variants, across 22
libraries, 3 assays and 8 treatment arms. Every variant class is scored, not only
the ones expressible as a single substitution: mid-codon 3-nt deletions,
frameshifts and multi-mutants are named by their protein consequence and kept.

## The notebooks

Run them in this order. Each writes the table the next one reads.

| notebook | does |
|---|---|
| **`Scoring.ipynb`** | barcode counts → per-variant scores. Canonical protein-level identity, the filtering and normalisation cascade, the replicate gain correction, the standard curve, and the empty-vector dominant-negative thresholds. |
| **`Annotations.ipynb`** | joins the annotation layers — dominant negatives, curated PDB interfaces, HSP90/CDC37 contacts, conservation, ΔΔG, kinase motifs, population and tumour observation — into the single table the figure notebooks read. |
| **`DN.ipynb`** | figure 3: what dominant-negative variants are, where they sit in the structure, and how they are depleted from population databases. |
| **`HSP90i.ipynb`** | figure 4: chaperone buffering — which variants are rescued by HSP90 and what predicts it. |

`utils.py` holds the shared loaders and the rendering helpers the figure
notebooks use, so the notebooks stay about the analysis rather than about
matplotlib.

## Two things worth knowing before reading the tables

**A row is a variant *effect*, not a variant.** The key is
`(variant, library, assay, assay_treatment)`; the same variant appears in up to
eight rows. Aggregating without grouping on that key mixes assays and treatments.

**`library` is not `protein`.** Several proteins were scanned in two halves —
`araf_cterm` and `araf_nterm` are different constructs of ARAF, each spanning the
full protein.

## The two score tables

The pipeline writes two tables. Which one you want depends on what you are doing.

| table | columns | what it is |
|---|---|---|
| **`raw_scores.tsv`** | 38 | The measurements. Identity, reference sequence (UniProt isoform, RefSeq, Ensembl, and GRCh38 coordinates where the protein change has a resolvable nucleotide route), barcode support, the **uncorrected** per-replicate ratios and scores with their standard curve, and the low/wt-like/high classification. Written by `scripts/export_raw_scores.py`. |
| **`scores_reannotated.tsv`** | 96 | Everything the figures need: the gain-corrected replicates, dominant-negative calls, the HSP90 dependence and buffering families, structure, conservation, PTMs, clinical and population annotation. Written by `scripts/reannotate_scores.py`. |

Every column of both tables — what it means, how it was derived and which primary
source it came from — is documented in
[`docs/score_table_columns.md`](docs/score_table_columns.md).

`raw_scores.tsv` is a strict column selection of the annotated table, so the values
agree exactly. Two names differ on purpose, and the difference has to survive a join
on `(variant, library, assay, assay_treatment)`:

- `score_j` is the raw WT-relative score in both tables. `average_score` in the raw
  table is the mean of those three; `average score` in the annotated table is the
  mean of the **corrected** replicates.
- `std_adj_score_j` in the raw table is the standard curve fitted on the raw
  replicates; `intercept_0_std_adj_score_j` in the annotated table is fitted on the
  corrected ones.

**The correction.** Two of 192 replicates resolve materially less of their assay's
dynamic range than their two siblings do, and are corrected by a single exponent
about wild type, clamped outside the measured range; the other 190 come through
bit-identical. `scripts/gain_correction.py` carries the method, the thresholds and
the evidence. The raw table has none of it.

## What is here, and what is not

Only what the pipeline needs to run, plus the column reference:

- the four notebooks, and `utils.py` (shared loaders and rendering helpers)
- `scripts/` — `gain_correction.py` (the replicate correction),
  `reannotate_scores.py` (builds the annotated table), `export_raw_scores.py`
  (builds the raw table)
- `src/labelseq_mapk/` — `config.py` and `annotation.py`, the two modules the
  above import
- `config/` — the four YAML files
- `data/dn_cutoffs_empty_vector.tsv` — the per-library empty-vector DN thresholds,
  the one data file small enough to version
- `docs/score_table_columns.md` — every column of both tables
- `environment.yaml`

**Not here: any data.** The barcode counts, the score tables and the annotated
tables are orders of magnitude too large for GitHub, and the annotation source
files are third-party downloads. `config/paths.yaml` is therefore a template:
its paths point into a `data/inputs/` tree that you populate, not at the
locations they were built from. The tables are available on request and will be
deposited with the manuscript.

## Environment

```
conda env create -f environment.yaml
conda activate labelseq_mapk
```
