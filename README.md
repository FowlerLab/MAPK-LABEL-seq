# MAPK LABEL-seq — a pathway-scale variant effect atlas

Functional characterisation of variant effects across 17 core MAPK pathway
proteins, measured with
[LABEL-seq](https://pmc.ncbi.nlm.nih.gov/articles/PMC11785348/).

## Authors and contributions

- **Sriram Pendyala** — design, data, and analysis.
- **Jessica Simon** — the LABEL-seq method
  ([Simon, Fowler & Maly, *Nature Methods* 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC11785348/))
  and experiments, data, the original scoring and annotation analysis this pipeline was rebuilt from,
  and the figure notebooks for figures 1–6e–i.
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

**Scale.** 531,797 variant effects over 200,361 distinct variants, across 22
libraries, 3 assays and 8 treatment arms. Every variant class is scored, not only
the ones expressible as a single substitution: mid-codon 3-nt deletions,
frameshifts and multi-mutants are named by their protein consequence and kept.

## The notebooks

Run them in this order. Each writes the table the next one reads.

| notebook | does |
|---|---|
| **`Scoring.ipynb`** | barcode counts → per-variant scores. Canonical protein-level identity, the filtering and normalisation cascade, the replicate gain correction, the standard curve, and the empty-vector dominant-negative thresholds. |
| **`Annotations.ipynb`** | joins the annotation layers — dominant negatives, curated PDB interfaces, HSP90/CDC37 contacts, conservation, ΔΔG, kinase motifs, population and tumour observation — and the per-variant HSP90 calls, and writes **Supplementary Table 2** and the annotated table the figure notebooks read. |
| **`Fig1_and_associated_ED.ipynb`** | figure 1 and extended data 1: the dataset, replicate agreement, per-protein score distributions. |
| **`Fig2_Fig3_and_associated_ED.ipynb`** | figures 2–3 and extended data 3–6: abundance and activity landscapes, per-position tracks, structural context. |
| **`Fig4_and_associated_ED.ipynb`** | figure 4 and extended data 7: dominant-negative variants — what they are, where they sit in the structure, how they are depleted from population databases. |
| **`Fig5_and_associated_ED.ipynb`** | figure 5 and extended data 8: chaperone buffering — which variants are rescued by HSP90 and what predicts it. |
| **`Fig6_a-d_and_associated_ED.ipynb`** | figure 6a–d and extended data 9: do variants at protein–protein interfaces score differently from matched surface controls, across experimental and predicted complexes. Runs in its own environment — see below. |
| **`Fig6_e-i_and_associated_ED.ipynb`** | figure 6e–i: KRAS — the LZTR1 and SOS1 interfaces, LZTR1 knockout and CIAR perturbation of abundance. |

The figure notebooks (Fig1–Fig6e–i) are Jessica Simon's, run here as she wrote
them apart from paths and the table read. They read the score table through
`src/labelseq_mapk/figure_data.py:load_scores`, which maps the Supplementary
Table 2 names back to the ones the figure code uses and derives its
`Mutation Type` from `variant_category` (so "deletion" is every single-codon
3-nt deletion, codon-straddling `delins_2for1` included). They run in the
`labelseq_mapk_figures` environment. `utils.py` holds the loaders and constants
`Annotations.ipynb` uses; `interactions/ppi_utils.py` holds the helpers for
`Fig6_a-d_and_associated_ED.ipynb`.

## The interaction analysis (figure 6)

`Fig6_a-d_and_associated_ED.ipynb` reads the same `output/annotated_combined.tsv` as the other
figure notebooks (through `load_scores`), and three sets of complexes containing a profiled protein:
RCSB PDB experimental structures, AlphaFold-Multimer models from Predictomes
(SPOC ≥ 0.69) and RoseTTAFold2-PPI hits from Zhang et al. (2025).

| step | where |
|---|---|
| choose the structures | `interactions/fetch_pdb_controls.py`, `add_homodimer_controls.py`, `extract_predictomes.py` — invocations in the notebook. Done once: the chosen sets are frozen in `data/interactions/` and deposited with the data, because RCSB grows and the prediction releases are third-party downloads. |
| interface vs. control statistics, per complex | `interactions/main.py`, once per source (configs in `config/`; `interactions/run_main.sge` submits them to SGE) → `output/interactions/` |
| inter-chain clash filter | `interactions/compute_interchain_clashes.py` → `output/interactions/interchain_cbeta_clashes.csv` |
| every panel | `Fig6_a-d_and_associated_ED.ipynb` → `output/figures/fig6/`; the structure renders are drawn with PyMOL through helpers in `interactions/ppi_utils.py` |

It has its own environment, `interactions/environment.yaml`
(`labelseq_mapk_interactions`), pinned to the versions its figures were made with:
the code was written against pandas 1.5, and pandas 3 changes its behaviour
without raising. The structure panels need PyMOL (`$PYMOL`, default
`~/pymol/pymol`) and `main.py` needs DSSP 3.1.4 (`mkdssp`); the figures use
Arial (falling back to Liberation Sans if it is not installed).

## Two things worth knowing before reading the tables

**A row is a variant *effect*, not a variant.** The key is
`(variant, library, assay, assay_treatment)`; the same variant appears in up to
eight rows. Aggregating without grouping on that key mixes assays and treatments.

**`library` is not `protein`.** Several proteins were scanned in two halves —
`araf_cterm` and `araf_nterm` are different constructs of ARAF, each spanning the
full protein.

## Supplementary Table 2

`output/Supplementary_Table_2.tsv` is the released score table: 531,797 variant
effects × 54 columns, one row per `(variant, library, assay, assay_treatment)`,
written by `Annotations.ipynb` (layout by Jessica Simon; the schema — every
column's source name and the order — is `src/labelseq_mapk/supplementary_table.py`).
`output/annotated_combined.tsv` is the same table, same names and row order,
plus the six columns a notebook or the interaction pipeline reads that the
release does not carry. `scripts/compare_supplementary_table_2.py` compares two
releases cell by cell, and `scripts/audit_supplementary_table_2.py` checks a
release against its own definitions before upload. Every column is described in
[`docs/supplementary_table_2_columns.md`](docs/supplementary_table_2_columns.md).

The four HSP90 columns come from `src/labelseq_mapk/hsp90_calls.py` (ported
from Jessica's Fig5 export, plus the gap-rule `dependent` the Fig 5 panels use,
which is the paper's definition of HSP90 dependence). `buffering_class` there is
computed on the standard-adjusted score; the Fig 5 panels use the WT-relative
score, and the two disagree for <1% of variants.

All of Us observation (`in_aou`) is not released: it is Controlled Tier data. It
stays in `output/annotated_combined.tsv`, which the Fig 4 depletion panels read,
so that table must not be deposited as is.

The table behind ED 8b is fitted by `scripts/fit_buffering_factors_per_protein.py`
(run it after `Annotations.ipynb`, before `Fig5_and_associated_ED.ipynb`).

Every dominant-negative number in the paper can be recounted from this table:
`dominant_negative == True` gives 14,412 calls (missense 11,514, 3nt deletion
1,592 of which 621 are `delins_2for1`, nonsense 1,306), all in baseline activity
rows, one row per variant.

## The two intermediate score tables

The pipeline also writes two intermediate tables. Which one you want depends on what you are doing.

| table | columns | what it is |
|---|---|---|
| **`raw_scores.tsv`** | 38 | The measurements. Identity, reference sequence (UniProt isoform, RefSeq, Ensembl, and GRCh38 coordinates where the protein change has a resolvable nucleotide route), barcode support, the **uncorrected** per-replicate ratios and scores with their standard curve, and the low/wt-like/high classification. Written by `scripts/export_raw_scores.py`. |
| **`scores_reannotated.tsv`** | 96 | Everything the figures need: the gain-corrected replicates, dominant-negative calls, the HSP90 dependence and buffering families, structure, conservation, PTMs, clinical and population annotation. Written by `scripts/reannotate_scores.py`. |

Each table has its own column reference — every column, in file order, with what
it means and where it came from:
[`docs/scores_reannotated_columns.md`](docs/scores_reannotated_columns.md) and
[`docs/raw_scores_columns.md`](docs/raw_scores_columns.md).

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

- the eight notebooks, and `utils.py` (loaders and constants for `Annotations.ipynb`)
- `scripts/` — `gain_correction.py` (the replicate correction),
  `reannotate_scores.py` (builds the annotated table), `export_raw_scores.py`
  (builds the raw table)
- `src/labelseq_mapk/` — `config.py` and `annotation.py` (the annotation engine),
  `supplementary_table.py` (the release schema), `hsp90_calls.py` (the HSP90
  columns) and `figure_data.py` (the figure notebooks' loader)
- `config/` — the four YAML files, plus the three interaction-analysis configs and
  `visualization_config.yaml`
- `interactions/` — the figure 6 code: structure selection, `main.py`, the clash
  cache, `ppi_utils.py` (loaders, statistics, plotting and PyMOL render helpers),
  and its `environment.yaml`
- `data/dn_cutoffs_empty_vector.tsv` — the per-library empty-vector DN thresholds,
  the one data file small enough to version
- `docs/` — a column reference for each of the two tables
- `environment.yaml`, `environment_figures.yaml`

**Not here: any data.** The barcode counts, the score tables and the annotated
tables are orders of magnitude too large for GitHub, and the annotation source
files are third-party downloads. `config/paths.yaml` is therefore a template:
its paths point into a `data/inputs/` tree that you populate, not at the
locations they were built from. The tables are available on request and will be
deposited with the manuscript. The same holds for the frozen structure sets in
`data/interactions/` (about 1 GB: 343 processed PDB complexes, 771 Predictomes
models, 355 Zhang et al. models, with their manifests).

## Environment

```
conda env create -f environment.yaml
conda activate labelseq_mapk

# the figure notebooks (Fig1–Fig6e–i): pandas 2.3, matplotlib 3.11.1 — the
# versions their figures were made with
conda env create -f environment_figures.yaml
python -m ipykernel install --user --name labelseq_mapk_figures
# ED 1a also needs R 4.5 with Bioconductor drawProteins

# figure 6a–d only
conda env create -f interactions/environment.yaml
python -m ipykernel install --user --name labelseq_mapk_interactions
```
