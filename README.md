# MAPK LABEL-seq — a pathway-scale variant effect atlas

Functional characterisation of variant effects across 17 core MAPK pathway
proteins, measured with
[LABEL-seq](https://pmc.ncbi.nlm.nih.gov/articles/PMC11785348/).

## Authors and contributions

- **Sriram Pendyala** — project design, experiments, data, and analysis.
- **Jessica Simon** — the LABEL-seq method
  ([Simon, Fowler & Maly, *Nature Methods* 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC11785348/))
  and the original scoring and annotation analysis this pipeline was rebuilt from.
- **Claude** (Anthropic, Opus 5) — pipeline, analysis scripts and documentation,
  written with Sriram.

Fowler Lab, Department of Genome Sciences, University of Washington.

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

## Data

The count tables, score tables and annotated tables are far too large for GitHub
and are not in this repository. They are available on request, and will be
deposited with the manuscript.

## Environment

```
conda env create -f environment.yaml
conda activate labelseq_mapk
```
