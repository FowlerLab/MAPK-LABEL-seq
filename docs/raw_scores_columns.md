# `raw_scores.tsv` — column reference

**531,509 rows × 38 columns**, tab-separated. Written by
`scripts/export_raw_scores.py`.

The measurements and identity, with no derived analysis. One row is a **variant
effect**: one variant in one library, one assay, one treatment arm. The key is
`(variant, library, assay, assay_treatment)` — the same variant appears in up to
eight rows.

**What is deliberately absent**: the replicate gain correction and everything built
on it, the dominant-negative calls, the HSP90 dependence and buffering families, and
every structural, conservation, PTM, clinical, population and published-DMS
annotation. Those live in `scores_reannotated.tsv`, documented in
`scores_reannotated_columns.md`.

Rows are not filtered by assay: all three assays and all treatment arms have raw
scores, so all are present.

Columns in file order. `filled` is the percentage of rows that are non-null.

| # | column | group | filled | description |
|---|---|---|---|---|
| 0 | `variant` | identity | 100.0% | Canonical protein-level identity: `A11C`, `A11-` (deletion), `A11A` (synonymous). |
| 1 | `hgvs_p` | identity | 99.9% | The same change in HGVS protein syntax, e.g. `p.Ala11Cys`. Unprefixed. |
| 2 | `Wild Type Residue` | identity | 99.9% | Reference residue. Null on control and wild-type rows. |
| 3 | `Mutation` | identity | 99.9% | Substituted residue; `-` deletion, `*` stop, plus `fs`, `multi`, `delins…`, `complex`. Null on control and wild-type rows. |
| 4 | `Position` | identity | 99.9% | Residue position in protein numbering. Integer; null on control and wild-type rows. |
| 5 | `Mutation Type` | identity | 100.0% | Fine-grained class: missense, synonymous wild type, deletion, frame shift, nonsense, delins_2for1, multi_change, standard, wild type, deletion_multi, unrecognised. |
| 6 | `variant_category` | identity | 100.0% | The 8 reporting classes: missense, synonymous, 3nt deletion, nonsense, frameshift, other, standard, WT. |
| 7 | `variant_class` | identity | 100.0% | `original` or `recovered` — whether the variant survived the older single-substitution filter, or is one the full recovery adds. |
| 8 | `library` | identity | 100.0% | 22 libraries. **Not** the same as protein: `araf_cterm` and `araf_nterm` are different constructs of ARAF. |
| 9 | `protein` | identity | 100.0% | 17 proteins. On a control row this records where the measurement was made, not what the construct is. |
| 10 | `assay` | identity | 100.0% | `abundance`, `activity` or `interaction`. |
| 11 | `assay_treatment` | identity | 100.0% | `No_treatment`, `HSP90i`, `DMSO`, `CIAR`, `SerumStarve`, `LZTR1koCIAR`, `LZTR1ko`, `LZTR1`. `No_treatment` and `DMSO` both mean untreated. |
| 12 | `count_data_delivery` | identity | 100.0% | Provenance stamp of the count tables. |
| 13 | `uniprot_id` | reference | 99.9% | The sequence-verified **isoform**, e.g. `P01116-2` for KRAS4B. Blank on control rows. |
| 14 | `uniprot_accession` | reference | 99.9% | Base UniProt accession, isoform-agnostic. Blank on control rows. |
| 15 | `ensembl_protein` | reference | 93.5% | Versioned ENSP. Null for KSR1, which has none, and on control rows. |
| 16 | `refseq_protein` | reference | 99.9% | Versioned RefSeq protein. Blank on control rows. |
| 17 | `mane_select` | reference | 99.9% | Whether this proteoform **is** the MANE Select one. Blank on control rows. |
| 18 | `hgvs_c` | reference | 32.4% | Transcript-level HGVS, e.g. `NM_001654.5:c.328_330del`. |
| 19 | `hgvs_g` | reference | 32.4% | Genomic HGVS on **GRCh38**. |
| 20 | `g_snvs_grch38` | reference | 29.1% | Every single-nucleotide route to this protein change on **GRCh38**, pipe-separated `chrom:pos:ref:alt`. A list, because a protein change is usually reachable by more than one codon change. |
| 21 | `clingen_allele_id` | reference | 91.1% | ClinGen Allele Registry identifier. |
| 22 | `Number of Barcodes` | support | 100.0% | Barcodes carrying this variant, before the per-replicate quantifiability filter. Integer. |
| 23 | `average_num_quant_bc` | support | 100.0% | Quantifiable barcodes, averaged over the three replicates. The `>= 5` inclusion cutoff is applied on this. |
| 24 | `variant_frequency` | support | 100.0% | This variant's share of all reads in its cell. |
| 25 | `ratio_1` | scores | 100.0% | Raw channel ratio for replicate 1, per barcode then averaged. Activity **pEM1/E40**, abundance **Flag/HT**, interaction **Strep/Flag**. Scales with sequencing depth, so **not comparable across replicates or libraries**. |
| 26 | `ratio_2` | scores | 100.0% | As `ratio_1`, replicate 2. |
| 27 | `ratio_3` | scores | 99.3% | As `ratio_1`, replicate 3. |
| 28 | `average ratio` | scores | 100.0% | Mean of the three ratios; inherits the same incomparability. |
| 29 | `score_1` | scores | 100.0% | **Raw** WT-relative score: `ratio_1` divided by the mean `ratio_1` over that replicate's wild-type barcodes. Wild type sits at 1 by construction. |
| 30 | `score_2` | scores | 100.0% | As `score_1`, replicate 2. |
| 31 | `score_3` | scores | 99.3% | As `score_1`, replicate 3. |
| 32 | `average_score` | scores | 100.0% | Mean of the three raw replicate scores. **Not** the corrected mean. |
| 33 | `std_adj_score_1` | scores | 99.3% | Standard-adjusted score for replicate 1: `score_1 / m_1`, with `m_1` fitted as `y = m·x` through the origin over the assigned standards, **on the raw replicate**. Puts libraries on a common axis. |
| 34 | `std_adj_score_2` | scores | 99.3% | As `std_adj_score_1`, replicate 2. |
| 35 | `std_adj_score_3` | scores | 99.3% | As `std_adj_score_1`, replicate 3. |
| 36 | `average_std_adj_score` | scores | 99.3% | Mean of the three standard-adjusted scores. |
| 37 | `classification_2.5pct` | classification | 100.0% | `low` / `wt-like` / `high` against the 2.5th–97.5th percentile of the cell's **synonymous** variants, computed on `average_score` — i.e. on the raw scores in this file. |

---

## Reading it correctly

* **A row is a variant effect, not a variant.** Group by
  `(library, assay, assay_treatment)` or you will mix assays and treatment arms.
* **`library` is not `protein`.** Several proteins were scanned in two halves,
  each spanning the full protein.
* **Control rows keep their measurement but carry no reference sequence.** The
  spiked BRAF standards, `empty_vector_std` and `NoVar_std` are measured in every
  library, so a BRAF standard appears under KRAS. Those rows are real and are kept,
  marked `standard` in `Mutation Type` and `variant_category`, with `library` and
  `protein` recording where the measurement was made — but the five reference
  columns are blank, because a spike-in is not a variant of the host protein.
  Filter on `variant_category != "standard"` for any variant-level analysis.
* **`std_adj_score_j` is empty for the KRAS interaction assay**, which has no
  assigned standards and therefore no curve to fit.
* **Genomic coordinates are partial by nature** — filled for 96.5% of synonymous
  variants, about 30% of missense and nonsense, and none of the frameshifts or
  multi-residue deletions, which no single substitution produces.
* **Nonsense and frameshift are absent for abundance and interaction in the six
  C-terminally MCP-tagged libraries** (`met`, `ret`, `egfr`, `erbb2`, `sos1`,
  `sos2`): truncation removes the tag before it can be translated, so the readout
  measures nothing.

## Names that differ from `scores_reannotated.tsv`

The two tables are joinable, so these distinctions have to survive a join.

| here | there |
|---|---|
| `average_score` — mean of the **raw** replicates | `average score` — mean of the **corrected** replicates |
| `std_adj_score_j` — curve on the **raw** replicates | `intercept_0_std_adj_score_j` — curve on the **corrected** ones (`raw_std_adj_score_j` is the raw-replicate version) |
| `classification_2.5pct` — computed on the raw mean | same name, computed on the corrected mean; the two differ on 531 rows |
