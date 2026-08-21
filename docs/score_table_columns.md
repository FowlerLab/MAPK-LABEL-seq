# Columns of the annotated score table

`output/scoring/scores_reannotated.tsv` — **531,797 rows × 92 columns**.

A row is one **variant effect**: one variant measured in one library, one assay and
one treatment arm. The key is `(variant, library, assay, assay_treatment)` — nothing
narrower is unique, since the same variant appears in up to eight rows. 168,896
distinct variants, 22 libraries, 17 proteins, 3 assays, 8 treatment arms.

`Origin` below is one of three, and is recorded per column in
`output/scoring/reannotation_provenance.tsv`:

| origin | produced by | n |
|---|---|---|
| **scoring** | `Scoring.ipynb`, from raw barcode counts | 38 |
| **annot** | `labelseq_mapk.annotation.run_annotation`, from primary annotation files | 49 |
| **struct** | `scripts/compute_structure_annotations.py`, from AlphaFold models | 5 |

`Filled` is the percentage of rows that are non-null.

## Two tables

| file | columns | what it is |
|---|---|---|
| `scores_reannotated.tsv` | 96 | everything: the gain-corrected replicates, the DN calls, the HSP90 families, structure, conservation, clinical, population. Documented below. |
| `raw_scores.tsv` | 38 | measurement and identity only, written by `scripts/export_raw_scores.py`. **531,509 rows.** Identity, reference sequence, barcode support, the **uncorrected** scores with their standard curve, and `classification_2.5pct`. No correction, no `DN_EV`, no HSP90, no structural or external annotation. |

`raw_scores.tsv` is a strict column selection, so every value in it equals the
corresponding value in the annotated table. Two names differ deliberately, because
they mean different things and the tables are joinable on
`(variant, library, assay, assay_treatment)`:

| in `raw_scores.tsv` | in `scores_reannotated.tsv` |
|---|---|
| `score_j`, `average_score` | `score_j` raw; `average score` is the mean of the **corrected** replicates |
| `std_adj_score_j`, `average_std_adj_score` | `raw_std_adj_score_j`; `intercept_0_std_adj_score_j` is fitted on the **corrected** replicates |
| `classification_2.5pct` **recomputed on the raw mean** | the same name, computed on the **corrected** mean — the two differ on 531 rows |

Four further differences, all in `raw_scores.tsv` and all deliberate:

* **Control rows keep their measurement but lose the reference sequence.** The
  spiked BRAF standards, `empty_vector_std` and `NoVar_std` are measured in every
  library, so a BRAF standard appears under KRAS. Those rows are real and are kept,
  marked by `Mutation Type` and `variant_category` == `standard`; `library` and
  `protein` still say where the measurement was made. But `uniprot_id`,
  `uniprot_accession`, `ensembl_protein`, `refseq_protein` and `mane_select` are
  **blank** on them — a spike-in is not a variant of the host protein, and leaving
  the host's accession there would tell a reader it was.
* **No sentinel strings.** Upstream, `Position`, `Wild Type Residue` and `Mutation`
  carry the literal strings `standard` and `wild type`, which makes `Position` a
  mixed-type column. Here those 531 rows are null in all three, and `Position` is
  an integer column.
* **`Number of Barcodes` is an integer**, not a float.
* **Rows with no score are dropped** — 288 of them, all controls. A variant effect
  record needs an effect.

`std_adj_score_j` is empty for the 3,817 rows of the KRAS interaction assay, which
has no assigned standards and therefore no curve to fit.

---

## Identity

| column | origin | filled | description |
|---|---|---|---|
| `variant` | scoring | 100% | Canonical protein-level identity: `A11C`, `A11-` (deletion), `A11A` (synonymous). 168,896 distinct. |
| `hgvs_p` | scoring | 99.9% | The same change in HGVS protein syntax, e.g. `p.Ala11Cys`. Unprefixed — the accession is a separate column. |
| `Wild Type Residue` | scoring | 100% | Reference residue at that position. |
| `Mutation` | scoring | 100% | Substituted residue; `-` for a deletion, `*` for a stop. |
| `Position` | scoring | 100% | Residue position in protein numbering, 1–1390. Stored as a string. |
| `Mutation Type` | scoring | 100% | Fine-grained class, 11 values: `missense` 430,858 · `synonymous wild type` 21,755 · `deletion` 20,777 · `frame shift` 20,703 · `nonsense` 17,235 · `delins_2for1` 12,474 · `multi_change` 7,165 · `standard` 755 · `wild type` 64 · `deletion_multi` 3 · `unrecognised` 8. |
| `variant_category` | scoring | 100% | The 8 reporting classes `Mutation Type` collapses to: `missense`, `synonymous`, `3nt deletion` (33,251 — codon-aligned deletions plus `delins_2for1`, the same 3-nt event out of frame with the codon grid), `nonsense`, `frameshift`, `other` (7,176), `standard`, `WT`. Every row falls in exactly one. |
| `variant_class` | scoring | 100% | `original` (491,444) or `recovered` (40,353) — whether the variant survived the older single-amino-acid-only filter, or is one the full recovery adds. |
| `library` | scoring | 100% | 22 libraries. **Not** the same as protein: `araf_cterm` and `araf_nterm` are different constructs of ARAF. |
| `assay` | scoring | 100% | `abundance` 316,461 · `activity` 211,509 · `interaction` 3,827. |
| `assay_treatment` | scoring | 100% | `No_treatment` 315,072 · `HSP90i` 111,374 · `DMSO` 57,412 · `CIAR` 25,264 · `SerumStarve` 10,372 · `LZTR1koCIAR` 4,238 · `LZTR1ko` 4,238 · `LZTR1` 3,827. `No_treatment` and `DMSO` both mean untreated — which label appears depends on whether the experiment had a drug arm, and no cell carries both. |
| `count_data_delivery` | scoring | 100% | Provenance stamp of the count tables: `260404 (April)`. |

## Reference sequence

| column | origin | filled | description |
|---|---|---|---|
| `protein` | annot | 100% | 17 proteins, from `config/proteins.yaml`. |
| `uniprot_accession` | annot | 100% | Base UniProt accession, e.g. `P01116`. |
| `uniprot_id` | scoring | 100% | The sequence-verified **isoform**, e.g. `P01116-2` for KRAS4B. Cite this one; the base accession alone is ambiguous where an isoform was used. |
| `ensembl_protein` | scoring | 93.6% | Versioned ENSP for the same proteoform. Null for KSR1, which has none. |
| `refseq_protein` | scoring | 100% | Versioned RefSeq protein. |
| `mane_select` | scoring | 100% | Whether our proteoform **is** the MANE Select one (400,640 rows True). Informational; it gates nothing. |
| `hgvs_c` | mapping | 32.4% | Transcript-level HGVS, e.g. `NM_001654.5:c.328_330del`. |
| `hgvs_g` | mapping | 32.4% | Genomic HGVS on **GRCh38**, e.g. `NC_000023.11:g.47565009_47565011del`. |
| `g_snvs_grch38` | mapping | 29.1% | Every single-nucleotide route to this protein change on **GRCh38**, pipe-separated `chrom:pos:ref:alt` (bare chromosome). A protein change is usually reachable by more than one codon change, so this is a list rather than one coordinate. |
| `clingen_allele_id` | mapping | 91.0% | ClinGen Allele Registry identifier. |

The three coordinate columns come from `output/variant_mapping.tsv` and are filled
only where the protein change has a resolvable nucleotide route: 96.5% of
synonymous variants, ~30% of missense and nonsense, and **none** of the
frameshifts, multi-residue deletions or multi-mutants, which no single substitution
produces. `mapping` is a fourth provenance tag alongside the three above.

## Barcode support

| column | origin | filled | description |
|---|---|---|---|
| `Number of Barcodes` | scoring | 100% | Barcodes carrying this variant, before the per-replicate quantifiability filter. Up to 33,010. |
| `average_num_quant_bc` | scoring | 100% | Barcodes actually quantifiable, averaged over the three replicates. The `>= 5` inclusion cutoff is applied on this; standards are exempt. |
| `variant_frequency` | scoring | 100% | This variant's share of all reads in its cell. |

## Scores

The chain is: barcode counts → per-replicate channel ratio → divide by wild type →
average over the variant's barcodes → gain correction → standard curve.

| column | origin | filled | description |
|---|---|---|---|
| `ratio_1/2/3` | scoring | 99.9 / 99.9 / 99.2% | Raw channel ratio for that replicate, computed per barcode then averaged over the variant's barcodes. `CHANNELS` gives `(numerator, denominator)`, so activity is **pEM1/E40**, abundance **Flag/HT**, interaction **Strep/Flag**. Its absolute value scales with each channel's sequencing depth, so it is **not comparable across replicates or libraries**. |
| `average ratio` | scoring | 99.9% | Mean of the three ratios; inherits the same incomparability. |
| `score_1/2/3` | scoring | 99.9 / 99.9 / 99.2% | Raw WT-relative score: `ratio_j` divided by the mean `ratio_j` over that replicate's wild-type barcodes. Wild type sits at 1 by construction, and dividing by a same-replicate quantity is what cancels the per-channel depth factor. |
| `corrected_score_1/2/3` | scoring | 99.9 / 99.9 / 99.2% | **The replicate scores to use.** Identical to `score_j` for 190 of 192 replicates. Two are corrected for a dynamic-range deficit by a single exponent about score = 1, clamped outside the 1st/99th percentiles: `ksr1_cterm` activity untreated rep 3 (×2.003) and `mras` activity untreated rep 2 (×1.979). Method and thresholds in `scripts/gain_correction.py`; per-replicate record in `output/scoring/replicate_correction_log.tsv`. |
| `average score` | scoring | 99.9% | Mean of the three **corrected** replicates. The canonical per-variant effect size, and what the classification and DN logic read. |
| `intercept_0_std_adj_score_1/2/3` | scoring | 99.2% | Standard-adjusted score: `corrected_score_j / m_j`, where `m_j` is fitted per replicate as `y = m·x` through the origin over the assigned standards. Puts libraries on a common axis. |
| `intercept_0_standard-adjusted score` | scoring | 99.2% | Mean of those three. |
| `raw_std_adj_score_1/2/3` | scoring | 99.2% | The same curve refitted on the **raw** replicates. Diagnostic only — it exists so the correction's effect on the calibrated scale is visible. No analysis reads it. |

Assigned standard values (the spiked BRAF controls): activity `R509Y_std` 0.396,
`Wild Type_std` 1.0, `G469T_std` 8.355, `G258E_std` 12.0, `K601D_std` 18.95;
abundance `E695*_std` 0.16, `I592S_std` 0.29, `G258N_std` 0.53, `S727K_std` 0.66,
`P367N_std` 0.71, `Wild Type_std` 1.0. `NoVar_std` and `empty_vector_std` are
controls, not calibrators, and are excluded from the fit.

**BRAF activity is exempt from the standard curve.** The standards *are* BRAF
variants and were never spiked into the BRAF activity libraries (9 and 18 standard
barcodes, against 639–1,682 elsewhere), so no curve can be fitted; for
`braf_cterm` and `braf_nterm` activity the standard-adjusted columns carry the
WT-relative score through unchanged.

**Nonsense and frameshift are unscorable for abundance and interaction in the six
C-terminally MCP-tagged libraries** (`met`, `ret`, `egfr`, `erbb2`, `sos1`, `sos2`)
— truncation removes the tag, so the readout measures nothing. Those rows are
dropped, and listed in `output/scoring/unscorable_cterm_mcp.tsv`. Activity is
unaffected: it reads ERK phosphorylation with antibodies against ERK, not the tag.

## Effect classification

| column | origin | filled | description |
|---|---|---|---|
| `classification_2.5pct` | scoring | 99.9% | `low` 140,270 · `wt-like` 337,836 · `high` 53,403, against the 2.5th–97.5th percentile of the cell's **synonymous** variants. A two-sided test. |
| `DN_EV` | annot | 21.2% | **The dominant-negative call.** Per (library, treatment), activity below the 2.5th percentile of a bootstrap over *empty-vector barcode* means — barcodes with no kinase cassette, i.e. the true "no kinase" pathway baseline inside that library. 14,906 True, 97,780 False. Thresholds in `data/dn_cutoffs_empty_vector.tsv`. It applies the `inhibitory_proteins` exclusion itself, and is **NaN rather than False** where a cell has no usable EV baseline, so "not dominant-negative" and "not assessable" stay distinguishable. |
| `average_z_score` | annot | 99.9% | `average score` z-scored within the cell. |
| `synon_wt_mean`, `synon_wt_std` | annot | 100% | The cell's synonymous-variant mean and SD, exposed so downstream code need not recompute them. |
| `average_z_score_from_syonWT` | annot | 99.9% | `(average score − synon_wt_mean) / synon_wt_std`. Spelling as in the original. |

## HSP90

`docs/hsp90_metric_definitions.md` is the authority on which of these sits on the
dependence scale and which on the buffering scale; they are not interchangeable.

| column | origin | filled | description |
|---|---|---|---|
| `client_status` | annot | 100% | HSP90 client strength per protein, from config: `strong` 187,882 · `weak` 126,213 · `non` 94,590 · `unknown` 123,112. |
| `Abund_chaperone_dependency` | annot | 66.6% | `abundance(HSP90i) − abundance(control)`. Negative means the variant loses abundance when HSP90 is inhibited. Filled only for libraries with an HSP90i arm. |
| `Abund_chaperone_dependency_zscore_from_synon_WT` | annot | 66.6% | The same, z-scored on the synonymous-WT distribution. |
| `chap_dep_classification` | annot | 67.4% | `decreased` 11,194 · `wt-like` 297,147 · `increased` 50,227, against the synonymous-WT band. |
| `buffered` | annot | 66.6% | Paired t-test over the three replicates, HSP90i vs control; True when p/2 < 0.05 **and** mean HSP90i < mean control. 222,830 True · 131,119 False. Computed on the standard-adjusted replicate columns. |
| `percent_buffered` | annot | 66.6% | Magnitude of that shift. |
| `percent_buffered_WT_norm` | annot | 66.6% | The same, wild-type-normalised. |

## Structure

| column | origin | filled | description |
|---|---|---|---|
| `plddt` | struct | 99.8% | AlphaFold per-residue confidence, 23.1–98.9. |
| `pdb_aa` | struct | 99.8% | The residue the model actually has at that position. |
| `pdb_aa_mismatch` | struct | 100% | True where model and construct disagree (3,104 rows). The guard against silent off-by-one numbering. |
| `inter_domain_contacts_all_atom` | struct | 100% | Intramolecular domain–domain contact at the project-wide **4 Å all-atom** cutoff, 5,603 rows. Every interface, dimer and chaperone contact definition uses this cutoff. |
| `inter_domain_partners` | struct | 1.1% | Which domain is contacted, 9 values. |
| `max_sasa` | annot | 99.8% | Reference maximum solvent-accessible area for that residue type, 97–265 Å². |
| `dssp_solvent_accessibility_angstroms^2` | annot | 99.8% | Absolute SASA from DSSP, 0–317 Å². |
| `relative_sasa` | annot | 99.8% | The ratio of the two. Exceeds 1 in places (max 1.52) — real, and a sign the residue is more exposed in the model than in the reference tripeptide. |
| `dssp_secondary_structure` | annot | 99.8% | 8 DSSP classes. |

## Domains, features and interfaces

| column | origin | filled | description |
|---|---|---|---|
| `domain` | annot | 100% | 14 values including `none`. From the **curated** bounds in `config/proteins.yaml` — deliberately not UniProt or CDD, which disagree (ARAF by 65 residues). |
| `feature` | annot | 11.5% | NCBI Conserved Domain features and active sites at that residue, as a list. |
| `active_site` | annot | 100% | True on 37,512 rows. |
| `protein_interface` | annot | 100% | True on 106,277 rows, from curated PDB complexes at 4 Å all-atom. |
| `interface_evidence` | annot | 20.0% | Which partner(s) the residue contacts, 215 distinct values. |
| `alignment_pos` | annot | 37.4% | Column index in the human kinase-domain alignment; kinases only. Conservation itself (`jsd_conservation`, Capra–Singh) is computed elsewhere and is not in this table. |

## Post-translational modification — PhosphoSitePlus

| column | origin | filled | description |
|---|---|---|---|
| `MOD_RSD` | annot | 8.0% | The modified residue as PhosphoSitePlus names it, e.g. `K104-ub`. 628 distinct. |
| `Modification` | annot | 8.0% | `phospho` or `ub`. |
| `LT_LIT` | annot | 3.1% | Low-throughput literature reports, 1–194. |
| `MS_LIT` | annot | 6.2% | Mass-spec literature reports, 1–54. |
| `MS_CST` | annot | 4.1% | Cell Signaling Technology mass-spec observations, 1–1,860. |
| `Ambiguous_Site` | annot | 8.0% | Flag: the site assignment is ambiguous. |
| `ON_FUNCTION` | annot | 2.3% | Curated molecular consequence of modifying this site. |
| `ON_PROCESS` | annot | 1.3% | Curated cellular process affected. |
| `regulatory_PTM_note` | annot | 2.5% | `ON_FUNCTION` and `ON_PROCESS` joined into one string. |
| `regulatory_PTM` | annot | 100% | True on 13,038 rows — the residue is a curated regulatory site. |
| `disease_associated_phosphosite` | annot | 100% | 0/1; the residue is a PhosphoSitePlus disease-associated phosphosite. |

## Clinical and predicted effect

| column | origin | filled | description |
|---|---|---|---|
| `ClinVar_Name` | annot | 4.4% | Full ClinVar variant name, 7,777 distinct. Most variants here have never been submitted, hence the low fill. |
| `ClinicalSignificance` | annot | 4.4% | 18 values as reported. |
| `PhenotypeList` | annot | 4.4% | Reported conditions, 1,375 distinct. |
| `Origin` | annot | 4.4% | ClinVar allele origin, 28 values. |
| `OriginSimple` | annot | 4.4% | Collapsed: `somatic`, `germline`, `germline/somatic`, `unknown`. |
| `ReviewStatus` | annot | 4.4% | ClinVar review status, 7 values. |
| `NumberSubmitters` | annot | 4.4% | Submitting labs, 1–75. |
| `am_pathogenicity` | annot | 80.5% | AlphaMissense pathogenicity score, 0.028–1.0. Missense with a prediction. Continuous only — threshold it yourself and state the threshold. |

## Expression and other published data

| column | origin | filled | description |
|---|---|---|---|
| `hek_rna_tpm` | annot | 100% | OpenCell HEK293 endogenous transcript abundance. |
| `hek_protein_conc_nm` | annot | 86.0% | OpenCell endogenous protein concentration, nM. |
| `hek_protein_copy_number` | annot | 86.0% | OpenCell copies per cell. |
| `ligandable_cys` | annot | 0.2% | CysDB: the cysteine is ligandable. True where present. |
| `PMID39091798_Enrichment(ave)_FLshp2_CDsrc` | annot | 4.0% | Shah et al. SHP2 DMS enrichment, full-length SHP2 / SRC catalytic domain. SHP2 only. |
| `PMID39091798_Enrichment(ave)_CDshp2_FLvsrc` | annot | 1.9% | The same for SHP2 catalytic domain / full-length SRC. |
