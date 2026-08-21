# `scores_reannotated.tsv` — column reference

**531,797 rows × 96 columns**, tab-separated.

One row is a **variant effect**: one variant measured in one library, one assay and
one treatment arm. The key is `(variant, library, assay, assay_treatment)` — nothing
narrower is unique, because the same variant is measured in up to eight rows.
168,896 distinct variants, 22 libraries, 17 proteins, 3 assays, 8 treatment arms.

Columns are listed **in file order**. `origin` is one of

| origin | produced by | n |
|---|---|---|
| `scoring` | scoring, from raw barcode counts | 38 |
| `annot` | the annotation engine, from primary annotation files | 49 |
| `struct` | per-residue structure, from AlphaFold models | 5 |
| `mapping` | reference identifiers and GRCh38 coordinates | 4 |

`filled` is the percentage of rows that are non-null.

---

| # | column | origin | filled | description |
|---|---|---|---|---|
| 0 | `variant` | scoring | 100% | Canonical protein-level identity: `A11C`, `A11-` (deletion), `A11A` (synonymous). |
| 1 | `Number of Barcodes` | scoring | 100% | Barcodes carrying this variant, before the per-replicate quantifiability filter. |
| 2–4 | `ratio_1` … `ratio_3` | scoring | 99.9 / 99.9 / 99.2% | Raw channel ratio per replicate, computed per barcode then averaged over the variant's barcodes. Activity is **pEM1/E40**, abundance **Flag/HT**, interaction **Strep/Flag**. Scales with each channel's sequencing depth, so **not comparable across replicates or libraries**. |
| 5 | `average ratio` | scoring | 99.9% | Mean of the three ratios; inherits the same incomparability. |
| 6–8 | `score_1` … `score_3` | scoring | 99.9 / 99.9 / 99.2% | **Raw** WT-relative score: `ratio_j` divided by the mean `ratio_j` over that replicate's wild-type barcodes. Wild type sits at 1 by construction; dividing by a same-replicate quantity cancels the per-channel depth factor. |
| 9 | `variant_frequency` | scoring | 100% | This variant's share of all reads in its cell. |
| 10 | `Mutation Type` | scoring | 100% | Fine-grained class: missense, synonymous wild type, deletion, frame shift, nonsense, delins_2for1, multi_change, standard, wild type, deletion_multi, unrecognised. |
| 11 | `Wild Type Residue` | scoring | 100% | Reference residue. `standard` / `wild type` on control and WT rows. |
| 12 | `Mutation` | scoring | 100% | Substituted residue; `-` deletion, `*` stop, plus `fs`, `multi`, `delins…`, `complex`. |
| 13 | `Position` | scoring | 100% | Residue position in protein numbering. **Mixed type**: the strings `standard` and `wild type` appear on 819 control and WT rows. |
| 14 | `hgvs_p` | scoring | 99.9% | The same change in HGVS protein syntax, e.g. `p.Ala11Cys`. Unprefixed; the accession is a separate column. |
| 15 | `average_num_quant_bc` | scoring | 100% | Quantifiable barcodes, averaged over the three replicates. The `>= 5` inclusion cutoff is applied on this; standards are exempt. |
| 16–18 | `corrected_score_1` … `_3` | scoring | 99.9 / 99.9 / 99.2% | **The replicate scores to use.** Identical to `score_j` for 190 of 192 replicates. Two are corrected for a dynamic-range deficit by one exponent about wild type, clamped outside the measured range: `ksr1_cterm` activity untreated rep 3 (×2.003) and `mras` activity untreated rep 2 (×1.979). |
| 19 | `average score` | scoring | 99.9% | Mean of the three **corrected** replicates. The canonical per-variant effect size; the classification and DN logic read this. |
| 20–22 | `intercept_0_std_adj_score_1` … `_3` | scoring | 99.2% | Standard-adjusted score: `corrected_score_j / m_j`, with `m_j` fitted per replicate as `y = m·x` through the origin over the assigned standards. Puts libraries on a common axis. |
| 23–25 | `raw_std_adj_score_1` … `_3` | scoring | 99.2% | The same curve refitted on the **raw** replicates. Diagnostic only — nothing downstream reads it. |
| 26 | `intercept_0_standard-adjusted score` | scoring | 99.2% | Mean of columns 20–22. |
| 27 | `library` | scoring | 100% | 22 libraries. **Not** the same as protein: `araf_cterm` and `araf_nterm` are different constructs of ARAF, each spanning the full protein. |
| 28 | `assay` | scoring | 100% | `abundance` 316,461 · `activity` 211,509 · `interaction` 3,827. |
| 29 | `assay_treatment` | scoring | 100% | `No_treatment`, `HSP90i`, `DMSO`, `CIAR`, `SerumStarve`, `LZTR1koCIAR`, `LZTR1ko`, `LZTR1`. `No_treatment` and `DMSO` both mean untreated; no cell carries both. |
| 30 | `classification_2.5pct` | scoring | 99.9% | `low` / `wt-like` / `high` against the 2.5th–97.5th percentile of the cell's **synonymous** variants. Computed on `average score`, i.e. on the corrected replicates. |
| 31 | `variant_category` | scoring | 100% | The 8 reporting classes: missense, synonymous, 3nt deletion, nonsense, frameshift, other, standard, WT. Every row falls in exactly one. |
| 32 | `uniprot_id` | scoring | 100% | The sequence-verified **isoform**, e.g. `P01116-2` for KRAS4B. Cite this; the base accession alone is ambiguous. |
| 33 | `ensembl_protein` | scoring | 93.6% | Versioned ENSP. Null for KSR1, which has none. |
| 34 | `refseq_protein` | scoring | 100% | Versioned RefSeq protein. |
| 35 | `mane_select` | scoring | 100% | Whether this proteoform **is** the MANE Select one. Informational. |
| 36 | `variant_class` | scoring | 100% | `original` (491,444) or `recovered` (40,353) — whether the variant survived the older single-substitution filter or is one the full recovery adds. |
| 37 | `count_data_delivery` | scoring | 100% | Provenance stamp of the count tables. |
| 38 | `protein` | annot | 100% | 17 proteins. |
| 39 | `uniprot_accession` | annot | 100% | Base UniProt accession, isoform-agnostic. |
| 40 | `average_z_score` | annot | 99.9% | `average score` z-scored within the cell. |
| 41–42 | `synon_wt_mean`, `synon_wt_std` | annot | 100% | The cell's synonymous mean and SD, exposed so downstream code need not recompute them. |
| 43 | `average_z_score_from_syonWT` | annot | 99.9% | `(average score − synon_wt_mean) / synon_wt_std`. Spelling as in the original. |
| 44 | `client_status` | annot | 100% | HSP90 client strength per protein: strong / weak / non / unknown. |
| 45 | `Abund_chaperone_dependency` | annot | 66.6% | `abundance(HSP90i) − abundance(control)`. Negative = loses abundance when HSP90 is inhibited. Filled only for libraries with an HSP90i arm. |
| 46 | `chap_dep_classification` | annot | 67.4% | `decreased` / `wt-like` / `increased` against the synonymous-WT band. |
| 47 | `Abund_chaperone_dependency_zscore_from_synon_WT` | annot | 66.6% | The same, z-scored on the synonymous-WT distribution. |
| 48 | `buffered` | annot | 66.6% | Paired t-test over the three replicates, HSP90i vs control; True when p/2 < 0.05 **and** mean HSP90i < mean control. Computed on the standard-adjusted columns. |
| 49–50 | `percent_buffered`, `percent_buffered_WT_norm` | annot | 66.6% | Magnitude of that shift, raw and wild-type-normalised. |
| 51 | `DN_EV` | annot | 21.2% | **The dominant-negative call.** Activity below the 2.5th percentile of a bootstrap over empty-vector barcode means, per library and treatment. Applies the inhibitory-protein exclusion, and is **NaN rather than False** where a cell has no usable baseline, so "not DN" and "not assessable" stay distinct. |
| 52 | `max_sasa` | annot | 99.8% | Reference maximum solvent-accessible area for that residue type. |
| 53 | `relative_sasa` | annot | 99.8% | `dssp_solvent_accessibility / max_sasa`. Exceeds 1 in places — real, meaning more exposed in the model than in the reference tripeptide. |
| 54 | `dssp_secondary_structure` | annot | 99.8% | 8 DSSP classes. |
| 55 | `dssp_solvent_accessibility_angstroms^2` | annot | 99.8% | Absolute SASA from DSSP. |
| 56 | `MOD_RSD` | annot | 8.0% | PhosphoSitePlus modified residue, e.g. `K104-ub`. |
| 57–59 | `LT_LIT`, `MS_LIT`, `MS_CST` | annot | 3.1 / 6.2 / 4.1% | PhosphoSitePlus evidence counts: low-throughput literature, mass-spec literature, CST mass-spec. |
| 60 | `Ambiguous_Site` | annot | 8.0% | The site assignment is ambiguous. |
| 61 | `Modification` | annot | 8.0% | `phospho` or `ub`. |
| 62–63 | `ON_FUNCTION`, `ON_PROCESS` | annot | 2.3 / 1.3% | Curated molecular consequence and cellular process for that site. |
| 64 | `regulatory_PTM_note` | annot | 2.5% | The two above joined into one string. |
| 65 | `regulatory_PTM` | annot | 100% | The residue is a curated regulatory site. |
| 66 | `disease_associated_phosphosite` | annot | 100% | 0/1; PhosphoSitePlus disease-associated phosphosite. |
| 67–68 | `PMID39091798_Enrichment(ave)_FLshp2_CDsrc`, `PMID39091798_Enrichment(ave)_CDshp2_FLvsrc` | annot | 4.0 / 1.9% | Shah et al. SHP2 DMS enrichment. SHP2 only. |
| 69–75 | `ClinVar_Name`, `ClinicalSignificance`, `PhenotypeList`, `Origin`, `OriginSimple`, `ReviewStatus`, `NumberSubmitters` | annot | 4.4% | ClinVar record as reported. Low fill because most variants here have never been submitted. |
| 76 | `domain` | annot | 100% | 14 values including `none`. From the **curated** bounds in `config/proteins.yaml` — deliberately not UniProt or CDD, which disagree. |
| 77–79 | `hek_rna_tpm`, `hek_protein_conc_nm`, `hek_protein_copy_number` | annot | 100 / 86 / 86% | OpenCell HEK293 endogenous expression. |
| 80 | `feature` | annot | 11.5% | NCBI Conserved Domain features and active sites at that residue, as a list. |
| 81 | `active_site` | annot | 100% | Boolean. |
| 82 | `protein_interface` | annot | 100% | Boolean, from curated PDB complexes at 4 Å all-atom. |
| 83 | `interface_evidence` | annot | 20.0% | Which partner(s) the residue contacts. |
| 84 | `am_pathogenicity` | annot | 80.5% | AlphaMissense pathogenicity score, continuous. Threshold it yourself and state the threshold; AlphaMissense's own three-way call is not carried. |
| 85 | `alignment_pos` | annot | 37.4% | Column index in the human kinase-domain alignment; kinases only. |
| 86 | `ligandable_cys` | annot | 0.2% | CysDB: the cysteine is ligandable. |
| 87 | `plddt` | struct | 99.8% | AlphaFold per-residue confidence. |
| 88 | `pdb_aa` | struct | 99.8% | The residue the model actually has at that position. |
| 89 | `inter_domain_contacts_all_atom` | struct | 100% | Intramolecular domain–domain contact at the project-wide **4 Å all-atom** cutoff. |
| 90 | `inter_domain_partners` | struct | 1.1% | Which domain is contacted. |
| 91 | `pdb_aa_mismatch` | struct | 100% | Model and construct disagree — the guard against silent off-by-one numbering. |
| 92 | `hgvs_c` | mapping | 32.4% | Transcript-level HGVS, e.g. `NM_001654.5:c.328_330del`. |
| 93 | `hgvs_g` | mapping | 32.4% | Genomic HGVS on **GRCh38**. |
| 94 | `clingen_allele_id` | mapping | 91.0% | ClinGen Allele Registry identifier. |
| 95 | `g_snvs_grch38` | mapping | 29.1% | Every single-nucleotide route to this protein change on **GRCh38**, pipe-separated `chrom:pos:ref:alt` (bare chromosome). A list, not one coordinate, because a protein change is usually reachable by more than one codon change. |

The three coordinate columns are filled only where a nucleotide route resolves:
96.5% of synonymous variants, ~30% of missense and nonsense, and **none** of the
frameshifts, multi-residue deletions or multi-mutants, which no single substitution
produces.

---

## Reading it correctly

* **A row is a variant effect, not a variant.** Group by
  `(library, assay, assay_treatment)` or you will mix assays and treatment arms.
* **`library` ≠ `protein`.** Several proteins were scanned in two halves.
* **Use `corrected_score_j` and `average score`**, not `score_j`, unless you
  specifically want the raw replicate. `raw_std_adj_score_j` is diagnostic.
* **`DN_EV` is the dominant-negative call**, and NaN means "not assessable".
* **Control rows carry the host library's identity.** The spiked BRAF standards,
  `empty_vector_std` and `NoVar_std` are measured in every library, so a BRAF
  standard appears under KRAS with KRAS accessions. They are identifiable by
  `Mutation Type` / `variant_category` == `standard`, and 288 of them have no score
  in any replicate. Filter them out for any variant-level analysis.
* **`Position` is mixed type**, carrying `standard` and `wild type` on 819 rows.
* **Nonsense and frameshift are unscorable for abundance and interaction in the six
  C-terminally MCP-tagged libraries** (`met`, `ret`, `egfr`, `erbb2`, `sos1`,
  `sos2`) — truncation removes the tag. Those rows are absent by construction.

A smaller table without any of the derived analysis —
identity, reference sequence, barcode support, the uncorrected scores and the
classification only — is written by `scripts/export_raw_scores.py` as
`raw_scores.tsv`, and documented in `raw_scores_columns.md`.
