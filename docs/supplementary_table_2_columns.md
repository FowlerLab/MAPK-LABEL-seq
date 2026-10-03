# Supplementary Table 2 — column reference

`Supplementary_Table_2.tsv` — **531,797 rows × 54 columns**, tab-separated, plain
ASCII, Unix line endings, no index column. A blank cell is a missing value.

One row is a **variant effect**: one variant measured in one library, one assay and
one treatment arm. The key is `(library, variant, assay, assay_treatment)` — nothing
narrower is unique, because the same variant is measured in up to eight rows.
200,361 distinct (protein, variant) pairs over 17 proteins, 22 libraries, 3 assays
and 8 treatment arms. (Counting distinct `variant` strings alone gives 168,883, an
undercount: the same string in two proteins names two different variants.)

Columns are listed **in file order**. `origin` is one of

| origin | produced by | n |
|---|---|---|
| `scoring` | scoring, from raw barcode counts (`Scoring.ipynb`) | 24 |
| `annot` | the annotation layer, from primary annotation files (`Annotations.ipynb`) | 24 |
| `struct` | per-residue structure, from the AlphaFold models | 2 |
| `hsp90` | the per-variant HSP90-inhibition calls (`src/labelseq_mapk/hsp90_calls.py`) | 4 |

`filled` is the percentage of rows that are non-blank. Booleans are written
`True` / `False`; blank means **not assessed** (the chaperone-contact columns are
the one exception, see below).

---

## Identity

| # | column | origin | filled | description |
|---|---|---|---|---|
| 1 | `protein` | annot | 100% | 17 proteins, lower case: `araf braf craf egfr erbb2 grb2 kras ksr1 ksr2 mek1 mek2 met mras ret shp2 sos1 sos2`. `met` is the receptor called HGFR in the paper; `kras` is KRAS4B; `mek1`/`mek2` are MAP2K1/MAP2K2; `shp2` is PTPN11. |
| 2 | `library` | scoring | 100% | 22 libraries. **Not** the same as protein: ARAF, BRAF, CRAF, KSR1 and KSR2 were each scanned in two libraries, `<protein>_nterm` and `<protein>_cterm`, which are different constructs of the full protein. A variant measured in both appears once per library. |
| 3 | `variant` | scoring | 100% | Protein-level identity in one-letter code, in the protein's own numbering: `A11C` missense, `A11A` synonymous, `A11*` nonsense, `A11-` single-residue deletion, `A11fs` frameshift, `A764_S765delinsG` a 3-nt deletion that straddles two codons, `E425_R426del` a two-residue deletion, `E62D\|K84N` a multi-mutant (changes joined by `\|`), `WT` wild type, plus the control names. |
| 4 | `assay` | scoring | 100% | `abundance` (Flag / HaloTag, how much protein there is) 316,461 · `activity` (pERK reporter, what the variant does to pathway output) 211,509 · `interaction` (Strep / Flag, KRAS binding to LZTR1) 3,827. |
| 5 | `assay_treatment` | scoring | 100% | `No_treatment` 315,072 · `HSP90i` (pimitespib) 111,374 · `DMSO` 57,412 · `CIAR` 25,264 · `SerumStarve` 10,372 · `LZTR1koCIAR` 4,238 · `LZTR1ko` 4,238 · `LZTR1` 3,827. `No_treatment` and `DMSO` are both the untreated baseline (DMSO is the vehicle control where a drug arm exists); no library × assay carries both. |
| 6 | `variant_category` | scoring | 100% | The 8 reporting classes; every row is in exactly one. `missense` 430,858 · `3nt deletion` 33,251 · `synonymous` 21,755 · `frameshift` 20,703 · `nonsense` 17,235 · `other` 7,176 (multi-mutants and deletions of more than one residue) · `standard` 755 · `WT` 64. `3nt deletion` is every in-frame loss of one codon's worth of sequence, **including** the 3-nt deletions that straddle two codons and so are written as a `delins`. |
| 7 | `wild_type_residue` | scoring | 100% | Reference residue at `position`, one-letter. Checked against the reference sequence at every position. On control rows it holds `standard` / `wild type`. |
| 8 | `mutation` | scoring | 100% | The new residue: an amino acid, `*` stop, `-` deletion, or `fs`, `multi`, `delins…`, `complex` for the other classes. |
| 9 | `position` | scoring | 100% | Residue number in the protein (`uniprot_id`) numbering. **Mixed type**: the strings `standard` and `wild type` appear on the 819 control and WT rows; every library variant has a number. A multi-residue change is placed at its first residue. |
| 10 | `hgvs_p` | scoring | 99.9% | The same change in HGVS protein syntax, e.g. `p.Ala11Cys`, `p.Ala11=`, `p.Ala11del`. Unprefixed; the accession is `uniprot_id`. Blank only on the spiked standards. |

## Scores

Every score is a ratio of channels read from barcode counts, averaged over the
variant's barcodes. Activity is pEM1 / E40, abundance Flag / HaloTag,
interaction Strep / Flag.

| # | column | origin | filled | description |
|---|---|---|---|---|
| 11–13 | `score_1` … `score_3` | scoring | 99.9 / 99.9 / 99.2% | **Raw** WT-relative score per replicate: the variant's ratio divided by the mean ratio over that replicate's wild-type barcodes. Wild type is 1 by construction. Kept for transparency; use 14–16. |
| 14–16 | `WT_relative_score_replicate_1` … `_3` | scoring | 99.9 / 99.9 / 99.2% | **The replicate scores to use.** Identical to `score_j` in 190 of the 192 replicates. Two replicates resolved materially less of their assay's dynamic range than their siblings and are corrected by a single exponent about wild type (clamped outside the measured range): `mras` activity, replicate 2 (exponent 1.979) and `ksr1_cterm` activity, replicate 3 (2.003). |
| 17 | `mean_WT_relative_score` | scoring | 99.9% | Mean of 14–16. **The per-variant effect size**: 1 = wild type, below 1 = less abundance / activity / binding than wild type. Re-anchored to wild type in every library × assay × treatment, so comparing it across treatment arms compares each to its own wild type. `variant_classification` and `dominant_negative` are computed on it. |
| 18–20 | `standardized_score_replicate_1` … `_3` | scoring | 99.2% | `WT_relative_score_replicate_j / m_j`, with `m_j` the slope of a line through the origin fitted per replicate over the spiked BRAF standards. Puts libraries and treatment arms on a common axis, so — unlike column 17 — it keeps a shift of wild type itself (e.g. wild type losing abundance under HSP90 inhibition). |
| 21 | `mean_standardized_score` | scoring | 99.2% | Mean of 18–20. The HSP90 calls (24–26, 54) are computed on this scale. |
| 22 | `variant_classification` | scoring | 99.9% | `low` 140,270 · `wt-like` 337,836 · `high` 53,403: column 17 below the 2.5th / above the 97.5th percentile of the **synonymous** variants of the same library × assay × treatment. |
| 23 | `dominant_negative` | annot | 22.5% | **The dominant-negative call** (14,412 True). True when baseline activity (`activity`, `No_treatment`/`DMSO`) is below the empty-vector threshold of its library: the 2.5th percentile of 500 bootstrap means of 10 empty-vector barcodes, median over 200 seeds — i.e. the variant lowers pathway output below what no added protein gives. Assessed only on baseline activity rows, only for missense, 3-nt deletions and nonsense, and not for the five proteins whose wild type itself lowers pathway output (GRB2, KSR1, KSR2, MEK1, MEK2), where "below empty vector" is the wild-type phenotype. Blank = not assessed. 11,514 missense · 1,592 3-nt deletion · 1,306 nonsense. |

## HSP90 inhibition

Computed for the ten proteins with an HSP90i abundance arm (ARAF, BRAF, CRAF,
EGFR, HGFR, KSR2, MEK1, MEK2, RET, SOS2) on the `mean_standardized_score` of
abundance, pairing each variant's untreated arm (DMSO where present) with its
HSP90i arm. Each call is per variant and is **repeated on every row of that
variant** in the library — abundance and activity alike — so filter to one row per
variant before counting.

| # | column | origin | filled | description |
|---|---|---|---|---|
| 24 | `buffering_class` | hsp90 | 64.2% | Against the synonymous 2.5th percentile of each arm: `Buffered` 50,896 — wild-type-like untreated, low under HSP90i; `Poorly buffered` 58,880 — low in both; `WT-like or high` 231,551 — low in neither. Blank where low untreated but not under HSP90i, or not measured. |
| 25 | `dependent` | hsp90 | 66.6% | **The paper's HSP90-dependence call** (Fig 5d, ED 8a and every "dependent" number in the text). True when abundance falls under HSP90 inhibition by at least the gap between wild type and the bottom of the synonymous distribution in the untreated arm: `untreated − HSP90i ≥ max(0, WT_untreated − synonymous 2.5th percentile_untreated)`, on `mean_standardized_score`, the gap computed per library. Reproduces the Fig 5d fractions exactly; ED 8a pools a split protein's two libraries into one gap, so its CRAF and BRAF fractions differ in the third decimal. Filled for every variant with both arms; the paper's fractions are over missense and 3-nt deletions. |
| 26 | `dependent_threshold` | hsp90 | 66.6% | True when abundance under HSP90i is below the HSP90i synonymous 2.5th percentile — i.e. "low under HSP90 inhibition". Equals `buffering_class` ∈ {Buffered, Poorly buffered} wherever that is filled. |

## Barcode support and reference

| # | column | origin | filled | description |
|---|---|---|---|---|
| 27 | `total_observed_barcodes` | scoring | 100% | Barcodes carrying this variant before any filter. |
| 28 | `mean_number_quantified_barcodes` | scoring | 100% | Barcodes that passed the per-replicate quantifiability filter, averaged over the three replicates. Every library variant has ≥ 5 (the inclusion rule); the spiked standards are exempt. |
| 29 | `uniprot_id` | scoring | 100% | The sequence-verified UniProt **isoform** the numbering refers to, e.g. `P01116-2` for KRAS4B. Cite this; the base accession alone can be ambiguous. Control rows carry their host library's accession. |
| 30 | `client_status_literature` | annot | 100% | The protein's HSP90-client strength **as reported in the literature**, not measured here: `strong` ARAF, CRAF, ERBB2, KSR1, KSR2 · `weak` BRAF, HGFR, RET · `non` EGFR, MEK1, MEK2 · `unknown` the rest. ERBB2 and KSR1 were not tested with HSP90i in this study. |

## Structure

Per residue, so constant across every row at a (protein, position). Blank on
control rows and positions the models do not cover.

| # | column | origin | filled | description |
|---|---|---|---|---|
| 31 | `max_sasa` | annot | 99.8% | Reference maximum solvent-accessible area of the wild-type residue type (Å²). |
| 32 | `relative_sasa` | annot | 99.8% | Column 33 / column 31. **Buried** in the paper means `relative_sasa < 0.25`. Exceeds 1 at 1,349 rows (max 1.52) — real: more exposed in the model than in the reference tripeptide. |
| 33 | `dssp_solvent_accessibility_angstroms^2` | annot | 99.8% | Absolute solvent-accessible area from DSSP on the AlphaFold model (Å²). |
| 34 | `domain` | annot | 100% | 14 values: `kinase`, `rbd`, `crd`, `braf_specific`, `g_domain`, `gef`, `phosphatase`, `n_sh3`, `c_sh3`, `sh2`, `sh2_1`, `sh2_2`, `sam`, `none`. Curated bounds (`config/proteins.yaml`), deliberately not UniProt or the NCBI CDD, which disagree with each other and with the literature for some proteins. |
| 35 | `HEK293_protein_concentration_nM` | annot | 86.0% | Endogenous protein concentration of that protein in HEK293 cells (OpenCell). Per protein. Blank for RET and KSR2, which OpenCell does not report. |
| 36 | `feature` | annot | 12.6% | NCBI Conserved Domain Database site features at this residue (e.g. `ATP binding site; active site`), semicolon-separated, from a CD-search of each protein's sequence. |
| 37 | `active_site` | annot | 100% | The residue is a catalytic, nucleotide-binding or substrate-binding site, from the CDD site features plus manual curation. |
| 38 | `kinase_alignment_position` | annot | 37.4% | Column of this residue in a human kinase-domain multiple alignment, so equivalent residues of different kinases share a number. Kinase domains only (ARAF, BRAF, CRAF, EGFR, ERBB2, HGFR, RET, KSR1, KSR2, MEK1, MEK2). |
| 39 | `AF_plddt` | struct | 99.8% | AlphaFold per-residue confidence, 0–100. For KRAS the model is of KRAS4A, so it describes the wrong isoform over the C-terminal hypervariable region (from residue 151). |
| 40 | `inter_domain_contacts_all_atom` | struct | 100% | The residue has any atom within 4 Å of an atom in a different domain of the same protein, in the AlphaFold model. |

## Chaperone contacts

From the cryo-EM HSP90–CDC37–BRAF (PDB 7ZR0) and HSP90–CDC37–CRAF (8U1L)
complexes, at 4 Å all-atom: the union of the two contact sets in kinase-alignment
columns, transferred to all 11 kinases through the alignment, and split at the
boundary of the unfolded N-lobe stretch (BRAF E533 / CRAF E425). **Only meaningful where
`chaperone_contact_resolved` is True** — elsewhere the three contact columns read
False because the residue could not be assessed, not because it makes no contact.

| # | column | origin | filled | description |
|---|---|---|---|---|
| 41 | `at_hsp90_unfolded` | annot | 100% | Contacts HSP90 within the N-lobe stretch that is unfolded and threaded through HSP90 in the complex (at or before the boundary). |
| 42 | `at_hsp90_folded` | annot | 100% | Contacts HSP90 from the folded part of the kinase (after the boundary). |
| 43 | `at_cdc37` | annot | 100% | Contacts CDC37 from the folded part of the kinase. A residue can contact both CDC37 and HSP90. |
| 44 | `at_any_chaperone_contact` | annot | 100% | Any of 41–43. |
| 45 | `chaperone_contact_resolved` | annot | 100% | The residue maps into the kinase alignment (column 38 filled), so 41–44 could be assessed. |

## Conservation and stability

| # | column | origin | filled | description |
|---|---|---|---|---|
| 46 | `phylop_vert` | annot | 72.9% | phyloP 100-vertebrate conservation, averaged over the codon's three bases. Higher = more conserved. Per position; computed for the 12 proteins assessable for dominant negatives, **blank for GRB2, KSR1, KSR2, MEK1 and MEK2**. |
| 47 | `jsd_conservation` | annot | 34.3% | Jensen–Shannon divergence conservation (Capra & Singh 2007) of the column-38 alignment column across the human kinome — conservation among kinase paralogs, a different axis from column 46 (Pearson r = 0.17). Kinase domains only. |
| 48 | `kinase_motif` | annot | 37.3% | Canonical kinase structural element at this residue: `P_loop`, `beta1`–`beta6`, `alphaC`–`alphaI`, `hinge`, `catalytic_loop`, `DFG`, `activation_loop`, `F_loop`. Kinase domains only; blank between elements. |
| 49 | `ddG_SPURS` | annot | 85.1% | Predicted change in folding free energy for this substitution (SPURS), kcal/mol, positive = destabilizing. Missense, plus 0 on synonymous; blank for other classes. |

## Population and tumour observation

Matched on GRCh38 genomic coordinates `(chromosome, position, reference, alternate)`
of every single-nucleotide route to the protein change, so observations are found
whatever transcript a database annotated them on. These databases record
variants one nucleotide change makes, so restrict any depletion analysis to them:
`min_nt_changes == 1` for missense, and `nmd_zone` filled for nonsense (the
single-nucleotide stops). Filled only for single-codon classes.

| # | column | origin | filled | description |
|---|---|---|---|---|
| 50 | `in_gnomad` | annot | 91.0% | Observed in gnomAD. Filled for missense, synonymous, nonsense and the 3-nt deletions with a resolvable genomic route; blank for frameshifts and multi-residue changes. |
| 51 | `min_nt_changes` | annot | 84.7% | The fewest nucleotide changes that turn the reference codon into one encoding this residue: 1, 2 or 3 for missense; 0 by convention for synonymous. Blank for other classes. |
| 52 | `nmd_zone` | annot | 1.0% | For single-nucleotide stops: `NMD-triggering` (more than 50 nt upstream of the last exon–exon junction, so the transcript would be degraded in a patient) or `NMD-escaping`. The assay expresses cDNA, so neither is subject to NMD here; the column is for interpreting the population data. |
| 53 | `genie_count` | annot | 77.9% | Number of AACR Project GENIE (v19.0) tumour samples carrying this missense variant, matched on GRCh37 coordinates; 0 = not observed. Missense only. |
| 54 | `dependent_significant` | hsp90 | 59.7% | One-sided Welch t-test of the three HSP90i against the three untreated replicate scores (column 18–20 scale), alternative HSP90i < untreated, p < 0.05, **uncorrected for multiple testing**; at least two replicates in each arm. Missense and 3-nt deletions only; repeated on the variant's rows like 24–26. |

---

## Reading it correctly

* **A row is a variant effect, not a variant.** Group by
  `(library, assay, assay_treatment)` before summarising, or assays and treatment
  arms get mixed.
* **`library` ≠ `protein`.** Several proteins were scanned in two libraries.
* **Use `WT_relative_score_replicate_j` and `mean_WT_relative_score`**, not
  `score_j`, unless you specifically want the uncorrected replicate.
* **Two score scales.** `mean_WT_relative_score` puts wild type at 1 in every arm;
  `mean_standardized_score` keeps how wild type itself moves between arms. A change
  between treatment arms means different things on the two.
* **Blank in a boolean means not assessed.** `dominant_negative` is blank
  everywhere except baseline activity rows of eligible variants and proteins.
* **`dependent` is the paper's HSP90 dependence.** `dependent_threshold` ("low
  under HSP90i") and `dependent_significant` (a per-variant t-test) are
  alternative definitions, released for comparison.
* **The HSP90 calls repeat across a variant's rows.** Count them on one row per
  `(library, variant)`.
* **Control rows carry the host library's identity.** The spiked BRAF standards
  and the empty-vector / no-variant controls are measured in every library, so a
  BRAF standard appears under KRAS with KRAS's accession. They are
  `variant_category == "standard"`; 288 of them have no score. Leave them out of
  any variant-level analysis.
* **`position` is mixed type**, holding `standard` / `wild type` on 819 rows.
* **Nonsense and frameshift abundance and interaction are absent in the six
  C-terminally tagged libraries** (`met`, `ret`, `egfr`, `erbb2`, `sos1`, `sos2`):
  a truncation removes the tag, so there is nothing to measure.
* **Per-position annotations (31–48) repeat** on every variant at a position, and
  across a protein's two libraries.
