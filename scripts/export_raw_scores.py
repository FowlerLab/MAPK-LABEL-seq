"""Export the raw score table: measurement and identity, no derived analysis.

`scores_reannotated.tsv` carries everything the figures need -- the gain-corrected
replicates, the dominant-negative calls, the chaperone-dependency and buffering
families, structure, conservation, population and tumour observation. That is the
right table for the analyses in this paper and the wrong one to hand to somebody
who wants the measurements.

This writes that smaller table. It is a strict column selection, so every value in
it is identical to the corresponding value in the annotated table -- nothing is
recomputed except the two averages named below, which the annotated table does not
carry in an uncorrected form.

WHAT IS IN IT

  identity        which variant, named as a protein consequence, and which cell
                  it was measured in
  reference       the sequence the numbering refers to: UniProt isoform, RefSeq
                  and Ensembl protein, plus GRCh38 genomic coordinates where the
                  protein change has a resolvable single-nucleotide route
  barcode support how many barcodes stand behind the number
  scores          the per-replicate ratios, the WT-relative scores, and the
                  standard-adjusted scores -- all UNCORRECTED
  classification  low / wt-like / high against the cell's synonymous distribution

WHAT IS DELIBERATELY NOT IN IT

  * the replicate gain correction, and anything built on it. `score_j` here is the
    raw WT-relative score; the corrected replicates and the `average score` and
    `intercept_0_std_adj_score_j` built from them live in the annotated table.
  * `DN_EV` and the dominant-negative thresholds.
  * the HSP90 families -- chaperone dependency, buffering, client status.
  * every structural, conservation, PTM, clinical, population and published-DMS
    annotation.

Rows are NOT filtered: all three assays and all treatment arms are present, since
a raw score exists for each. It is the derived columns that are dropped, not the
measurements.

TWO NAMES TO BE CAREFUL WITH. In the annotated table `average score` is the mean of
the CORRECTED replicates and `intercept_0_std_adj_score_j` is the standard curve
fitted on them. Neither name is reused here. This table has `average_score`, the
mean of the three raw replicates, and `std_adj_score_j` / `average_std_adj_score`,
the standard curve fitted on the raw replicates (`raw_std_adj_score_j` upstream).
The two tables are joinable on `(variant, library, assay, assay_treatment)`, so the
distinction has to survive the join.

Output
    output/scoring/raw_scores.tsv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "output" / "scoring" / "scores_reannotated.tsv"
OUT = ROOT / "output" / "scoring" / "raw_scores.tsv"

#: Copied through unchanged, in this order. Grouped as the docstring describes.
IDENTITY = ["variant", "hgvs_p", "Wild Type Residue", "Mutation", "Position",
            "Mutation Type", "variant_category", "variant_class",
            "library", "protein", "assay", "assay_treatment",
            "count_data_delivery"]
REFERENCE = ["uniprot_id", "uniprot_accession", "ensembl_protein",
             "refseq_protein", "mane_select",
             "hgvs_c", "hgvs_g", "g_snvs_grch38", "clingen_allele_id"]
SUPPORT = ["Number of Barcodes", "average_num_quant_bc", "variant_frequency"]
RATIOS = ["ratio_1", "ratio_2", "ratio_3", "average ratio"]
RAW = ["score_1", "score_2", "score_3"]
#: the standard curve fitted on the RAW replicates, renamed on the way out so it
#: cannot be mistaken for the canonical (corrected) calibration
STD_ADJ = {f"raw_std_adj_score_{j}": f"std_adj_score_{j}" for j in (1, 2, 3)}
CLASSIFICATION = ["classification_2.5pct"]

#: Never written, even if they appear upstream. Guarded rather than assumed: this
#: table is the one that goes out, so a new derived column must not reach it by
#: default.
FORBIDDEN_SUBSTRINGS = (
    "corrected", "DN_EV", "dn_threshold", "chaperone", "buffer", "client_status",
    "intercept_0", "average score", "z_score", "synon_wt",
    "plddt", "pdb_aa", "domain", "relative_sasa", "dssp", "active_site",
    "interface", "am_pathogenicity", "alignment_pos", "MOD_RSD", "Modification",
    "ClinVar", "Clinical", "Phenotype", "Origin", "ReviewStatus",
    "NumberSubmitters", "hek_", "feature", "ligandable", "regulatory_PTM",
    "disease_associated", "PMID", "motif", "phylop", "jsd", "ddG",
    "gnomad", "aou", "genie", "nmd",
)


def main() -> int:
    if not SRC.exists():
        print(f"{SRC} not found -- run the scoring and annotation stages first")
        return 1
    d = pd.read_csv(SRC, sep="\t", low_memory=False)
    print(f"read {len(d):,} rows x {len(d.columns)} columns from {SRC.name}")

    wanted = IDENTITY + REFERENCE + SUPPORT + RATIOS + RAW + list(STD_ADJ)
    missing = [c for c in wanted if c not in d.columns]
    assert not missing, f"columns absent from {SRC.name}: {missing}"

    out = d[wanted + CLASSIFICATION].rename(columns=STD_ADJ)

    # The two averages the annotated table has no uncorrected form of. Computed
    # with the same nan-mean as upstream, so a variant scored in two replicates
    # still gets a value.
    out["average_score"] = out[RAW].mean(axis=1)
    out["average_std_adj_score"] = out[list(STD_ADJ.values())].mean(axis=1)

    # order: identity, reference, support, scores, classification
    order = (IDENTITY + REFERENCE + SUPPORT + RATIOS + RAW + ["average_score"]
             + list(STD_ADJ.values()) + ["average_std_adj_score"]
             + CLASSIFICATION)
    out = out[order]

    leaked = [c for c in out.columns
              if any(f.lower() in c.lower() for f in FORBIDDEN_SUBSTRINGS)]
    assert not leaked, f"derived columns reached the raw table: {leaked}"

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, sep="\t", index=False)
    print(f"wrote {OUT.relative_to(ROOT)}")
    print(f"  {len(out):,} rows x {len(out.columns)} columns\n")

    print("column groups")
    for name, cols in (("identity", IDENTITY), ("reference", REFERENCE),
                       ("barcode support", SUPPORT),
                       ("scores", RATIOS + RAW + ["average_score"]
                        + list(STD_ADJ.values()) + ["average_std_adj_score"]),
                       ("classification", CLASSIFICATION)):
        print(f"  {name:16s} {len(cols):2d}  {', '.join(cols)}")

    print("\nfill")
    for c in out.columns:
        print(f"  {c:26s} {out[c].notna().mean():6.1%}")

    print("\nrows by assay x treatment")
    print(out.groupby(["assay", "assay_treatment"]).size()
          .rename("rows").to_frame().to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
