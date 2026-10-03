"""Schema of Supplementary Table 2 and of the annotated table built around it.

Supplementary Table 2 is the released score table (layout by Jessica Simon,
2026-09-29). ``output/annotated_combined.tsv`` is the same table, same names and
row order, plus the handful of columns a figure notebook or the interaction
pipeline reads that the release does not carry (``EXTRA_COLUMNS``). The
notebooks' variant-type column is derived from ``variant_category`` on load
(``figure_data.load_scores``), so the pipeline's fine-grained ``Mutation Type``
is not carried.
"""
from __future__ import annotations

import pandas as pd

#: Annotation-pipeline name -> Supplementary Table 2 name. Columns not listed
#: keep their name.
RENAME = {
    "Wild Type Residue": "wild_type_residue",
    "Mutation": "mutation",
    "Position": "position",
    "corrected_score_1": "WT_relative_score_replicate_1",
    "corrected_score_2": "WT_relative_score_replicate_2",
    "corrected_score_3": "WT_relative_score_replicate_3",
    "average score": "mean_WT_relative_score",
    "intercept_0_std_adj_score_1": "standardized_score_replicate_1",
    "intercept_0_std_adj_score_2": "standardized_score_replicate_2",
    "intercept_0_std_adj_score_3": "standardized_score_replicate_3",
    "intercept_0_standard-adjusted score": "mean_standardized_score",
    "classification_2.5pct": "variant_classification",
    "DN_EV": "dominant_negative",
    "Number of Barcodes": "total_observed_barcodes",
    "average_num_quant_bc": "mean_number_quantified_barcodes",
    "hek_protein_conc_nm": "HEK293_protein_concentration_nM",
    "alignment_pos": "kinase_alignment_position",
    "plddt": "AF_plddt",
}

#: Supplementary Table 2, in file order.
ST2_COLUMNS = [
    "protein", "library", "variant", "assay", "assay_treatment",
    "variant_category", "wild_type_residue", "mutation", "position", "hgvs_p",
    "score_1", "score_2", "score_3",
    "WT_relative_score_replicate_1", "WT_relative_score_replicate_2",
    "WT_relative_score_replicate_3", "mean_WT_relative_score",
    "standardized_score_replicate_1", "standardized_score_replicate_2",
    "standardized_score_replicate_3", "mean_standardized_score",
    "variant_classification", "dominant_negative", "buffering_class",
    "dependent", "dependent_threshold", "total_observed_barcodes",
    "mean_number_quantified_barcodes", "uniprot_id", "client_status_literature",
    "max_sasa", "relative_sasa", "dssp_solvent_accessibility_angstroms^2",
    "domain", "HEK293_protein_concentration_nM", "feature", "active_site",
    "kinase_alignment_position", "AF_plddt", "inter_domain_contacts_all_atom",
    "at_hsp90_unfolded", "at_hsp90_folded", "at_cdc37",
    "at_any_chaperone_contact", "chaperone_contact_resolved", "phylop_vert",
    "jsd_conservation", "kinase_motif", "ddG_SPURS", "in_gnomad",
    "min_nt_changes", "nmd_zone", "genie_count",
    "dependent_significant",
]

#: Not in the release, but read by a notebook or the interaction pipeline:
#:   uniprot_accession         gene-symbol lookup in interactions/main.py
#:   dssp_secondary_structure  Fig2, Fig4, Fig6
#:   interface_pdb_curated     Fig2 ED 3c/3e, Fig4, Fig5
#:   interface_partners, n_interface_partners, dn_threshold   Fig4
#:   in_aou                    Fig4 depletion. Derived from All of Us Controlled
#:                             Tier data, whose dissemination policy does not
#:                             allow per-variant observation to be released.
EXTRA_COLUMNS = [
    "uniprot_accession", "dssp_secondary_structure",
    "interface_pdb_curated", "interface_partners", "n_interface_partners",
    "dn_threshold", "in_aou",
]


def build(ann: pd.DataFrame, hsp90_calls: pd.DataFrame) -> pd.DataFrame:
    """The annotated table: Supplementary Table 2 columns, then the extras.

    ``ann`` is the annotation pipeline's table (pipeline names); ``hsp90_calls``
    is one row per (protein, library, variant) from ``hsp90_calls``.
    """
    key = ["protein", "library", "variant"]
    assert not hsp90_calls.duplicated(key).any()
    out = ann.rename(columns=RENAME).merge(hsp90_calls, on=key, how="left",
                                           validate="many_to_one")
    assert len(out) == len(ann)
    missing = [c for c in ST2_COLUMNS + EXTRA_COLUMNS if c not in out.columns]
    assert not missing, f"columns missing from the annotated table: {missing}"
    return out[ST2_COLUMNS + EXTRA_COLUMNS]


def write_supplementary_table_2(table: pd.DataFrame, path) -> None:
    """Write the release file: the ST2 columns, no index column."""
    table[ST2_COLUMNS].to_csv(path, sep="\t", index=False)
