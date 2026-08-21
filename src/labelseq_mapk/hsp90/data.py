"""HSP90 buffering data loading and feature computation.

Loads the annotated pipeline output and constructs a paired dataset
of control vs HSP90i scores for each variant, augmented with
interpretable biophysical features.

Data source: output/Annotated_Replicate_scores_masterframe.tsv
Domain annotations: pipeline `domain` column (curated boundaries)
"""

from pathlib import Path
from typing import Optional
import json

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ANNOTATED_FILE = PROJECT_ROOT / "output" / "Annotated_Replicate_scores_masterframe.tsv"
SPURS_DDG_FILE = PROJECT_ROOT / "data" / "inputs" / "spurs_ddg_all_proteins.tsv"
MOTIF_FILE = PROJECT_ROOT.parent / "claude_HSP90" / "kinase_motif_definitions.json"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Proteins with paired HSP90i data and their control treatments
HSP90I_PROTEINS = {
    "araf": "DMSO",
    "braf": "No_treatment",
    "craf": "No_treatment",
    "egfr": "DMSO",
    "ksr2": "DMSO",
    "mek1": "DMSO",
    "mek2": "DMSO",
    "met": "DMSO",
    "ret": "No_treatment",
    "sos2": "DMSO",
}

# Which of these are kinases (all except sos2, which has a GEF domain)
KINASE_PROTEINS = ["araf", "braf", "craf", "egfr", "ksr2", "mek1", "mek2", "met", "ret"]

# Primary catalytic/functional domain name per protein (from pipeline domain col)
CATALYTIC_DOMAINS = {
    "araf": "kinase", "braf": "kinase", "craf": "kinase",
    "egfr": "kinase", "erbb2": "kinase",
    "grb2": "sh2",
    "kras": "g_domain", "mras": "g_domain",
    "ksr1": "kinase", "ksr2": "kinase",
    "mek1": "kinase", "mek2": "kinase",
    "met": "kinase", "ret": "kinase",
    "shp2": "phosphatase",
    "sos1": "gef", "sos2": "gef",
}

# Known chaperone/co-chaperone UniProt IDs in the interface_evidence column
CHAPERONE_PARTNERS = {
    "p07900": "HSP90AA1",   # HSP90 alpha
    "p08238": "HSP90AB1",   # HSP90 beta
    "q16543": "CDC37",       # CDC37 co-chaperone
}

# Grantham distance matrix (Grantham 1974, Science 185:862)
# Symmetric matrix — indexed as GRANTHAM[(aa1, aa2)]
_GRANTHAM_RAW = {
    ('A','R'):112,('A','N'):111,('A','D'):126,('A','C'):195,('A','E'):107,
    ('A','Q'):91,('A','G'):60,('A','H'):86,('A','I'):94,('A','L'):96,
    ('A','K'):106,('A','M'):84,('A','F'):113,('A','P'):27,('A','S'):99,
    ('A','T'):58,('A','W'):148,('A','Y'):112,('A','V'):64,
    ('R','N'):86,('R','D'):96,('R','C'):180,('R','E'):54,('R','Q'):43,
    ('R','G'):125,('R','H'):29,('R','I'):97,('R','L'):102,('R','K'):26,
    ('R','M'):91,('R','F'):97,('R','P'):103,('R','S'):110,('R','T'):71,
    ('R','W'):101,('R','Y'):77,('R','V'):96,
    ('N','D'):23,('N','C'):139,('N','E'):42,('N','Q'):46,('N','G'):80,
    ('N','H'):68,('N','I'):149,('N','L'):153,('N','K'):94,('N','M'):142,
    ('N','F'):158,('N','P'):91,('N','S'):46,('N','T'):65,('N','W'):174,
    ('N','Y'):143,('N','V'):133,
    ('D','C'):154,('D','E'):45,('D','Q'):61,('D','G'):94,('D','H'):81,
    ('D','I'):168,('D','L'):172,('D','K'):101,('D','M'):160,('D','F'):177,
    ('D','P'):108,('D','S'):65,('D','T'):85,('D','W'):181,('D','Y'):160,
    ('D','V'):152,
    ('C','E'):170,('C','Q'):154,('C','G'):159,('C','H'):174,('C','I'):198,
    ('C','L'):198,('C','K'):202,('C','M'):196,('C','F'):205,('C','P'):169,
    ('C','S'):112,('C','T'):149,('C','W'):215,('C','Y'):194,('C','V'):192,
    ('E','Q'):29,('E','G'):98,('E','H'):40,('E','I'):134,('E','L'):138,
    ('E','K'):56,('E','M'):126,('E','F'):140,('E','P'):93,('E','S'):80,
    ('E','T'):65,('E','W'):152,('E','Y'):122,('E','V'):121,
    ('Q','G'):87,('Q','H'):24,('Q','I'):109,('Q','L'):113,('Q','K'):53,
    ('Q','M'):101,('Q','F'):116,('Q','P'):76,('Q','S'):68,('Q','T'):42,
    ('Q','W'):130,('Q','Y'):99,('Q','V'):96,
    ('G','H'):98,('G','I'):135,('G','L'):138,('G','K'):127,('G','M'):127,
    ('G','F'):153,('G','P'):42,('G','S'):56,('G','T'):59,('G','W'):184,
    ('G','Y'):147,('G','V'):109,
    ('H','I'):94,('H','L'):99,('H','K'):32,('H','M'):87,('H','F'):100,
    ('H','P'):77,('H','S'):89,('H','T'):47,('H','W'):115,('H','Y'):83,
    ('H','V'):84,
    ('I','L'):5,('I','K'):102,('I','M'):10,('I','F'):21,('I','P'):95,
    ('I','S'):142,('I','T'):89,('I','W'):61,('I','Y'):33,('I','V'):29,
    ('L','K'):107,('L','M'):15,('L','F'):22,('L','P'):98,('L','S'):145,
    ('L','T'):92,('L','W'):61,('L','Y'):36,('L','V'):32,
    ('K','M'):95,('K','F'):102,('K','P'):103,('K','S'):121,('K','T'):78,
    ('K','W'):110,('K','Y'):85,('K','V'):97,
    ('M','F'):28,('M','P'):87,('M','S'):135,('M','T'):81,('M','W'):67,
    ('M','Y'):36,('M','V'):21,
    ('F','P'):114,('F','S'):155,('F','T'):103,('F','W'):40,('F','Y'):22,
    ('F','V'):50,
    ('P','S'):74,('P','T'):38,('P','W'):147,('P','Y'):110,('P','V'):68,
    ('S','T'):58,('S','W'):177,('S','Y'):144,('S','V'):124,
    ('T','W'):128,('T','Y'):92,('T','V'):69,
    ('W','Y'):37,('W','V'):88,
    ('Y','V'):55,
}

GRANTHAM = {}
for (a, b), v in _GRANTHAM_RAW.items():
    GRANTHAM[(a, b)] = v
    GRANTHAM[(b, a)] = v
# Self-substitutions have distance 0
for aa in "ACDEFGHIKLMNPQRSTVWY":
    GRANTHAM[(aa, aa)] = 0

# Kyte-Doolittle hydrophobicity scale
HYDROPHOBICITY = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
    'E': -3.5, 'Q': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2,
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_annotated_data(
    annotated_file: Optional[Path] = None,
) -> pd.DataFrame:
    """Load the full annotated pipeline output.

    Args:
        annotated_file: Path to annotated TSV. Defaults to pipeline output.

    Returns:
        Full DataFrame (all variants, all assays, all treatments).
    """
    path = annotated_file or ANNOTATED_FILE
    df = pd.read_csv(path, sep="\t", low_memory=False)
    df["Position"] = pd.to_numeric(df["Position"], errors="coerce")
    return df


def get_paired_hsp90i_data(
    df: pd.DataFrame,
    score_col: str = "average score",
) -> pd.DataFrame:
    """Extract paired control/HSP90i scores for missense variants.

    For each HSP90i protein, matches variants across control and HSP90i
    treatments within the abundance assay. Returns one row per variant with
    both ctrl_score and hsp90i_score.

    Args:
        df: Full annotated DataFrame.
        score_col: Which score column to use.

    Returns:
        DataFrame with columns: protein, variant, Position, wt_aa, mut_aa,
        ctrl_score, hsp90i_score, delta_score, domain, relative_sasa,
        kinase_conservation, interface_evidence, ...
    """
    # Filter to abundance assay, missense variants
    abund = df[
        (df["assay"] == "abundance")
        & (df["Mutation Type"] == "missense")
    ].copy()
    abund = abund.dropna(subset=[score_col, "Position"])
    abund["Position"] = abund["Position"].astype(int)

    all_paired = []

    for protein, ctrl_treatment in HSP90I_PROTEINS.items():
        # Get control and HSP90i data for this protein
        ctrl = abund[
            (abund["protein"] == protein)
            & (abund["assay_treatment"] == ctrl_treatment)
        ][["variant", "Position", score_col]].drop_duplicates("variant")

        hsp = abund[
            (abund["protein"] == protein)
            & (abund["assay_treatment"] == "HSP90i")
        ][["variant", "Position", score_col]].drop_duplicates("variant")

        if ctrl.empty or hsp.empty:
            continue

        # Also grab annotation columns from the control rows
        # (domain, RSA, conservation, etc. are position-level, same across treatments)
        annot_cols = [
            "variant", "Position", "domain", "relative_sasa",
            "dssp_secondary_structure", "kinase_conservation",
            "interface_evidence", "active_site", "protein_interface",
            "am_pathogenicity", "hek_protein_conc_nm", "client_status",
        ]
        annot_cols = [c for c in annot_cols if c in abund.columns]
        annot = abund[
            (abund["protein"] == protein)
            & (abund["assay_treatment"] == ctrl_treatment)
        ][annot_cols].drop_duplicates("variant")

        # Merge ctrl and HSP90i scores
        merged = ctrl.merge(
            hsp[["variant", score_col]],
            on="variant",
            suffixes=("_ctrl", "_hsp90i"),
        )
        merged.rename(columns={
            f"{score_col}_ctrl": "ctrl_score",
            f"{score_col}_hsp90i": "hsp90i_score",
        }, inplace=True)

        # Merge annotation columns
        merged = merged.merge(annot.drop(columns=["Position"], errors="ignore"),
                              on="variant", how="left")

        merged["protein"] = protein
        merged["delta_score"] = merged["hsp90i_score"] - merged["ctrl_score"]
        all_paired.append(merged)

    paired = pd.concat(all_paired, ignore_index=True)

    # Extract wt/mut amino acids from variant name
    paired["wt_aa"] = paired["variant"].str[0]
    paired["mut_aa"] = paired["variant"].str[-1]

    # Derived features
    paired["is_catalytic_domain"] = paired.apply(
        lambda r: r["domain"] == CATALYTIC_DOMAINS.get(r["protein"], ""), axis=1
    ).astype(int)

    return paired


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

def add_grantham_distance(df: pd.DataFrame) -> pd.DataFrame:
    """Add Grantham distance for each wt→mut substitution.

    Args:
        df: DataFrame with wt_aa and mut_aa columns.

    Returns:
        DataFrame with grantham_distance column added.
    """
    df = df.copy()
    df["grantham_distance"] = df.apply(
        lambda r: GRANTHAM.get((r["wt_aa"], r["mut_aa"]), np.nan), axis=1
    )
    return df


def add_delta_hydrophobicity(df: pd.DataFrame) -> pd.DataFrame:
    """Add change in hydrophobicity (Kyte-Doolittle) for each substitution.

    Positive = mutation is more hydrophobic. Negative = mutation is less
    hydrophobic (hydrophobic loss at a buried site is destabilizing).

    Args:
        df: DataFrame with wt_aa and mut_aa columns.

    Returns:
        DataFrame with delta_hydrophobicity column added.
    """
    df = df.copy()
    df["delta_hydrophobicity"] = df.apply(
        lambda r: HYDROPHOBICITY.get(r["mut_aa"], 0) - HYDROPHOBICITY.get(r["wt_aa"], 0),
        axis=1,
    )
    return df


def add_spurs_ddg(
    df: pd.DataFrame,
    ddg_file: Optional[Path] = None,
) -> pd.DataFrame:
    """Merge SPURS-predicted ddG values.

    Positive ddG = destabilizing mutation.

    Args:
        df: Paired DataFrame with protein, variant columns.
        ddg_file: Path to SPURS TSV. Defaults to standard location.

    Returns:
        DataFrame with ddG_SPURS column added.
    """
    path = ddg_file or SPURS_DDG_FILE
    if not path.exists():
        print(f"WARNING: SPURS ddG file not found at {path}, skipping")
        df["ddG_SPURS"] = np.nan
        return df

    spurs = pd.read_csv(path, sep="\t")
    spurs["variant"] = spurs["wt_aa"] + spurs["position"].astype(str) + spurs["mut_aa"]
    spurs = spurs[["protein", "variant", "ddG_SPURS"]].drop_duplicates(
        subset=["protein", "variant"]
    )

    n_before = len(df)
    df = df.merge(spurs, on=["protein", "variant"], how="left")
    n_matched = df["ddG_SPURS"].notna().sum()
    print(f"  SPURS ddG: {n_matched}/{n_before} variants matched ({n_matched/n_before:.1%})")

    return df


def add_motif_annotations(
    df: pd.DataFrame,
    motif_file: Optional[Path] = None,
) -> pd.DataFrame:
    """Add binary kinase motif membership flags.

    Motif definitions from kinase_motif_definitions.json. Each motif
    is a range [start, end] of residue positions. A variant is in a motif
    if its Position falls within [start, end] inclusive.

    Args:
        df: Paired DataFrame with protein and Position columns.
        motif_file: Path to motif JSON. Defaults to standard location.

    Returns:
        DataFrame with motif_* binary columns added.
    """
    path = motif_file or MOTIF_FILE
    if not path.exists():
        print(f"WARNING: Motif definitions not found at {path}, skipping")
        return df

    with open(path) as f:
        motif_defs = json.load(f)

    # Motifs to annotate (excluding kinase_domain which we get from pipeline domain col,
    # and excluding atp_binding/hinge_residue which are point annotations)
    range_motifs = [
        "gly_rich_loop", "alphaC_helix", "beta4_beta5", "hinge",
        "DFG", "activation_loop", "catalytic_loop",
    ]

    df = df.copy()
    for motif in range_motifs:
        col = f"motif_{motif}"
        df[col] = 0
        for protein in df["protein"].unique():
            prot_motifs = motif_defs.get(protein, {})
            if motif in prot_motifs:
                rng = prot_motifs[motif]
                if isinstance(rng, list) and len(rng) == 2:
                    start, end = rng
                    mask = (
                        (df["protein"] == protein)
                        & (df["Position"] >= start)
                        & (df["Position"] <= end)
                    )
                    df.loc[mask, col] = 1

    # Also flag "any motif" for convenience
    motif_cols = [f"motif_{m}" for m in range_motifs]
    df["in_any_motif"] = (df[motif_cols].sum(axis=1) > 0).astype(int)

    return df


def add_chaperone_interface_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Flag variants at HSP90 and CDC37 interaction interfaces.

    Uses the interface_evidence column from the pipeline, which contains
    semicolon-separated UniProt IDs of interaction partners derived from
    PDB structures.

    Args:
        df: Paired DataFrame with interface_evidence column.

    Returns:
        DataFrame with at_HSP90_interface, at_CDC37_interface columns.
    """
    df = df.copy()

    for partner_id, partner_name in CHAPERONE_PARTNERS.items():
        col = f"at_{partner_name}_interface"
        df[col] = df["interface_evidence"].str.contains(
            partner_id, case=False, na=False
        ).astype(int)

    # Combined: at any chaperone interface
    chap_cols = [f"at_{name}_interface" for name in CHAPERONE_PARTNERS.values()]
    df["at_any_chaperone_interface"] = (df[chap_cols].sum(axis=1) > 0).astype(int)

    return df


def add_partner_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Parse interface_evidence into individual partner flags.

    Creates a binary column for each interaction partner found in the data
    (among HSP90i proteins). Returns a partner summary DataFrame alongside
    the modified main DataFrame.

    Args:
        df: Paired DataFrame with interface_evidence column.

    Returns:
        DataFrame with at_partner_{id} columns added, plus a
        'partner_list' column (list of partner IDs per variant).
    """
    df = df.copy()

    # Parse all partners
    def _parse_partners(evidence: str) -> list:
        if pd.isna(evidence) or evidence == "":
            return []
        return [p.strip() for p in str(evidence).split(";") if p.strip()]

    df["partner_list"] = df["interface_evidence"].apply(_parse_partners)
    df["n_partners"] = df["partner_list"].apply(len)
    df["at_any_interface"] = (df["n_partners"] > 0).astype(int)

    return df


# ---------------------------------------------------------------------------
# Buffering classification
# ---------------------------------------------------------------------------

def classify_buffering(
    df: pd.DataFrame,
    ctrl_threshold: float = 0.8,
) -> pd.DataFrame:
    """Classify variants by HSP90 buffering status.

    Uses a threshold-based definition:
    - HSP90-buffered: ctrl_score >= threshold AND hsp90i_score < threshold
    - High→High: ctrl_score >= threshold AND hsp90i_score >= threshold
    - Low→Low: ctrl_score < threshold AND hsp90i_score < threshold
    - Low→High: ctrl_score < threshold AND hsp90i_score >= threshold

    Args:
        df: Paired DataFrame with ctrl_score and hsp90i_score.
        ctrl_threshold: Score threshold for "stable" classification.

    Returns:
        DataFrame with buffering_category and buffered_binary columns.
    """
    df = df.copy()

    conditions = [
        (df["ctrl_score"] >= ctrl_threshold) & (df["hsp90i_score"] < ctrl_threshold),
        (df["ctrl_score"] >= ctrl_threshold) & (df["hsp90i_score"] >= ctrl_threshold),
        (df["ctrl_score"] < ctrl_threshold) & (df["hsp90i_score"] < ctrl_threshold),
        (df["ctrl_score"] < ctrl_threshold) & (df["hsp90i_score"] >= ctrl_threshold),
    ]
    labels = ["HSP90-buffered", "High→High", "Low→Low", "Low→High"]
    df["buffering_category"] = np.select(conditions, labels, default="unknown")

    # Binary: among ctrl >= threshold variants, is it buffered?
    df["buffered_binary"] = (
        (df["ctrl_score"] >= ctrl_threshold)
        & (df["hsp90i_score"] < ctrl_threshold)
    ).astype(int)

    return df


# ---------------------------------------------------------------------------
# Main build function
# ---------------------------------------------------------------------------

def build_hsp90_dataset(
    annotated_file: Optional[Path] = None,
    score_col: str = "average score",
) -> pd.DataFrame:
    """Build the complete HSP90 analysis dataset.

    Loads annotated data, pairs control/HSP90i scores, and adds all
    interpretable features: Grantham distance, delta hydrophobicity,
    SPURS ddG, motif annotations, chaperone interface flags, partner
    flags, and buffering classification.

    Args:
        annotated_file: Path to annotated TSV.
        score_col: Which score column to use.

    Returns:
        Paired DataFrame with all features, one row per variant.
    """
    print("Loading annotated data...")
    full_df = load_annotated_data(annotated_file)
    print(f"  {len(full_df)} rows, {len(full_df.columns)} columns")

    print("Pairing control/HSP90i scores...")
    paired = get_paired_hsp90i_data(full_df, score_col)
    print(f"  {len(paired)} paired missense variants across {paired['protein'].nunique()} proteins")

    print("Adding Grantham distance...")
    paired = add_grantham_distance(paired)

    print("Adding delta hydrophobicity...")
    paired = add_delta_hydrophobicity(paired)

    print("Adding SPURS ddG...")
    paired = add_spurs_ddg(paired)

    print("Adding motif annotations...")
    paired = add_motif_annotations(paired)

    print("Adding chaperone interface flags...")
    paired = add_chaperone_interface_flags(paired)

    print("Adding partner flags...")
    paired = add_partner_flags(paired)

    print("Classifying buffering status...")
    paired = classify_buffering(paired)

    # Summary
    n_total = len(paired)
    n_ctrl_high = (paired["ctrl_score"] >= 0.8).sum()
    n_buffered = paired["buffered_binary"].sum()
    print(f"\nDataset summary:")
    print(f"  Total paired variants: {n_total}")
    print(f"  ctrl >= 0.8: {n_ctrl_high} ({n_ctrl_high/n_total:.1%})")
    print(f"  HSP90-buffered (threshold): {n_buffered} ({n_buffered/n_ctrl_high:.1%} of ctrl-high)")

    per_protein = paired.groupby("protein").agg(
        n=("variant", "count"),
        n_buffered=("buffered_binary", "sum"),
        mean_delta=("delta_score", "mean"),
    ).reset_index()
    # Count ctrl >= 0.8 per protein separately (lambda aggs can break on older pandas)
    ctrl_high_counts = paired[paired["ctrl_score"] >= 0.8].groupby("protein").size()
    per_protein["n_ctrl_high"] = per_protein["protein"].map(ctrl_high_counts).fillna(0).astype(int)
    per_protein["pct_buffered"] = per_protein["n_buffered"] / per_protein["n_ctrl_high"] * 100
    print("\n  Per-protein breakdown:")
    for _, row in per_protein.sort_values("pct_buffered", ascending=False).iterrows():
        print(f"    {row['protein']:5s}: {row['n']:5d} paired, "
              f"{row['n_buffered']:4d} buffered ({row['pct_buffered']:.1f}%), "
              f"mean Δ = {row['mean_delta']:.3f}")

    return paired
