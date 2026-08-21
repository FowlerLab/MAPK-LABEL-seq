"""Population database depletion analysis for variant classification.

Provides utilities for merging LABEL-seq scores with gnomAD and All of Us
allele frequencies, computing minimum nucleotide changes for SNV accessibility,
and running Fisher's exact depletion tests across DN threshold definitions.

This module supports the DN threshold sensitivity analysis, which tests whether
dominant-negative variants are depleted in population databases across multiple
threshold definitions (pooled 5-50%, per-protein 5-50%).
"""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# DN column definitions
# ---------------------------------------------------------------------------

PERCENTILES: List[int] = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

DN_POOLED_COLS: List[str] = [f"DN_{p}pct" for p in PERCENTILES]
DN_PP_COLS: List[str] = [f"DNpp_{p}pct" for p in PERCENTILES]

# Empty-vector-derived per-(library, assay_treatment) DN classification.
# Added 2026-04-19; replaces the NoVar_std cold-shock-confounded thresholds for
# the manuscript. See add_dominant_negative_ev() in annotation.py and
# docs/open_questions.md Q8.
DN_EV_COL: str = "DN_EV"

DN_ALL_COLS: List[str] = [DN_EV_COL] + DN_POOLED_COLS + DN_PP_COLS

# Proteins excluded from DN classification (overexpression inhibits pathway).
# Updated 2026-04-19: list trimmed to the five true MEK-pathway negative
# regulators / scaffolds. ARAF and RET are pathway activators and were
# mistakenly included earlier.
INHIBITORY_PROTEINS: List[str] = [
    "grb2", "mek1", "mek2", "ksr1", "ksr2",
]

# Proteins eligible for DN classification. 12 of the 17 profiled proteins —
# everything minus the five inhibitors above. Now includes ARAF and RET.
DN_ELIGIBLE_PROTEINS: List[str] = [
    "araf", "braf", "craf", "egfr", "erbb2", "kras", "met", "mras",
    "ret", "shp2", "sos1", "sos2",
]

# ---------------------------------------------------------------------------
# Genetic code for SNV accessibility
# ---------------------------------------------------------------------------

CODON_TABLE: Dict[str, str] = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}

# Reverse lookup: amino acid → list of codons
AA_TO_CODONS: Dict[str, List[str]] = {}
for _codon, _aa in CODON_TABLE.items():
    AA_TO_CODONS.setdefault(_aa, []).append(_codon)


def _hamming(c1: str, c2: str) -> int:
    """Number of positions where two codons differ."""
    return sum(a != b for a, b in zip(c1, c2))


def _build_nt_change_lookup() -> Dict[Tuple[str, str], int]:
    """Build lookup: (wt_aa, mut_aa) -> minimum nucleotide changes.

    For each pair of amino acids, finds the minimum Hamming distance
    across all pairs of codons encoding them.
    """
    amino_acids = sorted(set(CODON_TABLE.values()) - {"*"})
    lookup: Dict[Tuple[str, str], int] = {}
    for wt_aa in amino_acids:
        for mut_aa in amino_acids:
            if wt_aa == mut_aa:
                lookup[(wt_aa, mut_aa)] = 0
                continue
            min_dist = 3
            for c1 in AA_TO_CODONS[wt_aa]:
                for c2 in AA_TO_CODONS[mut_aa]:
                    d = _hamming(c1, c2)
                    if d < min_dist:
                        min_dist = d
            lookup[(wt_aa, mut_aa)] = min_dist
    return lookup


NT_CHANGE_LOOKUP: Dict[Tuple[str, str], int] = _build_nt_change_lookup()

# ---------------------------------------------------------------------------
# Colors for plotting
# ---------------------------------------------------------------------------

CLASS_COLORS: Dict[str, str] = {
    "dominant_negative": "#d62728",  # red
    "low": "#ff7f0e",               # orange
    "wt-like": "#2ca02c",           # green
    "high": "#1f77b4",              # blue
    "all_non_dn": "#7f7f7f",        # gray
    "low_fixed": "#e377c2",         # pink — fixed Low reference group
}

POOLED_COLOR: str = "#1f77b4"   # blue for pooled thresholds
PP_COLOR: str = "#ff7f0e"       # orange for per-protein thresholds

# ---------------------------------------------------------------------------
# Data loading and merging
# ---------------------------------------------------------------------------


def compute_min_nt_changes(df: pd.DataFrame) -> pd.Series:
    """Compute minimum nucleotide changes for each variant.

    Args:
        df: DataFrame with 'Wild Type Residue' and 'Mutation' columns.

    Returns:
        Series of int (1, 2, or 3) or NaN for non-missense.
    """
    return df.apply(
        lambda r: NT_CHANGE_LOOKUP.get(
            (r["Wild Type Residue"], r["Mutation"]), np.nan
        ),
        axis=1,
    )


def load_position_mapping(path: Path) -> pd.DataFrame:
    """Load validated position mapping for gnomAD/AoU mappability.

    The position mapping translates between UniProt canonical positions
    (used by LABEL-seq) and Ensembl MANE select positions (used by gnomAD/AoU).

    Args:
        path: Path to position_mapping.tsv.

    Returns:
        DataFrame with validated positions (protein, our_position).
    """
    pm = pd.read_csv(path, sep="\t")
    pm = pm[pm["validated"] == True].copy()  # noqa: E712
    return pm


def build_gnomad_lookup(path: Path) -> pd.DataFrame:
    """Build gnomAD lookup table: (protein, Position, Mutation) -> AF/AC/AN.

    Aggregates by taking max AF and summing AC when duplicates exist
    (e.g., same variant on different transcripts).

    Args:
        path: Path to gnomad_variants.tsv.

    Returns:
        DataFrame with one row per unique (protein, Position, Mutation).
    """
    gv = pd.read_csv(path, sep="\t")
    gv = gv.rename(columns={"our_position": "Position"})
    gv["Position"] = gv["Position"].astype(str)

    lookup = (
        gv.groupby(["protein", "Position", "mutation"])
        .agg(gnomad_af=("gnomad_af", "max"),
             gnomad_ac=("gnomad_ac", "sum"),
             gnomad_an=("gnomad_an", "max"))
        .reset_index()
        .rename(columns={"mutation": "Mutation"})
    )
    return lookup


def build_aou_lookup(path: Path) -> pd.DataFrame:
    """Build AoU lookup table: (protein, Position, Mutation) -> AF/AC/AN.

    Args:
        path: Path to aou_variants.tsv.

    Returns:
        DataFrame with one row per unique (protein, Position, Mutation).
    """
    av = pd.read_csv(path, sep="\t")
    av = av.rename(columns={"our_position": "Position"})
    av["Position"] = av["Position"].astype(str)

    lookup = (
        av.groupby(["protein", "Position", "mutation"])
        .agg(aou_af=("aou_af", "max"),
             aou_ac=("aou_ac", "sum"),
             aou_an=("aou_an", "max"))
        .reset_index()
        .rename(columns={"mutation": "Mutation"})
    )
    return lookup


def merge_population_data(
    scores: pd.DataFrame,
    gnomad_path: Path,
    aou_path: Path,
    position_mapping_path: Path,
) -> pd.DataFrame:
    """Merge scored variants with gnomAD and AoU population allele frequencies.

    Adds columns: gnomad_af, gnomad_ac, gnomad_an, aou_af, aou_ac, aou_an,
    gnomad_mappable, min_nt_changes.

    Args:
        scores: Annotated scores DataFrame (from pipeline output).
        gnomad_path: Path to gnomad_variants.tsv.
        aou_path: Path to aou_variants.tsv.
        position_mapping_path: Path to position_mapping.tsv.

    Returns:
        DataFrame with population data merged and zero-filled for mappable
        missense variants not found in the databases.
    """
    df = scores.copy()
    df["Position"] = df["Position"].astype(str)

    # Mark mappable positions
    pm = load_position_mapping(position_mapping_path)
    pm_set = set(zip(pm["protein"].astype(str), pm["our_position"].astype(str)))
    df["gnomad_mappable"] = df.apply(
        lambda r: (str(r["protein"]), str(r["Position"])) in pm_set
        if pd.notna(r["protein"]) and pd.notna(r["Position"])
        else False,
        axis=1,
    )

    # Merge gnomAD
    gnomad_lookup = build_gnomad_lookup(gnomad_path)
    df = df.merge(gnomad_lookup, on=["protein", "Position", "Mutation"], how="left")

    # Merge AoU
    aou_lookup = build_aou_lookup(aou_path)
    df = df.merge(aou_lookup, on=["protein", "Position", "Mutation"], how="left")

    # Zero-fill AF for mappable missense variants not in databases
    mappable_missense = (
        (df["gnomad_mappable"] == True)  # noqa: E712
        & (df["Mutation Type"] == "missense")
    )
    for col in ["gnomad_af", "gnomad_ac", "gnomad_an", "aou_af", "aou_ac", "aou_an"]:
        df.loc[mappable_missense & df[col].isna(), col] = 0.0

    # Compute min nucleotide changes
    df["min_nt_changes"] = compute_min_nt_changes(df)

    return df


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


def filter_activity_1nt(
    df: pd.DataFrame,
    require_mappable: bool = True,
) -> pd.DataFrame:
    """Filter to 1-nt SNV-accessible missense variants in activity assay.

    Standard filter for all depletion analyses:
    - Mutation Type == 'missense'
    - assay == 'activity'
    - assay_treatment in ['No_treatment', 'DMSO']
    - min_nt_changes == 1
    - gnomad_mappable == True (if require_mappable)

    Args:
        df: Merged scores DataFrame with popdb columns.
        require_mappable: Whether to require gnomad_mappable == True.

    Returns:
        Filtered DataFrame (copy).
    """
    mask = (
        (df["Mutation Type"] == "missense")
        & (df["assay"] == "activity")
        & (df["assay_treatment"].isin(["No_treatment", "DMSO"]))
        & (df["min_nt_changes"] == 1)
    )
    if require_mappable:
        mask = mask & (df["gnomad_mappable"] == True)  # noqa: E712
    return df[mask].copy()


# ---------------------------------------------------------------------------
# DN mask helpers — critical for three-valued logic
# ---------------------------------------------------------------------------


def get_dn_mask(df: pd.DataFrame, col: str) -> pd.Series:
    """Get boolean mask for DN variants.

    IMPORTANT: DN columns contain True/False/None. We must use explicit
    equality to avoid treating None as False in truthy evaluation.

    Args:
        df: DataFrame with DN column.
        col: DN column name (e.g., 'DN_5pct').

    Returns:
        Boolean Series where True = variant IS dominant negative.
    """
    return df[col] == True  # noqa: E712


def get_non_dn_mask(df: pd.DataFrame, col: str) -> pd.Series:
    """Get boolean mask for classifiable non-DN variants.

    Args:
        df: DataFrame with DN column.
        col: DN column name.

    Returns:
        Boolean Series where True = variant is classifiable but NOT DN.
    """
    return df[col] == False  # noqa: E712


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------


def fisher_test(
    group1_obs: int,
    group1_total: int,
    group2_obs: int,
    group2_total: int,
) -> Dict[str, float]:
    """Fisher's exact test comparing observation rates between two groups.

    Args:
        group1_obs: Number observed in group 1 (e.g., DN variants in gnomAD).
        group1_total: Total in group 1.
        group2_obs: Number observed in group 2 (e.g., non-DN in gnomAD).
        group2_total: Total in group 2.

    Returns:
        Dict with keys: or, p, group1_frac, group2_frac, group1_obs,
        group1_total, group2_obs, group2_total.
    """
    a = group1_obs
    b = group1_total - group1_obs
    c = group2_obs
    d = group2_total - group2_obs

    table = np.array([[a, b], [c, d]])
    oddsratio, pvalue = stats.fisher_exact(table, alternative="two-sided")

    return {
        "or": oddsratio,
        "p": pvalue,
        "group1_frac": group1_obs / group1_total if group1_total > 0 else np.nan,
        "group2_frac": group2_obs / group2_total if group2_total > 0 else np.nan,
        "group1_obs": group1_obs,
        "group1_total": group1_total,
        "group2_obs": group2_obs,
        "group2_total": group2_total,
    }


def log_or_with_ci(
    a: int, b: int, c: int, d: int, z: float = 1.96
) -> Dict[str, float]:
    """Compute log odds ratio with 95% CI and continuity correction.

    Uses Haldane correction (add 0.5 to all cells) when any cell is zero.

    Args:
        a, b, c, d: 2x2 table cells [[a, b], [c, d]].
        z: Z-score for CI (1.96 = 95%).

    Returns:
        Dict with keys: log_or, or, ci_low, ci_high, se.
    """
    if a == 0 or b == 0 or c == 0 or d == 0:
        a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5

    log_or = np.log((a * d) / (b * c))
    se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)

    return {
        "log_or": log_or,
        "or": np.exp(log_or),
        "ci_low": np.exp(log_or - z * se),
        "ci_high": np.exp(log_or + z * se),
        "se": se,
    }


def depletion_test(
    df: pd.DataFrame,
    dn_mask: pd.Series,
    ref_mask: pd.Series,
    af_col: str,
) -> Dict[str, float]:
    """Run Fisher's exact test comparing population observation rates.

    Tests whether DN variants are observed in a population database at a
    different rate than the reference group.

    Args:
        df: Filtered DataFrame.
        dn_mask: Boolean mask for DN variants.
        ref_mask: Boolean mask for reference group.
        af_col: Column name for allele frequency ('gnomad_af' or 'aou_af').

    Returns:
        Dict with Fisher test results + log-OR CI.
    """
    dn = df[dn_mask]
    ref = df[ref_mask]

    dn_obs = int((dn[af_col] > 0).sum())
    dn_total = len(dn)
    ref_obs = int((ref[af_col] > 0).sum())
    ref_total = len(ref)

    if dn_total == 0 or ref_total == 0:
        return {
            "or": np.nan, "p": np.nan,
            "group1_frac": np.nan, "group2_frac": np.nan,
            "group1_obs": dn_obs, "group1_total": dn_total,
            "group2_obs": ref_obs, "group2_total": ref_total,
            "ci_low": np.nan, "ci_high": np.nan,
        }

    result = fisher_test(dn_obs, dn_total, ref_obs, ref_total)

    ci = log_or_with_ci(
        dn_obs, dn_total - dn_obs,
        ref_obs, ref_total - ref_obs,
    )
    result["ci_low"] = ci["ci_low"]
    result["ci_high"] = ci["ci_high"]

    return result
