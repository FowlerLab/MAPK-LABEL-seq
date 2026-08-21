"""Variant-level scoring for the LABEL-seq pipeline.

The scoring logic:
1. Ratio calculation (numerator / denominator per replicate)
2. Wild-type normalization (divide by WT mean ratio)
3. Aggregation to variant level (mean across barcodes)
4. Variant frequency computation
5. Standard curve fitting (intercept = 0, slope-only)
6. Barcode count cutoff
7. Percentile-based classification from synonymous WT distribution
8. Position offset correction
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def compute_ratios(
    df: pd.DataFrame, numerator: str, denominator: str
) -> pd.DataFrame:
    """Compute per-replicate ratios (numerator / denominator).

    Args:
        df: Barcode-level DataFrame with count columns.
        numerator: Channel name prefix (e.g., 'pEM1').
        denominator: Channel name prefix (e.g., 'E40').

    Returns:
        DataFrame with added 'ratio_1', 'ratio_2', 'ratio_3' columns.
    """
    df = df.copy()
    for j in range(1, 4):
        df[f"ratio_{j}"] = df[f"{numerator}_{j}"] / df[f"{denominator}_{j}"]
    return df


def normalize_to_wt(
    df: pd.DataFrame, numerator: str, denominator: str
) -> pd.DataFrame:
    """Normalize replicate ratios to the wild-type mean.

    For each replicate, computes the mean ratio across all barcodes
    classified as 'wild type', then divides every barcode's ratio by
    this value. The result is a score where WT ≈ 1.0.

    Args:
        df: Barcode-level DataFrame with 'Mutation Type' and count columns.
        numerator: Channel name prefix.
        denominator: Channel name prefix.

    Returns:
        DataFrame with added 'score_1', 'score_2', 'score_3' columns.
    """
    df = df.copy()
    wt = df[df["Mutation Type"] == "wild type"]

    for j in range(1, 4):
        num_col = f"{numerator}_{j}"
        denom_col = f"{denominator}_{j}"
        wt_mean_ratio = (wt[num_col] / wt[denom_col]).mean()
        df[f"score_{j}"] = (df[num_col] / df[denom_col]) / wt_mean_ratio

    return df


def aggregate_to_variants(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate barcode-level scores to variant-level means.

    Groups by variant and computes the mean of Number of Barcodes, ratios,
    and scores across all barcodes for that variant. This treats each barcode
    equally regardless of read depth (GOTCHA G9).

    Args:
        df: Barcode-level DataFrame with score columns.

    Returns:
        Variant-level DataFrame with one row per variant.
    """
    agg_cols = [
        "Number of Barcodes",
        "ratio_1", "ratio_2", "ratio_3",
        "average ratio",
        "score_1", "score_2", "score_3",
    ]
    # Only aggregate columns that exist
    agg_cols = [c for c in agg_cols if c in df.columns]

    replicate_scores = df.groupby("variant")[agg_cols].mean().reset_index()
    replicate_scores["average score"] = replicate_scores[
        ["score_1", "score_2", "score_3"]
    ].mean(axis=1)

    return replicate_scores


def compute_variant_frequency(
    barcode_df: pd.DataFrame,
    replicate_scores: pd.DataFrame,
    numerator: str,
    denominator: str,
) -> pd.DataFrame:
    """Compute the frequency of each variant based on total reads.

    Frequency = (sum of reads for variant) / (total reads across all variants).
    Reads are summed across both channels and all replicates.

    Args:
        barcode_df: Barcode-level DataFrame with count columns.
        replicate_scores: Variant-level DataFrame to merge frequency into.
        numerator: Channel name prefix.
        denominator: Channel name prefix.

    Returns:
        replicate_scores with added 'variant_frequency' column.
    """
    df = barcode_df.copy()

    for j in range(1, 4):
        df[f"reads_{j}"] = (
            df[f"{denominator}_{j}"].fillna(0) + df[f"{numerator}_{j}"].fillna(0)
        )
    df["total_reads"] = df[[f"reads_{j}" for j in range(1, 4)]].sum(axis=1)

    total_reads_all = df["total_reads"].sum()
    variant_reads = df.groupby("variant")["total_reads"].sum().reset_index()
    variant_reads["variant_frequency"] = variant_reads["total_reads"] / total_reads_all

    replicate_scores = replicate_scores.merge(
        variant_reads[["variant", "variant_frequency"]], on="variant"
    )
    return replicate_scores


def fit_standard_curve(
    replicate_scores: pd.DataFrame,
    standards_dict: Dict[str, float],
    skip: bool = False,
) -> pd.DataFrame:
    """Fit a zero-intercept standard curve and adjust scores.

    Uses known control variants with assigned scores to fit y = m*x
    (intercept forced to 0) via scipy.optimize.curve_fit. Then divides
    each replicate's score by the fitted slope.

    For BRAF activity, `skip=True` is passed because BRAF is the source
    of the standard controls — applying the curve would be circular
    (GOTCHA G1).

    GOTCHA G4: No bounds or sign constraints on the slope. A negative
    slope would invert all scores.

    Args:
        replicate_scores: Variant-level DataFrame with score columns.
        standards_dict: Mapping of variant name → assigned score value.
        skip: If True, copy raw scores directly (no standard curve).

    Returns:
        DataFrame with 'intercept_0_std_adj_score_1/2/3' and
        'intercept_0_standard-adjusted score' columns.
    """
    df = replicate_scores.copy()
    rep_cols = ["score_1", "score_2", "score_3"]

    if skip:
        # GOTCHA G1: BRAF activity — use raw WT-normalized scores
        for rep in rep_cols:
            df[f"intercept_0_std_adj_{rep}"] = df[rep]
            df[f"intercept_0_slope_{rep}"] = np.nan
        df["intercept_0_standard-adjusted score"] = df["average score"]
        return df

    def model(x: np.ndarray, m: float) -> np.ndarray:
        """Linear model with zero intercept: y = m * x."""
        return m * x

    for rep in rep_cols:
        # Get standard control variants
        standard_scores = df[
            (df["Mutation Type"] == "standard") & (df["variant"] != "NoVar_std")
        ].copy()

        standard_scores["Assigned Standard Score"] = standard_scores["variant"].map(
            standards_dict
        )
        standard_scores["Assigned Standard Score"] = pd.to_numeric(
            standard_scores["Assigned Standard Score"], errors="coerce"
        )
        standard_scores = standard_scores.dropna(subset=["Assigned Standard Score", rep])

        if standard_scores["Assigned Standard Score"].notna().sum() >= 2:
            x = standard_scores["Assigned Standard Score"].values
            y = standard_scores[rep].values

            # GOTCHA G4: No bounds on slope — could be negative
            slope, _ = curve_fit(model, x, y)
            slope = slope[0]

            df[f"intercept_0_slope_{rep}"] = slope
            df[f"intercept_0_std_adj_{rep}"] = df[rep] / slope
        else:
            df[f"intercept_0_slope_{rep}"] = np.nan
            df[f"intercept_0_std_adj_{rep}"] = np.nan

    df["intercept_0_standard-adjusted score"] = df[
        [f"intercept_0_std_adj_{rep}" for rep in rep_cols]
    ].mean(axis=1)

    return df


def compute_quantified_barcodes(
    barcode_df: pd.DataFrame,
    replicate_scores: pd.DataFrame,
) -> pd.DataFrame:
    """Count how many barcodes had measurable scores per variant per replicate.

    A barcode is "quantified" in a replicate if its score is not NaN.
    The average across replicates gives the mean number of quantified barcodes.

    Args:
        barcode_df: Barcode-level DataFrame with score columns.
        replicate_scores: Variant-level DataFrame to merge counts into.

    Returns:
        replicate_scores with added 'average_num_quant_bc' column.
    """
    score_cols = [f"score_{j}" for j in range(1, 4)]
    counts = barcode_df.groupby("variant")[score_cols].count().reset_index()
    counts["average_num_quant_bc"] = counts[score_cols].mean(axis=1)

    replicate_scores = replicate_scores.merge(
        counts[["variant", "average_num_quant_bc"]], on="variant"
    )
    return replicate_scores


def apply_barcode_cutoff(
    df: pd.DataFrame, min_barcodes: int = 5
) -> pd.DataFrame:
    """Remove variants with too few quantified barcodes.

    Variants need at least `min_barcodes` quantified barcodes (averaged
    across replicates) to be retained. This ensures scores are based on
    sufficient biological replicates.

    Args:
        df: Variant-level DataFrame with 'average_num_quant_bc' column.
        min_barcodes: Minimum threshold. Defaults to 5.

    Returns:
        Filtered DataFrame.
    """
    return df[df["average_num_quant_bc"] >= min_barcodes].copy()


def classify_variants(
    df: pd.DataFrame,
    percentile_configs: List[Dict[str, Any]],
) -> pd.DataFrame:
    """Classify variants using synonymous WT percentile thresholds.

    For each (library, assay, treatment) group, computes the specified
    percentiles of the synonymous wild-type 'average score' distribution.
    Variants below the lower percentile are 'low', above upper are 'high',
    and in between are 'wt-like'.

    Args:
        df: Variant-level DataFrame with 'average score', 'Mutation Type',
            'library', 'assay', 'assay_treatment' columns.
        percentile_configs: List of dicts, each with keys 'name', 'lower',
            'upper' (e.g., {'name': 'classification_5pct', 'lower': 0.05,
            'upper': 0.95}).

    Returns:
        DataFrame with classification columns added.
    """
    df = df.copy()
    syn_wt = df[df["Mutation Type"] == "synonymous wild type"]
    group_cols = ["library", "assay", "assay_treatment"]

    for pconf in percentile_configs:
        col_name = pconf["name"]
        lower_q = pconf["lower"]
        upper_q = pconf["upper"]

        thresholds = (
            syn_wt.groupby(group_cols)["average score"]
            .agg(
                lower_threshold=lambda x, q=lower_q: x.quantile(q),
                upper_threshold=lambda x, q=upper_q: x.quantile(q),
            )
            .reset_index()
        )

        df = df.merge(thresholds, on=group_cols, how="left")

        def _classify(row: pd.Series) -> Optional[str]:
            score = row["average score"]
            low = row["lower_threshold"]
            high = row["upper_threshold"]
            if pd.isna(score) or pd.isna(low) or pd.isna(high):
                return None
            if score < low:
                return "low"
            elif score > high:
                return "high"
            else:
                return "wt-like"

        df[col_name] = df.apply(_classify, axis=1)
        df = df.drop(columns=["lower_threshold", "upper_threshold"])

    return df


def apply_position_offsets(
    df: pd.DataFrame,
    config: Dict[str, Any],
) -> pd.DataFrame:
    """Add protein-specific position offsets and rebuild variant strings.

    Position offsets correct for libraries that cover only a portion of the
    full-length protein. For example, EGFR activity has offset +670 because
    the library starts at residue 670.

    Only applies to non-special variants (excludes standard, wild type,
    unknown mutation types).

    BUG B5: The original variant string is overwritten — no 'original_variant'
    column is preserved.

    Args:
        df: DataFrame with 'library', 'assay', 'Position', 'Mutation Type',
            'Wild Type Residue', 'Mutation', 'variant' columns.
        config: Full config dict (uses proteins section for offsets).

    Returns:
        DataFrame with corrected Position values and rebuilt variant strings.
    """
    from labelseq_mapk.config import get_position_offset

    df = df.copy()

    # Only modify non-special variants
    select_member = ~df["Mutation Type"].isin(["standard", "wild type", "unknown"])

    # Apply offsets for each (library, assay) combination
    for library in df["library"].unique():
        for assay in df["assay"].unique():
            offset = get_position_offset(config, library, assay)
            if offset == 0:
                continue
            mask = (
                (df["library"] == library)
                & (df["assay"] == assay)
                & select_member
            )
            df.loc[mask, "Position"] = df.loc[mask, "Position"] + offset

    # Convert Position to int where possible to avoid '600.0' in variant strings
    # This fixes BUG B6 (float positions from offset addition)
    numeric_pos = pd.to_numeric(df["Position"], errors="coerce")
    int_mask = numeric_pos.notna()
    if int_mask.any():
        df.loc[int_mask, "Position"] = numeric_pos[int_mask].astype(int).astype(str)

    # BUG B5: Rebuild variant string, overwriting original
    mask_any = select_member
    df.loc[mask_any, "variant"] = (
        df.loc[mask_any, "Wild Type Residue"].astype(str)
        + df.loc[mask_any, "Position"].astype(str)
        + df.loc[mask_any, "Mutation"].astype(str)
    )

    return df
