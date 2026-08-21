"""Barcode-level filtering operations for the LABEL-seq scoring pipeline.

The filtering cascade, in order:
1. Barcode deduplication (remove barcodes mapping to >1 variant)
2. Count cutoff (NaN counts below minimum threshold)
3. Frequency filter (NaN below frequency threshold per replicate)
4. Outlier removal (upper-only, ≥2 replicates)

All functions operate on barcode-level DataFrames and return modified copies.
"""

from typing import List

import numpy as np
import pandas as pd


def dedup_barcodes(df: pd.DataFrame) -> pd.DataFrame:
    """Remove barcodes that map to more than one variant.

    In the LABEL-seq library, each barcode should be uniquely assigned to one
    variant via PacBio long-read sequencing. Barcodes that map to multiple
    variants are ambiguous and must be removed.

    Args:
        df: Barcode-level DataFrame with 'barcode' and 'variant' columns.

    Returns:
        DataFrame with multi-mapping barcodes removed.
    """
    barcode_variant_counts = df.groupby("barcode")["variant"].nunique()
    duplicate_barcodes = barcode_variant_counts[barcode_variant_counts > 1].index
    return df[~df["barcode"].isin(duplicate_barcodes)].copy()


def apply_count_cutoff(
    df: pd.DataFrame,
    numerator: str,
    denominator: str,
    cutoff: int = 10,
) -> pd.DataFrame:
    """Set counts below the minimum threshold to NaN.

    Any replicate where either the numerator or denominator channel has
    fewer than `cutoff` reads is set to NaN for both channels. This prevents
    noisy low-count ratios from entering the scoring pipeline.

    Args:
        df: Barcode-level DataFrame with replicate count columns.
        numerator: Channel name prefix (e.g., 'pEM1', 'Flag', 'Strep').
        denominator: Channel name prefix (e.g., 'E40', 'HT', 'Flag').
        cutoff: Minimum read count. Defaults to 10.

    Returns:
        DataFrame with sub-threshold counts replaced by NaN.
    """
    df = df.copy()
    for j in range(1, 4):
        num_col = f"{numerator}_{j}"
        denom_col = f"{denominator}_{j}"
        df[denom_col] = np.where(df[denom_col] < cutoff, np.nan, df[denom_col])
        df[num_col] = np.where(df[num_col] < cutoff, np.nan, df[num_col])
    return df


def apply_freq_filter(
    df: pd.DataFrame,
    numerator: str,
    denominator: str,
    freq_cutoffs: List[float] = None,
) -> pd.DataFrame:
    """Filter barcodes below a per-replicate frequency threshold.

    Computes the frequency of each barcode as:
        freq = (numerator + denominator) / total_reads_in_replicate

    Barcodes below the threshold have their counts set to NaN.

    Note: In the current pipeline, freq_cutoffs = [0, 0, 0], meaning this
    filter has no effect (GOTCHA G3). The code runs but removes nothing.

    Args:
        df: Barcode-level DataFrame with replicate count columns.
        numerator: Channel name prefix.
        denominator: Channel name prefix.
        freq_cutoffs: Per-replicate frequency thresholds. Defaults to [0, 0, 0].

    Returns:
        DataFrame with frequency columns added and sub-threshold counts NaN'd.
    """
    if freq_cutoffs is None:
        freq_cutoffs = [0, 0, 0]

    df = df.copy()
    for value, j in zip(freq_cutoffs, range(1, 4)):
        num_col = f"{numerator}_{j}"
        denom_col = f"{denominator}_{j}"

        total = df[denom_col].sum() + df[num_col].sum()
        df[f"frequency_{j}"] = (df[denom_col] + df[num_col]) / total

        df[denom_col] = np.where(df[f"frequency_{j}"] < value, np.nan, df[denom_col])
        df[num_col] = np.where(df[f"frequency_{j}"] < value, np.nan, df[num_col])
    return df


def remove_outliers(
    df: pd.DataFrame,
    numerator: str,
    denominator: str,
    sd_threshold: float = 2.5,
    min_replicates: int = 2,
) -> pd.DataFrame:
    """Remove barcodes with extreme ratios across replicates.

    For each variant, identifies barcodes whose ratio exceeds
    median + sd_threshold * std within a replicate. A barcode is only
    removed if it's flagged as an outlier in at least `min_replicates`
    replicates.

    IMPORTANT (BUG B2): Only the UPPER bound is checked. The lower bound
    (median - 2.5*std) is commented out in the original notebook. This means
    barcodes with abnormally LOW ratios are retained, potentially pulling
    variant averages downward.

    When a barcode is flagged, ALL its replicate values (numerator,
    denominator, AND ratio) are set to NaN. This is critical because
    downstream WT normalization reads numerator/denominator directly —
    failing to NaN those would contaminate the WT mean.

    Args:
        df: Barcode-level DataFrame with 'ratio_1', 'ratio_2', 'ratio_3'
            columns and a 'variant' column.
        numerator: Channel name prefix (e.g., 'pEM1').
        denominator: Channel name prefix (e.g., 'E40').
        sd_threshold: Number of standard deviations above median to flag.
        min_replicates: Minimum number of replicates where a barcode must
            be an outlier to trigger removal.

    Returns:
        DataFrame with outlier barcodes' values set to NaN.
    """
    df = df.copy()

    outlier_flags = []
    for j in range(1, 4):
        ratio_col = f"ratio_{j}"
        median_ratios = df.groupby("variant")[ratio_col].transform("median")
        std_ratios = df.groupby("variant")[ratio_col].transform("std")

        # BUG B2: Only upper bound is checked. Lower bound is commented out
        # in the original notebook. See docs/known_bugs_and_gotchas.md B2.
        mask_j = df[ratio_col] > median_ratios + sd_threshold * std_ratios
        outlier_flags.append(mask_j)

    # Sum flags across replicates
    outlier_sum = (
        outlier_flags[0].astype(int)
        + outlier_flags[1].astype(int)
        + outlier_flags[2].astype(int)
    )

    # Only remove if outlier in >= min_replicates replicates
    final_outliers = outlier_sum >= min_replicates

    # Set ALL replicate values to NaN for flagged barcodes.
    # Must NaN numerator and denominator too — WT normalization reads these
    # directly, not the ratio columns.
    for j in range(1, 4):
        df.loc[final_outliers, f"{denominator}_{j}"] = np.nan
        df.loc[final_outliers, f"{numerator}_{j}"] = np.nan
        df.loc[final_outliers, f"ratio_{j}"] = np.nan

    return df
