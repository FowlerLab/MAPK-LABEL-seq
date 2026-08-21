"""End-to-end pipeline orchestration for LABEL-seq MAPK scoring.

Wires together io, filtering, scoring, and variant modules to replicate
the original filtering and scoring logic.

The pipeline processes raw barcode-count TSVs through:
  load → annotate variants → dedup → count cutoff → freq filter →
  compute ratios → outlier removal → WT normalize → position offsets →
  aggregate to variants → variant frequency → filter unknowns →
  standard curve → barcode cutoff → classification →
  final filtering → output

Usage:
    config = load_config(Path("config/"))
    config = resolve_paths(config, project_root)
    scores = run_scoring(config)
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from labelseq_mapk.annotation import run_annotation
from labelseq_mapk.config import (
    get_position_offset,
    load_config,
    resolve_paths,
)
from labelseq_mapk.filtering import (
    apply_count_cutoff,
    apply_freq_filter,
    dedup_barcodes,
    remove_outliers,
)
from labelseq_mapk.io import load_raw_dataframes, parse_filename
from labelseq_mapk.scoring import (
    aggregate_to_variants,
    apply_barcode_cutoff,
    apply_position_offsets,
    classify_variants,
    compute_quantified_barcodes,
    compute_ratios,
    compute_variant_frequency,
    fit_standard_curve,
    normalize_to_wt,
)
from labelseq_mapk.variants import (
    assign_mutation_types_vectorized,
    mutation_type,
    process_aa_variant,
    process_variants_vectorized,
)


def _process_single_group(
    data: pd.DataFrame,
    library: str,
    assay: str,
    assay_treatment: str,
    config: Dict[str, Any],
) -> pd.DataFrame:
    """Process a single (library, assay, treatment) group through the scoring pipeline.

    This implements the inner loop of Notebook 1: for each group of barcodes
    sharing the same library/assay/treatment, apply all filtering and scoring
    steps to produce variant-level scores.

    Args:
        data: Barcode-level DataFrame for this group.
        library: Library name (e.g., 'braf_cterm').
        assay: Assay type ('activity', 'abundance', 'interaction').
        assay_treatment: Treatment condition (e.g., 'No_treatment', 'HSP90i').
        config: Full config dict.

    Returns:
        Variant-level DataFrame with scores and classifications.
    """
    scoring_cfg = config["scoring"]
    channels = scoring_cfg["assay_channels"][assay]
    numerator = channels["numerator"]
    denominator = channels["denominator"]

    # --- Barcode-level processing ---

    # 1. Remove duplicate barcodes
    data = dedup_barcodes(data)

    # 2. Annotate variants (vectorized for speed)
    data = process_variants_vectorized(data)
    data["Number of Barcodes"] = data["variant"].map(data["variant"].value_counts())
    data = assign_mutation_types_vectorized(data)

    # 3. Apply count cutoff
    data = apply_count_cutoff(
        data, numerator, denominator, scoring_cfg["count_cutoff"]
    )

    # 4. Apply frequency filter
    data = apply_freq_filter(
        data, numerator, denominator, scoring_cfg["freq_cutoffs"]
    )

    # 5. Compute ratios
    data = compute_ratios(data, numerator, denominator)

    # 6. Remove outliers (must NaN numerator/denominator too for WT normalization)
    data = remove_outliers(
        data,
        numerator=numerator,
        denominator=denominator,
        sd_threshold=scoring_cfg["outlier"]["sd_threshold"],
        min_replicates=scoring_cfg["outlier"]["min_replicates"],
    )

    # 7. Compute average ratio
    data["average ratio"] = data[
        [f"ratio_{j}" for j in range(1, 4)]
    ].mean(axis=1)

    # 8. Normalize to wild type
    data = normalize_to_wt(data, numerator, denominator)

    # --- Variant-level aggregation ---

    # 9. Aggregate to variant level
    replicate_scores = aggregate_to_variants(data)

    # 10. Compute variant frequency
    replicate_scores = compute_variant_frequency(
        data, replicate_scores, numerator, denominator
    )

    # 11. Re-annotate variants (at variant level, vectorized)
    replicate_scores = process_variants_vectorized(replicate_scores)
    replicate_scores = assign_mutation_types_vectorized(replicate_scores)

    # 12. Filter out unknowns and frameshifts
    replicate_scores = replicate_scores[
        ~replicate_scores["Mutation Type"].isin(["unknown", "frame shift"])
    ].copy()

    # 13. Standard curve fitting
    standards_dict = scoring_cfg["standard_controls"].get(assay, {})

    # Check if this (library, assay) should skip standard curve
    skip_std = any(
        s["library"] == library and s["assay"] == assay
        for s in scoring_cfg.get("skip_standard_curve", [])
    )

    replicate_scores = fit_standard_curve(
        replicate_scores, standards_dict, skip=skip_std
    )

    # 14. Compute quantified barcode counts
    replicate_scores = compute_quantified_barcodes(data, replicate_scores)

    # 15. Apply barcode count cutoff
    replicate_scores = apply_barcode_cutoff(
        replicate_scores, scoring_cfg["mean_barcodes_cutoff"]
    )

    # 16. Add metadata columns
    replicate_scores["library"] = library
    replicate_scores["assay"] = assay
    replicate_scores["assay_treatment"] = assay_treatment

    return replicate_scores


def run_scoring(config: Dict[str, Any], verbose: bool = True) -> pd.DataFrame:
    """Run the full scoring pipeline: raw TSVs → Replicate_scores_masterframe.

    Loads all raw barcode-count TSVs, processes each (library, assay, treatment)
    group through the filtering/scoring pipeline, applies position offsets,
    classifies variants, and produces the final scored DataFrame.

    Args:
        config: Full config dict with resolved paths.
        verbose: If True, print progress messages.

    Returns:
        Variant-level DataFrame matching Replicate_scores_masterframe_autogenerated.tsv.
    """
    scoring_cfg = config["scoring"]
    data_dir = Path(config["paths"]["raw_dataframes"])

    # Load all raw data
    if verbose:
        print(f"Loading raw data from {data_dir}...")
    raw_data = load_raw_dataframes(data_dir)
    if verbose:
        print(f"  Loaded {len(raw_data):,} barcode rows across "
              f"{raw_data['library'].nunique()} libraries")

    # Process each (library, assay, treatment) group
    groups = raw_data.groupby(["library", "assay", "assay_treatment"])
    all_scores: List[pd.DataFrame] = []

    for (library, assay, treatment), group_df in groups:
        if verbose:
            print(f"  Processing {library} / {assay} / {treatment} "
                  f"({len(group_df):,} barcodes)...")

        try:
            scores = _process_single_group(
                group_df, library, assay, treatment, config
            )
            all_scores.append(scores)
        except Exception as e:
            print(f"  ERROR processing {library}/{assay}/{treatment}: {e}")
            continue

    # Concatenate all groups
    replicate_scores = pd.concat(all_scores, ignore_index=True)
    if verbose:
        print(f"\nScored {len(replicate_scores):,} variants across all groups")

    # Apply position offsets (operates on the full concatenated DataFrame)
    replicate_scores = apply_position_offsets(replicate_scores, config)

    # Classify variants using percentile thresholds
    replicate_scores = classify_variants(
        replicate_scores, scoring_cfg["classification_percentiles"]
    )

    # Drop intermediate columns (slope columns, etc.)
    cols_to_drop = [
        "intercept_0_slope_score_1",
        "intercept_0_slope_score_2",
        "intercept_0_slope_score_3",
    ]
    existing_drops = [c for c in cols_to_drop if c in replicate_scores.columns]
    if existing_drops:
        replicate_scores = replicate_scores.drop(columns=existing_drops)

    # Remove nonsense variants for specific libraries, ABUNDANCE assay only.
    # Investigation showed the golden file excludes nonsense for these 6
    # libraries in abundance but NOT activity. See GOTCHA G2 and Q11.
    exclude_libs = scoring_cfg.get("exclude_nonsense_libraries", [])
    nonsense_mask = (
        replicate_scores["library"].isin(exclude_libs)
        & (replicate_scores["Mutation Type"] == "nonsense")
        & (replicate_scores["assay"] == "abundance")
    )
    replicate_scores = replicate_scores[~nonsense_mask].copy()

    if verbose:
        print(f"Final output: {len(replicate_scores):,} variant scores")

    return replicate_scores


def run_full(
    config: Dict[str, Any], verbose: bool = True
) -> pd.DataFrame:
    """Run the full pipeline: raw TSVs → scored → annotated.

    Args:
        config: Full config dict with resolved paths.
        verbose: Print progress.

    Returns:
        Fully annotated variant-level DataFrame.
    """
    scores = run_scoring(config, verbose=verbose)

    if verbose:
        print("\nRunning annotation pipeline...")
    annotated = run_annotation(scores, config, verbose=verbose)

    return annotated
