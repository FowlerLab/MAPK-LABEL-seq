"""Data I/O for the LABEL-seq MAPK pipeline.

Handles loading raw barcode-count TSV files and extracting metadata
(date, library, assay, treatment) from filenames.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


# Known assay types — used to split filename into library vs. assay vs. treatment
_ASSAY_TYPES = {"activity", "abundance", "interaction"}

# Suffix stripped from all raw data filenames before parsing
_FILENAME_SUFFIX = "_dataframe_beforefreqcutoff"


def parse_filename(filename: str) -> Dict[str, str]:
    """Parse a raw data TSV filename into metadata components.

    Filename format: {date}_{library}_{assay}[_{treatment}]_dataframe_beforefreqcutoff.tsv

    The library name can contain underscores (e.g., braf_cterm, araf_nterm),
    so we scan tokens left-to-right and identify the assay keyword
    (activity/abundance/interaction) to split library from treatment.

    Args:
        filename: The filename (not full path) of a raw barcode-count TSV.

    Returns:
        Dictionary with keys: 'date', 'library', 'assay', 'assay_treatment'.
        Treatment defaults to 'No_treatment' if not present in the filename.

    Examples:
        >>> parse_filename('240917_kras_activity_dataframe_beforefreqcutoff.tsv')
        {'date': '240917', 'library': 'kras', 'assay': 'activity', 'assay_treatment': 'No_treatment'}

        >>> parse_filename('250314_braf_cterm_activity_dataframe_beforefreqcutoff.tsv')
        {'date': '250314', 'library': 'braf_cterm', 'assay': 'activity', 'assay_treatment': 'No_treatment'}

        >>> parse_filename('250106_egfr_abundance_HSP90i_dataframe_beforefreqcutoff.tsv')
        {'date': '250106', 'library': 'egfr', 'assay': 'abundance', 'assay_treatment': 'HSP90i'}
    """
    # Strip extension and suffix
    stem = filename.replace(".tsv", "")
    stem = stem.replace(_FILENAME_SUFFIX, "")

    tokens = stem.split("_")

    # First token is always the date (YYMMDD format)
    date = tokens[0]

    # Find the assay keyword by scanning left-to-right
    assay_idx = None
    for i, token in enumerate(tokens):
        if token in _ASSAY_TYPES:
            assay_idx = i
            break

    if assay_idx is None:
        raise ValueError(
            f"Could not find assay type (activity/abundance/interaction) "
            f"in filename: {filename}"
        )

    # Library is everything between date and assay
    library = "_".join(tokens[1:assay_idx])
    assay = tokens[assay_idx]

    # Treatment is everything after assay (if anything)
    treatment_tokens = tokens[assay_idx + 1 :]
    if treatment_tokens:
        assay_treatment = "_".join(treatment_tokens)
    else:
        assay_treatment = "No_treatment"

    return {
        "date": date,
        "library": library,
        "assay": assay,
        "assay_treatment": assay_treatment,
    }


def load_raw_dataframes(
    data_dir: Path, skip_unused: bool = True
) -> pd.DataFrame:
    """Load all raw barcode-count TSV files and concatenate with metadata.

    Reads every .tsv file in data_dir, parses the filename to extract
    metadata (date, library, assay, treatment), and returns a single
    concatenated DataFrame.

    Args:
        data_dir: Path to the directory containing raw TSV files
            (e.g., all_dataframe_beforefreqcutoff_manually_curated/).
        skip_unused: If True, skip files in the 'unused/' subdirectory.

    Returns:
        Combined DataFrame with all barcode-level data plus metadata columns:
        'date', 'library', 'assay', 'assay_treatment'.
    """
    data_dir = Path(data_dir)
    tsv_files = sorted(data_dir.glob("*.tsv"))

    if skip_unused:
        # Also collect files from subdirectories but exclude 'unused/'
        tsv_files = [f for f in tsv_files if "unused" not in f.parts]

    if not tsv_files:
        raise FileNotFoundError(f"No .tsv files found in {data_dir}")

    frames: List[pd.DataFrame] = []
    for filepath in tsv_files:
        metadata = parse_filename(filepath.name)
        df = pd.read_csv(filepath, sep="\t")

        # Add metadata columns
        for key, value in metadata.items():
            df[key] = value

        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    return combined
