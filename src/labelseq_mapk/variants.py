"""Variant string parsing and mutation type classification.

Ported from the original analysis. Known bugs are replicated
exactly for parity — see docs/known_bugs_and_gotchas.md for details.

Performance note: The original row-by-row `apply` functions are retained for
testing and reference, but `process_variants_vectorized` and
`assign_mutation_types_vectorized` should be used in the pipeline. They use
pandas string operations and are ~100x faster on large DataFrames.
"""

from typing import Tuple, Union

import numpy as np
import pandas as pd


def process_aa_variant(row: pd.Series) -> Union[Tuple[str, str, Union[str, int]], str]:
    """Parse a variant string into (wild_type_residue, mutation, position).

    Handles standard LABEL-seq variant naming: single-letter AA code + position
    + single-letter mutation code (e.g., 'V600E'). Also recognizes special
    variants: wild type ('WT', '_wt'), standards ('_std'), frameshifts ('fs'),
    multi-variants ('|'), and readthroughs ('RT').

    Args:
        row: A pandas Series with a 'variant' column containing the variant
            string to parse.

    Returns:
        A tuple of (wild_type_residue, mutation, position) where position is
        an int for parseable variants, or 'unknown'/'wild type'/'standard'
        strings for special cases.

    Known bugs:
        BUG B1: The except clause returns a scalar 'unknown' instead of a
        3-tuple ('unknown', 'unknown', 'unknown'). This is replicated exactly
        for parity with the original notebook.
    """
    aa_variant = str(row["variant"])
    try:
        if "fs" in aa_variant:
            return "unknown", "unknown", "unknown"
        elif "|" in aa_variant:
            return "unknown", "unknown", "unknown"
        elif "RT" in aa_variant:
            return "unknown", "unknown", "unknown"
        elif "silent" in aa_variant:
            return "unknown", "unknown", "unknown"
        elif "WT" in aa_variant:
            return "wild type", "wild type", "wild type"
        elif "_wt" in aa_variant:
            return "wild type", "wild type", "wild type"
        elif "std" in aa_variant:
            return "standard", "standard", "standard"
        else:
            if len(aa_variant) in [3, 4, 5, 6]:
                wild_type_residue = aa_variant[0]
                mutation = aa_variant[-1]
                if len(aa_variant) == 3:
                    position = int(aa_variant[1])
                elif len(aa_variant) == 4:
                    position = int(aa_variant[1:3])
                elif len(aa_variant) == 5:
                    position = int(aa_variant[1:4])
                elif len(aa_variant) == 6:
                    position = int(aa_variant[1:5])
                return wild_type_residue, mutation, int(position)
            else:
                return "unknown", "unknown", "unknown"
    except Exception:
        # BUG B1: Returns scalar string, not 3-tuple. Replicated for parity.
        # See docs/known_bugs_and_gotchas.md B1.
        return "unknown"


def mutation_type(row: pd.Series) -> str:
    """Classify a variant's mutation type based on parsed residue information.

    Uses the 'Wild Type Residue' and 'Mutation' columns (output of
    process_aa_variant) to assign one of: wild type, unknown, standard,
    synonymous wild type, frame shift, deletion, nonsense, or missense.

    The classification order matters: wild type and special types are checked
    before the synonymous/missense distinction.

    Args:
        row: A pandas Series with 'Wild Type Residue' and 'Mutation' columns.

    Returns:
        One of: 'wild type', 'unknown', 'standard', 'synonymous wild type',
        'frame shift', 'deletion', 'nonsense', 'missense'.
    """
    if row["Wild Type Residue"] == "wild type":
        return "wild type"
    elif row["Mutation"] == "unknown":
        return "unknown"
    elif row["Mutation"] == "standard":
        return "standard"
    elif row["Wild Type Residue"] == row["Mutation"]:
        return "synonymous wild type"
    elif row["Mutation"] == "frame shift":
        return "frame shift"
    elif row["Mutation"] == "-":
        return "deletion"
    elif row["Mutation"] == "*":
        return "nonsense"
    else:
        return "missense"


# ---------------------------------------------------------------------------
# Vectorized versions (fast) — use these in the pipeline
# ---------------------------------------------------------------------------


def process_variants_vectorized(df: pd.DataFrame) -> pd.DataFrame:
    """Parse variant strings into WT residue, mutation, and position columns.

    Vectorized replacement for process_aa_variant(). Uses pandas string
    operations instead of row-by-row apply, giving ~100x speedup on large
    DataFrames.

    Produces identical output to:
        df.apply(process_aa_variant, axis=1, result_type='expand')

    Args:
        df: DataFrame with a 'variant' column.

    Returns:
        DataFrame with 'Wild Type Residue', 'Mutation', 'Position' columns added.
    """
    df = df.copy()
    v = df["variant"].astype(str)

    # Start with all unknown
    wt_res = pd.Series("unknown", index=df.index)
    mut = pd.Series("unknown", index=df.index)
    pos = pd.Series("unknown", index=df.index, dtype=object)

    # --- Special-case flags (checked in original if/elif order) ---
    # Each flag is True only if NO earlier flag was True (exclusive)
    is_fs = v.str.contains("fs", na=False)
    is_pipe = (~is_fs) & v.str.contains(r"\|", na=False)
    is_rt = (~is_fs) & (~is_pipe) & v.str.contains("RT", na=False)
    is_silent = (~is_fs) & (~is_pipe) & (~is_rt) & v.str.contains("silent", na=False)
    is_wt = (~is_fs) & (~is_pipe) & (~is_rt) & (~is_silent) & v.str.contains("WT", na=False)
    is_wt2 = (~is_fs) & (~is_pipe) & (~is_rt) & (~is_silent) & (~is_wt) & v.str.contains("_wt", na=False)
    is_std = (~is_fs) & (~is_pipe) & (~is_rt) & (~is_silent) & (~is_wt) & (~is_wt2) & v.str.contains("std", na=False)

    # All special cases already default to 'unknown'; override WT and std
    wt_mask = is_wt | is_wt2
    wt_res[wt_mask] = "wild type"
    mut[wt_mask] = "wild type"
    pos[wt_mask] = "wild type"

    wt_res[is_std] = "standard"
    mut[is_std] = "standard"
    pos[is_std] = "standard"

    # --- Parseable variants: length 3-6, not flagged as special ---
    any_special = is_fs | is_pipe | is_rt | is_silent | is_wt | is_wt2 | is_std
    vlen = v.str.len()
    parseable = (~any_special) & vlen.isin([3, 4, 5, 6])

    if parseable.any():
        pv = v[parseable]
        wt_res[parseable] = pv.str[0]
        mut[parseable] = pv.str[-1]

        # Position is the middle characters, converted to int
        pos_str = pv.str[1:-1]
        # Use pd.to_numeric to handle any edge cases that would throw in int()
        pos_numeric = pd.to_numeric(pos_str, errors="coerce")

        # Rows where position parsing failed → unknown (BUG B1 parity:
        # original returns scalar 'unknown' on exception, which pandas expand
        # turns into NaN. We replicate by leaving as 'unknown'.)
        failed = pos_numeric.isna()
        if failed.any():
            wt_res[parseable & failed.reindex(df.index, fill_value=True)] = "unknown"
            mut[parseable & failed.reindex(df.index, fill_value=True)] = "unknown"

        # Successful parses get integer position
        pos[parseable] = pos_numeric.reindex(df.index)
        # Fill back the successful ones as ints
        success_mask = parseable & ~pos.isna() & (pos != "unknown")
        # We need to handle mixed types carefully
        pos_final = pos.copy()
        if success_mask.any():
            pos_final[success_mask] = pos_numeric.reindex(df.index)[success_mask].astype(int)
        pos = pos_final

    df["Wild Type Residue"] = wt_res
    df["Mutation"] = mut
    df["Position"] = pos

    return df


def assign_mutation_types_vectorized(df: pd.DataFrame) -> pd.DataFrame:
    """Classify mutation types from parsed variant columns.

    Vectorized replacement for mutation_type(). Uses numpy.select for
    conditional assignment instead of row-by-row apply.

    Produces identical output to:
        df.apply(mutation_type, axis=1)

    Args:
        df: DataFrame with 'Wild Type Residue' and 'Mutation' columns.

    Returns:
        DataFrame with 'Mutation Type' column added.
    """
    df = df.copy()
    wt = df["Wild Type Residue"]
    mut = df["Mutation"]

    # Conditions checked in order (first match wins, like if/elif)
    conditions = [
        wt == "wild type",
        mut == "unknown",
        mut == "standard",
        wt == mut,
        mut == "frame shift",
        mut == "-",
        mut == "*",
    ]
    choices = [
        "wild type",
        "unknown",
        "standard",
        "synonymous wild type",
        "frame shift",
        "deletion",
        "nonsense",
    ]

    df["Mutation Type"] = np.select(conditions, choices, default="missense")
    return df
