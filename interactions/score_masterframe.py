"""
Normalisation of the annotated LABEL-seq score table for the interaction
analysis (used by ``main.py``):

1. **Annotation fan-out collapse.** In an earlier release of the annotated
   table, the per-residue UniProt annotation join emitted one row per *feature* rather than
   one row per *measurement*. 135 positions carry two overlapping features (e.g.
   ARAF 535 is both "activation loop (A-loop)" / active site and "polypeptide
   substrate binding site"), so all 19,740 measurements at those positions
   appear twice. Left alone they get double weight in every variant-level
   statistic — Mann-Whitney n, Cohen's d, log fold change — which quietly
   inflates significance at exactly the functionally annotated positions we care
   most about. :func:`collapse_annotation_fanout` merges those rows back to one
   measurement, unioning the feature lists and OR-ing the boolean flags.

2. **Gene symbol canonicalisation.** The table's ``protein`` column holds
   library-style names (``craf``, ``mek1``, ``mek2``, ``shp2``) while every
   downstream artefact — ``domain_info.json``, the prior-evidence tables, PDB
   filenames — is keyed on HGNC symbols (``RAF1``,
   ``MAP2K1``, ``MAP2K2``, ``PTPN11``). :func:`load_gene_symbols` reads the
   accession→symbol map so callers can derive ``gene_name`` from the accession
   instead of trusting the label in the file.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

# Columns that identify a single measurement. Two rows agreeing on all of these
# are the same experimental observation, not two observations.
MEASUREMENT_KEY: Sequence[str] = (
    "protein",
    "library",
    "assay",
    "assay_treatment",
    "variant",
)

# Annotation columns the fan-out duplicated. `feature` is a stringified Python
# list; the two boolean flags are derived from it upstream.
FANOUT_LIST_COL = "feature"
FANOUT_BOOL_COLS: Sequence[str] = ("active_site", "protein_interface")

DEFAULT_GENE_SYMBOL_MAP = "gene_symbols.json"


def parse_feature_list(value) -> List[str]:
    """
    Parse the stringified list in the ``feature`` column into real strings.

    Returns ``[]`` for the sentinel ``['none']`` and for anything unparseable,
    so callers can treat "no annotation" and "annotation missing" alike.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, (list, tuple)):
        items = list(value)
    else:
        text = str(value).strip()
        if not text:
            return []
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            # Not a list literal — treat the raw string as a single feature.
            return [] if text.lower() in {"none", "nan"} else [text]
        items = list(parsed) if isinstance(parsed, (list, tuple)) else [parsed]
    return [str(x) for x in items if str(x).strip().lower() not in {"none", "nan", ""}]


def format_feature_list(features: Sequence[str]) -> str:
    """Render a feature list back into the file's ``"['a', 'b']"`` convention."""
    return repr(["none"] if not features else sorted(dict.fromkeys(features)))


def _coerce_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    return str(value).strip().lower() in {"true", "yes", "1"}


def collapse_annotation_fanout(
    df: pd.DataFrame,
    key: Sequence[str] = MEASUREMENT_KEY,
    logger=None,
) -> pd.DataFrame:
    """
    Collapse annotation-join duplicates to one row per measurement.

    Rows sharing ``key`` are merged: ``feature`` becomes the union of the
    per-row feature lists, ``active_site`` / ``protein_interface`` become the
    logical OR, and every other column is taken from the first row (they are
    identical within a group by construction — this is a join fan-out, not
    replicate measurements).

    Row order is preserved and the index is reset. A frame with no duplicates is
    returned unchanged apart from the index, so this is safe to call on any
    masterframe version.
    """
    present_key = [c for c in key if c in df.columns]
    if not present_key:
        return df.reset_index(drop=True)

    dup_mask = df.duplicated(subset=present_key, keep=False)
    n_dup_rows = int(dup_mask.sum())
    if n_dup_rows == 0:
        return df.reset_index(drop=True)

    # Guard: this collapse is only valid when the duplicate rows differ *only*
    # in the annotation columns. Anything else means the file has genuinely
    # repeated measurements and silently keeping the first row would drop data.
    dup = df[dup_mask]
    annotation_cols = {FANOUT_LIST_COL, *FANOUT_BOOL_COLS}
    conflicting = [
        col
        for col in df.columns
        if col not in present_key
        and col not in annotation_cols
        and (dup.groupby(present_key, dropna=False)[col].nunique(dropna=False) > 1).any()
    ]
    if conflicting:
        raise ValueError(
            f"{n_dup_rows} rows share {list(present_key)} but disagree on "
            f"non-annotation column(s) {conflicting}. These are not annotation "
            "fan-out duplicates — resolve them upstream before loading."
        )

    keep = df[~dup_mask]
    merged_rows = []
    for _, group in dup.groupby(present_key, dropna=False, sort=False):
        row = group.iloc[0].copy()
        if FANOUT_LIST_COL in group.columns:
            features: List[str] = []
            for value in group[FANOUT_LIST_COL]:
                features.extend(parse_feature_list(value))
            row[FANOUT_LIST_COL] = format_feature_list(features)
        for col in FANOUT_BOOL_COLS:
            if col in group.columns:
                row[col] = bool(group[col].map(_coerce_bool).any())
        merged_rows.append(row)

    out = pd.concat([keep, pd.DataFrame(merged_rows)], ignore_index=False)
    out = out.sort_index().reset_index(drop=True)

    msg = (
        f"Collapsed annotation fan-out: {n_dup_rows} duplicated rows across "
        f"{len(merged_rows)} measurements -> {len(df)} - {len(df) - len(out)} = "
        f"{len(out)} rows (features unioned, flags OR-ed)"
    )
    if logger is not None:
        logger.info(msg)
    return out


def load_gene_symbols(path: Optional[Path | str] = None) -> Dict[str, str]:
    """Load the ``{uniprot_accession: HGNC symbol}`` map."""
    path = Path(path or DEFAULT_GENE_SYMBOL_MAP)
    with open(path) as fh:
        return json.load(fh)


def apply_gene_symbols(
    df: pd.DataFrame,
    symbols: Dict[str, str],
    accession_col: str = "uniprot_accession",
    out_col: str = "gene_name",
) -> pd.DataFrame:
    """
    Set ``out_col`` to the HGNC symbol for each row's accession.

    Isoform suffixes (``P01116-2``) are stripped before lookup. Raises if any
    accession is unmapped, so a new protein in the masterframe fails loudly
    rather than silently producing pages named after an accession.
    """
    accessions = df[accession_col].astype(str).str.split("-").str[0]
    unmapped = sorted(set(accessions) - set(symbols))
    if unmapped:
        raise ValueError(
            f"No HGNC symbol for accession(s) {unmapped}. "
            f"Add them to {DEFAULT_GENE_SYMBOL_MAP}."
        )
    out = df.copy()
    out[out_col] = accessions.map(symbols).values
    return out
