"""The score table as the figure notebooks read it.

``output/annotated_combined.tsv`` (written by Annotations.ipynb) carries
Supplementary Table 2's column names. The figure code predates those names, so
:func:`load_scores` maps them back, and derives the notebooks' ``Mutation Type``
from ``variant_category``: "deletion" there means every in-frame 3-nt deletion,
the codon-straddling delins_2for1 included.

Python 3.8 compatible: Interactions.ipynb runs in its own, older environment.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .supplementary_table import RENAME

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANNOTATED_COMBINED = PROJECT_ROOT / "output" / "annotated_combined.tsv"

#: Supplementary Table 2 name -> the name the figure code uses.
CODE_NAMES = {v: k for k, v in RENAME.items()}

#: variant_category -> the notebooks' Mutation Type vocabulary.
MUTATION_TYPE = {
    "missense": "missense",
    "3nt deletion": "deletion",
    "nonsense": "nonsense",
    "synonymous": "synonymous wild type",
    "frameshift": "frame shift",
    "other": "other",
    "standard": "standard",
    "WT": "wild type",
}


def load_scores(path=None, usecols=None, **read_csv_kwargs) -> pd.DataFrame:
    """Read the annotated score table in the figure code's column names.

    ``usecols`` is given in the figure code's names; ``Mutation Type`` may be
    among them, and is always derived rather than read.
    """
    path = Path(path) if path is not None else ANNOTATED_COMBINED
    read_cols = None
    if usecols is not None:
        release = {v: k for k, v in CODE_NAMES.items()}
        wanted = list(dict.fromkeys(usecols))
        read_cols = [release.get(c, c) for c in wanted if c != "Mutation Type"]
        if "variant_category" not in read_cols:
            read_cols.append("variant_category")
    df = pd.read_csv(path, sep="\t", low_memory=False, usecols=read_cols,
                     **read_csv_kwargs)
    df = df.rename(columns=CODE_NAMES)
    unmapped = sorted(set(df["variant_category"].dropna()) - set(MUTATION_TYPE))
    if unmapped:
        raise ValueError(f"variant_category values with no Mutation Type: {unmapped}")
    df["Mutation Type"] = df["variant_category"].map(MUTATION_TYPE)
    if usecols is not None:
        df = df[[c for c in dict.fromkeys(usecols)]]
    return df
