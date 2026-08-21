"""HGVSp normalisation utilities for joining scored variants with gnomAD/AoU.

The canonical protein-level identifier in this codebase is the HGVSp produced
by dcd_mapping2 (https://github.com/VariantEffect/dcd_mapping2), which lives
in the `hgvs_p` column of ``merged_master_with_hgvs_p.csv``. Population
databases (gnomAD ``hgvsp`` field, AoU annotatedcsv ``achange`` column) use
slightly different conventions:

    merged_master:       "NP_001645.1:p.Ala334Asp"     (with accession)
    gnomAD hgvsp:        "p.Lys807Thr"                 (no accession)
    AoU achange:         "p.Arg351Ter"                 (no accession)
    enumerated stops:    (construct "p.Glu2Ter" from wt_residue + position)

This module provides one function, ``normalise_hgvsp``, that produces a
canonical form suitable for exact equality joins across all four sources:

    * strip any "<accession>:" prefix
    * convert "*" stop glyph to "Ter"
    * leave synonymous "=" as-is (kept distinct from missense / stops)
    * preserve three-letter amino acid codes

and a helper ``build_stop_hgvsp`` for constructing HGVSp from enumerated
stop-SNV rows (which carry wt_residue as a one-letter code + position).
"""

from __future__ import annotations

from typing import Optional

# Single-letter to three-letter amino acid codes.
AA1_TO_AA3 = {
    "A": "Ala", "R": "Arg", "N": "Asn", "D": "Asp", "C": "Cys",
    "Q": "Gln", "E": "Glu", "G": "Gly", "H": "His", "I": "Ile",
    "L": "Leu", "K": "Lys", "M": "Met", "F": "Phe", "P": "Pro",
    "S": "Ser", "T": "Thr", "W": "Trp", "Y": "Tyr", "V": "Val",
    "*": "Ter",
}


def normalise_hgvsp(hgvsp: Optional[str]) -> Optional[str]:
    """Return a canonical HGVSp string for cross-database joins.

    Strips the accession prefix (everything up to and including the first
    colon), normalises "*" to "Ter", and returns None for empty / NaN input.

    Args:
        hgvsp: Raw HGVSp string in any of the accepted forms.

    Returns:
        Canonical form ("p.XxxNNNYyy") or None if the input is empty.

    Examples:
        >>> normalise_hgvsp("NP_001645.1:p.Ala334Asp")
        'p.Ala334Asp'
        >>> normalise_hgvsp("p.Arg100*")
        'p.Arg100Ter'
        >>> normalise_hgvsp("p.Ala334=")
        'p.Ala334='
        >>> normalise_hgvsp(None) is None
        True
        >>> normalise_hgvsp("") is None
        True
    """
    if hgvsp is None:
        return None
    s = str(hgvsp).strip()
    if not s or s.lower() == "nan":
        return None
    # Strip accession prefix (e.g., "NP_001645.1:p.Ala334Asp" → "p.Ala334Asp").
    if ":" in s:
        s = s.split(":", 1)[1]
    # Normalise stop glyph.
    s = s.replace("*", "Ter")
    return s


def build_stop_hgvsp(wt_residue: str, position: int) -> str:
    """Construct canonical stop-gained HGVSp from an enumerated stop SNV.

    The nonsense_all_snvs.tsv enumeration stores wt_residue as a one-letter
    code and position as an integer; the population databases report
    stop_gained variants as "p.Xxx<N>Ter". This helper converts the former
    to the latter.

    Args:
        wt_residue: One-letter amino acid code at the position (e.g. "R").
        position: 1-based protein position.

    Returns:
        Canonical HGVSp string, e.g. "p.Arg100Ter".

    Raises:
        KeyError: If wt_residue is not a recognised one-letter code.
    """
    aa3 = AA1_TO_AA3[wt_residue.upper()]
    return f"p.{aa3}{int(position)}Ter"
