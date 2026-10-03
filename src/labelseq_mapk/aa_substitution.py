"""Amino-acid substitution chemistry features (defensible, basis-independent).

Pure functions of (wild-type AA, mutant AA) from standard biochemical lookup
tables and matrices — ported verbatim from the prior `claude_HSP90`
`hsp90_features.compute_substitution_features` so the paper-1 factor analysis
reuses the exact same definitions. Nothing here depends on structure or on the
measured phenotype, so these are reusable across projects without a basis
caveat (Grantham/ΔΔG were verified identical old-vs-new).

`compute_substitution_features(wt, mut)` returns the per-variant feature dict.
"""

from __future__ import annotations

import numpy as np

# Kyte-Doolittle hydrophobicity.
HYDROPHOBICITY = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}
VOLUME = {
    "A": 88.6, "R": 173.4, "N": 114.1, "D": 111.1, "C": 108.5,
    "Q": 143.8, "E": 138.4, "G": 60.1, "H": 153.2, "I": 166.7,
    "L": 166.7, "K": 168.6, "M": 162.9, "F": 189.9, "P": 112.7,
    "S": 89.0, "T": 116.1, "W": 227.8, "Y": 193.6, "V": 140.0,
}
CHARGE = {
    "A": 0, "R": 1, "N": 0, "D": -1, "C": 0, "Q": 0, "E": -1,
    "G": 0, "H": 0, "I": 0, "L": 0, "K": 1, "M": 0, "F": 0,
    "P": 0, "S": 0, "T": 0, "W": 0, "Y": 0, "V": 0,
}
POLARITY = {
    "A": 0.0, "R": 52.0, "N": 3.38, "D": 49.7, "C": 1.48,
    "Q": 3.53, "E": 49.9, "G": 0.0, "H": 51.6, "I": 0.13,
    "L": 0.13, "K": 49.5, "M": 1.43, "F": 0.35, "P": 1.58,
    "S": 1.67, "T": 1.66, "W": 2.10, "Y": 1.61, "V": 0.13,
}
MOLECULAR_WEIGHT = {
    "A": 89.1, "R": 174.2, "N": 132.1, "D": 133.1, "C": 121.2,
    "Q": 146.2, "E": 147.1, "G": 75.0, "H": 155.2, "I": 131.2,
    "L": 131.2, "K": 146.2, "M": 149.2, "F": 165.2, "P": 115.1,
    "S": 105.1, "T": 119.1, "W": 204.2, "Y": 181.2, "V": 117.1,
}
AROMATIC = set("FWY")

# Grantham distance (composition/polarity/volume); symmetric.
_GRANTHAM = {
    ("A", "R"): 112, ("A", "N"): 111, ("A", "D"): 126, ("A", "C"): 195, ("A", "Q"): 91, ("A", "E"): 107,
    ("A", "G"): 60, ("A", "H"): 86, ("A", "I"): 94, ("A", "L"): 96, ("A", "K"): 106, ("A", "M"): 84,
    ("A", "F"): 113, ("A", "P"): 27, ("A", "S"): 99, ("A", "T"): 58, ("A", "W"): 148, ("A", "Y"): 112,
    ("A", "V"): 64, ("R", "N"): 86, ("R", "D"): 96, ("R", "C"): 180, ("R", "Q"): 43, ("R", "E"): 54,
    ("R", "G"): 125, ("R", "H"): 29, ("R", "I"): 97, ("R", "L"): 102, ("R", "K"): 26, ("R", "M"): 91,
    ("R", "F"): 97, ("R", "P"): 103, ("R", "S"): 110, ("R", "T"): 71, ("R", "W"): 101, ("R", "Y"): 77,
    ("R", "V"): 96, ("N", "D"): 23, ("N", "C"): 139, ("N", "Q"): 46, ("N", "E"): 42, ("N", "G"): 80,
    ("N", "H"): 68, ("N", "I"): 149, ("N", "L"): 153, ("N", "K"): 94, ("N", "M"): 142, ("N", "F"): 158,
    ("N", "P"): 91, ("N", "S"): 46, ("N", "T"): 65, ("N", "W"): 174, ("N", "Y"): 143, ("N", "V"): 133,
    ("D", "C"): 154, ("D", "Q"): 61, ("D", "E"): 45, ("D", "G"): 94, ("D", "H"): 81, ("D", "I"): 168,
    ("D", "L"): 172, ("D", "K"): 101, ("D", "M"): 160, ("D", "F"): 177, ("D", "P"): 108, ("D", "S"): 65,
    ("D", "T"): 85, ("D", "W"): 181, ("D", "Y"): 160, ("D", "V"): 152, ("C", "Q"): 154, ("C", "E"): 170,
    ("C", "G"): 159, ("C", "H"): 174, ("C", "I"): 198, ("C", "L"): 198, ("C", "K"): 202, ("C", "M"): 196,
    ("C", "F"): 205, ("C", "P"): 169, ("C", "S"): 112, ("C", "T"): 149, ("C", "W"): 215, ("C", "Y"): 194,
    ("C", "V"): 192, ("Q", "E"): 29, ("Q", "G"): 87, ("Q", "H"): 24, ("Q", "I"): 109, ("Q", "L"): 113,
    ("Q", "K"): 53, ("Q", "M"): 101, ("Q", "F"): 116, ("Q", "P"): 76, ("Q", "S"): 68, ("Q", "T"): 42,
    ("Q", "W"): 130, ("Q", "Y"): 99, ("Q", "V"): 96, ("E", "G"): 98, ("E", "H"): 40, ("E", "I"): 134,
    ("E", "L"): 138, ("E", "K"): 56, ("E", "M"): 126, ("E", "F"): 140, ("E", "P"): 93, ("E", "S"): 80,
    ("E", "T"): 65, ("E", "W"): 152, ("E", "Y"): 122, ("E", "V"): 121, ("G", "H"): 98, ("G", "I"): 135,
    ("G", "L"): 138, ("G", "K"): 127, ("G", "M"): 127, ("G", "F"): 153, ("G", "P"): 42, ("G", "S"): 56,
    ("G", "T"): 59, ("G", "W"): 184, ("G", "Y"): 147, ("G", "V"): 109, ("H", "I"): 94, ("H", "L"): 99,
    ("H", "K"): 32, ("H", "M"): 87, ("H", "F"): 100, ("H", "P"): 77, ("H", "S"): 89, ("H", "T"): 47,
    ("H", "W"): 115, ("H", "Y"): 83, ("H", "V"): 84, ("I", "L"): 5, ("I", "K"): 102, ("I", "M"): 10,
    ("I", "F"): 21, ("I", "P"): 95, ("I", "S"): 142, ("I", "T"): 89, ("I", "W"): 61, ("I", "Y"): 33,
    ("I", "V"): 29, ("L", "K"): 107, ("L", "M"): 15, ("L", "F"): 22, ("L", "P"): 98, ("L", "S"): 145,
    ("L", "T"): 92, ("L", "W"): 61, ("L", "Y"): 36, ("L", "V"): 32, ("K", "M"): 95, ("K", "F"): 102,
    ("K", "P"): 103, ("K", "S"): 121, ("K", "T"): 78, ("K", "W"): 110, ("K", "Y"): 85, ("K", "V"): 97,
    ("M", "F"): 28, ("M", "P"): 87, ("M", "S"): 135, ("M", "T"): 81, ("M", "W"): 67, ("M", "Y"): 36,
    ("M", "V"): 21, ("F", "P"): 114, ("F", "S"): 155, ("F", "T"): 103, ("F", "W"): 40, ("F", "Y"): 22,
    ("F", "V"): 50, ("P", "S"): 74, ("P", "T"): 38, ("P", "W"): 147, ("P", "Y"): 110, ("P", "V"): 68,
    ("S", "T"): 58, ("S", "W"): 177, ("S", "Y"): 144, ("S", "V"): 124, ("T", "W"): 128, ("T", "Y"): 92,
    ("T", "V"): 69, ("W", "Y"): 37, ("W", "V"): 88, ("Y", "V"): 55,
}

# BLOSUM62 (subset of pairs needed; symmetric, includes diagonal).
_BLOSUM62 = {
    ("A", "A"): 4, ("A", "R"): -1, ("A", "N"): -2, ("A", "D"): -2, ("A", "C"): 0, ("A", "Q"): -1,
    ("A", "E"): -1, ("A", "G"): 0, ("A", "H"): -2, ("A", "I"): -1, ("A", "L"): -1, ("A", "K"): -1,
    ("A", "M"): -1, ("A", "F"): -2, ("A", "P"): -1, ("A", "S"): 1, ("A", "T"): 0, ("A", "W"): -3,
    ("A", "Y"): -2, ("A", "V"): 0, ("R", "R"): 5, ("R", "N"): 0, ("R", "D"): -2, ("R", "C"): -3,
    ("R", "Q"): 1, ("R", "E"): 0, ("R", "G"): -2, ("R", "H"): 0, ("R", "I"): -3, ("R", "L"): -2,
    ("R", "K"): 2, ("R", "M"): -1, ("R", "F"): -3, ("R", "P"): -2, ("R", "S"): -1, ("R", "T"): -1,
    ("R", "W"): -3, ("R", "Y"): -2, ("R", "V"): -3, ("N", "N"): 6, ("N", "D"): 1, ("N", "C"): -3,
    ("N", "Q"): 0, ("N", "E"): 0, ("N", "G"): 0, ("N", "H"): 1, ("N", "I"): -3, ("N", "L"): -3,
    ("N", "K"): 0, ("N", "M"): -2, ("N", "F"): -3, ("N", "P"): -2, ("N", "S"): 1, ("N", "T"): 0,
    ("N", "W"): -4, ("N", "Y"): -2, ("N", "V"): -3, ("D", "D"): 6, ("D", "C"): -3, ("D", "Q"): 0,
    ("D", "E"): 2, ("D", "G"): -1, ("D", "H"): -1, ("D", "I"): -3, ("D", "L"): -4, ("D", "K"): -1,
    ("D", "M"): -3, ("D", "F"): -3, ("D", "P"): -1, ("D", "S"): 0, ("D", "T"): -1, ("D", "W"): -4,
    ("D", "Y"): -3, ("D", "V"): -3, ("C", "C"): 9, ("C", "Q"): -3, ("C", "E"): -4, ("C", "G"): -3,
    ("C", "H"): -3, ("C", "I"): -1, ("C", "L"): -1, ("C", "K"): -3, ("C", "M"): -1, ("C", "F"): -2,
    ("C", "P"): -3, ("C", "S"): -1, ("C", "T"): -1, ("C", "W"): -2, ("C", "Y"): -2, ("C", "V"): -1,
    ("Q", "Q"): 5, ("Q", "E"): 2, ("Q", "G"): -2, ("Q", "H"): 0, ("Q", "I"): -3, ("Q", "L"): -2,
    ("Q", "K"): 1, ("Q", "M"): 0, ("Q", "F"): -3, ("Q", "P"): -1, ("Q", "S"): 0, ("Q", "T"): -1,
    ("Q", "W"): -2, ("Q", "Y"): -1, ("Q", "V"): -2, ("E", "E"): 5, ("E", "G"): -2, ("E", "H"): 0,
    ("E", "I"): -3, ("E", "L"): -3, ("E", "K"): 1, ("E", "M"): -2, ("E", "F"): -3, ("E", "P"): -1,
    ("E", "S"): 0, ("E", "T"): -1, ("E", "W"): -3, ("E", "Y"): -2, ("E", "V"): -2, ("G", "G"): 6,
    ("G", "H"): -2, ("G", "I"): -4, ("G", "L"): -4, ("G", "K"): -2, ("G", "M"): -3, ("G", "F"): -3,
    ("G", "P"): -2, ("G", "S"): 0, ("G", "T"): -2, ("G", "W"): -2, ("G", "Y"): -3, ("G", "V"): -3,
    ("H", "H"): 8, ("H", "I"): -3, ("H", "L"): -3, ("H", "K"): -1, ("H", "M"): -2, ("H", "F"): -1,
    ("H", "P"): -2, ("H", "S"): -1, ("H", "T"): -2, ("H", "W"): -2, ("H", "Y"): 2, ("H", "V"): -3,
    ("I", "I"): 4, ("I", "L"): 2, ("I", "K"): -3, ("I", "M"): 1, ("I", "F"): 0, ("I", "P"): -3,
    ("I", "S"): -2, ("I", "T"): -1, ("I", "W"): -3, ("I", "Y"): -1, ("I", "V"): 3, ("L", "L"): 4,
    ("L", "K"): -2, ("L", "M"): 2, ("L", "F"): 0, ("L", "P"): -3, ("L", "S"): -2, ("L", "T"): -1,
    ("L", "W"): -2, ("L", "Y"): -1, ("L", "V"): 1, ("K", "K"): 5, ("K", "M"): -1, ("K", "F"): -3,
    ("K", "P"): -1, ("K", "S"): 0, ("K", "T"): -1, ("K", "W"): -3, ("K", "Y"): -2, ("K", "V"): -2,
    ("M", "M"): 5, ("M", "F"): 0, ("M", "P"): -2, ("M", "S"): -1, ("M", "T"): -1, ("M", "W"): -1,
    ("M", "Y"): -1, ("M", "V"): 1, ("F", "F"): 6, ("F", "P"): -4, ("F", "S"): -2, ("F", "T"): -2,
    ("F", "W"): 1, ("F", "Y"): 3, ("F", "V"): -1, ("P", "P"): 7, ("P", "S"): -1, ("P", "T"): -1,
    ("P", "W"): -4, ("P", "Y"): -3, ("P", "V"): -2, ("S", "S"): 4, ("S", "T"): 1, ("S", "W"): -3,
    ("S", "Y"): -2, ("S", "V"): -2, ("T", "T"): 5, ("T", "W"): -2, ("T", "Y"): -2, ("T", "V"): 0,
    ("W", "W"): 11, ("W", "Y"): 2, ("W", "V"): -3, ("Y", "Y"): 7, ("Y", "V"): -1, ("V", "V"): 4,
}


def grantham_distance(aa1: str, aa2: str) -> float:
    """Grantham distance between two residues (0 if identical)."""
    if aa1 == aa2:
        return 0.0
    key = (aa1, aa2) if (aa1, aa2) in _GRANTHAM else (aa2, aa1)
    return float(_GRANTHAM.get(key, np.nan))


def blosum62_score(aa1: str, aa2: str) -> float:
    """BLOSUM62 substitution score (symmetric; diagonal for identical)."""
    key = (aa1, aa2) if (aa1, aa2) in _BLOSUM62 else (aa2, aa1)
    return float(_BLOSUM62.get(key, np.nan))


def compute_substitution_features(wt_aa: str, mut_aa: str) -> dict[str, float]:
    """Per-variant substitution-chemistry features.

    Args:
        wt_aa: wild-type residue (1-letter).
        mut_aa: mutant residue (1-letter).

    Returns:
        Dict of 17 features (delta properties, Grantham, BLOSUM62, special-residue
        flags, WT context). NaN for non-standard residues.
    """
    f: dict[str, float] = {}
    f["delta_hydrophobicity"] = HYDROPHOBICITY.get(mut_aa, np.nan) - HYDROPHOBICITY.get(wt_aa, np.nan)
    f["delta_volume"] = VOLUME.get(mut_aa, np.nan) - VOLUME.get(wt_aa, np.nan)
    f["delta_charge"] = CHARGE.get(mut_aa, np.nan) - CHARGE.get(wt_aa, np.nan)
    f["delta_polarity"] = POLARITY.get(mut_aa, np.nan) - POLARITY.get(wt_aa, np.nan)
    f["delta_mw"] = MOLECULAR_WEIGHT.get(mut_aa, np.nan) - MOLECULAR_WEIGHT.get(wt_aa, np.nan)
    f["grantham_distance"] = grantham_distance(wt_aa, mut_aa)
    f["blosum62_score"] = blosum62_score(wt_aa, mut_aa)
    f["to_proline"] = float(mut_aa == "P" and wt_aa != "P")
    f["from_proline"] = float(wt_aa == "P" and mut_aa != "P")
    f["to_glycine"] = float(mut_aa == "G" and wt_aa != "G")
    f["from_glycine"] = float(wt_aa == "G" and mut_aa != "G")
    f["charge_reversal"] = float(
        (CHARGE.get(wt_aa, 0) > 0 and CHARGE.get(mut_aa, 0) < 0)
        or (CHARGE.get(wt_aa, 0) < 0 and CHARGE.get(mut_aa, 0) > 0)
    )
    f["aromatic_loss"] = float(wt_aa in AROMATIC and mut_aa not in AROMATIC)
    f["aromatic_gain"] = float(wt_aa not in AROMATIC and mut_aa in AROMATIC)
    f["wt_hydrophobicity"] = HYDROPHOBICITY.get(wt_aa, np.nan)
    f["mut_hydrophobicity"] = HYDROPHOBICITY.get(mut_aa, np.nan)
    f["wt_volume"] = VOLUME.get(wt_aa, np.nan)
    return f
