"""Cell-by-cell comparison of two Supplementary Table 2 files.

Floats are compared to within 4 ulp (so the 1-ulp drift of pandas' default CSV
parser does not count as a difference); every other cell must be identical as
text. Prints, per column, how many cells differ and a few examples.

usage: python scripts/compare_supplementary_table_2.py OURS THEIRS
"""
import sys

import numpy as np
import pandas as pd

KEY = ["library", "variant", "assay", "assay_treatment"]


def read_release(path):
    """Read a release as text; drop the unnamed pandas index earlier releases carry."""
    with open(path) as fh:
        has_index = fh.readline().startswith("\t")
    return pd.read_csv(path, sep="\t", index_col=0 if has_index else None,
                       dtype=str, keep_default_na=False).reset_index(drop=True)


def main(ours_path, theirs_path):
    a, b = read_release(ours_path), read_release(theirs_path)
    print(f"ours   {a.shape}  {ours_path}\ntheirs {b.shape}  {theirs_path}")
    only_a = [c for c in a.columns if c not in b.columns]
    only_b = [c for c in b.columns if c not in a.columns]
    if only_a or only_b:
        print(f"columns only in ours: {only_a}\ncolumns only in theirs: {only_b}")
    shared = [c for c in a.columns if c in b.columns]
    assert shared == [c for c in b.columns if c in a.columns], "shared columns are in a different order"
    a, b = a[shared], b[shared]
    assert a[KEY].equals(b[KEY]), "row keys/order differ"
    n_diff_total = 0
    for c in a.columns:
        x, y = a[c], b[c]
        same = x == y
        if not same.all():
            xf = pd.to_numeric(x.where(x != ""), errors="coerce")
            yf = pd.to_numeric(y.where(y != ""), errors="coerce")
            num = xf.notna() & yf.notna()
            close = num & np.isclose(xf, yf, rtol=4 * np.finfo(float).eps, atol=0)
            same = same | close
        n = int((~same).sum())
        n_diff_total += n
        if n:
            ex = pd.DataFrame({"ours": x[~same], "theirs": y[~same]}).head(3)
            print(f"  {c}: {n:,} cells differ\n{ex.to_string()}")
    print(f"\n{n_diff_total:,} differing cells beyond float rounding")
    return n_diff_total


if __name__ == "__main__":
    sys.exit(1 if main(*sys.argv[1:3]) else 0)
