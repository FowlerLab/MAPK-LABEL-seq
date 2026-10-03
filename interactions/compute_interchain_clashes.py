#!/usr/bin/env python
"""
Compute inter-chain C-beta clash counts per structure and maintain the cache.

`interchain_cbeta_clashes.csv` gates the clash filter in
`ppi_utils.load_combined_results`, which drops any structure with
a confident inter-chain C-beta contact below 3 A as a physically implausible pose
(chains interpenetrating). The cache had no generator in the repo — it was
produced ad hoc in April 2026 — which is a problem, because `load_combined_results`
fills structures MISSING from the cache with zero clashes. A new structure is
therefore assumed clash-free and passes the filter unchecked. This script closes
that gap so the cache can be rebuilt or extended.

Geometry deliberately reuses `main.calculate_cbeta_distances`, so "C-beta" means
exactly what it means in the analysis (C-beta, or C-alpha for glycine) and the
counts are directly comparable with the existing rows rather than a parallel
reimplementation that might drift.

A clash is "confident" when BOTH residues have B-factor >= 60. For AlphaFold
models that column is pLDDT, so this is the intended confidence filter. For
experimental structures it is a crystallographic temperature factor and the
threshold has no confidence meaning — that is the convention the existing cache
already uses (has_plddt is True for RCSB rows), and experimental structures
essentially never clash, so it is preserved here rather than silently changed.
A model with no confidence column at all (B-factor 0 throughout) counts every
contact as confident.

Usage
-----
    # add only structures missing from the cache (default, cheap)
    python compute_interchain_clashes.py

    # recompute everything from scratch
    python compute_interchain_clashes.py --refresh

    # report what would change, write nothing
    python compute_interchain_clashes.py --dry-run
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

import main as M

CACHE = Path("output/interactions/interchain_cbeta_clashes.csv")

# Both residues must reach this B-factor/pLDDT for a contact to count as
# "confident". 60 matches the pipeline's plddt_threshold.
CONFIDENT_MIN = 60.0

COLUMNS = [
    "structure", "source", "scored_protein", "interactor", "has_plddt",
    "n_all_clashes_lt2A", "n_all_clashes_lt3A",
    "n_confident_clashes_lt2A", "n_confident_clashes_lt3A", "min_cbeta_dist",
]


def structure_dirs(config: dict) -> Dict[str, List[Path]]:
    """Map predictor name -> directories that may hold its PDB files."""
    s = config.get("structures", {})
    return {
        "RCSB_PDB": [Path(s.get("rcsb_pdb_dir", "data/interactions/pdb_controls"))],
        "Predictomes": [Path(s.get("pdb_dir", "data/interactions/predictomes_MAPK"))],
        "Zhang_et_al": [Path(s.get("zhangetal_pdb_dir", "data/interactions/Zhangetal_structures"))],
    }


def find_structure(name: str, dirs: List[Path]) -> Optional[Path]:
    """Locate a structure file by name, searching nested layouts if needed."""
    for d in dirs:
        p = d / name
        if p.exists():
            return p
        # Zhang files live in per-pair subdirectories.
        hits = list(d.glob(f"*/{name}"))
        if hits:
            return hits[0]
    return None


def clashes_for_structure(path: Path) -> Optional[dict]:
    """
    Count inter-chain C-beta contacts below 2 A and 3 A.

    Returns None when the structure cannot be read or has no two-chain contact
    within the 15 A window `calculate_cbeta_distances` keeps — callers should
    treat that as "not assessable" rather than as zero clashes.
    """
    try:
        structure = M.load_structure(path)
        dist = M.calculate_cbeta_distances(structure)
    except Exception:
        return None
    if dist.empty:
        return None

    bf_a = M.get_residue_plddt(structure, "A")
    bf_b = M.get_residue_plddt(structure, "B")
    has_bf = bool(bf_a) and bool(bf_b) and (
        max(list(bf_a.values()) + list(bf_b.values())) > 0
    )

    # A model with no confidence column (the Zhang et al. "AF" models carry
    # B-factor 0 throughout) gives no grounds to discount a clash, so every
    # contact counts, as it does in the ad hoc April cache this replaces. Treating
    # them as never confident instead let 25 Zhang models with C-beta contacts
    # down to 1.4 A through the filter.
    confident = dist.apply(
        lambda r: (bf_a.get(r["res_a"], 0.0) >= CONFIDENT_MIN
                   and bf_b.get(r["res_b"], 0.0) >= CONFIDENT_MIN),
        axis=1,
    ) if has_bf else pd.Series(True, index=dist.index)

    return {
        "has_plddt": has_bf,
        "n_all_clashes_lt2A": int((dist["distance"] < 2.0).sum()),
        "n_all_clashes_lt3A": int((dist["distance"] < 3.0).sum()),
        "n_confident_clashes_lt2A": int(((dist["distance"] < 2.0) & confident).sum()),
        "n_confident_clashes_lt3A": int(((dist["distance"] < 3.0) & confident).sum()),
        "min_cbeta_dist": round(float(dist["distance"].min()), 1),
    }


def wanted_structures(config: dict) -> pd.DataFrame:
    """Every structure that produced statistics, with its predictor and proteins."""
    frames = []
    for key in ("pdb_output_dir", "predictomes_dir", "zhangetal_dir"):
        d = config.get(key)
        if not d:
            continue
        f = Path(d) / "all_interactions.csv"
        if f.exists():
            frames.append(pd.read_csv(f))
    df = pd.concat(frames, ignore_index=True)
    df = df[df["skipped"] != True]  # noqa: E712
    cols = ["structure", "predictor", "scored_protein_name", "interactor_name"]
    out = df[cols].dropna(subset=["structure"]).drop_duplicates("structure")
    return out.rename(columns={
        "predictor": "source",
        "scored_protein_name": "scored_protein",
        "interactor_name": "interactor",
    })


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--config", type=Path, default=Path("config/visualization_config.yaml"))
    ap.add_argument("--refresh", action="store_true",
                    help="Recompute every structure instead of only the missing ones")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would change without writing the cache")
    args = ap.parse_args(argv)

    with open(args.config) as fh:
        config = yaml.safe_load(fh)

    want = wanted_structures(config)
    dirs = structure_dirs(config)
    existing = pd.read_csv(CACHE) if CACHE.exists() and not args.refresh else pd.DataFrame()

    have = set(existing["structure"]) if not existing.empty else set()
    todo = want[~want["structure"].isin(have)]
    stale = have - set(want["structure"])

    print(f"structures with statistics : {len(want)}")
    print(f"already cached             : {len(have & set(want['structure']))}")
    print(f"to compute                 : {len(todo)}")
    print(f"stale cache rows           : {len(stale)}")
    if args.dry_run:
        for s in sorted(todo["structure"]):
            print("   would compute:", s)
        return 0

    rows, unreadable = [], []
    for i, r in enumerate(todo.itertuples(index=False), start=1):
        path = find_structure(r.structure, dirs.get(r.source, []))
        if path is None:
            unreadable.append((r.structure, "file not found"))
            continue
        res = clashes_for_structure(path)
        if res is None:
            unreadable.append((r.structure, "no two-chain C-beta contact"))
            continue
        rows.append({"structure": r.structure, "source": r.source,
                     "scored_protein": r.scored_protein, "interactor": r.interactor,
                     **res})
        if i % 25 == 0 or i == len(todo):
            print(f"   {i}/{len(todo)}")

    fresh = pd.DataFrame(rows, columns=COLUMNS)
    combined = (pd.concat([existing, fresh], ignore_index=True)
                if not existing.empty else fresh)
    # Drop rows for structures no longer in the analysis so the cache mirrors it.
    combined = combined[combined["structure"].isin(set(want["structure"]))]
    combined = combined.drop_duplicates("structure", keep="last")
    combined[COLUMNS].to_csv(CACHE, index=False)

    print(f"\nwrote {len(combined)} rows -> {CACHE} (added {len(fresh)}, dropped {len(stale)})")
    if unreadable:
        print(f"WARNING: {len(unreadable)} structures could not be assessed and are "
              "absent from the cache; load_combined_results will treat them as "
              "clash-free:")
        for s, why in unreadable[:10]:
            print(f"   {s}: {why}")
    clashing = combined[combined["n_confident_clashes_lt3A"] > 0]
    print(f"structures with confident clashes < 3 A: {len(clashing)} "
          f"({len(clashing) / len(combined):.1%}) -> excluded downstream")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
