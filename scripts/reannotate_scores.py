"""Annotate OUR scores from primary sources. No masterframe anywhere.

Stage 2 of three:

1. ``Scoring.ipynb`` (+ ``score_cell.py`` / ``finalize_scores.py``) builds the
   masterframe from raw barcode counts. Scores and identity only.
2. **this script**, driven by ``Annotations.ipynb`` -- annotates it.
3. ``DN.ipynb`` / ``HSP90i.ipynb`` -- draw the figures.

Every annotation is rebuilt from a primary file: the AlphaFold models, the DSSP
outputs, the NCBI conserved-domain tables, PhosphoSitePlus, ClinVar,
AlphaMissense, CysDB, the published DMS tables, the kinase alignment, and the
domain bounds in ``config/proteins.yaml``. Nothing is transferred out of
Jessica's delivery.

Two sources are combined:

* ``labelseq_mapk.annotation.run_annotation`` -- the port of her
  ``Generate_table.ipynb``: DSSP/RSA, domains, NCBI features and active sites,
  PPI interfaces, PTMs, ClinVar, AlphaMissense, CysDB, OpenCell, alignment
  positions, published DMS comparisons.
* ``compute_structure_annotations.py`` -- the per-residue structural columns
  ``run_annotation`` has no function for: pLDDT, the structure's own residue
  identity, and intramolecular domain-domain contacts. Both pLDDT and residue
  identity were verified to reproduce her inherited values exactly (100.0000%),
  and the contact flags to 98.6-98.9% under our own stated cutoffs.

``--compare-to`` runs an agreement check against her delivery. It is QC output
only and never feeds the pipeline; a script may reference her table, a pipeline
stage may not.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from labelseq_mapk.annotation import run_annotation  # noqa: E402
from labelseq_mapk.config import load_config  # noqa: E402
# the merge itself lives in utils so the notebook and this script cannot drift
from utils import add_structure_annotations  # noqa: E402

SCORES = ROOT / "output" / "scoring" / "scores_masterframe_recovered.tsv"
STRUCT = ROOT / "output" / "structure_annotations.tsv"
OUT = ROOT / "output" / "scoring" / "scores_reannotated.tsv"
#: Reference identifiers and genomic coordinates, keyed on (protein, variant).
#: Built by scripts/build_variant_mapping.py.
MAPPING = ROOT / "output" / "variant_mapping.tsv"
#: What we take from it. `hgvs_g` and `g_snvs_grch38` are GRCh38; a variant has
#: them only where its protein change is reachable by a single-nucleotide route
#: that could be resolved, which is why they are ~a third filled rather than all.
MAPPING_COLS = ["hgvs_c", "hgvs_g", "g_snvs_grch38", "clingen_allele_id"]
PROV = ROOT / "output" / "scoring" / "reannotation_provenance.tsv"

#: Columns carried by compute_structure_annotations.py, keyed (protein, position).
#: The merge itself is utils.add_structure_annotations, shared with the notebook.
#: `inter_domain_contacts` (the older, non-all-atom cutoff) is deliberately not
#: here. Every interface, dimer and chaperone contact definition in the project
#: was unified to 4 A all-atom on 2026-08-10, so carrying the superseded flag
#: alongside invited picking the wrong one -- and the two differ on 3,919 rows.
#: It is still computed into output/structure_annotations.tsv as a diagnostic.
STRUCT_COLS = ["plddt", "pdb_aa",
               "inter_domain_contacts_all_atom", "inter_domain_partners"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", type=Path, default=SCORES)
    ap.add_argument("--struct", type=Path, default=STRUCT)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--compare-to", type=Path,
                    help="her delivery, for a QC agreement check only")
    args = ap.parse_args()

    cfg = load_config(ROOT / "config")
    scores = pd.read_csv(args.scores, sep="\t", low_memory=False)
    n_in, cols_in = len(scores), set(scores.columns)
    print(f"loaded {n_in:,} rows x {len(cols_in)} cols from {args.scores.name}")

    leaked = sorted(cols_in & {"domain", "relative_sasa", "active_site", "feature",
                               "protein_interface", "client_status", "plddt"})
    assert not leaked, (
        f"{args.scores.name} already carries annotation columns {leaked}; stage 1 "
        "should emit scores and identity only")
    print("stage 1 table is annotation-free, as expected\n")

    ann = run_annotation(scores, cfg, verbose=True)
    print()
    ann = add_structure_annotations(ann, args.struct)
    print(f"structure annotations merged from {args.struct.name}: "
          f"plddt non-null {ann.plddt.notna().mean():.1%}, "
          f"inter-domain (4 A) {int(ann.inter_domain_contacts_all_atom.sum()):,} rows")

    # `annotation_source` used to be stamped here as a constant string. Dropped
    # 2026-08-20: a column with one value carries no information, and the
    # provenance it asserted is recorded properly in reannotation_provenance.tsv.

    # ---- reference identifiers and GRCh38 coordinates ----------------------
    if MAPPING.exists():
        m = pd.read_csv(MAPPING, sep="\t", low_memory=False,
                        usecols=["protein", "variant"] + MAPPING_COLS)
        m = m.drop_duplicates(["protein", "variant"])
        before = len(ann)
        ann = ann.merge(m, on=["protein", "variant"], how="left")
        assert len(ann) == before, "variant mapping merge changed the row count"
        print(f"\ngenomic coordinates merged from {MAPPING.name}: "
              + ", ".join(f"{c} {ann[c].notna().mean():.1%}" for c in MAPPING_COLS))
    else:
        for c in MAPPING_COLS:
            ann[c] = pd.NA
        print(f"\n{MAPPING.name} absent -- genomic coordinate columns left empty")

    rows = [(c, "from scoring (stage 1)" if c in cols_in
             else "structural (AlphaFold models)" if c in STRUCT_COLS
             or c == "pdb_aa_mismatch"
             else "reference identifiers / GRCh38 coordinates"
             if c in MAPPING_COLS
             else "annotation engine (primary files)")
            for c in ann.columns]
    prov = pd.DataFrame(rows, columns=["column", "origin"]).sort_values(
        ["origin", "column"])
    prov.to_csv(PROV, sep="\t", index=False)
    print("\ncolumn provenance:")
    for origin, g in prov.groupby("origin"):
        print(f"  {origin:36s} {len(g):>3d} columns")

    assert len(ann) == n_in, f"row count changed: {n_in:,} -> {len(ann):,}"
    ann.to_csv(args.out, sep="\t", index=False)
    print(f"\nwrote {args.out.relative_to(ROOT)}")
    print(f"  {len(ann):,} rows x {len(ann.columns)} columns")
    print(f"wrote {PROV.relative_to(ROOT)}")

    if args.compare_to and args.compare_to.exists():
        hers = pd.read_csv(args.compare_to, sep="\t", low_memory=False)
        shared = [c for c in ann.columns if c in hers.columns
                  and c not in ("variant", "library", "assay", "assay_treatment")]
        keys = ["variant", "library", "assay", "assay_treatment"]
        m = ann[keys + shared].merge(hers[keys + shared], on=keys,
                                     suffixes=("_ours", "_hers"))
        print(f"\nQC — agreement with {args.compare_to.name} "
              f"on {len(m):,} shared rows:")
        out_rows = []
        for c in shared:
            a, b = f"{c}_ours", f"{c}_hers"
            ok = m[a].notna() & m[b].notna()
            if not ok.any():
                out_rows.append((c, 0, float("nan")))
                continue
            x, y = m.loc[ok, a], m.loc[ok, b]
            numeric = (pd.api.types.is_numeric_dtype(x)
                       and pd.api.types.is_numeric_dtype(y)
                       and not pd.api.types.is_bool_dtype(x)
                       and not pd.api.types.is_bool_dtype(y))
            if numeric:
                agree = ((x - y).abs() <= 1e-6 * y.abs().clip(lower=1)).mean()
            else:
                # her booleans are "yes"/"no" strings where ours are True/False,
                # so normalise both to a truth value before comparing -- without
                # this every boolean column reads as 0% agreement
                TRUE, FALSE = {"true", "yes", "1", "1.0"}, {"false", "no", "0", "0.0"}
                nx = x.astype(str).str.strip().str.lower()
                ny = y.astype(str).str.strip().str.lower()
                if nx.isin(TRUE | FALSE).all() and ny.isin(TRUE | FALSE).all():
                    agree = (nx.isin(TRUE) == ny.isin(TRUE)).mean()
                else:
                    agree = (nx == ny).mean()
            out_rows.append((c, int(ok.sum()), agree))
        rep = pd.DataFrame(out_rows, columns=["column", "n", "agree"]
                           ).sort_values("agree")
        for r in rep.itertuples():
            print(f"  {r.column:44s} {r.agree:8.4%}  (n={r.n:,})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
