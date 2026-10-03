#!/usr/bin/env python3
"""Per-protein univariate model of HSP90 buffering — the table behind ED 8b.

For each (protein, feature), a multinomial logit ``buffering_class ~ feature`` on
that protein's kinase-domain missense variants (WT-like or high = reference), as
in the factor analysis of 2026-08 (LABELseq_MAPK
``scripts/plot_hsp90_factor_per_protein_importance.py``), refitted on the current
tables:

* **Cohort** — exactly the one Fig 5f and ED 8c plot: abundance paired untreated
  vs HSP90i within library, buffering classes on the WT-relative score against
  each arm's synonymous 2.5th percentile (``hsp90_calls``), missense, curated
  kinase domain, classified.
* **Features** — substitution chemistry from the residue pair
  (``aa_substitution``); per-position AlphaFold geometry and alignment-assigned
  secondary structure / lobe / motif (``output/hsp90/structural_features_afv4.tsv``);
  phyloP (``output/hsp90/phylop_per_position.tsv``, which equals the table's
  ``phylop_vert`` wherever both exist and also covers KSR2, MEK1 and MEK2); and
  from the annotated table, kinase JSD, SPURS ddG, relative SASA, the active site
  (the corrected NCBI CD-search), the curated 4 Å PDB interface and its partner
  count, and the three chaperone contacts.

Continuous features are z-scored on the pooled cohort, so coefficients are per
pooled SD and comparable across proteins. A binary feature is fitted for a
protein only if every class has at least ``MIN_POSITIONS`` distinct positions
carrying it (positions, not variants: substitutions at one residue are not
independent observations).

Writes ``output/hsp90/factor_analysis/buffering_factor_per_protein_importance.tsv``
(protein, feature, contrast, coef, p, pseudo_r2), read by ED 8b in
``Fig5_and_associated_ED.ipynb``.

    python scripts/fit_buffering_factors_per_protein.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from labelseq_mapk import hsp90_calls  # noqa: E402
from labelseq_mapk.aa_substitution import compute_substitution_features  # noqa: E402
from labelseq_mapk.figure_data import load_scores  # noqa: E402

HSP90 = ROOT / "output" / "hsp90"
OUT = HSP90 / "factor_analysis" / "buffering_factor_per_protein_importance.tsv"
STRUCT = HSP90 / "structural_features_afv4.tsv"
PHYLOP = HSP90 / "phylop_per_position.tsv"

SCORE_WT_REL = "average score"
PROTEINS = ["craf", "araf", "ret", "ksr2", "egfr", "met", "braf", "mek2", "mek1"]
CLASS_CODE = {"WT-like or high": 0, "Buffered": 1, "Poorly buffered": 2}
CONTRASTS = ["buffered", "poor"]          # MNLogit columns for classes 1 and 2
MIN_POSITIONS = 5

CONTINUOUS = [
    "delta_hydrophobicity", "delta_volume", "delta_charge", "delta_polarity",
    "delta_mw", "grantham_distance", "blosum62_score", "wt_hydrophobicity",
    "wt_volume", "jsd_conservation", "phylop_vert", "ddG_SPURS", "relative_sasa",
    "plddt", "contact_number_8A", "contact_number_12A", "depth",
    "n_backbone_hbonds", "n_sidechain_hbonds", "hbond_network_fraction",
    "long_range_contacts", "long_range_contact_fraction", "betweenness_centrality",
    "clustering_coefficient", "n_partners", "dist_to_atp_site",
    "dist_to_activation_loop",
]
BINARY = [
    "to_proline", "from_proline", "to_glycine", "from_glycine", "charge_reversal",
    "aromatic_loss", "aromatic_gain", "ss_helix", "ss_strand", "is_n_lobe",
    "is_c_lobe", "motif_gly_rich_loop", "motif_alphaC_helix", "motif_beta4_beta5",
    "motif_hinge", "motif_DFG", "motif_activation_loop", "motif_catalytic_loop",
    "active_site", "interface_pdb_curated", "at_hsp90_unfolded", "at_hsp90_folded",
    "at_cdc37",
]
STRUCT_COLS = [
    "plddt", "contact_number_8A", "contact_number_12A", "depth",
    "n_backbone_hbonds", "n_sidechain_hbonds", "hbond_network_fraction",
    "long_range_contacts", "long_range_contact_fraction", "betweenness_centrality",
    "clustering_coefficient", "ss_helix", "ss_strand", "is_n_lobe", "is_c_lobe",
    "dist_to_atp_site", "dist_to_activation_loop", "motif_gly_rich_loop",
    "motif_alphaC_helix", "motif_beta4_beta5", "motif_hinge", "motif_DFG",
    "motif_activation_loop", "motif_catalytic_loop",
]


def build_cohort() -> tuple[pd.DataFrame, list[str]]:
    ann = load_scores()
    paired = hsp90_calls.pair_abundance(ann, SCORE_WT_REL)
    paired = hsp90_calls.assign_buffering(paired, ann, SCORE_WT_REL)
    df = paired[(paired["Mutation Type"] == "missense") & (paired.domain == "kinase")
                & paired.protein.isin(PROTEINS)
                & paired.buffering_class.notna()].copy()
    n_cohort = len(df)
    df["Position"] = pd.to_numeric(df["Position"])

    chem = pd.DataFrame([compute_substitution_features(w, m) for w, m in
                         zip(df["Wild Type Residue"], df["Mutation"])], index=df.index)
    df = pd.concat([df, chem[[c for c in chem.columns if c not in df.columns]]], axis=1)

    st = pd.read_csv(STRUCT, sep="\t").rename(columns={"position": "Position"})
    # The structure file's pLDDT is from the same AlphaFold models as the table's;
    # take every structural column from one source.
    df = df.drop(columns=[c for c in STRUCT_COLS if c in df.columns]).merge(st[["protein", "Position", *STRUCT_COLS]], on=["protein", "Position"],
                  how="left", validate="many_to_one")
    ph = pd.read_csv(PHYLOP, sep="\t")[["protein", "Position", "phylop_vert"]]
    df = df.drop(columns="phylop_vert").merge(ph, on=["protein", "Position"],
                                              how="left", validate="many_to_one")
    # The interface degree is the number of curated PDB partners at 4 A.
    df["n_partners"] = df["n_interface_partners"].fillna(0)

    df = df.dropna(subset=CONTINUOUS + BINARY).copy()
    print(f"cohort: {n_cohort:,} classified kinase-domain missense; "
          f"{len(df):,} with every predictor ({n_cohort - len(df):,} dropped)")
    for c in BINARY:
        df[c] = df[c].astype(float)

    terms = []
    for c in CONTINUOUS:
        sd = df[c].std()
        if sd > 1e-9 * (abs(df[c].mean()) + 1):
            df[f"{c}_z"] = (df[c] - df[c].mean()) / sd
            terms.append(f"{c}_z")
    n = len(df)
    for c in BINARY:   # drop near-constant flags, as in the August analysis
        minority = min(df[c].sum(), n - df[c].sum())
        if minority >= 30 and minority / n >= 0.005:
            terms.append(c)
        else:
            print(f"  dropped near-constant binary {c} (minority {int(minority)})")
    df["y"] = df.buffering_class.map(CLASS_CODE).astype(int)
    return df, terms


def per_protein_univariate(df: pd.DataFrame, terms: list[str]) -> pd.DataFrame:
    rows = []
    for protein in PROTEINS:
        sub = df[df.protein == protein]
        y = sub["y"].to_numpy(dtype=float)
        if len(np.unique(y)) < 3:
            continue
        for feat in terms:
            col = sub[feat].astype(float)
            if col.nunique() < 2:
                continue
            if set(col.unique()) <= {0.0, 1.0}:
                pos = sub.loc[col == 1].groupby("y")["Position"].nunique()
                if len(pos) < 3 or pos.min() < MIN_POSITIONS:
                    continue
            X = sm.add_constant(col.reset_index(drop=True), has_constant="add")
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = sm.MNLogit(y, X).fit(method="lbfgs", maxiter=2000, disp=0)
            except Exception:   # degenerate within-protein fit
                continue
            try:
                pvals = res.pvalues
            except (ValueError, np.linalg.LinAlgError):
                pvals = None
            for j, contrast in enumerate(CONTRASTS):
                rows.append({
                    "protein": protein, "feature": feat, "contrast": contrast,
                    "coef": float(res.params.iloc[1, j]),
                    "p": float(pvals.iloc[1, j]) if pvals is not None else np.nan,
                    "pseudo_r2": max(0.0, float(res.prsquared)),
                })
    return pd.DataFrame(rows)


def main() -> None:
    df, terms = build_cohort()
    print(df.groupby("protein").buffering_class.value_counts().unstack().reindex(PROTEINS)
          .to_string())
    out = per_protein_univariate(df, terms)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, sep="\t", index=False)
    print(f"wrote {OUT.relative_to(ROOT)}: {out.feature.nunique()} features x "
          f"{out.protein.nunique()} proteins, {len(out)} rows")


if __name__ == "__main__":
    main()
