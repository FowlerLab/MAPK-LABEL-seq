"""Per-variant HSP90 inhibition calls written into Supplementary Table 2.

Ported from the export cell of ``Fig5_and_associated_ED.ipynb`` (Jessica
Simon, 2026-09-29), plus the dependence call the Fig 5 panels make. Four calls
per (protein, library, variant):

``buffering_class``
    Buffered / Poorly buffered / WT-like or high, from whether the control and
    the HSP90i abundance each fall below the synonymous-WT 2.5th percentile of
    their own (library, assay, treatment). Computed on the standard-adjusted
    score (``SCORE_ABS``), not the WT-relative score the Fig 5 panels use; the
    two disagree for <1% of variants.
``dependent``
    The paper's dependence call (the "gap rule" of Fig 5d / ED 8a): abundance
    falls under HSP90 inhibition by at least the gap between wild type and the
    bottom of the synonymous distribution in the untreated arm,
    ``ctrl - HSP90i >= max(0, WT_ctrl - syn_2.5pct_ctrl)``, on the
    standard-adjusted score and per library. Fig 5d pools nothing (one library
    per kinase domain); ED 8a pools a split protein's two libraries into one gap,
    so its fractions can differ slightly for ARAF, BRAF, CRAF and KSR2.
``dependent_threshold``
    HSP90i abundance below the HSP90i synonymous-WT 2.5th percentile.
``dependent_significant``
    One-sided Welch t-test of the three HSP90i replicates against the three
    control replicates (H_a: HSP90i < control), p < 0.05, no multiple-testing
    correction. Missense and 3-nt deletions (``variant_category``, so including
    delins_2for1) only.

The table these functions read uses the annotation pipeline's column names
(``intercept_0_standard-adjusted score``, ...), because Annotations.ipynb renames
to the Supplementary Table 2 names only when it writes. Variant types are read
from ``variant_category``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

SCORE_ABS = "intercept_0_standard-adjusted score"
SCORE_ABS_REPS = [f"intercept_0_std_adj_score_{j}" for j in (1, 2, 3)]
HSP90I_PROTEINS = ["araf", "braf", "craf", "egfr", "ksr2", "mek1", "mek2",
                   "met", "ret", "sos2"]
BASELINE_TREATMENTS = {"No_treatment", "DMSO"}
KEY = ["protein", "library", "variant"]


def pair_abundance(ann, score_col, proteins=tuple(HSP90I_PROTEINS)):
    """Pair each variant's control and HSP90i abundance within its library.

    Control is DMSO where the library has it, else No_treatment.
    """
    ab = ann[(ann.assay == "abundance") & ann.protein.isin(set(proteins))].copy()
    ctrl = ab[ab.assay_treatment.isin(BASELINE_TREATMENTS)].copy()
    hsp = ab[ab.assay_treatment == "HSP90i"]
    ctrl["_pref"] = (ctrl.assay_treatment == "DMSO").astype(int)
    ctrl = (ctrl.sort_values("_pref", ascending=False)
            .drop_duplicates(KEY).drop(columns="_pref"))
    out = ctrl.merge(hsp[KEY + [score_col]].rename(columns={score_col: "hsp90i_score"}),
                     on=KEY, how="inner")
    out = out.rename(columns={score_col: "ctrl_score",
                              "assay_treatment": "ctrl_treatment"})
    return out.dropna(subset=["ctrl_score", "hsp90i_score"]).reset_index(drop=True)


def syn_wt_low_threshold(ann, score_col):
    """Synonymous-WT 2.5th percentile per (library, assay, treatment)."""
    syn = ann[(ann["variant_category"] == "synonymous")].dropna(subset=[score_col])
    return (syn.groupby(["library", "assay", "assay_treatment"])[score_col]
            .quantile(0.025).rename("threshold").reset_index())


def assign_buffering(paired, ann, score_col):
    """Add ctrl/HSP90i thresholds and the three-way ``buffering_class``.

    Control-low but HSP90i-not-low is left unclassified (NaN).
    """
    thr = syn_wt_low_threshold(ann, score_col)
    ab = thr[thr.assay == "abundance"]
    ctrl_thr = (ab[ab.assay_treatment.isin(BASELINE_TREATMENTS)]
                .set_index(["library", "assay_treatment"])["threshold"])
    hsp_thr = ab[ab.assay_treatment == "HSP90i"].set_index("library")["threshold"]
    out = paired.copy()
    out["ctrl_threshold"] = pd.MultiIndex.from_arrays(
        [out.library, out.ctrl_treatment]).map(ctrl_thr)
    out["hsp90i_threshold"] = out.library.map(hsp_thr)
    ctrl_low = out.ctrl_score < out.ctrl_threshold
    hsp_low = out.hsp90i_score < out.hsp90i_threshold
    cls = pd.Series(np.nan, index=out.index, dtype=object)
    cls[(~ctrl_low) & hsp_low] = "Buffered"
    cls[ctrl_low & hsp_low] = "Poorly buffered"
    cls[(~ctrl_low) & (~hsp_low)] = "WT-like or high"
    out["buffering_class"] = cls
    return out


def classify_dependent_statistical(df, *, proteins=None, assay="abundance",
                                   variant_types=("missense", "3nt deletion"),
                                   alpha=0.05, min_reps=2):
    """One-sided Welch t-test per variant, HSP90i replicates < control replicates."""
    sub = df[(df.assay == assay) & df["variant_category"].isin(variant_types)].copy()
    if proteins is not None:
        sub = sub[sub.protein.isin(set(proteins))]
    ctrl = sub[sub.assay_treatment.isin(BASELINE_TREATMENTS)].copy()
    hsp = sub[sub.assay_treatment == "HSP90i"]
    ctrl["_pref"] = (ctrl.assay_treatment == "DMSO").astype(int)
    ctrl = (ctrl.sort_values("_pref", ascending=False)
            .drop_duplicates(KEY).drop(columns="_pref"))
    merged = ctrl[KEY + SCORE_ABS_REPS].merge(hsp[KEY + SCORE_ABS_REPS], on=KEY,
                                              suffixes=("_ctrl", "_hsp"))
    ctrl_cols = [f"{c}_ctrl" for c in SCORE_ABS_REPS]
    hsp_cols = [f"{c}_hsp" for c in SCORE_ABS_REPS]
    records = []
    for row in merged.itertuples(index=False):
        ctrl_vals = np.array([getattr(row, c) for c in ctrl_cols], dtype=float)
        hsp_vals = np.array([getattr(row, c) for c in hsp_cols], dtype=float)
        ctrl_vals = ctrl_vals[np.isfinite(ctrl_vals)]
        hsp_vals = hsp_vals[np.isfinite(hsp_vals)]
        if len(ctrl_vals) < min_reps or len(hsp_vals) < min_reps:
            continue
        t_stat, p_two = ttest_ind(hsp_vals, ctrl_vals, equal_var=False)
        p_one = p_two / 2 if t_stat < 0 else 1.0 - p_two / 2
        records.append({"protein": row.protein, "library": row.library,
                        "variant": row.variant, "t_stat": t_stat, "p_value": p_one})
    result = pd.DataFrame(records)
    result["dependent"] = result.p_value < alpha
    return result


def dependence_gap(ann, score_col=SCORE_ABS):
    """Per library: max(0, mean WT control score - control synonymous 2.5th pct).

    The margin by which a variant's abundance must fall under HSP90i to count as
    dependent; as in ``violin_inputs`` of Fig5_and_associated_ED.ipynb.
    """
    ab = ann[(ann.assay == "abundance")
             & ann.assay_treatment.isin(BASELINE_TREATMENTS)]
    wt = (ab[ab.variant_category == "WT"].groupby("library")[score_col].mean())
    syn = (ab[ab.variant_category == "synonymous"].dropna(subset=[score_col])
           .groupby("library")[score_col]
           .agg(lambda s: np.percentile(s.to_numpy(dtype=float), 2.5)))
    return (wt - syn).clip(lower=0.0).rename("dependence_gap")


def hsp90_derived_annotations(ann):
    """One row per (protein, library, variant): the four calls above."""
    paired = pair_abundance(ann, SCORE_ABS)
    paired = assign_buffering(paired, ann, SCORE_ABS)
    paired["dependent_threshold"] = paired["hsp90i_score"] < paired["hsp90i_threshold"]
    gap = paired.library.map(dependence_gap(ann))
    assert gap.notna().all(), "a paired library has no WT or synonymous control"
    paired["dependent"] = (paired.ctrl_score - paired.hsp90i_score) >= gap
    threshold_calls = paired[KEY + ["buffering_class", "dependent",
                                    "dependent_threshold"]].drop_duplicates(KEY)
    stat_calls = classify_dependent_statistical(
        ann, proteins=HSP90I_PROTEINS)[KEY + ["dependent"]].rename(
        columns={"dependent": "dependent_significant"})
    return threshold_calls.merge(stat_calls, on=KEY, how="outer")
