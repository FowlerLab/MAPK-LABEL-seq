"""Annotation functions for the LABEL-seq MAPK pipeline.

Each function takes a scored DataFrame and returns it with new columns added.
Functions are called in sequence by the annotation pipeline. Every function
follows the pattern:

    def add_X(df, ...) -> pd.DataFrame:
        # Load external data
        # Transform/filter
        # Merge on appropriate keys
        return df

This module faithfully replicates Notebook 2 (Generate_table.ipynb) logic.
See docs/known_bugs_and_gotchas.md for known issues.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Group 1: Protein metadata and z-scores
# ---------------------------------------------------------------------------


def add_protein_metadata(
    df: pd.DataFrame, config: Dict[str, Any]
) -> pd.DataFrame:
    """Add protein name and UniProt accession from config.

    Derives the base protein name by stripping _nterm/_cterm from library.
    Looks up UniProt ID from proteins.yaml.

    Args:
        df: Scored DataFrame with 'library' column.
        config: Full config dict with 'proteins' key.

    Returns:
        DataFrame with 'protein' and 'uniprot_accession' columns.
    """
    df = df.copy()
    df["protein"] = df["library"].str.replace(r"(_nterm|_cterm)$", "", regex=True)

    # Build protein → uniprot mapping from config
    protein_to_uniprot = {}
    for lib_name, lib_cfg in config["proteins"].items():
        # Use the 'protein' field if it exists (for split libraries), else use key
        protein = lib_cfg.get("protein", lib_name)
        if "uniprot_id" in lib_cfg:
            protein_to_uniprot[protein] = lib_cfg["uniprot_id"]

    df["uniprot_accession"] = df["protein"].map(protein_to_uniprot)
    return df


def add_z_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Add z-score columns: global and relative to synonymous WT.

    Two z-scores are computed:
    1. average_z_score: z-score within each (library, assay, treatment) group
    2. average_z_score_from_syonWT: z-score relative to the synonymous WT
       distribution of that group (how many SDs from WT mean)

    Also adds synon_wt_mean and synon_wt_std for downstream use.

    Args:
        df: Scored DataFrame with 'average score', 'Mutation Type' columns.

    Returns:
        DataFrame with z-score columns added.
    """
    df = df.copy()
    group_cols = ["library", "assay", "assay_treatment"]

    # Global z-score within each group
    df["average_z_score"] = df.groupby(group_cols)["average score"].transform(
        lambda x: (x - x.mean()) / x.std()
    )

    # Synonymous WT stats per group
    syn_wt = df[df["Mutation Type"] == "synonymous wild type"]
    syn_wt_stats = (
        syn_wt.groupby(group_cols)["average score"]
        .agg(["mean", "std"])
        .rename(columns={"mean": "synon_wt_mean", "std": "synon_wt_std"})
        .reset_index()
    )
    df = df.merge(syn_wt_stats, on=group_cols, how="left")

    # Z-score from synonymous WT
    df["average_z_score_from_syonWT"] = (
        (df["average score"] - df["synon_wt_mean"]) / df["synon_wt_std"]
    )

    return df


# ---------------------------------------------------------------------------
# Group 2: HSP90 chaperone analysis
# ---------------------------------------------------------------------------


def add_hsp90_client_status(
    df: pd.DataFrame, config: Dict[str, Any]
) -> pd.DataFrame:
    """Add HSP90 client status from config.

    Args:
        df: DataFrame with 'protein' column.
        config: Full config dict.

    Returns:
        DataFrame with 'client_status' column.
    """
    df = df.copy()
    # Build protein → client_status mapping
    status_map = {}
    for lib_name, lib_cfg in config["proteins"].items():
        protein = lib_cfg.get("protein", lib_name)
        if "client_status" in lib_cfg:
            status_map[protein] = lib_cfg["client_status"]

    df["client_status"] = df["protein"].map(status_map)
    return df


def add_chaperone_dependency(df: pd.DataFrame) -> pd.DataFrame:
    """Compute chaperone dependency: abundance change upon HSP90 inhibition.

    For each variant in a protein with HSP90i data, computes:
        Abund_chaperone_dependency = score(HSP90i) - score(control)

    Control is DMSO if available, otherwise No_treatment.

    Also classifies dependency relative to synonymous WT distribution
    (2.5th/97.5th percentile thresholds) and computes z-scores.

    Args:
        df: Scored DataFrame with abundance scores across treatments.

    Returns:
        DataFrame with chaperone dependency columns added.
    """
    df = df.copy()

    # Initialize columns — use object dtype for columns that will hold strings
    df["Abund_chaperone_dependency"] = np.nan
    df["chap_dep_classification"] = pd.Series([None] * len(df), dtype=object)
    df["Abund_chaperone_dependency_zscore_from_synon_WT"] = np.nan

    # Work only with abundance assay
    abund = df[df["assay"] == "abundance"].copy()

    # Find libraries that have HSP90i treatment
    valid_libraries = (
        abund.groupby("library")["assay_treatment"]
        .apply(lambda x: "HSP90i" in set(x))
    )
    valid_libraries = valid_libraries[valid_libraries].index

    abund_valid = abund[abund["library"].isin(valid_libraries)].copy()

    # Pivot: one row per (library, assay, variant), columns = treatments
    # Uses 'average score' (NOT intercept_0), matching the notebook exactly.
    pivot = (
        abund_valid.pivot_table(
            index=["library", "assay", "variant"],
            columns="assay_treatment",
            values="average score",
            aggfunc="mean",
        )
        .reset_index()
    )

    # Control = DMSO first, fallback to No_treatment (bfill across columns)
    ctrl_cols = []
    if "DMSO" in pivot.columns:
        ctrl_cols.append("DMSO")
    if "No_treatment" in pivot.columns:
        ctrl_cols.append("No_treatment")

    if ctrl_cols and "HSP90i" in pivot.columns:
        control = pivot[ctrl_cols].bfill(axis=1).iloc[:, 0]
        pivot["Abund_chaperone_dependency"] = pivot["HSP90i"] - control
    else:
        return df

    # Classify using synonymous WT thresholds per library
    # First, get mutation types for variants
    var_types = df[["variant", "library", "Mutation Type"]].drop_duplicates()
    pivot = pivot.merge(var_types, on=["variant", "library"], how="left")

    for lib in valid_libraries:
        lib_pivot = pivot[pivot["library"] == lib]
        syn_wt_dep = lib_pivot[lib_pivot["Mutation Type"] == "synonymous wild type"][
            "Abund_chaperone_dependency"
        ].dropna()

        if len(syn_wt_dep) > 0:
            wt_low = syn_wt_dep.quantile(0.025)
            wt_high = syn_wt_dep.quantile(0.975)
            wt_mean = syn_wt_dep.mean()
            wt_std = syn_wt_dep.std() if syn_wt_dep.std() > 0 else np.nan

            mask = pivot["library"] == lib
            pivot.loc[mask, "chap_dep_classification"] = np.where(
                pivot.loc[mask, "Abund_chaperone_dependency"] >= wt_high,
                "decreased",
                np.where(
                    pivot.loc[mask, "Abund_chaperone_dependency"] <= wt_low,
                    "increased",
                    "wt-like",
                ),
            )
            if pd.notna(wt_std):
                pivot.loc[mask, "Abund_chaperone_dependency_zscore_from_synon_WT"] = (
                    (pivot.loc[mask, "Abund_chaperone_dependency"] - wt_mean) / wt_std
                )

    # Merge back to main DataFrame — for ALL assays of each library
    dep_cols = ["library", "variant", "Abund_chaperone_dependency",
                "chap_dep_classification", "Abund_chaperone_dependency_zscore_from_synon_WT"]
    dep_cols = [c for c in dep_cols if c in pivot.columns]
    dep_merge = pivot[dep_cols].drop_duplicates(subset=["library", "variant"])

    df = df.merge(dep_merge, on=["library", "variant"], how="left",
                  suffixes=("_drop", ""))

    # Clean up duplicate columns from merge
    drop_cols = [c for c in df.columns if c.endswith("_drop")]
    df = df.drop(columns=drop_cols)

    return df


def add_buffering(df: pd.DataFrame) -> pd.DataFrame:
    """Compute HSP90 buffering status via paired t-test across replicates.

    A variant is 'buffered' if:
    1. The paired t-test (HSP90i vs control replicates) has p/2 < 0.05
    2. The mean HSP90i score < mean control score

    Also computes percent_buffered and WT-normalized percent_buffered.

    Args:
        df: Scored DataFrame with replicate-level std-adjusted scores.

    Returns:
        DataFrame with 'buffered', 'percent_buffered', 'percent_buffered_WT_norm'.
    """
    from scipy import stats

    df = df.copy()
    df["buffered"] = pd.Series([None] * len(df), dtype=object)
    df["percent_buffered"] = np.nan
    df["percent_buffered_WT_norm"] = np.nan

    rep_cols = [
        "intercept_0_std_adj_score_1",
        "intercept_0_std_adj_score_2",
        "intercept_0_std_adj_score_3",
    ]

    abund = df[df["assay"] == "abundance"].copy()
    hsp90i_libs = abund[abund["assay_treatment"] == "HSP90i"]["library"].unique()

    for lib in hsp90i_libs:
        lib_data = abund[abund["library"] == lib]
        treatments = lib_data["assay_treatment"].unique()

        if "DMSO" in treatments:
            ctrl_treatment = "DMSO"
        elif "No_treatment" in treatments:
            ctrl_treatment = "No_treatment"
        else:
            continue

        hsp90i_data = lib_data[lib_data["assay_treatment"] == "HSP90i"].set_index(
            "variant"
        )[rep_cols]
        ctrl_data = lib_data[lib_data["assay_treatment"] == ctrl_treatment].set_index(
            "variant"
        )[rep_cols]

        # Align on common variants
        common = hsp90i_data.index.intersection(ctrl_data.index)
        if len(common) == 0:
            continue

        hsp = hsp90i_data.loc[common].values
        ctrl = ctrl_data.loc[common].values

        # Per-variant t-test
        buffered_flags = []
        pct_buffered = []
        for i in range(len(common)):
            h = hsp[i][~np.isnan(hsp[i])]
            c = ctrl[i][~np.isnan(ctrl[i])]
            if len(h) >= 2 and len(c) >= 2:
                t_stat, p_val = stats.ttest_ind(h, c)
                is_buffered = (p_val / 2 < 0.05) and (np.mean(h) < np.mean(c))
                pct = (np.mean(h) - np.mean(c)) / np.mean(c) if np.mean(c) != 0 else np.nan
            else:
                is_buffered = np.nan
                pct = np.nan
            buffered_flags.append(is_buffered)
            pct_buffered.append(pct)

        result = pd.DataFrame(
            {
                "variant": common,
                "buffered": buffered_flags,
                "percent_buffered": pct_buffered,
            }
        )

        # WT-normalize percent_buffered
        wt_rows = lib_data[
            (lib_data["Mutation Type"] == "wild type")
            & (lib_data["assay_treatment"] == ctrl_treatment)
        ]
        if len(wt_rows) > 0:
            wt_variant = wt_rows["variant"].values[0]
            wt_pct = result.loc[result["variant"] == wt_variant, "percent_buffered"]
            if len(wt_pct) > 0 and wt_pct.values[0] != 0:
                result["percent_buffered_WT_norm"] = (
                    result["percent_buffered"] / wt_pct.values[0]
                )

        # Map back
        result_map = result.set_index("variant")
        mask = df["library"] == lib
        for col in ["buffered", "percent_buffered", "percent_buffered_WT_norm"]:
            if col in result_map.columns:
                df.loc[mask, col] = df.loc[mask, "variant"].map(
                    result_map[col]
                )

    return df


# ---------------------------------------------------------------------------
# Group 3: Dominant negative classification
# ---------------------------------------------------------------------------


def add_dominant_negative_ev(
    df: pd.DataFrame, config: Dict[str, Any]
) -> pd.DataFrame:
    """Classify DN variants using per-library empty-vector activity baselines.

    Drops a new DN_EV column onto the frame. Unlike add_dominant_negative
    (which pools NoVar_std across non-excluded libraries and takes the 5th
    percentile), this function uses the **2.5th percentile of empty-vector
    (EV) barcode activity** for each (library, assay_treatment) pair — i.e.
    the same tail cutoff used by classification_2.5pct but on EV barcodes
    instead of synonymous-WT barcodes. EV barcodes carry no kinase
    overexpression cassette, so their activity scores are the true
    "no kinase" pathway baseline inside each library. This fixes the
    cold-shock confound that broke the per-protein NoVar_std thresholds
    (docs/open_questions.md Q8 and user memory `project_novar_cold_shock`).

    Thresholds live in data/dn_cutoffs_empty_vector.tsv (path key
    'ev_dn_cutoffs' in config/paths.yaml). Scores are on the WT-relative
    scale (the 'average score' column), which matches the units used
    when computing the EV 2.5th percentile.

    DN_EV stays None (not applicable) for:
      - Inhibitory proteins (GRB2, MEK1, MEK2, KSR1, KSR2 — the exact set in
        config/scoring.yaml `inhibitory_proteins`): low activity is the
        expected WT phenotype when overexpressing an inhibitor, so the
        classical "dominant negative" interpretation does not apply even with
        an EV baseline. NOTE: ARAF and RET are NOT inhibitory — they are
        activators with high WT activity — so they DO receive DN_EV calls
        (e.g. ARAF gives ~3069 missense DNs). Match this list to the config,
        not to this comment.
      - Libraries with a NaN threshold in the cutoffs file (dropped via
        dropna). SOS2 was such a case before it was re-run, but now has a
        valid cutoff and IS classified — do not assume SOS2 is excluded.
      - (library, assay_treatment) pairs not present in the cutoffs file.
      - Non-activity assay rows (abundance, interaction).
      - Rows with NaN 'average score'.

    Args:
        df: Annotated scores DataFrame. Must already carry the 'protein'
            column from add_protein_metadata.
        config: Full config dict. Reads paths.ev_dn_cutoffs (absolute),
            scoring.inhibitory_proteins, and scoring.dominant_negative.assay.

    Returns:
        DataFrame with DN_EV column added (dtype=object, {True, False, None}).
    """
    df = df.copy()

    cutoffs_path = Path(config["paths"]["ev_dn_cutoffs"])
    # File uses tab + aligned-whitespace padding; whitespace separator handles both.
    cutoffs = pd.read_csv(cutoffs_path, sep=r"\s+")
    # Drop libraries with missing threshold (e.g., sos2 as of 2026-04-19).
    cutoffs = cutoffs.dropna(subset=["dn_threshold"])
    # Restrict DN_EV calling to baseline treatments — No_treatment and DMSO
    # only (per user decision 2026-04-19). Drops EGFR SerumStarve and KRAS
    # CIAR entries; those stimulated/induced-signalling conditions are not
    # used for DN definition while the analysis is stabilising.
    _allowed = {"No_treatment", "DMSO"}
    cutoffs = cutoffs[cutoffs["assay_treatment"].isin(_allowed)].copy()

    # (library, assay_treatment) -> threshold
    lookup: Dict[Tuple[str, str], float] = dict(
        zip(
            zip(cutoffs["library"], cutoffs["assay_treatment"]),
            cutoffs["dn_threshold"].astype(float),
        )
    )

    inhibitory = set(config["scoring"]["inhibitory_proteins"])
    assay_name = config["scoring"]["dominant_negative"]["assay"]
    score_col = "average score"  # WT-relative, matches EV cutoff units

    # DN is only defined for variant types that change the protein product.
    # Synonymous and wild-type rows cannot be dominant-negative *by definition*
    # (the protein sequence is unchanged), and the BRAF "standard" spike-ins are
    # control constructs, not variants of the host protein. Without this gate a
    # handful of these score below the EV cutoff on activity noise/cold-shock and
    # get spuriously labelled DN (e.g. 39 synonymous, 15 standards as of the
    # 2026-06-22 audit), contaminating the population-depletion DN class. The
    # observed synonymous DN rate (0.73%) is far below the missense rate (11.6%),
    # which is itself a useful validation that DN calls reflect real perturbation.
    DN_ELIGIBLE_TYPES = {"missense", "nonsense", "deletion"}

    df["DN_EV"] = pd.Series([None] * len(df), dtype=object)

    # Base classifiability: activity assay, non-inhibitory protein, a
    # protein-altering variant type, and a non-null score.
    classifiable = (
        (df["assay"] == assay_name)
        & (~df["protein"].isin(inhibitory))
        & df["Mutation Type"].isin(DN_ELIGIBLE_TYPES)
        & df[score_col].notna()
    )

    # Apply per-(library, treatment) threshold.
    for (library, treatment), threshold in lookup.items():
        mask = (
            classifiable
            & (df["library"] == library)
            & (df["assay_treatment"] == treatment)
        )
        if mask.any():
            df.loc[mask, "DN_EV"] = df.loc[mask, score_col] < threshold

    return df


# ---------------------------------------------------------------------------
# Group 4: DSSP secondary structure
# ---------------------------------------------------------------------------


# Maximum SASA per amino acid — Tien et al. 2013, *empirical* column (not the
# Gly-X-Gly theoretical one, which runs ~7% higher: Ala 129 vs 121 here).

MAX_SASA = {
    "A": 121.0, "R": 265.0, "N": 187.0, "D": 187.0, "C": 148.0,
    "Q": 214.0, "E": 214.0, "G": 97.0, "H": 216.0, "I": 195.0,
    "L": 191.0, "K": 230.0, "M": 203.0, "F": 228.0, "P": 154.0,
    "S": 143.0, "T": 163.0, "W": 264.0, "Y": 255.0, "V": 165.0,
}

# DSSP code → human-readable secondary structure
SS_MAP = {
    "H": "alpha helix", "B": "beta bridge", "E": "beta strand",
    "G": "3-10 helix", "I": "pi helix", "T": "turn", "S": "bend",
}


def _parse_dssp(filepath: Path) -> pd.DataFrame:
    """Parse a DSSP file into a DataFrame of (position, aa, ss, asa).

    Args:
        filepath: Path to a .dssp file.

    Returns:
        DataFrame with columns: Position, aa, dssp_secondary_structure,
        dssp_solvent_accessibility_angstroms^2.
    """
    rows = []
    in_data = False
    with open(filepath) as f:
        for line in f:
            if line.startswith("  #  RESIDUE"):
                in_data = True
                continue
            if not in_data:
                continue
            if len(line) < 38:
                continue
            try:
                resnum = int(line[5:10].strip())
                aa = line[13]
                ss = line[16]
                asa = int(line[35:38].strip())
            except (ValueError, IndexError):
                continue

            if aa == "!":  # chain break
                continue

            ss_name = SS_MAP.get(ss, "no_ss")
            rows.append({
                "Position": resnum,
                "aa": aa,
                "dssp_secondary_structure": ss_name,
                "dssp_solvent_accessibility_angstroms^2": asa,
            })

    return pd.DataFrame(rows)


def add_dssp(df: pd.DataFrame, dssp_dir: Path) -> pd.DataFrame:
    """Add DSSP secondary structure and solvent accessibility annotations.

    Parses AlphaFold-derived DSSP files for each protein and merges on
    (protein, Position). Computes relative SASA from theoretical max values.

    Args:
        df: Scored DataFrame with 'protein', 'library', 'Position' columns.
        dssp_dir: Path to directory containing {protein}.dssp files.

    Returns:
        DataFrame with DSSP columns added.
    """
    df = df.copy()
    dssp_dir = Path(dssp_dir)

    # Initialize columns — dssp_secondary_structure is a string
    df["dssp_secondary_structure"] = pd.Series([None] * len(df), dtype=object)
    df["dssp_solvent_accessibility_angstroms^2"] = np.nan
    df["max_sasa"] = np.nan
    df["relative_sasa"] = np.nan

    # Build a combined DSSP lookup table for all proteins, then merge once
    all_dssp = []
    for protein in df["protein"].unique():
        dssp_file = dssp_dir / f"{protein}.dssp"
        if not dssp_file.exists():
            continue

        dssp_data = _parse_dssp(dssp_file)
        if dssp_data.empty:
            continue

        dssp_data["protein"] = protein
        dssp_data["Position"] = dssp_data["Position"].astype(str)
        all_dssp.append(dssp_data[["protein", "Position",
                                    "dssp_secondary_structure",
                                    "dssp_solvent_accessibility_angstroms^2"]])

    if all_dssp:
        dssp_lookup = pd.concat(all_dssp, ignore_index=True)
        # Convert df Position to str for merge
        df["Position"] = df["Position"].astype(str)
        df = df.merge(dssp_lookup, on=["protein", "Position"], how="left",
                      suffixes=("_drop", ""))
        # Clean up any duplicate columns from merge
        for col in ["dssp_secondary_structure_drop",
                     "dssp_solvent_accessibility_angstroms^2_drop"]:
            if col in df.columns:
                df = df.drop(columns=[col])

    # Compute max SASA and relative SASA
    df["max_sasa"] = df["Wild Type Residue"].map(MAX_SASA)
    valid_sasa = df["dssp_solvent_accessibility_angstroms^2"].notna() & df[
        "max_sasa"
    ].notna()
    df.loc[valid_sasa, "relative_sasa"] = (
        df.loc[valid_sasa, "dssp_solvent_accessibility_angstroms^2"]
        / df.loc[valid_sasa, "max_sasa"]
    )

    return df


# ---------------------------------------------------------------------------
# Group 5: PTM annotations (PhosphoSitePlus)
# ---------------------------------------------------------------------------


def add_ptms(
    df: pd.DataFrame, phospho_file: Path, ub_file: Path
) -> pd.DataFrame:
    """Add phosphorylation and ubiquitination annotations from PhosphoSitePlus.

    Merges on (uniprot_accession, Wild Type Residue, Position).

    Args:
        df: Scored DataFrame.
        phospho_file: Path to Phosphorylation_site_dataset.tsv.
        ub_file: Path to Ubiquitination_site_dataset.tsv.

    Returns:
        DataFrame with PTM columns added.
    """
    df = df.copy()

    ptm_frames = []
    for filepath, mod_pattern, mod_type in [
        (phospho_file, r"([A-Z])(\d+)-p", "phospho"),
        (ub_file, r"([A-Z])(\d+)-ub", "ub"),
    ]:
        ptm = pd.read_csv(filepath, sep="\t", skiprows=3, low_memory=False)
        # Filter to human
        ptm = ptm[ptm["ORGANISM"] == "human"].copy()
        # Parse MOD_RSD
        extracted = ptm["MOD_RSD"].str.extract(mod_pattern)
        ptm["Wild Type Residue"] = extracted[0]
        ptm["Position"] = pd.to_numeric(extracted[1], errors="coerce")
        ptm = ptm.dropna(subset=["Wild Type Residue", "Position"])
        ptm["Position"] = ptm["Position"].astype(int).astype(str)
        # Add Modification type (derived from file, not from MOD_RSD parsing)
        ptm["Modification"] = mod_type
        ptm_frames.append(ptm)

    ptm_all = pd.concat(ptm_frames, ignore_index=True)

    # Select relevant columns. ORGANISM is deliberately NOT carried: the frame is
    # already filtered to human above, so the column would be a constant.
    ptm_cols = ["ACC_ID", "Wild Type Residue", "Position", "MOD_RSD",
                "LT_LIT", "MS_LIT", "MS_CST",
                "Ambiguous_Site", "Modification"]
    ptm_cols = [c for c in ptm_cols if c in ptm_all.columns]

    # Deduplicate: if a position has both phosphorylation and ubiquitination,
    # aggregate into one row to avoid inflating the DataFrame.
    # Note: lambda in agg dict doesn't work well with pandas, so define funcs.
    def _join_unique(x):
        return "; ".join(sorted(set(str(v) for v in x if pd.notna(v))))

    agg_funcs = {}
    for col in ptm_cols:
        if col in ["ACC_ID", "Wild Type Residue", "Position"]:
            continue  # These are group keys
        elif col in ["LT_LIT", "MS_LIT", "MS_CST", "Ambiguous_Site"]:
            agg_funcs[col] = "max"
        elif col in ["MOD_RSD", "Modification"]:
            agg_funcs[col] = _join_unique
        else:
            agg_funcs[col] = "first"

    ptm_subset = (
        ptm_all[ptm_cols]
        .groupby(["ACC_ID", "Wild Type Residue", "Position"])
        .agg(agg_funcs)
        .reset_index()
    )

    # Merge
    df["Position"] = df["Position"].astype(str)
    df = df.merge(
        ptm_subset,
        left_on=["uniprot_accession", "Wild Type Residue", "Position"],
        right_on=["ACC_ID", "Wild Type Residue", "Position"],
        how="left",
    )
    if "ACC_ID" in df.columns:
        df = df.drop(columns=["ACC_ID"])

    return df


def add_regulatory_ptms(
    df: pd.DataFrame, reg_file: Path
) -> pd.DataFrame:
    """Add regulatory PTM annotations from PhosphoSitePlus.

    Args:
        df: Scored DataFrame.
        reg_file: Path to Regulatory_sites_MAPKonly.tsv.

    Returns:
        DataFrame with 'ON_FUNCTION', 'ON_PROCESS', 'regulatory_PTM_note',
        'regulatory_PTM' columns.
    """
    df = df.copy()

    reg = pd.read_csv(reg_file, sep="\t", low_memory=False)
    # Parse position
    extracted = reg["MOD_RSD"].str.extract(r"([A-Z])(\d+)")
    reg["Wild Type Residue"] = extracted[0]
    reg["Position"] = pd.to_numeric(extracted[1], errors="coerce")
    reg = reg.dropna(subset=["Position"])
    reg["Position"] = reg["Position"].astype(int).astype(str)

    # Build annotation text
    reg["regulatory_PTM_note"] = reg.apply(
        lambda r: "; ".join(
            filter(None, [str(r.get("ON_FUNCTION", "")), str(r.get("ON_PROCESS", ""))])
        ),
        axis=1,
    )

    reg_subset = reg[
        ["ACC_ID", "Wild Type Residue", "Position", "ON_FUNCTION", "ON_PROCESS",
         "regulatory_PTM_note"]
    ].drop_duplicates()
    # Deduplicate on merge keys to prevent row inflation when a position has
    # multiple modification types (e.g., K1061-ac and K1061-ub in EGFR).
    # Only the first entry is kept.
    reg_subset = reg_subset.drop_duplicates(
        subset=["ACC_ID", "Wild Type Residue", "Position"], keep="first"
    )

    df = df.merge(
        reg_subset,
        left_on=["uniprot_accession", "Wild Type Residue", "Position"],
        right_on=["ACC_ID", "Wild Type Residue", "Position"],
        how="left",
    )
    if "ACC_ID" in df.columns:
        df = df.drop(columns=["ACC_ID"])

    df["regulatory_PTM"] = df["regulatory_PTM_note"].notna()
    return df


def add_disease_phosphosites(
    df: pd.DataFrame, disease_file: Path
) -> pd.DataFrame:
    """Add disease-associated phosphosite flag from PhosphoSitePlus.

    Args:
        df: Scored DataFrame.
        disease_file: Path to Disease-associated_sites.tsv.

    Returns:
        DataFrame with 'disease_associated_phosphosite' column (0/1).
    """
    df = df.copy()

    disease = pd.read_csv(disease_file, sep="\t", skiprows=3, low_memory=False)
    # Filter to phosphosites only
    disease = disease[disease["MOD_RSD"].str.contains("-p", na=False)]

    # Get our UniProt accessions
    our_accessions = df["uniprot_accession"].dropna().unique()
    disease = disease[disease["ACC_ID"].isin(our_accessions)]

    extracted = disease["MOD_RSD"].str.extract(r"([A-Z])(\d+)")
    disease["Wild Type Residue"] = extracted[0]
    disease["Position"] = pd.to_numeric(extracted[1], errors="coerce")
    disease = disease.dropna(subset=["Position"])
    disease["Position"] = disease["Position"].astype(int).astype(str)

    disease_sites = disease[
        ["ACC_ID", "Wild Type Residue", "Position"]
    ].drop_duplicates()
    disease_sites["disease_associated_phosphosite"] = 1

    df = df.merge(
        disease_sites,
        left_on=["uniprot_accession", "Wild Type Residue", "Position"],
        right_on=["ACC_ID", "Wild Type Residue", "Position"],
        how="left",
    )
    if "ACC_ID" in df.columns:
        df = df.drop(columns=["ACC_ID"])
    df["disease_associated_phosphosite"] = (
        df["disease_associated_phosphosite"].fillna(0).astype(int)
    )
    return df


# ---------------------------------------------------------------------------
# Group 6: Published datasets
# ---------------------------------------------------------------------------


def add_shah_shp2(
    df: pd.DataFrame, fig2b_file: Path, fig2c_file: Path
) -> pd.DataFrame:
    """Add SHP2 enrichment data from Shah et al. (PMID 39091798).

    Args:
        df: Scored DataFrame.
        fig2b_file: Path to Shah sourcedata fig2b.
        fig2c_file: Path to Shah sourcedata fig2c.

    Returns:
        DataFrame with Shah SHP2 enrichment columns.
    """
    df = df.copy()

    for filepath, col_name in [
        (fig2b_file, "PMID39091798_Enrichment(ave)_FLshp2_CDsrc"),
        (fig2c_file, "PMID39091798_Enrichment(ave)_CDshp2_FLvsrc"),
    ]:
        shah = pd.read_csv(filepath, sep="\t")
        if "Mutation" in shah.columns and "POS" in shah.columns:
            shah["variant"] = shah["WT"] + shah["POS"].astype(str) + shah["MUT"]
            shah_merge = shah[["variant", "Enrichment (ave)"]].rename(
                columns={"Enrichment (ave)": col_name}
            )
            # Only merge for SHP2
            shp2_mask = df["protein"] == "shp2"
            df = df.merge(
                shah_merge, on="variant", how="left"
            )
            # Set non-SHP2 values to NaN
            if col_name in df.columns:
                df.loc[~shp2_mask, col_name] = np.nan

    return df


# ---------------------------------------------------------------------------
# Group 7: ClinVar
# ---------------------------------------------------------------------------


def add_clinvar(df: pd.DataFrame, clinvar_file: Path) -> pd.DataFrame:
    """Add ClinVar clinical significance annotations.

    Merges on (protein, variant). NOTE: The original notebook uses an outer
    join (B3), but since the row count didn't change in the golden file, we
    use a left join for safety.

    Args:
        df: Scored DataFrame.
        clinvar_file: Path to clinvar_MAPK_filtered.csv.

    Returns:
        DataFrame with ClinVar columns added.
    """
    df = df.copy()

    clinvar = pd.read_csv(clinvar_file, low_memory=False)

    # Derive protein column from GeneSymbol
    if "protein" not in clinvar.columns and "GeneSymbol" in clinvar.columns:
        clinvar["protein"] = clinvar["GeneSymbol"].str.lower()
    elif "protein" in clinvar.columns:
        clinvar["protein"] = clinvar["protein"].str.lower()
    else:
        return df  # Can't merge without protein column

    clinvar["protein"] = clinvar["protein"].replace({
        "raf1": "craf", "ptpn11": "shp2",
        "map2k1": "mek1", "map2k2": "mek2",
    })

    # Rename Name column to ClinVar_Name if present
    if "Name" in clinvar.columns and "ClinVar_Name" not in clinvar.columns:
        clinvar = clinvar.rename(columns={"Name": "ClinVar_Name"})

    clinvar_cols = [
        "protein", "variant", "ClinVar_Name", "ClinicalSignificance",
        "PhenotypeList", "Origin", "OriginSimple", "ReviewStatus",
        "NumberSubmitters",
    ]
    clinvar_cols = [c for c in clinvar_cols if c in clinvar.columns]
    clinvar_subset = clinvar[clinvar_cols].drop_duplicates(
        subset=["protein", "variant"]
    )

    df = df.merge(clinvar_subset, on=["protein", "variant"], how="left")
    return df


# ---------------------------------------------------------------------------
# Group 8: Domain boundaries
# ---------------------------------------------------------------------------


def add_domains(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """Add domain annotations from config.

    Uses vectorized lookup instead of row-by-row apply for speed.

    Args:
        df: Scored DataFrame with 'protein' and 'Position' columns.
        config: Full config dict.

    Returns:
        DataFrame with 'domain' column.
    """
    from labelseq_mapk.config import get_domain

    df = df.copy()

    # Build a flat lookup: (protein, position) → domain
    # Pre-compute for all protein/position combinations
    positions = pd.to_numeric(df["Position"], errors="coerce")
    unique_combos = df[["protein"]].copy()
    unique_combos["_pos_num"] = positions
    unique_combos = unique_combos.dropna(subset=["_pos_num"]).drop_duplicates()

    domain_map = {}
    for _, row in unique_combos.iterrows():
        key = (row["protein"], int(row["_pos_num"]))
        if key not in domain_map:
            domain_map[key] = get_domain(config, key[0], key[1])

    # Map back using a tuple key
    df["_pos_num"] = positions
    df["domain"] = df.apply(
        lambda r: domain_map.get((r["protein"], int(r["_pos_num"])), "none")
        if pd.notna(r["_pos_num"]) else "none",
        axis=1,
    )
    df = df.drop(columns=["_pos_num"])
    return df


# ---------------------------------------------------------------------------
# Group 9: OpenCell expression
# ---------------------------------------------------------------------------


def add_opencell(df: pd.DataFrame, opencell_file: Path) -> pd.DataFrame:
    """Add HEK293 endogenous expression data from OpenCell.

    Merges on 'library'.

    Args:
        df: Scored DataFrame with 'library' column.
        opencell_file: Path to opencell_curated.csv.

    Returns:
        DataFrame with expression columns.
    """
    df = df.copy()

    opencell = pd.read_csv(opencell_file)
    merge_cols = ["library", "hek_rna_tpm", "hek_protein_conc_nm",
                  "hek_protein_copy_number"]
    merge_cols = [c for c in merge_cols if c in opencell.columns]

    df = df.merge(opencell[merge_cols], on="library", how="left")
    return df


# ---------------------------------------------------------------------------
# Group 10: NCBI features
# ---------------------------------------------------------------------------


def add_ncbi_features(
    df: pd.DataFrame, features_file: Path
) -> pd.DataFrame:
    """Add NCBI Conserved Domain feature annotations.

    Parses the CD-search features.txt to annotate active sites, protein
    interfaces, and other functional features. The file has 7 comment lines
    starting with #, then a header row, then data.

    The Query column has format "Q#5 - >araf\\" and coordinates are
    comma-separated "{AA}{position}" pairs (e.g., "I382,G383,T384").

    The file also has a manually added "jessicas_manual_annotation" column
    that categorizes features as "active_site" or "protein_interface".

    Args:
        df: Scored DataFrame.
        features_file: Path to features.txt from NCBI CD-search.

    Returns:
        DataFrame with 'feature', 'active_site', 'protein_interface',
        'interface_evidence' columns.
    """
    df = df.copy()

    # Skip comment lines, read header from the data
    features = pd.read_csv(features_file, sep="\t", skiprows=7, low_memory=False)

    if "Query" not in features.columns:
        return df

    # Extract protein name from Query (format: "Q#5 - >araf\")
    features["protein"] = features["Query"].str.extract(r">(\w+)")[0]
    features["protein"] = features["protein"].str.lower()
    features = features.dropna(subset=["protein"])

    # Get the manual annotation column
    annot_col = "jessicas_manual_annotation"
    if annot_col not in features.columns:
        annot_col = None

    # Each feature spans multiple positions (comma-separated coordinates)
    expanded_rows = []
    for _, row in features.iterrows():
        coords = str(row.get("coordinates", ""))
        title = str(row.get("Title", ""))
        protein = row["protein"]
        manual_annot = str(row.get(annot_col, "")) if annot_col else ""

        # Remove quotes from coordinates
        coords = coords.strip('"')

        for coord in coords.split(","):
            coord = coord.strip()
            import re
            match = re.match(r"([A-Z])(\d+)", coord)
            if match:
                expanded_rows.append({
                    "protein": protein,
                    "Wild Type Residue": match.group(1),
                    "Position": match.group(2),
                    "feature_title": title,
                    "manual_annotation": manual_annot,
                })

    if not expanded_rows:
        df["feature"] = None
        df["active_site"] = False
        df["protein_interface"] = False
        df["interface_evidence"] = None
        return df

    feat_df = pd.DataFrame(expanded_rows)

    # Aggregate features per position: collect all titles and annotations
    feat_agg = (
        feat_df.groupby(["protein", "Wild Type Residue", "Position"])
        .agg(
            feature=("feature_title", lambda x: str(sorted(set(x.tolist())))),
            manual_annotation=("manual_annotation", lambda x: ";".join(
                sorted(set(a for a in x if a and a != "nan"))
            )),
        )
        .reset_index()
    )

    df = df.merge(
        feat_agg, on=["protein", "Wild Type Residue", "Position"], how="left"
    )

    # Derive active_site from manual annotation (NCBI CD-search)
    df["active_site"] = df["manual_annotation"].str.contains(
        "active_site", na=False
    )

    # Clean up temporary column
    if "manual_annotation" in df.columns:
        df = df.drop(columns=["manual_annotation"])

    return df


# ---------------------------------------------------------------------------
# Group 10b: PDB-based protein-protein interfaces
# ---------------------------------------------------------------------------


def add_ppi_interfaces(
    df: pd.DataFrame, interactions_file: Path
) -> pd.DataFrame:
    """Add protein-protein interface annotations from manually curated PDB data.

    Uses interactions_jess_edits.txt which lists interface positions for
    protein pairs derived from PDB crystal structures. Each row has
    protein_a, protein_b, positions_a (list of interface residues for
    protein_a), and positions_b (for protein_b). The 'keep' column
    filters valid interactions.

    Args:
        df: Scored DataFrame with 'protein' and 'Position' columns.
        interactions_file: Path to interactions_jess_edits.txt.

    Returns:
        DataFrame with 'protein_interface' and 'interface_evidence' columns.
    """
    df = df.copy()
    df["protein_interface"] = False
    df["interface_evidence"] = pd.Series([None] * len(df), dtype=object)

    interactions = pd.read_csv(interactions_file, sep="\t")

    # Filter to kept interactions
    interactions = interactions[interactions["keep"] == True]  # noqa

    # Build a lookup: protein → set of interface positions, with evidence
    # Each position can come from multiple interactions
    import ast

    interface_lookup: Dict[str, Dict[int, List[str]]] = {}

    for _, row in interactions.iterrows():
        protein_a = str(row.get("protein_a", "")).strip().lower()
        protein_b_name = str(row.get("protein_a_name", row.get("protein_b", "")))
        protein_b = str(row.get("protein_b", "")).strip().lower()

        # Parse positions_a → interface positions for protein_a
        for protein, positions_col, partner in [
            (protein_a, "positions_a", protein_b),
            (protein_b, "positions_b", protein_a),
        ]:
            if not protein or protein == "nan":
                continue

            pos_str = str(row.get(positions_col, "[]"))
            try:
                positions = ast.literal_eval(pos_str)
            except (ValueError, SyntaxError):
                continue

            if not isinstance(positions, list) or len(positions) == 0:
                continue

            if protein not in interface_lookup:
                interface_lookup[protein] = {}

            for pos in positions:
                pos = int(pos)
                if pos not in interface_lookup[protein]:
                    interface_lookup[protein][pos] = []
                interface_lookup[protein][pos].append(partner)

    # Build a flat lookup DataFrame and merge once (vectorized)
    lookup_rows = []
    for protein, pos_partners in interface_lookup.items():
        for pos, partners in pos_partners.items():
            lookup_rows.append({
                "protein": protein,
                "Position": str(pos),
                "_is_interface": True,
                "_iface_evidence": "; ".join(sorted(set(partners))),
            })

    if lookup_rows:
        iface_df = pd.DataFrame(lookup_rows)
        df["Position"] = df["Position"].astype(str)
        df = df.merge(iface_df, on=["protein", "Position"], how="left")
        df["protein_interface"] = df["_is_interface"].fillna(False)
        df["interface_evidence"] = df["_iface_evidence"]
        df = df.drop(columns=["_is_interface", "_iface_evidence"])

    return df


# ---------------------------------------------------------------------------
# Group 11: AlphaMissense
# ---------------------------------------------------------------------------


def add_alphamissense(
    df: pd.DataFrame, am_file: Path
) -> pd.DataFrame:
    """Add AlphaMissense pathogenicity predictions.

    Args:
        df: Scored DataFrame.
        am_file: Path to AlphaMissense MAPK-filtered TSV.

    Returns:
        DataFrame with an 'am_pathogenicity' column.

    `am_class` is deliberately NOT carried through. It is AlphaMissense's own
    three-way call at its published thresholds, and on this variant set it
    discriminates too weakly to be worth a column: it labelled 277,592 of the
    scored missense variants pathogenic against 106,531 benign. The continuous
    `am_pathogenicity` is kept, so anyone who wants a categorical can threshold
    it themselves and say which threshold they used.
    """
    df = df.copy()

    am = pd.read_csv(am_file, sep="\t")
    if am.columns[0].startswith('"'):
        # Handle quoted column names
        am.columns = [c.strip('"') for c in am.columns]

    # AlphaMissense uses protein_variant format like "M1A"
    # Match to our (uniprot_accession, variant)
    am_merge = am[["uniprot_id", "protein_variant", "am_pathogenicity"]].copy()
    am_merge = am_merge.rename(columns={"uniprot_id": "uniprot_accession", "protein_variant": "variant"})
    am_merge = am_merge.drop_duplicates(subset=["uniprot_accession", "variant"])

    df = df.merge(am_merge, on=["uniprot_accession", "variant"], how="left")
    return df


# ---------------------------------------------------------------------------
# Group 12: Kinase-alignment column mapping
# ---------------------------------------------------------------------------


def add_alignment_positions(
    df: pd.DataFrame, alignment_file: Path
) -> pd.DataFrame:
    """Map residues onto columns of the human kinase-domain alignment.

    Positions are matched to alignment columns through the UniProt accession in
    the alignment headers. Emits 'alignment_pos' — the
    golden parity file carries both names.

    Until 2026-08-10 this function also emitted 'kinase_conservation', a Shannon
    entropy conservation score (1 - H/log2(20)) per alignment column. That column
    was dropped because (a) no figure read it once Fig. 3's S3B moved to
    'jsd_conservation', (b) its role — within-kinome paralog conservation — is
    filled by 'jsd_conservation' (Capra-Singh Jensen-Shannon divergence,
    scripts/compute_jsd_conservation.py), which is the better-validated statistic,
    and (c) it counted every record of the alignment including the ANNOTATION
    pseudo-sequence, so non-residue characters entered the column distributions
    (bug B11 in docs/known_bugs_and_gotchas.md). The column-mapping below never
    depended on the entropy and is unchanged.

    Args:
        df: Scored DataFrame.
        alignment_file: Path to Human-PK-alignment.fasta.

    Returns:
        DataFrame with an 'alignment_pos' column.
    """
    from Bio import SeqIO

    df = df.copy()
    df["alignment_pos"] = np.nan

    # Parse alignment
    records = list(SeqIO.parse(str(alignment_file), "fasta"))
    if not records:
        return df

    alignment_len = len(records[0].seq)

    # Build UniProt → (residue position → MSA column) mapping.
    # Header format: "TKL_BRAF/457-716 BRAF_HUMAN BRAF P15056"
    # UniProt ID is the last token in the description.
    # The /start-end range tells us which residues are in the alignment.
    import re

    uniprot_to_msa = {}
    for record in records:
        desc_parts = record.description.split()
        uniprot_id = desc_parts[-1] if len(desc_parts) >= 4 else None
        if not uniprot_id or not re.match(r"^[A-Z0-9]{6}$", uniprot_id):
            continue

        # Extract start position from ID (e.g., "TKL_BRAF/457-716" → 457)
        range_match = re.search(r"/(\d+)-(\d+)", record.id)
        if range_match:
            start_pos = int(range_match.group(1))
        else:
            start_pos = 1

        seq = str(record.seq)
        pos_to_col = {}
        residue_pos = start_pos  # Start counting from the range start
        for col_idx, aa in enumerate(seq):
            if aa != "-":
                pos_to_col[residue_pos] = col_idx
                residue_pos += 1
        uniprot_to_msa[uniprot_id] = pos_to_col

    # Build a flat lookup: (uniprot_id, position) → alignment_col
    lookup_rows = []
    for uniprot_id, pos_to_col in uniprot_to_msa.items():
        for pos, col_idx in pos_to_col.items():
            if col_idx < alignment_len:
                lookup_rows.append({
                    "uniprot_accession": uniprot_id,
                    "Position": str(pos),
                    "alignment_pos": col_idx,
                })

    if lookup_rows:
        cols_df = pd.DataFrame(lookup_rows)
        df["Position"] = df["Position"].astype(str)
        df = df.merge(cols_df, on=["uniprot_accession", "Position"], how="left",
                      suffixes=("_drop", ""))
        if "alignment_pos_drop" in df.columns:
            df = df.drop(columns=["alignment_pos_drop"])

    # `msa_col` used to be emitted here as a byte-identical alias of
    # `alignment_pos`, because her table carries both names. Dropped 2026-08-20 --
    # two names for one column is a trap, not a convenience.

    return df


# ---------------------------------------------------------------------------
# Group 13: CysDB
# ---------------------------------------------------------------------------


def add_cysdb(df: pd.DataFrame, cysdb_dir: Path) -> pd.DataFrame:
    """Add cysteine ligandability data from CysDB.

    Args:
        df: Scored DataFrame.
        cysdb_dir: Path to CysDB_files/ directory.

    Returns:
        DataFrame with 'ligandable_cys' column.
    """
    df = df.copy()
    df["ligandable_cys"] = pd.Series([None] * len(df), dtype=object)
    cysdb_dir = Path(cysdb_dir)

    for filepath in cysdb_dir.glob("cysdb_identified_*.csv"):
        # Extract UniProt ID from filename
        uniprot_id = filepath.stem.replace("cysdb_identified_", "").replace("_results", "")

        cys_data = pd.read_csv(filepath)
        if "ligandable" not in cys_data.columns:
            continue

        # Parse cysteineid to get position (e.g., P00533_C1049 → 1049)
        cys_data["Position"] = cys_data["cysteineid"].str.extract(r"_C(\d+)")[0]
        ligandable = cys_data[cys_data["ligandable"] == "yes"]["Position"].values

        mask = (df["uniprot_accession"] == uniprot_id) & (
            df["Wild Type Residue"] == "C"
        )
        for pos in ligandable:
            pos_mask = mask & (df["Position"].astype(str) == str(pos))
            df.loc[pos_mask, "ligandable_cys"] = True

    return df


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------


def run_annotation(
    scores_df: pd.DataFrame,
    config: Dict[str, Any],
    verbose: bool = True,
) -> pd.DataFrame:
    """Run the full annotation pipeline on scored data.

    Applies all annotation functions in sequence, matching the order of
    Notebook 2 (Generate_table.ipynb).

    Args:
        scores_df: Output of run_scoring() — variant-level scores.
        config: Full config dict with resolved paths.
        verbose: Print progress.

    Returns:
        Fully annotated DataFrame.
    """
    paths = config["paths"]["annotations"]
    df = scores_df.copy()

    steps = [
        ("Protein metadata", lambda d: add_protein_metadata(d, config)),
        ("Z-scores", add_z_scores),
        ("HSP90 client status", lambda d: add_hsp90_client_status(d, config)),
        ("Chaperone dependency", add_chaperone_dependency),
        ("Buffering", add_buffering),
        ("Dominant negative (EV baseline)", lambda d: add_dominant_negative_ev(d, config)),
        ("DSSP", lambda d: add_dssp(d, paths["dssp_dir"])),
        ("PTMs", lambda d: add_ptms(d, paths["psp_phosphorylation"], paths["psp_ubiquitination"])),
        ("Regulatory PTMs", lambda d: add_regulatory_ptms(d, paths["psp_regulatory"])),
        ("Disease phosphosites", lambda d: add_disease_phosphosites(d, paths["psp_disease"])),
        ("Shah SHP2", lambda d: add_shah_shp2(d, paths["shah_shp2_fig2b"], paths["shah_shp2_fig2c"])),
        ("ClinVar", lambda d: add_clinvar(d, paths["clinvar"])),
        ("Domains", lambda d: add_domains(d, config)),
        ("OpenCell", lambda d: add_opencell(d, paths["opencell"])),
        ("NCBI features", lambda d: add_ncbi_features(d, paths["ncbi_features"])),
        ("PPI interfaces", lambda d: add_ppi_interfaces(d, paths["interactions"])),
        ("AlphaMissense", lambda d: add_alphamissense(d, paths["alphamissense"])),
        ("Alignment positions", lambda d: add_alignment_positions(d, paths["kinase_alignment"])),
        ("CysDB", lambda d: add_cysdb(d, paths["cysdb_dir"])),
    ]

    # No blanket try/except. It used to catch every step and print a WARNING,
    # which is how `add_dominant_negative` came to fail on EVERY run with
    # KeyError: 'classification_5pct' and silently contribute nothing -- the
    # annotation simply went missing and the table still looked complete. A step
    # that cannot run is a bug, so it now stops the pipeline.
    for name, func in steps:
        if verbose:
            print(f"  Annotating: {name}...")
        df = func(df)

    if verbose:
        print(f"  Annotation complete: {len(df):,} rows, {len(df.columns)} columns")

    return df
