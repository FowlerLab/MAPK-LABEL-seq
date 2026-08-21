"""Shared analysis and plotting utilities for the LABEL-seq MAPK manuscript.

This module is the single place where the project's *conventions* live — score
columns, class definitions, thresholds, colour palettes, protein labels — so
that the figure notebooks (``Annotations.ipynb``, ``DN.ipynb``, ``HSP90i.ipynb``)
contain only the logic specific to each panel.

Design rules, all of them learned the hard way (see
``docs/hsp90_metric_definitions.md`` and ``docs/known_bugs_and_gotchas.md``):

1. **Every threshold is computed per (library, assay, assay_treatment).**
   ``classification_2.5pct`` is computed at that granularity in the scoring
   pipeline, so anything derived from a synonymous-WT percentile must match it.
   Pooling libraries would mix ARAF cterm (DMSO control) with ARAF nterm
   (No_treatment control) — see gotcha G7.

2. **Two scales, never mixed.** *Buffering* is wild-type-relative (built on
   ``average score``, which re-anchors WT to 1.0 within each condition, so the
   global WT shift cancels). *Dependence* is absolute (built on
   ``intercept_0_standard-adjusted score``, which retains it). Function names
   here always say which scale they are on.

3. **MET is displayed as "HGFR"** in every figure. Use :func:`protein_label`
   for any user-visible protein string; never print the raw key.

**Rendering, conventions, and the annotation plumbing.** Most of this module is
plot builders that take already-computed data and turn it into a figure, plus
the shared conventions (palettes, protein order and display names, score column
names, thresholds) and the style/save helpers.

Since 2026-08-14 it also carries the *input paths* and the small annotation
helpers that attach a file to a scored table — :func:`annotation_sources`,
:func:`input_manifest`, :func:`add_structure_annotations`. They live here so
that a notebook can declare and check every file it reads in one place instead
of leaving paths buried in scripts. The rule they do **not** break: no
classification and no statistics live here. Anything that decides what a variant
*is* — a threshold, a class, a test — stays in the notebook, next to the result
it produces.

Import at the top of a notebook::

    import utils
    utils.apply_style()
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib.colors as mcolors
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

__all__ = ["PROJECT_ROOT", "DN_ORDER", "HSP90_ORDER", "PROTEIN_KEYS",
    "DN_ELIGIBLE", "INHIBITORY_PROTEINS", "HSP90I_PROTEINS",
    "BASELINE_TREATMENTS", "PROTEIN_ALTERING_TYPES", "SCORE_WT_REL",
    "SCORE_ABS", "SCORE_ABS_REPS", "BUFFERING_COLORS", "BUFFERING_ORDER",
    "BUFFERING_CMAPS", "ACTIVITY_COLORS", "ACTIVITY_ORDER", "DB_COLORS",
    "VARIANT_TYPE_COLORS", "VARIANT_TYPE_ORDER", "PROTEIN_SPHERE_COLOR",
    "STRUCT_CARTOON_COLOR", "STRUCT_PARTNER_COLORS", "ACTIVITY_GROUP_ORDER",
    "ACTIVITY_GROUP_COLOR", "MIN_N_PER_PROTEIN", "ONTOLOGY_BUCKET_ORDER",
    "ONTOLOGY_BUCKET_STYLE", "ONTOLOGY_BUCKET_LABELS",
    "ONTOLOGY_LEGEND_ROWS", "apply_style", "save_figure", "protein_label",
    "protein_labels", "stars_4tier", "stars_4tier_ns",
    "fixed_pitch_bar_axes", "plot_stacked_vbar", "plot_paired_score_violins",
    "plot_density_panel", "plot_beta_heatmap", "BUFFERING_STACK_ORDER",
    "plot_forest", "plot_stacked_hbar", "plot_bubble_grid", "plot_heatmap",
    "plot_violin_panel", "plot_donut", "plot_univariate_profiles",
    "plot_dn_positions", "plot_activity_histogram", "plot_dn_summary_table",
    "plot_structure", "plot_sphere_legend", "PYMOL_BIN",
    "plot_class_depletion_heatmap", "plot_class_depletion_pooled",
    "DEPLETION_DB_STYLE", "DEPLETION_DB_OFFSET",
]


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
DATA = PROJECT_ROOT / "data"
OUTPUT = PROJECT_ROOT / "output"

#: Stage 1: scores and canonical identity, from the barcode counts. Written by
#: Scoring.ipynb (or scripts/score_cell.py + finalize_scores.py). No annotations.
SCORES_RECOVERED = OUTPUT / "scoring" / "scores_masterframe_recovered.tsv"
#: Stage 2 output and the base layer for the figures: our scores with every
#: annotation rebuilt from primary sources by scripts/reannotate_scores.py
#: (the annotation engine over the 22 annotation files, plus
#: scripts/compute_structure_annotations.py over the AlphaFold models).
SCORES_REANNOTATED = OUTPUT / "scoring" / "scores_reannotated.tsv"
#: Written by Annotations.ipynb: the combined annotation set the figures use.
ANNOTATED_COMBINED = OUTPUT / "annotated_combined.tsv"

#: Barcode-level scores for both no-variant controls (`empty_vector_std` and
#: `NoVar_std`), written by Scoring.ipynb Step 11. The raw material the DN
#: threshold is derived from, kept so the derivation can be shown rather than
#: asserted.
CONTROL_BARCODE_SCORES = OUTPUT / "scoring" / "control_barcode_scores.tsv"
#: DN thresholds recomputed from those barcodes: the 2.5th percentile of 200
#: draws of a 10-barcode mean, i.e. the distribution of a *variant-like* score
#: under an empty construct. Reproduces the delivered cutoffs to a median ratio
#: of 0.99, which is what identifies it as the method that produced them.
DN_THRESHOLDS = OUTPUT / "scoring" / "dn_thresholds_recomputed.tsv"
#: The delivered cutoffs. No longer an input — retained only for the agreement
#: check in Scoring.ipynb Step 11.
EV_CUTOFFS = DATA / "dn_cutoffs_empty_vector.tsv"

INTERACTIONS_AUGMENTED = OUTPUT / "dn" / "interactions_curated_augmented.tsv"
POPULATION_HGVSP = OUTPUT / "population_data_hgvsp.tsv"

#: Per-residue structural annotations from the raw AlphaFold models, written by
#: ``scripts/compute_structure_annotations.py``: pLDDT, the structure's own
#: residue identity, and intramolecular domain-domain contacts.
STRUCTURE_ANNOTATIONS = OUTPUT / "structure_annotations.tsv"
#: The AlphaFold v4 monomer models the structural annotations are computed from,
#: and which the DSSP outputs (hence every RSA in the project) came from.
AF_STRUCTURES = (PROJECT_ROOT / "data" / "inputs" / "annotations" / "AF_MAPK_structures")

#: Quantitative predictors and contact sets, each produced by its own script.
SPURS_DDG = PROJECT_ROOT / "data" / "inputs" / "spurs_ddg_all_proteins.tsv"
PHYLOP_PER_POSITION = OUTPUT / "dn" / "phylop_per_position.tsv"
KINASE_JSD = OUTPUT / "hsp90" / "kinase_jsd_per_position.tsv"
TRANSFERRED_CONTACTS = OUTPUT / "hsp90" / "transferred_contacts.tsv"
KINASE_MOTIFS = OUTPUT / "hsp90" / "kinase_motif_assignments.tsv"
GENIE_JOINED = OUTPUT / "clinical" / "activity_vs_genie_joined.tsv"
REFERENCE_FASTA = DATA / "reference_protein_sequences.fasta"
PROTEIN_ACCESSIONS = PROJECT_ROOT / "config" / "protein_accessions.yaml"


def annotation_sources() -> dict:
    """The primary annotation files, resolved from ``config/paths.yaml``.

    The 22 downloaded/derived inputs the annotation engine reads — DSSP outputs,
    NCBI conserved domains, PhosphoSitePlus, ClinVar, AlphaMissense, CysDB, the
    published DMS tables, the kinase alignment. Exposed so a notebook can print
    and check every input it depends on rather than trusting a path buried in a
    script.
    """
    import sys
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from labelseq_mapk.config import load_config
    return load_config(PROJECT_ROOT / "config")["paths"]["annotations"]


def input_manifest(extra: dict | None = None) -> "pd.DataFrame":
    """Every input file this pipeline reads, with a size and an existence check.

    Printed at the top of a notebook so that a missing or truncated input is
    caught before it becomes a puzzling number in a figure.
    """
    named = {
        "scores (stage 1)": SCORES_RECOVERED,
        "scores re-annotated (stage 2a)": SCORES_REANNOTATED,
        "structure annotations": STRUCTURE_ANNOTATIONS,
        "AlphaFold models": AF_STRUCTURES,
        "control barcode scores": CONTROL_BARCODE_SCORES,
        "DN thresholds (recomputed)": DN_THRESHOLDS,
        "curated interactions": INTERACTIONS_AUGMENTED,
        "HSP90/CDC37 contacts": TRANSFERRED_CONTACTS,
        "kinase motifs": KINASE_MOTIFS,
        "phyloP": PHYLOP_PER_POSITION,
        "kinase JSD": KINASE_JSD,
        "SPURS ddG": SPURS_DDG,
        "population (gnomAD/AoU)": POPULATION_HGVSP,
        "GENIE": GENIE_JOINED,
        "reference sequences": REFERENCE_FASTA,
        "protein accessions": PROTEIN_ACCESSIONS,
    }
    named.update({f"annotation source: {k}": Path(v)
                  for k, v in annotation_sources().items()})
    if extra:
        named.update({k: Path(v) for k, v in extra.items()})
    rows = []
    for label, p in named.items():
        p = Path(p)
        if p.is_dir():
            n = len(list(p.glob("*")))
            size = f"{n} files"
        elif p.exists():
            size = f"{p.stat().st_size / 1e6:,.1f} MB"
        else:
            size = "—"
        rows.append({"input": label, "exists": p.exists(), "size": size,
                     "path": str(p)})
    return pd.DataFrame(rows)


def add_structure_annotations(df: "pd.DataFrame",
                              path: "Path | None" = None) -> "pd.DataFrame":
    """Merge the AlphaFold-derived per-residue columns onto a scored table.

    Keyed on ``(protein, Position)``. Adds ``plddt``, ``pdb_aa``,
    ``inter_domain_contacts``, ``inter_domain_contacts_all_atom``,
    ``inter_domain_partners``, and derives ``pdb_aa_mismatch`` — the structure's
    residue disagreeing with our wild-type call, which means the model and our
    numbering describe different isoforms.
    """
    path = Path(path or STRUCTURE_ANNOTATIONS)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run scripts/compute_structure_annotations.py")
    # 4 A all-atom only -- the older `inter_domain_contacts` cutoff was
    # superseded on 2026-08-10 and is no longer merged onto scores.
    cols = ["plddt", "pdb_aa",
            "inter_domain_contacts_all_atom", "inter_domain_partners"]
    st = (pd.read_csv(path, sep="\t")[["protein", "position"] + cols]
          .drop_duplicates(["protein", "position"]))
    out = df.copy()
    out["_pos"] = pd.to_numeric(out["Position"], errors="coerce")
    out = out.merge(st.rename(columns={"position": "_pos"}),
                    on=["protein", "_pos"], how="left").drop(columns="_pos")
    out["pdb_aa_mismatch"] = (out["pdb_aa"].notna()
                              & (out["pdb_aa"] != out["Wild Type Residue"]))
    for c in ("inter_domain_contacts_all_atom",):
        out[c] = out[c].fillna(False).astype(bool)
    return out


# ---------------------------------------------------------------------------
# Score columns
# ---------------------------------------------------------------------------

#: WT-normalised score. WT == 1.0 **within each condition**, so differences on
#: this column are wild-type-relative (the *buffering* scale).
SCORE_WT_REL = "average score"
#: Standard-curve-corrected score. Retains the global WT shift between
#: conditions, so differences on this column are absolute (the *dependence*
#: scale). Compressed by the zero-intercept fit — fine for ranking within a
#: library, treat cross-protein absolute values with care.
SCORE_ABS = "intercept_0_standard-adjusted score"
#: Per-replicate companions of SCORE_ABS. New in the 080126 delivery; these are
#: what make a per-variant significance test on the absolute scale possible.
SCORE_ABS_REPS = [f"intercept_0_std_adj_score_{j}" for j in (1, 2, 3)]


# ---------------------------------------------------------------------------
# Protein sets and display names
# ---------------------------------------------------------------------------

#: Figure-3 (dominant-negative) protein order: RAFs, RTKs, GTPases, GEFs,
#: phosphatase, then the scaffolds/adaptor. Shared by the DN panels so their
#: columns line up when stacked.
#:
#: NOT a general-purpose protein list — it contains no MEK, because MEK is not
#: DN-eligible. Ordering a figure-4 panel through it drops MEK1 and MEK2
#: silently and sorts KSR2 to the end; use :data:`HSP90_ORDER` there.
DN_ORDER = [
    "araf", "braf", "craf",
    "egfr", "erbb2", "met", "ret",
    "kras", "mras",
    "sos1", "sos2",
    "shp2",
    "grb2", "ksr1", "ksr2",
]

#: Figure-4 (HSP90) protein order: the nine kinase-domain-bearing proteins
#: profiled under HSP90 inhibition, grouped by how much their WILD TYPE depends
#: on HSP90 — high (CRAF, ARAF, RET), moderate (KSR2, EGFR, HGFR), low (BRAF,
#: MEK2, MEK1). This is the order the violin panels use, so every figure-4 panel
#: that resolves by protein reads in the same sequence.
#:
#: SOS2 is not here: it has no kinase domain, and its response is inverted
#: (wild-type SOS2 rises under HSP90 inhibition). The all-variant panels append
#: it as its own tier.
HSP90_ORDER = [
    "craf", "araf", "ret",
    "ksr2", "egfr", "met",
    "braf", "mek2", "mek1",
]

#: Every protein key either order knows about — for "is this string a protein?"
#: checks, where ordering is irrelevant.
PROTEIN_KEYS = DN_ORDER + [p for p in HSP90_ORDER if p not in DN_ORDER]

#: The 12 proteins whose wild-type overexpression raises pathway activity, so a
#: variant scoring below the empty-vector baseline is interpretable as dominant
#: negative. Excludes the inhibitory set below.
DN_ELIGIBLE = [
    "araf", "braf", "craf", "egfr", "erbb2", "kras",
    "met", "mras", "ret", "shp2", "sos1", "sos2",
]

#: Overexpressing these *lowers* pathway activity, so "below empty vector" is
#: the expected wild-type phenotype, not dominant negativity. Matches
#: ``config/scoring.yaml: inhibitory_proteins``.
INHIBITORY_PROTEINS = {"grb2", "ksr1", "ksr2", "mek1", "mek2"}

#: The 9 kinases with paired control/HSP90i abundance data.
#: Every protein profiled under HSP90 inhibition — ten, including SOS2.
#:
#: SOS2 belongs here (8,397 abundance measurements) even though it is absent
#: from :data:`HSP90_ORDER`: it has no kinase domain, so it cannot appear in the
#: kinase-domain panels, and its response is inverted. Omitting it from this
#: list silently dropped its wild type from the dependence panel.
HSP90I_PROTEINS = ["araf", "braf", "craf", "egfr", "ksr2", "mek1", "mek2",
                   "met", "ret", "sos2"]

#: Unperturbed activity conditions. DN is defined only here: SerumStarve
#: collapses EGFR-WT onto the empty-vector baseline, and CIAR drives the pathway
#: independently of variant identity, so neither reports variant function
#: against a meaningful no-kinase floor.
BASELINE_TREATMENTS = {"No_treatment", "DMSO"}

#: Variant types that change the protein product. DN is restricted to these:
#: synonymous and the BRAF spike-in standards cannot be dominant negative.
PROTEIN_ALTERING_TYPES = {"missense", "nonsense", "deletion"}

#: PyMOL executable. Override with the ``PYMOL`` environment variable.
PYMOL_BIN = Path(os.environ.get("PYMOL", Path.home() / "pymol" / "pymol"))
#: ``fetch`` writes to the cwd unless told otherwise; keep downloads together.
PYMOL_FETCH_CACHE = DATA / "structures" / "pymol_fetch_cache"

#: MET's gene symbol is MET but the receptor is conventionally HGFR in this
#: manuscript. Only this one protein differs from an uppercase key.
_DISPLAY_OVERRIDES = {"met": "HGFR", "shp2": "SHP2", "kras": "KRAS", "mras": "MRAS"}


def protein_label(protein: str) -> str:
    """Display name for a protein key. MET renders as **HGFR**.

    Also handles the synthetic scopes (``egfr_ss``, ``kras_ciar``) and split
    libraries (``braf_cterm``) so any key in the data can be labelled.

    >>> protein_label("met"), protein_label("braf_cterm")
    ('HGFR', 'BRAF (C)')
    """
    key = str(protein).strip().lower()
    suffix = ""
    if key.endswith("_cterm"):
        key, suffix = key[:-6], " (C)"
    elif key.endswith("_nterm"):
        key, suffix = key[:-6], " (N)"
    elif key.endswith("_ss"):
        key, suffix = key[:-3], " (serum-starved)"
    elif key.endswith("_ciar"):
        key, suffix = key[:-5], " (CIAR)"
    return _DISPLAY_OVERRIDES.get(key, key.upper()) + suffix


def protein_labels(proteins: Iterable[str]) -> list[str]:
    """Vectorised :func:`protein_label`, for axis tick labels."""
    return [protein_label(p) for p in proteins]


# ---------------------------------------------------------------------------
# Palettes
# ---------------------------------------------------------------------------

#: Locked across every HSP90 figure (docs/hsp90_metric_definitions.md).
BUFFERING_COLORS = {
    "Buffered": "#bd2c2c",
    "Poorly buffered": "#d9892d",
    "WT-like or high": "#3a6fb5",
}
BUFFERING_ORDER = ["Buffered", "Poorly buffered", "WT-like or high"]

#: White -> class-colour ramps for the structure renders, so a painted fold
#: reads in the same colour as the class does in every 2D panel.
#: Bottom-to-top stack order for the buffering proportion bars (4C, 4G):
#: the WT-like majority sits at the base so the two buffered classes stack
#: against a common baseline and can be compared across bars by eye.
BUFFERING_STACK_ORDER = ["WT-like or high", "Buffered", "Poorly buffered"]

BUFFERING_CMAPS = {"Buffered": "Reds", "Poorly buffered": "Oranges",
                   "WT-like or high": "Blues"}

#: Activity-class palette, kept stable across figures.
ACTIVITY_COLORS = {
    "DN": "#C0392B",
    "low": "#26A69A",
    "wt-like": "#F4E04D",
    "high": "#E64A19",
}
ACTIVITY_ORDER = ["DN", "low", "wt-like", "high"]

#: Title-cased activity groups as used by the by-activity-group panels
#: (``scripts/analyze_structure_by_activity_group.py`` and the phyloP / RSA /
#: kinase-conservation companions). Same colours, ordered GOF -> most severe.
ACTIVITY_GROUP_ORDER = ["High", "WT-like", "Low", "DN"]
ACTIVITY_GROUP_COLOR = {
    "High": "#E64A19", "WT-like": "#F4E04D", "Low": "#26A69A", "DN": "#C0392B",
}
#: Minimum variants in a (protein, group) cell for it to contribute a
#: per-protein median dot — the Simpson guard.
MIN_N_PER_PROTEIN = 10

#: Population / cancer databases.
DB_COLORS = {"gnomAD": "#3571b6", "AoU": "#b67a35", "GENIE": "#4f8f4f"}

#: Per-protein sphere colours for the structural renders, carried over from
#: ``scripts/paint_dn_counts_pdb_structures.py`` so a protein keeps one colour
#: across every structure it appears in.
PROTEIN_SPHERE_COLOR = {
    "araf":  (0.85, 0.65, 0.10),   # mustard
    "braf":  (0.10, 0.28, 0.65),   # deep blue
    "craf":  (0.25, 0.55, 0.85),   # cornflower
    "egfr":  (0.70, 0.10, 0.15),   # ruby
    "erbb2": (0.65, 0.08, 0.12),   # maroon
    "kras":  (0.90, 0.45, 0.05),   # dark orange
    "met":   (0.55, 0.35, 0.10),   # bronze
    "mras":  (0.85, 0.50, 0.10),   # amber
    "ret":   (0.45, 0.05, 0.45),   # dark magenta
    "shp2":  (0.15, 0.60, 0.30),   # forest green
    "sos1":  (0.50, 0.05, 0.65),   # purple
    "sos2":  (0.05, 0.55, 0.45),   # teal
}
#: Cartoon and partner colours for the structural renders.
STRUCT_CARTOON_COLOR = "gray80"
STRUCT_PARTNER_COLORS = ("skyblue", "wheat", "lightpink")

#: Per-variant-type colours, carried over from
#: ``scripts/plot_dn_structural_enrichment_forest*.py`` so the DN forests match.
VARIANT_TYPE_COLORS = {
    "missense": "#2166ac",   # blue
    "deletion": "#1b7837",   # green
    "nonsense": "#762a83",   # purple
}
VARIANT_TYPE_ORDER = ["missense", "deletion", "nonsense"]

#: The same palette keyed by the `variant_category` vocabulary the scoring
#: pipeline emits, which names the single-codon class "3nt deletion" and keeps an
#: explicit "other" bucket (multi-mutants and larger in-frame deletions).
#: Synonymous, WT and the spiked standards are deliberately absent: they are
#: controls rather than library variants, so the panels using this palette plot
#: only the protein-altering classes.
VARIANT_CATEGORY_COLORS = {
    "missense": "#2166ac",       # blue
    "3nt deletion": "#1b7837",   # green
    "nonsense": "#762a83",       # purple
    "frameshift": "#b2182b",     # red
    "other": "#8c8c8c",          # grey
}
VARIANT_CATEGORY_ORDER = ["missense", "3nt deletion", "nonsense", "frameshift",
                          "other"]

#: DN structural-ontology (v4) bucket styling, carried over from
#: ``scripts/plot_dn_ontology_v4_coverage.py``. Each entry is
#: ``(facecolor, hatch_colour, hatch_pattern)``; the two-category combos are
#: drawn as the first category's fill hatched in the second's colour.
#:
#: Note there are six buckets and not seven: v4 defines Buried as
#: ``rSASA < 0.25 and not active site``, so B and AS are mutually exclusive by
#: construction and neither B+AS nor B+AS+I can occur.
ONTOLOGY_COLOR_B = "#d97b1f"      # orange
ONTOLOGY_COLOR_AS = "#a64ca6"     # purple
ONTOLOGY_COLOR_I = "#3a6fb5"      # blue
ONTOLOGY_COLOR_NONE = "#ffffff"   # white -- unexplained
ONTOLOGY_NONE_EDGE = "#999999"

ONTOLOGY_BUCKET_ORDER = ["B_only", "AS_only", "I_only", "B_I", "AS_I", "none"]
ONTOLOGY_BUCKET_STYLE: dict[str, tuple] = {
    "B_only":  (ONTOLOGY_COLOR_B, None, None),
    "AS_only": (ONTOLOGY_COLOR_AS, None, None),
    "I_only":  (ONTOLOGY_COLOR_I, None, None),
    "B_I":     (ONTOLOGY_COLOR_B, ONTOLOGY_COLOR_I, "////"),
    "AS_I":    (ONTOLOGY_COLOR_AS, ONTOLOGY_COLOR_I, "////"),
    "none":    (ONTOLOGY_COLOR_NONE, None, None),
}
ONTOLOGY_BUCKET_LABELS = {
    "B_only": "Buried (B)", "AS_only": "Active site (AS)",
    "I_only": "Interface (I)", "B_I": "B + I", "AS_I": "AS + I",
    "none": "Other",
}
#: Legend layout: single categories + Other on the top row, combos beneath.
ONTOLOGY_LEGEND_ROWS = [["B_only", "AS_only", "I_only", "none"],
                        ["B_I", "AS_I"]]


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

_HELVETICA_DIR = Path.home() / ".local" / "share" / "fonts"


def apply_style(*, base_fontsize: float = 8.0, strict: bool = False) -> None:
    """Register Helvetica and set publication rcParams. Idempotent.

    ``pdf.fonttype=42`` and ``svg.fonttype="none"`` keep text as *editable
    text* rather than outlined paths, which is what lets the panels be
    assembled and re-labelled in Illustrator.

    Args:
        base_fontsize: Default text size in points.
        strict: If True, raise when Helvetica is missing instead of falling
            back to Arial/DejaVu. Use in the final figure run.
    """
    ttfs = sorted(_HELVETICA_DIR.glob("Helvetica*.ttf"))
    if ttfs:
        for f in ttfs:
            fm.fontManager.addfont(str(f))
    elif strict:
        raise FileNotFoundError(f"No Helvetica*.ttf in {_HELVETICA_DIR}")

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "font.size": base_fontsize,
        "axes.labelsize": base_fontsize,
        "axes.titlesize": base_fontsize + 1,
        "xtick.labelsize": base_fontsize - 1,
        "ytick.labelsize": base_fontsize - 1,
        "legend.fontsize": base_fontsize - 1,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "figure.dpi": 110,
        "savefig.dpi": 300,
    })


def save_figure(
    fig: Figure,
    stem: str | Path,
    *,
    formats: Sequence[str] = ("pdf", "svg", "png"),
    tight: bool = True,
) -> list[Path]:
    """Save a figure to several formats beside one another.

    PDF and SVG are the assembly formats (editable text); PNG is for quick
    viewing in the notebook.

    Args:
        fig: Figure to write.
        stem: Output path *without* extension. Parent dirs are created.
        formats: Extensions to write.
        tight: Use ``bbox_inches="tight"``. Safe with the fixed-pitch helpers
            — it trims surrounding whitespace without resizing the data area.

    Returns:
        The paths written.
    """
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    kw = {"bbox_inches": "tight"} if tight else {}
    written = []
    for ext in formats:
        p = stem.with_suffix(f".{ext}")
        fig.savefig(p, **kw)
        written.append(p)
    return written


def stars_4tier(p: float) -> str:
    """``****`` <1e-4, ``***`` <1e-3, ``**`` <1e-2, ``*`` <0.05, else ``ns``.

    Convention of ``plot_class_depletion_global.py``. Non-finite -> "".
    """
    if p is None or not np.isfinite(p):
        return ""
    return ("****" if p < 1e-4 else "***" if p < 1e-3 else
            "**" if p < 1e-2 else "*" if p < 0.05 else "ns")


def stars_4tier_ns(p: float) -> str:
    """As :func:`stars_4tier` but ``n.s.`` for non-significant *and* for
    non-finite. Convention of the by-activity-group conservation panels."""
    if p is None or not np.isfinite(p):
        return "n.s."
    return ("****" if p < 1e-4 else "***" if p < 1e-3 else
            "**" if p < 1e-2 else "*" if p < 0.05 else "n.s.")


# ---------------------------------------------------------------------------
# Plots
#
# Every builder below is a port of an existing project figure — colours, font
# sizes, marker sizes and panel dimensions are carried over verbatim so the new
# panels are interchangeable with the ones already in the manuscript. The
# defaults reproduce the established figure; the keyword arguments are there so
# a panel can be adjusted without a new function being written.
#
# Source of each port is named in the function docstring.
# ---------------------------------------------------------------------------


def plot_forest(
    df: pd.DataFrame,
    *,
    label_col: str,
    or_col: str = "or",
    lo_col: str = "ci_low",
    hi_col: str = "ci_high",
    series_col: str | None = None,
    label_order: Sequence[str] | None = None,
    series_order: Sequence[str] | None = None,
    series_colors: Mapping[str, str] | None = None,
    series_dy: Mapping[str, float] | float = 0.24,
    series_n: Mapping[str, int] | None = None,
    figsize: tuple[float, float] = (7.6, 4.7),
    xticks: Sequence[float] = (0.25, 0.5, 1, 2, 4, 8),
    xlim_pad: tuple[float, float] = (0.80, 1.65),
    xlabel: str = "Dominant negative odds ratio (pooled, 95% CI)",
    title: str = "DN structural enrichment",
    annotate_or: bool = True,
    or_fmt: str = "{:.2f}",
    annot_col: str | None = None,
    markersize: float = 7.0,
    annot_x_mult: float = 1.06,
    elinewidth: float = 1.4,
    capsize: float = 3.5,
    legend_loc: str = "lower right",
    legend_fmt: str = "{series} DN (n={n:,})",
    fs: Mapping[str, float] | None = None,
) -> tuple[Figure, Axes]:
    """Dot + 95% CI odds-ratio forest on a log axis.

    Port of ``scripts/plot_dn_structural_enrichment_forest_pooled.py`` — same
    7.6x4.7 in panel, per-variant-type colours, +/-0.24 vertical offsets,
    7.0 pt markers, 1.4 pt whiskers with 3.5 pt caps, dashed OR=1 reference at
    ``#999999``, log ticks 0.25-8, the OR value printed at 1.06x the CI upper
    bound, and a light-framed legend in the lower-right corner. Font sizes are
    ~1.5x print-final so the panel stays legible after the 67% downscale used
    when placing it.

    Also used for the population-depletion forest by passing
    ``series_colors=DB_COLORS`` and a different ``xlabel``/``title``.

    Args:
        df: One row per (label, series) with the OR and its interval.
        label_col: Column holding the y-axis category.
        series_col: Optional column splitting into vertically-offset series.
        label_order / series_order: Explicit ordering; defaults to order of
            appearance. ``label_order`` is top-to-bottom.
        series_colors: series -> colour. Defaults to
            :data:`VARIANT_TYPE_COLORS`.
        series_dy: Vertical offset per series, or a scalar half-spread that is
            spread evenly across the series.
        series_n: series -> n, for the legend.
        xlim_pad: Multiplicative padding on (min CI low, max CI high).
        fs: Font-size overrides; keys ``title``, ``y``, ``x``, ``tick``,
            ``or``, ``legend``.
    """
    F = {"title": 16, "y": 13, "x": 14, "tick": 12, "or": 9.5, "legend": 11}
    F.update(fs or {})

    labels = list(label_order or dict.fromkeys(df[label_col]))
    series = list(series_order or (dict.fromkeys(df[series_col])
                                  if series_col else [None]))
    colors = dict(series_colors or VARIANT_TYPE_COLORS)
    for i, s in enumerate(series):
        colors.setdefault(s, list(VARIANT_TYPE_COLORS.values())[
            i % len(VARIANT_TYPE_COLORS)])

    if isinstance(series_dy, Mapping):
        dy = dict(series_dy)
    elif len(series) > 1:
        dy = dict(zip(series, np.linspace(series_dy, -series_dy, len(series))))
    else:
        dy = {series[0]: 0.0}

    n_lab = len(labels)
    y_of = {k: n_lab - i for i, k in enumerate(labels)}

    fig, ax = plt.subplots(figsize=figsize)
    xmin, xmax = np.inf, -np.inf
    for s in series:
        sub = df if s is None else df[df[series_col] == s]
        col = colors[s]
        for _, r in sub.iterrows():
            if r[label_col] not in y_of or not np.isfinite(r[or_col]):
                continue
            y = y_of[r[label_col]] + dy.get(s, 0.0)
            v, lo, hi = float(r[or_col]), float(r[lo_col]), float(r[hi_col])
            if not (np.isfinite(lo) and np.isfinite(hi)):
                lo = hi = v
            xmin, xmax = min(xmin, lo), max(xmax, hi)
            ax.errorbar(v, y, xerr=[[max(v - lo, 0)], [max(hi - v, 0)]],
                        fmt="none", ecolor=col, elinewidth=elinewidth,
                        capsize=capsize, capthick=elinewidth, alpha=0.9,
                        zorder=2)
            ax.plot(v, y, marker="o", markersize=markersize, linestyle="",
                    color=col, markeredgecolor="white", markeredgewidth=0.7,
                    zorder=3)
            if annot_col is not None:
                txt = str(r.get(annot_col, "") or "")
            elif annotate_or:
                txt = or_fmt.format(v)
            else:
                txt = ""
            if txt:
                ax.text(hi * annot_x_mult, y, txt, ha="left", va="center",
                        fontsize=F["or"], color=col, zorder=3, clip_on=False)

    ax.axvline(1.0, color="#999999", linewidth=1.0, linestyle="--", zorder=1)
    ax.set_xscale("log")
    if np.isfinite(xmin):
        ax.set_xlim(xmin * xlim_pad[0], xmax * xlim_pad[1])
    ax.set_xticks(list(xticks))
    ax.set_xticklabels([str(t) for t in xticks], fontsize=F["tick"])
    ax.minorticks_off()

    ax.set_yticks([y_of[k] for k in labels])
    ax.set_yticklabels(labels, fontsize=F["y"])
    ax.set_ylim(0.4, n_lab + 0.6)
    ax.set_xlabel(xlabel, fontsize=F["x"])
    if title:
        ax.set_title(title, fontsize=F["title"], pad=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=3)

    if series_col is not None:
        handles = [
            plt.Line2D([0], [0], marker="o", linestyle="",
                       markersize=markersize, markerfacecolor=colors[s],
                       markeredgecolor=colors[s],
                       label=legend_fmt.format(series=s,
                                               n=(series_n or {}).get(s, 0)))
            for s in series]
        leg = ax.legend(handles=handles, fontsize=F["legend"], loc=legend_loc,
                        frameon=True, handletextpad=0.4, labelspacing=0.4,
                        borderaxespad=0.7)
        leg.get_frame().set_edgecolor("#cccccc")
        leg.get_frame().set_facecolor("white")
        leg.get_frame().set_linewidth(0.6)
    fig.tight_layout()
    return fig, ax


def fixed_pitch_bar_axes(
    n_bars: int,
    *,
    pitch_in: float = 0.95,
    axes_h_in: float = 3.40,
    left_in: float = 0.95,
    right_in: float = 0.30,
    bottom_in: float = 1.15,
    top_in: float = 0.50,
    legend_in: float = 0.0,
) -> tuple[Figure, Axes]:
    """Figure + axes whose data area is a fixed physical width *per category*.

    Port of the helper in ``scripts/_plot_style.py``. Two stacked-bar panels can
    both pass ``width=0.62`` and still draw bars of different physical width,
    because a data unit maps to a different number of inches when the axes spans
    a different x-range (4 bars vs 3) or is squeezed by an outside legend. This
    pins the axes to an explicit inch rectangle and sets ``xlim`` so one
    category is exactly one data unit == ``pitch_in`` inches; a bar drawn at
    ``width=w`` is then always ``w * pitch_in`` inches wide whatever ``n_bars``
    is. That is what lets 4C and 4G show bars of identical width.

    Bars go at integer x positions ``0 .. n_bars-1``. Because the axes is placed
    with ``add_axes``, callers must **not** call ``fig.tight_layout()``;
    ``savefig(bbox_inches="tight")`` is fine — it trims surrounding whitespace
    without resizing the data area.

    Args:
        n_bars: Number of bar categories.
        pitch_in: Inches per category slot. Equal across panels => equal bars.
        axes_h_in: Height of the data area, in inches.
        left_in, right_in, bottom_in, top_in: Figure margins, in inches.
        legend_in: Extra width reserved at the right for an outside legend.
            Does not affect bar width; it only stops the legend being clipped.
    """
    data_w = n_bars * pitch_in
    fig_w = left_in + data_w + right_in + legend_in
    fig_h = bottom_in + axes_h_in + top_in
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes([left_in / fig_w, bottom_in / fig_h,
                       data_w / fig_w, axes_h_in / fig_h])
    ax.set_xlim(-0.5, n_bars - 0.5)
    return fig, ax


def plot_stacked_vbar(
    props: pd.DataFrame,
    *,
    colors: Mapping[str, str],
    n_by_row: Mapping[str, int] | pd.Series | None = None,
    xtick_fmt: str = "{label}\n(n={n:,})",
    ylabel: str = "Proportion of missense variants",
    legend_title: str = "WT-relative category",
    annot_min: float = 0.05,
    annot_fmt: str = "{:.2f}",
    bar_width: float = 0.62,
    ylim: tuple[float, float] = (0.0, 1.02),
    show_legend: bool = True,
    yticks: Sequence[float] | None = None,
    pitch_in: float = 0.95,
    legend_in: float = 1.70,
    axes_h_in: float = 3.40,
    fs: Mapping[str, float] | None = None,
) -> tuple[Figure, Axes]:
    """Vertical stacked proportion bars on a fixed physical bar pitch.

    Port of ``plot_stacked_bar`` in
    ``scripts/plot_hsp90_wt_category_by_domain.py`` — bars at ``width=0.62``
    with white 0.6 pt separators, every segment at or above ``annot_min``
    labelled in bold white inside the segment, x-tick labels carrying the group
    N, and a frameless titled legend outside the axes on the right.

    Args:
        props: Rows = bar groups in order, columns = stack segments in bottom-
            to-top order. Values are proportions.
        colors: Segment name -> colour.
        n_by_row: Group -> N, for the x-tick labels. Omit to label bare.
        annot_min: Segments smaller than this share are left unannotated.
        show_legend: Draw the outside legend. The activity-class panel omits it
            and inherits its colour key from the panel beside it.
        yticks: Explicit y ticks; ``None`` leaves matplotlib's choice.
        fs: Font-size overrides; keys ``annot``, ``xtick``, ``ylabel``,
            ``ytick``, ``legend``, ``legend_title``.
    """
    F = {"annot": 11.0, "xtick": 12.0, "ylabel": 13.0, "ytick": 11.0,
         "legend": 11.0, "legend_title": 12.0}
    F.update(fs or {})
    groups = list(props.index)
    fig, ax = fixed_pitch_bar_axes(len(groups), pitch_in=pitch_in,
                                   axes_h_in=axes_h_in, legend_in=legend_in)

    bottoms = np.zeros(len(groups))
    for seg in props.columns:
        vals = props[seg].to_numpy(dtype=float)
        ax.bar(range(len(groups)), vals, bottom=bottoms, label=str(seg),
               color=colors[seg], width=bar_width, linewidth=0.6,
               edgecolor="white")
        for i, (v, b) in enumerate(zip(vals, bottoms)):
            if v >= annot_min:
                ax.text(i, b + v / 2, annot_fmt.format(v), ha="center",
                        va="center", fontsize=F["annot"], color="white",
                        fontweight="bold")
        bottoms += vals

    ax.set_xticks(range(len(groups)))
    if n_by_row is not None:
        ax.set_xticklabels([xtick_fmt.format(label=g, n=int(n_by_row[g]))
                            for g in groups], fontsize=F["xtick"])
    else:
        ax.set_xticklabels([str(g) for g in groups], fontsize=F["xtick"])
    ax.set_ylabel(ylabel, fontsize=F["ylabel"])
    ax.set_ylim(*ylim)
    ax.set_xlabel("")
    ax.tick_params(axis="y", labelsize=F["ytick"])
    if yticks is not None:
        ax.set_yticks(list(yticks))
    if show_legend:
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0),
                  fontsize=F["legend"], frameon=False, title=legend_title,
                  title_fontsize=F["legend_title"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig, ax


def plot_stacked_hbar(
    frac: pd.DataFrame,
    *,
    counts: Mapping[str, int] | pd.Series | None = None,
    bucket_style: Mapping[str, tuple] | None = None,
    bucket_order: Sequence[str] | None = None,
    bucket_labels: Mapping[str, str] | None = None,
    legend_rows: Sequence[Sequence[str]] | None = None,
    pooled_row: str | None = None,
    bar_h: float = 0.82,
    pooled_gutter_mult: float = 1.5,
    annotate_threshold: float = 0.10,
    fig_w: float = 8.0,
    bar_pitch_in: float = 0.33,
    top_margin_in: float = 0.20,
    bottom_margin_in: float = 0.06,
    legend_in: float = 0.55,
    xlabel_gap_in: float = 0.55,
    xlabel: str = "Fraction of dominant negatives",
    n_fmt: str = "n = {:,}",
    fs: Mapping[str, float] | None = None,
) -> tuple[Figure, Axes]:
    """Horizontal stacked composition bars, one per category.

    Port of ``scripts/plot_dn_ontology_v4_pooled_coverage_horizontal.py``:
    8.0 in wide, 0.82-thick bars so rows nearly touch, hatched overlays for
    the two-category combos, white segment labels above 10%, ``n = X`` at the
    right edge, and a flow legend band beneath. An optional pooled
    "ALL PROTEINS" bar is separated by 1.5x the normal gutter.

    ``bucket_style`` maps bucket -> ``(facecolor, hatch_colour, hatch_pattern)``
    with the last two None for a solid fill; defaults to
    :data:`ONTOLOGY_BUCKET_STYLE`.

    Args:
        frac: Rows are bars, columns are buckets; values are fractions.
        counts: Per-bar n, printed at the right edge.
        pooled_row: Index label to place last and separate by a wider gutter.
        annotate_threshold: Segments at or below this fraction go unlabelled.
        fs: Font-size overrides; keys ``cat``, ``tick``, ``axis``, ``seg``,
            ``n``, ``legend``.
    """
    F = {"cat": 14, "tick": 13, "axis": 14, "seg": 11, "n": 9, "legend": 13}
    F.update(fs or {})
    style = dict(bucket_style or ONTOLOGY_BUCKET_STYLE)
    order = list(bucket_order or [b for b in ONTOLOGY_BUCKET_ORDER
                                  if b in frac.columns])
    labels = dict(bucket_labels or ONTOLOGY_BUCKET_LABELS)

    rows = [r for r in frac.index if r != pooled_row]
    y_of = {r: i for i, r in enumerate(rows)}
    normal_gutter = 1.0 - bar_h
    if pooled_row is not None and pooled_row in frac.index:
        y_of[pooled_row] = (len(rows) - 1) + bar_h + pooled_gutter_mult * normal_gutter
    n_rows_eq = max(y_of.values()) + 1

    bars_in = bar_pitch_in * n_rows_eq
    fig_h = top_margin_in + bars_in + xlabel_gap_in + legend_in + bottom_margin_in
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes([0.11, (bottom_margin_in + legend_in + xlabel_gap_in) / fig_h,
                       0.80, bars_in / fig_h])
    legend_ax = fig.add_axes([0.07, bottom_margin_in / fig_h, 0.92,
                              legend_in / fig_h])

    for row, y in y_of.items():
        cum = 0.0
        for b in order:
            f = float(frac.loc[row, b]) if b in frac.columns else 0.0
            if f <= 0 or not np.isfinite(f):
                continue
            face, hatch_col, hatch_pat = style.get(b, ("#999999", None, None))
            is_none = b in ("none", "Other")
            ax.add_patch(plt.Rectangle(
                (cum, y - bar_h / 2), f, bar_h, facecolor=face,
                edgecolor=ONTOLOGY_NONE_EDGE if is_none else "white",
                linewidth=0.5 if is_none else 0.3, zorder=2))
            if hatch_pat:
                ax.add_patch(plt.Rectangle(
                    (cum, y - bar_h / 2), f, bar_h, facecolor="none",
                    edgecolor=hatch_col, linewidth=0, hatch=hatch_pat, zorder=3))
            if f > annotate_threshold:
                ax.text(cum + f / 2, y, f"{int(round(f * 100))}%",
                        ha="center", va="center", fontsize=F["seg"],
                        color="black" if is_none else "white", zorder=4)
            cum += f
        if counts is not None and row in counts:
            ax.text(1.01, y, n_fmt.format(int(counts[row])), ha="left",
                    va="center", fontsize=F["n"], color="#444444")

    # xlim to 1.15 leaves room for the "n = X" labels at x=1.01.
    ax.set_xlim(0, 1.15)
    ax.set_ylim(-0.5, max(y_of.values()) + 0.5)
    ax.invert_yaxis()
    ax.set_yticks(list(y_of.values()))
    ax.set_yticklabels([protein_label(r) if str(r).lower() in PROTEIN_KEYS
                        else str(r) for r in y_of], fontsize=F["cat"])
    if pooled_row is not None and pooled_row in frac.index:
        ax.get_yticklabels()[-1].set_fontweight("bold")
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"], fontsize=F["tick"])
    ax.set_xlabel(xlabel, fontsize=F["axis"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#888")
    ax.tick_params(axis="y", length=0)

    # Two-row flowed legend, each row centred. Item widths are *measured* from
    # the renderer rather than estimated, so long names ("Active site (AS)")
    # and short ones ("AS + I") coexist without a fixed grid clipping either.
    legend_ax.set_xlim(0, 1)
    legend_ax.set_ylim(0, 1)
    legend_ax.axis("off")
    sw, sh, gap, item_gap = 0.045, 0.24, 0.010, 0.024
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    ax_w = legend_ax.get_window_extent(rend).width

    def _text_w(s: str) -> float:
        probe = legend_ax.text(0, -2, s, fontsize=F["legend"])
        w = probe.get_window_extent(rend).width / ax_w
        probe.remove()
        return w

    rows = legend_rows or [[b for b in order]]
    for row_buckets, y0 in zip(rows, (0.70, 0.24)):
        row_buckets = [b for b in row_buckets if b in style]
        if not row_buckets:
            continue
        widths = [sw + gap + _text_w(labels.get(b, b)) for b in row_buckets]
        total = sum(widths) + item_gap * (len(row_buckets) - 1)
        x = (1.0 - total) / 2.0
        for b, w in zip(row_buckets, widths):
            face, hatch_col, hatch_pat = style[b]
            is_none = b in ("none", "Other")
            legend_ax.add_patch(plt.Rectangle(
                (x, y0 - sh / 2), sw, sh, facecolor=face,
                edgecolor=ONTOLOGY_NONE_EDGE if is_none else "#888",
                linewidth=0.6 if is_none else 0.3))
            if hatch_pat:
                legend_ax.add_patch(plt.Rectangle(
                    (x, y0 - sh / 2), sw, sh, facecolor="none",
                    edgecolor=hatch_col, linewidth=0, hatch=hatch_pat))
            legend_ax.text(x + sw + gap, y0, labels.get(b, b), va="center",
                           ha="left", fontsize=F["legend"])
            x += w + item_gap
    return fig, ax


def plot_bubble_grid(
    entries: pd.DataFrame,
    *,
    col_col: str = "protein",
    row_col: str = "category",
    size_col: str = "n_dn_in",
    or_col: str = "or",
    fires_col: str = "fires",
    partner_col: str | None = "partner",
    col_order: Sequence[str] | None = None,
    row_order: Sequence[str] = ("B", "AS", "I"),
    row_labels: Mapping[str, str] | None = None,
    figsize: tuple[float, float] = (3.5, 1.75),
    axes_rect: Sequence[float] = (0.155, 0.25, 0.65, 0.47),
    cbar_rect: Sequence[float] = (0.83, 0.27, 0.017, 0.52),
    log2_lim: tuple[float, float] = (-3.0, 3.0),
    n_ref_area: float = 28.0,
    n_ref_count: float = 100.0,
    area_gamma: float = 0.6,
    legend_counts: Sequence[int] = (30, 100, 1000),
    or_label: str = "Odds ratio (log₂)",
    size_label: str = "# missense and deletion dominant negatives",
    fs: Mapping[str, float] | None = None,
) -> tuple[Figure, Axes]:
    """Grid of squares: area = count, fill = log2 OR, border = significance.

    Port of ``scripts/plot_dn_ontology_v4_combined_bubble_horizontal.py`` — the
    fixed 3.5x1.75 in panel with the axes at an absolute rectangle, RdBu_r on a
    two-slope norm over OR 1/8 to 8 with extend arrows, sub-linear marker area
    (``28 * (n/100)**0.6``) so large counts do not dwarf small ones, black
    border at significance and ``#888`` otherwise, protein labels on top at 45
    degrees, interface partner names written horizontally under the bottom row,
    a vertical colourbar at the right and a size legend along the bottom.

    Saved deliberately **without** ``bbox_inches="tight"`` in the original so
    the panel is exactly the requested size; pass ``tight=False`` to
    :func:`save_figure` to preserve that.

    Args:
        entries: One row per cell. Rows with a non-positive ``size_col`` are
            skipped, as are rows where ``or_col`` is not positive.
        partner_col: If given, its value is printed beneath the bottom row
            (used for the driving interface partner).
        fs: Font-size overrides; keys ``row``, ``col``, ``cbar``,
            ``cbar_tick``, ``legend``, ``partner``, ``size_label``.
    """
    import matplotlib.colors as mcolors

    F = {"row": 5.5, "col": 6.8, "cbar": 5.5, "cbar_tick": 4.5, "legend": 5.0,
         "partner": 3.4, "size_label": 5.5}
    F.update(fs or {})
    labels = dict(row_labels or {"B": "Buried", "AS": "Active\nsite",
                                 "I": "Interface"})
    cols = list(col_order or [p for p in DN_ORDER
                              if p in set(entries[col_col])])
    rows = list(row_order)
    x_of = {c: i for i, c in enumerate(cols)}
    y_of = {r: len(rows) - 1 - i for i, r in enumerate(rows)}

    def _area(n):
        return 0.0 if n <= 0 else n_ref_area * (n / n_ref_count) ** area_gamma

    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes(list(axes_rect))
    cmap = plt.get_cmap("RdBu_r")
    norm = mcolors.TwoSlopeNorm(vcenter=0.0, vmin=log2_lim[0], vmax=log2_lim[1])

    for _, e in entries.iterrows():
        if e[col_col] not in x_of or e[row_col] not in y_of:
            continue
        n = e[size_col]
        v = e[or_col]
        if not np.isfinite(n) or n <= 0 or not np.isfinite(v) or v <= 0:
            continue
        ax.scatter([x_of[e[col_col]]], [y_of[e[row_col]]], s=_area(n),
                   marker="s", facecolor=cmap(norm(np.log2(v))),
                   edgecolor="black" if bool(e.get(fires_col)) else "#888888",
                   linewidth=0.9 if bool(e.get(fires_col)) else 0.3, zorder=3)
        if partner_col and e[row_col] == rows[-1] and e.get(partner_col):
            p = str(e[partner_col])
            txt = (protein_label(e[col_col]) if p == "dimer"
                   else "pY\npeptide" if p == "pY_peptide" else p.upper())
            ax.text(x_of[e[col_col]], -0.36, txt, ha="center", va="top",
                    fontsize=F["partner"], color="#333333", linespacing=1.0,
                    zorder=4)

    ax.set_yticks([y_of[r] for r in rows])
    ax.set_yticklabels([labels.get(r, r) for r in rows], fontsize=F["row"],
                       linespacing=1.0)
    ax.set_xticks(range(len(cols)))
    ax.xaxis.tick_top()
    ax.set_xticklabels(protein_labels(cols), fontsize=F["col"], rotation=45,
                       ha="left", rotation_mode="anchor")
    ax.set_xlim(-0.5, len(cols) - 0.5)
    ax.set_ylim(-0.45, len(rows) - 0.6)
    ax.tick_params(axis="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cax = fig.add_axes(list(cbar_rect))
    ticks = list(range(int(log2_lim[0]), int(log2_lim[1]) + 1))
    cbar = fig.colorbar(sm, cax=cax, ticks=ticks, extend="both")
    cbar.set_label(or_label, fontsize=F["cbar"], labelpad=2)
    cbar.ax.set_yticklabels([f"{2.0 ** t:g}" for t in ticks],
                            fontsize=F["cbar_tick"])
    cbar.outline.set_linewidth(0.3)
    cbar.ax.tick_params(length=1.5, pad=1)

    handles = [plt.scatter([], [], s=_area(n), marker="s", facecolor="#cccccc",
                           edgecolor="#666666", linewidth=0.4, label=f"{n}")
               for n in legend_counts]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.45, 0.015),
               ncol=len(legend_counts), frameon=False, fontsize=F["legend"],
               handletextpad=0.4, columnspacing=2.8, borderpad=0.15)
    fig.text(0.45, 0.125, size_label, ha="center", va="bottom",
             fontsize=F["size_label"])
    return fig, ax


def plot_heatmap(
    mat: pd.DataFrame,
    *,
    cell_text: pd.DataFrame | None = None,
    q: pd.DataFrame | None = None,
    cmap: str = "RdBu_r",
    vmin: float | None = None,
    vmax: float | None = None,
    center: float | None = None,
    nodata_color: str = "#dddddd",   # NODATA_COLOR in the source script
    sig_q: float = 0.05,
    sig_q_strong: float = 0.01,
    sig_edge_weak: str = "#555555",
    sig_edge_strong: str = "black",
    sig_inset: float = 0.085,
    row_gaps: Mapping[int, float] | None = None,
    col_gaps: Mapping[int, float] | None = None,
    group_gap: float = 0.6,
    bold_cols: Sequence[str] = (),
    bold_rows: Sequence[str] = (),
    row_groups: Sequence[tuple[str, int, int]] = (),
    row_group_x: float = -0.255,
    row_group_fs: float = 14.0,
    subplots_adjust: Mapping[str, float] | None = None,
    anchor: str | None = None,
    suptitle: str = "",
    two_slope: bool = False,
    figsize: tuple[float, float] = (8.4, 7.2),
    cbar_label: str = "",
    title: str = "",
    fs: Mapping[str, float] | None = None,
) -> tuple[Figure, Axes]:
    """Square-cell heatmap with inset significance borders.

    Port of ``scripts/plot_class_depletion_heatmap.py`` — RdBu_r fill,
    ``#dddddd`` for no data, and significance carried by a **border inset**
    ``0.085`` inside the cell (dark grey at q<0.05, black at q<0.01) so a grey
    border is never overdrawn by a neighbour's black one. Borders rather than
    asterisks keep cells readable at this size. Blank spacer rows/columns
    separate database blocks and the pooled reference column.

    Args:
        mat: Values to colour.
        cell_text: Same-shaped strings to print in cells.
        q: Same-shaped q-values driving the border tiers.
        center: If given (and vmin/vmax are not), a symmetric diverging scale
            centred here.
        row_gaps / col_gaps: ``{index: gap}`` extra space before that
            row/column, in cell units. ``group_gap`` is the conventional value.
        fs: Font-size overrides; keys ``row``, ``col``, ``cell``, ``cbar``,
            ``title``.
    """
    F = {"row": 12, "col": 11, "cell": 7.5, "cbar": 11, "title": 13,
         "suptitle": 12}
    F.update(fs or {})
    vals = mat.values.astype(float)
    if two_slope and vmin is not None and vmax is not None:
        pass
    elif vmin is None or vmax is None:
        if center is not None:
            span = np.nanmax(np.abs(vals - center))
            vmin, vmax = center - span, center + span
        else:
            vmin = np.nanmin(vals) if vmin is None else vmin
            vmax = np.nanmax(vals) if vmax is None else vmax
    if two_slope:
        import matplotlib.colors as mcolors
        norm = mcolors.TwoSlopeNorm(vcenter=center if center is not None else 0.0,
                                    vmin=vmin, vmax=vmax)
    else:
        norm = plt.Normalize(vmin, vmax)
    cm = plt.get_cmap(cmap)

    def _positions(n, gaps):
        out, p = {}, 0.0
        for i in range(n):
            p += (gaps or {}).get(i, 0.0)
            out[i] = p
            p += 1.0
        return out

    y_of = _positions(mat.shape[0], row_gaps)
    x_of = _positions(mat.shape[1], col_gaps)

    fig, ax = plt.subplots(figsize=figsize)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = vals[i, j]
            xx, yy = x_of[j], y_of[i]
            ax.add_patch(plt.Rectangle(
                (xx - 0.5, yy - 0.5), 1, 1,
                facecolor=nodata_color if not np.isfinite(v) else cm(norm(v)),
                edgecolor="white", linewidth=0.5, zorder=2))
            if q is not None:
                qv = q.iat[i, j]
                if np.isfinite(qv) and qv < sig_q:
                    strong = qv < sig_q_strong
                    ax.add_patch(plt.Rectangle(
                        (xx - 0.5 + sig_inset, yy - 0.5 + sig_inset),
                        1 - 2 * sig_inset, 1 - 2 * sig_inset, facecolor="none",
                        edgecolor=sig_edge_strong if strong else sig_edge_weak,
                        linewidth=1.1 if strong else 0.8, zorder=4))
            if cell_text is not None:
                t = cell_text.iat[i, j]
                if isinstance(t, str) and t:
                    shade = 0.5 if not np.isfinite(v) else norm(v)
                    ax.text(xx, yy, t, ha="center", va="center",
                            color="white" if (shade < 0.18 or shade > 0.82)
                            else "#111111", fontsize=F["cell"], zorder=5)

    ax.set_xlim(min(x_of.values()) - 0.55, max(x_of.values()) + 0.55)
    ax.set_ylim(max(y_of.values()) + 0.55, min(y_of.values()) - 0.55)
    ax.set_xticks(list(x_of.values()))
    ax.set_xticklabels([protein_label(c) if str(c).lower() in PROTEIN_KEYS
                        else str(c) for c in mat.columns],
                       rotation=45, ha="left", fontsize=F["col"])
    ax.xaxis.set_ticks_position("top")
    for lab, col in zip(ax.get_xticklabels(), mat.columns):
        if col in set(bold_cols):
            lab.set_fontweight("bold")
    ax.set_yticks(list(y_of.values()))
    ax.set_yticklabels([str(r) for r in mat.index], fontsize=F["row"])
    for lab, row in zip(ax.get_yticklabels(), mat.index):
        if row in set(bold_rows):
            lab.set_fontweight("bold")
    ax.set_aspect("equal")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)

    # Rotated database super-labels, pushed left of the long class labels so
    # they cannot collide with "Nonsense (NMD)".
    for name, i0, i1 in row_groups:
        ax.annotate(name, xy=(row_group_x, (y_of[i0] + y_of[i1]) / 2),
                    xycoords=("axes fraction", "data"), rotation=90,
                    ha="center", va="center", fontsize=row_group_fs,
                    fontweight="bold", annotation_clip=False)

    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cm), ax=ax,
                        fraction=0.030, pad=0.03)
    cbar.set_label(cbar_label, fontsize=F["cbar"])
    cbar.outline.set_linewidth(0.5)
    if subplots_adjust:
        fig.subplots_adjust(**subplots_adjust)
    if anchor:
        ax.set_anchor(anchor)
    if suptitle:
        fig.suptitle(suptitle, fontsize=F.get("suptitle", 12),
                     fontweight="bold", y=0.99)
    if title:
        ax.set_title(title, loc="left", pad=24, fontsize=F["title"])
    return fig, ax


def plot_violin_panel(
    groups: Mapping[str, Sequence[float]] | None = None,
    *,
    series_groups: Mapping[str, Mapping[str, Sequence[float]]] | None = None,
    series_colors: Mapping[str, str] | None = None,
    series_dy: float = 0.20,
    series_width: float = 0.34,
    group_colors: Mapping[str, str] | None = None,
    median_bar: bool = False,
    bracket_vs: str | None = None,
    bracket_targets: Sequence[str] = (),
    bracket_gt_counts: Mapping[str, tuple[int, int]] | None = None,
    bracket_base: float | None = None,
    bracket_base_pad: float = 0.4,
    bracket_step: float = 0.9,
    bracket_tick: float = 0.2,
    bracket_dx: float = 0.14,
    bracket_fs: float = 6.5,
    bracket_stars_fn=stars_4tier_ns,
    ylim: tuple[float, float] | None = None,
    ylim_lo: float | None = None,
    ylim_lo_pct: float = 0.5,
    ylim_lo_pad: float = 0.5,
    ylim_top_pad: float = 0.5,
    xlim: tuple[float, float] | None = None,
    n_below: bool = False,
    ref_line: float | None = None,
    ref_color: str = "#999999",
    ref_ls: str = "--",
    ref_lw: float = 0.7,
    tight_layout: bool = False,
    ref_label: str = "",
    class_frac: pd.DataFrame | None = None,
    class_colors: Mapping[str, str] | None = None,
    class_order: Sequence[str] | None = None,
    point_overlay: pd.DataFrame | None = None,
    overlay_value_col: str | None = None,
    overlay_group_col: str | None = None,
    marker_values: Mapping[str, float] | None = None,
    marker_style: str = "dash",
    tiers: Sequence[tuple[str, Sequence[str]]] = (),
    tier_colors: Mapping[str, str] | None = None,
    tier_fs: float = 10.0,
    y_percent: bool = False,
    title_pad: float | None = None,
    violin_color: str = "#b9c2cf",
    violin_edge: str = "#4a5567",
    figsize: tuple[float, float] = (14.5, 9.4),
    height_ratios: Sequence[float] = (1.0, 4.5),
    widths: float = 0.78,
    show_violin_medians: bool = True,
    subplots_adjust: Mapping[str, float] | None = None,
    tick_labelsize: float | None = None,
    spine_lw: float | None = None,
    log_y: bool = False,
    hline: float | None = 1.0,
    ylabel: str = "Abundance score",
    show_n: bool = True,
    title: str = "",
    fs: Mapping[str, float] | None = None,
) -> tuple[Figure, np.ndarray]:
    """Violins per category, optionally with a class strip above and/or points.

    Port of ``scripts/plot_hsp90_score_violin.py`` — 14.5x9.4 in, class-fraction
    strip above the violins at a 1 : 4.5 height ratio so the violins remain the
    dominant element while the composition strip's columns stay locked to the
    violin x-positions.

    One function covers three panels by argument:

    * ``class_frac`` given -> the composition strip is drawn above (Fig 4D, S4A).
    * ``point_overlay`` given -> per-group medians are scattered as open
      circles over the violins, the check that a pooled signal is not a
      composition artifact (Fig S3B).
    * ``marker_values`` given -> a horizontal dash per category, used for the
      wild-type value on the dependence panel (Fig 4B).

    Args:
        groups: Ordered mapping of category -> values.
        class_frac: Rows indexed by the same categories, columns the classes.
        point_overlay: Long frame from which per-(group, category) medians are
            taken, using ``overlay_value_col`` and ``overlay_group_col``.
        marker_values: category -> value for the wild-type marker.
        marker_style: ``"dash"`` for a horizontal line spanning the slot, or
            ``"star"`` for the open star used by the fractional-change panel.
        tiers: ``(caption, [category, ...])`` groups, drawn with a bold coloured
            caption above the panel and a faint divider between groups — used
            when the categories are themselves grouped (the WT HSP90
            dependence tiers).
        tier_colors: caption -> colour, so a caption can be matched to the
            violin fills it describes.
        y_percent: Format the y-axis ticks as percentages.
        ref_color / ref_ls / ref_lw: Style of the ``ref_line``.
        tight_layout: Call ``fig.tight_layout()`` at the end. Only safe on the
            single-axes form — the strip form places its axes explicitly.
        title_pad: Padding for the title, in points.
        fs: Font-size overrides; keys ``tick``, ``axis``, ``legend``, ``title``.
    """
    F = {"tick": 11, "axis": 13, "legend": 11, "title": 14}
    F.update(fs or {})
    if (groups is None) == (series_groups is None):
        raise ValueError("pass exactly one of `groups` or `series_groups`")

    # Normalise both call styles to {series: {category: values}}.
    nested = ({None: groups} if series_groups is None
              else {k: v for k, v in series_groups.items()})
    cats = list(dict.fromkeys(c for g in nested.values() for c in g))
    pos = np.arange(len(cats))
    scolors = dict(series_colors or VARIANT_TYPE_COLORS)
    group_colors = dict(group_colors) if group_colors else None
    series = list(nested)
    offs = (np.linspace(series_dy, -series_dy, len(series)) if len(series) > 1
            else np.zeros(1))
    w = series_width if len(series) > 1 else widths

    def _vals(s, c):
        return np.asarray([v for v in nested[s].get(c, []) if np.isfinite(v)],
                          dtype=float)

    # Per-category n for the tick labels: total across series.
    data = [np.concatenate([_vals(s, c) for s in series]) if series else
            np.array([]) for c in cats]

    if class_frac is not None:
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 1, height_ratios=list(height_ratios),
                              hspace=0.06)
        ax_strip, ax = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    else:
        fig, ax = plt.subplots(figsize=figsize)
        ax_strip = None

    for si, s in enumerate(series):
        vv = [_vals(s, c) for c in cats]
        keep = [i for i, d in enumerate(vv) if d.size > 1]
        if not keep:
            continue
        parts = ax.violinplot([vv[i] for i in keep],
                              positions=pos[keep] + offs[si], widths=w,
                              showextrema=False,
                              showmedians=show_violin_medians)
        face = scolors.get(s, violin_color) if s is not None else violin_color
        for bi, b in zip(keep, parts["bodies"]):
            b.set_facecolor(group_colors.get(cats[bi], face) if group_colors
                            else face)
            b.set_edgecolor("#333333" if group_colors else
                            (violin_edge if s is None else "none"))
            b.set_alpha(0.85 if s is None else 0.8)
            b.set_linewidth(0.6 if group_colors else 0.5)
        if show_violin_medians:
            parts["cmedians"].set_color("#111111")
            parts["cmedians"].set_linewidth(1.1 if s is None else 0.9)
        if median_bar:
            # Wide flat median bar instead of the violin's thin default.
            if show_violin_medians:
                parts["cmedians"].set_alpha(0.0)
            for i in keep:
                ax.hlines(np.median(vv[i]), i - 0.34, i + 0.34,
                          color="#111111", linewidth=1.6, zorder=4)

    if marker_values:
        for i, c in enumerate(cats):
            v = marker_values.get(c)
            if v is None or not np.isfinite(v):
                continue
            if marker_style == "star":
                # Open star: reads as a distinct annotation against a filled
                # violin, where a dash would be mistaken for the median.
                ax.scatter(i, v, marker="*", s=120, facecolor="white",
                           edgecolor="black", linewidths=1.0, zorder=6)
            else:
                ax.plot([i - 0.30, i + 0.30], [v, v], color="#111111", lw=1.6,
                        zorder=6)

    if point_overlay is not None and overlay_value_col and overlay_group_col:
        rng = np.random.default_rng(0)
        for i, c in enumerate(cats):
            sub = point_overlay[point_overlay[overlay_group_col] == c]
            if not len(sub):
                continue
            ax.scatter(i + (rng.random(len(sub)) - 0.5) * 0.28,
                       sub[overlay_value_col].values, s=16, facecolor="white",
                       edgecolor="#333333", linewidth=0.6, zorder=5)

    if hline is not None:
        ax.axhline(hline, color="#666666", ls=":", lw=0.9)
    if log_y:
        ax.set_yscale("log")

    if ref_line is not None:
        ax.axhline(ref_line, color=ref_color, linewidth=ref_lw, linestyle=ref_ls,
                   zorder=1)
        if ref_label:
            ax.text(len(cats) - 0.55, ref_line + 0.25, ref_label, fontsize=7,
                    color="#777777", ha="right", va="bottom")

    ytop = None
    if bracket_vs is not None and bracket_targets:
        from scipy.stats import mannwhitneyu
        i_ref = cats.index(bracket_vs)
        vals = {c: data[cats.index(c)] for c in cats}
        base = (bracket_base if bracket_base is not None else
                max(np.percentile(v, 99.5) for v in vals.values() if v.size)
                + bracket_base_pad)
        step, tickh = bracket_step, bracket_tick
        for k, g in enumerate(bracket_targets):
            if g not in cats:
                continue
            i_g, y = cats.index(g), base + k * step
            _, pv = mannwhitneyu(vals[bracket_vs], vals[g],
                                 alternative="two-sided")
            note = bracket_stars_fn(pv)
            if bracket_gt_counts and g in bracket_gt_counts:
                n_gt, n_tot = bracket_gt_counts[g]
                note = f"{note}  > in {n_gt}/{n_tot}"
            ax.plot([i_g, i_g, i_ref, i_ref],
                    [y, y + tickh, y + tickh, y], color="#333333",
                    linewidth=0.9)
            ax.text(i_ref + bracket_dx, y + tickh, note, ha="left",
                    va="center", fontsize=bracket_fs)
        ytop = base + len(bracket_targets) * step + tickh

    if series_groups is not None:
        handles = [plt.Line2D([], [], color=scolors.get(s, violin_color), lw=5,
                              alpha=0.8, label=str(s)) for s in series]
        if marker_values:
            handles.append(plt.Line2D([], [], color="#111111", lw=1.6,
                                      label="wild type"))
        ax.legend(handles=handles, frameon=False, loc="upper right",
                  fontsize=F["legend"])

    # With brackets on, leave a full slot to the right so the annotations are
    # not clipped — the source figures use (-0.6, 4.6) for four groups.
    ax.set_xlim(*(xlim if xlim is not None else
                  ((-0.6, len(cats) + 0.6) if bracket_vs is not None
                   else (-0.6, len(cats) - 0.4))))
    ax.set_xticks(pos)
    tick = [protein_label(c) if str(c).lower() in PROTEIN_KEYS else str(c)
            for c in cats]
    if show_n:
        tick = [f"{t}\n(n={d.size:,})" for t, d in zip(tick, data)]
    if n_below:
        ax.set_xticklabels([protein_label(c) if str(c).lower() in PROTEIN_KEYS
                            else str(c) for c in cats], fontsize=F["tick"])
        for i, d in enumerate(data):
            ax.annotate(f"n={d.size:,}", xy=(i, 0),
                        xycoords=("data", "axes fraction"), xytext=(0, -15),
                        textcoords="offset points", ha="center", va="top",
                        fontsize=6.5, color="#333333", annotation_clip=False)
    else:
        ax.set_xticklabels(tick, rotation=45, ha="right", fontsize=F["tick"])
    if ylim is not None:
        ax.set_ylim(*ylim)
    elif ytop is not None:
        # Pooled percentile, not the minimum of the per-group percentiles.
        pooled = np.concatenate([d for d in data if d.size])
        lo = (ylim_lo if ylim_lo is not None
              else np.percentile(pooled, ylim_lo_pct) - ylim_lo_pad)
        ax.set_ylim(lo, ytop + ylim_top_pad)
    ax.set_ylabel(ylabel, fontsize=F["axis"])
    if tiers:
        # Dividers on the boundaries between tiers, and one bold caption per
        # tier in its own colour, anchored in axes fraction so it stays at the
        # visual top of the panel.
        edge = 0
        for _, members in list(tiers)[:-1]:
            edge += len(members)
            ax.axvline(edge - 0.5, color="0.85", lw=0.9, zorder=0)
        for caption, members in tiers:
            idxs = [cats.index(m) for m in members if m in cats]
            if not idxs:
                continue
            ax.text((idxs[0] + idxs[-1]) / 2, 1.02, caption, ha="center",
                    va="bottom", fontsize=tier_fs, fontweight="bold",
                    transform=ax.get_xaxis_transform(),
                    color=(tier_colors or {}).get(caption, "#111111"))

    if y_percent:
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _p: f"{v * 100:.0f}%"))

    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    if ax_strip is not None:
        colors = dict(class_colors or BUFFERING_COLORS)
        order = list(class_order or [c for c in BUFFERING_ORDER
                                     if c in class_frac.columns])
        base = np.zeros(len(cats))
        for cls in order:
            v = class_frac.reindex(cats)[cls].fillna(0.0).values
            ax_strip.bar(pos, v, bottom=base, width=widths,
                         color=colors.get(cls, "#999999"), edgecolor="white",
                         linewidth=0.3, label=cls)
            base = base + v
        ax_strip.set_xlim(-0.6, len(cats) - 0.4)
        ax_strip.set_ylim(0, 1)
        ax_strip.set_xticks([])
        ax_strip.set_yticks([0, 0.5, 1])
        ax_strip.set_yticklabels(["0", "", "1"], fontsize=F["tick"] - 2)
        ax_strip.set_ylabel("fraction", fontsize=F["axis"] - 3)
        for s in ("top", "right"):
            ax_strip.spines[s].set_visible(False)
        ax_strip.legend(frameon=False, ncol=len(order),
                        bbox_to_anchor=(0.0, 1.02), loc="lower left",
                        fontsize=F["legend"])
        if title:
            ax_strip.set_title(title, loc="left", pad=26, fontsize=F["title"])
    elif title:
        ax.set_title(title, loc="left", fontsize=F["title"],
                     **({"pad": title_pad} if title_pad else {}))
    if tick_labelsize is not None:
        ax.tick_params(axis="both", labelsize=tick_labelsize)
    if spine_lw is not None:
        for s in ("left", "bottom"):
            ax.spines[s].set_linewidth(spine_lw)
    if subplots_adjust:
        fig.subplots_adjust(**subplots_adjust)
    if tight_layout:
        fig.tight_layout()
    return fig, np.array([a for a in (ax_strip, ax) if a is not None],
                         dtype=object)


def plot_paired_score_violins(
    arrays: Mapping[tuple[str, str], np.ndarray],
    wt: Mapping[tuple[str, str], float],
    thresholds: Mapping[tuple[str, str], float],
    heatmap: Mapping[str, Mapping[str, float]],
    *,
    groups: Sequence[tuple[str, Sequence[str]]],
    group_fill: Mapping[str, str],
    heatmap_rows: Sequence[str],
    heatmap_labels: Mapping[str, str],
    n_standard_rows: int = 2,
    figsize: tuple[float, float] = (16.0, 8.9),
    height_ratios: tuple[float, float] = (1.0, 4.2),
    margins: tuple[float, float, float, float] = (0.20, 0.98, 0.10, 0.90),
    stride: float = 1.25,
    half: float = 0.27,
    width: float = 0.50,
    xtick_fmt: str = "{label}\n(n={n:,})",
    trim_pct: tuple[float, float] | None = None,
    ylim_pct: tuple[float, float] = (0.5, 99.5),
    ylabel: str = "Standard-adjusted abundance score (log)",
    low_legend_label: str = "Low (< Synonymous 2.5th pct)",
    bracket_labels: tuple[str, str] = ("Standard\nscale", "WT-relative\nscale"),
    legend_y: float = -0.18,
    fs: Mapping[str, float] | None = None,
) -> tuple[Figure, Axes, Axes]:
    """Paired control/HSP90i score violins under a composition heatmap.

    Verbatim port of ``plot_violin`` in
    ``scripts/plot_hsp90_score_violin_all_missense.py`` (the generalised
    sibling of ``plot_hsp90_score_violin.py``, which hard-codes what this one
    derives from ``groups``). Used for both 4D and S4A, which differ only in
    cohort, figure size and tick label.

    The figure is two stacked axes sharing x, so the heatmap's column centres
    are locked to the violins' protein positions. Per protein there are two
    violins — control on the left in the tier colour, HSP90i on the right in a
    55% lightened version, so treatment reads as shade and dependence tier as
    hue. Two annotations sit on each violin:

    * a **hatched sub-polygon** covering the part of the violin below that
      cell's synonymous-WT 2.5th-percentile threshold, clipped out of the drawn
      polygon itself rather than drawn as a band, so the textured area is
      literally the "low"-classified mass;
    * a **horizontal WT line** spanning the violin's true lateral extent at the
      wild-type score, interpolated from the body polygon so it shrinks in the
      tails instead of overhanging.

    The WT line is what makes the panel readable: a distribution sitting far
    from zero but on top of its own WT dash has no variant-specific effect.

    Args:
        arrays: ``(protein, {"ctrl"|"hsp90i"}) -> scores`` in linear units;
            log10 is taken here.
        wt: ``(protein, treatment) -> wild-type score``, linear units.
        thresholds: ``(protein, treatment) -> low cutoff``, linear units.
        heatmap: ``protein -> {row_key: fraction}``, plus ``_n`` for the tick
            label. Non-finite cells are left blank.
        trim_pct: Percentile window each series is trimmed to *before* the KDE
            is computed, e.g. ``(1, 99)``. ``violinplot`` evaluates its kernel
            over the full min-to-max of the data, so a handful of extreme
            variants drags a thin sliver of density out across the panel and
            every violin reads as long-tailed. Trimming bounds the drawn support
            to the bulk. Note the consequence: the hatched "low" region is
            derived from the drawn polygon, so trimming makes it a qualitative
            marker of where the threshold cuts rather than a faithful area — the
            exact fractions are in the heatmap above. ``None`` keeps the full
            range.
        ylim_pct: Percentiles of the pooled data setting the y-range.
        groups: ``(tier label, [protein, ...])`` left to right. Tier dividers,
            heatmap group gaps and the tier captions are all derived from this.
        group_fill: tier label -> colour.
        heatmap_rows: Row keys top to bottom; the first ``n_standard_rows`` are
            bracketed as the absolute scale and the rest as WT-relative.
        fs: Font-size overrides; keys ``xtick``, ``ytick``, ``ylabel``,
            ``cell``, ``row``, ``bracket``, ``group``, ``legend``.

    Returns:
        ``(fig, ax_violin, ax_heatmap)``.
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Polygon, Rectangle
    import matplotlib.ticker as mticker

    F = {"xtick": 15.0, "ytick": 15.0, "ylabel": 15.0, "cell": 12.0,
         "row": 13.0, "bracket": 13.0, "group": 15.0, "legend": 15.0}
    F.update(fs or {})
    proteins = [p for _, ps in groups for p in ps]
    group_of = {p: g for g, ps in groups for p in ps}
    L, R, B, T = margins

    plt.rcParams["hatch.linewidth"] = 0.7
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 1, height_ratios=list(height_ratios), hspace=0.02,
                          left=L, right=R, top=T, bottom=B)
    ax_top = fig.add_subplot(gs[0])
    ax = fig.add_subplot(gs[1], sharex=ax_top)

    def _trim(v, pct):
        """Restrict a series to a percentile window, for the KDE's support."""
        if pct is None or v.size == 0:
            return v
        lo, hi = np.nanpercentile(v, pct[0]), np.nanpercentile(v, pct[1])
        out = v[(v >= lo) & (v <= hi)]
        return out if out.size > 1 else v

    pos_ctrl, pos_hsp, ser_ctrl, ser_hsp, fill_ctrl, fill_hsp = [], [], [], [], [], []
    for i, p in enumerate(proteins, start=1):
        center = i * stride
        pos_ctrl.append(center - half)
        pos_hsp.append(center + half)
        ser_ctrl.append(_trim(np.log10(arrays[(p, "ctrl")]), trim_pct))
        ser_hsp.append(_trim(np.log10(arrays[(p, "hsp90i")]), trim_pct))
        tier = mcolors.to_rgba(group_fill[group_of[p]])
        fill_ctrl.append(tier)
        # Lighten toward white for HSP90i: dark = control, light = inhibited.
        fill_hsp.append(tuple(c + (1.0 - c) * 0.55 for c in tier[:3]) + (tier[3],))

    def _draw(positions, series, fills):
        # showmedians=False: the WT line is the only horizontal annotation, so
        # a median tick would be read as the WT marker.
        parts = ax.violinplot(series, positions=positions, showmeans=False,
                              showmedians=False, showextrema=False, widths=width)
        for body, fc in zip(parts["bodies"], fills):
            body.set_facecolor(fc)
            body.set_edgecolor("black")
            body.set_linewidth(0.6)
            body.set_alpha(0.92)
        return parts

    parts_ctrl = _draw(pos_ctrl, ser_ctrl, fill_ctrl)
    parts_hsp = _draw(pos_hsp, ser_hsp, fill_hsp)

    def _edges(body, center_x):
        """Violin polygon split into (right, left) edges, each sorted by y."""
        verts = body.get_paths()[0].vertices
        right = verts[verts[:, 0] > center_x]
        left = verts[verts[:, 0] < center_x]
        if len(right) == 0 or len(left) == 0:
            return None, None
        return right[np.argsort(right[:, 1])], left[np.argsort(left[:, 1])]

    def _halfwidth_at(body, y_target, center_x):
        """Lateral half-width of the violin at ``y_target``; 0 if outside it."""
        right, left = _edges(body, center_x)
        if right is None:
            return 0.0
        y_lo, y_hi = max(right[0, 1], left[0, 1]), min(right[-1, 1], left[-1, 1])
        if y_target < y_lo or y_target > y_hi:
            return 0.0
        return float(np.interp(y_target, right[:, 1], right[:, 0])
                     - np.interp(y_target, left[:, 1], left[:, 0])) / 2.0

    def _sub_polygon_below(body, y_threshold, center_x):
        """Closed sub-polygon of the violin below ``y_threshold``, flat-topped."""
        right, left = _edges(body, center_x)
        if right is None:
            return None
        y_lo, y_hi = max(right[0, 1], left[0, 1]), min(right[-1, 1], left[-1, 1])
        if y_threshold <= y_lo:
            return None
        if y_threshold >= y_hi:
            return np.concatenate([right, left[::-1]])
        r_cross = float(np.interp(y_threshold, right[:, 1], right[:, 0]))
        l_cross = float(np.interp(y_threshold, left[:, 1], left[:, 0]))
        return np.array(
            list(right[right[:, 1] < y_threshold].tolist())
            + [[r_cross, y_threshold], [l_cross, y_threshold]]
            + list(left[left[:, 1] < y_threshold][::-1].tolist()))

    # Hatch the "low" region of every violin. facecolor="none" + lw=0 keeps the
    # violin's own colour showing through, so only the texture is added.
    for which, parts, positions in (("ctrl", parts_ctrl, pos_ctrl),
                                    ("hsp90i", parts_hsp, pos_hsp)):
        for i, p in enumerate(proteins):
            thr = thresholds.get((p, which), np.nan)
            wt_v = wt.get((p, which), np.nan)
            if not (np.isfinite(thr) and thr > 0):
                continue
            # The cutoff should always sit below WT (2.5th pct of syn-WT <
            # median syn-WT ~ WT); skip if a noisy WT row inverts that.
            if np.isfinite(wt_v) and thr >= wt_v:
                continue
            sub = _sub_polygon_below(parts["bodies"][i], float(np.log10(thr)),
                                     positions[i])
            if sub is None or len(sub) < 3:
                continue
            ax.add_patch(Polygon(sub, closed=True, facecolor="none",
                                 edgecolor="black", linewidth=0.0,
                                 hatch="///", zorder=2.5))

    for i, p in enumerate(proteins, start=1):
        center = i * stride
        for tx, pos, body in (("ctrl", center - half, parts_ctrl["bodies"][i - 1]),
                              ("hsp90i", center + half, parts_hsp["bodies"][i - 1])):
            v = wt.get((p, tx), np.nan)
            if not (np.isfinite(v) and v > 0):
                continue
            y = float(np.log10(v))
            hw = _halfwidth_at(body, y, pos)
            if hw > 0:
                ax.hlines(y, pos - hw, pos + hw, color="black", linewidth=1.4,
                          zorder=6)

    # Tier dividers, derived from group sizes so membership changes track.
    edge = 0
    for _, ps in groups[:-1]:
        edge += len(ps)
        ax.axvline((edge + 0.5) * stride, color="0.85", lw=0.9, zorder=0)

    ax.set_xticks(np.arange(1, len(proteins) + 1) * stride)
    ax.set_xticklabels([xtick_fmt.format(label=protein_label(p),
                                        n=int(heatmap[p].get("_n", 0)))
                        for p in proteins], fontsize=F["xtick"])
    ax.set_xlim(0.4 * stride, (len(proteins) + 0.6) * stride)

    pooled = np.concatenate(ser_ctrl + ser_hsp)
    y_lo = float(np.nanpercentile(pooled, ylim_pct[0]))
    y_hi = float(np.nanpercentile(pooled, ylim_pct[1]))
    pad = (y_hi - y_lo) * 0.06
    ax.set_ylim(y_lo - pad, y_hi + pad)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(1.0))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(
        lambda v, _p: f"$10^{{{int(round(v))}}}$"
        if abs(v - round(v)) < 1e-6 else ""))
    ax.yaxis.set_minor_locator(mticker.FixedLocator(
        [np.log10(b) + d for d in range(int(np.floor(y_lo)) - 1,
                                        int(np.ceil(y_hi)) + 1)
         for b in range(2, 10)]))
    ax.tick_params(axis="y", labelsize=F["ytick"])
    ax.set_ylabel(ylabel, fontsize=F["ylabel"])

    # --- composition heatmap ------------------------------------------------
    rows = list(heatmap_rows)
    data = np.array([[heatmap[p].get(k, np.nan) for p in proteins] for k in rows],
                    dtype=float)
    n_rows, n_cols = len(rows), len(proteins)
    wtrel_n = n_rows - n_standard_rows
    V_GAP = 0.18
    total_h = n_rows + V_GAP
    standard_y = (0.0, float(n_standard_rows))
    wtrel_y = (n_standard_rows + V_GAP, n_standard_rows + V_GAP + wtrel_n)
    cmap = plt.get_cmap("viridis")

    # Drawn as explicit rectangles, not imshow, so the inter-cell gaps can be
    # painted: a small gap inside a tier and a 3x gap between tiers. The
    # horizontal gap is converted through the axes' physical size so it renders
    # at the same *width* as V_GAP renders in height — data-x and data-y do not
    # occupy the same physical size on this wide, short axes.
    ax_w_in = (R - L) * figsize[0]
    ax_h_in = (T - B) * figsize[1] * (height_ratios[0] / sum(height_ratios))
    h_gap_small = V_GAP * (((n_cols * stride) / ax_w_in) / (total_h / ax_h_in))
    h_gap_large = 3.0 * h_gap_small
    group_end = set(np.cumsum([len(ps) for _, ps in groups])[:-1] - 1)

    def _cell_lr(c):
        center = (c + 1) * stride
        right = (h_gap_small / 2 if c == n_cols - 1 else
                 h_gap_large / 2 if c in group_end else h_gap_small / 2)
        left = (h_gap_small / 2 if c == 0 else
                h_gap_large / 2 if (c - 1) in group_end else h_gap_small / 2)
        return center - 0.5 * stride + left, center + 0.5 * stride - right

    row_centers = ([r + 0.5 for r in range(n_standard_rows)]
                   + [n_standard_rows + V_GAP + (r - n_standard_rows) + 0.5
                      for r in range(n_standard_rows, n_rows)])
    for r in range(n_rows):
        y_top = (standard_y[0] + r if r < n_standard_rows
                 else wtrel_y[0] + (r - n_standard_rows))
        for c in range(n_cols):
            v = data[r, c]
            if not np.isfinite(v):
                continue
            lx, rx = _cell_lr(c)
            ax_top.add_patch(Rectangle((lx, y_top), rx - lx, 1.0,
                                       facecolor=cmap(v), edgecolor="none"))
            # Perceptual luminance picks the text colour so labels stay legible
            # across the whole viridis range.
            rgba = cmap(v)
            lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            ax_top.text((c + 1) * stride, row_centers[r], f"{v:.2f}",
                        ha="center", va="center", fontsize=F["cell"],
                        color="white" if lum < 0.55 else "black")

    ax_top.set_xlim(0.5 * stride, (n_cols + 0.5) * stride)
    ax_top.set_ylim(total_h, 0.0)
    ax_top.set_yticks(row_centers)
    ax_top.set_yticklabels([heatmap_labels[k] for k in rows], fontsize=F["row"])
    ax_top.tick_params(axis="y", which="both", left=False, pad=2)
    ax_top.tick_params(axis="x", which="both", bottom=False, top=False,
                       labelbottom=False)
    for sp in ("top", "right", "left", "bottom"):
        ax_top.spines[sp].set_visible(False)

    # Scale brackets, outside the row-label gutter. x in axes fraction, y in
    # data, so they stay attached to the rows once the group gap shifts them.
    blended = ax_top.get_yaxis_transform()
    for label, (y0, y1) in zip(bracket_labels, (standard_y, wtrel_y)):
        ax_top.plot([-0.125, -0.125], [y0, y1], transform=blended, color="black",
                    linewidth=1.4, clip_on=False, solid_capstyle="butt")
        ax_top.text(-0.140, (y0 + y1) / 2, label, transform=blended,
                    ha="right", va="center", fontsize=F["bracket"])

    for label, ps in groups:
        idxs = [proteins.index(p) + 1 for p in ps]
        ax_top.text((idxs[0] + idxs[-1]) / 2 * stride, 1.05, label,
                    ha="center", va="bottom", fontsize=F["group"],
                    fontweight="bold", transform=ax_top.get_xaxis_transform(),
                    color=group_fill[label])

    # Neutral grey/black swatches rather than one tier's colour, so the
    # dark = control / light = HSP90i convention reads independent of palette.
    handles = [
        Patch(facecolor="#4d4d4d", edgecolor="black", label="DMSO"),
        Patch(facecolor="#bfbfbf", edgecolor="black", label="HSP90i"),
        Patch(facecolor="#bfbfbf", edgecolor="black", hatch="///",
              label=low_legend_label),
        Line2D([0], [0], color="black", linewidth=1.4, label="WT"),
    ]
    ax.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
              bbox_to_anchor=(0.5, legend_y), fontsize=F["legend"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig, ax, ax_top


def plot_donut(
    outer_sizes: Sequence[float],
    outer_colors: Sequence,
    outer_labels: Sequence[str],
    *,
    inner_sizes: Sequence[float] = (),
    inner_colors: Sequence = (),
    inner_labels: Sequence[str] = (),
    figsize: tuple[float, float] = (6.5, 6.5),
    outer_width: float = 0.32,
    outer_label_r: float = 0.84,
    inner_radius: float = 0.66,
    inner_width: float = 0.22,
    edge_lw: float = 1.2,
    startangle: float = 90.0,
    center_label: str = "",
    inner_rotate: bool = True,
    inner_bold: bool = True,
    fs: Mapping[str, float] | None = None,
) -> tuple[Figure, Axes]:
    """Two-ring donut: an outer ring of slices inside an inner ring of groups.

    Port of ``plot_donut`` in ``scripts/plot_hsp90_client_donut.py``. The outer
    ring is drawn first so the inner ring's white edge sits cleanly on top of
    it. Outer labels go at ``outer_label_r``; inner labels sit at the inner
    band's mid-radius, rotated *tangentially* (long axis along the arc) and
    flipped in the lower half so they never read upside-down.

    Ring alignment is the caller's job: pass outer sizes that already sum,
    within each group, to that group's inner-ring span, so the two rings are
    concentric. The notebook does that arithmetic — see Fig 4A.

    Args:
        outer_sizes: Angular size per outer slice, in the drawing order.
        outer_colors: One colour per outer slice.
        outer_labels: One label per outer slice (``""`` to leave a slice bare).
        inner_sizes / inner_colors / inner_labels: The same, for the inner
            ring. Empty draws a single-ring donut.
        center_label: Text for the hole; typically the overall total.
        fs: Font-size overrides; keys ``outer``, ``inner``, ``center``.
    """
    F = {"outer": 12.0, "inner": 12.0, "center": 17.0}
    F.update(fs or {})
    fig, ax = plt.subplots(figsize=figsize)

    outer_wedges, _ = ax.pie(
        list(outer_sizes), radius=1.0, startangle=startangle,
        colors=list(outer_colors),
        wedgeprops={"width": outer_width, "edgecolor": "white",
                    "linewidth": edge_lw})

    inner_wedges = []
    if len(inner_sizes):
        inner_wedges, _ = ax.pie(
            list(inner_sizes), radius=inner_radius, startangle=startangle,
            colors=list(inner_colors),
            wedgeprops={"width": inner_width, "edgecolor": "white",
                        "linewidth": edge_lw})

    for w, label in zip(outer_wedges, outer_labels):
        if not label:
            continue
        a = np.deg2rad((w.theta1 + w.theta2) / 2)
        ax.text(outer_label_r * np.cos(a), outer_label_r * np.sin(a), label,
                ha="center", va="center", fontsize=F["outer"], color="black")

    r_in = inner_radius - inner_width / 2
    for w, label in zip(inner_wedges, inner_labels):
        if not label:
            continue
        angle = (w.theta1 + w.theta2) / 2
        a = np.deg2rad(angle)
        rot = 0.0
        if inner_rotate:
            # Tangential = radial angle - 90 deg, normalised to [-180, 180],
            # then flipped 180 deg in the lower half so text stays upright.
            rot = ((angle - 90) + 180) % 360 - 180
            if rot > 90 or rot < -90:
                rot += 180
        ax.text(r_in * np.cos(a), r_in * np.sin(a), label, ha="center",
                va="center", fontsize=F["inner"],
                fontweight="bold" if inner_bold else "normal", color="black",
                rotation=rot, rotation_mode="anchor")

    if center_label:
        ax.text(0, 0, center_label, ha="center", va="center",
                fontsize=F["center"], color="black")
    ax.set(aspect="equal")
    fig.tight_layout()
    return fig, ax

def plot_univariate_profiles(
    profiles: Mapping[tuple, pd.DataFrame],
    edges: Mapping[str, Sequence[float]],
    *,
    predictors: Sequence[Mapping],
    categories: Sequence[tuple[str, str, str]],
    proteins: Sequence[str] = (),
    ylabel: str = "Proportion of kinase-domain missense",
    row_ylabel_fmt: str = "{label} missense\nproportion",
    suptitle: str = "",
    fs: Mapping[str, float] | None = None,
) -> tuple[Figure, Axes]:
    """Binned fraction-in-class profiles over predictors, with count strips.

    Verbatim port of ``plot_global`` / ``plot_per_protein`` in
    ``scripts/plot_hsp90_univariate_simplified_definitions.py``. One column per
    predictor; each cell is a profile over uniform-width bins with a grey
    per-bin count strip beneath it sharing the x-axis. The strip is what makes
    the profile honest — a fraction from a bin holding twelve variants looks
    identical to one from twelve hundred until you can see the n.

    Bins are uniform width rather than quantile: the *shape* of the
    relationship is the finding (the ddG profile is an inverted U), and quantile
    bins would flatten it.

    Args:
        profiles: ``(protein_or_None, predictor_col) -> frame`` with columns
            ``bin_center``, ``n`` and ``frac_<key>`` per category. Empty frames
            hide their cell.
        edges: predictor column -> bin edges, shared down a column so panels
            align horizontally.
        predictors: One mapping per column with ``col``, ``label`` keys.
        categories: ``(key, label, colour)`` per line, in legend order.
        proteins: One row per protein for the per-protein grid; empty gives the
            single pooled row with a legend in the first panel.
        row_ylabel_fmt: Row label for the per-protein grid; ``{label}`` is the
            display name.
        fs: Font-size overrides; keys ``axis``, ``tick``, ``hist_axis``,
            ``hist_tick``, ``legend``, ``row``, ``suptitle``.

    Returns:
        ``(fig, first_axes)``.
    """
    from matplotlib.lines import Line2D
    import matplotlib.ticker as mticker

    F = {"axis": 17.0, "tick": 14.0, "hist_axis": 15.0, "hist_tick": 13.0,
         "legend": 16.0, "row": 15.0, "suptitle": 15.0}
    F.update(fs or {})
    per_protein = bool(len(proteins))
    rows = list(proteins) if per_protein else [None]
    n_rows, n_cols = len(rows), len(predictors)

    def _round_ceiling(v: float) -> int:
        """Round up to one significant figure, for the strip's single top tick."""
        if v <= 0:
            return 1
        mag = 10 ** int(np.floor(np.log10(v)))
        return int(np.ceil(v / mag) * mag)

    if per_protein:
        fig = plt.figure(figsize=(4.7 * n_cols, 2.5 * n_rows))
        outer = fig.add_gridspec(n_rows, n_cols, hspace=0.45, wspace=0.36,
                                 top=0.955, bottom=0.05)
    else:
        fig = plt.figure(figsize=(5.8 * n_cols, 6.4))
        outer = fig.add_gridspec(1, n_cols, wspace=0.32)

    first = None
    for r, prot in enumerate(rows):
        for c, pcfg in enumerate(predictors):
            prof = profiles.get((prot, pcfg["col"]))
            e = np.asarray(edges[pcfg["col"]], dtype=float)
            inner = outer[r, c].subgridspec(
                2, 1, height_ratios=[4, 1],
                hspace=0.08 if per_protein else 0.05)
            ax = fig.add_subplot(inner[0])
            ax_h = fig.add_subplot(inner[1], sharex=ax)
            first = first or ax
            if prof is None or prof.empty:
                ax.set_visible(False)
                ax_h.set_visible(False)
                continue

            for key, label, colour in categories:
                ax.plot(prof["bin_center"], prof[f"frac_{key}"], "-o",
                        color=colour, markersize=4, lw=1.5, label=label)
            ax.set_ylim(0.0, 1.0)
            if per_protein:
                # The strip's top count label sits right under the profile's
                # "0.0" in a cell this small, so drop that one tick label.
                ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
                ax.set_yticklabels(["", "0.2", "0.4", "0.6", "0.8", "1.0"])
            if c == 0:
                ax.set_ylabel(
                    row_ylabel_fmt.format(label=protein_label(prot))
                    if per_protein else ylabel,
                    fontsize=F["row"] if per_protein else F["axis"],
                    fontweight="bold" if per_protein else "normal")
            ax.tick_params(labelsize=F["tick"])
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            if not per_protein and c == 0:
                ax.legend(loc="upper right", frameon=False, fontsize=F["legend"])
            plt.setp(ax.get_xticklabels(), visible=False)

            widths = np.diff(e)
            centers = prof["bin_center"].to_numpy()
            counts = prof["n"].to_numpy()
            # Match each surviving centre to its own bin width: empty bins may
            # have been dropped, so positional indexing would misalign.
            bw = np.array([widths[int(np.searchsorted(e, x, side="right") - 1)]
                           for x in centers])
            ax_h.bar(centers, counts, width=bw * 0.95, color="0.55",
                     edgecolor="white", linewidth=0.4)
            last_row = (r == n_rows - 1)
            if last_row:
                ax_h.set_xlabel(pcfg["label"], fontsize=F["axis"])
            if c == 0:
                ax_h.set_ylabel("n", fontsize=F["hist_axis"])
            ax_h.tick_params(axis="y", labelsize=F["hist_tick"])
            ax_h.tick_params(axis="x", labelsize=F["tick"])
            for sp in ("top", "right"):
                ax_h.spines[sp].set_visible(False)
            if counts.size and per_protein:
                top = _round_ceiling(float(counts.max()))
                ax_h.set_ylim(0, top)
                ax_h.set_yticks([0, top])
            elif counts.size:
                ax_h.set_ylim(0, counts.max() * 1.10)
                ax_h.yaxis.set_major_locator(
                    mticker.MaxNLocator(nbins=3, integer=True))
            else:
                ax_h.set_ylim(0, 1)
            if per_protein:
                ax.set_xlim(e[0], e[-1])
                ax_h.set_xlim(e[0], e[-1])

    if per_protein:
        # One shared legend in the top margin, so it cannot overlap a panel.
        handles = [Line2D([0], [0], color=col, marker="o", markersize=6, lw=2,
                          label=lab) for _k, lab, col in categories]
        fig.legend(handles=handles, loc="lower center", ncol=len(categories),
                   bbox_to_anchor=(0.5, 0.957), frameon=False,
                   fontsize=F["legend"], columnspacing=1.6, handletextpad=0.5)
    if suptitle:
        fig.suptitle(suptitle, fontsize=F["suptitle"],
                     y=0.995 if per_protein else 1.00)
    return fig, first


def plot_density_panel(
    series: Mapping[str, Sequence[float]],
    *,
    colors: Mapping[str, str],
    marker_lines: Mapping[str, float] = {},
    xlabel: str = "",
    ylabel: str = "Density",
    xlim: tuple[float, float] | None = None,
    figsize: tuple[float, float] = (11.0, 5.0),
    bw_adjust: float = 1.0,
    grid_n: int = 512,
    fill_alpha: float = 0.18,
    lw: float = 1.8,
    ref_line: float | None = None,
    marker_color: str = "#444444",
    marker_fs: float = 9.0,
    n_label_rows: int = 4,
    x_percent: bool = False,
    legend_fmt: str = "{label} (n={n:,})",
    fs: Mapping[str, float] | None = None,
) -> tuple[Figure, Axes]:
    """Smoothed density curves over one axis, with labelled marker lines.

    Built for the dependence panel: the distributions of a per-variant change
    overlaid by variant type, with a dotted vertical line per protein at where
    its WILD TYPE sits. The wild-type lines are the point of the panel — a
    distribution far from zero says nothing on its own if its own wild type has
    moved just as far, so the reader needs both in one frame.

    Densities are Gaussian KDEs rather than histograms so three overlapping
    distributions stay readable; each is drawn as a line with a light fill.

    Marker labels are dealt out over ``n_label_rows`` staggered heights in
    ascending x order, which keeps neighbouring protein names from colliding
    without hand-placing any of them.

    Args:
        series: label -> values. Non-finite values are dropped per series.
        colors: label -> colour.
        marker_lines: name -> x position for the dotted vertical lines.
        ref_line: x for a neutral reference (no change), drawn behind the rest.
        n_label_rows: How many stacked heights to spread marker labels across.
        x_percent: Format x ticks as percentages.
        fs: Font-size overrides; keys ``axis``, ``tick``, ``legend``.
    """
    from scipy import stats as _st

    F = {"axis": 12.0, "tick": 11.0, "legend": 11.0}
    F.update(fs or {})
    clean = {k: np.asarray([v for v in vals if np.isfinite(v)], dtype=float)
             for k, vals in series.items()}
    clean = {k: v for k, v in clean.items() if v.size > 1}
    if xlim is None:
        allv = np.concatenate(list(clean.values()))
        xlim = (float(np.nanpercentile(allv, 0.5)),
                float(np.nanpercentile(allv, 99.5)))

    fig, ax = plt.subplots(figsize=figsize)
    if ref_line is not None:
        ax.axvline(ref_line, ls=":", color="0.45", lw=0.9, zorder=1)

    grid = np.linspace(xlim[0], xlim[1], grid_n)
    for label, vals in clean.items():
        kde = _st.gaussian_kde(vals)
        kde.set_bandwidth(kde.factor * bw_adjust)
        y = kde(grid)
        col = colors.get(label, "#666666")
        ax.plot(grid, y, color=col, lw=lw, zorder=4,
                label=legend_fmt.format(label=label, n=int(vals.size)))
        ax.fill_between(grid, y, color=col, alpha=fill_alpha, lw=0, zorder=2)

    ax.set_xlim(*xlim)
    ax.set_ylim(bottom=0.0)
    top = ax.get_ylim()[1]
    # Stagger the labels over a few heights, in x order, so adjacent proteins
    # never write over one another.
    for rank, (name, x) in enumerate(sorted(marker_lines.items(),
                                            key=lambda kv: kv[1])):
        if not np.isfinite(x):
            continue
        ax.axvline(x, ls=":", color=marker_color, lw=0.9, zorder=3)
        y = top * (0.97 - 0.075 * (rank % n_label_rows))
        ax.text(x, y, f" {name}", ha="left", va="top", fontsize=marker_fs,
                color=marker_color, rotation=90, zorder=5)

    if x_percent:
        ax.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _p: f"{v * 100:.0f}%"))
    ax.set_xlabel(xlabel, fontsize=F["axis"])
    ax.set_ylabel(ylabel, fontsize=F["axis"])
    ax.tick_params(labelsize=F["tick"])
    ax.legend(frameon=False, fontsize=F["legend"], loc="upper left")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    return fig, ax


def plot_beta_heatmap(
    betas: pd.DataFrame,
    importance: Mapping[str, float],
    *,
    col_order: Sequence[str],
    feature_labels: Mapping[str, str],
    feature_groups: Sequence[tuple[str, Sequence[tuple[str, str]]]],
    group_colors: Mapping[str, str],
    cell_in: float = 0.12,
    cbar_label: str = "β (log-odds)\nBuffered vs WT-like",
    imp_label: str = "mean R²",
    imp_ticks: tuple[float, float] = (0.0, 0.15),
    robust_pct: float = 98.0,
    fs: Mapping[str, float] | None = None,
) -> tuple[Figure, Axes]:
    """Per-protein coefficient heatmap with an importance bar panel beside it.

    Verbatim port of ``scripts/plot_hsp90_factor_beta_heatmap.py``. Features run
    down the y-axis ordered by pooled importance, proteins across the x-axis,
    and each cell is that protein's univariate coefficient. Three things sit
    around it: a black bar panel on the left giving each feature's mean R² (so a
    strong-looking coefficient on a worthless feature is visible as such), a
    colour strip on the right binding each feature to its group, and a
    diverging colourbar with triangle caps.

    Cells are laid out at a fixed physical size so they are **square** and the
    group strip aligns exactly with the heatmap rows; the figure size follows
    from the matrix shape rather than the reverse.

    Args:
        betas: Long frame with ``feature``, ``protein``, ``coef``.
        importance: feature -> mean pseudo-R², used for the row order and bars.
        col_order: Protein order across the x-axis.
        feature_labels / feature_groups / group_colors: Display names, the
            ``(group, [(feature, label), ...])`` spec, and one colour per group.
        robust_pct: Percentile of |coef| used for the symmetric colour limit, so
            one extreme protein cannot wash the map out.
    """
    import matplotlib.colors as _mc
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.patches import Patch

    F = {"feature": 5.0, "protein": 6.0, "imp_tick": 4.5, "imp_label": 5.0,
         "cbar": 5.5, "cbar_tick": 5.0, "legend": 5.0, "legend_title": 5.5}
    F.update(fs or {})
    group_of = {f: g for g, feats in feature_groups for f, _l in feats}
    present = set(betas["feature"])
    order = [f for f, _ in sorted(importance.items(), key=lambda kv: -kv[1])
             if f in present]
    r2 = [float(importance.get(f, 0.0)) for f in order]
    M = (betas.pivot_table(index="feature", columns="protein", values="coef")
         .reindex(index=order, columns=list(col_order)).to_numpy(dtype=float))

    vmax = float(np.nanpercentile(np.abs(M), robust_pct))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("0.9")

    n_row, n_col = len(order), len(col_order)
    s = cell_in
    left, xlab, top = 1.20, 0.12, 0.52
    bar_w, gap_b = 0.32, 0.06
    gap1, strip_w, gap2, cbar_w, right = 0.05, 0.15, 0.16, 0.12, 0.18
    data_w, data_h = n_col * s, n_row * s
    data_x = left + bar_w + gap_b
    fig_w = data_x + data_w + gap1 + strip_w + gap2 + cbar_w + right
    fig_h = xlab + data_h + top
    fig = plt.figure(figsize=(fig_w, fig_h))
    rect = lambda x, y, w, h: [x / fig_w, y / fig_h, w / fig_w, h / fig_h]
    y0 = xlab

    axb = fig.add_axes(rect(left, y0, bar_w, data_h))
    axb.barh(range(n_row), r2, color="black", height=0.8)
    axb.set_ylim(n_row - 0.5, -0.5)          # row 0 on top, matching imshow
    axb.set_yticks(range(n_row))
    axb.set_yticklabels([feature_labels.get(f, f) for f in order],
                        fontsize=F["feature"])
    axb.tick_params(length=0)
    axb.set_xlim(0, max(max(r2, default=imp_ticks[1]), imp_ticks[1]) * 1.03)
    axb.set_xticks(list(imp_ticks))
    axb.set_xticklabels([f"{v:g}" for v in imp_ticks])
    axb.xaxis.tick_top()
    axb.xaxis.set_label_position("top")
    axb.tick_params(axis="x", labelsize=F["imp_tick"], length=2)
    axb.set_xlabel(imp_label, fontsize=F["imp_label"])
    for sp in ("bottom", "right", "left"):
        axb.spines[sp].set_visible(False)

    ax = fig.add_axes(rect(data_x, y0, data_w, data_h))
    im = ax.imshow(np.ma.masked_invalid(M), aspect="auto", cmap=cmap, norm=norm)
    ax.set_xticks(range(n_col))
    ax.set_xticklabels(protein_labels(col_order), fontsize=F["protein"],
                       rotation=90)
    ax.xaxis.tick_top()
    ax.set_yticks([])                        # feature names live on the bars
    ax.tick_params(length=0)
    ax.set_xticks(np.arange(-0.5, n_col, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_row, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.4)
    ax.tick_params(which="minor", length=0)

    axg = fig.add_axes(rect(data_x + data_w + gap1, y0, strip_w, data_h))
    axg.imshow(np.array([[_mc.to_rgb(group_colors.get(group_of.get(f), "#999999"))]
                         for f in order]), aspect="auto")
    axg.set_xticks([])
    axg.set_yticks([])
    for sp in axg.spines.values():
        sp.set_visible(False)

    rx = data_x + data_w + gap1 + strip_w + gap2
    cax = fig.add_axes(rect(rx, y0 + data_h * 0.45, cbar_w, data_h * 0.35))
    cbar = fig.colorbar(im, cax=cax, extend="both")
    cbar.set_label(cbar_label, fontsize=F["cbar"])
    cbar.ax.tick_params(labelsize=F["cbar_tick"])

    groups_present = [g for g, _ in feature_groups
                      if any(group_of.get(f) == g for f in order)]
    fig.legend(handles=[Patch(facecolor=group_colors[g], label=g)
                        for g in groups_present],
               loc="upper left",
               bbox_to_anchor=((rx - gap2) / fig_w, (y0 + data_h) / fig_h),
               ncol=1, fontsize=F["legend"], frameon=False, handlelength=1.0,
               handletextpad=0.4, title="Feature group",
               title_fontsize=F["legend_title"])
    return fig, ax

def plot_dn_positions(
    positions_by_type: Mapping[str, Sequence[int]],
    *,
    xlim: tuple[int, int],
    domains: Sequence[tuple[str, int, int, str]] = (),
    text_only_domains: Sequence[tuple[str, int]] = (),
    row_types: Sequence[str] = ("missense", "deletion", "nonsense"),
    row_labels: Sequence[str] = ("Mis", "Del", "Non"),
    dot_size: float = 26.0,
    col_width: float = 0.32,
    domain_alpha: float = 0.18,
    default_color: str = "#8a8f99",
    figsize: tuple[float, float] = (7.5, 1.3),
    margins: Mapping[str, float] | None = None,
    xlabel: str = "Position",
    fs: Mapping[str, float] | None = None,
) -> tuple[Figure, Axes]:
    """DN variants along the primary sequence, horizontal — one row per type.

    Port of ``render_protein_horizontal`` in
    ``scripts/plot_dn_position_distributions.py``: 7.5 x 1.3 in landscape,
    N-terminus on the left, three discrete rows (Mis on top, Del, Non) spaced
    ``col_width`` apart, domains as vertical shaded bands spanning all rows
    with bold labels above, row labels at the left, and a small y-jitter so
    variants at the same position do not completely overlap.

    Dots default to 26 pt^2 rather than the original 14 so they read at panel
    size; pass ``dot_size`` to change it.

    The axes are placed by ``subplots_adjust``, not ``tight_layout``, and the
    original saves **without** ``bbox_inches="tight"`` so every protein's panel
    comes out at identical dimensions and can be stacked. Use
    ``save_figure(..., tight=False)`` to preserve that.

    Args:
        positions_by_type: variant type -> residue positions carrying a DN.
        xlim: ``(first, last)`` residue of the profiled window. Trimming to the
            scanned region rather than the full protein keeps the density
            honest.
        domains: ``(name, start, end, colour)`` shaded bands.
        text_only_domains: ``(name, position)`` italic grey motif labels.
        dot_size: Scatter point area.
        margins: Overrides for ``subplots_adjust``; defaults to the locked
            ``left=0.06, right=0.99, top=0.78, bottom=0.32``.
        fs: Font-size overrides; keys ``domain``, ``motif``, ``row``,
            ``axis``, ``tick``.
    """
    F = {"domain": 10, "motif": 9, "row": 8, "axis": 11, "tick": 9}
    F.update(fs or {})
    M = {"left": 0.06, "right": 0.99, "top": 0.78, "bottom": 0.32}
    M.update(margins or {})

    pmin, pmax = xlim
    span = max(pmax - pmin, 1)
    fig, ax = plt.subplots(figsize=figsize)

    n_rows = len(row_types)
    row_y = {t: (n_rows - 1 - i) * col_width for i, t in enumerate(row_types)}
    y_min = -col_width / 2
    y_max = (n_rows - 1) * col_width + col_width / 2
    label_y = y_max + 0.10

    def _colour_of(pos: int) -> str:
        for name, start, end, colour in domains:
            if start <= pos <= end:
                return colour
        return default_color

    for name, start, end, colour in domains:
        mid = (max(start, pmin) + min(end, pmax)) / 2
        if mid < pmin or mid > pmax:
            continue
        ax.axvspan(start, end, ymin=0.0, ymax=1.0, facecolor=colour,
                   alpha=domain_alpha, zorder=0)
        ax.text(mid, label_y, name.replace(" ", "\n"), ha="center",
                va="bottom", fontsize=F["domain"], color="#333333",
                fontweight="bold")
    for name, mid in text_only_domains:
        if pmin <= mid <= pmax:
            ax.text(mid, label_y, name.replace(" ", "\n"), ha="center",
                    va="bottom", fontsize=F["motif"], color="#888888",
                    style="italic")

    for t in row_types:
        pos = list(positions_by_type.get(t, []))
        if not pos:
            continue
        y = row_y[t]
        ys = [y + ((i % 5) - 2) * 0.015 for i in range(len(pos))]
        ax.scatter(pos, ys, s=dot_size, c=[_colour_of(p) for p in pos],
                   edgecolors="white", linewidths=0.4, alpha=0.92, zorder=3)

    header_x = pmin - 0.02 * span
    for t, lab in zip(row_types, row_labels):
        ax.text(header_x, row_y[t], lab, ha="right", va="center",
                fontsize=F["row"], color="#333333")

    ax.set_xlim(pmin - 0.06 * span, pmax + 0.01 * span)
    ax.set_ylim(y_min - 0.05, y_max + 0.40)
    ax.set_yticks([])
    ax.set_xlabel(xlabel, fontsize=F["axis"])
    ax.set_xticks([tk for tk in ax.get_xticks() if pmin <= tk <= pmax])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="x", labelsize=F["tick"])
    fig.subplots_adjust(**M)
    return fig, ax


def plot_activity_histogram(
    scores: Sequence[float],
    *,
    dn_threshold: float,
    box_distributions: Sequence[tuple[str, Mapping[str, float], str]] = (),
    groups: Sequence[tuple[str, Sequence[float], str]] = (),
    density_groups: Sequence[tuple[str, Sequence[float], str]] = (),
    smooth: bool = False,
    smooth_bw: float | str | None = None,
    dn_color: str = "#C0392B",
    bar_color: str = "#222222",
    title: str = "",
    xlabel: str = "Pathway activity score (WT-normalised)",
    bins: int = 60,
    log_x: bool = True,
    figsize: tuple[float, float] = (3.6, 2.9),
    note: str = "",
    fs: Mapping[str, float] | None = None,
) -> tuple[Figure, np.ndarray]:
    """Activity histogram with the DN threshold and control boxplots beneath.

    New panel (no prior version). Variants below ``dn_threshold`` are drawn in
    the DN colour, the rest dark; a dotted line marks the threshold. Bin edges
    are split exactly at the threshold so the coloured block ends there rather
    than at the nearest bin boundary.

    ``box_distributions`` is a sequence of ``(label, five_number, colour)``
    drawn top-to-bottom in the lower strip on the shared x-axis. The
    five-number mapping uses keys ``p2.5``, ``p25``, ``median``, ``p75``,
    ``p97.5``; missing quartiles collapse the box to a line rather than
    failing, so a distribution known only by its outer percentiles still
    renders.

    ``groups`` switches the bars from DN-vs-not colouring to a **stacked**
    histogram of variant classes, given as ``(label, values, colour)``. The
    threshold line and its annotation stay, and the region below the threshold
    is shaded, so which variants are DN is still readable — but the panel now
    also answers *what kind* of variant sits where on the activity axis, which
    the two-colour version could not. ``scores`` should be the concatenation of
    the group values; it is what the smoothed curve and the counts describe.

    ``density_groups`` takes the same ``(label, values, colour)`` shape but
    replaces the bars with one **area-normalised** smoothed density per class.
    Use it to compare where classes sit on the activity axis: stacked counts are
    dominated by whichever class is most numerous, so a class an order of
    magnitude rarer is invisible. The trade is that the curves no longer carry
    abundance, so class sizes are put in the legend and the per-class fraction
    below the threshold is annotated instead.

    ``smooth`` overlays a Gaussian KDE scaled to counts. It is fitted in the
    axis's own space — log10 when ``log_x``, linear otherwise — so the curve is
    not distorted by the transform. Scaling uses a *uniform* bin width in that
    space rather than the actual (threshold-split, possibly uneven) edges, so
    the curve's area matches the histogram's regardless of where the split
    lands.

    Args:
        scores: Values to histogram.
        dn_threshold: Empty-vector cutoff on the same scale.
        box_distributions: Control distributions for the lower strip.
        groups: ``(label, values, colour)`` per variant class, stacked.
        smooth: Overlay a count-scaled KDE.
        smooth_bw: Bandwidth passed to ``gaussian_kde`` (default: Scott's rule).
        note: Small grey caption, e.g. to record a missing input.
        fs: Font-size overrides; keys ``axis``, ``tick``, ``ann``, ``title``.
    """
    F = {"axis": 8, "tick": 7, "ann": 6.5, "title": 9}
    F.update(fs or {})
    scores = np.asarray([s for s in scores if np.isfinite(s)], dtype=float)
    n_box = max(len(box_distributions), 1)

    fig, axes = plt.subplots(
        2, 1, figsize=figsize, sharex=True,
        gridspec_kw={"height_ratios": [3.2, 0.42 + 0.30 * n_box],
                     "hspace": 0.12})
    ax, axb = axes

    if log_x:
        lo = max(np.nanmin(scores), 1e-3)
        edges = np.logspace(np.log10(lo), np.log10(np.nanmax(scores)), bins + 1)
    else:
        edges = np.linspace(np.nanmin(scores), np.nanmax(scores), bins + 1)
    edges = np.unique(np.sort(np.append(edges, dn_threshold)))
    centres = 0.5 * (edges[:-1] + edges[1:])

    if density_groups:
        # One smoothed density per variant class, each normalised to unit area,
        # rather than stacked counts. Stacking answers "how many", which the
        # class sizes already say (they are in the legend); overlaid densities
        # answer "where does each class sit", which is the actual question — and
        # a class 13x rarer than missense is invisible on a count axis.
        from scipy.stats import gaussian_kde
        ax.axvspan(edges[0], dn_threshold, color=dn_color, alpha=0.07, lw=0,
                   zorder=0)
        for entry in density_groups:
            # a 4th element is an optional linestyle, used to mark a control
            # (e.g. synonymous) as not one of the library variant classes
            label, vals, colour = entry[:3]
            ls = entry[3] if len(entry) > 3 else "-"
            v = np.asarray([x for x in vals if np.isfinite(x)], dtype=float)
            if log_x:
                v = v[v > 0]
            t = np.log10(v) if log_x else v
            if t.size < 5 or np.ptp(t) == 0:
                continue
            grid = np.linspace(np.log10(edges[0]) if log_x else edges[0],
                               np.log10(edges[-1]) if log_x else edges[-1], 400)
            y = gaussian_kde(t, bw_method=smooth_bw)(grid)
            x = 10 ** grid if log_x else grid
            ax.plot(x, y, color=colour, lw=1.4, ls=ls, zorder=3,
                    label=f"{label} ({len(v):,})")
            if ls == "-":
                ax.fill_between(x, y, color=colour, alpha=0.13, lw=0, zorder=1)
        counts = np.zeros(len(edges) - 1)
        ax.set_ylim(bottom=0)
        ax.legend(loc="upper left", fontsize=F["ann"], frameon=False,
                  handlelength=1.1, handletextpad=0.5, borderaxespad=0.3,
                  labelspacing=0.25)
    elif groups:
        # Stacked by variant class. The below-threshold region is shaded instead
        # of recoloured, so the DN split stays legible without spending the
        # colour channel that now carries variant class.
        ax.axvspan(edges[0], dn_threshold, color=dn_color, alpha=0.07, lw=0,
                   zorder=0)
        bottom = np.zeros(len(edges) - 1)
        for label, vals, colour in groups:
            v = np.asarray([x for x in vals if np.isfinite(x)], dtype=float)
            c, _ = np.histogram(v, bins=edges)
            ax.bar(edges[:-1], c, width=np.diff(edges), align="edge",
                   bottom=bottom, color=colour, linewidth=0, zorder=2,
                   label=f"{label} ({len(v):,})")
            bottom += c
        counts = bottom
        ax.legend(loc="upper left", fontsize=F["ann"], frameon=False,
                  handlelength=0.9, handletextpad=0.5, borderaxespad=0.3,
                  labelspacing=0.25)
    else:
        counts, _ = np.histogram(scores, bins=edges)
        ax.bar(edges[:-1], counts, width=np.diff(edges), align="edge",
               color=np.where(centres < dn_threshold, dn_color, bar_color),
               linewidth=0, zorder=2)

    if smooth and not density_groups and len(scores) > 5:
        from scipy.stats import gaussian_kde
        t = np.log10(scores[scores > 0]) if log_x else scores
        if t.size > 5 and np.ptp(t) > 0:
            grid = np.linspace(t.min(), t.max(), 400)
            # scale a density to counts with a UNIFORM bin width in this space:
            # the real edges carry an inserted split at the threshold, so their
            # widths are uneven and would misscale the curve
            width = np.ptp(t) / bins
            y = gaussian_kde(t, bw_method=smooth_bw)(grid) * t.size * width
            ax.plot(10 ** grid if log_x else grid, y, color="#111111",
                    lw=1.0, alpha=0.85, zorder=4)

    ax.axvline(dn_threshold, color=dn_color, ls=":", lw=1.1, zorder=5)
    # Headroom for the threshold caption and the per-class block, both of which
    # hang from the top of the axes. Without it a tall peak near the threshold
    # runs straight through the text.
    ax.set_ylim(top=ax.get_ylim()[1]
                * (1.26 if (groups or density_groups) else 1.10))
    n_dn = int((scores < dn_threshold).sum())
    ax.annotate(f"DN threshold {dn_threshold:.3g}\n{n_dn:,} of "
                f"{len(scores):,} below",
                xy=(dn_threshold, ax.get_ylim()[1]), xytext=(4, -4),
                textcoords="offset points", ha="left", va="top",
                color=dn_color, fontsize=F["ann"])
    _classes = density_groups or groups
    if _classes and n_dn:
        # What fraction of each class falls below the threshold. On a density
        # panel this is the number the curves cannot give: the curves are
        # area-normalised, so they show where each class sits but not how much
        # of it is DN. Anchored in axes fraction at the top right, since a
        # multi-class breakdown beside the threshold line runs off the panel.
        comp = []
        for entry in _classes:
            label, vals = entry[0], entry[1]
            v = np.asarray([x for x in vals if np.isfinite(x)], dtype=float)
            if not len(v):
                continue
            k = int((v < dn_threshold).sum())
            comp.append(f"{label} {100 * k / len(v):.0f}%"
                        if density_groups else
                        f"{label} {100 * k / n_dn:.0f}%")
        if comp:
            head = ("% of each class below:" if density_groups
                    else "of those below:")
            ax.annotate(head + "\n" + "\n".join(comp),
                        xy=(0.99, 0.97), xycoords="axes fraction",
                        ha="right", va="top", color=dn_color,
                        fontsize=F["ann"], linespacing=1.35)
    ax.set_ylabel("Density" if density_groups else "Variants",
                  fontsize=F["axis"])
    ax.tick_params(labelsize=F["tick"])
    if title:
        ax.set_title(title, loc="left", fontsize=F["title"])

    for i, (label, five, colour) in enumerate(box_distributions):
        y = -float(i)
        lo_, q1, med, q3, hi_ = (five.get(k, np.nan) for k in
                                 ("p2.5", "p25", "median", "p75", "p97.5"))
        q1 = q1 if np.isfinite(q1) else med
        q3 = q3 if np.isfinite(q3) else med
        axb.plot([lo_, hi_], [y, y], color=colour, lw=0.8, zorder=2)
        axb.add_patch(plt.Rectangle((q1, y - 0.14), max(q3 - q1, 1e-9), 0.28,
                                    facecolor=colour, edgecolor=colour,
                                    alpha=0.35, zorder=3))
        axb.plot([med, med], [y - 0.16, y + 0.16], color=colour, lw=1.3,
                 zorder=4)
        axb.annotate(label, xy=(0.0, y), xycoords=("axes fraction", "data"),
                     xytext=(2, 0), textcoords="offset points", va="center",
                     ha="left", fontsize=F["ann"], color=colour)

    axb.axvline(dn_threshold, color=dn_color, ls=":", lw=1.1)
    axb.set_ylim(-(n_box - 1) - 0.6, 0.6)
    axb.set_yticks([])
    axb.set_xlabel(xlabel, fontsize=F["axis"])
    axb.tick_params(labelsize=F["tick"])
    for side in ("left", "top", "right"):
        axb.spines[side].set_visible(False)
    if log_x:
        ax.set_xscale("log")
        # A decade-only axis labels just 10^0 over the range these scores span,
        # which leaves the reader unable to place the threshold or read the
        # spread. Label the 1-2-5 subdivisions in plain decimals instead.
        from matplotlib.ticker import (FuncFormatter, LogLocator,
                                       NullFormatter)
        axb.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0),
                                               numticks=12))
        axb.xaxis.set_minor_locator(LogLocator(base=10, subs="auto",
                                               numticks=12))
        axb.xaxis.set_minor_formatter(NullFormatter())
        axb.xaxis.set_major_formatter(FuncFormatter(
            lambda v, _p: (f"{v:g}" if v >= 1 else f"{v:.2g}")))
    if note:
        axb.annotate(note, xy=(1.0, 1.0), xycoords="axes fraction", ha="right",
                     va="bottom", fontsize=F["ann"] - 1.5, color="#999999")
    return fig, axes


def plot_density_grid(
    panels,
    *,
    ncols: int = 4,
    panel_size: tuple[float, float] = (2.5, 1.85),
    dn_color: str = "#C0392B",
    xlabel: str = "Pathway activity score (WT-normalised)",
    log_x: bool = True,
    xlim: tuple[float, float] | None = None,
    suptitle: str = "",
    fs: Mapping[str, float] | None = None,
) -> tuple[Figure, np.ndarray]:
    """Small multiples of :func:`plot_activity_histogram`'s density mode.

    One panel per library, so the DN threshold can be judged against each
    library's own distributions rather than against a single exemplar. Each
    ``panels`` entry is ``(title, density_groups, dn_threshold, note)`` with
    ``density_groups`` as ``(label, values, colour[, linestyle])``.

    Every panel shares one x-axis range and one legend, and each is
    area-normalised independently -- the comparison being made across panels is
    *shape against the threshold*, not abundance, which varies by orders of
    magnitude between libraries.
    """
    from scipy.stats import gaussian_kde
    F = {"axis": 7, "tick": 6, "ann": 6, "title": 7.5}
    F.update(fs or {})

    vals = [v for _t, gs, _thr, _n in panels for _l, v, *_ in gs
            for v in np.asarray(v, dtype=float) if np.isfinite(v)]
    if xlim is not None:
        lo, hi = xlim
    else:
        # A shared percentile range clips whichever protein reaches lowest --
        # here HGFR and KRAS, whose thresholds sit near 0.09 and 0.18. Pass
        # `xlim` to set the window deliberately instead.
        lo, hi = np.percentile([v for v in vals if v > 0] if log_x else vals,
                               [0.2, 99.8])
    grid_t = np.linspace(np.log10(lo) if log_x else lo,
                         np.log10(hi) if log_x else hi, 400)
    xs = 10 ** grid_t if log_x else grid_t

    n = len(panels)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, sharex=True,
                             figsize=(panel_size[0] * ncols,
                                      panel_size[1] * nrows))
    axes = np.atleast_1d(axes).ravel()
    seen: dict = {}
    for ax, (title, groups, thr, note) in zip(axes, panels):
        ax.axvspan(lo, thr, color=dn_color, alpha=0.07, lw=0, zorder=0)
        for entry in groups:
            label, v, colour = entry[:3]
            ls = entry[3] if len(entry) > 3 else "-"
            v = np.asarray([x for x in v if np.isfinite(x)], dtype=float)
            if log_x:
                v = v[v > 0]
            t = np.log10(v) if log_x else v
            if t.size < 5 or np.ptp(t) == 0:
                continue
            y = gaussian_kde(t)(grid_t)
            (ln,) = ax.plot(xs, y, color=colour, lw=1.2, ls=ls, zorder=3)
            if ls == "-":
                ax.fill_between(xs, y, color=colour, alpha=0.13, lw=0, zorder=1)
            seen.setdefault(label, ln)
        ax.axvline(thr, color=dn_color, ls=":", lw=1.0, zorder=5)
        ax.set_title(title, loc="left", fontsize=F["title"])
        if note:
            # top-LEFT: the synonymous peak sits near x=1 in every panel and the
            # top-right annotation was landing on it. The shaded DN region is
            # empty up there in all of them.
            ax.annotate(note, xy=(0.03, 0.94), xycoords="axes fraction",
                        ha="left", va="top", fontsize=F["ann"], color=dn_color)
        ax.set_yticks([])
        ax.set_xlim(lo, hi)
        ax.tick_params(labelsize=F["tick"])
        for side in ("left", "top", "right"):
            ax.spines[side].set_visible(False)
    for ax in axes[n:]:
        ax.set_visible(False)
    if log_x:
        from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
        for ax in axes[:n]:
            ax.set_xscale("log")
            ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 3.0),
                                                  numticks=8))
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.xaxis.set_major_formatter(FuncFormatter(
                lambda v, _p: (f"{v:g}" if v >= 1 else f"{v:.2g}")))
    if suptitle:
        fig.suptitle(suptitle, x=0.01, ha="left", fontsize=F["title"] + 1.5)
    fig.tight_layout(rect=(0, 0.085, 1, 0.98))
    # placed after tight_layout, in figure coords, so the legend sits below the
    # axis label instead of on top of it
    fig.supxlabel(xlabel, fontsize=F["axis"], y=0.055)
    fig.legend(seen.values(), seen.keys(), loc="lower center", ncol=len(seen),
               fontsize=F["ann"], frameon=False, bbox_to_anchor=(0.5, 0.0))
    return fig, axes


def plot_dn_summary_table(
    table: pd.DataFrame,
    *,
    col_widths: Sequence[float] = (0.10, 0.16, 0.15, 0.15, 0.15, 0.13),
    left_cols: set[int] = frozenset({0, 1}),
    row_h: float = 0.1625,
    header_h: float = 0.17,
    width_scale: float = 7.5,
    fontsize: float = 7.0,
    zebra: str = "#f5f5f5",
    rule_color: str = "black",
    rule_lw: float = 0.7,
) -> tuple[Figure, Axes]:
    """Compact line-ruled summary table.

    Verbatim port of ``render_table`` in ``scripts/make_dn_summary_table.py``,
    which follows ``Dominant_negative.ipynb`` cell 19: three horizontal rules
    (top, under the header, bottom), 7 pt text, faint zebra shading, no cell
    borders and no filled header. The first two columns are left-aligned and
    the rest centred, and the figure width is set by the column widths so the
    panel comes out at a fixed compact size.

    Args:
        table: Already-formatted strings; column order is taken as given.
        col_widths: Relative widths, one per column. Their sum times
            ``width_scale`` is the figure width in inches.
        left_cols: Indices of left-aligned columns; others are centred.
    """
    labels = list(table.columns)
    n_rows = len(table)
    total_w = sum(col_widths)
    fig_w = total_w * width_scale
    fig_h = header_h + row_h * n_rows + 0.10

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    x_starts, x = [], 0.0
    for w in col_widths:
        x_starts.append(x / total_w)
        x += w

    y_top = 1.0
    y_header = y_top - header_h / fig_h
    row_step = row_h / fig_h

    def draw_line(y: float) -> None:
        ax.plot([0, 1], [y, y], color=rule_color, linewidth=rule_lw,
                transform=ax.transAxes, clip_on=False)

    def x_pos(j: int) -> tuple[float, str]:
        if j in left_cols:
            return x_starts[j] + 0.008, "left"
        return x_starts[j] + col_widths[j] / total_w / 2, "center"

    for j, label in enumerate(labels):
        xp, ha = x_pos(j)
        ax.text(xp, (y_top + y_header) / 2, label, transform=ax.transAxes,
                ha=ha, va="center", fontsize=fontsize, fontweight="bold")
    draw_line(y_top)
    draw_line(y_header)

    for i, (_, row) in enumerate(table.iterrows()):
        y_mid = y_header - (i + 0.5) * row_step
        if i % 2 == 1:
            ax.axhspan(y_header - (i + 1) * row_step,
                       y_header - i * row_step, xmin=0, xmax=1,
                       color=zebra, zorder=0)
        for j, label in enumerate(labels):
            xp, ha = x_pos(j)
            ax.text(xp, y_mid, str(row[label]), transform=ax.transAxes,
                    ha=ha, va="center", fontsize=fontsize)
    draw_line(y_header - n_rows * row_step)
    return fig, ax


def plot_class_depletion_heatmap(
    lut: Mapping[tuple[str, str, str], tuple[float, float, int]],
    *,
    groups: Sequence[tuple[str, Sequence[str]]],
    class_labels: Mapping[str, str],
    col_order: Sequence[str],
    pairwise: pd.DataFrame | None = None,
    pooled_key: str = "Pooled",
    genie_min_count: int = 1,
    suptitle: str | None = None,
    cbar_label: str = "Odds ratio vs WT-like missense (log₂ scale)",
    figsize: tuple[float, float] = (8.4, 7.2),
    group_gap: float = 0.6,
    col_gap: float = 0.6,
    log2_lim: tuple[float, float] = (-3.0, 3.0),
    nodata_color: str = "#dddddd",   # NODATA_COLOR in the source script
    sig_q: float = 0.05,
    sig_q_strong: float = 0.01,
    sig_edge_weak: str = "#555555",
    sig_edge_strong: str = "black",
    sig_inset: float = 0.085,
    cbar_dy: float = 0.0,
    cbar_h_frac: float = 0.55,
) -> tuple[Figure, Axes]:
    """Per-scope x (database x class) depletion heatmap.

    Verbatim port of ``build_figure`` and ``draw_pooled_brackets`` from
    ``scripts/plot_class_depletion_heatmap.py`` — the drawing is unchanged so
    the panel is identical to the published figure. Only the inputs differ:
    the class/database definitions, the column order and the between-class
    statistics are computed in the notebook and passed in, so no analysis
    happens here.

    Placement is deliberately staged and order-dependent. The axes is
    aspect-equal, so its realised rectangle is only known after
    ``fig.canvas.draw()``; the colourbar and significance key are then placed
    against that realised right edge, which is why they sit snug regardless of
    how far the bracket ladder extends the x-range.

    Args:
        lut: ``(scope, database, class) -> (odds_ratio, q, n_total)``. Missing
            keys, ``n_total == 0`` and non-finite odds ratios all render as
            no-data grey.
        groups: ``(database, [class, ...])`` top to bottom. Blocks may differ
            in length — GENIE has no nonsense classes because stop-gains have
            no cancer observations.
        class_labels: class key -> row label.
        col_order: scope columns left to right; ``pooled_key`` is appended
            after a gap and its tick label is bolded.
        pairwise: between-class tests with ``database``, ``class_a``,
            ``class_b``, ``q``. Significant pairs get a bracket in packed lanes
            right of the pooled column; ns pairs are omitted. None skips them.
        genie_min_count: tumour-count threshold, for the GENIE super-label.
        cbar_dy: Raise the colourbar (and so its rotated label) by this
            fraction of the figure height. The bar is bottom-aligned with the
            heatmap, which puts a long label hard against the bottom edge where
            a tight bbox can clip it.
        cbar_h_frac: Colourbar height as a fraction of the heatmap height.

    Returns:
        ``(fig, ax)``.
    """
    from matplotlib.cm import ScalarMappable
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle
    import matplotlib.colors as mcolors

    LOG2_MIN, LOG2_MAX = log2_lim
    CMAP = plt.get_cmap("RdBu_r")
    NORM = mcolors.TwoSlopeNorm(vcenter=0.0, vmin=LOG2_MIN, vmax=LOG2_MAX)
    CBAR_EXP = list(range(int(LOG2_MIN), int(LOG2_MAX) + 1))
    CBAR_LABELS = [f"{2.0 ** e:g}" for e in CBAR_EXP]

    proteins = list(col_order)
    col_x: dict[str, float] = {}
    x = 0.0
    for p in proteins:
        col_x[p] = x
        x += 1.0
    x += col_gap
    col_x[pooled_key] = x
    all_cols = proteins + [pooled_key]

    row_y: list[tuple[str, str, float]] = []
    group_spans: list[tuple[str, float, float]] = []
    y = 0.0
    for gi, (db, classes) in enumerate(groups):
        if gi > 0:
            y += group_gap
        y0 = y
        for cls in classes:
            row_y.append((db, cls, y))
            y += 1.0
        group_spans.append((db, y0, y - 1.0))
    y_bottom = y - 1.0

    fig, ax = plt.subplots(figsize=figsize)

    # Two passes so significant borders are never clipped by a later neighbour.
    sig_weak: list[tuple[float, float]] = []
    sig_strong: list[tuple[float, float]] = []
    for (db, cls, yy) in row_y:
        for col in all_cols:
            xx = col_x[col]
            rec = lut.get((col, db, cls))
            if rec is None or rec[2] == 0 or not np.isfinite(rec[0]):
                ax.add_patch(Rectangle((xx - 0.5, yy - 0.5), 1, 1,
                                       facecolor=nodata_color,
                                       edgecolor="white", lw=0.4, zorder=2))
                continue
            or_, q, _ = rec
            val = np.clip(np.log2(or_) if or_ > 0 else LOG2_MIN,
                          LOG2_MIN, LOG2_MAX)
            ax.add_patch(Rectangle((xx - 0.5, yy - 0.5), 1, 1,
                                   facecolor=CMAP(NORM(val)),
                                   edgecolor="#cfcfcf", lw=0.4, zorder=2))
            if np.isfinite(q):
                if q < sig_q_strong:
                    sig_strong.append((xx, yy))
                elif q < sig_q:
                    sig_weak.append((xx, yy))
    side = 1 - 2 * sig_inset
    for (xx, yy) in sig_weak:
        ax.add_patch(Rectangle((xx - 0.5 + sig_inset, yy - 0.5 + sig_inset),
                               side, side, facecolor="none",
                               edgecolor=sig_edge_weak, lw=1.3, zorder=5))
    for (xx, yy) in sig_strong:
        ax.add_patch(Rectangle((xx - 0.5 + sig_inset, yy - 0.5 + sig_inset),
                               side, side, facecolor="none",
                               edgecolor=sig_edge_strong, lw=1.6, zorder=6))

    # ---- bracket ladder right of the pooled column ----------------------
    bracket_x_max = col_x[pooled_key] + 0.85
    if pairwise is not None and len(pairwise):
        y_of = {(db, cls): yy for (db, cls, yy) in row_y}
        x0, lane_w, nub = col_x[pooled_key] + 0.85, 0.17, 0.10
        for db, _classes in groups:
            sub = pairwise[pairwise["database"] == db]
            if sub.empty:
                continue
            pairs = []
            for _, r in sub.iterrows():
                q = r["q"]
                if not np.isfinite(q) or q >= sig_q:
                    continue
                key_a, key_b = (db, r["class_a"]), (db, r["class_b"])
                if key_a not in y_of or key_b not in y_of:
                    continue
                ya, yb = y_of[key_a], y_of[key_b]
                pairs.append((abs(ya - yb), min(ya, yb), max(ya, yb),
                              q < sig_q_strong))
            pairs.sort(key=lambda t: (t[0], t[1]))
            for lane, (_span, ytop, ybot, strong) in enumerate(pairs):
                xx = x0 + lane * lane_w
                bracket_x_max = max(bracket_x_max, xx)
                col = sig_edge_strong if strong else sig_edge_weak
                lw = 1.3 if strong else 1.0
                ax.plot([xx, xx], [ytop, ybot], color=col, lw=lw, zorder=7,
                        clip_on=False)
                for yend in (ytop, ybot):
                    ax.plot([xx - nub, xx], [yend, yend], color=col, lw=lw,
                            zorder=7, clip_on=False)

    ax.set_xlim(-0.6, bracket_x_max + 0.7)
    ax.set_ylim(y_bottom + 0.6, -0.6)
    ax.set_aspect("equal")

    ax.set_xticks([col_x[c] for c in all_cols])
    ax.xaxis.tick_top()
    ax.set_xticklabels([protein_label(c) if str(c).lower() in PROTEIN_KEYS
                        else str(c).upper() for c in all_cols],
                       fontsize=13, rotation=45, ha="left",
                       rotation_mode="anchor")
    ax.get_xticklabels()[-1].set_fontweight("bold")
    ax.tick_params(axis="x", length=0)

    ax.set_yticks([yy for (_, _, yy) in row_y])
    ax.set_yticklabels([class_labels.get(c, c) for (_, c, _) in row_y],
                       fontsize=12)
    ax.tick_params(axis="y", length=0)
    for (db, y0, y1) in group_spans:
        db_label = f"GENIE (≥{genie_min_count})" if db == "GENIE" else db
        ax.annotate(db_label, xy=(-0.255, (y0 + y1) / 2),
                    xycoords=("axes fraction", "data"), rotation=90,
                    ha="center", va="center", fontsize=14, fontweight="bold",
                    annotation_clip=False)

    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)

    fig.subplots_adjust(left=0.26, right=0.96, top=0.80, bottom=0.06)
    ax.set_anchor("W")
    if suptitle:
        fig.suptitle(suptitle, fontsize=12, fontweight="bold", y=0.99)

    fig.canvas.draw()   # realise the aspect-shrunk axes before placing legends
    pos = ax.get_position()
    legend_x = pos.x1 + 0.015

    sm = ScalarMappable(norm=NORM, cmap=CMAP)
    sm.set_array([])
    cax = fig.add_axes([legend_x, pos.y0 + cbar_dy, 0.016,
                        pos.height * cbar_h_frac])
    cbar = fig.colorbar(sm, cax=cax, ticks=CBAR_EXP, extend="both")
    cbar.ax.set_yticklabels(CBAR_LABELS, fontsize=10.5)
    cbar.set_label(cbar_label, fontsize=11.5)
    cbar.outline.set_linewidth(0.5)

    # Significance key: a box at the same drawn size as the cells' inset
    # outline, plus a matching bracket glyph, so one key serves both.
    ax_bb = ax.get_window_extent()
    xr, yr = ax.get_xlim(), ax.get_ylim()
    cell_px = min(ax_bb.width / abs(xr[1] - xr[0]),
                  ax_bb.height / abs(yr[1] - yr[0]))
    fw_px, fh_px = fig.get_size_inches() * fig.dpi
    cell_fw, cell_fh = cell_px / fw_px, cell_px / fh_px
    side_fw, side_fh = cell_fw * (1 - 2 * sig_inset), cell_fh * (1 - 2 * sig_inset)

    key_left, y_top, pitch = legend_x, pos.y1, cell_fh + 0.060
    for i, (edge, lw, label) in enumerate(
            [(sig_edge_weak, 1.3, f"BH q < {sig_q:g}"),
             (sig_edge_strong, 1.6, f"BH q < {sig_q_strong:g}")]):
        top = y_top - i * pitch
        fig.add_artist(Rectangle(
            (key_left, top - side_fh), side_fw, side_fh, facecolor="#ededed",
            edgecolor=edge, linewidth=lw, transform=fig.transFigure,
            clip_on=False))
        yc, half = top - side_fh / 2, side_fh * 0.45
        sx, nub = key_left + side_fw + 0.024, 0.009
        for (x0, y0), (x1, y1) in (((sx, yc - half), (sx, yc + half)),
                                   ((sx - nub, yc - half), (sx, yc - half)),
                                   ((sx - nub, yc + half), (sx, yc + half))):
            fig.add_artist(Line2D([x0, x1], [y0, y1], color=edge, lw=lw,
                                  transform=fig.transFigure, clip_on=False))
        fig.text((key_left + sx) / 2, top - side_fh - 0.012, label,
                 ha="center", va="top", fontsize=10)
    return fig, ax


#: Database styling for the pooled depletion forest, carried over from
#: ``scripts/plot_class_depletion_global.py``. Germline vs cancer is stated in
#: the label because the two are different kinds of evidence.
DEPLETION_DB_STYLE = {
    "gnomAD": {"color": "#1F3F73", "marker": "o", "label": "gnomAD (germline)"},
    "AoU":    {"color": "#E08328", "marker": "s", "label": "AoU (germline)"},
    "GENIE":  {"color": "#51b949", "marker": "D", "label": "GENIE (cancer)"},
}
#: Vertical offset within a class slot; first listed sits on top.
DEPLETION_DB_OFFSET = {"gnomAD": +0.22, "AoU": 0.0, "GENIE": -0.22}


def plot_class_depletion_pooled(
    df: pd.DataFrame,
    *,
    class_order: Sequence[tuple[str, str]],
    db_style: Mapping[str, Mapping[str, str]] | None = None,
    db_offset: Mapping[str, float] | None = None,
    figsize: tuple[float, float] = (5.5, 2.6),
    xlim: tuple[float, float] = (0.15, 3.5),
    xticks: Sequence[float] = (0.2, 0.5, 1.0, 2.0),
    xlabel: str = "Odds ratio vs WT-like missense  (95% CI)",
    genie_legend_label: str | None = None,
    stars_fn=stars_4tier,
) -> tuple[Figure, Axes]:
    """Pooled class-depletion forest, three databases per class.

    Verbatim port of ``render`` in ``scripts/plot_class_depletion_global.py`` —
    5.5 x 2.6 in, per-database colour/marker/offset, 5.5 pt markers with white
    edges, 1.0 pt whiskers and 2.5 pt caps, the ``{stars}  {n_obs}/{n_tot}``
    annotation at 1.05x the CI upper bound in the series colour, a dashed grey
    OR=1 reference, log x fixed to 0.15-3.5, and the legend inside the
    upper-right corner (empty because the data only reaches OR ~1.6).

    There is deliberately **no WT-like row**: WT-like is the reference the odds
    ratios are computed against, so its OR is 1 by construction and plotting it
    would imply a measurement.

    Classes with no coverage in a database simply render no marker for it — the
    nonsense rows have no GENIE data because stop-gains are not recurrent
    cancer drivers.

    Args:
        df: One row per (class, database) with ``class``, ``database``, ``or``,
            ``ci_low``, ``ci_high``, ``p``, ``n_obs``, ``n_tot``. All computed
            in the notebook.
        class_order: ``(class_key, display_label)`` top to bottom; the labels
            carry their own line breaks.
        genie_legend_label: Override for the GENIE legend entry, e.g.
            ``"GENIE >=2 (cancer)"`` to flag the tumour-count filter.
    """
    style = {k: dict(v) for k, v in (db_style or DEPLETION_DB_STYLE).items()}
    offset = dict(db_offset or DEPLETION_DB_OFFSET)
    if genie_legend_label and "GENIE" in style:
        style["GENIE"]["label"] = genie_legend_label

    class_keys = [c[0] for c in class_order]
    class_labels = [c[1] for c in class_order]
    df = df[df["class"].isin(class_keys)].copy()

    fig, ax = plt.subplots(figsize=figsize)
    n_classes = len(class_keys)
    y_centres = np.arange(n_classes - 1, -1, -1, dtype=float)

    for db, st in style.items():
        for i, cls in enumerate(class_keys):
            row = df[(df["class"] == cls) & (df["database"] == db)]
            if row.empty:
                continue
            r = row.iloc[0]
            y = y_centres[i] + offset.get(db, 0.0)
            or_val = float(r["or"])
            ci_lo, ci_hi = float(r["ci_low"]), float(r["ci_high"])
            if not np.isfinite(or_val):
                continue
            if not (np.isfinite(ci_lo) and np.isfinite(ci_hi)):
                ci_lo = ci_hi = or_val
            ax.errorbar(or_val, y,
                        xerr=[[max(or_val - ci_lo, 0)], [max(ci_hi - or_val, 0)]],
                        fmt="none", ecolor=st["color"], elinewidth=1.0,
                        capsize=2.5, capthick=1.0, alpha=0.9, zorder=2)
            ax.plot(or_val, y, marker=st["marker"], markersize=5.5,
                    color=st["color"], markeredgecolor="white",
                    markeredgewidth=0.4, zorder=3)
            ax.text(ci_hi * 1.05, y,
                    f"{stars_fn(float(r['p']))}  {int(r['n_obs'])}/{int(r['n_tot'])}",
                    ha="left", va="center", fontsize=6.5, color=st["color"],
                    clip_on=False)

    ax.axvline(1.0, color="#999999", linewidth=0.7, linestyle="--", zorder=1)
    ax.set_xscale("log")
    ax.set_xlim(*xlim)
    ax.set_xticks(list(xticks))
    ax.set_xticklabels([f"{t:g}" if t != 1.0 else "1.0" for t in xticks],
                       fontsize=8)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_yticks(y_centres)
    ax.set_yticklabels(class_labels, fontsize=9)
    ax.set_ylim(-0.7, n_classes - 0.3)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)

    handles = [plt.Line2D([0], [0], marker=style[db]["marker"],
                          color=style[db]["color"], linestyle="",
                          markersize=5.5, markeredgecolor="white",
                          markeredgewidth=0.4, label=style[db]["label"])
               for db in ("gnomAD", "AoU", "GENIE") if db in style]
    ax.legend(handles=handles, loc="upper right", fontsize=8, frameon=False,
              handletextpad=0.3, labelspacing=0.4, borderaxespad=0.4)
    fig.subplots_adjust(left=0.20, right=0.98, top=0.96, bottom=0.20)
    return fig, ax

def plot_structure(
    out_stem: str | Path,
    *,
    structure: str,
    query_chains: Sequence[str] = ("A",),
    sphere_counts: Mapping[int, float] | None = None,
    residue_values: Mapping[int, float] | None = None,
    sphere_color: tuple[float, float, float] | str = (0.75, 0.16, 0.16),
    cmap: str = "Reds",
    vmin: float = 0.0,
    vmax: float = 1.0,
    nodata_color: str = "gray35",
    radius_range: tuple[float, float] = (0.6, 3.5),
    count_range: tuple[float, float] = (1.0, 20.0),
    partners: Sequence[tuple[str, str, str] | Mapping] = (),
    ligands: Sequence[str] = (),
    graft: Mapping | None = None,
    ligand_color: str = "yellow",
    ion_vdw: float = 0.9,
    cartoon_color: str = STRUCT_CARTOON_COLOR,
    view: Sequence[float] | None = None,
    align_to: str | None = None,
    align_chain: str = "A",
    align_resi: tuple[int, int] | None = None,
    align_rotations: Sequence[tuple[str, float]] = (),
    px_per_angstrom: float = 14.0,
    pad_angstrom: float = 6.0,
    cartoon_pad_angstrom: float = 4.0,
    max_px: int = 3000,
    bg_color: str = "white",
    save_pse: bool = True,
    run: bool = True,
    extra_commands: Sequence[str] = (),
) -> Path:
    """Render a structure with per-residue spheres and/or a colour ramp.

    The one structure renderer. It writes a ``.pml`` and then invokes PyMOL on
    it, so the exact commands stay inspectable and re-runnable outside the
    notebook.

    Two channels, independent and combinable:

    * ``sphere_counts`` -> C-alpha spheres whose VDW radius is a **linear,
      absolute** function of the count (``count_range`` mapped onto
      ``radius_range``, saturating above). Absolute rather than per-structure
      normalisation is what lets one legend serve every render.
    * ``residue_values`` -> the cartoon coloured by a value in
      ``[vmin, vmax]`` through ``cmap``, with unmeasured residues in
      ``nodata_color`` (mid-grey, so it is distinguishable from the white end
      of the ramp).

    **Constant scale, varying canvas.** For the sphere legend to be honest, a
    3.5 A sphere must occupy the same number of page pixels in every panel. The
    camera is therefore zoomed to fit the molecule and the image is then sized
    at ``px_per_angstrom`` pixels per Angstrom of molecular extent — so a large
    protein yields a physically larger image rather than being shrunk to a
    common frame. Panels will not all be the same dimensions, and will not
    match renders made with a fixed image size.

    Args:
        out_stem: Output path without extension; ``.pml``, ``.png`` and
            optionally ``.pse`` are written beside it.
        structure: ``"fetch:9AXM"`` for a PDB ID, else a path to a local file.
            Fetches use the asymmetric unit, which preserves the deposited
            chain IDs — the biological assembly renames them and silently
            produced a blank ARAF render once.
        partners: Either ``(chain, label, colour)`` or a mapping with those
            keys plus optional ``sphere_counts`` / ``sphere_color``. Shown as
            sticks for a short peptide, cartoon otherwise. Giving a partner its
            own spheres is how a two-chain panel shows DNs on both sides: each
            chain keeps its own cartoon colour and its own sphere colour, so
            which protein a sphere belongs to is unambiguous.
        ligands: Residue names to draw from the query chains, e.g.
            ``("ATP", "MG")``. Polyatomic ligands are drawn as sticks with
            carbons in ``ligand_color`` and other elements by element; single
            atoms (metals, halides) are drawn as small spheres in
            ``ligand_color``. Nothing is shown by default: a deposited ligand is
            loaded but invisible unless asked for, which is how MET's bound
            ATP went unrendered.
        graft: Borrow a chain from a SECOND structure and show it in this one's
            frame, for when the ligand-bound structure covers only part of the
            protein. Keys: ``structure``, ``align_resi`` (the residue range
            shared by both, used for the superposition), ``keep_chain``,
            ``label``, ``color``, and optional ``align_chain`` (default "A").
            The donor is superposed onto the query over ``align_resi``, then
            everything but ``keep_chain`` is discarded. Caveat worth stating in
            any caption: donor and query are different conformations, so a
            grafted ligand marks *where* a site is rather than reproducing that
            complex's geometry exactly.
        ligand_color: Carbon / ion colour for ``ligands``. Chosen to contrast
            with both the grey cartoon and the per-protein DN spheres.
        ion_vdw: Sphere radius for single-atom ligands. Deliberately distinct
            from ``radius_range`` so an ion is not misread as a DN sphere; the
            colour is the primary cue.
        view: 18-tuple from PyMOL's ``get_view``. Its **orientation and
            centring** are honoured exactly; its zoom is refit to the molecule
            so nothing is cropped, since the view's own field of view trims
            anything taller than it and enlarging the image at a fixed field of
            view would change the scale rather than reveal more. Without a view the camera is zoomed to fit and
            the canvas sized from the molecular extent; both routes land on the
            same scale, which is what makes one sphere legend valid throughout.
        align_to / align_chain / align_resi / align_rotations: Superpose onto
            an anchor so several proteins share a frame. The rotations are
            applied to the **anchor's atoms** before superposition — order
            matters, because ``cmd.rotate(axis, angle, selection)`` moves only
            that selection, so rotating afterwards would leave the subject
            behind.
        px_per_angstrom: Pixels per Angstrom. Constant across panels — this is
            what the shared sphere legend rests on.
        pad_angstrom: Clear space beyond the ink, in Angstroms.
        cartoon_pad_angstrom: Allowance for the cartoon ribbon, which is drawn
            around the backbone and so reaches past every atom centre. Added to
            ``pad_angstrom`` together with the largest sphere radius.
        max_px: Cap on either image dimension, to bound ray-trace time.
        run: Execute PyMOL. False writes the ``.pml`` only.

    Returns:
        Path to the written ``.pml``.
    """
    out_stem = Path(out_stem)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    pml = out_stem.with_suffix(".pml")
    obj = out_stem.name.upper().replace("-", "_").replace(".", "_")

    if str(structure).startswith("fetch:"):
        PYMOL_FETCH_CACHE.mkdir(parents=True, exist_ok=True)
        load = f"fetch {str(structure).split(':', 1)[1]}, {obj}, async=0"
    else:
        load = f"load {Path(structure).resolve()}, {obj}"

    q_sel = " or ".join(f"chain {c}" for c in query_chains)
    L = [f"# {out_stem.name}",
         f"set fetch_path, {PYMOL_FETCH_CACHE.resolve()}"]

    if align_to is not None:
        if str(align_to).startswith("fetch:"):
            L.append(f"fetch {str(align_to).split(':', 1)[1]}, _anchor_raw, async=0")
        else:
            L.append(f"load {Path(align_to).resolve()}, _anchor_raw")
        rng = f" and resi {align_resi[0]}-{align_resi[1]}" if align_resi else ""
        L += ["remove _anchor_raw and (solvent or hetatm)",
              f"create _pose_anchor, _anchor_raw and chain {align_chain} "
              f"and polymer{rng}",
              "delete _anchor_raw", "orient _pose_anchor"]
        for axis, angle in align_rotations:
            L.append(f"rotate {axis}, {angle}, _pose_anchor")

    L += [load, "remove solvent", "remove hydro", "hide everything",
          f"select query_sel, {obj} and ({q_sel})",
          "show cartoon, query_sel",
          f"color {cartoon_color}, query_sel"]

    if align_to is not None:
        # cealign gets within a few Angstrom; `super` refines it. cealign alone
        # leaves ~3 A RMSD, which visibly tilts the kinase.
        L += ["cealign _pose_anchor, query_sel",
              "super query_sel, _pose_anchor, cycles=5",
              "hide everything, _pose_anchor", "delete _pose_anchor"]

    if graft:
        g_src = str(graft["structure"])
        if g_src.startswith("fetch:"):
            L.append(f"fetch {g_src.split(':', 1)[1]}, _graft_raw, async=0")
        else:
            L.append(f"load {Path(g_src).resolve()}, _graft_raw")
        g_chain = graft.get("align_chain", "A")
        lo, hi = graft["align_resi"]
        L += ["remove _graft_raw and solvent",
              # Superpose the donor onto the query over the range they share, so
              # the borrowed chain lands in the query's frame. `align` returns
              # the transform; applying it to the whole donor object carries the
              # ligand along with the domain that was fitted.
              f"create _graft_ref, _graft_raw and chain {g_chain} and polymer "
              f"and resi {lo}-{hi}",
              f"align _graft_ref, {obj} and ({q_sel}) and resi {lo}-{hi}, "
              "cycles=5, object=_graft_aln",
              "python",
              "m = cmd.get_object_matrix('_graft_ref')",
              "cmd.transform_object('_graft_raw', m)",
              "python end",
              f"create graft_sel, _graft_raw and chain {graft['keep_chain']}",
              "delete _graft_ref", "delete _graft_aln", "delete _graft_raw",
              "hide everything, graft_sel",
              "show sticks, graft_sel",
              f"color {graft.get('color', 'yellow')}, graft_sel",
              "python",
              "from pymol import util",
              "util.cnc('graft_sel')",
              f"print('grafted {graft.get('label', 'ligand')}: %d atoms' % "
              "cmd.count_atoms('graft_sel'))",
              "python end"]

    partner_specs = []
    for i, rec in enumerate(partners):
        d = (dict(zip(("chain", "label", "color"), rec))
             if isinstance(rec, (tuple, list)) else dict(rec))
        safe = "".join(c if c.isalnum() or c in "._" else "_"
                       for c in str(d["label"]))
        d["sel"] = f"partner_{i}_{safe}"
        partner_specs.append(d)
        L += [f"select {d['sel']}, {obj} and chain {d['chain']}",
              f"color {d['color']}, {d['sel']}", "python",
              f"n_ca = cmd.count_atoms('{d['sel']} and name CA')",
              f"cmd.show('sticks' if n_ca <= 15 else 'cartoon', '{d['sel']}')",
              "python end"]

    if residue_values:
        cm, span = plt.get_cmap(cmap), (vmax - vmin) or 1.0
        L += [f"color {nodata_color}, query_sel",
              "set cartoon_smooth_loops, 0", "set cartoon_discrete_colors, 1"]
        by_colour: dict[tuple, list[int]] = {}
        for pos, val in residue_values.items():
            if val is None or not np.isfinite(val):
                continue
            rgb = tuple(round(c, 3) for c in
                        cm(min(max((val - vmin) / span, 0.0), 1.0))[:3])
            by_colour.setdefault(rgb, []).append(int(pos))
        for k, (rgb, positions) in enumerate(sorted(by_colour.items())):
            L += [f"set_color ramp_{k}, [{rgb[0]}, {rgb[1]}, {rgb[2]}]",
                  f"color ramp_{k}, {obj} and ({q_sel}) and resi "
                  + "+".join(str(x) for x in sorted(positions))]

    def _sphere_cmds(sel_expr: str, counts, colour, tag: str) -> list[str]:
        """Commands sizing and showing C-alpha spheres on one selection.

        Radius is a linear, ABSOLUTE function of the count — the same mapping
        for every selection in every panel — which is what the shared legend
        depends on. Counts at or above ``count_range[1]`` saturate rather than
        rescaling the panel.
        """
        c0, c1 = count_range
        r0, r1 = radius_range
        out: list[str] = []
        if isinstance(colour, str):
            name = colour
        else:
            name = f"sphere_col_{tag}"
            out.append(f"set_color {name}, [{colour[0]:.3f}, "
                       f"{colour[1]:.3f}, {colour[2]:.3f}]")
        by_radius: dict[float, list[int]] = {}
        for pos, n in counts.items():
            if n is None or not np.isfinite(n) or n < c0:
                continue
            frac = (min(float(n), c1) - c0) / max(c1 - c0, 1e-9)
            by_radius.setdefault(round(r0 + frac * (r1 - r0), 3),
                                 []).append(int(pos))
        for radius, positions in sorted(by_radius.items()):
            out.append(f"alter (({sel_expr}) and resi "
                       + "+".join(str(x) for x in sorted(positions))
                       + f" and name CA), vdw={radius:.3f}")
        allp = sorted(x for ps in by_radius.values() for x in ps)
        if allp:
            out += ["rebuild",
                    f"select dn_spheres_{tag}, ({sel_expr}) and resi "
                    + "+".join(str(x) for x in allp) + " and name CA",
                    f"show spheres, dn_spheres_{tag}",
                    f"color {name}, dn_spheres_{tag}",
                    "python",
                    f"print('spheres {tag}: %d of {len(allp)} positions present "
                    f"in this structure' % cmd.count_atoms('dn_spheres_{tag}'))",
                    "python end"]
        return out

    partner_sphere_specs = [d for d in partner_specs if d.get("sphere_counts")]
    if sphere_counts or partner_sphere_specs:
        # Shrink every atom first, once, so only the residues given a radius
        # below render as spheres.
        L += ["alter all, vdw=0.01", "rebuild"]
        if sphere_counts:
            L += _sphere_cmds(f"{obj} and ({q_sel})", sphere_counts,
                              sphere_color, "query")
        for i, d in enumerate(partner_sphere_specs):
            L += _sphere_cmds(f"{obj} and chain {d['chain']}",
                              d["sphere_counts"],
                              d.get("sphere_color", sphere_color), f"p{i}")

    if ligands:
        # After the sphere pass, because that sets vdw=0.01 on every atom to
        # keep unmarked residues from rendering as spheres — the ions need their
        # radius restored afterwards or they vanish.
        resn = "+".join(str(r).upper() for r in ligands)
        L += [f"select ligand_sel, {obj} and ({q_sel}) and resn {resn}",
              "python",
              "from pymol import util",
              "cmd.show('sticks', 'ligand_sel and not (elem MG+ZN+CA+MN+NA+K+CL+FE)')",
              "cmd.show('spheres', 'ligand_sel and elem MG+ZN+CA+MN+NA+K+CL+FE')",
              f"cmd.alter('ligand_sel and elem MG+ZN+CA+MN+NA+K+CL+FE', 'vdw={ion_vdw}')",
              "cmd.rebuild()",
              f"cmd.color('{ligand_color}', 'ligand_sel')",
              # Non-carbon atoms back to element colours, so the phosphates and
              # ribose of a nucleotide stay legible instead of a yellow blob.
              "util.cnc('ligand_sel and not (elem MG+ZN+CA+MN+NA+K+CL+FE)')",
              "print('ligand atoms shown: %d' % cmd.count_atoms('ligand_sel'))",
              "python end",
              "set stick_radius, 0.20"]

    L += [f"bg_color {bg_color}", "set ray_shadows, 0", "set antialias, 2",
          "set ray_opaque_background, 1", "set sphere_quality, 4",
          "set cartoon_fancy_helices, 1", "set depth_cue, 0",
          "set specular, 0.25", "set orthoscopic, on", *extra_commands]

    png = out_stem.with_suffix(".png").resolve()
    # Everything that carries a sphere must be inside the frame, not just the
    # query chain — otherwise a partner's DNs would be cropped.
    framed = " or ".join([f"{obj} and ({q_sel})"]
                         + [f"{obj} and chain {d['chain']}"
                            for d in partner_specs]
                         + (["ligand_sel"] if ligands else [])
                         + (["graft_sel"] if graft else []))
    L.append(f"select framed_sel, {framed}")

    # ONE sizing rule for both routes, so every panel lands on exactly
    # px_per_angstrom and the shared sphere legend is valid throughout.
    #
    # The canvas is SQUARE. PyMOL maps the field of view to the *smaller* pixel
    # dimension, so on a portrait canvas it governs width and on a landscape one
    # height; getting that backwards silently clips the other axis. On a square
    # canvas both readings coincide.
    #
    # The camera distance is then recomputed from the extent the molecule needs
    # about the view's own centre. With a fixed field of view the visible span is
    # 2 * distance * tan(fov / 2), so anything larger is trimmed, and enlarging
    # the image at a fixed field of view would rescale Angstroms-per-pixel rather
    # than reveal more.
    #
    # Padding covers the largest sphere radius and the cartoon ribbon as well as
    # pad_angstrom, because get_coords returns atom CENTRES and neither of those
    # has an atom at its outer edge.
    _fit = ["python",
            "import math",
            "v = list(cmd.get_view())",
            "fov = abs(v[17]) or 20.0",
            "R, org = v[0:9], v[12:15]",
            "P = cmd.get_coords('framed_sel')",
            "def _proj(k, off):",
            "    return [R[3 * k] * (q[0] - org[0]) + R[3 * k + 1] * (q[1] - org[1])"
            " + R[3 * k + 2] * (q[2] - org[2]) + off for q in P]",
            "xs, ys, zs = _proj(0, v[9]), _proj(1, v[10]), _proj(2, 0.0)",
            f"pad = {pad_angstrom} + {radius_range[1]} + {cartoon_pad_angstrom}",
            "half = lambda s: max(abs(min(s)), abs(max(s)))",
            "side = 2 * max(half(xs), half(ys)) + 2 * pad",
            "d = side / (2.0 * math.tan(math.radians(fov) / 2.0))",
            "v[11] = -d",
            "v[15] = max(1.0, d - half(zs) - pad)",
            "v[16] = d + half(zs) + pad",
            "cmd.set_view(v)",
            f"n = min(int(round(side * {px_per_angstrom})), {max_px})",
            "print('fit %.1f A square (pad %.1f) -> %d px (%.2f px/A)'"
            " % (side, pad, n, n / side))",
            f"cmd.png(r'{png}', width=n, height=n, dpi=300, ray=1)",
            "python end"]

    if view is not None:
        # A pinned view supplies the ORIENTATION and CENTRING; its zoom is
        # refit above so nothing is cropped.
        L += ["set_view (" + ", ".join(f"{v:.6f}" for v in view) + ")"] + _fit
    else:
        # No curated view: PyMOL's own principal-axes orientation, which is
        # reproducible but arbitrary. Same sizing rule from there.
        L += ["orient framed_sel"] + _fit

    if save_pse:
        L.append(f"save {out_stem.with_suffix('.pse').resolve()}")

    pml.write_text("\n".join(L) + "\n")
    if run:
        if not PYMOL_BIN.exists():
            raise FileNotFoundError(
                f"PyMOL not found at {PYMOL_BIN}. Set the PYMOL env var.")
        res = subprocess.run([str(PYMOL_BIN), "-cq", str(pml)],
                             capture_output=True, text=True, timeout=900)
        if res.returncode != 0:
            raise RuntimeError(f"PyMOL failed on {pml.name}:\n"
                               f"{res.stdout[-800:]}\n{res.stderr[-800:]}")
    return pml


def plot_sphere_legend(
    counts: Sequence[int] = (1, 5, 10, 20),
    *,
    radius_range: tuple[float, float] = (0.6, 3.5),
    count_range: tuple[float, float] = (1.0, 20.0),
    px_per_angstrom: float = 14.0,
    color: str = "#b0b0b0",
    figsize: tuple[float, float] = (1.5, 2.0),
    label: str = "DN variants\nat position",
    fs: Mapping[str, float] | None = None,
) -> tuple[Figure, Axes]:
    """Sphere-size key for :func:`plot_structure`.

    One legend serves every render because the radius map is absolute and the
    structures are drawn at a constant pixels-per-Angstrom. Drawn in **grey**,
    since the spheres are coloured per protein and this key is about size only.

    Marker areas are computed from the same Angstrom radii and the same
    ``px_per_angstrom`` the renders use, so a circle here is the on-page size
    of that sphere in the panels.
    """
    F = {"label": 9, "tick": 8}
    F.update(fs or {})
    c0, c1 = count_range
    r0, r1 = radius_range
    fig, ax = plt.subplots(figsize=figsize)
    for i, n in enumerate(counts):
        frac = (min(float(n), c1) - c0) / max(c1 - c0, 1e-9)
        r_a = r0 + frac * (r1 - r0)                 # Angstrom
        d_pt = 2 * r_a * px_per_angstrom * 72.0 / 300.0   # px -> points at 300 dpi
        y = 0.88 - i * 0.18
        ax.scatter([0.28], [y], s=d_pt ** 2, facecolor=color,
                   edgecolor="white", linewidth=0.6)
        ax.text(0.62, y, f"{n}", va="center", ha="left", fontsize=F["tick"])
    ax.text(0.0, 0.99, label, va="bottom", ha="left", fontsize=F["label"])
    ax.set_xlim(0, 1)
    ax.set_ylim(0.88 - len(counts) * 0.18, 1.10)
    ax.set_axis_off()
    return fig, ax
