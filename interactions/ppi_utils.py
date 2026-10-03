"""
Shared loaders, statistics and drawing helpers for the protein-interaction
figures (Figure 6 and Extended Data 9), imported by Fig6_a-d_and_associated_ED.ipynb. The
structure panels are drawn by `render_structure` and `render_surface`, which
drive PyMOL.

Like the top-level utils.py, this holds the loaders and the drawing code, so the
notebook reads as the analysis.

Importing this module applies the publication style (Arial, editable PDF
text). Run from the repository root: paths are relative to it.
"""
from __future__ import annotations

import matplotlib
matplotlib.use('Agg')
import json
import os
import subprocess
import tempfile
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.colors as mcolors
import matplotlib.font_manager as fm
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd
import yaml
from Bio.PDB import PDBParser
from matplotlib.cm import ScalarMappable
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Patch, Rectangle
from matplotlib.ticker import FuncFormatter
from matplotlib_venn import venn3, venn3_circles
from scipy.cluster.hierarchy import dendrogram, fcluster, leaves_list, linkage
from scipy.spatial.distance import squareform
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

def reset_style(publication: bool = True) -> None:
    """Start a panel from matplotlib's defaults plus the publication style.

    Panels change rcParams as they draw, so without this each would inherit the
    state the previous one left behind. The Venn diagram is drawn without the
    publication style (publication=False).
    """
    matplotlib.rcdefaults()
    if publication:
        setup_publication_style()


# ===========================================================================
# Figure style
# ===========================================================================

# Where Arial is usually installed: the Microsoft core-fonts package on Linux,
# the per-user font directory, and the macOS / Windows system font folders.
ARIAL_DIRS = [Path('/usr/share/fonts/truetype/msttcorefonts'),
              Path.home() / '.local' / 'share' / 'fonts',
              Path('/Library/Fonts'), Path('/System/Library/Fonts/Supplemental'),
              Path('C:/Windows/Fonts')]


def apply_arial() -> None:
    """Register Arial and make it the default sans-serif font.

    Idempotent: safe to call multiple times. If Arial is not installed the
    figures fall back to Liberation Sans (metrically identical to Arial) or
    DejaVu Sans, and a note is printed.
    """
    for d in ARIAL_DIRS:
        if d.is_dir():
            for f in sorted(d.glob('[Aa]rial*.ttf')):
                fm.fontManager.addfont(str(f))
    if 'Arial' not in {f.name for f in fm.fontManager.ttflist}:
        print('Arial not found; figures will use Liberation Sans or DejaVu Sans. '
              'Install Arial (e.g. the ttf-mscorefonts package) to match the paper.')
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Liberation Sans", "DejaVu Sans"]
    # Math text (e.g. the volcano's -log10 axis label) in Arial too, rather than
    # matplotlib's default DejaVu math font.
    plt.rcParams.update({"mathtext.fontset": "custom", "mathtext.rm": "Arial",
                         "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold"})
    # 42 = TrueType, embeds as editable text in PDFs (default 3 = Type 3,
    # which Illustrator can't edit cleanly).
    plt.rcParams["pdf.fonttype"] = 42
    # "none" tells matplotlib to leave SVG text as <text> elements rather
    # than converting glyphs to <path> outlines.
    plt.rcParams["svg.fonttype"] = "none"



# The violin figures are assembled side by side in the manuscript, so they must
# all drop in at the same scale. Two rules make that work:
#
#   1. A FIXED height and margins specified in INCHES, not axes fractions.
#      Fractional margins scale with the canvas, so a narrower figure would get
#      a narrower left margin and clip its y-axis label. Absolute margins give
#      every panel identical space for labels whatever its width.
#
#   2. Width proportional to the number of violins, so each violin is the same
#      physical width everywhere: a 3-category panel is narrower than a
#      4-category one rather than stretching three violins over the same space.
#
# Callers must NOT call subplots_adjust afterwards - violin_figure() has already
# set it, and a second call would silently discard this geometry. That matters for
# more than tidiness: the margins fix the AXES size, so a panel that widens one
# margin to fit extra text ends up with smaller violins than its neighbours even
# though the canvas still matches. Any text that does not fit belongs in the
# figure legend, not in the figure. They must also
# save WITHOUT bbox_inches='tight': that flag expands the saved box to enclose
# any artist overflowing the canvas (these figures' italic captions do), which
# makes the final PDF size depend on caption length rather than on figsize and is
# exactly how the panels drifted out of alignment in the first place.
VIOLIN_FIG_H = 5.12          # in - shared height for every violin panel
VIOLIN_PER_CAT_W = 1.28      # in - horizontal space per violin
VIOLIN_PANEL_GAP = 1.13      # in - gap between the activity and abundance panels
VIOLIN_MARGIN_IN = {         # in - reserved for labels, title and caption
    "left": 1.15,            #      y-axis label + tick labels
    "right": 0.26,
    "top": 0.95,             #      two-line suptitle
    "bottom": 0.95,          #      two-line category labels + counts
}



def violin_figure(n_cats: int, n_panels: int = 2):
    """
    Create a violin figure with the standard manuscript geometry.

    Parameters
    ----------
    n_cats : int
        Number of violin categories per panel. Sets the width.
    n_panels : int, default 2
        Number of side-by-side panels (activity, abundance).

    Returns
    -------
    (fig, axes) with subplots_adjust already applied. Do not adjust again.
    """
    panel_w = VIOLIN_PER_CAT_W * n_cats
    fig_w = (VIOLIN_MARGIN_IN["left"] + VIOLIN_MARGIN_IN["right"]
             + n_panels * panel_w + (n_panels - 1) * VIOLIN_PANEL_GAP)
    fig, axes = plt.subplots(1, n_panels, figsize=(fig_w, VIOLIN_FIG_H))
    fig.subplots_adjust(
        left=VIOLIN_MARGIN_IN["left"] / fig_w,
        right=1.0 - VIOLIN_MARGIN_IN["right"] / fig_w,
        top=1.0 - VIOLIN_MARGIN_IN["top"] / VIOLIN_FIG_H,
        bottom=VIOLIN_MARGIN_IN["bottom"] / VIOLIN_FIG_H,
        wspace=VIOLIN_PANEL_GAP / panel_w,
    )
    return fig, axes



# ===========================================================================
# Structure-source colours
# ===========================================================================

PDB = '#6a51a3'   # RCSB PDB (experimental)          — purple
AF  = '#c51b7a'   # Predictomes / AlphaFold-Multimer  — magenta
RF  = '#b8860b'   # RoseTTAFold2-PPI / Zhang et al.   — gold



# ===========================================================================
# Loading per-structure results; the significance rule
# ===========================================================================

# Every manuscript figure (Fig 5a Goodsell/barplots, 5b/c volcanoes and the
# interaction network) must call an interaction "significant"
# the same way, or the same data tells different stories in different panels.
# This is the one place that rule lives.
#
# The rule operates on the *aggregated* interaction table produced by
# aggregate_volcano_data (one row per scored x interactor x
# predictor x assay, with BH-FDR recomputed across the aggregated tests per
# assay). An interaction is significant iff:
#
#     BH-FDR < SIG_ALPHA   AND   |Cohen's d| >= SIG_EFFECT
#
# Cohen's d (a true magnitude effect size in pooled-SD units) is used rather
# than log fold change so the effect threshold is comparable across proteins
# with different DMS dynamic range.
SIG_ALPHA: float = 0.05
SIG_EFFECT: float = 0.2



def flag_significant(
    agg: pd.DataFrame,
    *,
    alpha: float = SIG_ALPHA,
    effect: float = SIG_EFFECT,
) -> pd.Series:
    """
    Boolean mask of aggregated interactions passing the canonical rule.

    Parameters
    ----------
    agg : pd.DataFrame
        Aggregated interaction table from aggregate_volcano_data. Must contain
        ``neg_log_pval`` (= -log10 of the aggregated BH-FDR) and ``cohens_d``.
    alpha : float, default SIG_ALPHA
        FDR cutoff. An interaction passes when BH-FDR < alpha, i.e.
        ``neg_log_pval >= -log10(alpha)``.
    effect : float, default SIG_EFFECT
        Minimum |Cohen's d|.

    Returns
    -------
    pd.Series
        Boolean mask aligned to ``agg``'s index. NaN p / effect -> False.
    """
    fdr_cutoff = -np.log10(alpha)
    mask = (agg["neg_log_pval"] >= fdr_cutoff) & (agg["cohens_d"].abs() >= effect)
    return mask.fillna(False)



def load_combined_results(
    predictomes_dir: Union[str, Path],
    zhangetal_dir: Union[str, Path],
    pdb_dir: Optional[Union[str, Path]] = None,
    include_skipped: bool = False,
    apply_clash_filter: bool = True,
) -> pd.DataFrame:
    """
    Load and combine interaction results from multiple output directories.

    Parameters
    ----------
    predictomes_dir : str or Path
        Path to predictomes_output directory
    zhangetal_dir : str or Path
        Path to Zhangetal_output directory
    pdb_dir : str or Path, optional
        Path to pdb_output directory (RCSB experimental structures)
    include_skipped : bool, default False
        Whether to include skipped interactions

    Returns
    -------
    pd.DataFrame
        Combined DataFrame with all interactions
    """
    predictomes_dir = Path(predictomes_dir)
    zhangetal_dir = Path(zhangetal_dir)

    dfs = []

    # Load predictomes results
    predictomes_file = predictomes_dir / "all_interactions.csv"
    if predictomes_file.exists():
        df_pred = pd.read_csv(predictomes_file)
        dfs.append(df_pred)

    # Load Zhang et al. results
    zhangetal_file = zhangetal_dir / "all_interactions.csv"
    if zhangetal_file.exists():
        df_zhang = pd.read_csv(zhangetal_file)
        dfs.append(df_zhang)

    # Load RCSB PDB results (optional third source)
    if pdb_dir is not None:
        pdb_dir = Path(pdb_dir)
        pdb_file = pdb_dir / "all_interactions.csv"
        if pdb_file.exists():
            df_pdb = pd.read_csv(pdb_file)
            dfs.append(df_pdb)

    if not dfs:
        raise FileNotFoundError(
            f"No all_interactions.csv files found in {predictomes_dir} or {zhangetal_dir}"
        )

    # Combine
    df = pd.concat(dfs, ignore_index=True)

    # Filter skipped if requested
    if not include_skipped:
        df = df[df['skipped'] == False].copy()

    # Add interaction name column
    df['interaction_name'] = df['scored_protein_name'] + '-' + df['interactor_name']

    # Merge inter-chain clash data if available
    clash_file = Path('output/interactions/interchain_cbeta_clashes.csv')
    if clash_file.exists():
        clash_df = pd.read_csv(clash_file)
        # Join on structure name (one clash row per unique structure)
        clash_cols = ['structure', 'n_confident_clashes_lt2A', 'n_confident_clashes_lt3A', 'min_cbeta_dist']
        clash_subset = clash_df[clash_cols].drop_duplicates(subset='structure')
        df = df.merge(clash_subset, on='structure', how='left')
        # Fill NaN (structures not in clash file) with 0 clashes
        df['n_confident_clashes_lt2A'] = df['n_confident_clashes_lt2A'].fillna(0).astype(int)
        df['n_confident_clashes_lt3A'] = df['n_confident_clashes_lt3A'].fillna(0).astype(int)
        df['min_cbeta_dist'] = df['min_cbeta_dist'].fillna(99.0)

        # Exclude structures with confident inter-chain C-beta clashes (<3 A,
        # both residues pLDDT >= 60).  These represent physically implausible
        # docking poses where chains interpenetrate at confident residues.
        # `apply_clash_filter=False` keeps the merged clash columns but skips the
        # drop, so callers can count pre- vs post-filter structures themselves.
        if apply_clash_filter:
            n_before = len(df)
            df = df[df['n_confident_clashes_lt3A'] == 0].copy()
            n_excluded = n_before - len(df)
            if n_excluded > 0:
                print(f"  Excluded {n_excluded} interactions with inter-chain C-beta clashes")

    return df



def prepare_volcano_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data for volcano plots by adding derived columns.

    Adds:
    - neg_log_pval: -log10(pval_mannwhitney) for per-structure data. For
      AGGREGATED data this column is already set by aggregate_volcano_data()
      from a properly recomputed BH-FDR, and is left untouched (filled only
      when absent) so the aggregated volcano y-axis stays FDR-based.
    - neg_log_pval_fdr: -log10(pval_fdr)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with interaction results

    Returns
    -------
    pd.DataFrame
        DataFrame with additional columns for volcano plots
    """
    df = df.copy()

    # Add -log10(p-value) columns, handling zeros.
    #
    # `neg_log_pval` is the volcano y-axis. For per-structure data it is the
    # raw Mann-Whitney p. For aggregated data, aggregate_volcano_data() has
    # already set it from the recomputed BH-FDR — so only fill it when absent
    # to avoid clobbering that value. `neg_log_pval_fdr` always tracks pval_fdr.
    if 'neg_log_pval' not in df.columns:
        df['neg_log_pval'] = -np.log10(df['pval_mannwhitney'].clip(lower=1e-300))
    df['neg_log_pval_fdr'] = -np.log10(df['pval_fdr'].clip(lower=1e-300))

    return df



# ===========================================================================
# Pooling structures into interactions; volcano panels
# ===========================================================================

# Try to import adjustText for label adjustment
try:
    from adjustText import adjust_text
    HAS_ADJUSTTEXT = True
except ImportError:
    HAS_ADJUSTTEXT = False



# Predictor → palette mapping used across publication figures.
# ColorBrewer Dark2: Predictomes (purple), Zhang (orange), RCSB PDB (green).
PREDICTOR_PALETTE = {
    'Predictomes': '#c51b7a',
    'Zhang_et_al': '#b8860b',
    'RCSB_PDB':    '#6a51a3',
}
# Up/down significance-box colors. Volcano convention: increased-at-interface
# (Cohen's d > 0) on the RIGHT in red; decreased (d < 0) on the LEFT in blue.
# The box is a light tint of these; the in-box count is drawn in a darker shade
# of the SAME hue ("similar colored but not same color font to the background").
UP_BOX_COLOR = '#c0392b'    # red  — increased at interface (gain)
DOWN_BOX_COLOR = '#2c6fa6'  # blue — decreased at interface (loss)
# Display gene-symbol aliases used in the manuscript figures, so highlighted
# interaction labels read as the names the paper uses (e.g. RAF1 -> CRAF).
DISPLAY_NAME = {'RAF1': 'CRAF', 'PTPN11': 'SHP2',
                'MAP2K1': 'MEK1', 'MAP2K2': 'MEK2', 'MET': 'HGFR'}



def _disp(name: str) -> str:
    """Map a UniProt gene symbol to its manuscript display name."""
    return DISPLAY_NAME.get(name, name)



# Distinct colours for the featured (highlighted) interactions. Each featured
# interaction's structures are ringed in one of these and its name label is
# drawn in the same colour, so the reader ties a label to its circles by colour
# rather than tracing leader lines. Chosen to contrast with the three predictor
# point fills (purple/orange/green) and with each other.
# Front-loaded so the first (most-used) colours are maximally distinct from each
# other AND from the three predictor point fills (purple PDB / pink AF / gold RF);
# the fill-adjacent hues (magenta, violet, olive) sit at the end so they're only
# reached if many interactions are featured.
HIGHLIGHT_PALETTE = [
    '#e6194B',  # red
    '#4363d8',  # blue
    '#3cb44b',  # green
    '#f58231',  # orange
    '#42d4f4',  # cyan
    '#9A6324',  # brown
    '#469990',  # teal
    '#000000',  # black
    '#800000',  # maroon
    '#000075',  # navy
    '#f032e6',  # magenta
    '#911eb4',  # violet
    '#808000',  # olive
]



def build_highlight_color_map(
    df: pd.DataFrame,
    highlight_pairs: Optional[List[Tuple[str, str]]],
) -> Dict[Tuple[str, str], str]:
    """
    Assign a distinct colour to each featured interaction present in ``df``.

    Returns a map from a *directed* (scored_protein_name, interactor_name) pair
    to a hex colour. Matching against ``highlight_pairs`` is direction-agnostic
    (a pair A:B claims both the A-scored and B-scored points), and each present
    direction gets its own colour so the two readouts of the same interaction
    are individually identifiable.

    The order is deterministic — by the order of ``highlight_pairs`` then by the
    directed pair — so passing the *full* aggregated table once and reusing the
    map across every panel/figure keeps a given interaction the same colour
    everywhere.
    """
    cols = {'scored_protein_name', 'interactor_name'}
    if not highlight_pairs or not cols <= set(df.columns):
        return {}
    # Only interactions with >=1 significant structure are ever ringed/labelled
    # (non-significant complexes are not called out), so don't spend palette
    # colours on pairs that will never be drawn — that keeps the drawn rings as
    # colour-diverse as possible. Uses the same rule as the panels
    # (FDR<0.05 & |Cohen's d|>=0.2); falls back to all-present if the needed
    # columns aren't available.
    d = df
    if {'neg_log_pval', 'cohens_d'} <= set(df.columns):
        sig = ((df['neg_log_pval'] >= -np.log10(0.05)) &
               (df['cohens_d'].abs() >= 0.2))
        if sig.any():
            d = df[sig]
    present = set(zip(d['scored_protein_name'], d['interactor_name']))
    ordered: List[Tuple[str, str]] = []
    for s, i in highlight_pairs:
        fs = frozenset((s, i))
        for g in sorted(p for p in present if frozenset(p) == fs):
            if g not in ordered:
                ordered.append(g)
    return {g: HIGHLIGHT_PALETTE[k % len(HIGHLIGHT_PALETTE)]
            for k, g in enumerate(ordered)}



def _darken(color, factor: float = 0.5):
    """Return a darker shade of `color` (same hue, RGB scaled toward black).

    Used so a count drawn inside a light-tinted box is the same colour family
    as the box but clearly darker/readable.
    """
    r, g, b = mcolors.to_rgb(color)
    return (r * factor, g * factor, b * factor)



def _trim_zeros_formatter(value, _pos=None) -> str:
    """Tick formatter that strips trailing zeros: 1.0->'1', 0.5->'0.5', -2.0->'-2'.

    Applied to the Cohen's d (x) axis of every volcano panel so the tick
    labels read identically across panels even when matplotlib picks integer
    ticks for a wide range and half-integer ticks for a narrow one (e.g. the
    activity panel showed '-2, -1' while abundance showed '-2.0, -1.0'). The
    axis *limits* can still differ; only the label style is unified.
    """
    if value == 0:
        return '0'              # avoid '-0' when a tick lands on negative zero
    return f'{value:g}'         # %g drops trailing zeros and the decimal point



def setup_publication_style() -> None:
    """
    Configure matplotlib rcParams for publication figures.

    Delegates font registration to `apply_arial()` (above). That helper:
      - Registers Arial with matplotlib's font manager (falling back to
        Liberation Sans, then DejaVu Sans, if Arial is not installed).
      - Sets ``pdf.fonttype = 42`` and ``svg.fonttype = 'none'`` so PDF/SVG
        exports keep text as editable objects rather than glyph paths.

    On top of that, this function applies volcano-figure-specific tick /
    label sizes and turns off the top + right spines globally.

    Notes
    -----
    Idempotent. Safe to call at module import time and/or before each figure.
    """
    apply_arial()
    plt.rcParams.update({
        'ps.fonttype': 42,
        'axes.linewidth': 0.8,
        'axes.labelsize': 8,
        'axes.titlesize': 9,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.major.size': 3,
        'ytick.major.size': 3,
        'legend.fontsize': 7,
        'legend.frameon': False,
        'mathtext.default': 'regular',  # match body sans-serif (no italic)
    })



def _parse_positions(pos_str) -> set:
    """Parse comma-separated position string into a set of ints."""
    if pd.isna(pos_str) or str(pos_str).strip() == '':
        return set()
    return {int(x) for x in str(pos_str).split(',')}



def _mean_pairwise_jaccard(sets: List[set]) -> float:
    """
    Mean pairwise Jaccard index across a list of sets.

    Returns NaN if fewer than 2 non-empty sets, 0.0 if all pairs
    have zero intersection.
    """
    non_empty = [s for s in sets if len(s) > 0]
    if len(non_empty) < 2:
        return np.nan
    jaccards = []
    for i in range(len(non_empty)):
        for j in range(i + 1, len(non_empty)):
            union = len(non_empty[i] | non_empty[j])
            if union == 0:
                continue
            jaccards.append(len(non_empty[i] & non_empty[j]) / union)
    return np.mean(jaccards) if jaccards else 0.0



def _assign_binding_mode_clusters(
    dfp: pd.DataFrame,
    group_cols: List[str],
    *,
    similarity_threshold: float = 0.3,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Within each (scored × interactor × assay × predictor) group, sub-cluster
    structures by interface-position similarity so multi-mode binding (e.g.,
    SOS1–KRAS REM-domain vs CDC25-catalytic-domain in Zhang et al.) is not
    averaged together.

    Method
    ------
    For each group with ≥2 structures, compute pairwise Jaccard distance
    between the interface position sets of the structures (distance =
    1 − Jaccard similarity). Run **complete-linkage hierarchical clustering**
    at distance threshold `1 − similarity_threshold` (default 0.7, i.e.
    Jaccard sim < 0.3 ⇒ different mode).

    Why complete linkage: two structures end up in the same cluster only if
    *every* pair within the cluster has Jaccard ≥ similarity_threshold. This
    prevents chaining (A–B sim 0.5, B–C sim 0.5, A–C sim 0.05 — which single
    linkage would merge into one mode despite A and C disagreeing).

    Why Jaccard 0.3 cutoff: matches the existing diagnostic threshold used
    to flag "inconsistent interface predictions" in NOTEBOOK 2026-04-08.
    Below this, structures genuinely place the interface elsewhere.

    Parameters
    ----------
    dfp : pd.DataFrame
        Per-structure interaction rows. Must contain 'interface_positions'
        as comma-separated position strings.
    group_cols : list of str
        Group key columns. Clustering is independent per group.
    similarity_threshold : float
        Jaccard similarity below this is treated as "different mode."
    verbose : bool
        Print per-group split events for inspection.

    Returns
    -------
    pd.DataFrame
        Copy of `dfp` with a new integer column '_cluster_id' assigning each
        row to a binding-mode cluster within its group (zero-indexed).
    """
    out = dfp.copy()
    out['_cluster_id'] = 0  # singletons and same-mode groups stay 0

    distance_threshold = 1.0 - similarity_threshold
    n_split = 0

    for key, grp in out.groupby(group_cols, dropna=False, sort=False):
        if len(grp) <= 1:
            continue  # singleton — no clustering to do

        iface_sets = [_parse_positions(row['interface_positions'])
                      for _, row in grp.iterrows()]
        n = len(iface_sets)

        # Pairwise Jaccard distance matrix.
        dist = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                a, b = iface_sets[i], iface_sets[j]
                u = len(a | b)
                sim = (len(a & b) / u) if u > 0 else 0.0
                dist[i, j] = dist[j, i] = 1.0 - sim

        # If all distances are zero (identical interfaces), no need to cluster.
        if dist.max() == 0:
            continue

        # scipy expects a condensed distance vector for linkage().
        condensed = squareform(dist, checks=False)
        Z = linkage(condensed, method='complete')
        cluster_labels = fcluster(Z, t=distance_threshold, criterion='distance')

        # Write back zero-indexed cluster IDs to the group's rows.
        for idx, cid in zip(grp.index, cluster_labels):
            out.at[idx, '_cluster_id'] = int(cid) - 1

        n_distinct = len(set(cluster_labels))
        if n_distinct > 1:
            n_split += 1
            if verbose:
                key_str = ' / '.join(str(k) for k in (key if isinstance(key, tuple) else (key,)))
                sizes = Counter(cluster_labels)
                size_str = ', '.join(f'{c} structures' for c in sorted(sizes.values(), reverse=True))
                print(f'    SPLIT [{key_str}] -> {n_distinct} modes ({size_str})')

    if verbose:
        print(f'  Sub-clustering: {n_split} groups split into multiple modes')

    return out



def _load_domain_info(
    source: Optional[Union[dict, str, Path]],
) -> Optional[Dict[str, Dict[str, str]]]:
    """Load domain_info.json if a path is given; pass through dicts; tolerate None."""
    if source is None:
        # Try the conventional project location.
        default_path = Path('interactions/domain_info.json')
        if default_path.exists():
            with open(default_path) as fh:
                return json.load(fh)
        return None
    if isinstance(source, dict):
        return source
    with open(source) as fh:
        return json.load(fh)



def _label_binding_mode(
    interface_positions: set,
    scored_protein_name: str,
    domain_info: Optional[Dict[str, Dict[str, str]]],
) -> str:
    """
    Produce a short, biologically meaningful mode label from an interface.

    Strategy
    --------
    1. If `domain_info` has an entry for `scored_protein_name`, look up each
       interface position and tally domain hits.
       a. If the top single domain covers ≥40% of positions → use it.
       b. Else if top-2 domains together cover ≥50% AND each individual
          share ≥20% → return combined label "DomA+DomB" (handles e.g.
          SOS1's RasGEFN+RasGEF inter-domain hinge mode).
    2. Fallback to a position range like "res 12-40".

    Parameters
    ----------
    interface_positions : set of int
        Consensus interface positions for the cluster (typically the union
        across structures in the cluster).
    scored_protein_name : str
        Gene symbol for the scored protein (which side the interface is on).
    domain_info : dict, optional
        {protein_name: {position_str: domain_name}}.

    Returns
    -------
    str
        A short label suitable for appending to the interaction name.
    """
    if not interface_positions:
        return 'unknown'

    if domain_info and scored_protein_name in domain_info:
        protein_map = domain_info[scored_protein_name]
        domains = [protein_map.get(str(p), '') for p in interface_positions]
        domains = [_clean_domain_name(d) for d in domains if d]
        if domains:
            n_total = len(interface_positions)
            counts = Counter(domains)
            top_items = counts.most_common(2)
            top_dom, top_n = top_items[0]
            top_share = top_n / n_total
            if top_share >= 0.4:
                return top_dom
            # Combined-domain fallback for inter-domain interfaces.
            if len(top_items) >= 2:
                second_dom, second_n = top_items[1]
                if (top_n + second_n) / n_total >= 0.5 and (second_n / n_total) >= 0.2:
                    # Order alphabetically for stable labels across runs.
                    a, b = sorted([top_dom, second_dom])
                    return f'{a}+{b}'

    lo, hi = min(interface_positions), max(interface_positions)
    return f'res {lo}-{hi}'



def _clean_domain_name(raw: str) -> str:
    """Trim suffixes like '_rpt1', ' superfamily' from domain names."""
    name = raw.replace(' superfamily', '').strip()
    # Pfam-style numeric suffixes (e.g. 'HFD_SOS1_rpt1') stay informative;
    # keep them. Just drop leading underscores or trailing whitespace.
    return name



def aggregate_volcano_data(
    df: pd.DataFrame,
    *,
    subcluster_modes: bool = True,
    mode_similarity_threshold: float = 0.3,
    domain_info: Optional[Union[dict, str, Path]] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Collapse per-structure rows into one row per biological binding mode per
    (scored_protein, interactor, assay, predictor).

    With `subcluster_modes=True` (default), within-group sub-clustering by
    interface-position Jaccard distance ensures that genuinely distinct
    binding modes (e.g., SOS1–KRAS REM-domain vs CDC25-domain in Zhang et al.)
    are emitted as **separate points** instead of being medianed together.
    See `_assign_binding_mode_clusters` for the algorithm.

    Aggregation rules (within each cluster):
    - pval_mannwhitney: geometric mean across structures (= exp of mean of
      log p-values) — the per-interaction evidence summary
    - pval_fdr: Benjamini-Hochberg FDR *recomputed* across the aggregated
      interaction tests per assay (all predictors pooled). NOT the geometric
      mean of main.py's per-structure FDR, which is uncalibrated once
      structures are collapsed. This is the volcano y-axis source.
    - neg_log_pval: -log10(pval_fdr) — the volcano y-axis
    - log_fold_change: median
    - cliffs_delta, auroc, cohens_d: median
    - n_interface_positions: median (for point sizing)
    - n_structures: count of structures contributing

    Output columns added by sub-clustering:
    - binding_mode: short mode label (dominant domain or residue range)
    - n_modes: number of distinct modes detected for the parent
      (scored × interactor × assay × predictor) group
    - cluster_size: number of structures in this mode's cluster

    Overlap flags (for multi-structure clusters — *within-cluster*):
    - fragment_jaccard: mean pairwise Jaccard of covered positions
      (interface + control). Low → structures in this cluster model
      different protein regions despite shared mode.
    - interface_jaccard: mean pairwise Jaccard of interface positions
      (within cluster). After sub-clustering, this should be high (≥
      `mode_similarity_threshold`) by construction.

    Parameters
    ----------
    df : pd.DataFrame
        Per-structure interaction results. Must contain 'interface_positions'
        and 'control_positions' columns (comma-separated position strings).
    subcluster_modes : bool, default True
        Whether to sub-cluster structures by binding mode within each group.
        Set False to recover original behavior (one row per group, possibly
        averaging across distinct binding modes).
    mode_similarity_threshold : float, default 0.3
        Jaccard similarity threshold below which two structures are deemed
        to represent different binding modes.
    domain_info : dict or path, optional
        Domain annotations for biological mode labels. If None, attempts to
        load `domain_info.json` from the working directory.
    verbose : bool, default False
        Print per-group splits and a summary count.

    Returns
    -------
    pd.DataFrame
        One row per unique biological mode per predictor, with aggregated
        statistics and overlap flags. Mode-disambiguated `interaction_name`
        appended with mode label only when n_modes > 1 for the parent group.
    """
    dfp = df.copy()

    # Ensure numeric columns
    numeric_cols = [
        'log_fold_change', 'pval_fdr', 'pval_mannwhitney',
        'cliffs_delta', 'auroc', 'cohens_d', 'n_interface_positions',
        'n_control_positions',
    ]
    for c in numeric_cols:
        if c in dfp.columns:
            dfp[c] = pd.to_numeric(dfp[c], errors='coerce')

    # Drop rows without valid statistics
    dfp = dfp.dropna(subset=['log_fold_change', 'pval_fdr'])

    group_cols = ['scored_protein', 'scored_protein_name', 'interactor',
                  'interactor_name', 'assay', 'predictor']

    has_positions = ('interface_positions' in dfp.columns and
                     'control_positions' in dfp.columns)

    # --- Phase 0: assign per-row cluster ID (binding mode) within each group ---
    # When sub-clustering is enabled, structures within a group with disagreeing
    # interface positions land in different clusters; the aggregation downstream
    # then emits one row per cluster instead of one row per group.
    if subcluster_modes and has_positions:
        dfp = _assign_binding_mode_clusters(
            dfp, group_cols,
            similarity_threshold=mode_similarity_threshold,
            verbose=verbose,
        )
    else:
        dfp['_cluster_id'] = 0

    # Extended group key for the aggregation: include cluster ID.
    cluster_group_cols = group_cols + ['_cluster_id']

    def _geomean_pval(s):
        """Geometric mean of p-values via log-space averaging."""
        log_vals = np.log(s.clip(lower=1e-300))
        return np.exp(log_vals.mean())

    def _cv(s):
        """Coefficient of variation (std/mean). Returns NaN for single values."""
        if len(s) < 2 or s.mean() == 0:
            return np.nan
        return s.std() / s.mean()

    # --- Phase 1: numeric aggregation per (group × cluster) ---
    agg = dfp.groupby(cluster_group_cols, dropna=False).agg(
        log_fold_change=('log_fold_change', 'median'),
        log_fold_change_iqr=('log_fold_change', lambda s: s.quantile(0.75) - s.quantile(0.25) if len(s) > 1 else 0.0),
        pval_fdr=('pval_fdr', _geomean_pval),
        pval_mannwhitney=('pval_mannwhitney', _geomean_pval),
        cliffs_delta=('cliffs_delta', 'median'),
        auroc=('auroc', 'median'),
        cohens_d=('cohens_d', 'median'),
        n_interface_positions=('n_interface_positions', 'median'),
        n_interface_positions_cv=('n_interface_positions', _cv),
        n_control_positions=('n_control_positions', 'median'),
        n_structures=('structure', 'nunique'),
    ).reset_index()

    # cluster_size = same as n_structures here (every structure contributes
    # to exactly one cluster), but we track it explicitly for clarity.
    agg['cluster_size'] = agg['n_structures']

    # n_modes = how many distinct clusters exist for the parent group.
    n_modes_per_group = (
        dfp.groupby(group_cols, dropna=False, sort=False)['_cluster_id']
        .nunique()
        .reset_index(name='n_modes')
    )
    agg = agg.merge(n_modes_per_group, on=group_cols, how='left')

    # --- Phase 2: overlap metrics + binding-mode label per cluster ---
    domain_map = _load_domain_info(domain_info) if subcluster_modes else None

    if has_positions:
        overlap_records = []
        for key, grp in dfp.groupby(cluster_group_cols, dropna=False, sort=False):
            iface_sets = [_parse_positions(row['interface_positions'])
                          for _, row in grp.iterrows()]
            ctrl_sets = [_parse_positions(row['control_positions'])
                         for _, row in grp.iterrows()]
            frag_sets = [iface | ctrl for iface, ctrl in zip(iface_sets, ctrl_sets)]

            # Consensus interface = union across structures within cluster
            consensus_iface = set().union(*iface_sets) if iface_sets else set()

            # Mode label: domain-based if possible, else residue range.
            scored_name = dict(zip(cluster_group_cols, key))['scored_protein_name']
            mode_label = _label_binding_mode(
                consensus_iface, scored_name, domain_map,
            )

            overlap_records.append({
                **dict(zip(cluster_group_cols, key)),
                'interface_jaccard': _mean_pairwise_jaccard(iface_sets),
                'fragment_jaccard': _mean_pairwise_jaccard(frag_sets),
                'binding_mode': mode_label,
                'consensus_interface_positions': ','.join(
                    str(p) for p in sorted(consensus_iface)
                ),
            })

        overlap_df = pd.DataFrame(overlap_records)
        agg = agg.merge(overlap_df, on=cluster_group_cols, how='left')
    else:
        agg['interface_jaccard'] = np.nan
        agg['fragment_jaccard'] = np.nan
        agg['binding_mode'] = 'unknown'
        agg['consensus_interface_positions'] = ''

    # Recompute multiple-testing correction AFTER collapsing structures.
    #
    # main.py computes BH-FDR across all per-STRUCTURE tests. Once we collapse
    # structures to one row per biological interaction (above), that per-
    # structure FDR is no longer calibrated, and geometric-mean-ing it across
    # structures (the `pval_fdr` aggregation in Phase 1) does not restore
    # calibration. So we re-apply Benjamini-Hochberg across the *aggregated*
    # interaction tests, treating each assay as one multiple-testing family
    # (all predictors pooled — the volcano panels present experimental and
    # predicted interactions together, so they belong to the same family).
    # The per-interaction input is the geometric-mean raw Mann-Whitney p.
    from statsmodels.stats.multitest import multipletests
    agg['pval_fdr'] = np.nan
    for _assay in agg['assay'].dropna().unique():
        # Only correct over rows with a finite raw p; NaN raw p -> NaN FDR,
        # which the plotters drop. Guards against geomean producing NaN when a
        # structure had a missing Mann-Whitney p.
        mask = (agg['assay'] == _assay) & agg['pval_mannwhitney'].notna()
        if mask.any():
            pvals = agg.loc[mask, 'pval_mannwhitney'].clip(lower=1e-300)
            agg.loc[mask, 'pval_fdr'] = multipletests(pvals, method='fdr_bh')[1]

    # Volcano y-axis = -log10(aggregated BH-FDR). prepare_volcano_data() must
    # NOT overwrite this (it is guarded to fill neg_log_pval only when absent).
    agg['neg_log_pval'] = -np.log10(agg['pval_fdr'].clip(lower=1e-300))

    # Disambiguate within-group duplicate labels. RCSB self-interactions like
    # MAP2K1–MAP2K1 (4 different crystal-packing interfaces all on PKc_MAP2K1)
    # produce the same domain-derived label for multiple clusters. Suffix them
    # by descending cluster size so the largest mode keeps the bare label.
    if subcluster_modes:
        def _disambiguate(grp):
            if len(grp) <= 1:
                return grp
            grp = grp.sort_values(['cluster_size', 'cohens_d'],
                                  ascending=[False, True], kind='stable')
            seen: Counter = Counter()
            new_labels = []
            for label in grp['binding_mode']:
                seen[label] += 1
                new_labels.append(label if seen[label] == 1
                                  else f'{label} #{seen[label]}')
            grp = grp.copy()
            grp['binding_mode'] = new_labels
            return grp
        agg = (agg.groupby(group_cols, group_keys=False, sort=False)
               .apply(_disambiguate)
               .reset_index(drop=True))

    # Mode-disambiguated interaction name: append mode label *only when*
    # the parent group resolved into multiple modes. Single-mode groups keep
    # the clean "SCORED_INTERACTOR" label so the volcano isn't cluttered.
    base_name = agg['scored_protein_name'].astype(str) + '_' + agg['interactor_name'].astype(str)
    multi_mode_mask = agg['n_modes'].fillna(1).astype(int) > 1
    agg['interaction_name'] = np.where(
        multi_mode_mask,
        base_name + ' (' + agg['binding_mode'].fillna('mode?').astype(str) + ')',
        base_name,
    )

    # Drop the internal cluster ID before returning.
    return agg.drop(columns=['_cluster_id'])



def _draw_volcano_panel(
    ax: plt.Axes,
    groups: List[Tuple[pd.DataFrame, str, str]],
    *,
    x_col: str,
    effect_threshold: float,
    fdr_alpha: float,
    size_col: str,
    size_range: Tuple[float, float],
    annotate: bool,
    max_labels: int,
    panel_title: Optional[str] = None,
    title_color: Optional[str] = None,
    show_color_key: bool = False,
    color_key_order: Optional[List[int]] = None,
    count_split: Optional[List[Tuple[str, pd.DataFrame]]] = None,
    highlight_pairs: Optional[List[Tuple[str, str]]] = None,
    highlight_colors: Optional[Dict[Tuple[str, str], str]] = None,
    auto_place: bool = True,
    title_fs: float = 8,
    tick_fs: Optional[float] = None,
    hl_label_fs: float = 6.5,
    legend_fs: float = 6,
    show_counts: bool = True,
    corner_counts: Optional[Dict[str, Tuple[int, int]]] = None,
) -> Dict[str, float]:
    """
    Render a volcano panel onto an existing Axes.

    Supports either a single predictor (`groups` of length 1) or multiple
    predictors overlaid in the same panel (e.g., Predictomes + Zhang combined
    as "Predicted"). Each predictor is drawn in its own color but shares the
    same threshold line (the y-axis is already BH-FDR-corrected across the
    aggregated tests per assay, with all predictors pooled into one family).

    Plots:
      - x = effect size column (typically Cohen's d)
      - y = -log10(BH-FDR q-value)
      - dashed verticals at ±effect_threshold (e.g., Cohen's d=±0.2)
      - solid horizontal at -log10(fdr_alpha)  [FDR cutoff]
      - two shaded significance boxes: right (red, increased/"up") and left
        (blue, decreased/"down"); each is annotated with the number of
        significant interactions in it, in a darker shade of the box colour
      - top labels by |effect| among significant points (FDR < fdr_alpha)

    Parameters
    ----------
    ax : plt.Axes
        Target axes.
    groups : list of (pd.DataFrame, str, str)
        One or more (sub, color, predictor_name) triples. Each `sub` must
        contain x_col, 'neg_log_pval', size_col, 'interaction_name'.
    panel_title : str, optional
        Title above the panel. If None and `groups` has length 1, falls back
        to that predictor's name.
    title_color : str, optional
        Title text color. Defaults to the single group's color, or 'black'
        when multiple groups are overlaid.
    show_color_key : bool, default False
        If True (or auto-True when len(groups) > 1), draw an inline color key
        in the upper-left of the panel showing each predictor's color and
        name. Useful for combined panels.
    color_key_order : list of int, optional
        Indices into `groups` specifying the order of entries in the inline
        color key. Decoupled from draw order — useful when the most
        important / smallest series is drawn *last* (so it sits on top of
        the others) but should appear *first* in the legend. Default None
        uses the order of `groups`. Indices pointing at empty groups are
        skipped silently.
    count_split : list of (label, pd.DataFrame), optional
        If provided, replaces the single "n / sig" line above the panel with
        a per-subset breakdown — one line per (label, sub_df) pair, e.g.:
            PDB: n = 327, sig = 12
            Predicted: n = 1957, sig = 145
        Each sub_df is scored against the *same* FDR cutoff line drawn for
        the panel (the BH-FDR correction was already done across the union of
        all aggregated tests for this assay — one multiple-testing family —
        the *display* of counts is merely split). Each sub_df must contain
        `x_col` and
        `'neg_log_pval'`. Useful when a panel overlays experimental and
        predicted sources and you want the reader to see hit rates for each
        source separately.

    Returns
    -------
    dict
        Summary stats (n, n_sig) across all groups in the panel.
    """
    # Combine all group rows once for joint statistics (Bonferroni N, total
    # significant count, annotation pool). Per-group plotting still uses each
    # subset individually so points keep their predictor color.
    non_empty_groups = [(sub, color, name) for sub, color, name in groups
                        if len(sub) > 0]
    if not non_empty_groups:
        ax.text(0.5, 0.5, 'no data', transform=ax.transAxes,
                ha='center', va='center', color='gray', fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        return {'n': 0, 'n_sig': 0,
                'x_min': np.nan, 'x_max': np.nan,
                'y_min': np.nan, 'y_max': np.nan}

    combined = pd.concat([sub for sub, _, _ in non_empty_groups], ignore_index=True)
    n_total = len(combined)

    # Point sizes from median interface positions (same encoding across panels).
    # Compute the size scale across the combined set so the same residue count
    # produces the same dot size whether plotted alone or alongside another
    # predictor.
    if size_col in combined.columns:
        sz_all = pd.to_numeric(combined[size_col], errors='coerce').fillna(3)
        sz_min, sz_max = sz_all.min(), sz_all.max()

        def _size_for(sub):
            sz = pd.to_numeric(sub[size_col], errors='coerce').fillna(3)
            if sz_max > sz_min:
                return (size_range[0]
                        + (sz - sz_min) / (sz_max - sz_min)
                        * (size_range[1] - size_range[0]))
            return np.full(len(sub), (size_range[0] + size_range[1]) / 2)
    else:
        def _size_for(sub):
            return np.full(len(sub), 30)

    # Scatter each predictor group in its own color.
    for sub, color, _name in non_empty_groups:
        ax.scatter(
            sub[x_col], sub['neg_log_pval'],
            c=color, s=_size_for(sub), alpha=0.65,
            edgecolors='white', linewidths=0.3,
            zorder=3,
        )

    # Reference lines.  The y-axis is -log10(BH-FDR), recomputed across the
    # aggregated interaction tests per assay (see aggregate_volcano_data), so
    # the single meaningful horizontal cutoff is FDR = fdr_alpha. There is no
    # Bonferroni line: the points are already FDR-corrected and a second
    # Bonferroni pass on top would double-correct.
    fdr_y = -np.log10(fdr_alpha)
    ax.axvline(effect_threshold, color='gray', ls='--', lw=0.6, zorder=1)
    ax.axvline(-effect_threshold, color='gray', ls='--', lw=0.6, zorder=1)
    ax.axhline(fdr_y, color='black', ls='-', lw=0.5, zorder=1, alpha=0.5)

    # Significance: FDR < fdr_alpha AND |effect| >= effect_threshold.
    sig_mask = ((combined['neg_log_pval'] >= fdr_y) &
                (combined[x_col].abs() >= effect_threshold))
    n_sig = int(sig_mask.sum())

    # Up / down significance boxes. Shade the two significant quadrants — right
    # (d >= +effect_threshold, "up"/increased) in red, left (d <= -threshold,
    # "down"/decreased) in blue — and write how many interactions fall in each.
    # Capture the (autoscaled) limits first, shade within them, then re-lock so
    # the fills don't expand the view (same pattern as plot_volcano).
    n_up = int((sig_mask & (combined[x_col] >= effect_threshold)).sum())
    n_down = int((sig_mask & (combined[x_col] <= -effect_threshold)).sum())
    _xlim, _ylim = ax.get_xlim(), ax.get_ylim()
    if _xlim[1] > effect_threshold:
        ax.fill_between([effect_threshold, _xlim[1]], fdr_y, _ylim[1],
                        color=UP_BOX_COLOR, alpha=0.07, lw=0, zorder=0)
    if _xlim[0] < -effect_threshold:
        ax.fill_between([_xlim[0], -effect_threshold], fdr_y, _ylim[1],
                        color=DOWN_BOX_COLOR, alpha=0.07, lw=0, zorder=0)
    ax.set_xlim(_xlim)
    ax.set_ylim(_ylim)
    # Counts, top corners inside each box, in a darker shade of the box colour.
    # By default the plotted-point count per side; when `corner_counts` is given
    # (a dict {'down': (n_interfaces, n_pairs), 'up': (...)}), report interface
    # and unique-pair totals instead — so the numbers describe interfaces/pairs
    # even when points are grouped for display.
    if corner_counts:
        up_i, up_p = corner_counts['up']
        dn_i, dn_p = corner_counts['down']
        up_txt = f'{up_i} interfaces\n{up_p} pairs'
        dn_txt = f'{dn_i} interfaces\n{dn_p} pairs'
        corner_fs = 8.5
    else:
        up_txt, dn_txt, corner_fs = str(n_up), str(n_down), 9
    ax.text(0.985, 0.985, up_txt, transform=ax.transAxes,
            ha='right', va='top', fontsize=corner_fs, fontweight='bold',
            color=_darken(UP_BOX_COLOR), zorder=5, multialignment='right',
            linespacing=1.25)
    ax.text(0.015, 0.985, dn_txt, transform=ax.transAxes,
            ha='left', va='top', fontsize=corner_fs, fontweight='bold',
            color=_darken(DOWN_BOX_COLOR), zorder=5, multialignment='left',
            linespacing=1.25)

    # Resolve which rows are explicitly highlighted (featured in the paper),
    # so we can (a) exclude them from the generic auto-labels below and
    # (b) draw them with emphasis afterwards.
    has_pair_cols = {'scored_protein_name', 'interactor_name'} <= set(combined.columns)
    # An explicit 'volcano_label' column (set by the caller) names exactly which
    # points each featured label covers, e.g. one binding mode of a pair.
    by_label = 'volcano_label' in combined.columns
    if by_label:
        hl_mask = combined['volcano_label'].notna()
    elif highlight_pairs and has_pair_cols:
        # Match direction-agnostically: a highlight pair (A, B) flags both the
        # A-scored×B-interactor point AND the B-scored×A-interactor point, since
        # either protein may have been the DMS-scored side (e.g. KRAS–SOS1 is
        # tested both as KRAS scored and as SOS1 scored).
        hl_set = {frozenset((s, i)) for s, i in highlight_pairs}
        hl_mask = combined.apply(
            lambda r: frozenset((r['scored_protein_name'],
                                 r['interactor_name'])) in hl_set,
            axis=1,
        )
    else:
        hl_mask = pd.Series(False, index=combined.index)

    # Collect every label (generic + highlighted) into one list and position
    # them with a single adjust_text de-collision pass. No leader lines/arrows:
    # featured interactions are identified by colour (each interaction's rings
    # and its label share a colour), so labels just need to sit near — but
    # clear of — their circles. The de-collision pass uses a generous
    # point-expansion so labels are pushed off the (large) rings.
    label_texts: List = []
    fallback: List[Tuple[float, float, str, dict]] = []

    hl_colors = highlight_colors or {}

    # Generic: top points by |effect| among significant (highlighted excluded —
    # they get their own bold labels below).
    if annotate and n_sig > 0 and 'interaction_name' in combined.columns:
        sig = combined[sig_mask & ~hl_mask].copy()
        sig['_abs_eff'] = sig[x_col].abs()
        labels_per_side = max(max_labels // 2, 1)
        top_pos = sig[sig[x_col] > 0].nlargest(labels_per_side, '_abs_eff')
        top_neg = sig[sig[x_col] < 0].nlargest(labels_per_side, '_abs_eff')
        labelled = pd.concat([top_pos, top_neg]).head(max_labels)
        for _, row in labelled.iterrows():
            txt = row['interaction_name'].replace('_', '–')
            pt = (float(row[x_col]), float(row['neg_log_pval']))
            if HAS_ADJUSTTEXT:
                label_texts.append(ax.text(
                    pt[0], pt[1], txt, fontsize=5.5,
                    ha='center', va='bottom', style='italic'))
            else:
                fallback.append((pt[0], pt[1], txt,
                                 dict(fontsize=5.5, alpha=0.9, style='italic')))

    # Emphasise explicitly highlighted interactions (the paper-featured pairs,
    # e.g. KRAS/MRAS–LZTR1, SOS1–KRAS, CDC37–BRAF/CRAF, BRAF–ITCH, KSR1–DCAF1).
    # Each featured interaction gets ONE colour (from highlight_colors): all of
    # its structures — across predictors and binding modes — are ringed in that
    # colour, and the name label is drawn in the same colour, so correspondence
    # is read by colour. Points keep their predictor colour underneath the ring.
    if hl_mask.any() and 'interaction_name' in combined.columns:
        hl = combined[hl_mask]
        keys = ['volcano_label'] if by_label else ['scored_protein_name', 'interactor_name']
        # Labels already on the panel (corner counts), then each placed label,
        # are obstacles: a new label that overlaps one steps outward until clear.
        renderer = ax.figure.canvas.get_renderer()
        placed = [t.get_window_extent(renderer) for t in ax.texts]
        for key, grp in hl.groupby(keys, sort=False):
            # Ring/label an interaction if it is significant in AT LEAST ONE
            # assay: the supplied colour map is built from the full (both-assay)
            # table with the significance filter, so membership means
            # "significant somewhere" and the interaction is then labelled on
            # BOTH assay panels. Without a colour map, fall back to this panel's
            # own significance.
            grp_sig = sig_mask.loc[grp.index]
            if by_label:
                txt = key[0] if isinstance(key, tuple) else key
            else:
                sp, intr = key
                if hl_colors:
                    if (sp, intr) not in hl_colors:
                        continue
                elif not grp_sig.any():
                    continue
                txt = f'{_disp(sp)}–{_disp(intr)}'
            # Labels and leader lines in dark grey, no rings: each label sits at
            # a fixed offset from the interaction's most significant point (up
            # and away from d = 0), with a thin line to every one of its points,
            # so each text object and line starts at a known place for hand
            # placement.
            color = '#333333'
            gx = grp[x_col].astype(float)
            gy = grp['neg_log_pval'].astype(float)
            # Anchor at the most-significant structure in this panel if any,
            # else at this panel's top structure for the interaction.
            src = grp[grp_sig.to_numpy()] if grp_sig.any() else grp
            anchor = src.loc[src['neg_log_pval'].idxmax()]
            ax0, ay0 = float(anchor[x_col]), float(anchor['neg_log_pval'])
            dx = -12 if ax0 < 0 else 12
            # Try the default offset, then steps out along the label's side and
            # up/down, keeping the first position that overlaps nothing placed.
            steps = [(0, 0)] + [(sx * k, sy * k) for k in range(1, 9)
                                for sx, sy in ((0, 1), (0, -1), (1, 0), (1, 1), (1, -1))]
            for sx, sy in steps:
                label_tf = mtransforms.offset_copy(
                    ax.transData, fig=ax.figure, x=dx + np.sign(dx) * 14 * sx,
                    y=14 + 12 * sy, units='points')
                t = ax.text(ax0, ay0, txt, transform=label_tf, fontsize=hl_label_fs,
                            fontweight='bold', color=color, multialignment='center',
                            ha='right' if dx < 0 else 'left', va='bottom', zorder=7,
                            bbox=dict(boxstyle='round,pad=0.15', fc='white',
                                      ec='none', alpha=0.85))
                bb = t.get_window_extent(renderer).expanded(1.04, 1.1)
                if not any(bb.overlaps(o) for o in placed):
                    break
                t.remove()
            else:
                t = ax.text(ax0, ay0, txt, transform=label_tf, fontsize=hl_label_fs,
                            fontweight='bold', color=color, multialignment='center',
                            ha='right' if dx < 0 else 'left', va='bottom', zorder=7,
                            bbox=dict(boxstyle='round,pad=0.15', fc='white',
                                      ec='none', alpha=0.85))
            placed.append(t.get_window_extent(renderer))
            for px, py in zip(gx, gy):
                ax.annotate('', xy=(px, py), xytext=(ax0, ay0),
                            textcoords=label_tf, zorder=6,
                            arrowprops=dict(arrowstyle='-', color=color, lw=0.6,
                                            shrinkA=0, shrinkB=2))

    # Single de-collision pass over all labels (no arrows). Pass the panel's
    # data points as static obstacles and bump the static-repulsion / expand
    # margin modestly above adjustText's defaults so labels are pushed a bit
    # further off the circles, while a small pull keeps each label near its own
    # circle (correspondence is by colour regardless).
    #
    # With auto_place=False the de-collision pass is skipped entirely: every
    # label is left sitting exactly at its circle's centre. This is for hand-
    # editing the figure in Illustrator — each editable text object starts at a
    # known anchor (its circle) so the user can drag it out to taste.
    if not auto_place:
        pass
    elif HAS_ADJUSTTEXT and label_texts:
        adjust_text(
            label_texts, ax=ax,
            x=combined[x_col].to_numpy(dtype=float),
            y=combined['neg_log_pval'].to_numpy(dtype=float),
            force_text=(0.3, 0.3),
            force_static=(0.35, 0.35),
            force_pull=(0.03, 0.03),
            expand=(1.3, 1.4),
        )
    else:
        for x, y, txt, kw in fallback:
            ax.annotate(txt, (x, y), xytext=(8, 8),
                        textcoords='offset points', **kw)

    # Panel title: defaults to the single predictor's name (in its color)
    # for single-group panels, or `panel_title` for combined panels.
    if panel_title is None and len(non_empty_groups) == 1:
        panel_title = non_empty_groups[0][2].replace('_', ' ')
        if title_color is None:
            title_color = non_empty_groups[0][1]
    if title_color is None:
        title_color = 'black'
    if panel_title is not None:
        ax.set_title(panel_title, color=title_color, fontsize=title_fs,
                     fontweight='bold', pad=4, loc='center')

    # Inline color key for combined (multi-group) panels.  Tucked into the
    # upper-right corner *inside* the axes, just below the n/sig text that
    # sits above the spine.  Vertical stack (ncol=1) keeps the footprint
    # narrow.  The previous above-spine placement was tried but it
    # overlapped the centered panel title — the upper-right interior of a
    # volcano is the empty quadrant (high effect × low significance is rare)
    # so this placement won't fight data points.  Caller decides via
    # `show_color_key`; we don't auto-enable it for multi-group panels
    # because callers driving multi-row figures usually want it on row 0
    # only.
    if show_color_key and len(non_empty_groups) > 1:
        if color_key_order is not None:
            # Caller-specified legend order, decoupled from draw order.
            # Indices reference the original `groups` list; empty groups
            # (filtered out of `non_empty_groups`) are skipped silently.
            ordered = [groups[i] for i in color_key_order
                       if 0 <= i < len(groups) and len(groups[i][0]) > 0]
        else:
            ordered = non_empty_groups
        for _sub, color, name in ordered:
            ax.scatter([], [], c=color, s=18, label=name.replace('_', ' '))
        ax.legend(
            loc='upper right',
            # Sits below the up-count number in the top-right significance box.
            bbox_to_anchor=(1.0, 0.90),
            frameon=False, fontsize=legend_fs,
            handletextpad=0.3, borderaxespad=0, labelspacing=0.2,
        )

    # n / n_sig — placed *just above* the top spine, right-aligned, so it
    # sits on the same horizontal band as the centered panel title without
    # overlapping point labels for highly-significant interactions near the
    # top of the data range.  Uses Arial per the rcParams set by
    # apply_arial().
    #
    # n / n_sig — placed *just above* the top spine, right-aligned. When
    # `count_split` is supplied, the single "n | sig" line is replaced with one
    # line per (label, sub_df) so experimental vs predicted hit rates can be
    # reported separately. `show_counts=False` suppresses this text entirely.
    if show_counts and count_split:
        lines = []
        for label, sub_df in count_split:
            sub_n = len(sub_df)
            if sub_n == 0:
                lines.append(f'{label}: $n$ = 0, sig = 0')
                continue
            sub_sig = int(((sub_df['neg_log_pval'] >= fdr_y) &
                           (sub_df[x_col].abs() >= effect_threshold)).sum())
            lines.append(f'{label}: $n$ = {sub_n}, sig = {sub_sig}')
        ax.text(1.0, 1.005,
                '\n'.join(lines),
                transform=ax.transAxes, ha='right', va='bottom',
                fontsize=6, color='gray', family='Arial',
                linespacing=1.25)
    elif show_counts:
        ax.text(1.0, 1.005,
                f'$n$ = {n_total} | sig = {n_sig}',
                transform=ax.transAxes, ha='right', va='bottom',
                fontsize=6, color='gray', family='Arial')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if tick_fs is not None:
        ax.tick_params(axis='both', labelsize=tick_fs)

    # Unify Cohen's d tick-label style across panels (strip trailing zeros),
    # independent of each panel's auto-chosen tick locations / limits.
    ax.xaxis.set_major_formatter(FuncFormatter(_trim_zeros_formatter))

    return {
        'n': n_total,
        'n_sig': n_sig,
        'x_min': float(combined[x_col].min()),
        'x_max': float(combined[x_col].max()),
        'y_min': float(combined['neg_log_pval'].min()),
        'y_max': float(combined['neg_log_pval'].max()),
    }



def plot_volcano_overlay_three_sources(
    df: pd.DataFrame,
    *,
    assays: Optional[List[str]] = None,
    x_col: str = 'cohens_d',
    effect_threshold: float = 0.2,
    fdr_alpha: float = 0.05,
    size_col: str = 'n_interface_positions',
    size_range: Tuple[float, float] = (10, 180),
    annotate_significant: bool = True,
    max_labels_per_panel: int = 10,
    figsize_per_panel: Tuple[float, float] = (4.5, 4.5),
    show_legend: bool = True,
    highlight_pairs: Optional[List[Tuple[str, str]]] = None,
    highlight_colors: Optional[Dict[Tuple[str, str], str]] = None,
    auto_place: bool = True,
    corner_counts: Optional[Dict[str, Dict[str, Tuple[int, int]]]] = None,
    output_path: Optional[Union[str, Path]] = None,
    dpi: int = 300,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Single-panel volcano figure overlaying all three structure sources for
    one (or more) assays.

    Layout
    ------
    For each assay in `assays`, a single axes is drawn with three series
    overlaid:
      - RCSB PDB (green)        — experimental ground truth
      - Schmid et al. / AF      — Predictomes (purple)
      - Zhang et al. / RF       — Zhang RosettaFold (orange)

    The inline color key (upper-right of the panel) names all three sources
    using the same italic-`et al.` mathtext style as
    `plot_volcano_pdb_vs_predicted`. The color key can be suppressed per
    panel via `show_legend=False` — useful when the figure is part of a
    multi-panel layout and one shared legend on a sibling panel is enough.

    Counts above the panel are split via `_draw_volcano_panel`'s
    `count_split` into:

        PDB:       n = …, sig = …
        Predicted: n = …, sig = …

    where "Predicted" is Predictomes + Zhang concatenated. The Bonferroni
    horizontal cutoff is computed across the union (single multiple-testing
    family per panel) so all three series share one significance line. The
    split counts are shown on every panel even when the color key is
    hidden — they are the per-source totals the reader needs in order to
    interpret each panel on its own.

    This view sits between the per-predictor 1×3 facet
    (`plot_volcano_publication_faceted`) and the experimental-vs-predicted
    2-column split (`plot_volcano_pdb_vs_predicted`): a single overlaid
    panel that lets the reader compare experimental and predicted points
    side-by-side without column-jumping, while still seeing the totals for
    each source separately.

    Parameters
    ----------
    df : pd.DataFrame
        Aggregated per-(scored, interactor, assay, predictor) data, as
        produced by `aggregate_volcano_data`. Must contain `predictor`,
        `assay`, x_col, `neg_log_pval` (or `pval_fdr`), `interaction_name`.
    assays : list of str, optional
        Assays to include as rows. Default uses every assay in `df`.
    x_col, effect_threshold, fdr_alpha, size_col, size_range,
    annotate_significant, max_labels_per_panel, figsize_per_panel,
    output_path, dpi
        Same semantics as `plot_volcano_pdb_vs_predicted`.
    show_legend : bool, default True
        Whether to draw the inline three-entry color key in the panel.
        Pass False when the assay is rendered alongside another assay that
        already carries the legend (saves visual clutter and label–legend
        collisions).

    Returns
    -------
    tuple
        (fig, axes_2d) where axes_2d has shape (n_assays, 1).
    """
    setup_publication_style()

    dfp = df.copy()

    if 'neg_log_pval' not in dfp.columns:
        if 'pval_fdr' in dfp.columns:
            dfp['neg_log_pval'] = -np.log10(
                pd.to_numeric(dfp['pval_fdr'], errors='coerce').clip(lower=1e-300)
            )
        elif 'pval_mannwhitney' in dfp.columns:
            dfp['neg_log_pval'] = -np.log10(
                pd.to_numeric(dfp['pval_mannwhitney'], errors='coerce').clip(lower=1e-300)
            )
        else:
            raise ValueError(
                "df must contain 'neg_log_pval', 'pval_fdr', or 'pval_mannwhitney'"
            )

    for c in [x_col, 'neg_log_pval', size_col]:
        if c in dfp.columns:
            dfp[c] = pd.to_numeric(dfp[c], errors='coerce')
    dfp = dfp.dropna(subset=[x_col, 'neg_log_pval'])

    # One stable colour per featured interaction, shared across every panel.
    # Compute from the full (multi-panel) frame so a given interaction is the
    # same colour in every panel; callers may pass a precomputed map to keep
    # colours consistent across separate figures too.
    hcolors = (highlight_colors if highlight_colors is not None
               else build_highlight_color_map(dfp, highlight_pairs))

    if assays is None:
        assays = sorted(dfp['assay'].dropna().unique().tolist())

    n_rows = len(assays)
    fig, axes = plt.subplots(
        n_rows, 1,
        figsize=(figsize_per_panel[0], figsize_per_panel[1] * n_rows),
        squeeze=False,
        sharex='all',
    )

    pdb_color = PREDICTOR_PALETTE['RCSB_PDB']
    pred_color = PREDICTOR_PALETTE['Predictomes']
    zhang_color = PREDICTOR_PALETTE['Zhang_et_al']

    for i, assay in enumerate(assays):
        pdb_sub = dfp[(dfp['assay'] == assay) &
                      (dfp['predictor'] == 'RCSB_PDB')].copy()
        pred_sub = dfp[(dfp['assay'] == assay) &
                       (dfp['predictor'] == 'Predictomes')].copy()
        zhang_sub = dfp[(dfp['assay'] == assay) &
                        (dfp['predictor'] == 'Zhang_et_al')].copy()

        # "Predicted" subset for the count split — Predictomes + Zhang
        # concatenated into a single per-source bucket. The reader's
        # mental category is "experimental vs predicted", not
        # "experimental vs AF vs RF", so the count line collapses the two
        # predicted sources together while the dot colors keep them
        # distinguishable in the plot itself.
        predicted_sub = pd.concat([pred_sub, zhang_sub], ignore_index=True)

        _draw_volcano_panel(
            axes[i, 0],
            groups=[
                # Order here controls draw order (later = on top). Putting
                # PDB last makes the green experimental dots most visible
                # since they're the smallest set. Legend order is decoupled
                # below via color_key_order so PDB still reads first.
                # Labels match the source Venn diagram (PDB / AF-M / RF2-PPI).
                (pred_sub, pred_color, 'AF-M'),
                (zhang_sub, zhang_color, 'RF2-PPI'),
                (pdb_sub, pdb_color, 'PDB'),
            ],
            x_col=x_col,
            effect_threshold=effect_threshold,
            fdr_alpha=fdr_alpha,
            size_col=size_col,
            size_range=size_range,
            annotate=annotate_significant,
            max_labels=max_labels_per_panel,
            panel_title=assay.capitalize(),
            title_color='black',
            show_color_key=show_legend,
            color_key_order=[2, 0, 1],   # legend reads PDB → Schmid → Zhang
            count_split=[
                ('PDB', pdb_sub),
                ('Predicted', predicted_sub),
            ],
            highlight_pairs=highlight_pairs,
            highlight_colors=hcolors,
            auto_place=auto_place,
            # Larger type for this manuscript overlay panel; drop the top-right
            # PDB/Predicted count text.
            title_fs=14,
            tick_fs=10,
            hl_label_fs=8.5,
            legend_fs=11,
            show_counts=False,
            corner_counts=(corner_counts or {}).get(assay),
        )

    # Axis labels — y-axis on every row, x-axis on bottom only.
    x_label = ("Cohen's $d$ (interface vs control)"
               if x_col == 'cohens_d'
               else f"{x_col} (interface vs control)")
    y_label = '$-\\log_{10}$(FDR $q$)'

    for i in range(n_rows):
        axes[i, 0].set_ylabel(y_label, fontsize=12)
        if i == n_rows - 1:
            axes[i, 0].set_xlabel(x_label, fontsize=12)
        else:
            axes[i, 0].set_xlabel('')

    fig.tight_layout(rect=[0, 0, 1, 0.98])

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
        if str(output_path).endswith('.pdf'):
            fig.savefig(output_path.with_suffix('.png'),
                        dpi=dpi, bbox_inches='tight')

    return fig, axes



setup_publication_style()



# ===========================================================================
# Structures and the interaction table
# ===========================================================================

_parser = PDBParser(QUIET=True)



def load_interactions() -> pd.DataFrame:
    """Load the combined, clash-filtered interaction table from all three sources.

    Delegates to ``load_combined_results`` so the patch analysis sees exactly
    the same interaction universe as Fig 5a/b/c and the network: skipped rows
    dropped, and the 317 inter-chain C-beta clash structures excluded
    (``interchain_cbeta_clashes.csv``). Reading the raw all_interactions.csv
    directly — as this did previously — silently re-included those non-viable
    poses, inflating per-patch interactor counts relative to the rest of the
    figure.
    """
    return load_combined_results(
        predictomes_dir='output/interactions/predictomes_output_dom_r25_gt10',
        zhangetal_dir='output/interactions/Zhangetal_output_dom_r25_gt10',
        pdb_dir='output/interactions/pdb_output_ctrl_domains_rsa25_gt10A',
        include_skipped=False,
    )



def compute_significant_pairs(df: pd.DataFrame) -> set:
    """Return the set of significant (scored_protein_name, interactor) pairs.

    Significance is the canonical rule (``flag_significant`` on the aggregated
    interaction table: BH-FDR < 0.05 and |Cohen's d| >= 0.2), with a pair
    counted significant if it passes in EITHER functional assay — **activity or
    (basal) abundance**. The HSP90-inhibitor abundance readout
    (``abundance_HSP90i``) is excluded so this matches the volcanoes / main
    figure. Keyed on (scored_protein_name, interactor).
    """
    analyzed = df.dropna(subset=['log_fold_change', 'pval_fdr', 'cohens_d'])
    if analyzed.empty:
        return set()
    agg = aggregate_volcano_data(analyzed)
    agg['is_sig'] = flag_significant(agg)
    sig = agg[agg['is_sig'] & agg['assay'].isin(['activity', 'abundance'])]
    return set(zip(sig['scored_protein_name'], sig['interactor']))



def parse_positions(pos_str: str) -> set:
    """Parse comma-separated position string into a set of ints."""
    if pd.isna(pos_str) or str(pos_str).strip() == '':
        return set()
    return {int(x) for x in str(pos_str).split(',')}



def resolve_pdb_path(structure_name: str) -> Path:
    """
    Find the PDB file for a structure name across the three source directories.

    Returns
    -------
    Path or None
    """
    for base in [Path('data/interactions/predictomes_MAPK'), Path('data/interactions/pdb_controls')]:
        for suffix in ['', '.pdb']:
            p = base / (structure_name + suffix)
            if p.exists():
                return p
    # Zhang et al: nested pair-dir layout
    parts = structure_name.replace('.pdb', '').split('__')
    if len(parts) >= 2:
        u1 = parts[0].split('_')[0]
        u2 = parts[1].split('_')[0]
        for suffix in ['', '.pdb']:
            p = Path('data/interactions/Zhangetal_structures') / f'{u1}_{u2}' / (structure_name + suffix)
            if p.exists():
                return p
    return None



def get_cbeta_coords(structure, chain_id: str) -> dict:
    """
    Get C-beta coordinates per residue (C-alpha for glycine).

    Returns
    -------
    dict
        resnum (int) -> numpy array of shape (3,)
    """
    coords = {}
    for residue in structure[0][chain_id]:
        if residue.id[0] != ' ':
            continue
        resnum = residue.id[1]
        if 'CB' in residue:
            coords[resnum] = residue['CB'].get_vector().get_array()
        elif 'CA' in residue:
            coords[resnum] = residue['CA'].get_vector().get_array()
    return coords



def select_representative_structures(df: pd.DataFrame) -> dict:
    """
    Select one Predictomes AF2 structure per protein for patch definition.

    Prefers Predictomes (full-length AF2) for consistency across all proteins.
    Falls back to RCSB_PDB or Zhang_et_al if no Predictomes structure exists.
    Within each predictor, selects the structure with the most interface
    positions (best coverage of the protein surface).

    Parameters
    ----------
    df : pd.DataFrame
        Valid (non-skipped) interactions with interface_positions

    Returns
    -------
    dict
        protein_name -> {'structure': str, 'chain': str, 'pdb_path': Path,
                         'predictor': str}
    """
    representatives = {}
    for protein in sorted(df['scored_protein_name'].unique()):
        sub = df[df['scored_protein_name'] == protein]
        for pred in ['Predictomes', 'RCSB_PDB', 'Zhang_et_al']:
            pred_sub = sub[sub['predictor'] == pred]
            if len(pred_sub) > 0:
                best = pred_sub.loc[pred_sub['n_interface_positions'].idxmax()]
                pdb_path = resolve_pdb_path(str(best['structure']))
                if pdb_path is not None:
                    representatives[protein] = {
                        'structure': str(best['structure']),
                        'chain': best['scored_chain'],
                        'pdb_path': pdb_path,
                        'predictor': pred,
                    }
                    break
    return representatives



warnings.filterwarnings('ignore')



# ===========================================================================
# Interface footprints and clusters
# ===========================================================================

DIST_TOL = 6.0       # A — two residues "co-located" if C-beta within this
MIN_FOOTPRINT = 5    # min scored interface residues to assess an interaction
SITE_CUTOFF = 0.5    # IoU to link two footprints into the same site. 0.5 chosen



def build_footprints(valid) -> dict:
    """{scored_protein_name: {interactor: set(int positions)}} — union over structures."""
    fp: dict = defaultdict(lambda: defaultdict(set))
    for _, row in valid.iterrows():
        positions = parse_positions(row['interface_positions'])
        fp[row['scored_protein_name']][row['interactor']] |= positions
    return {p: dict(d) for p, d in fp.items()}



def _n_near(src: set, dst: set, coords: dict, dist_tol: float) -> int:
    """Number of residues in `src` that are co-located with `dst`.

    A residue counts if it is identical to a `dst` residue, or (with coords)
    within `dist_tol` A of one. Residues missing from `coords` only match on
    exact identity. dist_tol == 0 reduces to set intersection.
    """
    dst_xyz = [coords[r] for r in dst if r in coords] if dist_tol > 0 else []
    n = 0
    for r in src:
        if r in dst:                           # identical residue
            n += 1
            continue
        if dist_tol > 0 and r in coords and dst_xyz:
            if min(np.linalg.norm(coords[r] - x) for x in dst_xyz) <= dist_tol:
                n += 1
    return n



def jaccard_overlap(a: set, b: set, coords: dict, dist_tol: float = 0.0) -> float:
    """Jaccard / intersection-over-union of two interface footprints.

    With dist_tol == 0 this is the exact residue-set IoU |A∩B| / |A∪B| — the
    same measure used as `interface_jaccard` elsewhere in the codebase. With
    dist_tol > 0 the intersection is softened to count spatially adjacent
    residues (so a site bound one residue off-register still scores), via the
    symmetric overlap (|A near B| + |B near A|) / 2; it reduces to exact IoU
    when dist_tol == 0.
    """
    if not a or not b:
        return 0.0
    inter = (_n_near(a, b, coords, dist_tol) + _n_near(b, a, coords, dist_tol)) / 2.0
    union = len(a) + len(b) - inter
    return inter / union if union > 0 else 0.0



def _load_inputs():
    """Load footprints, significant pairs, per-protein C-beta coords, and an
    interactor UniProt -> gene-name map, once."""
    df = load_interactions()  # clash-filtered, skipped dropped
    valid = df[(df['skipped'] != True) & df['interface_positions'].notna()].copy()
    sig_pairs = compute_significant_pairs(df)        # (scored_name, interactor)
    footprints = build_footprints(valid)
    iname = dict(zip(valid['interactor'], valid['interactor_name']))
    reps = select_representative_structures(valid)   # {protein: {pdb_path, chain}}
    coords_by_protein: dict = {}
    for protein, info in reps.items():
        try:
            structure = _parser.get_structure('s', str(info['pdb_path']))
            coords_by_protein[protein] = get_cbeta_coords(structure, info['chain'])
        except Exception:
            coords_by_protein[protein] = {}
    return footprints, sig_pairs, coords_by_protein, iname



def representative_cbeta_coords(valid) -> dict:
    """{protein: {residue: C-beta coordinates}} from one representative structure
    per scored protein. jaccard_overlap uses them for its 6 A tolerance, so a site
    engaged one residue off-register still counts as shared."""
    reps = select_representative_structures(valid)   # {protein: {pdb_path, chain}}
    coords_by_protein: dict = {}
    for protein, info in reps.items():
        try:
            structure = _parser.get_structure('s', str(info['pdb_path']))
            coords_by_protein[protein] = get_cbeta_coords(structure, info['chain'])
        except Exception:
            coords_by_protein[protein] = {}
    return coords_by_protein


def analyse(footprints, sig_pairs, coords_by_protein,
            jaccard_cutoff: float, dist_tol: float, floor: int) -> list:
    """Classify each (protein, partner) interaction by IoU site-sharing.

    Two partners share a site iff jaccard_overlap(.) >= jaccard_cutoff.
    Returns one record per interaction.
    """
    results = []
    for protein, partners in footprints.items():
        coords = coords_by_protein.get(protein, {})
        assessable = {x: fp for x, fp in partners.items() if len(fp) >= floor}
        for x, fp_x in partners.items():
            assessed = len(fp_x) >= floor
            n_any = n_sup = 0
            if assessed:
                for y, fp_y in assessable.items():
                    if y == x:
                        continue
                    if jaccard_overlap(fp_x, fp_y, coords, dist_tol) >= jaccard_cutoff:
                        n_any += 1
                        if (protein, y) in sig_pairs:
                            n_sup += 1
            results.append({
                'protein': protein, 'partner': x,
                'supported': (protein, x) in sig_pairs,
                'assessed': assessed,
                'footprint': len(fp_x),
                'n_shared_any': n_any, 'n_shared_sup': n_sup,
            })
    return results



def cluster_sites(footprints, sig_pairs, coords_by_protein,
                  jaccard_cutoff: float, dist_tol: float, floor: int) -> list:
    """Cluster each scored protein's partner footprints into distinct 'sites'.

    Uses **complete-linkage** hierarchical clustering on the pairwise IoU
    distance (1 - Jaccard) of partner footprints, cut at distance
    1 - `jaccard_cutoff`. Complete linkage means two partners share a site only
    if EVERY pair in that site overlaps at IoU >= cutoff — so it does NOT chain
    (the failure mode of connected components, where a hub like KRAS collapsed
    to a single site because every partner overlapped *some* other partner).
    Distinct reused surfaces (e.g. KRAS Switch I vs Switch II) therefore split
    into separate sites. This matches the binding-mode clustering convention in
    _assign_binding_mode_clusters (ppi_utils).

    Only partners with footprint >= `floor` residues participate. Returns one
    dict per site: {protein, partners: set, n_partners, n_supported}.
    """
    sites = []
    dist_threshold = 1.0 - jaccard_cutoff
    for protein, partners in footprints.items():
        coords = coords_by_protein.get(protein, {})
        items = [(x, fp) for x, fp in partners.items() if len(fp) >= floor]
        n = len(items)
        if n == 0:
            continue
        if n == 1:
            labels = [1]
        else:
            dist = np.zeros((n, n))
            for i in range(n):
                for j in range(i + 1, n):
                    iou = jaccard_overlap(items[i][1], items[j][1], coords, dist_tol)
                    dist[i, j] = dist[j, i] = 1.0 - iou
            if dist.max() == 0:                     # all footprints identical
                labels = [1] * n
            else:
                Z = linkage(squareform(dist, checks=False), method='complete')
                labels = fcluster(Z, t=dist_threshold, criterion='distance')

        comps = defaultdict(list)
        for (x, _fp), lab in zip(items, labels):
            comps[int(lab)].append(x)
        for members in comps.values():
            n_sup = sum(1 for x in members if (protein, x) in sig_pairs)
            # Characterize the site by its footprint, not just min-max extent:
            # core = residues shared by >= half the partners (the consensus
            # contact surface). cov = per-residue partner count.
            cov = Counter()
            allpos = set()
            for x in members:
                f = footprints[protein][x]
                allpos |= f
                for p in f:
                    cov[p] += 1
            thresh = max(2, (len(members) + 1) // 2)
            core = sorted(p for p, c in cov.items() if c >= thresh)
            rng = f'{min(allpos)}-{max(allpos)}' if allpos else ''
            sites.append({'protein': protein, 'partners': set(members),
                          'n_partners': len(members), 'n_supported': n_sup,
                          'range': rng, 'core_residues': core})
    return sites



def disp(name: str) -> str:
    return DISPLAY_NAME.get(name, name)



# Partner-count categories for a site (number of partners sharing the site).
SITE_CATS = ['1', '2-4', '5+']
SITE_CAT_COLOR = {'1': '#74c476', '2-4': '#fd8d3c', '5+': '#6a51a3'}
SITE_CAT_DESC = {'1': '1 partner (specific)', '2-4': '2–4 partners (promiscuous)',
                 '5+': '5+ partners (hub)'}



# The ED9i cluster bars bin sites two ways: one partner, or shared by two or more.
SITE_CATS_2 = ['1', '2+']
SITE_CAT_COLOR_2 = {'1': '#74c476', '2+': '#fd8d3c'}
SITE_CAT_DESC_2 = {'1': '1 partner (specific)', '2+': '2+ partners (promiscuous)'}


def site_category_2(n: int) -> str:
    """Bin a site's partner count into 1 / 2+ (promiscuous)."""
    return '1' if n == 1 else '2+'


def site_category(n: int) -> str:
    """Bin a site's partner count into 1 / 2-4 / 5+ (hub)."""
    if n == 1:
        return '1'
    if n <= 4:
        return '2-4'
    return '5+'



def plot_sites_bar(sites: list, output_path: Path, cutoff: float, floor: int,
                   examples=None, *, categories=None):
    """Per-protein stacked bar of interacting sites binned by partner count.
    Bar height = total sites on that protein; global per-category totals are in
    the legend/title. `categories` is (bin function, category order, colours,
    legend labels); the default is 1 / 2-4 / 5+. `examples` (list of (protein,
    text)) draws bent-arrow callouts from a protein's hub (5+) segment to a box
    naming example partners at its largest hub site (three-way binning only).
    """
    site_category, SITE_CATS, SITE_CAT_COLOR, SITE_CAT_DESC = (
        categories or (globals()['site_category'], globals()['SITE_CATS'],
                       globals()['SITE_CAT_COLOR'], globals()['SITE_CAT_DESC']))
    reset_style(publication=True)
    apply_arial()
    by_protein = defaultdict(lambda: {c: 0 for c in SITE_CATS})
    for s in sites:
        by_protein[s['protein']][site_category(s['n_partners'])] += 1

    proteins = sorted(by_protein, key=lambda p: -sum(by_protein[p].values()))
    counts = {c: [by_protein[p][c] for p in proteins] for c in SITE_CATS}
    g = {c: sum(counts[c]) for c in SITE_CATS}
    total = sum(g.values())
    promisc = total - g['1']

    # Fonts/sizes are deliberately large: intended for the supplement at ~50%.
    x = np.arange(len(proteins))
    fig, ax = plt.subplots(figsize=(max(11, 0.7 * len(proteins) + 4), 7.0))
    bottom = np.zeros(len(proteins))
    for c in SITE_CATS:
        ax.bar(x, counts[c], 0.72, bottom=bottom,
               label=f'{SITE_CAT_DESC[c]} — {g[c]}',
               color=SITE_CAT_COLOR[c], edgecolor='white', linewidth=0.8)
        bottom += np.array(counts[c])
    ymax = bottom.max()

    # Hub-site example callouts. Each box sits just above-right of its bar and
    # connects with an elbow leader: horizontal out of the box, one kink, then a
    # short diagonal into the hub (5+) segment.
    if examples:
        ax.set_ylim(0, ymax * 1.5)
        prot_idx = {p: i for i, p in enumerate(proteins)}
        ex = [(p, t) for p, t in examples if p in prot_idx]
        seen = defaultdict(int)
        for prot, text in ex:
            i = prot_idx[prot]
            seg_lo = counts['1'][i] + counts['2-4'][i]
            seg_hi = seg_lo + counts['5+'][i]
            row = seen[prot]; seen[prot] += 1          # stack multiple per protein
            # target a different height within the hub band for stacked callouts
            ty = seg_lo + counts['5+'][i] * (0.7 - 0.4 * row)
            lx = i + 2.2                               # box just to the right
            ly = seg_hi + 3.0 + row * 5.5              # just above the bar, stacked
            ax.annotate(
                text, xy=(i + 0.25, ty), xytext=(lx, ly),
                fontsize=11, ha='left', va='center', zorder=6,
                bbox=dict(boxstyle='round,pad=0.3', fc='#efeaf6',
                          ec=SITE_CAT_COLOR['5+'], lw=1.2),
                arrowprops=dict(arrowstyle='-|>', color=SITE_CAT_COLOR['5+'],
                                lw=1.8, shrinkA=3, shrinkB=4,
                                connectionstyle='angle,angleA=0,angleB=60'))

    ax.set_xticks(x)
    ax.set_xticklabels([disp(p) for p in proteins], rotation=45, ha='right', fontsize=18)
    ax.tick_params(axis='y', labelsize=18)
    ax.set_ylabel('Number of interface clusters', fontsize=22)
    ax.set_title(
        f'Interface clusters by number of partners\n'
        f'(IoU ≥ {cutoff:g} cluster definition, footprint ≥ {floor} positions; '
        f'{total} clusters, {g["1"]} single, {promisc} promiscuous '
        f'[{100*promisc/total:.0f}%])',
        fontsize=20, fontweight='bold')
    # Legend just outside the right edge so it never collides with the callouts.
    ax.legend(frameon=False, fontsize=14, loc='upper left',
              bbox_to_anchor=(1.005, 1.0), title='partners / cluster', title_fontsize=14)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches='tight')
    fig.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}  ({total} clusters: " +
          ", ".join(f'{c}={g[c]}' for c in SITE_CATS) + ")")
    return g



def plot_sites_bar_compact(sites: list, output_path: Path, proteins=('KRAS',), *,
                           categories=None, figsize=(1.7, 2.3)):
    """Small companion to plot_sites_bar: one bar for all proteins and one per
    protein in `proteins`, each stacked by partner-count category as a
    percentage of that bar's clusters, with the counts on the segments and n
    above. `categories` is as in plot_sites_bar."""
    site_category, cats, cat_color, cat_desc = (
        categories or (globals()['site_category'], globals()['SITE_CATS'],
                       globals()['SITE_CAT_COLOR'], globals()['SITE_CAT_DESC']))
    reset_style(publication=True)
    apply_arial()
    groups = [('All', sites)] + [(disp(p), [s for s in sites if s['protein'] == p])
                                 for p in proteins]
    fig, ax = plt.subplots(figsize=figsize)
    for i, (name, ss) in enumerate(groups):
        n = len(ss)
        bottom = 0.0
        for c in cats:
            k = sum(site_category(s['n_partners']) == c for s in ss)
            frac = 100 * k / n
            ax.bar(i, frac, 0.7, bottom=bottom, color=cat_color[c],
                   edgecolor='white', linewidth=0.6)
            if k:
                ax.text(i, bottom + frac / 2, str(k), ha='center', va='center',
                        fontsize=8, color='#1f2933')
            bottom += frac
        ax.text(i, 101.5, f'n = {n}', ha='center', va='bottom', fontsize=7.5,
                color='#5b6770')
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g for g, _ in groups], fontsize=9)
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.set_ylim(0, 108)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.tick_params(axis='y', labelsize=8)
    ax.set_ylabel('Interface clusters (%)', fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(handles=[Patch(facecolor=cat_color[c], label=cat_desc[c]) for c in cats],
              frameon=False, fontsize=7.5, loc='upper center',
              bbox_to_anchor=(0.5, -0.14), ncol=1, handlelength=1.0)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    return output_path



def report(res: list, label: str):
    assessed = [r for r in res if r['assessed']]
    sup = [r for r in assessed if r['supported']]
    alli = assessed
    def frac(rows, key):
        n = sum(1 for r in rows if r[key] >= 1)
        return n, len(rows), (100 * n / len(rows) if rows else float('nan'))
    print(f"\n=== {label} ===")
    for name, rows in [('SUPPORTED', sup), ('ALL assessed (baseline)', alli)]:
        na, da, pa = frac(rows, 'n_shared_any')
        ns, ds, ps = frac(rows, 'n_shared_sup')
        print(f"  {name}: n={len(rows)}")
        print(f"    share a site with >=1 ANY partner:       {na}/{da} ({pa:.1f}%)")
        print(f"    share a site with >=1 SUPPORTED partner: {ns}/{ds} ({ps:.1f}%)")
    # Distribution of #partners shared with, among supported.
    shared = [r['n_shared_any'] for r in sup]
    if shared:
        arr = np.array(shared)
        print(f"  supported partners-shared-with: median={np.median(arr):.0f} "
              f"mean={arr.mean():.1f} max={arr.max()} "
              f"(0:{int((arr==0).sum())}, 1:{int((arr==1).sum())}, "
              f">=2:{int((arr>=2).sum())})")



# ===========================================================================
# Interface-overlap heatmaps
# ===========================================================================

def iou_matrix(items, coords):
    """Pairwise IoU matrix for a list of (name, footprint)."""
    n = len(items)
    M = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            M[i, j] = M[j, i] = jaccard_overlap(items[i][1], items[j][1],
                                                  coords, DIST_TOL)
    return M



def contiguous_blocks(labels_in_order):
    """Yield (start, end) inclusive index ranges of equal consecutive labels."""
    start = 0
    for i in range(1, len(labels_in_order) + 1):
        if i == len(labels_in_order) or labels_in_order[i] != labels_in_order[start]:
            yield start, i - 1
            start = i



# Curated functional labels for the paper-figure panels. Each entry maps a
# *signature* partner (a gene that uniquely identifies one site at IoU 0.5) to a
# short family label. At IoU 0.5 the clustering partitions partners, so a
# signature gene picks out exactly one site. LZTR1 is labelled by surface
# (Switch II) rather than a family because its co-clustered partners are not a
# clean functional group — see footprint analysis: LZTR1 is most similar to that
# Switch-II cluster (IoU 0.68-0.85) and only partially overlaps the GAP site.
HUB_OVERRIDES = {
    'KRAS': [
        ('SOS1', 'Ras / Ral GEFs'),
        ('PLXNA1', 'GAPs & Plexins'),
        ('NF1', 'Ras / Rho GAPs'),
        ('HRAS', 'Ras family & PI3K'),
        ('BRAF', 'RAF kinases / effectors'),
        ('TIAM1', 'Rho GEFs'),
        ('LZTR1', 'LZTR1 (Switch II)'),
    ],
    'BRAF': [
        ('MEK1', 'MEK / MAP2K'),
        ('KRAS', 'Ras family (RBD)'),
        ('ARAF', 'RAF dimer'),            # ARAF/CRAF — separate from KSR at 0.5
        ('KSR1', 'KSR pseudokinases'),    # KSR1/KSR2 — own site at 0.5
        ('CDC37', 'CDC37 (cochaperone)'),  # singleton — featured in Supp Fig 5
        ('ITCH', 'ITCH (E3 ligase)'),      # singleton — featured in Supp Fig 5
    ],
    'GRB2': [
        ('GAB2', 'GAB2 / PAK'),           # GAB2, PAK1 — GAB2 scaffold site
        ('VAV3', 'Vav GEFs'),             # Vav1, VAV3
        ('LAT', 'LAT adaptors'),          # LAT, LAT2
        ('PTPN6', 'SHP phosphatases'),    # PTPN6 (SHP1), SHP2
        ('ARHGAP9', 'Rho GAPs'),          # ARHGAP9, ARHGAP12
    ],
}



def load_effect_directions():
    """Per-(scored protein, partner, assay) functional effect: (median Cohen's d,
    q-value). The q-value is the geometric mean of the member's aggregated BH-FDR
    across its predictor rows; the Cohen's d is the median across those rows.
    Uses the canonical aggregate_volcano_data (BH-FDR recomputed per assay), so
    it matches the volcano definitions.
    """
    df = load_interactions()
    analyzed = df.dropna(subset=['log_fold_change', 'pval_fdr', 'cohens_d'])
    agg = aggregate_volcano_data(analyzed)
    eff = {}
    for (sp, intr, assay), g in agg.groupby(
            ['scored_protein_name', 'interactor', 'assay']):
        d = float(g['cohens_d'].median())
        q = float(np.exp(np.log(g['pval_fdr'].clip(lower=1e-300)).mean()))
        eff[(sp, intr, assay)] = (d, q)
    return eff



# A site is called significant in an assay when the geometric-mean q-value
# across its members < SITE_Q_CUT AND the |median Cohen's d| >= SITE_D_CUT.
SITE_Q_CUT = 0.01
SITE_D_CUT = 0.2



def site_effect_pair(protein, member_unis, eff,
                     q_cut=SITE_Q_CUT, d_cut=SITE_D_CUT):
    """(activity_sym, abundance_sym) for a site. Per assay, the site is called
    significant when the geometric-mean q-value across its members < `q_cut` AND
    |median Cohen's d| >= `d_cut`; the symbol is then the sign of the median
    Cohen's d, else 'ns'. Symbols: '+', '-', 'ns'.
    """
    syms = []
    for assay in ('activity', 'abundance'):
        vals = [eff[(protein, u, assay)] for u in member_unis
                if (protein, u, assay) in eff]
        if not vals:
            syms.append('ns')
            continue
        med = float(np.median([d for d, _ in vals]))
        gq = float(np.exp(np.mean(np.log(
            np.clip([q for _, q in vals], 1e-300, None)))))
        if gq < q_cut and abs(med) >= d_cut:
            syms.append('+' if med >= 0 else '-')
        else:
            syms.append('ns')
    return syms[0], syms[1]



# Diverging colour scale for the per-interactor Cohen's d effect strip
# (blue = negative/loss, red = positive/gain, centred on 0); non-significant
# cells (FDR q >= 0.05) are greyed. Shared by the full and paper heatmaps and
# their 'Cohen's d' colourbars.
EFFECT_D_VLIM = 1.5
EFFECT_D_CMAP = plt.cm.bwr
EFFECT_NS_GREY = '#b4b8bc'



def _effect_norm():
    return mcolors.TwoSlopeNorm(vmin=-EFFECT_D_VLIM, vcenter=0.0, vmax=EFFECT_D_VLIM)



def _effect_facecolor(protein, u, assay, eff):
    """Cell facecolour for interactor `u` in `assay`: bwr(d) when significant
    (FDR q < 0.05), grey when not (q >= 0.05), white when not measured. The
    effect-size floor is intentionally not applied here — every significant
    interactor is shown coloured by its Cohen's d, however small."""
    v = eff.get((protein, u, assay)) if eff else None
    if v is None:
        return 'white'
    d, q = v
    return EFFECT_D_CMAP(_effect_norm()(d)) if q < 0.05 else EFFECT_NS_GREY



def _draw_effect_strip(fig, protein, members_uni_ord, n, eff, *, x0, y0, w, h,
                       header_fs=8):
    """Two-column per-interactor Cohen's d heatmap (activity | abundance) drawn
    at figure rectangle [x0, y0, w, h], aligned to `n` heatmap rows (row 0 at
    top). No numeric text — the colour is the value. Returns
    (right_edge_x, row_to_figy, eax)."""
    eax = fig.add_axes([x0, y0, w, h])
    eax.set_xlim(0, 2); eax.set_ylim(n - 0.5, -0.5); eax.axis('off')
    for i, u in enumerate(members_uni_ord):
        for assay, xc in (('activity', 0.5), ('abundance', 1.5)):
            eax.add_patch(Rectangle(
                (xc - 0.5, i - 0.5), 1, 1,
                facecolor=_effect_facecolor(protein, u, assay, eff),
                edgecolor='white', lw=0.3))
    for xc, hdr in ((0.5, 'Activity'), (1.5, 'Abundance')):
        eax.text(xc, -0.7, hdr, ha='center', va='bottom', rotation=90,
                 fontsize=header_fs, fontweight='bold', color='#222222')

    def row_to_figy(rc):
        return y0 + h * (1 - (rc + 0.5) / n)
    return x0 + w, row_to_figy, eax



def _paper_panel_data(protein, items, coords, iname, cutoff, eff=None):
    """Cluster one protein's partners and assign curated family labels to hubs.

    Returns (Mr, n, multi_sites, labeled, members, Z): the IoU matrix in cluster
    order, the partner count, all multi-partner sites, the labelled (curated)
    subset numbered in diagonal order, the partners in row order, and the
    complete-linkage tree on 1 - IoU (None when no two partners overlap).
    """
    pdisp = disp(protein)
    pcoords = coords.get(protein, {})
    M = iou_matrix(items, pcoords)
    D = 1.0 - M
    np.fill_diagonal(D, 0.0)
    Z = None
    if D.max() == 0:
        order = np.arange(len(items))
        labels = np.ones(len(items), dtype=int)
    else:
        Z = linkage(squareform(D, checks=False), method='complete')
        order = leaves_list(Z)
        labels = fcluster(Z, t=1.0 - cutoff, criterion='distance')

    Mr = M[np.ix_(order, order)]
    lab_ord = labels[order]
    members_uni_ord = [items[o][0] for o in order]
    overrides = HUB_OVERRIDES.get(pdisp, [])

    multi_sites = []
    for s, e in contiguous_blocks(lab_ord):
        members = members_uni_ord[s:e + 1]
        gene_set = {disp(iname.get(u, u)) for u in members}
        label, signature = None, None
        for sig, lab in overrides:        # curated order = label priority
            if sig in gene_set:
                label, signature = lab, sig
                break
        # Multi-partner sites are always shown; single-partner sites only when
        # explicitly curated (e.g. BRAF CDC37 / ITCH, featured in Supp Fig 5).
        if e == s and label is None:
            continue
        multi_sites.append({
            's': s, 'e': e, 'members': members, 'genes': gene_set,
            'cat': site_category(len(members)), 'n': len(members),
            'label': label, 'signature': signature,
            'act_abund': site_effect_pair(protein, members, eff) if eff else None,
        })
    labeled = [st for st in multi_sites if st['label']]
    labeled.sort(key=lambda st: (st['s'] + st['e']) / 2.0)   # top -> bottom
    for i, st in enumerate(labeled):
        st['num'] = i + 1                                     # diagonal order
    return Mr, len(items), multi_sites, labeled, members_uni_ord, Z



def _is_accession(g):
    """True for UniProt-accession-like tokens (e.g. A0A1B0GUL7, V9P4T4) rather
    than gene symbols, so they can be deprioritised in example lists."""
    return len(g) >= 6 and g[1].isdigit()



def _examples(st):
    """'SIG, other, other  +N' example-partner string for a site (gene symbols
    preferred over UniProt accessions)."""
    sig = st['signature']
    others = sorted((g for g in st['genes'] if g != sig),
                    key=lambda g: (_is_accession(g), g.lower()))
    ex = [sig] + others[:2]
    extra = st['n'] - len(ex)
    return ', '.join(ex) + (f'  +{extra}' if extra > 0 else '')



def _draw_paper_panel(fig, ax, Mr, n, multi_sites, labeled, *,
                      protein, members_uni_ord, eff,
                      link_style, title, label_x0=None, label_x1=None):
    """Draw one paper heatmap panel into `ax`, with family labels in the figure
    region starting at `label_x0` (auto = just right of the axis). Returns `im`
    (for a shared colorbar). Does NOT add a colorbar.
    """
    im = ax.imshow(Mr, cmap='magma', vmin=0, vmax=1, aspect='equal')

    for st in multi_sites:
        s, e = st['s'], st['e']
        if st['label']:
            col, lw = SITE_CAT_COLOR_2[site_category_2(st['n'])], 3.2
        else:
            col, lw = '#9aa0a6', 1.6      # unlabelled multi-partner site, faint
        ax.add_patch(Rectangle((s - 0.5, s - 0.5), e - s + 1, e - s + 1,
                               fill=False, edgecolor=col, lw=lw, zorder=4))
        if st['label'] and link_style == 'badge':
            ax.text(s, s, str(st['num']), ha='center', va='center',
                    fontsize=11, fontweight='bold', color='white', zorder=6,
                    bbox=dict(boxstyle='circle,pad=0.28', fc=col,
                              ec='white', lw=1.0))

    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=17, fontweight='bold')

    fig.canvas.draw()
    bbox = ax.get_window_extent().transformed(fig.transFigure.inverted())
    fig_w_in, fig_h_in = fig.get_size_inches()

    # Per-interactor effect strip (Activity | Abundance Cohen's d heatmap) just
    # right of the correlation heatmap. The cluster boxes are drawn ON the strip
    # (spanning both columns over each site's rows) and the family labels point
    # to those boxes with leader lines.
    strip_gap = 0.006
    strip_w = 2 * 0.16 / fig_w_in
    strip_right, row_to_figy, eax = _draw_effect_strip(
        fig, protein, members_uni_ord, n, eff,
        x0=bbox.x1 + strip_gap, y0=bbox.y0, w=strip_w, h=bbox.height,
        header_fs=10.5)
    # Cluster box around both columns (which span x=0..2) for the site's rows,
    # with a small outward margin so it brackets the cells instead of cutting
    # through them. clip_on=False so the full linewidth shows at the strip edges.
    for st in labeled:
        col = SITE_CAT_COLOR_2[site_category_2(st['n'])]
        eax.add_patch(Rectangle((-0.06, st['s'] - 0.56), 2.12,
                                st['e'] - st['s'] + 1.12, fill=False,
                                edgecolor=col, lw=1.0, zorder=6, clip_on=False))

    lx0 = strip_right + 0.024              # labels start past the strip + leaders
    n_lab = len(labeled)
    ys = ([(bbox.y0 + bbox.y1) / 2] if n_lab == 1
          else list(np.linspace(bbox.y1 - 0.03, bbox.y0 + 0.03, n_lab)))
    dy = 13.0 / 72.0 / fig_h_in
    FS_FAM, FS_EX = 11.5, 9.0

    if link_style == 'badge':
        nx, tx = lx0, lx0 + 0.018
        for st, ly in zip(labeled, ys):
            col = SITE_CAT_COLOR_2[site_category_2(st['n'])]
            fig.text(nx, ly + dy, str(st['num']), ha='center', va='center',
                     fontsize=FS_FAM, fontweight='bold', color='white', zorder=6,
                     bbox=dict(boxstyle='circle,pad=0.30', fc=col, ec='none'))
            fig.text(tx, ly + dy, f"{st['label']}  ({st['n']})", ha='left',
                     va='center', fontsize=FS_FAM, fontweight='bold', color=col)
            if st['n'] > 1:
                fig.text(tx, ly, _examples(st), ha='left', va='center',
                         fontsize=FS_EX, color='#333333')
    else:
        x_label = lx0
        # Stagger each connector's vertical "elbow" to a distinct x so the rails
        # don't stack into one line. Rails live in the gap between the effect
        # strip's right edge and the label text, and point at the strip boxes.
        rail_lo, rail_hi = strip_right + 0.005, lx0 - 0.007
        rails = ([(rail_lo + rail_hi) / 2] if len(labeled) <= 1
                 else list(np.linspace(rail_lo, rail_hi, len(labeled))))
        for st, ly, x_rail in zip(labeled, ys, rails):
            col = SITE_CAT_COLOR_2[site_category_2(st['n'])]
            row_c = (st['s'] + st['e']) / 2.0
            x_edge, y_row = strip_right, row_to_figy(row_c)
            y_fam = ly + dy
            for xs, ysg in (([x_edge, x_rail], [y_row, y_row]),
                            ([x_rail, x_rail], [y_row, y_fam]),
                            ([x_rail, x_label - 0.004], [y_fam, y_fam])):
                fig.add_artist(Line2D(xs, ysg, transform=fig.transFigure,
                                      color=col, lw=1.3, zorder=5,
                                      solid_capstyle='round'))
            fig.text(x_label, y_fam, f"{st['label']}  ({st['n']})", ha='left',
                     va='center', fontsize=FS_FAM, fontweight='bold', color=col)
            if st['n'] > 1:
                fig.text(x_label, ly, _examples(st), ha='left', va='center',
                         fontsize=FS_EX, color='#333333')
    return im



def _draw_dendrogram(fig, Z, n, rect, cutoff):
    """Complete-linkage tree on 1 - IoU at figure rectangle `rect`, root at the
    top and leaves aligned to the `n` heatmap columns below it; a dashed line
    marks the 1 - `cutoff` cut that defines the clusters."""
    dax = fig.add_axes(rect)
    dendrogram(Z, orientation='top', ax=dax, no_labels=True,
               link_color_func=lambda k: '#4a4f55')
    for coll in dax.collections:
        coll.set_linewidth(0.6)
    dax.set_xlim(0, 10 * n)                 # leaf i at 10i + 5, column 0 at left
    dax.set_ylim(0, 1.02)
    dax.axhline(1 - cutoff, color='#9aa0a6', lw=0.7, ls=(0, (3, 2)), zorder=0)
    dax.axis('off')
    return dax


def plot_pair_paper(panels, coords, footprints, sig_pairs, iname, cutoff,
                    link_style='badge', eff=None):
    """One figure, N protein panels side-by-side sharing a single colorscale.

    `panels` is a list of (protein, items) drawn left-to-right in the given
    order. Every heatmap is the same physical square (larger hubs just have
    smaller cells), with the same fonts, so panels sit at matched scale. Each
    panel = heatmap + Cohen's d effect strip + leaders + family labels, with the
    clustering dendrogram above it; panels are packed with minimal whitespace
    between them. Shared vertical colourbars
    (interface IoU + effect-strip Cohen's d) sit at the far left.
    """
    n_panels = len(panels)
    hm_in = 2.85                             # heatmap square (inches)
    panel_in = 6.2                           # heatmap + strip + leaders + labels
    dendro_in, dendro_gap = 0.5, 0.03        # dendrogram above each heatmap
    left_in, gap_in, H = 1.4, 0.12, 4.6
    W = left_in + n_panels * panel_in + (n_panels - 1) * gap_in
    h, w, y0 = hm_in / H, hm_in / W, 0.18
    fig = plt.figure(figsize=(W, H))
    im = None
    for i, (protein, items) in enumerate(panels):
        ax_x0 = (left_in + i * (panel_in + gap_in)) / W
        ax = fig.add_axes([ax_x0, y0, w, h])
        Mr, n, multi_sites, labeled, members_uni_ord, Z = _paper_panel_data(
            protein, items, coords, iname, cutoff, eff)
        im = _draw_paper_panel(fig, ax, Mr, n, multi_sites, labeled,
                               protein=protein, members_uni_ord=members_uni_ord,
                               eff=eff, link_style=link_style,
                               title=disp(protein))
        if Z is not None:                  # tree above the heatmap, title above it
            dax = _draw_dendrogram(fig, Z, n, [ax_x0, y0 + h + dendro_gap / H, w,
                                               dendro_in / H], cutoff)
            ax.set_title('')
            dax.set_title(disp(protein), fontsize=17, fontweight='bold')
    # Shared colourbars at the far left: interface IoU (top) + the effect-strip
    # Cohen's d scale (bottom). Placed with room for their left-side tick + axis
    # labels inside the left margin.
    cbx, cbw = 0.6 / W, 0.11 / W
    cax = fig.add_axes([cbx, y0 + 0.34, cbw, 0.26])
    cb = fig.colorbar(im, cax=cax, orientation='vertical')
    cb.set_label('interface IoU', fontsize=11)
    cb.ax.yaxis.set_ticks_position('left')
    cb.ax.yaxis.set_label_position('left')
    cb.ax.tick_params(labelsize=9)

    dcax = fig.add_axes([cbx, y0, cbw, 0.26])
    sm = ScalarMappable(norm=_effect_norm(), cmap=EFFECT_D_CMAP); sm.set_array([])
    dcb = fig.colorbar(sm, cax=dcax, orientation='vertical',
                       ticks=[-EFFECT_D_VLIM, 0, EFFECT_D_VLIM])
    dcb.set_label("Cohen's d", fontsize=11)
    dcb.ax.yaxis.set_ticks_position('left')
    dcb.ax.yaxis.set_label_position('left')
    dcb.ax.tick_params(labelsize=9)
    return fig



# ===========================================================================
# BRAF dimer: categories, interface/control sets, violin helpers
# ===========================================================================

# One (structure, protomer) per contact. These are exactly the rows main.py
# tested, so each interface already has a control set that satisfies the >= 10 A
# criterion against that same chain's interface.
REAL_ID, REAL_CHAIN = '4MNE', 'B'               # physiological R509 dimer, BRAF:MEK1
ALT_ID, ALT_CHAIN = '7K0V', 'B'                 # alternate contact, C-lobe patch
ALT_CHAIN_N = 'A'                               # alternate contact, N-lobe patch



def lighten(hexcol, f=0.55):
    """Blend a colour toward white by fraction `f`. Shared with the sphere figure."""
    return tuple(c + (1.0 - c) * f for c in matplotlib.colors.to_rgb(hexcol))



def darken(hexcol, f=0.35):
    """Blend a colour toward black by fraction `f`."""
    return tuple(c * (1.0 - f) for c in matplotlib.colors.to_rgb(hexcol))


def average_color(c1, c2):
    """Midpoint of two colours in RGB."""
    a, b = matplotlib.colors.to_rgb(c1), matplotlib.colors.to_rgb(c2)
    return tuple((x + y) / 2.0 for x, y in zip(a, b))



REAL = 'Side-to-side\ndimer (4MNE)'
RCTRL = 'Control\n(4MNE)'
ALT = 'Alternate\ncontact (7K0V)'
ACTRL = 'Control\n(7K0V)'
ALLD = 'All\npositions'   # every scored position of the protein; no domain filter
REAL_BASE = '#3fa89f'                            # teal
ALT_BASE = '#e08214'                             # orange (alt/artifact contact)
# Per-protomer interface shades, {structure: {chain: colour}}. Both structures get
# two shades so each protomer's contribution to the contact is visible in the
# ball-and-stick figure; the chain named by REAL_CHAIN / ALT_CHAIN takes the base
# hue and the other a lightened one.
IFACE_SHADES = {
    REAL_ID: {REAL_CHAIN: matplotlib.colors.to_rgb(REAL_BASE),
              ALT_CHAIN_N: lighten(REAL_BASE)},
    ALT_ID: {ALT_CHAIN: matplotlib.colors.to_rgb(ALT_BASE),
             ALT_CHAIN_N: lighten(ALT_BASE)},
}
# The violin pools both protomers, so its colour is the midpoint of that
# structure's two sphere shades -- the violin cannot then be read as belonging to
# one protomer rather than the other.
IFACE_MEAN = {k: average_color(*v.values()) for k, v in IFACE_SHADES.items()}
# Two greys, one per structure, matching the sphere figure's control shades.
CTRL_COLOR_REAL = '#c4c4c4'                      # light grey - 4MNE controls
CTRL_COLOR_ALT = '#5f5f5f'                       # dark grey  - 7K0V controls
CAT_COLOR = {REAL: IFACE_MEAN[REAL_ID],
             RCTRL: CTRL_COLOR_REAL,
             ALT: IFACE_MEAN[ALT_ID],
             ACTRL: CTRL_COLOR_ALT,
             ALLD: '#6ba3d6'}



def braf_sets():
    """
    {structure_id: {'interface': set, 'control': set}}, pooled over both protomers.

    One rule for both structures: interface = union of the two protomers' contact
    residues; controls = union of the two protomers' control sets, minus that
    interface. Single source of truth shared with the ball-and-stick figure.

    THE COST OF POOLING -- disclose this, do not bury it. Each protomer's controls
    were selected >= 10 A from *that protomer's* interface. The pooled interface is
    larger, so some controls now fall inside its exclusion shell:

        4MNE   6 of 113 controls (5%),  worst 6.2 A  -- interface barely grows
                                                        (Jaccard 0.889, symmetric)
        7K0V  31 of  94 controls (33%), worst 4.5 A  -- interface doubles
                                                        (Jaccard 0.037, head-to-tail)

    The rule's purpose is to exclude residues that "may be
    allosterically or sterically constrained by the interaction and therefore biased
    toward the interface DMS-score distribution, diluting the statistical contrast".
    So 7K0V's contaminated controls bias its comparison TOWARD null: its
    non-significance here is partly a property of the control set, not only of the
    contact. The per-protomer split is the rigorous version and belongs in the
    supplement -- there, the C-lobe patch is null (d = -0.14 / -0.02) while the
    N-lobe patch appears significant purely through V600 (activity d = +0.48,
    falling to +0.13 once V600 is dropped).

    Pooling is used here anyway because a reader cannot hold an asymmetric
    interface in mind from a violin plot, and 4MNE pays almost nothing for it.
    """
    df = load_interactions()
    bb = df[(df['scored_protein_name'] == 'BRAF') &
            (df['interactor_name'] == 'BRAF') & (df['skipped'] != True) &
            df['interface_positions'].notna()]
    out = {}
    for key in (REAL_ID, ALT_ID):
        sub = bb[bb['structure'].str.contains(key, na=False)]
        if sub.empty:
            raise SystemExit(f'No {key} BRAF-BRAF row found')
        iface, ctrl = set(), set()
        for _, r in sub.iterrows():
            iface |= parse_positions(r['interface_positions'])
            if pd.notna(r['control_positions']):
                ctrl |= parse_positions(r['control_positions'])
        out[key] = {'interface': iface, 'control': ctrl - iface}
    return out



def braf_positions():
    s = braf_sets()
    return {REAL: s[REAL_ID]['interface'], RCTRL: s[REAL_ID]['control'],
            ALT: s[ALT_ID]['interface'], ACTRL: s[ALT_ID]['control']}



def _cohens_d(a, b):
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float('nan')
    sp = np.sqrt(((na - 1) * np.var(a, ddof=1) +
                  (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2))
    return (np.mean(a) - np.mean(b)) / sp if sp else float('nan')



def _stars(p, d=None):
    """
    Canonical significance rule, not bare p-stars: BH-FDR < 0.05 AND
    |Cohen's d| >= 0.2 (flag_significant).

    The effect-size gate is not decoration. With ~250 interface variants against
    ~1500 controls, a rank test clears p < 0.01 on a shift far too small to
    mean anything -- 7K0V activity is p = 1.2e-3 at |d| = 0.14, and its d and
    Cliff's delta do not even agree in sign. Annotating that as ** would have
    the figure assert significance for the very contact it exists to call
    non-functional.
    """
    if d is not None and not np.isnan(d) and abs(d) < SIG_EFFECT:
        return 'ns'
    return ('***' if p < 1e-3 else '**' if p < 1e-2 else '*' if p < 0.05
            else 'ns')



def _log_ticks(lo, hi):
    """Tick positions (in log10 units) + original-value labels spanning [lo, hi]."""
    cand = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1, 1.5, 2, 3, 5, 7, 10, 15, 20, 30]
    vals = [c for c in cand if lo <= c <= hi]
    if not vals:
        vals = [round(lo, 2), round(hi, 2)]
    return np.log10(vals), [f'{v:g}' for v in vals]



def _bracket(ax, x1, x2, y, h, text, fs=15):
    """Draw a significance bracket spanning x1..x2 at height y (tick height h)."""
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.0,
            color='#3a4047', clip_on=False)
    ax.text((x1 + x2) / 2.0, y + h, text, ha='center', va='bottom',
            fontsize=fs, color='#3a4047', clip_on=False)



def _sig_brackets(ax, data, data_raw, cats, pairs, q=None):
    """
    Bracket the explicit (cat1, cat2) `pairs` with the canonical significance
    verdict plus Cohen's d. `q` maps a pair to its BH-FDR q-value, which then
    sets the verdict in place of the pair's raw Mann-Whitney p.

    `data` is what is drawn (log10 for activity) and sets bracket positions;
    `data_raw` is the untransformed score and is what d is computed from, so the
    printed d matches main.py and the supplementary table rather than being a
    log-scale number. Mann-Whitney is rank-based, so p is identical either way.
    """
    comps = []
    for c1, c2 in pairs:
        i1, i2 = cats.index(c1), cats.index(c2)
        v1, v2 = data[i1], data[i2]
        r1, r2 = data_raw[i1], data_raw[i2]
        if len(v1) > 5 and len(v2) > 5:
            p = mannwhitneyu(v1, v2, alternative='two-sided').pvalue
            if q is not None:
                p = q[(c1, c2)]
            d = _cohens_d(r1, r2)
            comps.append((abs(i2 - i1), min(i1, i2) + 1, max(i1, i2) + 1, p, d))
    if not comps:
        return
    comps.sort()
    y0, y1 = ax.get_ylim()
    step = (y1 - y0) * 0.07
    # Shortest spans lowest; brackets that do not overlap in x share a level, so
    # the two interface-vs-own-control comparisons sit side by side.
    placed, n_lvl = [], 0
    for _, x1, x2, p, d in comps:
        lvl = 0
        while any(l == lvl and not (x2 < px1 or x1 > px2)
                  for l, px1, px2 in placed):
            lvl += 1
        placed.append((lvl, x1, x2))
        n_lvl = max(n_lvl, lvl + 1)
        # Show d alongside the verdict so the effect-size gate is auditable.
        label = _stars(p, d)
        if not np.isnan(d):
            label = f'{label}  d={d:+.2f}'
        _bracket(ax, x1, x2, y1 + step * (0.4 + lvl), step * 0.3, label,
                 fs=12)
    ax.set_ylim(y0, y1 + step * (0.9 + n_lvl))



# ===========================================================================
# Pipeline schematic
# ===========================================================================

INK, MUTED, HILITE = '#1f2933', '#5b6770', '#b30000'
# (predictor name in the data, display label) — order = left-to-right in labels.
# The RoseTTAFold2-PPI source is shown compactly as 'RF2-PPI' in the per-source
# summary lines (the source box below spells out the full name).
SRC = [('RCSB_PDB', 'PDB'), ('Predictomes', 'AlphaFold'), ('Zhang_et_al', 'RF2-PPI')]



def compute_funnel(config_path='config/visualization_config.yaml'):
    """Compute every count shown in the schematic from the canonical pipeline,
    so nothing is hardcoded (and nothing can go stale).

    - structures: unique structure files per source, INCLUDING skipped
      interactions (the raw structural input), pre- and post- clash filter.
    - interactions: distinct UNORDERED protein pairs with an assayable
      interface (valid stats) after the clash filter and dropping skipped.
      A pair scored from both sides (e.g. SOS1->KRAS and KRAS->SOS1) is ONE
      interaction here; a homodimer (scored==interactor) is one interaction.
      Keyed on gene names so allele/species-variant accessions (HLA-A,
      MAP2K1) collapse. The per-DIRECTION count (SOS1->KRAS and KRAS->SOS1 as
      two edges) is intentionally kept separate for the pathway diagram.
    - significant: canonical rule via flag_significant (BH-FDR<0.05 &
      |Cohen's d|>=0.2), counted in >=1 assay in >=1 direction.
    """
    cfg = yaml.safe_load(open(config_path))
    kw = dict(predictomes_dir=cfg['predictomes_dir'],
              zhangetal_dir=cfg['zhangetal_dir'],
              pdb_dir=cfg.get('pdb_output_dir'))

    def by_src(df):
        u = df.drop_duplicates('structure')
        return {lab: int((u['predictor'] == p).sum()) for p, lab in SRC}

    pre = load_combined_results(**kw, include_skipped=True, apply_clash_filter=False)
    raw = by_src(pre)

    analyzed = load_combined_results(
        **kw, include_skipped=False, apply_clash_filter=True
    ).dropna(subset=['log_fold_change', 'pval_fdr', 'cohens_d'])
    agg = aggregate_volcano_data(analyzed)
    agg['is_sig'] = flag_significant(agg)
    # Count over the two assays, activity and abundance, so the funnel matches
    # the volcanoes.
    agg = agg[agg['assay'].isin(['activity', 'abundance'])].copy()

    # Canonical UNORDERED protein-pair key (frozenset of gene names). Collapses
    # both-directions-scored pairs and homodimers to one interaction; keyed on
    # names so allele/species-variant accessions merge. See docstring.
    def _pair(df):
        return [frozenset((a, b)) for a, b in
                zip(df['scored_protein_name'], df['interactor_name'])]
    agg['_pair'] = _pair(agg)
    key = '_pair'

    def sig_by_src(a):
        return {lab: int(a[a['predictor'] == p].groupby(key)['is_sig'].any().sum())
                for p, lab in SRC}

    def assay_sig(name):
        return int(agg[agg['assay'] == name].groupby(key)['is_sig'].any().sum())

    # Assessable structures = post clash-filter AND with an assayable interface
    # (the structures that actually enter the analysis), counted per source.
    assess = (analyzed[analyzed['assay'].isin(['activity', 'abundance'])]
              .drop_duplicates('structure')).copy()
    assess['_pair'] = _pair(assess)
    assessable = {lab: int((assess['predictor'] == p).sum()) for p, lab in SRC}
    sc = assess.groupby(key).size()                # structures per interaction

    # A candidate interface = one (interaction, predictor, binding mode) — the
    # structures proposing that interface pooled into one test. Counting is
    # per-predictor (same physical interface from 2 predictors = 2 candidates);
    # the next step collapses redundant candidates into distinct interactions.
    # The grouping is structural (assay-independent), so we report ONE count
    # (union across assays), then the fraction significant in each assay.
    gk = ['scored_protein', 'interactor', 'predictor', 'binding_mode']
    grouped = int(agg.drop_duplicates(gk).shape[0])
    grouped_sig = {name: int(agg[agg['assay'] == name].groupby(gk)['is_sig']
                             .any().sum())
                   for name in ('activity', 'abundance')}

    return dict(
        raw=raw, assessable=assessable,
        tested_union=agg.groupby(key).ngroups,
        grouped=grouped, grouped_sig=grouped_sig,
        dots={name: int((agg['assay'] == name).sum())
              for name in ('activity', 'abundance')},
        sig=sig_by_src(agg), sig_union=int(agg.groupby(key)['is_sig'].any().sum()),
        activity=assay_sig('activity'), abundance=assay_sig('abundance'),
        multi_struct=int((sc > 1).sum()), med_struct=int(sc.median()),
        max_struct=int(sc.max()),
        multi_src=int((agg.groupby(key)['predictor'].nunique() > 1).sum()),
    )



def box(ax, x, y, w, h, title, subtitle=None, *, fc='white', ec=INK, fs=16,
        sub_fs=11, lw=1.5):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle='round,pad=0.010,rounding_size=0.018',
                                fc=fc, ec=ec, lw=lw, zorder=2))
    if subtitle:
        ax.text(x, y + h * 0.17, title, ha='center', va='center',
                fontsize=fs, fontweight='bold', color=INK, zorder=3)
        ax.text(x, y - h * 0.27, subtitle, ha='center', va='center',
                fontsize=sub_fs, color=MUTED, zorder=3)
    else:
        ax.text(x, y, title, ha='center', va='center',
                fontsize=fs, fontweight='bold', color=INK, zorder=3)



def draw_portrait(F):
    """Single-column PORTRAIT funnel sized for a ~1.5 in (w) x 3 in (h) figure
    slot. Source names are abbreviated to match the Venn/volcano legends
    (PDB / AF-M / RF2-PPI); the arrow annotations are kept terse (fuller detail
    lives in the figure legend). Rendered at the target size so the point sizes
    seen here are what print."""
    raw = F['raw']; raw_tot = sum(raw.values())
    assess = F['assessable']; assess_tot = sum(assess.values())
    tested_u, sig_u = F['tested_union'], F['sig_union']
    pct = round(100 * sig_u / tested_u)

    fig, ax = plt.subplots(figsize=(1.5, 3.0))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    CX = 0.5

    ax.text(CX, 0.975, 'MAPK-pathway interaction\nstructures (17 proteins)',
            ha='center', va='center', fontsize=5.0, fontweight='bold',
            color=INK, linespacing=0.95)

    # 1. three sources (abbreviated) -> combined pool
    sw, sh, sy = 0.305, 0.075, 0.885
    sxs = [0.17, 0.5, 0.83]
    box(ax, sxs[0], sy, sw, sh, f"PDB\n{raw['PDB']}", fc='#ebe6f4', ec=PDB,
        fs=4.8, lw=0.7)
    box(ax, sxs[1], sy, sw, sh, f"AF-M\n{raw['AlphaFold']}", fc='#f9e5f0', ec=AF,
        fs=4.8, lw=0.7)
    box(ax, sxs[2], sy, sw, sh, f"RF2-PPI\n{raw['RF2-PPI']}", fc='#f7efd8', ec=RF,
        fs=4.8, lw=0.7)
    for sx in sxs:
        ax.add_patch(FancyArrowPatch((sx, sy - sh / 2), (CX, 0.812),
                     arrowstyle='-|>', mutation_scale=5, lw=0.7, color=MUTED,
                     zorder=1))
    box(ax, CX, 0.775, 0.94, 0.058, f'{raw_tot:,} candidate structures',
        fc='#f3f4f6', fs=5.2, lw=0.8)

    # 2. clash + assayable-interface filter -> assessable
    ax.add_patch(FancyArrowPatch((CX, 0.746), (CX, 0.607), arrowstyle='-|>',
                 mutation_scale=6, lw=1.1, color=MUTED, zorder=1))
    ax.text(0.55, 0.677, f'clash + interface\nfilter (−{raw_tot - assess_tot})',
            ha='left', va='center', fontsize=4.0, color=MUTED, style='italic',
            linespacing=0.95)
    box(ax, CX, 0.575, 0.94, 0.058, f'{assess_tot} assessable structures',
        fc='#eef4ea', ec=PDB, fs=5.2, lw=0.8)

    # 3. test every complex, then collapse to unique pairs
    ax.add_patch(FancyArrowPatch((CX, 0.546), (CX, 0.26), arrowstyle='-|>',
                 mutation_scale=6, lw=1.1, color=MUTED, zorder=1))
    ax.text(0.55, 0.40,
            f'test each of the\n{assess_tot} complexes\n(interface vs.\n'
            'control), then\ncollapse to pairs',
            ha='left', va='center', fontsize=4.0, color=MUTED, style='italic',
            linespacing=0.95)
    box(ax, CX, 0.165, 0.96, 0.11,
        f'{sig_u} / {tested_u} distinct\ninteractions ({pct}%)\n'
        'functionally supported',
        fc='#fde8e8', ec=HILITE, lw=1.2, fs=5.0)
    return fig



# ===========================================================================
# Volcano inputs
# ===========================================================================

def load_aggregated(config: dict, *, assays: Optional[List[str]] = None,
                    group: bool = True) -> pd.DataFrame:
    """
    Load combined per-structure interaction results for the volcano panels.

    With ``group=True`` (default), structures are pooled by binding mode within
    each predictor (IoU>0.3 sub-clustering, see aggregate_volcano_data) to one
    row per (scored × interactor × assay × predictor × binding mode) — the ~546
    "candidate interfaces". With ``group=False`` NO pooling is done: every
    assessable structure is kept as its own point (~914), and BH-FDR is
    recomputed per assay across those per-structure Mann-Whitney p-values (the
    same one-family-per-assay rule the grouped path uses), so the two views are
    directly comparable.

    Parameters
    ----------
    config : dict
        Loaded `visualization_config.yaml`.
    assays : list of str, optional
        If given, restrict to these assay names first.
    group : bool, default True
        Pool structures within IoU>0.3 binding modes (True) or keep every
        assessable structure as an individual point (False).

    Returns
    -------
    pd.DataFrame
        Data ready for plotting. Columns include cohens_d, pval_fdr,
        neg_log_pval, n_interface_positions, interaction_name.
    """
    raw = load_combined_results(
        predictomes_dir=config['predictomes_dir'],
        zhangetal_dir=config['zhangetal_dir'],
        pdb_dir=config.get('pdb_output_dir'),
        include_skipped=False,
    )

    # Drop rows that the analysis flagged as skipped or have no statistics.
    raw = raw.dropna(subset=['log_fold_change', 'pval_fdr', 'cohens_d'])

    if assays is not None:
        raw = raw[raw['assay'].isin(assays)].copy()

    if group:
        # One point per biological interaction per predictor (binding-mode pooled).
        agg = aggregate_volcano_data(raw)
        return prepare_volcano_data(agg)

    # Ungrouped: every assessable structure is its own point. Re-apply BH-FDR
    # per assay across the per-structure Mann-Whitney p (one family per assay,
    # all predictors pooled), mirroring aggregate_volcano_data's recomputation.
    import numpy as np
    from statsmodels.stats.multitest import multipletests
    df = raw.copy()
    df['pval_fdr'] = np.nan
    for _assay in df['assay'].dropna().unique():
        mask = (df['assay'] == _assay) & df['pval_mannwhitney'].notna()
        if mask.any():
            pvals = df.loc[mask, 'pval_mannwhitney'].clip(lower=1e-300)
            df.loc[mask, 'pval_fdr'] = multipletests(pvals, method='fdr_bh')[1]
    df['neg_log_pval'] = -np.log10(df['pval_fdr'].clip(lower=1e-300))
    return prepare_volcano_data(df)



# ===========================================================================
# CRAF chaperone-interface sets and violin helpers
# ===========================================================================

def craf_positions():
    """Return {category: set(int positions)} for CRAF."""
    df = load_interactions()
    raf = df[(df['scored_protein_name'] == 'RAF1') & (df['skipped'] != True) &
             df['interface_positions'].notna()]

    def pooled(partners, col):
        s = set()
        for _, r in raf[raf['interactor_name'].isin(partners)].iterrows():
            if pd.notna(r[col]):
                s |= parse_positions(r[col])
        return s

    cdc37 = pooled(['CDC37'], 'interface_positions')
    hsp90 = pooled(['HSP90AB1', 'HSP90AA1'], 'interface_positions')
    # Union of the matched controls from all three interactions, minus BOTH
    # interfaces: a control residue for one interaction can be an interface
    # residue of the other (all 26 such residues), so subtract them or the
    # control is contaminated with functionally-active interface positions.
    control = (pooled(['CDC37', 'HSP90AB1', 'HSP90AA1'], 'control_positions')
               - cdc37 - hsp90)
    return {'CDC37\ninterface': cdc37, 'HSP90\ninterface': hsp90,
            'Control': control}



def _stars_mw(p):
    return ('***' if p < 1e-3 else '**' if p < 1e-2 else '*' if p < 0.05
            else 'ns')



def _sig_brackets_vs_control(ax, data, cats):
    """Bracket each 'interface' category against Control and All.

    `data` is the list of per-category value arrays (log-transformed for the log
    panel; the Mann-Whitney U test is rank-based so p-values are unchanged).
    Brackets are stacked shortest-span-lowest just above the current axis top,
    and the y-limit is extended to make room.
    """
    iface = [c for c in cats if 'interface' in c]
    targets = [t for t in ('Control', ALLD) if t in cats]
    comps = []
    for ic in iface:
        vi = data[cats.index(ic)]
        for tc in targets:
            vt = data[cats.index(tc)]
            if len(vi) > 5 and len(vt) > 5:
                p = mannwhitneyu(vi, vt, alternative='two-sided').pvalue
                x1, x2 = cats.index(ic) + 1, cats.index(tc) + 1
                comps.append((abs(x2 - x1), min(x1, x2), max(x1, x2), p))
    if not comps:
        return
    comps.sort()
    y0, y1 = ax.get_ylim()
    step = (y1 - y0) * 0.07
    for lvl, (_, x1, x2, p) in enumerate(comps):
        _bracket(ax, x1, x2, y1 + step * (0.4 + lvl), step * 0.3, _stars_mw(p))
    ax.set_ylim(y0, y1 + step * (0.9 + len(comps)))



# ===========================================================================
# Panel drawing: each takes what the notebook computed and writes PDF + PNG
# ===========================================================================

FIG6 = Path('output/figures/fig6')


def plot_funnel(F, out=FIG6 / 'pipeline_funnel.pdf'):
    """5a: the funnel, from the counts in `F`."""
    reset_style(publication=True)
    apply_arial()
    out.parent.mkdir(parents=True, exist_ok=True)
    figp = draw_portrait(F)
    figp.savefig(out, bbox_inches='tight')
    figp.savefig(out.with_suffix('.png'), dpi=400, bbox_inches='tight')
    plt.close(figp)


def plot_volcanoes(points, corner_counts, highlight, out_dir=FIG6, *,
                   assays=('abundance', 'activity'), effect_threshold=0.2,
                   fdr_alpha=0.05):
    """6c/6d: one volcano per assay, the three sources overlaid. Featured
    interactions are labelled in dark grey, each label joined by a thin line to
    every point of that interaction; labels start at a fixed offset, to be
    placed by hand. The colour key is drawn on the activity panel."""
    reset_style(publication=True)
    setup_publication_style()
    highlight_colors = build_highlight_color_map(points, highlight)
    for assay in assays:
        fig, axes = plot_volcano_overlay_three_sources(
            points[points['assay'] == assay],
            assays=[assay],
            effect_threshold=effect_threshold,
            fdr_alpha=fdr_alpha,
            annotate_significant=False,
            max_labels_per_panel=8,
            highlight_pairs=highlight,
            highlight_colors=highlight_colors,
            auto_place=False,
            figsize_per_panel=(4.5, 4.5),
            show_legend=(assay == 'activity'),
            corner_counts=corner_counts,
            output_path=out_dir / f'volcano_pub_{assay}_all_predictors.pdf',
        )
        plt.close(fig)


def plot_volcano_size_key(points, out=FIG6 / 'volcano_size_legend'):
    """Marker-size key for 5c/5d, on the same scale as the volcanoes (the
    interface-residue range of `points`)."""
    reset_style(publication=True)
    SIZE_RANGE = (10, 180)          # matches plot_volcano_overlay_three_sources
    REPS = [5, 15, 25]              # representative interface-position counts
    FS = 11                         # label font size (matches overlay legend)
    apply_arial()
    s = pd.to_numeric(points['n_interface_positions'], errors='coerce').dropna()
    smin, smax = float(s.min()), float(s.max())

    def s_of(v):
        return (SIZE_RANGE[0] + (v - smin) / (smax - smin)
                * (SIZE_RANGE[1] - SIZE_RANGE[0]))

    # Manual layout so the circles sit close together: place each marker at a
    # y just far enough below the previous one to clear both radii + a small gap.
    fig, ax = plt.subplots(figsize=(1.15, 1.35))
    ax.set_xlim(0, 1); ax.axis('off')
    r_pts = [np.sqrt(s_of(v) / np.pi) for v in REPS]     # marker radii (points)
    gap = 4.0                                            # points between edges
    centers, y = [], 0.0
    for i, r in enumerate(r_pts):
        if i:
            y -= (r_pts[i - 1] + r + gap)               # descend by both radii
        centers.append(y)
    # Convert point offsets to data units via the figure height.
    fig_h_pts = fig.get_size_inches()[1] * 72
    pad = max(r_pts) + 6
    ax.set_ylim((min(centers) - pad) / fig_h_pts, (max(centers) + pad) / fig_h_pts)
    xm = 0.28
    for v, yc, r in zip(REPS, centers, r_pts):
        yy = yc / fig_h_pts
        ax.scatter([xm], [yy], s=s_of(v), facecolor='#b0b0b0',
                   edgecolor='white', linewidths=0.4, alpha=0.9, zorder=3)
        ax.text(xm + 0.22, yy, str(v), ha='left', va='center', fontsize=FS)
    ax.text(xm + 0.11, (max(centers) + pad * 0.7) / fig_h_pts,
            'interface\npositions', ha='center', va='bottom', fontsize=FS,
            multialignment='center')
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ('.pdf', '.png'):
        fig.savefig(out.with_suffix(ext), dpi=300, bbox_inches='tight',
                    transparent=True)
    plt.close(fig)


def plot_source_venn(subsets, avg, n_pairs, out=FIG6 / 'source_pair_venn'):
    """ED9a: `subsets` in matplotlib-venn order (PDB only, AF only, PDB&AF, RF
    only, PDB&RF, AF&RF, all three); `avg` = mean complexes per pair by source."""
    reset_style(publication=False)
    COLORS = {'PDB': '#6a51a3', 'AF-M': '#c51b7a', 'RF2-PPI': '#b8860b'}
    apply_arial()
    labels = tuple(f'{lab}\n({avg[lab]:.1f} complexes/pair)'
                   for lab in ('PDB', 'AF-M', 'RF2-PPI'))
    fig, ax = plt.subplots(figsize=(8.2, 7.2))
    v = venn3(subsets=subsets, set_labels=labels,
              set_colors=(COLORS['PDB'], COLORS['AF-M'], COLORS['RF2-PPI']),
              alpha=0.55, ax=ax)
    venn3_circles(subsets=subsets, lw=1.2, color='#3a4047', ax=ax)
    for t in v.subset_labels:
        if t:
            t.set_fontsize(15); t.set_fontweight('bold')
    for t in v.set_labels:
        if t:
            t.set_fontsize(14); t.set_fontweight('bold')
    ax.set_title(f'Unique interacting protein pairs by structural source '
                 f'(n = {n_pairs})',
                 fontsize=15, fontweight='bold')
    fig.text(0.5, 0.02,
             'Assessable unordered protein pairs; a pair is in a circle if that '
             'source contributes ≥1 assessable complex.\n"complexes/pair" = mean '
             'distinct structure models per pair within that source.',
             ha='center', va='bottom', fontsize=10.5, color='#5b6770',
             style='italic')
    fig.subplots_adjust(bottom=0.13)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(out.with_suffix('.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)


# Protein classes, left to right, for ED9b.
PROTEIN_CLASSES = {
    'RTKs': ['EGFR', 'ERBB2', 'MET', 'RET'],
    'Adaptor/\nPhosphatase': ['GRB2', 'PTPN11'],
    'GEFs': ['SOS1', 'SOS2'],
    'RASs': ['KRAS', 'MRAS'],
    'RAFs': ['ARAF', 'BRAF', 'RAF1'],
    'Scaffolds': ['KSR1', 'KSR2'],
    'MEKs': ['MAP2K1', 'MAP2K2'],
}


def plot_interactions_by_class(tot, sig, out=FIG6 / 'interactor_class_summary'):
    """ED9b: `tot` and `sig` map (scored protein, source in PDB/AF/RF) to the
    number of partners, and of functionally supported partners."""
    reset_style(publication=True)
    DISP = DISPLAY_NAME
    SRC_ORDER = ['PDB', 'AF', 'RF']
    DARK = {'PDB': PDB, 'AF': AF, 'RF': RF}
    LIGHT = {'PDB': '#b9a9d6', 'AF': '#e3a6c9', 'RF': '#ddc47a'}   # pale tints
    SRC_LABEL = {'PDB': 'PDB', 'AF': 'AF-M', 'RF': 'RF2-PPI'}
    CLASSES = PROTEIN_CLASSES
    apply_arial()

    fig, ax = plt.subplots(figsize=(5.25, 2.25))
    x = 0.0
    xticks, xlabels, class_spans = [], [], []
    for cls, prots in CLASSES.items():
        start = x
        for pr in prots:
            y0 = 0.0; ssum = 0.0; bounds = []
            for s in SRC_ORDER:
                nt = tot.get((pr, s), 0); ns = sig.get((pr, s), 0)
                if nt == 0:
                    continue
                ax.bar(x, ns, bottom=y0, width=0.82, color=DARK[s], lw=0, zorder=3)
                ax.bar(x, nt - ns, bottom=y0 + ns, width=0.82, color=LIGHT[s],
                       lw=0, zorder=3)
                y0 += nt; ssum += ns
                bounds.append(y0)
            # thin black separators BETWEEN predictor blocks only (the top of
            # each block except the last); no line between sig/non-sig.
            for b in bounds[:-1]:
                ax.plot([x - 0.41, x + 0.41], [b, b], color='black', lw=0.6,
                        zorder=6)
            if y0 > 0:
                ax.text(x, y0 + 2, f'{int(ssum)}/{int(y0)}', ha='center',
                        va='bottom', fontsize=4.4, color='#5b6770')
            xticks.append(x); xlabels.append(DISP.get(pr, pr))
            x += 1.0
        class_spans.append((cls, start, x - 1.0))
        x += 0.85

    ax.set_ylim(0, 132)
    ax.set_xlim(-0.7, x - 0.85 + 0.7)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, rotation=90, fontsize=5.4)
    ax.tick_params(axis='x', length=0, pad=1)
    ax.tick_params(axis='y', labelsize=5.4, length=2)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel('Interactions by predictor', fontsize=6.5)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)

    for cls, x0, x1 in class_spans:
        xc = (x0 + x1) / 2.0
        ax.plot([x0 - 0.42, x1 + 0.42], [126, 126], color='#5b6770', lw=1.1,
                clip_on=False, zorder=4)
        ax.text(xc, 128.5, cls, ha='center', va='bottom', fontsize=6,
                fontweight='bold', color='#1f2933', linespacing=0.9)

    # legend: one solid swatch per predictor, then a solid/greyed pair showing
    # the significant vs not-significant shading.
    handles = [Patch(fc=DARK[s], ec='none', label=SRC_LABEL[s]) for s in SRC_ORDER]
    handles.append(Patch(fc='none', ec='none', label=' '))          # spacer
    handles.append(Patch(fc='#4d4d4d', ec='none', label='significant'))
    handles.append(Patch(fc='#cccccc', ec='none', label='not significant'))
    ax.legend(handles=handles, fontsize=5, frameon=False, loc='upper left',
              bbox_to_anchor=(0.0, 0.88), ncol=1, handlelength=0.9,
              handletextpad=0.4, labelspacing=0.25, borderaxespad=0.15,
              title='predictor', title_fontsize=5)
    fig.text(0.5, -0.02,
             'a partner is counted once per predictor that models it.',
             ha='center', va='top', fontsize=5, color='#5b6770', style='italic')

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(out.with_suffix('.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)


VIOLIN_ASSAYS = [('activity', 'No_treatment', 'Activity'),
                 ('abundance', 'No_treatment', 'Abundance')]


ALL_COLOR = CAT_COLOR[ALLD]                  # 'All' violin


def substituted_q(complexes, assay, drop, new_p):
    """BH q-values for tests that stand in for some complexes' own tests.

    The BH family is every complex's Mann-Whitney p in `assay` (as in the
    per-complex rule), minus the rows selected by `drop` (a function of that
    table returning a boolean mask), plus the `new_p` tests ({key: p}).
    Returns {key: q}.
    """
    rows = complexes[(complexes['assay'] == assay) & complexes['pval_mannwhitney'].notna()]
    keep = rows.loc[~drop(rows).to_numpy(), 'pval_mannwhitney'].to_list()
    keys = list(new_p)
    ps = np.clip(keep + [new_p[k] for k in keys], 1e-300, None)
    q = multipletests(ps, method='fdr_bh')[1]
    return dict(zip(keys, q[len(keep):]))


def plot_interface_violins(pos, scores, *, cats, colors, pairs, protein, title, out,
                           q=None):
    """Missense score distributions per position category, activity (log axis)
    beside abundance (linear), with significance brackets.

    pos       : {category: positions}; the ALLD category, if listed in `cats`, is
                filled with every scored position.
    scores    : one protein's score table (library_scores).
    cats      : categories left to right; colors: {category: colour}.
    pairs     : (category, category) comparisons to bracket.
    q         : {assay: {pair: BH-FDR q}}; when given, q sets each verdict in
                place of the pair's raw Mann-Whitney p.
    """
    reset_style(publication=True)
    SCORE_COL = 'average score'
    apply_arial()

    fig, axes = violin_figure(len(cats))
    for ax, (assay, treat, label) in zip(axes, VIOLIN_ASSAYS):
        sub = scores[(scores['assay'] == assay) & (scores['assay_treatment'] == treat) &
                     (scores['Mutation Type'] == 'missense')].dropna(subset=[SCORE_COL])
        catpos = dict(pos); catpos[ALLD] = set(sub['Position'])
        data_raw, counts = [], []
        for c in cats:
            vals = sub[sub['Position'].isin(catpos[c])][SCORE_COL].values
            data_raw.append(vals)
            counts.append(f'{len(catpos[c])} pos\n{len(vals)} var')

        # Activity spans a wide multiplicative range (long gain-of-function
        # tail) -> log y-axis; abundance is tightly bounded -> linear. The log
        # panel's violin KDE is computed in log10 space (correct density shape
        # for multiplicative data); the axis is relabelled with original values.
        use_log = (assay == 'activity')
        data = [np.log10(v) if use_log else v for v in data_raw]

        parts = ax.violinplot(data, showmedians=True, showextrema=False,
                              widths=0.85)
        for b, c in zip(parts['bodies'], cats):
            b.set_facecolor(colors[c]); b.set_alpha(0.8); b.set_edgecolor('#3a4047')
            b.set_linewidth(0.6)
        parts['cmedians'].set_color('#1f2933'); parts['cmedians'].set_linewidth(1.4)
        # WT reference (1.0; = 0 in log10 space) + median markers
        ax.axhline(0.0 if use_log else 1.0, color='#b0b0b0', lw=0.9, ls='--',
                   zorder=0)
        for i, vals in enumerate(data, 1):
            if len(vals):
                ax.scatter([i], [np.median(vals)], s=16, color='white',
                           edgecolors='#1f2933', linewidths=0.8, zorder=5)

        # Mann-Whitney is rank-based, so unaffected by the log transform.
        _sig_brackets(ax, data, data_raw, cats, pairs, q=q[assay] if q else None)

        ax.set_xticks(range(1, len(cats) + 1))
        ax.set_xticklabels(cats, fontsize=15)
        # counts on a smaller separate line beneath each category name
        trans = ax.get_xaxis_transform()
        for i, cnt in enumerate(counts, 1):
            ax.text(i, -0.155, cnt, transform=trans,
                    ha='center', va='top', fontsize=11, color='#5b6770')
        ax.set_title(f'{protein} {label}', fontsize=18, fontweight='bold',
                     color='#1f2933')
        if use_log:
            allv = np.concatenate([v for v in data_raw if len(v)])
            tpos, tlab = _log_ticks(allv.min(), allv.max())
            ax.set_yticks(tpos); ax.set_yticklabels(tlab)
        ax.set_ylabel(f'WT-relative {label.lower()} score', fontsize=16)
        ax.tick_params(labelsize=14)
        for s_ in ('top', 'right'):
            ax.spines[s_].set_visible(False)

    fig.suptitle(title, fontsize=18, fontweight='bold', y=0.99)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix('.pdf'))
    fig.savefig(out.with_suffix('.png'), dpi=200)
    plt.close(fig)
    return out


def plot_iou_heatmaps(panels, coords, footprints, sig_pairs, iname, eff,
                      out_dir=FIG6 / 'site_heatmaps', cutoff=SITE_CUTOFF):
    """ED9j: one IoU heatmap per (protein, partners) in `panels`, side by side on
    a shared colour scale; hub clusters carry family labels on leader lines."""
    reset_style(publication=True)
    apply_arial()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig = plot_pair_paper(panels, coords, footprints, sig_pairs, iname,
                          cutoff, link_style='leader', eff=eff)
    out = out_dir / ('_'.join(disp(p) for p, _ in panels) + '_sites_iou0p5_paper_leaders_combined')
    fig.savefig(out.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(out.with_suffix('.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)


# ===========================================================================
# Structure renders (PyMOL): one renderer and one panel composer for every
# structure figure; the notebook says what each one shows
# ===========================================================================

PYMOL = os.environ.get('PYMOL', str(Path.home() / 'pymol' / 'pymol'))
PYMOL_MAX_THREADS = int(os.environ.get('PYMOL_MAX_THREADS', '2'))

# Hand-chosen cameras (PyMOL get_view, in each model's own frame).
STRUCTURE_VIEWS = {
    # exposes the side-to-side interface instead of burying it between protomers
    '4MNE': (0.263974041, 0.896977603, 0.354608297,
             0.101454109, -0.391430616, 0.914597869,
             0.959178984, -0.205453381, -0.194329247,
             0.000000000, 0.000000000, -263.887420654,
             9.183140755, -21.212041855, -81.313529968,
             208.050857544, 319.723999023, -20.000000000),
    # shows both surfaces of the head-to-tail contact at once
    '7K0V': (0.302750289, 0.947531343, -0.102611691,
             -0.927981317, 0.317607433, 0.194875121,
             0.217239425, 0.036221396, 0.975443602,
             0.000000000, 0.000000000, -262.312072754,
             35.626865387, 12.323307037, -39.570312500,
             206.808837891, 317.815307617, -20.000000000),
    '7Z38': (-0.896922588, -0.314111918, 0.311212987,
             0.439872652, -0.562099695, 0.700393200,
             -0.045067810, 0.765097320, 0.642331481,
             0.000773020, 0.000166602, -262.293823242,
             114.617507935, 57.499362946, 50.620651245,
             -16.127233505, 542.863830566, -20.000000000),
    'BRAF-ITCH': (-0.660545528, 0.640572190, 0.391581446,
                  -0.735192537, -0.446143210, -0.510336339,
                  -0.152208954, -0.624988437, 0.765653133,
                  -0.000007629, -0.000011444, -80.614593506,
                  -2.019355774, -10.405792236, 13.966537476,
                  -1137.969116211, 1299.198242188, -20.000000000),
}

# Partner colours, kept away from the blue-white-red score scale.
PARTNER_COLOR = {'HSP90': '#7fbf7b', 'CDC37': '#d9a44e', 'ITCH': '#9b72c7'}
PARTNER_TRANSPARENCY = 0.5
CONTACT_CUTOFF = 4.5      # A: partner residues shown as sticks near the interface


def residue_scores(scores, libraries, assay, treatment='No_treatment',
                   score_col='average score'):
    """(mean missense score per position, WT centre) for one protein and assay.
    The centre is the mean score of the wild-type and synonymous barcodes."""
    s = scores[scores['library'].isin(libraries) & (scores['assay'] == assay)
               & (scores['assay_treatment'] == treatment)].copy()
    s['Position'] = pd.to_numeric(s['Position'], errors='coerce')
    s = s.dropna(subset=['Position'])
    s['Position'] = s['Position'].astype(int)
    pos = s[s['Mutation Type'] == 'missense'].groupby('Position')[score_col].mean()
    wt = s[s['Mutation Type'].isin(['synonymous wild type', 'wild type'])]
    centre = float(wt[score_col].mean()) if len(wt) else 1.0
    return pos, centre


def score_scale(pos, centre, cap=None):
    """Colour-scale limits symmetric about the WT centre: the 95th percentile of
    |score - centre|, optionally capped (BRAF activity's long gain-of-function
    tail would otherwise wash out the loss-of-function colours)."""
    dev = float(np.percentile(np.abs(pos.values - centre), 95))
    if cap is not None:
        dev = min(dev, cap)
    return centre - dev, centre + dev


def plddt_keep(pdb, min_plddt, always=None):
    """{chain: residues} with pLDDT >= `min_plddt` (read from the Cb, or Ca, B
    factor), plus the residues in `always` ({chain: residues}) whatever their
    pLDDT. Pass to render_structure(keep=...) to hide disordered regions."""
    s = PDBParser(QUIET=True).get_structure('m', str(pdb))
    keep = {}
    for c in s[0]:
        ks = set()
        for r in c:
            if r.id[0] != ' ':
                continue
            a = 'CB' if 'CB' in r else ('CA' if 'CA' in r else None)
            if a and r[a].get_bfactor() >= min_plddt:
                ks.add(r.id[1])
        keep[c.id] = ks | set((always or {}).get(c.id, ()))
    return keep


def _resi(positions):
    return '+'.join(str(int(r)) for r in sorted(positions))


def render_structure(pdb, out_png, *, scored_chains, scores, centre, vmin, vmax,
                     partners=None, sticks=None, contacts_from=None,
                     spheres=None, keep=None, view=None, size=(2400, 2200),
                     set_viewport=True):
    """Ray-trace one structure with PyMOL and return the PNG path.

    scored_chains : chains coloured per residue by `scores` (position -> score),
        blue-white-red between vmin and vmax, centred on `centre`; residues
        without a score take the centre colour.
    partners      : {PyMOL chain selection: colour} drawn as semi-transparent
        cartoon, e.g. {'A+B': PARTNER_COLOR['HSP90']}.
    sticks        : {chain: positions} shown as ball-and-stick.
    contacts_from : PyMOL chain selection whose residues within CONTACT_CUTOFF
        of the stick residues are also shown as ball-and-stick (the partner's
        side of the interface).
    spheres       : [(chain, positions, colour, scale)] as C-beta spheres (C-alpha
        for glycine), e.g. interface and control positions.
    keep          : {chain: residues}; everything else is removed (plddt_keep).
    view          : 18-float PyMOL view (get_view); None = automatic orient.
    set_viewport  : set the viewport to the render size before applying `view`,
        so the ray reproduces the framing chosen in a session of that size.
    A PyMOL session (.pse) is saved beside the PNG, for choosing a view.
    """
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    w, h = size
    cmds = [f'cmd.viewport({w}, {h})'] if set_viewport else []
    cmds += [f'cmd.set("max_threads", {PYMOL_MAX_THREADS})',
             f'cmd.load({str(pdb)!r}, "mol")']
    for ch, res in (keep or {}).items():
        cmds.append(f'cmd.remove("mol and chain {ch} and not resi {_resi(res)}")')
    cmds += ['cmd.hide("everything")', 'cmd.show("cartoon")', 'cmd.bg_color("white")',
             'cmd.set("ray_opaque_background", 0)', 'cmd.set("ray_shadows", 0)',
             'cmd.set("ray_interior_color", "grey80")', 'cmd.set("cartoon_gap_cutoff", 0)',
             'cmd.set("ambient", 0.4)', 'cmd.set("specular", 0.15)']
    for i, (sel, col) in enumerate((partners or {}).items()):
        cmds += [f'cmd.set_color("partner{i}", {list(mcolors.to_rgb(col))!r})',
                 f'cmd.color("partner{i}", "mol and chain {sel}")',
                 f'cmd.set("cartoon_transparency", {PARTNER_TRANSPARENCY}, "mol and chain {sel}")']
    scored = '+'.join(scored_chains)
    cmds.append(f'cmd.alter("mol and chain {scored}", "b={centre:.4f}")')
    cmds += [f'cmd.alter("mol and chain {scored} and resi {int(p)}", "b={v:.4f}")'
             for p, v in scores.items()]
    cmds += ['cmd.rebuild()',
             f'cmd.spectrum("b", "blue_white_red", "mol and chain {scored}", '
             f'minimum={vmin:.4f}, maximum={vmax:.4f})',
             'cmd.set("stick_radius", 0.14)', 'cmd.set("stick_ball", 1)',
             'cmd.set("stick_ball_ratio", 1.7)']
    stick_sel = [f'(mol and chain {ch} and resi {_resi(res)})'
                 for ch, res in (sticks or {}).items() if res]
    for s_ in stick_sel:
        cmds.append(f'cmd.show("sticks", "{s_}")')
    if contacts_from and stick_sel:
        cmds.append(f'cmd.show("sticks", "byres ((mol and chain {contacts_from}) '
                    f'within {CONTACT_CUTOFF} of ({" or ".join(stick_sel)}))")')
    cb = '(name CB or (resn GLY and name CA))'
    for i, (ch, res, col, scale) in enumerate(spheres or []):
        if not res:
            continue
        cmds += [f'cmd.set_color("sph{i}", {list(mcolors.to_rgb(col))!r})',
                 f'cmd.select("sph{i}", "mol and chain {ch} and resi {_resi(res)} and {cb}")',
                 f'cmd.show("spheres", "sph{i}")', f'cmd.set("sphere_scale", {scale}, "sph{i}")',
                 f'cmd.color("sph{i}", "sph{i}")']
    cmds.append(f'cmd.set_view({tuple(float(v) for v in view)!r})' if view
                else 'cmd.orient("mol")\ncmd.zoom("mol", 3)')
    cmds += ['cmd.deselect()', f'cmd.save({str(out_png.with_suffix(".pse"))!r})',
             f'cmd.ray({w}, {h})', f'cmd.png({str(out_png)!r}, dpi=200)']
    with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False) as fh:
        fh.write('from pymol import cmd\n' + '\n'.join(cmds) + '\n')
    try:
        subprocess.run([PYMOL, '-cq', fh.name], check=True, capture_output=True, text=True)
    finally:
        os.unlink(fh.name)
    return out_png


def _crop_render(png, pad=8):
    im = mpimg.imread(str(png))
    if im.shape[-1] == 4:
        m = im[..., 3] > 0.02
        ys, xs = np.where(m.any(axis=1))[0], np.where(m.any(axis=0))[0]
        im = im[max(ys[0] - pad, 0):ys[-1] + pad, max(xs[0] - pad, 0):xs[-1] + pad]
    return im


def legend_patch(color, label, alpha=1 - PARTNER_TRANSPARENCY):
    """Legend entry for a partner chain drawn as transparent cartoon."""
    return Patch(facecolor=color, alpha=alpha, edgecolor='none', label=label)


def legend_dot(color, label, edge='none', size=11):
    """Legend entry for C-beta spheres (or painted surface residues)."""
    return Line2D([0], [0], color='none', marker='o', markerfacecolor=color,
                  markeredgecolor=edge, markersize=size, label=label)


# Band layouts under the renders (inches from the bottom of the figure):
# colourbar position and size, legend row, caption. 'row' is for several renders
# side by side at a fixed height; the others fit a single render to the width.
PANEL_LAYOUTS = {
    'row':     dict(band=1.95, cbar=(0.40, 1.35, 0.20, 0.11), legend_y=0.42,
                    legend_size=11, legend_spacing=1.4, legend_pad=0.5, caption_size=10, title_size=15, title_pad=6),
    'single':  dict(band=2.9, cbar=(0.42, 2.05, 0.16, 0.16), legend_y=0.75,
                    legend_size=12.5, legend_spacing=2.0, legend_pad=0.5, caption_y=0.22, caption_size=10,
                    title_size=16, title_pad=8),
    'compact': dict(band=2.6, cbar=(0.42, 1.85, 0.16, 0.15), legend_y=0.70,
                    legend_size=12.5, legend_spacing=2.0, legend_pad=0.8, caption_y=0.22, caption_size=9,
                    title_size=16, title_pad=8),
}


def plot_structure_panel(pngs, titles, out, *, centre, vmin, vmax, cbar_label,
                         legend=(), caption='', layout='row', width=11.0, panel_h=4.4):
    """Lay out renders with a blue-white-red colourbar (vmin, WT = 1, vmax), a
    legend row and a caption, and save `out`.pdf/.png. layout='row' puts the
    renders side by side at `panel_h` inches; 'single' / 'compact' fit one
    render to `width` inches."""
    reset_style(publication=True)
    apply_arial()
    L = PANEL_LAYOUTS[layout]
    ims = [_crop_render(p) for p in pngs]
    asps = [im.shape[1] / im.shape[0] for im in ims]
    band = L['band']
    if layout == 'row':
        gap = 0.3
        widths = [panel_h * a for a in asps]
        fig_w = sum(widths) + gap * (len(ims) - 1) + 0.2
        fig_h = panel_h + band
        fig = plt.figure(figsize=(fig_w, fig_h))
        x = 0.1
        axes = []
        for w in widths:
            axes.append(fig.add_axes([x / fig_w, band / fig_h, w / fig_w, panel_h / fig_h]))
            x += w + gap
    else:
        fig_w = width
        img_h = fig_w * 0.96 / asps[0]
        fig_h = img_h + band + 0.5
        fig = plt.figure(figsize=(fig_w, fig_h))
        axes = [fig.add_axes([0.02, band / fig_h, 0.96, img_h / fig_h])]
    for ax, im, title in zip(axes, ims, titles):
        ax.imshow(im)
        ax.axis('off')
        ax.set_title(title, fontsize=L['title_size'], fontweight='bold', color='#1f2933',
                     pad=L['title_pad'])
    cx, cy, cw, ch = L['cbar']
    cax = fig.add_axes([cx, cy / fig_h, cw, ch / fig_h])
    cbar = ColorbarBase(cax, cmap=plt.cm.bwr, orientation='horizontal',
                        norm=TwoSlopeNorm(vmin=vmin, vcenter=centre, vmax=vmax))
    cbar.set_ticks([vmin, centre, vmax])
    cbar.set_ticklabels([f'{vmin:.1f}', '1', f'{vmax:.1f}'])
    cbar.set_label(cbar_label, fontsize=15, labelpad=6)
    cbar.ax.tick_params(labelsize=13)
    if legend:
        fig.legend(handles=list(legend), loc='lower center', ncol=len(legend),
                   frameon=False, fontsize=L['legend_size'],
                   bbox_to_anchor=(0.5, L['legend_y'] / fig_h),
                   handletextpad=L['legend_pad'], columnspacing=L['legend_spacing'])
    if caption:
        fig.text(0.5, L.get('caption_y', 0.14) / fig_h, caption, ha='center', va='center',
                 fontsize=L['caption_size'], color='#5b6770', style='italic')
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(out.with_suffix('.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    return out


def _shaded_sphere(ax, x, y, r, color, zorder=3):
    """A lit sphere (diffuse shading plus a soft highlight) of radius `r` at
    (x, y) in data units, so the key reads like the ray-traced spheres."""
    n = 160
    u = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(u, -u)
    R2 = X ** 2 + Y ** 2
    Z = np.sqrt(np.clip(1 - R2, 0, None))
    light = np.array([-0.45, 0.55, 0.70]); light /= np.linalg.norm(light)
    diffuse = np.clip(X * light[0] + Y * light[1] + Z * light[2], 0, 1)
    spec = diffuse ** 40
    base = np.array(mcolors.to_rgb(color))
    rgb = base * (0.45 + 0.55 * diffuse[..., None]) + 0.35 * spec[..., None]
    rgba = np.dstack([np.clip(rgb, 0, 1), (R2 <= 1).astype(float)])
    im = ax.imshow(rgba, extent=(x - r, x + r, y - r, y + r), zorder=zorder,
                   interpolation='bilinear')
    im.set_clip_path(Circle((x, y), r, transform=ax.transData))


def plot_structure_key(out, *, centre, vmin, vmax, cbar_label, spheres=(),
                       partners=(), sticks=None, width=2.4):
    """Stand-alone key for a structure panel, to place beside it: partner
    cartoons (semi-transparent swatches), C-beta sphere categories drawn as
    shaded spheres sized by their render scale, an optional ball-and-stick
    entry, and the blue-white-red score colourbar (vmin, WT = 1, vmax).

    spheres  : [(colour, label, scale)], scale as passed to render_structure.
    partners : [(colour, label)]; sticks: label for the ball-and-stick entry.
    """
    reset_style(publication=True)
    apply_arial()
    row, fs = 0.26, 9
    rows = [('partner', c, l, None) for c, l in partners]
    rows += [('sphere', c, l, sc) for c, l, sc in spheres]
    if sticks:
        rows.append(('sticks', '#6b7178', sticks, None))
    cbar_h = 0.8
    H = row * len(rows) + cbar_h + 0.1
    fig = plt.figure(figsize=(width, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width); ax.set_ylim(0, H); ax.axis('off')
    ax.set_aspect('equal')
    x_icon, x_text = 0.2, 0.42
    for k, (kind, col, label, sc) in enumerate(rows):
        y = H - 0.08 - row * (k + 0.5)
        if kind == 'partner':
            ax.add_patch(Rectangle((x_icon - 0.13, y - 0.06), 0.26, 0.12,
                                   facecolor=col, alpha=1 - PARTNER_TRANSPARENCY,
                                   edgecolor='none'))
        elif kind == 'sphere':
            _shaded_sphere(ax, x_icon, y, 0.1 * sc / 0.85, col)
        else:
            ax.plot([x_icon - 0.1, x_icon + 0.1], [y - 0.03, y + 0.03], color=col,
                    lw=1.4, solid_capstyle='round', zorder=2)
            for xx, yy in ((x_icon - 0.1, y - 0.03), (x_icon + 0.1, y + 0.03)):
                _shaded_sphere(ax, xx, yy, 0.028, col)
        ax.text(x_text, y, label, va='center', ha='left', fontsize=fs, color='#1f2933')
    # colourbar, with direction words at its ends
    cw, ch = 1.7, 0.11
    cx, cy = (width - cw) / 2, 0.44
    cax = fig.add_axes([cx / width, cy / H, cw / width, ch / H])
    cb = ColorbarBase(cax, cmap=plt.cm.bwr, orientation='horizontal',
                      norm=TwoSlopeNorm(vmin=vmin, vcenter=centre, vmax=vmax))
    cb.set_ticks([vmin, centre, vmax])
    cb.set_ticklabels([f'{vmin:.1f}', '1 (WT)', f'{vmax:.1f}'])
    cb.ax.tick_params(labelsize=fs - 1, length=2, pad=1.5)
    cb.outline.set_linewidth(0.5)
    ax.text(cx, cy + ch + 0.05, 'decreased', ha='left', va='bottom', fontsize=fs - 1.5,
            color='#2c6fa6', style='italic')
    ax.text(cx + cw, cy + ch + 0.05, 'increased', ha='right', va='bottom',
            fontsize=fs - 1.5, color='#c0392b', style='italic')
    ax.text(width / 2, 0.1, cbar_label, ha='center', va='center', fontsize=fs)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix('.pdf'), transparent=True)
    fig.savefig(out.with_suffix('.png'), dpi=300, transparent=True)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Surface renders: residues painted by interface family (ED9k)
# ---------------------------------------------------------------------------

# Base camera for the KRAS surface (native frame of its representative model);
# the three panels are y-rotations of it.
STRUCTURE_VIEWS['KRAS-surface'] = (
    0.403187692, -0.178249106, 0.897578597,
    0.069076188, 0.983964622, 0.164375156,
    -0.912489176, -0.004273857, 0.409039646,
    0.000000000, 0.000000000, -198.822296143,
    -14.682649612, -10.665683746, -1.372977376,
    159.520736694, 238.123840332, 20.000000000)

# Family colours (Paul Tol bright + vibrant), and a purple ramp for residues
# shared by several families, darkening with the number of families.
FAMILY_PALETTE = ['#4477AA', '#EE6677', '#228833', '#CCBB44',
                  '#66CCEE', '#AA3377', '#EE7733', '#882255', '#44AA99']
SHARED_TIERS = [(2, 2, '#9e9ac8', '2 families'),
                (3, 4, '#6a51a3', '3–4 families'),
                (5, 7, '#3f007d', '5–7 families')]
SURFACE_GREY = '#b8bcc2'


def shared_tier(n_families):
    """(colour, label) for a residue shared by `n_families` families."""
    for lo, hi, col, lab in SHARED_TIERS:
        if lo <= n_families <= hi:
            return col, lab
    return SHARED_TIERS[-1][2], SHARED_TIERS[-1][3]


def surface_residues(coords, pdb, chain, predictor, keep_always=(), min_plddt=70,
                     max_low_fraction=0.30):
    """Residues to draw as a clean surface. For a predicted model that is more
    than `max_low_fraction` low-confidence (disordered linkers), keep pLDDT >=
    `min_plddt`; otherwise drop residues further than 2.1x the median distance
    from the centroid (flexible termini). `keep_always` residues are kept."""
    keep_always = set(keep_always)
    if predictor in ('Predictomes', 'Zhang_et_al'):
        pl = {}
        for r in PDBParser(QUIET=True).get_structure('m', str(pdb))[0][chain]:
            a = 'CB' if 'CB' in r else ('CA' if 'CA' in r else None)
            if r.id[0] == ' ' and a:
                pl[r.id[1]] = r[a].get_bfactor()
        low = sum(1 for r in coords if pl.get(r, 100) < min_plddt) / max(len(coords), 1)
        if low > max_low_fraction:
            return {r for r in coords if pl.get(r, 0) >= min_plddt or r in keep_always}
    res = sorted(coords)
    P = np.array([coords[r] for r in res])
    d = np.linalg.norm(P - P.mean(axis=0), axis=1)
    return {r for r, di in zip(res, d) if di <= np.median(d) * 2.1} | (keep_always & set(coords))


def render_surface(pdb, chain, residues, colors, out_dir, stem, *, view,
                   yaws=(-90, 0, 90), size=1100, base=SURFACE_GREY, fov=20.0):
    """Ray-trace the molecular surface of `residues` on `chain`, painted by
    `colors` ({colour: residues}; everything else `base`), once per y-rotation
    of `view`. Returns the image paths; a .pse is saved for choosing a view."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    imgs = [out_dir / f'{stem}_yaw{y:+d}.png' for y in yaws]
    cmds = [f'cmd.set("max_threads", {PYMOL_MAX_THREADS})',
            f'cmd.load({str(pdb)!r}, "mol")',
            f'cmd.remove("mol and not chain {chain}")',
            f'cmd.remove("mol and not resi {_resi(residues)}")',
            'cmd.hide("everything", "mol")', 'cmd.show("surface", "mol")',
            f'cmd.set_color("base", {list(mcolors.to_rgb(base))!r})', 'cmd.color("base", "mol")']
    for i, (col, res) in enumerate(colors.items()):
        cmds += [f'cmd.set_color("c{i}", {list(mcolors.to_rgb(col))!r})',
                 f'cmd.color("c{i}", "mol and resi {_resi(res)}")']
    cmds += ['cmd.set("orthoscopic", 1)', f'cmd.set("field_of_view", {fov})',
             'cmd.set("ray_opaque_background", 0)', 'cmd.set("ray_shadows", 0)',
             'cmd.set("ambient", 0.45)', 'cmd.set("specular", 0.12)',
             'cmd.set("two_sided_lighting", 1)', 'cmd.set("surface_quality", 1)',
             f'cmd.set_view({tuple(view)!r})',
             f'cmd.save({str(out_dir.parent / (stem + ".pse"))!r})']
    for y, img in zip(yaws, imgs):
        cmds.append(f'cmd.set_view({tuple(view)!r})')
        if y:
            cmds.append(f'cmd.turn("y", {y})')
        cmds += [f'cmd.ray({size}, {size})', f'cmd.png({str(img)!r}, dpi=150)']
    with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False) as fh:
        fh.write('from pymol import cmd\n' + '\n'.join(cmds) + '\n')
    try:
        subprocess.run([PYMOL, '-cq', fh.name], check=True, capture_output=True, text=True)
    finally:
        os.unlink(fh.name)
    return imgs


def plot_surface_views(imgs, titles, out, *, legend=(), title='', caption=''):
    """Three (or more) surface views side by side, cropped to a common box so
    they share one scale, with a legend row, title and caption."""
    reset_style(publication=True)
    apply_arial()
    raw = [mpimg.imread(str(i)) for i in imgs]
    union = np.zeros(raw[0].shape[:2], bool)
    for im in raw:
        union |= (im[..., 3] > 0.02) if im.shape[-1] == 4 else np.ones_like(union)
    ys, xs = np.where(union.any(axis=1))[0], np.where(union.any(axis=0))[0]
    p = 8
    y0, y1 = max(ys[0] - p, 0), min(ys[-1] + p, union.shape[0])
    x0, x1 = max(xs[0] - p, 0), min(xs[-1] + p, union.shape[1])
    cropped = [im[y0:y1, x0:x1] for im in raw]
    aspect = (x1 - x0) / (y1 - y0)
    n = len(cropped)
    fig_h, top, bot, legr = 5.7, 0.90, 0.02, 0.20
    row_h = fig_h * (top - bot) / (1.0 + legr)
    fig_w = min(max(n * row_h * aspect / 0.99, 10.0), 16.5)
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(2, n, height_ratios=[1.0, legr], hspace=0.0, wspace=0.0,
                          left=0.005, right=0.995, top=top, bottom=bot)
    for j, (im, t_) in enumerate(zip(cropped, titles)):
        ax = fig.add_subplot(gs[0, j])
        ax.imshow(im)
        ax.axis('off')
        ax.set_title(t_, fontsize=15, fontweight='bold', color='#1f2933', pad=1)
    axl = fig.add_subplot(gs[1, :])
    axl.axis('off')
    if legend:
        axl.legend(handles=list(legend), loc='upper center', ncol=4, frameon=False,
                   fontsize=14, handletextpad=0.4, columnspacing=1.4, borderaxespad=0.0)
    if title:
        fig.suptitle(title, fontsize=12.5, fontweight='bold', y=0.995)
    if caption:
        fig.text(0.5, 0.055, caption, ha='center', va='center', fontsize=8,
                 color='#5b6770', style='italic')
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(out.with_suffix('.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    return out
