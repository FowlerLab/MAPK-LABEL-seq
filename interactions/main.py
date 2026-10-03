#!/usr/bin/env python3
"""
DMS Interface Interaction Analysis Pipeline

Analyzes whether predicted protein-protein interaction interfaces show distinct
LABEL-seq (DMS) score distributions compared to control surface residues.

Core Question:
    For a predicted A-B interaction where protein A has LABEL-seq data,
    do the residues at the interface (where A contacts B) show significantly
    different scores compared to control surface residues on A?

Usage:
    python main.py --config config.yaml
    python main.py --config config.yaml --protein P15056
    python main.py --config config.yaml --generate-sge --sge-output sge_scripts/
    python main.py --config config.yaml --aggregate --input-dirs output_*/
    python main.py --config config.yaml --dry-run
"""

import argparse
import csv
import json
import logging
import os
import re
import subprocess
import sys
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import yaml
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.metrics import roc_auc_score

from score_masterframe import (
    apply_gene_symbols,
    collapse_annotation_fanout,
    load_gene_symbols,
)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)

# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure logging for the pipeline."""
    logger = logging.getLogger("dms_interface")
    logger.setLevel(getattr(logging, level.upper()))

    # Console handler
    handler = logging.StreamHandler()
    handler.setLevel(getattr(logging, level.upper()))
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger

logger = setup_logging()


# =============================================================================
# UNIPROT GENE NAME LOOKUP
# =============================================================================

def query_uniprot_gene_names(uniprot_ids: List[str], batch_size: int = 100) -> Dict[str, str]:
    """
    Query UniProt REST API to get gene names for a list of UniProt IDs.

    Args:
        uniprot_ids: List of UniProt accession IDs
        batch_size: Number of IDs to query per request

    Returns:
        Dict mapping UniProt ID -> primary gene name
    """
    import urllib.request
    import urllib.parse
    import time

    if not uniprot_ids:
        return {}

    # Remove duplicates while preserving order
    unique_ids = list(dict.fromkeys(uniprot_ids))
    gene_names = {}

    logger.info(f"Querying UniProt for {len(unique_ids)} gene names...")

    # Process in batches
    for i in range(0, len(unique_ids), batch_size):
        batch = unique_ids[i:i + batch_size]

        # Build query for UniProt REST API
        query = " OR ".join([f"accession:{uid}" for uid in batch])
        params = {
            'query': query,
            'format': 'tsv',
            'fields': 'accession,gene_primary',
            'size': str(batch_size)  # Ensure we get all results for the batch
        }

        url = f"https://rest.uniprot.org/uniprotkb/search?{urllib.parse.urlencode(params)}"

        try:
            req = urllib.request.Request(url)
            req.add_header('User-Agent', 'Python DMS-Interface-Pipeline')

            with urllib.request.urlopen(req, timeout=30) as response:
                content = response.read().decode('utf-8')

                # Parse TSV response
                lines = content.strip().split('\n')
                if len(lines) > 1:  # Skip header
                    for line in lines[1:]:
                        parts = line.split('\t')
                        if len(parts) >= 2:
                            accession = parts[0]
                            gene = parts[1] if parts[1] else accession
                            gene_names[accession] = gene

            # Be nice to the API
            if i + batch_size < len(unique_ids):
                time.sleep(0.5)

        except Exception as e:
            logger.warning(f"Failed to query UniProt for batch {i//batch_size + 1}: {e}")
            # Fall back to using UniProt IDs as names for this batch
            for uid in batch:
                if uid not in gene_names:
                    gene_names[uid] = uid

    # Fill in any missing IDs with themselves
    for uid in unique_ids:
        if uid not in gene_names:
            gene_names[uid] = uid

    found_names = sum(1 for k, v in gene_names.items() if k != v)
    logger.info(f"Retrieved {found_names}/{len(unique_ids)} gene names from UniProt")

    return gene_names


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Config:
    """Pipeline configuration."""
    # Input files
    scores_file: Path
    scores_columns: Dict[str, str]
    scores_filters: Dict[str, List[str]]
    predictions: List[Dict[str, Any]]
    domains_info_file: Optional[Path]
    domains_colors_file: Optional[Path]

    # Parameters
    contact_distance: float
    surface_rsa: float
    control_spatial_distance: float
    control_spatial_distance_min: float  # Minimum C-beta distance from interface for controls; 0 = no minimum (pre-2026-02-26 behaviour)
    control_use_global_interface: bool   # If True, controls exclude residues near ANY known interface for this protein across all structures (requires first-pass pre-computation)
    control_within_domains: bool         # If True, restrict control pool to residues that fall within annotated protein domains (from domain_info.json)
    min_positions_per_group: int

    # Structure quality filters
    plddt_threshold: Optional[float]  # Filter residues below this pLDDT (B-factor in AlphaFold PDBs)
    min_residues_after_filter: int  # Skip structure if fewer residues pass pLDDT filter

    # Statistics settings
    statistics: Dict[str, Any]

    # Thresholds
    thresholds: Dict[str, float]

    # Output settings
    output_directory: Path
    generate_plots: bool
    generate_pymol: bool
    generate_heatmaps: bool

    # Parallelization
    n_cores: int

    # SGE settings
    sge: Dict[str, Any]

    # Paths
    pymol_path: str
    dssp_path: str

    # Optional JSON mapping {uniprot_accession: HGNC symbol}. When set, gene_name
    # is derived from uniprot_id rather than trusted from the scores file. The
    # score table labels proteins with library-style names (craf, mek1,
    # shp2); every downstream artefact (domain_info.json, prior-evidence tables,
    # structure filenames) is keyed on HGNC symbols, so we remap.
    scores_gene_symbol_map: Optional[Path] = None


@dataclass
class InteractionPair:
    """Represents a single analysis: one scored protein at one interface."""
    pdb_file: Path
    protein_a_uniprot: str      # Chain A in structure
    protein_b_uniprot: str      # Chain B in structure
    protein_a_name: str
    protein_b_name: str
    scored_chain: str           # 'A' or 'B' - which chain has LABEL-seq data
    scored_protein: str         # UniProt ID of scored protein
    scored_protein_name: str    # Gene name of scored protein
    interactor: str             # UniProt ID of partner
    interactor_name: str        # Gene name of partner
    predictor: str              # Source (e.g., "Zhang_et_al")
    skip_plddt_filter: bool = False  # True for experimental PDB structures


@dataclass
class ResidueGroups:
    """Residue classifications for an interaction."""
    interface_positions: List[int]
    control_positions: List[int]
    surface_positions: List[int]
    all_scored_positions: List[int]


@dataclass
class StatisticsResult:
    """Statistical comparison results for one assay."""
    # Counts
    n_interface_positions: int
    n_control_positions: int
    n_interface_variants: int
    n_control_variants: int

    # Descriptive statistics
    mean_interface: float
    mean_control: float
    mean_all: float
    std_interface: float
    std_control: float

    # Effect sizes
    log_fold_change: float = np.nan
    cohens_d: float = np.nan
    cliffs_delta: float = np.nan
    auroc: float = np.nan
    auroc_sign: int = 0  # 1 if interface > control, -1 otherwise
    wasserstein_dist: float = np.nan

    # Significance tests
    pval_mannwhitney: float = np.nan
    pval_mannwhitney_vs_all: float = np.nan
    pval_ks: float = np.nan
    pval_permutation: float = np.nan
    pval_fdr: float = np.nan

    # Confidence intervals
    bootstrap_mean_diff: float = np.nan
    bootstrap_ci_low: float = np.nan
    bootstrap_ci_high: float = np.nan


@dataclass
class InteractionResult:
    """Complete results for one interaction pair."""
    pair: InteractionPair
    residue_groups: ResidueGroups
    statistics_by_assay: Dict[str, StatisticsResult]
    skipped: bool = False
    skip_reason: str = ""


# =============================================================================
# CONFIGURATION LOADING
# =============================================================================

def load_config(config_path: Path) -> Config:
    """Load and validate configuration from YAML file."""
    logger.info(f"Loading configuration from {config_path}")

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Resolve paths relative to config file directory
    config_dir = config_path.parent

    def resolve_path(p: str) -> Path:
        """Resolve path relative to config directory."""
        path = Path(os.path.expandvars(p))
        if not path.is_absolute():
            path = config_dir / path
        return path

    # Parse structure filters
    structure_filters = cfg.get('structure_filters', {})
    plddt_threshold = structure_filters.get('plddt_threshold', None)
    min_residues_after_filter = structure_filters.get('min_residues_after_filter', 10)

    # Build Config object
    config = Config(
        # Input files
        scores_file=resolve_path(cfg['scores']['file']),
        scores_columns=cfg['scores']['columns'],
        scores_filters=cfg['scores'].get('filters', {}),
        scores_gene_symbol_map=(resolve_path(cfg['scores']['gene_symbol_map'])
                                if cfg['scores'].get('gene_symbol_map') else None),
        predictions=cfg['predictions'],
        domains_info_file=resolve_path(cfg['domains']['info_file']) if cfg.get('domains', {}).get('info_file') else None,
        domains_colors_file=resolve_path(cfg['domains']['colors_file']) if cfg.get('domains', {}).get('colors_file') else None,

        # Parameters
        contact_distance=cfg['parameters']['contact_distance'],
        surface_rsa=cfg['parameters']['surface_rsa'],
        control_spatial_distance=cfg['parameters']['control_spatial_distance'],
        # control_spatial_distance_min is optional; defaults to 0.0 (no minimum) so that
        # configs written before 2026-02-26 produce identical results without modification.
        control_spatial_distance_min=cfg['parameters'].get('control_spatial_distance_min', 0.0),
        # control_use_global_interface is optional; defaults to False (per-structure controls).
        # When True, a pre-computation pass aggregates interface positions across all structures
        # for each protein, and controls exclude any residue near any of those positions.
        control_use_global_interface=cfg['parameters'].get('control_use_global_interface', False),
        # control_within_domains is optional; defaults to False (all surface residues eligible).
        # When True, control candidates are restricted to positions within annotated protein
        # domains (from domain_info.json). Positions outside any domain are excluded from
        # the control pool even if they are surface-exposed.
        control_within_domains=cfg['parameters'].get('control_within_domains', False),
        min_positions_per_group=cfg['parameters']['min_positions_per_group'],

        # Structure quality filters
        plddt_threshold=plddt_threshold,
        min_residues_after_filter=min_residues_after_filter,

        # Statistics
        statistics=cfg['statistics'],

        # Thresholds
        thresholds=cfg['thresholds'],

        # Output
        output_directory=resolve_path(cfg['output']['directory']),
        generate_plots=cfg['output'].get('generate_plots', True),
        generate_pymol=cfg['output'].get('generate_pymol', True),
        generate_heatmaps=cfg['output'].get('generate_heatmaps', False),

        # Parallelization
        n_cores=cfg['parallel'].get('n_cores', 1),

        # SGE
        sge=cfg.get('sge', {}),

        # Paths
        pymol_path=os.path.expandvars(cfg['paths'].get('pymol', 'pymol')),
        dssp_path=os.path.expandvars(cfg['paths'].get('dssp', 'mkdssp')),
    )

    # Resolve prediction directories
    for pred in config.predictions:
        pred['directory'] = resolve_path(pred['directory'])
        if 'metadata' in pred:
            pred['metadata'] = resolve_path(pred['metadata'])

    logger.info(f"Configuration loaded: {len(config.predictions)} prediction source(s)")
    return config


# =============================================================================
# DATA LOADING
# =============================================================================

def load_scores(config: Config) -> pd.DataFrame:
    """Load and filter LABEL-seq scores."""
    logger.info(f"Loading scores from {config.scores_file}")

    df = pd.read_csv(config.scores_file, sep='\t', low_memory=False)
    logger.info(f"Loaded {len(df)} rows")

    # One row per measurement. An annotation join upstream can fan
    # measurements at multi-feature positions into duplicate rows; left alone
    # they get double weight in every statistic below.
    df = collapse_annotation_fanout(df, logger=logger)

    # Validate required columns exist
    cols = config.scores_columns
    required = ['variant', 'position', 'score', 'assay', 'uniprot_id']
    for col_key in required:
        col_name = cols.get(col_key)
        if col_name and col_name not in df.columns:
            raise ValueError(f"Required column '{col_name}' (mapped from '{col_key}') not found in scores file")

    # Apply filters
    if config.scores_filters:
        mutation_types = config.scores_filters.get('mutation_types', [])
        if mutation_types and cols.get('mutation_type'):
            mutation_col = cols['mutation_type']
            if mutation_col in df.columns:
                before = len(df)
                df = df[df[mutation_col].isin(mutation_types)]
                logger.info(f"Filtered to {mutation_types}: {before} -> {len(df)} rows")

        # Treatment condition filtering: select which assay_treatment values
        # to keep per library×assay. Supports a default allowlist and
        # per-library overrides for specific assays (e.g., EGFR activity
        # uses SerumStarve while everything else uses No_treatment/DMSO).
        treatment_filters = config.scores_filters.get('treatment_filters', {})
        treatment_col = cols.get('assay_treatment')
        library_col = cols.get('library')
        assay_col = cols.get('assay')
        if treatment_filters and treatment_col and treatment_col in df.columns:
            default_treatments = treatment_filters.get('default', [])
            overrides = treatment_filters.get('overrides', {})
            before = len(df)

            # Start with default: keep rows whose treatment is in the default list
            if default_treatments:
                mask = df[treatment_col].isin(default_treatments)
            else:
                # No default specified — keep all rows unless overridden
                mask = pd.Series(True, index=df.index)

            # Apply per-library×assay overrides: replace the default mask
            # for matching rows with the override's allowlist
            for lib, assay_overrides in overrides.items():
                for assay, treatments in assay_overrides.items():
                    lib_assay_rows = (df[library_col] == lib) & (df[assay_col] == assay)
                    # Clear default for these rows, apply override instead
                    mask = mask & ~lib_assay_rows
                    mask = mask | (lib_assay_rows & df[treatment_col].isin(treatments))

            df = df[mask]
            overrides_str = ", ".join(f"{k}: {v}" for k, v in overrides.items())
            logger.info(f"Treatment filter: {before} -> {len(df)} rows "
                        f"(default={default_treatments}, overrides={overrides_str})")

            # Additional conditions: load extra rows from alternative treatments
            # and append them with a renamed assay (e.g., abundance → abundance_HSP90i).
            # This lets the rest of the pipeline treat them as independent assays
            # without any downstream code changes.
            additional_conditions = treatment_filters.get('additional_conditions', [])
            for cond in additional_conditions:
                cond_name = cond['name']
                cond_treatments = cond['treatments']
                cond_assays = cond.get('assays', [])  # which assays to duplicate
                cond_suffix = cond.get('suffix', f"_{cond_name}")

                # Reload from the pre-mutation-filtered data: we need the full
                # DataFrame before treatment filtering was applied. Re-read and
                # apply the same mutation filter.
                df_extra = pd.read_csv(config.scores_file, sep='\t', low_memory=False)
                df_extra = collapse_annotation_fanout(df_extra, logger=logger)
                if mutation_types and cols.get('mutation_type'):
                    mutation_col = cols['mutation_type']
                    if mutation_col in df_extra.columns:
                        df_extra = df_extra[df_extra[mutation_col].isin(mutation_types)]

                # Filter to the specified treatments and assays
                extra_mask = df_extra[treatment_col].isin(cond_treatments)
                if cond_assays:
                    extra_mask = extra_mask & df_extra[assay_col].isin(cond_assays)
                df_extra = df_extra[extra_mask].copy()

                # Rename assay to include the condition suffix
                df_extra[assay_col] = df_extra[assay_col] + cond_suffix

                n_extra = len(df_extra)
                n_libs = df_extra[library_col].nunique()
                logger.info(f"Additional condition '{cond_name}': {n_extra} rows "
                            f"from {n_libs} libraries (suffix='{cond_suffix}')")

                df = pd.concat([df, df_extra], ignore_index=True)

    # Rename columns for internal use
    rename_map = {
        cols['variant']: 'variant',
        cols['position']: 'position',
        cols['score']: 'score',
        cols['assay']: 'assay',
        cols['uniprot_id']: 'uniprot_id',
    }
    if cols.get('gene_name') and cols['gene_name'] in df.columns:
        rename_map[cols['gene_name']] = 'gene_name'
    if cols.get('library') and cols['library'] in df.columns:
        rename_map[cols['library']] = 'library'

    df = df.drop(columns=[new for old, new in rename_map.items()
                          if old != new and new in df.columns and old in df.columns])
    df = df.rename(columns=rename_map)

    # Canonicalise gene_name from the accession when a symbol map is configured.
    if config.scores_gene_symbol_map:
        symbols = load_gene_symbols(config.scores_gene_symbol_map)
        before_names = sorted(df['gene_name'].unique()) if 'gene_name' in df.columns else []
        df = apply_gene_symbols(df, symbols, accession_col='uniprot_id')
        logger.info(f"Remapped gene_name from accession: {before_names} -> "
                    f"{sorted(df['gene_name'].unique())}")

    # Ensure position is integer
    df['position'] = df['position'].astype(int)

    # Drop rows with missing scores
    before = len(df)
    df = df.dropna(subset=['score'])
    if len(df) < before:
        logger.info(f"Dropped {before - len(df)} rows with missing scores")

    return df


def get_scored_proteins(scores_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Build mapping of proteins that have LABEL-seq data.

    Returns:
        Dict mapping uniprot_id -> {
            'gene_name': str,
            'assays': List[str],
            'positions': Set[int]
        }
    """
    proteins = {}

    for uniprot_id, group in scores_df.groupby('uniprot_id'):
        # Strip isoform suffix if present
        uniprot_id_clean = str(uniprot_id).split('-')[0]

        gene_name = group['gene_name'].iloc[0] if 'gene_name' in group.columns else uniprot_id_clean
        assays = group['assay'].unique().tolist()
        positions = set(group['position'].unique())

        proteins[uniprot_id_clean] = {
            'gene_name': gene_name,
            'assays': assays,
            'positions': positions
        }

    logger.info(f"Found {len(proteins)} proteins with LABEL-seq data")
    return proteins


def load_domain_info(config: Config) -> Tuple[Optional[Dict], Optional[Dict]]:
    """Load domain annotations if available."""
    domain_info = None
    domain_colors = None

    if config.domains_info_file and config.domains_info_file.exists():
        with open(config.domains_info_file) as f:
            domain_info = json.load(f)
        logger.info(f"Loaded domain info for {len(domain_info)} proteins")

    if config.domains_colors_file and config.domains_colors_file.exists():
        with open(config.domains_colors_file) as f:
            domain_colors = json.load(f)
        logger.info(f"Loaded {len(domain_colors)} domain colors")

    return domain_info, domain_colors


# =============================================================================
# PAIR DISCOVERY
# =============================================================================

def load_predictomes_metadata(metadata_path: Path) -> Dict[str, Tuple[str, str]]:
    """
    Load predictomes metadata mapping complex names to UniProt IDs.

    Args:
        metadata_path: Path to CSV with columns: complex_name, uniprot_ids, spoc_score

    Returns:
        Dict mapping complex_name -> (uniprot_a, uniprot_b)
    """
    metadata = {}
    with open(metadata_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            complex_name = row['complex_name']
            uniprot_ids = row['uniprot_ids'].split(':')
            if len(uniprot_ids) >= 2:
                metadata[complex_name] = (uniprot_ids[0], uniprot_ids[1])
    return metadata


def discover_interaction_pairs(
    config: Config,
    scored_proteins: Dict[str, Dict[str, Any]]
) -> List[InteractionPair]:
    """
    Discover all InteractionPair objects from prediction directories.

    A single PDB file can generate 0, 1, or 2 InteractionPair objects
    depending on which proteins have LABEL-seq data.

    Queries UniProt API to get gene names for interactor proteins that
    don't have LABEL-seq data.

    Supports two metadata formats:
    - Default: UniProt IDs extracted directly from filename via regex
    - Predictomes: complex_name extracted from filename, UniProt IDs looked up from metadata CSV
    """
    # First pass: collect all parsed pairs and identify UniProt IDs needing names
    parsed_pairs = []  # List of (pdb_file, uniprot_a, uniprot_b, pred_name, skip_plddt)
    unknown_uniprots = set()  # UniProt IDs not in scored_proteins

    for pred_source in config.predictions:
        pred_name = pred_source['name']
        pred_dir = pred_source['directory']
        metadata_format = pred_source.get('metadata_format', 'default')
        skip_plddt = pred_source.get('skip_plddt_filter', False)

        logger.info(f"Scanning predictions from {pred_name}: {pred_dir}")

        if not pred_dir.exists():
            logger.warning(f"Prediction directory not found: {pred_dir}")
            continue

        # Load metadata for predictomes format
        predictomes_metadata = None
        if metadata_format == 'predictomes' and 'metadata' in pred_source:
            metadata_path = pred_source['metadata']
            if metadata_path.exists():
                predictomes_metadata = load_predictomes_metadata(metadata_path)
                logger.info(f"Loaded {len(predictomes_metadata)} complexes from predictomes metadata")
            else:
                logger.warning(f"Predictomes metadata file not found: {metadata_path}")
                continue

        # Find all PDB files (handle nested directories)
        pdb_files = list(pred_dir.glob("**/*.pdb"))
        logger.info(f"Found {len(pdb_files)} PDB files in {pred_name}")

        # Get filename pattern (default: simple UniProtA_UniProtB format)
        filename_pattern = pred_source.get(
            'filename_pattern',
            r"(?P<uniprot_a>[A-Z0-9]+)_(?P<uniprot_b>[A-Z0-9]+).*"
        )
        pattern_re = re.compile(filename_pattern)

        for pdb_file in pdb_files:
            # Parse filename using configurable regex
            stem = pdb_file.stem
            match = pattern_re.match(stem)

            if not match:
                logger.debug(f"Skipping {pdb_file.name}: does not match pattern")
                continue

            # Get UniProt IDs based on metadata format
            if metadata_format == 'predictomes' and predictomes_metadata is not None:
                # Extract complex_name from filename and look up UniProt IDs
                try:
                    complex_name = match.group('complex_name')
                except IndexError:
                    logger.warning(f"Pattern missing complex_name group for predictomes format: {filename_pattern}")
                    continue

                if complex_name not in predictomes_metadata:
                    logger.debug(f"Skipping {pdb_file.name}: complex_name '{complex_name}' not in metadata")
                    continue

                uniprot_a, uniprot_b = predictomes_metadata[complex_name]
            else:
                # Default format: extract UniProt IDs directly from filename
                try:
                    uniprot_a = match.group('uniprot_a').split('-')[0]  # Strip isoform
                    uniprot_b = match.group('uniprot_b').split('-')[0]  # Strip isoform
                except IndexError:
                    logger.warning(f"Pattern missing uniprot_a or uniprot_b groups: {filename_pattern}")
                    continue

            # Track UniProt IDs not in scored_proteins
            if uniprot_a not in scored_proteins:
                unknown_uniprots.add(uniprot_a)
            if uniprot_b not in scored_proteins:
                unknown_uniprots.add(uniprot_b)

            # Only keep pairs where at least one protein has LABEL-seq data
            if uniprot_a in scored_proteins or uniprot_b in scored_proteins:
                parsed_pairs.append((pdb_file, uniprot_a, uniprot_b, pred_name, skip_plddt))

    # Query UniProt for gene names of unknown proteins
    uniprot_gene_names = {}
    if unknown_uniprots:
        uniprot_gene_names = query_uniprot_gene_names(list(unknown_uniprots))

    # Second pass: create InteractionPair objects with resolved names
    pairs = []
    for pdb_file, uniprot_a, uniprot_b, pred_name, skip_plddt in parsed_pairs:
        # Get gene names from scored_proteins or UniProt lookup
        if uniprot_a in scored_proteins:
            name_a = scored_proteins[uniprot_a].get('gene_name', uniprot_a)
        else:
            name_a = uniprot_gene_names.get(uniprot_a, uniprot_a)

        if uniprot_b in scored_proteins:
            name_b = scored_proteins[uniprot_b].get('gene_name', uniprot_b)
        else:
            name_b = uniprot_gene_names.get(uniprot_b, uniprot_b)

        # Check if A has LABEL-seq data
        if uniprot_a in scored_proteins:
            pairs.append(InteractionPair(
                pdb_file=pdb_file,
                protein_a_uniprot=uniprot_a,
                protein_b_uniprot=uniprot_b,
                protein_a_name=name_a,
                protein_b_name=name_b,
                scored_chain='A',
                scored_protein=uniprot_a,
                scored_protein_name=name_a,
                interactor=uniprot_b,
                interactor_name=name_b,
                predictor=pred_name,
                skip_plddt_filter=skip_plddt,
            ))

        # Check if B has LABEL-seq data (creates second analysis)
        if uniprot_b in scored_proteins:
            pairs.append(InteractionPair(
                pdb_file=pdb_file,
                protein_a_uniprot=uniprot_a,
                protein_b_uniprot=uniprot_b,
                protein_a_name=name_a,
                protein_b_name=name_b,
                scored_chain='B',
                scored_protein=uniprot_b,
                scored_protein_name=name_b,
                interactor=uniprot_a,
                interactor_name=name_a,
                predictor=pred_name,
                skip_plddt_filter=skip_plddt,
            ))

    logger.info(f"Discovered {len(pairs)} interaction pairs to analyze")
    return pairs


# =============================================================================
# STRUCTURE PROCESSING
# =============================================================================

def load_structure(pdb_path: Path):
    """Load PDB structure using Biopython."""
    from Bio.PDB import PDBParser

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('structure', pdb_path)
    return structure


def get_residue_coords(structure, chain_id: str) -> Dict[int, np.ndarray]:
    """
    Get C-beta coordinates for each residue (C-alpha for glycines).

    Returns:
        Dict mapping residue number -> 3D coordinates
    """
    coords = {}

    for model in structure:
        if chain_id in model:
            chain = model[chain_id]
            for residue in chain:
                # Skip hetero residues and water
                if residue.id[0] != ' ':
                    continue

                resnum = residue.id[1]

                # Use CB if available, otherwise CA (for glycines)
                if 'CB' in residue:
                    coords[resnum] = residue['CB'].get_coord()
                elif 'CA' in residue:
                    coords[resnum] = residue['CA'].get_coord()
        break  # Only first model

    return coords


def get_residue_plddt(structure, chain_id: str) -> Dict[int, float]:
    """
    Get pLDDT scores for each residue from B-factor column.

    AlphaFold stores per-residue pLDDT in the B-factor column (0-100 scale).

    Returns:
        Dict mapping residue number -> pLDDT score
    """
    plddt_scores = {}

    for model in structure:
        if chain_id in model:
            chain = model[chain_id]
            for residue in chain:
                # Skip hetero residues and water
                if residue.id[0] != ' ':
                    continue

                resnum = residue.id[1]

                # Get B-factor from CA atom (all atoms in residue should have same pLDDT)
                if 'CA' in residue:
                    plddt_scores[resnum] = residue['CA'].get_bfactor()
                elif 'CB' in residue:
                    plddt_scores[resnum] = residue['CB'].get_bfactor()
        break  # Only first model

    return plddt_scores


def filter_residues_by_plddt(
    structure,
    chain_id: str,
    plddt_threshold: float
) -> set:
    """
    Get residue numbers that pass the pLDDT threshold.

    If all B-factors in the chain are 0, pLDDT data is assumed to be absent
    (e.g., some RosettaFold structures don't store confidence scores).  In
    that case, ALL residues are returned (no filtering applied).

    Args:
        structure: Biopython structure object
        chain_id: Chain to filter
        plddt_threshold: Minimum pLDDT score (0-100)

    Returns:
        Set of residue numbers passing the threshold
    """
    plddt_scores = get_residue_plddt(structure, chain_id)

    # If all B-factors are 0, pLDDT data is absent — skip filtering
    if plddt_scores and max(plddt_scores.values()) == 0:
        return set(plddt_scores.keys())

    passing_residues = set()
    for resnum, plddt in plddt_scores.items():
        if plddt >= plddt_threshold:
            passing_residues.add(resnum)

    return passing_residues


def calculate_cbeta_distances(structure) -> pd.DataFrame:
    """
    Calculate C-beta distances between chains A and B.

    Returns:
        DataFrame with columns: res_a, res_b, distance
    """
    coords_a = get_residue_coords(structure, 'A')
    coords_b = get_residue_coords(structure, 'B')

    if not coords_a or not coords_b:
        return pd.DataFrame(columns=['res_a', 'res_b', 'distance'])

    # Build coordinate arrays
    res_a = list(coords_a.keys())
    res_b = list(coords_b.keys())
    arr_a = np.array([coords_a[r] for r in res_a])
    arr_b = np.array([coords_b[r] for r in res_b])

    # Calculate all pairwise distances
    dist_matrix = cdist(arr_a, arr_b)

    # Build DataFrame
    rows = []
    for i, ra in enumerate(res_a):
        for j, rb in enumerate(res_b):
            d = dist_matrix[i, j]
            # Only keep close contacts (efficiency)
            if d <= 15.0:  # Generous cutoff for downstream filtering
                rows.append({'res_a': ra, 'res_b': rb, 'distance': d})

    return pd.DataFrame(rows)


def calculate_intrachain_distances(structure, chain_id: str) -> Dict[Tuple[int, int], float]:
    """
    Calculate distances between residues within the same chain.
    Used for finding spatially matched control residues.

    Returns:
        Dict mapping (res1, res2) -> distance
    """
    coords = get_residue_coords(structure, chain_id)

    if not coords:
        return {}

    residues = list(coords.keys())
    distances = {}

    for i, r1 in enumerate(residues):
        for r2 in residues[i+1:]:
            d = np.linalg.norm(coords[r1] - coords[r2])
            distances[(r1, r2)] = d
            distances[(r2, r1)] = d

    return distances


def run_dssp(structure, pdb_path: Path, dssp_path: str = "mkdssp") -> Dict[int, Dict[str, Any]]:
    """
    Run DSSP to get secondary structure and solvent accessibility.

    Returns:
        Dict mapping (chain_id, resnum) -> {
            'ss': secondary structure,
            'asa': absolute solvent accessibility,
            'rsa': relative solvent accessibility
        }
    """
    from Bio.PDB.DSSP import DSSP

    try:
        model = structure[0]
        dssp = DSSP(model, pdb_path, dssp=dssp_path)

        results = {}
        for key in dssp.keys():
            chain_id, res_id = key
            resnum = res_id[1]

            dssp_data = dssp[key]
            ss = dssp_data[2]
            asa = dssp_data[3]  # Absolute ASA

            # Relative ASA (DSSP provides this normalized)
            # Values typically 0-1, but can exceed 1 for extended conformations
            rsa = asa  # DSSP already normalizes in recent versions

            results[(chain_id, resnum)] = {
                'ss': ss,
                'asa': asa,
                'rsa': rsa
            }

        return results

    except Exception as e:
        logger.warning(f"DSSP failed for {pdb_path}: {e}")
        return {}


def load_domain_positions(domain_info_file: Optional[Path]) -> Dict[str, Set[int]]:
    """
    Load per-protein domain position sets from domain_info.json.

    The JSON maps gene_name -> {position_str -> domain_name}.
    Positions with an empty domain_name string are outside any annotated domain.

    Returns:
        Dict mapping gene_name -> set of integer positions within any domain.
        Returns empty dict if file is None or cannot be read.
    """
    if domain_info_file is None or not domain_info_file.exists():
        return {}

    try:
        import json
        with open(domain_info_file) as f:
            raw = json.load(f)

        domain_map: Dict[str, Set[int]] = {}
        for gene, positions in raw.items():
            in_domain = {int(pos) for pos, label in positions.items() if label}
            if in_domain:
                domain_map[gene] = in_domain
                logger.debug(f"Domain positions loaded: {gene} → {len(in_domain)} positions")

        logger.info(f"Loaded domain positions for {len(domain_map)} proteins from {domain_info_file}")
        return domain_map

    except Exception as e:
        logger.warning(f"Could not load domain info from {domain_info_file}: {e}")
        return {}


def get_interface_positions(
    distances_df: pd.DataFrame,
    scored_chain: str,
    threshold: float
) -> List[int]:
    """
    Get interface positions on the scored chain.

    Interface = residues on scored chain within threshold distance of partner chain.
    """
    if distances_df.empty:
        return []

    if scored_chain == 'A':
        interface_df = distances_df[distances_df['distance'] <= threshold]
        positions = interface_df['res_a'].unique().tolist()
    else:  # Chain B
        interface_df = distances_df[distances_df['distance'] <= threshold]
        positions = interface_df['res_b'].unique().tolist()

    return sorted(positions)


def define_residue_groups(
    structure,
    distances_df: pd.DataFrame,
    dssp_results: Dict,
    intrachain_distances: Dict[Tuple[int, int], float],
    scored_chain: str,
    scored_positions: set,
    config: Config,
    plddt_passing_residues: Optional[set] = None,
    global_interface_positions: Optional[Set[int]] = None,
    domain_positions: Optional[Set[int]] = None,
) -> ResidueGroups:
    """
    Define interface, control, and surface residue groups.

    Interface: positions on scored chain <= contact_distance from partner.
    Surface:   positions with RSA > surface_rsa (config.surface_rsa).
    Control:   surface, non-interface, selected by one of two modes:

      Mode A — per-structure spatial window (control_use_global_interface=False):
        Nearest interface residue distance in
        [control_spatial_distance_min, control_spatial_distance].

      Mode B — global exclusion (control_use_global_interface=True):
        Not in global_interface_positions, nearest global interface distance
        >= control_spatial_distance_min. No outer distance cap.

    Domain filter (control_within_domains=True):
        Before control selection, the surface pool is intersected with
        domain_positions, restricting controls to annotated domain regions.
        Interface positions are NOT filtered by domain.

    Args:
        plddt_passing_residues:     Optional set of residues passing pLDDT.
        global_interface_positions: Union of interface positions across ALL
                                    structures for this protein.
        domain_positions:           Set of positions within annotated domains
                                    for this protein. When provided and
                                    config.control_within_domains=True, only
                                    these positions are eligible as controls.
    """
    # Get interface positions
    interface = set(get_interface_positions(
        distances_df, scored_chain, config.contact_distance
    ))

    # Get surface positions from DSSP
    surface = set()
    for (chain_id, resnum), data in dssp_results.items():
        if chain_id == scored_chain and data['rsa'] > config.surface_rsa:
            surface.add(resnum)

    # If DSSP failed, use all residues
    if not surface:
        coords = get_residue_coords(structure, scored_chain)
        surface = set(coords.keys())
        logger.warning(f"DSSP gave no surface residues, using all {len(surface)} residues")

    # Filter surface residues by pLDDT if threshold is set
    if plddt_passing_residues is not None:
        before_filter = len(surface)
        surface = surface & plddt_passing_residues
        if before_filter > len(surface):
            logger.debug(f"pLDDT filter: surface {before_filter} -> {len(surface)} residues")

    # Restrict control candidate pool to annotated domain positions.
    # This filter applies ONLY to the control pool — interface positions are
    # always defined by proximity to the partner chain regardless of domain.
    # If domain_positions is None (no domain info) or control_within_domains
    # is False, the full surface pool is used unchanged.
    control_surface = surface
    if config.control_within_domains and domain_positions is not None:
        before_domain = len(control_surface)
        control_surface = control_surface & domain_positions
        logger.debug(f"Domain filter: control surface {before_domain} -> {len(control_surface)} residues")

    # Select control residues.
    # Two modes, selected by config.control_use_global_interface:
    #
    # MODE A — per-structure spatial window (default, control_use_global_interface=False):
    #   Controls are surface, non-interface residues whose distance to the
    #   NEAREST interface residue (in THIS structure) falls within
    #   [control_spatial_distance_min, control_spatial_distance].
    #   control_spatial_distance_min=0 is identical to pre-2026-02-26 behaviour.
    #
    # MODE B — global exclusion (control_use_global_interface=True):
    #   Controls are surface residues that satisfy all of:
    #     (1) not in this structure's interface
    #     (2) not in global_interface_positions (not at ANY known interface
    #         for this protein across the entire dataset)
    #     (3) nearest intrachain distance to any global interface residue
    #         >= control_spatial_distance_min (exclusion zone)
    #   There is no outer distance cap; controls can be anywhere on the
    #   surface as long as they are sufficiently distal from all known interfaces.
    controls = set()

    if config.control_use_global_interface and global_interface_positions is not None:
        # Mode B: global exclusion
        for pos in control_surface:
            if pos in interface:
                continue
            if pos in global_interface_positions:
                continue  # This position is a known interface residue in some structure

            # Find distance to nearest global interface residue
            min_dist_to_global = float('inf')
            for g_pos in global_interface_positions:
                key = (pos, g_pos)
                if key in intrachain_distances:
                    d = intrachain_distances[key]
                    if d < min_dist_to_global:
                        min_dist_to_global = d

            if min_dist_to_global >= config.control_spatial_distance_min:
                controls.add(pos)
    else:
        # Mode A: per-structure spatial window
        for pos in control_surface:
            if pos in interface:
                continue

            # Find distance to nearest interface residue in this structure
            min_dist_to_interface = float('inf')
            for iface_pos in interface:
                key = (pos, iface_pos)
                if key in intrachain_distances:
                    d = intrachain_distances[key]
                    if d < min_dist_to_interface:
                        min_dist_to_interface = d

            if (min_dist_to_interface <= config.control_spatial_distance and
                    min_dist_to_interface >= config.control_spatial_distance_min):
                controls.add(pos)

    # Intersect with scored positions
    all_scored = list(scored_positions)
    interface_scored = sorted(interface & scored_positions)
    control_scored = sorted(controls & scored_positions)
    surface_scored = sorted(surface & scored_positions)

    return ResidueGroups(
        interface_positions=interface_scored,
        control_positions=control_scored,
        surface_positions=surface_scored,
        all_scored_positions=all_scored
    )


def check_data_sufficiency(
    groups: ResidueGroups,
    min_n: int
) -> Tuple[bool, str]:
    """Check if there are enough positions in each group."""
    if len(groups.interface_positions) < min_n:
        return False, f"Only {len(groups.interface_positions)} interface positions (need {min_n})"

    if len(groups.control_positions) < min_n:
        return False, f"Only {len(groups.control_positions)} control positions (need {min_n})"

    return True, ""


# =============================================================================
# STATISTICS
# =============================================================================

def compute_mannwhitney(interface_scores: np.ndarray, control_scores: np.ndarray) -> float:
    """Compute Mann-Whitney U test p-value (two-sided)."""
    if len(interface_scores) < 2 or len(control_scores) < 2:
        return np.nan

    try:
        _, pval = stats.mannwhitneyu(interface_scores, control_scores, alternative='two-sided')
        return pval
    except:
        return np.nan


def compute_ks_test(interface_scores: np.ndarray, control_scores: np.ndarray) -> float:
    """Compute Kolmogorov-Smirnov test p-value."""
    if len(interface_scores) < 2 or len(control_scores) < 2:
        return np.nan

    try:
        _, pval = stats.ks_2samp(interface_scores, control_scores)
        return pval
    except:
        return np.nan


def compute_permutation_pvalue(
    interface_scores: np.ndarray,
    control_scores: np.ndarray,
    n_permutations: int = 1000
) -> float:
    """Compute permutation test p-value for difference in means."""
    if len(interface_scores) < 2 or len(control_scores) < 2:
        return np.nan

    observed_diff = np.mean(interface_scores) - np.mean(control_scores)
    combined = np.concatenate([interface_scores, control_scores])
    n_interface = len(interface_scores)

    count = 0
    for _ in range(n_permutations):
        np.random.shuffle(combined)
        perm_diff = np.mean(combined[:n_interface]) - np.mean(combined[n_interface:])
        if abs(perm_diff) >= abs(observed_diff):
            count += 1

    return (count + 1) / (n_permutations + 1)


def compute_cohens_d(interface_scores: np.ndarray, control_scores: np.ndarray) -> float:
    """Compute Cohen's d effect size."""
    if len(interface_scores) < 2 or len(control_scores) < 2:
        return np.nan

    n1, n2 = len(interface_scores), len(control_scores)
    var1, var2 = np.var(interface_scores, ddof=1), np.var(control_scores, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return np.nan

    return (np.mean(interface_scores) - np.mean(control_scores)) / pooled_std


def compute_cliffs_delta(interface_scores: np.ndarray, control_scores: np.ndarray) -> float:
    """Compute Cliff's delta effect size."""
    if len(interface_scores) == 0 or len(control_scores) == 0:
        return np.nan

    # Count pairs where interface > control, interface < control
    n_greater = 0
    n_less = 0

    for i in interface_scores:
        for c in control_scores:
            if i > c:
                n_greater += 1
            elif i < c:
                n_less += 1

    total = len(interface_scores) * len(control_scores)
    if total == 0:
        return np.nan

    return (n_greater - n_less) / total


def compute_auroc(
    interface_scores: np.ndarray,
    control_scores: np.ndarray
) -> Tuple[float, int]:
    """
    Compute AUROC for classifying interface vs control.

    Returns:
        (auroc, sign) where sign indicates direction (1 if interface > control)
    """
    if len(interface_scores) < 2 or len(control_scores) < 2:
        return np.nan, 0

    # Labels: 1 for interface, 0 for control
    y_true = np.concatenate([
        np.ones(len(interface_scores)),
        np.zeros(len(control_scores))
    ])
    y_scores = np.concatenate([interface_scores, control_scores])

    try:
        auroc = roc_auc_score(y_true, y_scores)
        # Determine direction
        sign = 1 if np.mean(interface_scores) > np.mean(control_scores) else -1
        return auroc, sign
    except:
        return np.nan, 0


def compute_wasserstein(interface_scores: np.ndarray, control_scores: np.ndarray) -> float:
    """Compute Wasserstein (Earth Mover's) distance."""
    if len(interface_scores) == 0 or len(control_scores) == 0:
        return np.nan

    try:
        return stats.wasserstein_distance(interface_scores, control_scores)
    except:
        return np.nan


def compute_log_fold_change(interface_scores: np.ndarray, control_scores: np.ndarray) -> float:
    """Compute log2 fold change of means."""
    mean_iface = np.mean(interface_scores)
    mean_ctrl = np.mean(control_scores)

    # Handle edge cases
    if mean_ctrl == 0 or mean_iface == 0:
        return np.nan
    if mean_ctrl < 0 or mean_iface < 0:
        # Can't take log of negative, use difference instead
        return np.nan

    return np.log2(mean_iface / mean_ctrl)


def compute_bootstrap_ci(
    interface_scores: np.ndarray,
    control_scores: np.ndarray,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for difference in means.

    Returns:
        (mean_diff, ci_low, ci_high)
    """
    if len(interface_scores) < 2 or len(control_scores) < 2:
        return np.nan, np.nan, np.nan

    diffs = []
    for _ in range(n_bootstrap):
        iface_sample = np.random.choice(interface_scores, size=len(interface_scores), replace=True)
        ctrl_sample = np.random.choice(control_scores, size=len(control_scores), replace=True)
        diffs.append(np.mean(iface_sample) - np.mean(ctrl_sample))

    diffs = np.array(diffs)
    alpha = 1 - ci_level
    ci_low = np.percentile(diffs, 100 * alpha / 2)
    ci_high = np.percentile(diffs, 100 * (1 - alpha / 2))
    mean_diff = np.mean(diffs)

    return mean_diff, ci_low, ci_high


def compute_all_statistics(
    interface_scores: np.ndarray,
    control_scores: np.ndarray,
    all_scores: np.ndarray,
    config: Config
) -> StatisticsResult:
    """Compute all configured statistics."""
    stats_cfg = config.statistics

    # Basic descriptive stats
    result = StatisticsResult(
        n_interface_positions=0,  # Set by caller
        n_control_positions=0,    # Set by caller
        n_interface_variants=len(interface_scores),
        n_control_variants=len(control_scores),
        mean_interface=np.mean(interface_scores) if len(interface_scores) > 0 else np.nan,
        mean_control=np.mean(control_scores) if len(control_scores) > 0 else np.nan,
        mean_all=np.mean(all_scores) if len(all_scores) > 0 else np.nan,
        std_interface=np.std(interface_scores, ddof=1) if len(interface_scores) > 1 else np.nan,
        std_control=np.std(control_scores, ddof=1) if len(control_scores) > 1 else np.nan,
    )

    # Log fold change
    if stats_cfg.get('log_fold_change', True):
        result.log_fold_change = compute_log_fold_change(interface_scores, control_scores)

    # Mann-Whitney U
    if stats_cfg.get('mann_whitney', True):
        result.pval_mannwhitney = compute_mannwhitney(interface_scores, control_scores)
        result.pval_mannwhitney_vs_all = compute_mannwhitney(interface_scores, all_scores)

    # KS test
    if stats_cfg.get('ks_test', True):
        result.pval_ks = compute_ks_test(interface_scores, control_scores)

    # Permutation test
    if stats_cfg.get('permutation_test', True):
        n_perm = stats_cfg.get('permutation_n', 1000)
        result.pval_permutation = compute_permutation_pvalue(interface_scores, control_scores, n_perm)

    # Cohen's d
    if stats_cfg.get('cohens_d', True):
        result.cohens_d = compute_cohens_d(interface_scores, control_scores)

    # Cliff's delta
    if stats_cfg.get('cliffs_delta', True):
        result.cliffs_delta = compute_cliffs_delta(interface_scores, control_scores)

    # AUROC
    if stats_cfg.get('auroc', True):
        result.auroc, result.auroc_sign = compute_auroc(interface_scores, control_scores)

    # Wasserstein distance
    if stats_cfg.get('wasserstein', True):
        result.wasserstein_dist = compute_wasserstein(interface_scores, control_scores)

    # Bootstrap CI
    if stats_cfg.get('bootstrap_ci', True):
        n_boot = stats_cfg.get('bootstrap_n', 1000)
        ci_level = stats_cfg.get('bootstrap_ci_level', 0.95)
        result.bootstrap_mean_diff, result.bootstrap_ci_low, result.bootstrap_ci_high = \
            compute_bootstrap_ci(interface_scores, control_scores, n_boot, ci_level)

    return result


def apply_fdr_correction(results: List[InteractionResult]) -> List[InteractionResult]:
    """Apply Benjamini-Hochberg FDR correction across all p-values."""
    from statsmodels.stats.multitest import multipletests

    # Collect all p-values with their locations
    pvals = []
    locations = []  # (result_idx, assay, pval_type)

    for i, result in enumerate(results):
        if result.skipped:
            continue
        for assay, stats in result.statistics_by_assay.items():
            if not np.isnan(stats.pval_mannwhitney):
                pvals.append(stats.pval_mannwhitney)
                locations.append((i, assay, 'mannwhitney'))

    if not pvals:
        return results

    # Apply FDR correction
    _, pvals_corrected, _, _ = multipletests(pvals, method='fdr_bh')

    # Update results
    for (i, assay, pval_type), corrected_pval in zip(locations, pvals_corrected):
        results[i].statistics_by_assay[assay].pval_fdr = corrected_pval

    logger.info(f"Applied FDR correction to {len(pvals)} p-values")
    return results


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_boxplot(
    interface_scores: np.ndarray,
    control_scores: np.ndarray,
    all_scores: np.ndarray,
    pair: InteractionPair,
    assay: str,
    stats_result: StatisticsResult,
    output_path: Path
):
    """Generate boxplot comparing interface vs control scores."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(8, 6))

    # Prepare data
    data = []
    for score in interface_scores:
        data.append({'Group': 'Interface', 'Score': score})
    for score in control_scores:
        data.append({'Group': 'Control', 'Score': score})
    for score in all_scores:
        data.append({'Group': 'All', 'Score': score})

    df = pd.DataFrame(data)

    # Plot
    palette = {'Interface': '#e74c3c', 'Control': '#3498db', 'All': '#95a5a6'}
    sns.boxplot(data=df, x='Group', y='Score', palette=palette, ax=ax)
    sns.stripplot(data=df, x='Group', y='Score', color='black', alpha=0.5, size=3, ax=ax)

    # Title and labels
    title = f"{pair.scored_protein_name} - {pair.interactor_name} ({assay})"
    subtitle = f"p={stats_result.pval_mannwhitney:.2e}, Cliff's d={stats_result.cliffs_delta:.3f}"
    ax.set_title(f"{title}\n{subtitle}")
    ax.set_ylabel("Score")
    ax.set_xlabel("")

    # Add sample sizes
    for i, (group, n) in enumerate([
        ('Interface', len(interface_scores)),
        ('Control', len(control_scores)),
        ('All', len(all_scores))
    ]):
        ax.annotate(f'n={n}', xy=(i, ax.get_ylim()[0]),
                   ha='center', va='top', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def generate_pymol_session(
    pair: InteractionPair,
    groups: ResidueGroups,
    output_path: Path,
    pymol_path: str
):
    """Generate PyMOL session highlighting interface residues."""
    # Create PyMOL script
    pml_script = output_path.with_suffix('.pml')

    interface_sel = '+'.join(str(p) for p in groups.interface_positions) if groups.interface_positions else 'none'
    control_sel = '+'.join(str(p) for p in groups.control_positions) if groups.control_positions else 'none'

    script_content = f"""
# PyMOL script for {pair.scored_protein_name}-{pair.interactor_name}
# Generated by DMS Interface Analysis Pipeline

# Load structure
load {pair.pdb_file}, complex

# Basic display
hide everything
show cartoon, complex
color gray80, complex

# Color chains
color palegreen, chain A
color lightblue, chain B

# Highlight interface residues (red)
select interface, chain {pair.scored_chain} and resi {interface_sel}
show sticks, interface
color red, interface

# Highlight control residues (blue)
select controls, chain {pair.scored_chain} and resi {control_sel}
show sticks, controls
color blue, controls

# Labels
set label_size, 12
set label_color, black

# Orient view
orient complex
zoom complex, 5

# Save session
save {output_path}
"""

    with open(pml_script, 'w') as f:
        f.write(script_content)

    # Run PyMOL headless
    try:
        cmd = f"{pymol_path} -c -q {pml_script}"
        subprocess.run(cmd, shell=True, check=True, capture_output=True, timeout=60)
        pml_script.unlink()  # Clean up script
    except Exception as e:
        logger.warning(f"PyMOL session generation failed: {e}")


# =============================================================================
# OUTPUT GENERATION
# =============================================================================

def results_to_dataframe(results: List[InteractionResult]) -> pd.DataFrame:
    """Convert results to a flat DataFrame."""
    rows = []

    for result in results:
        if result.skipped:
            # Include skipped pairs with minimal info
            rows.append({
                'scored_protein': result.pair.scored_protein,
                'scored_protein_name': result.pair.scored_protein_name,
                'interactor': result.pair.interactor,
                'interactor_name': result.pair.interactor_name,
                'scored_chain': result.pair.scored_chain,
                'predictor': result.pair.predictor,
                'structure': result.pair.pdb_file.name,
                'skipped': True,
                'skip_reason': result.skip_reason
            })
            continue

        pair = result.pair
        groups = result.residue_groups

        for assay, stats in result.statistics_by_assay.items():
            row = {
                # Identifiers
                'scored_protein': pair.scored_protein,
                'scored_protein_name': pair.scored_protein_name,
                'interactor': pair.interactor,
                'interactor_name': pair.interactor_name,
                'scored_chain': pair.scored_chain,
                'predictor': pair.predictor,
                'structure': pair.pdb_file.name,
                'assay': assay,
                'skipped': False,
                'skip_reason': '',

                # Counts
                'n_interface_positions': stats.n_interface_positions,
                'n_control_positions': stats.n_control_positions,
                'n_interface_variants': stats.n_interface_variants,
                'n_control_variants': stats.n_control_variants,
                'interface_positions': ','.join(map(str, groups.interface_positions)),
                'control_positions': ','.join(map(str, groups.control_positions)),

                # Descriptive stats
                'mean_interface': stats.mean_interface,
                'mean_control': stats.mean_control,
                'mean_all': stats.mean_all,
                'std_interface': stats.std_interface,
                'std_control': stats.std_control,
                'log_fold_change': stats.log_fold_change,

                # Significance tests
                'pval_mannwhitney': stats.pval_mannwhitney,
                'pval_mannwhitney_vs_all': stats.pval_mannwhitney_vs_all,
                'pval_ks': stats.pval_ks,
                'pval_permutation': stats.pval_permutation,
                'pval_fdr': stats.pval_fdr,

                # Effect sizes
                'cohens_d': stats.cohens_d,
                'cliffs_delta': stats.cliffs_delta,
                'auroc': stats.auroc,
                'auroc_sign': stats.auroc_sign,
                'wasserstein_dist': stats.wasserstein_dist,

                # Bootstrap CI
                'bootstrap_mean_diff': stats.bootstrap_mean_diff,
                'bootstrap_ci_low': stats.bootstrap_ci_low,
                'bootstrap_ci_high': stats.bootstrap_ci_high,
            }
            rows.append(row)

    return pd.DataFrame(rows)


def filter_significant_results(df: pd.DataFrame, thresholds: Dict[str, float]) -> pd.DataFrame:
    """Filter results by significance thresholds."""
    if df.empty:
        return df

    # Start with non-skipped results
    significant = df[~df['skipped']].copy()

    # Apply thresholds
    if 'pvalue' in thresholds and 'pval_mannwhitney' in significant.columns:
        significant = significant[significant['pval_mannwhitney'] <= thresholds['pvalue']]

    if 'fdr_pvalue' in thresholds and 'pval_fdr' in significant.columns:
        significant = significant[significant['pval_fdr'] <= thresholds['fdr_pvalue']]

    if 'auroc' in thresholds and 'auroc' in significant.columns:
        # Handle bidirectional effects: AUROC < 0.5 means interface < control
        # Use max(auroc, 1-auroc) to capture discriminative power in either direction
        auroc_power = significant['auroc'].apply(lambda x: max(x, 1-x) if pd.notna(x) else np.nan)
        significant = significant[auroc_power >= thresholds['auroc']]

    if 'min_effect_size' in thresholds and 'cliffs_delta' in significant.columns:
        significant = significant[significant['cliffs_delta'].abs() >= thresholds['min_effect_size']]

    return significant


def results_to_json(results: List[InteractionResult]) -> Dict:
    """Convert results to nested JSON structure."""
    output = {
        'metadata': {
            'n_total_pairs': len(results),
            'n_analyzed': sum(1 for r in results if not r.skipped),
            'n_skipped': sum(1 for r in results if r.skipped)
        },
        'results': []
    }

    for result in results:
        if result.skipped:
            output['results'].append({
                'pair': {
                    'scored_protein': result.pair.scored_protein,
                    'interactor': result.pair.interactor,
                    'predictor': result.pair.predictor
                },
                'skipped': True,
                'skip_reason': result.skip_reason
            })
            continue

        pair_data = {
            'pair': {
                'scored_protein': result.pair.scored_protein,
                'scored_protein_name': result.pair.scored_protein_name,
                'interactor': result.pair.interactor,
                'interactor_name': result.pair.interactor_name,
                'scored_chain': result.pair.scored_chain,
                'predictor': result.pair.predictor,
                'structure': result.pair.pdb_file.name
            },
            'residue_groups': {
                'interface_positions': result.residue_groups.interface_positions,
                'control_positions': result.residue_groups.control_positions,
                'n_interface': len(result.residue_groups.interface_positions),
                'n_control': len(result.residue_groups.control_positions)
            },
            'statistics_by_assay': {}
        }

        for assay, stats in result.statistics_by_assay.items():
            pair_data['statistics_by_assay'][assay] = asdict(stats)

        output['results'].append(pair_data)

    return output


def write_outputs(
    results: List[InteractionResult],
    config: Config,
    scores_df: pd.DataFrame
):
    """Write all output files."""
    output_dir = config.output_directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert to DataFrame
    df = results_to_dataframe(results)

    # Write full results
    all_csv = output_dir / "all_interactions.csv"
    df.to_csv(all_csv, index=False)
    logger.info(f"Wrote {len(df)} rows to {all_csv}")

    # Write significant results
    sig_df = filter_significant_results(df, config.thresholds)
    sig_csv = output_dir / "significant_interactions.csv"
    sig_df.to_csv(sig_csv, index=False)
    logger.info(f"Wrote {len(sig_df)} significant results to {sig_csv}")

    # Write JSON summary
    json_output = results_to_json(results)
    json_path = output_dir / "summary.json"
    with open(json_path, 'w') as f:
        json.dump(json_output, f, indent=2, default=str)
    logger.info(f"Wrote JSON summary to {json_path}")

    # Generate plots
    if config.generate_plots:
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)

        for result in results:
            if result.skipped:
                continue

            for assay, stats in result.statistics_by_assay.items():
                # Get scores for plotting
                pair = result.pair
                groups = result.residue_groups

                protein_scores = scores_df[
                    (scores_df['uniprot_id'].str.split('-').str[0] == pair.scored_protein) &
                    (scores_df['assay'] == assay)
                ]

                interface_scores = protein_scores[
                    protein_scores['position'].isin(groups.interface_positions)
                ]['score'].values

                control_scores = protein_scores[
                    protein_scores['position'].isin(groups.control_positions)
                ]['score'].values

                all_scores = protein_scores['score'].values

                # Generate plot
                plot_name = f"{pair.scored_protein_name}_{pair.interactor_name}_{assay}_{pair.predictor}.boxplot.pdf"
                plot_path = plots_dir / plot_name

                try:
                    plot_boxplot(
                        interface_scores, control_scores, all_scores,
                        pair, assay, stats, plot_path
                    )
                except Exception as e:
                    logger.warning(f"Failed to generate plot {plot_name}: {e}")

        logger.info(f"Generated plots in {plots_dir}")

    # Generate PyMOL sessions
    if config.generate_pymol:
        pymol_dir = output_dir / "pymol_sessions"
        pymol_dir.mkdir(exist_ok=True)

        # One session per structure (not per assay)
        generated = set()
        for result in results:
            if result.skipped:
                continue

            pair = result.pair
            key = (pair.pdb_file, pair.scored_chain)
            if key in generated:
                continue
            generated.add(key)

            # Include PDB file stem to differentiate between multiple structures (e.g., different ranks)
            session_name = f"{pair.pdb_file.stem}_{pair.scored_chain}.pse"
            session_path = pymol_dir / session_name

            try:
                generate_pymol_session(pair, result.residue_groups, session_path, config.pymol_path)
            except Exception as e:
                logger.warning(f"Failed to generate PyMOL session {session_name}: {e}")

        logger.info(f"Generated PyMOL sessions in {pymol_dir}")


# =============================================================================
# SGE SCRIPT GENERATION
# =============================================================================

def generate_sge_scripts(
    config: Config,
    proteins: List[str],
    output_dir: Path,
    config_path: Path
):
    """Generate SGE job scripts for parallel execution."""
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(exist_ok=True)

    sge_cfg = config.sge
    memory = sge_cfg.get('memory_per_core', '10G')
    n_cores = sge_cfg.get('n_cores', 10)
    runtime = sge_cfg.get('runtime', '4:00:00')
    project = sge_cfg.get('project')

    job_scripts = []

    for protein in proteins:
        script_name = f"run_{protein}.sh"
        script_path = output_dir / script_name

        project_line = f"#$ -P {project}" if project else "# No project specified"

        script_content = f"""#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -l mfree={memory}
#$ -pe serial {n_cores}
#$ -l h_rt={runtime}
#$ -o {logs_dir}/{protein}.out
#$ -e {logs_dir}/{protein}.err
{project_line}

# DMS Interface Analysis - {protein}

python {Path(__file__).resolve()} \\
    --config {config_path.resolve()} \\
    --protein {protein} \\
    --output output_{protein}/
"""

        with open(script_path, 'w') as f:
            f.write(script_content)

        script_path.chmod(0o755)
        job_scripts.append(script_path)

    # Master submission script
    submit_script = output_dir / "submit_all.sh"
    with open(submit_script, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("# Submit all DMS interface analysis jobs\n\n")
        for script in job_scripts:
            f.write(f"qsub {script.name}\n")
        f.write(f"\necho 'Submitted {len(job_scripts)} jobs'\n")

    submit_script.chmod(0o755)

    logger.info(f"Generated {len(job_scripts)} SGE scripts in {output_dir}")
    logger.info(f"Submit with: {submit_script}")


def aggregate_parallel_results(
    input_dirs: List[Path],
    output_dir: Path,
    config: Config
):
    """Aggregate results from parallel SGE jobs."""
    output_dir.mkdir(parents=True, exist_ok=True)

    all_dfs = []
    all_json_results = []

    for dir_path in input_dirs:
        csv_path = dir_path / "all_interactions.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            all_dfs.append(df)
            logger.info(f"Loaded {len(df)} rows from {csv_path}")

        json_path = dir_path / "summary.json"
        if json_path.exists():
            with open(json_path) as f:
                data = json.load(f)
                all_json_results.extend(data.get('results', []))

    if not all_dfs:
        logger.warning("No results found to aggregate")
        return

    # Combine DataFrames
    combined_df = pd.concat(all_dfs, ignore_index=True)

    # Apply FDR correction to combined results
    if config.statistics.get('fdr_correction', True):
        pvals = combined_df['pval_mannwhitney'].dropna().values
        if len(pvals) > 0:
            from statsmodels.stats.multitest import multipletests
            mask = ~combined_df['pval_mannwhitney'].isna()
            _, corrected, _, _ = multipletests(
                combined_df.loc[mask, 'pval_mannwhitney'].values,
                method='fdr_bh'
            )
            combined_df.loc[mask, 'pval_fdr'] = corrected

    # Write combined results
    combined_df.to_csv(output_dir / "all_interactions.csv", index=False)
    logger.info(f"Wrote {len(combined_df)} combined rows")

    # Filter significant
    sig_df = filter_significant_results(combined_df, config.thresholds)
    sig_df.to_csv(output_dir / "significant_interactions.csv", index=False)
    logger.info(f"Wrote {len(sig_df)} significant results")

    # Combined JSON
    json_output = {
        'metadata': {
            'n_total_pairs': len(all_json_results),
            'n_analyzed': sum(1 for r in all_json_results if not r.get('skipped', False)),
            'n_skipped': sum(1 for r in all_json_results if r.get('skipped', False)),
            'aggregated_from': [str(d) for d in input_dirs]
        },
        'results': all_json_results
    }

    with open(output_dir / "summary.json", 'w') as f:
        json.dump(json_output, f, indent=2, default=str)


# =============================================================================
# MAIN ANALYSIS FUNCTION
# =============================================================================

def analyze_interaction_pair(
    pair: InteractionPair,
    scores_df: pd.DataFrame,
    config: Config,
    global_interface_map: Optional[Dict[str, Set[int]]] = None,
    domain_positions_map: Optional[Dict[str, Set[int]]] = None,
) -> InteractionResult:
    """Analyze a single interaction pair."""
    logger.info(f"Analyzing {pair.scored_protein_name} (chain {pair.scored_chain}) - {pair.interactor_name}")

    # Initialize result
    result = InteractionResult(
        pair=pair,
        residue_groups=ResidueGroups([], [], [], []),
        statistics_by_assay={}
    )

    try:
        # Load structure
        structure = load_structure(pair.pdb_file)

        # Apply pLDDT filtering if threshold is set (skip for experimental PDB structures)
        plddt_passing_a = None
        plddt_passing_b = None
        if config.plddt_threshold is not None and not pair.skip_plddt_filter:
            plddt_passing_a = filter_residues_by_plddt(structure, 'A', config.plddt_threshold)
            plddt_passing_b = filter_residues_by_plddt(structure, 'B', config.plddt_threshold)

            # Check if enough residues pass in each chain
            if len(plddt_passing_a) < config.min_residues_after_filter:
                result.skipped = True
                result.skip_reason = f"Only {len(plddt_passing_a)} chain A residues pass pLDDT >= {config.plddt_threshold}"
                return result
            if len(plddt_passing_b) < config.min_residues_after_filter:
                result.skipped = True
                result.skip_reason = f"Only {len(plddt_passing_b)} chain B residues pass pLDDT >= {config.plddt_threshold}"
                return result

            logger.debug(f"pLDDT filter: {len(plddt_passing_a)} chain A, {len(plddt_passing_b)} chain B residues pass >= {config.plddt_threshold}")

        # Calculate inter-chain distances
        distances_df = calculate_cbeta_distances(structure)
        if distances_df.empty:
            result.skipped = True
            result.skip_reason = "Could not calculate inter-chain distances"
            return result

        # Filter distances to only include residues passing pLDDT threshold
        if config.plddt_threshold is not None and not pair.skip_plddt_filter:
            before_filter = len(distances_df)
            distances_df = distances_df[
                distances_df['res_a'].isin(plddt_passing_a) &
                distances_df['res_b'].isin(plddt_passing_b)
            ]
            if distances_df.empty:
                result.skipped = True
                result.skip_reason = f"No inter-chain contacts after pLDDT filtering (had {before_filter} before)"
                return result

        # Run DSSP
        dssp_results = run_dssp(structure, pair.pdb_file, config.dssp_path)

        # Calculate intra-chain distances for control selection
        intrachain_distances = calculate_intrachain_distances(structure, pair.scored_chain)

        # Get scored positions for this protein
        protein_scores = scores_df[
            scores_df['uniprot_id'].str.split('-').str[0] == pair.scored_protein
        ]
        scored_positions = set(protein_scores['position'].unique())

        if not scored_positions:
            result.skipped = True
            result.skip_reason = "No scored positions found"
            return result

        # Define residue groups
        # Pass pLDDT-passing residues for the scored chain to filter surface/control
        plddt_passing_scored = None
        if config.plddt_threshold is not None and not pair.skip_plddt_filter:
            plddt_passing_scored = plddt_passing_a if pair.scored_chain == 'A' else plddt_passing_b

        # Resolve global interface positions for this protein (Mode B only)
        global_iface = None
        if global_interface_map is not None:
            global_iface = global_interface_map.get(pair.scored_protein, set())

        # Resolve domain positions for this protein (domain filter only)
        domain_pos = None
        if domain_positions_map is not None:
            domain_pos = domain_positions_map.get(pair.scored_protein_name)

        groups = define_residue_groups(
            structure, distances_df, dssp_results, intrachain_distances,
            pair.scored_chain, scored_positions, config,
            plddt_passing_residues=plddt_passing_scored,
            global_interface_positions=global_iface,
            domain_positions=domain_pos,
        )
        result.residue_groups = groups

        # Check data sufficiency
        sufficient, reason = check_data_sufficiency(groups, config.min_positions_per_group)
        if not sufficient:
            result.skipped = True
            result.skip_reason = reason
            return result

        # Get assays for this protein
        assays = protein_scores['assay'].unique()

        for assay in assays:
            assay_scores = protein_scores[protein_scores['assay'] == assay]

            # Get scores by group
            interface_scores = assay_scores[
                assay_scores['position'].isin(groups.interface_positions)
            ]['score'].values

            control_scores = assay_scores[
                assay_scores['position'].isin(groups.control_positions)
            ]['score'].values

            all_scores = assay_scores['score'].values

            # Skip if insufficient data for this assay
            if len(interface_scores) < config.min_positions_per_group:
                logger.debug(f"Skipping {assay}: insufficient interface scores")
                continue
            if len(control_scores) < config.min_positions_per_group:
                logger.debug(f"Skipping {assay}: insufficient control scores")
                continue

            # Compute statistics
            stats = compute_all_statistics(interface_scores, control_scores, all_scores, config)
            stats.n_interface_positions = len(groups.interface_positions)
            stats.n_control_positions = len(groups.control_positions)

            result.statistics_by_assay[assay] = stats

        if not result.statistics_by_assay:
            result.skipped = True
            result.skip_reason = "No assays had sufficient data"

        return result

    except Exception as e:
        logger.error(f"Error analyzing {pair.scored_protein_name}-{pair.interactor_name}: {e}")
        result.skipped = True
        result.skip_reason = str(e)
        return result


def compute_pair_interface_positions(
    pair: InteractionPair,
    config: Config,
) -> Tuple[str, Set[int]]:
    """
    Lightweight first-pass: load one structure and return the interface positions
    on the scored chain.  No DSSP, no statistics.

    Returns:
        (scored_protein_uniprot, set_of_interface_residue_numbers)
        On any failure, returns (scored_protein_uniprot, empty set).

    Used by build_global_interface_map() to aggregate all known interface
    positions per protein before the main analysis pass.
    """
    try:
        structure = load_structure(pair.pdb_file)

        plddt_passing_a, plddt_passing_b = None, None
        if config.plddt_threshold is not None and not pair.skip_plddt_filter:
            plddt_passing_a = filter_residues_by_plddt(structure, 'A', config.plddt_threshold)
            plddt_passing_b = filter_residues_by_plddt(structure, 'B', config.plddt_threshold)

        distances_df = calculate_cbeta_distances(structure)
        if distances_df.empty:
            return (pair.scored_protein, set())

        if config.plddt_threshold is not None and not pair.skip_plddt_filter:
            distances_df = distances_df[
                distances_df['res_a'].isin(plddt_passing_a) &
                distances_df['res_b'].isin(plddt_passing_b)
            ]

        interface = set(get_interface_positions(
            distances_df, pair.scored_chain, config.contact_distance
        ))
        return (pair.scored_protein, interface)

    except Exception as e:
        logger.debug(f"compute_pair_interface_positions failed for {pair.pdb_file}: {e}")
        return (pair.scored_protein, set())


def build_global_interface_map(
    pairs: List[InteractionPair],
    config: Config,
    n_cores: int,
) -> Dict[str, Set[int]]:
    """
    Pre-computation pass: aggregate all interface positions per scored protein
    across every structure in the dataset.

    Returns:
        Dict mapping scored_protein UniProt ID -> set of all residue numbers
        that are at the interface in at least one structure.

    Used when config.control_use_global_interface=True so that control residues
    can be required to be distant from ALL known interfaces, not just the one
    in the current structure.
    """
    logger.info(f"Building global interface map from {len(pairs)} structures "
                f"(first-pass pre-computation)...")

    global_map: Dict[str, Set[int]] = {}

    if n_cores == 1:
        for pair in pairs:
            protein, positions = compute_pair_interface_positions(pair, config)
            global_map.setdefault(protein, set()).update(positions)
    else:
        with ProcessPoolExecutor(max_workers=n_cores) as executor:
            futures = {
                executor.submit(compute_pair_interface_positions, pair, config): pair
                for pair in pairs
            }
            for future in as_completed(futures):
                try:
                    protein, positions = future.result()
                    global_map.setdefault(protein, set()).update(positions)
                except Exception as e:
                    logger.warning(f"Global interface pre-pass failed for a pair: {e}")

    for protein, positions in sorted(global_map.items()):
        logger.info(f"  Global interface: {protein} → {len(positions)} unique positions "
                    f"across all structures")

    logger.info(f"Global interface map complete: {len(global_map)} proteins")
    return global_map


def run_parallel_analysis(
    pairs: List[InteractionPair],
    scores_df: pd.DataFrame,
    config: Config,
    n_cores: int
) -> List[InteractionResult]:
    """Run analysis on multiple pairs in parallel."""

    # If global exclusion mode is requested, do a fast first-pass over all
    # structures to accumulate the full per-protein interface map before any
    # statistics are computed.
    global_interface_map: Optional[Dict[str, Set[int]]] = None
    if config.control_use_global_interface:
        global_interface_map = build_global_interface_map(pairs, config, n_cores)

    # Load domain positions if domain-restricted controls are requested.
    domain_positions_map: Optional[Dict[str, Set[int]]] = None
    if config.control_within_domains:
        domain_positions_map = load_domain_positions(config.domains_info_file)
        if not domain_positions_map:
            logger.warning("control_within_domains=True but no domain positions loaded; "
                           "falling back to all surface residues")

    results = []

    if n_cores == 1:
        # Sequential execution
        for pair in pairs:
            result = analyze_interaction_pair(
                pair, scores_df, config, global_interface_map, domain_positions_map
            )
            results.append(result)
    else:
        # Parallel execution
        with ProcessPoolExecutor(max_workers=n_cores) as executor:
            futures = {
                executor.submit(
                    analyze_interaction_pair, pair, scores_df, config,
                    global_interface_map, domain_positions_map
                ): pair
                for pair in pairs
            }

            for future in as_completed(futures):
                pair = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Analysis failed for {pair.scored_protein}: {e}")
                    results.append(InteractionResult(
                        pair=pair,
                        residue_groups=ResidueGroups([], [], [], []),
                        statistics_by_assay={},
                        skipped=True,
                        skip_reason=str(e)
                    ))

    return results


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="DMS Interface Interaction Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--config', '-c',
        type=Path,
        required=True,
        help='Path to configuration YAML file'
    )

    parser.add_argument(
        '--protein', '-p',
        type=str,
        help='Analyze only interactions involving this protein (UniProt ID)'
    )

    parser.add_argument(
        '--output', '-o',
        type=Path,
        help='Override output directory from config'
    )

    parser.add_argument(
        '--generate-sge',
        action='store_true',
        help='Generate SGE job scripts instead of running analysis'
    )

    parser.add_argument(
        '--sge-output',
        type=Path,
        default=Path('sge_scripts'),
        help='Directory for SGE scripts (default: sge_scripts/)'
    )

    parser.add_argument(
        '--aggregate',
        action='store_true',
        help='Aggregate results from parallel runs'
    )

    parser.add_argument(
        '--input-dirs',
        type=Path,
        nargs='+',
        help='Input directories for aggregation'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be analyzed without running'
    )

    parser.add_argument(
        '--cores',
        type=int,
        help='Override number of cores from config'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Setup logging
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        for handler in logger.handlers:
            handler.setLevel(logging.DEBUG)

    # Load configuration
    config = load_config(args.config)

    # Override config with CLI args
    if args.output:
        config.output_directory = args.output
    if args.cores:
        config.n_cores = args.cores

    # Handle aggregation mode
    if args.aggregate:
        if not args.input_dirs:
            logger.error("--input-dirs required for aggregation")
            return 1
        aggregate_parallel_results(args.input_dirs, config.output_directory, config)
        return 0

    # Load scores
    scores_df = load_scores(config)
    scored_proteins = get_scored_proteins(scores_df)

    # Handle SGE script generation
    if args.generate_sge:
        proteins = list(scored_proteins.keys())
        generate_sge_scripts(config, proteins, args.sge_output, args.config)
        return 0

    # Discover interaction pairs
    pairs = discover_interaction_pairs(config, scored_proteins)

    # Filter to single protein if specified
    if args.protein:
        protein_clean = args.protein.split('-')[0]
        pairs = [p for p in pairs if p.scored_protein == protein_clean]
        logger.info(f"Filtered to {len(pairs)} pairs for protein {args.protein}")

    if not pairs:
        logger.warning("No interaction pairs to analyze")
        return 0

    # Dry run
    if args.dry_run:
        logger.info("=== DRY RUN ===")
        logger.info(f"Would analyze {len(pairs)} interaction pairs:")
        for pair in pairs[:20]:  # Show first 20
            logger.info(f"  {pair.scored_protein_name} (chain {pair.scored_chain}) - {pair.interactor_name}")
        if len(pairs) > 20:
            logger.info(f"  ... and {len(pairs) - 20} more")
        return 0

    # Run analysis
    logger.info(f"Starting analysis of {len(pairs)} interaction pairs")
    results = run_parallel_analysis(pairs, scores_df, config, config.n_cores)

    # Apply FDR correction
    if config.statistics.get('fdr_correction', True):
        results = apply_fdr_correction(results)

    # Write outputs
    write_outputs(results, config, scores_df)

    # Summary
    n_analyzed = sum(1 for r in results if not r.skipped)
    n_skipped = sum(1 for r in results if r.skipped)
    logger.info(f"Analysis complete: {n_analyzed} analyzed, {n_skipped} skipped")

    return 0


if __name__ == "__main__":
    sys.exit(main())
