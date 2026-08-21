"""Configuration loader for the LABEL-seq MAPK pipeline.

Loads scoring parameters, protein metadata, and file paths from YAML configs.
Provides helper functions to look up per-protein settings by library name.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def load_config(config_dir: Path) -> Dict[str, Any]:
    """Load all YAML config files into a unified dictionary.

    Args:
        config_dir: Path to the config/ directory containing scoring.yaml,
            proteins.yaml, and paths.yaml.

    Returns:
        Dictionary with keys 'scoring', 'proteins', 'paths', each containing
        the parsed contents of the corresponding YAML file.
    """
    config = {}
    for name in ("scoring", "proteins", "paths"):
        filepath = config_dir / f"{name}.yaml"
        with open(filepath) as f:
            config[name] = yaml.safe_load(f)
    return config


def resolve_paths(config: Dict[str, Any], project_root: Path) -> Dict[str, Any]:
    """Convert relative paths in the paths config to absolute paths.

    Recursively walks the 'paths' section of the config and resolves any
    string value that looks like a relative path against project_root.

    Args:
        config: Full config dict (modified in place).
        project_root: Absolute path to the project root directory.

    Returns:
        The same config dict with paths resolved.
    """
    def _resolve(obj: Any) -> Any:
        if isinstance(obj, str) and "/" in obj:
            resolved = project_root / obj
            return str(resolved)
        elif isinstance(obj, dict):
            return {k: _resolve(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_resolve(item) for item in obj]
        return obj

    config["paths"] = _resolve(config["paths"])
    return config


def get_protein_config(config: Dict[str, Any], library: str) -> Optional[Dict[str, Any]]:
    """Look up protein metadata by library name.

    For split libraries (e.g., 'braf_cterm'), the library name is a direct
    key in proteins.yaml. For non-split libraries (e.g., 'kras'), the
    library name is also a direct key.

    Args:
        config: Full config dict containing 'proteins' key.
        library: Library name as it appears in the raw data filenames.

    Returns:
        Dictionary of protein metadata, or None if not found.
    """
    return config["proteins"].get(library)


def get_position_offset(config: Dict[str, Any], library: str, assay: str) -> int:
    """Get the position offset for a (library, assay) pair.

    Position offsets correct for the fact that some libraries cover only a
    portion of the full-length protein. For example, EGFR activity has
    offset +670 because the library starts at residue 670 in the full-length
    protein, but the raw data numbers positions starting from 1.

    Args:
        config: Full config dict containing 'proteins' key.
        library: Library name (e.g., 'egfr', 'braf_cterm').
        assay: Assay type ('activity', 'abundance', 'interaction').

    Returns:
        Integer offset to add to raw positions. Returns 0 if the library
        or assay is not found in the config (meaning no offset needed).
    """
    protein_cfg = get_protein_config(config, library)
    if protein_cfg is None:
        return 0
    offsets = protein_cfg.get("position_offsets", {})
    if offsets is None:
        return 0
    return offsets.get(assay, 0)


def get_domain(config: Dict[str, Any], protein: str, position: Any) -> str:
    """Look up which domain a position falls in for a given protein.

    Uses the domain boundaries defined in proteins.yaml. For split libraries,
    the caller should pass the base protein name (e.g., 'braf' not 'braf_cterm').

    Args:
        config: Full config dict containing 'proteins' key.
        protein: Base protein name (e.g., 'braf', 'kras').
        position: Residue position (int or numeric). Non-numeric values
            return 'none'.

    Returns:
        Domain name string, or 'none' if the position is outside all domains
        or the protein has no domain definitions.
    """
    try:
        pos = int(position)
    except (ValueError, TypeError):
        return "none"

    # Try base protein name first, then check split library entries
    protein_cfg = config["proteins"].get(protein)
    if protein_cfg is None:
        # Try with _nterm or _cterm suffix entries and merge domains
        domains = {}
        for suffix in ("_nterm", "_cterm", ""):
            key = f"{protein}{suffix}" if suffix else protein
            cfg = config["proteins"].get(key)
            if cfg and "domains" in cfg:
                domains.update(cfg["domains"])
        if not domains:
            return "none"
    else:
        domains = protein_cfg.get("domains", {})

    if not domains:
        return "none"

    for domain_name, (start, end) in domains.items():
        if start <= pos <= end:
            return domain_name
    return "none"
