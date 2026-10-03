#!/usr/bin/env python3
"""
Fetch RCSB PDB experimental structures as a control dataset.

For each UniProt ID, queries RCSB PDB for heterocomplex experimental structures
containing that protein (resolution/method/entity-count filters applied at
search time to minimise API calls). Downloads each structure, enumerates valid
two-chain pairs, and writes one two-chain PDB file per pair with chains
relabeled A/B and residues renumbered to UniProt positions via SIFTS.

Per-entry processing is parallelised with a thread pool.

Input (one of three options):
    --uniprot-ids O14807,P00533,...      comma-separated
    --uniprot-ids-file ids.txt           one per line
    --scores-file scores.tsv             extract from --uniprot-col column

Usage:
    python fetch_pdb_controls.py \\
        --uniprot-ids O14807,P00533,P01116,P04049,P04626,P07949,P08581,P10398,P15056,P36507,P62993,Q02750,Q06124,Q07889,Q07890 \\
        --output-dir pdb_controls/ \\
        --resolution-cutoff 3.5 \\
        --em-resolution-cutoff 4.5 \\
        --experimental-methods "X-RAY DIFFRACTION,ELECTRON MICROSCOPY" \\
        --min-sifts-coverage 0.5 \\
        --min-partner-length 10 \\
        --max-per-pair 5 \\
        --threads 20 \\
        --extra-pdb-ids 3KUC,9AXM,6EPL,1MW4,4QSY,6VJJ \\
        --uniprot-aliases Q6VAB6:P15056,Q8IVT5:P15056
"""

import argparse
import csv
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

# =============================================================================
# API ENDPOINTS
# =============================================================================

RCSB_SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
RCSB_DATA_URL   = "https://data.rcsb.org/rest/v1/core"
RCSB_FILES_URL  = "https://files.rcsb.org/download"
PDBE_SIFTS_URL  = "https://www.ebi.ac.uk/pdbe/api/mappings/uniprot"

# EM methods use a separate (looser) resolution cutoff; NMR has no resolution
EM_METHODS      = {'ELECTRON MICROSCOPY', 'ELECTRON CRYSTALLOGRAPHY'}
NO_RES_METHODS  = {'SOLUTION NMR', 'SOLID-STATE NMR'}

# =============================================================================
# INPUT: UNIPROT ID LOADING
# =============================================================================

def load_uniprot_ids_from_scores(scores_file: str, uniprot_col: str = "uniprot_id") -> List[str]:
    ids = set()
    with open(scores_file, newline='') as fh:
        for row in csv.DictReader(fh, delimiter='\t'):
            uid = row.get(uniprot_col, '').strip()
            if uid:
                ids.add(uid.split('-')[0])
    return sorted(ids)


def load_uniprot_ids_from_file(ids_file: str) -> List[str]:
    ids = set()
    with open(ids_file) as fh:
        for line in fh:
            uid = line.strip().split('-')[0]
            if uid and not uid.startswith('#'):
                ids.add(uid)
    return sorted(ids)


def parse_uniprot_ids(args) -> List[str]:
    if args.uniprot_ids:
        ids = sorted({u.strip().split('-')[0] for u in args.uniprot_ids.split(',') if u.strip()})
    elif args.uniprot_ids_file:
        ids = load_uniprot_ids_from_file(args.uniprot_ids_file)
    elif args.scores_file:
        ids = load_uniprot_ids_from_scores(args.scores_file, args.uniprot_col)
    else:
        raise ValueError("Provide --uniprot-ids, --uniprot-ids-file, or --scores-file")
    logger.info(f"Loaded {len(ids)} UniProt IDs: {ids}")
    return ids

# =============================================================================
# RCSB SEARCH — with resolution/method/entity-count pre-filters
# =============================================================================

def _method_node(method: str) -> dict:
    return {
        "type": "terminal",
        "service": "text",
        "parameters": {
            "attribute": "exptl.method",
            "operator": "exact_match",
            "value": method,
        }
    }


def _resolution_node(cutoff: float) -> dict:
    return {
        "type": "terminal",
        "service": "text",
        "parameters": {
            "attribute": "rcsb_entry_info.resolution_combined",
            "operator": "less_or_equal",
            "value": cutoff,
            "negation": False,
        }
    }


def query_rcsb_for_uniprot(
    uniprot_id: str,
    xray_resolution_cutoff: float,
    em_resolution_cutoff: float,
    method_allowlist: List[str],
) -> List[str]:
    """
    Search RCSB for PDB entries that:
      - contain the given UniProt ID
      - have >= 2 deposited polymer chain instances
      - pass per-method resolution filters (X-ray vs EM have different cutoffs;
        NMR-type methods are included without a resolution constraint)

    All filtering is done server-side to minimise the number of entries
    returned (and therefore the number of follow-up API calls needed).
    """
    # Node: UniProt accession
    uniprot_node = {
        "type": "terminal",
        "service": "text",
        "parameters": {
            "attribute": (
                "rcsb_polymer_entity_container_identifiers"
                ".reference_sequence_identifiers.database_accession"
            ),
            "operator": "exact_match",
            "value": uniprot_id,
        }
    }

    # Node: at least 2 polymer chain instances (covers both homodimers and heterocomplexes)
    heterocomplex_node = {
        "type": "terminal",
        "service": "text",
        "parameters": {
            "attribute": "rcsb_entry_info.deposited_polymer_entity_instance_count",
            "operator": "greater_or_equal",
            "value": 2,
        }
    }

    # Build per-method (method AND resolution) groups; NMR-type has no resolution filter
    method_res_groups = []
    for method in method_allowlist:
        m = method.upper()
        if m in NO_RES_METHODS:
            method_res_groups.append(_method_node(method))
        else:
            cutoff = em_resolution_cutoff if m in EM_METHODS else xray_resolution_cutoff
            method_res_groups.append({
                "type": "group",
                "logical_operator": "and",
                "nodes": [_method_node(method), _resolution_node(cutoff)],
            })

    if not method_res_groups:
        return []

    if len(method_res_groups) == 1:
        method_res_node = method_res_groups[0]
    else:
        method_res_node = {
            "type": "group",
            "logical_operator": "or",
            "nodes": method_res_groups,
        }

    query = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [uniprot_node, heterocomplex_node, method_res_node],
        },
        "return_type": "entry",
        "request_options": {"return_all_hits": True},
    }

    try:
        resp = requests.post(RCSB_SEARCH_URL, json=query, timeout=30)
        resp.raise_for_status()
        pdb_ids = [r['identifier'] for r in resp.json().get('result_set', [])]
        logger.info(f"  {uniprot_id}: {len(pdb_ids)} entries pass search filters")
        return pdb_ids
    except requests.RequestException as e:
        logger.warning(f"  RCSB search failed for {uniprot_id}: {e}")
        return []

# =============================================================================
# PER-ENTRY DATA FETCHING
# =============================================================================

def get_entry_entities(pdb_id: str) -> List[Dict]:
    """
    Fetch entity info for a PDB entry.

    Returns list of dicts:
        entity_id, chain_ids, uniprot_ids, length, has_mutation,
        resolution_angstroms, experimental_method

    Entities without UniProt mappings are still included (length/mutation
    data is needed even for partner chains not in our scored set).
    """
    try:
        resp = requests.get(f"{RCSB_DATA_URL}/entry/{pdb_id}", timeout=30)
        resp.raise_for_status()
        entry_data = resp.json()
    except requests.RequestException as e:
        logger.warning(f"  Entry fetch failed {pdb_id}: {e}")
        return []

    # Resolution — rcsb_entry_info.resolution_combined is the canonical field
    info = entry_data.get('rcsb_entry_info', {})
    res_list = info.get('resolution_combined') or []
    resolution = res_list[0] if res_list else None

    exp_method = None
    exptl = entry_data.get('exptl', [{}])
    if exptl:
        exp_method = exptl[0].get('method', '').upper()

    polymer_entity_ids = entry_data.get(
        'rcsb_entry_container_identifiers', {}
    ).get('polymer_entity_ids', [])

    entities = []
    for entity_id in polymer_entity_ids:
        try:
            eresp = requests.get(
                f"{RCSB_DATA_URL}/polymer_entity/{pdb_id}/{entity_id}", timeout=30
            )
            eresp.raise_for_status()
            edata = eresp.json()
        except requests.RequestException as e:
            logger.warning(f"  Entity fetch failed {pdb_id}/{entity_id}: {e}")
            continue

        container = edata.get('rcsb_polymer_entity_container_identifiers', {})
        chain_ids   = container.get('auth_asym_ids', [])

        # uniprot_ids is the direct container field (most reliable)
        uniprot_ids = [u for u in (container.get('uniprot_ids') or []) if u]

        # Sequence length — lives in entity_poly, not rcsb_polymer_entity
        seq = (edata.get('entity_poly', {}).get('pdbx_seq_one_letter_code_can') or '')
        length = len(seq)

        poly = edata.get('rcsb_polymer_entity', {})
        has_mutation = bool((poly.get('pdbx_mutation') or '').strip())

        if chain_ids:          # include even if no UniProt (need length/mutation for partners)
            entities.append({
                'entity_id':            entity_id,
                'chain_ids':            chain_ids,
                'uniprot_ids':          uniprot_ids,
                'length':               length,
                'has_mutation':         has_mutation,
                'resolution_angstroms': resolution,
                'experimental_method':  exp_method,
            })

    return entities


def get_cif_label_auth_map(pdb_id: str, raw_dir: Path) -> Dict[str, Dict[int, int]]:
    """
    Download the mmCIF file for *pdb_id* and parse the _atom_site block to
    build a {auth_asym_id: {label_seq_id: auth_seq_id}} mapping.

    This is only needed when SIFTS segments have author_residue_number=None
    (i.e., the PDB entry uses non-standard author residue numbering and SIFTS
    only records the sequential label_seq_id instead of the auth_seq_id that
    Biopython's res.id[1] exposes).
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    cif_path = raw_dir / f"{pdb_id}.cif"

    if not (cif_path.exists() and cif_path.stat().st_size > 100):
        try:
            resp = requests.get(
                f"{RCSB_FILES_URL}/{pdb_id}.cif", timeout=120
            )
            resp.raise_for_status()
            cif_path.write_bytes(resp.content)
            logger.debug(f"  Downloaded CIF for {pdb_id} ({cif_path.stat().st_size} bytes)")
        except requests.RequestException as e:
            logger.warning(f"  CIF download failed {pdb_id}: {e}")
            return {}

    result: Dict[str, Dict[int, int]] = {}
    # State machine to parse the _atom_site loop_ block
    state = 'idle'   # idle | loop_start | atom_cols | atom_data | other_cols
    col_names: List[str] = []
    col_idx: Dict[str, int] = {}

    try:
        with open(cif_path, errors='replace') as fh:
            for raw_line in fh:
                line = raw_line.strip()
                if not line or line.startswith('#'):
                    continue

                if line.startswith('data_'):
                    state = 'idle'; col_names = []; col_idx = {}
                    continue

                if line == 'loop_':
                    state = 'loop_start'; col_names = []; col_idx = {}
                    continue

                if state == 'loop_start':
                    if line.startswith('_atom_site.'):
                        col_names.append(line[len('_atom_site.'):])
                        state = 'atom_cols'
                    elif line.startswith('_'):
                        state = 'other_cols'
                    else:
                        state = 'idle'
                    continue

                if state == 'atom_cols':
                    if line.startswith('_atom_site.'):
                        col_names.append(line[len('_atom_site.'):])
                        continue
                    elif line.startswith('_'):
                        state = 'idle'
                        continue
                    else:
                        # First data line — build column index and fall through
                        col_idx = {n: i for i, n in enumerate(col_names)}
                        state = 'atom_data'
                        # fall through to atom_data block

                if state == 'atom_data':
                    if line.startswith('_') or line == 'loop_' or line.startswith('data_'):
                        state = 'idle'; col_names = []; col_idx = {}
                        continue
                    if not (line.startswith('ATOM') or line.startswith('HETATM')):
                        continue
                    if not all(k in col_idx for k in
                               ('auth_asym_id', 'label_seq_id', 'auth_seq_id')):
                        continue
                    parts = line.split()
                    if len(parts) <= max(
                        col_idx['auth_asym_id'],
                        col_idx['label_seq_id'],
                        col_idx['auth_seq_id'],
                    ):
                        continue
                    # Only model 1 for NMR / multi-model structures
                    if 'pdbx_PDB_model_num' in col_idx:
                        if parts[col_idx['pdbx_PDB_model_num']] != '1':
                            continue
                    auth_chain = parts[col_idx['auth_asym_id']]
                    label_seq  = parts[col_idx['label_seq_id']]
                    auth_seq   = parts[col_idx['auth_seq_id']]
                    if label_seq in ('.', '?') or auth_seq in ('.', '?'):
                        continue
                    try:
                        result.setdefault(auth_chain, {})[int(label_seq)] = int(auth_seq)
                    except (ValueError, TypeError):
                        pass
    except OSError as e:
        logger.warning(f"  CIF parse failed {pdb_id}: {e}")
        return {}

    logger.debug(
        f"  CIF label→auth map for {pdb_id}: "
        f"{sum(len(v) for v in result.values())} residues across {len(result)} chains"
    )
    return result


def get_sifts_mapping(
    pdb_id: str,
    raw_dir: Optional[Path] = None,
) -> Dict[str, List[Dict]]:
    """
    Fetch per-chain UniProt residue mapping from PDBe SIFTS.

    PDBe response structure:
        {pdb_id: {"UniProt": {uniprot_id: {"mappings": [...]}}}}

    When SIFTS reports author_residue_number=None for a segment (which happens
    for structures with non-standard author residue numbering), we fall back to
    residue_number (label_seq_id) and then convert to auth_seq_id using the
    mmCIF file (downloaded to raw_dir if provided).
    """
    try:
        resp = requests.get(f"{PDBE_SIFTS_URL}/{pdb_id.lower()}", timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        logger.warning(f"  SIFTS fetch failed {pdb_id}: {e}")
        return {}

    result: Dict[str, List[Dict]] = {}
    chains_needing_conversion: set = set()

    uniprot_section = data.get(pdb_id.lower(), {}).get('UniProt', {})
    for uniprot_id, udata in uniprot_section.items():
        for m in udata.get('mappings', []):
            chain_id = m.get('chain_id', '')
            if not chain_id:
                continue
            s, e = m.get('start', {}), m.get('end', {})
            unp_s, unp_e = m.get('unp_start'), m.get('unp_end')
            if None in (unp_s, unp_e):
                continue

            auth_s = s.get('author_residue_number')
            auth_e = e.get('author_residue_number')
            label_s = s.get('residue_number')
            label_e = e.get('residue_number')

            if auth_s is not None and auth_e is not None:
                pdb_s, pdb_e = auth_s, auth_e
                use_label = False
            elif label_s is not None and label_e is not None:
                pdb_s, pdb_e = label_s, label_e
                use_label = True
                chains_needing_conversion.add(chain_id)
            else:
                continue

            result.setdefault(chain_id, []).append({
                'uniprot_id':    uniprot_id,
                'uniprot_start': int(unp_s),
                'uniprot_end':   int(unp_e),
                'pdb_start':     int(pdb_s),
                'pdb_end':       int(pdb_e),
                '_use_label':    use_label,
            })

    # Convert label_seq_id → auth_seq_id for chains that need it
    if chains_needing_conversion and raw_dir is not None:
        logger.debug(
            f"  {pdb_id}: fetching CIF to fix label→auth for chains "
            f"{chains_needing_conversion}"
        )
        l2a_by_chain = get_cif_label_auth_map(pdb_id, raw_dir)
        for chain_id in chains_needing_conversion:
            l2a = l2a_by_chain.get(chain_id, {})
            converted = []
            for seg in result.get(chain_id, []):
                if not seg.get('_use_label'):
                    converted.append(seg)
                    continue
                # SIFTS segment endpoints are label_seq_ids; CIF may not have
                # ATOM records for every label_seq_id (unresolved residues).
                # Use first/last label_seq_id within the range that ARE present
                # in ATOM records, adjusting uniprot_start/end accordingly.
                label_start = seg['pdb_start']
                label_end   = seg['pdb_end']
                # Find first and last label_seq_id with ATOM records
                first_label = next(
                    (lid for lid in range(label_start, label_end + 1) if lid in l2a), None
                )
                last_label = next(
                    (lid for lid in range(label_end, label_start - 1, -1) if lid in l2a), None
                )
                if first_label is None or last_label is None:
                    logger.debug(
                        f"  {pdb_id} {chain_id}: no ATOM records for label "
                        f"range {label_start}-{label_end} — dropping segment"
                    )
                    continue
                # Adjust uniprot boundaries to match actual resolved residues
                uniprot_adj_start = seg['uniprot_start'] + (first_label - label_start)
                uniprot_adj_end   = seg['uniprot_end']   - (label_end   - last_label)
                seg['pdb_start']     = l2a[first_label]
                seg['pdb_end']       = l2a[last_label]
                seg['uniprot_start'] = uniprot_adj_start
                seg['uniprot_end']   = uniprot_adj_end
                converted.append(seg)
            result[chain_id] = converted
    elif chains_needing_conversion:
        logger.debug(
            f"  {pdb_id}: chains {chains_needing_conversion} have label-only "
            f"SIFTS resnums but no raw_dir provided for CIF fallback"
        )

    # Remove internal flag before returning
    for chain_segs in result.values():
        for seg in chain_segs:
            seg.pop('_use_label', None)

    return result

# =============================================================================
# CHAIN PAIR ENUMERATION
# =============================================================================

def enumerate_chain_pairs(
    entities: List[Dict],
    sifts: Dict[str, List[Dict]],
    scored_uniprots: List[str],
    min_partner_length: int = 10,
    aliases: Optional[Dict[str, str]] = None,
) -> List[Tuple]:
    """
    Return valid (uniprot_a, chain_a, uniprot_b, chain_b) pairs where:
      - scored chain is chain A; partner is chain B
      - partner chain sequence length >= min_partner_length
      - both chains have SIFTS mappings

    Deduplicates at the entity level: chains sharing an entity_id are
    crystallographic copies of the same protein. For each unique entity-entity
    interaction we pick one representative chain per entity (the one with the
    most SIFTS-mapped residues). This reduces O(N²) redundant copies of the
    same interface to exactly one pair.

    Direction is normalised: when both proteins are in the scored set, the
    alphabetically-first UniProt ID is always uniprot_a, eliminating
    (MEK1, BRAF) + (BRAF, MEK1) double-counting.

    aliases maps non-canonical UniProt IDs (as used in RCSB/SIFTS) to their
    canonical equivalents (e.g. {'Q6VAB6': 'P15056'}).
    """
    if aliases is None:
        aliases = {}

    # Primary: chain→UniProt from SIFTS, normalised through aliases
    chain_to_uniprot: Dict[str, str] = {
        chain: aliases.get(segs[0]['uniprot_id'], segs[0]['uniprot_id'])
        for chain, segs in sifts.items()
        if segs
    }

    # Supplementary: length from entity data
    chain_to_length: Dict[str, int] = {}
    for ent in entities:
        for ch in ent['chain_ids']:
            chain_to_length[ch] = ent['length']

    # Only consider chains that have SIFTS mappings
    sifts_chains = set(chain_to_uniprot.keys())
    scored_set   = set(scored_uniprots)

    # Build entity → chains sorted by SIFTS-mapped residue count (best first)
    # Chains within the same entity are crystallographic copies of the same protein.
    def sifts_count(ch: str) -> int:
        return sum(
            seg['pdb_end'] - seg['pdb_start'] + 1
            for seg in sifts.get(ch, [])
        )

    entity_sifts_chains: Dict[str, List[str]] = {}
    for ent in entities:
        eid = ent['entity_id']
        chains_ok = sorted(
            [ch for ch in ent['chain_ids'] if ch in sifts_chains],
            key=sifts_count, reverse=True,
        )
        if chains_ok:
            entity_sifts_chains[eid] = chains_ok

    entity_ids = sorted(entity_sifts_chains.keys())
    pairs: List[Tuple] = []

    for i, ei in enumerate(entity_ids):
        chains_i = entity_sifts_chains[ei]
        ui = chain_to_uniprot[chains_i[0]]  # all chains in entity share the same UniProt

        for j in range(i, len(entity_ids)):
            ej = entity_ids[j]
            chains_j = entity_sifts_chains[ej]
            uj = chain_to_uniprot[chains_j[0]]

            if ei == ej:
                # Homodimer: need two distinct chains from the same entity
                if ui not in scored_set:
                    continue
                if len(chains_i) < 2:
                    continue
                ca, cb = chains_i[0], chains_i[1]
                ua, ub = ui, uj
            else:
                # Heterocomplex: one representative per entity
                i_scored = ui in scored_set
                j_scored = uj in scored_set
                if not (i_scored or j_scored):
                    continue

                ca, cb = chains_i[0], chains_j[0]

                # Canonical direction: when both scored, alphabetically-first UniProt → A
                if i_scored and j_scored:
                    if ui <= uj:
                        ua, chain_a, ub, chain_b = ui, ca, uj, cb
                    else:
                        ua, chain_a, ub, chain_b = uj, cb, ui, ca
                    ca, cb = chain_a, chain_b
                    ua, ub = ua, ub
                elif i_scored:
                    ua, ub = ui, uj
                else:
                    # j is scored → swap so scored is always chain A
                    ua, ca, ub, cb = uj, cb, ui, ca

            # Partner must be large enough
            partner_len = chain_to_length.get(cb, 0)
            if partner_len < min_partner_length:
                logger.debug(f"  Skip {ca}/{cb}: partner {ub} length {partner_len} < {min_partner_length}")
                continue

            pairs.append((ua, ca, ub, cb))

    return pairs

# =============================================================================
# DOWNLOAD + SIFTS MAP
# =============================================================================

def download_pdb(pdb_id: str, raw_dir: Path) -> Optional[Path]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = raw_dir / f"{pdb_id}.pdb"
    if dest.exists() and dest.stat().st_size > 1000:
        return dest
    try:
        resp = requests.get(f"{RCSB_FILES_URL}/{pdb_id}.pdb", timeout=60)
        if resp.status_code == 404:
            # Newer structures may only have mmCIF format — convert CIF → PDB
            return _download_cif_as_pdb(pdb_id, raw_dir, dest)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        return dest
    except requests.RequestException as e:
        logger.warning(f"  Download failed {pdb_id}: {e}")
        return None


def _download_cif_as_pdb(pdb_id: str, raw_dir: Path, dest: Path) -> Optional[Path]:
    """Download mmCIF for pdb_id and convert to legacy PDB format via Biopython."""
    cif_path = raw_dir / f"{pdb_id}.cif"
    if not (cif_path.exists() and cif_path.stat().st_size > 100):
        try:
            resp = requests.get(f"{RCSB_FILES_URL}/{pdb_id}.cif", timeout=120)
            if resp.status_code == 404:
                logger.warning(f"  404 for both PDB and CIF: {pdb_id}")
                return None
            resp.raise_for_status()
            cif_path.write_bytes(resp.content)
        except requests.RequestException as e:
            logger.warning(f"  CIF download failed {pdb_id}: {e}")
            return None
    try:
        from Bio.PDB import MMCIFParser, PDBIO
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure(pdb_id, str(cif_path))
        io = PDBIO()
        io.set_structure(structure)
        io.save(str(dest))
        logger.info(f"  Converted CIF → PDB for {pdb_id} ({dest.stat().st_size} bytes)")
        return dest
    except Exception as e:
        logger.warning(f"  CIF→PDB conversion failed {pdb_id}: {e}")
        return None


def build_resnum_map(segments: List[Dict]) -> Dict[int, int]:
    """pdb_resnum → uniprot_position via linear SIFTS segments."""
    resmap: Dict[int, int] = {}
    for seg in segments:
        ps, pe = seg['pdb_start'], seg['pdb_end']
        us = seg['uniprot_start']
        if (pe - ps) != (seg['uniprot_end'] - us):
            continue   # skip indel segments
        offset = us - ps
        for r in range(ps, pe + 1):
            resmap[r] = r + offset
    return resmap

# =============================================================================
# STRUCTURE EXTRACTION
# =============================================================================

def extract_renumber_save(
    raw_pdb: Path,
    chain_a_orig: str,
    chain_b_orig: str,
    sifts_map_a: Dict[int, int],
    sifts_map_b: Dict[int, int],
    uniprot_a: str,
    uniprot_b: str,
    pdb_id: str,
    output_dir: Path,
    min_sifts_coverage: float = 0.5,
) -> Optional[Tuple[Path, Dict]]:
    """Extract two chains, relabel A/B, renumber to UniProt positions, save."""
    try:
        from Bio.PDB import PDBParser, PDBIO, Structure, Model, Chain
        from Bio.PDB.Residue import Residue as BioResidue
    except ImportError:
        logger.error("Biopython not installed: pip install biopython")
        sys.exit(1)

    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure(pdb_id, str(raw_pdb))
    except Exception as e:
        logger.warning(f"  Parse failed {raw_pdb}: {e}")
        return None

    model = structure[0]
    present = {c.id for c in model.get_chains()}
    if chain_a_orig not in present or chain_b_orig not in present:
        logger.warning(f"  Chains {chain_a_orig},{chain_b_orig} missing from {pdb_id}")
        return None

    def renumber(chain, resmap):
        kept = []
        for res in chain.get_residues():
            if res.id[0] != ' ':
                continue
            rn = res.id[1]
            if rn in resmap:
                kept.append((resmap[rn], res))
        return kept

    chain_a = model[chain_a_orig]
    chain_b = model[chain_b_orig]
    kept_a  = renumber(chain_a, sifts_map_a)
    kept_b  = renumber(chain_b, sifts_map_b)

    n_total_a = sum(1 for r in chain_a.get_residues() if r.id[0] == ' ')
    n_total_b = sum(1 for r in chain_b.get_residues() if r.id[0] == ' ')
    if n_total_a == 0:
        return None

    cov_a = len(kept_a) / n_total_a
    cov_b = len(kept_b) / n_total_b if n_total_b else 0.0

    if cov_a < min_sifts_coverage:
        logger.warning(f"  Skip {pdb_id} {chain_a_orig}+{chain_b_orig}: coverage A={cov_a:.2f}")
        return None
    if not kept_a or not kept_b:
        return None

    out_struct = Structure.Structure(pdb_id)
    out_model  = Model.Model(0)
    out_struct.add(out_model)

    for new_id, residues in [('A', kept_a), ('B', kept_b)]:
        new_chain = Chain.Chain(new_id)
        out_model.add(new_chain)
        for new_rn, orig_res in residues:
            new_res = BioResidue((' ', new_rn, ' '), orig_res.resname, orig_res.segid)
            for atom in orig_res.get_atoms():
                new_res.add(atom.copy())
            new_chain.add(new_res)

    output_dir.mkdir(parents=True, exist_ok=True)
    fname  = f"{uniprot_a}_{uniprot_b}_{pdb_id}_{chain_a_orig}{chain_b_orig}.pdb"
    fpath  = output_dir / fname
    io = PDBIO()
    io.set_structure(out_struct)
    io.save(str(fpath))

    return fpath, {
        'pdb_id': pdb_id, 'uniprot_a': uniprot_a, 'chain_a_orig': chain_a_orig,
        'uniprot_b': uniprot_b, 'chain_b_orig': chain_b_orig,
        'resolution_angstroms': None, 'experimental_method': None,
        'n_residues_a_mapped': len(kept_a), 'n_residues_b_mapped': len(kept_b),
        'sifts_coverage_a': round(cov_a, 3), 'sifts_coverage_b': round(cov_b, 3),
        'output_file': fname,
    }

# =============================================================================
# PER-ENTRY WORKER (runs in thread pool)
# =============================================================================

def process_entry(
    pdb_id: str,
    scored_uniprots: List[str],
    raw_dir: Path,
    output_dir: Path,
    min_partner_length: int,
    min_sifts_coverage: float,
    aliases: Optional[Dict[str, str]] = None,
) -> List[Dict]:
    """
    Full pipeline for one PDB entry: fetch entities → SIFTS → enumerate pairs
    → download → extract/renumber → return list of metadata dicts.
    """
    if aliases is None:
        aliases = {}
    results = []

    entities = get_entry_entities(pdb_id)
    if not entities:
        return results

    resolution = entities[0].get('resolution_angstroms')
    exp_method = entities[0].get('experimental_method', '')

    # Pass raw_dir so get_sifts_mapping can download the CIF when needed to fix
    # segments where SIFTS has author_residue_number=None (label_seq_id fallback)
    sifts = get_sifts_mapping(pdb_id, raw_dir=raw_dir)
    if not sifts:
        logger.debug(f"  No SIFTS for {pdb_id}")
        return results

    chain_pairs = enumerate_chain_pairs(
        entities, sifts, scored_uniprots, min_partner_length, aliases=aliases
    )
    if not chain_pairs:
        logger.debug(f"  No valid pairs in {pdb_id}")
        return results

    logger.info(f"  {pdb_id}: {len(chain_pairs)} pair(s)  [{exp_method}, {resolution}Å]")

    raw_pdb = download_pdb(pdb_id, raw_dir)
    if raw_pdb is None:
        return results

    for ua, ca, ub, cb in chain_pairs:
        sifts_map_a = build_resnum_map(sifts.get(ca, []))
        sifts_map_b = build_resnum_map(sifts.get(cb, []))
        if not sifts_map_a:
            continue

        result = extract_renumber_save(
            raw_pdb, ca, cb, sifts_map_a, sifts_map_b,
            ua, ub, pdb_id, output_dir, min_sifts_coverage,
        )
        if result is None:
            continue

        out_path, meta = result
        meta['resolution_angstroms'] = resolution
        meta['experimental_method']  = exp_method
        results.append(meta)
        logger.info(
            f"    Wrote {out_path.name} "
            f"({meta['n_residues_a_mapped']}A/{meta['n_residues_b_mapped']}B res, "
            f"covA={meta['sifts_coverage_a']:.2f})"
        )

    return results

# =============================================================================
# METADATA OUTPUT
# =============================================================================

def write_metadata(records: List[Dict], output_csv: Path) -> None:
    if not records:
        logger.warning("No records to write")
        return
    fields = [
        'pdb_id','uniprot_a','chain_a_orig','uniprot_b','chain_b_orig',
        'resolution_angstroms','experimental_method',
        'n_residues_a_mapped','n_residues_b_mapped',
        'sifts_coverage_a','sifts_coverage_b','output_file',
    ]
    with open(output_csv, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for rec in records:
            w.writerow({k: rec.get(k, '') for k in fields})
    logger.info(f"Wrote metadata → {output_csv}")

# =============================================================================
# MAIN
# =============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Fetch RCSB PDB heterocomplex structures for a set of UniProt IDs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    inp = ap.add_mutually_exclusive_group(required=True)
    inp.add_argument('--uniprot-ids',      metavar='IDS',  help='Comma-separated UniProt IDs')
    inp.add_argument('--uniprot-ids-file', metavar='FILE', help='File with one UniProt ID per line')
    inp.add_argument('--scores-file',      metavar='TSV',  help='Scores TSV; IDs from --uniprot-col')

    ap.add_argument('--output-dir',            default='data/interactions/pdb_controls/')
    ap.add_argument('--resolution-cutoff',     type=float, default=3.5,
                    help='Resolution cutoff for X-ray structures (default: 3.5Å)')
    ap.add_argument('--em-resolution-cutoff',  type=float, default=4.5,
                    help='Resolution cutoff for EM structures (default: 4.5Å)')
    ap.add_argument('--experimental-methods',  default='X-RAY DIFFRACTION,ELECTRON MICROSCOPY')
    ap.add_argument('--min-sifts-coverage',    type=float, default=0.5)
    ap.add_argument('--min-partner-length',    type=int,   default=10)
    ap.add_argument('--max-per-pair',          type=int,   default=5)
    ap.add_argument('--threads',               type=int,   default=20,
                    help='Parallel threads for per-entry processing (default: 20)')
    ap.add_argument('--extra-pdb-ids',         default='',
                    help='Comma-separated PDB IDs to force-include, bypassing RCSB search '
                         '(e.g. 3KUC,1MW4,6EPL)')
    ap.add_argument('--uniprot-aliases',       default='',
                    help='Comma-separated ALIAS:CANONICAL pairs to normalize non-canonical '
                         'UniProt IDs in SIFTS (e.g. Q6VAB6:P15056,Q8IVT5:P15056)')
    ap.add_argument('--uniprot-col',  default='uniprot_id')
    ap.add_argument('--log-level',    default='INFO',
                    choices=['DEBUG','INFO','WARNING','ERROR'])
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
    )

    method_allowlist = [m.strip().upper() for m in args.experimental_methods.split(',')]
    output_dir = Path(args.output_dir)
    raw_dir    = output_dir / '_raw'

    # Parse UniProt aliases (e.g. Q6VAB6:P15056 → RCSB indexes some BRAF constructs
    # under non-canonical accessions instead of canonical P15056)
    aliases: Dict[str, str] = {}
    if args.uniprot_aliases:
        for pair in args.uniprot_aliases.split(','):
            pair = pair.strip()
            if ':' in pair:
                alias, canonical = pair.split(':', 1)
                aliases[alias.strip()] = canonical.strip()
    if aliases:
        logger.info(f"UniProt aliases: {aliases}")

    scored_uniprots = parse_uniprot_ids(args)

    # Collect all pdb_id → set-of-scored-uniprots-that-found-it mappings
    all_pdb_ids: Dict[str, set] = {}
    for uid in scored_uniprots:
        logger.info(f"\nQuerying: {uid}")
        pdb_ids = query_rcsb_for_uniprot(
            uid, args.resolution_cutoff, args.em_resolution_cutoff, method_allowlist
        )
        for pid in pdb_ids:
            all_pdb_ids.setdefault(pid, set()).add(uid)

    # Force-include extra PDB IDs specified on the command line (bypass search filters)
    if args.extra_pdb_ids:
        for pid in args.extra_pdb_ids.split(','):
            pid = pid.strip().upper()
            if pid:
                logger.info(f"  Force-including extra PDB ID: {pid}")
                all_pdb_ids.setdefault(pid, set())

    unique_pdb_ids = sorted(all_pdb_ids)
    logger.info(f"\n{len(unique_pdb_ids)} unique PDB entries to process (after search filters)")

    # Process entries in parallel — collect ALL results first, without cap
    all_raw_metadata: List[Dict] = []

    def worker(pdb_id):
        return process_entry(
            pdb_id, scored_uniprots, raw_dir, output_dir,
            args.min_partner_length, args.min_sifts_coverage,
            aliases=aliases,
        )

    with ThreadPoolExecutor(max_workers=args.threads) as pool:
        futures = {pool.submit(worker, pid): pid for pid in unique_pdb_ids}
        for future in as_completed(futures):
            pid = futures[future]
            try:
                records = future.result()
            except Exception as e:
                logger.warning(f"  Error processing {pid}: {e}")
                continue
            all_raw_metadata.extend(records)

    # Sort by resolution ascending (None/missing last) so better-resolved
    # structures fill the max-per-pair cap slots first
    all_raw_metadata.sort(
        key=lambda m: (
            m['resolution_angstroms'] is None,
            m['resolution_angstroms'] if m['resolution_angstroms'] is not None else 999,
        )
    )

    # Apply max-per-pair cap after sorting
    pair_counts: Dict[Tuple[str, str], int] = {}
    all_metadata: List[Dict] = []
    for meta in all_raw_metadata:
        pair_key = (meta['uniprot_a'], meta['uniprot_b'])
        if pair_counts.get(pair_key, 0) >= args.max_per_pair:
            logger.debug(
                f"  max-per-pair reached for {pair_key}, skipping {meta['pdb_id']}"
            )
            continue
        pair_counts[pair_key] = pair_counts.get(pair_key, 0) + 1
        all_metadata.append(meta)

    metadata_csv = output_dir / "pdb_metadata.csv"
    write_metadata(all_metadata, metadata_csv)

    # Remove any PDB files written to disk that didn't make the cap.
    # extract_renumber_save writes files before the cap is applied, so we
    # clean up the extras here to keep the output directory in sync with
    # the metadata CSV (and avoid downstream tools processing stale files).
    kept_files = {m['output_file'] for m in all_metadata}
    n_removed = 0
    for pdb_file in output_dir.glob("*.pdb"):
        if pdb_file.name not in kept_files:
            pdb_file.unlink()
            n_removed += 1
    if n_removed:
        logger.info(f"Removed {n_removed} uncapped PDB file(s) not in metadata")

    logger.info(
        f"\nDone. {len(all_metadata)} chain pairs from "
        f"{len(set(r['pdb_id'] for r in all_metadata))} PDB entries."
    )


if __name__ == '__main__':
    main()
