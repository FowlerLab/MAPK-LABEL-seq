#!/usr/bin/env python
"""
Add homodimer control structures that ``fetch_pdb_controls.py`` cannot reach.

Why this exists
---------------
``enumerate_chain_pairs`` deduplicates at the entity level: for a homodimer it
emits exactly ONE pair per entity, ``chains_i[0], chains_i[1]``, ranked by
SIFTS-mapped span. That is the right call for a two-chain crystal, but in a
crystal with four copies of the same protein the chosen pair need not be the
one that forms the biological dimer. Two consequences were found in the BRAF
set:

* **3NY5** — the pair chosen (B/C) touch at 2 and 1 residues and were dropped
  by the >= 3-interface-position rule, while the crystal's real contact is A-B
  at 18/19 residues.
* **4MNE** (BRAF:MEK1) — its two BRAF-BRAF dimers, B-C (17/22 side-to-side core
  residues) and F-G (15/22), were never enumerated at all, because the entry's
  BRAF entity was consumed by the BRAF/MEK1 heterocomplex pair.

**6UAN** is a third case with a different cause: ``enumerate_chain_pairs`` does
emit its BRAF-BRAF pair today, and the pair extracts cleanly (SIFTS coverage
1.00/1.00), yet only the BRAF/14-3-3 pair reached ``pdb_metadata.csv``. The
historical reason is not recoverable from the artifacts on disk.

Rather than modify ``fetch_pdb_controls.py``, this driver reuses its functions and lets the chain pair be
named explicitly, so the choice is recorded rather than left to a ranking that
has already drifted once -- the pairs in ``pdb_metadata.csv`` (3NY5 ``_BC``,
7K0V ``_CA``, 8C7X ``_BA``) no longer match what the current code returns
(A-B in all three cases).

Chain pairs are chosen as the pair with the largest interface that carries the
side-to-side (R509) dimer core, measured in UniProt numbering at a 6 A C-beta
cutoff -- the same contact definition ``main.py`` uses.

DELIBERATE DEPARTURE FROM THE SELECTION POLICY -- READ BEFORE RE-RUNNING
------------------------------------------------------------------------
``fetch_pdb_controls.py`` would not select 4MNE for BRAF-BRAF, for two reasons:

1. Entity-level deduplication treats a crystal's repeated copies as redundancy
   and keeps one representative pair per entry. The doc's own example is
   "4MNE MEK1-BRAF tetramer 10 -> 1 pair", and that one pair went to BRAF-MEK1.
2. ``--max-per-pair`` (default 5) is applied after sorting by resolution, and
   BRAF-BRAF already has exactly five better-resolved entries: 8C7X 1.65,
   8C7Y 1.65, 7K0V 1.93, 5ITA 1.95, 3NY5 1.99 A. 4MNE at 2.85 A ranks sixth.

4MNE is added anyway because resolution is not the criterion that matters for
the BRAF dimer figure: 4MNE captures the dimer with MEK1 bound, so it is the
signalling dimer rather than a crystal form of the isolated kinase domain, and
its 9 interface residues are a strict subset of 8C7X's 14. That is a biological
argument the resolution-ranked cap cannot express.

Consequence: re-running ``fetch_pdb_controls.py`` will delete the file this
script writes, because unselected files are pruned to keep pdb_controls/ in sync
with pdb_metadata.csv. Re-run THIS script afterwards, then main.py.

After running this, ``main.py`` must be re-run so the new structures get
interface/control positions and statistics; nothing downstream sees them until
then. Note that adding rows shifts the BH-FDR denominator slightly for every
interaction in the same assay, so borderline calls elsewhere can move.

Usage
-----
    python add_homodimer_controls.py --dry-run
    python add_homodimer_controls.py                      # the curated set
    python add_homodimer_controls.py --pdb 4MNE --chains B,C --uniprot P15056
"""
from __future__ import annotations

import argparse
import csv
import logging
import shutil
import sys
from pathlib import Path

import fetch_pdb_controls as F

logger = logging.getLogger('add_homodimer_controls')

# pdb_id -> (chain_a, chain_b, uniprot, why this pair)
CURATED = {
    '4MNE': ('B', 'C', 'P15056',
             'BRAF:MEK1 complex; B-C is the larger of two BRAF dimers in the AU '
             '(17/22 core residues vs 15/22 for F-G). TRUE WILD TYPE: zero '
             'conflict records in struct_ref_seq_dif, where 8C7X, 8C7Y, 6U2H and '
             '7K0V all share a 14-substitution crystallisation construct that '
             'includes Q562R and L588N -- two of the eleven side-to-side core '
             'residues. Since DMS scores come from wild-type BRAF, 4MNE avoids '
             'measuring positions the crystal has mutated. Present only as '
             'BRAF-MEK1 until now.'),
    '6UAN': ('B', 'C', 'P15056',
             'Dimeric B-Raf:14-3-3 cryo-EM complex, 3.9 A; 17/19 interface '
             'residues carrying 6 side-to-side core residues per protomer. '
             'Asymmetric -- one protomer contributes 738-750 -- so it forms its '
             'own binding mode rather than merging with the 8C7X mode.'),
    '3NY5': ('A', 'B', 'P15056',
             'RBD-only construct (UniProt 153-235). A-B is 18/19 residues where '
             'the previously selected B-C pair was 2/1 and got skipped. NOT the '
             'kinase-domain dimer -- include only to document the RBD contact.'),
}
DEFAULT_SET = ['4MNE']

METADATA_FIELDS = [
    'pdb_id', 'uniprot_a', 'chain_a_orig', 'uniprot_b', 'chain_b_orig',
    'resolution_angstroms', 'experimental_method',
    'n_residues_a_mapped', 'n_residues_b_mapped',
    'sifts_coverage_a', 'sifts_coverage_b', 'output_file',
]


def existing_rows(metadata_csv: Path):
    if not metadata_csv.exists():
        return []
    with open(metadata_csv, newline='') as fh:
        return list(csv.DictReader(fh))


def already_present(rows, pdb_id, ua, ub) -> bool:
    """A homodimer row for this entry and pair of accessions already exists."""
    return any(r['pdb_id'] == pdb_id and r['uniprot_a'] == ua and
               r['uniprot_b'] == ub for r in rows)


def build_one(pdb_id, chain_a, chain_b, uniprot, output_dir: Path,
              min_sifts_coverage: float, dry_run: bool):
    entities = F.get_entry_entities(pdb_id)
    if not entities:
        logger.error(f'{pdb_id}: entity fetch failed')
        return None
    resolution = entities[0].get('resolution_angstroms')
    exp_method = entities[0].get('experimental_method', '')

    raw_dir = output_dir / '_raw'
    sifts = F.get_sifts_mapping(pdb_id, raw_dir=raw_dir)
    if not sifts:
        logger.error(f'{pdb_id}: no SIFTS mapping')
        return None
    for ch in (chain_a, chain_b):
        if ch not in sifts:
            logger.error(f'{pdb_id}: chain {ch} has no SIFTS mapping '
                         f'(mapped chains: {sorted(sifts)})')
            return None
        acc = sifts[ch][0]['uniprot_id']
        if acc != uniprot:
            logger.warning(f'{pdb_id} chain {ch}: SIFTS says {acc}, '
                           f'not {uniprot} -- proceeding, verify numbering')

    map_a = F.build_resnum_map(sifts.get(chain_a, []))
    map_b = F.build_resnum_map(sifts.get(chain_b, []))
    if not map_a or not map_b:
        logger.error(f'{pdb_id}: empty residue map for {chain_a} or {chain_b}')
        return None

    if dry_run:
        logger.info(f'{pdb_id} {chain_a}/{chain_b}: would write '
                    f'{uniprot}_{uniprot}_{pdb_id}_{chain_a}{chain_b}.pdb '
                    f'({len(map_a)}/{len(map_b)} mapped residues, '
                    f'{resolution} A {exp_method})')
        return None

    raw_pdb = F.download_pdb(pdb_id, raw_dir)
    if raw_pdb is None:
        logger.error(f'{pdb_id}: raw download failed')
        return None

    result = F.extract_renumber_save(
        raw_pdb, chain_a, chain_b, map_a, map_b,
        uniprot, uniprot, pdb_id, output_dir, min_sifts_coverage,
    )
    if result is None:
        logger.error(f'{pdb_id} {chain_a}/{chain_b}: extract_renumber_save '
                     f'declined (likely below {min_sifts_coverage} coverage)')
        return None
    out_path, meta = result
    meta['resolution_angstroms'] = resolution
    meta['experimental_method'] = exp_method
    logger.info(f'{pdb_id}: wrote {out_path.name} '
                f"({meta['n_residues_a_mapped']}A/{meta['n_residues_b_mapped']}B "
                f"residues, covA={meta['sifts_coverage_a']})")
    return meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--output-dir', type=Path, default=Path('data/interactions/pdb_controls'))
    ap.add_argument('--pdb', action='append', default=[],
                    help='PDB ID (repeatable). Defaults to the curated set.')
    ap.add_argument('--chains', action='append', default=[],
                    help='Chain pair "A,B" for the matching --pdb')
    ap.add_argument('--uniprot', action='append', default=[],
                    help='UniProt accession for the matching --pdb')
    ap.add_argument('--min-sifts-coverage', type=float, default=0.5,
                    help='Match fetch_pdb_controls.py (default: 0.5)')
    ap.add_argument('--dry-run', action='store_true',
                    help='Report what would be written; touch nothing')
    ap.add_argument('--log-level', default='INFO')
    args = ap.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format='%(levelname)s %(message)s')

    # Build the work list: explicit --pdb/--chains/--uniprot, else CURATED.
    jobs = []
    if args.pdb:
        for i, pid in enumerate(args.pdb):
            pid = pid.upper()
            if i < len(args.chains):
                ca, cb = [c.strip() for c in args.chains[i].split(',')]
            elif pid in CURATED:
                ca, cb = CURATED[pid][0], CURATED[pid][1]
            else:
                logger.error(f'{pid}: no --chains given and not in CURATED')
                return 2
            uni = (args.uniprot[i] if i < len(args.uniprot)
                   else CURATED.get(pid, (None, None, 'P15056'))[2])
            jobs.append((pid, ca, cb, uni))
    else:
        jobs = [(p, CURATED[p][0], CURATED[p][1], CURATED[p][2])
                for p in DEFAULT_SET]

    metadata_csv = args.output_dir / 'pdb_metadata.csv'
    rows = existing_rows(metadata_csv)
    new_meta = []
    for pid, ca, cb, uni in jobs:
        if already_present(rows, pid, uni, uni):
            logger.info(f'{pid}: {uni}/{uni} row already in pdb_metadata.csv, skipping')
            continue
        if pid in CURATED:
            logger.info(f'{pid} {ca}/{cb}: {CURATED[pid][3]}')
        meta = build_one(pid, ca, cb, uni, args.output_dir,
                         args.min_sifts_coverage, args.dry_run)
        if meta:
            new_meta.append(meta)

    if not new_meta:
        logger.info('Nothing written.')
        return 0

    # Append, never rewrite: pdb_metadata.csv is the record for every structure
    # in the set, so a full rewrite here would discard the rest of it.
    backup = metadata_csv.with_suffix('.csv.bak')
    shutil.copy2(metadata_csv, backup)
    logger.info(f'Backed up {metadata_csv.name} -> {backup.name}')
    with open(metadata_csv, 'a', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=METADATA_FIELDS)
        for meta in new_meta:
            w.writerow({k: meta.get(k, '') for k in METADATA_FIELDS})
    logger.info(f'Appended {len(new_meta)} row(s) to {metadata_csv}')
    logger.info('Re-run main.py so the new structures get interface/control '
                'positions and statistics.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
