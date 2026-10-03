#!/usr/bin/env python3
"""
Extract predictomes interaction structures for specified UniProt IDs.
Filters by SPOC score threshold and extracts matching files from tar archives.
"""

import csv
import subprocess
import os
import argparse
from pathlib import Path
from collections import defaultdict


def parse_uniprot_ids(uniprot_string: str) -> set:
    """Parse comma-separated UniProt IDs string into a set."""
    ids = set()
    for uid in uniprot_string.split(','):
        uid = uid.strip()
        if uid:
            ids.add(uid)
    return ids


def load_tar_indices(tar_dir: str) -> dict:
    """
    Load all index_data_split_*.csv files to build a mapping of
    complex_name -> tar file path.
    """
    index = {}
    index_files = sorted(Path(tar_dir).glob('index_data_split_*.csv'))

    for index_file in index_files:
        # Extract split number from filename (e.g., index_data_split_00.csv -> 00)
        split_num = index_file.stem.replace('index_data_split_', '')
        tar_file = Path(tar_dir) / f'data_split_{split_num}.tar'

        with open(index_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Build complex name from protein1, protein2, id
                complex_name = f"{row['protein1']}__{row['protein2']}__{row['id']}"
                index[complex_name] = str(tar_file)

    return index


def filter_predictomes_csv(csv_file: str, target_uniprots: set, spoc_threshold: float) -> list:
    """
    Filter predictomes CSV for interactions involving target UniProt IDs
    with SPOC score above threshold.

    Returns list of dicts with complex_name, uniprot_ids, spoc_score.
    """
    matches = []
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            uniprot_ids = row['uniprot_ids'].split(':')
            spoc_score = float(row['spoc_score'])

            # Check if any of the UniProt IDs match our targets
            if spoc_score >= spoc_threshold:
                if any(uid in target_uniprots for uid in uniprot_ids):
                    matches.append({
                        'complex_name': row['complex_name'],
                        'uniprot_ids': row['uniprot_ids'],
                        'spoc_score': spoc_score
                    })
    return matches


def extract_complexes_from_tar(tar_file: str, complex_names: list, output_dir: str):
    """
    Extract all files for multiple complexes from a single tar archive in one pass.
    This is more efficient than extracting one complex at a time.
    """
    tar_basename = Path(tar_file).stem  # e.g., "data_split_00"
    os.makedirs(output_dir, exist_ok=True)

    # Build wildcard patterns for all complexes
    patterns = [f"{tar_basename}/{name}*" for name in complex_names]

    # Use tar with multiple --wildcards patterns
    cmd = ['tar', '-xf', tar_file, '-C', output_dir, '--strip-components=1', '--wildcards']
    cmd.extend(patterns)

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0 and result.stderr:
        # Filter out "not found" warnings which are expected for some patterns
        errors = [line for line in result.stderr.strip().split('\n')
                  if 'Not found in archive' not in line and line.strip()]
        if errors:
            print(f"  Warnings: {'; '.join(errors)}")

    return len(complex_names)


def extract_predictomes(
    uniprot_ids: str,
    predictomes_csv: str,
    tar_dir: str,
    output_dir: str,
    spoc_threshold: float = 0.5,
    matches_only: bool = False,
    max_tar_files: int = None
) -> dict:
    """
    Main function to extract predictomes for specified UniProt IDs.

    Args:
        uniprot_ids: Comma-separated string of UniProt IDs
        predictomes_csv: Path to predictomes pair scores CSV
        tar_dir: Directory containing tar files and index files
        output_dir: Output directory for extracted files
        spoc_threshold: Minimum SPOC score (default 0.5)
        matches_only: If True, only find matches without extracting
        max_tar_files: If set, only process this many tar files (for testing)

    Returns:
        Dictionary with 'matches', 'extracted', 'not_found' counts
    """
    # Parse UniProt IDs
    target_uniprots = parse_uniprot_ids(uniprot_ids)
    print(f"Target UniProt IDs ({len(target_uniprots)}): {sorted(target_uniprots)}")

    # Filter predictomes CSV
    print(f"\nFiltering predictomes for SPOC >= {spoc_threshold}...")
    matches = filter_predictomes_csv(predictomes_csv, target_uniprots, spoc_threshold)
    print(f"Found {len(matches)} matching interactions")

    # Save matches to a file
    os.makedirs(output_dir, exist_ok=True)
    matches_file = Path(output_dir) / 'matching_interactions.csv'
    with open(matches_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['complex_name', 'uniprot_ids', 'spoc_score'])
        writer.writeheader()
        writer.writerows(matches)
    print(f"Wrote matches to {matches_file}")

    result = {
        'matches': len(matches),
        'extracted': 0,
        'not_found': 0
    }

    if matches_only:
        return result

    # Load tar indices
    print(f"\nLoading tar indices from {tar_dir}...")
    tar_index = load_tar_indices(tar_dir)
    print(f"Loaded index with {len(tar_index)} complexes")

    # Group matches by tar file for efficient extraction
    tar_to_complexes = defaultdict(list)
    not_found = 0

    for match in matches:
        complex_name = match['complex_name']
        if complex_name in tar_index:
            tar_file = tar_index[complex_name]
            tar_to_complexes[tar_file].append(complex_name)
        else:
            print(f"  WARNING: {complex_name} not found in tar index")
            not_found += 1

    print(f"\nGrouped {len(matches) - not_found} complexes across {len(tar_to_complexes)} tar files")

    # Sort tar files for consistent ordering
    tar_files_to_process = sorted(tar_to_complexes.keys())

    # Optionally limit number of tar files for testing
    if max_tar_files is not None:
        tar_files_to_process = tar_files_to_process[:max_tar_files]
        print(f"  (Limited to {max_tar_files} tar files for testing)")

    # Extract from each tar file
    print(f"\nExtracting files to {output_dir}...")
    extracted = 0

    for i, tar_file in enumerate(tar_files_to_process):
        complex_names = tar_to_complexes[tar_file]
        print(f"[{i+1}/{len(tar_files_to_process)}] Extracting {len(complex_names)} complexes from {Path(tar_file).name}...")
        num_extracted = extract_complexes_from_tar(tar_file, complex_names, output_dir)
        extracted += num_extracted

    print(f"\nDone! Extracted {extracted} complexes, {not_found} not found in index")

    result['extracted'] = extracted
    result['not_found'] = not_found

    return result


def main():
    parser = argparse.ArgumentParser(description='Extract predictomes for specified UniProt IDs')
    parser.add_argument('--uniprot-ids', required=True,
                        help='Comma-separated list of UniProt IDs (e.g., "P00533,P01116,P15056")')
    parser.add_argument('--predictomes-csv', required=True, help='Predictomes pair scores CSV')
    parser.add_argument('--tar-dir', required=True, help='Directory containing tar files and index files')
    parser.add_argument('--output-dir', required=True, help='Output directory for extracted files')
    parser.add_argument('--spoc-threshold', type=float, default=0.5, help='Minimum SPOC score (default: 0.5)')
    parser.add_argument('--matches-only', action='store_true', help='Only find matches, do not extract')
    parser.add_argument('--max-tar-files', type=int, default=None,
                        help='Maximum number of tar files to process (for testing)')

    args = parser.parse_args()

    extract_predictomes(
        uniprot_ids=args.uniprot_ids,
        predictomes_csv=args.predictomes_csv,
        tar_dir=args.tar_dir,
        output_dir=args.output_dir,
        spoc_threshold=args.spoc_threshold,
        matches_only=args.matches_only,
        max_tar_files=args.max_tar_files
    )


if __name__ == '__main__':
    main()
