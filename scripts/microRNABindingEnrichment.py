# microRNA Binding Enrichment Analysis
# Updated to match motifEnrichment.py style with length distribution matching and Mann-Whitney U tests

import argparse
import pandas as pd
import yaml
import os
from typing import Tuple
import random
from collections import Counter
import numpy as np
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
import tqdm

def split_regions_to_lengths_proportional(control_df: pd.DataFrame, motif_lengths, random_state: int = 42):
    """Split control regions into segments whose length distribution mirrors motif_lengths.
    Strategy:
      - Maintain empirical proportion of selected lengths close to motif length frequencies.
      - For each region, iteratively allocate a length that fits remaining span.
      - Lookahead avoids creating an unusable tail (0 < tail < min_len).
      - Selection favors lengths currently underrepresented relative to motif distribution.
    """
    if control_df.empty or not motif_lengths:
        return control_df.copy()
    rnd = random.Random(random_state)
    counts = Counter(motif_lengths)  # motif length multiset
    total_motif_segments = len(motif_lengths)
    motif_freq = {L: counts[L] / total_motif_segments for L in counts}
    min_len = min(counts)
    used = Counter()
    total_segments = 0
    records = []

    for _, row in tqdm.tqdm(control_df.iterrows(), desc="Splitting control regions", total=len(control_df)):
        region_start = int(row['start'])
        region_end = int(row['end'])
        remaining = region_end - region_start
        current_start = region_start
        if remaining < min_len:
            continue  # skip too-small region
        while remaining >= min_len:
            # Candidate lengths that fit
            candidates = [L for L in counts if L <= remaining]
            if not candidates:
                break
            # Apply lookahead feasibility: avoid leaving an unusable tail
            feasible = [L for L in candidates if (remaining - L == 0) or (remaining - L >= min_len)]
            if feasible:
                candidates = feasible
            # Rank candidates by how underrepresented they would be after selection
            best_list = []
            best_score = None
            for L in candidates:
                new_prop = (used[L] + 1) / (total_segments + 1)
                imbalance = new_prop - motif_freq[L]
                # Prefer imbalance <= 0 (still not exceeding expected). Among those, closest to 0.
                if imbalance <= 0:
                    score = -imbalance
                    tag = (0, score)  # category 0 = non-overrepresented
                else:
                    score = imbalance
                    tag = (1, score)  # category 1 = overrepresented
                if best_score is None or tag < best_score:
                    best_score = tag
                    best_list = [L]
                elif tag == best_score:
                    best_list.append(L)
            # Random tie-break within best category
            chosen = rnd.choice(best_list)
            # Emit segment
            new_row = row.copy()
            new_row['orig_start'] = row['start']
            new_row['orig_end'] = row['end']
            new_row['start'] = current_start
            new_row['end'] = current_start + chosen
            new_row['length'] = chosen
            records.append(new_row)
            # Update trackers
            used[chosen] += 1
            total_segments += 1
            current_start += chosen
            remaining -= chosen
            if remaining < min_len:
                break
    if not records:
        return control_df.copy()
    return pd.DataFrame(records)


def overlap(interest_df: pd.DataFrame, mirna_df: pd.DataFrame) -> pd.DataFrame:
    """Compute overlaps between interest regions (expected 3'UTR) and microRNA predictions.
    Coordinates in the predictions (UTR start/end) are relative to the 3'UTR start, so we
    shift interest coordinates by sepEnd to be on the same scale.
    """
    if interest_df.empty:
        return pd.DataFrame(columns=['trID', 'geneSymbol', 'start', 'end', 'microRNA'])

    # Derive 3'UTR-relative coordinates
    interest_df = interest_df.copy()
    interest_df['corStart'] = interest_df['start'] - interest_df['sepEnd']
    interest_df['corEnd'] = interest_df['end'] - interest_df['sepEnd']
    # Keep only transcripts present in interest set
    mirna_df = mirna_df[mirna_df['Transcript ID'].isin(interest_df['trID'].unique())]

    hits = []
    for _, row in mirna_df.iterrows():
        micro_rna_len = row['UTR end'] - row['UTR start']
        # Candidate regions for this transcript
        subset = interest_df[interest_df['trID'] == row['Transcript ID']]
        if subset.empty:
            continue
        for _, region in subset.iterrows():
            overlap_start = max(region['corStart'], row['UTR start'])
            overlap_end = min(region['corEnd'], row['UTR end'])
            if overlap_end - overlap_start > int(micro_rna_len / 2):
                hits.append({
                    'trID': region['trID'],
                    'geneSymbol': row['Gene Symbol'],
                    'start': overlap_start,
                    'end': overlap_end,
                    'microRNA': row['miR Family']
                })
    return pd.DataFrame(hits)


def split_by_utr(df_regions: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return two DataFrames: 5'UTR (start <= sepStart) and 3'UTR (start > sepStart)."""
    if df_regions.empty:
        return df_regions.copy(), df_regions.copy()
    five = df_regions[df_regions['start'] <= df_regions['sepStart']].copy()
    three = df_regions[df_regions['start'] > df_regions['sepStart']].copy()
    five['length'] = five['end'] - five['start']
    three['length'] = three['end'] - three['start']
    return five, three


def count_mirna_overlaps_per_region(regions_df: pd.DataFrame, mirna_df: pd.DataFrame) -> pd.DataFrame:
    """Build a count matrix: rows are region indices, columns are microRNA families.
    Each cell contains the count of overlapping microRNA binding sites for that region.
    """
    if regions_df.empty:
        return pd.DataFrame()
    
    # Get all unique microRNA families
    all_mirnas = sorted(mirna_df['miR Family'].unique())
    
    # Create descriptive index
    regions_df = regions_df.copy()
    regions_df.index = regions_df['trID'] + '_' + regions_df['start'].astype(str) + '-' + regions_df['end'].astype(str)
    
    # Initialize matrix
    count_matrix = pd.DataFrame(0, index=regions_df.index, columns=all_mirnas)
    
    # Get overlaps using the existing overlap function
    overlaps = overlap(regions_df, mirna_df)
    
    # Fill in counts
    if not overlaps.empty:
        for _, hit in overlaps.iterrows():
            # Find matching region indices for this transcript
            matching_indices = [idx for idx in count_matrix.index if idx.startswith(hit['trID'] + '_')]
            for idx in matching_indices:
                # Parse the region coordinates from the index
                parts = idx.split('_')[-1].split('-')
                region_start = int(parts[0])
                region_end = int(parts[1])
                
                # Get the sepEnd for this transcript to convert coordinates
                region_row = regions_df[regions_df.index == idx].iloc[0]
                hit_abs_start = hit['start'] + region_row['sepEnd']
                hit_abs_end = hit['end'] + region_row['sepEnd']
                
                # Check if this hit overlaps with this specific region
                if (hit_abs_start < region_end and hit_abs_end > region_start):
                    if hit['microRNA'] in count_matrix.columns:
                        count_matrix.loc[idx, hit['microRNA']] += 1
    
    return count_matrix


def run_multi(args, yaml_params):
    """Run microRNA enrichment analysis with length-matched controls and Mann-Whitney U tests."""
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load microRNA predictions
    mirna_df = pd.read_csv(args.mirna_file, sep='\t')
    mirna_filtered = mirna_df[mirna_df['Species ID'] == 7227]
    print(f"microRNA predictions (species=7227): {len(mirna_filtered)}")
    
    # Count unique microRNA families
    unique_mirnas = mirna_filtered['miR Family'].nunique()
    print(f"Number of unique microRNA families: {unique_mirnas}")
    
    # Load high and low SHAP regions
    highSHAP = pd.read_csv(args.highSHAP, sep='\t', engine='pyarrow', dtype_backend='pyarrow')
    highControl = pd.read_csv(args.highControl, sep='\t', engine='pyarrow', dtype_backend='pyarrow')
    lowSHAP = pd.read_csv(args.lowSHAP, sep='\t', engine='pyarrow', dtype_backend='pyarrow')
    lowControl = pd.read_csv(args.lowControl, sep='\t', engine='pyarrow', dtype_backend='pyarrow')
    
    # Split by UTR
    highSHAP_5utr, highSHAP_3utr = split_by_utr(highSHAP)
    highControl_5utr, highControl_3utr = split_by_utr(highControl)
    lowSHAP_5utr, lowSHAP_3utr = split_by_utr(lowSHAP)
    lowControl_5utr, lowControl_3utr = split_by_utr(lowControl)
    
    # Generate matched control sets for 3'UTR using length distribution matching
    print("Generating length-matched control sets for 3'UTR...")
    highControl_3utr_matched = split_regions_to_lengths_proportional(
        highControl_3utr, highSHAP_3utr['length'].astype(int).tolist()
    )
    lowControl_3utr_matched = split_regions_to_lengths_proportional(
        lowControl_3utr, lowSHAP_3utr['length'].astype(int).tolist()
    )
    
    print("Matched control segment counts (proportional):",
          "pos3", len(highSHAP_3utr),
          "pos3_control", len(highControl_3utr_matched),
          "neg3", len(lowSHAP_3utr),
          "neg3_control", len(lowControl_3utr_matched))
    
    # Build count matrices for each set
    print("\nBuilding microRNA count matrices...")
    pos3 = count_mirna_overlaps_per_region(highSHAP_3utr, mirna_filtered)
    pcontrol3 = count_mirna_overlaps_per_region(highControl_3utr_matched, mirna_filtered)
    neg3 = count_mirna_overlaps_per_region(lowSHAP_3utr, mirna_filtered)
    ncontrol3 = count_mirna_overlaps_per_region(lowControl_3utr_matched, mirna_filtered)
    
    # # Save count matrices
    # print("\nSaving count matrices...")
    # pos3.to_parquet(f"{args.output_dir}/pos3.parquet")
    # pcontrol3.to_parquet(f"{args.output_dir}/pcontrol3.parquet")
    # neg3.to_parquet(f"{args.output_dir}/neg3.parquet")
    # ncontrol3.to_parquet(f"{args.output_dir}/ncontrol3.parquet")
    
    # Perform Mann-Whitney U tests
    print("\nPerforming Mann-Whitney U tests...")
    test_pairs = [
        (pos3, pcontrol3, 'pos3'),
        (neg3, ncontrol3, 'neg3')
    ]
    
    results = {}
    for motif_df, control_df, pair_name in tqdm.tqdm(test_pairs, desc="Running tests"):
        pair_results = {}
        for col in motif_df.columns:
            if col in control_df.columns:
                motif_mean = motif_df[col].mean()
                control_mean = control_df[col].mean()
                if motif_mean == 0 and control_mean == 0:
                    continue  # skip uninformative
                stat, p = mannwhitneyu(motif_df[col], control_df[col], alternative='two-sided')
                pair_results[col] = {
                    'motif_mean': motif_mean,
                    'control_mean': control_mean,
                    'statistic': stat,
                    'p_value': p
                }
        # Create DataFrame for this pair
        df_results = pd.DataFrame.from_dict(pair_results, orient='index')
        results[pair_name] = df_results
    
    # Apply FDR correction and save results
    print("\nApplying FDR correction and saving results...")
    for k in results.keys():
        results[k]['p_adj'] = multipletests(results[k]['p_value'], method='fdr_bh')[1]
        results[k].sort_values('p_value', inplace=True)
        results[k].to_csv(f"{args.output_dir}/{args.name}_{k}_mannwhitneyu_results.csv")
        print(f"\n{k}:")
        print(results[k][results[k]['p_adj'] < 0.07])


def main():
    parser = argparse.ArgumentParser(
        description='Analyze microRNA binding enrichment with length-matched controls and Mann-Whitney U tests.'
    )
    parser.add_argument('--params', type=str, required=True, help='Path to params YAML file')
    parser.add_argument('--name', type=str, required=False, help='Study name')
    parser.add_argument('--mirna_file', type=str, required=False, 
                        help='Path to microRNA predictions file')
    parser.add_argument('--highSHAP', type=str, required=False, help='Path to highSHAP file')
    parser.add_argument('--highControl', type=str, required=False, help='Path to highControl file')
    parser.add_argument('--lowSHAP', type=str, required=False, help='Path to lowSHAP file')
    parser.add_argument('--lowControl', type=str, required=False, help='Path to lowControl file')
    parser.add_argument('--output_dir', type=str, required=False, help='Path to output directory')
    
    args = parser.parse_known_args()[0]
    
    # Read parameters from YAML file
    yaml_params = None
    if args.params:
        with open(args.params, 'r') as file:
            yaml_params = yaml.safe_load(file)
            if 'microRNABindingEnrichment' in yaml_params:
                for key, value in yaml_params['microRNABindingEnrichment'].items():
                    parser.set_defaults(**{key: value})
    
    args = parser.parse_args()
    
    # Check if we have the new-style arguments
    multi_mode = all([
        args.highSHAP,
        args.highControl,
        args.lowSHAP,
        args.lowControl,
        args.name
    ])
    
    if multi_mode:
        run_multi(args, yaml_params)
    else:
        raise ValueError("Missing required arguments. Need either (highSHAP, highControl, lowSHAP, lowControl, name).")


if __name__ == '__main__':
    main()