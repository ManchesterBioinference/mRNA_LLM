# Read the file, skipping the first two lines that start with #
# %%
import pandas as pd
import os
from Bio import SeqIO
import argparse
import yaml

parser = argparse.ArgumentParser(description='Process motif enrichment analysis.')
parser.add_argument("--params", type=str, required=True, help="Path to the extra features .csv file")
parser.add_argument('--name', type=str, required=False, help='Study name')
parser.add_argument('--motifLocations', type=str, required=False, help='Path to MAST output file')
parser.add_argument('--highSHAP', type=str, required=False, help='Path to highSHAP file')
parser.add_argument('--highControl', type=str, required=False, help='Path to highControl file')
parser.add_argument('--lowSHAP', type=str, required=False, help='Path to lowSHAP file')
parser.add_argument('--lowControl', type=str, required=False, help='Path to lowControl file')
parser.add_argument('--output_dir', type=str, required=False, help='Path to output directory')
args = parser.parse_known_args()[0]

# Read parameters from YAML file
if args.params:
    with open(args.params, 'r') as file:
        yaml_params = yaml.safe_load(file)
        for key, value in yaml_params['motifEnrichment'].items():
            parser.set_defaults(**{key: value})

args = parser.parse_args()

def _reorder_lengths_median_alternating(lengths_list):
    """
    Reorders a list of numbers to have the middle number first,
    then alternates one lower and one higher, so that the median
    number is first and the extreme values come last.
    """
    if not lengths_list:
        return []

    # Sort the list to easily find the median and subsequent elements
    sorted_lengths = sorted(lengths_list)
    n = len(sorted_lengths)
    
    new_order_lengths = []
    
    # Determine the starting middle index.
    # For odd n, (n-1)//2 is the exact middle.
    # For even n, (n-1)//2 is the lower of the two middle elements, which will be picked first.
    mid_idx = (n - 1) // 2
    
    # Add the first middle element
    new_order_lengths.append(sorted_lengths[mid_idx])
    
    # Initialize pointers for elements to the left and right of the initial middle element
    l_ptr = mid_idx - 1
    r_ptr = mid_idx + 1
    
    # Loop until all elements from sorted_lengths are added to new_order_lengths,
    # alternating between picking from the left and right sides of the initial middle.
    while l_ptr >= 0 or r_ptr < n:
        # Add element from the left side (lower than current median elements)
        if l_ptr >= 0:
            new_order_lengths.append(sorted_lengths[l_ptr])
            l_ptr -= 1
        
        # Add element from the right side (higher than current median elements)
        if r_ptr < n: 
            new_order_lengths.append(sorted_lengths[r_ptr])
            r_ptr += 1
            
    return new_order_lengths

#try:
#    df = pd.read_csv('output/mast_out/dev_mast_results.txt', sep='\s+', skiprows=2, skipfooter=1, engine='python')
#except:
#    os.chdir('..')
#    df = pd.read_csv('output/mast_out/dev_mast_results.txt', sep='\s+', skiprows=2, skipfooter=1, engine='python')
df = pd.read_csv(args.motifLocations, sep='\s+', skiprows=2, skipfooter=1, engine='python')
df.columns = ['sequence_name', 'strand', 'id', 'alt_id', 'hit_start','hit_end','score','p_value']

# Count unique RBPs in alt_id column
unique_rbps = df['alt_id'].nunique()
print(f"Number of unique RBPs: {unique_rbps}")

# Create new column 'trID' by splitting sequence_name on '_' and taking index 0
df['trID'] = df['sequence_name'].str.split('_').str[0]

highSHAP = pd.read_csv(args.highSHAP, sep='\t',engine='pyarrow',dtype_backend = 'pyarrow')
highControl = pd.read_csv(args.highControl, sep='\t',engine='pyarrow',dtype_backend = 'pyarrow')
lowSHAP = pd.read_csv(args.lowSHAP, sep='\t',engine='pyarrow',dtype_backend = 'pyarrow')
lowControl = pd.read_csv(args.lowControl, sep='\t',engine='pyarrow',dtype_backend = 'pyarrow')

# --- New code: split each into 5' and 3' UTR dataframes and match length distributions ---

def split_by_utr(df_regions: pd.DataFrame):
    """Return two DataFrames: 5'UTR (start <= sepStart) and 3'UTR (start > sepStart)."""
    if df_regions.empty:
        return df_regions.copy(), df_regions.copy()
    five = df_regions[df_regions['start'] <= df_regions['sepStart']].copy()
    three = df_regions[df_regions['start'] > df_regions['sepStart']].copy()
    five['length'] = five['end'] - five['start']
    three['length'] = three['end'] - three['start']
    return five, three

highSHAP_5utr, highSHAP_3utr = split_by_utr(highSHAP)
highControl_5utr, highControl_3utr = split_by_utr(highControl)
lowSHAP_5utr, lowSHAP_3utr = split_by_utr(lowSHAP)
lowControl_5utr, lowControl_3utr = split_by_utr(lowControl)

# Helper to get reordered motif lengths

def get_reordered_lengths(motif_df: pd.DataFrame):
    if motif_df.empty:
        return []
    lengths = (motif_df['end'] - motif_df['start']).astype(int)
    lengths = lengths[lengths > 0].tolist()
    return _reorder_lengths_median_alternating(lengths)

pos5_lengths = get_reordered_lengths(highSHAP_5utr)
pos3_lengths = get_reordered_lengths(highSHAP_3utr)
neg5_lengths = get_reordered_lengths(lowSHAP_5utr)
neg3_lengths = get_reordered_lengths(lowSHAP_3utr)

print("Length counts (motif): pos5", len(pos5_lengths), "pos3", len(pos3_lengths), "neg5", len(neg5_lengths), "neg3", len(neg3_lengths))

# Function to split control regions into segments matching motif length distribution (proportion-controlled, lookahead)
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

    for _, row in tqdm.tqdm(control_df.iterrows()):
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
            # imbalance = ( (used[L]+1)/(total_segments+1) ) - motif_freq[L]
            best_list = []
            best_score = None
            for L in candidates:
                new_prop = (used[L] + 1) / (total_segments + 1)
                imbalance = new_prop - motif_freq[L]
                # Prefer imbalance <= 0 (still not exceeding expected). Among those, closest to 0.
                if imbalance <= 0:
                    score = -imbalance  # smaller magnitude preferred -> larger score negative? invert
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

# Generate matched control sets with new proportional splitter
highControl_5utr_matched = split_regions_to_lengths_proportional(highControl_5utr, highSHAP_5utr['length'].astype(int).tolist())
highControl_3utr_matched = split_regions_to_lengths_proportional(highControl_3utr, highSHAP_3utr['length'].astype(int).tolist())
lowControl_5utr_matched = split_regions_to_lengths_proportional(lowControl_5utr, lowSHAP_5utr['length'].astype(int).tolist())
lowControl_3utr_matched = split_regions_to_lengths_proportional(lowControl_3utr, lowSHAP_3utr['length'].astype(int).tolist())

# create a more descriptive index
for d in [highSHAP_5utr, highSHAP_3utr, highControl_5utr_matched, highControl_3utr_matched, lowSHAP_5utr, lowSHAP_3utr, lowControl_5utr_matched, lowControl_3utr_matched]:
    d.index = d['trID'] + '_'+ d['start'].astype(str) + '-' + d['end'].astype(str)

print("Matched control segment counts (proportional):",
      "pos5", len(highControl_5utr_matched),
      "pos3", len(highControl_3utr_matched),
      "neg5", len(lowControl_5utr_matched),
      "neg3", len(lowControl_3utr_matched))

# These new DataFrames are now available:
# highSHAP_5utr, highSHAP_3utr, lowSHAP_5utr, lowSHAP_3utr
# highControl_5utr_matched, highControl_3utr_matched, lowControl_5utr_matched, lowControl_3utr_matched
# -----------------------------------------------------------------------------

# %%
#import matplotlib.pyplot as plt
#
## Define the pairs
#pairs = [
#    (highSHAP_5utr, highControl_5utr_matched, "5'UTR Positive"),
#    (highSHAP_3utr, highControl_3utr_matched, "3'UTR Positive"),
#    (lowSHAP_5utr, lowControl_5utr_matched, "5'UTR Negative"),
#    (lowSHAP_3utr, lowControl_3utr_matched, "3'UTR Negative")
#]
#
#fig, axes = plt.subplots(4, 2, figsize=(12, 16))
#fig.suptitle('Histograms of Motif and Control Lengths')
#
#for i, (motif_df, control_df, title) in tqdm.tqdm(enumerate(pairs)):
#    motif_lengths = None
#    control_lengths = None
#    
#    if not motif_df.empty and 'length' in motif_df.columns:
#        motif_lengths = motif_df['length'].dropna()
#    
#    if not control_df.empty and 'length' in control_df.columns:
#        control_lengths = control_df['length'].dropna()
#    
#    # Compute shared bins and xlim if both have data
#    if motif_lengths is not None and control_lengths is not None:
#        all_lengths = pd.concat([motif_lengths, control_lengths])
#        overall_min = all_lengths.min()
#        overall_max = all_lengths.max()
#        bins = np.linspace(overall_min, overall_max, 21)  # 20 bins
#        xlim = (overall_min, overall_max)
#    elif motif_lengths is not None:
#        overall_min = motif_lengths.min()
#        overall_max = motif_lengths.max()
#        bins = np.linspace(overall_min, overall_max, 21)
#        xlim = (overall_min, overall_max)
#    elif control_lengths is not None:
#        overall_min = control_lengths.min()
#        overall_max = control_lengths.max()
#        bins = np.linspace(overall_min, overall_max, 21)
#        xlim = (overall_min, overall_max)
#    else:
#        continue  # Skip if no data
#    
#    if motif_lengths is not None:
#        axes[i, 0].hist(motif_lengths, bins=bins, alpha=0.7, color='blue', edgecolor='black')
#        axes[i, 0].set_title(f'Motif {title}')
#        axes[i, 0].set_xlabel('Length')
#        axes[i, 0].set_ylabel('Frequency')
#        axes[i, 0].set_xlim(xlim)
#    
#    if control_lengths is not None:
#        axes[i, 1].hist(control_lengths, bins=bins, alpha=0.7, color='red', edgecolor='black')
#        axes[i, 1].set_title(f'Control {title}')
#        axes[i, 1].set_xlabel('Length')
#        axes[i, 1].set_ylabel('Frequency')
#        axes[i, 1].set_xlim(xlim)
#
#plt.tight_layout()
#plt.show()

# %%
pos5 = pd.DataFrame(0, index=highSHAP_5utr.index, columns=df['alt_id'].unique())
pos3 = pd.DataFrame(0, index=highSHAP_3utr.index, columns=df['alt_id'].unique())
pcontrol5 = pd.DataFrame(0, index=highControl_5utr_matched.index, columns=df['alt_id'].unique())
pcontrol3 = pd.DataFrame(0, index=highControl_3utr_matched.index, columns=df['alt_id'].unique())
neg5 = pd.DataFrame(0, index=lowSHAP_5utr.index, columns=df['alt_id'].unique())
neg3 = pd.DataFrame(0, index=lowSHAP_3utr.index, columns=df['alt_id'].unique())
ncontrol5 = pd.DataFrame(0, index=lowControl_5utr_matched.index, columns=df['alt_id'].unique())
ncontrol3 = pd.DataFrame(0, index=lowControl_3utr_matched.index, columns=df['alt_id'].unique())


for regions, tmpDF in tqdm.tqdm(zip([highSHAP_5utr, highSHAP_3utr, highControl_5utr_matched, highControl_3utr_matched, lowSHAP_5utr,lowSHAP_3utr, lowControl_5utr_matched, lowControl_3utr_matched], [pos5, pos3, pcontrol5, pcontrol3, neg5, neg3, ncontrol5, ncontrol3])):
    for r in regions.iterrows():
        if r[1]['start'] > r[1]['sepStart']:
            id = r[1]['trID']+'_3utr'
            r[1]['start'] = r[1]['start'] - r[1]['sepEnd']
            r[1]['end'] = r[1]['end'] - r[1]['sepEnd']
        else:
            id = r[1]['trID']+'_5utr'
        if id in df['sequence_name'].values:
            tmp = df[df['sequence_name'] == id]
            for t in tmp.iterrows():
                if (t[1]['hit_start'] >= r[1]['start'] and t[1]['hit_start'] <= r[1]['end']) or (t[1]['hit_end'] >= r[1]['start'] and t[1]['hit_end'] <= r[1]['end']):
                    tmpDF.loc[r[0],t[1]['alt_id']] += 1
# %%
# List of (DataFrame, name) pairs for saving
df_pairs = [
    (pos5, 'pos5'),
    (pos3, 'pos3'),
    (pcontrol5, 'pcontrol5'),
    (pcontrol3, 'pcontrol3'),
    (neg5, 'neg5'),
    (neg3, 'neg3'),
    (ncontrol5, 'ncontrol5'),
    (ncontrol3, 'ncontrol3')
]

for d, name in df_pairs:
    d.to_parquet(f"{args.output_dir}/{name}.parquet")
# %%
# Define the pairs for Mann-Whitney U tests
test_pairs = [
    (pos5, pcontrol5, 'pos5'),
    (pos3, pcontrol3, 'pos3'),
    (neg5, ncontrol5, 'neg5'),
    (neg3, ncontrol3, 'neg3')
]

# Perform Mann-Whitney U test for each pair and column
results = {}
for motif_df, control_df, pair_name in tqdm.tqdm(test_pairs):
    pair_results = {}
    for col in motif_df.columns:
        if col in control_df.columns:
            motif_mean = motif_df[col].mean()
            control_mean = control_df[col].mean()
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

for k in results.keys():
    results[k]['p_adj'] = multipletests(results[k]['p_value'], method='fdr_bh')[1]
    results[k].sort_values('p_value', inplace=True)
    results[k].to_csv(f"{args.output_dir}/{args.name}_{k}_mannwhitneyu_results.csv")
    print(k)
    print(results[k][results[k]['p_adj'] < 0.07])



# # Optionally, print or save results
# for pair_name, pair_results in results.items():
#     print(f"\n{pair_name}:")
#     for col, res in pair_results.items():
#         print(f"  {col}: U={res['statistic']:.2f}, p={res['p_value']:.4e}")
# 
# # %%
# import matplotlib.pyplot as plt
# 
# # Plot histogram for MSI in pos5 and pcontrol5
# fig, axes = plt.subplots(1, 2, figsize=(10, 5))
# fig.suptitle('Histograms of MSI Counts in pos5 and pcontrol5')
# 
# # pos5 MSI
# gene = 'SHEP'
# if gene in pos5.columns:
#     axes[0].hist(pos5[gene], bins=20, alpha=0.7, color='blue', edgecolor='black')
#     axes[0].set_title(f'pos5 {gene}')
#     axes[0].set_xlabel('Count')
#     axes[0].set_ylabel('Frequency')
# 
# # pcontrol5 MSI
# if gene in pcontrol5.columns:
#     axes[1].hist(pcontrol5[gene], bins=20, alpha=0.7, color='red', edgecolor='black')
#     axes[1].set_title(f'pcontrol5 {gene}')
#     axes[1].set_xlabel('Count')
#     axes[1].set_ylabel('Frequency')
# 
# plt.tight_layout()
# plt.show()
# # %%
# 