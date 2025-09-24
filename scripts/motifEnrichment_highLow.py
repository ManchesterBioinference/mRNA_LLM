# Read the file, skipping the first two lines that start with #
# %%
import pandas as pd
import os
from Bio import SeqIO
import argparse
import yaml
import tqdm
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
import numpy as np

parser = argparse.ArgumentParser(description='Process motif enrichment analysis.')
parser.add_argument("--params", type=str, help="Path to the extra features .csv file")
parser.add_argument('--name', type=str, required=False, help='Study name')
parser.add_argument('--motifLocations', default='output/mast_out/train_mast_results.txt',type=str, required=False, help='Path to MAST output file')
parser.add_argument('--highTE', default= 'output/splitDecay/train_highDecay.fasta', type=str, required=False, help='Path to highTE file')
parser.add_argument('--lowTE', default = 'output/splitDecay/train_lowDecay.fasta', type=str, required=False, help='Path to lowTE file')
parser.add_argument('--output_dir', type=str, required=False, help='Path to output directory')
args = parser.parse_known_args()[0]
# %%

# Read parameters from YAML file
if args.params:
    with open(args.params, 'r') as file:
        yaml_params = yaml.safe_load(file)
        for key, value in yaml_params['motifEnrichment'].items():
            parser.set_defaults(**{key: value})

args = parser.parse_args()
#%%
df = pd.read_csv(args.motifLocations, sep='\s+', skiprows=2, skipfooter=1, engine='python')
df.columns = ['sequence_name', 'strand', 'id', 'alt_id', 'hit_start','hit_end','score','p_value']

# Count unique RBPs in alt_id column
unique_rbps = df['alt_id'].nunique()
print(f"Number of unique RBPs: {unique_rbps}")

# Create new column 'trID' by splitting sequence_name on '_' and taking index 0
df.index = df['sequence_name'].str.split('_').str[0]
df['utr'] = df['sequence_name'].str.split('_').str[1]
df.drop(columns=['sequence_name'], inplace=True)

highTE = [r for r in SeqIO.parse(args.highTE, "fasta")]
lowTE = [r for r in SeqIO.parse(args.lowTE, "fasta")]

high5Len = np.log(1+np.array([len(r.seq.split(',')[0]) for r in highTE])).tolist()
high3Len = np.log(1+np.array([len(r.seq.split(',')[1]) for r in highTE])).tolist()
low5Len =  np.log(1+np.array([len(r.seq.split(',')[0]) for r in lowTE]) ).tolist()
low3Len =  np.log(1+np.array([len(r.seq.split(',')[1]) for r in lowTE]) ).tolist()

## %%
#import matplotlib.pyplot as plt
#
## Compute shared bin edges across all lengths
#all_lens = high5Len + low5Len + high3Len + low3Len
#xmin, xmax = min(all_lens), max(all_lens)
#bins = np.linspace(xmin, xmax, 31)
#
## 5'UTR
#plt.hist(high5Len, alpha=0.5, label='highTE 5UTR', bins=bins)
#plt.hist(low5Len,  alpha=0.5, label='lowTE 5UTR',  bins=bins)
#plt.xlim(xmin, xmax)
#plt.legend()
#plt.show()
#
## 3'UTR
#plt.hist(high3Len, alpha=0.5, label='highTE 3UTR', bins=bins)
#plt.hist(low3Len,  alpha=0.5, label='lowTE 3UTR',  bins=bins)
#plt.xlim(xmin, xmax)
#plt.legend()
#plt.show()

# %%
highTE_ids = [record.description.split()[1] for record in highTE]
lowTE_ids = [record.description.split()[1] for record in lowTE]
# %%
high = df.loc[highTE_ids]
low = df.loc[lowTE_ids]
high5 = high[high['utr'] == '5utr']
high3 = high[high['utr'] == '3utr']
low5 = low[low['utr'] == '5utr']
low3 = low[low['utr'] == '3utr']
del high, low

# %%
high5Len = [len(r.seq.split(',')[0]) for r in highTE]
high3Len = [len(r.seq.split(',')[1]) for r in highTE]
low5Len =  [len(r.seq.split(',')[0]) for r in lowTE]
low3Len =  [len(r.seq.split(',')[1]) for r in lowTE]
#%%

# TODO - i need to make sure that the distribution of lengths are the same between high5TE and low5TE as well as high3TE and low3TE
def break_sequences(fullDF, TE_ids, lengths, medianLen=25):
    def break_sequences(fullDF: pd.DataFrame, TE_ids: list[str], lengths: list[int], medianLen: int = 25) -> tuple[pd.DataFrame, list[int]]:
        """
        Breaks sequences into segments of approximately medianLen length, avoiding regions defined by 'hit_start' and 'hit_end' in the DataFrame.

        This function processes a DataFrame containing sequence hits, breaks each sequence into segments while skipping hit regions,
        and updates the DataFrame's index to reflect the new segmented names. It also computes the lengths of these new segments.

        Parameters:
        - fullDF (pd.DataFrame): DataFrame with sequence hits, indexed by sequence IDs, containing columns 'hit_start' and 'hit_end'.
        - TE_ids (list[str]): List of sequence IDs corresponding to the lengths.
        - lengths (list[int]): List of total lengths for each sequence ID.
        - medianLen (int, optional): Target length for each segment. Defaults to 25.

        Returns:
        - tuple[pd.DataFrame, list[int]]: A tuple containing the modified DataFrame with updated index and a list of new segment lengths.
        """
    idsToLengths = {k:v for k,v in zip(TE_ids,lengths)}
    newLengths = []
    newIndex = []
    for id in fullDF.index.unique():
        tmp = fullDF.loc[[id]]
        ranges = []
        for _, row in tmp.iterrows():
            ranges += list(range(row['hit_start'], row['hit_end']))
        length = idsToLengths[id]
        indices = [x for x in range(length) if x not in ranges]
        idx = medianLen
        myBreaks = [0]
        while idx < length:
            # find the nearest available index
            nearest = min(indices, key=lambda x: abs(x - idx))
            myBreaks.append(nearest)
            idx = nearest
            idx += medianLen
        myBreaks.append(length+1)
        newLengths += [b - a for a, b in zip(myBreaks[:-1], myBreaks[1:])]

        pointer = 1 # skip the first break (0)
        newNames = []
        for name, row in tmp.iterrows():
            while row['hit_end'] > myBreaks[pointer]:
                pointer += 1
            newNames.append(name+f'_{myBreaks[pointer]}')
        newIndex += newNames
    fullDF.index = newIndex
    return(fullDF, newLengths)

high5, high5Len = break_sequences(high5, highTE_ids, high5Len)
# %%
high3, high3Len = break_sequences(high3, highTE_ids, high3Len)
low5, low5Len = break_sequences(low5, lowTE_ids, low5Len)
low3, low3Len = break_sequences(low3, lowTE_ids, low3Len)

# # %%
# import matplotlib.pyplot as plt
# #high5Len = np.log(1+np.array([len(r.seq.split(',')[0]) for r in highTE])).tolist()
# #high3Len = np.log(1+np.array([len(r.seq.split(',')[1]) for r in highTE])).tolist()
# #low5Len =  np.log(1+np.array([len(r.seq.split(',')[0]) for r in lowTE]) ).tolist()
# #low3Len =  np.log(1+np.array([len(r.seq.split(',')[1]) for r in lowTE]) ).tolist()
# 
# # Compute shared bin edges across all lengths
# all_lens = high5Len + low5Len + high3Len + low3Len
# xmin, xmax = min(all_lens), max(all_lens)
# bins = np.linspace(xmin, xmax, 31)
# 
# # 5'UTR
# plt.hist(high5Len, alpha=0.5, label='highTE 5UTR', bins=bins)
# plt.hist(low5Len,  alpha=0.5, label='lowTE 5UTR',  bins=bins)
# plt.xlim(xmin, xmax)
# plt.legend()
# plt.show()
# 
# # 3'UTR
# plt.hist(high3Len, alpha=0.5, label='highTE 3UTR', bins=bins)
# plt.hist(low3Len,  alpha=0.5, label='lowTE 3UTR',  bins=bins)
# plt.xlim(xmin, xmax)
# plt.legend()
# plt.show()

#%%

# Convert to matrix: index=sequence_name, columns=alt_id, values=count of 'utr'
high5_counts = ( high5.pivot_table(index = high5.index, columns='alt_id', values='utr', aggfunc='count', fill_value=0).astype(int))
high3_counts = ( high3.pivot_table( index=high3.index, columns='alt_id', values='utr', aggfunc='count', fill_value=0).astype(int))
low5_counts = ( low5.pivot_table( index=low5.index, columns='alt_id', values='utr', aggfunc='count', fill_value=0).astype(int))
low3_counts = ( low3.pivot_table( index=low3.index, columns='alt_id', values='utr', aggfunc='count', fill_value=0).astype(int))
# %%
# correct for differing sequence lengths
# high5LenDict = {k:v for k,v in zip([record.description.split()[1] for record in highTE], high5Len)}
# high3LenDict = {k:v for k,v in zip([record.description.split()[1] for record in highTE], high3Len)}
# low5LenDict = {k:v for k,v in zip([record.description.split()[1] for record in lowTE], low5Len)}
# low3LenDict = {k:v for k,v in zip([record.description.split()[1] for record in lowTE], low3Len)}
# 
# high5_counts['Len'] = [high5LenDict[ind] for ind in high5_counts.index]
# high3_counts['Len'] = [high3LenDict[ind] for ind in high3_counts.index]
# low5_counts['Len'] = [low5LenDict[ind] for ind in low5_counts.index]
# low3_counts['Len'] = [low3LenDict[ind] for ind in low3_counts.index]
# 
# high5_counts = high5_counts.drop(columns=['Len']).div(high5_counts['Len'], axis=0)
# high3_counts = high3_counts.drop(columns=['Len']).div(high3_counts['Len'], axis=0)
# low5_counts = low5_counts.drop(columns=['Len']).div(low5_counts['Len'], axis=0)
# low3_counts = low3_counts.drop(columns=['Len']).div(low3_counts['Len'], axis=0)


# %%

# Define the pairs for Mann-Whitney U tests
test_pairs = [
    (high5_counts, low5_counts, 'utr5'),
    (high3_counts, low3_counts, 'utr3'),
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
            #plt.hist(np.log(1+1000*motif_df[col]), alpha=0.5, label='motif', bins=30)
            #plt.hist(np.log(1+1000*control_df[col]), alpha=0.5, label='control', bins=16)
            #plt.show()
            #break
            pair_results[col] = {
                'motif_mean': motif_mean,
                'control_mean': control_mean,
                'statistic': stat,
                'p_value': p
            }

    #break
    # Create DataFrame for this pair
    df_results = pd.DataFrame.from_dict(pair_results, orient='index')
    results[pair_name] = df_results
#%%

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