import argparse
import pandas as pd
import yaml

def overlap(interest_df, mirna_df):
    interest_df['corStart'] = interest_df['start'] - interest_df['sepEnd']
    interest_df['corEnd'] = interest_df['end'] - interest_df['sepEnd']
    mirna_df = mirna_df[mirna_df['Transcript ID'].isin(interest_df['trID'].unique())]

    hitsDF = pd.DataFrame(columns=['trID', 'start', 'end', 'microRNA'])
    for i, row in mirna_df.iterrows():
        interest_hits = interest_df[(interest_df['trID'] == row['Transcript ID'])]
        # check if the microRNA binding site overlaps with the interest regions
        if not interest_hits.empty:
            microRNALength = row['UTR end'] - row['UTR start']
            for _, interest_row in interest_hits.iterrows():
                overlap_start = max(interest_row['corStart'], row['UTR start'])
                overlap_end = min(interest_row['corEnd'], row['UTR end'])
                if overlap_end - overlap_start > int(microRNALength/2):  # There is an overlap
                    hitsDF = hitsDF._append({'trID': interest_row['trID'],
                                             'geneSymbol': row['Gene Symbol'],
                                             'start': overlap_start,
                                             'end': overlap_end,
                                             'microRNA': row['miR Family']}, ignore_index=True)

    return hitsDF


def main():
    parser = argparse.ArgumentParser(description='Analyze microRNA binding enrichment')
    parser.add_argument("--params", default='params.yaml', type=str, help="Path to the YAML file containing parameters.",)
    parser.add_argument('--mirna_file', default='data/microRNA/Predicted_Targets_Info.default_predictions.txt',
                        help='Path to microRNA predictions file')
    parser.add_argument('--control_file', default='output/motifs/positive/control_positions.txt',
                        help='Path to control positions file')
    parser.add_argument('--interest_file', default='output/motifs/positive/motif_positions.txt',
                        help='Path to interest positions file')
    parser.add_argument('--output_dir', default='output/microRNA/', help='Directory to save results')
    parser.add_argument('--control_overlap_file', default='control_overlap.csv', help='File to save control overlap results')
    parser.add_argument('--interest_overlap_file', default='interest_overlap.csv', help='File to save interest overlap results')
    parser.add_argument('--fisher_file', default='fisher_results.txt', help='File to save Fisher results')

    args = parser.parse_known_args()[0]

    # Read parameters from YAML file
    if args.params:
        with open(args.params, 'r') as file:
            yaml_params = yaml.safe_load(file)
            for key, value in yaml_params['microRNABindingEnrichment'].items():
                parser.set_defaults(**{key: value})

    args = parser.parse_args()

    # Ensure output directory exists
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Read microRNA predictions and filter for species ID 7227
    mirna_df = pd.read_csv(args.mirna_file, sep='\t')
    mirna_filtered = mirna_df[mirna_df['Species ID'] == 7227]
    
    # Read control and motif positions
    control_df = pd.read_csv(args.control_file, sep='\t')
    motif_df = pd.read_csv(args.interest_file, sep='\t')

    # only keep regions in the 3'UTR to match the microRNA binding data
    control_df = control_df[(control_df['sepEnd'] < control_df['start'])]
    motif_df = motif_df[(motif_df['sepEnd'] < motif_df['start'])]

    total_mirna = len(mirna_filtered)
    total_control = len(control_df)
    total_motif = len(motif_df)
    print(f"Filtered microRNA predictions: {total_mirna} rows")
    print(f"Control positions: {total_control} rows")
    print(f"Motif positions: {total_motif} rows")

    controlOverlap = overlap(control_df, mirna_filtered)
    controlOverlap.to_csv(os.path.join(args.output_dir, args.control_overlap_file), sep=',', index=False)
    motifOverlap = overlap(motif_df, mirna_filtered)
    motifOverlap.to_csv(os.path.join(args.output_dir, args.interest_overlap_file), sep=',', index=False)

    total_control_overlap = len(controlOverlap)
    total_motif_overlap = len(motifOverlap)

    # Fisher exact test for each microRNA
    from scipy.stats import fisher_exact
    control_counts = controlOverlap['microRNA'].value_counts()
    motif_counts = motifOverlap['microRNA'].value_counts()
    all_microRNAs = set(control_counts.index).union(set(motif_counts.index))
    fisher_results = []
    for mirna in all_microRNAs:
        control_count = control_counts.get(mirna, 0)
        motif_count = motif_counts.get(mirna, 0)
        not_control_count = total_control - control_count
        not_motif_count = total_motif - motif_count
        
        # Create a contingency table
        contingency_table = [[control_count, motif_count],
                             [not_control_count, not_motif_count]]
        
        odds_ratio, p_value = fisher_exact(contingency_table)
        fisher_results.append({'microRNA': mirna, 'totalInterest': total_motif, 'TP': motif_count, '%TP': motif_count / total_motif * 100 if total_motif > 0 else 0,
                               'totalControl': total_control, 'FP': control_count, '%FP': control_count / total_control * 100 if total_control > 0 else 0,
                               'odds_ratio': odds_ratio, 'p_value': p_value})

    # Save results
    fisher_results_df = pd.DataFrame(fisher_results)
    fisher_results_df.sort_values(by='p_value', inplace=True)
    fisher_results_df['p_value'] = fisher_results_df['p_value'].apply(lambda x: f"{x:.2e}")
    fisher_results_df.round(2).to_csv(os.path.join(args.output_dir, args.fisher_file), sep=',', index=False)

if __name__ == "__main__":
    main()