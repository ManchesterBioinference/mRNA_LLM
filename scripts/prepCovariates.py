import argparse
import pandas as pd
from Bio import SeqIO

def main():
    parser = argparse.ArgumentParser(description='Filter covariates based on transcript IDs from FASTA file')
    parser.add_argument('--data_dir', required=True, help='Path to FASTA file')
    parser.add_argument('--covariates', required=True, help='Path to covariates CSV file')
    parser.add_argument('--output_file', required=True, help='Path to output TSV file')
    
    args = parser.parse_args()
    
    # Extract transcript IDs from FASTA file
    tr_ids = []
    with open(args.data_dir, 'r') as fasta_file:
        for record in SeqIO.parse(fasta_file, 'fasta'):
            # Get the 2nd element from the description (split by space or other delimiter)
            description_parts = record.description.split()
            if len(description_parts) > 1:
                tr_ids.append(description_parts[1])
    
    # Read covariates file and set first column as index
    covariates_df = pd.read_csv(args.covariates, index_col=0).drop(columns=['Residuals','Decay Rate'])
    
    # Filter covariates to only include rows with transcript IDs found in FASTA
    filtered_covariates = covariates_df.loc[tr_ids]
    
    # Save to TSV without headers or row names
    filtered_covariates.to_csv(args.output_file, sep='\t', header=False, index=False)

if __name__ == '__main__':
    main()