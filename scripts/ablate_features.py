import pandas as pd
import argparse
import os

def get_codon_columns():
    bases = ['A', 'C', 'G', 'T']
    codons = [a + b + c for a in bases for b in bases for c in bases]
    stop_codons = ['TAA', 'TAG', 'TGA']
    return [c for c in codons if c not in stop_codons]

def main():
    parser = argparse.ArgumentParser(description='Ablate features from the extra features file.')
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input extra features file.')
    parser.add_argument('--output_file', type=str, required=True, help='Path to save the ablated features file.')
    parser.add_argument('--ablate_lengths', action='store_true', help='Ablate length features.')
    parser.add_argument('--ablate_gc_content', action='store_true', help='Ablate GC content features.')
    parser.add_argument('--ablate_codons', action='store_true', help='Ablate codon frequency features.')
    
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    features_df = pd.read_csv(args.input_file, index_col=0)
    
    cols_to_drop = []
    if args.ablate_lengths:
        cols_to_drop.extend(["5' UTR Length", "CDS Length", "3' UTR Length"])
    if args.ablate_gc_content:
        cols_to_drop.extend(["5' UTR GC Content", "CDS GC Content", "3' UTR GC Content"])
    if args.ablate_codons:
        cols_to_drop.extend(get_codon_columns())

    # Ensure we only try to drop columns that exist
    cols_to_drop = [col for col in cols_to_drop if col in features_df.columns]
    
    ablated_features_df = features_df.drop(columns=cols_to_drop)
    
    ablated_features_df.to_csv(args.output_file)

if __name__ == '__main__':
    main()
