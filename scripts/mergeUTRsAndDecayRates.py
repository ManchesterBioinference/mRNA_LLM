import argparse
import yaml
import pandas as pd
from Bio import SeqIO
import numpy as np
import os

def loadParams(parser):
    args = parser.parse_known_args()[0]
        # Read parameters from YAML file
    if args.params:
        with open(args.params, 'r') as file:
            yaml_params = yaml.safe_load(file)
            for key, value in yaml_params['mergeUTRsAndDecayRates'].items():
                parser.set_defaults(**{key: value})

    return(parser.parse_args())

def merge_utr_and_decay_rates(fasta_dir, decay_rates_file, output_file):
    # Load decay rates
    decay_rates_df = pd.read_csv(decay_rates_file)
    # Merge UTR sequences with decay rates
    id_decay = {}
    for index, row in decay_rates_df.iterrows():
        tr_id = row['tr_id']
        decay = row['D'] #if row['D'] > 0.001 else 0.001 #TODO - ask Magnus about this
        halfLife = np.log(1+np.log(2)/decay)
        id_decay[tr_id] = halfLife
    
    # Load UTR sequences
    decaySeqs = []
    for file in os.listdir(fasta_dir):
        filepath = os.path.join(fasta_dir, file)
        for record in SeqIO.parse(filepath, "fasta"):
            if record.id in id_decay:
                description = record.description
                record.id = str(id_decay[record.id])
                record.description = description
                decaySeqs.append(record)
    
    # Write decaySeqs to output file
    with open(output_file, 'w') as file:
        SeqIO.write(decaySeqs, file, "fasta-2line")
    
def main():
    parser = argparse.ArgumentParser(description="Merge UTRs and Decay Rates")
    parser.add_argument('--params', default='params.yaml', help='Path to the parameters YAML file')
    parser.add_argument('--fasta', help='Path to the input FASTA file')
    parser.add_argument('--decay_rates', help='Path to the input decay rates file')
    parser.add_argument('--output_file', help='Path to the output file')
    args = loadParams(parser)
    
    merge_utr_and_decay_rates(args.fasta, args.decay_rates, args.output_file)

if __name__ == "__main__":
    main()