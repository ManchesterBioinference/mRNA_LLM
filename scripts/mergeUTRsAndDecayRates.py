import argparse
import yaml
import pandas as pd
from Bio import SeqIO
import numpy as np
import os
import re
from collections import defaultdict
import gzip as gz

def load_transcripts_from_gtf(gtf_path):
    gene_to_transcripts = defaultdict(list)
    with gz.open(gtf_path, 'rt') as f:
        for line in f:
            if line.startswith("#"):
                continue
            fields = line.strip().split("\t")
            if len(fields) != 9:
                continue
            feature_type = fields[2]
            if feature_type != "transcript":
                continue
            attr_str = fields[8]
            # Use regex to extract gene_id and transcript_id
            gene_match = re.search(r'gene_id "([^"]+)"', attr_str)
            tx_match = re.search(r'transcript_id "([^"]+)"', attr_str)
            if gene_match and tx_match:
                gene_id = gene_match.group(1)
                transcript_id = tx_match.group(1)
                gene_to_transcripts[gene_id].append(transcript_id)
    return gene_to_transcripts

def loadParams(parser):
    args = parser.parse_known_args()[0]
        # Read parameters from YAML file
    if args.params:
        with open(args.params, 'r') as file:
            yaml_params = yaml.safe_load(file)
            for key, value in yaml_params['mergeUTRsAndDecayRates'].items():
                parser.set_defaults(**{key: value})

    return(parser.parse_args())

def findHighlyExpressedTranscript(gene_to_transcripts, transcriptExpression, geneList):
    # Load transcript expression data
    transcript_expression_df = pd.read_csv(transcriptExpression, index_col=0)
    # Create a dictionary to store the most highly expressed transcript for each gene
    gene_to_highest_transcript = {}
    for gene in geneList:
        transcripts = gene_to_transcripts.get(gene, [])
        # Get the expression values for the transcripts
        expression_values = transcript_expression_df.loc[[t for t in transcripts if t in transcript_expression_df.index]].mean(axis=1).values.flatten()
        # Find the transcript with the highest expression
        if len(expression_values) == 0:
            highest_transcript = transcripts[0] if transcripts else None
        else:
            highest_transcript = transcripts[np.argmax(expression_values)]
        gene_to_highest_transcript[gene] = highest_transcript
    return gene_to_highest_transcript

def merge_utr_and_decay_rates(args):
    #construct a dictionary of gene_id to transcript
    gene_to_transcripts = load_transcripts_from_gtf(args.gtf_file)
    # Load decay rates
    decay_rates_df = pd.read_csv(args.decay_rates, index_col=0)
    # identify most highly expressed transcript for each gene
    gene_to_transcripts = findHighlyExpressedTranscript(gene_to_transcripts, args.transcriptExpression, decay_rates_df.index.tolist())
    # Merge UTR sequences with decay rates
    id_decay = {}
    for index, row in decay_rates_df.iterrows():
        tr_id = gene_to_transcripts[index]
        decay = row.iloc[-3:].mean()#['D'] #if row['D'] > 0.001 else 0.001 #TODO - ask Magnus about this
        halfLife = np.log(1+decay)
        id_decay[tr_id] = halfLife
    
    # Load UTR sequences
    decaySeqs = []
    for file in os.listdir(args.fasta):
        filepath = os.path.join(args.fasta, file)
        for record in SeqIO.parse(filepath, "fasta"):
            if record.id in id_decay:
                description = record.description
                record.id = str(id_decay[record.id])
                record.description = description
                decaySeqs.append(record)
    
    # Write decaySeqs to output file
    with open(args.output_file, 'w') as file:
        SeqIO.write(decaySeqs, file, "fasta-2line")
    
def main():
    parser = argparse.ArgumentParser(description="Merge UTRs and Decay Rates")
    parser.add_argument('--params', default='params.yaml', help='Path to the parameters YAML file')
    parser.add_argument('--fasta', help='Path to the input FASTA file')
    parser.add_argument('--decay_rates', help='Path to the input decay rates file')
    parser.add_argument('--gtf_file', help='Path to the GTF file')
    parser.add_argument('--transcriptExpression', help='Path to the transcript expression csv file')
    parser.add_argument('--output_file', help='Path to the output file')
    args = loadParams(parser)
    
    merge_utr_and_decay_rates(args)

if __name__ == "__main__":
    main()