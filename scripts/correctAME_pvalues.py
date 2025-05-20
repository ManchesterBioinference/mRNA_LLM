import pandas as pd
from Bio import SeqIO
from scipy.stats import fisher_exact
import argparse

parser = argparse.ArgumentParser(description='Correct AME p-values using Fisher\'s exact test.')
parser.add_argument('--params', type=str, default= 'params.yaml', help='Path to the params.yaml file.')
parser.add_argument('--ame_input', type=str, help='Path to the AME output file in TSV format.')
parser.add_argument('--motif_seqs', type=str, help='Path to the motif sequences file in FASTA format.')
parser.add_argument('--control_seqs', type=str, help='Path to the control sequences file in FASTA format.')
parser.add_argument('--ame_corrected', type=str, help='Path to the AME corrected output file in TSV format.')
args = parser.parse_args()

motifs = list(SeqIO.parse(args.motif_seqs, 'fasta'))
counts = set()
for motif in motifs:
    counts.add(motif.id.split('_')[2])

control_motifs = list(SeqIO.parse(args.control_seqs, 'fasta'))
control_counts = set()
for control_motif in control_motifs:
    control_counts.add(control_motif.id.split('_')[2])

data = pd.read_csv(args.ame_input, sep='\t', comment='#')

numMotifs = data.shape[0]
newPvalues = []
newPadj = []
newE = []
for index, row in data.iterrows():
    # run fisher's exact test
    totalPos = len(counts)#96 #row['pos']#
    totalNeg = len(control_counts) #row['neg']
    truePos = row['TP']
    falsePos = row['FP']
    p = fisher_exact([[truePos, falsePos], [totalPos-truePos, totalNeg-falsePos]], alternative='greater')[1]
    padj = 1 - (1-p)**row['tests']
    e = padj * numMotifs
    newPvalues.append(p)
    newPadj.append(padj)
    newE.append(e)

data['newPos'] = len(counts)
data['newNeg'] = len(control_counts)
data['newPvalue'] = newPvalues
data['newPadj'] = newPadj
data['newE'] = newE
data.sort_values(by=['newE'], ascending=True, inplace=True)

data.to_csv(args.ame_corrected, sep='\t', index=False)