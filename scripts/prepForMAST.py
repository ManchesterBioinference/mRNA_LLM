import argparse
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from os import makedirs, path

def main():
    parser = argparse.ArgumentParser(description="Split sequences in a FASTA file on ',' and write two new sequences per entry.")
    parser.add_argument("--input_fasta", help="Input FASTA file")
    parser.add_argument("--output_fasta", help="Output FASTA file")
    args = parser.parse_args()

    records = []
    for record in SeqIO.parse(args.input_fasta, "fasta"):
        parts = str(record.seq).split(',')
        if len(parts) != 2:
            continue  # skip if not exactly two parts
        id_base = record.description.split()[1]
        if len(parts[0]) > 0:
            records.append(SeqRecord(Seq(parts[0]), id=f"{id_base}_5utr", description=record.description))
        if len(parts[1]) > 0:
            records.append(SeqRecord(Seq(parts[1]), id=f"{id_base}_3utr", description=record.description))

    makedirs(path.dirname(args.output_fasta), exist_ok=True)
    SeqIO.write(records, args.output_fasta, "fasta")

if __name__ == "__main__":
    main()