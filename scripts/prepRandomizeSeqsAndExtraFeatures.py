import argparse
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord

def main():
    parser = argparse.ArgumentParser(description="Process FASTA files based on specified method.")
    parser.add_argument("--params", help="Path to a parameters file (not used in current logic).")
    parser.add_argument("--sequence_file", required=True, help="Path to the input FASTA file.")
    parser.add_argument("--process", required=True, choices=['originalSeqs', 'sameSeq', 'same5UTR', 'same3UTR'],
                        help="Processing mode: 'originalSeqs', 'sameSeq', 'same5UTR', 'same3UTR'.")
    parser.add_argument("--output_file", required=True, help="Path to the output FASTA file.")

    args = parser.parse_args()

    try:
        records = list(SeqIO.parse(args.sequence_file, "fasta"))
    except FileNotFoundError:
        print(f"Error: Input sequence file not found at {args.sequence_file}")
        return
    except Exception as e:
        print(f"Error reading sequence file: {e}")
        return

    if not records:
        print("Input sequence file is empty.")
        # Create an empty output file or handle as per desired behavior
        with open(args.output_file, "w") as out_handle:
            pass
        return

    if args.process == "originalSeqs":
        first_record_description = records[0].description.split(' ')
        id = first_record_description[0]
        first_record_description = ' '.join(first_record_description[1:])
        for record in records:
            record.id = id
            record.description = first_record_description
    elif args.process == "sameSeq":
        first_record_sequence = records[0].seq
        for record in records:
            record.seq = first_record_sequence
    elif args.process == "same5UTR":
        first_record_5utr = str(records[0].seq.split(',')[0])
        first_record = records[0]
        for record in records:
            variable3UTR = str(record.seq).split(',')[1]
            record.id = first_record.id
            record.description = ' '.join(first_record.description.split(' ')[1:])
            record.seq = ','.join([first_record_5utr, variable3UTR])
    elif args.process == "same3UTR":
        first_record_3utr = str(records[0].seq.split(',')[1])
        first_record = records[0]
        for record in records:
            variable5UTR = str(record.seq).split(',')[0]
            record.id = first_record.id
            record.description = ' '.join(first_record.description.split(' ')[1:])
            record.seq = ','.join([variable5UTR, first_record_3utr])
    else:
        # This case should not be reached due to argparse choices
        print(f"Error: Unknown process type '{args.process}'.")
        return

    try:
        SeqIO.write(records, args.output_file, "fasta")
        print(f"Processed sequences saved to {args.output_file}")
    except Exception as e:
        print(f"Error writing output file: {e}")

if __name__ == "__main__":
    main()