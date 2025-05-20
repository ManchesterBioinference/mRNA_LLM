from Bio import SeqIO
import math
import os
import argparse
import yaml

def split_fasta_by_decay(input_fasta_path, high_decay_fasta_path, low_decay_fasta_path):
    """
    Reads a FASTA file, extracts numeric IDs, ranks them,
    and splits the data into high and low decay groups.

    Args:
        input_fasta_path (str): Path to the input FASTA file.
        high_decay_fasta_path (str): Path to save the high decay subset.
        low_decay_fasta_path (str): Path to save the low decay subset.
    """
    records_with_ids = []
    for record in SeqIO.parse(input_fasta_path, "fasta"):
        try:
            # Assuming the ID is a number or can be converted to one.
            # If the ID has other characters, you might need more specific parsing.
            # For example, if ID is "gene_123.45", you might need to extract "123.45".
            # Here, we'll try a direct conversion.
            numeric_id = float(record.id)
            records_with_ids.append((numeric_id, record))
        except ValueError:
            print(f"Warning: Could not convert ID '{record.id}' to a number. Skipping this record.")
            continue

    if not records_with_ids:
        print("No valid records found or IDs could not be converted to numbers.")
        return

    # Sort records by the numeric ID in ascending order
    records_with_ids.sort(key=lambda x: x[0])

    # Extract sorted records
    sorted_records = [record for _, record in records_with_ids]
    n_records = len(sorted_records)

    if n_records < 5:
        print(f"Warning: Only {n_records} records found. Cannot split into 5 chunks as requested. Skipping file creation.")
        return

    chunk_size = math.ceil(n_records / 5) # Use math.ceil to ensure all records are covered

    # Define chunks
    # Chunk 0 (lowest) to Chunk 4 (highest)
    # Low decay: chunks 0 and 1
    # High decay: chunks 3 and 4

    low_decay_records = []
    high_decay_records = []

    for i in range(5):
        start_index = i * chunk_size
        end_index = min((i + 1) * chunk_size, n_records)
        current_chunk_records = sorted_records[start_index:end_index]

        if i < 2: # Chunks 0 and 1 are low
            low_decay_records.extend(current_chunk_records)
        elif i > 2: # Chunks 3 and 4 are high
            high_decay_records.extend(current_chunk_records)

    # Save the subsets
    if low_decay_records:
        SeqIO.write(low_decay_records, low_decay_fasta_path, "fasta")
        print(f"Saved {len(low_decay_records)} low decay records to {low_decay_fasta_path}")
    else:
        print("No records classified as low decay.")

    if high_decay_records:
        SeqIO.write(high_decay_records, high_decay_fasta_path, "fasta")
        print(f"Saved {len(high_decay_records)} high decay records to {high_decay_fasta_path}")
    else:
        print("No records classified as high decay.")

if __name__ == '__main__':
    # Define file paths
    input_file = "output/data/decay/test.fasta"
    high_decay_file = "output/data/decay/high_decay_test.fasta"
    low_decay_file = "output/data/decay/low_decay_test.fasta"
    parser = argparse.ArgumentParser(description="Split FASTA file into high and low decay groups based on numeric IDs.")
    parser.add_argument("--params", type=str, default='params.yaml', help="Path to the input FASTA file.")
    parser.add_argument("--input", type=str, default=input_file, help="Path to the input FASTA file.")
    parser.add_argument("--output", type=str, default=input_file, help="Path to the output dir.")
    parser.add_argument("--high", type=str, default=high_decay_file, help="Path to save the high decay subset.")
    parser.add_argument("--low", type=str, default=low_decay_file, help="Path to save the low decay subset.")

    args = parser.parse_known_args()[0]

    # Read parameters from YAML file
    if args.params:
        with open(args.params, 'r') as file:
            yaml_params = yaml.safe_load(file)
            for key, value in yaml_params['splitHighLowDecay'].items():
                parser.set_defaults(**{key: value})

    args = parser.parse_args()

    # # Create dummy input file for testing if it doesn't exist
    # os.makedirs(os.path.dirname(input_file), exist_ok=True)
    # if not os.path.exists(input_file):
    #     print(f"Creating dummy input file: {input_file}")
    #     with open(input_file, "w") as f:
    #         for i in range(1, 21): # Create 20 records for easy splitting
    #             f.write(f">{i*10}\n") # ID is a number
    #             f.write(f"ATGCATGC{i}\n")

    os.makedirs(args.output, exist_ok=True)
    split_fasta_by_decay(args.input, os.path.join(args.output, args.high), os.path.join(args.output, args.low))