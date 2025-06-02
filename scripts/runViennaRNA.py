import argparse
import subprocess
import re
import pandas as pd
from Bio import SeqIO
import os
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
import yaml

# Define a worker function for multiprocessing - MOVED TO GLOBAL SCOPE
def process_sequence(seq_data, condaPath_arg, condaEnv_arg, RNAfold_path_arg):
    seq_id, sequence = seq_data
    structure, mfe = run_rnafold(sequence, condaPath=condaPath_arg, condaEnv=condaEnv_arg, RNAfold_path=RNAfold_path_arg)
    return {
    "id": seq_id,
    "sequence_length": len(sequence),
    "structure": structure,
    "mfe": mfe
    }

def run_rnafold(sequence, condaPath,condaEnv,RNAfold_path):
    """
    Runs RNAfold on a given RNA sequence and returns the structure and MFE.
    """
    try:
        # Use full path to RNAfold or ensure it's in your PATH
        process = subprocess.run(
            [RNAfold_path, '-b','200'],
            #[condaPath, 'run','-n',condaEnv, RNAfold_path, '-b','200'], # Update this path to where RNAfold is installed
            #[condaPath, 'run','-n',condaEnv, RNAfold_path, "--maxBPspan=200"], # Update this path to where RNAfold is installed
            input=sequence,
            text=True,
            capture_output=True,
            check=True,
        )
        output_lines = process.stdout.strip().split('\n')
        if len(output_lines) >= 2:
            # First line is the sequence, second line is structure and MFE
            # Example: .((...)) (-1.23)
            structure_mfe_line = output_lines[1]
            match = re.search(r'([().]+)\s*\(?([-+]?\d*\.\d+|\d+)\)?', structure_mfe_line)
            if match:
                structure = match.group(1).strip()
                mfe = float(match.group(2))
                return structure, mfe
            else:
                print(f"Warning: Could not parse structure/MFE from: {structure_mfe_line}")
                return None, None
        else:
            print(f"Warning: Unexpected RNAfold output for sequence: {sequence[:30]}...")
            return None, None
    except FileNotFoundError:
        print("Error: RNAfold command not found. Please ensure ViennaRNA is installed and in PATH.")
        raise
    except subprocess.CalledProcessError as e:
        print(f"Error running RNAfold for sequence: {sequence[:30]}...")
        print(f"Stderr: {e.stderr}")
        return None, None
    except Exception as e:
        print(f"An unexpected error occurred with RNAfold: {e}")
        return None, None

def main():
    parser = argparse.ArgumentParser(description="Run ViennaRNA RNAfold on sequences in a FASTA file and extract MFE and secondary structure.")
    parser.add_argument('--params', default='params.yaml', help='Path to the parameters YAML file')
    parser.add_argument( "--input_fasta", type=str, default="/mnt/mr01-home01/m65338lb/projects/mRNA_LLM-worktrees/translationEfficiency/data/dmel-all-three_prime_UTR-r6.59.fasta", help="Path to the input FASTA file.")
    parser.add_argument("--condaPath",type=str, help="Path to the conda executable.")
    parser.add_argument("--condaEnv",type=str, help="Name of the conda environment.")
    parser.add_argument("--RNAfold_path",type=str, help="Path to the RNAfold executable.")
    parser.add_argument("--maxWorkers",type=int, default = 4, help="Maximum number of workers for parallel processing.")
    parser.add_argument( "--output_csv", type=str, default="data/vienna_features.csv", help="Path to save the output CSV file.")

    args = parser.parse_known_args()[0]
    # Read parameters from YAML file
    if args.params:
        with open(args.params, 'r') as file:
            yaml_params = yaml.safe_load(file)
            for key, value in yaml_params['runViennaRNA'].items():
                parser.set_defaults(**{key: value})

    args = parser.parse_args()

    print(f"Reading sequences from: {args.input_fasta}")
    
    all_sequence_data = [] # Store all sequence data here
    processed_ids = set()

    # First, parse all sequences from the FASTA file
    print("Parsing FASTA file to gather all sequences...")
    for record in tqdm(list(SeqIO.parse(args.input_fasta, "fasta")), desc="Reading FASTA records"):
        seq_id = record.id
        sequence = str(record.seq).upper().replace('T', 'U') # Ensure RNA

        if not sequence:
            print(f"Warning: Empty sequence for ID {seq_id}. Skipping.")
            continue
        
        if seq_id in processed_ids:
            print(f"Warning: Duplicate sequence ID {seq_id}. Skipping.")
            continue
        processed_ids.add(seq_id)
        all_sequence_data.append((seq_id, sequence))

    if not all_sequence_data:
        print("No valid sequences found to process.")
        return

    print(f"Finished parsing. Found {len(all_sequence_data)} sequences to process.")

    # Use about 75% of available cores
    num_cores = min(max(1, int(cpu_count() * 0.75)), args.maxWorkers)
    print(f"Processing {len(all_sequence_data)} sequences in parallel using {num_cores} cores.")

    # Create a partial function with fixed arguments for the global process_sequence
    partial_process_func = partial(process_sequence, 
                                   condaPath_arg=args.condaPath, 
                                   condaEnv_arg=args.condaEnv, 
                                   RNAfold_path_arg=args.RNAfold_path)
    
    results = []
    # Process all sequences in parallel using a single pool
    with Pool(processes=num_cores) as pool:
        results = list(tqdm(
            pool.imap(partial_process_func, all_sequence_data),
            total=len(all_sequence_data),
            desc="Folding all sequences"
        ))

    df = pd.DataFrame(results)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"ViennaRNA features saved to: {args.output_csv}")
    print(f"Processed {len(df)} sequences.")
    if df['mfe'].isnull().any():
        print(f"Warning: Some sequences ({df['mfe'].isnull().sum()}) could not be processed for MFE by RNAfold.")

if __name__ == "__main__":
    main()
