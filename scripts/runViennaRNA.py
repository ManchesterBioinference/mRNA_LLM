import argparse
import subprocess
import re
import pandas as pd
from Bio import SeqIO
import os

def run_rnafold(sequence, condaPath,condaEnv,RNAfold_path):
    """
    Runs RNAfold on a given RNA sequence and returns the structure and MFE.
    """
    try:
        # Use full path to RNAfold or ensure it's in your PATH
        process = subprocess.run(
            [condaPath, 'run','-n',condaEnv, RNAfold_path, '-b','200'], # Update this path to where RNAfold is installed
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
    parser.add_argument(
        "--input_fasta",
        type=str,
        default="/mnt/mr01-home01/m65338lb/projects/mRNA_LLM-worktrees/translationEfficiency/data/dmel-all-three_prime_UTR-r6.59.fasta",
        help="Path to the input FASTA file."
    )
    parser.add_argument("--condaPath",type=str, help="Path to the conda executable.")
    parser.add_argument("--condaEnv",type=str, help="Name of the conda environment.")
    parser.add_argument("--RNAfold_path",type=str, help="Path to the RNAfold executable.")
    parser.add_argument(
        "--output_csv",
        type=str,
        default="/mnt/mr01-home01/m65338lb/projects/mRNA_LLM-worktrees/translationEfficiency/data/vienna_features.csv",
        help="Path to save the output CSV file."
    )
    args = parser.parse_args()

    print(f"Reading sequences from: {args.input_fasta}")
    
    results = []
    processed_ids = set()

    for record in SeqIO.parse(args.input_fasta, "fasta"):
        seq_id = record.id
        sequence = str(record.seq).upper().replace('T', 'U') # Ensure RNA

        if not sequence:
            print(f"Warning: Empty sequence for ID {seq_id}. Skipping.")
            continue
        
        if seq_id in processed_ids:
            print(f"Warning: Duplicate sequence ID {seq_id}. Skipping.")
            continue
        processed_ids.add(seq_id)

        print(f"Processing {seq_id}... of length {len(sequence)}")
        structure, mfe = run_rnafold(sequence, condaPath=args.condaPath, condaEnv=args.condaEnv, RNAfold_path=args.RNAfold_path)
        print(mfe)
        if seq_id == 'FBtr0113386':
            break
        
        results.append({
            "id": seq_id,
            "sequence_length": len(sequence),
            "structure": structure,
            "mfe": mfe
        })

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
