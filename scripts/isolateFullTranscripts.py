import os
import gzip
import argparse
import yaml
from Bio import SeqIO

def read_target_transcript_ids(fasta_file_path):
    """Reads transcript IDs from a FASTA file.
    Assumes the first word in the description line (after '>') is the transcript ID.
    """
    target_ids = set()
    if not os.path.isfile(fasta_file_path):
        print(f"Warning: Target IDs FASTA file not found: {fasta_file_path}")
        return target_ids
    try:
        # Use 'rt' mode with gzip.open if file could be gzipped, otherwise 'r' for open
        # For simplicity, assuming plain text .fasta file
        with open(fasta_file_path, 'r') as f_in:
            for record in SeqIO.parse(f_in, 'fasta'):
                if record.description:
                    # The first item in the description is the transcript id
                    target_ids.add(record.description.split()[1])
                elif record.id: # Fallback if description is empty but ID exists
                    target_ids.add(record.id)
    except Exception as e:
        print(f"Error reading target IDs FASTA file {fasta_file_path}: {e}")
    return target_ids

def parse_args():
    parser = argparse.ArgumentParser(description="Isolate full transcript sequences from cDNA and GTF files.")
    parser.add_argument('--params', default='params.yaml', help='Path to the yaml file with specified arguments.')
    parser.add_argument('--data_dir', default='data/downloaded', help='Path to the directory containing the data files.')
    parser.add_argument('--output_file', default='output/data/full_transcripts/Dmela-full_transcripts.fa', help='Path to the output directory for full transcripts.')
    #parser.add_argument('--max_workers', type=int, default=4, help='Maximum number of processes to run concurrently.')
    parser.add_argument('--target_ids_fasta', default='output/data/utrTE.fasta', help='Path to FASTA file containing target transcript IDs.')

    # Parse initial arguments to get params file path
    known_args, _ = parser.parse_known_args()

    if os.path.isfile(known_args.params):
        with open(known_args.params, 'r') as file:
            yaml_params = yaml.safe_load(file)
            # Use a relevant section from YAML, e.g., 'isolateFullTranscripts' or fallback
            config_section = yaml_params.get('isolateFullTranscripts',{})
            
            yaml_defaults = {k: v for k, v in config_section.items() if hasattr(known_args, k)}
            parser.set_defaults(**yaml_defaults)

    return parser.parse_args()

def main():
    args = parse_args()

    target_ids = read_target_transcript_ids(args.target_ids_fasta)
    if not target_ids:
        print(f"No target transcript IDs loaded from {args.target_ids_fasta}. "
              "Ensure the file exists and is correctly formatted. No transcripts will be extracted if list is empty.")
        # Consider exiting if target_ids are essential: return

    all_files = os.listdir(args.data_dir)
    all_files.sort() # Sort to ensure GTF/FASTA pairs are processed together
    
    utr3 = SeqIO.to_dict(SeqIO.parse(os.path.join(args.data_dir,all_files[0]), 'fasta')) 
    utr5 = SeqIO.to_dict(SeqIO.parse(os.path.join(args.data_dir,all_files[1]), 'fasta')) 
    cds = SeqIO.to_dict(SeqIO.parse(os.path.join(args.data_dir,all_files[2]), 'fasta')) 

    seqs = []
    for t_id in target_ids:
        if t_id in utr3 and t_id in utr5 and t_id in cds:
            record = utr5[t_id]
            record.seq += cds[t_id].seq + utr3[t_id].seq
            seqs.append(record)
        else:
            print(f"Transcript ID {t_id} not found in all files.")

    # Write the combined sequences to the output file using SeqIO
    with open(args.output_file, 'w') as out_f:
        SeqIO.write(seqs, out_f, 'fasta')

if __name__ == '__main__':
    main()