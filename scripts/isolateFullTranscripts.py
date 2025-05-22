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

def reverse_complement(dna_sequence):
    complement_map = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}
    return "".join(complement_map.get(base.upper(), base.upper()) for base in reversed(dna_sequence))

def read_gtf(gtf_file, target_transcript_ids):
    utr_dict = {}
    with gzip.open(gtf_file, 'rt') as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                parts = line.split('\t')
                if parts[2] == 'transcript': #'three_prime_utr':
                    strand = 1 if parts[6] == '+' else -1
                    attributes = parts[8].strip().split(';')
                    attributes = {x.split('"')[0].strip():x.split('"')[1].strip() for x in attributes if x != ''}
                    gene_id = attributes['gene_id']
                    transcript_id = attributes['transcript_id']
                    if transcript_id not in target_transcript_ids:
                        continue

                    start = int(parts[3])
                    end = int(parts[4])
                    if parts[0] not in utr_dict:
                        utr_dict[parts[0]] = {}
                    if (transcript_id, strand, gene_id) not in utr_dict[parts[0]]:
                        utr_dict[parts[0]][(transcript_id, strand, gene_id)] = []
                    utr_dict[parts[0]][(transcript_id, strand, gene_id)].append((start, end))
    return utr_dict

def extract_3utr_sequences(fa_file, utr_dict):
    utr_sequences = {}
    with gzip.open(fa_file, 'rt') as f:
        for record in SeqIO.parse(f, 'fasta'):
            name = record.name 
            if name in utr_dict:
                sequence = str(record.seq)
                for tID_strand_gID in utr_dict[name].keys():
                    utr = ''.join([sequence[start - 1:end] for start, end in utr_dict[name][tID_strand_gID]])
                    if 'N' in utr:
                        continue
                    utr_sequences[tID_strand_gID] = utr if tID_strand_gID[1] == 1 else reverse_complement(utr)
    return utr_sequences

def process_files(data_dir, output_file, files, start_index, target_transcript_ids):
    """Processes a pair of GTF and genomic FASTA files to extract full transcripts."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    gtf_filename = files[start_index]
    fasta_filename = files[start_index + 1]
    
    # Determine base name for output file
    base_name = gtf_filename.split('-')[0] if '-' in gtf_filename else gtf_filename.split('.')[0]

    gtf_file_path = os.path.join(data_dir, gtf_filename)
    genomic_fasta_file_path = os.path.join(data_dir, fasta_filename)

    print(f"Processing GTF: {gtf_file_path}, FASTA: {genomic_fasta_file_path}")

    full_transcripts = read_gtf(gtf_file_path, target_transcript_ids)
    
    if not full_transcripts:
        print(f"No target transcripts extracted for {base_name} (Source: {gtf_filename}, {fasta_filename})")
        return

    sequences = extract_3utr_sequences(os.path.join(data_dir, files[start_index + 1]), full_transcripts)

    #output_file = os.path.join(output_file, f'{base_name}-full_transcript.fa')
    with open(output_file, 'w') as out_f:
        for tID_strand_gID, sequence in sequences.items():
            out_f.write(f'>{tID_strand_gID[0]} {tID_strand_gID[2]}\n{sequence}\n')
    print(f"Written {len(sequences)} full transcripts to {output_file}")

def main():
    args = parse_args()

    target_ids = read_target_transcript_ids(args.target_ids_fasta)
    if not target_ids:
        print(f"No target transcript IDs loaded from {args.target_ids_fasta}. "
              "Ensure the file exists and is correctly formatted. No transcripts will be extracted if list is empty.")
        # Consider exiting if target_ids are essential: return

    all_files = os.listdir(args.data_dir)
    all_files.sort() # Sort to ensure GTF/FASTA pairs are processed together
    
    # Filter for likely GTF and FASTA files to form pairs
    # This assumes a naming convention like file.gtf.gz and file.fa.gz
    # A more robust pairing might be needed depending on actual filenames
    file_pairs = []
    i = 0
    while i < len(all_files) -1:
        # Basic check for gtf/fa(sta) extensions and matching base names
        f1_base, f1_ext1, f1_ext2 = all_files[i].partition('.gtf')
        f2_base, f2_ext1, f2_ext2 = all_files[i+1].partition('.fa') # .fa or .fasta
        
        is_gtf = f1_ext1 == '.gtf' 
        is_fasta = f2_ext1 == '.fa'

        # Check if basenames match (e.g. "genome.gtf.gz" and "genome.fa.gz")
        # This is a simple check; complex names might need refined logic
        f1_core_name = all_files[i].split('.')[0]
        f2_core_name = all_files[i+1].split('.')[0]

        if is_gtf and is_fasta and f1_core_name == f2_core_name :
            file_pairs.append((all_files[i], all_files[i+1]))
            i += 2 # Move to next potential pair
        else:
            # print(f"Skipping non-matching pair or unrecognized files: {all_files[i]}, {all_files[i+1] if i+1 < len(all_files) else ''}")
            i += 1 # Try next file

    process_files(args.data_dir, args.output_file, all_files, 0, target_ids) # Process first pair
    # with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
    #     futures = []
    #     # The original script submitted based on index, assuming files list was already paired by sorting.
    #     # Reverting to simpler indexing for submission if `files.sort()` is reliable for pairing.
    #     for i in range(0, len(all_files) - 1, 2):
    #         # This assumes files[i] is GTF and files[i+1] is FASTA after sorting.
    #         # Add checks if necessary, e.g. files[i].endswith('.gtf.gz')
    #         futures.append(executor.submit(process_files, args.data_dir, args.output_file, all_files, i, target_ids))
    #     
    #     for future in futures:
    #         try:
    #             future.result() # Retrieve result or exception
    #         except Exception as e:
    #             print(f"A process raised an exception: {e}")

if __name__ == '__main__':
    main()