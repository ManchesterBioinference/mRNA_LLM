from Bio import SeqIO
import os
import random
import math
import argparse
import yaml
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager
from threading import Thread
import gzip

random.seed(42)

def load_ncbi_oma_mapping(file_path):
    ncbi_to_oma = {}
    with gzip.open(file_path, 'rt') as f:
        for line in f:
            if line.startswith("#"):
                continue
            oma_id, ncbi_id = line.strip().split()
            ncbi_to_oma[ncbi_id] = oma_id
    return ncbi_to_oma

def load_geneID_ncbi_mapping(file_path, ncbi_to_oma):
    geneID_to_ncbi = {}
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip().split()
            if line[5] in ncbi_to_oma.keys():
                geneID_to_ncbi[line[1]] = line[5]
    return geneID_to_ncbi

def load_flybase_geneID_mapping(file_path):
    flybase_to_geneID = {}
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip().split()
            flybase_to_geneID[line[1]] = line[0]
    return flybase_to_geneID

# Load ortholog groups and assign to splits
def load_and_split_ortholog_groups(file_path):
    ortholog_splits = {}
    with gzip.open(file_path, 'rt') as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            group_id = parts[0]
            oma_ids = parts[2:]

            # Randomly assign each group to a split
            split = random.choices(['train', 'dev', 'test'], weights=[0.7, 0.15, 0.15], k=1)[0]
            for oma_id in oma_ids:
                ortholog_splits[oma_id] = (split, group_id)
    return ortholog_splits

# Convert sequence to k-mers
def seq2kmer(seq, k):
    return seq #" ".join([seq[x:x+k] for x in range(len(seq)+1-k)])

# Process individual fasta file
def process_fasta_file(fasta_files, ortholog_splits, ncbi_to_oma, flybase_to_geneID, geneID_to_ncbi, kmer_size, queue):
    species = "[D" + fasta_files[0].split('/')[-1].replace('-','_').split('_')[1][:4] + "]"
    seq_list = {}
    utrs5 = SeqIO.to_dict(SeqIO.parse([f for f in fasta_files if '5utr' in f][0], "fasta"))
    utrs3 = [f for f in fasta_files if '3utr' in f][0]
    missing_count = 0
    mOmaGroups = 0
    mNcbiToOma = 0
    mGeneIDToNcbi = 0
    mFlybaseToGeneID = 0
    groupsToKeep = set()
    with open(utrs3, 'r') as file:
        for seq_record in SeqIO.parse(file, "fasta"):
            tr_id = seq_record.id
            ncbi_id = (seq_record.description.split()[1],'')  # Extract NCBI gene ID
            if species == "[Dmela]":
                if ncbi_id[0] in flybase_to_geneID:
                    gene_id = flybase_to_geneID[ncbi_id[0]]
                    if gene_id in geneID_to_ncbi:
                        ncbi_id_tmp = geneID_to_ncbi[gene_id]
                        if ncbi_id_tmp in ncbi_to_oma:
                            oma_id = ncbi_to_oma[ncbi_id_tmp]
                            if oma_id in ortholog_splits:
                                groupsToKeep.add(ortholog_splits[oma_id])
                            else:
                                split = random.choices(['train', 'dev', 'test'], weights=[0.7, 0.15, 0.15], k=1)[0]
                                name = 'missing_'+str(mOmaGroups)+'_omaGroups'
                                ortholog_splits[oma_id] = (split, name)
                                groupsToKeep.add((split, name))
                                missing_count += 1
                                mOmaGroups += 1
                        else:
                            split = random.choices(['train', 'dev', 'test'], weights=[0.7, 0.15, 0.15], k=1)[0]
                            name = 'missing_'+str(mNcbiToOma)+'_ncbiToOma'
                            ncbi_to_oma[ncbi_id] = name
                            ortholog_splits[name] = (split, name)
                            groupsToKeep.add((split, name))
                            missing_count += 1
                            mNcbiToOma += 1
                    else:
                        split = random.choices(['train', 'dev', 'test'], weights=[0.7, 0.15, 0.15], k=1)[0]
                        name = 'missing_'+str(mGeneIDToNcbi)+'_geneIDToNcbi'
                        geneID_to_ncbi[gene_id] = name
                        ncbi_to_oma[name] = name
                        ortholog_splits[name] = (split, name)
                        groupsToKeep.add((split, name))
                        missing_count += 1
                        mGeneIDToNcbi += 1
                else:
                    split = random.choices(['train', 'dev', 'test'], weights=[0.7, 0.15, 0.15], k=1)[0]
                    name = 'missing_'+str(mFlybaseToGeneID)+'_flybaseToGeneID'
                    flybase_to_geneID[ncbi_id[0]] = name
                    geneID_to_ncbi[name] = name
                    ncbi_to_oma[name] = name
                    ortholog_splits[name] = (split, name)
                    groupsToKeep.add((split, name))
                    missing_count += 1
                    mFlybaseToGeneID += 1
                ncbi_id = (geneID_to_ncbi.get(flybase_to_geneID.get(ncbi_id[0], None),None),ncbi_id[0])
            oma_id = ncbi_to_oma.get(ncbi_id[0], None)

            # Skip if no matching OMA ID found
            if oma_id is None or oma_id not in ortholog_splits:
                continue
            
            # Determine the split for this sequence
            split, group_id = ortholog_splits[oma_id]
            kmer_sequence = seq2kmer(str(seq_record.seq).replace('T','U'), kmer_size)
            kmer_sequence_5utr = seq2kmer(str(utrs5[tr_id].seq).replace('T','U'), kmer_size) if tr_id in utrs5 else ""
            # num_tokens = math.ceil((kmer_sequence.count(' ') + 3)/(512-10))  #make this dynamic #3 = cls + sep +1 because we are counting the spaces between tokens, 512 = block size, 10 = num memory tokens

            # # Put the result in the queue
            # queue.put((split, kmer_sequence, species, num_tokens, group_id, oma_id, ncbi_id))
            fasta_header = f">{tr_id} {ncbi_id[1]} {ncbi_id[0]}|{oma_id}|{group_id}|{species}"
            fasta_sequence = kmer_sequence_5utr + "," + kmer_sequence
            seq_list[fasta_sequence] = True
            fasta_format = f"{fasta_header}\n{fasta_sequence}\n"
            queue.put((split,fasta_format))

# Writer thread function
def writer_thread(output_dir, queue):
    files = {
        'train': open(os.path.join(output_dir, 'train.fasta'), 'a'),
        'dev': open(os.path.join(output_dir, 'dev.fasta'), 'a'),
        'test': open(os.path.join(output_dir, 'test.fasta'), 'a')
    }
    while True:
        item = queue.get()
        if item is None:
            break
        split, kmer_sequence = item
        #split, kmer_sequence, species, num_tokens, group_id, oma_id, ncbi_id = item
        files[split].write(f"{kmer_sequence}") #\t{species}\t{num_tokens}\t{group_id}\t{oma_id}\t{ncbi_id}\n")
    for f in files.values():
        f.close()

# Main function for parallel processing
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default='params.yaml', help="Path to the YAML file containing parameters.")
    parser.add_argument("--data_dir", default='data/3UTRs', help="Directory with species FASTA files.")
    parser.add_argument("--output_dir", help="Directory to save the .fasta files.")
    parser.add_argument("--kmer", type=int, default=4, help="K-mer size.")
    parser.add_argument("--oma_groups", help="Path to oma-groups.txt file.")
    parser.add_argument("--oma_ncbi_map", help="Path to oma-ncbi.txt file.")
    parser.add_argument("--ncbi_geneID_map", help="Path to gene2refseq_Dmel.txt file.")
    parser.add_argument("--geneID_flybase_map", help="Path to GeneID_to_Flybase.txt file.")
    args = parser.parse_args()

    # Load parameters
    with open(args.params, 'r') as file:
        yaml_params = yaml.safe_load(file)
        for key, value in yaml_params['preprocessFlyUTRs'].items():
            parser.set_defaults(**{key: value})
    args = parser.parse_args()

    # Load mappings and ortholog groups
    ncbi_to_oma = load_ncbi_oma_mapping(args.oma_ncbi_map)
    geneID_to_ncbi = load_geneID_ncbi_mapping(args.ncbi_geneID_map, ncbi_to_oma)
    flybase_to_geneID = load_flybase_geneID_mapping(args.geneID_flybase_map)
    ortholog_splits = load_and_split_ortholog_groups(args.oma_groups)

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Remove existing .fasta files
    for file in os.listdir(args.output_dir):
        if file.endswith(".fasta"):
            os.remove(os.path.join(args.output_dir, file))

    # Initialize manager for communication between processes and writer thread
    manager = Manager()
    queue = manager.Queue()

    
    writer = Thread(target=writer_thread, args=(args.output_dir, queue))
    writer.start()

    # Get list of FASTA files
    fasta_files = [os.path.join(args.data_dir, file) for file in os.listdir(args.data_dir) if file.endswith(".fa")]
    speciesList = list(set([os.path.basename(file).split('-')[0] for file in fasta_files]))

    #process_fasta_file('data/3UTRs/Drosophila_melanogaster-3utr.fa', ortholog_splits, oma_to_ncbi, flybase_to_ncbi, args.kmer, args.maxSeqLen, queue)
    # Process files in parallel
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(process_fasta_file, [f for f in fasta_files if species in f and 'utr' in f], ortholog_splits, ncbi_to_oma, flybase_to_geneID, geneID_to_ncbi, args.kmer, queue)
            for species in speciesList
        ]
        for future in tqdm(futures, desc="Processing FASTA files"):
            future.result()  # Wait for each file to complete

    # Signal the writer thread to exit
    queue.put(None)
    writer.join()

if __name__ == "__main__":
    main()