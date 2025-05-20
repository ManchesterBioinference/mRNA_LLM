# extract 20% independent_test_set, do 5-fold split in remaining data
from Bio import SeqIO
import os
from tqdm import tqdm
import random
import argparse
import yaml
import gzip



# Load NCBI-to-OMA gene ID mappings
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
    missingCount = 0
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip().split()
            if line[5] in ncbi_to_oma.keys():
                geneID_to_ncbi[line[1]] = line[5]
            #else:
            #    geneID_to_ncbi[line[1]] = 'missing_'+str(missingCount)
            #    missingCount += 1
            #    ncbi_to_oma['missing_'+str(missingCount)] = 'missing_'+str(missingCount)
    return geneID_to_ncbi

def load_flybase_geneID_mapping(file_path):
    flybase_to_geneID = {}
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip().split()
            flybase_to_geneID[line[1]] = line[0]
    return flybase_to_geneID

# Load ortholog groups and assign to splits
def load_and_split_ortholog_groups(file_path, geneSet,ncbi_to_oma,geneID_to_ncbi,flybase_to_geneID):

    # Load oma to oma groups dict
    ortholog_splits = {}
    with gzip.open(file_path, 'rt') as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            group_id = parts[0]
            oma_ids = parts[2:]

            for oma_id in oma_ids:
                ortholog_splits[oma_id] = group_id #(split, group_id)

    # filter out oma Groups not in geneSet
    groupsToKeep = set()
    missing_count = 0
    mOmaGroups = 0
    mNcbiToOma = 0
    mGeneIDToNcbi = 0
    mFlybaseToGeneID = 0
    for gene in sorted(geneSet):
        if gene in flybase_to_geneID:
            gene_id = flybase_to_geneID[gene]
            if gene_id in geneID_to_ncbi:
                ncbi_id = geneID_to_ncbi[gene_id]
                if ncbi_id in ncbi_to_oma:
                    oma_id = ncbi_to_oma[ncbi_id]
                    if oma_id in ortholog_splits:
                        groupsToKeep.add(ortholog_splits[oma_id])
                    else:
                        name = 'missing_'+str(mOmaGroups)+'_omaGroups'
                        ortholog_splits[oma_id] = name
                        groupsToKeep.add(name)
                        missing_count += 1
                        mOmaGroups += 1
                else:
                    name = 'missing_'+str(mNcbiToOma)+'_ncbiToOma'
                    ncbi_to_oma[ncbi_id] = name
                    ortholog_splits[name] = name
                    groupsToKeep.add(name)
                    missing_count += 1
                    mNcbiToOma += 1
            else:
                name = 'missing_'+str(mGeneIDToNcbi)+'_geneIDToNcbi'
                geneID_to_ncbi[gene_id] = name
                ncbi_to_oma[name] = name
                ortholog_splits[name] = name
                groupsToKeep.add(name)
                missing_count += 1
                mGeneIDToNcbi += 1
        else:
            name = 'missing_'+str(mFlybaseToGeneID)+'_flybaseToGeneID'
            flybase_to_geneID[gene] = name
            geneID_to_ncbi[name] = name
            ncbi_to_oma[name] = name
            ortholog_splits[name] = name
            groupsToKeep.add(name)
            missing_count += 1
            mFlybaseToGeneID += 1

    # Shuffle the OMA group IDs
    groupsToKeep = sorted(list(groupsToKeep))
    random.seed(42)
    random.shuffle(groupsToKeep)

    # Determine split indices
    train_split = int(0.7 * len(groupsToKeep))
    dev_split = int(0.85 * len(groupsToKeep))

    # Assign splits based on shuffled order
    omaG_to_split = {}
    for i, omaG in enumerate(groupsToKeep):
        if i < train_split:
            split = 'train'
        elif i < dev_split:
            split = 'dev'
        else:
            split = 'test'
        omaG_to_split[omaG] = split

    # add the split to the oma to oma groups dict
    trID_to_split = {}
    for g in geneSet:
        omaG = ortholog_splits[ncbi_to_oma[geneID_to_ncbi[flybase_to_geneID[g]]]]
        if omaG in groupsToKeep:
            trID_to_split[g] = (omaG_to_split[omaG],omaG)

    return trID_to_split

def seq2kmer(seq, k):
    return seq
    kmer = [seq[x:x+k] for x in range(len(seq)+1-k)]
    kmers = " ".join(kmer)
    return kmers

def loadData(input_path):
    input_file = input_path

    seq_list = []
    geneSet = set()
    
    for seq_record in tqdm(SeqIO.parse(input_file, "fasta")):
        seq_record.seq = seq_record.seq.upper().replace('T', 'U')
        seq_list.append(seq_record)
        geneSet.add(seq_record.description.split(' ')[2])
    
    return seq_list, geneSet

def extract_test_and_split_train_vali(seq_list, ortholog_splits,save_path, kmer):
    splitDict = {'train': [],'dev':[],'test':[]}
    for s in seq_list:
        gn_id = s.description.split(' ')[2]
        s.seq = seq2kmer(s.seq,kmer)
        split,omaG = ortholog_splits[gn_id]
        s.description += ' ' +str(omaG)
        splitDict[split].append(s)

    os.makedirs(save_path, exist_ok=True)
    for k in splitDict.keys():
        print(f'{k}: {len(splitDict[k])}')
        with open(f"{save_path}/{k}.fasta", 'w') as f:
            SeqIO.write(splitDict[k], f, "fasta-2line")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument( "--params", default='params.yaml', type=str, help="Path to the YAML file containing parameters.",)
    parser.add_argument( "--data_dir", default="output/data/utrDecayRates.fasta", type=str, help="The input fasta data dir. Should contain the sequences for the task.",)
    parser.add_argument( "--output_dir", default=None, type=str, help="Path where the processed .tsv files are saved.",)
    parser.add_argument( "--kmer", default=4, type=int, help="The kmer used by the model",)
    parser.add_argument("--oma_groups", help="Path to oma-groups.txt file.")
    parser.add_argument("--oma_ncbi_map", help="Path to oma-ncbi.txt file.")
    parser.add_argument("--ncbi_geneID_map", help="Path to gene2refseq_Dmel.txt file.")
    parser.add_argument("--geneID_flybase_map", help="Path to GeneID_to_Flybase.txt file.")

    args = parser.parse_known_args()[0]

    # Read parameters from YAML file
    if args.params:
        with open(args.params, 'r') as file:
            yaml_params = yaml.safe_load(file)
            for key, value in yaml_params['preprocessFlyUTRs'].items():
                if key != 'data_dir':
                    parser.set_defaults(**{key: value})
            for key, value in yaml_params['preprocessData'].items():
                parser.set_defaults(**{key: value})

    args = parser.parse_args()

    seqList, geneSet = loadData(args.data_dir)

    # Load mappings and ortholog groups
    ncbi_to_oma = load_ncbi_oma_mapping(args.oma_ncbi_map)
    geneID_to_ncbi = load_geneID_ncbi_mapping(args.ncbi_geneID_map, ncbi_to_oma)
    flybase_to_geneID = load_flybase_geneID_mapping(args.geneID_flybase_map)
    ortholog_splits = load_and_split_ortholog_groups(args.oma_groups, geneSet, ncbi_to_oma, geneID_to_ncbi, flybase_to_geneID)
    
    extract_test_and_split_train_vali(seqList, ortholog_splits, args.output_dir, args.kmer)

if __name__ == "__main__":
    main()
