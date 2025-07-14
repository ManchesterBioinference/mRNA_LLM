import argparse

def extract_motif_ids(input_file):
    motif_ids = []
    with open(input_file, 'r') as f:
        lines = f.readlines()
    # Find the section header
    section_found = False
    table_start = False
    for line in lines:
        if not section_found and line.strip().startswith('DATABASE AND MOTIFS'):
            section_found = True
        elif section_found and line.strip().startswith('MOTIF ID'):
            table_start = True
            continue
        elif table_start and line.strip().startswith('-----'):
            continue
        elif table_start:
            if not line.strip():
                break  # End of table
            parts = line.split()
            if parts:
                motif_ids.append(parts[1])  # ID column is the second column
    return motif_ids

def main():
    parser = argparse.ArgumentParser(description='Extract motif IDs from mast.txt')
    parser.add_argument('--input', default='output/mastFilter_out/mast.txt', help='Input mast.txt file')
    parser.add_argument('--output', default='output/mastFilter_out/motifsToKeep.txt', help='Output file for motif IDs')
    args = parser.parse_args()

    motif_ids = extract_motif_ids(args.input)
    with open(args.output, 'w') as out_f:
        for motif_id in motif_ids:
            out_f.write(' --inc ' +motif_id)

if __name__ == '__main__':
    main()