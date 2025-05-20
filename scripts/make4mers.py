def read_input(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return lines

def identify_original_string(kmers):
    return ''.join(kmer[0] for kmer in kmers) + kmers[-1][1:]

def create_kmers(original_string, k):
    return [original_string[i:i+k] for i in range(0, len(original_string) - k + 1)]

def process_file(file_path):
    lines = read_input(file_path)
    
    # Skip the header line
    header = lines[0]
    data_lines = lines[1:]
    
    results = []
    for line in data_lines:
        parts = line.strip().split('\t')
        kmers = parts[0].split()
        label = parts[1]
        
        original_string = identify_original_string(kmers)
        kmers_of_4 = create_kmers(original_string, 4)
        results.append((' '.join(kmers_of_4), label))
    
    return header, results

def write_output(file_path, header, results):
    with open(file_path, 'w') as file:
        file.write(header)
        for kmers_of_4, label in results:
            file.write(f"{kmers_of_4}\t{label}\n")

def main():
    input_file_path = ['output/data/train_copy.tsv','output/data/test_copy.tsv','output/data/dev_copy.tsv']
    output_file_path = ['output/data/train.tsv','output/data/test.tsv','output/data/dev.tsv']
    
    for inF,outF in zip(input_file_path,output_file_path):
        header, results = process_file(inF)
        write_output(outF, header, results)

if __name__ == "__main__":
    main()