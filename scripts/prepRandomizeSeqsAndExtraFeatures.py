import argparse
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
import pickle

from myShap.ferret_extras import BaseExplainer, Explanation

def main():
    parser = argparse.ArgumentParser(description="Process FASTA files based on specified method.")
    parser.add_argument("--params", help="Path to a parameters file (not used in current logic).")
    parser.add_argument("--sequence_file", required=True, help="Path to the input FASTA file.")
    parser.add_argument("--process", required=True, choices=['seqImpact', 'extraFeatImpact', '5UTRImpact', '3UTRImpact'], help="Processing mode: 'seqImpact', 'extraFeatImpact', '5UTRImpact', '3UTRImpact'.")
    parser.add_argument("--output_file", required=True, help="Path to the output FASTA file.")
    parser.add_argument("--importance", help="path to the importance file")

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

    importance = pickle.load(open(args.importance, 'rb'))
    idToPrediction = {imp.id: imp.prediction for imp in importance}

    newRecords = []
    if args.process == "seqImpact":
        # keep the original prediction associated with the extraFeatures (E1S1) and the extraFeatures the same (E1). Change the sequence (S)
        # id is prediction from E1S1, sample is E1S2 through E1SN so that the E1SN prediction can be compared to the E1S1 prediction. 
        for i in range(len(records)):
            first_record_description = records[i].description.split(' ') 
            if first_record_description[1] not in idToPrediction:
                print(f"Warning: No prediction found for {first_record_description[1]}. Skipping this record.")
                continue
            id = idToPrediction[first_record_description[1]] #original prediction (E1S1)
            actualTE = records[i].id
            first_record_description = ' '.join(first_record_description[1:])
            for j,record in enumerate(records):
                if j == i:
                    continue
                pertID = record.description.split(' ')[1] # ID for the perturbed sequence
                if pertID not in idToPrediction:
                    continue
                r = SeqRecord(record.seq, record.id, record.name, record.description, record.dbxrefs, record.features, record.annotations, record.letter_annotations)
                r.id = str(id)
                r.description = first_record_description+' '+pertID+' '+actualTE #move the actual TE to the end # defines the extraFeatures (E)
                newRecords.append(r)
    elif args.process == "extraFeatImpact":
        # keep the original prediction associated with the sequence (E1S1) and the sequence the same (S1). Change the extraFeatures (E).
        # id is prediction from E1S1, sample is E2S1 through ENS1 so that the E2S1 prediction can be compared to the E1S1 prediction.
        for i in range(len(records)):
            if records[i].description.split(' ')[1] not in idToPrediction:
                continue
            id = str(idToPrediction[records[i].description.split(' ')[1]]) #original prediction (E1S1)
            actualTE = records[i].id
            first_record_sequence = records[i].seq # defines the sequence (S)
            origTRID = records[i].description.split(' ')[1] # original extraFeatures (E1)
            for j,record in enumerate(records):
                if j == i:
                    continue
                r = SeqRecord(first_record_sequence, id, record.name, ' '.join(record.description.split()[1:])+' '+origTRID+' '+actualTE, record.dbxrefs, record.features, record.annotations, record.letter_annotations)
                newRecords.append(r)
    elif args.process == "5UTRImpact":
        # keep the original prediction associated with the sequence (E1S1) and the 3' UTR the same (3UTR). Change the 5' UTR (5UTR).
        # id is prediction from E1S1, sample is E1S(5.2,3.1) through E1S(5.N,3.1) so that the E1S(5.N,3.1) prediction can be compared to the E1S1 prediction.
        for i in range(len(records)):
            if records[i].description.split(' ')[1] not in idToPrediction:
                continue
            id = idToPrediction[records[i].description.split(' ')[1]] #original prediction (E1S1)
            actualTE = records[i].id
            first_record_3utr = str(records[i].seq.split(',')[1]) #3.1
            first_record = records[i] # E1
            for j,record in enumerate(records):
                if j == i:
                    continue
                r = SeqRecord(record.seq, record.id, record.name, record.description, record.dbxrefs, record.features, record.annotations, record.letter_annotations)
                pertID = record.description.split(' ')[1] # ID for the perturbed 5' UTR
                variable5UTR = str(record.seq).split(',')[0]
                r.id = str(id)
                r.description = ' '.join(first_record.description.split(' ')[1:])+' '+pertID+' '+actualTE #move the actual TE to the end
                r.seq = ','.join([variable5UTR, first_record_3utr])
                newRecords.append(r)
    elif args.process == "3UTRImpact":
        # keep the original prediction associated with the sequence (E1S1) and the 5' UTR the same (5UTR). Change the 3' UTR (3UTR).
        # id is prediction from E1S1, sample is E1S(5.1,3.2) through E1S(5.1,3.N) so that the E1S(5.1,3.N) prediction can be compared to the E1S1 prediction.
        for i in range(len(records)):
            if records[i].description.split(' ')[1] not in idToPrediction:
                continue
            id = idToPrediction[records[i].description.split(' ')[1]] #original prediction (E1S1)
            actualTE = records[i].id
            first_record_5utr = str(records[i].seq.split(',')[0])
            first_record = records[i]
            for j,record in enumerate(records):
                if j == i:
                    continue
                r = SeqRecord(record.seq, record.id, record.name, record.description, record.dbxrefs, record.features, record.annotations, record.letter_annotations)
                pertID = record.description.split(' ')[1] # ID for the perturbed 3' UTR
                variable3UTR = str(record.seq).split(',')[1]
                r.id = str(id)
                r.description = ' '.join(first_record.description.split(' ')[1:])+' '+pertID+' '+actualTE #move the actual TE to the end
                r.seq = ','.join([first_record_5utr, variable3UTR])
                newRecords.append(r)
    else:
        # This case should not be reached due to argparse choices
        print(f"Error: Unknown process type '{args.process}'.")
        return

    try:
        SeqIO.write(newRecords, args.output_file, "fasta")
        print(f"Processed sequences saved to {args.output_file}")
    except Exception as e:
        print(f"Error writing output file: {e}")

if __name__ == "__main__":
    main()