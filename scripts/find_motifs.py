import os
import pandas as pd
import numpy as np
import argparse
import pickle
import yaml


def filterWithLOWESS(importance, frac=0.3, save_file_dir=None):
    """ Filters the importance scores using LOWESS smoothing to remove noise.
    Args:
        importance (list): List of Importance objects containing attention scores.
    Returns:
        list: Importance objects that have been filtered using LOWESS.
    """
    import dvclive
    import statsmodels.api as sm
    import matplotlib.pyplot as plt

    actuals_arr = []
    preds_arr = []
    for x in importance:
        actuals_arr.append(x.actual)
        preds_arr.append(x.prediction)
    actuals_arr = np.array(actuals_arr)
    preds_arr = np.array(preds_arr)
    residuals = preds_arr - actuals_arr

    # Fit LOESS model: residuals ~ actuals
    # The frac parameter controls the smoothing. Adjust as needed.
    lowess_fit = sm.nonparametric.lowess(residuals, actuals_arr, frac=frac)

    # Get the smoothed y-values (predicted residuals from LOESS)
    loess_x = lowess_fit[:, 0]
    loess_y_pred_residuals = lowess_fit[:, 1]

    # Sort by actuals_arr for correct plotting of LOESS curve
    sort_idx = np.argsort(actuals_arr)
    actuals_sorted = actuals_arr[sort_idx]
    residuals_sorted_by_actuals = residuals[sort_idx]

    # Interpolate LOESS predictions to match the original residuals' order for deviation calculation
    # This is important because lowess returns sorted x values
    loess_pred_residuals_for_deviation = np.interp(actuals_arr, loess_x, loess_y_pred_residuals)

    # Calculate the deviation of each residual from the LOESS curve
    deviations_from_loess = residuals - loess_pred_residuals_for_deviation

    # Calculate the standard deviation of these deviations
    std_dev_of_deviations = np.std(deviations_from_loess)

    # Define a threshold for flagging significant deviations (e.g., 2 standard deviations)
    threshold = 2 * std_dev_of_deviations

    # Flag samples whose residuals deviate significantly
    outlier_indices = np.where(np.abs(deviations_from_loess) > threshold)[0]
    outlier_actuals = actuals_arr[outlier_indices]
    outlier_residuals = residuals[outlier_indices]

    print(f"Number of points flagged as outliers: {len(outlier_indices)}")
    print(f"Standard deviation of deviations from LOESS: {std_dev_of_deviations:.4f}")
    print(f"Threshold for outliers (2*std_dev): {threshold:.4f}")
    for i in outlier_indices:
        print(f"  Filtered out sequence: {importance[i].id}")

    # Plotting
    fig = plt.figure(figsize=(12, 7))
    plt.scatter(actuals_arr, residuals, label='Residuals (Actual - Predicted)', alpha=0.5, s=10)
    plt.plot(actuals_sorted, loess_y_pred_residuals[np.argsort(loess_x)], color='red', linewidth=2, label='LOESS fit to residuals')
    plt.scatter(outlier_actuals, outlier_residuals, color='green', s=50, label=f'Flagged Outliers ({len(outlier_indices)})', edgecolor='black')

    # Plot lines for threshold
    plt.plot(actuals_sorted, loess_y_pred_residuals[np.argsort(loess_x)] + threshold, color='orange', linestyle='--', label=f'+{threshold:.2f} (Threshold)')
    plt.plot(actuals_sorted, loess_y_pred_residuals[np.argsort(loess_x)] - threshold, color='orange', linestyle='--', label=f'-{threshold:.2f} (Threshold)')


    plt.xlabel("Actual Values")
    plt.ylabel("Residuals (Actual - Predicted)")
    plt.title("Residual Analysis with LOESS Fit")
    plt.axhline(0, color='gray', linestyle=':', linewidth=0.8)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    if save_file_dir is not None:
        os.makedirs(save_file_dir, exist_ok=True)
        fig.savefig(os.path.join(save_file_dir, "residuals_with_loess_fit.png"))
    else:
        with dvclive.Live(os.path.join(os.getcwd(), 'dvclive/findMotifs')) as live:
            live.log_image("residuals_with_loess_fit.png", fig)
    plt.close(fig)

    return [importance[i] for i in range(len(importance)) if i not in outlier_indices]

def get_minimal_supersets(sequence_list):
    """
    Filters a list of sequences, returning only those that are not
    subsequences of any other sequence in the input list.
    If sequence A is a subsequence of B (and A != B), A is removed.
    """
    if not sequence_list:
        return []

    # Get unique sequences and sort by length descending (longer sequences first)
    # This helps in the subsequence check logic.
    unique_sequences = sorted(list(set(sequence_list)), key=len, reverse=True)
    
    minimal_supersets = []
    for i, current_seq in enumerate(unique_sequences):
        is_subsequence_of_another = False
        for j, other_seq in enumerate(unique_sequences):
            if i == j:
                continue
            # Check if current_seq is a subsequence of other_seq
            if current_seq in other_seq and len(current_seq) < len(other_seq):
                is_subsequence_of_another = True
                break

        if not is_subsequence_of_another:
            minimal_supersets.append(current_seq)
        
    return minimal_supersets

def _update_sequences_in_dict(target_dict, current_processing_key, new_sequences_raw):
    """
    Updates the target_dict with new sequences, ensuring no duplicates or
    subsequences are added, and removing existing subsequences if a new
    supersequence is added.
    """
    if not new_sequences_raw:
        return

    # 1. Filter new_sequences_raw to get minimal supersets among them
    filtered_new_sequences = get_minimal_supersets(list(set(new_sequences_raw)))

    sequences_to_potentially_add = []

    for new_seq in filtered_new_sequences:
        is_sub_or_dup_of_existing = False
        # 2a. Check if new_seq is a duplicate or subsequence of any sequence already in target_dict
        for existing_sequences_list in target_dict.values():
            for existing_seq in existing_sequences_list:
                if new_seq == existing_seq or (new_seq in existing_seq and len(new_seq) < len(existing_seq)):
                    is_sub_or_dup_of_existing = True
                    break
            if is_sub_or_dup_of_existing:
                break
        
        if is_sub_or_dup_of_existing:
            continue # Discard new_seq

        # 2b. If new_seq is not discarded, identify and remove existing sequences in target_dict
        # that are subsequences of new_seq.
        
        # Iterate over a copy of items for safe modification
        for key, existing_list in list(target_dict.items()): 
            indices_to_remove_from_this_list = []
            for idx, existing_seq in enumerate(existing_list):
                if existing_seq in new_seq and len(existing_seq) < len(new_seq):
                    indices_to_remove_from_this_list.append(idx)
            
            # Remove in reverse order to maintain correct indices
            for idx in sorted(indices_to_remove_from_this_list, reverse=True):
                del target_dict[key][idx]
            
            if not target_dict[key]: # If list becomes empty
                del target_dict[key]

        sequences_to_potentially_add.append(new_seq)

    # 3. Consolidate sequences for the current_processing_key
    if sequences_to_potentially_add:
        current_key_existing_sequences = target_dict.get(current_processing_key, [])
        combined_sequences_for_key = current_key_existing_sequences + sequences_to_potentially_add
        
        # Final filter for the specific key's list
        final_sequences_for_key = get_minimal_supersets(list(set(combined_sequences_for_key)))
        
        if final_sequences_for_key:
            target_dict[current_processing_key] = final_sequences_for_key
        elif current_processing_key in target_dict: # If list became empty after all operations
            del target_dict[current_processing_key]


def contiguous_regions(imp, condition, len_thres=6, max_len=15):
    """
    Modified from and credit to: https://stackoverflow.com/a/4495197/3751373
    Finds contiguous True regions of the boolean array "condition". Returns
    a 2D array where the first column is the start index of the region and the
    second column is the end index.

    Arguments:
    condition -- custom conditions to filter/select high attention 
            (list of boolean arrays)
    
    Keyword arguments:
    len_thres -- int, specified minimum length threshold for contiguous region 
        (default 5)
    max_len -- int, specified maximum length threshold for contiguous region 
        (default 15)

    Returns:
    idx -- Index of contiguous regions in sequence

    """
    
    # Find the indicies of changes in "condition"
    d = np.diff(condition) 
 
    idx, = d.nonzero() 


    idx += 1

    if condition[0]:
        # If the start of condition is True prepend a 0
        idx = np.r_[0, idx] 

    if condition[-1]:
        # If the end of condition is True, append the length of the array
        idx = np.r_[idx, condition.size] # Edit

    # Reshape the result into two columns
    idx.shape = (-1,2)

    new_idx = []
    def _split_region_if_needed(imp, start, end, len_thres, max_len, new_idx):
        """
        Recursively checks if a region is within length limits (len_thres, max_len).
        If too long, splits it based on the minimum score and recurses on sub-regions.
        Appends valid regions to new_idx.
        """
        if start >= end: # Base case: empty or invalid region
            return

        # Calculate the actual sequence length of the region using token lengths
        region_len = sum(imp.lengths[start:end])

        if region_len < len_thres:
            # Base case: Region is too short, discard.
            return
        elif max_len is None or end-start == 1 or region_len <= max_len:
            # Base case: Region is within the valid length range, accept it.
            new_idx.append([start, end])
            return
        else:
            # Recursive case: Region is too long, split it.
            scores = imp.scores.flatten()[start:end]

            # Find the index of the minimum score relative to the start of the slice
            if np.all(scores == scores[0]):
                # If all scores are the same, split near the middle token count
                min_score_relative_idx = (end - start) // 2
            else:
                # Find all indices with the minimum score relative to the start of the slice
                min_score = np.min(scores)
                min_indices_relative = np.where(scores == min_score)[0]

                # If there's only one minimum, use it directly
                if len(min_indices_relative) == 1:
                    min_score_relative_idx = min_indices_relative[0]
                else:
                    # Calculate the center index of the scores slice
                    center_index_relative = (len(scores) - 1) / 2.0
                    # Calculate distances from the center for each minimum index
                    distances = np.abs(min_indices_relative - center_index_relative)
                    # Find the index within min_indices_relative that corresponds to the minimum distance
                    closest_idx_in_min_indices = np.argmin(distances)
                    # Get the actual relative index of the minimum score closest to the center
                    min_score_relative_idx = min_indices_relative[closest_idx_in_min_indices]

            split_idx = start + min_score_relative_idx

            # Ensure the split point guarantees progress to avoid infinite recursion
            # If the minimum score is at the very beginning (index 0 relative to start),
            # the split point would be 'start'. We must advance it to split effectively.
            if split_idx == start:
                split_idx = start + 1 # Split after the first token

            # Recursively process the left and right sub-regions
            # Note: The split ensures that neither sub-region is identical to the original [start, end)
            # if split_idx was advanced from start.
            _split_region_if_needed(imp, start, split_idx, len_thres, max_len, new_idx)
            _split_region_if_needed(imp, split_idx, end, len_thres, max_len, new_idx)


    # In contiguous_regions function, replace the original loop with this:
    for start, end in idx:
        _split_region_if_needed(imp, start, end, len_thres, max_len, new_idx)

    
    seq_idx = [(int(sum(imp.lengths[:tokenCoord[0]])), int(sum(imp.lengths[:tokenCoord[1]]))) for tokenCoord in new_idx]
    return np.array(seq_idx)

def find_high_attention(imp, min_len=5, positive = True, trainSHAPmean=0.0, trainSHAPstd=1.0, numStds=1.25, **kwargs):
    """
    With an array of attention scores as input, finds contiguous high attention 
    sub-regions indices having length greater than min_len.
    
    Arguments:
    score -- numpy array of attention scores for a sequence

    Keyword arguments:
    min_len -- int, specified minimum length threshold for contiguous region 
        (default 5)
    **kwargs -- other input arguments:
        cond -- custom conditions to filter/select high attention 
            (list of boolean arrays)
    
    Returns:
    motif_regions -- indices of high attention regions in sequence

    """
    scores = imp.scores.flatten()*len(imp.tokens)
    if np.sum(scores) == 0:
        return [], [], [], [], [], []
    if positive:
        cond1 = (scores > trainSHAPmean + numStds*trainSHAPstd)
        cond2 = (scores > 0)
        controlCond = (scores < trainSHAPmean)
    else:
        cond1 = (scores < trainSHAPmean - numStds*trainSHAPstd)
        cond2 = (scores < 0)
        controlCond = (scores > trainSHAPmean)

    cond = np.asarray(list(map(all, zip(cond1, cond2))))

    # find important contiguous region with high attention
    motif_regions = contiguous_regions(imp,cond,min_len,max_len=None)

    # isolate regions of interest and control for motif enrichment analysis
    #TODO - confirm that these thresholds work well
    interestIdx = [] #contiguous_regions(imp,score > 0,10,max_len=None)
    controlIdx = contiguous_regions(imp,controlCond,10,max_len=None)

    fullSeq = ''.join(imp.tokens)

    interestRegions = []
    for x in interestIdx:
        seq = fullSeq[x[0]:x[1]]
        if '[SEP]' in seq:
            seq = seq.split('[SEP]')
            for s in seq:
                if len(s) > 0:
                    interestRegions.append(s)
        else:
            interestRegions.append(seq)

    controlRegions = []
    controlPositions = []
    for x in controlIdx:
        controlPositions.append((fullSeq.index('['), fullSeq.index(']'), x[0], x[1]))
        seq = fullSeq[x[0]:x[1]]
        if '[SEP]' in seq:
            seq = seq.split('[SEP]')
            for s in seq:
                if len(s) > 0:
                    controlRegions.append(s)
        else:
            controlRegions.append(seq)
    motif_seqs = []
    motifPositions = []
    for x in motif_regions:
        motifPositions.append((fullSeq.index('['), fullSeq.index(']'), x[0], x[1]))
        seq = fullSeq[x[0]:x[1]]
        if '[SEP]' in seq:
            seq = seq.split('[SEP]')
            for s in seq:
                if len(s) > 0:
                    motif_seqs.append(s)
        else:
            motif_seqs.append(seq)

    return motif_regions, controlRegions, interestRegions, motif_seqs, motifPositions, controlPositions

def _reorder_lengths_median_alternating(lengths_list):
    """
    Reorders a list of numbers to have the middle number first,
    then alternates one lower and one higher, so that the median
    number is first and the extreme values come last.
    """
    if not lengths_list:
        return []

    # Sort the list to easily find the median and subsequent elements
    sorted_lengths = sorted(lengths_list)
    n = len(sorted_lengths)
    
    new_order_lengths = []
    
    # Determine the starting middle index.
    # For odd n, (n-1)//2 is the exact middle.
    # For even n, (n-1)//2 is the lower of the two middle elements, which will be picked first.
    mid_idx = (n - 1) // 2
    
    # Add the first middle element
    new_order_lengths.append(sorted_lengths[mid_idx])
    
    # Initialize pointers for elements to the left and right of the initial middle element
    l_ptr = mid_idx - 1
    r_ptr = mid_idx + 1
    
    # Loop until all elements from sorted_lengths are added to new_order_lengths,
    # alternating between picking from the left and right sides of the initial middle.
    while l_ptr >= 0 or r_ptr < n:
        # Add element from the left side (lower than current median elements)
        if l_ptr >= 0:
            new_order_lengths.append(sorted_lengths[l_ptr])
            l_ptr -= 1
        
        # Add element from the right side (higher than current median elements)
        if r_ptr < n: 
            new_order_lengths.append(sorted_lengths[r_ptr])
            r_ptr += 1
            
    return new_order_lengths
### make full pipeline
def motif_analysis(importances,
                   neg_seqs, # Note: This argument seems unused currently, relying on control_seqs derived from importances
                   pos_atten_scores, # Note: This argument seems unused currently
                   rbp_using, # Note: This argument seems unused currently
                   window_size = 24,
                   min_len = 4,
                   pval_cutoff = 0.005,
                   min_n_motif = 3,
                   align_all_ties = True,
                   save_file_dir = None,
                   positive = True,
                   trainSHAPmean = None,
                   trainSHAPstd = None,
                   args = None,
                   **kwargs
                  ):
 
    
    from Bio import motifs
    from Bio.Seq import Seq
    from scipy.stats import hypergeom
    import statsmodels.stats.multitest as multi
    import random
    random.seed(42)
    
    verbose = False
    if 'verbose' in kwargs:
        verbose = kwargs['verbose']
    
    allow_multi_match = kwargs.get('allow_multi_match', False) # Get allow_multi_match from kwargs
    p_adjust = kwargs.get('p_adjust', 'fdr_bh') # Get p_adjust method

    if verbose:
        print("*** Begin motif analysis ***")

    ## find the motif regions
    if verbose:
        print("* Finding high attention motif regions and control/interest sequences")
    control_seqs = {}
    interest_seqs = {}
    motif_seq = {} # This dictionary will store motif sequences per importance score index
    control_positions = {}
    motif_positions = {}
    for i, imp in enumerate(importances):
        # handle kwargs
        if 'atten_cond' in kwargs:
            motif_regions, controlRegions, interestRegions, current_motif_s_list, motifPositions, controlPositions = find_high_attention(imp, min_len=min_len, positive=positive, trainSHAPmean=trainSHAPmean, trainSHAPstd=trainSHAPstd, cond=kwargs['atten_cond'])
        else:
            motif_regions, controlRegions, interestRegions, current_motif_s_list, motifPositions, controlPositions = find_high_attention(imp, min_len=min_len, positive=positive, trainSHAPmean=trainSHAPmean, trainSHAPstd=trainSHAPstd)
        
        # Collect control and interest sequences (regions) using the new helper function
        if controlRegions: # Check if the list is not empty
            _update_sequences_in_dict(control_seqs, str(imp.id), controlRegions,)
        if interestRegions: # Check if the list is not empty
            interest_seqs[str(imp.id)] = interestRegions
        if current_motif_s_list: # Check if the list is not empty
            _update_sequences_in_dict(motif_seq, str(imp.id), current_motif_s_list)
        if motifPositions: # Check if the list is not empty
            motif_positions[str(imp.id)] = motifPositions
        if controlPositions: # Check if the list is not empty
            control_positions[str(imp.id)] = controlPositions

    positive_label = 'positive' if positive else 'negative'
    output_dir = os.path.join(save_file_dir, positive_label)
    os.makedirs(output_dir, exist_ok=True)
    # save control and interest sequences to fasta files
    count = 0
    with open(os.path.join(output_dir,args.control_file), 'w') as f:
        #for k in motif_seq.keys():
        #    for j, s in enumerate(motif_seq[k]):
        #        strng = ''.join(interest_seqs[k])
        #        shuffled_string = [random.choice(strng) for _ in range(len(s))]
        #        new_string = ''.join(shuffled_string)
        #        f.write(f">interest_shuffled_{k+1000}_{j}\n{new_string}\n")
        # for k in control_seqs.keys():
        #     for j, s in enumerate(control_seqs[k]):
        #         f.write(f">control_seq_{k}_{j}\n{s}\n")
        contSeq = ''.join([s for k in control_seqs.keys() for s in control_seqs[k]])
        # Collect all motif sequences and their lengths
        motif_lengths = [len(s) for seq_list in motif_seq.values() for s in seq_list if len(s) >= 8]

        if motif_lengths:
            # Order the list longest to shortest
            motif_lengths = _reorder_lengths_median_alternating(motif_lengths)
            
            num_unique_motif_lengths = len(motif_lengths)
            current_motif_length_idx = 0 # To cycle through the sorted motif_lengths
            
            current_pos_in_contSeq = 0
            # Loop through contSeq, grabbing chunks with lengths from the circularized motif_lengths list
            while current_pos_in_contSeq < len(contSeq):
                # Get the next length from the (circular) list of motif_lengths
                chunk_len = motif_lengths[current_motif_length_idx]
            
                # Check if the remaining contSeq is long enough for a chunk of this_length
                if current_pos_in_contSeq + chunk_len <= len(contSeq):
                    chunk = contSeq[current_pos_in_contSeq : current_pos_in_contSeq + chunk_len]
                    f.write(f">control_chunk_motif_len_pattern_{count}\n{chunk}\n")
                    current_pos_in_contSeq += chunk_len
                    count += 1 # Increment the unique counter for FASTA headers
                # Move to the next length for the next iteration, cycling back to the start if needed
                    current_motif_length_idx = (current_motif_length_idx + 1) % num_unique_motif_lengths
            
                else:
                    if current_motif_length_idx != num_unique_motif_lengths -1:
                        # Move to the next length for the next iteration, cycling back to the start if needed
                        current_motif_length_idx = (current_motif_length_idx + 1) % num_unique_motif_lengths
            
                        continue
                    # Not enough contSeq left to make a chunk of the current chunk_len
                    break 

    with open(os.path.join(output_dir,args.interest_file), 'w') as f:
        for k in interest_seqs.keys():
            for j, s in enumerate(interest_seqs[k]):
                f.write(f">interestSeq_{k}.{j}\n{s}\n")
    # save motif sequences to fasta files
    with open(os.path.join(output_dir,args.motif_file), 'w') as f:
        for k in motif_seq.keys(): # This uses the updated motif_seq dictionary
            for j, s in enumerate(motif_seq[k]):
                if len(s) >= 8: # Only save sequences longer than 8
                    f.write(f">motifSeq_{k}.{j}\n{s}\n")
    with open(os.path.join(output_dir,args.control_positions), 'w') as f:
        f.write('trID\tsepStart\tsepEnd\tstart\tend\n') # Header for control positions
        for k in control_positions.keys():
            for j, s in enumerate(control_positions[k]):
                f.write(f"{k}\t{s[0]}\t{s[1]}\t{s[2]}\t{s[3]}\n")
    with open(os.path.join(output_dir,args.motif_positions), 'w') as f:
        f.write('trID\tsepStart\tsepEnd\tstart\tend\n') # Header for motif positions
        for k in motif_positions.keys():
            for j, s in enumerate(motif_positions[k]):
                f.write(f"{k}\t{s[0]}\t{s[1]}\t{s[2]}\t{s[3]}\n")
    return

def getTrainSHAPStats(trainSHAPpath):
    importances = pickle.load(open(trainSHAPpath, 'rb'))
    all_scores = []
    for imp in importances:
        scores = imp.scores.flatten()*len(imp.tokens)
        all_scores.extend(scores)
    all_scores = np.array(all_scores)
    mean_score = np.mean(all_scores)
    std_score = np.std(all_scores)
    return mean_score, std_score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default='params.yaml', type=str, help="Path to the YAML file containing parameters.",)
    parser.add_argument( "--rbp", default=None, type=str, help="rbp using",)
    parser.add_argument( "--data_dir", default=None, type=str, help="The input data dir. Should contain the sequence+label .tsv files (or other data files) for the task.",)
    parser.add_argument( "--predict_dir", default=None, type=str, help="Path where the attention scores were saved. Should contain both pred_results.npy and atten.npy",)
    parser.add_argument( "--window_size", default=10, type=int, help="Specified window size to be final motif length",)
    parser.add_argument( "--min_len", default=5, type=int, help="Specified minimum length threshold for contiguous region",)
    parser.add_argument( "--pval_cutoff", default=0.05, type=float, help="Cutoff FDR/p-value to declare statistical significance",)
    parser.add_argument( "--p_adjust", default='fdr_bh', type=str, help="Multiple testing correction method (e.g., fdr_bh, bonferroni, None)",) # Changed default
    parser.add_argument( "--min_n_motif", default=3, type=int, help="Minimum instance inside motif to be filtered",)
    parser.add_argument( "--align_all_ties", action='store_true', help="Whether to keep all best alignments when ties encountered",)
    parser.add_argument( "--save_file_dir", default='.', type=str, help="Path to save outputs",)
    parser.add_argument( "--motif_file", default='.', type=str, help="Path to save outputs",)
    parser.add_argument( "--control_file", default='.', type=str, help="Path to save outputs",)
    parser.add_argument( "--motif_positions", default='.', type=str, help="Path to save outputs",)
    parser.add_argument( "--control_positions", default='.', type=str, help="Path to save outputs",)
    parser.add_argument( "--interest_file", default='.', type=str, help="Path to save outputs",)
    parser.add_argument( "--verbose", action='store_true', help="Verbosity controller",)
    parser.add_argument("--SHAP", default="output/importance/shap.pkl", type=str, help="The path to the pickled SHAP data for each sample")
    parser.add_argument("--trainSHAP", default=None, type=str, help="The path to the pickled SHAP data for the training set")
    parser.add_argument("--allow_multi_match", action='store_true', help="Allow multiple matches of a motif within a single sequence during counting for hypergeometric test.")


    # TODO: add the conditions
    args = parser.parse_known_args()[0]

    # Read parameters from YAML file
    if args.params:
        with open(args.params, 'r') as file:
            yaml_params = yaml.safe_load(file)
            for key, value in yaml_params['findMotifs'].items():
                parser.set_defaults(**{key: value})

    args = parser.parse_args()

    trainSHAPmean, trainSHAPstd = getTrainSHAPStats(args.trainSHAP)

    importance = pickle.load(open(args.SHAP, 'rb'))
    #importance = filterWithLOWESS(importance, save_file_dir=args.save_file_dir)
    iWithLen = []
    for x in importance:
        x.tokens = [t.replace('T','U') for t in x.tokens] # Ensure U instead of T
        x.lengths = [len(token) for token in x.tokens]
        iWithLen.append(x)


    # Removed loading of unused data (atten_scores, pred, dev)
    # pos_atten_scores = atten_scores[dev_pos.index.values]
    # neg_atten_scores = atten_scores[dev_neg.index.values]
    # assert len(dev_pos) == len(pos_atten_scores)

    # run motif analysis for positive and negative scores
    all_results = {}
    for pos in [True, False]:
        if pos:
            print("\n--- Finding motifs for POSITIVE attention scores ---")
        else:
            print("\n--- Finding motifs for NEGATIVE attention scores ---")
        
        # Pass allow_multi_match and p_adjust via kwargs
        merged_motif_seqs = motif_analysis(iWithLen, 
                                    None, # neg_seqs - unused
                                    None, # pos_atten_scores - unused
                                    args.rbp, # rbp_using - unused
                                    window_size = args.window_size,
                                    min_len = args.min_len,
                                    pval_cutoff = args.pval_cutoff,
                                    min_n_motif = args.min_n_motif,
                                    align_all_ties = args.align_all_ties,
                                    save_file_dir = args.save_file_dir,
                                    verbose = args.verbose,
                                    positive = pos,
                                    p_adjust = args.p_adjust, # Pass p_adjust method
                                    allow_multi_match=args.allow_multi_match,
                                    trainSHAPmean=trainSHAPmean,
                                    trainSHAPstd=trainSHAPstd,
                                    args = args # Pass counting option
                                )
        # label = 'positive' if pos else 'negative'
        # all_results[label] = merged_motif_seqs
        # print(f"--- Completed analysis for {label} scores. Found {len(merged_motif_seqs)} motifs. ---")

    # Optionally, save the combined results dictionary
    # with open(os.path.join(args.save_file_dir, "all_motif_results.pkl"), "wb") as f:
    #    pickle.dump(all_results, f)

if __name__ == "__main__":
    main()


