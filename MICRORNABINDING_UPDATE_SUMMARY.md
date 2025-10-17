# microRNABindingEnrichment.py Update Summary

## Overview
Updated `microRNABindingEnrichment.py` to match the style and methodology of `motifEnrichment.py`.

## Key Changes

### 1. **Length Distribution Matching for Control Regions**
- Added `split_regions_to_lengths_proportional()` function (identical to motifEnrichment.py)
- Control regions for 3'UTR are now split into segments matching the length distribution of the SHAP regions
- Uses a sophisticated algorithm that:
  - Maintains empirical proportion of selected lengths close to motif length frequencies
  - Uses lookahead to avoid creating unusable tails
  - Favors lengths that are currently underrepresented

### 2. **Replaced Fisher Exact Test with Mann-Whitney U Test**
- Removed the Fisher exact test approach that compared transcript-level presence/absence
- Implemented Mann-Whitney U test to compare distributions of microRNA binding site counts
- Applied FDR (False Discovery Rate) correction using Benjamini-Hochberg method
- This matches the statistical approach used in motifEnrichment.py

### 3. **Count Matrix Approach**
- Added `count_mirna_overlaps_per_region()` function
- Creates count matrices where:
  - Rows = individual regions (with descriptive indices: `trID_start-end`)
  - Columns = microRNA families
  - Values = count of overlapping microRNA binding sites
- Saved as parquet files for further analysis

### 4. **Updated Function Names and Structure**
- Replaced `split_5p_3p()` with `split_by_utr()` to match motifEnrichment.py naming
- Both functions do the same thing but split_by_utr also adds a 'length' column

### 5. **Updated Argument Structure**
- Changed from old arguments (`pos_control_file`, `pos_interest_file`, etc.)
- To new arguments matching motifEnrichment.py:
  - `--highSHAP`: Path to high SHAP regions
  - `--highControl`: Path to high SHAP control regions  
  - `--lowSHAP`: Path to low SHAP regions
  - `--lowControl`: Path to low SHAP control regions
  - `--name`: Study name (e.g., 'train', 'dev', 'test')
  - `--mirna_file`: Path to microRNA predictions
  - `--output_dir`: Output directory

### 6. **Output Files**
The script now produces:
- **Count matrices** (parquet format):
  - `pos3.parquet`: High SHAP 3'UTR regions
  - `pcontrol3.parquet`: High SHAP 3'UTR matched controls
  - `neg3.parquet`: Low SHAP 3'UTR regions
  - `ncontrol3.parquet`: Low SHAP 3'UTR matched controls

- **Statistical results** (CSV format):
  - `{name}_pos3_mannwhitneyu_results.csv`: Results comparing high SHAP vs controls
  - `{name}_neg3_mannwhitneyu_results.csv`: Results comparing low SHAP vs controls
  
  Each CSV contains:
  - `motif_mean`: Mean count in SHAP regions
  - `control_mean`: Mean count in control regions
  - `statistic`: Mann-Whitney U statistic
  - `p_value`: Raw p-value
  - `p_adj`: FDR-adjusted p-value

### 7. **Improved Console Output**
- Shows progress bars for control region splitting and overlap counting
- Prints summary statistics at each step
- Displays significant results (p_adj < 0.07) at the end

## Backward Compatibility
- Kept the `run_legacy()` function for backward compatibility with old-style arguments
- Legacy mode still uses Fisher exact test
- The script automatically detects which mode to use based on provided arguments

## Usage Example
```bash
python scripts/microRNABindingEnrichment.py \
    --params params.yaml \
    --name train \
    --mirna_file data/microRNA/Predicted_Targets_Info.default_predictions.txt \
    --highSHAP output/shap/train_high_shap_regions.tsv \
    --highControl output/shap/train_high_control_regions.tsv \
    --lowSHAP output/shap/train_low_shap_regions.tsv \
    --lowControl output/shap/train_low_control_regions.tsv \
    --output_dir output/microRNA
```

## Notes
- Only 3'UTR regions are analyzed (microRNA binding sites are primarily in 3'UTRs)
- The length matching ensures fair comparison between SHAP and control regions
- Mann-Whitney U test is more appropriate for comparing count distributions than Fisher exact test
