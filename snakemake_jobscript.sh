#!/bin/bash --login

#SBATCH -p gpuL # multicore #
#SBATCH -G 2
#SBATCH -n 8 #--cpus-per-task=10  # Use this instead of -n for Ray Tune #SBATCH --ntasks=1          # Single task (Ray will handle parallelization)
#SBATCH -t 4-0 #:00:00

#####################################################
### run command
#####################################################
source ~/.bashrc
conda activate rayTune
 python scripts/manualTokenShuffle.py --params params.yaml --model_name_or_path output/ftModel_no_codons/best_spearmanr/ --data_dir output/data/decay --output_dir output/predict --scaler output/ftModel_no_codons/best_spearmanr/scaler.joblib --label tokenShuffle --extraFeatures output/data/ablations/no_codons.csv --mfe output/data/codons/vienna_features.csv
#~/.local/share/mamba/bin/dvc repro motifEnrichment
# ~/.local/share/mamba/bin/dvc repro predict
# ~/.local/share/mamba/bin/dvc repro motifEnrichment_highLowDecay
# ~/.local/share/mamba/bin/dvc repro codonOnlyClassificationMLP
# ~/.local/share/mamba/bin/dvc repro allExtraFeaturesClassificationMLP
# ~/.local/share/mamba/bin/dvc repro extraFeaturesAndMotifCountsClassificationMLP


# ~/.local/share/mamba/bin/git add -u
# ~/.local/share/mamba/bin/git commit -m "(ablations) default - all extra features included"
# 
# ~/.local/share/mamba/bin/dvc repro -s -f fineTuneModel_ablations@0 # visualizeImportance #runViennaRNA #
# ~/.local/share/mamba/bin/git add -u
# ~/.local/share/mamba/bin/git commit -m "(ablations) no_mfe"
# 
# ~/.local/share/mamba/bin/dvc repro -s -f fineTuneModel_ablations@1 # visualizeImportance #runViennaRNA #
# ~/.local/share/mamba/bin/git add -u
#30
#31
# ~/.local/share/mamba/bin/git commit -m "(ablations) no_lengths"
# 
# ~/.local/share/mamba/bin/dvc repro -s -f fineTuneModel_ablations@2 # visualizeImportance #runViennaRNA #
# ~/.local/share/mamba/bin/git add -u
# ~/.local/share/mamba/bin/git commit -m "(ablations) no_gc_content"
# 
# ~/.local/share/mamba/bin/dvc repro -s -f fineTuneModel_ablations@3 # visualizeImportance #runViennaRNA #
# ~/.local/share/mamba/bin/git add -u
# ~/.local/share/mamba/bin/git commit -m "(ablations) no_codons"
# 
# ~/.local/share/mamba/bin/dvc repro -s -f fineTuneModel_ablations@4 # visualizeImportance #runViennaRNA #
# ~/.local/share/mamba/bin/git add -u
# ~/.local/share/mamba/bin/git commit -m "(ablations) seq_only"
# 
# ~/.local/share/mamba/bin/dvc repro -s -f fineTuneModel_ablations@5 # visualizeImportance #runViennaRNA #
# ~/.local/share/mamba/bin/git add -u
# ~/.local/share/mamba/bin/git commit -m "(ablations) codons_only - no lengths, GC, mfe"