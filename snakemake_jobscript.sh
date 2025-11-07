#!/bin/bash --login

#SBATCH -p gpuL # multicore #
#SBATCH -G 1
#SBATCH -n 1 #--cpus-per-task=10  # Use this instead of -n for Ray Tune #SBATCH --ntasks=1          # Single task (Ray will handle parallelization)
#SBATCH -t 1:00

#####################################################
### run command
#####################################################
echo "Running on host $(hostname)"
#source ~/.bashrc
#conda activate raytune
#python scripts/manualTokenShuffle.py --params params.yaml --model_name_or_path output/ftModel/best_spearmanr/ --data_dir output/data/decay --output_dir output/predict --scaler output/ftModel/best_spearmanr/scaler.joblib --label tokenShuffle --extraFeatures output/data/codons/extraFeatures.csv --mfe output/data/codons/vienna_features.csv
#~/miniconda3/bin/dvc repro motifEnrichment
#~/.local/share/mamba/bin/dvc repro codonOnlyClassificationMLP
#~/.local/share/mamba/bin/dvc repro allExtraFeaturesClassificationMLP
#~/.local/share/mamba/bin/dvc repro extraFeaturesAndMotifCountsClassificationMLP


# ~/.local/share/mamba/bin/git add -u
# ~/.local/share/mamba/bin/git commit -m "(ablations) default - all extra features included"
# 
# ~/.local/share/mamba/bin/dvc repro -s -f fineTuneModel_ablations@0 # visualizeImportance #runViennaRNA #
# ~/.local/share/mamba/bin/git add -u
# ~/.local/share/mamba/bin/git commit -m "(ablations) no_mfe"
# 
# ~/.local/share/mamba/bin/dvc repro -s -f fineTuneModel_ablations@1 # visualizeImportance #runViennaRNA #
# ~/.local/share/mamba/bin/git add -u
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