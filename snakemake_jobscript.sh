#!/bin/bash --login

#SBATCH -p gpuL #multicore #
#SBATCH -G 2
#SBATCH -n 4
#SBATCH -t 6:00:00

#####################################################
### run command
#####################################################
source ~/.bashrc
conda activate inseq
# ~/.local/share/mamba/bin/dvc repro fineTuneModel # visualizeImportance #runViennaRNA #
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
~/.local/share/mamba/bin/dvc repro -s -f fineTuneModel_ablations@5 # visualizeImportance #runViennaRNA #
~/.local/share/mamba/bin/git add -u
~/.local/share/mamba/bin/git commit -m "(ablations) codons_only - no lengths, GC, mfe"