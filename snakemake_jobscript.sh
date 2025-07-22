#!/bin/bash --login

#SBATCH -p gpuL #multicore #
#SBATCH -G 1
#SBATCH -n 4
#SBATCH -t 1:00:00

#####################################################
### run command
#####################################################
source ~/.bashrc
conda activate inseq
~/.local/share/mamba/bin/dvc repro predict@0 # visualizeImportance #runViennaRNA #
~/.local/share/mamba/bin/git add -u
~/.local/share/mamba/bin/git commit -m "no_mfe"

~/.local/share/mamba/bin/dvc repro predict@1 # visualizeImportance #runViennaRNA #
~/.local/share/mamba/bin/git add -u
~/.local/share/mamba/bin/git commit -m "no_lengths"

~/.local/share/mamba/bin/dvc repro predict@2 # visualizeImportance #runViennaRNA #
~/.local/share/mamba/bin/git add -u
~/.local/share/mamba/bin/git commit -m "no_gc_content"

~/.local/share/mamba/bin/dvc repro predict@3 # visualizeImportance #runViennaRNA #
~/.local/share/mamba/bin/git add -u
~/.local/share/mamba/bin/git commit -m "no_codons"

~/.local/share/mamba/bin/dvc repro predict@4 # visualizeImportance #runViennaRNA #
~/.local/share/mamba/bin/git add -u
~/.local/share/mamba/bin/git commit -m "seq_only"

~/.local/share/mamba/bin/dvc repro predict@5 # visualizeImportance #runViennaRNA #
~/.local/share/mamba/bin/git add -u
~/.local/share/mamba/bin/git commit -m "default - all extra features"