#!/bin/bash --login

#SBATCH -p gpuL #multicore
#SBATCH -G 2
#SBATCH -n 4
#SBATCH -t 03:00:00

#####################################################
### run command
#####################################################
source ~/.bashrc
conda activate inseq
~/.local/share/mamba/bin/dvc repro fineTuneModel
#~/.local/share/mamba/bin/dvc repro randomizeSeqsAndExtraFeatures #dvc exp run
#~/.local/share/mamba/bin/dvc exp run --run-all --jobs 1
#~/.local/share/mamba/bin/git add -u
#~/.local/share/mamba/bin/git add dvclive/
#~/.local/share/mamba/bin/git add data/
#~/.local/share/mamba/bin/git add output/data/
#~/.local/share/mamba/bin/git commit -m "1 count in each replicate (mRNA and ribo), double log transform"