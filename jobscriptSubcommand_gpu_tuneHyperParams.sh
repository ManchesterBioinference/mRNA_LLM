#!/bin/bash --login

#SBATCH -p gpuL
#SBATCH -G 2
#SBATCH -n 4
#SBATCH -t 06:00:00

source ~/.bashrc
conda activate inseq
~/.local/share/mamba/bin/dvc repro fineTuneModel
#~/.local/share/mamba/bin/git add -u
#~/.local/share/mamba/bin/git add dvclive/
#~/.local/share/mamba/bin/git add data/
#~/.local/share/mamba/bin/git add output/data/
#~/.local/share/mamba/bin/git commit -m "corrected attention mask, lr${1} epochs${2} wUpPerc${3}"
#./find_max_spearman.py

touch jobscriptStatus/lr${1}_n${2}_warm${3}
