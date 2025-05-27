#!/bin/bash --login

#SBATCH -p gpuL
#SBATCH -G 2
#SBATCH -n 4
#SBATCH -t 05:00:00

#####################################################
### run command
#####################################################
#singularity exec -B $(readlink motif_seqs.fasta):/tmp/motif_seqs.fasta -B $(readlink control_seqs.fasta):/tmp/control_seqs.fasta docker://memesuite/memesuite ame --oc ame_output --control /tmp/control_seqs.fasta --evalue-report-threshold 100 --method fisher --rna /tmp/motif_seqs.fasta /opt/meme/share/meme-5.5.7/db/motif_databases/RNA/Ray2013_rbp_Drosophila_melanogaster.meme
#  --hit-lo-fraction 0.1 --scoring max 
conda activate inseq
#~/.local/share/mamba/bin/dvc repro runAME_highLowDecay #dvc exp run
#~/.local/share/mamba/bin/dvc repro randomizeSeqsAndExtraFeatures #dvc exp run
~/.local/share/mamba/bin/dvc repro fineTuneModel
~/.local/share/mamba/bin/git add -u
~/.local/share/mamba/bin/git add dvclive/
~/.local/share/mamba/bin/git add data/
~/.local/share/mamba/bin/git add output/data/
#~/.local/share/mamba/bin/git commit -m "1 count in each replicate (mRNA and ribo), double log transform"