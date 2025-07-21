#!/bin/bash

# Script to train SVM model with LSGKM-SVR
# Usage: train_svm.sh <outprefix> <memory> <threads>

set -e  # Exit on any error

# Check if required arguments are provided
if [ $# -ne 3 ]; then
    echo "Usage: $0 <outprefix> <memory> <threads>"
    echo "Example: $0 output/svm/model 8 4"
    exit 1
fi

OUTPREFIX="$1"
MEMORY="$2"
THREADS="$3"

# Required files
TRAIN_DEV_FASTA="${OUTPREFIX}/train_dev.fasta"
SEQLABELS="${OUTPREFIX}.seqlabels.txt"
COVARIATES="${OUTPREFIX}/covariates.tsv"

# Check if required files exist
if [ ! -f "$TRAIN_DEV_FASTA" ]; then
    echo "Error: train_dev.fasta not found at $TRAIN_DEV_FASTA"
    exit 1
fi

if [ ! -f "$SEQLABELS" ]; then
    echo "Error: seqlabels.txt not found at $SEQLABELS"
    exit 1
fi

if [ ! -f "$COVARIATES" ]; then
    echo "Error: covariates.tsv not found at $COVARIATES"
    exit 1
fi

if [ ! -f "lsgkm-svr/src/gkmtrain" ]; then
    echo "Error: gkmtrain executable not found. Please run compileLSGKM_SVR stage first."
    exit 1
fi

echo "Training SVM model with LSGKM-SVR..."

# Set CUDA runtime library path if available
CUDA_PATH=$(bash scripts/detect_cuda.sh)

if [ ! -z "$CUDA_PATH" ]; then
    export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH
    echo "Using CUDA runtime from: $CUDA_PATH"
else
    echo "Using CPU-only version"
fi

# Note: all the ',' will be converted to 'A' by lsgkm-svr
echo "Running gkmtrain with parameters:"
echo "  - Memory: ${MEMORY}GB"
echo "  - Threads: ${THREADS}"
echo "  - Input FASTA: ${TRAIN_DEV_FASTA}"
echo "  - Labels: ${SEQLABELS}"
echo "  - Covariates: ${COVARIATES}"
echo "  - Output prefix: ${OUTPREFIX}"

lsgkm-svr/src/gkmtrain \
    -y 3 \
    -m ${MEMORY} \
    -T ${THREADS} \
    -N \
    -t 7 \
    ${TRAIN_DEV_FASTA} \
    ${SEQLABELS} \
    ${OUTPREFIX} \
    ${COVARIATES}

# Verify the model was created
if [ -f "${OUTPREFIX}.model.txt" ]; then
    echo "SVM model training completed successfully!"
    echo "Model saved to: ${OUTPREFIX}.model.txt"
else
    echo "Error: Model file not found after training"
    exit 1
fi
