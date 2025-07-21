#!/bin/bash

# Script to compile LSGKM-SVR with CUDA support if available
# Usage: compile_lsgkm_svr.sh <data_dir_with_train_fasta>

set -e  # Exit on any error

# Check if data directory argument is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <data_dir_with_train_fasta>"
    echo "Example: $0 output/data/preprocessed"
    exit 1
fi

DATA_DIR="$1"
TRAIN_FASTA="${DATA_DIR}/train.fasta"

# Check if train.fasta exists
if [ ! -f "$TRAIN_FASTA" ]; then
    echo "Error: train.fasta not found at $TRAIN_FASTA"
    exit 1
fi

echo "Compiling LSGKM-SVR..."

# Detect CUDA installation
CUDA_PATH=$(bash scripts/detect_cuda.sh)

if [ ! -z "$CUDA_PATH" ]; then 
    echo "CUDA found at $CUDA_PATH, building with GPU support" 
    export PATH=$CUDA_PATH/bin:$PATH 
    export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH
else 
    echo "CUDA not found, building CPU-only version"
fi 

# Get maximum sequence length from train.fasta
echo "Determining maximum sequence length from $TRAIN_FASTA..."
MAX_LEN=$(wc -L "$TRAIN_FASTA" | cut -d' ' -f1)
echo "Maximum sequence length: $MAX_LEN"

# Update MAX_SEQ_LENGTH in the header file
echo "Updating MAX_SEQ_LENGTH in lsgkm-svr/src/libsvm_gkm.h..."
sed -i "s/#define MAX_SEQ_LENGTH 2048/#define MAX_SEQ_LENGTH $MAX_LEN/" lsgkm-svr/src/libsvm_gkm.h

# Compile the software
echo "Compiling LSGKM-SVR..."
cd lsgkm-svr/src

if [ ! -z "$CUDA_PATH" ]; then
    echo "Building with CUDA support..."
    make CUDA_PATH=$CUDA_PATH CUDA_ARCH=sm_70
else
    echo "Building CPU-only version..."
    make CUDA_ENABLED=0
fi

cd ../..

echo "LSGKM-SVR compilation completed successfully!"

# Verify the executable was created
if [ -f "lsgkm-svr/src/gkmtrain" ]; then
    echo "gkmtrain executable created successfully"
else
    echo "Error: gkmtrain executable not found after compilation"
    exit 1
fi
