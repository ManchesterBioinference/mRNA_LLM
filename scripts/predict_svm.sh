#!/bin/bash

# Script to make predictions with trained SVM model using LSGKM-SVR
# Usage: predict_svm.sh <test_fasta> <model_file> <outprefix> <threads> <output_file>

set -e  # Exit on any error

# Check if required arguments are provided
if [ $# -ne 5 ]; then
    echo "Usage: $0 <test_fasta> <model_file> <outprefix> <threads> <output_file>"
    echo "Example: $0 data/test.fasta output/svm/model.model.txt output/svm/model 4 predictions.tsv"
    exit 1
fi

TEST_FASTA="$1"
MODEL_FILE="$2"
OUTPREFIX="$3"
THREADS="$4"
OUTPUT_FILE="$5"

# Check if required files exist
if [ ! -f "$TEST_FASTA" ]; then
    echo "Error: Test FASTA file not found at $TEST_FASTA"
    exit 1
fi

if [ ! -f "$MODEL_FILE" ]; then
    echo "Error: Model file not found at $MODEL_FILE"
    exit 1
fi

if [ ! -f "lsgkm-svr/src/gkmpredict" ]; then
    echo "Error: gkmpredict executable not found. Please run compileLSGKM_SVR stage first."
    exit 1
fi

echo "Making predictions with SVM model using LSGKM-SVR..."

# Create output directory if it doesn't exist
OUTPUT_DIR=$(dirname "${OUTPREFIX}/${OUTPUT_FILE}")
mkdir -p "$OUTPUT_DIR"

# Set CUDA runtime library path if available
CUDA_PATH=$(bash scripts/detect_cuda.sh)

if [ ! -z "$CUDA_PATH" ]; then
    export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH
    echo "Using CUDA runtime from: $CUDA_PATH"
else
    echo "Using CPU-only version"
fi

# Convert U to T in test FASTA (required by lsgkm-svr)
TEST_FASTA_CONVERTED="${OUTPREFIX}/test.fasta"
echo "Converting U to T in test FASTA..."
sed 's/U/T/g' "$TEST_FASTA" > "$TEST_FASTA_CONVERTED"

echo "Running gkmpredict with parameters:"
echo "  - Threads: ${THREADS}"
echo "  - Test FASTA: ${TEST_FASTA_CONVERTED}"
echo "  - Model: ${MODEL_FILE}"
echo "  - Output: ${OUTPREFIX}_${OUTPUT_FILE}"

lsgkm-svr/src/gkmpredict \
    -T ${THREADS} \
    ${TEST_FASTA_CONVERTED} \
    ${MODEL_FILE} \
    ${OUTPREFIX}_${OUTPUT_FILE}

# Verify the predictions were created
if [ -f "${OUTPREFIX}_${OUTPUT_FILE}" ]; then
    echo "SVM predictions completed successfully!"
    echo "Predictions saved to: ${OUTPREFIX}_${OUTPUT_FILE}"
else
    echo "Error: Predictions file not found after running gkmpredict"
    exit 1
fi
