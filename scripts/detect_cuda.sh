#!/bin/bash

# Script to detect CUDA installation and set up environment variables
# Returns the CUDA path if found, empty string if not found

# Function to check if CUDA path is valid
check_cuda_path() {
    local cuda_path="$1"
    if [ -d "$cuda_path" ] && [ -f "$cuda_path/bin/nvcc" ] && [ -d "$cuda_path/lib64" ]; then
        return 0
    fi
    return 1
}

# List of common CUDA installation paths to check
CUDA_PATHS=(
    "/opt/apps/pkg/compilers/cuda/12.6.2"
    "/opt/apps/libs/nvidia-cuda/toolkit/12.6.2"
    "/opt/apps/libs/nvidia-cuda/toolkit/12.4.1"
    "/opt/apps/libs/nvidia-cuda/toolkit/12.2.2"
    "/opt/apps/libs/nvidia-cuda/toolkit/12.0.1"
    "/usr/local/cuda"
    "/opt/cuda"
    "/usr/lib/cuda"
)

# Check if CUDA_PATH is already set and valid
if [ ! -z "$CUDA_PATH" ] && check_cuda_path "$CUDA_PATH"; then
    echo "$CUDA_PATH"
    exit 0
fi

# Check each common path
for cuda_path in "${CUDA_PATHS[@]}"; do
    if check_cuda_path "$cuda_path"; then
        echo "$cuda_path"
        exit 0
    fi
done

# Try to find nvcc in PATH
NVCC_PATH=$(which nvcc 2>/dev/null)
if [ ! -z "$NVCC_PATH" ]; then
    # Extract CUDA path from nvcc location
    CUDA_PATH=$(dirname $(dirname "$NVCC_PATH"))
    if check_cuda_path "$CUDA_PATH"; then
        echo "$CUDA_PATH"
        exit 0
    fi
fi

# CUDA not found
echo ""
exit 1
