#!/bin/bash

# Script to run vLLM GPU tests

echo "=================================="
echo "vLLM GPU Testing Suite"
echo "=================================="
echo ""

# Get script directory to use relative paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PYTHON="${SCRIPT_DIR}/venv/bin/python3"

# Verify Python exists
if [ ! -f "$PYTHON" ]; then
    echo "Error: Virtual environment not found at ${SCRIPT_DIR}/venv"
    echo "Please ensure the virtual environment is properly set up."
    exit 1
fi

# Check GPU availability
echo "Checking GPU availability..."
$PYTHON -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU Count: {torch.cuda.device_count()}')"
echo ""

# Check which test to run
if [ "$1" == "with" ]; then
    echo "Running tests WITH vLLM on GPU..."
    $PYTHON test_with_vllm_gpu.py
elif [ "$1" == "without" ]; then
    echo "Running tests WITHOUT vLLM (transformers only) on GPU..."
    $PYTHON test_without_vllm_gpu.py
elif [ "$1" == "compare" ]; then
    echo "Running full GPU comparison tests..."
    $PYTHON test_gpu_comparison.py
else
    echo "Usage: ./run_gpu_tests.sh [with|without|compare]"
    echo ""
    echo "Options:"
    echo "  with     - Run tests WITH vLLM on GPU"
    echo "  without  - Run tests WITHOUT vLLM (transformers only) on GPU"
    echo "  compare  - Run full GPU comparison"
    echo ""
    echo "Running full comparison by default..."
    $PYTHON test_gpu_comparison.py
fi
