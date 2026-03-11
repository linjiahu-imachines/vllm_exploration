#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "${SCRIPT_DIR}/.." && pwd )"

PYTHON_BIN="${VLLM_CPU_PYTHON:-}"
if [ -z "${PYTHON_BIN}" ]; then
  if [ -x "${PROJECT_ROOT}/vllm_cpu_venv/bin/python3" ]; then
    PYTHON_BIN="${PROJECT_ROOT}/vllm_cpu_venv/bin/python3"
  elif [ -x "${PROJECT_ROOT}/vllm_test/venv/bin/python3" ]; then
    PYTHON_BIN="${PROJECT_ROOT}/vllm_test/venv/bin/python3"
  else
    echo "No CPU test interpreter found."
    echo "Set VLLM_CPU_PYTHON or create one of:"
    echo "  ${PROJECT_ROOT}/vllm_cpu_venv/bin/python3"
    echo "  ${PROJECT_ROOT}/vllm_test/venv/bin/python3"
    exit 1
  fi
fi

echo "Using Python: ${PYTHON_BIN}"
echo "Running Qwen3-1.7B CPU-only comparison..."

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/test_qwen3_cpu_compare.py" \
  --single-batch-size 1 \
  --large-batch-size 500 \
  --max-new-tokens 30 \
  --vllm-max-num-seqs 32
