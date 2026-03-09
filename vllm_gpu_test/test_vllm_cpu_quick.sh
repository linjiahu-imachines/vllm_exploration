#!/bin/bash
# Quick test: vLLM CPU with V0 engine, 10 prompts
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_V1=0
export VLLM_CPU_KVCACHE_SPACE=4

PYTHON=/home/linhu/projects/vllm_exploration/vllm_test/venv/bin/python3

exec $PYTHON -u /home/linhu/projects/vllm_exploration/vllm_gpu_test/test_vllm_cpu_quick.py 2>&1
