# Qwen3 1.7B CPU-Only Evaluation

This folder contains a dedicated CPU-only benchmark for:

- `Qwen/Qwen3-1.7B`

It compares two named experiments:

1. `qwen3-1.7B without vLLM`
2. `qwen3-1.7B with vLLM`

Each run executes:

- Single-query scenario (`batch_size=1`)
- Batch-query scenario (`batch_size=500`)

## Quick Run

From project root:

```bash
cd /home/azureuser/projects/vllm_exploration
./vllm_cpu_qwen3_1_7b_test/run_qwen3_cpu_test.sh
```

## Common Options

```bash
cd /home/azureuser/projects/vllm_exploration/vllm_cpu_qwen3_1_7b_test
"/home/azureuser/projects/vllm_exploration/vllm_cpu_venv/bin/python3" test_qwen3_cpu_compare.py \
  --single-batch-size 1 \
  --large-batch-size 500 \
  --max-new-tokens 30 \
  --vllm-max-num-seqs 32
```

## Output

Results are written to:

- `vllm_cpu_qwen3_1_7b_test/qwen3_1_7b_cpu_results.json`

## Notes

- Use file-based execution for vLLM CPU tests (not heredoc stdin execution).
- `VLLM_CPU_SGL_KERNEL=1` is optional tuning, not required.
- This test is CPU-only by design.
- For the 500-query path, the non-vLLM side uses deterministic micro-batching to avoid OOM while preserving a 500-query total workload.
