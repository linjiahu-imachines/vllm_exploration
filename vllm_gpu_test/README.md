# vLLM GPU Testing Suite

This project compares LLM inference performance with and without vLLM on GPU deployment.

## Hardware Requirements

- NVIDIA GPU(s) with CUDA support
- CUDA 12.0 or higher
- Sufficient VRAM (at least 2GB per GPU for opt-125m)

## Setup

### 1. Virtual environment is already created and configured

The environment is located at: `/home/linhu/projects/vllm_gpu_test/venv/`

### 2. Activate and verify

```bash
cd /home/linhu/projects/vllm_gpu_test
source venv/bin/activate
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Test Files

- **test_with_vllm_gpu.py** - Tests using vLLM on GPU
  - Single GPU inference
  - Single GPU batch inference
  - Multi-GPU tensor parallelism (if 2+ GPUs available)

- **test_without_vllm_gpu.py** - Tests using transformers directly on GPU
  - Single GPU inference
  - Single GPU batch inference
  - Multi-GPU DataParallel (if 2+ GPUs available)

- **test_gpu_comparison.py** - Full comparison suite

## Running Tests

### Run individual tests

```bash
source venv/bin/activate

# Test WITH vLLM on GPU
python test_with_vllm_gpu.py

# Test WITHOUT vLLM (using transformers) on GPU
python test_without_vllm_gpu.py

# Run full comparison
python test_gpu_comparison.py
```

## What's Being Tested

1. **Single GPU Inference**: Processing 3 prompts sequentially on one GPU
2. **Single GPU Batch**: Processing 5 prompts in a batch on one GPU
3. **Multi-GPU**: Leveraging multiple GPUs (if available)
   - vLLM uses tensor parallelism
   - Transformers uses DataParallel

## Model Used

- **facebook/opt-125m**: A small 125M parameter model
- Larger models typically show bigger vLLM performance gains

## Expected Results

On GPU, vLLM should demonstrate:
- Faster inference through optimized kernels
- Better batching with continuous batching
- Improved multi-GPU scaling with tensor parallelism
- PagedAttention for efficient memory usage

**Note**: For very small models like opt-125m, the advantage may be modest. Larger models (>1B params) typically show 2-10x speedup with vLLM.

## GPU Configuration

### Detected GPUs
Check your system with:
```bash
nvidia-smi
```

### CUDA Setup
- CUDA Version: 12.4
- PyTorch: Built with CUDA 12.4 support
- vLLM: GPU-accelerated version

## Comparison with CPU Results

This GPU testing complements the CPU testing in `/home/linhu/projects/vllm_test/`:
- CPU results showed transformers 8-11x faster than vLLM
- GPU results should show vLLM faster than transformers
- Demonstrates hardware-dependent performance characteristics

## Troubleshooting

### CUDA Out of Memory
- Reduce batch size
- Use smaller model
- Reduce `gpu_memory_utilization` parameter

### Multi-GPU Issues
- Check all GPUs are visible: `nvidia-smi`
- Verify CUDA_VISIBLE_DEVICES if needed
- Ensure sufficient VRAM on all GPUs

## Notes

- First run will download the model (~500MB for opt-125m)
- Model will be cached for subsequent runs
- vLLM on GPU is optimized for throughput and latency
- Results will be saved for comparison report generation
