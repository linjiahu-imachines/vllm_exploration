# CPU-Only Batch-500 Fair Comparison Report

**Date**: 2026-03-10  
**Machine**: Azure VM (`Intel Xeon Platinum 8573C`, 4 physical cores / 8 logical threads)  
**Model**: `facebook/opt-125m`  
**Workload**: 500 prompts, `max_new_tokens=30`  
**Goal**: Fairly compare CPU inference performance **with** and **without** vLLM under the same thread budget.

---

## 1) Fairness-Controlled Test Setup

To make the comparison fair, both paths were constrained to the same CPU resources:

- `CPU_THREADS=4`
- Process CPU affinity pinned to physical-core sibling set:
  - `taskset -c 1,3,5,7`
- Same model and batch size for both runs

Execution command:

```bash
export CPU_THREADS=4
taskset -c 1,3,5,7 python test_batch500_complete.py
```

---

## 2) Environment Summary

| Item | Value |
|---|---|
| CPU | Intel Xeon Platinum 8573C |
| Logical CPUs | 8 |
| Physical Cores | 4 |
| RAM | 31.3 GB |
| GPU | None (CPU-only node) |
| CUDA | Not available |
| vLLM CPU interpreter | `/home/azureuser/projects/vllm_exploration/vllm_cpu_venv/bin/python3` |

---

## 3) Fair Comparison Results (Batch=500)

| Scenario | Total Time | Per Prompt | Throughput | Active Threads (>5%) | Avg CPU Util |
|---|---:|---:|---:|---:|---:|
| Transformers on CPU (without vLLM) | **29.00s** | **58.00ms** | **17.24 prompts/sec** | 4/8 | 32.2% |
| vLLM on CPU | 63.57s | 127.14ms | 7.87 prompts/sec | 4/8 | 39.8% |

---

## 4) Performance Delta

- **Time ratio**: `63.57 / 29.00 = 2.19x`
  - vLLM CPU is **2.19x slower**
- **Throughput ratio**: `17.24 / 7.87 = 2.19x`
  - vLLM CPU has **2.19x lower throughput**

---

## 5) Key Observations

1. **Thread fairness is achieved**
   - Both runs show `4/8` active threads after applying affinity + thread cap.

2. **Conclusion remains unchanged under fair constraints**
   - Even after controlling threads, vLLM CPU is still significantly slower for this workload/model.

3. **vLLM CPU is functioning correctly**
   - Platform detected as CPU.
   - Stable init and inference.
   - Auto OMP binding matches pinned core set.

4. **This result is workload-specific**
   - This is for `opt-125m`, batch-500, offline generation.
   - Results may differ for larger models or online serving-style mixed traffic.

---

## 6) Final Conclusion

For this CPU-only machine and this batch-500 workload:

- **Best performance**: Transformers CPU (without vLLM)
- **vLLM CPU**: valid and stable, but about **2.2x slower**

Recommended default for this node and workload: **Transformers CPU**.

