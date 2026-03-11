"""
CPU-only comparison for Qwen/Qwen3-1.7B:
1) qwen3-1.7B without vLLM
2) qwen3-1.7B with vLLM
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone

import torch
import psutil
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen3-1.7B"


def build_prompts(batch_size: int) -> list[str]:
    return [f"Question {i}: Explain CPU inference trade-offs in one paragraph." for i in range(batch_size)]


def _read_lscpu_key_values() -> dict:
    try:
        proc = subprocess.run(["lscpu"], capture_output=True, text=True, timeout=10, check=True)
    except Exception:
        return {}
    data = {}
    for line in proc.stdout.splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        data[k.strip()] = v.strip()
    return data


def collect_system_info() -> dict:
    lscpu = _read_lscpu_key_values()
    vm = psutil.virtual_memory()
    return {
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": sys.version.split()[0],
        },
        "cpu": {
            "vendor": lscpu.get("Vendor ID"),
            "model_name": lscpu.get("Model name"),
            "architecture": lscpu.get("Architecture"),
            "sockets": lscpu.get("Socket(s)"),
            "cores_per_socket": lscpu.get("Core(s) per socket"),
            "threads_per_core": lscpu.get("Thread(s) per core"),
            "logical_cpus": lscpu.get("CPU(s)", str(psutil.cpu_count(logical=True))),
            "physical_cores": str(psutil.cpu_count(logical=False)),
            "max_mhz": lscpu.get("CPU max MHz"),
            "min_mhz": lscpu.get("CPU min MHz"),
        },
        "cache": {
            "l1d": lscpu.get("L1d cache"),
            "l1i": lscpu.get("L1i cache"),
            "l2": lscpu.get("L2 cache"),
            "l3": lscpu.get("L3 cache"),
        },
        "memory": {
            "total_gb": round(vm.total / (1024**3), 2),
            "available_gb": round(vm.available / (1024**3), 2),
        },
    }


def run_transformers_cpu(model: str, batch_size: int, max_new_tokens: int) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(model)
    model_obj = AutoModelForCausalLM.from_pretrained(model, dtype=torch.bfloat16)
    model_obj = model_obj.to("cpu")
    model_obj.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompts = build_prompts(batch_size)
    t0 = time.time()
    # Avoid OOM for large total batch by using deterministic micro-batching.
    micro_batch_size = 1 if batch_size == 1 else min(25, batch_size)
    for i in range(0, batch_size, micro_batch_size):
        chunk = prompts[i : i + micro_batch_size]
        inputs = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=128).to("cpu")
        with torch.no_grad():
            _ = model_obj.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
    elapsed = time.time() - t0

    return {
        "engine": "qwen3-1.7B without vLLM",
        "model": model,
        "batch_size": batch_size,
        "max_new_tokens": max_new_tokens,
        "micro_batch_size": micro_batch_size,
        "time_s": round(elapsed, 2),
        "per_prompt_ms": round((elapsed / batch_size) * 1000, 2),
        "throughput_prompts_per_s": round(batch_size / elapsed, 2),
    }


def resolve_vllm_python(script_dir: str) -> str:
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    candidates = [
        os.environ.get("VLLM_CPU_PYTHON", ""),
        os.path.join(project_root, "vllm_cpu_venv", "bin", "python3"),
        os.path.join(project_root, "vllm_test", "venv", "bin", "python3"),
        sys.executable,
    ]
    for path in candidates:
        if path and os.path.exists(path):
            return path
    raise RuntimeError("No usable Python interpreter found for vLLM CPU test")


def run_vllm_cpu(
    model: str, batch_size: int, max_new_tokens: int, max_num_seqs: int, script_dir: str
) -> dict:
    python_bin = resolve_vllm_python(script_dir)
    worker_script = os.path.join(script_dir, "test_qwen3_vllm_cpu.py")

    proc = subprocess.run(
        [
            python_bin,
            worker_script,
            "--model",
            model,
            "--batch-size",
            str(batch_size),
            "--max-new-tokens",
            str(max_new_tokens),
            "--max-num-seqs",
            str(max_num_seqs),
        ],
        cwd=script_dir,
        text=True,
        capture_output=True,
        timeout=3600,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"vLLM subprocess failed with exit code {proc.returncode}: {proc.stderr[-1000:]}")

    for line in proc.stdout.splitlines():
        if line.startswith("__RESULT_JSON__:"):
            return json.loads(line.split(":", 1)[1])
    raise RuntimeError("vLLM subprocess did not return JSON result")


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3-1.7B CPU-only comparison")
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--single-batch-size", type=int, default=1)
    parser.add_argument("--large-batch-size", type=int, default=500)
    parser.add_argument("--max-new-tokens", type=int, default=30)
    parser.add_argument("--vllm-max-num-seqs", type=int, default=32)
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    started_at = datetime.now(timezone.utc).isoformat()

    print("=" * 80)
    print("Qwen3-1.7B CPU-only comparison")
    print("=" * 80)
    print(f"Model: {args.model}")
    system_info = collect_system_info()

    print(f"Single batch size: {args.single_batch_size}")
    print(f"Large batch size: {args.large_batch_size}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"vLLM max_num_seqs: {args.vllm_max_num_seqs}")
    print(
        f"CPU: {system_info['cpu']['model_name']} | "
        f"{system_info['cpu']['physical_cores']} physical / {system_info['cpu']['logical_cpus']} logical"
    )
    print(
        f"Cache: L1d {system_info['cache']['l1d']}, L1i {system_info['cache']['l1i']}, "
        f"L2 {system_info['cache']['l2']}, L3 {system_info['cache']['l3']}"
    )
    print(
        f"Memory: total {system_info['memory']['total_gb']} GB, "
        f"available {system_info['memory']['available_gb']} GB"
    )
    print()

    transformers_single = run_transformers_cpu(args.model, args.single_batch_size, args.max_new_tokens)
    vllm_single = run_vllm_cpu(
        args.model, args.single_batch_size, args.max_new_tokens, args.vllm_max_num_seqs, script_dir
    )
    print(
        f"[Single Query] without vLLM: {transformers_single['time_s']}s | "
        f"with vLLM: {vllm_single['time_s']}s (init {vllm_single['init_time_s']}s)"
    )

    transformers_batch500 = run_transformers_cpu(args.model, args.large_batch_size, args.max_new_tokens)
    vllm_batch500 = run_vllm_cpu(
        args.model, args.large_batch_size, args.max_new_tokens, args.vllm_max_num_seqs, script_dir
    )
    print(
        f"[Batch {args.large_batch_size}] without vLLM: {transformers_batch500['time_s']}s | "
        f"with vLLM: {vllm_batch500['time_s']}s (init {vllm_batch500['init_time_s']}s)"
    )

    summary = {
        "meta": {
            "started_at_utc": started_at,
            "model": args.model,
            "single_batch_size": args.single_batch_size,
            "large_batch_size": args.large_batch_size,
            "max_new_tokens": args.max_new_tokens,
            "experiment_names": [
                "qwen3-1.7B without vLLM",
                "qwen3-1.7B with vLLM",
            ],
            "system_info": system_info,
        },
        "single_query": {
            "qwen3-1.7B without vLLM": transformers_single,
            "qwen3-1.7B with vLLM": vllm_single,
        },
        "batch_500_query": {
            "qwen3-1.7B without vLLM": transformers_batch500,
            "qwen3-1.7B with vLLM": vllm_batch500,
        },
        "ratios": {
            "single_query": {
                "time_ratio_with_over_without_vllm": round(
                    vllm_single["time_s"] / transformers_single["time_s"], 3
                ),
                "throughput_ratio_with_over_without_vllm": round(
                    vllm_single["throughput_prompts_per_s"] / transformers_single["throughput_prompts_per_s"], 3
                ),
            },
            "batch_500_query": {
                "time_ratio_with_over_without_vllm": round(
                    vllm_batch500["time_s"] / transformers_batch500["time_s"], 3
                ),
                "throughput_ratio_with_over_without_vllm": round(
                    vllm_batch500["throughput_prompts_per_s"]
                    / transformers_batch500["throughput_prompts_per_s"],
                    3,
                ),
            },
        },
    }

    out_path = os.path.join(script_dir, "qwen3_1_7b_cpu_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("-" * 80)
    print(f"Saved results: {out_path}")
    print(json.dumps(summary["ratios"], indent=2))


if __name__ == "__main__":
    main()
