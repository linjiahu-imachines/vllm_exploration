"""
Standalone vLLM CPU test for Qwen/Qwen3-1.7B.
This script is intended to be called from a file-based entrypoint.
"""

import argparse
import json
import os
import time

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def build_prompts(batch_size: int) -> list[str]:
    return [f"Question {i}: Explain CPU inference trade-offs in one paragraph." for i in range(batch_size)]


def main() -> None:
    parser = argparse.ArgumentParser(description="vLLM CPU test for Qwen3-1.7B")
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=30)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    args = parser.parse_args()

    from vllm import LLM, SamplingParams

    prompts = build_prompts(args.batch_size)

    init_t0 = time.time()
    llm = LLM(
        model=args.model,
        max_num_seqs=max(1, min(args.batch_size, args.max_num_seqs)),
        enforce_eager=True,
    )
    init_time = time.time() - init_t0

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.max_new_tokens,
    )

    t0 = time.time()
    _ = llm.generate(prompts, sampling_params)
    elapsed = time.time() - t0

    result = {
        "engine": "qwen3-1.7B with vLLM",
        "model": args.model,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "max_num_seqs": max(1, min(args.batch_size, args.max_num_seqs)),
        "init_time_s": round(init_time, 2),
        "time_s": round(elapsed, 2),
        "per_prompt_ms": round((elapsed / args.batch_size) * 1000, 2),
        "throughput_prompts_per_s": round(args.batch_size / elapsed, 2),
    }
    print(f"__RESULT_JSON__:{json.dumps(result)}", flush=True)


if __name__ == "__main__":
    main()
