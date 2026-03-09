"""
Comparison test suite for vLLM vs transformers on GPU
"""
import time
import torch
from test_with_vllm_gpu import test_vllm_gpu_single, test_vllm_gpu_batch, test_vllm_multi_gpu
from test_without_vllm_gpu import test_transformers_gpu_single, test_transformers_gpu_batch, test_transformers_multi_gpu


def run_full_gpu_comparison():
    """Run full GPU comparison suite"""
    print("\n" + "=" * 80)
    print("FULL GPU COMPARISON TEST SUITE")
    print("vLLM vs Transformers on GPU")
    print("=" * 80)
    
    print(f"\nGPU Information:")
    print(f"  CUDA Available: {torch.cuda.is_available()}")
    print(f"  Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    print()
    
    results = {}
    
    start_time = time.time()
    
    try:
        # Test 1: Single GPU, Single Inference
        print("\n" + "=" * 80)
        print("TEST 1: Single GPU - Single Inference (3 prompts)")
        print("=" * 80)
        
        print("\n[1/2] Testing Transformers...")
        trans_single = test_transformers_gpu_single()
        results['trans_single'] = trans_single
        
        print("\n[2/2] Testing vLLM...")
        vllm_single = test_vllm_gpu_single()
        results['vllm_single'] = vllm_single
        
        print("\n" + "-" * 80)
        print("COMPARISON - Single GPU Inference")
        print("-" * 80)
        print(f"Transformers: {trans_single:.4f}s")
        print(f"vLLM:         {vllm_single:.4f}s")
        
        if vllm_single < trans_single:
            speedup = trans_single / vllm_single
            improvement = ((trans_single - vllm_single) / trans_single) * 100
            print(f"\n✅ vLLM is {speedup:.2f}x faster ({improvement:.1f}% improvement)")
        else:
            slowdown = vllm_single / trans_single
            degradation = ((vllm_single - trans_single) / trans_single) * 100
            print(f"\n⚠️  vLLM is {slowdown:.2f}x slower ({degradation:.1f}% slower)")
        
        # Test 2: Single GPU, Batch Inference
        print("\n" + "=" * 80)
        print("TEST 2: Single GPU - Batch Inference (5 prompts)")
        print("=" * 80)
        
        print("\n[1/2] Testing Transformers batch...")
        trans_batch = test_transformers_gpu_batch()
        results['trans_batch'] = trans_batch
        
        print("\n[2/2] Testing vLLM batch...")
        vllm_batch = test_vllm_gpu_batch()
        results['vllm_batch'] = vllm_batch
        
        print("\n" + "-" * 80)
        print("COMPARISON - Single GPU Batch Inference")
        print("-" * 80)
        print(f"Transformers: {trans_batch:.4f}s")
        print(f"vLLM:         {vllm_batch:.4f}s")
        
        if vllm_batch < trans_batch:
            speedup = trans_batch / vllm_batch
            improvement = ((trans_batch - vllm_batch) / trans_batch) * 100
            print(f"\n✅ vLLM is {speedup:.2f}x faster ({improvement:.1f}% improvement)")
        else:
            slowdown = vllm_batch / trans_batch
            degradation = ((vllm_batch - trans_batch) / trans_batch) * 100
            print(f"\n⚠️  vLLM is {slowdown:.2f}x slower ({degradation:.1f}% slower)")
        
        # Test 3: Multi-GPU (if available)
        if torch.cuda.device_count() >= 2:
            print("\n" + "=" * 80)
            print("TEST 3: Multi-GPU Inference (3 prompts)")
            print("=" * 80)
            
            print("\n[1/2] Testing Transformers multi-GPU...")
            trans_multi = test_transformers_multi_gpu()
            results['trans_multi'] = trans_multi
            
            print("\n[2/2] Testing vLLM multi-GPU...")
            vllm_multi = test_vllm_multi_gpu()
            results['vllm_multi'] = vllm_multi
            
            if trans_multi and vllm_multi:
                print("\n" + "-" * 80)
                print("COMPARISON - Multi-GPU Inference")
                print("-" * 80)
                print(f"Transformers: {trans_multi:.4f}s")
                print(f"vLLM:         {vllm_multi:.4f}s")
                
                if vllm_multi < trans_multi:
                    speedup = trans_multi / vllm_multi
                    improvement = ((trans_multi - vllm_multi) / trans_multi) * 100
                    print(f"\n✅ vLLM is {speedup:.2f}x faster ({improvement:.1f}% improvement)")
                else:
                    slowdown = vllm_multi / trans_multi
                    degradation = ((vllm_multi - trans_multi) / trans_multi) * 100
                    print(f"\n⚠️  vLLM is {slowdown:.2f}x slower ({degradation:.1f}% slower)")
        
        total_time = time.time() - start_time
        
        # Final Summary
        print("\n" + "=" * 80)
        print("FINAL SUMMARY - GPU COMPARISON")
        print("=" * 80)
        print("\nSingle GPU Performance:")
        print(f"  Single Inference: vLLM {vllm_single:.4f}s vs Transformers {trans_single:.4f}s")
        print(f"  Batch Inference:  vLLM {vllm_batch:.4f}s vs Transformers {trans_batch:.4f}s")
        
        if 'vllm_multi' in results and 'trans_multi' in results and results['vllm_multi'] and results['trans_multi']:
            print(f"\nMulti-GPU Performance:")
            print(f"  Inference: vLLM {results['vllm_multi']:.4f}s vs Transformers {results['trans_multi']:.4f}s")
        
        print(f"\nTotal test duration: {total_time:.2f}s")
        
        print("\n" + "=" * 80)
        print("Key Findings:")
        print("=" * 80)
        
        if vllm_single < trans_single:
            print("✅ vLLM shows performance advantage on GPU")
            print("✅ vLLM's optimizations work well on GPU architecture")
            print("✅ Recommended for GPU-based deployments")
        else:
            print("⚠️  In this test, vLLM did not show expected GPU advantage")
            print("⚠️  This may be due to model size (opt-125m is very small)")
            print("💡 vLLM typically shows larger gains with bigger models (>1B params)")
        
        return results
        
    except Exception as e:
        print(f"\nError during GPU comparison: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    run_full_gpu_comparison()
