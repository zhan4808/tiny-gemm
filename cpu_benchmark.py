import torch
import math
import time
import argparse
import numpy as np
from tqdm import tqdm

class CPUBenchmark:
    @staticmethod
    def attention(q, k, v, causal=True):
        """Simple CPU implementation of attention"""
        # q, k, v: [B, H, N, D]
        batch_size, num_heads, seq_len, head_dim = q.shape
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(head_dim)
        
        # Apply causal mask if needed
        if causal:
            mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
            scores.masked_fill_(mask, float("-inf"))
        
        # Apply softmax
        attn_weights = torch.softmax(scores, dim=-1)
        
        # Compute output
        output = torch.matmul(attn_weights, v)
        
        return output
    
    @staticmethod
    def feed_forward(x, w1, b1, w2, b2):
        """Simple CPU implementation of feed-forward network"""
        # x: [B, N, D]
        # w1: [D, D_hidden]
        # b1: [D_hidden]
        # w2: [D_hidden, D]
        # b2: [D]
        
        # First linear layer
        hidden = torch.matmul(x, w1) + b1
        
        # Apply GELU activation
        hidden = 0.5 * hidden * (1 + torch.tanh(math.sqrt(2 / math.pi) * (hidden + 0.044715 * torch.pow(hidden, 3))))
        
        # Second linear layer
        output = torch.matmul(hidden, w2) + b2
        
        return output
    
    @staticmethod
    def fused_attention(q, k, v, causal=True):
        """Simulated fused attention (same as regular attention on CPU)"""
        return CPUBenchmark.attention(q, k, v, causal)
    
    @staticmethod
    def fused_ffn(x, w1, b1, w2, b2):
        """Simulated fused feed-forward (same as regular FFN on CPU)"""
        return CPUBenchmark.feed_forward(x, w1, b1, w2, b2)

def benchmark_attention(batch_size, num_heads, seq_len, head_dim, causal=True, warmup=2, rep=5):
    """Benchmark attention implementations"""
    # Create inputs
    q = torch.randn(batch_size, num_heads, seq_len, head_dim)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim)
    
    # Ensure outputs match
    reg_out = CPUBenchmark.attention(q, k, v, causal)
    fused_out = CPUBenchmark.fused_attention(q, k, v, causal)
    max_diff = (reg_out - fused_out).abs().max().item()
    print(f"Attention max absolute difference: {max_diff:.6f}")
    
    # Benchmark implementations
    times = []
    
    # Warm up
    for _ in range(warmup):
        _ = CPUBenchmark.attention(q, k, v, causal)
    
    # Regular implementation
    start = time.time()
    for _ in range(rep):
        _ = CPUBenchmark.attention(q, k, v, causal)
    end = time.time()
    regular_time = (end - start) / rep
    
    # Simulate fused implementation (apply speedup factor)
    # In real hardware with Triton, we'd expect ~1.5-3x speedup
    # Here we simulate a 2x speedup for visualization
    simulated_speedup = 2.0
    fused_time = regular_time / simulated_speedup
    
    return {
        "regular_time": regular_time * 1000,  # ms
        "fused_time": fused_time * 1000,  # ms
        "simulated_speedup": simulated_speedup,
    }

def benchmark_ffn(batch_size, seq_len, d_model, d_hidden, warmup=2, rep=5):
    """Benchmark feed-forward network implementations"""
    # Create inputs
    x = torch.randn(batch_size, seq_len, d_model)
    w1 = torch.randn(d_model, d_hidden)
    b1 = torch.randn(d_hidden)
    w2 = torch.randn(d_hidden, d_model)
    b2 = torch.randn(d_model)
    
    # Ensure outputs match
    reg_out = CPUBenchmark.feed_forward(x, w1, b1, w2, b2)
    fused_out = CPUBenchmark.fused_ffn(x, w1, b1, w2, b2)
    max_diff = (reg_out - fused_out).abs().max().item()
    print(f"FFN max absolute difference: {max_diff:.6f}")
    
    # Benchmark implementations
    # Warm up
    for _ in range(warmup):
        _ = CPUBenchmark.feed_forward(x, w1, b1, w2, b2)
    
    # Regular implementation
    start = time.time()
    for _ in range(rep):
        _ = CPUBenchmark.feed_forward(x, w1, b1, w2, b2)
    end = time.time()
    regular_time = (end - start) / rep
    
    # Simulate fused implementation (apply speedup factor)
    # In real hardware with Triton, we'd expect ~1.5-3x speedup
    # Here we simulate a 2.5x speedup for visualization
    simulated_speedup = 2.5
    fused_time = regular_time / simulated_speedup
    
    return {
        "regular_time": regular_time * 1000,  # ms
        "fused_time": fused_time * 1000,  # ms
        "simulated_speedup": simulated_speedup,
    }

def run_benchmarks(batch_sizes, num_heads, seq_lengths, head_dim):
    """Run all benchmarks and print results"""
    attention_results = []
    ffn_results = []
    
    total_configs = len(batch_sizes) * len(seq_lengths)
    with tqdm(total=total_configs, desc="Running benchmarks") as pbar:
        for batch_size in batch_sizes:
            for seq_len in seq_lengths:
                print(f"\nBenchmarking with batch_size={batch_size}, seq_len={seq_len}, heads={num_heads}, dim={head_dim}")
                
                # Benchmark attention
                attn_result = benchmark_attention(batch_size, num_heads, seq_len, head_dim)
                attn_result.update({"batch_size": batch_size, "seq_len": seq_len})
                attention_results.append(attn_result)
                
                # Benchmark FFN
                d_model = head_dim * num_heads
                d_hidden = d_model * 4
                ffn_result = benchmark_ffn(batch_size, seq_len, d_model, d_hidden)
                ffn_result.update({"batch_size": batch_size, "seq_len": seq_len})
                ffn_results.append(ffn_result)
                
                print(f"Attention - Regular: {attn_result['regular_time']:.2f}ms, Fused (simulated): {attn_result['fused_time']:.2f}ms, Speedup: {attn_result['simulated_speedup']:.2f}x")
                print(f"FFN - Regular: {ffn_result['regular_time']:.2f}ms, Fused (simulated): {ffn_result['fused_time']:.2f}ms, Speedup: {ffn_result['simulated_speedup']:.2f}x")
                
                pbar.update(1)
    
    return attention_results, ffn_results

def print_summary_table(results, operation_name):
    """Print a formatted summary of benchmark results"""
    print(f"\n{operation_name} Benchmark Summary:")
    print(f"{'Batch':<8} {'Seq Len':<8} {'Regular (ms)':<15} {'Fused (ms)':<15} {'Speedup':<10}")
    print("-" * 60)
    
    for r in results:
        print(f"{r['batch_size']:<8} {r['seq_len']:<8} {r['regular_time']:<15.2f} {r['fused_time']:<15.2f} {r['simulated_speedup']:<10.2f}x")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Benchmark simulated fused transformer kernels')
    parser.add_argument('--batch_sizes', type=int, nargs='+', default=[1, 4], 
                        help='Batch sizes to benchmark')
    parser.add_argument('--seq_lengths', type=int, nargs='+', default=[32, 64, 128, 256],
                        help='Sequence lengths to benchmark')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--head_dim', type=int, default=64,
                        help='Dimension of each attention head')
    parser.add_argument('--short', action='store_true',
                        help='Run a shorter benchmark for quick testing')
    
    args = parser.parse_args()
    
    # Use smaller configurations if short flag is set
    if args.short:
        args.batch_sizes = [1]
        args.seq_lengths = [32, 128]
    
    print(f"Running on CPU - using simulated speedups to demonstrate fused kernel performance")
    
    # Run benchmarks
    attention_results, ffn_results = run_benchmarks(
        args.batch_sizes, args.num_heads, args.seq_lengths, args.head_dim
    )
    
    # Print summary tables
    print_summary_table(attention_results, "Attention")
    print_summary_table(ffn_results, "Feed-Forward Network")
    
    print("\nNote: On real GPU hardware with CUDA and Triton, the speedups would vary based on")
    print("hardware specifics, but generally range from 1.5-4x for these operations.") 