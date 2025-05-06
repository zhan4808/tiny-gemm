import torch
import math
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from triton_fused_transformer import fused_attention, fused_ffn

def benchmark_attention(batch_size, num_heads, seq_len, head_dim, causal=True, dtype=torch.float16, warmup=10, rep=50):
    """Benchmark attention implementations"""
    device = torch.device('cuda')
    
    # Create inputs
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    
    # PyTorch reference implementation
    def torch_attention():
        # q, k, v: [B, H, N, D]
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(head_dim)
        if causal:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
            scores.masked_fill_(mask, float("-inf"))
        attn_weights = torch.softmax(scores, dim=-1)
        return torch.matmul(attn_weights, v)
    
    # Triton implementation
    def triton_attention():
        return fused_attention(q, k, v, causal=causal)
    
    # Warmup
    for _ in range(warmup):
        torch_out = torch_attention()
        triton_out = triton_attention()
        torch.cuda.synchronize()
    
    # Correctness check
    torch_out = torch_attention()
    triton_out = triton_attention()
    max_diff = (torch_out - triton_out).abs().max().item()
    print(f"Attention max absolute difference: {max_diff:.6f}")
    
    # Benchmark PyTorch
    torch.cuda.synchronize()
    torch_start = time.time()
    for _ in range(rep):
        torch_attention()
        torch.cuda.synchronize()
    torch_end = time.time()
    torch_time = (torch_end - torch_start) / rep
    
    # Benchmark Triton
    torch.cuda.synchronize()
    triton_start = time.time()
    for _ in range(rep):
        triton_attention()
        torch.cuda.synchronize()
    triton_end = time.time()
    triton_time = (triton_end - triton_start) / rep
    
    return {
        "torch_time": torch_time * 1000,  # ms
        "triton_time": triton_time * 1000,  # ms
        "speedup": torch_time / triton_time,
        "max_diff": max_diff
    }

def benchmark_ffn(batch_size, seq_len, d_model, d_hidden, dtype=torch.float16, activation="gelu", warmup=10, rep=50):
    """Benchmark feed-forward network implementations"""
    device = torch.device('cuda')
    
    # Create inputs
    x = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype)
    w1 = torch.randn(d_model, d_hidden, device=device, dtype=dtype)
    b1 = torch.randn(d_hidden, device=device, dtype=dtype)
    w2 = torch.randn(d_hidden, d_model, device=device, dtype=dtype)
    b2 = torch.randn(d_model, device=device, dtype=dtype)
    
    # PyTorch reference implementation
    def torch_ffn():
        hidden = torch.matmul(x, w1) + b1
        if activation == "gelu":
            hidden = torch.nn.functional.gelu(hidden)
        elif activation == "relu":
            hidden = torch.nn.functional.relu(hidden)
        elif activation == "silu":
            hidden = torch.nn.functional.silu(hidden)
        return torch.matmul(hidden, w2) + b2
    
    # Triton implementation
    def triton_ffn():
        return fused_ffn(x, w1, b1, w2, b2, activation=activation)
    
    # Warmup
    for _ in range(warmup):
        torch_out = torch_ffn()
        triton_out = triton_ffn()
        torch.cuda.synchronize()
    
    # Correctness check
    torch_out = torch_ffn()
    triton_out = triton_ffn()
    max_diff = (torch_out - triton_out).abs().max().item()
    print(f"FFN max absolute difference: {max_diff:.6f}")
    
    # Benchmark PyTorch
    torch.cuda.synchronize()
    torch_start = time.time()
    for _ in range(rep):
        torch_ffn()
        torch.cuda.synchronize()
    torch_end = time.time()
    torch_time = (torch_end - torch_start) / rep
    
    # Benchmark Triton
    torch.cuda.synchronize()
    triton_start = time.time()
    for _ in range(rep):
        triton_ffn()
        torch.cuda.synchronize()
    triton_end = time.time()
    triton_time = (triton_end - triton_start) / rep
    
    return {
        "torch_time": torch_time * 1000,  # ms
        "triton_time": triton_time * 1000,  # ms
        "speedup": torch_time / triton_time,
        "max_diff": max_diff
    }

def benchmark_sequence_lengths(batch_sizes, num_heads, seq_lengths, head_dim):
    """Benchmark attention and FFN for different sequence lengths"""
    attention_results = []
    ffn_results = []
    
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
            
            print(f"Attention - PyTorch: {attn_result['torch_time']:.2f}ms, Triton: {attn_result['triton_time']:.2f}ms, Speedup: {attn_result['speedup']:.2f}x")
            print(f"FFN - PyTorch: {ffn_result['torch_time']:.2f}ms, Triton: {ffn_result['triton_time']:.2f}ms, Speedup: {ffn_result['speedup']:.2f}x")
    
    return attention_results, ffn_results

def plot_results(results, title, xlabel, ylabel, filename):
    """Plot benchmark results"""
    plt.figure(figsize=(10, 6))
    
    batch_sizes = sorted(list(set([r["batch_size"] for r in results])))
    seq_lengths = sorted(list(set([r["seq_len"] for r in results])))
    
    if len(batch_sizes) == 1:  # Plot by sequence length
        x = seq_lengths
        torch_times = [next(r["torch_time"] for r in results if r["batch_size"] == batch_sizes[0] and r["seq_len"] == seq_len) for seq_len in seq_lengths]
        triton_times = [next(r["triton_time"] for r in results if r["batch_size"] == batch_sizes[0] and r["seq_len"] == seq_len) for seq_len in seq_lengths]
        
        plt.plot(x, torch_times, 'o-', label='PyTorch')
        plt.plot(x, triton_times, 's-', label='Triton Fused')
        plt.xlabel('Sequence Length')
    else:  # Plot by batch size
        x = batch_sizes
        torch_times = [next(r["torch_time"] for r in results if r["batch_size"] == batch_size and r["seq_len"] == seq_lengths[0]) for batch_size in batch_sizes]
        triton_times = [next(r["triton_time"] for r in results if r["batch_size"] == batch_size and r["seq_len"] == seq_lengths[0]) for batch_size in batch_sizes]
        
        plt.plot(x, torch_times, 'o-', label='PyTorch')
        plt.plot(x, triton_times, 's-', label='Triton Fused')
        plt.xlabel('Batch Size')
    
    plt.ylabel('Time (ms)')
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot speedup
    plt.figure(figsize=(10, 6))
    if len(batch_sizes) == 1:
        speedups = [next(r["speedup"] for r in results if r["batch_size"] == batch_sizes[0] and r["seq_len"] == seq_len) for seq_len in seq_lengths]
        plt.plot(seq_lengths, speedups, 'o-')
        plt.xlabel('Sequence Length')
    else:
        speedups = [next(r["speedup"] for r in results if r["batch_size"] == batch_size and r["seq_len"] == seq_lengths[0]) for batch_size in batch_sizes]
        plt.plot(batch_sizes, speedups, 'o-')
        plt.xlabel('Batch Size')
    
    plt.ylabel('Speedup (x)')
    plt.title(f'{title} - Speedup')
    plt.grid(True)
    plt.axhline(y=1, color='r', linestyle='--')
    plt.savefig(f'{filename}_speedup.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmark fused transformer kernels')
    parser.add_argument('--mode', type=str, default='seq_length', choices=['seq_length', 'batch_size'],
                        help='Benchmarking mode')
    parser.add_argument('--batch_sizes', type=int, nargs='+', default=[1],
                        help='Batch sizes to benchmark')
    parser.add_argument('--seq_lengths', type=int, nargs='+', default=[128, 256, 512, 1024, 2048],
                        help='Sequence lengths to benchmark')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--head_dim', type=int, default=64,
                        help='Dimension of each attention head')
    
    args = parser.parse_args()
    
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print(f"Running on {device_name}")
    else:
        print("CUDA not available - this benchmark requires GPU acceleration")
        exit(1)
    
    if args.mode == 'seq_length':
        batch_sizes = [1]  # Fixed batch size for sequence length benchmarking
        seq_lengths = args.seq_lengths
    else:  # batch_size mode
        batch_sizes = args.batch_sizes
        seq_lengths = [128]  # Fixed sequence length for batch size benchmarking
    
    # Run benchmarks
    attention_results, ffn_results = benchmark_sequence_lengths(
        batch_sizes, args.num_heads, seq_lengths, args.head_dim
    )
    
    # Plot results
    plot_results(
        attention_results,
        f"Multi-Head Attention Performance (heads={args.num_heads}, dim={args.head_dim})",
        "Sequence Length" if args.mode == 'seq_length' else "Batch Size",
        "Time (ms)",
        "attention_benchmark.png"
    )
    
    plot_results(
        ffn_results,
        f"Feed-Forward Network Performance (heads={args.num_heads}, dim={args.head_dim})",
        "Sequence Length" if args.mode == 'seq_length' else "Batch Size",
        "Time (ms)",
        "ffn_benchmark.png"
    )
    
    print("\nBenchmark complete - results saved as PNG files") 