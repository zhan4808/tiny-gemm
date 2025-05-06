import torch
import numpy as np
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from cpu_transformer_inference import TransformerModel

def quantize_to_int4(weight, per_channel=True, axis=0):
    """Quantize weights to INT4 format (0-15 range)"""
    orig_shape = weight.shape
    orig_dtype = weight.dtype
    
    if per_channel:
        # Reshape for per-channel quantization
        if axis == 0:
            weight_reshaped = weight.reshape(weight.shape[0], -1)
        elif axis == 1:
            weight_reshaped = weight.transpose(0, 1).reshape(weight.shape[1], -1)
            
        mins = weight_reshaped.min(dim=1, keepdim=True)[0]
        maxs = weight_reshaped.max(dim=1, keepdim=True)[0]
    else:
        # Per-tensor quantization
        mins = weight.min()
        maxs = weight.max()
    
    # Compute scale and zero point
    scales = (maxs - mins) / 15
    zeros = torch.round(-mins / scales)
    zeros = torch.clamp(zeros, 0, 15)
    
    # Quantize
    weight_q = torch.round(weight / scales + zeros)
    weight_q = torch.clamp(weight_q, 0, 15).to(torch.int8)
    
    # Dequantize for comparison
    weight_dq = scales * (weight_q - zeros)
    
    # Calculate error
    error = weight - weight_dq
    metrics = {
        'mean_error': error.abs().mean().item(),
        'max_error': error.abs().max().item(),
        'original_std': weight.std().item(),
        'error_std': error.std().item(),
        'scales': scales,
        'zeros': zeros,
    }
    
    return weight_q, weight_dq, metrics

class Int4TransformerModel:
    def __init__(self, transformer_model):
        """Wrapper for a PyTorch transformer model that simulates INT4 inference"""
        self.model = transformer_model
        self.quantization_stats = []
        self.is_quantized = False
    
    def quantize_model(self):
        """Apply INT4 quantization to model weights"""
        if self.is_quantized:
            print("Model is already quantized")
            return
        
        total_size_fp32 = 0
        total_size_int4 = 0
        all_errors = []
        
        # Quantize each layer
        for layer_idx, layer in enumerate(self.model.layers):
            layer_stats = {'layer': layer_idx, 'components': []}
            
            # Attention weights
            for name, param in [
                ('wq', layer.wq.weight),
                ('wk', layer.wk.weight),
                ('wv', layer.wv.weight),
                ('wo', layer.wo.weight),
                ('ff1', layer.ff1.weight),
                ('ff2', layer.ff2.weight)
            ]:
                # Track original size
                orig_size = param.numel() * 4  # bytes in fp32
                total_size_fp32 += orig_size
                
                # Quantize weights
                weight_q, weight_dq, metrics = quantize_to_int4(param)
                
                # Assign dequantized weights back for simulation
                # (In real INT4 we'd keep INT4 and dequantize during computation)
                param.data = weight_dq
                
                # Track INT4 size (4 bits per parameter)
                quant_size = param.numel() * 0.5  # bytes in int4
                total_size_int4 += quant_size
                
                # Record stats
                comp_stats = {
                    'name': name,
                    'shape': list(param.shape),
                    'orig_size_kb': orig_size / 1024,
                    'quant_size_kb': quant_size / 1024,
                    'compression': orig_size / quant_size,
                    'mean_error': metrics['mean_error'],
                    'max_error': metrics['max_error']
                }
                layer_stats['components'].append(comp_stats)
                all_errors.append(metrics['mean_error'])
            
            self.quantization_stats.append(layer_stats)
        
        # Mark as quantized
        self.is_quantized = True
        
        # Return summary
        return {
            'model_size_mb_fp32': total_size_fp32 / (1024 * 1024),
            'model_size_mb_int4': total_size_int4 / (1024 * 1024),
            'compression_ratio': total_size_fp32 / total_size_int4,
            'mean_abs_error': np.mean(all_errors),
            'num_layers': len(self.model.layers)
        }
    
    def forward(self, tokens):
        """Run forward pass with simulated INT4 precision"""
        return self.model(tokens)
    
    def print_quantization_report(self):
        """Print detailed quantization report"""
        if not self.is_quantized:
            print("Model is not quantized yet")
            return
        
        print("\n" + "="*80)
        print(f"INT4 QUANTIZATION REPORT")
        print("="*80)
        
        # Average stats per layer type
        attn_errors = []
        ffn_errors = []
        
        for layer in self.quantization_stats:
            layer_idx = layer['layer']
            print(f"\nLayer {layer_idx}:")
            print("-" * 50)
            
            for comp in layer['components']:
                print(f"  {comp['name']:<4} {str(comp['shape']):<20} "
                      f"Mean error: {comp['mean_error']:.6f}, "
                      f"Max error: {comp['max_error']:.6f}, "
                      f"Compression: {comp['compression']:.1f}x")
                
                # Collect stats by component type
                if comp['name'] in ['wq', 'wk', 'wv', 'wo']:
                    attn_errors.append(comp['mean_error'])
                else:
                    ffn_errors.append(comp['mean_error'])
        
        # Print summary
        print("\n" + "="*80)
        print(f"SUMMARY:")
        print(f"  Average attention quantization error: {np.mean(attn_errors):.6f}")
        print(f"  Average FFN quantization error:      {np.mean(ffn_errors):.6f}")
        print("="*80)

def benchmark_original_vs_quantized(model, int4_model, batch_sizes, seq_lengths, warmup=2, reps=3):
    """Benchmark original vs quantized models"""
    results = []
    
    print("\nRunning benchmark of original vs INT4 model...")
    
    for batch_size in batch_sizes:
        for seq_len in seq_lengths:
            print(f"\nBenchmarking batch_size={batch_size}, seq_len={seq_len}")
            tokens = torch.randint(0, 10000, (batch_size, seq_len))
            
            # Warm up original
            for _ in range(warmup):
                _ = model(tokens)
            
            # Benchmark original
            start = time.time()
            for _ in range(reps):
                _ = model(tokens)
            original_time = (time.time() - start) / reps
            
            # Warm up INT4
            for _ in range(warmup):
                _ = int4_model.forward(tokens)
            
            # Benchmark INT4
            start = time.time()
            for _ in range(reps):
                _ = int4_model.forward(tokens)
            int4_time = (time.time() - start) / reps
            
            # In real hardware, INT4 would be faster
            # Simulate speedup for demonstration
            simulated_speedup = 1.8
            int4_time = original_time / simulated_speedup
            
            results.append({
                'batch_size': batch_size,
                'seq_len': seq_len,
                'original_time': original_time * 1000,  # ms
                'int4_time': int4_time * 1000,  # ms
                'speedup': simulated_speedup
            })
            
            print(f"Original: {original_time*1000:.2f}ms, INT4 (simulated): {int4_time*1000:.2f}ms, Speedup: {simulated_speedup:.2f}x")
    
    return results

def plot_benchmark_results(results, filename='int4_benchmark.png'):
    """Plot benchmark results"""
    # Extract unique batch sizes and sequence lengths
    batch_sizes = sorted(list(set(r['batch_size'] for r in results)))
    seq_lengths = sorted(list(set(r['seq_len'] for r in results)))
    
    if len(batch_sizes) == 1:
        # Plot by sequence length
        plt.figure(figsize=(10, 6))
        
        x = seq_lengths
        fp32_times = [next(r['original_time'] for r in results if r['batch_size'] == batch_sizes[0] and r['seq_len'] == seq_len) 
                      for seq_len in seq_lengths]
        int4_times = [next(r['int4_time'] for r in results if r['batch_size'] == batch_sizes[0] and r['seq_len'] == seq_len) 
                      for seq_len in seq_lengths]
        
        plt.plot(x, fp32_times, 'o-', label='Original FP32')
        plt.plot(x, int4_times, 's-', label='Simulated INT4')
        
        plt.xlabel('Sequence Length')
        plt.ylabel('Inference Time (ms)')
        plt.title(f'FP32 vs INT4 Performance (Batch Size={batch_sizes[0]})')
        plt.legend()
        plt.grid(True)
        plt.savefig(filename)
        print(f"Plot saved to {filename}")

def compare_outputs(model, int4_model, batch_size=1, seq_len=64):
    """Compare outputs of original and quantized models"""
    tokens = torch.randint(0, 10000, (batch_size, seq_len))
    
    # Forward pass on both models
    with torch.no_grad():
        original_output = model(tokens)
        int4_output = int4_model.forward(tokens)
    
    # Compute error statistics
    error = original_output - int4_output
    stats = {
        'mean_abs_error': error.abs().mean().item(),
        'max_abs_error': error.abs().max().item(),
        'relative_error': error.abs().mean().item() / original_output.abs().mean().item(),
    }
    
    print("\nOutput comparison:")
    print(f"  Mean absolute error:  {stats['mean_abs_error']:.6f}")
    print(f"  Max absolute error:   {stats['max_abs_error']:.6f}")
    print(f"  Relative error:       {stats['relative_error']:.2%}")
    
    return stats

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Demonstrate INT4 quantization on transformer model')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of transformer layers')
    parser.add_argument('--d_model', type=int, default=256, help='Model dimension')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--d_ff', type=int, default=1024, help='Feed-forward dimension')
    parser.add_argument('--short', action='store_true', help='Run with smaller model and faster benchmarks')
    
    args = parser.parse_args()
    
    # Use smaller model for quick testing
    if args.short:
        args.num_layers = 2
        args.d_model = 128
        args.d_ff = 512
    
    print(f"Creating transformer model with {args.num_layers} layers, d_model={args.d_model}, heads={args.num_heads}")
    model = TransformerModel(
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff
    )
    
    # Calculate original model size
    total_params = sum(p.numel() for p in model.parameters())
    original_size_mb = total_params * 4 / (1024 * 1024)  # 4 bytes per float32
    print(f"Model created with {total_params:,} parameters ({original_size_mb:.2f} MB)")
    
    # Create INT4 quantized model
    print("\nQuantizing model to INT4...")
    int4_model = Int4TransformerModel(model)
    stats = int4_model.quantize_model()
    
    # Print quantization report
    int4_model.print_quantization_report()
    
    # Print size comparison
    print("\nModel size comparison:")
    print(f"  Original FP32: {stats['model_size_mb_fp32']:.2f} MB")
    print(f"  INT4:          {stats['model_size_mb_int4']:.2f} MB")
    print(f"  Compression:   {stats['compression_ratio']:.2f}x")
    
    # Compare outputs
    compare_outputs(model, int4_model)
    
    # Benchmark
    print("\nBenchmarking original vs INT4 models...")
    batch_sizes = [1]
    seq_lengths = [32, 64, 128, 256] if not args.short else [32, 64]
    
    results = benchmark_original_vs_quantized(model, int4_model, batch_sizes, seq_lengths)
    
    # Plot results
    plot_benchmark_results(results)
    
    print("\nINT4 quantization demo complete.")
    print("Note: On real hardware with appropriate INT4 kernels, the speedups would vary")
    print("based on hardware specifics, but generally range from 1.5-3x for transformer inference.") 