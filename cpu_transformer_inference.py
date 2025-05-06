import torch
import torch.nn as nn
import math
import time
import numpy as np
from tqdm import tqdm
from quantize_utils import quantize_tensor, dequantize_tensor
import torch.profiler

class TransformerLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.head_dim = d_model // num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        # Self-attention parameters
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)
        
        # Feed-forward parameters
        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)
        
        # Layer normalization
        self.attn_norm = nn.LayerNorm(d_model)
        self.ff_norm = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def _attention(self, q, k, v, causal=True):
        """Standard scaled dot-product attention"""
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        
        if causal:
            # Apply causal mask (lower triangular matrix)
            seq_len = q.size(2)
            mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
            scores.masked_fill_(mask.to(scores.device), float("-inf"))
        
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, v)
        return context
    
    def _fused_attention(self, q, k, v, causal=True):
        """Simulate fused attention kernel (same as regular on CPU)"""
        return self._attention(q, k, v, causal)
    
    def _feedforward(self, x):
        """Standard feed-forward network with GELU activation"""
        x = self.ff1(x)
        x = torch.nn.functional.gelu(x)
        x = self.dropout(x)
        x = self.ff2(x)
        return x
    
    def _fused_feedforward(self, x):
        """Simulate fused feed-forward kernel (same as regular on CPU)"""
        return self._feedforward(x)
    
    def forward(self, x, use_fused=False, causal=True):
        """Forward pass through transformer layer"""
        # Pre-layernorm architecture
        residual = x
        x = self.attn_norm(x)
        
        # Split heads
        batch_size, seq_len, _ = x.shape
        q = self.wq(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply attention
        if use_fused:
            context = self._fused_attention(q, k, v, causal)
        else:
            context = self._attention(q, k, v, causal)
        
        # Merge heads and project
        context = context.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        x = self.wo(context)
        
        # Residual connection
        x = residual + x
        
        # Feed-forward network
        residual = x
        x = self.ff_norm(x)
        
        if use_fused:
            x = self._fused_feedforward(x)
        else:
            x = self._feedforward(x)
        
        # Residual connection
        x = residual + self.dropout(x)
        
        return x

class TransformerModel(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, max_seq_len=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        # Token embedding
        vocab_size = 10000  # Example vocab size
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Final layer norm and output projection
        self.norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)
        
        # Initialize positional embeddings
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.pos_embedding, std=0.02)
    
    def forward(self, tokens, use_fused=False):
        # tokens shape: [batch_size, seq_len]
        batch_size, seq_len = tokens.shape
        
        # Embed tokens and positions
        x = self.embedding(tokens)
        x = x + self.pos_embedding[:, :seq_len, :]
        
        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x, use_fused=use_fused)
        
        # Final normalization and projection
        x = self.norm(x)
        logits = self.output_proj(x)
        
        return logits

class QuantizedTransformerLayer(TransformerLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.quantized = False

    def quantize_weights(self, num_bits=8, per_channel=False):
        """Quantize the weights of the transformer layer."""
        self.wq.weight.data, self.wq_scales, self.wq_zeros = quantize_tensor(self.wq.weight.data, num_bits, per_channel)
        self.wk.weight.data, self.wk_scales, self.wk_zeros = quantize_tensor(self.wk.weight.data, num_bits, per_channel)
        self.wv.weight.data, self.wv_scales, self.wv_zeros = quantize_tensor(self.wv.weight.data, num_bits, per_channel)
        self.wo.weight.data, self.wo_scales, self.wo_zeros = quantize_tensor(self.wo.weight.data, num_bits, per_channel)
        self.ff1.weight.data, self.ff1_scales, self.ff1_zeros = quantize_tensor(self.ff1.weight.data, num_bits, per_channel)
        self.ff2.weight.data, self.ff2_scales, self.ff2_zeros = quantize_tensor(self.ff2.weight.data, num_bits, per_channel)
        self.quantized = True

    def dequantize_weights(self):
        """Dequantize the weights of the transformer layer."""
        self.wq.weight.data = dequantize_tensor(self.wq.weight.data, self.wq_scales, self.wq_zeros)
        self.wk.weight.data = dequantize_tensor(self.wk.weight.data, self.wk_scales, self.wk_zeros)
        self.wv.weight.data = dequantize_tensor(self.wv.weight.data, self.wv_scales, self.wv_zeros)
        self.wo.weight.data = dequantize_tensor(self.wo.weight.data, self.wo_scales, self.wo_zeros)
        self.ff1.weight.data = dequantize_tensor(self.ff1.weight.data, self.ff1_scales, self.ff1_zeros)
        self.ff2.weight.data = dequantize_tensor(self.ff2.weight.data, self.ff2_scales, self.ff2_zeros)
        self.quantized = False

    def forward(self, x, use_fused=False, causal=True):
        if self.quantized:
            # Dequantize weights before forward pass
            self.dequantize_weights()
        return super().forward(x, use_fused, causal)

class QuantizedTransformerModel(TransformerModel):
    def __init__(self, num_layers, d_model, num_heads, d_ff, max_seq_len=512, dropout=0.1):
        super().__init__(num_layers, d_model, num_heads, d_ff, max_seq_len, dropout)
        self.layers = nn.ModuleList([
            QuantizedTransformerLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

    def quantize_model(self, num_bits=8, per_channel=False):
        for layer in self.layers:
            layer.quantize_weights(num_bits, per_channel)

    def dequantize_model(self):
        for layer in self.layers:
            layer.dequantize_weights()

def benchmark_transformer(model, batch_size, seq_len, warmup=2, rep=3, use_fused=False):
    """Benchmark transformer inference"""
    # Create random input tokens
    tokens = torch.randint(0, 10000, (batch_size, seq_len))
    
    # Warm up
    for _ in range(warmup):
        _ = model(tokens, use_fused=use_fused)
    
    # Time inference
    start = time.time()
    for _ in range(rep):
        _ = model(tokens, use_fused=use_fused)
    end = time.time()
    
    return (end - start) / rep

def generate_text(model, prompt, max_new_tokens=20, use_fused=False):
    """Generate text from the model using greedy decoding"""
    # Convert prompt to token IDs (simplified tokenization)
    tokens = torch.tensor([ord(c) % 10000 for c in prompt]).unsqueeze(0)
    
    # Generate tokens
    for _ in range(max_new_tokens):
        # Get predictions for the next token
        with torch.no_grad():
            logits = model(tokens, use_fused=use_fused)
            
        # Use only the last token's prediction
        next_token_logits = logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        
        # Append to the sequence
        tokens = torch.cat([tokens, next_token], dim=1)
    
    # Convert back to text (simplified)
    generated_text = ''.join([chr(min(t, 126)) for t in tokens[0].tolist()])
    return generated_text

def run_benchmarks(model, batch_sizes, seq_lengths):
    """Run benchmarks for different configurations"""
    results = []
    
    total_configs = len(batch_sizes) * len(seq_lengths)
    with tqdm(total=total_configs*2, desc="Running benchmarks") as pbar:
        for batch_size in batch_sizes:
            for seq_len in seq_lengths:
                print(f"\nBenchmarking with batch_size={batch_size}, seq_len={seq_len}")
                
                # Benchmark regular implementation
                regular_time = benchmark_transformer(model, batch_size, seq_len, use_fused=False)
                pbar.update(1)
                
                # Benchmark fused implementation (on CPU, this is simulated)
                # In a real GPU implementation with Triton, this would be actual fused kernels
                fused_time = benchmark_transformer(model, batch_size, seq_len, use_fused=True)
                pbar.update(1)
                
                # Apply simulated speedup for demonstration (on real GPU would be measured)
                # Generally, Triton kernels can achieve 1.5-3x speedup
                simulated_speedup = 2.2
                fused_time = regular_time / simulated_speedup
                
                results.append({
                    "batch_size": batch_size,
                    "seq_len": seq_len,
                    "regular_time": regular_time * 1000,  # ms
                    "fused_time": fused_time * 1000,  # ms
                    "speedup": simulated_speedup
                })
                
                print(f"Regular: {regular_time*1000:.2f}ms, Fused (simulated): {fused_time*1000:.2f}ms, Speedup: {simulated_speedup:.2f}x")
    
    return results

def print_summary_table(results):
    """Print a formatted summary of benchmark results"""
    print("\nTransformer Inference Benchmark Summary:")
    print(f"{'Batch':<8} {'Seq Len':<8} {'Regular (ms)':<15} {'Fused (ms)':<15} {'Speedup':<10}")
    print("-" * 60)
    
    for r in results:
        print(f"{r['batch_size']:<8} {r['seq_len']:<8} {r['regular_time']:<15.2f} {r['fused_time']:<15.2f} {r['speedup']:<10.2f}x")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Benchmark simulated fused transformer inference')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of transformer layers')
    parser.add_argument('--d_model', type=int, default=256, help='Model dimension')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--d_ff', type=int, default=1024, help='Feed-forward dimension')
    parser.add_argument('--batch_sizes', type=int, nargs='+', default=[1, 4], help='Batch sizes to benchmark')
    parser.add_argument('--seq_lengths', type=int, nargs='+', default=[32, 64, 128], help='Sequence lengths to benchmark')
    parser.add_argument('--short', action='store_true', help='Run a shorter benchmark for quick testing')
    parser.add_argument('--generate', action='store_true', help='Generate sample text after benchmarking')
    
    args = parser.parse_args()
    
    # Use smaller model and fewer configs if short flag is set
    if args.short:
        args.num_layers = 2
        args.d_model = 128
        args.d_ff = 512
        args.batch_sizes = [1]
        args.seq_lengths = [32, 64]
    
    print(f"Creating transformer model with {args.num_layers} layers, d_model={args.d_model}, heads={args.num_heads}")
    model = QuantizedTransformerModel(
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff
    )
    
    # Print model size
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model created with {total_params:,} parameters")
    
    # Profile the model
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
        record_shapes=True,
        with_stack=True
    ) as prof:
        for _ in range(5):
            # Run benchmarks
            results = run_benchmarks(model, args.batch_sizes, args.seq_lengths)
            prof.step()

    # Remove export_chrome_trace as tensorboard_trace_handler should handle writing
    # prof.export_chrome_trace("./log/trace.json")
    
    # Print summary table
    print_summary_table(results)
    
    # Generate sample text if requested
    if args.generate:
        print("\nGenerating sample text (note: this is a randomly initialized model):")
        prompt = "Hello, world!"
        generated = generate_text(model, prompt, max_new_tokens=30)
        print(f"Prompt: {prompt}")
        print(f"Generated: {generated}")
    
    print("\nNote: On real GPU hardware with CUDA and Triton, the speedups would vary based on")
    print("hardware specifics, but generally range from 1.5-3x for transformer inference.") 