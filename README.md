# Tiny-GEMM: Optimized Triton GEMM for Small Batch Transformer Inference

A collection of optimized matrix multiplication kernels using Triton for efficient transformer model inference on resource-constrained hardware.

## Features

- Fused attention mechanism optimized for small batch sizes and low-latency inference
- Fused feed-forward network with optimized memory access patterns
- Support for typical transformer operations (GELU, causal masking, etc.)
- Benchmarking utilities to compare with PyTorch
- Custom quantization framework for INT4 quantization
- Profiling setup for identifying performance bottlenecks

## Implementation Details

This project implements two key transformer components as fused operations in Triton:

1. **Fused Multi-Head Attention**: Computes `Softmax(Q·K^T/sqrt(d_k))·V` in a single kernel, with optimizations:
   - Efficient tiling for small batch operation
   - Memory locality optimizations for Q, K, V matrices
   - Causal masking for autoregressive models
   - Optional attention dropout

2. **Fused Feed-Forward Network**: Computes `Act(X·W1 + B1)·W2 + B2` in a single kernel, with:
   - Activation functions (GELU, ReLU, SiLU)
   - Weight-stationary design to minimize memory transfer
   - Blocking strategy for efficient matrix multiplication
   - Support for standard transformer hidden dimension patterns

## Optimizations

The implementation includes several key optimizations for small-batch transformer inference:

- **Memory Hierarchy Utilization**: Efficient use of L1/L2 cache and shared memory
- **Tiling Strategies**: Blocking techniques to maximize data reuse
- **Fused Operations**: Combined multiple operations to reduce memory traffic
- **Reduced Precision Support**: FP16 computation for improved throughput
- **Cache-Aware Layout**: Memory access patterns designed for coalesced memory access

## Custom Quantization Framework

- Implemented custom quantization and dequantization functions for INT4 quantization.
- Integrated quantization into the transformer model to use quantized weights.
- Demonstrated performance improvements with quantized models.

## Profiling and Bottleneck Identification

- Integrated PyTorch profiler to identify performance bottlenecks.
- Set up TensorBoard for visualizing profiling data.
- Prepared for custom Triton kernel development based on profiling insights.

## Usage

### Fused Attention

```python
from triton_fused_transformer import fused_attention

# [B, H, N, D] format tensors
# batch_size = 1, num_heads = 8, seq_len = 512, head_dim = 64
q = torch.randn(1, 8, 512, 64, device='cuda', dtype=torch.float16)
k = torch.randn(1, 8, 512, 64, device='cuda', dtype=torch.float16)
v = torch.randn(1, 8, 512, 64, device='cuda', dtype=torch.float16)

# Compute attention with causal masking
output = fused_attention(q, k, v, causal=True)
```

### Fused FFN

```python
from triton_fused_transformer import fused_ffn

# Input: [B, N, D]
# batch_size = 1, seq_len = 512, d_model = 512
x = torch.randn(1, 512, 512, device='cuda', dtype=torch.float16)
w1 = torch.randn(512, 2048, device='cuda', dtype=torch.float16)  # Expand to 4x hidden dimension
b1 = torch.randn(2048, device='cuda', dtype=torch.float16)
w2 = torch.randn(2048, 512, device='cuda', dtype=torch.float16)  # Project back to model dimension
b2 = torch.randn(512, device='cuda', dtype=torch.float16)

# Compute FFN with GELU activation
output = fused_ffn(x, w1, b1, w2, b2, activation="gelu")
```

## Benchmarking

Run benchmarks to compare performance against PyTorch:

```bash
# Benchmark across sequence lengths
python benchmark_fused_transformer.py --mode=seq_length --seq_lengths 128 256 512 1024 2048

# Benchmark across batch sizes
python benchmark_fused_transformer.py --mode=batch_size --batch_sizes 1 2 4 8 16
```

## Requirements

- PyTorch >= 1.13
- Triton >= 2.0
- CUDA compatible GPU
- numpy
- matplotlib (for benchmarking visualization)

## Project Structure

- `triton_fused_transformer.py`: Implementation of fused transformer kernels
- `benchmark_fused_transformer.py`: Benchmarking utilities
- `triton_gemm.py`: Base GEMM implementation
- `cpu_transformer_inference.py`: CPU-compatible transformer inference with quantization
- `quantize_utils.py`: Custom quantization utilities

## Future Work

- Fine-tuned custom Triton kernel development based on profiling insights
- Support for more activation functions
- Flash Attention 2 style optimizations
- Additional transformer components (layer norm, residual blocks)

# triton_gemm.py
**Author:** Robert Zhang

---

## 1. Tiling & Block Sizes

We define three block-size parameters: `BLOCK_M`, `BLOCK_N`, and `BLOCK_K`.  
In typical transformer layers, you would tune these block sizes to suit the shapes of your **Q**, **K**, **V**, and feedforward matrices.  

Different GPUs benefit from different tile configurations. For example, on an A100 you might try:
- `(BLOCK_M, BLOCK_N, BLOCK_K) = (128, 128, 32)`, or 
- `(128, 256, 64)`.

---

## 2. Grouping

The kernel uses a `GROUP_SIZE_M` concept to group blocks along the M dimension. This helps reuse data in L2/L1 caches effectively and can boost performance for large M. Tuning `GROUP_SIZE_M` is a knob you can experiment with.

---

## 3. Accumulator Precision

We accumulate partial sums in `tl.float32` to improve numerical stability when multiplying FP16 operands. This is standard practice in ML frameworks: multiply in FP16, accumulate in FP32 (a.k.a. "FP16->FP32 mixed-precision").

At the end of the loop, we cast the accumulator back to FP16. If you want BF16, you can change the `to(tl.float16)` calls to `to(tl.bfloat16)`.

---

## 4. Memory Access and Masking

We explicitly compute pointer addresses with A_ptr + row * stride_am + col * stride_ak.
For large or small matrices, some indices might fall out of the valid range. We use tl.load(…, mask=…)
to mask out-of-bounds threads, setting them to zero.

We do the same in the store path for writing to C.

---

## 5. **Looping over K**
   - A `for kb in range(k_blocks)` loop chunks the `K` dimension into tiles of size `BLOCK_K`.
   - Each iteration loads sub-tiles from `A` and `B`, then performs a `tl.dot` to accumulate partial sums into `acc`.

---

## 6. **Kernel Launch Configuration**
   - Define the "grid" lambda for Triton to know how many 2D blocks to launch.
   - Each block corresponds to a tile `[BLOCK_M, BLOCK_N]` in the `C` matrix.
   - Incorporates `GROUP_SIZE_M` for grouping.
   - **Performance tuning knobs**:
     - `num_stages` (e.g., 2 or 3).
     - `num_warps` (e.g., 4 or 8), depending on your GPU.
   - Profiling tools like Nsight Systems/Compute are recommended.

---

## 7. **Benchmarks**
   - Includes a Python timing harness comparing the Triton kernel with `torch.matmul()` (uses cuBLAS under-the-hood).
   - Metrics to measure:
     - Throughput (GFLOPs/s).
     - Memory bandwidth utilization.
     - Other relevant metrics via Nsight.
    
---

## 8. **Extensions**
   - **Specialized self-attention kernel**:
     - Fuse "Q x K^T" with softmax scaling or "(QK^T) x V" for "FlashAttention"-style fusion.
     - Incorporate head dimension (`B` or `H`) in indexing for multi-head attention.
     - Use shared memory for advanced memory reuse.
   - **Specialized MLP feedforward kernel**:
     - Handle two GEMMs in a row (e.g., `xW1 + b1`, activation, and `outputW2 + b2`) for partial fusion.
     - 
---

## 9. **Next Steps**
   1. Tune block sizes:
      - Adjust `(BLOCK_M, BLOCK_N, BLOCK_K)` to match workload shapes and GPU architecture.
   2. Check occupancy:
      - Use `triton.testing.test_kernel`, Nsight Systems, or `nvprof` to ensure saturation of GPU SMs.
   3. Experiment with data layouts:
      - For example, use a transposed layout for `B` for more coalesced loads.
   4. Warp-level matrix ops:
      - Experiment with Tensor Cores for advanced patterns.
   5. Benchmark on real workloads:
      - Validate performance on transformer blocks using your custom GEMM, replacing cuBLAS/cuBLASLt calls.
      - Measure training throughput, memory usage, and numerical stability.

---
