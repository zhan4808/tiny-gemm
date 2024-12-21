import math
import torch
import triton
import triton.language as tl

###############################################################################
# FP16 GEMM Kernel in Triton
###############################################################################
# This kernel computes: C = alpha * A x B + beta * C
#    - A is [M, K], B is [K, N], C is [M, N]
#    - All are assumed to be contiguous in row-major layout for simplicity.
#    - For best performance, you should ensure:
#         - alignment of pointers where possible
#         - block sizes match GPU architecture
#         - correct num_warps, num_stages, etc. for your hardware
#    - We'll focus on FP16. You can adjust for BF16 or FP32 with minimal changes.
#
# NOTE: The kernel is a template. You will need to thoroughly benchmark and tune.
###############################################################################

@triton.jit
def kernel_gemm_fp16(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,  # Tile size along the M dimension
    BLOCK_N: tl.constexpr,  # Tile size along the N dimension
    BLOCK_K: tl.constexpr,  # Tile size along the K dimension
    GROUP_SIZE_M: tl.constexpr  # How many blocks to group together along M
):
    """Each program instance handles a tile of C of size [BLOCK_M, BLOCK_N]."""

    # ------------------------------------------------------
    # Program IDs: we will decompose the 2D grid into (M, N)
    # ------------------------------------------------------
    pid = tl.program_id(axis=0)
    # Warp the pid into 2D: first dim is the M block index, second dim is the N block index
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    # Group the blocks along M dimension for better L2 locality
    group_id = pid // grid_n
    group_id = group_id * GROUP_SIZE_M + (pid % GROUP_SIZE_M)
    pid_n = pid % grid_n  # block id in the N dimension
    pid_m = group_id      # block id in the M dimension

    # If the group goes out of range, we skip
    if pid_m >= grid_m:
        return

    # ------------------------------------------------------
    # Compute the actual indices in M and N for the tile
    # ------------------------------------------------------
    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    # ------------------------------------------------------
    # Create a block of pointers for A and B in increments
    # We'll accumulate partial sums in registers
    # ------------------------------------------------------
    # Range for M, K dimension in A
    # size [BLOCK_M]
    offs_m = m_start + tl.arange(0, BLOCK_M)  # row offsets in A
    # Range for K dimension
    offs_k = tl.arange(0, BLOCK_K)  # col offsets in A / row offset in B

    # Range for N dimension in B
    offs_n = n_start + tl.arange(0, BLOCK_N)

    # For partial accumulators:
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # ------------------------------------------------------
    # We loop over K dimension in steps of BLOCK_K
    # ------------------------------------------------------
    # For each step, we load a tile of size [BLOCK_M, BLOCK_K] from A
    # and a tile of size [BLOCK_K, BLOCK_N] from B
    # Then compute partial accumulations into 'acc'
    # We'll do as many steps as needed to cover the entire K dimension.
    # ------------------------------------------------------
    k_blocks = (K + BLOCK_K - 1) // BLOCK_K

    # Start pointer for A and B in the K dimension
    # We'll increment by BLOCK_K each iteration
    for kb in range(k_blocks):
        # Current offset in K dimension
        k_offset = kb * BLOCK_K

        # Load tile from A
        # We create 2D indices for [BLOCK_M, BLOCK_K]
        # clamp rows if they go out-of-bound
        a_row_mask = offs_m[:, None] < M
        a_col_mask = (k_offset + offs_k[None, :]) < K
        a_mask = a_row_mask & a_col_mask
        # Compute pointer offsets
        # address(A + row*stride_am + (k_offset+col)*stride_ak)
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am) + ((k_offset + offs_k[None, :]) * stride_ak)
        # Load, conditionally with mask
        a_tile = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float16)

        # Load tile from B
        # We create 2D indices for [BLOCK_K, BLOCK_N]
        b_row_mask = (k_offset + offs_k[:, None]) < K
        b_col_mask = offs_n[None, :] < N
        b_mask = b_row_mask & b_col_mask
        # address(B + (k_offset+row)*stride_bk + col*stride_bn)
        b_ptrs = B_ptr + ((k_offset + offs_k[:, None]) * stride_bk) + (offs_n[None, :] * stride_bn)
        b_tile = tl.load(b_ptrs, mask=b_mask, other=0.0).to(tl.float16)

        # ------------------------------------------------------
        # Compute partial matmul on the tile
        # We'll multiply a_tile [BLOCK_M, BLOCK_K] with
        #                 b_tile [BLOCK_K, BLOCK_N]
        # Result is accum [BLOCK_M, BLOCK_N]
        # ------------------------------------------------------
        acc += tl.dot(a_tile, b_tile)

    # After the K loop, we have accumulated partial sums in acc.
    # We now write these partial sums back to C (in FP16 or BF16).
    # We'll clamp rows/cols to not exceed M, N boundaries.

    # Convert from float32 accumulator to FP16
    # For BF16, use .to(tl.bfloat16)
    c_tile = acc.to(tl.float16)

    # Compute pointer addresses for writing
    c_row_mask = offs_m[:, None] < M
    c_col_mask = offs_n[None, :] < N
    c_mask = c_row_mask & c_col_mask
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm) + (offs_n[None, :] * stride_cn)

    # Store the results
    tl.store(c_ptrs, c_tile, mask=c_mask)


def triton_gemm_fp16(
    A: torch.Tensor, 
    B: torch.Tensor, 
    C: torch.Tensor, 
    BLOCK_M=128, 
    BLOCK_N=128, 
    BLOCK_K=32,
    GROUP_SIZE_M=8
):
    """
    Host-side Python wrapper that launches kernel_gemm_fp16.
    - A: [M, K]
    - B: [K, N]
    - C: [M, N] (output)
    
    The kernel expects:
    - All tensors on GPU
    - FP16 data type
    - Row-major layout (strides are [K,1], [N,1], [N,1] for A, B, C respectively)
    """
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must be on GPU"
    assert A.dtype == torch.float16, "A must be FP16"
    assert B.dtype == torch.float16, "B must be FP16"
    assert C.dtype == torch.float16, "C must be FP16"

    M, K = A.shape
    Kb, N = B.shape
    Mb, Nb = C.shape
    assert K == Kb, "A and B dimensions mismatch"
    assert M == Mb and N == Nb, "Output C dimension mismatch"

    # Get pointer addresses
    A_ptr = A.data_ptr()
    B_ptr = B.data_ptr()
    C_ptr = C.data_ptr()

    # Strides for row-major layout
    stride_am = A.stride(0)
    stride_ak = A.stride(1)
    stride_bk = B.stride(0)
    stride_bn = B.stride(1)
    stride_cm = C.stride(0)
    stride_cn = C.stride(1)

    # Compute grid size
    grid = lambda meta: (
        ( (M + meta['BLOCK_M'] - 1) // meta['BLOCK_M'] ) * 
        ( (N + meta['BLOCK_N'] - 1) // meta['BLOCK_N'] ) // 
        meta['GROUP_SIZE_M'],
    )

    # Launch Triton kernel
    kernel_gemm_fp16[grid](
        A_ptr, B_ptr, C_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_M=BLOCK_M, 
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K, 
        GROUP_SIZE_M=GROUP_SIZE_M,
        num_stages=2,   # typical choice, tune as needed
        num_warps=4     # typical choice, tune based on GPU
    )


###############################################################################
# Example usage / basic test
###############################################################################
if __name__ == "__main__":
    # Problem size
    M, N, K = 256, 256, 256
    
    # Create random input
    A = torch.randn((M, K), device='cuda', dtype=torch.float16)
    B = torch.randn((K, N), device='cuda', dtype=torch.float16)
    # Output buffer
    C = torch.zeros((M, N), device='cuda', dtype=torch.float16)

    # Ground truth from PyTorch (FP16) with cuBLAS
    # For validation only. For large M,N,K this can be slow in Python mode.
    # Also note that default cublas might accumulate in FP32 internally,
    # so they should match fairly closely.
    C_ref = (A @ B)

    # Run our Triton kernel
    triton_gemm_fp16(A, B, C, BLOCK_M=128, BLOCK_N=128, BLOCK_K=32, GROUP_SIZE_M=8)

    # Compare
    max_abs_diff = (C - C_ref).abs().max().item()
    print(f"Max abs diff: {max_abs_diff:.6f}")
    if max_abs_diff < 1e-2:
        print("Success: Triton kernel and cuBLAS match within tolerance!")
    else:
        print("Warning: Triton kernel and cuBLAS differ. Tune or debug further.")

    # ------------------------------------------------------------------------
    # Benchmark the kernel vs. cuBLAS with torch.matmul
    # ------------------------------------------------------------------------
    import time

    def benchmark_op(op, warmup=3, rep=10):
        # Warmup
        for _ in range(warmup):
            op()
            torch.cuda.synchronize()
        # Timing
        start = time.time()
        for _ in range(rep):
            op()
        torch.cuda.synchronize()
        end = time.time()
        return (end - start) / rep

    # Triton kernel
    def triton_run():
        triton_gemm_fp16(A, B, C)

    # cuBLAS kernel
    def cublas_run():
        torch.matmul(A, B)

    triton_time = benchmark_op(triton_run)
    cublas_time = benchmark_op(cublas_run)
    print(f"Triton GEMM time:  {triton_time*1e3:.3f} ms")
    print(f"cuBLAS GEMM time:  {cublas_time*1e3:.3f} ms")