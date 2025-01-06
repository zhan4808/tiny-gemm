import math
import torch
import triton
import triton.language as tl

@triton.jit
def kernel_gemm_int4(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr
):
    pid = tl.program_id(axis=0)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    group_id = pid // grid_n
    group_id = group_id * GROUP_SIZE_M + (pid % GROUP_SIZE_M)
    pid_n = pid % grid_n
    pid_m = group_id

    if pid_m >= grid_m:
        return

    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    offs_m = m_start + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    offs_n = n_start + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k_blocks = (K + BLOCK_K - 1) // BLOCK_K

    for kb in range(k_blocks):
        k_offset = kb * BLOCK_K

        a_row_mask = offs_m[:, None] < M
        a_col_mask = (k_offset + offs_k[None, :]) < K
        a_mask = a_row_mask & a_col_mask

        a_ptrs = A_ptr + (offs_m[:, None] * stride_am) + ((k_offset + offs_k[None, :]) * stride_ak)
        a_tile = tl.load(a_ptrs, mask=a_mask, other=0).to(tl.int4)

        b_row_mask = (k_offset + offs_k[:, None]) < K
        b_col_mask = offs_n[None, :] < N
        b_mask = b_row_mask & b_col_mask

        b_ptrs = B_ptr + ((k_offset + offs_k[:, None]) * stride_bk) + (offs_n[None, :] * stride_bn)
        b_tile = tl.load(b_ptrs, mask=b_mask, other=0).to(tl.int4)

        acc += tl.dot(a_tile.to(tl.float32), b_tile.to(tl.float32))

    c_tile = acc.to(tl.int4)

    c_row_mask = offs_m[:, None] < M
    c_col_mask = offs_n[None, :] < N
    c_mask = c_row_mask & c_col_mask

    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm) + (offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, c_tile, mask=c_mask)


def triton_gemm_int4(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    BLOCK_M=128,
    BLOCK_N=128,
    BLOCK_K=32,
    GROUP_SIZE_M=8
):
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must be on GPU"
    assert A.dtype == torch.int4 and B.dtype == torch.int4 and C.dtype == torch.int4

    M, K = A.shape
    Kb, N = B.shape
    Mb, Nb = C.shape
    assert K == Kb and M == Mb and N == Nb

    A_ptr = A.data_ptr()
    B_ptr = B.data_ptr()
    C_ptr = C.data_ptr()

    stride_am, stride_ak = A.stride()
    stride_bk, stride_bn = B.stride()
    stride_cm, stride_cn = C.stride()

    grid_size_fn = lambda meta: (
        (M + meta['BLOCK_M'] - 1) // meta['BLOCK_M'] *
        (N + meta['BLOCK_N'] - 1) // meta['BLOCK_N'] //
        meta['GROUP_SIZE_M'],
     )

    kernel_gemm_int4[grid_size_fn](
        A_ptr, B_ptr, C_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        num_stages=2,
        num_warps=2
     )



###############################################################################
# Example usage / basic test
###############################################################################
if __name__ == "__main__":
    # Problem size
    M, N, K = 256, 256, 256
    
    # Create random input
    A = torch.randn((M, K), device='cuda', dtype=torch.int)
    B = torch.randn((K, N), device='cuda', dtype=torch.int)
    # Output buffer
    C = torch.zeros((M, N), device='cuda', dtype=torch.int)

    # Ground truth from PyTorch (FP16) with cuBLAS
    # For validation only. For large M,N,K this can be slow in Python mode.
    # Also note that default cublas might accumulate in FP32 internally,
    # so they should match fairly closely.
    C_ref = (A @ B)

    # Run our Triton kernel
    triton_gemm_int4(A, B, C, BLOCK_M=128, BLOCK_N=128, BLOCK_K=32, GROUP_SIZE_M=8)

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
        triton_gemm_int4(A, B, C)

    # cuBLAS kernel
    def cublas_run():
        torch.matmul(A, B)

    triton_time = benchmark_op(triton_run)
    cublas_time = benchmark_op(cublas_run)
    print(f"Triton GEMM time:  {triton_time*1e3:.3f} ms")
    print(f"cuBLAS GEMM time:  {cublas_time*1e3:.3f} ms")