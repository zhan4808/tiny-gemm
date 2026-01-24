import argparse
import time

import torch

from tiny_gemm.quantization.packed_int4 import pack_int4_signed, quantize_per_tensor_int4
from triton_gemm import triton_gemm_packed_int4


def benchmark_op(op, warmup=5, rep=20):
    for _ in range(warmup):
        op()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.time()
    for _ in range(rep):
        op()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end = time.time()
    return (end - start) / rep


def main():
    parser = argparse.ArgumentParser(description="Benchmark packed INT4 GEMM")
    parser.add_argument("--m", type=int, default=512)
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--k", type=int, default=512)
    args = parser.parse_args()

    if args.k % 2 != 0:
        raise ValueError("K must be even for packed INT4")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    A_fp = torch.randn((args.m, args.k), device=device, dtype=torch.float16)
    B_fp = torch.randn((args.k, args.n), device=device, dtype=torch.float16)

    A_q, A_scale = quantize_per_tensor_int4(A_fp)
    B_q, B_scale = quantize_per_tensor_int4(B_fp)
    A_packed = pack_int4_signed(A_q, axis=1)
    B_packed = pack_int4_signed(B_q, axis=0)

    def triton_run():
        triton_gemm_packed_int4(A_packed, B_packed, args.k)

    def ref_run():
        torch.matmul(A_q.float() * A_scale, B_q.float() * B_scale)

    triton_time = benchmark_op(triton_run) if device.type == "cuda" else None
    ref_time = benchmark_op(ref_run)

    print(f"Reference (dequantized) GEMM: {ref_time*1e3:.3f} ms")
    if triton_time is not None:
        print(f"Triton packed INT4 GEMM:      {triton_time*1e3:.3f} ms")
        print(f"Speedup:                     {ref_time / triton_time:.2f}x")
    else:
        print("Triton benchmark skipped (CUDA not available).")


if __name__ == "__main__":
    main()
