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


SMALLMODEL_M_VALUES = [1, 2, 4, 8, 16, 64, 128, 512]
SMALLMODEL_KN_PAIRS = [
    (1024, 2048),
    (1024, 1024),
    (1152, 1024),
    (1152, 256),
    (2048, 2048),
    (2048, 512),
    (2048, 1024),
    (3072, 3072),
    (3072, 768),
    (3072, 1024),
    (4096, 4096),
    (4096, 1024),
    (5120, 5120),
    (5120, 1280),
    (5376, 4096),
    (5376, 2048),
    (1024, 3072),
    (3072, 1024),
    (1152, 6912),
    (6912, 1152),
    (2048, 8192),
    (8192, 2048),
    (2048, 16384),
    (16384, 2048),
    (2048, 11008),
    (11008, 2048),
    (3072, 8192),
    (8192, 3072),
    (4096, 14336),
    (14336, 4096),
    (5120, 14336),
    (14336, 5120),
    (5376, 21504),
    (21504, 5376),
]


def _run_single(device, m, n, k, warmup, rep):
    if k % 2 != 0:
        raise ValueError("K must be even for packed INT4")

    A_fp = torch.randn((m, k), device=device, dtype=torch.float16)
    B_fp = torch.randn((k, n), device=device, dtype=torch.float16)

    A_q, A_scale = quantize_per_tensor_int4(A_fp)
    B_q, B_scale = quantize_per_tensor_int4(B_fp)
    A_packed = pack_int4_signed(A_q, axis=1)
    B_packed = pack_int4_signed(B_q, axis=0)

    def triton_run():
        triton_gemm_packed_int4(A_packed, B_packed, k)

    def ref_run():
        torch.matmul(A_q.float() * A_scale, B_q.float() * B_scale)

    triton_time = (
        benchmark_op(triton_run, warmup=warmup, rep=rep)
        if device.type == "cuda"
        else None
    )
    ref_time = benchmark_op(ref_run, warmup=warmup, rep=rep)
    return ref_time, triton_time


def _print_result(m, n, k, ref_time, triton_time):
    print(f"M={m} N={n} K={k}")
    print(f"  Reference (dequantized) GEMM: {ref_time*1e3:.3f} ms")
    if triton_time is not None:
        print(f"  Triton packed INT4 GEMM:      {triton_time*1e3:.3f} ms")
        print(f"  Speedup:                     {ref_time / triton_time:.2f}x")
    else:
        print("  Triton benchmark skipped (CUDA not available).")


def main():
    parser = argparse.ArgumentParser(description="Benchmark packed INT4 GEMM")
    parser.add_argument("--suite", choices=["single", "smallmodel"], default="single")
    parser.add_argument("--m", type=int, default=512)
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--k", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rep", type=int, default=20)
    parser.add_argument("--max_shapes", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.suite == "single":
        ref_time, triton_time = _run_single(
            device, args.m, args.n, args.k, args.warmup, args.rep
        )
        _print_result(args.m, args.n, args.k, ref_time, triton_time)
        return

    count = 0
    for m in SMALLMODEL_M_VALUES:
        for k, n in SMALLMODEL_KN_PAIRS:
            if args.max_shapes and count >= args.max_shapes:
                return
            ref_time, triton_time = _run_single(
                device, m, n, k, args.warmup, args.rep
            )
            _print_result(m, n, k, ref_time, triton_time)
            count += 1


if __name__ == "__main__":
    main()
