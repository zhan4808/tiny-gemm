import argparse

import torch

from tiny_gemm.quantization.packed_int4 import pack_int4_signed, quantize_per_tensor_int4
from triton_gemm import triton_gemm_packed_int4


def main() -> None:
    parser = argparse.ArgumentParser(description="Run INT4 GEMM for profiling")
    parser.add_argument("--m", type=int, default=1)
    parser.add_argument("--k", type=int, default=3072)
    parser.add_argument("--n", type=int, default=3072)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--use_static", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run GEMM.")

    a = torch.randn((args.m, args.k), device="cuda", dtype=torch.float16)
    b = torch.randn((args.k, args.n), device="cuda", dtype=torch.float16)
    a_q, _ = quantize_per_tensor_int4(a)
    b_q, _ = quantize_per_tensor_int4(b)
    a_p = pack_int4_signed(a_q, axis=1)
    b_p = pack_int4_signed(b_q, axis=0)

    def run():
        triton_gemm_packed_int4(
            a_p, b_p, args.k, use_static_config=args.use_static
        )

    for _ in range(args.warmup):
        run()
    torch.cuda.synchronize()

    torch.cuda.nvtx.range_push("gemm")
    for _ in range(args.iters):
        run()
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
