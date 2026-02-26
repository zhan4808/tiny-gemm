import argparse
import csv
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


def parse_shape_list(shape_list: str):
    shapes = []
    for raw in shape_list.split(";"):
        raw = raw.strip()
        if not raw:
            continue
        parts = [p.strip() for p in raw.replace("x", ",").split(",") if p.strip()]
        if len(parts) != 3:
            raise ValueError(f"Invalid shape entry: '{raw}' (expected M,K,N)")
        m, k, n = (int(parts[0]), int(parts[1]), int(parts[2]))
        shapes.append((m, k, n))
    return shapes


def main() -> None:
    parser = argparse.ArgumentParser(description="Dequantization breakdown microbenchmark")
    parser.add_argument(
        "--shape_list",
        default="1,4096,1024;1,4096,14336",
        help="Semicolon-separated shapes M,K,N",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rep", type=int, default=20)
    parser.add_argument("--csv", type=str, default="dequant_breakdown.csv")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    shapes = parse_shape_list(args.shape_list)

    with open(args.csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["M", "K", "N", "dequant_ms", "fp16_ms", "int4_ms"])
        for m, k, n in shapes:
            if k % 2 != 0:
                raise ValueError("K must be even for packed INT4")
            a_fp = torch.randn((m, k), device=device, dtype=torch.float16)
            b_fp = torch.randn((k, n), device=device, dtype=torch.float16)

            a_q, a_scale = quantize_per_tensor_int4(a_fp)
            b_q, b_scale = quantize_per_tensor_int4(b_fp)
            a_packed = pack_int4_signed(a_q, axis=1)
            b_packed = pack_int4_signed(b_q, axis=0)

            def dequant_run():
                _ = a_q.float() * a_scale
                _ = b_q.float() * b_scale

            def fp16_run():
                torch.matmul(a_fp, b_fp)

            def int4_run():
                triton_gemm_packed_int4(a_packed, b_packed, k)

            dequant_time = benchmark_op(dequant_run, args.warmup, args.rep)
            fp16_time = benchmark_op(fp16_run, args.warmup, args.rep)
            int4_time = benchmark_op(int4_run, args.warmup, args.rep)

            writer.writerow(
                [
                    m,
                    k,
                    n,
                    f"{dequant_time * 1e3:.6f}",
                    f"{fp16_time * 1e3:.6f}",
                    f"{int4_time * 1e3:.6f}",
                ]
            )

            print(
                f"M={m} K={k} N={n} dequant={dequant_time*1e3:.3f} ms "
                f"fp16={fp16_time*1e3:.3f} ms int4={int4_time*1e3:.3f} ms"
            )


if __name__ == "__main__":
    main()
