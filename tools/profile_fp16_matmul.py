import argparse
import time

import torch


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


def main() -> None:
    parser = argparse.ArgumentParser(description="FP16 matmul microbenchmark")
    parser.add_argument("--m", type=int, default=1)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--n", type=int, default=14336)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rep", type=int, default=20)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a = torch.randn((args.m, args.k), device=device, dtype=torch.float16)
    b = torch.randn((args.k, args.n), device=device, dtype=torch.float16)

    def fp16_run():
        torch.matmul(a, b)

    avg = benchmark_op(fp16_run, warmup=args.warmup, rep=args.rep)
    print(f"FP16 matmul: {avg * 1e3:.3f} ms (M={args.m}, K={args.k}, N={args.n})")


if __name__ == "__main__":
    main()
