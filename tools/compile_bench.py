import argparse
import time

import torch

import tiny_gemm.ops  # registers ops


def _bench(fn, reps=50):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.time() - start) / reps


def _make_attention_inputs(device):
    q = torch.randn(1, 8, 128, 64, device=device, dtype=torch.float16)
    k = torch.randn(1, 8, 128, 64, device=device, dtype=torch.float16)
    v = torch.randn(1, 8, 128, 64, device=device, dtype=torch.float16)
    return q, k, v


def _make_ffn_inputs(device):
    x = torch.randn(1, 128, 512, device=device, dtype=torch.float16)
    w1 = torch.randn(512, 2048, device=device, dtype=torch.float16)
    b1 = torch.randn(2048, device=device, dtype=torch.float16)
    w2 = torch.randn(2048, 512, device=device, dtype=torch.float16)
    b2 = torch.randn(512, device=device, dtype=torch.float16)
    return x, w1, b1, w2, b2


def main() -> None:
    parser = argparse.ArgumentParser(description="torch.compile experiment")
    parser.add_argument("--mode", choices=["attention", "ffn"], default="attention")
    parser.add_argument("--reps", type=int, default=50)
    parser.add_argument("--backend", type=str, default="inductor")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for compile experiments.")

    device = torch.device("cuda")

    if args.mode == "attention":
        q, k, v = _make_attention_inputs(device)

        def eager():
            return torch.ops.tiny_gemm.fused_attention(q, k, v, True, 0.0)

    else:
        x, w1, b1, w2, b2 = _make_ffn_inputs(device)

        def eager():
            return torch.ops.tiny_gemm.fused_ffn(x, w1, b1, w2, b2, 0)

    compiled = torch.compile(eager, backend=args.backend)
    eager_time = _bench(eager, reps=args.reps)
    compiled_time = _bench(compiled, reps=args.reps)

    print(f"Mode: {args.mode}")
    print(f"Eager:    {eager_time*1e3:.3f} ms")
    print(f"Compile:  {compiled_time*1e3:.3f} ms")
    print(f"Speedup:  {eager_time / compiled_time:.2f}x")


if __name__ == "__main__":
    main()
