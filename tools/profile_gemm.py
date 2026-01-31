import argparse
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

from tiny_gemm.quantization.packed_int4 import pack_int4_signed, quantize_per_tensor_int4
from triton_gemm import triton_gemm_packed_int4


def _make_inputs(m: int, k: int, n: int):
    a = torch.randn((m, k), device="cuda", dtype=torch.float16)
    b = torch.randn((k, n), device="cuda", dtype=torch.float16)
    a_q, _ = quantize_per_tensor_int4(a)
    b_q, _ = quantize_per_tensor_int4(b)
    a_packed = pack_int4_signed(a_q, axis=1)
    b_packed = pack_int4_signed(b_q, axis=0)
    return a_packed, b_packed


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile INT4 GEMM with torch.profiler")
    parser.add_argument("--m", type=int, default=1)
    parser.add_argument("--k", type=int, default=3072)
    parser.add_argument("--n", type=int, default=3072)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--active", type=int, default=20)
    parser.add_argument("--use_static", action="store_true")
    parser.add_argument("--out_dir", type=str, default="profiles")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for profiling.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    a_packed, b_packed = _make_inputs(args.m, args.k, args.n)

    def run():
        triton_gemm_packed_int4(
            a_packed, b_packed, args.k, use_static_config=args.use_static
        )

    for _ in range(args.warmup):
        run()
    torch.cuda.synchronize()

    trace_path = out_dir / f"profile_m{args.m}_k{args.k}_n{args.n}.json"
    table_path = out_dir / f"profile_m{args.m}_k{args.k}_n{args.n}.txt"

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(args.active):
            run()
            torch.cuda.synchronize()
    prof.export_chrome_trace(str(trace_path))
    table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=15)
    table_path.write_text(table)
    print(table)
    print(f"Saved trace: {trace_path}")
    print(f"Saved table: {table_path}")


if __name__ == "__main__":
    main()
