import argparse
import csv
import re
import time

import torch

from tiny_gemm.quantization.packed_int4 import pack_int4_signed, quantize_per_tensor_int4
from triton_gemm import get_best_config, triton_gemm_packed_int4


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


def _format_config(config):
    if config is None:
        return ""
    parts = []
    for key in sorted(config.kwargs.keys()):
        parts.append(f"{key}={config.kwargs[key]}")
    parts.append(f"num_warps={config.num_warps}")
    parts.append(f"num_stages={config.num_stages}")
    return ",".join(parts)


def _run_single(device, m, n, k, warmup, rep, use_static_config, check_correctness):
    if k % 2 != 0:
        raise ValueError("K must be even for packed INT4")

    A_fp = torch.randn((m, k), device=device, dtype=torch.float16)
    B_fp = torch.randn((k, n), device=device, dtype=torch.float16)

    A_q, A_scale = quantize_per_tensor_int4(A_fp)
    B_q, B_scale = quantize_per_tensor_int4(B_fp)
    A_packed = pack_int4_signed(A_q, axis=1)
    B_packed = pack_int4_signed(B_q, axis=0)

    def triton_run():
        triton_gemm_packed_int4(
            A_packed, B_packed, k, use_static_config=use_static_config
        )

    def fp16_run():
        torch.matmul(A_fp, B_fp)

    def dequant_run():
        torch.matmul(A_q.float() * A_scale, B_q.float() * B_scale)

    triton_time = (
        benchmark_op(triton_run, warmup=warmup, rep=rep)
        if device.type == "cuda"
        else None
    )
    fp16_time = benchmark_op(fp16_run, warmup=warmup, rep=rep)
    dequant_time = benchmark_op(dequant_run, warmup=warmup, rep=rep)
    best_config = None
    if device.type == "cuda":
        best_config = get_best_config(
            m,
            n,
            k,
            str(A_packed.dtype),
            str(B_packed.dtype),
            "torch.float32",
        )
    max_abs_diff_fp16 = None
    max_abs_diff_dequant = None
    if check_correctness and device.type == "cuda":
        with torch.no_grad():
            c_fp16 = torch.matmul(A_fp, B_fp).float()
            c_dequant = torch.matmul(A_q.float() * A_scale, B_q.float() * B_scale)
            c_triton = triton_gemm_packed_int4(
                A_packed, B_packed, k, use_static_config=use_static_config
            )
            max_abs_diff_fp16 = (c_triton - c_fp16).abs().max().item()
            max_abs_diff_dequant = (c_triton - c_dequant).abs().max().item()
    return (
        fp16_time,
        dequant_time,
        triton_time,
        best_config,
        max_abs_diff_fp16,
        max_abs_diff_dequant,
    )


def _print_result(m, n, k, fp16_time, dequant_time, triton_time):
    print(f"M={m} N={n} K={k}")
    print(f"  FP16 GEMM:                  {fp16_time*1e3:.3f} ms")
    print(f"  Dequantized FP16 GEMM:      {dequant_time*1e3:.3f} ms")
    if triton_time is not None:
        print(f"  Triton packed INT4 GEMM:      {triton_time*1e3:.3f} ms")
        print(f"  Speedup vs FP16:              {fp16_time / triton_time:.2f}x")
        print(f"  Speedup vs dequantized:       {dequant_time / triton_time:.2f}x")
    else:
        print("  Triton benchmark skipped (CUDA not available).")


def _parse_m_values(m_values: str):
    if not m_values:
        return SMALLMODEL_M_VALUES
    return [int(v.strip()) for v in m_values.split(",") if v.strip()]


def _parse_shape_list(shape_list: str):
    shapes = []
    if not shape_list:
        return shapes
    for raw in shape_list.split(";"):
        raw = raw.strip()
        if not raw:
            continue
        parts = [p.strip() for p in re.split(r"[x,]", raw) if p.strip()]
        if len(parts) != 3:
            raise ValueError(f"Invalid shape entry: '{raw}' (expected M,K,N)")
        m, k, n = (int(parts[0]), int(parts[1]), int(parts[2]))
        shapes.append((m, k, n))
    return shapes


def main():
    parser = argparse.ArgumentParser(description="Benchmark packed INT4 GEMM")
    parser.add_argument("--suite", choices=["single", "smallmodel"], default="single")
    parser.add_argument("--m", type=int, default=512)
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--k", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rep", type=int, default=20)
    parser.add_argument("--max_shapes", type=int, default=0)
    parser.add_argument("--csv", type=str, default="")
    parser.add_argument("--group_size", type=str, default="per_tensor")
    parser.add_argument("--m_values", type=str, default="")
    parser.add_argument("--shape_list", type=str, default="")
    parser.add_argument("--disable_static", action="store_true")
    parser.add_argument("--skip_correctness", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    csv_writer = None
    csv_file = None
    if args.csv:
        csv_file = open(args.csv, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(
            [
                "M",
                "K",
                "N",
                "group_size",
                "ref_ms",
                "fp16_ms",
                "triton_ms",
                "speedup",
                "speedup_fp16",
                "max_abs_diff_fp16",
                "max_abs_diff_dequant",
                "best_config",
            ]
        )
    use_static_config = not args.disable_static
    check_correctness = not args.skip_correctness
    shapes = _parse_shape_list(args.shape_list)
    if shapes:
        for m, k, n in shapes:
            (
                fp16_time,
                dequant_time,
                triton_time,
                best_config,
                max_abs_diff_fp16,
                max_abs_diff_dequant,
            ) = _run_single(
                device, m, n, k, args.warmup, args.rep, use_static_config, check_correctness
            )
            _print_result(m, n, k, fp16_time, dequant_time, triton_time)
            if csv_writer is not None:
                speedup = dequant_time / triton_time if triton_time else ""
                speedup_fp16 = fp16_time / triton_time if triton_time else ""
                csv_writer.writerow(
                    [
                        m,
                        k,
                        n,
                        args.group_size,
                        f"{dequant_time*1e3:.6f}",
                        f"{fp16_time*1e3:.6f}",
                        f"{triton_time*1e3:.6f}" if triton_time else "",
                        f"{speedup:.6f}" if triton_time else "",
                        f"{speedup_fp16:.6f}" if triton_time else "",
                        f"{max_abs_diff_fp16:.6f}"
                        if max_abs_diff_fp16 is not None
                        else "",
                        f"{max_abs_diff_dequant:.6f}"
                        if max_abs_diff_dequant is not None
                        else "",
                        _format_config(best_config),
                    ]
                )
        if csv_file is not None:
            csv_file.close()
        return

    if args.suite == "single":
        (
            fp16_time,
            dequant_time,
            triton_time,
            best_config,
            max_abs_diff_fp16,
            max_abs_diff_dequant,
        ) = _run_single(
            device,
            args.m,
            args.n,
            args.k,
            args.warmup,
            args.rep,
            use_static_config,
            check_correctness,
        )
        _print_result(args.m, args.n, args.k, fp16_time, dequant_time, triton_time)
        if csv_writer is not None:
            speedup = dequant_time / triton_time if triton_time else ""
            speedup_fp16 = fp16_time / triton_time if triton_time else ""
            csv_writer.writerow(
                [
                    args.m,
                    args.k,
                    args.n,
                    args.group_size,
                    f"{dequant_time*1e3:.6f}",
                    f"{fp16_time*1e3:.6f}",
                    f"{triton_time*1e3:.6f}" if triton_time else "",
                    f"{speedup:.6f}" if triton_time else "",
                    f"{speedup_fp16:.6f}" if triton_time else "",
                    f"{max_abs_diff_fp16:.6f}"
                    if max_abs_diff_fp16 is not None
                    else "",
                    f"{max_abs_diff_dequant:.6f}"
                    if max_abs_diff_dequant is not None
                    else "",
                    _format_config(best_config),
                ]
            )
            csv_file.close()
        return

    count = 0
    m_values = _parse_m_values(args.m_values)
    for m in m_values:
        for k, n in SMALLMODEL_KN_PAIRS:
            if args.max_shapes and count >= args.max_shapes:
                if csv_file is not None:
                    csv_file.close()
                return
            (
                fp16_time,
                dequant_time,
                triton_time,
                best_config,
                max_abs_diff_fp16,
                max_abs_diff_dequant,
            ) = _run_single(
                device, m, n, k, args.warmup, args.rep, use_static_config, check_correctness
            )
            _print_result(m, n, k, fp16_time, dequant_time, triton_time)
            if csv_writer is not None:
                speedup = dequant_time / triton_time if triton_time else ""
                speedup_fp16 = fp16_time / triton_time if triton_time else ""
                csv_writer.writerow(
                    [
                        m,
                        k,
                        n,
                        args.group_size,
                        f"{dequant_time*1e3:.6f}",
                        f"{fp16_time*1e3:.6f}",
                        f"{triton_time*1e3:.6f}" if triton_time else "",
                        f"{speedup:.6f}" if triton_time else "",
                        f"{speedup_fp16:.6f}" if triton_time else "",
                        f"{max_abs_diff_fp16:.6f}"
                        if max_abs_diff_fp16 is not None
                        else "",
                        f"{max_abs_diff_dequant:.6f}"
                        if max_abs_diff_dequant is not None
                        else "",
                        _format_config(best_config),
                    ]
                )
            count += 1
    if csv_file is not None:
        csv_file.close()


if __name__ == "__main__":
    main()
