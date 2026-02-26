import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_SHAPES = [
    (1, 4096, 4096),
    (1, 4096, 1024),
    (1, 4096, 14336),
    (1, 14336, 4096),
    (8, 4096, 4096),
    (8, 4096, 1024),
    (8, 4096, 14336),
    (8, 14336, 4096),
]

FAMILY_ORDER = ["q_proj", "kv_proj", "ffn_up", "ffn_down"]


def _annotate_bars(ax, bars, fmt="{:.2f}", y_offset=0.02):
    ymax = ax.get_ylim()[1]
    for bar in bars:
        height = bar.get_height()
        if not np.isfinite(height):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + ymax * y_offset,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=8,
        )


def shape_family(n: int, k: int) -> str:
    if n > k:
        return "ffn_up"
    if n == k:
        return "q_proj"
    if n <= 1280:
        return "kv_proj"
    return "ffn_down"


def parse_shape_list(shape_list: str):
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


def load_rows(csv_path: Path):
    rows = []
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fp16_ms = float(row["fp16_ms"]) if row.get("fp16_ms") else np.nan
            dequant_ms = float(row["ref_ms"]) if row.get("ref_ms") else np.nan
            triton_ms = float(row["triton_ms"]) if row.get("triton_ms") else np.nan
            speedup_fp16 = (
                float(row["speedup_fp16"])
                if row.get("speedup_fp16")
                else (fp16_ms / triton_ms if triton_ms else np.nan)
            )
            rows.append(
                {
                    "M": int(row["M"]),
                    "K": int(row["K"]),
                    "N": int(row["N"]),
                    "fp16_ms": fp16_ms,
                    "dequant_ms": dequant_ms,
                    "triton_ms": triton_ms,
                    "speedup_fp16": speedup_fp16,
                }
            )
    return rows


def plot_decode_bars(rows, shapes, out_dir: Path) -> None:
    rows_by_shape = {(r["M"], r["K"], r["N"]): r for r in rows}
    shapes_by_m = defaultdict(list)
    for m, k, n in shapes:
        shapes_by_m[m].append((m, k, n))

    for m, shape_list in shapes_by_m.items():
        shape_list = sorted(shape_list, key=lambda s: FAMILY_ORDER.index(shape_family(s[2], s[1])))
        labels = [shape_family(n, k) for _, k, n in shape_list]
        fp16_vals = []
        dequant_vals = []
        triton_vals = []
        for key in shape_list:
            row = rows_by_shape.get(key)
            if row is None:
                fp16_vals.append(np.nan)
                dequant_vals.append(np.nan)
                triton_vals.append(np.nan)
                continue
            fp16_vals.append(row["fp16_ms"])
            dequant_vals.append(row["dequant_ms"])
            triton_vals.append(row["triton_ms"])

        x = np.arange(len(labels))
        width = 0.26
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(x - width, fp16_vals, width, label="FP16", color="#4C78A8")
        ax.bar(x, dequant_vals, width, label="Dequant FP16", color="#F58518")
        ax.bar(x + width, triton_vals, width, label="Tiny-GEMM INT4", color="#54A24B")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Latency (ms)")
        ax.set_title(f"Decode GEMM Latency (M={m})")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"decode_latency_m{m}.png", dpi=200)
        plt.close(fig)


def plot_family_slopes(rows, m_values, out_dir: Path) -> None:
    for m in m_values:
        family_data = defaultdict(lambda: {"fp16": [], "dequant": [], "triton": []})
        for row in rows:
            if row["M"] != m:
                continue
            fam = shape_family(row["N"], row["K"])
            family_data[fam]["fp16"].append(row["fp16_ms"])
            family_data[fam]["dequant"].append(row["dequant_ms"])
            family_data[fam]["triton"].append(row["triton_ms"])

        fp16_median = [np.median(family_data[f]["fp16"]) for f in FAMILY_ORDER]
        dequant_median = [np.median(family_data[f]["dequant"]) for f in FAMILY_ORDER]
        triton_median = [np.median(family_data[f]["triton"]) for f in FAMILY_ORDER]

        x = np.arange(len(FAMILY_ORDER))
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x, fp16_median, marker="o", label="FP16", color="#4C78A8")
        ax.plot(x, dequant_median, marker="o", label="Dequant FP16", color="#F58518")
        ax.plot(x, triton_median, marker="o", label="Tiny-GEMM INT4", color="#54A24B")
        ax.set_xticks(x)
        ax.set_xticklabels(FAMILY_ORDER)
        ax.set_ylabel("Median latency (ms)")
        ax.set_title(f"Median Latency by Family (M={m})")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"family_latency_m{m}.png", dpi=200)
        plt.close(fig)


def plot_prefill_vs_decode(rows, decode_m_max, prefill_m_min, out_dir: Path) -> None:
    decode_data = defaultdict(list)
    prefill_data = defaultdict(list)
    for row in rows:
        fam = shape_family(row["N"], row["K"])
        if row["M"] <= decode_m_max:
            decode_data[fam].append(row["speedup_fp16"])
        if row["M"] >= prefill_m_min:
            prefill_data[fam].append(row["speedup_fp16"])

    decode_median = [
        np.median(decode_data[fam]) if decode_data[fam] else np.nan
        for fam in FAMILY_ORDER
    ]
    prefill_median = [
        np.median(prefill_data[fam]) if prefill_data[fam] else np.nan
        for fam in FAMILY_ORDER
    ]

    x = np.arange(len(FAMILY_ORDER))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4))
    decode_bars = ax.bar(
        x - width / 2, decode_median, width, label="Decode", color="#54A24B"
    )
    prefill_bars = ax.bar(
        x + width / 2, prefill_median, width, label="Prefill", color="#4C78A8"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(FAMILY_ORDER)
    ax.set_ylabel("Speedup vs FP16 (median)")
    ax.set_title("Prefill vs Decode Speedup by Family")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend()
    _annotate_bars(ax, decode_bars, fmt="{:.2f}x")
    _annotate_bars(ax, prefill_bars, fmt="{:.2f}x")
    fig.tight_layout()
    fig.savefig(out_dir / "prefill_vs_decode_speedup.png", dpi=200)
    plt.close(fig)


def plot_speedup_vs_n(rows, k_fixed, out_dir: Path) -> None:
    by_m = defaultdict(list)
    for row in rows:
        if row["K"] != k_fixed:
            continue
        by_m[row["M"]].append(row)

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(by_m)))
    for m, m_rows in sorted(by_m.items()):
        m_rows = sorted(m_rows, key=lambda r: r["N"])
        n_vals = [r["N"] for r in m_rows]
        speedups = [r["speedup_fp16"] for r in m_rows]
        color = colors[list(sorted(by_m.keys())).index(m)]
        ax.plot(n_vals, speedups, marker="o", label=f"M={m}", color=color)

    ax.set_xlabel("N")
    ax.set_ylabel("Speedup vs FP16")
    ax.set_title(f"Speedup vs Output Width (K={k_fixed})")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"speedup_vs_n_k{k_fixed}.png", dpi=200)
    plt.close(fig)


def plot_batch_scaling(rows, k_fixed, n_fixed, out_dir: Path) -> None:
    data = [r for r in rows if r["K"] == k_fixed and r["N"] == n_fixed]
    if not data:
        return
    data = sorted(data, key=lambda r: r["M"])
    m_vals = [r["M"] for r in data]
    fp16 = [r["fp16_ms"] for r in data]
    triton = [r["triton_ms"] for r in data]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(m_vals, fp16, marker="o", label="FP16", color="#4C78A8")
    ax.plot(m_vals, triton, marker="o", label="Tiny-GEMM INT4", color="#54A24B")
    ax.set_xlabel("M (batch size)")
    ax.set_ylabel("Latency (ms)")
    ax.set_title(f"Batch Scaling (K={k_fixed}, N={n_fixed})")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    if m_vals[-1] / max(m_vals[0], 1) >= 32:
        ax.set_xscale("log", base=2)
        ax.set_xticks(m_vals)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"batch_scaling_k{k_fixed}_n{n_fixed}.png", dpi=200)
    plt.close(fig)


def plot_tokens_per_sec(rows, k_fixed, n_fixed, out_dir: Path) -> None:
    data = [r for r in rows if r["K"] == k_fixed and r["N"] == n_fixed]
    if not data:
        return
    data = sorted(data, key=lambda r: r["M"])
    m_vals = np.array([r["M"] for r in data], dtype=float)
    fp16 = np.array([r["fp16_ms"] for r in data], dtype=float)
    triton = np.array([r["triton_ms"] for r in data], dtype=float)

    fp16_tps = (m_vals / fp16) * 1e3
    triton_tps = (m_vals / triton) * 1e3

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(m_vals, fp16_tps, marker="o", label="FP16", color="#4C78A8")
    ax.plot(m_vals, triton_tps, marker="o", label="Tiny-GEMM INT4", color="#54A24B")
    ax.set_xlabel("M (batch size)")
    ax.set_ylabel("Tokens/sec (batch / latency)")
    ax.set_title(f"Decode Throughput (K={k_fixed}, N={n_fixed})")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    if m_vals[-1] / max(m_vals[0], 1) >= 32:
        ax.set_xscale("log", base=2)
        ax.set_xticks(m_vals)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"tokens_per_sec_k{k_fixed}_n{n_fixed}.png", dpi=200)
    plt.close(fig)


def _estimate_bytes_fp16(m, k, n):
    a_bytes = m * k * 2
    b_bytes = k * n * 2
    c_bytes = m * n * 2
    return a_bytes + b_bytes + c_bytes


def _estimate_bytes_int4(m, k, n):
    a_bytes = m * k * 0.5
    b_bytes = k * n * 0.5
    c_bytes = m * n * 4
    return a_bytes + b_bytes + c_bytes


def plot_memory_traffic(rows, out_dir: Path) -> None:
    fp16_points = []
    int4_points = []
    for row in rows:
        m, k, n = row["M"], row["K"], row["N"]
        if not np.isfinite(row["fp16_ms"]) or not np.isfinite(row["triton_ms"]):
            continue
        fp16_points.append(
            (_estimate_bytes_fp16(m, k, n) / 1e6, row["fp16_ms"])
        )
        int4_points.append(
            (_estimate_bytes_int4(m, k, n) / 1e6, row["triton_ms"])
        )

    fig, ax = plt.subplots(figsize=(8, 4))
    if fp16_points:
        fp16_x, fp16_y = zip(*fp16_points)
        ax.scatter(fp16_x, fp16_y, label="FP16", color="#4C78A8")
    if int4_points:
        int4_x, int4_y = zip(*int4_points)
        ax.scatter(int4_x, int4_y, label="Tiny-GEMM INT4", color="#54A24B")
    ax.set_xlabel("Estimated bytes moved (MB)")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Latency vs Estimated Memory Traffic")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "memory_traffic_scatter.png", dpi=200)
    plt.close(fig)


def plot_simple_roofline(rows, out_dir: Path) -> None:
    fp16_points = []
    int4_points = []
    for row in rows:
        m, k, n = row["M"], row["K"], row["N"]
        flops = 2.0 * m * k * n
        if np.isfinite(row["fp16_ms"]) and row["fp16_ms"] > 0:
            ai_fp16 = flops / _estimate_bytes_fp16(m, k, n)
            tflops_fp16 = flops / (row["fp16_ms"] * 1e-3) / 1e12
            fp16_points.append((ai_fp16, tflops_fp16))
        if np.isfinite(row["triton_ms"]) and row["triton_ms"] > 0:
            ai_int4 = flops / _estimate_bytes_int4(m, k, n)
            tflops_int4 = flops / (row["triton_ms"] * 1e-3) / 1e12
            int4_points.append((ai_int4, tflops_int4))

    fig, ax = plt.subplots(figsize=(8, 4))
    if fp16_points:
        x, y = zip(*fp16_points)
        ax.scatter(x, y, label="FP16", color="#4C78A8")
    if int4_points:
        x, y = zip(*int4_points)
        ax.scatter(x, y, label="Tiny-GEMM INT4", color="#54A24B")
    ax.set_xlabel("Estimated arithmetic intensity (FLOPs/byte)")
    ax.set_ylabel("Achieved TFLOPs/s")
    ax.set_title("Estimated Roofline Scatter")
    ax.grid(axis="both", linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "roofline_scatter.png", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot decode-focused Tiny-GEMM results")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--shape_list", default="")
    parser.add_argument("--m_values", default="1,8")
    parser.add_argument("--n_scale_k", type=int, default=4096)
    parser.add_argument("--batch_k", type=int, default=4096)
    parser.add_argument("--batch_n", type=int, default=4096)
    parser.add_argument("--decode_m_max", type=int, default=8)
    parser.add_argument("--prefill_m_min", type=int, default=256)
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    shapes = parse_shape_list(args.shape_list) or DEFAULT_SHAPES
    m_values = [int(v.strip()) for v in args.m_values.split(",") if v.strip()]

    rows = load_rows(csv_path)
    plot_decode_bars(rows, shapes, out_dir)
    plot_family_slopes(rows, m_values, out_dir)
    plot_speedup_vs_n(rows, args.n_scale_k, out_dir)
    plot_batch_scaling(rows, args.batch_k, args.batch_n, out_dir)
    plot_tokens_per_sec(rows, args.batch_k, args.batch_n, out_dir)
    plot_memory_traffic(rows, out_dir)
    plot_simple_roofline(rows, out_dir)
    plot_prefill_vs_decode(rows, args.decode_m_max, args.prefill_m_min, out_dir)


if __name__ == "__main__":
    main()
