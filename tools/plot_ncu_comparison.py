import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BASE_METRICS = {
    "dram__throughput.avg.pct_of_peak_sustained_elapsed": "DRAM BW (% peak)",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed": "SM Throughput (% peak)",
    "sm__warps_active.avg.pct_of_peak_sustained_elapsed": "SM Warps Active (% peak)",
}

TENSOR_METRICS = {
    "smsp__inst_executed_pipe_tensor.sum": "Tensor Insts",
    "smsp__inst_executed.sum": "All Insts",
}


def _load_csv_rows(csv_path: Path):
    with csv_path.open() as f:
        lines = f.read().splitlines()
    header_idx = None
    for idx, line in enumerate(lines):
        if line.startswith('"ID"'):
            header_idx = idx
            break
    if header_idx is None:
        return []
    reader = csv.DictReader(lines[header_idx:])
    return list(reader)


def _pick_kernel(rows, metrics):
    counts = defaultdict(int)
    for row in rows:
        metric = row.get("Metric Name")
        if metric not in metrics:
            continue
        kernel = row.get("Kernel Name")
        if kernel:
            counts[kernel] += 1
    if not counts:
        return None
    preferred = ["cutlass", "cublas", "gemm"]
    preferred_counts = {
        k: v
        for k, v in counts.items()
        if any(token in k.lower() for token in preferred)
    }
    if preferred_counts:
        return max(preferred_counts.items(), key=lambda kv: kv[1])[0]
    return max(counts.items(), key=lambda kv: kv[1])[0]


def _mean_metric(rows, kernel_name, metric):
    values = []
    for row in rows:
        if row.get("Kernel Name") != kernel_name:
            continue
        if row.get("Metric Name") != metric:
            continue
        try:
            values.append(float(row.get("Metric Value", "")))
        except ValueError:
            continue
    return float(np.mean(values)) if values else np.nan


def load_utilization(csv_path: Path, kernel_name: str | None):
    rows = _load_csv_rows(csv_path)
    if kernel_name == "auto" or kernel_name is None:
        kernel_name = _pick_kernel(rows, set(BASE_METRICS) | set(TENSOR_METRICS))
    metrics = {}
    for metric in BASE_METRICS:
        metrics[metric] = _mean_metric(rows, kernel_name, metric)

    tensor_inst = _mean_metric(rows, kernel_name, "smsp__inst_executed_pipe_tensor.sum")
    all_inst = _mean_metric(rows, kernel_name, "smsp__inst_executed.sum")
    if np.isfinite(tensor_inst) and np.isfinite(all_inst) and all_inst > 0:
        metrics["tensor_inst_ratio"] = (tensor_inst / all_inst) * 100.0
    else:
        metrics["tensor_inst_ratio"] = np.nan

    return kernel_name, metrics


def plot_comparison(fp16_csv: Path, int4_csv: Path, out_path: Path, kernel_fp16, kernel_int4):
    kernel_fp16, fp16_metrics = load_utilization(fp16_csv, kernel_fp16)
    kernel_int4, int4_metrics = load_utilization(int4_csv, kernel_int4)

    labels = list(BASE_METRICS.values()) + ["Tensor Insts (% of total)"]
    fp16_vals = [fp16_metrics[m] for m in BASE_METRICS] + [fp16_metrics["tensor_inst_ratio"]]
    int4_vals = [int4_metrics[m] for m in BASE_METRICS] + [int4_metrics["tensor_inst_ratio"]]

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - width / 2, fp16_vals, width, label="FP16", color="#4C78A8")
    ax.bar(x + width / 2, int4_vals, width, label="INT4", color="#54A24B")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Percent / Ratio")
    ax.set_title("Hardware Utilization Comparison")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    print(f"FP16 kernel: {kernel_fp16}")
    print(f"INT4 kernel: {kernel_int4}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare NCU utilization FP16 vs INT4")
    parser.add_argument("--fp16_csv", required=True)
    parser.add_argument("--int4_csv", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--fp16_kernel", default="auto")
    parser.add_argument("--int4_kernel", default="kernel_gemm_packed_int4_static")
    args = parser.parse_args()

    plot_comparison(
        Path(args.fp16_csv),
        Path(args.int4_csv),
        Path(args.out),
        args.fp16_kernel,
        args.int4_kernel,
    )


if __name__ == "__main__":
    main()
