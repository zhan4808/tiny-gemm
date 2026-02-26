import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


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


def _pick_kernel(rows):
    counts = defaultdict(int)
    for row in rows:
        kernel = row.get("Kernel Name")
        if kernel:
            counts[kernel] += 1
    if not counts:
        return None
    preferred = ["cutlass", "cublas", "gemm"]
    preferred_counts = {
        k: v for k, v in counts.items() if any(p in k.lower() for p in preferred)
    }
    if preferred_counts:
        return max(preferred_counts.items(), key=lambda kv: kv[1])[0]
    return max(counts.items(), key=lambda kv: kv[1])[0]


def _mean_metric(rows, kernel_name, metric_name):
    values = []
    for row in rows:
        if row.get("Kernel Name") != kernel_name:
            continue
        if row.get("Metric Name") != metric_name:
            continue
        try:
            values.append(float(row.get("Metric Value", "")))
        except ValueError:
            continue
    return float(np.mean(values)) if values else np.nan


def load_peak_compute(csv_path: Path, kernel_name: str | None):
    rows = _load_csv_rows(csv_path)
    if not kernel_name or kernel_name == "auto":
        kernel_name = _pick_kernel(rows)
    if kernel_name is None:
        return None, np.nan
    metric = "sm__throughput.avg.pct_of_peak_sustained_elapsed"
    value = _mean_metric(rows, kernel_name, metric)
    return kernel_name, value


def plot_peak_compute(fp16_csv: Path, int4_csv: Path, out_path: Path, fp16_kernel, int4_kernel):
    fp16_kernel, fp16_val = load_peak_compute(fp16_csv, fp16_kernel)
    int4_kernel, int4_val = load_peak_compute(int4_csv, int4_kernel)

    labels = ["FP16", "INT4"]
    values = [fp16_val, int4_val]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.bar(labels, values, color=["#4C78A8", "#54A24B"])
    ax.set_ylabel("% Peak SM Throughput")
    ax.set_title("Peak Compute Utilization (Proxy)")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    print(f"FP16 kernel: {fp16_kernel}")
    print(f"INT4 kernel: {int4_kernel}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot peak compute utilization from NCU CSVs")
    parser.add_argument("--fp16_csv", required=True)
    parser.add_argument("--int4_csv", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--fp16_kernel", default="auto")
    parser.add_argument("--int4_kernel", default="kernel_gemm_packed_int4_static")
    args = parser.parse_args()

    plot_peak_compute(
        Path(args.fp16_csv),
        Path(args.int4_csv),
        Path(args.out),
        args.fp16_kernel,
        args.int4_kernel,
    )


if __name__ == "__main__":
    main()
