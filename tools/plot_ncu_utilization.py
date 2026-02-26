import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METRIC_LABELS = {
    "dram__throughput.avg.pct_of_peak_sustained_elapsed": "DRAM BW (% peak)",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed": "SM Throughput (% peak)",
    "sm__warps_active.avg.pct_of_peak_sustained_elapsed": "SM Warps Active (% peak)",
}


def load_metrics(csv_path: Path, kernel_name: str):
    by_kernel = defaultdict(lambda: defaultdict(list))
    with csv_path.open() as f:
        lines = f.read().splitlines()
    header_idx = None
    for idx, line in enumerate(lines):
        if line.startswith('"ID"'):
            header_idx = idx
            break
    if header_idx is None:
        return by_kernel
    reader = csv.DictReader(lines[header_idx:])
    for row in reader:
        if row.get("Kernel Name") != kernel_name:
            continue
        metric = row.get("Metric Name")
        if metric not in METRIC_LABELS:
            continue
        try:
            value = float(row.get("Metric Value", ""))
        except ValueError:
            continue
        by_kernel[kernel_name][metric].append(value)
    return by_kernel


def plot_utilization(csv_path: Path, kernel_name: str, out_path: Path) -> None:
    data = load_metrics(csv_path, kernel_name)
    metrics = METRIC_LABELS.keys()
    values = []
    for metric in metrics:
        samples = data.get(kernel_name, {}).get(metric, [])
        values.append(float(np.mean(samples)) if samples else np.nan)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.bar(range(len(values)), values, color="#4C78A8")
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels([METRIC_LABELS[m] for m in metrics], rotation=20, ha="right")
    ax.set_ylabel("Percent of peak")
    ax.set_title(f"Hardware Utilization ({kernel_name})")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Nsight Compute utilization bars")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--kernel", default="kernel_gemm_packed_int4_static")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    plot_utilization(Path(args.csv), args.kernel, Path(args.out))


if __name__ == "__main__":
    main()
