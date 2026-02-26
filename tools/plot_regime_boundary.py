import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_rows(csv_path: Path):
    rows = []
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fp16_ms = float(row["fp16_ms"]) if row.get("fp16_ms") else np.nan
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
                    "speedup_fp16": speedup_fp16,
                }
            )
    return rows


def estimate_bytes_fp16(m, k, n):
    a_bytes = m * k * 2
    b_bytes = k * n * 2
    c_bytes = m * n * 2
    return a_bytes + b_bytes + c_bytes


def plot_boundary(rows, out_path: Path) -> None:
    xs = []
    ys = []
    colors = []
    for row in rows:
        m, k, n = row["M"], row["K"], row["N"]
        flops = 2.0 * m * k * n
        ai = flops / estimate_bytes_fp16(m, k, n)
        speedup = row["speedup_fp16"]
        if not np.isfinite(ai) or not np.isfinite(speedup):
            continue
        xs.append(ai)
        ys.append(speedup)
        colors.append("#54A24B" if speedup >= 1.0 else "#E45756")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(xs, ys, c=colors, alpha=0.8, edgecolor="black", linewidth=0.3)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel("Arithmetic Intensity (FLOPs/byte, FP16 bytes)")
    ax.set_ylabel("Speedup vs FP16")
    ax.set_title("INT4 Regime Boundary")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot INT4 regime boundary")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows = load_rows(Path(args.csv))
    plot_boundary(rows, Path(args.out))


if __name__ == "__main__":
    main()
