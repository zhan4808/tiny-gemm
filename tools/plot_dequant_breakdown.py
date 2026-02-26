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
            rows.append(
                {
                    "M": int(row["M"]),
                    "K": int(row["K"]),
                    "N": int(row["N"]),
                    "dequant_ms": float(row["dequant_ms"]),
                    "fp16_ms": float(row["fp16_ms"]),
                    "int4_ms": float(row["int4_ms"]),
                }
            )
    return rows


def plot_breakdown(rows, out_path: Path) -> None:
    labels = [f"M={r['M']} N={r['N']}" for r in rows]
    fp16 = [r["fp16_ms"] for r in rows]
    int4 = [r["int4_ms"] for r in rows]
    dequant = [r["dequant_ms"] for r in rows]

    dequant_share = [min(d, t) for d, t in zip(dequant, int4)]
    other_share = [max(t - d, 0.0) for d, t in zip(dequant, int4)]

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4))
    fp16_bars = ax.bar(x - width / 2, fp16, width, label="FP16", color="#4C78A8")
    dequant_bars = ax.bar(
        x + width / 2,
        dequant_share,
        width,
        label="INT4 dequant (est)",
        color="#F58518",
    )
    other_bars = ax.bar(
        x + width / 2,
        other_share,
        width,
        bottom=dequant_share,
        label="INT4 other (est)",
        color="#54A24B",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Latency (ms)")
    ax.set_title("INT4 Dequantization Cost (Estimated)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend()
    for bar in fp16_bars:
        height = bar.get_height()
        if np.isfinite(height):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.02 * max(fp16 + int4),
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    for bar, total in zip(other_bars, int4):
        if np.isfinite(total):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                total + 0.02 * max(fp16 + int4),
                f"{total:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    for bar, dq in zip(dequant_bars, dequant_share):
        if np.isfinite(dq):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                dq / 2,
                f"{dq:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="black",
            )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot dequant breakdown bars")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows = load_rows(Path(args.csv))
    plot_breakdown(rows, Path(args.out))


if __name__ == "__main__":
    main()
