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
    ax.bar(x - width / 2, fp16, width, label="FP16", color="#4C78A8")
    ax.bar(x + width / 2, dequant_share, width, label="INT4 dequant (est)", color="#F58518")
    ax.bar(
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
