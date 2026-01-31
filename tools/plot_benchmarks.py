import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def shape_family(n: int, k: int) -> str:
    if n > k:
        return "ffn_up"
    if n == k:
        return "q_proj"
    if n <= 1280:
        return "kv_proj"
    return "ffn_down"


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
                    "speedup": float(row["speedup"]) if row["speedup"] else np.nan,
                }
            )
    return rows


def plot_speedup_by_family(rows, out_path: Path) -> None:
    families = ["kv_proj", "q_proj", "ffn_down", "ffn_up"]
    m_values = sorted({row["M"] for row in rows})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    axes = axes.flatten()

    for idx, m in enumerate(m_values[:4]):
        ax = axes[idx]
        data = defaultdict(list)
        for row in rows:
            if row["M"] != m:
                continue
            fam = shape_family(row["N"], row["K"])
            data[fam].append(row["speedup"])
        box_data = [data[fam] for fam in families]
        ax.boxplot(box_data, tick_labels=families, showfliers=False)
        ax.set_title(f"M={m}")
        ax.set_ylabel("Speedup (ref / triton)")
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    for idx in range(len(m_values[:4]), len(axes)):
        fig.delaxes(axes[idx])

    fig.suptitle("INT4 GEMM Speedup by Shape Family")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_speedup_heatmap_m1(rows, out_path: Path) -> None:
    m1_rows = [row for row in rows if row["M"] == 1]
    k_vals = sorted({row["K"] for row in m1_rows})
    n_vals = sorted({row["N"] for row in m1_rows})
    grid = np.full((len(k_vals), len(n_vals)), np.nan, dtype=float)

    k_index = {k: i for i, k in enumerate(k_vals)}
    n_index = {n: i for i, n in enumerate(n_vals)}
    for row in m1_rows:
        grid[k_index[row["K"]], n_index[row["N"]]] = row["speedup"]

    fig, ax = plt.subplots(figsize=(12, 7))
    im = ax.imshow(grid, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(n_vals)))
    ax.set_xticklabels(n_vals, rotation=45, ha="right")
    ax.set_yticks(range(len(k_vals)))
    ax.set_yticklabels(k_vals)
    ax.set_xlabel("N")
    ax.set_ylabel("K")
    ax.set_title("M=1 Speedup Heatmap (ref / triton)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Speedup")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Tiny-GEMM benchmark graphs")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(csv_path)
    plot_speedup_by_family(rows, out_dir / "int4_speedup_by_family.png")
    plot_speedup_heatmap_m1(rows, out_dir / "int4_speedup_heatmap_m1.png")


if __name__ == "__main__":
    main()
