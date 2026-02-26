import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_rows(md_path: Path):
    rows = []
    lines = md_path.read_text().splitlines()
    for line in lines:
        if not line.startswith("|"):
            continue
        parts = [p.strip() for p in line.strip().strip("|").split("|")]
        if parts[0] == "M" or parts[0].startswith("---"):
            continue
        if len(parts) < 6:
            continue
        m, k, n = int(parts[0]), int(parts[1]), int(parts[2])
        family = parts[3]
        total_ms = float(parts[5])
        rows.append({"M": m, "K": k, "N": n, "family": family, "total_ms": total_ms})
    return rows


def plot_top_kernels(rows, out_path: Path, top_k=10) -> None:
    rows = sorted(rows, key=lambda r: r["total_ms"], reverse=True)[:top_k]
    labels = [f"{r['M']}x{r['K']}x{r['N']}" for r in rows]
    values = [r["total_ms"] for r in rows]
    families = [r["family"] for r in rows]

    palette = {
        "ffn_up": "#54A24B",
        "ffn_down": "#4C78A8",
        "q_proj": "#F58518",
        "kv_proj": "#E45756",
    }
    colors = [palette.get(fam, "#9D9D9D") for fam in families]

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(range(len(values)), values, color=colors)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Total CUDA time (ms)")
    ax.set_title(f"Top {top_k} Shapes by CUDA Time")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.01 * max(values),
            f"{height:.1f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=palette[k], label=k)
        for k in palette
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot top kernels by CUDA time")
    parser.add_argument("--md", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--top_k", type=int, default=10)
    args = parser.parse_args()

    rows = load_rows(Path(args.md))
    plot_top_kernels(rows, Path(args.out), top_k=args.top_k)


if __name__ == "__main__":
    main()
