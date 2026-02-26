import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow, FancyBboxPatch


def _box(ax, xy, text, color):
    x, y = xy
    box = FancyBboxPatch(
        (x, y),
        0.22,
        0.18,
        boxstyle="round,pad=0.02",
        linewidth=1,
        edgecolor="black",
        facecolor=color,
    )
    ax.add_patch(box)
    ax.text(x + 0.11, y + 0.09, text, ha="center", va="center", fontsize=9)


def plot_diagram(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 1.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    _box(ax, (0.02, 0.41), "Activations\nFP16", "#72B7B2")
    _box(ax, (0.26, 0.41), "Packed INT4\nWeights", "#F58518")
    _box(ax, (0.50, 0.41), "Shared Mem\nUnpack", "#E45756")
    _box(ax, (0.74, 0.41), "FP32\nAccumulate", "#4C78A8")

    ax.add_patch(FancyArrow(0.20, 0.5, 0.05, 0, width=0.012, length_includes_head=True))
    ax.add_patch(FancyArrow(0.44, 0.5, 0.05, 0, width=0.012, length_includes_head=True))
    ax.add_patch(FancyArrow(0.68, 0.5, 0.05, 0, width=0.012, length_includes_head=True))
    ax.add_patch(FancyArrow(0.92, 0.5, 0.05, 0, width=0.012, length_includes_head=True))
    ax.text(0.96, 0.5, "Output", ha="left", va="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Tiny-GEMM architecture diagram")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    plot_diagram(Path(args.out))


if __name__ == "__main__":
    main()
