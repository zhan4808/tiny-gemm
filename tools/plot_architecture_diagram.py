import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow, FancyBboxPatch


def _box(ax, xy, text, color):
    x, y = xy
    box = FancyBboxPatch(
        (x, y),
        0.26,
        0.16,
        boxstyle="round,pad=0.02",
        linewidth=1,
        edgecolor="black",
        facecolor=color,
    )
    ax.add_patch(box)
    ax.text(x + 0.13, y + 0.08, text, ha="center", va="center", fontsize=9)


def plot_diagram(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 2.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    _box(ax, (0.02, 0.42), "Packed INT4\nWeights", "#F58518")
    _box(ax, (0.34, 0.42), "Shared Mem\nUnpack", "#E45756")
    _box(ax, (0.66, 0.42), "FP32\nAccumulate", "#4C78A8")
    _box(ax, (0.86, 0.42), "Output", "#54A24B")

    ax.add_patch(FancyArrow(0.28, 0.5, 0.06, 0, width=0.01, length_includes_head=True))
    ax.add_patch(FancyArrow(0.60, 0.5, 0.06, 0, width=0.01, length_includes_head=True))
    ax.add_patch(FancyArrow(0.84, 0.5, 0.02, 0, width=0.01, length_includes_head=True))

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
