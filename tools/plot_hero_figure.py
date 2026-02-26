import argparse
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt


def plot_hero(fig_a: Path, fig_b: Path, fig_c: Path, out_path: Path) -> None:
    imgs = [mpimg.imread(p) for p in (fig_a, fig_b, fig_c)]
    titles = ["(A) Speedup vs N", "(B) Utilization Shift", "(C) Dequant Breakdown"]

    fig, axes = plt.subplots(3, 1, figsize=(6, 10))
    for ax, img, title in zip(axes, imgs, titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create hero figure (A/B/C panel)")
    parser.add_argument("--a", required=True)
    parser.add_argument("--b", required=True)
    parser.add_argument("--c", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    plot_hero(Path(args.a), Path(args.b), Path(args.c), Path(args.out))


if __name__ == "__main__":
    main()
