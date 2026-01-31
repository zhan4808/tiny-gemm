import argparse
from pathlib import Path

import matplotlib.pyplot as plt


def read_markdown_table(path: Path):
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    lines = [line for line in lines if line.startswith("|")]
    if len(lines) < 2:
        raise ValueError(f"No table found in {path}")
    header = [cell.strip() for cell in lines[0].strip("|").split("|")]
    rows = []
    for line in lines[2:]:
        rows.append([cell.strip() for cell in line.strip("|").split("|")])
    return header, rows


def render_table(header, rows, out_path: Path, title: str):
    fig, ax = plt.subplots(figsize=(12, 0.4 * (len(rows) + 2)))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=header,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.2)
    ax.set_title(title, pad=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render markdown tables to PNG")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--title", default="")
    args = parser.parse_args()

    header, rows = read_markdown_table(Path(args.input))
    title = args.title or Path(args.input).stem.replace("_", " ").title()
    render_table(header, rows, Path(args.output), title)


if __name__ == "__main__":
    main()
