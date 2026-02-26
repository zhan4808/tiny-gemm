import argparse
import csv
from pathlib import Path


def load_rows(csv_path: Path):
    rows = []
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def find_row(rows, m, k, n):
    for row in rows:
        if int(row["M"]) == m and int(row["K"]) == k and int(row["N"]) == n:
            return row
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LaTeX summary table")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows = load_rows(Path(args.csv))
    shapes = [
        ("KV proj", 1, 4096, 1024, "Dequant"),
        ("Q proj", 1, 4096, 4096, "Mixed"),
        ("FFN up", 1, 4096, 14336, "Memory"),
        ("FFN down", 1, 14336, 4096, "Memory"),
    ]

    lines = [
        "\\begin{table}[t]",
        "  \\centering",
        "  \\caption{Summary of key decode shapes (M=1).}",
        "  \\label{tab:summary}",
        "  \\begin{tabular}{lccccl}",
        "    \\toprule",
        "    Shape & FP16 (ms) & INT4 (ms) & Speedup & Bound \\\\",
        "    \\midrule",
    ]

    for name, m, k, n, bound in shapes:
        row = find_row(rows, m, k, n)
        if row is None:
            fp16 = "N/A"
            int4 = "N/A"
            speedup = "N/A"
        else:
            fp16 = f"{float(row['fp16_ms']):.3f}"
            int4 = f"{float(row['triton_ms']):.3f}"
            speedup = f"{float(row['speedup_fp16']):.2f}x"
        lines.append(f"    {name} & {fp16} & {int4} & {speedup} & {bound} \\\\")

    lines += [
        "    \\bottomrule",
        "  \\end{tabular}",
        "\\end{table}",
        "",
    ]

    Path(args.out).write_text("\n".join(lines))


if __name__ == "__main__":
    main()
