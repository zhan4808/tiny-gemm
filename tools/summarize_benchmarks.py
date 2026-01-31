import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import median


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
                    "ref_ms": float(row["ref_ms"]),
                    "triton_ms": float(row["triton_ms"]),
                    "speedup": float(row["speedup"]),
                    "best_config": row["best_config"],
                    "family": shape_family(int(row["N"]), int(row["K"])),
                }
            )
    return rows


def write_family_summary(rows, out_path: Path) -> None:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["family"], row["M"])].append(row)

    lines = []
    lines.append("| family | M | count | median triton_ms | median speedup |")
    lines.append("|---|---:|---:|---:|---:|")
    for (family, m) in sorted(groups.keys()):
        data = groups[(family, m)]
        triton_median = median([r["triton_ms"] for r in data])
        speedup_median = median([r["speedup"] for r in data])
        lines.append(
            f"| {family} | {m} | {len(data)} | {triton_median:.6f} | {speedup_median:.2f} |"
        )

    out_path.write_text("\n".join(lines) + "\n")


def write_kernel_table(rows, out_path: Path) -> None:
    lines = []
    lines.append(
        "| M | K | N | family | triton_ms | ref_ms | speedup | best_config |"
    )
    lines.append("|---:|---:|---:|---|---:|---:|---:|---|")
    for row in rows:
        lines.append(
            "| {M} | {K} | {N} | {family} | {triton_ms:.6f} | {ref_ms:.6f} | {speedup:.2f} | {best_config} |".format(
                **row
            )
        )
    out_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize benchmark CSV")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(Path(args.csv))
    write_family_summary(rows, out_dir / "family_summary.md")
    write_kernel_table(rows, out_dir / "kernel_time_by_shape.md")


if __name__ == "__main__":
    main()
