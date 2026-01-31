import argparse
import csv
import subprocess
from pathlib import Path


def shape_family(n: int, k: int) -> str:
    if n > k:
        return "ffn_up"
    if n == k:
        return "q_proj"
    if n <= 1280:
        return "kv_proj"
    return "ffn_down"


def parse_cuda_kern_sum(csv_text: str):
    lines = [line for line in csv_text.splitlines() if line.strip()]
    header_idx = None
    for idx, line in enumerate(lines):
        if line.startswith("Time (%)"):
            header_idx = idx
            break
    if header_idx is None or header_idx + 1 >= len(lines):
        return []
    header = lines[header_idx].split(",")
    rows = []
    for line in lines[header_idx + 1 :]:
        parts = list(csv.reader([line]))[0]
        row = dict(zip(header, parts))
        rows.append(row)
    return rows


def extract_kernel_time(report_path: Path):
    cmd = [
        "nsys",
        "stats",
        "--force-export=true",
        "--report",
        "cuda_gpu_kern_sum",
        "--format",
        "csv",
        str(report_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 and not result.stdout:
        return None
    rows = parse_cuda_kern_sum(result.stdout)
    if not rows:
        return None
    rows = [r for r in rows if r.get("Name")]
    if not rows:
        return None
    for row in rows:
        if "kernel_gemm_packed_int4" in row["Name"]:
            return row
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Nsight Systems reports")
    parser.add_argument("--reports_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    reports_dir = Path(args.reports_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_shape_rows = []
    for report in sorted(reports_dir.glob("nsys_m*_k*_n*.nsys-rep")):
        name = report.stem.replace("nsys_", "")
        parts = name.split("_")
        m = int(parts[0].replace("m", ""))
        k = int(parts[1].replace("k", ""))
        n = int(parts[2].replace("n", ""))
        top = extract_kernel_time(report)
        if top is None:
            continue
        total_ns = float(top["Total Time (ns)"])
        per_shape_rows.append(
            {
                "M": m,
                "K": k,
                "N": n,
                "family": shape_family(n, k),
                "kernel": top["Name"],
                "total_ms": total_ns / 1e6,
            }
        )

    per_shape_rows.sort(key=lambda r: (r["M"], r["K"], r["N"]))

    kernel_table = out_dir / "kernel_time_by_shape_nsys.md"
    lines = []
    lines.append("| M | K | N | family | kernel | total_ms |")
    lines.append("|---:|---:|---:|---|---|---:|")
    for row in per_shape_rows:
        lines.append(
            "| {M} | {K} | {N} | {family} | {kernel} | {total_ms:.6f} |".format(
                **row
            )
        )
    kernel_table.write_text("\n".join(lines) + "\n")

    ranked = sorted(per_shape_rows, key=lambda r: r["total_ms"], reverse=True)
    top_table = out_dir / "top_kernels_by_cuda_time.md"
    lines = []
    lines.append("| M | K | N | family | kernel | total_ms |")
    lines.append("|---:|---:|---:|---|---|---:|")
    for row in ranked[:20]:
        lines.append(
            "| {M} | {K} | {N} | {family} | {kernel} | {total_ms:.6f} |".format(
                **row
            )
        )
    top_table.write_text("\n".join(lines) + "\n")

    # Per-family summary
    family_rows = {}
    for row in per_shape_rows:
        key = (row["family"], row["M"])
        family_rows.setdefault(key, []).append(row["total_ms"])

    lines = []
    lines.append("| family | M | count | median total_ms |")
    lines.append("|---|---:|---:|---:|")
    for (family, m) in sorted(family_rows.keys()):
        values = sorted(family_rows[(family, m)])
        mid = len(values) // 2
        if len(values) % 2 == 0:
            median = (values[mid - 1] + values[mid]) / 2
        else:
            median = values[mid]
        lines.append(
            f"| {family} | {m} | {len(values)} | {median:.6f} |"
        )
    (out_dir / "family_summary_nsys.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
