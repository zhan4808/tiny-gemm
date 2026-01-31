import argparse
import csv
from pathlib import Path
import subprocess


def main() -> None:
    parser = argparse.ArgumentParser(description="Run nsys profiling for shapes")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--m_values", default="1,2,4,8")
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    m_values = {int(v.strip()) for v in args.m_values.split(",") if v.strip()}

    shapes = []
    with open(args.csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            m = int(row["M"])
            if m not in m_values:
                continue
            k = int(row["K"])
            n = int(row["N"])
            shapes.append((m, k, n))

    for m, k, n in shapes:
        out_path = out_dir / f"nsys_m{m}_k{k}_n{n}"
        cmd = [
            "nsys",
            "profile",
            "--force-overwrite=true",
            "--sample=none",
            "--trace=cuda,nvtx",
            "-o",
            str(out_path),
            "python3",
            "tools/run_gemm.py",
            "--m",
            str(m),
            "--k",
            str(k),
            "--n",
            str(n),
            "--iters",
            str(args.iters),
        ]
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
