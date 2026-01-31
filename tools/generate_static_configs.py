import argparse
import csv
from collections import defaultdict
from pathlib import Path


def bucket_m(m: int) -> str | None:
    if m <= 1:
        return "m1"
    if m <= 4:
        return "small"
    if m <= 8:
        return "medium"
    return None


def shape_family(n: int, k: int) -> str:
    if n > k:
        return "ffn_up"
    if n == k:
        return "q_proj"
    if n <= 1280:
        return "kv_proj"
    return "ffn_down"


def parse_config(config_str: str) -> dict[str, int]:
    if not config_str:
        return {}
    items = config_str.split(",")
    parsed = {}
    for item in items:
        key, value = item.split("=")
        parsed[key] = int(value)
    return parsed


def format_configs(configs: dict[tuple[str, str], dict[str, int]]) -> str:
    lines = ["_STATIC_CONFIGS: dict[tuple[str, str], dict[str, int]] = {"]
    for key in sorted(configs.keys()):
        m_bucket, family = key
        cfg = configs[key]
        lines.append(f'    ("{m_bucket}", "{family}"): {{')
        for cfg_key in sorted(cfg.keys()):
            lines.append(f'        "{cfg_key}": {cfg[cfg_key]},')
        lines.append("    },")
    lines.append("}")
    return "\n".join(lines)


def load_best_configs(csv_path: Path) -> dict[tuple[str, str], dict[str, int]]:
    buckets = defaultdict(lambda: defaultdict(list))
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("best_config"):
                continue
            m = int(row["M"])
            n = int(row["N"])
            k = int(row["K"])
            speed = float(row["speedup"]) if row.get("speedup") else 0.0
            m_bucket = bucket_m(m)
            if m_bucket is None:
                continue
            family = shape_family(n, k)
            buckets[(m_bucket, family)][row["best_config"]].append(speed)

    best_configs = {}
    for key, configs in buckets.items():
        best_cfg = None
        best_avg = -1.0
        for cfg_str, speeds in configs.items():
            avg = sum(speeds) / len(speeds)
            if avg > best_avg:
                best_avg = avg
                best_cfg = cfg_str
        if best_cfg:
            best_configs[key] = parse_config(best_cfg)
    return best_configs


def replace_block(text: str, new_block: str) -> str:
    start = "# BEGIN STATIC CONFIGS"
    end = "# END STATIC CONFIGS"
    start_idx = text.find(start)
    end_idx = text.find(end)
    if start_idx == -1 or end_idx == -1:
        raise RuntimeError("Missing static config markers in triton_gemm.py")
    end_idx += len(end)
    return text[:start_idx] + start + "\n" + new_block + "\n" + end + text[end_idx:]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate static config table")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    csv_path = Path(args.csv)
    output_path = Path(args.output)
    configs = load_best_configs(csv_path)
    new_block = format_configs(configs)

    text = output_path.read_text()
    updated = replace_block(text, new_block)
    output_path.write_text(updated)


if __name__ == "__main__":
    main()
