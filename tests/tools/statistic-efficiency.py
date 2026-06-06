from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


DEFAULT_INPUT_DIR = Path(
    "/home/vincent/hyper-simulation-try/tests/tasks/memory/locomo/data/hyper_simulation/efficiency"
)
DEFAULT_OUTPUT_PATH = DEFAULT_INPUT_DIR / "summary.jsonl"

DIR_PATTERN = re.compile(
    r"^(?P<method>[^-]+)-sigma(?P<sigma>\d+p\d+)-b(?P<b>\d+)-delta(?P<delta>\d+p\d+)$"
)
SWEEP_ORDER = ("sigma", "b", "delta")

SIGMA_SWEEP_PAIRS = {
    (5, 0.65),
    (7, 0.75),
    (10, 0.85),
}
B_SWEEP_PAIRS = {
    (0.55, 0.5),
    (0.65, 0.7),
    (0.75, 0.8),
}
DELTA_SWEEP_PAIRS = {
    (0.8, 5),
    (0.7, 7),
    (0.6, 10),
}


def _decode_number(token: str) -> float:
    return float(token.replace("p", "."))


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Top-level JSON must be object: {path}")
    return payload


def _sum_prepared_elapsed(prepared_json_path: Path) -> float:
    payload = _load_json(prepared_json_path)
    results = payload.get("results", [])
    if not isinstance(results, list):
        return 0.0

    total = 0.0
    for row in results:
        if not isinstance(row, dict):
            continue
        elapsed = row.get("prepared_elapsed_seconds")
        if isinstance(elapsed, (int, float)):
            total += float(elapsed)
    return total


def _collect_rows(input_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for prepared_json in sorted(input_dir.glob("*/prepared.json")):
        folder_name = prepared_json.parent.name
        match = DIR_PATTERN.match(folder_name)
        if match is None:
            continue

        method = match.group("method")
        sigma_value = _decode_number(match.group("sigma"))
        b_value = int(match.group("b"))
        delta_value = _decode_number(match.group("delta"))
        total_time = _sum_prepared_elapsed(prepared_json)

        sweep_matches: list[str] = []
        if (b_value, delta_value) in SIGMA_SWEEP_PAIRS:
            sweep_matches.append("sigma")
        if (sigma_value, delta_value) in B_SWEEP_PAIRS:
            sweep_matches.append("b")
        if (sigma_value, b_value) in DELTA_SWEEP_PAIRS:
            sweep_matches.append("delta")

        if not sweep_matches:
            sweep_matches.append("unknown")

        for sweep in sweep_matches:
            rows.append(
                {
                    "sweep": sweep,
                    "method": method,
                    "sigma": sigma_value,
                    "b": b_value,
                    "delta": delta_value,
                    "total_prepared_elapsed_seconds": round(total_time, 6),
                }
            )

    sweep_rank = {name: index for index, name in enumerate(SWEEP_ORDER)}
    rows.sort(
        key=lambda row: (
            sweep_rank.get(str(row["sweep"]), len(SWEEP_ORDER)),
            str(row["method"]),
            float(row["sigma"]),
            int(row["b"]),
            float(row["delta"]),
        )
    )
    return rows


def _write_jsonl(rows: list[dict[str, Any]], output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="汇总 efficiency prepared.json 中的 prepared_elapsed_seconds 总和，并按 sweep/method/sigma/b/delta 排序输出。"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="efficiency 目录路径。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="输出 jsonl 文件路径。",
    )
    args = parser.parse_args()

    rows = _collect_rows(args.input_dir)
    row_count = _write_jsonl(rows, args.output)
    print(f"Saved {row_count} rows to: {args.output}")


if __name__ == "__main__":
    main()
