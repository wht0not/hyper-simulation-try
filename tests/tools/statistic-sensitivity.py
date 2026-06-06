from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import mean
from typing import Any


DEFAULT_INPUT_DIR = Path(
    "/home/vincent/hyper-simulation-try/tests/tasks/memory/locomo/data/hyper_simulation/sensitivity"
)
DEFAULT_OUTPUT_PATH = DEFAULT_INPUT_DIR / "llm_judge_sensitivity_stats.jsonl"

DIR_PATTERN = re.compile(
    r"^(?P<method>[^-]+)-sigma(?P<sigma>\d+p\d+)-b(?P<b>\d+)-delta(?P<delta>\d+p\d+)$"
)
SWEEP_ORDER = ("sigma", "delta", "b")


def _decode_number(token: str) -> float:
    return float(token.replace("p", "."))


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Top-level JSON must be object: {path}")
    return payload


def _extract_scores_and_elapsed(final_json_path: Path) -> tuple[list[float], list[float]]:
    payload = _load_json(final_json_path)
    results = payload.get("results", [])
    if not isinstance(results, list):
        return [], []

    scores: list[float] = []
    elapsed_seconds: list[float] = []
    for row in results:
        if not isinstance(row, dict):
            continue
        metrics = row.get("metrics")
        if not isinstance(metrics, dict):
            continue
        judge = metrics.get("llm_as_judge")
        if not isinstance(judge, dict):
            continue
        score = judge.get("score")
        elapsed = row.get("prepared_elapsed_seconds")
        if isinstance(score, (int, float)):
            scores.append(float(score))
        if isinstance(elapsed, (int, float)):
            elapsed_seconds.append(float(elapsed))
    return scores, elapsed_seconds


def _collect_scores(input_dir: Path) -> dict[tuple[str, float, int, float], dict[str, list[float]]]:
    # (method, sigma, b, delta) -> {"scores": [...], "elapsed_seconds": [...]}
    grouped: dict[tuple[str, float, int, float], dict[str, list[float]]] = {}

    for final_json in sorted(input_dir.glob("*/final.json")):
        folder_name = final_json.parent.name
        match = DIR_PATTERN.match(folder_name)
        if match is None:
            continue

        method = match.group("method")
        sigma_value = _decode_number(match.group("sigma"))
        b_value = int(match.group("b"))
        delta_value = _decode_number(match.group("delta"))

        scores, elapsed_seconds = _extract_scores_and_elapsed(final_json)
        if not scores:
            continue

        key = (method, sigma_value, b_value, delta_value)
        if key not in grouped:
            grouped[key] = {"scores": [], "elapsed_seconds": []}
        grouped[key]["scores"].extend(scores)
        grouped[key]["elapsed_seconds"].extend(elapsed_seconds)

    return grouped


def _write_jsonl(grouped: dict[tuple[str, float, int, float], dict[str, list[float]]], output_path: Path) -> int:
    base_rows: list[dict[str, Any]] = []
    for (method, sigma_value, b_value, delta_value), stats in grouped.items():
        scores = stats["scores"]
        elapsed_seconds = stats["elapsed_seconds"]
        base_rows.append(
            {
                "method": method,
                "sigma": sigma_value,
                "b": b_value,
                "delta": delta_value,
                "llm_judge_mean": mean(scores),
                "score_count": len(scores),
                "prepared_elapsed_mean_seconds": (sum(elapsed_seconds) / len(scores)),
            }
        )

    methods = sorted({row["method"] for row in base_rows})
    rows: list[dict[str, Any]] = []
    for method in methods:
        method_rows = [row for row in base_rows if row["method"] == method]

        sigma_rows = sorted(
            [row for row in method_rows if row["b"] == 5 and row["delta"] == 0.7],
            key=lambda row: row["sigma"],
        )
        delta_rows = sorted(
            [row for row in method_rows if row["sigma"] == 0.75 and row["b"] == 5],
            key=lambda row: row["delta"],
        )
        b_rows = sorted(
            [row for row in method_rows if row["sigma"] == 0.75 and row["delta"] == 0.7],
            key=lambda row: row["b"],
        )

        sweep_rows = {
            "sigma": sigma_rows,
            "delta": delta_rows,
            "b": b_rows,
        }
        for sweep in SWEEP_ORDER:
            for row in sweep_rows[sweep]:
                row_with_sweep = dict(row)
                row_with_sweep["sweep"] = sweep
                rows.append(row_with_sweep)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="按 method 分组，并按 sigma/delta/b 扫描顺序输出 llm_as_judge 与 prepared_elapsed_seconds 平均值。"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="sensitivity 目录路径。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="输出 jsonl 文件路径。",
    )
    args = parser.parse_args()

    grouped = _collect_scores(args.input_dir)
    row_count = _write_jsonl(grouped, args.output)
    print(f"Saved {row_count} rows to: {args.output}")


if __name__ == "__main__":
    main()
