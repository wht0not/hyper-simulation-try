from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .metrics import (
    LLM_JUDGE_REPEAT,
    compute_base_metrics,
    compute_llm_judge_metrics,
    normalize_answer,
)
from .utils import coerce_category, entry_key, load_existing_results, safe_write_json


def evaluate_answer(prediction: str, golden: Any, category: Any) -> dict[str, float]:
    return compute_base_metrics(
        prediction=prediction,
        golden=golden,
        category=category,
    )


def summarize_results(
    rows: list[dict[str, Any]],
    method: str,
    model_name: str = "",
    source_path: str = "",
    window: str = "",
) -> dict[str, Any]:
    category_counts: dict[int, int] = {k: 0 for k in [1, 2, 3, 4, 5]}
    metric_aliases = {
        "cosine_similarity": "cosine_similarity",
        "f1": "F1",
        "rouge_l": "rouge_L",
        "bleu1": "BLEU",
    }
    metric_sums: dict[str, float] = {k: 0.0 for k in metric_aliases}
    metric_counts: dict[str, int] = {k: 0 for k in metric_aliases}
    llm_judge_sum = 0.0
    llm_judge_count = 0
    llm_judge_std_sum = 0.0
    llm_judge_std_count = 0
    by_category_metric_sums: dict[int, dict[str, float]] = {
        k: {metric_name: 0.0 for metric_name in metric_aliases}
        for k in [1, 2, 3, 4, 5]
    }
    by_category_metric_counts: dict[int, dict[str, int]] = {
        k: {metric_name: 0 for metric_name in metric_aliases}
        for k in [1, 2, 3, 4, 5]
    }
    by_category_llm_judge_sum: dict[int, float] = {k: 0.0 for k in [1, 2, 3, 4, 5]}
    by_category_llm_judge_count: dict[int, int] = {k: 0 for k in [1, 2, 3, 4, 5]}
    by_category_llm_judge_std_sum: dict[int, float] = {k: 0.0 for k in [1, 2, 3, 4, 5]}
    by_category_llm_judge_std_count: dict[int, int] = {k: 0 for k in [1, 2, 3, 4, 5]}

    for row in rows:
        category = coerce_category(row.get("category"))
        if category not in category_counts:
            continue
        metrics = row.get("metrics", {}) or {}
        category_counts[category] += 1
        for metric_name in ["f1", "bleu1", "rouge_l", "cosine_similarity"]:
            metric_value = metrics.get(metric_name)
            if metric_name in metrics and metric_value is not None:
                metric_sums[metric_name] += float(metrics.get(metric_name, 0.0))
                metric_counts[metric_name] += 1
                by_category_metric_sums[category][metric_name] += float(metrics.get(metric_name, 0.0))
                by_category_metric_counts[category][metric_name] += 1
        llm_judge = metrics.get("llm_as_judge", {})
        if isinstance(llm_judge, dict) and "mean" in llm_judge:
            llm_judge_sum += float(llm_judge.get("mean", 0.0))
            llm_judge_count += 1
            by_category_llm_judge_sum[category] += float(llm_judge.get("mean", 0.0))
            by_category_llm_judge_count[category] += 1
        if isinstance(llm_judge, dict) and "std" in llm_judge:
            llm_judge_std_sum += float(llm_judge.get("std", 0.0))
            llm_judge_std_count += 1
            by_category_llm_judge_std_sum[category] += float(llm_judge.get("std", 0.0))
            by_category_llm_judge_std_count[category] += 1

    total_q = sum(category_counts.values())
    cat_summary = {}
    for k in [4, 1, 2, 3, 5]:
        c_total = category_counts[k]
        cat_summary[str(k)] = {"total": c_total}
        for metric_name, summary_name in metric_aliases.items():
            c_metric_count = by_category_metric_counts[k][metric_name]
            c_metric_sum = by_category_metric_sums[k][metric_name]
            cat_summary[str(k)][summary_name] = (
                round(c_metric_sum / c_metric_count, 4) if c_metric_count > 0 else 0.0
            )
        cat_summary[str(k)]["LLM-as-judge_mean"] = (
            round(by_category_llm_judge_sum[k] / by_category_llm_judge_count[k], 4)
            if by_category_llm_judge_count[k] > 0
            else None
        )
        cat_summary[str(k)]["LLM-as-judge_std"] = (
            round(by_category_llm_judge_std_sum[k] / by_category_llm_judge_std_count[k], 4)
            if by_category_llm_judge_std_count[k] > 0
            else None
        )

    summary = {
        "method": method,
        "total": total_q,
        "by_category": cat_summary,
        "overall": {
            summary_name: round(metric_sums[metric_name] / metric_counts[metric_name], 4)
            if metric_counts[metric_name] > 0
            else None
            for metric_name, summary_name in metric_aliases.items()
        },
    }
    summary["overall"]["LLM-as-judge_mean"] = (
        round(llm_judge_sum / llm_judge_count, 4) if llm_judge_count > 0 else None
    )
    summary["overall"]["LLM-as-judge_std"] = (
        round(llm_judge_std_sum / llm_judge_std_count, 4) if llm_judge_std_count > 0 else None
    )
    if model_name:
        summary["model_name"] = model_name
    if source_path:
        summary["source_path"] = source_path
    if window:
        summary["window"] = window
    return summary


def evaluate_single_row(
    row: dict[str, Any],
    llm_judge_repeat: int = LLM_JUDGE_REPEAT,
) -> dict[str, Any]:
    out_row = dict(row)
    out_row["prediction"] = normalize_answer(out_row.get("prediction", ""))
    try:
        out_row["metrics"] = compute_base_metrics(
            prediction=out_row.get("prediction", ""),
            golden=out_row.get("answer"),
            category=out_row.get("category"),
        )
    except Exception as exc:
        out_row["metrics"] = {
            "f1": 0.0,
            "bleu1": 0.0,
            "rouge_l": 0.0,
            "cosine_similarity": 0.0,
            "error": f"{type(exc).__name__}: {exc}",
        }
    if coerce_category(out_row.get("category")) != 5:
        try:
            out_row["metrics"]["llm_as_judge"] = compute_llm_judge_metrics(
                question=str(out_row.get("q", "")),
                prediction=out_row.get("prediction", ""),
                golden=out_row.get("answer"),
                category=out_row.get("category"),
                llm_judge_repeat=llm_judge_repeat,
            )
        except Exception as exc:
            out_row["metrics"]["llm_as_judge"] = {
                "score": 0.0,
                "mean": 0.0,
                "std": 0.0,
                "runs": [{"score": 0.0, "label": "ERROR", "raw": f"{type(exc).__name__}: {exc}"}],
                "model": None,
                "repeat": 0,
            }
    return out_row


def _metrics_sidecar(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "summary": payload["summary"],
        "results": [
            {
                "sample_id": row.get("sample_id"),
                "qa_id": row.get("qa_id"),
                "q": row.get("q"),
                "category": row.get("category"),
                "answer": row.get("answer"),
                "prediction": row.get("prediction"),
                "metrics": row.get("metrics", {}),
            }
            for row in payload.get("results", [])
            if isinstance(row, dict)
        ],
    }


def _materialize_rows(row_order: list[str], row_map: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row_map[key]
        for key in row_order
        if key in row_map and isinstance(row_map[key], dict) and isinstance(row_map[key].get("metrics"), dict)
    ]


def evaluate_results_file(
    answers_path: str | Path,
    output_path: str | Path,
    method: str,
    model_name: str = "",
    source_path: str = "",
    window: str = "",
    judge_max_workers: int = 4,
    llm_judge_repeat: int = LLM_JUDGE_REPEAT,
) -> dict[str, Any]:
    answers_file = Path(answers_path)
    payload = json.loads(answers_file.read_text(encoding="utf-8"))
    rows = payload.get("results", []) if isinstance(payload, dict) else []
    if not isinstance(rows, list):
        rows = []

    if not model_name:
        model_name = str(payload.get("summary", {}).get("model_name", ""))
    if not source_path:
        source_path = str(payload.get("summary", {}).get("source_path", ""))
    if not window:
        window = str(payload.get("summary", {}).get("window", ""))

    output_file = Path(output_path)
    existing_rows = [
        row for row in load_existing_results(output_file) if isinstance(row.get("metrics"), dict)
    ]
    evaluated_map = {
        entry_key(row): row
        for row in existing_rows
        if entry_key(row) and isinstance(row, dict) and isinstance(row.get("metrics"), dict)
    }
    valid_rows = [row for row in rows if isinstance(row, dict)]
    row_order: list[str] = []
    seen_keys: set[str] = set()
    pending_rows: list[tuple[str, dict[str, Any]]] = []

    for row in valid_rows:
        key = entry_key(row)
        if not key:
            continue
        if key not in seen_keys:
            row_order.append(key)
            seen_keys.add(key)
        existing = evaluated_map.get(key)
        if isinstance(existing, dict) and isinstance(existing.get("metrics"), dict):
            continue
        pending_rows.append((key, row))

    max_workers = max(1, int(judge_max_workers))
    future_map: dict[Any, tuple[str, dict[str, Any]]] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for key, row in pending_rows:
            future = executor.submit(evaluate_single_row, row, llm_judge_repeat)
            future_map[future] = (key, row)

        for future in tqdm(as_completed(future_map), total=len(future_map), desc=f"locomo/evaluate/{method}", unit="q"):
            key, row = future_map[future]
            try:
                out_row = future.result()
            except Exception as exc:
                out_row = dict(row)
                out_row["prediction"] = normalize_answer(out_row.get("prediction", ""))
                out_row["metrics"] = {
                    "f1": 0.0,
                    "bleu1": 0.0,
                    "rouge_l": 0.0,
                    "cosine_similarity": 0.0,
                    "llm_as_judge": {
                        "score": 0.0,
                        "mean": 0.0,
                        "std": 0.0,
                        "runs": [{"score": 0.0, "label": "ERROR", "raw": f"{type(exc).__name__}: {exc}"}],
                        "model": None,
                        "repeat": 0,
                    },
                }
            evaluated_map[key] = out_row
            evaluated_rows = _materialize_rows(row_order, evaluated_map)
            evaluated_payload = {
                "summary": summarize_results(
                    evaluated_rows,
                    method=method,
                    model_name=model_name,
                    source_path=source_path,
                    window=window,
                ),
                "results": evaluated_rows,
            }
            safe_write_json(output_file, evaluated_payload)
            safe_write_json(output_file.with_name(output_file.stem + "_metrics.json"), _metrics_sidecar(evaluated_payload))

    evaluated_rows = _materialize_rows(row_order, evaluated_map)
    evaluated_payload = {
        "summary": summarize_results(
            evaluated_rows,
            method=method,
            model_name=model_name,
            source_path=source_path,
            window=window,
        ),
        "results": evaluated_rows,
    }
    safe_write_json(output_file, evaluated_payload)
    safe_write_json(output_file.with_name(output_file.stem + "_metrics.json"), _metrics_sidecar(evaluated_payload))
    return evaluated_payload
