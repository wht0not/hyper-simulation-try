from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm

from prompt.base import QA_PROMPT
from .qa_utils import build_question_text, resolve_qa_answer
from .utils import (
    coerce_category,
    entry_key,
    load_entries,
    load_existing_results,
    prepared_output_path,
    safe_write_json,
    window_tag,
)


def _item_metrics_from_texts(items: list[str]) -> tuple[int, float]:
    cleaned = [str(item).strip() for item in items if str(item).strip()]
    if not cleaned:
        return 0, 0.0
    avg_chunk_size = sum(len(item) for item in cleaned) / len(cleaned)
    return len(cleaned), round(avg_chunk_size, 2)


def _prepared_payload(
    rows: list[dict[str, Any]],
    dataset_file: Path,
    out_file: Path,
) -> dict[str, Any]:
    def _mean_std(values: list[float]) -> tuple[float, float]:
        if not values:
            return 0.0, 0.0
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        return round(mean, 6), round(math.sqrt(variance), 6)

    top_k_values = [float(row.get("top_k", 0.0) or 0.0) for row in rows if isinstance(row, dict)]
    chunk_size_values = [float(row.get("chunk_size", 0.0) or 0.0) for row in rows if isinstance(row, dict)]
    prepared_elapsed_values = [
        float(row.get("prepared_elapsed_seconds", 0.0) or 0.0) for row in rows if isinstance(row, dict)
    ]
    top_k_mean, top_k_std = _mean_std(top_k_values)
    chunk_size_mean, chunk_size_std = _mean_std(chunk_size_values)
    prepared_elapsed_mean, prepared_elapsed_std = _mean_std(prepared_elapsed_values)
    return {
        "summary": {
            "method": "context",
            "window": window_tag(dataset_file),
            "source_path": str(dataset_file),
            "prepared_file": str(out_file),
            "total": len(rows),
            "top_k_mean": top_k_mean,
            "top_k_std": top_k_std,
            "chunk_size_mean": chunk_size_mean,
            "chunk_size_std": chunk_size_std,
            "prepared_elapsed_seconds_mean": prepared_elapsed_mean,
            "prepared_elapsed_seconds_std": prepared_elapsed_std,
        },
        "results": rows,
    }


def _compose_context(entry: dict[str, Any]) -> str:
    d_val = entry.get("d")
    d_start = str(entry.get("d_start", "")).strip()
    if isinstance(d_val, list):
        sessions = [str(one).strip() for one in d_val if str(one).strip()]
        if d_start and sessions:
            return d_start + "\n\n" + "\n\n".join(sessions)
        if d_start:
            return d_start
        return "\n\n".join(sessions)
    if isinstance(d_val, str):
        return d_val.strip()
    return ""


def compose_context_entry(entry: dict[str, Any], method_name: str = "context") -> dict[str, Any] | None:
    q = str(entry.get("q", "")).strip()
    d = entry.get("d", [])
    answer = resolve_qa_answer(entry)
    if not q or not d:
        return None

    category = coerce_category(entry.get("category"))
    if category == 5:
        return None
    context = _compose_context(entry)
    prepared: dict[str, Any] = {
        "sample_id": entry.get("sample_id"),
        "qa_id": entry.get("qa_id"),
        "q": q,
        "answer": answer,
        "category": category,
        "method": method_name,
    }

    prepared["prompt"] = f"{context}\n\n{QA_PROMPT.format(build_question_text(q, category))}"
    if isinstance(d, list):
        item_texts = [str(one).strip() for one in d if str(one).strip()]
    else:
        item_texts = [str(d).strip()] if str(d).strip() else []
    top_k, chunk_size = _item_metrics_from_texts(item_texts)
    prepared["top_k"] = top_k
    prepared["chunk_size"] = chunk_size

    return prepared


def prepare_context_dataset(
    dataset_path: str,
    output_dir: str,
    limit: int | None = None,
    checkpoint_every: int = 500,
) -> dict[str, Any]:
    dataset_file = Path(dataset_path)
    out_file = prepared_output_path(output_dir, "context", dataset_file)
    entries = load_entries(dataset_file, limit=limit)
    prepared_rows = [
        row
        for row in load_existing_results(out_file)
        if coerce_category(row.get("category", -1)) != 5
    ]
    existing_map = {entry_key(row): row for row in prepared_rows if entry_key(row)}
    checkpoint_every = max(1, int(checkpoint_every))
    pending_writes = 0

    for entry in tqdm(entries, desc="locomo/compose/context", unit="q"):
        if entry_key(entry) in existing_map:
            continue
        item_started_at = time.perf_counter()
        prepared = compose_context_entry(entry)
        if prepared is not None:
            prepared["prepared_elapsed_seconds"] = round(time.perf_counter() - item_started_at, 6)
            prepared_rows.append(prepared)
            existing_map[entry_key(prepared)] = prepared
            pending_writes += 1
            if pending_writes >= checkpoint_every:
                safe_write_json(out_file, _prepared_payload(prepared_rows, dataset_file, out_file))
                pending_writes = 0

    payload = _prepared_payload(prepared_rows, dataset_file, out_file)
    safe_write_json(out_file, payload)
    return payload
