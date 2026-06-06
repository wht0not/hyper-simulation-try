from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


DEFAULT_DATASET_PATHS = [
    "/home/vincent/hyper-simulation-try/tests/tasks/memory/locomo/data/context/locomo_context.json",
]
DEFAULT_OUTPUT_DIR = "/home/vincent/hyper-simulation-try/tests/tasks/memory/locomo/data"
DEFAULT_INSTANCES_ROOT = "/home/vincent/hyper-simulation-try/data/hypergraphs/locomo/context"
DEFAULT_RAG_SOURCE_PATH = "/home/vincent/hyper-simulation-try/tests/tasks/memory/locomo/data/rag/locomo10_rag.json"
DEFAULT_LANGMEM_DATASET_PATH = "/home/vincent/hyper-simulation-try/tests/tasks/memory/locomo/data/langmem/locomo10_rag.json"
DEFAULT_MODEL_NAME = "qwen3.5:9b"
DEFAULT_TEMPERATURE = 0.1


def safe_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(path)


def entry_key(entry: dict[str, Any]) -> str:
    sample_id = str(entry.get("sample_id", "")).strip()
    qa_id = str(entry.get("qa_id", "")).strip()
    q = str(entry.get("q", "")).strip()
    return f"{sample_id}::{qa_id}" if sample_id and qa_id else q


def _category_filter_env(name: str = "LOCOMO_ALLOWED_CATEGORIES") -> set[int] | None:
    raw = os.environ.get(name)
    if raw is None:
        return None
    categories: set[int] = set()
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        try:
            categories.add(int(token))
        except Exception:
            continue
    return categories or None


def _positive_int_env(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return None
    try:
        value = int(raw)
    except Exception:
        return None
    return value if value > 0 else None


def filtered_row_limit(limit: int | None = None) -> int | None:
    max_rows = _positive_int_env("LOCOMO_MAX_ROWS")
    candidates = [value for value in (limit, max_rows) if value is not None and value > 0]
    return min(candidates) if candidates else None


def filter_entry_rows(rows: list[dict[str, Any]], limit: int | None = None) -> list[dict[str, Any]]:
    allowed_categories = _category_filter_env()
    row_limit = filtered_row_limit(limit)
    filtered_rows: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if allowed_categories is not None and coerce_category(row.get("category", -1)) not in allowed_categories:
            continue
        filtered_rows.append(row)
        if row_limit is not None and len(filtered_rows) >= row_limit:
            break
    return filtered_rows


def load_entries(dataset_path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    if not dataset_path.exists():
        return []
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    entries = payload.get("entries", payload.get("results", [])) if isinstance(payload, dict) else []
    if not isinstance(entries, list):
        return []
    valid_entries = [one for one in entries if isinstance(one, dict)]
    return filter_entry_rows(valid_entries, limit=limit)


def iter_filtered_sample_questions(
    payload: Any,
    limit: int | None = None,
) -> list[tuple[str, dict[str, Any], list[dict[str, Any]]]]:
    sample_items: list[tuple[str, dict[str, Any]]] = []
    if isinstance(payload, dict):
        sample_items = [(str(key), value) for key, value in payload.items() if isinstance(value, dict)]
    elif isinstance(payload, list):
        sample_items = [
            (str(value.get("sample_id", idx)), value)
            for idx, value in enumerate(payload)
            if isinstance(value, dict)
        ]
    if limit is not None and limit > 0:
        sample_items = sample_items[:limit]

    allowed_categories = _category_filter_env()
    row_limit = filtered_row_limit()
    filtered_samples: list[tuple[str, dict[str, Any], list[dict[str, Any]]]] = []
    total_rows = 0

    for sample_key, sample in sample_items:
        questions = sample.get("question", sample.get("qa", []))
        if not isinstance(questions, list):
            continue
        kept_questions: list[dict[str, Any]] = []
        for question in questions:
            if not isinstance(question, dict):
                continue
            if allowed_categories is not None and coerce_category(question.get("category", -1)) not in allowed_categories:
                continue
            kept_questions.append(question)
            total_rows += 1
            if row_limit is not None and total_rows >= row_limit:
                break
        if kept_questions:
            filtered_samples.append((sample_key, sample, kept_questions))
        if row_limit is not None and total_rows >= row_limit:
            break

    return filtered_samples


def load_payload_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    rows = payload.get("results", payload.get("entries", [])) if isinstance(payload, dict) else []
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict)]


def load_existing_result_map(path: Path) -> dict[str, dict[str, Any]]:
    existing: dict[str, dict[str, Any]] = {}
    for row in load_payload_rows(path):
        key = entry_key(row)
        if key:
            existing[key] = row
    return existing


def load_existing_results(path: Path) -> list[dict[str, Any]]:
    return load_payload_rows(path)


def window_tag(path_like: str | Path) -> str:
    stem = Path(path_like).stem
    name = stem.lower()
    if name in {"locomo10_rag", "locomo_context", "locomo_context_raw"}:
        return ""
    if name.startswith("locomo_"):
        suffix = stem[len("locomo_") :].strip("_")
        if suffix:
            return suffix
    return stem


def coerce_category(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return -1


def result_basename(method: str, tag: str) -> str:
    tag = str(tag).strip()
    if not tag:
        return ""
    return tag


def prepared_output_path(output_dir: str | Path, method: str, source_path: str | Path) -> Path:
    return Path(output_dir) / "prepared.json"


def retrieved_output_path(output_dir: str | Path, method: str, source_path: str | Path) -> Path:
    return Path(output_dir) / "retrieved.json"


def answers_output_path(output_dir: str | Path, method: str, source_path: str | Path) -> Path:
    return Path(output_dir) / "answers.json"


def final_output_path(output_dir: str | Path, method: str, source_path: str | Path) -> Path:
    return Path(output_dir) / "final.json"


def rag_retrieved_output_path(
    output_dir: str | Path,
    rag_source_path: str | Path,
    chunk_size: int,
    top_k: int,
) -> Path:
    chunk_size = max(1, int(chunk_size))
    top_k = max(1, int(top_k))
    source_name = Path(rag_source_path).name
    return Path(output_dir) / "rag" / f"{top_k}_{chunk_size}" / source_name
