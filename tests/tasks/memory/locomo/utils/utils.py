from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_DATASET_PATHS = [
    "/home/vincent/hyper-simulation-try/tests/tasks/memory/locomo/data/context/locomo_context_raw.json",
]
DEFAULT_OUTPUT_DIR = "/home/vincent/hyper-simulation-try/tests/tasks/memory/locomo/data"
DEFAULT_INSTANCES_ROOT = "/home/vincent/hyper-simulation-try/data/hypergraphs/locomo_context"
DEFAULT_RAG_SOURCE_PATH = "/home/vincent/hyper-simulation-try/tests/tasks/memory/locomo/data/rag/locomo10_rag.json"
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


def load_entries(dataset_path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    if not dataset_path.exists():
        return []
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    entries = payload.get("entries", payload.get("results", [])) if isinstance(payload, dict) else []
    if not isinstance(entries, list):
        return []
    valid_entries = [one for one in entries if isinstance(one, dict)]
    if limit is not None and limit > 0:
        return valid_entries[:limit]
    return valid_entries


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
    method = str(method).strip()
    tag = str(tag).strip()
    if not method:
        raise ValueError("method is required")
    if not tag:
        return f"locomo_{method}"
    if tag == method or tag.startswith(f"{method}_"):
        return f"locomo_{tag}"
    return f"locomo_{method}_{tag}"


def prepared_output_path(output_dir: str | Path, method: str, source_path: str | Path) -> Path:
    tag = window_tag(source_path)
    return Path(output_dir) / f"{result_basename(method, tag)}_prepared.json"


def answers_output_path(output_dir: str | Path, method: str, source_path: str | Path) -> Path:
    tag = window_tag(source_path)
    return Path(output_dir) / f"{result_basename(method, tag)}_answers.json"


def final_output_path(output_dir: str | Path, method: str, source_path: str | Path) -> Path:
    tag = window_tag(source_path)
    return Path(output_dir) / f"{result_basename(method, tag)}.json"


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
