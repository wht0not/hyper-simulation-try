from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from .memory import build_memorybank_memory_dataset, load_memorybank_memory_payload
from utils.utils import (
    coerce_category,
    entry_key,
    load_entries,
    load_existing_result_map,
    load_existing_results,
    retrieved_output_path,
    safe_write_json,
    window_tag,
)


def _iter_memorybank_samples(payload: Any) -> list[tuple[str, dict[str, Any]]]:
    if isinstance(payload, dict):
        return [(str(key), value) for key, value in payload.items() if isinstance(value, dict)]
    if isinstance(payload, list):
        items: list[tuple[str, dict[str, Any]]] = []
        for idx, value in enumerate(payload):
            if isinstance(value, dict):
                items.append((str(value.get("sample_id", idx)), value))
        return items
    return []


def count_memorybank_questions(dataset_path: str | Path, limit: int | None = None) -> int:
    dataset_file = Path(dataset_path)
    if not dataset_file.exists():
        return 0
    try:
        payload = json.loads(dataset_file.read_text(encoding="utf-8"))
    except Exception:
        return 0
    samples = _iter_memorybank_samples(payload)
    if limit is not None and limit > 0:
        samples = samples[:limit]
    total = 0
    for _, sample in samples:
        questions = sample.get("question", sample.get("qa", []))
        if isinstance(questions, list):
            total += len([row for row in questions if isinstance(row, dict)])
    return total


def _retrieved_payload(
    rows: list[dict[str, Any]],
    dataset_file: Path,
    out_file: Path,
    model_name: str,
    elapsed_seconds: float | None = None,
) -> dict[str, Any]:
    summary = {
        "method": "memorybank",
        "stage": "retrieve",
        "window": window_tag(dataset_file),
        "source_path": str(dataset_file),
        "retrieved_file": str(out_file),
        "memorybank_model_name": model_name,
        "total": len(rows),
    }
    if elapsed_seconds is not None:
        summary["elapsed_seconds"] = round(float(elapsed_seconds), 4)
    return {"summary": summary, "entries": rows}


def _embed_texts(texts: list[str]) -> np.ndarray:
    from hyper_simulation.component.embedding import get_embedding_batch

    if not texts:
        return np.zeros((0, 1), dtype=np.float32)
    return np.asarray(get_embedding_batch(texts), dtype=np.float32)


def _top_k_summaries(question: str, summary_rows: list[dict[str, Any]], top_k: int = 3) -> list[dict[str, Any]]:
    if not summary_rows:
        return []
    from hyper_simulation.component.embedding import get_embedding_batch

    question_embedding = np.asarray(get_embedding_batch([question])[0], dtype=np.float32)
    summary_texts = [str(row.get("content", "")).strip() for row in summary_rows]
    summary_embeddings = _embed_texts(summary_texts)
    scores = np.matmul(summary_embeddings, question_embedding)
    top_k = max(1, min(int(top_k), len(summary_rows)))
    sorted_indices = np.argsort(scores)[::-1][:top_k]
    return [summary_rows[int(idx)] for idx in sorted_indices.tolist()]


def retrieve_memorybank_dataset(
    dataset_path: str,
    output_dir: str,
    model_name: str,
    output_path: str | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    dataset_file = Path(dataset_path)
    out_file = Path(output_path) if output_path else retrieved_output_path(output_dir, "memorybank", dataset_file)

    existing_entries = load_entries(dataset_file, limit=limit)
    try:
        raw_payload = json.loads(dataset_file.read_text(encoding="utf-8"))
    except Exception:
        raw_payload = None
    summary = raw_payload.get("summary", {}) if isinstance(raw_payload, dict) else {}
    if isinstance(summary, dict) and summary.get("stage") == "retrieve" and existing_entries:
        payload = {
            "summary": {**summary, "retrieved_file": str(dataset_file), "total": len(existing_entries)},
            "entries": existing_entries,
        }
        safe_write_json(out_file, payload)
        return payload

    samples = _iter_memorybank_samples(raw_payload)
    if limit is not None and limit > 0:
        samples = samples[:limit]

    build_memorybank_memory_dataset(
        dataset_path=dataset_path,
        output_dir=output_dir,
        model_name=model_name,
        limit=limit,
    )
    total_questions = count_memorybank_questions(dataset_file, limit=limit)
    existing_map = load_existing_result_map(out_file)
    retrieved_rows = load_existing_results(out_file)
    pbar = tqdm(total=total_questions, desc="locomo/retrieve/memorybank", unit="q")

    try:
        for sample_key, sample in samples:
            chat_history = sample.get("conversation", [])
            questions = sample.get("question", sample.get("qa", []))
            if not isinstance(chat_history, list) or not isinstance(questions, list):
                continue

            sample_id = str(sample.get("sample_id", sample_key)).strip() or str(sample_key)
            valid_questions = [row for row in questions if isinstance(row, dict)]
            pending_questions: list[dict[str, Any]] = []
            for qa_idx, question_item in enumerate(valid_questions):
                row_stub = {
                    "sample_id": sample_id,
                    "qa_id": str(question_item.get("qa_id", qa_idx)),
                    "q": str(question_item.get("question", "")).strip(),
                }
                if entry_key(row_stub) in existing_map:
                    pbar.update(1)
                    continue
                pending_questions.append(question_item)
            if not pending_questions:
                continue

            cache_payload = load_memorybank_memory_payload(output_dir=output_dir, sample_id=sample_id)
            memory_payload = cache_payload.get("memory_payload", {}) if isinstance(cache_payload, dict) else {}
            if not isinstance(memory_payload, dict):
                memory_payload = {}

            summary_rows = memory_payload.get("summary_by_date", [])
            personality_rows = memory_payload.get("personality_by_date", [])
            overall_history = str(memory_payload.get("overall_history", ""))
            overall_personality = str(memory_payload.get("overall_personality", ""))
            if not isinstance(summary_rows, list):
                summary_rows = []
            if not isinstance(personality_rows, list):
                personality_rows = []
            user_name = str(cache_payload.get("metadata", {}).get("user_name", sample.get("conversation", [{}])[0].get("speaker", "User")))
            ai_name = str(cache_payload.get("metadata", {}).get("ai_name", "Assistant"))

            for qa_idx, question_item in enumerate(valid_questions):
                question = str(question_item.get("question", "")).strip()
                if not question:
                    pbar.update(1)
                    continue
                qa_id = str(question_item.get("qa_id", qa_idx))
                row_stub = {"sample_id": sample_id, "qa_id": qa_id, "q": question}
                if entry_key(row_stub) in existing_map:
                    pbar.update(1)
                    continue

                retrieve_started_at = time.perf_counter()
                top_summary_rows = _top_k_summaries(question, summary_rows, top_k=3)
                retrieval_time = round(time.perf_counter() - retrieve_started_at, 4)
                d_list = [f"Date: {row['date']}\nSummary: {row['content']}" for row in top_summary_rows]

                row = {
                    "sample_id": sample_id,
                    "qa_id": qa_id,
                    "q": question,
                    "answer": question_item.get("answer"),
                    "category": coerce_category(question_item.get("category")),
                    "method": "memorybank",
                    "d": d_list,
                    "memorybank_context": {
                        "user_name": user_name,
                        "ai_name": ai_name,
                        "overall_history": overall_history,
                        "overall_personality": overall_personality,
                        "search_time": retrieval_time,
                    },
                }
                retrieved_rows.append(row)
                existing_map[entry_key(row)] = row
                safe_write_json(
                    out_file,
                    _retrieved_payload(
                        retrieved_rows,
                        dataset_file,
                        out_file,
                        model_name=model_name,
                        elapsed_seconds=time.perf_counter() - started_at,
                    ),
                )
                pbar.update(1)
    finally:
        pbar.close()

    payload = _retrieved_payload(
        retrieved_rows,
        dataset_file,
        out_file,
        model_name=model_name,
        elapsed_seconds=time.perf_counter() - started_at,
    )
    safe_write_json(out_file, payload)
    return payload
