from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from tqdm import tqdm

from prompt.memorybank import build_memorybank_answer_prompt
from utils.utils import (
    entry_key,
    load_entries,
    load_existing_result_map,
    load_existing_results,
    prepared_output_path,
    safe_write_json,
    window_tag,
)
from .retrieval import count_memorybank_questions, retrieve_memorybank_dataset


def _prepared_payload(
    rows: list[dict[str, Any]],
    source_file: Path,
    out_file: Path,
    model_name: str,
    elapsed_seconds: float | None = None,
) -> dict[str, Any]:
    summary = {
        "method": "memorybank",
        "window": window_tag(source_file),
        "source_path": str(source_file),
        "prepared_file": str(out_file),
        "memorybank_model_name": model_name,
        "total": len(rows),
    }
    if elapsed_seconds is not None:
        summary["elapsed_seconds"] = round(float(elapsed_seconds), 4)
    return {"summary": summary, "results": rows}


def _prepare_memorybank_row(
    sample_id: str,
    qa_id: str,
    question: str,
    answer: Any,
    category: int,
    d_list: list[str],
    memory_payload: dict[str, Any],
) -> dict[str, Any]:
    user_name = str(memory_payload.get("user_name", "User"))
    overall_history = str(memory_payload.get("overall_history", "")).strip()
    overall_personality = str(memory_payload.get("overall_personality", "")).strip()
    related_memory = "\n\n".join([str(one).strip() for one in d_list if str(one).strip()]).strip()
    search_time = float(memory_payload.get("search_time", 0.0) or 0.0)

    prepared: dict[str, Any] = {
        "sample_id": sample_id,
        "qa_id": qa_id,
        "q": question,
        "answer": answer,
        "category": category,
        "method": "memorybank",
    }
    answer_prompt_payload = build_memorybank_answer_prompt(
        user_name=user_name,
        overall_history=overall_history,
        overall_personality=overall_personality,
        related_memory=related_memory,
        question=question,
        category=category,
        answer=str(answer or ""),
        sample_id=sample_id,
        qa_id=qa_id,
    )
    prepared["prompt"] = str(answer_prompt_payload["prompt"])
    prepared["answer_temperature"] = float(answer_prompt_payload.get("temperature", 0.1))
    if isinstance(answer_prompt_payload.get("cat5_answer_key"), dict):
        prepared["cat5_answer_key"] = answer_prompt_payload["cat5_answer_key"]
    return prepared


def prepare_memorybank_dataset(
    dataset_path: str,
    output_dir: str,
    model_name: str,
    limit: int | None = None,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    dataset_file = Path(dataset_path)
    retrieved_payload = retrieve_memorybank_dataset(
        dataset_path=dataset_path,
        output_dir=output_dir,
        model_name=model_name,
        limit=limit,
    )
    source_path = str(retrieved_payload.get("summary", {}).get("source_path", dataset_file))
    source_file = Path(source_path)
    retrieved_file = Path(str(retrieved_payload.get("summary", {}).get("retrieved_file", dataset_file)))
    out_file = prepared_output_path(output_dir, "memorybank", source_file)
    entries = load_entries(retrieved_file, limit=limit)

    total_questions = len(entries) if entries else count_memorybank_questions(dataset_file, limit=limit)
    existing_map = load_existing_result_map(out_file)
    prepared_rows = load_existing_results(out_file)
    pbar = tqdm(total=total_questions, desc="locomo/compose/memorybank", unit="q")

    try:
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            if entry_key(entry) in existing_map:
                pbar.update(1)
                continue
            memory_payload = entry.get("memorybank_context", {})
            if not isinstance(memory_payload, dict):
                memory_payload = {}
            d_list = entry.get("d", [])
            if isinstance(d_list, list):
                d_list = [str(one).strip() for one in d_list if str(one).strip()]
            else:
                d_list = []
            prepared = _prepare_memorybank_row(
                sample_id=str(entry.get("sample_id", "")),
                qa_id=str(entry.get("qa_id", "")),
                question=str(entry.get("q", "")).strip(),
                answer=entry.get("answer"),
                category=int(entry.get("category", -1)),
                d_list=d_list,
                memory_payload=memory_payload,
            )
            prepared_rows.append(prepared)
            existing_map[entry_key(prepared)] = prepared
            safe_write_json(
                out_file,
                _prepared_payload(
                    prepared_rows,
                    source_file,
                    out_file,
                    model_name=model_name,
                    elapsed_seconds=time.perf_counter() - started_at,
                ),
            )
            pbar.update(1)
    finally:
        pbar.close()

    payload = _prepared_payload(
        prepared_rows,
        source_file,
        out_file,
        model_name=model_name,
        elapsed_seconds=time.perf_counter() - started_at,
    )
    safe_write_json(out_file, payload)
    return payload
