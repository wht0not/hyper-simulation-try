from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from tqdm import tqdm

from prompt.amem import (
    build_amem_answer_prompt,
)
from utils.utils import (
    entry_key,
    load_entries,
    load_existing_result_map,
    load_existing_results,
    prepared_output_path,
    safe_write_json,
    window_tag,
)
from .retrieval import count_amem_questions, retrieve_amem_dataset


def _prepared_payload(
    rows: list[dict[str, Any]],
    source_file: Path,
    out_file: Path,
    model_name: str,
    elapsed_seconds: float | None = None,
) -> dict[str, Any]:
    summary = {
        "method": "amem",
        "window": window_tag(source_file),
        "source_path": str(source_file),
        "prepared_file": str(out_file),
        "amem_model_name": model_name,
        "total": len(rows),
    }
    if elapsed_seconds is not None:
        summary["elapsed_seconds"] = round(float(elapsed_seconds), 4)
    return {"summary": summary, "results": rows}


def _prepare_amem_row(
    sample_id: str,
    qa_id: str,
    question: str,
    answer: Any,
    category: int,
    speaker_1_name: str,
    speaker_1_memories: str,
    speaker_1_search_time: float,
    speaker_2_name: str,
    speaker_2_memories: str,
    speaker_2_search_time: float,
) -> dict[str, Any]:
    context_text = (
        f"Memories for user {speaker_1_name}:\n{speaker_1_memories or 'No relevant memories found.'}\n\n"
        f"Memories for user {speaker_2_name}:\n{speaker_2_memories or 'No relevant memories found.'}"
    )
    prepared: dict[str, Any] = {
        "sample_id": sample_id,
        "qa_id": qa_id,
        "q": question,
        "answer": answer,
        "category": category,
        "method": "amem",
        "d": [speaker_1_memories, speaker_2_memories],
        "speaker_memory": {
            "speaker_1": {
                "name": speaker_1_name,
                "memory": speaker_1_memories,
                "search_time": speaker_1_search_time,
            },
            "speaker_2": {
                "name": speaker_2_name,
                "memory": speaker_2_memories,
                "search_time": speaker_2_search_time,
            },
        },
    }
    answer_prompt_payload = build_amem_answer_prompt(
        context_text=context_text,
        question=question,
        category=category,
        answer=str(answer or ""),
        sample_id=sample_id,
        qa_id=qa_id,
    )
    prepared["prompt"] = str(answer_prompt_payload["prompt"])
    prepared["answer_temperature"] = float(answer_prompt_payload.get("temperature", 0.7))
    if isinstance(answer_prompt_payload.get("cat5_answer_key"), dict):
        prepared["cat5_answer_key"] = answer_prompt_payload["cat5_answer_key"]
    return prepared


def prepare_amem_dataset(
    dataset_path: str,
    output_dir: str,
    model_name: str,
    limit: int | None = None,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    dataset_file = Path(dataset_path)
    retrieved_payload = retrieve_amem_dataset(
        dataset_path=dataset_path,
        output_dir=output_dir,
        model_name=model_name,
        limit=limit,
    )
    source_path = str(retrieved_payload.get("summary", {}).get("source_path", dataset_file))
    source_file = Path(source_path)
    retrieved_file = Path(str(retrieved_payload.get("summary", {}).get("retrieved_file", dataset_file)))
    out_file = prepared_output_path(output_dir, "amem", source_file)
    entries = load_entries(retrieved_file, limit=limit)

    total_questions = len(entries) if entries else count_amem_questions(dataset_file, limit=limit)
    existing_map = load_existing_result_map(out_file)
    prepared_rows = load_existing_results(out_file)
    pbar = tqdm(total=total_questions, desc="locomo/compose/amem", unit="q")

    try:
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            if entry_key(entry) in existing_map:
                pbar.update(1)
                continue
            speaker_memory = entry.get("speaker_memory", {})
            speaker_1 = speaker_memory.get("speaker_1", {}) if isinstance(speaker_memory, dict) else {}
            speaker_2 = speaker_memory.get("speaker_2", {}) if isinstance(speaker_memory, dict) else {}
            d_list = entry.get("d", [])
            speaker_1_memories = str(d_list[0]) if isinstance(d_list, list) and len(d_list) > 0 else ""
            speaker_2_memories = str(d_list[1]) if isinstance(d_list, list) and len(d_list) > 1 else ""
            prepared = _prepare_amem_row(
                sample_id=str(entry.get("sample_id", "")),
                qa_id=str(entry.get("qa_id", "")),
                question=str(entry.get("q", "")).strip(),
                answer=entry.get("answer"),
                category=int(entry.get("category", -1)),
                speaker_1_name=str(speaker_1.get("name", "speaker_1")),
                speaker_1_memories=speaker_1_memories,
                speaker_1_search_time=float(speaker_1.get("search_time", 0.0) or 0.0),
                speaker_2_name=str(speaker_2.get("name", "speaker_2")),
                speaker_2_memories=speaker_2_memories,
                speaker_2_search_time=float(speaker_2.get("search_time", 0.0) or 0.0),
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
