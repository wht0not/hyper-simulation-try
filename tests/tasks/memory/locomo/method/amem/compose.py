from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from tqdm import tqdm

from prompt.amem import (
    build_amem_answer_prompt,
)
from utils.qa_utils import build_question_text, resolve_qa_answer
from utils.utils import (
    coerce_category,
    entry_key,
    load_entries,
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
    speaker_2_name: str,
    speaker_2_memories: str,
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
    }
    answer_prompt_payload = build_amem_answer_prompt(
        context_text=context_text,
        question=build_question_text(question, category),
    )
    prepared["prompt"] = str(answer_prompt_payload["prompt"])
    prepared["answer_temperature"] = float(answer_prompt_payload.get("temperature", 0.1))
    return prepared


def prepare_amem_dataset(
    dataset_path: str,
    output_dir: str,
    model_name: str,
    limit: int | None = None,
    skip_retrieve: bool = False,
    checkpoint_every: int = 500,
    retrieve_k: int = 10,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    dataset_file = Path(dataset_path)
    if skip_retrieve:
        source_file = dataset_file
        retrieved_file = dataset_file
    else:
        retrieved_payload = retrieve_amem_dataset(
            dataset_path=dataset_path,
            output_dir=output_dir,
            model_name=model_name,
            limit=limit,
            retrieve_k=retrieve_k,
        )
        source_path = str(retrieved_payload.get("summary", {}).get("source_path", dataset_file))
        source_file = Path(source_path)
        retrieved_file = Path(str(retrieved_payload.get("summary", {}).get("retrieved_file", dataset_file)))
    out_file = prepared_output_path(output_dir, "amem", source_file)
    entries = load_entries(retrieved_file, limit=limit)

    total_questions = len(entries) if entries else count_amem_questions(dataset_file, limit=limit)
    prepared_rows = [
        row
        for row in load_existing_results(out_file)
        if coerce_category(row.get("category", -1)) != 5
    ]
    existing_map = {entry_key(row): row for row in prepared_rows if entry_key(row)}
    checkpoint_every = max(1, int(checkpoint_every))
    pending_writes = 0
    pbar = tqdm(total=total_questions, desc="locomo/compose/amem", unit="q")

    try:
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            if coerce_category(entry.get("category", -1)) == 5:
                pbar.update(1)
                continue
            if entry_key(entry) in existing_map:
                pbar.update(1)
                continue
            speakers = entry.get("speakers", {})
            speaker_1_name = str(speakers.get("speaker_1", "speaker_1")) if isinstance(speakers, dict) else "speaker_1"
            speaker_2_name = str(speakers.get("speaker_2", "speaker_2")) if isinstance(speakers, dict) else "speaker_2"
            d_list = entry.get("d", [])
            speaker_1_memories = str(d_list[0]) if isinstance(d_list, list) and len(d_list) > 0 else ""
            speaker_2_memories = str(d_list[1]) if isinstance(d_list, list) and len(d_list) > 1 else ""
            item_started_at = time.perf_counter()
            prepared = _prepare_amem_row(
                sample_id=str(entry.get("sample_id", "")),
                qa_id=str(entry.get("qa_id", "")),
                question=str(entry.get("q", "")).strip(),
                answer=resolve_qa_answer(entry),
                category=coerce_category(entry.get("category", -1)),
                speaker_1_name=speaker_1_name,
                speaker_1_memories=speaker_1_memories,
                speaker_2_name=speaker_2_name,
                speaker_2_memories=speaker_2_memories,
            )
            prepared["prepared_elapsed_seconds"] = round(time.perf_counter() - item_started_at, 6)
            prepared_rows.append(prepared)
            existing_map[entry_key(prepared)] = prepared
            pending_writes += 1
            if pending_writes >= checkpoint_every:
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
                pending_writes = 0
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
