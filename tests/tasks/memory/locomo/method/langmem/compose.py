from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from tqdm import tqdm

from prompt.langmem import LOCOMO_LANGMEM_PROMPT, LOCOMO_LANGMEM_PROMPT_CAT_5
from utils.qa_utils import build_cat5_choice_question, build_question_text
from utils.utils import (
    coerce_category,
    entry_key,
    load_entries,
    load_existing_result_map,
    load_existing_results,
    prepared_output_path,
    safe_write_json,
    window_tag,
)
from .retrieval import count_langmem_questions, retrieve_langmem_dataset


def _prepared_payload(
    rows: list[dict[str, Any]],
    dataset_file: Path,
    out_file: Path,
    model_name: str,
    embedding_model_name: str,
    elapsed_seconds: float | None = None,
) -> dict[str, Any]:
    summary = {
        "method": "langmem",
        "window": window_tag(dataset_file),
        "source_path": str(dataset_file),
        "prepared_file": str(out_file),
        "langmem_model_name": model_name,
        "langmem_embedding_model_name": embedding_model_name,
        "total": len(rows),
    }
    if elapsed_seconds is not None:
        summary["elapsed_seconds"] = round(float(elapsed_seconds), 4)
    return {
        "summary": summary,
        "results": rows,
    }


def _prepare_langmem_row(
    sample_id: str,
    qa_id: str,
    question: str,
    answer: Any,
    category: Any,
    speaker_1_name: str,
    speaker_1_memories: str,
    speaker_1_search_time: float,
    speaker_2_name: str,
    speaker_2_memories: str,
    speaker_2_search_time: float,
) -> dict[str, Any]:
    category_int = coerce_category(category)
    prepared: dict[str, Any] = {
        "sample_id": sample_id,
        "qa_id": qa_id,
        "q": question,
        "answer": answer,
        "category": category_int,
        "method": "langmem",
    }

    if category_int == 5:
        cat5_question, cat5_answer_key = build_cat5_choice_question(
            question,
            str(answer or ""),
            sample_id=sample_id,
            qa_id=qa_id,
        )
        prepared["cat5_answer_key"] = cat5_answer_key
        prepared["prompt"] = LOCOMO_LANGMEM_PROMPT_CAT_5.format(
            speaker_1_user_id=speaker_1_name,
            speaker_1_memories=speaker_1_memories or "No relevant memories found.",
            speaker_2_user_id=speaker_2_name,
            speaker_2_memories=speaker_2_memories or "No relevant memories found.",
            question=cat5_question,
        )
    else:
        prepared["prompt"] = LOCOMO_LANGMEM_PROMPT.format(
            speaker_1_user_id=speaker_1_name,
            speaker_1_memories=speaker_1_memories or "No relevant memories found.",
            speaker_2_user_id=speaker_2_name,
            speaker_2_memories=speaker_2_memories or "No relevant memories found.",
            question=build_question_text(question, category_int),
        )
    return prepared


def prepare_langmem_dataset(
    dataset_path: str,
    output_dir: str,
    model_name: str,
    limit: int | None = None,
    embedding_model_name: str = "qwen3-embedding:0.6b",
) -> dict[str, Any]:
    started_at = time.perf_counter()
    dataset_file = Path(dataset_path)
    retrieved_payload = retrieve_langmem_dataset(
        dataset_path=dataset_path,
        output_dir=output_dir,
        model_name=model_name,
        limit=limit,
        embedding_model_name=embedding_model_name,
    )
    source_path = str(retrieved_payload.get("summary", {}).get("source_path", dataset_file))
    source_file = Path(source_path)
    retrieved_file = Path(str(retrieved_payload.get("summary", {}).get("retrieved_file", dataset_file)))
    out_file = prepared_output_path(output_dir, "langmem", source_file)
    entries = load_entries(retrieved_file, limit=limit)

    total_questions = len(entries) if entries else count_langmem_questions(dataset_file, limit=limit)

    existing_map = load_existing_result_map(out_file)
    prepared_rows = load_existing_results(out_file)
    pbar = tqdm(total=total_questions, desc="locomo/compose/langmem", unit="q")

    try:
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            if entry_key(entry) in existing_map:
                pbar.update(1)
                continue
            speakers = entry.get("speakers", {})
            search_time = entry.get("search_time", {})
            speaker_1_name = str(speakers.get("speaker_1", "speaker_1")) if isinstance(speakers, dict) else "speaker_1"
            speaker_2_name = str(speakers.get("speaker_2", "speaker_2")) if isinstance(speakers, dict) else "speaker_2"
            d_list = entry.get("d", [])
            speaker_1_memories = str(d_list[0]) if isinstance(d_list, list) and len(d_list) > 0 else ""
            speaker_2_memories = str(d_list[1]) if isinstance(d_list, list) and len(d_list) > 1 else ""
            prepared = _prepare_langmem_row(
                sample_id=str(entry.get("sample_id", "")),
                qa_id=str(entry.get("qa_id", "")),
                question=str(entry.get("q", "")).strip(),
                answer=entry.get("answer"),
                category=coerce_category(entry.get("category")),
                speaker_1_name=speaker_1_name,
                speaker_1_memories=speaker_1_memories,
                speaker_1_search_time=float(search_time.get("speaker_1", 0.0) or 0.0)
                if isinstance(search_time, dict)
                else 0.0,
                speaker_2_name=speaker_2_name,
                speaker_2_memories=speaker_2_memories,
                speaker_2_search_time=float(search_time.get("speaker_2", 0.0) or 0.0)
                if isinstance(search_time, dict)
                else 0.0,
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
                    embedding_model_name=embedding_model_name,
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
        embedding_model_name=embedding_model_name,
        elapsed_seconds=time.perf_counter() - started_at,
    )
    safe_write_json(out_file, payload)
    return payload
