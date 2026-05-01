from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .memory import build_amem_memory_dataset, load_amem_memory_system
from prompt.amem import build_amem_generate_query_prompt, build_amem_relevant_parts_prompt
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


def _iter_amem_samples(payload: Any) -> list[tuple[str, dict[str, Any]]]:
    if isinstance(payload, dict):
        return [(str(key), value) for key, value in payload.items() if isinstance(value, dict)]
    if isinstance(payload, list):
        items: list[tuple[str, dict[str, Any]]] = []
        for idx, value in enumerate(payload):
            if isinstance(value, dict):
                items.append((str(value.get("sample_id", idx)), value))
        return items
    return []


def count_amem_questions(dataset_path: str | Path, limit: int | None = None) -> int:
    dataset_file = Path(dataset_path)
    if not dataset_file.exists():
        return 0
    try:
        payload = json.loads(dataset_file.read_text(encoding="utf-8"))
    except Exception:
        return 0
    samples = _iter_amem_samples(payload)
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
        "method": "amem",
        "stage": "retrieve",
        "window": window_tag(dataset_file),
        "source_path": str(dataset_file),
        "retrieved_file": str(out_file),
        "amem_model_name": model_name,
        "total": len(rows),
    }
    if elapsed_seconds is not None:
        summary["elapsed_seconds"] = round(float(elapsed_seconds), 4)
    return {"summary": summary, "entries": rows}


def _ordered_speakers(chat_history: list[dict[str, Any]]) -> list[str]:
    speakers: list[str] = []
    for turn in chat_history:
        if not isinstance(turn, dict):
            continue
        speaker = str(turn.get("speaker", "")).strip()
        if speaker and speaker not in speakers:
            speakers.append(speaker)
    return speakers


def _format_memory_message(turn: dict[str, Any]) -> str:
    timestamp = str(turn.get("timestamp", turn.get("date_time", ""))).strip()
    speaker = str(turn.get("speaker", "")).strip() or "Unknown"
    text = str(turn.get("text", "")).strip()
    if timestamp:
        return f"{timestamp} | {speaker}: {text}"
    return f"{speaker}: {text}"


def _count_speaker_turns(chat_history: list[dict[str, Any]], speaker_name: str) -> int:
    count = 0
    for turn in chat_history:
        if not isinstance(turn, dict):
            continue
        if str(turn.get("speaker", "")).strip() != speaker_name:
            continue
        message = _format_memory_message(turn)
        if message:
            count += 1
    return count


def _extract_json_block(raw_text: str) -> dict[str, Any] | None:
    text = str(raw_text or "").strip()
    if not text:
        return None
    if not text.startswith("{"):
        start_idx = text.find("{")
        if start_idx >= 0:
            text = text[start_idx:]
    if not text.endswith("}"):
        end_idx = text.rfind("}")
        if end_idx >= 0:
            text = text[: end_idx + 1]
    try:
        payload = json.loads(text)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _search_memories(memory_system: Any, query: str, retrieve_k: int = 10) -> tuple[str, float]:
    started_at = time.perf_counter()
    try:
        memories = memory_system.find_related_memories_raw(query, k=retrieve_k)
        return str(memories or "").strip(), round(time.perf_counter() - started_at, 4)
    except Exception as exc:
        return f"[ERROR] {type(exc).__name__}: {exc}", round(time.perf_counter() - started_at, 4)


def _generate_query_keywords(memory_system: Any, question: str) -> str:
    prompt = build_amem_generate_query_prompt(question)
    try:
        raw = memory_system.llm_controller.llm.get_completion(prompt, temperature=0.2)
    except Exception:
        return question

    payload = _extract_json_block(raw)
    if payload is not None and isinstance(payload.get("keywords"), str):
        val = str(payload["keywords"]).strip()
        return val if val else question
    val = str(raw).strip()
    return val if val else question


def _select_relevant_parts(memory_system: Any, memories_text: str, query: str) -> str:
    prompt = build_amem_relevant_parts_prompt(memories_text=memories_text, query=query)
    try:
        raw = memory_system.llm_controller.llm.get_completion(prompt, temperature=0.2)
    except Exception:
        return memories_text

    payload = _extract_json_block(raw)
    if payload is not None and isinstance(payload.get("relevant_parts"), str):
        val = str(payload["relevant_parts"]).strip()
        return val if val else memories_text
    val = str(raw).strip()
    return val if val else memories_text


def retrieve_amem_dataset(
    dataset_path: str,
    output_dir: str,
    model_name: str,
    output_path: str | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    dataset_file = Path(dataset_path)
    if output_path:
        out_file = Path(output_path)
    else:
        out_file = retrieved_output_path(output_dir, "amem", dataset_file)

    # If this is already a retrieved entries file, just normalize and return it.
    existing_entries = load_entries(dataset_file, limit=limit)
    try:
        raw_payload = json.loads(dataset_file.read_text(encoding="utf-8"))
    except Exception:
        raw_payload = None
    summary = raw_payload.get("summary", {}) if isinstance(raw_payload, dict) else {}
    if isinstance(summary, dict) and summary.get("stage") == "retrieve" and existing_entries:
        payload = {
            "summary": {
                **summary,
                "retrieved_file": str(dataset_file),
                "total": len(existing_entries),
            },
            "entries": existing_entries,
        }
        safe_write_json(out_file, payload)
        return payload

    build_amem_memory_dataset(
        dataset_path=dataset_path,
        output_dir=output_dir,
        model_name=model_name,
        limit=limit,
    )

    samples = _iter_amem_samples(raw_payload)
    if limit is not None and limit > 0:
        samples = samples[:limit]

    total_questions = count_amem_questions(dataset_file, limit=limit)
    existing_map = load_existing_result_map(out_file)
    retrieved_rows = load_existing_results(out_file)
    pbar = tqdm(total=total_questions, desc="locomo/retrieve/amem", unit="q")

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

            speakers = _ordered_speakers(chat_history)
            if len(speakers) != 2:
                for _ in pending_questions:
                    pbar.update(1)
                continue

            speaker_1_name, speaker_2_name = speakers[0], speakers[1]
            speaker_1_system = load_amem_memory_system(output_dir, sample_id, "speaker_1", speaker_1_name, model_name)
            speaker_2_system = load_amem_memory_system(output_dir, sample_id, "speaker_2", speaker_2_name, model_name)

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

                query_for_retrieval_1 = _generate_query_keywords(speaker_1_system, question)
                raw_speaker_1_memories, speaker_1_search_time = _search_memories(speaker_1_system, query_for_retrieval_1)
                selected_speaker_1_memories = _select_relevant_parts(
                    speaker_1_system, raw_speaker_1_memories, question
                )

                query_for_retrieval_2 = _generate_query_keywords(speaker_2_system, question)
                raw_speaker_2_memories, speaker_2_search_time = _search_memories(speaker_2_system, query_for_retrieval_2)
                selected_speaker_2_memories = _select_relevant_parts(
                    speaker_2_system, raw_speaker_2_memories, question
                )

                row = {
                    "sample_id": sample_id,
                    "qa_id": qa_id,
                    "q": question,
                    "answer": question_item.get("answer"),
                    "category": coerce_category(question_item.get("category")),
                    "method": "amem",
                    "retrieval_query": {
                        "speaker_1": query_for_retrieval_1,
                        "speaker_2": query_for_retrieval_2,
                    },
                    "d": [selected_speaker_1_memories, selected_speaker_2_memories],
                    "speakers": {
                        "speaker_1": speaker_1_name,
                        "speaker_2": speaker_2_name,
                    },
                    "search_time": {
                        "speaker_1": speaker_1_search_time,
                        "speaker_2": speaker_2_search_time,
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
