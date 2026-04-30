from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from langchain_ollama import ChatOllama
from tqdm import tqdm

from prompt.memorybank import (
    build_memorybank_overall_history_prompt,
    build_memorybank_overall_personality_prompt,
    build_memorybank_personality_summary_prompt,
    build_memorybank_session_summary_prompt,
)
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


def _memory_cache_file(output_dir: str, sample_id: str) -> Path:
    return Path(output_dir) / "memory" / f"{sample_id}.json"


def _conversation_digest(chat_history: list[dict[str, Any]]) -> str:
    payload = json.dumps(chat_history, ensure_ascii=False, sort_keys=True)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def _question_cache_key(qa_id: str, question: str) -> str:
    return f"{qa_id}::{question}"


def _load_sample_memory_cache(
    output_dir: str,
    sample_id: str,
    chat_history: list[dict[str, Any]],
    model_name: str,
) -> dict[str, Any]:
    cache_file = _memory_cache_file(output_dir, sample_id)
    if not cache_file.exists():
        return {}
    try:
        payload = json.loads(cache_file.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    meta = payload.get("meta", {})
    if not isinstance(meta, dict):
        return {}
    if meta.get("conversation_digest") != _conversation_digest(chat_history):
        return {}
    if str(meta.get("model_name", "")) != str(model_name):
        return {}
    return payload


def _save_sample_memory_cache(
    output_dir: str,
    sample_id: str,
    chat_history: list[dict[str, Any]],
    model_name: str,
    summary_rows: list[dict[str, Any]],
    personality_rows: list[dict[str, Any]],
    overall_history: str,
    overall_personality: str,
    question_cache: dict[str, Any],
) -> None:
    cache_file = _memory_cache_file(output_dir, sample_id)
    payload = {
        "meta": {
            "method": "memorybank",
            "sample_id": sample_id,
            "model_name": model_name,
            "conversation_digest": _conversation_digest(chat_history),
        },
        "memory_payload": {
            "summary_by_date": summary_rows,
            "personality_by_date": personality_rows,
            "overall_history": overall_history,
            "overall_personality": overall_personality,
        },
        "question_cache": question_cache,
    }
    safe_write_json(cache_file, payload)


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


def _ordered_speakers(chat_history: list[dict[str, Any]]) -> list[str]:
    speakers: list[str] = []
    for turn in chat_history:
        if not isinstance(turn, dict):
            continue
        speaker = str(turn.get("speaker", "")).strip()
        if speaker and speaker not in speakers:
            speakers.append(speaker)
    return speakers


def _group_turns_by_date(chat_history: list[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    order: list[str] = []
    for turn in chat_history:
        if not isinstance(turn, dict):
            continue
        timestamp = str(turn.get("timestamp", turn.get("date_time", ""))).strip() or "unknown_date"
        if timestamp not in grouped:
            grouped[timestamp] = []
            order.append(timestamp)
        grouped[timestamp].append(turn)
    return [(date, grouped[date]) for date in order]


def _session_dialogue_text(turns: list[dict[str, Any]], user_name: str, ai_name: str) -> str:
    lines: list[str] = []
    for turn in turns:
        speaker = str(turn.get("speaker", "")).strip()
        text = str(turn.get("text", "")).strip()
        if not text:
            continue
        speaker_name = user_name if speaker == user_name else ai_name if speaker == ai_name else speaker or "Unknown"
        lines.append(f"{speaker_name}: {text}")
    return "\n".join(lines).strip()


def _call_text_llm(model: ChatOllama, prompt: str, temperature: float = 0.1) -> str:
    try:
        response = model.invoke(prompt, temperature=temperature)
        return str(getattr(response, "content", response) or "").strip()
    except TypeError:
        response = ChatOllama(model=model.model, temperature=temperature, reasoning=False, num_predict=4096).invoke(prompt)
        return str(getattr(response, "content", response) or "").strip()
    except Exception:
        return ""


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

    llm = ChatOllama(model=model_name, temperature=0.1, reasoning=False, num_predict=4096)
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

            speakers = _ordered_speakers(chat_history)
            if len(speakers) != 2:
                for _ in pending_questions:
                    pbar.update(1)
                continue
            user_name, ai_name = speakers[0], speakers[1]

            dated_turns = _group_turns_by_date(chat_history)
            cache_payload = _load_sample_memory_cache(
                output_dir=output_dir,
                sample_id=sample_id,
                chat_history=chat_history,
                model_name=model_name,
            )
            memory_payload = cache_payload.get("memory_payload", {}) if isinstance(cache_payload, dict) else {}
            question_cache = cache_payload.get("question_cache", {}) if isinstance(cache_payload, dict) else {}
            if not isinstance(memory_payload, dict):
                memory_payload = {}
            if not isinstance(question_cache, dict):
                question_cache = {}
            cache_dirty = False

            summary_rows = memory_payload.get("summary_by_date", [])
            personality_rows = memory_payload.get("personality_by_date", [])
            overall_history = str(memory_payload.get("overall_history", ""))
            overall_personality = str(memory_payload.get("overall_personality", ""))
            if not isinstance(summary_rows, list):
                summary_rows = []
            if not isinstance(personality_rows, list):
                personality_rows = []

            has_memory_payload = bool(summary_rows) and bool(personality_rows)
            if not has_memory_payload:
                summary_rows = []
                personality_rows = []
                memory_pbar = (
                    tqdm(
                        total=len(dated_turns) * 2,
                        desc=f"locomo/memory/memorybank/{sample_id}",
                        unit="step",
                        leave=False,
                    )
                    if dated_turns
                    else None
                )
                try:
                    for date, turns in dated_turns:
                        dialogue_text = _session_dialogue_text(turns, user_name=user_name, ai_name=ai_name)
                        if not dialogue_text:
                            if memory_pbar is not None:
                                memory_pbar.update(2)
                            continue
                        session_summary = _call_text_llm(llm, build_memorybank_session_summary_prompt(dialogue_text))
                        if memory_pbar is not None:
                            memory_pbar.update(1)
                        personality_summary = _call_text_llm(
                            llm,
                            build_memorybank_personality_summary_prompt(dialogue_text, user_name=user_name),
                        )
                        if memory_pbar is not None:
                            memory_pbar.update(1)
                        summary_rows.append(
                            {
                                "date": date,
                                "content": session_summary or dialogue_text,
                                "raw_dialogue": dialogue_text,
                            }
                        )
                        personality_rows.append(
                            {
                                "date": date,
                                "content": personality_summary,
                            }
                        )
                finally:
                    if memory_pbar is not None:
                        memory_pbar.close()

                dated_summaries = "\n".join(
                    f"At {row['date']}, the events are {row['content']}"
                    for row in summary_rows
                    if str(row.get("content", "")).strip()
                )
                dated_personality = "\n".join(
                    f"At {row['date']}, the analysis shows {row['content']}"
                    for row in personality_rows
                    if str(row.get("content", "")).strip()
                )
                overall_history = (
                    _call_text_llm(llm, build_memorybank_overall_history_prompt(dated_summaries))
                    if dated_summaries
                    else ""
                )
                overall_personality = (
                    _call_text_llm(llm, build_memorybank_overall_personality_prompt(dated_personality))
                    if dated_personality
                    else ""
                )
                cache_dirty = True

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

                cache_key = _question_cache_key(qa_id, question)
                cached_item = question_cache.get(cache_key)
                if isinstance(cached_item, dict):
                    d_list = cached_item.get("d", [])
                    retrieval_time = float(cached_item.get("retrieval_time", 0.0) or 0.0)
                    if not isinstance(d_list, list):
                        d_list = []
                    d_list = [str(item) for item in d_list]
                else:
                    retrieve_started_at = time.perf_counter()
                    top_summary_rows = _top_k_summaries(question, summary_rows, top_k=3)
                    retrieval_time = round(time.perf_counter() - retrieve_started_at, 4)
                    d_list = [
                        f"Date: {row['date']}\nSummary: {row['content']}\nDialogue: {row['raw_dialogue']}"
                        for row in top_summary_rows
                    ]
                    question_cache[cache_key] = {
                        "d": d_list,
                        "retrieval_time": retrieval_time,
                    }
                    cache_dirty = True
                related_memory = "\n\n".join(d_list).strip()

                row = {
                    "sample_id": sample_id,
                    "qa_id": qa_id,
                    "q": question,
                    "answer": question_item.get("answer"),
                    "category": coerce_category(question_item.get("category")),
                    "method": "memorybank",
                    "d": d_list,
                    "memorybank_memory": {
                        "user_name": user_name,
                        "ai_name": ai_name,
                        "overall_history": overall_history,
                        "overall_personality": overall_personality,
                        "related_memory": related_memory,
                        "summary_by_date": summary_rows,
                        "personality_by_date": personality_rows,
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
            if cache_dirty:
                _save_sample_memory_cache(
                    output_dir=output_dir,
                    sample_id=sample_id,
                    chat_history=chat_history,
                    model_name=model_name,
                    summary_rows=summary_rows,
                    personality_rows=personality_rows,
                    overall_history=overall_history,
                    overall_personality=overall_personality,
                    question_cache=question_cache,
                )
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
