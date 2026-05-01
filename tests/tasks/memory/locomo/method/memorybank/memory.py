from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from langchain_ollama import ChatOllama
from tqdm import tqdm

from prompt.memorybank import (
    build_memorybank_overall_history_prompt,
    build_memorybank_overall_personality_prompt,
    build_memorybank_personality_summary_prompt,
    build_memorybank_session_summary_prompt,
)
from utils.utils import safe_write_json, window_tag


def _memory_file(output_dir: str, sample_id: str) -> Path:
    memory_dir = Path(output_dir) / "memory"
    memory_dir.mkdir(parents=True, exist_ok=True)
    safe_sample = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in sample_id)
    return memory_dir / f"memorybank_{safe_sample}.json"


def _iter_memorybank_samples(payload: Any) -> list[tuple[str, dict[str, Any]]]:
    if isinstance(payload, dict):
        return [(str(key), value) for key, value in payload.items() if isinstance(value, dict)]
    if isinstance(payload, list):
        rows: list[tuple[str, dict[str, Any]]] = []
        for idx, value in enumerate(payload):
            if isinstance(value, dict):
                rows.append((str(value.get("sample_id", idx)), value))
        return rows
    return []


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


def load_memorybank_memory_payload(output_dir: str, sample_id: str) -> dict[str, Any]:
    memory_file = _memory_file(output_dir, sample_id)
    if not memory_file.exists():
        return {}
    try:
        payload = json.loads(memory_file.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _save_memorybank_memory_payload(
    output_dir: str,
    sample_id: str,
    user_name: str,
    ai_name: str,
    summary_rows: list[dict[str, Any]],
    personality_rows: list[dict[str, Any]],
    overall_history: str,
    overall_personality: str,
) -> None:
    safe_write_json(
        _memory_file(output_dir, sample_id),
        {
            "method": "memorybank",
            "stage": "memory",
            "memory_id": sample_id,
            "summary": f"memorybank memory for {sample_id}",
            "timestamp": str(int(time.time())),
            "metadata": {"sample_id": sample_id, "user_name": user_name, "ai_name": ai_name},
            "memory_payload": {
                "summary_by_date": summary_rows,
                "personality_by_date": personality_rows,
                "overall_history": str(overall_history or ""),
                "overall_personality": str(overall_personality or ""),
            },
        },
    )


def build_memorybank_memory_dataset(
    dataset_path: str,
    output_dir: str,
    model_name: str,
    limit: int | None = None,
) -> dict[str, Any]:
    dataset_file = Path(dataset_path)
    try:
        raw_payload = json.loads(dataset_file.read_text(encoding="utf-8"))
    except Exception:
        raw_payload = None
    samples = _iter_memorybank_samples(raw_payload)
    if limit is not None and limit > 0:
        samples = samples[:limit]

    llm = ChatOllama(model=model_name, temperature=0.1, reasoning=False, num_predict=4096)
    rows: list[dict[str, Any]] = []
    total_added = 0
    for sample_key, sample in samples:
        chat_history = sample.get("conversation", [])
        if not isinstance(chat_history, list):
            continue
        speakers = _ordered_speakers(chat_history)
        if len(speakers) != 2:
            continue
        sample_id = str(sample.get("sample_id", sample_key)).strip() or str(sample_key)
        user_name, ai_name = speakers[0], speakers[1]
        cached_payload = load_memorybank_memory_payload(output_dir, sample_id)
        memory_payload = cached_payload.get("memory_payload", {}) if isinstance(cached_payload, dict) else {}
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
        dated_turns = _group_turns_by_date(chat_history)
        existing_dates = {str(row.get("date", "")) for row in summary_rows if isinstance(row, dict)}
        memory_pbar = (
            tqdm(total=len(dated_turns) * 2, desc=f"locomo/memory/memorybank/{sample_id}", unit="step", leave=False)
            if dated_turns
            else None
        )
        added = 0
        try:
            for date, turns in dated_turns:
                if date in existing_dates:
                    if memory_pbar is not None:
                        memory_pbar.update(2)
                    continue
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
                        "memory_id": f"{sample_id}::summary::{date}",
                        "date": date,
                        "content": session_summary or dialogue_text,
                        "timestamp": date,
                    }
                )
                personality_rows.append(
                    {
                        "memory_id": f"{sample_id}::personality::{date}",
                        "date": date,
                        "content": personality_summary,
                        "timestamp": date,
                    }
                )
                existing_dates.add(date)
                added += 2
                _save_memorybank_memory_payload(
                    output_dir=output_dir,
                    sample_id=sample_id,
                    user_name=user_name,
                    ai_name=ai_name,
                    summary_rows=summary_rows,
                    personality_rows=personality_rows,
                    overall_history=overall_history,
                    overall_personality=overall_personality,
                )
        finally:
            if memory_pbar is not None:
                memory_pbar.close()

        dated_summaries = "\n".join(
            f"At {row['date']}, the events are {row['content']}"
            for row in summary_rows
            if isinstance(row, dict) and str(row.get("content", "")).strip()
        )
        dated_personality = "\n".join(
            f"At {row['date']}, the analysis shows {row['content']}"
            for row in personality_rows
            if isinstance(row, dict) and str(row.get("content", "")).strip()
        )
        if dated_summaries:
            overall_history = _call_text_llm(llm, build_memorybank_overall_history_prompt(dated_summaries))
        if dated_personality:
            overall_personality = _call_text_llm(llm, build_memorybank_overall_personality_prompt(dated_personality))
        _save_memorybank_memory_payload(
            output_dir=output_dir,
            sample_id=sample_id,
            user_name=user_name,
            ai_name=ai_name,
            summary_rows=summary_rows,
            personality_rows=personality_rows,
            overall_history=overall_history,
            overall_personality=overall_personality,
        )
        total_added += added
        rows.append(
            {
                "sample_id": sample_id,
                "user_name": user_name,
                "ai_name": ai_name,
                "added_memories": added,
                "total_memories": len(summary_rows) + len(personality_rows),
            }
        )
    return {
        "summary": {
            "method": "memorybank",
            "stage": "memory",
            "source_path": str(dataset_file),
            "memory_dir": str(Path(output_dir) / "memory"),
            "window": window_tag(dataset_file),
            "memorybank_model_name": model_name,
            "total": len(rows),
            "added_memories": total_added,
        },
        "entries": rows,
    }
