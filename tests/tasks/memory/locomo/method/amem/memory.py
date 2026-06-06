from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm

from utils.utils import window_tag


def _iter_amem_samples(payload: Any) -> list[tuple[str, dict[str, Any]]]:
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


def _format_memory_message(turn: dict[str, Any]) -> str:
    timestamp = str(turn.get("timestamp", turn.get("date_time", ""))).strip()
    speaker = str(turn.get("speaker", "")).strip() or "Unknown"
    text = str(turn.get("text", "")).strip()
    if timestamp:
        return f"{timestamp} | {speaker}: {text}"
    return f"{speaker}: {text}"


def _count_speaker_turns(chat_history: list[dict[str, Any]], speaker_name: str) -> int:
    total = 0
    for turn in chat_history:
        if not isinstance(turn, dict):
            continue
        if str(turn.get("speaker", "")).strip() != speaker_name:
            continue
        if _format_memory_message(turn):
            total += 1
    return total


def _build_memory_id(sample_id: str, speaker_name: str, turn_idx: int, turn: dict[str, Any]) -> str:
    raw = json.dumps(
        {
            "method": "amem",
            "sample_id": sample_id,
            "speaker": speaker_name,
            "turn_idx": turn_idx,
            "timestamp": str(turn.get("timestamp", turn.get("date_time", ""))).strip(),
            "text": str(turn.get("text", "")).strip(),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return f"amem_{hashlib.md5(raw.encode('utf-8')).hexdigest()}"


def _namespace(sample_id: str, speaker_tag: str, speaker_name: str) -> str:
    return f"{sample_id}::{speaker_tag}::{speaker_name}"


def load_amem_memory_system(output_dir: str, sample_id: str, speaker_tag: str, speaker_name: str, model_name: str):
    from .memory_layer import AgenticMemorySystem

    return AgenticMemorySystem(
        model_name="qwen3-embedding:0.6b",
        llm_backend="ollama",
        llm_model=model_name,
        output_dir=output_dir,
        namespace=_namespace(sample_id, speaker_tag, speaker_name),
    )


def _add_sample_memories(
    memory_system: Any,
    chat_history: list[dict[str, Any]],
    sample_id: str,
    speaker_name: str,
    memory_pbar: Any | None = None,
) -> int:
    debug_log = os.getenv("AMEM_DEBUG_LOG", "0") == "1"
    added = 0
    for turn_idx, turn in enumerate(chat_history):
        if not isinstance(turn, dict):
            continue
        if str(turn.get("speaker", "")).strip() != speaker_name:
            continue
        message = _format_memory_message(turn)
        if not message:
            continue
        timestamp = str(turn.get("timestamp", turn.get("date_time", ""))).strip() or None
        memory_id = _build_memory_id(sample_id=sample_id, speaker_name=speaker_name, turn_idx=turn_idx, turn=turn)
        before = len(memory_system.memories)
        started_at = time.perf_counter()
        if debug_log:
            print(
                f"[amem/memory] add_note start sample={sample_id} speaker={speaker_name} turn_idx={turn_idx} memory_id={memory_id}",
                flush=True,
            )
        memory_system.add_note(message, time=timestamp, id=memory_id)
        elapsed_ms = (time.perf_counter() - started_at) * 1000
        if debug_log:
            print(
                f"[amem/memory] add_note done sample={sample_id} speaker={speaker_name} turn_idx={turn_idx} memory_id={memory_id} elapsed_ms={elapsed_ms:.1f}",
                flush=True,
            )
        if len(memory_system.memories) > before:
            added += 1
        if memory_pbar is not None:
            memory_pbar.update(1)
    return added


def build_amem_memory_dataset(
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
    samples = _iter_amem_samples(raw_payload)
    if limit is not None and limit > 0:
        samples = samples[:limit]

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
        speaker_1_name, speaker_2_name = speakers[0], speakers[1]
        speaker_1_system = load_amem_memory_system(output_dir, sample_id, "speaker_1", speaker_1_name, model_name)
        speaker_2_system = load_amem_memory_system(output_dir, sample_id, "speaker_2", speaker_2_name, model_name)
        memory_total = _count_speaker_turns(chat_history, speaker_1_name) + _count_speaker_turns(chat_history, speaker_2_name)
        memory_pbar = (
            tqdm(total=memory_total, desc=f"locomo/memory/amem/{sample_id}", unit="turn", leave=True)
            if memory_total > 0
            else None
        )
        try:
            added_1 = _add_sample_memories(speaker_1_system, chat_history, sample_id, speaker_1_name, memory_pbar)
            added_2 = _add_sample_memories(speaker_2_system, chat_history, sample_id, speaker_2_name, memory_pbar)
        finally:
            if memory_pbar is not None:
                memory_pbar.close()
        total_added += added_1 + added_2
        rows.append(
            {
                "sample_id": sample_id,
                "speaker_1": speaker_1_name,
                "speaker_2": speaker_2_name,
                "added_memories": added_1 + added_2,
                "total_memories": len(speaker_1_system.memories) + len(speaker_2_system.memories),
            }
        )
    return {
        "summary": {
            "method": "amem",
            "stage": "memory",
            "source_path": str(dataset_file),
            "memory_dir": str(Path(output_dir) / "memory"),
            "window": window_tag(dataset_file),
            "amem_model_name": model_name,
            "total": len(rows),
            "added_memories": total_added,
        },
        "entries": rows,
    }
