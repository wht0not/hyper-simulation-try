from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from tqdm import tqdm

from utils.utils import safe_write_json, window_tag


def _iter_langmem_samples(payload: Any) -> list[tuple[str, dict[str, Any]]]:
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
        if _format_memory_message(turn).strip():
            total += 1
    return total


def _build_memory_id(sample_id: str, speaker_name: str, turn_idx: int, turn: dict[str, Any]) -> str:
    raw = json.dumps(
        {
            "method": "langmem",
            "sample_id": sample_id,
            "speaker": speaker_name,
            "turn_idx": turn_idx,
            "timestamp": str(turn.get("timestamp", turn.get("date_time", ""))).strip(),
            "text": str(turn.get("text", "")).strip(),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return f"langmem_{hashlib.md5(raw.encode('utf-8')).hexdigest()}"


def _memory_dir(output_dir: str) -> Path:
    path = Path(output_dir) / "memory"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _memory_file(output_dir: str, sample_id: str, speaker_name: str) -> Path:
    safe_sample = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in sample_id)
    safe_speaker = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in speaker_name)
    return _memory_dir(output_dir) / f"langmem_{safe_sample}_{safe_speaker}.json"


def load_langmem_memories(output_dir: str, sample_id: str, speaker_name: str) -> list[dict[str, Any]]:
    memory_file = _memory_file(output_dir, sample_id, speaker_name)
    if not memory_file.exists():
        return []
    try:
        payload = json.loads(memory_file.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(payload, dict):
        return []
    rows = payload.get("memories", [])
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict)]


def _save_langmem_memories(output_dir: str, sample_id: str, speaker_name: str, rows: list[dict[str, Any]]) -> None:
    safe_write_json(
        _memory_file(output_dir, sample_id, speaker_name),
        {
            "method": "langmem",
            "stage": "memory",
            "memory_id": f"{sample_id}::{speaker_name}",
            "summary": f"langmem memory list for {sample_id}/{speaker_name}",
            "metadata": {"sample_id": sample_id, "speaker": speaker_name},
            "memories": rows,
        },
    )


def build_langmem_memory_dataset(
    dataset_path: str,
    output_dir: str,
    model_name: str,
    limit: int | None = None,
) -> dict[str, Any]:
    _ = model_name
    dataset_file = Path(dataset_path)
    try:
        raw_payload = json.loads(dataset_file.read_text(encoding="utf-8"))
    except Exception:
        raw_payload = None
    samples = _iter_langmem_samples(raw_payload)
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
        memory_total = _count_speaker_turns(chat_history, speaker_1_name) + _count_speaker_turns(chat_history, speaker_2_name)
        memory_pbar = (
            tqdm(total=memory_total, desc=f"locomo/memory/langmem/{sample_id}", unit="turn", leave=False)
            if memory_total > 0
            else None
        )
        added = 0
        for speaker_name in (speaker_1_name, speaker_2_name):
            existing_rows = load_langmem_memories(output_dir, sample_id, speaker_name)
            existing_ids = {
                str(row.get("memory_id", "")).strip()
                for row in existing_rows
                if isinstance(row, dict) and str(row.get("memory_id", "")).strip()
            }
            for turn_idx, turn in enumerate(chat_history):
                if not isinstance(turn, dict):
                    continue
                if str(turn.get("speaker", "")).strip() != speaker_name:
                    continue
                message = _format_memory_message(turn)
                if not message.strip():
                    continue
                memory_id = _build_memory_id(sample_id, speaker_name, turn_idx, turn)
                if memory_id not in existing_ids:
                    existing_rows.append(
                        {
                            "memory_id": memory_id,
                            "memory": message,
                            "timestamp": str(turn.get("timestamp", turn.get("date_time", ""))).strip(),
                        }
                    )
                    existing_ids.add(memory_id)
                    added += 1
                    _save_langmem_memories(output_dir, sample_id, speaker_name, existing_rows)
                if memory_pbar is not None:
                    memory_pbar.update(1)
        if memory_pbar is not None:
            memory_pbar.close()
        total_added += added
        rows.append(
            {
                "sample_id": sample_id,
                "speaker_1": speaker_1_name,
                "speaker_2": speaker_2_name,
                "added_memories": added,
                "total_memories": len(load_langmem_memories(output_dir, sample_id, speaker_1_name))
                + len(load_langmem_memories(output_dir, sample_id, speaker_2_name)),
            }
        )
    return {
        "summary": {
            "method": "langmem",
            "stage": "memory",
            "source_path": str(dataset_file),
            "memory_dir": str(Path(output_dir) / "memory"),
            "window": window_tag(dataset_file),
            "langmem_model_name": model_name,
            "total": len(rows),
            "added_memories": total_added,
        },
        "entries": rows,
    }
