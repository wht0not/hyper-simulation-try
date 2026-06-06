from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[6]
SRC_ROOT = PROJECT_ROOT / "src"
LOCOMO_ROOT = Path(__file__).resolve().parents[2]
for candidate in (SRC_ROOT, PROJECT_ROOT, LOCOMO_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from utils.utils import (
    coerce_category,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RAG_SOURCE_PATH,
    entry_key,
    load_existing_result_map,
    load_existing_results,
    rag_retrieved_output_path,
    safe_write_json,
    window_tag,
)
from utils.qa_utils import resolve_qa_answer


def _get_memory_cache_path(source_file: Path, chunk_size: int) -> Path:
    chunk_size = max(1, int(chunk_size))
    source_hash = hashlib.md5(str(source_file.resolve()).encode()).hexdigest()[:8]
    cache_dir = source_file.parent / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{source_file.stem}_cache_{source_hash}_chunk{chunk_size}.json"


def _load_memory_cache(cache_path: Path) -> dict[str, Any] | None:
    if not cache_path.exists():
        return None
    try:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _save_memory_cache(cache_path: Path, cache_data: dict[str, Any]) -> None:
    safe_write_json(cache_path, cache_data)

CONV_START_PROMPT = (
    "Below is a conversation between two people: {speaker_a} and {speaker_b}. "
    "The conversation takes place over multiple days and the date of each conversation is written at the beginning of the conversation."
)


def _format_turn(timestamp: str, speaker: str, text: str) -> str:
    ts = str(timestamp or "").strip()
    spk = str(speaker or "").strip() or "Unknown"
    body = str(text or "").strip()
    if ts:
        return f"{ts} | {spk}: {body}"
    return f"{spk}: {body}"


def _get_encoding():
    return None


def _token_count(text: str, encoding: Any) -> int:
    _ = encoding
    return len(str(text or ""))


def _chunk_turns(turns: list[str], chunk_size: int) -> tuple[list[str], list[list[int]]]:
    chunk_size = max(1, int(chunk_size))
    if not turns:
        return [], []
    encoding = _get_encoding()
    chunk_texts: list[str] = []
    chunk_turn_indices: list[list[int]] = []
    current_lines: list[str] = []
    current_indices: list[int] = []
    current_tokens = 0
    for idx, line in enumerate(turns):
        line_tokens = max(1, _token_count(line, encoding))
        if current_lines and current_tokens + line_tokens > chunk_size:
            chunk_texts.append("\n".join(current_lines))
            chunk_turn_indices.append(current_indices)
            current_lines = []
            current_indices = []
            current_tokens = 0
        current_lines.append(line)
        current_indices.append(idx)
        current_tokens += line_tokens
    if current_lines:
        chunk_texts.append("\n".join(current_lines))
        chunk_turn_indices.append(current_indices)
    return chunk_texts, chunk_turn_indices


def _top_k_chunks(
    query: str,
    chunk_texts: list[str],
    top_k: int,
    cache: dict[str, Any],
) -> tuple[list[int], list[float]]:
    if not chunk_texts:
        return [], []
    import numpy as np
    from hyper_simulation.component.embedding import get_embedding_batch

    top_k = max(1, min(int(top_k), len(chunk_texts)))
    query_embedding = np.asarray(get_embedding_batch([query], cache=cache)[0], dtype=np.float32)
    chunk_embeddings = np.asarray(get_embedding_batch(chunk_texts, cache=cache), dtype=np.float32)
    scores = np.matmul(chunk_embeddings, query_embedding)
    sorted_indices = np.argsort(scores)[::-1][:top_k]
    top_indices = [int(i) for i in sorted_indices.tolist()]
    top_scores = [float(scores[i]) for i in top_indices]
    return top_indices, top_scores


def _iter_rag_samples(payload: Any) -> list[tuple[str, dict[str, Any]]]:
    if isinstance(payload, dict):
        return [(str(k), v) for k, v in payload.items() if isinstance(v, dict)]
    if isinstance(payload, list):
        items: list[tuple[str, dict[str, Any]]] = []
        for idx, one in enumerate(payload):
            if isinstance(one, dict):
                sid = str(one.get("sample_id", idx))
                items.append((sid, one))
        return items
    return []


def _build_conv_prefix(turns: list[dict[str, Any]]) -> str:
    speakers: list[str] = []
    for turn in turns:
        speaker = str(turn.get("speaker", "")).strip()
        if speaker and speaker not in speakers:
            speakers.append(speaker)
        if len(speakers) >= 2:
            break
    if len(speakers) >= 2:
        return CONV_START_PROMPT.format(speaker_a=speakers[0], speaker_b=speakers[1]).strip()
    if len(speakers) == 1:
        return CONV_START_PROMPT.format(speaker_a=speakers[0], speaker_b=speakers[0]).strip()
    return ""


def retrieve_rag_dataset(
    rag_source_path: str,
    output_path: str | None,
    output_dir: str,
    chunk_size: int = 128,
    top_k: int = 5,
    limit: int | None = None,
) -> dict[str, Any]:
    source_file = Path(rag_source_path)
    if not source_file.exists():
        raise FileNotFoundError(f"rag source file not found: {source_file}")
    if output_path:
        out_file = Path(output_path)
    else:
        out_file = rag_retrieved_output_path(output_dir, source_file, chunk_size, top_k)
    raw_payload = json.loads(source_file.read_text(encoding="utf-8"))
    samples = _iter_rag_samples(raw_payload)
    if limit is not None and limit > 0:
        samples = samples[:limit]

    total_questions = 0
    for _, sample in samples:
        questions_raw = sample.get("question", sample.get("qa", []))
        if isinstance(questions_raw, list):
            total_questions += len([qa for qa in questions_raw if isinstance(qa, dict)])

    existing_map = load_existing_result_map(out_file)
    retrieved_rows = load_existing_results(out_file)
    embedding_cache: dict[str, Any] = {}
    
    cache_path = _get_memory_cache_path(source_file, chunk_size)
    chunk_cache = _load_memory_cache(cache_path) or {}
    cache_dirty = False
    
    pbar = tqdm(total=total_questions, desc="locomo/retrieve/rag", unit="q")

    try:
        for sample_id, sample in samples:
            turns_raw = sample.get("conversation", [])
            questions_raw = sample.get("question", sample.get("qa", []))
            if not isinstance(turns_raw, list) or not isinstance(questions_raw, list):
                continue
            
            d_start = _build_conv_prefix(turns_raw)
            turns = [
                _format_turn(
                    timestamp=str(turn.get("timestamp", turn.get("date_time", ""))),
                    speaker=str(turn.get("speaker", "")),
                    text=str(turn.get("text", "")),
                )
                for turn in turns_raw
                if isinstance(turn, dict)
            ]
            
            if sample_id in chunk_cache:
                chunk_texts = chunk_cache[sample_id]["chunk_texts"]
                chunk_turn_indices = chunk_cache[sample_id]["chunk_turn_indices"]
            else:
                chunk_texts, chunk_turn_indices = _chunk_turns(turns, chunk_size=chunk_size)
                chunk_cache[sample_id] = {
                    "chunk_texts": chunk_texts,
                    "chunk_turn_indices": chunk_turn_indices
                }
                cache_dirty = True
            
            if not chunk_texts:
                for qa in questions_raw:
                    if isinstance(qa, dict):
                        pbar.update(1)
                continue

            for qa_idx, qa in enumerate(questions_raw):
                if not isinstance(qa, dict):
                    continue
                q = str(qa.get("question", "")).strip()
                if not q:
                    pbar.update(1)
                    continue
                qa_id = str(qa.get("qa_id", qa_idx))
                row_stub = {"sample_id": sample_id, "qa_id": qa_id, "q": q}
                if entry_key(row_stub) in existing_map:
                    pbar.update(1)
                    continue
                top_indices, top_scores = _top_k_chunks(
                    query=q,
                    chunk_texts=chunk_texts,
                    top_k=top_k,
                    cache=embedding_cache,
                )
                retrieved_rows.append(
                    {
                        "sample_id": sample_id,
                        "qa_id": qa_id,
                        "q": q,
                        "answer": resolve_qa_answer(qa),
                        "category": coerce_category(qa.get("category", -1)),
                        "method": "rag",
                        "rag_mode": "chunk",
                        "chunk_size": int(chunk_size),
                        "top_k": int(top_k),
                        "d_start": d_start,
                        "d": [chunk_texts[idx] for idx in top_indices],
                        "retrieved_chunk_indices": top_indices,
                        "retrieval_scores": [round(float(score), 6) for score in top_scores],
                        "retrieved_turn_indices": [chunk_turn_indices[idx] for idx in top_indices],
                    }
                )
                pbar.update(1)
    finally:
        pbar.close()
        if cache_dirty:
            _save_memory_cache(cache_path, chunk_cache)

    payload = {
        "summary": {
            "method": "rag",
            "stage": "retrieve",
            "rag_mode": "chunk",
            "chunk_size": int(chunk_size),
            "top_k": int(top_k),
            "window": window_tag(source_file),
            "source_path": str(source_file),
            "retrieved_file": str(out_file),
            "total": len(retrieved_rows),
        },
        "entries": retrieved_rows,
    }
    safe_write_json(out_file, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieve LoCoMo rag chunks into a generic entries dataset")
    parser.add_argument("--rag-source-path", type=str, default=DEFAULT_RAG_SOURCE_PATH)
    parser.add_argument("--output-path", type=str, default="")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    payload = retrieve_rag_dataset(
        rag_source_path=args.rag_source_path,
        output_path=(args.output_path or None),
        output_dir=args.output_dir,
        chunk_size=args.chunk_size,
        top_k=args.top_k,
        limit=(args.limit or None),
    )
    print(json.dumps(payload["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
