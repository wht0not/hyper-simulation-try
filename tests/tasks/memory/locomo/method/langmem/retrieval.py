from __future__ import annotations

import sys
import typing

if not hasattr(typing, "NotRequired"):
    try:
        from typing_extensions import NotRequired
        typing.NotRequired = NotRequired
    except ImportError:
        from typing import Optional
        typing.NotRequired = Optional

import json
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .memory import build_langmem_memory_dataset, load_langmem_memories
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


def _iter_langmem_samples(payload: Any) -> list[tuple[str, dict[str, Any]]]:
    if isinstance(payload, dict):
        return [(str(key), value) for key, value in payload.items() if isinstance(value, dict)]
    if isinstance(payload, list):
        items: list[tuple[str, dict[str, Any]]] = []
        for idx, value in enumerate(payload):
            if isinstance(value, dict):
                items.append((str(value.get("sample_id", idx)), value))
        return items
    return []


def count_langmem_questions(dataset_path: str | Path, limit: int | None = None) -> int:
    dataset_file = Path(dataset_path)
    if not dataset_file.exists():
        return 0
    try:
        payload = json.loads(dataset_file.read_text(encoding="utf-8"))
    except Exception:
        return 0
    samples = _iter_langmem_samples(payload)
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
    embedding_model_name: str,
    elapsed_seconds: float | None = None,
) -> dict[str, Any]:
    summary = {
        "method": "langmem",
        "stage": "retrieve",
        "window": window_tag(dataset_file),
        "source_path": str(dataset_file),
        "retrieved_file": str(out_file),
        "langmem_model_name": model_name,
        "langmem_embedding_model_name": embedding_model_name,
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
        if message.strip():
            count += 1
    return count


def _build_embed_fn(embedding_model_name: str) -> tuple[int, Any]:
    from langchain_ollama import OllamaEmbeddings

    embeddings = OllamaEmbeddings(model=embedding_model_name)
    sample_embedding = embeddings.embed_query("memory")
    dims = len(sample_embedding)

    def embed_texts(texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return embeddings.embed_documents(texts)

    return dims, embed_texts


def _build_agent(model_name: str, embedding_model_name: str):
    import typing
    if not hasattr(typing, "NotRequired"):
        try:
            from typing_extensions import NotRequired
            typing.NotRequired = NotRequired
        except ImportError:
            from typing import Optional
            typing.NotRequired = Optional
    
    from langchain_ollama import ChatOllama
    from langchain.agents import create_agent
    from langchain.agents.middleware import before_model, AgentState
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.store.memory import InMemoryStore
    from langmem import create_manage_memory_tool, create_search_memory_tool

    @before_model
    def pre_model_hook(state: AgentState, runtime) -> dict[str, AgentState] | None:
        store = runtime.store
        memories = store.search(("memories",), query=state["messages"][-1].content)
        system_msg = f"""You are a helpful assistant.

## Memories
<memories>
{memories}
</memories>
"""
        return {"messages": [{"role": "system", "content": system_msg}]}

    dims, embed_fn = _build_embed_fn(embedding_model_name)
    store = InMemoryStore(index={"dims": dims, "embed": embed_fn})
    llm = ChatOllama(model=model_name, temperature=0.0, reasoning=False)
    tools = [
        create_manage_memory_tool(namespace=("memories",), store=store),
        create_search_memory_tool(namespace=("memories",), store=store),
    ]
    return create_agent(
        model=llm,
        tools=tools,
        store=store,
        checkpointer=MemorySaver(),
        middleware=[pre_model_hook],
    )


def _load_agent_memories(
    agent: Any,
    output_dir: str,
    sample_id: str,
    speaker_name: str,
    thread_id: str,
) -> None:
    config = {"configurable": {"thread_id": thread_id}}
    existing_memories = load_langmem_memories(output_dir=output_dir, sample_id=sample_id, speaker_name=speaker_name)
    for memory_payload in existing_memories:
        content = str(memory_payload.get("memory", memory_payload.get("summary", ""))).strip()
        if not content:
            continue
        agent.invoke({"messages": [{"role": "user", "content": content}]}, config=config)


def _search_memories(agent: Any, query: str, thread_id: str) -> tuple[str, float]:
    config = {"configurable": {"thread_id": thread_id}}
    started_at = time.perf_counter()
    try:
        response = agent.invoke({"messages": [{"role": "user", "content": query}]}, config=config)
        messages = response.get("messages", []) if isinstance(response, dict) else []
        if messages:
            last_message = messages[-1]
            if isinstance(last_message, dict):
                content = last_message.get("content", "")
            else:
                content = getattr(last_message, "content", "")
            return str(content or "").strip(), round(time.perf_counter() - started_at, 4)
    except Exception as exc:
        return f"[ERROR] {type(exc).__name__}: {exc}", round(time.perf_counter() - started_at, 4)
    return "", round(time.perf_counter() - started_at, 4)


def retrieve_langmem_dataset(
    dataset_path: str,
    output_dir: str,
    model_name: str,
    output_path: str | None = None,
    limit: int | None = None,
    embedding_model_name: str = "qwen3-embedding:0.6b",
) -> dict[str, Any]:
    started_at = time.perf_counter()
    dataset_file = Path(dataset_path)
    if output_path:
        out_file = Path(output_path)
    else:
        out_file = retrieved_output_path(output_dir, "langmem", dataset_file)

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

    build_langmem_memory_dataset(
        dataset_path=dataset_path,
        output_dir=output_dir,
        model_name=model_name,
        limit=limit,
    )

    samples = _iter_langmem_samples(raw_payload)
    if limit is not None and limit > 0:
        samples = samples[:limit]

    total_questions = count_langmem_questions(dataset_file, limit=limit)
    existing_map = load_existing_result_map(out_file)
    retrieved_rows = load_existing_results(out_file)
    pbar = tqdm(total=total_questions, desc="locomo/retrieve/langmem", unit="q")

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
            speaker_1_agent = _build_agent(model_name=model_name, embedding_model_name=embedding_model_name)
            speaker_2_agent = _build_agent(model_name=model_name, embedding_model_name=embedding_model_name)
            _load_agent_memories(
                speaker_1_agent,
                output_dir=output_dir,
                sample_id=sample_id,
                speaker_name=speaker_1_name,
                thread_id=f"langmem-{sample_id}-speaker-1",
            )
            _load_agent_memories(
                speaker_2_agent,
                output_dir=output_dir,
                sample_id=sample_id,
                speaker_name=speaker_2_name,
                thread_id=f"langmem-{sample_id}-speaker-2",
            )

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

                speaker_1_memories, speaker_1_search_time = _search_memories(
                    speaker_1_agent, question, thread_id=f"langmem-{sample_id}-speaker-1"
                )
                speaker_2_memories, speaker_2_search_time = _search_memories(
                    speaker_2_agent, question, thread_id=f"langmem-{sample_id}-speaker-2"
                )

                row = {
                    "sample_id": sample_id,
                    "qa_id": qa_id,
                    "q": question,
                    "answer": question_item.get("answer"),
                    "category": coerce_category(question_item.get("category")),
                    "method": "langmem",
                    "d": [speaker_1_memories, speaker_2_memories],
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
                        embedding_model_name=embedding_model_name,
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
        embedding_model_name=embedding_model_name,
        elapsed_seconds=time.perf_counter() - started_at,
    )
    safe_write_json(out_file, payload)
    return payload
