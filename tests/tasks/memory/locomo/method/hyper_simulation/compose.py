from __future__ import annotations

import json
import os
import re
import hashlib
from pathlib import Path
import time
from typing import Any

from tqdm import tqdm

from hyper_simulation.component.hyper_simulation import compute_hyper_simulation
from hyper_simulation.component.postprocess import get_simulation_slice
from hyper_simulation.hypergraph.hypergraph import Hypergraph as LocalHypergraph, Vertex
from hyper_simulation.hypergraph.union import MultiHopFusion

from prompt.amem import build_hyper_amem_answer_prompt
from prompt.hyper_simulation import LOCOMO_HYPER_PROMPT, LOCOMO_HYPER_RAG_PROMPT
from prompt.langmem import LOCOMO_LANGMEM_PROMPT
from prompt.memorybank import build_hyper_memorybank_answer_prompt
from utils.utils import (
    coerce_category,
    entry_key,
    load_existing_results,
    prepared_output_path,
    safe_write_json,
    window_tag,
)

LEGACY_HYPERSIM_FIELDS = {
    "consistent_context",
    "simulation_pair_count",
    "ranked_slice_indices",
    "selected_context_indices",
    "slice_hit_counts",
    "slice_critical_hit_counts",
}
QUERY_DIRNAME = "query"


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return float(raw)
    except Exception:
        return default


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return int(raw)
    except Exception:
        return default


def _category_filter_env(name: str = "HYPERSIM_ALLOWED_CATEGORIES") -> set[int] | None:
    raw = os.environ.get(name)
    if raw is None:
        return None
    tokens = [token.strip() for token in str(raw).split(",")]
    categories: set[int] = set()
    for token in tokens:
        if not token:
            continue
        try:
            categories.add(int(token))
        except Exception:
            continue
    return categories or None


def locomo_root_from_instances_root(instances_root: str | Path) -> Path:
    path = Path(instances_root)
    for candidate in (path, *path.parents):
        if candidate.name == "locomo":
            return candidate
    raise ValueError(f"instances_root must be under locomo directory: {path}")


def shared_query_output_dir(instances_root: str | Path) -> Path:
    return locomo_root_from_instances_root(instances_root) / QUERY_DIRNAME


def query_key_from_question(question: str) -> str:
    normalized = " ".join(str(question or "").strip().split())
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def _prepared_payload(
    rows: list[dict[str, Any]],
    instances_dir: Path,
    out_file: Path,
    elapsed_seconds: float | None = None,
) -> dict[str, Any]:
    summary = {
        "method": "hyper_simulation",
        "window": window_tag(instances_dir),
        "source_path": str(instances_dir),
        "prepared_file": str(out_file),
        "total": len(rows),
    }
    if elapsed_seconds is not None:
        summary["elapsed_seconds"] = round(float(elapsed_seconds), 4)
    return {
        "summary": {
            **summary,
        },
        "results": rows,
    }


def sanitize_hypersim_row(row: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(row)
    if "slice" not in cleaned:
        ranked_slice_indices = cleaned.get("ranked_slice_indices", [])
        slice_hit_counts = cleaned.get("slice_hit_counts", {}) or {}
        slice_critical_hit_counts = cleaned.get("slice_critical_hit_counts", {}) or {}
        slice_summary: dict[str, list[int]] = {}
        if isinstance(ranked_slice_indices, list):
            for idx in ranked_slice_indices:
                idx_str = str(idx)
                slice_summary[idx_str] = [
                    int(slice_critical_hit_counts.get(idx_str, slice_critical_hit_counts.get(idx, 0))),
                    int(slice_hit_counts.get(idx_str, slice_hit_counts.get(idx, 0))),
                ]
        cleaned["slice"] = slice_summary
    for field_name in LEGACY_HYPERSIM_FIELDS:
        cleaned.pop(field_name, None)
    return cleaned


def _sorted_index_from_name(path: Path) -> int:
    match_obj = re.fullmatch(r"data_hypergraph(\d+)\.pkl", path.name)
    if match_obj is None:
        return 10**9
    return int(match_obj.group(1))


def _load_query_hypergraph(instances_root: Path, question: str) -> LocalHypergraph | None:
    query_root = shared_query_output_dir(instances_root)
    query_key = query_key_from_question(question)
    query_path = query_root / f"{query_key}.pkl"
    if not query_path.exists():
        return None
    try:
        return LocalHypergraph.load(str(query_path))
    except Exception:
        return None


def _load_data_hypergraphs(instance_dir: Path) -> list[dict[str, Any]]:
    data_paths = sorted(instance_dir.glob("data_hypergraph*.pkl"), key=_sorted_index_from_name)
    evidence_items: list[dict[str, Any]] = []
    for data_path in data_paths:
        match_obj = re.fullmatch(r"data_hypergraph(\d+)\.pkl", data_path.name)
        if match_obj is None:
            continue
        data_idx = int(match_obj.group(1))
        try:
            data_hg = LocalHypergraph.load(str(data_path))
        except Exception:
            continue
        evidence_items.append({"index": data_idx, "path": str(data_path), "hypergraph": data_hg})
    return evidence_items


def _build_slice_text(
    slice_index: int,
    evidence_item: dict[str, Any],
) -> str:
    hg = evidence_item.get("hypergraph")
    if hg is not None and hasattr(hg, "original_text") and hg.original_text:
        return _render_session_block(str(hg.original_text).strip(), slice_index + 1)
    fallback = f"[slice {slice_index}]"
    return fallback


def _normalize_hyper_items(raw_items: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_items, list):
        raw_items = []
    normalized: list[dict[str, Any]] = []
    for idx, item in enumerate(raw_items):
        if isinstance(item, dict):
            text = str(item.get("text", item.get("memory", ""))).strip()
            if not text:
                continue
            normalized.append(
                {
                    "index": int(item.get("index", idx)),
                    "text": text,
                    "speaker_tag": str(item.get("speaker_tag", "")).strip(),
                    "speaker_name": str(item.get("speaker_name", "")).strip(),
                }
            )
            continue
        text = str(item).strip()
        if not text:
            continue
        normalized.append({"index": idx, "text": text, "speaker_tag": "", "speaker_name": ""})
    return normalized


def _hyper_items_from_evidence_items(evidence_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    derived_items: list[dict[str, Any]] = []
    for idx, evidence_item in enumerate(evidence_items):
        hg = evidence_item.get("hypergraph")
        text = ""
        if hg is not None and hasattr(hg, "original_text"):
            text = str(getattr(hg, "original_text", "") or "").strip()
        if not text:
            continue
        derived_items.append(
            {
                "index": int(evidence_item.get("index", idx)),
                "text": text,
                "speaker_tag": "",
                "speaker_name": "",
            }
        )
    return derived_items


def _render_session_block(session_text: str, session_index: int) -> str:
    text = str(session_text or "").strip()
    if not text:
        return ""
    lines = [line.rstrip() for line in text.splitlines()]
    session_date = ""
    body_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("DATE:"):
            session_date = stripped[len("DATE:") :].strip()
            continue
        if stripped == "CONVERSATION:":
            continue
        body_lines.append(stripped)
    rendered = [f"Session {session_index}"]
    if session_date:
        rendered.append(f"Session Date: {session_date}")
        rendered.append("All utterances below happened on this date.")
    if body_lines:
        rendered.append("Conversation:")
        rendered.extend(body_lines)
    return "\n".join(rendered).strip()


def _render_ranked_plain_context(hyper_items: list[dict[str, Any]], d_start: str = "") -> str:
    blocks: list[str] = []
    for idx, item in enumerate(hyper_items):
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        session_index = int(item.get("index", idx)) + 1
        rendered_text = _render_session_block(text, session_index)
        if rendered_text:
            blocks.append(rendered_text)
    context_text = "\n\n".join(blocks).strip()
    if d_start and context_text:
        return f"{d_start}\n\n{context_text}"
    if d_start:
        return d_start
    return context_text


def _ordered_items_from_indices(hyper_items: list[dict[str, Any]], ranked_indices: list[int]) -> list[dict[str, Any]]:
    if not ranked_indices:
        return list(hyper_items)
    ordered: list[dict[str, Any]] = []
    seen: set[int] = set()
    for idx in ranked_indices:
        if idx < 0 or idx >= len(hyper_items) or idx in seen:
            continue
        ordered.append(hyper_items[idx])
        seen.add(idx)
    for idx, item in enumerate(hyper_items):
        if idx in seen:
            continue
        ordered.append(item)
    return ordered


def _prepare_generic_hyper_row(
    sample_id: str,
    qa_id: str,
    question: str,
    answer: Any,
    category: int,
    context_text: str,
    source_method: str,
) -> dict[str, Any]:
    prepared: dict[str, Any] = {
        "sample_id": sample_id,
        "qa_id": qa_id,
        "q": question,
        "answer": answer,
        "category": category,
        "method": "hyper_simulation",
    }
    prompt_template = LOCOMO_HYPER_RAG_PROMPT if str(source_method).strip() == "rag" else LOCOMO_HYPER_PROMPT
    prompt_text = prompt_template.format(
        context_text=context_text,
        question=str(question).strip(),
    )
    prepared["prompt"] = prompt_text
    return prepared


def _prepare_memorybank_hyper_row(
    source_entry: dict[str, Any],
    sample_id: str,
    qa_id: str,
    question: str,
    answer: Any,
    category: int,
    ordered_hyper_items: list[dict[str, Any]],
) -> dict[str, Any]:
    memory_payload = source_entry.get("memorybank_context", {})
    if not isinstance(memory_payload, dict):
        memory_payload = {}
    ranked_memory_body = "\n\n".join(
        [str(item.get("text", "")).strip() for item in ordered_hyper_items if str(item.get("text", "")).strip()]
    ).strip()
    related_memory = (
        "Memories (Ranked by Relevance, Highest to Lowest):\n\n" + ranked_memory_body
        if ranked_memory_body
        else ranked_memory_body
    )
    answer_prompt_payload = build_hyper_memorybank_answer_prompt(
        user_name=str(memory_payload.get("user_name", "User")),
        overall_history=str(memory_payload.get("overall_history", "")).strip(),
        related_memory=related_memory,
        question=str(question).strip(),
    )
    prepared: dict[str, Any] = {
        "sample_id": sample_id,
        "qa_id": qa_id,
        "q": question,
        "answer": answer,
        "category": category,
        "method": "hyper_simulation",
        "source_method": "memorybank",
        "prompt": str(answer_prompt_payload["prompt"]),
        "answer_temperature": float(answer_prompt_payload.get("temperature", 0.1)),
    }
    return prepared


def _speaker_memory_text(
    ordered_hyper_items: list[dict[str, Any]],
    speaker_tag: str,
    speaker_name: str,
    fallback_text: str = "",
) -> str:
    selected = [
        str(item.get("text", "")).strip()
        for item in ordered_hyper_items
        if str(item.get("text", "")).strip()
        and (
            str(item.get("speaker_tag", "")).strip() == speaker_tag
            or str(item.get("speaker_name", "")).strip() == speaker_name
        )
    ]
    if selected:
        return "Memories (Ranked by Relevance, Highest to Lowest):\n\n" + "\n\n".join(selected).strip()
    return str(fallback_text or "").strip()


def _ensure_ranked_memory_header(text: str) -> str:
    body = str(text or "").strip()
    if not body:
        return body
    canonical_header = "Memories (Ranked by Relevance, Highest to Lowest):"
    if body.startswith(canonical_header):
        return body
    if body.startswith("Memories ranked by relevance:"):
        body = body[len("Memories ranked by relevance:") :].strip()
    return f"{canonical_header}\n\n{body}"


def _prepare_amem_hyper_row(
    source_entry: dict[str, Any],
    sample_id: str,
    qa_id: str,
    question: str,
    answer: Any,
    category: int,
    ordered_hyper_items: list[dict[str, Any]],
) -> dict[str, Any]:
    speakers = source_entry.get("speakers", {})
    d_list = source_entry.get("d", [])
    speaker_1_name = str(speakers.get("speaker_1", "speaker_1")) if isinstance(speakers, dict) else "speaker_1"
    speaker_2_name = str(speakers.get("speaker_2", "speaker_2")) if isinstance(speakers, dict) else "speaker_2"
    speaker_1_fallback = str(d_list[0]) if isinstance(d_list, list) and len(d_list) > 0 else ""
    speaker_2_fallback = str(d_list[1]) if isinstance(d_list, list) and len(d_list) > 1 else ""
    speaker_1_memories = _speaker_memory_text(
        ordered_hyper_items,
        speaker_tag="speaker_1",
        speaker_name=speaker_1_name,
        fallback_text=speaker_1_fallback,
    )
    speaker_2_memories = _speaker_memory_text(
        ordered_hyper_items,
        speaker_tag="speaker_2",
        speaker_name=speaker_2_name,
        fallback_text=speaker_2_fallback,
    )
    speaker_1_memories = _ensure_ranked_memory_header(speaker_1_memories)
    speaker_2_memories = _ensure_ranked_memory_header(speaker_2_memories)
    context_text = (
        f"Memories for user {speaker_1_name}:\n{speaker_1_memories or 'No relevant memories found.'}\n\n"
        f"Memories for user {speaker_2_name}:\n{speaker_2_memories or 'No relevant memories found.'}"
    )
    answer_prompt_payload = build_hyper_amem_answer_prompt(
        context_text=context_text,
        question=str(question).strip(),
    )
    prepared: dict[str, Any] = {
        "sample_id": sample_id,
        "qa_id": qa_id,
        "q": question,
        "answer": answer,
        "category": category,
        "method": "hyper_simulation",
        "source_method": "amem",
        "prompt": str(answer_prompt_payload["prompt"]),
        "answer_temperature": float(answer_prompt_payload.get("temperature", 0.1)),
    }
    return prepared


def _prepare_langmem_hyper_row(
    source_entry: dict[str, Any],
    sample_id: str,
    qa_id: str,
    question: str,
    answer: Any,
    category: int,
    ordered_hyper_items: list[dict[str, Any]],
) -> dict[str, Any]:
    speakers = source_entry.get("speakers", {})
    d_list = source_entry.get("d", [])
    speaker_1_name = str(speakers.get("speaker_1", "speaker_1")) if isinstance(speakers, dict) else "speaker_1"
    speaker_2_name = str(speakers.get("speaker_2", "speaker_2")) if isinstance(speakers, dict) else "speaker_2"
    speaker_1_fallback = str(d_list[0]) if isinstance(d_list, list) and len(d_list) > 0 else ""
    speaker_2_fallback = str(d_list[1]) if isinstance(d_list, list) and len(d_list) > 1 else ""
    speaker_1_memories = _speaker_memory_text(
        ordered_hyper_items,
        speaker_tag="speaker_1",
        speaker_name=speaker_1_name,
        fallback_text=speaker_1_fallback,
    ) or "No relevant memories found."
    speaker_2_memories = _speaker_memory_text(
        ordered_hyper_items,
        speaker_tag="speaker_2",
        speaker_name=speaker_2_name,
        fallback_text=speaker_2_fallback,
    ) or "No relevant memories found."
    prepared: dict[str, Any] = {
        "sample_id": sample_id,
        "qa_id": qa_id,
        "q": question,
        "answer": answer,
        "category": category,
        "method": "hyper_simulation",
        "source_method": "langmem",
    }
    prepared["prompt"] = LOCOMO_LANGMEM_PROMPT.format(
        speaker_1_user_id=speaker_1_name,
        speaker_1_memories=speaker_1_memories,
        speaker_2_user_id=speaker_2_name,
        speaker_2_memories=speaker_2_memories,
        question=str(question).strip(),
    )
    return prepared


def _is_content_vertex(vertex: Vertex) -> bool:
    return not (vertex.is_verb() or vertex.is_virtual())


def _is_entity_fact_anchor(vertex: Vertex) -> bool:
    return _is_content_vertex(vertex) and vertex.has_entity()


def _is_critical_query_vertex(query_hg: LocalHypergraph, vertex: Vertex) -> bool:
    if not _is_content_vertex(vertex):
        return False
    if any(hyperedge.root == vertex for hyperedge in query_hg.contained_edges.get(vertex, [])):
        return True
    return _is_entity_fact_anchor(vertex)


def _build_context_block(
    simulation_slices: list[list[tuple[Vertex, Vertex]]],
    evidence_items: list[dict[str, Any]],
    all_vertex_ids: set[int],
    critical_vertex_ids: set[int],
) -> tuple[str, list[int]]:
    scored_indices: list[tuple[int, int, int]] = []
    for idx, simulation_slice in enumerate(simulation_slices):
        matched_data_vertex_ids = {
            v.id for u, v in simulation_slice if u is not None and v is not None and u.id in all_vertex_ids
        }
        matched_critical_data_vertex_ids = {
            v.id for u, v in simulation_slice if u is not None and v is not None and u.id in critical_vertex_ids
        }
        total_hit_count = len(matched_data_vertex_ids)
        critical_hit_count = len(matched_critical_data_vertex_ids)
        scored_indices.append((idx, critical_hit_count, total_hit_count))
    scored_indices.sort(key=lambda item: (-item[1], -item[2], item[0]))
    if not scored_indices:
        ranked_slice_indices = []
    else:
        # Keep all evidence slices and only rerank by hit counts.
        ranked_slice_indices = [idx for idx, _, _ in scored_indices]
    rendered_slices: list[str] = [
        _build_slice_text(idx, evidence_items[idx])
        for idx in ranked_slice_indices
        if 0 <= idx < len(evidence_items)
    ]
    ranked_item_indices = [
        int(evidence_items[idx].get("index", idx))
        for idx in ranked_slice_indices
        if 0 <= idx < len(evidence_items)
    ]
    return "\n\n".join(rendered_slices), ranked_item_indices


def _filtered_qa_items(
    qa_list: Any,
    sample_id: str,
    existing_map: dict[str, dict[str, Any]],
    allowed_categories: set[int] | None,
) -> list[dict[str, Any]]:
    if not isinstance(qa_list, list):
        return []
    filtered: list[dict[str, Any]] = []
    for qa_item in qa_list:
        if not isinstance(qa_item, dict):
            continue
        qa_id = str(qa_item.get("qa_id", ""))
        question = str(qa_item.get("question", ""))
        category = coerce_category(qa_item.get("category", -1))
        if category == 5:
            continue
        if allowed_categories is not None and category not in allowed_categories:
            continue
        row_stub = {"sample_id": sample_id, "qa_id": qa_id, "q": question}
        if existing_map.get(entry_key(row_stub)) is not None:
            continue
        filtered.append(qa_item)
    return filtered


def _qa_requires_rerank(
    qa_item: dict[str, Any],
    hyper_items: list[dict[str, Any]],
    low_d_bypass: bool,
) -> bool:
    if low_d_bypass:
        return False
    source_entry = qa_item.get("source_entry", {})
    if not isinstance(source_entry, dict):
        source_entry = {}
    qa_hyper_items = _normalize_hyper_items(source_entry.get("hyper_d_items", hyper_items))
    if not qa_hyper_items and hyper_items:
        qa_hyper_items = list(hyper_items)
    return len(qa_hyper_items) > 3


def compose_hypersim_instance(instances_root: Path, instance_dir: Path, existing_map: dict[str, dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    existing_map = existing_map or {}
    sigma_threshold = _float_env("HYPERSIM_SIGMA_THRESHOLD", 0.75)
    b_threshold = _int_env("HYPERSIM_B_THRESHOLD", 5)
    delta_threshold = _float_env("HYPERSIM_DELTA_THRESHOLD", 0.7)
    allowed_categories = _category_filter_env()
    meta_path = instance_dir / "metadata.json"
    if not meta_path.exists():
        return []
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    sample_id = str(meta.get("sample_id", ""))
    qa_list = meta.get("qa_list", [])
    d_start = meta.get("d_start", "")
    source_method = str(meta.get("source_method", "context")).strip() or "context"
    low_d_bypass = bool(meta.get("low_d_bypass", False))
    qa_items = _filtered_qa_items(
        qa_list=qa_list,
        sample_id=sample_id,
        existing_map=existing_map,
        allowed_categories=allowed_categories,
    )
    if not qa_items:
        return []
    hyper_items = _normalize_hyper_items(meta.get("hyper_items", []))
    instance_merged_hg = None
    evidence_items: list[dict[str, Any]] = []
    use_generic_source = source_method not in {"memorybank", "amem", "langmem"}
    needs_evidence_hypergraphs = (
        (use_generic_source and not hyper_items)
        or any(
        _qa_requires_rerank(qa_item, hyper_items, low_d_bypass=low_d_bypass)
        for qa_item in qa_items
        )
    )
    if needs_evidence_hypergraphs:
        evidence_items = _load_data_hypergraphs(instance_dir)
        if not hyper_items and evidence_items:
            hyper_items = _hyper_items_from_evidence_items(evidence_items)
        if evidence_items:
            data_hgs = [ev["hypergraph"] for ev in evidence_items]
            fusion = MultiHopFusion()
            instance_merged_hg, _ = fusion.merge_hypergraphs(data_hgs)
    prepared_rows: list[dict[str, Any]] = []
    for qa_item in qa_items:
        item_started_at = time.perf_counter()
        qa_id = str(qa_item.get("qa_id", ""))
        q = str(qa_item.get("question", ""))
        answer = qa_item.get("answer", "")
        category = coerce_category(qa_item.get("category", -1))
        source_entry = qa_item.get("source_entry", {})
        if not isinstance(source_entry, dict):
            source_entry = {}
        effective_source_method = str(source_entry.get("source_method", source_entry.get("method", source_method))).strip() or source_method
        qa_hyper_items = _normalize_hyper_items(source_entry.get("hyper_d_items", hyper_items))
        if not qa_hyper_items and hyper_items:
            qa_hyper_items = list(hyper_items)
        use_generic_prompt = source_method not in {"memorybank", "amem", "langmem"}
        ordered_hyper_items = list(qa_hyper_items)
        context_text = (
            _render_ranked_plain_context(ordered_hyper_items, d_start=str(d_start or ""))
            if use_generic_prompt
            else ""
        )
        can_rerank = (
            not low_d_bypass
            and len(qa_hyper_items) > 3
            and bool(evidence_items)
            and instance_merged_hg is not None
        )
        try:
            if can_rerank:
                query_hg = _load_query_hypergraph(instances_root, question=q)
                if query_hg is not None:
                    mapping, q_map, d_map = compute_hyper_simulation(
                        query_hg,
                        instance_merged_hg,
                        sigma_threshold=sigma_threshold,
                        b_threshold=b_threshold,
                        delta_threshold=delta_threshold,
                    )
                    simulation = [
                        (q_map[q_vertex_id], d_map[d_vertex_id])
                        for q_vertex_id, d_ids in mapping.items()
                        for d_vertex_id in d_ids
                        if q_vertex_id in q_map and d_vertex_id in d_map
                    ]
                    simulation_slices = get_simulation_slice(query_hg, instance_merged_hg, simulation, len(evidence_items))
                    full_query_vertex_ids = {vertex.id for vertex in query_hg.vertices if _is_content_vertex(vertex)}
                    critical_query_vertex_ids = {
                        vertex.id for vertex in query_hg.vertices if _is_critical_query_vertex(query_hg, vertex)
                    }
                    ranked_context, ranked_item_indices = _build_context_block(
                        simulation_slices=simulation_slices,
                        evidence_items=evidence_items,
                        all_vertex_ids=full_query_vertex_ids,
                        critical_vertex_ids=critical_query_vertex_ids,
                    )
                    ordered_hyper_items = _ordered_items_from_indices(qa_hyper_items, ranked_item_indices)
                    if use_generic_prompt:
                        if ranked_context.strip():
                            context_text = f"{d_start}\n\n{ranked_context}".strip() if d_start else ranked_context
                        else:
                            context_text = _render_ranked_plain_context(ordered_hyper_items, d_start=str(d_start or ""))
            if source_method == "memorybank":
                prepared = _prepare_memorybank_hyper_row(
                    source_entry=source_entry,
                    sample_id=sample_id,
                    qa_id=qa_id,
                    question=q,
                    answer=answer,
                    category=category,
                    ordered_hyper_items=ordered_hyper_items,
                )
            elif source_method == "amem":
                prepared = _prepare_amem_hyper_row(
                    source_entry=source_entry,
                    sample_id=sample_id,
                    qa_id=qa_id,
                    question=q,
                    answer=answer,
                    category=category,
                    ordered_hyper_items=ordered_hyper_items,
                )
            elif source_method == "langmem":
                prepared = _prepare_langmem_hyper_row(
                    source_entry=source_entry,
                    sample_id=sample_id,
                    qa_id=qa_id,
                    question=q,
                    answer=answer,
                    category=category,
                    ordered_hyper_items=ordered_hyper_items,
                )
            else:
                prepared = _prepare_generic_hyper_row(
                    sample_id=sample_id,
                    qa_id=qa_id,
                    question=q,
                    answer=answer,
                    category=category,
                    context_text=context_text,
                    source_method=effective_source_method,
                )
            prepared["source_method"] = effective_source_method
            prepared["prepared_elapsed_seconds"] = round(time.perf_counter() - item_started_at, 6)
            prepared_rows.append(prepared)
        except Exception:
            continue
    return prepared_rows


def prepare_hypersim_instances(
    instances_root: str,
    output_dir: str,
    limit: int | None = None,
    checkpoint_every: int = 500,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    instances_dir = Path(instances_root)
    out_file = prepared_output_path(output_dir, "hyper_simulation", instances_dir)
    allowed_categories = _category_filter_env()
    existing_rows = [
        sanitize_hypersim_row(row)
        for row in load_existing_results(out_file)
        if coerce_category(row.get("category", -1)) != 5
        and (allowed_categories is None or coerce_category(row.get("category", -1)) in allowed_categories)
    ]
    existing_map = {entry_key(row): row for row in existing_rows if entry_key(row)}
    if not instances_dir.exists():
        payload = {
            "summary": {
                "method": "hyper_simulation",
                "window": window_tag(instances_dir),
                "source_path": str(instances_dir),
                "prepared_file": str(out_file),
                "total": 0,
                "error": "instances dir not found",
            },
            "results": [],
        }
        safe_write_json(out_file, payload)
        return payload
    dirs = [d for d in instances_dir.iterdir() if d.is_dir()]
    if limit is not None and limit > 0:
        dirs = dirs[:limit]
    prepared_rows = list(existing_rows)
    checkpoint_every = max(1, int(checkpoint_every))
    pending_writes = 0
    for instance_dir in tqdm(dirs, desc="locomo/compose/hyper_simulation", unit="inst"):
        for prepared in compose_hypersim_instance(instances_dir, instance_dir, existing_map=existing_map):
            prepared_rows.append(prepared)
            existing_map[entry_key(prepared)] = prepared
            pending_writes += 1
            if pending_writes >= checkpoint_every:
                safe_write_json(
                    out_file,
                    _prepared_payload(
                        prepared_rows,
                        instances_dir,
                        out_file,
                        elapsed_seconds=time.perf_counter() - started_at,
                    ),
                )
                pending_writes = 0
    payload = _prepared_payload(
        prepared_rows,
        instances_dir,
        out_file,
        elapsed_seconds=time.perf_counter() - started_at,
    )
    safe_write_json(out_file, payload)
    return payload
