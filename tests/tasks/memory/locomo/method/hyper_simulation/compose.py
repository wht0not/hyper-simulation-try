from __future__ import annotations

import json
import re
from pathlib import Path
import time
from typing import Any

from tqdm import tqdm

from hyper_simulation.component.hyper_simulation import compute_hyper_simulation
from hyper_simulation.component.postprocess import get_simulation_slice
from hyper_simulation.hypergraph.hypergraph import Hypergraph as LocalHypergraph, Vertex
from hyper_simulation.hypergraph.union import MultiHopFusion

from prompt.hyper_simulation import LOCOMO_HYPER_PROMPT, LOCOMO_HYPER_PROMPT_CAT_5
from utils.qa_utils import build_cat5_choice_question, build_question_text
from utils.utils import (
    coerce_category,
    entry_key,
    load_existing_results,
    prepared_output_path,
    safe_write_json,
    window_tag,
)

MIN_RELATIVE_CRITICAL_HITS = 0.4
MIN_RELATIVE_TOTAL_HITS = 0.3
LEGACY_HYPERSIM_FIELDS = {
    "consistent_context",
    "simulation_pair_count",
    "ranked_slice_indices",
    "selected_context_indices",
    "slice_hit_counts",
    "slice_critical_hit_counts",
}
QUERY_DIRNAME = "query"


def locomo_root_from_instances_root(instances_root: str | Path) -> Path:
    path = Path(instances_root)
    for candidate in (path, *path.parents):
        if candidate.name == "locomo":
            return candidate
    raise ValueError(f"instances_root must be under locomo directory: {path}")


def shared_query_output_dir(instances_root: str | Path) -> Path:
    return locomo_root_from_instances_root(instances_root) / QUERY_DIRNAME


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


def _load_query_hypergraph(instances_root: Path, qa_id: str) -> LocalHypergraph | None:
    query_path = shared_query_output_dir(instances_root) / f"query_hypergraph_{qa_id}.pkl"
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


def _build_slice_text(
    slice_index: int,
    evidence_item: dict[str, Any],
    rank_position: int | None = None,
    critical_hit_count: int | None = None,
    total_hit_count: int | None = None,
) -> str:
    hg = evidence_item.get("hypergraph")
    score_lines: list[str] = []
    if rank_position is not None:
        score_lines.append(f"Match Rank: {rank_position}")
    if critical_hit_count is not None or total_hit_count is not None:
        crit = 0 if critical_hit_count is None else int(critical_hit_count)
        total = 0 if total_hit_count is None else int(total_hit_count)
        score_lines.append(f"Graph Match Score: critical={crit}, total={total}")
    if hg is not None and hasattr(hg, "original_text") and hg.original_text:
        rendered = _render_session_block(str(hg.original_text).strip(), int(evidence_item.get("index", slice_index)) + 1)
        if score_lines:
            return "\n".join(score_lines + [rendered]).strip()
        return rendered
    fallback = f"[slice {slice_index}]"
    if score_lines:
        return "\n".join(score_lines + [fallback]).strip()
    return fallback


def _build_fallback_context(evidence_items: list[dict[str, Any]]) -> str:
    rendered = [_build_slice_text(i, item) for i, item in enumerate(evidence_items)]
    return "\n\n".join([text for text in rendered if text.strip()])


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
) -> tuple[str, dict[str, list[int]]]:
    scored_indices: list[tuple[int, int, int]] = []
    slice_hit_counts: dict[int, int] = {}
    slice_critical_hit_counts: dict[int, int] = {}
    for idx, simulation_slice in enumerate(simulation_slices):
        matched_data_vertex_ids = {
            v.id for u, v in simulation_slice if u is not None and v is not None and u.id in all_vertex_ids
        }
        matched_critical_data_vertex_ids = {
            v.id for u, v in simulation_slice if u is not None and v is not None and u.id in critical_vertex_ids
        }
        total_hit_count = len(matched_data_vertex_ids)
        critical_hit_count = len(matched_critical_data_vertex_ids)
        slice_hit_counts[idx] = total_hit_count
        slice_critical_hit_counts[idx] = critical_hit_count
        scored_indices.append((idx, critical_hit_count, total_hit_count))
    scored_indices.sort(key=lambda item: (-item[1], -item[2], item[0]))
    if not scored_indices:
        ranked_slice_indices = []
    else:
        best_critical_hit_count = scored_indices[0][1]
        best_total_hit_count = scored_indices[0][2]
        min_critical_hit_count = max(1, int(best_critical_hit_count * MIN_RELATIVE_CRITICAL_HITS)) if best_critical_hit_count > 0 else 0
        min_total_hit_count = max(1, int(best_total_hit_count * MIN_RELATIVE_TOTAL_HITS)) if best_total_hit_count > 0 else 0
        ranked_slice_indices = [
            idx
            for idx, critical_hit_count, total_hit_count in scored_indices
            if critical_hit_count >= min_critical_hit_count or total_hit_count >= min_total_hit_count
        ]
    rendered_slices: list[str] = []
    ranked_slice_summary: dict[str, list[int]] = {}
    for rank_position, idx in enumerate(ranked_slice_indices, start=1):
        if idx >= len(evidence_items):
            continue
        ranked_slice_summary[str(int(evidence_items[idx].get("index", idx)))] = [
            int(slice_critical_hit_counts.get(idx, 0)),
            int(slice_hit_counts.get(idx, 0)),
        ]
        rendered_slices.append(
            _build_slice_text(
                idx,
                evidence_items[idx],
                rank_position=rank_position,
                critical_hit_count=slice_critical_hit_counts.get(idx, 0),
                total_hit_count=slice_hit_counts.get(idx, 0),
            )
        )
    return "\n\n".join(rendered_slices), ranked_slice_summary


def compose_hypersim_instance(instances_root: Path, instance_dir: Path, existing_map: dict[str, dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    existing_map = existing_map or {}
    meta_path = instance_dir / "metadata.json"
    if not meta_path.exists():
        return []
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    sample_id = meta.get("sample_id", "")
    qa_list = meta.get("qa_list", [])
    d_start = meta.get("d_start", "")
    evidence_items = _load_data_hypergraphs(instance_dir)
    if not evidence_items:
        return []
    data_hgs = [ev["hypergraph"] for ev in evidence_items]
    fusion = MultiHopFusion()
    instance_merged_hg, _ = fusion.merge_hypergraphs(data_hgs)
    prepared_rows: list[dict[str, Any]] = []
    for qa_item in qa_list:
        qa_id = str(qa_item.get("qa_id", ""))
        q = str(qa_item.get("question", ""))
        answer = qa_item.get("answer", "")
        category = coerce_category(qa_item.get("category", -1))
        row_stub = {"sample_id": sample_id, "qa_id": qa_id, "q": q}
        if existing_map.get(entry_key(row_stub)) is not None:
            continue
        query_hg = _load_query_hypergraph(instances_root, qa_id)
        if query_hg is None:
            continue
        try:
            mapping, q_map, d_map = compute_hyper_simulation(query_hg, instance_merged_hg)
            simulation = [
                (q_map[q_id], d_map[d_id])
                for q_id, d_ids in mapping.items()
                for d_id in d_ids
                if q_id in q_map and d_id in d_map
            ]
            simulation_slices = get_simulation_slice(query_hg, instance_merged_hg, simulation, len(data_hgs))
            full_query_vertex_ids = {vertex.id for vertex in query_hg.vertices if _is_content_vertex(vertex)}
            critical_query_vertex_ids = {vertex.id for vertex in query_hg.vertices if _is_critical_query_vertex(query_hg, vertex)}
            ranked_context, ranked_slice_summary = _build_context_block(
                simulation_slices=simulation_slices,
                evidence_items=evidence_items,
                all_vertex_ids=full_query_vertex_ids,
                critical_vertex_ids=critical_query_vertex_ids,
            )
            if ranked_context.strip():
                context_text = ranked_context
            else:
                ranked_slice_summary = {str(int(ev.get("index", i))): [0, 0] for i, ev in enumerate(evidence_items)}
                context_text = _build_fallback_context(evidence_items)
            if d_start:
                context_text = d_start + "\n\n" + context_text
            prepared: dict[str, Any] = {
                "sample_id": sample_id,
                "qa_id": qa_id,
                "q": q,
                "answer": answer,
                "category": category,
                "method": "hyper_simulation",
                "slice": ranked_slice_summary,
            }
            if category == 5:
                cat5_question, cat5_answer_key = build_cat5_choice_question(
                    q, str(answer or ""), sample_id=sample_id, qa_id=qa_id
                )
                prepared["cat5_answer_key"] = cat5_answer_key
                prepared["prompt"] = LOCOMO_HYPER_PROMPT_CAT_5.format(context_text=context_text, question=cat5_question)
            else:
                prepared["prompt"] = LOCOMO_HYPER_PROMPT.format(
                    context_text=context_text, question=build_question_text(q, category)
                )
            prepared_rows.append(prepared)
        except Exception:
            continue
    return prepared_rows


def prepare_hypersim_instances(instances_root: str, output_dir: str, limit: int | None = None) -> dict[str, Any]:
    started_at = time.perf_counter()
    instances_dir = Path(instances_root)
    out_file = prepared_output_path(output_dir, "hyper_simulation", instances_dir)
    existing_rows = [sanitize_hypersim_row(row) for row in load_existing_results(out_file)]
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
    for instance_dir in tqdm(dirs, desc="locomo/compose/hyper_simulation", unit="inst"):
        for prepared in compose_hypersim_instance(instances_dir, instance_dir, existing_map=existing_map):
            prepared_rows.append(prepared)
            existing_map[entry_key(prepared)] = prepared
            safe_write_json(
                out_file,
                _prepared_payload(
                    prepared_rows,
                    instances_dir,
                    out_file,
                    elapsed_seconds=time.perf_counter() - started_at,
                ),
            )
    payload = _prepared_payload(
        prepared_rows,
        instances_dir,
        out_file,
        elapsed_seconds=time.perf_counter() - started_at,
    )
    safe_write_json(out_file, payload)
    return payload
