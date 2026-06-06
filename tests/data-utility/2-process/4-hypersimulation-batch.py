from __future__ import annotations

import json
import logging
import sys
import time
import typing
from pathlib import Path

if not hasattr(typing, "NotRequired"):
    try:
        from typing_extensions import NotRequired

        typing.NotRequired = NotRequired
    except ImportError:
        from typing import Optional

        typing.NotRequired = Optional


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
LOCOMO_ROOT = REPO_ROOT / "tests" / "tasks" / "memory" / "locomo"
INSTANCES_ROOT = REPO_ROOT / "tests" / "data-utility" / "data" / "hypergraphs" / "locomo" / "context"
OUTPUT_DIR = REPO_ROOT / "tests" / "data-utility" / "data" / "context"

for candidate in (REPO_ROOT, SRC_ROOT, LOCOMO_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from method.hyper_simulation.compose import (
    _prepared_payload,
    _build_context_block,
    _hyper_items_from_evidence_items,
    _is_content_vertex,
    _is_critical_query_vertex,
    _load_data_hypergraphs,
    _ordered_items_from_indices,
    _prepare_generic_hyper_row,
    _render_ranked_plain_context,
    sanitize_hypersim_row,
)
from utils.utils import coerce_category, entry_key, safe_write_json
from hyper_simulation.component.hyper_simulation import compute_hyper_simulation
from hyper_simulation.component.postprocess import get_simulation_slice
from hyper_simulation.hypergraph.hypergraph import Hypergraph as LocalHypergraph
from hyper_simulation.hypergraph.union import MultiHopFusion


def _hypergraph_size_text(hypergraph: LocalHypergraph | None) -> str:
    if hypergraph is None:
        return "vertex: 0, edge: 0"
    return f"vertex: {len(hypergraph.vertices)}, edge: {len(hypergraph.hyperedges)}"


def _load_local_query_hypergraph(instance_dir: Path, qa_id: str) -> LocalHypergraph | None:
    query_path = instance_dir / f"query_hypergraph{qa_id}.pkl"
    if not query_path.exists():
        return None
    try:
        return LocalHypergraph.load(str(query_path))
    except Exception:
        return None


def _compose_conv_instance(
    instance_dir: Path,
    existing_map: dict[str, dict],
) -> list[dict]:
    meta_path = instance_dir / "metadata.json"
    if not meta_path.exists():
        return []

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return []

    sample_id = str(meta.get("sample_id", "")).strip()
    d_start = str(meta.get("d_start", "")).strip()
    qa_list = meta.get("qa_list", [])
    if not sample_id or not isinstance(qa_list, list):
        return []

    evidence_items = _load_data_hypergraphs(instance_dir)
    if not evidence_items:
        return []

    data_hgs = [ev["hypergraph"] for ev in evidence_items]
    fusion = MultiHopFusion()
    merged_hg, _ = fusion.merge_hypergraphs(data_hgs)
    hyper_items = _hyper_items_from_evidence_items(evidence_items)
    print(
        f"[instance {instance_dir.name}] merged ready; "
        f"data slices: {len(evidence_items)}, {_hypergraph_size_text(merged_hg)}"
    )

    prepared_rows: list[dict] = []
    for qa_item in qa_list:
        if not isinstance(qa_item, dict):
            continue
        qa_id = str(qa_item.get("qa_id", "")).strip()
        question = str(qa_item.get("question", "")).strip()
        category = coerce_category(qa_item.get("category", -1))
        if not qa_id or not question or category == 5:
            continue
        row_stub = {"sample_id": sample_id, "qa_id": qa_id, "q": question}
        if existing_map.get(entry_key(row_stub)) is not None:
            continue

        item_started_at = time.perf_counter()
        answer = qa_item.get("answer")
        ordered_hyper_items = list(hyper_items)
        context_text = _render_ranked_plain_context(ordered_hyper_items, d_start=d_start)

        query_hg = _load_local_query_hypergraph(instance_dir, qa_id)
        matched_vertex_pairs = 0
        slice_count = 0
        if query_hg is not None and len(ordered_hyper_items) > 3:
            mapping, q_map, d_map = compute_hyper_simulation(query_hg, merged_hg)
            simulation = [
                (q_map[q_vertex_id], d_map[d_vertex_id])
                for q_vertex_id, d_ids in mapping.items()
                for d_vertex_id in d_ids
                if q_vertex_id in q_map and d_vertex_id in d_map
            ]
            matched_vertex_pairs = len(simulation)
            simulation_slices = get_simulation_slice(query_hg, merged_hg, simulation, len(evidence_items))
            slice_count = len(simulation_slices)
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
            ordered_hyper_items = _ordered_items_from_indices(ordered_hyper_items, ranked_item_indices)
            if ranked_context.strip():
                context_text = f"{d_start}\n\n{ranked_context}".strip() if d_start else ranked_context
            else:
                context_text = _render_ranked_plain_context(ordered_hyper_items, d_start=d_start)

        prepared = _prepare_generic_hyper_row(
            sample_id=sample_id,
            qa_id=qa_id,
            question=question,
            answer=answer,
            category=category,
            context_text=context_text,
            source_method="context",
        )
        prepared["source_method"] = "context"
        prepared["prepared_elapsed_seconds"] = round(time.perf_counter() - item_started_at, 6)
        prepared_rows.append(prepared)
        print(
            f"[qa {sample_id}/{qa_id}] hypersimulation ready; "
            f"query({_hypergraph_size_text(query_hg)}), "
            f"matched pairs: {matched_vertex_pairs}, slices: {slice_count}, "
            f"elapsed: {prepared['prepared_elapsed_seconds']}s"
        )

    return prepared_rows


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    out_file = OUTPUT_DIR / "prepared.json"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not INSTANCES_ROOT.exists():
        print(f"instances root does not exist: {INSTANCES_ROOT}")
        print("Run 3-hypergraph-batch.py first.")
        return

    instance_dirs = sorted([path for path in INSTANCES_ROOT.iterdir() if path.is_dir()])
    if not instance_dirs:
        print(f"no instance directories found under {INSTANCES_ROOT}")
        print("Run 3-hypergraph-batch.py first.")
        return

    print(f"instances root: {INSTANCES_ROOT}")
    print(f"output file: {out_file}")
    print(f"pending hypersimulation instances: {len(instance_dirs)}")

    started_at = time.perf_counter()
    prepared_rows: list[dict] = []
    existing_map: dict[str, dict] = {}

    for idx, instance_dir in enumerate(instance_dirs, start=1):
        try:
            rows = _compose_conv_instance(instance_dir=instance_dir, existing_map=existing_map)
        except Exception as exc:
            print(f"[{idx}/{len(instance_dirs)}] hypersimulation failed: {instance_dir.name} :: {type(exc).__name__}: {exc}")
            continue

        cleaned_rows = [sanitize_hypersim_row(row) for row in rows]
        prepared_rows.extend(cleaned_rows)
        for row in cleaned_rows:
            key = entry_key(row)
            if key:
                existing_map[key] = row

        safe_write_json(
            out_file,
            _prepared_payload(
                prepared_rows,
                INSTANCES_ROOT,
                out_file,
                elapsed_seconds=time.perf_counter() - started_at,
            ),
        )
        print(f"[{idx}/{len(instance_dirs)}] hypersimulation finished: {instance_dir.name}, rows={len(cleaned_rows)}")

    print(f"batch hypersimulation finished: total_rows={len(prepared_rows)}, output={out_file}")


if __name__ == "__main__":
    main()
