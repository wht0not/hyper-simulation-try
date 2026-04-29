from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from langchain_ollama import ChatOllama
from tqdm import tqdm

from hyper_simulation.component.hyper_simulation import compute_hyper_simulation
from hyper_simulation.component.postprocess import get_simulation_slice, ranking_slices
from hyper_simulation.hypergraph.hypergraph import Hypergraph as LocalHypergraph, Vertex
from hyper_simulation.hypergraph.union import MultiHopFusion
from hyper_simulation.utils.chat_completion import get_invoke
from prompt import LOCOMO_HYPER_PROMPT
# Reuse metric functions from locomo_baseline
from locomo_baseline import (
    _evaluate_answer,
    _normalize_answer,
    _safe_write_json,
)

MIN_TOTAL_ANSWERABLE_COVERAGE = 0.5
MIN_NON_CRITICAL_COVERAGE = 0.6

def _sorted_index_from_name(path: Path) -> int:
    match_obj = re.fullmatch(r"data_hypergraph(\d+)\.pkl", path.name)
    if match_obj is None:
        return 10**9
    return int(match_obj.group(1))


def _load_query_hypergraph(instance_dir: Path, qa_id: str) -> LocalHypergraph | None:
    query_path = instance_dir / f"query_hypergraph_{qa_id}.pkl"
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

        evidence_items.append(
            {
                "index": data_idx,
                "path": str(data_path),
                "hypergraph": data_hg,
            }
        )

    return evidence_items


def _build_slice_text(slice_index: int, evidence_item: dict[str, Any]) -> str:
    hg = evidence_item.get("hypergraph")
    if hg is not None and hasattr(hg, "original_text") and hg.original_text:
        return _render_session_block(str(hg.original_text).strip(), int(evidence_item.get("index", slice_index)) + 1)
    return f"[slice {slice_index}]"


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
            session_date = stripped[len("DATE:"):].strip()
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


def _build_context_block(
    query: LocalHypergraph,
    simulation_slices: list[list[tuple[Vertex, Vertex]]],
    evidence_items: list[dict[str, Any]],
    vertex_ids: set[int],
    k: int = 10,
) -> tuple[list[int], str]:
    ranked_slice_indices = ranking_slices(query, simulation_slices, vertex_ids, k=k)
    consistent_indices: list[int] = []
    rendered_slices: list[str] = []
    for idx in ranked_slice_indices:
        if idx >= len(evidence_items):
            continue
        consistent_indices.append(int(evidence_items[idx].get("index", idx)))
        rendered_slices.append(_build_slice_text(idx, evidence_items[idx]))
    return consistent_indices, "\n\n".join(rendered_slices)


def _build_fallback_context(evidence_items: list[dict[str, Any]]) -> str:
    rendered = [_build_slice_text(i, item) for i, item in enumerate(evidence_items)]
    return "\n\n".join([text for text in rendered if text.strip()])


def _is_content_vertex(vertex: Vertex) -> bool:
    return not (vertex.is_verb() or vertex.is_virtual())


def _is_question_scaffold_vertex(vertex: Vertex) -> bool:
    text = vertex.text().strip()
    if not text:
        return True
    lowered = text.lower()
    if vertex.is_query():
        return True
    if lowered.startswith("?"):
        return True
    if lowered in {
        "what",
        "what kind",
        "what type",
        "which",
        "which kind",
        "which type",
        "where",
        "when",
        "who",
        "why",
        "how",
        "how many",
        "how much",
    }:
        return True
    return False


def _is_entity_fact_anchor(vertex: Vertex) -> bool:
    if not _is_content_vertex(vertex):
        return False
    if _is_question_scaffold_vertex(vertex):
        return False
    return vertex.has_entity()


def _is_critical_query_vertex(query_hg: LocalHypergraph, vertex: Vertex) -> bool:
    if not _is_content_vertex(vertex):
        return False
    if _is_question_scaffold_vertex(vertex):
        return False
    if any(hyperedge.root == vertex for hyperedge in query_hg.contained_edges.get(vertex, [])):
        return True
    return _is_entity_fact_anchor(vertex)


def _assess_answerability_from_hypersim(
    query_hg: LocalHypergraph,
    mapping: dict[int, set[int] | list[int]],
    q_map: dict[int, Vertex],
) -> dict[str, Any]:
    matched_qids = {int(q_id) for q_id, d_ids in mapping.items() if d_ids}
    matched_items: list[str] = []
    unmatched_items: list[str] = []
    matched_critical_items: list[str] = []
    unmatched_critical_items: list[str] = []
    total_items = 0
    critical_total_items = 0
    non_critical_total_items = 0
    non_critical_matched_items = 0
    for vertex in query_hg.vertices:
        if not _is_content_vertex(vertex):
            continue
        text = vertex.text().strip()
        if not text:
            continue
        is_critical = _is_critical_query_vertex(query_hg, vertex)
        total_items += 1
        if is_critical:
            critical_total_items += 1
        if vertex.id in matched_qids and vertex.id in q_map:
            matched_items.append(text)
            if is_critical:
                matched_critical_items.append(text)
            else:
                non_critical_matched_items += 1
        else:
            unmatched_items.append(text)
            if is_critical:
                unmatched_critical_items.append(text)
        if not is_critical:
            non_critical_total_items += 1
    coverage = (len(matched_items) / total_items) if total_items > 0 else 0.0
    non_critical_coverage = (
        non_critical_matched_items / non_critical_total_items
        if non_critical_total_items > 0
        else 1.0
    )
    critical_all_matched = critical_total_items > 0 and not unmatched_critical_items
    answerability = "answerable" if (
        critical_all_matched
        or (coverage >= MIN_TOTAL_ANSWERABLE_COVERAGE
        and non_critical_coverage >= MIN_NON_CRITICAL_COVERAGE)
    ) else "not_answerable"
    return {
        "decision": answerability,
        "critical_all_matched": critical_all_matched,
        "coverage": coverage,
        "non_critical_coverage": non_critical_coverage,
        "matched_query_items": matched_items,
        "unmatched_query_items": unmatched_items,
        "matched_critical_query_items": matched_critical_items,
        "unmatched_critical_query_items": unmatched_critical_items,
        "total_query_items": total_items,
    }

def _window_tag(dataset_path: Path) -> str:
    name = dataset_path.stem.lower()
    if "1k" in name:
        return "1K"
    if "4k" in name:
        return "4K"
    if "8k" in name:
        return "8K"
    if "16k" in name:
        return "16K"
    if "32k" in name:
        return "32K"
    return dataset_path.stem

def run_locomo_hyper_simulation(
    instances_root: str = "/home/vincent/hyper-simulation-try/data/hypergraphs/locomo-1K",
    output_dir: str = "/home/vincent/hyper-simulation-try/tests/tasks/memory/locomo/results/test",
    model_name: str = "qwen3.5:9b",
    temperature: float = 0.1,
    limit: int | None = None,
) -> dict[str, Any]:
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    tag = _window_tag(Path(instances_root))
    out_file = out_root / f"locomo_hyper_simulation_{tag}.json"
    
    # Try to load existing
    existing_results = []
    category_counts: dict[int, int] = {k: 0 for k in [1, 2, 3, 4, 5]}
    category_acc: dict[int, float] = {k: 0.0 for k in [1, 2, 3, 4, 5]}
    
    if out_file.exists():
        try:
            payload = json.loads(out_file.read_text(encoding="utf-8"))
            for row in payload.get("results", []):
                existing_results.append(row)
                cat = row.get("category", -1)
                if cat in category_counts:
                    category_counts[cat] += 1
                    try:
                        metrics = row.get("metrics", {})
                        category_acc[cat] += float(metrics.get("locomo_score", 0.0))
                    except Exception:
                        pass
        except Exception:
            pass

    def _summary() -> dict[str, Any]:
        total_q = sum(category_counts.values())
        total_acc = sum(category_acc.values())
        cat_summary = {}
        for k in [4, 1, 2, 3, 5]:
            c_total = category_counts.get(k, 0)
            c_acc = category_acc.get(k, 0.0)
            cat_summary[str(k)] = {
                "total": c_total,
                "accuracy": round(c_acc / c_total, 3) if c_total > 0 else 0.0
            }
        return {
            "method": "hyper_simulation",
            "model_name": model_name,
            "total": total_q,
            "overall_accuracy": round(total_acc / total_q, 3) if total_q > 0 else 0.0,
            "by_category": cat_summary
        }

    instances_dir = Path(instances_root)
    if not instances_dir.exists():
        return {"error": "instances dir not found"}

    model = ChatOllama(model=model_name, temperature=temperature, reasoning=False, num_predict=8192)
    
    dirs = [d for d in instances_dir.iterdir() if d.is_dir()]
    if limit is not None and limit > 0:
        dirs = dirs[:limit]
        
    pbar = tqdm(dirs, desc="locomo/hyper_simulation")
    
    for instance_dir in pbar:
        meta_path = instance_dir / "metadata.json"
        if not meta_path.exists():
            continue
            
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
            
        sample_id = meta.get("sample_id", "")
        qa_list = meta.get("qa_list", [])
        d_start = meta.get("d_start", "")
        evidence_items = _load_data_hypergraphs(instance_dir)
        if not evidence_items:
            continue

        data_hgs = [ev["hypergraph"] for ev in evidence_items]
        fusion = MultiHopFusion()
        instance_merged_hg, _ = fusion.merge_hypergraphs(data_hgs)

        for qa_item in qa_list:
            qa_id = qa_item.get("qa_id", "")
            q = qa_item.get("question", "")
            answer = qa_item.get("answer", "")
            category = qa_item.get("category", -1)
            
            # Check if already evaluated
            already_done = any(r.get("sample_id") == sample_id and r.get("qa_id") == qa_id for r in existing_results)
            if already_done:
                continue

            query_hg = _load_query_hypergraph(instance_dir, qa_id)
            if query_hg is None:
                continue
                
            try:
                # compute_hyper_simulation returns (mapping, q_map, d_map)
                mapping, q_map, d_map = compute_hyper_simulation(query_hg, instance_merged_hg)
                simulation = [
                    (q_map[q_id], d_map[d_id])
                    for q_id, d_ids in mapping.items()
                    for d_id in d_ids
                    if q_id in q_map and d_id in d_map
                ]
                simulation_slices = get_simulation_slice(query_hg, instance_merged_hg, simulation, len(data_hgs))
                full_query_vertex_ids = {
                    vertex.id for vertex in query_hg.vertices if _is_content_vertex(vertex)
                }
                consistent_indices, ranked_context = _build_context_block(
                    query=query_hg,
                    simulation_slices=simulation_slices,
                    evidence_items=evidence_items,
                    vertex_ids=full_query_vertex_ids,
                )
                if ranked_context.strip():
                    context_text = ranked_context
                else:
                    consistent_indices = [int(ev.get("index", i)) for i, ev in enumerate(evidence_items)]
                    context_text = _build_fallback_context(evidence_items)
                if d_start:
                    context_text = d_start + "\n\n" + context_text

                answerability = _assess_answerability_from_hypersim(
                    query_hg=query_hg,
                    mapping=mapping,
                    q_map=q_map,
                )
                answerability_decision = str(answerability["decision"])
                critical_all_matched = bool(answerability["critical_all_matched"])
                answerability_coverage = float(answerability["coverage"])
                non_critical_coverage = float(answerability["non_critical_coverage"])
                matched_query_items = list(answerability["matched_query_items"])
                unmatched_query_items = list(answerability["unmatched_query_items"])
                matched_critical_query_items = list(answerability["matched_critical_query_items"])
                unmatched_critical_query_items = list(answerability["unmatched_critical_query_items"])
                total_query_items = int(answerability["total_query_items"])
                if answerability_decision != "answerable":
                    prediction = "no information available"
                    metrics = _evaluate_answer(prediction, answer, category)
                    if category in category_counts:
                        category_counts[category] += 1
                        category_acc[category] += float(metrics["locomo_score"])
                    out_row = {
                        "sample_id": sample_id,
                        "qa_id": qa_id,
                        "q": q,
                        "answer": answer,
                        "prediction": prediction,
                        "category": category,
                        "metrics": metrics,
                        "consistent_context": consistent_indices,
                        "stage1_prediction": "",
                        "answerability_decision": answerability_decision,
                        "critical_all_matched": critical_all_matched,
                        "answerability_coverage": answerability_coverage,
                        "non_critical_coverage": non_critical_coverage,
                        "matched_query_items": matched_query_items,
                        "unmatched_query_items": unmatched_query_items,
                        "matched_critical_query_items": matched_critical_query_items,
                        "unmatched_critical_query_items": unmatched_critical_query_items,
                        "total_query_items": total_query_items,
                    }
                    existing_results.append(out_row)
                    _safe_write_json(out_file, {"summary": _summary(), "results": existing_results})
                    continue
                
                # 2. Build Prompt
                prompt = LOCOMO_HYPER_PROMPT.format(
                    context_text=context_text,
                    question=q,
                )
                
                # 3. LLM invoke
                raw = get_invoke(model, prompt)
                stage1_prediction = _normalize_answer(raw)
                prediction = stage1_prediction
                
                # 4. Metrics
                metrics = _evaluate_answer(prediction, answer, category)
                
                if category in category_counts:
                    category_counts[category] += 1
                    category_acc[category] += float(metrics["locomo_score"])
                    
                out_row = {
                    "sample_id": sample_id,
                    "qa_id": qa_id,
                    "q": q,
                    "answer": answer,
                    "prediction": prediction,
                    "category": category,
                    "metrics": metrics,
                    "consistent_context": consistent_indices,
                    "stage1_prediction": stage1_prediction,
                    "answerability_decision": answerability_decision,
                    "critical_all_matched": critical_all_matched,
                    "answerability_coverage": answerability_coverage,
                    "non_critical_coverage": non_critical_coverage,
                    "matched_query_items": matched_query_items,
                    "unmatched_query_items": unmatched_query_items,
                    "matched_critical_query_items": matched_critical_query_items,
                    "unmatched_critical_query_items": unmatched_critical_query_items,
                    "total_query_items": total_query_items,
                }
                existing_results.append(out_row)
                
                _safe_write_json(out_file, {"summary": _summary(), "results": existing_results})
                
            except Exception as e:
                tqdm.write(f"[ERROR] {instance_dir.name} qa_id={qa_id}: {e}")
                continue

    summary_data = _summary()
    _safe_write_json(out_file, {"summary": summary_data, "results": existing_results})
    return summary_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LoCoMo Hyper Simulation")
    parser.add_argument("--instances-root", type=str, default="/home/vincent/hyper-simulation-try/data/hypergraphs/locomo-1K")
    parser.add_argument("--output-dir", type=str, default="/home/vincent/hyper-simulation-try/tests/tasks/memory/locomo/results/test")
    parser.add_argument("--model-name", type=str, default="qwen3.5:9b")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    report = run_locomo_hyper_simulation(
        instances_root=args.instances_root,
        output_dir=args.output_dir,
        model_name=args.model_name,
        temperature=args.temperature,
        limit=(args.limit or None),
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
