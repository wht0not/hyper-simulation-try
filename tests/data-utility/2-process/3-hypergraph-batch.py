from __future__ import annotations

import json
import logging
import sys
import typing
from collections import defaultdict
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
DATASET_PATH = REPO_ROOT / "tests" / "data-utility" / "data" / "sample-locomo.json"
INSTANCES_ROOT = REPO_ROOT / "tests" / "data-utility" / "data" / "hypergraphs" / "locomo" / "context"
BATCH_SIZE = 8

for candidate in (REPO_ROOT, SRC_ROOT, LOCOMO_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from method.hyper_simulation.build import (
    batch_text_to_hypergraph,
    load_entries_for_build,
    setup_gpu_nlp,
)


def _hypergraph_size_text(hypergraph) -> str:
    vertex_count = len(getattr(hypergraph, "vertices", []) or [])
    edge_count = len(getattr(hypergraph, "hyperedges", []) or [])
    return f"vertex: {vertex_count}, edge: {edge_count}"


def _group_entries_by_sample(entries: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        sample_id = str(entry.get("sample_id", "")).strip()
        question = str(entry.get("q", "")).strip()
        d_items = entry.get("d", [])
        if not sample_id or not question or not isinstance(d_items, list) or not d_items:
            continue
        grouped[sample_id].append(entry)
    return dict(grouped)


def _write_sample_metadata(sample_dir: Path, sample_id: str, sample_entries: list[dict]) -> None:
    qa_list = []
    for entry in sample_entries:
        qa_list.append(
            {
                "qa_id": str(entry.get("qa_id", "")).strip(),
                "question": str(entry.get("q", "")).strip(),
                "answer": entry.get("answer"),
                "category": entry.get("category"),
            }
        )
    payload = {
        "sample_id": sample_id,
        "d_start": str(sample_entries[0].get("d_start", "")).strip() if sample_entries else "",
        "qa_list": qa_list,
    }
    sample_dir.mkdir(parents=True, exist_ok=True)
    (sample_dir / "metadata.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _build_query_hypergraphs(
    nlp,
    tasks_query: list[tuple[Path, str, str]],
) -> int:
    if not tasks_query:
        return 0
    payload = [
        {"text": question, "meta": {"sample_dir": sample_dir, "qa_id": qa_id}}
        for sample_dir, qa_id, question in tasks_query
    ]
    built = 0
    total = len(tasks_query)
    for meta, hypergraph in batch_text_to_hypergraph(
        nlp=nlp,
        texts_with_metadata=payload,
        batch_size=BATCH_SIZE,
        is_query=True,
    ):
        if hypergraph is None:
            print(f"[query] failed: qa_id={meta.get('qa_id', 'unknown')} :: {meta.get('error', 'unknown error')}")
            continue
        sample_dir = Path(meta["sample_dir"])
        qa_id = str(meta["qa_id"])
        hypergraph.save(str(sample_dir / f"query_hypergraph{qa_id}.pkl"))
        built += 1
        print(
            f"[query {built}/{total}] hypergraph ready: "
            f"{sample_dir.name}/query_hypergraph{qa_id}.pkl; {_hypergraph_size_text(hypergraph)}"
        )
    return built


def _build_data_hypergraphs(
    nlp,
    tasks_data: list[tuple[Path, int, str]],
) -> int:
    if not tasks_data:
        return 0
    built = 0
    total = len(tasks_data)
    chunk_size = 64
    for offset in range(0, total, chunk_size):
        sub_tasks = tasks_data[offset : offset + chunk_size]
        payload = [
            {"text": text, "meta": {"instance_dir": instance_dir, "d_idx": d_idx}}
            for instance_dir, d_idx, text in sub_tasks
        ]
        for meta, hypergraph in batch_text_to_hypergraph(
            nlp=nlp,
            texts_with_metadata=payload,
            batch_size=BATCH_SIZE,
            is_query=False,
        ):
            if hypergraph is None:
                print(
                    f"[data] failed: {Path(meta.get('instance_dir', '.')).name}/"
                    f"data_hypergraph{meta.get('d_idx', 'x')}.pkl :: {meta.get('error', 'unknown error')}"
                )
                continue
            instance_dir = Path(meta["instance_dir"])
            d_idx = int(meta["d_idx"])
            hypergraph.save(str(instance_dir / f"data_hypergraph{d_idx}.pkl"))
            built += 1
            print(
                f"[data {built}/{total}] hypergraph ready: "
                f"{instance_dir.name}/data_hypergraph{d_idx}.pkl; {_hypergraph_size_text(hypergraph)}"
            )
    return built


def _prepare_conv_tasks(entries: list[dict]) -> tuple[list[tuple[Path, str, str]], list[tuple[Path, int, str]]]:
    tasks_query: list[tuple[Path, str, str]] = []
    tasks_data: list[tuple[Path, int, str]] = []
    grouped_entries = _group_entries_by_sample(entries)

    for sample_id, sample_entries in grouped_entries.items():
        sample_dir = INSTANCES_ROOT / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        _write_sample_metadata(sample_dir, sample_id, sample_entries)

        first_entry = sample_entries[0]
        data_items = first_entry.get("d", [])
        for d_idx, text in enumerate(data_items):
            text = str(text).strip()
            if not text:
                continue
            tasks_data.append((sample_dir, d_idx, text))

        for entry in sample_entries:
            qa_id = str(entry.get("qa_id", "")).strip()
            question = str(entry.get("q", "")).strip()
            if not qa_id or not question:
                continue
            tasks_query.append((sample_dir, qa_id, question))

    return tasks_query, tasks_data


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    dataset_file, entries = load_entries_for_build(DATASET_PATH)
    INSTANCES_ROOT.mkdir(parents=True, exist_ok=True)
    tasks_query, tasks_data = _prepare_conv_tasks(entries)

    print(f"sample dataset: {dataset_file}")
    print(f"instances root: {INSTANCES_ROOT}")
    print("layout: <context>/<sample_id>/query_hypergraph<qa_id>.pkl + data_hypergraph<idx>.pkl")
    print(f"pending query hypergraphs: {len(tasks_query)}")
    print(f"pending data hypergraphs: {len(tasks_data)}")

    if not tasks_query and not tasks_data:
        print("No pending hypergraphs to build.")
        return

    nlp = setup_gpu_nlp()
    built_queries = _build_query_hypergraphs(nlp, tasks_query)
    built_data = _build_data_hypergraphs(nlp, tasks_data)

    print(
        "batch hypergraph build finished: "
        f"queries={built_queries}/{len(tasks_query)}, "
        f"data={built_data}/{len(tasks_data)}"
    )


if __name__ == "__main__":
    main()
