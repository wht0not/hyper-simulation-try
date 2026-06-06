from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from statistics import fmean
from typing import Any

from hyper_simulation.hypergraph.hypergraph import Hypergraph as LocalHypergraph


def _sorted_index_from_name(path: Path) -> int:
    match_obj = re.fullmatch(r"data_hypergraph(\d+)\.pkl", path.name)
    if match_obj is not None:
        return int(match_obj.group(1))
    match_obj = re.fullmatch(r"data_(\d+)\.pkl", path.name)
    if match_obj is not None:
        return int(match_obj.group(1))
    return 10**9


def _query_key_from_question(question: str) -> str:
    normalized = " ".join(str(question or "").strip().split())
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def _list_data_files(instance_dir: Path) -> list[Path]:
    all_files: dict[str, Path] = {}
    for pattern in ("data_hypergraph*.pkl", "data_*.pkl"):
        for path in instance_dir.glob(pattern):
            all_files[path.name] = path
    return sorted(all_files.values(), key=lambda path: (_sorted_index_from_name(path), path.name))


def _collect_instance_dirs(context_root: Path) -> list[Path]:
    return sorted([path for path in context_root.iterdir() if path.is_dir() and (path / "metadata.json").exists()])


def _coerce_category(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return -1


def _load_data_stats(instance_dir: Path) -> dict[str, int]:
    data_paths = _list_data_files(instance_dir)
    data_vertices_sum = 0
    data_hyperedges_sum = 0
    loaded_data_count = 0
    for data_path in data_paths:
        try:
            data_hg = LocalHypergraph.load(str(data_path))
            data_vertices_sum += len(data_hg.vertices)
            data_hyperedges_sum += len(data_hg.hyperedges)
            loaded_data_count += 1
        except Exception as exc:
            print(f"[Warn] {instance_dir.name}: failed to load {data_path.name} ({type(exc).__name__})")
    return {
        "data_vertices_sum": data_vertices_sum,
        "data_hyperedges_sum": data_hyperedges_sum,
        "data_hypergraph_count": loaded_data_count,
    }


def _query_stats(query_path: Path, cache: dict[str, dict[str, int]]) -> dict[str, int] | None:
    key = query_path.stem
    if key in cache:
        return cache[key]
    try:
        query_hg = LocalHypergraph.load(str(query_path))
    except Exception as exc:
        print(f"[Warn] failed to load query {query_path.name} ({type(exc).__name__})")
        return None
    stat = {
        "query_vertices": len(query_hg.vertices),
        "query_hyperedges": len(query_hg.hyperedges),
        "query_hypergraph_count": 1,
    }
    cache[key] = stat
    return stat


def _print_group(title: str, rows: list[dict[str, float]]) -> None:
    if not rows:
        print(f"{title}: no valid rows")
        return
    avg_query_vertices = fmean(row["query_vertices"] for row in rows)
    avg_query_hyperedges = fmean(row["query_hyperedges"] for row in rows)
    avg_query_hg_count = fmean(row["query_hypergraph_count"] for row in rows)
    avg_data_vertices_sum = fmean(row["data_vertices_sum"] for row in rows)
    avg_data_hyperedges_sum = fmean(row["data_hyperedges_sum"] for row in rows)
    avg_data_hg_count = fmean(row["data_hypergraph_count"] for row in rows)
    print(f"{title} (query count={len(rows)}):")
    print(f"  |V_Q| (query vertices):       {avg_query_vertices:.4f}")
    print(f"  |E_Q| (query hyperedges):     {avg_query_hyperedges:.4f}")
    print(f"  query_hypergraph count:       {avg_query_hg_count:.4f}")
    print(f"  |V| (data vertices sum):      {avg_data_vertices_sum:.4f}")
    print(f"  |E| (data hyperedges sum):    {avg_data_hyperedges_sum:.4f}")
    print(f"  data_hypergraph count:        {avg_data_hg_count:.4f}")


def count_hypergraphs(query_dir: str, context_dir: str) -> None:
    query_root = Path(query_dir)
    context_root = Path(context_dir)
    if not query_root.exists():
        raise FileNotFoundError(f"Query directory not found: {query_root}")
    if not context_root.exists():
        raise FileNotFoundError(f"Context directory not found: {context_root}")

    instance_dirs = _collect_instance_dirs(context_root)
    if not instance_dirs:
        raise FileNotFoundError(f"No metadata.json found under context dir: {context_root}")

    query_cache: dict[str, dict[str, int]] = {}
    per_query_rows: list[dict[str, float]] = []
    missing_query_count = 0

    for instance_dir in instance_dirs:
        meta_path = instance_dir / "metadata.json"
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[Warn] {instance_dir.name}: failed to read metadata ({type(exc).__name__})")
            continue
        qa_list = meta.get("qa_list", [])
        if not isinstance(qa_list, list):
            continue
        data_stats = _load_data_stats(instance_dir)
        for qa in qa_list:
            if not isinstance(qa, dict):
                continue
            question = str(qa.get("question", "")).strip()
            if not question:
                continue
            category = _coerce_category(qa.get("category", -1))
            query_key = _query_key_from_question(question)
            query_path = query_root / f"{query_key}.pkl"
            if not query_path.exists():
                missing_query_count += 1
                continue
            q_stats = _query_stats(query_path, query_cache)
            if q_stats is None:
                continue
            per_query_rows.append(
                {
                    "category": float(category),
                    **{key: float(value) for key, value in q_stats.items()},
                    **{key: float(value) for key, value in data_stats.items()},
                }
            )

    if not per_query_rows:
        raise RuntimeError("No valid query-mapped rows could be loaded.")

    categories = sorted({int(row["category"]) for row in per_query_rows})
    print("=" * 72)
    print(f"Query dir:   {query_root.resolve()}")
    print(f"Context dir: {context_root.resolve()}")
    print(f"Total context instances: {len(instance_dirs)}")
    print(f"Valid query-mapped rows: {len(per_query_rows)}")
    print(f"Missing query hypergraphs: {missing_query_count}")
    print("=" * 72)
    _print_group("All categories", per_query_rows)
    print("-" * 72)
    for category in categories:
        rows = [row for row in per_query_rows if int(row["category"]) == category]
        _print_group(f"Category {category}", rows)
        print("-" * 72)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Count query/data hypergraph statistics for LoCoMo with query-based categorization. "
            "Query hypergraphs are loaded from query dir, data hypergraphs are loaded from context instance dirs."
        )
    )
    parser.add_argument(
        "--query-dir",
        type=str,
        default="/home/vincent/hyper-simulation-try/data/hypergraphs/locomo/query",
        help="Directory containing query hypergraph .pkl files.",
    )
    parser.add_argument(
        "--context-dir",
        type=str,
        default="/home/vincent/hyper-simulation-try/data/hypergraphs/locomo/context",
        help="Directory containing context instance subfolders with metadata/data_hypergraph files.",
    )
    args = parser.parse_args()

    count_hypergraphs(query_dir=args.query_dir, context_dir=args.context_dir)


if __name__ == "__main__":
    main()
