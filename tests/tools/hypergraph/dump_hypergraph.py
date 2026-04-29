import argparse
import pprint
import re
from pathlib import Path
from hyper_simulation.hypergraph.hypergraph import Hypergraph as LocalHypergraph
import json

RESULT_DIR = Path("/home/vincent/hyper-simulation-try/tests/tasks/memory/locomo/results/test")


def _save_structured_dump(branch: str, payload: dict) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULT_DIR / f"{branch}_dump.txt"
    out_path.write_text(pprint.pformat(payload, sort_dicts=False, width=140), encoding="utf-8")


def _node_to_dict(node) -> dict:
    return {
        "text": getattr(node, "text", ""),
        "index": getattr(node, "index", None),
        "source_id": getattr(node, "source_id", None),
        "pos": getattr(getattr(node, "pos", None), "name", str(getattr(node, "pos", None))),
        "dep": getattr(getattr(node, "dep", None), "name", str(getattr(node, "dep", None))),
        "ent": getattr(getattr(node, "ent", None), "name", str(getattr(node, "ent", None))),
        "is_query": getattr(node, "is_query", False),
    }


def _vertex_to_dict(vertex, idx: int) -> dict:
    return {
        "index": idx,
        "id": getattr(vertex, "id", None),
        "text": vertex.text() if hasattr(vertex, "text") else "",
        "type": vertex.type() if hasattr(vertex, "type") else "",
        "query_type": vertex.query_type() if hasattr(vertex, "query_type") else None,
        "is_query": vertex.is_query() if hasattr(vertex, "is_query") else False,
        "is_virtual": vertex.is_virtual() if hasattr(vertex, "is_virtual") else False,
        "is_verb": vertex.is_verb() if hasattr(vertex, "is_verb") else False,
        "provenance_ids": sorted(list(getattr(vertex, "provenance_ids", set()))),
        "nodes": [_node_to_dict(n) for n in getattr(vertex, "nodes", [])],
    }


def _hyperedge_to_dict(hyperedge, idx: int) -> dict:
    father = getattr(hyperedge, "father", None)
    return {
        "index": idx,
        "desc": getattr(hyperedge, "desc", ""),
        "full_desc": getattr(hyperedge, "full_desc", ""),
        "start": getattr(hyperedge, "start", None),
        "end": getattr(hyperedge, "end", None),
        "root_id": getattr(getattr(hyperedge, "root", None), "id", None),
        "vertex_ids": [getattr(v, "id", None) for v in getattr(hyperedge, "vertices", [])],
        "father_root_id": getattr(getattr(father, "root", None), "id", None) if father is not None else None,
        "hypergraph_id": getattr(hyperedge, "hypergraph_id", None),
        "text": hyperedge.text() if hasattr(hyperedge, "text") else "",
    }


def _contained_edges_summary(hg: LocalHypergraph) -> list[dict]:
    items = []
    contained = getattr(hg, "contained_edges", {})
    for vertex, edges in contained.items():
        items.append(
            {
                "vertex_id": getattr(vertex, "id", None),
                "vertex_text": vertex.text() if hasattr(vertex, "text") else "",
                "edge_count": len(edges),
                "edge_descs": [getattr(e, "desc", "") for e in edges],
            }
        )
    return items


def _hypergraph_full_dump(hg: LocalHypergraph, with_original_text: bool = True) -> dict:
    payload = {
        "vertex_count": len(getattr(hg, "vertices", [])),
        "hyperedge_count": len(getattr(hg, "hyperedges", [])),
        "vertices": [_vertex_to_dict(v, i) for i, v in enumerate(getattr(hg, "vertices", []))],
        "hyperedges": [_hyperedge_to_dict(e, i) for i, e in enumerate(getattr(hg, "hyperedges", []))],
        "contained_edges": _contained_edges_summary(hg),
        "doc": {
            "sentence_count": len(getattr(getattr(hg, "doc", None), "sentences", []) or []),
            "token_count": len(getattr(getattr(hg, "doc", None), "nodes", []) or []),
        },
    }
    if with_original_text:
        payload["original_text"] = getattr(hg, "original_text", None)
    return payload

def _dump_locomo_instance(instance_dir: str, qa_id: str) -> None:
    base = Path(instance_dir)
    query_path = base / f"query_hypergraph_{qa_id}.pkl"
    if not query_path.exists():
        raise FileNotFoundError(f"query hypergraph not found: {query_path}")

    query_hg = LocalHypergraph.load(str(query_path))
    structured_dump = {
        "branch": "locomo",
        "instance_dir": str(base),
        "qa_id": str(qa_id),
        "qa_item": None,
        "query_hypergraph": {},
        "data_hypergraphs": [],
    }
    print(f"LoCoMo Instance: {base}")
    print(f"QA ID: {qa_id}")

    metadata_path = base / "metadata.json"
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            qa_list = metadata.get("qa_list", [])
            qa_item = next((item for item in qa_list if str(item.get("qa_id", "")) == str(qa_id)), None)
            if qa_item is not None:
                structured_dump["qa_item"] = {
                    "question": qa_item.get("question", ""),
                    "answer": qa_item.get("answer", ""),
                    "category": qa_item.get("category", ""),
                }
                print(f"Question: {qa_item.get('question', '')}")
                print(f"Answer: {qa_item.get('answer', '')}")
                print(f"Category: {qa_item.get('category', '')}")
        except Exception:
            pass

    print("\nQuery Hypergraph:")
    print(f"Vertices ({len(query_hg.vertices)}):")
    for i, v in enumerate(query_hg.vertices):
        if v.is_query():
            print(f"  - [{i}] '{v.text()}' TYPE: {v.query_type()}")
            continue
        print(f"  - [{i}] '{v.text()}' TYPE: {v.type()}")
    structured_dump["query_hypergraph"] = _hypergraph_full_dump(query_hg, with_original_text=False)

    data_paths = sorted(
        base.glob("data_hypergraph*.pkl"),
        key=lambda p: int(re.fullmatch(r"data_hypergraph(\d+)\.pkl", p.name).group(1))
        if re.fullmatch(r"data_hypergraph(\d+)\.pkl", p.name) else 10**9,
    )

    for data_path in data_paths:
        match_obj = re.fullmatch(r"data_hypergraph(\d+)\.pkl", data_path.name)
        if match_obj is None:
            continue
        idx = int(match_obj.group(1))
        try:
            data_hg = LocalHypergraph.load(str(data_path))
        except Exception as e:
            print(f"\nData Hypergraph {idx}: LOAD FAILED ({e})")
            structured_dump["data_hypergraphs"].append(
                {"index": idx, "path": str(data_path), "status": "load_failed", "error": str(e)}
            )
            continue
        print(f"\nData Hypergraph {idx}:")
        print(f"Path: {data_path}")
        if hasattr(data_hg, "original_text") and data_hg.original_text:
            print(data_hg.original_text)
        print(f"Vertices ({len(data_hg.vertices)}):")
        item = {
            "index": idx,
            "path": str(data_path),
            "status": "ok",
            "hypergraph": _hypergraph_full_dump(data_hg, with_original_text=True),
        }
        for i, v in enumerate(data_hg.vertices):
            print(f"  - [{i}] '{v.text()}' TYPE: {v.type()}")
        structured_dump["data_hypergraphs"].append(item)

    _save_structured_dump("locomo", structured_dump)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dump hypergraph debug info")
    parser.add_argument("--locomo-instance-dir", type=str, default="/home/vincent/hyper-simulation-try/data/hypergraphs/locomo-1K/6ceb0737e9804f98")
    parser.add_argument("--qa-id", type=str, default="1")
    args = parser.parse_args()
    if not args.locomo_instance_dir or not args.qa_id:
        raise ValueError("--locomo-instance-dir and --qa-id must be provided together")
    _dump_locomo_instance(args.locomo_instance_dir, args.qa_id)
    
