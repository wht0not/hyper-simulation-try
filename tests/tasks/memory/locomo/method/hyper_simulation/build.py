"""
Build Locomo hypergraphs with batch support.

Supported input schema:
  data/bench/locomo-main/locomo-data/locomo_8K.json
  (also 16K/32K)

Output layout:
  data/hypergraphs/locomo/query/
    - <sha1(question)>.pkl
  data/hypergraphs/locomo/<source>/<instance_id>/
    - data_hypergraph0.pkl (session 1)
    - data_hypergraph1.pkl (session 2)
    - metadata.json
"""

import json
import logging
import hashlib
import os
from argparse import ArgumentParser
from pathlib import Path
import time
from typing import Any, Iterator

import spacy
from spacy.language import Language
from tqdm import tqdm

from hyper_simulation.component.build_hypergraph import (
    clean_text_for_spacy,
    doc_to_hypergraph,
    generate_instance_id,
    text_to_hypergraph,
)

logger = logging.getLogger(__name__)
local_model_path = "/home/vincent/.cache/huggingface/hub/models--biu-nlp--lingmess-coref/snapshots/fa5d8a827a09388d03adbe9e800c7d8c509c3935"
METHOD_NAME = "hyper_simulation"
QUERY_DIRNAME = "query"
LOW_D_BYPASS_THRESHOLD = 3


def _coerce_category(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return -1


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


def _resolve_entry_answer(entry: dict[str, Any]) -> Any:
    category = _coerce_category(entry.get("category"))
    answer = entry.get("answer")
    adversarial_answer = entry.get("adversarial_answer")
    if category == 5:
        if adversarial_answer is not None and str(adversarial_answer).strip():
            return adversarial_answer
        return answer
    if answer is not None:
        return answer
    if adversarial_answer is not None and str(adversarial_answer).strip():
        return adversarial_answer
    return answer


def locomo_root_from_instances_root(instances_root: str | Path) -> Path:
    instances_root_path = Path(instances_root)
    for candidate in (instances_root_path, *instances_root_path.parents):
        if candidate.name == "locomo":
            return candidate
    raise ValueError(f"instances_root must be under a 'locomo' directory: {instances_root_path}")


def shared_query_output_dir(instances_root: str | Path) -> Path:
    return locomo_root_from_instances_root(instances_root) / QUERY_DIRNAME


def query_key_from_question(question: str) -> str:
    normalized = " ".join(str(question or "").strip().split())
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def _entry_instance_id(sample_id: str, qa_id: str, question: str) -> str:
    stable_id = f"{str(sample_id).strip()}::{str(qa_id).strip()}::{str(question).strip()}"
    return generate_instance_id(stable_id)


def _normalize_hyper_items(entry: dict[str, Any]) -> list[dict[str, Any]]:
    raw_items = entry.get("hyper_d_items")
    if not isinstance(raw_items, list):
        raw_items = entry.get("hyper_d")
    if not isinstance(raw_items, list):
        raw_items = entry.get("d", [])
    if not isinstance(raw_items, list):
        raw_items = [raw_items]

    normalized_items: list[dict[str, Any]] = []
    for idx, item in enumerate(raw_items):
        if isinstance(item, dict):
            text = str(item.get("text", item.get("memory", ""))).strip()
            if not text:
                continue
            normalized_items.append(
                {
                    "index": idx,
                    "text": text,
                    "speaker_tag": str(item.get("speaker_tag", "")).strip(),
                    "speaker_name": str(item.get("speaker_name", "")).strip(),
                }
            )
            continue
        text = str(item).strip()
        if not text:
            continue
        normalized_items.append({"index": idx, "text": text, "speaker_tag": "", "speaker_name": ""})
    return normalized_items


def _compact_source_entry(entry: dict[str, Any], hyper_items: list[dict[str, Any]]) -> dict[str, Any]:
    excluded = {
        "prompt",
        "prediction",
        "raw_prediction",
        "metrics",
    }
    compact = {key: value for key, value in entry.items() if key not in excluded}
    compact["hyper_d_items"] = hyper_items
    return compact


def load_entries_for_build(dataset_path: str | Path) -> tuple[Path, list[dict[str, Any]]]:
    """Load any generic entries/results dataset for hypergraph build, including rag retrieval outputs."""
    dataset_file = Path(dataset_path)
    if not dataset_file.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_file}")

    payload = json.loads(dataset_file.read_text(encoding="utf-8"))
    rows = payload.get("entries", payload.get("results", [])) if isinstance(payload, dict) else []
    if not isinstance(rows, list):
        rows = []
    entries = [row for row in rows if isinstance(row, dict)]
    return dataset_file, entries


def setup_gpu_nlp(model_name: str = "en_core_web_trf") -> Language:
	"""Initialize spaCy model with optional GPU and fastcoref."""
	try:
		require_gpu_fn = getattr(spacy, "require_gpu", None)
		if callable(require_gpu_fn) and require_gpu_fn():
			logger.info("GPU is enabled for spaCy")
		else:
			logger.warning("GPU not available for spaCy, fallback to CPU")
	except Exception as exc:
		logger.warning("GPU check failed (%s), fallback to CPU", exc)

	try:
		nlp = spacy.load(model_name)
	except OSError as exc:
		logger.error("spaCy model %s not found", model_name)
		raise exc

	if "fastcoref" not in nlp.pipe_names:
		try:
			nlp.add_pipe(
				"fastcoref",
				config={
					"model_architecture": "LingMessCoref",
					"model_path": local_model_path,
					"device": "cuda",
				},
			)
			logger.info("fastcoref added with CUDA")
		except Exception as exc:
			logger.warning("Failed to add fastcoref: %s", exc)

	return nlp


def batch_text_to_hypergraph(
    nlp: Language,
    texts_with_metadata: list[dict],
    batch_size: int = 16,
    is_query: bool = False,
) -> Iterator[tuple[dict, Any]]:
    texts = [clean_text_for_spacy(item["text"]) for item in texts_with_metadata]
    metadatas = [item["meta"] for item in texts_with_metadata]
    original_texts = [item["text"] for item in texts_with_metadata]

    component_cfg = {"fastcoref": {"resolve_text": True}} if "fastcoref" in nlp.pipe_names else {}

    try:
        docs_list = list(
            nlp.pipe(
                texts,
                component_cfg=component_cfg,
                batch_size=max(1, batch_size),
            )
        )

        for doc, metadata, original_text in zip(docs_list, metadatas, original_texts):
            try:
                hypergraph = doc_to_hypergraph(doc, original_text, is_query=is_query)
                yield metadata, hypergraph
            except Exception as e:
                error_msg = f"{type(e).__name__}: {e}"
                metadata["error"] = error_msg
                logger.error(f"Error converting doc to hypergraph: {error_msg}")
                yield metadata, None

    except Exception as e:
        logger.warning(f"Batch processing failed: {type(e).__name__}: {e}. Falling back to per-text processing.")

        for text, metadata, original_text in zip(texts, metadatas, original_texts):
            try:
                doc = nlp(text)
                hypergraph = doc_to_hypergraph(doc, original_text, is_query=is_query)
                yield metadata, hypergraph
            except Exception as e2:
                error_msg = f"{type(e2).__name__}: {e2}"
                metadata["error"] = error_msg
                logger.error(f"Error processing individual text: {error_msg}")
                yield metadata, None


def _prepare_tasks(
    entries: list[dict],
    instances_root: Path,
    query_output_dir: Path,
    force_rebuild: bool = False,
) -> tuple[list[tuple[Path, str, str]], list[tuple[Path, int, str]]]:
    tasks_query: list[tuple[Path, str, str]] = []
    tasks_data: list[tuple[Path, int, str]] = []
    sample_metadata: dict[str, dict[str, Any]] = {}
    seen_query_keys: set[str] = set()
    allowed_categories = _category_filter_env()

    for entry in entries:
        category = _coerce_category(entry.get("category"))
        if category == 5:
            continue
        if allowed_categories is not None and category not in allowed_categories:
            continue
        sample_id = str(entry.get("sample_id", ""))
        qa_id = str(entry.get("qa_id", ""))
        q = str(entry.get("q", "")).strip()
        query_key = query_key_from_question(q)
        hyper_items = _normalize_hyper_items(entry)
        d_start = str(entry.get("d_start", "")).strip()
        source_method = str(entry.get("source_method", entry.get("method", ""))).strip() or "context"
        source_entry = _compact_source_entry(entry, hyper_items)
        low_d_bypass = len(hyper_items) <= LOW_D_BYPASS_THRESHOLD

        if not q or not hyper_items:
            continue

        instance_id = _entry_instance_id(sample_id, qa_id, q)
        instance_dir = instances_root / instance_id

        if instance_id not in sample_metadata:
            sample_metadata[instance_id] = {
                "sample_id": sample_id,
                "qa_id": qa_id,
                "question": q,
                "d_start": d_start,
                "source_method": source_method,
                "low_d_bypass": low_d_bypass,
                "hyper_items": hyper_items,
                "qa_list": [],
            }
            if not low_d_bypass:
                for d_idx, item in enumerate(hyper_items):
                    full_chunk = str(item.get("text", "")).strip()
                    if not full_chunk:
                        continue
                    if not force_rebuild and instance_dir.exists() and (instance_dir / f"data_hypergraph{d_idx}.pkl").exists():
                        continue
                    tasks_data.append((instance_dir, d_idx, full_chunk))

        sample_metadata[instance_id]["qa_list"].append(
            {
                "qa_id": qa_id,
                "category": category,
                "question": q,
                "answer": _resolve_entry_answer(entry),
                "source_entry": source_entry,
            }
        )

        if low_d_bypass:
            continue
        if query_key in seen_query_keys:
            continue
        if not force_rebuild and (query_output_dir / f"{query_key}.pkl").exists():
            continue
        seen_query_keys.add(query_key)
        tasks_query.append((instance_dir, query_key, q))

    for instance_id, meta in sample_metadata.items():
        instance_dir = instances_root / instance_id
        instance_dir.mkdir(parents=True, exist_ok=True)
        meta_path = instance_dir / "metadata.json"
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    return tasks_query, tasks_data


def build_all_hypergraphs_gpu_batch(
    nlp: Language,
    entries: list[dict],
    instances_root: Path,
    batch_size: int = 16,
    force_rebuild: bool = False,
) -> dict[str, Any]:
    instances_root.mkdir(parents=True, exist_ok=True)
    query_output_dir = shared_query_output_dir(instances_root)
    query_output_dir.mkdir(parents=True, exist_ok=True)
    tasks_query, tasks_data = _prepare_tasks(
        entries,
        instances_root,
        query_output_dir,
        force_rebuild=force_rebuild,
    )
    hypergraph_build_elapsed_seconds = 0.0

    logger.info(f"Processing {len(tasks_query)} queries and {len(tasks_data)} data chunks in batch...")

    # Process Queries
    if tasks_query:
        q_texts_with_meta = [
            {"text": q, "meta": {"instance_dir": instance_dir, "query_key": query_key}}
            for instance_dir, query_key, q in tasks_query
        ]
        query_build_started_at = time.perf_counter()
        q_hgs = list(
            batch_text_to_hypergraph(
                nlp, q_texts_with_meta, batch_size=batch_size, is_query=True
            )
        )
        hypergraph_build_elapsed_seconds += max(0.0, time.perf_counter() - query_build_started_at)
        for meta, hg in q_hgs:
            if hg is not None:
                query_key = meta["query_key"]
                hg.save(str(query_output_dir / f"{query_key}.pkl"))

    # Process Data
    if tasks_data:
        # Group by chunks to avoid huge memory spikes
        chunk_size = 1000
        for i in range(0, len(tasks_data), chunk_size):
            sub_tasks = tasks_data[i:i+chunk_size]
            d_texts_with_meta = [
                {"text": txt, "meta": {"instance_dir": instance_dir, "d_idx": d_idx}}
                for instance_dir, d_idx, txt in sub_tasks
            ]
            data_build_started_at = time.perf_counter()
            d_hgs = list(
                batch_text_to_hypergraph(
                    nlp, d_texts_with_meta, batch_size=batch_size, is_query=False
                )
            )
            hypergraph_build_elapsed_seconds += max(0.0, time.perf_counter() - data_build_started_at)
            for meta, hg in d_hgs:
                if hg is not None:
                    instance_dir = meta["instance_dir"]
                    d_idx = meta["d_idx"]
                    hg.save(str(instance_dir / f"data_hypergraph{d_idx}.pkl"))
            logger.info(f"Finished {min(i+chunk_size, len(tasks_data))} / {len(tasks_data)} data chunks")
    return {
        "hypergraph_build_elapsed_seconds": round(hypergraph_build_elapsed_seconds, 4),
    }


def build_all_hypergraphs_single(
    entries: list[dict],
    instances_root: Path,
    force_rebuild: bool = False,
) -> None:
    instances_root.mkdir(parents=True, exist_ok=True)
    query_output_dir = shared_query_output_dir(instances_root)
    query_output_dir.mkdir(parents=True, exist_ok=True)
    tasks_query, tasks_data = _prepare_tasks(
        entries,
        instances_root,
        query_output_dir,
        force_rebuild=force_rebuild,
    )

    logger.info(f"Processing {len(tasks_query)} queries and {len(tasks_data)} data chunks in single mode...")

    for instance_dir, query_key, q in tqdm(tasks_query, desc="Building query hypergraphs"):
        try:
            hg = text_to_hypergraph(q, is_query=True)
            hg.save(str(query_output_dir / f"{query_key}.pkl"))
        except Exception as exc:
            logger.error("Failed query hypergraph for %s query_key=%s: %s", instance_dir.name, query_key, exc)

    for instance_dir, d_idx, d_text in tqdm(tasks_data, desc="Building data hypergraphs"):
        try:
            hg = text_to_hypergraph(d_text, is_query=False)
            hg.save(str(instance_dir / f"data_hypergraph{d_idx}.pkl"))
        except Exception as exc:
            logger.error("Failed data hypergraph for %s d_idx=%s: %s", instance_dir.name, d_idx, exc)


def build_hypergraphs_from_dataset(
    dataset_path: str | Path,
    instances_root: str | Path,
    batch_size: int = 128,
    force_rebuild: bool = False,
) -> dict[str, Any]:
    """Build hypergraphs from any generic LoCoMo entries dataset."""
    started_at = time.perf_counter()
    dataset_file, entries = load_entries_for_build(dataset_path)
    instances_root_path = Path(instances_root)
    nlp = setup_gpu_nlp()
    build_stats = build_all_hypergraphs_gpu_batch(
        nlp,
        entries,
        instances_root_path,
        batch_size=batch_size,
        force_rebuild=force_rebuild,
    )
    hypergraph_build_elapsed_seconds = float(build_stats.get("hypergraph_build_elapsed_seconds", 0.0))
    summary_file = instances_root_path / "summary.json"
    summary_file.write_text(
        json.dumps(
            {
                "hypergraph_build_elapsed_seconds": hypergraph_build_elapsed_seconds,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "summary": {
            "method": METHOD_NAME,
            "stage": "build",
            "dataset_path": str(dataset_file),
            "instances_root": str(instances_root_path),
            "query_root": str(shared_query_output_dir(instances_root_path)),
            "use_gpu_batch": True,
            "batch_size": batch_size,
            "force_rebuild": force_rebuild,
            "total_entries": len(entries),
            "hypergraph_build_elapsed_seconds": hypergraph_build_elapsed_seconds,
            "summary_file": str(summary_file),
            "elapsed_seconds": round(time.perf_counter() - started_at, 4),
        }
    }


def main():
    parser = ArgumentParser(description="Build Locomo Hypergraphs")
    parser.add_argument("--dataset", type=str, default="/home/vincent/hyper-simulation-try/data/bench/locomo-main/locomo-data/locomo_32K.json")
    parser.add_argument("--instances-root", type=str, default="/home/vincent/hyper-simulation-try/data/hypergraphs/locomo/context")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--force-rebuild", action="store_true")
    args = parser.parse_args()

    build_hypergraphs_from_dataset(
        dataset_path=args.dataset,
        instances_root=args.instances_root,
        batch_size=args.batch_size,
        force_rebuild=args.force_rebuild,
    )
    logger.info("Done building locomo hypergraphs.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
