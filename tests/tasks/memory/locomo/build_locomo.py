"""
Build Locomo hypergraphs with batch support.

Supported input schema:
  data/bench/locomo-main/locomo-data/locomo_8K.json
  (also 16K/32K)

Output layout:
  tests/tasks/memory/locomo/instances/<instance_id>/
    - query_hypergraph.pkl
    - data_hypergraph0.pkl (session 1)
    - data_hypergraph1.pkl (session 2)
    - metadata.json
"""

import json
import logging
from argparse import ArgumentParser
from pathlib import Path
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
    output_dir: Path,
    force_rebuild: bool = False,
) -> tuple[list[tuple[Path, str, str]], list[tuple[Path, int, str]]]:
    tasks_query: list[tuple[Path, str, str]] = []
    tasks_data: list[tuple[Path, int, str]] = []
    sample_metadata: dict[str, dict[str, Any]] = {}

    for entry in entries:
        sample_id = str(entry.get("sample_id", ""))
        qa_id = str(entry.get("qa_id", ""))
        q = str(entry.get("q", "")).strip()
        d_list = entry.get("d", [])
        d_start = str(entry.get("d_start", "")).strip()

        if not isinstance(d_list, list):
            d_list = [str(d_list)]
        if not q or not d_list:
            continue

        instance_id = generate_instance_id(sample_id)
        instance_dir = output_dir / instance_id

        if instance_id not in sample_metadata:
            sample_metadata[instance_id] = {
                "sample_id": sample_id,
                "d_start": d_start,
                "qa_list": [],
            }
            for d_idx, d_text in enumerate(d_list):
                full_chunk = f"{d_text}"
                if not force_rebuild and instance_dir.exists() and (instance_dir / f"data_hypergraph{d_idx}.pkl").exists():
                    continue
                tasks_data.append((instance_dir, d_idx, full_chunk))

        sample_metadata[instance_id]["qa_list"].append(
            {
                "qa_id": qa_id,
                "category": entry.get("category"),
                "question": q,
                "answer": entry.get("answer"),
            }
        )

        if not force_rebuild and instance_dir.exists() and (instance_dir / f"query_hypergraph_{qa_id}.pkl").exists():
            continue
        tasks_query.append((instance_dir, qa_id, q))

    for instance_id, meta in sample_metadata.items():
        instance_dir = output_dir / instance_id
        instance_dir.mkdir(parents=True, exist_ok=True)
        meta_path = instance_dir / "metadata.json"
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    return tasks_query, tasks_data


def build_all_hypergraphs_gpu_batch(
    nlp: Language,
    entries: list[dict],
    output_dir: Path,
    batch_size: int = 16,
    force_rebuild: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks_query, tasks_data = _prepare_tasks(entries, output_dir, force_rebuild=force_rebuild)

    logger.info(f"Processing {len(tasks_query)} queries and {len(tasks_data)} data chunks in batch...")

    # Process Queries
    if tasks_query:
        q_texts_with_meta = [
            {"text": q, "meta": {"instance_dir": instance_dir, "qa_id": qa_id}}
            for instance_dir, qa_id, q in tasks_query
        ]
        q_hgs_iter = batch_text_to_hypergraph(
            nlp, q_texts_with_meta, batch_size=batch_size, is_query=True
        )
        for meta, hg in q_hgs_iter:
            if hg is not None:
                instance_dir = meta["instance_dir"]
                qa_id = meta["qa_id"]
                hg.save(str(instance_dir / f"query_hypergraph_{qa_id}.pkl"))

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
            d_hgs_iter = batch_text_to_hypergraph(
                nlp, d_texts_with_meta, batch_size=batch_size, is_query=False
            )
            for meta, hg in d_hgs_iter:
                if hg is not None:
                    instance_dir = meta["instance_dir"]
                    d_idx = meta["d_idx"]
                    hg.save(str(instance_dir / f"data_hypergraph{d_idx}.pkl"))
            logger.info(f"Finished {min(i+chunk_size, len(tasks_data))} / {len(tasks_data)} data chunks")


def build_all_hypergraphs_single(
    entries: list[dict],
    output_dir: Path,
    force_rebuild: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks_query, tasks_data = _prepare_tasks(entries, output_dir, force_rebuild=force_rebuild)

    logger.info(f"Processing {len(tasks_query)} queries and {len(tasks_data)} data chunks in single mode...")

    for instance_dir, qa_id, q in tqdm(tasks_query, desc="Building query hypergraphs"):
        try:
            hg = text_to_hypergraph(q, is_query=True)
            hg.save(str(instance_dir / f"query_hypergraph_{qa_id}.pkl"))
        except Exception as exc:
            logger.error("Failed query hypergraph for %s qa_id=%s: %s", instance_dir.name, qa_id, exc)

    for instance_dir, d_idx, d_text in tqdm(tasks_data, desc="Building data hypergraphs"):
        try:
            hg = text_to_hypergraph(d_text, is_query=False)
            hg.save(str(instance_dir / f"data_hypergraph{d_idx}.pkl"))
        except Exception as exc:
            logger.error("Failed data hypergraph for %s d_idx=%s: %s", instance_dir.name, d_idx, exc)


def main():
    parser = ArgumentParser(description="Build Locomo Hypergraphs")
    parser.add_argument("--dataset", type=str, default="/home/vincent/hyper-simulation-try/data/bench/locomo-main/locomo-data/locomo_32K.json")
    parser.add_argument("--output-dir", type=str, default="/home/vincent/hyper-simulation-try/data/hypergraphs/locomo")
    parser.add_argument("--use-gpu-batch", action="store_true")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--force-rebuild", action="store_true")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        logger.error(f"Dataset not found: {dataset_path}")
        return

    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    entries = payload.get("entries", [])
    
    out_dir = Path(args.output_dir)
    if args.use_gpu_batch:
        nlp = setup_gpu_nlp()
        build_all_hypergraphs_gpu_batch(nlp, entries, out_dir, batch_size=args.batch_size, force_rebuild=args.force_rebuild)
    else:
        build_all_hypergraphs_single(entries, out_dir, force_rebuild=args.force_rebuild)
    logger.info("Done building locomo hypergraphs.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
