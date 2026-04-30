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

import argparse
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = PROJECT_ROOT / "src"
LOCOMO_ROOT = Path(__file__).resolve().parent
for candidate in (SRC_ROOT, PROJECT_ROOT, LOCOMO_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from utils.utils import (
    DEFAULT_INSTANCES_ROOT,
    DEFAULT_MODEL_NAME,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_TEMPERATURE,
    answers_output_path,
    final_output_path,
    load_entries,
    load_payload_rows,
    prepared_output_path,
    retrieved_output_path,
)


def _count_payload_rows(path_like: str | Path) -> int:
    return len(load_payload_rows(Path(path_like)))


def _count_answer_predictions(path_like: str | Path) -> int:
    count = 0
    for row in load_payload_rows(Path(path_like)):
        if isinstance(row, dict) and "prediction" in row:
            count += 1
    return count


def _count_hypersim_questions(instances_root: str | Path, limit: int | None = None) -> int:
    instances_dir = Path(instances_root)
    if not instances_dir.exists():
        return 0
    dirs = [d for d in instances_dir.iterdir() if d.is_dir()]
    if limit is not None and limit > 0:
        dirs = dirs[:limit]
    total = 0
    for instance_dir in dirs:
        meta_path = instance_dir / "metadata.json"
        if not meta_path.exists():
            continue
        try:
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        qa_list = payload.get("qa_list", [])
        if isinstance(qa_list, list):
            total += len([row for row in qa_list if isinstance(row, dict)])
    return total


def _load_summary(path_like: str | Path) -> dict[str, Any]:
    path = Path(path_like)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    summary = payload.get("summary", {}) if isinstance(payload, dict) else {}
    return summary if isinstance(summary, dict) else {}


def _expected_count(method: str, dataset_path: str, instances_root: str, limit: int | None = None) -> int:
    if method == "context":
        return len(load_entries(Path(dataset_path), limit=limit))
    if method == "langmem":
        from method.langmem.compose import count_langmem_questions
        return count_langmem_questions(dataset_path, limit=limit)
    if method == "amem":
        from method.amem.compose import count_amem_questions
        return count_amem_questions(dataset_path, limit=limit)
    if method == "memorybank":
        from method.memorybank.compose import count_memorybank_questions
        return count_memorybank_questions(dataset_path, limit=limit)
    if method == "hyper_simulation":
        return _count_hypersim_questions(instances_root, limit=limit)
    return 0


def run_pipeline(
    method: str,
    stage: str,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    dataset_path: str = "",
    instances_root: str = DEFAULT_INSTANCES_ROOT,
    prepared_path: str = "",
    answers_path: str = "",
    model_name: str = DEFAULT_MODEL_NAME,
    temperature: float = DEFAULT_TEMPERATURE,
    limit: int | None = None,
    batch_size: int = 128,
    force_rebuild: bool = False,
    answer_batch_size: int = 1,
    judge_max_workers: int = 4,
    llm_judge_repeat: int = 5,
) -> dict[str, Any]:
    source_path = dataset_path if method in {"context", "langmem", "amem", "memorybank"} else instances_root
    effective_output_dir = output_dir
    if not source_path and stage in {"build", "retrieve", "compose", "all"}:
        raise ValueError("source path is required for build/retrieve/compose/all stage")

    build_payload: dict[str, Any] | None = None
    retrieval_payload: dict[str, Any] | None = None
    compose_payload: dict[str, Any] | None = None
    answer_payload: dict[str, Any] | None = None
    evaluate_payload: dict[str, Any] | None = None

    if stage == "build":
        if method != "hyper_simulation":
            raise ValueError("build stage is only supported for hyper_simulation")
        if not dataset_path:
            raise ValueError("--dataset-path is required for hyper_simulation build stage")
        from method.hyper_simulation.build import build_hypergraphs_from_dataset

        build_payload = build_hypergraphs_from_dataset(
            dataset_path=dataset_path,
            output_dir=instances_root,
            batch_size=batch_size,
            force_rebuild=force_rebuild,
        )
        return {
            "build": build_payload,
            "retrieve": retrieval_payload,
            "compose": compose_payload,
            "answer": answer_payload,
            "evaluate": evaluate_payload,
        }

    if stage == "retrieve" and method not in {"langmem", "amem", "memorybank"}:
        raise ValueError("retrieve stage is only supported for langmem, amem and memorybank")

    expected_total = _expected_count(
        method=method,
        dataset_path=dataset_path,
        instances_root=instances_root,
        limit=limit,
    )
    if not source_path and prepared_path:
        source_path = str(_load_summary(prepared_path).get("source_path", ""))
    if not source_path and answers_path:
        source_path = str(_load_summary(answers_path).get("source_path", ""))

    default_source = source_path or prepared_path or answers_path or method
    default_retrieved_path = ""
    if method in {"langmem", "amem", "memorybank"} and source_path:
        default_retrieved_path = str(retrieved_output_path(effective_output_dir, method, source_path))
    default_prepared_path = str(prepared_output_path(effective_output_dir, method, default_source))
    prepared_path = prepared_path or default_prepared_path

    if not source_path and prepared_path:
        source_path = str(_load_summary(prepared_path).get("source_path", ""))

    default_answers_source = source_path or prepared_path or default_source
    default_answers_path = str(answers_output_path(effective_output_dir, method, default_answers_source))
    answers_path = answers_path or default_answers_path

    if not source_path and answers_path:
        source_path = str(_load_summary(answers_path).get("source_path", ""))

    default_final_source = source_path or answers_path or default_answers_source
    default_final_path = str(final_output_path(effective_output_dir, method, default_final_source))

    has_retrieved = _count_payload_rows(default_retrieved_path) > 0 if default_retrieved_path else False
    has_complete_retrieved = has_retrieved and (
        expected_total <= 0 or _count_payload_rows(default_retrieved_path) >= expected_total
    )
    has_prepared = _count_payload_rows(prepared_path) > 0
    has_complete_prepared = has_prepared and (expected_total <= 0 or _count_payload_rows(prepared_path) >= expected_total)
    has_answers = _count_answer_predictions(answers_path) > 0
    has_complete_answers = has_answers and (
        _count_answer_predictions(answers_path) >= max(_count_payload_rows(prepared_path), expected_total)
    )
    has_evaluated = _count_payload_rows(default_final_path) > 0

    compose_input_path = dataset_path
    if method in {"langmem", "amem", "memorybank"} and stage in {"retrieve", "all"}:
        if stage == "all" and has_complete_retrieved:
            compose_input_path = default_retrieved_path
        else:
            if method == "langmem":
                from method.langmem.retrieval import retrieve_langmem_dataset

                retrieval_payload = retrieve_langmem_dataset(
                    dataset_path=dataset_path,
                    output_dir=output_dir,
                    model_name=model_name,
                    limit=limit,
                )
            elif method == "amem":
                from method.amem.retrieval import retrieve_amem_dataset

                retrieval_payload = retrieve_amem_dataset(
                    dataset_path=dataset_path,
                    output_dir=output_dir,
                    model_name=model_name,
                    limit=limit,
                )
            else:
                from method.memorybank.retrieval import retrieve_memorybank_dataset

                retrieval_payload = retrieve_memorybank_dataset(
                    dataset_path=dataset_path,
                    output_dir=output_dir,
                    model_name=model_name,
                    limit=limit,
                )
            compose_input_path = str(retrieval_payload["summary"]["retrieved_file"])

        if stage == "retrieve":
            return {
                "build": build_payload,
                "retrieve": retrieval_payload,
                "compose": compose_payload,
                "answer": answer_payload,
                "evaluate": evaluate_payload,
            }

    if stage in {"compose", "all"}:
        if stage == "all" and has_complete_prepared:
            compose_payload = None
        else:
            if method == "context":
                from utils.context import prepare_context_dataset

                compose_payload = prepare_context_dataset(
                    dataset_path=dataset_path,
                    output_dir=output_dir,
                    limit=limit,
                )
            elif method == "langmem":
                from method.langmem.compose import prepare_langmem_dataset

                compose_payload = prepare_langmem_dataset(
                    dataset_path=compose_input_path,
                    output_dir=output_dir,
                    model_name=model_name,
                    limit=limit,
                )
            elif method == "amem":
                from method.amem.compose import prepare_amem_dataset

                compose_payload = prepare_amem_dataset(
                    dataset_path=compose_input_path,
                    output_dir=output_dir,
                    model_name=model_name,
                    limit=limit,
                )
            elif method == "memorybank":
                from method.memorybank.compose import prepare_memorybank_dataset

                compose_payload = prepare_memorybank_dataset(
                    dataset_path=compose_input_path,
                    output_dir=output_dir,
                    model_name=model_name,
                    limit=limit,
                )
            else:
                from method.hyper_simulation.compose import prepare_hypersim_instances

                compose_payload = prepare_hypersim_instances(
                    instances_root=instances_root,
                    output_dir=output_dir,
                    limit=limit,
                )
            prepared_path = str(compose_payload["summary"]["prepared_file"])

    if stage in {"answer", "all"}:
        if stage == "all" and has_complete_answers:
            answer_payload = None
        else:
            from utils.answer import run_answers

            answer_payload = run_answers(
                prepared_path=prepared_path,
                output_path=answers_path,
                model_name=model_name,
                temperature=temperature,
                limit=limit,
                batch_size=answer_batch_size,
            )

    if stage in {"evaluate", "all"}:
        if stage == "all" and has_evaluated and has_complete_answers:
            evaluate_payload = None
        else:
            from utils.evaluate import evaluate_results_file

            evaluate_payload = evaluate_results_file(
                answers_path=answers_path,
                output_path=default_final_path,
                method=method,
                model_name=model_name,
                source_path=source_path,
                judge_max_workers=judge_max_workers,
                llm_judge_repeat=llm_judge_repeat,
            )

    return {
        "build": build_payload,
        "retrieve": retrieval_payload,
        "compose": compose_payload,
        "answer": answer_payload,
        "evaluate": evaluate_payload,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LoCoMo experiments")
    parser.add_argument("--method", choices=["context", "hyper_simulation", "langmem", "amem", "memorybank"], required=True)
    parser.add_argument("--stage", choices=["build", "retrieve", "compose", "answer", "evaluate", "all"], default="all")
    parser.add_argument("--dataset-path", type=str, default="")
    parser.add_argument("--instances-root", type=str, default=DEFAULT_INSTANCES_ROOT)
    parser.add_argument("--prepared-path", type=str, default="")
    parser.add_argument("--answers-path", type=str, default="")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--answer-batch-size", type=int, default=1)
    parser.add_argument("--judge-max-workers", type=int, default=4)
    parser.add_argument("--llm-judge-repeat", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    payload = run_pipeline(
        method=args.method,
        stage=args.stage,
        output_dir=args.output_dir,
        dataset_path=args.dataset_path,
        instances_root=args.instances_root,
        prepared_path=args.prepared_path,
        answers_path=args.answers_path,
        model_name=args.model_name,
        temperature=args.temperature,
        limit=(args.limit or None),
        batch_size=args.batch_size,
        force_rebuild=args.force_rebuild,
        answer_batch_size=args.answer_batch_size,
        judge_max_workers=args.judge_max_workers,
        llm_judge_repeat=args.llm_judge_repeat,
    )
    # print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
