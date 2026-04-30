from __future__ import annotations

from pathlib import Path
from typing import Any

from tqdm import tqdm

from .prompt import QA_PROMPT, QA_PROMPT_CAT_5
from .qa_utils import build_cat5_choice_question, build_question_text
from .utils import (
    coerce_category,
    entry_key,
    load_entries,
    load_existing_result_map,
    load_existing_results,
    prepared_output_path,
    safe_write_json,
    window_tag,
)


def _prepared_payload(
    rows: list[dict[str, Any]],
    dataset_file: Path,
    out_file: Path,
) -> dict[str, Any]:
    return {
        "summary": {
            "method": "context",
            "window": window_tag(dataset_file),
            "source_path": str(dataset_file),
            "prepared_file": str(out_file),
            "total": len(rows),
        },
        "results": rows,
    }


def _compose_context(entry: dict[str, Any]) -> str:
    d_val = entry.get("d")
    d_start = str(entry.get("d_start", "")).strip()
    if isinstance(d_val, list):
        sessions = [str(one).strip() for one in d_val if str(one).strip()]
        if d_start and sessions:
            return d_start + "\n\n" + "\n\n".join(sessions)
        if d_start:
            return d_start
        return "\n\n".join(sessions)
    if isinstance(d_val, str):
        return d_val.strip()
    return ""


def compose_context_entry(entry: dict[str, Any], method_name: str = "context") -> dict[str, Any] | None:
    q = str(entry.get("q", "")).strip()
    d = entry.get("d", [])
    answer = entry.get("answer")
    if not q or not d:
        return None

    category = coerce_category(entry.get("category"))
    context = _compose_context(entry)
    prepared: dict[str, Any] = {
        "sample_id": entry.get("sample_id"),
        "qa_id": entry.get("qa_id"),
        "q": q,
        "answer": answer,
        "category": category,
        "method": method_name,
    }

    if category == 5:
        cat5_question, cat5_answer_key = build_cat5_choice_question(
            q,
            str(answer or ""),
            sample_id=entry.get("sample_id", ""),
            qa_id=entry.get("qa_id", ""),
        )
        prepared["cat5_answer_key"] = cat5_answer_key
        prepared["prompt"] = f"{context}\n\n{QA_PROMPT_CAT_5.format(cat5_question)}"
    else:
        prepared["prompt"] = f"{context}\n\n{QA_PROMPT.format(build_question_text(q, category))}"

    return prepared


def prepare_context_dataset(
    dataset_path: str,
    output_dir: str,
    limit: int | None = None,
) -> dict[str, Any]:
    dataset_file = Path(dataset_path)
    out_file = prepared_output_path(output_dir, "context", dataset_file)
    entries = load_entries(dataset_file, limit=limit)
    existing_map = load_existing_result_map(out_file)
    prepared_rows = load_existing_results(out_file)

    for entry in tqdm(entries, desc="locomo/compose/context", unit="q"):
        if entry_key(entry) in existing_map:
            continue
        prepared = compose_context_entry(entry)
        if prepared is not None:
            prepared_rows.append(prepared)
            existing_map[entry_key(prepared)] = prepared
            safe_write_json(out_file, _prepared_payload(prepared_rows, dataset_file, out_file))

    payload = _prepared_payload(prepared_rows, dataset_file, out_file)
    safe_write_json(out_file, payload)
    return payload
