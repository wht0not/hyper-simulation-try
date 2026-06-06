from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

from langchain_ollama import ChatOllama
from tqdm import tqdm

from method.hyper_simulation.compose import sanitize_hypersim_row
from hyper_simulation.utils.chat_completion import get_generate

from .metrics import normalize_answer
from .utils import (
    DEFAULT_MODEL_NAME,
    DEFAULT_TEMPERATURE,
    answers_output_path,
    entry_key,
    load_existing_result_map,
    load_existing_results,
    safe_write_json,
)

SLICE_RELATED_FIELDS = {
    "slice",
    "consistent_context",
    "simulation_pair_count",
    "ranked_slice_indices",
    "selected_context_indices",
    "slice_hit_counts",
    "slice_critical_hit_counts",
}


def _resolve_row_temperature(row: dict[str, Any], default_temperature: float) -> float:
    try:
        return float(row.get("answer_temperature", default_temperature))
    except Exception:
        return float(default_temperature)


def _answer_summary(
    rows: list[dict[str, Any]],
    method: str,
    model_name: str,
    prepared_path: str,
    source_path: str,
    window: str,
    elapsed_seconds: float | None = None,
) -> dict[str, Any]:
    summary = {
        "method": method,
        "model_name": model_name,
        "prepared_path": prepared_path,
        "source_path": source_path,
        "window": window,
        "total": len(rows),
    }
    if elapsed_seconds is not None:
        summary["elapsed_seconds"] = round(float(elapsed_seconds), 4)
    return summary


def _answer_payload(
    rows: list[dict[str, Any]],
    method: str,
    model_name: str,
    prepared_file: Path,
    source_path: str,
    window: str,
    elapsed_seconds: float | None = None,
) -> dict[str, Any]:
    return {
        "summary": _answer_summary(
            rows=rows,
            method=method,
            model_name=model_name,
            prepared_path=str(prepared_file),
            source_path=source_path,
            window=window,
            elapsed_seconds=elapsed_seconds,
        ),
        "results": rows,
    }


def run_answers(
    prepared_path: str | Path,
    output_path: str | Path | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
    temperature: float = DEFAULT_TEMPERATURE,
    limit: int | None = None,
    batch_size: int = 1,
    checkpoint_every: int = 500,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    prepared_file = Path(prepared_path)
    payload = json.loads(prepared_file.read_text(encoding="utf-8"))
    prepared_rows = payload.get("results", []) if isinstance(payload, dict) else []
    if not isinstance(prepared_rows, list):
        prepared_rows = []

    summary = payload.get("summary", {}) if isinstance(payload, dict) else {}
    method = str(summary.get("method", ""))
    source_path = str(summary.get("source_path", ""))
    window = str(summary.get("window", ""))
    if output_path is None:
        output_path = answers_output_path(prepared_file.parent, method, source_path or prepared_file)
    out_file = Path(output_path)

    model_cache: dict[float, ChatOllama] = {}

    def get_model(temp: float) -> ChatOllama:
        rounded_temp = round(float(temp), 4)
        cached = model_cache.get(rounded_temp)
        if cached is None:
            cached = ChatOllama(
                model=model_name,
                temperature=rounded_temp,
                reasoning=False,
                num_predict=8192,
            )
            model_cache[rounded_temp] = cached
        return cached
    results: list[dict[str, Any]] = load_existing_results(out_file)
    existing_map = load_existing_result_map(out_file)
    checkpoint_every = max(1, int(checkpoint_every))
    pending_writes = 0
    if method == "hyper_simulation":
        existing_map = {
            key: sanitize_hypersim_row(row) if isinstance(row, dict) else row
            for key, row in existing_map.items()
        }

    iterable_rows = [row for row in prepared_rows if isinstance(row, dict)]
    if limit is not None and limit > 0:
        iterable_rows = iterable_rows[:limit]

    batch_size = max(1, int(batch_size))
    pbar = tqdm(total=len(iterable_rows), desc=f"locomo/answer/{method}", unit="q")
    for start_idx in range(0, len(iterable_rows), batch_size):
        batch_rows = iterable_rows[start_idx : start_idx + batch_size]
        pending_rows: list[dict[str, Any]] = []
        pending_prompts: list[str] = []

        for row in batch_rows:
            key = entry_key(row)
            existing = existing_map.get(key)
            if isinstance(existing, dict) and "prediction" in existing:
                pbar.update(1)
                continue

            prompt = str(row.get("prompt", "")).strip()
            if not prompt:
                pbar.update(1)
                continue
            pending_rows.append(row)
            pending_prompts.append(prompt)

        raw_outputs: list[str] = []
        infer_elapsed_seconds: list[float] = []
        if pending_prompts:
            try:
                row_temperatures = [_resolve_row_temperature(row, temperature) for row in pending_rows]
                unique_temperatures = {round(one, 4) for one in row_temperatures}
                if len(unique_temperatures) == 1:
                    infer_started_at = time.perf_counter()
                    raw_outputs = get_generate(pending_prompts, get_model(row_temperatures[0]))
                    infer_total = max(0.0, time.perf_counter() - infer_started_at)
                    if len(raw_outputs) != len(pending_prompts):
                        raise ValueError(
                            f"batch output size mismatch: expected {len(pending_prompts)}, got {len(raw_outputs)}"
                        )
                    avg_infer = infer_total / len(pending_prompts) if pending_prompts else 0.0
                    infer_elapsed_seconds = [avg_infer] * len(pending_prompts)
                else:
                    raw_outputs = []
                    infer_elapsed_seconds = []
                    for row, prompt in zip(pending_rows, pending_prompts):
                        row_model = get_model(_resolve_row_temperature(row, temperature))
                        infer_started_at = time.perf_counter()
                        response = row_model.invoke(prompt)
                        raw_outputs.append(str(getattr(response, "content", response) or ""))
                        infer_elapsed_seconds.append(max(0.0, time.perf_counter() - infer_started_at))
            except Exception as exc:
                tqdm.write(f"[ERROR][locomo/answer/{method}] batch_start={start_idx} err={type(exc).__name__}: {exc}")
                raw_outputs = [""] * len(pending_prompts)
                infer_elapsed_seconds = [0.0] * len(pending_prompts)

        for row, raw, infer_elapsed in zip(pending_rows, raw_outputs, infer_elapsed_seconds):
            item_started_at = time.perf_counter()
            try:
                prediction = normalize_answer(raw)
            except Exception as exc:
                tqdm.write(
                    f"[ERROR][locomo/answer/{method}] sample_id={row.get('sample_id')} qa_id={row.get('qa_id')} err={type(exc).__name__}: {exc}"
                )
                pbar.update(1)
                continue

            out_row = {
                k: v
                for k, v in row.items()
                if k not in {"prompt", "cat5_answer_key"} and k not in SLICE_RELATED_FIELDS
            }
            out_row["raw_prediction"] = raw
            out_row["prediction"] = prediction
            out_row["answer_elapsed_seconds"] = round(
                max(0.0, infer_elapsed) + max(0.0, time.perf_counter() - item_started_at),
                6,
            )
            results.append(out_row)
            existing_map[entry_key(out_row)] = out_row
            pending_writes += 1
            if pending_writes >= checkpoint_every:
                safe_write_json(
                    out_file,
                    _answer_payload(
                        rows=results,
                        method=method,
                        model_name=model_name,
                        prepared_file=prepared_file,
                        source_path=source_path,
                        window=window,
                        elapsed_seconds=time.perf_counter() - started_at,
                    ),
                )
                pending_writes = 0
            pbar.update(1)
    pbar.close()

    answer_payload = _answer_payload(
        rows=results,
        method=method,
        model_name=model_name,
        prepared_file=prepared_file,
        source_path=source_path,
        window=window,
        elapsed_seconds=time.perf_counter() - started_at,
    )
    safe_write_json(out_file, answer_payload)
    return answer_payload
