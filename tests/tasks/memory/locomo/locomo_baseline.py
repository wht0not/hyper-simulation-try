from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import fmean
from typing import Any
from collections import Counter

from langchain_ollama import ChatOllama
from tqdm import tqdm

from hyper_simulation.utils.chat_completion import get_invoke
import string
import regex

from nltk.stem import PorterStemmer
_ps = PorterStemmer()


DEFAULT_DATASET_PATHS = [
    "/home/vincent/hyper-simulation-try/data/bench/locomo-main/locomo-data/locomo_1K.json",
    # "/home/vincent/hyper-simulation-try/data/bench/locomo-main/locomo-data/locomo_4K.json",
    # "/home/vincent/hyper-simulation-try/data/bench/locomo-main/locomo-data/locomo_8K.json",
    # "/home/vincent/hyper-simulation-try/data/bench/locomo-main/locomo-data/locomo_16K.json",
    # "/home/vincent/hyper-simulation-try/data/bench/locomo-main/locomo-data/locomo_32K.json",
]
DEFAULT_OUTPUT_DIR = "/home/vincent/hyper-simulation-try/tests/tasks/memory/locomo/results/test"
from prompt import QA_PROMPT

def _safe_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(path)


def _entry_key(entry: dict[str, Any]) -> str:
    sample_id = str(entry.get("sample_id", "")).strip()
    qa_id = str(entry.get("qa_id", "")).strip()
    q = str(entry.get("q", "")).strip()
    return f"{sample_id}::{qa_id}" if sample_id and qa_id else q


def _load_entries(dataset_path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    if not dataset_path.exists():
        return []
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    entries = payload.get("entries", []) if isinstance(payload, dict) else []
    if not isinstance(entries, list):
        return []
    if limit is not None and limit > 0:
        return [one for one in entries if isinstance(one, dict)][:limit]
    return [one for one in entries if isinstance(one, dict)]


def _load_existing_result_map(out_file: Path) -> dict[str, dict[str, Any]]:
    if not out_file.exists():
        return {}
    try:
        payload = json.loads(out_file.read_text(encoding="utf-8"))
    except Exception:
        return {}
    rows = payload.get("results", []) if isinstance(payload, dict) else []
    if not isinstance(rows, list):
        return {}
    existing: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        key = _entry_key(row)
        if key:
            existing[key] = row
    return existing


def _normalize_answer(text: Any) -> str:
    if text is None:
        return ""
    cleaned = str(text).replace("</s>", "").replace("</think>", "").strip()
    patterns = [
        r"###\s*Final\s*Answer:\s*(.+?)(?:\n|$)",
        r"ANSWER:\s*(.+?)(?:\n|$)",
        r"Answer:\s*(.+?)(?:\n|$)",
    ]
    for pattern in patterns:
        m = re.search(pattern, cleaned, re.IGNORECASE)
        if m:
            cleaned = m.group(1).strip()
            break
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    if lines:
        cleaned = lines[-1]
    return cleaned.strip(" .,;:!?\"'()[]")


def _normalize_answer_official(s: str) -> str:
    s = (s or "").replace(",", "")

    def remove_articles(text: str) -> str:
        return regex.sub(r"\b(a|an|the|and)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def _stem_token(token: str) -> str:
    if _ps is None:
        return token
    return _ps.stem(token)


def _f1_score_official(prediction: str, ground_truth: str) -> float:
    prediction_tokens = [_stem_token(w) for w in _normalize_answer_official(prediction).split()]
    ground_truth_tokens = [_stem_token(w) for w in _normalize_answer_official(ground_truth).split()]
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    return (2 * precision * recall) / (precision + recall)


def _f1_multi_official(prediction: str, ground_truth: str) -> float:
    predictions = [p.strip() for p in str(prediction).split(",")]
    ground_truths = [g.strip() for g in str(ground_truth).split(",")]
    if not predictions or not ground_truths:
        return 0.0
    return fmean([max([_f1_score_official(p, gt) for p in predictions]) for gt in ground_truths])


def _evaluate_answer(prediction: str, golden: Any, category: Any) -> dict[str, float]:
    answer = str(golden).strip() if golden is not None else ""
    pred = str(prediction or "")

    try:
        category_int = int(category)
    except Exception:
        category_int = -1

    if category_int == 3:
        answer = answer.split(";")[0].strip()

    if category_int in [2, 3, 4]:
        locomo_score = _f1_score_official(pred, answer)
    elif category_int in [1]:
        locomo_score = _f1_multi_official(pred, answer)
    elif category_int in [5]:
        pred_low = pred.lower()
        locomo_score = 1.0 if ("no information available" in pred_low or "not mentioned" in pred_low) else 0.0
    else:
        locomo_score = _f1_score_official(pred, answer)

    return {
        "locomo_score": float(locomo_score),
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

def _window_tag(dataset_path: Path) -> str:
    name = dataset_path.stem.lower()
    if "8k" in name:
        return "8K"
    if "16k" in name:
        return "16K"
    if "32k" in name:
        return "32K"
    return dataset_path.stem


def run_locomo_vanilla_baseline(
    dataset_paths: list[str] | None = None,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    model_name: str = "qwen3.5:9b",
    temperature: float = 0.1,
    limit: int | None = None,
) -> dict[str, Any]:
    dataset_paths = dataset_paths or list(DEFAULT_DATASET_PATHS)
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    model = ChatOllama(model=model_name, temperature=temperature, top_p=1, reasoning=False, num_predict=512)

    global_summary: dict[str, Any] = {
        "task": "locomo_vanilla",
        "dataset_paths": dataset_paths,
        "model_name": model_name,
        "temperature": temperature,
        "results": {},
    }

    for one_path in dataset_paths:
        dataset_path = Path(one_path)
        tag = _window_tag(dataset_path)
        out_file = out_root / f"locomo_vanilla_{tag}.json"
        existing_map = _load_existing_result_map(out_file)

        entries = _load_entries(dataset_path, limit=limit)
        if not entries:
            global_summary["results"][tag] = {
                "status": "skipped",
                "reason": f"empty_or_missing: {dataset_path}",
            }
            _safe_write_json(out_root / "locomo_vanilla_summary.json", global_summary)
            continue

        results: list[dict[str, Any]] = []
        
        # Track official category-wise metrics
        category_counts: dict[int, int] = {k: 0 for k in [1, 2, 3, 4, 5]}
        category_acc: dict[int, float] = {k: 0.0 for k in [1, 2, 3, 4, 5]}

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
                "window": tag,
                "method": "vanilla",
                "dataset_path": str(dataset_path),
                "total": total_q,
                "overall_accuracy": round(total_acc / total_q, 3) if total_q > 0 else 0.0,
                "by_category": cat_summary
            }

        _safe_write_json(out_file, {"summary": _summary(), "results": results})

        pbar = tqdm(entries, desc=f"locomo/vanilla/{tag}", unit="q")
        for entry in pbar:
            q = str(entry.get("q", "")).strip()
            d = entry.get("d", [])
            answer = entry.get("answer")
            if not q or not d:
                continue

            key = _entry_key(entry)
            existing = existing_map.get(key)
            if isinstance(existing, dict) and "prediction" in existing and "metrics" in existing:
                metrics = existing.get("metrics", {}) or {}
                cat = entry.get("category", -1)
                
                # Add existing result to category counts
                if cat in category_counts:
                    category_counts[cat] += 1
                    try:
                        category_acc[cat] += float(metrics.get("locomo_score", 0.0))
                    except Exception:
                        pass
                
                results.append(existing)
                continue

            context = _compose_context(entry)
            prompt = f"{context}\n\n{QA_PROMPT.format(q)}"
            try:
                raw = get_invoke(model, prompt)
                prediction = _normalize_answer(raw)
            except Exception as exc:
                tqdm.write(
                    f"[ERROR][locomo/vanilla/{tag}] sample_id={entry.get('sample_id')} qa_id={entry.get('qa_id')} err={type(exc).__name__}: {exc}"
                )
                continue

            metrics = _evaluate_answer(prediction, answer, entry.get("category"))
            cat = entry.get("category", -1)
            if cat in category_counts:
                category_counts[cat] += 1
                category_acc[cat] += float(metrics["locomo_score"])

            out_row = {
                "sample_id": entry.get("sample_id"),
                "qa_id": entry.get("qa_id"),
                "q": q,
                "answer": answer,
                "prediction": prediction,
                "category": entry.get("category"),
                "metrics": metrics,
            }
            results.append(out_row)
            _safe_write_json(out_file, {"summary": _summary(), "results": results})
            global_summary["results"][tag] = {
                "status": "running",
                "output_file": str(out_file.resolve()),
                **_summary(),
            }
            _safe_write_json(out_root / "locomo_vanilla_summary.json", global_summary)

        payload = {"summary": _summary(), "results": results}
        _safe_write_json(out_file, payload)
        global_summary["results"][tag] = {
            "status": "ok",
            "output_file": str(out_file.resolve()),
            **_summary(),
        }
        _safe_write_json(out_root / "locomo_vanilla_summary.json", global_summary)

    overall_file = out_root / "locomo_vanilla_summary.json"
    _safe_write_json(overall_file, global_summary)
    global_summary["summary_file"] = str(overall_file.resolve())
    return global_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LoCoMo vanilla baseline on 8K/16K/32K datasets")
    parser.add_argument("--dataset-paths", type=str, default=",".join(DEFAULT_DATASET_PATHS))
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-name", type=str, default="qwen3.5:9b")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    dataset_paths = [one.strip() for one in args.dataset_paths.split(",") if one.strip()]
    report = run_locomo_vanilla_baseline(
        dataset_paths=dataset_paths,
        output_dir=args.output_dir,
        model_name=args.model_name,
        temperature=args.temperature,
        limit=(args.limit or None),
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
