from __future__ import annotations

import hashlib
from typing import Any

from .utils import coerce_category


def build_question_text(question: str, category: Any) -> str:
    category_int = coerce_category(category)
    base_question = str(question).strip()
    if category_int == 2:
        return base_question + " Use DATE of CONVERSATION to answer with an approximate date."
    return base_question


def build_cat5_answer_key(
    question: str,
    answer: str,
    sample_id: Any = "",
    qa_id: Any = "",
) -> dict[str, str]:
    gold_answer = str(answer).strip()
    stable_key = f"{sample_id}::{qa_id}::{question}"
    pick = int(hashlib.md5(stable_key.encode("utf-8")).hexdigest(), 16) % 2
    if pick == 0:
        return {"a": "No information available", "b": gold_answer}
    return {"a": gold_answer, "b": "No information available"}


def build_cat5_choice_question(
    question: str,
    answer: str,
    sample_id: Any = "",
    qa_id: Any = "",
) -> tuple[str, dict[str, str]]:
    prompt_question = build_question_text(question, 5) + " (a) {} (b) {}. Select the correct answer by writing (a) or (b)."
    answer_key = build_cat5_answer_key(question, answer, sample_id=sample_id, qa_id=qa_id)
    prompt_question = prompt_question.format(answer_key["a"], answer_key["b"])
    return prompt_question, answer_key


def decode_cat5_choice(model_prediction: Any, answer_key: dict[str, str]) -> str:
    raw_prediction = str(model_prediction or "").strip()
    normalized = raw_prediction.lower()
    compact = normalized.replace(" ", "")
    if compact in {"a", "(a)", "optiona", "choicea"}:
        return answer_key["a"]
    if compact in {"b", "(b)", "optionb", "choiceb"}:
        return answer_key["b"]
    if normalized.startswith("(a)") or normalized.startswith("a)"):
        return answer_key["a"]
    if normalized.startswith("(b)") or normalized.startswith("b)"):
        return answer_key["b"]
    if normalized == str(answer_key.get("a", "")).strip().lower():
        return answer_key["a"]
    if normalized == str(answer_key.get("b", "")).strip().lower():
        return answer_key["b"]
    if "no information available" in normalized or "not mentioned" in normalized:
        return "No information available"
    return raw_prediction
