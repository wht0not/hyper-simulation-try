from __future__ import annotations

from typing import Any

from .utils import coerce_category


def build_question_text(question: str, category: Any) -> str:
    category_int = coerce_category(category)
    base_question = str(question).strip()
    if category_int == 2:
        return base_question + " Use DATE of CONVERSATION to convert relative time to an approximate date."
    return base_question


def resolve_qa_answer(payload: dict[str, Any]) -> Any:
    answer = payload.get("answer")
    adversarial_answer = payload.get("adversarial_answer")
    if answer is not None:
        return answer
    if adversarial_answer is not None and str(adversarial_answer).strip():
        return adversarial_answer
    return answer
