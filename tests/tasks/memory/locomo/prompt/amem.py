from __future__ import annotations

from typing import Any

from utils.qa_utils import build_cat5_answer_key


AMEM_GENERATE_QUERY_PROMPT = """Given the following question, generate several keywords separated by commas.

Question: {question}

Keywords:"""

AMEM_RELEVANT_PARTS_PROMPT = """Given the following conversation memories and a question, select the most relevant parts of the conversation that would help answer the question. Include the date/time if available.

Conversation memories:
{memories_text}

Question: {query}

Return only the relevant parts of the conversation that would help answer this specific question.
If no parts are relevant, return the input unchanged."""

AMEM_ANSWER_PROMPT_DEFAULT = """Based on the context: {context_text}, write an answer in the form of a short phrase for the following question. Answer with exact words from the context whenever possible.

Question: {question} Short answer:"""

AMEM_ANSWER_PROMPT_CAT_2 = """Based on the context: {context_text}, answer the following question. Use DATE of CONVERSATION to answer with an approximate date.
Please generate the shortest possible answer, using words from the conversation where possible, and avoid using any subjects.

Question: {question} Short answer:"""

AMEM_ANSWER_PROMPT_CAT_3 = """Based on the context: {context_text}, write an answer in the form of a short phrase for the following question. Answer with exact words from the context whenever possible.

Question: {question} Short answer:"""

AMEM_ANSWER_PROMPT_CAT_5 = """Based on the context: {context_text}, answer the following question. {question}

Select the correct answer: {option_a} or {option_b}  Short answer:"""


def build_amem_generate_query_prompt(question: str) -> str:
    return AMEM_GENERATE_QUERY_PROMPT.format(question=str(question).strip())


def build_amem_relevant_parts_prompt(memories_text: str, query: str) -> str:
    return AMEM_RELEVANT_PARTS_PROMPT.format(
        memories_text=str(memories_text).strip(),
        query=str(query).strip(),
    )


def build_amem_answer_prompt(
    context_text: str,
    question: str,
    category: Any,
    answer: str = "",
    sample_id: Any = "",
    qa_id: Any = "",
) -> dict[str, Any]:
    category_int = int(category)
    payload: dict[str, Any] = {"temperature": 0.7}
    if category_int == 5:
        answer_key = build_cat5_answer_key(question, answer, sample_id=sample_id, qa_id=qa_id)
        payload["cat5_answer_key"] = answer_key
        payload["temperature"] = 0.5
        payload["prompt"] = AMEM_ANSWER_PROMPT_CAT_5.format(
            context_text=context_text,
            question=str(question).strip(),
            option_a=answer_key["a"],
            option_b=answer_key["b"],
        )
        return payload
    if category_int == 2:
        payload["prompt"] = AMEM_ANSWER_PROMPT_CAT_2.format(
            context_text=context_text,
            question=str(question).strip(),
        )
        return payload
    if category_int == 3:
        payload["prompt"] = AMEM_ANSWER_PROMPT_CAT_3.format(
            context_text=context_text,
            question=str(question).strip(),
        )
        return payload
    payload["prompt"] = AMEM_ANSWER_PROMPT_DEFAULT.format(
        context_text=context_text,
        question=str(question).strip(),
    )
    return payload
