from __future__ import annotations

from typing import Any

AMEM_GENERATE_QUERY_PROMPT = """Given the following question, generate several keywords separated by commas.

Question: {question}

Keywords:"""

AMEM_RELEVANT_PARTS_PROMPT = """Given the following conversation memories and a question, select the most relevant parts of the conversation that would help answer the question. Include the date/time if available.

Conversation memories:
{memories_text}

Question: {query}

Return only the relevant parts of the conversation that would help answer this specific question.
If no parts are relevant, return the input unchanged."""

AMEM_ANSWER_PROMPT = """Based on the context: {context}, write an answer in the form of a short phrase for the following question. Answer with exact words from the context whenever possible.

Question: {question} Short answer:"""

HYPER_AMEM_ANSWER_PROMPT = """
Context:
{context_text}

Answer with a short phrase.
Question: {question}
Short answer:"""


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
) -> dict[str, Any]:
    prompt = AMEM_ANSWER_PROMPT.format(
        context=context_text,
        question=str(question).strip(),
    )
    return {"temperature": 0.1, "prompt": prompt}


def build_hyper_amem_answer_prompt(
    context_text: str,
    question: str,
) -> dict[str, Any]:
    prompt = HYPER_AMEM_ANSWER_PROMPT.format(
        context_text=context_text,
        question=str(question).strip(),
    )
    return {"temperature": 0.1, "prompt": prompt}
