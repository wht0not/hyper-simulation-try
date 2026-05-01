from __future__ import annotations

from typing import Any

from utils.qa_utils import build_cat5_choice_question


MEMORYBANK_SUMMARIZE_SESSION_PROMPT = """Please summarize the following dialogue as concisely as possible, extracting the main themes and key information. If there are multiple key events, you may summarize them separately.

Dialogue content:
{dialogue_text}

Summarization:"""


MEMORYBANK_SUMMARIZE_PERSONALITY_PROMPT = """Based on the following dialogue, please summarize the user's personality traits and emotions, and devise response strategies based on your speculation.

Dialogue content:
{dialogue_text}

{user_name}'s personality traits, emotions, and AI response strategy are:"""


MEMORYBANK_OVERALL_HISTORY_PROMPT = """Please provide a highly concise summary of the following dated events, capturing the essential key information as succinctly as possible.

{dated_summaries}

Summarization:"""


MEMORYBANK_OVERALL_PERSONALITY_PROMPT = """The following are the user's exhibited personality traits and emotions throughout multiple dialogues, along with appropriate response strategies for the current situation:

{dated_personality}

Please provide a highly concise and general summary of the user's personality and the most appropriate response strategy for the AI companion, summarized as:"""


MEMORYBANK_ANSWER_PROMPT = """Now you will play the role of an AI companion for user {user_name}. You should understand past memory and extract information from memory when it is relevant to the question.

The summary of your past memories with the user is:
{overall_history}

The user's personality and response strategy are:
{overall_personality}

The memory most relevant to the question is:
{related_memory}

Answer the following question briefly and accurately using exact words from memory whenever possible.

Question: {question}

Short answer:"""


MEMORYBANK_ANSWER_PROMPT_CAT_2 = """Now you will play the role of an AI companion for user {user_name}. You should understand past memory and extract information from memory when it is relevant to the question.

The summary of your past memories with the user is:
{overall_history}

The user's personality and response strategy are:
{overall_personality}

The memory most relevant to the question is:
{related_memory}

Answer the following question using DATE of CONVERSATION to give an approximate date. Please generate the shortest possible answer, using words from the memory where possible, and avoid using any subjects.

Question: {question}

Short answer:"""


MEMORYBANK_ANSWER_PROMPT_CAT_5 = """Now you will play the role of an AI companion for user {user_name}. You should understand past memory and extract information from memory when it is relevant to the question.

The summary of your past memories with the user is:
{overall_history}

The user's personality and response strategy are:
{overall_personality}

The memory most relevant to the question is:
{related_memory}

Answer the following question by selecting the correct answer: {option_a} or {option_b}

Question: {question}

Short answer:"""


def build_memorybank_session_summary_prompt(dialogue_text: str) -> str:
    return MEMORYBANK_SUMMARIZE_SESSION_PROMPT.format(dialogue_text=str(dialogue_text).strip())


def build_memorybank_personality_summary_prompt(dialogue_text: str, user_name: str) -> str:
    return MEMORYBANK_SUMMARIZE_PERSONALITY_PROMPT.format(
        dialogue_text=str(dialogue_text).strip(),
        user_name=str(user_name).strip() or "User",
    )


def build_memorybank_overall_history_prompt(dated_summaries: str) -> str:
    return MEMORYBANK_OVERALL_HISTORY_PROMPT.format(dated_summaries=str(dated_summaries).strip())


def build_memorybank_overall_personality_prompt(dated_personality: str) -> str:
    return MEMORYBANK_OVERALL_PERSONALITY_PROMPT.format(dated_personality=str(dated_personality).strip())


def build_memorybank_answer_prompt(
    user_name: str,
    overall_history: str,
    overall_personality: str,
    related_memory: str,
    question: str,
    category: Any,
    answer: str = "",
    sample_id: Any = "",
    qa_id: Any = "",
) -> dict[str, Any]:
    category_int = int(category)
    base_kwargs = {
        "user_name": str(user_name).strip() or "User",
        "overall_history": str(overall_history).strip() or "No past summary available.",
        "overall_personality": str(overall_personality).strip() or "No personality summary available.",
        "related_memory": str(related_memory).strip() or "No relevant memory found.",
        "question": str(question).strip(),
    }
    payload: dict[str, Any] = {"temperature": 0.1}
    if category_int == 5:
        cat5_question, cat5_answer_key = build_cat5_choice_question(
            question,
            str(answer or ""),
            sample_id=sample_id,
            qa_id=qa_id,
        )
        payload["cat5_answer_key"] = cat5_answer_key
        cat5_kwargs = base_kwargs.copy()
        cat5_kwargs["question"] = cat5_question
        cat5_kwargs["option_a"] = cat5_answer_key["a"]
        cat5_kwargs["option_b"] = cat5_answer_key["b"]
        payload["prompt"] = MEMORYBANK_ANSWER_PROMPT_CAT_5.format(**cat5_kwargs)
        return payload
    if category_int == 2:
        payload["prompt"] = MEMORYBANK_ANSWER_PROMPT_CAT_2.format(**base_kwargs)
        return payload
    payload["prompt"] = MEMORYBANK_ANSWER_PROMPT.format(**base_kwargs)
    return payload
