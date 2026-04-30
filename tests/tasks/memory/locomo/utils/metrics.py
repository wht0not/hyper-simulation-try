from __future__ import annotations

import json
import re
import string
from collections import Counter
from statistics import fmean
from typing import Any

import numpy as np
import regex
from nltk.stem import PorterStemmer
from ollama import chat

from hyper_simulation.component.embedding import get_embedding_batch
from .utils import coerce_category


_ps = PorterStemmer()
LLM_JUDGE_MODEL = "atla/selene-mini"
LLM_JUDGE_REPEAT = 5

LLM_JUDGE_PROMPT = """
Your task is to label an answer to a question as CORRECT or WRONG.
You will be given:
1. The question.
2. The category.
3. The candidate answer field.
4. The model prediction.

Rules:
- For categories 1, 2, 3, 4: the candidate answer field is the ground-truth answer. Be generous about wording differences if the prediction clearly refers to the same fact.
- For category 5: the candidate answer field is intentionally misleading. The correct behavior is to say that the information is unavailable, not mentioned, unknown, or cannot be determined. If the model prediction gives the misleading candidate answer as if it were true, that is WRONG.
- For time expressions, accept equivalent dates or time periods even if wording differs.

Question: {question}
Category: {category}
Candidate answer field: {gold_answer}
Model prediction: {prediction}

Return only a JSON object in this format:
{{"label": "CORRECT"}} or {{"label": "WRONG"}}
""".strip()


def normalize_answer(text: Any) -> str:
    if text is None:
        return ""
    cleaned = str(text).replace("</s>", "").replace("</think>", "").strip()
    patterns = [
        r"###\s*Final\s*Answer:\s*(.+?)(?:\n|$)",
        r"ANSWER:\s*(.+?)(?:\n|$)",
        r"Answer:\s*(.+?)(?:\n|$)",
    ]
    for pattern in patterns:
        match_obj = re.search(pattern, cleaned, re.IGNORECASE)
        if match_obj:
            cleaned = match_obj.group(1).strip()
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
    return _ps.stem(token)


def f1_score_official(prediction: str, ground_truth: str) -> float:
    prediction_tokens = [_stem_token(w) for w in _normalize_answer_official(prediction).split()]
    ground_truth_tokens = [_stem_token(w) for w in _normalize_answer_official(ground_truth).split()]
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    return (2 * precision * recall) / (precision + recall)


def f1_multi_official(prediction: str, ground_truth: str) -> float:
    predictions = [p.strip() for p in str(prediction).split(",")]
    ground_truths = [g.strip() for g in str(ground_truth).split(",")]
    if not predictions or not ground_truths:
        return 0.0
    return fmean([max([f1_score_official(p, gt) for p in predictions]) for gt in ground_truths])


def locomo_f1(prediction: str, golden: Any, category: Any) -> float:
    answer = str(golden).strip() if golden is not None else ""
    pred = str(prediction or "")
    category_int = coerce_category(category)

    if category_int == 3:
        answer = answer.split(";")[0].strip()

    if category_int in [2, 3, 4]:
        return float(f1_score_official(pred, answer))
    if category_int in [1]:
        return float(f1_multi_official(pred, answer))
    if category_int in [5]:
        pred_low = pred.lower()
        return 1.0 if ("no information available" in pred_low or "not mentioned" in pred_low) else 0.0
    return float(f1_score_official(pred, answer))


def bleu1_score(prediction: str, golden: Any) -> float:
    pred_tokens = _normalize_answer_official(str(prediction or "")).split()
    ref_tokens = _normalize_answer_official(str(golden or "")).split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    ref_counter = Counter(ref_tokens)
    pred_counter = Counter(pred_tokens)
    overlap = sum(min(pred_counter[token], ref_counter[token]) for token in pred_counter)
    return float(overlap / max(len(pred_tokens), 1))


def _lcs_length(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i, a_tok in enumerate(a, start=1):
        for j, b_tok in enumerate(b, start=1):
            if a_tok == b_tok:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1]


def rouge_l_score(prediction: str, golden: Any) -> float:
    pred_tokens = _normalize_answer_official(str(prediction or "")).split()
    ref_tokens = _normalize_answer_official(str(golden or "")).split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    lcs = _lcs_length(pred_tokens, ref_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    beta = 1.2
    return float(((1 + beta**2) * precision * recall) / (recall + beta**2 * precision))


def bert_score_f1(prediction: str, golden: Any) -> float:
    pred_tokens = _normalize_answer_official(str(prediction or "")).split()
    ref_tokens = _normalize_answer_official(str(golden or "")).split()
    if not pred_tokens or not ref_tokens:
        return 0.0

    pred_embeddings = np.stack(get_embedding_batch(pred_tokens))
    ref_embeddings = np.stack(get_embedding_batch(ref_tokens))
    pred_norm = np.linalg.norm(pred_embeddings, axis=1, keepdims=True)
    ref_norm = np.linalg.norm(ref_embeddings, axis=1, keepdims=True)
    pred_norm[pred_norm == 0] = 1e-12
    ref_norm[ref_norm == 0] = 1e-12
    pred_embeddings = pred_embeddings / pred_norm
    ref_embeddings = ref_embeddings / ref_norm
    similarity_matrix = np.matmul(pred_embeddings, ref_embeddings.T)
    similarity_matrix = np.clip(similarity_matrix, -1.0, 1.0)
    precision = float(np.mean(np.max(similarity_matrix, axis=1)))
    recall = float(np.mean(np.max(similarity_matrix, axis=0)))
    if precision + recall == 0:
        return 0.0
    return float((2 * precision * recall) / (precision + recall))


def _bert_score_from_tokens(
    pred_tokens: list[str],
    ref_tokens: list[str],
    embedding_cache: dict[str, np.ndarray] | None = None,
) -> float:
    if not pred_tokens or not ref_tokens:
        return 0.0
    embedding_cache = embedding_cache or {}
    missing_tokens = [token for token in set(pred_tokens + ref_tokens) if token not in embedding_cache]
    if missing_tokens:
        get_embedding_batch(missing_tokens, cache=embedding_cache)
    pred_embeddings = np.stack([embedding_cache[token] for token in pred_tokens])
    ref_embeddings = np.stack([embedding_cache[token] for token in ref_tokens])
    pred_norm = np.linalg.norm(pred_embeddings, axis=1, keepdims=True)
    ref_norm = np.linalg.norm(ref_embeddings, axis=1, keepdims=True)
    pred_norm[pred_norm == 0] = 1e-12
    ref_norm[ref_norm == 0] = 1e-12
    pred_embeddings = pred_embeddings / pred_norm
    ref_embeddings = ref_embeddings / ref_norm
    similarity_matrix = np.matmul(pred_embeddings, ref_embeddings.T)
    similarity_matrix = np.clip(similarity_matrix, -1.0, 1.0)
    precision = float(np.mean(np.max(similarity_matrix, axis=1)))
    recall = float(np.mean(np.max(similarity_matrix, axis=0)))
    if precision + recall == 0:
        return 0.0
    return float((2 * precision * recall) / (precision + recall))


def _extract_llm_label(text: str) -> str:
    content = str(text or "").strip()
    try:
        payload = json.loads(content)
        label = str(payload.get("label", "")).strip().upper()
        if label in {"CORRECT", "WRONG"}:
            return label
    except Exception:
        pass
    json_match = re.search(r'\{\s*"label"\s*:\s*"([^"]+)"\s*\}', content, re.IGNORECASE)
    if json_match:
        label = json_match.group(1).strip().upper()
        if label in {"CORRECT", "WRONG"}:
            return label
    word_match = re.search(r"\b(CORRECT|WRONG)\b", content, re.IGNORECASE)
    if word_match:
        return word_match.group(1).upper()
    return "WRONG"


def llm_judge(
    question: str,
    gold_answer: Any,
    prediction: str,
    category: Any,
    model_name: str = LLM_JUDGE_MODEL,
    repeat: int = LLM_JUDGE_REPEAT,
) -> dict[str, Any]:
    prompt = LLM_JUDGE_PROMPT.format(
        question=str(question or "").strip(),
        category=coerce_category(category),
        gold_answer=str(gold_answer or "").strip(),
        prediction=str(prediction or "").strip(),
    )
    runs: list[dict[str, Any]] = []
    for _ in range(max(1, int(repeat))):
        try:
            response = chat(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0},
            )
            raw_content = response.message.content if hasattr(response, "message") else response["message"]["content"]
            label = _extract_llm_label(raw_content)
            runs.append(
                {
                    "score": 1.0 if label == "CORRECT" else 0.0,
                    "label": label,
                    "raw": raw_content,
                }
            )
        except Exception as exc:
            runs.append(
                {
                    "score": 0.0,
                    "label": "ERROR",
                    "raw": f"{type(exc).__name__}: {exc}",
                }
            )
    scores = [float(run.get("score", 0.0)) for run in runs]
    mean_score = float(np.mean(scores)) if scores else 0.0
    std_score = float(np.std(scores)) if scores else 0.0
    return {
        "score": mean_score,
        "mean": mean_score,
        "std": std_score,
        "runs": runs,
        "model": model_name,
        "repeat": len(runs),
    }


def compute_metrics(
    question: str,
    prediction: str,
    golden: Any,
    category: Any,
    llm_judge_model: str = LLM_JUDGE_MODEL,
    llm_judge_repeat: int = LLM_JUDGE_REPEAT,
) -> dict[str, Any]:
    prediction = normalize_answer(prediction)
    category_int = coerce_category(category)
    locomo_score = locomo_f1(prediction, golden, category)
    if category_int == 5:
        bleu1 = None
        rouge_l = None
        bert_f1 = None
        judge = None
    else:
        bleu1 = bleu1_score(prediction, golden)
        rouge_l = rouge_l_score(prediction, golden)
        bert_f1 = bert_score_f1(prediction, golden)
        judge = llm_judge(
            question=question,
            gold_answer=golden,
            prediction=prediction,
            category=category,
            model_name=llm_judge_model,
            repeat=llm_judge_repeat,
        )
    return {
        "locomo_score": float(locomo_score),
        "f1": float(locomo_score),
        "bleu1": None if bleu1 is None else float(bleu1),
        "rouge_l": None if rouge_l is None else float(rouge_l),
        "bert_score_f1": None if bert_f1 is None else float(bert_f1),
        "llm_as_judge": judge,
    }


def compute_base_metrics(
    prediction: str,
    golden: Any,
    category: Any,
) -> dict[str, Any]:
    prediction = normalize_answer(prediction)
    category_int = coerce_category(category)
    locomo_score = locomo_f1(prediction, golden, category)
    if category_int == 5:
        bleu1 = None
        rouge_l = None
        bert_f1 = None
    else:
        bleu1 = bleu1_score(prediction, golden)
        rouge_l = rouge_l_score(prediction, golden)
        bert_f1 = bert_score_f1(prediction, golden)
    return {
        "locomo_score": float(locomo_score),
        "f1": float(locomo_score),
        "bleu1": None if bleu1 is None else float(bleu1),
        "rouge_l": None if rouge_l is None else float(rouge_l),
        "bert_score_f1": None if bert_f1 is None else float(bert_f1),
    }


def compute_base_metrics_batch(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized_predictions = [normalize_answer(row.get("prediction", "")) for row in rows]
    token_pairs: list[tuple[list[str], list[str]]] = []
    all_tokens: list[str] = []
    for row, prediction in zip(rows, normalized_predictions):
        category_int = coerce_category(row.get("category"))
        if category_int == 5:
            token_pairs.append(([], []))
            continue
        pred_tokens = _normalize_answer_official(str(prediction or "")).split()
        ref_tokens = _normalize_answer_official(str(row.get("answer") or "")).split()
        token_pairs.append((pred_tokens, ref_tokens))
        all_tokens.extend(pred_tokens)
        all_tokens.extend(ref_tokens)

    embedding_cache: dict[str, np.ndarray] = {}
    if all_tokens:
        get_embedding_batch(all_tokens, cache=embedding_cache)

    metrics_list: list[dict[str, Any]] = []
    for row, prediction, (pred_tokens, ref_tokens) in zip(rows, normalized_predictions, token_pairs):
        category_int = coerce_category(row.get("category"))
        locomo_score = locomo_f1(prediction, row.get("answer"), category_int)
        if category_int == 5:
            bleu1 = None
            rouge_l = None
            bert_f1 = None
        else:
            bleu1 = bleu1_score(prediction, row.get("answer"))
            rouge_l = rouge_l_score(prediction, row.get("answer"))
            bert_f1 = _bert_score_from_tokens(pred_tokens, ref_tokens, embedding_cache=embedding_cache)
        metrics_list.append(
            {
                "locomo_score": float(locomo_score),
                "f1": float(locomo_score),
                "bleu1": None if bleu1 is None else float(bleu1),
                "rouge_l": None if rouge_l is None else float(rouge_l),
                "bert_score_f1": None if bert_f1 is None else float(bert_f1),
            }
        )
    return metrics_list


def compute_llm_judge_metrics(
    question: str,
    prediction: str,
    golden: Any,
    category: Any,
    llm_judge_model: str = LLM_JUDGE_MODEL,
    llm_judge_repeat: int = LLM_JUDGE_REPEAT,
) -> dict[str, Any] | None:
    if coerce_category(category) == 5:
        return None
    return llm_judge(
        question=question,
        gold_answer=golden,
        prediction=normalize_answer(prediction),
        category=category,
        model_name=llm_judge_model,
        repeat=llm_judge_repeat,
    )
