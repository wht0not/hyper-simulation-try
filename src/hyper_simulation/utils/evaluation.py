"""
Evaluation metrics for QA tasks (salvaged from question_answer module)
"""
import numpy as np
import string
import re
from collections import Counter
from typing import Dict, List, Union


def normalize_answer(s: Union[str, List]) -> str:
    """Normalize answer for comparison"""
    if isinstance(s, list):
        s = s[0] if len(s) > 0 else ""
    if not isinstance(s, str):
        s = str(s)

    def remove_articles(text):
        if text.strip() in ['a', 'an', 'the']:
            return text
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match_score(prediction: str, ground_truth: str) -> bool:
    """Check if prediction exactly matches ground truth after normalization"""
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def qa_f1_score(prediction: str, ground_truth: str) -> float:
    """Calculate token-level F1 score"""
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def match(prediction: str, ground_truth: List[str]) -> int:
    """Check if prediction contains any ground truth (substring match)"""
    if not isinstance(prediction, str):
        prediction = str(prediction)

    for gt in ground_truth:
        if isinstance(gt, list):
            gt = gt[0] if gt else ""
        if not isinstance(gt, str):
            gt = str(gt)

        if not gt:
            continue

        pred_lower = prediction.lower()
        gt_lower = gt.lower()

        # Strict matching for short answers
        if len(gt) <= 2:
            if re.search(rf'\b{re.escape(gt_lower)}\b', pred_lower):
                return 1
        else:
            if gt_lower in pred_lower:
                return 1
    return 0


def metric_max_over_ground_truths(metric_fn, prediction: str, ground_truths: List[str]) -> float:
    """Calculate maximum metric score over multiple ground truths"""
    scores = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores.append(score)
    return max(scores) if scores else 0.0


def evaluate_answer(prediction: str, ground_truth: Union[str, List[str]]) -> Dict[str, float]:
    """
    Evaluate generated answer against ground truth

    Args:
        prediction: Model generated answer
        ground_truth: Correct answer(s)

    Returns:
        Dictionary containing evaluation metrics
    """
    if isinstance(ground_truth, list):
        ground_truths = ground_truth
    else:
        ground_truths = [ground_truth]

    # Calculate Exact Match
    em_score = metric_max_over_ground_truths(
        exact_match_score, prediction, ground_truths
    )

    # Calculate F1 score
    f1_score = max([qa_f1_score(prediction, gt) for gt in ground_truths])

    # Calculate Match (partial match)
    match_score = match(prediction, ground_truths)

    return {
        "exact_match": em_score,
        "f1": f1_score,
        "match": match_score
    }


def accuracy(predictions: List[str], labels: List[List[str]]) -> float:
    """Calculate accuracy for multiple predictions"""
    match_count = 0
    for pred, label in zip(predictions, labels):
        target = label[0] if label else ""
        if pred == target:
            match_count += 1

    return 100 * (match_count / len(predictions)) if predictions else 0.0


def f1_batch(predictions: List[str], labels: List[Union[str, List[str]]]) -> float:
    """Calculate average F1 score for a batch"""
    f1_all = []
    for prediction, answers in zip(predictions, labels):
        if isinstance(answers, list):
            if len(answers) == 0:
                f1_all.append(0)
            else:
                f1_all.append(np.max([qa_f1_score(prediction, gt) for gt in answers]))
        else:
            f1_all.append(qa_f1_score(prediction, answers))

    return 100 * np.mean(f1_all) if f1_all else 0.0
