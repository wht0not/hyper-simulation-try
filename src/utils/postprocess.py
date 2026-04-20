from utils.metrics import exact_match_score, metric_max_over_ground_truths, qa_f1_score, match
from typing import Dict
import re
import logging

def evaluate_answer(prediction: str, ground_truth: list | str) -> Dict[str, float]:
    """评估生成的答案"""
    if isinstance(ground_truth, list):
        ground_truths = ground_truth
    else:
        ground_truths = [ground_truth]
    
    em_score = metric_max_over_ground_truths(
        exact_match_score, prediction, ground_truths
    )
    
    f1_score = max([qa_f1_score(prediction, gt) for gt in ground_truths])
    match_score = match(prediction, ground_truths)
    
    return {
        "exact_match": em_score,
        "f1": f1_score,
        "match": match_score
    }


def postprocess_answer(answer: str) -> tuple[str, str, bool]:
    """从 LLM 输出中提取最终答案"""
    logger = logging.getLogger(__name__)
    
    if not answer:
        return "unanswerable", "empty_input", True
    
    answer = answer.replace("</s>", "").replace("</think>", "").strip()
    
    final_answer_pattern = r"###\s*Final\s*Answer:\s*(.+?)(?:\n|$)"
    extracted = answer
    parse_status = "fallback_truncated"
    match_count = 0
    
    while True:
        match = re.search(final_answer_pattern, extracted, re.IGNORECASE)
        if match:
            extracted = match.group(1).strip()
            match_count += 1
            parse_status = "parsed_final_answer"
        else:
            break
    
    if match_count > 0:
        cleaned = extracted.strip(" .,;:!?\"'")
        if cleaned:
            return cleaned, parse_status, False
    
    lines = answer.strip().split('\n')
    exclude_keywords = ['step', 'reason', 'explain', 'note', 'context', 
                       'paragraph', 'think', 'analysis', 'conclusion']
    
    for line in reversed(lines):
        line = line.strip()
        if (line and 
            len(line) < 100 and 
            not any(k in line.lower() for k in exclude_keywords) and
            not line.startswith('#')):
            cleaned = line.strip(" .,;:!?\"'")
            if cleaned:
                return cleaned, "parsed_last_line", False
    
    logger.warning(f"⚠️ Could not parse answer (match_count={match_count}), using fallback. Output preview: {answer[:100]}...")
    
    cleaned = answer.strip(" .,;:!?\"'")[:200]
    return cleaned if cleaned else "unanswerable", "fallback_truncated", True
