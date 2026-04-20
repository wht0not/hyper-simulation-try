# coding: utf-8
import json
import jsonlines
from pathlib import Path
from typing import List, Dict, Any

def load_hotpotqa_data(file_path: str) -> List[Dict[str, Any]]:
    """
    加载HotpotQA数据集
    
    支持的schema:
    - id: 问题ID
    - question: 问题文本
    - answer: 正确答案
    - type: 问题类型 (comparison, bridge)
    - level: 难度级别 (easy, medium, hard)
    - supporting_facts: {title: [...], sent_id: [...]}
    - context: {title: [...], sentences: [[...], ...]}
    """
    data = []
    path = Path(file_path)
    
    paths = []
    if path.is_dir():
        paths = list(path.glob('hotpot_*.jsonl'))
    else:
        paths = [path]
    
    for path in paths:
        if path.suffix == '.json':
            with open(path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
                for item in raw_data:
                    formatted_item = {
                        '_id': item.get('id', ''),
                        'question': item.get('question', '').strip(),
                        'answer': item.get('answer', '').strip(),
                        'type': item.get('type', 'unknown'),
                        'level': item.get('level', 'unknown'),
                        'context': [
                            (title, sentences)
                            for title, sentences in zip(
                                item['context']['title'],
                                item['context']['sentences']
                            )
                        ],
                        'supporting_facts': item.get('supporting_facts', {}),
                    }
                    data.append(formatted_item)
        elif path.suffix == '.jsonl':
            with jsonlines.open(path, 'r') as reader:
                for item in reader:
                    formatted_item = {
                        '_id': item.get('id', ''),
                        'question': item.get('question', '').strip(),
                        'answer': item.get('answer', '').strip(),
                        'type': item.get('type', 'unknown'),
                        'level': item.get('level', 'unknown'),
                        'context': [
                            (title, sentences)
                            for title, sentences in zip(
                                item['context']['title'],
                                item['context']['sentences']
                            )
                        ],
                        'supporting_facts': item.get('supporting_facts', {}),
                    }
                    data.append(formatted_item)
        else:
            raise ValueError("Unsupported file format. Please use .json or .jsonl")
    
    return data

def load_musique_data(file_path: str, use_supporting_only: bool = True) -> List[Dict[str, Any]]:
    """加载MuSiQue数据集（jsonl）"""
    data: List[Dict[str, Any]] = []
    path = Path(file_path)

    paths = []
    if path.is_dir():
        paths = list(path.glob("*.jsonl"))
    else:
        paths = [path]

    for path in paths:
        if path.suffix != ".jsonl":
            raise ValueError("Unsupported file format. Please use .jsonl")

        with jsonlines.open(path, "r") as reader:
            for item in reader:
                paragraphs = item.get("paragraphs", []) or []
                context = []
                supporting_flags = []

                for p in paragraphs:
                    is_supporting = bool(p.get("is_supporting", False))
                    if use_supporting_only and not is_supporting:
                        continue
                    title = (p.get("title") or "").strip()
                    paragraph_text = (p.get("paragraph_text") or "").strip()

                    if title or paragraph_text:
                        context.append((title, [paragraph_text] if paragraph_text else []))
                        supporting_flags.append(is_supporting)

                formatted_item = {
                    "_id": item.get("id", ""),
                    "question": (item.get("question") or "").strip(),
                    "answer": (item.get("answer") or "").strip(),
                    "answer_alias": item.get("answer_alias", []) or [],
                    "answerable": item.get("answerable", True),
                    "question_decomposition": item.get("question_decomposition", []) or [],
                    "context": context,
                    "supporting_flags": supporting_flags,
                }
                data.append(formatted_item)

    return data

def load_multihop_data(file_path: str) -> List[Dict[str, Any]]:
    """加载多跳推理数据集（jsonl）"""
    data: List[Dict[str, Any]] = []
    path = Path(file_path)

    paths = []
    if path.is_dir():
        paths = list(path.glob("*.jsonl"))
    else:
        paths = [path]

    for path in paths:
        if path.suffix != ".jsonl":
            raise ValueError("Unsupported file format. Please use .jsonl")

        with jsonlines.open(path, "r") as reader:
            for item in reader:
                evidence_list = item.get("evidence_list", []) or []
                context = []
                supporting_flags = []
                for evidence in evidence_list:
                    title = (evidence.get("title") or "").strip()
                    text = (evidence.get("text") or evidence.get("fact") or "").strip()
                    if title or text:
                        context.append((title, [text] if text else []))
                        supporting_flags.append(True)

                formatted_item = {
                    "_id": item.get("id", ""),
                    "question": (item.get("query") or "").strip(),
                    "answer": (item.get("answer") or "").strip(),
                    "question_type": item.get("question_type", ""),
                    "metadata": item.get("metadata", []) or [],
                    "context": context,
                    "supporting_flags": supporting_flags,
                }
                data.append(formatted_item)

    return data

def load_arc_data(file_path: str) -> List[Dict[str, Any]]:
    """加载 ARC 数据集（选择题）"""
    data: List[Dict[str, Any]] = []
    path = Path(file_path)

    paths = []
    if path.is_dir():
        paths = list(path.glob("*.jsonl"))
    else:
        paths = [path]

    for path in paths:
        if path.suffix != ".jsonl":
            raise ValueError("Unsupported file format. Please use .jsonl")

        with jsonlines.open(path, "r") as reader:
            for item in reader:
                choices = item.get("choices", {})
                options_text = choices.get("text", [])
                options_label = choices.get("label", [])
                
                options_str = "\n".join([
                    f"{label}) {text}" 
                    for label, text in zip(options_label, options_text)
                ]) if options_label and options_text else ""
                
                question_with_options = f"{item.get('question', '').strip()}\n\nOptions:\n{options_str}"
                
                answer_label = item.get("answerKey", "")
                answer_text = ""
                if answer_label and options_label and options_text:
                    try:
                        idx = options_label.index(answer_label)
                        answer_text = options_text[idx]
                    except (ValueError, IndexError):
                        answer_text = answer_label
                
                formatted_item = {
                    "_id": item.get("id", ""),
                    "question": question_with_options,
                    "answer": answer_text,
                    "answer_label": answer_label,
                    "options": options_text,
                    "option_labels": options_label,
                    "context": [],
                    "supporting_flags": [],
                }
                data.append(formatted_item)

    return data

def load_contract_qa_data(file_path: str) -> List[Dict[str, Any]]:
    """加载 Contract QA 数据（合同条款问答）"""
    data: List[Dict[str, Any]] = []
    path = Path(file_path)
    
    paths = [path] if not path.is_dir() else list(path.glob("*.jsonl"))
    
    for filepath in paths:
        if filepath.suffix != ".jsonl":
            continue
        
        with jsonlines.open(filepath, "r") as reader:
            for idx, item in enumerate(reader):
                text = (item.get("text") or "").strip()
                question = (item.get("question") or "").strip()
                answer = (item.get("answer") or "").strip()
                
                if not text or not question:
                    continue
                
                formatted_item = {
                    "_id": item.get("id", f"contract_qa_{idx}"),
                    "question": question,
                    "answer": answer,
                    "context": [("contract", [text])],
                    "context_type": "contract",
                }
                data.append(formatted_item)
    
    return data

def load_consumer_contracts_qa_data(file_path: str) -> List[Dict[str, Any]]:
    """加载消费者合同QA数据"""
    data: List[Dict[str, Any]] = []
    path = Path(file_path)
    paths = [path] if not path.is_dir() else list(path.glob("*.jsonl"))
    for filepath in paths:
        if filepath.suffix != ".jsonl":
            continue
        with jsonlines.open(filepath, "r") as reader:
            for idx, item in enumerate(reader):
                text = (item.get("text") or item.get("contract") or "").strip()
                question = (item.get("question") or "").strip()
                answer = (item.get("answer") or "").strip()
                
                if not text or not question:
                    continue
                
                formatted_item = {
                    "_id": item.get("id", f"consumer_qa_{idx}"),
                    "question": question,
                    "answer": answer,
                    "context": [("terms_of_service", [text])],
                    "context_type": "tos",
                }
                data.append(formatted_item)
    return data

def load_privacy_policy_qa_data(file_path: str) -> List[Dict[str, Any]]:
    """加载隐私政策QA数据"""
    data: List[Dict[str, Any]] = []
    path = Path(file_path)
    
    paths = [path] if not path.is_dir() else list(path.glob("*.jsonl"))
    
    for filepath in paths:
        if filepath.suffix != ".jsonl":
            continue
        
        with jsonlines.open(filepath, "r") as reader:
            for idx, item in enumerate(reader):
                text = (item.get("text") or "").strip()
                question = (item.get("question") or "").strip()
                answer = (item.get("answer") or "").strip()
                
                if not text or not question:
                    continue
                
                formatted_item = {
                    "_id": item.get("id", f"privacy_qa_{idx}"),
                    "question": question,
                    "answer": answer,
                    "context": [("privacy_policy", [text])],
                    "context_type": "privacy_policy",
                }
                data.append(formatted_item)
    
    return data

def load_rule_qa_data(file_path: str) -> List[Dict[str, Any]]:
    """加载规则QA数据"""
    data: List[Dict[str, Any]] = []
    path = Path(file_path)
    paths = [path] if not path.is_dir() else list(path.glob("*.jsonl"))
    for filepath in paths:
        if filepath.suffix != ".jsonl":
            continue
        with jsonlines.open(filepath, "r") as reader:
            for idx, item in enumerate(reader):
                question = (item.get("question") or "").strip()
                text = (item.get("text") or "").strip()
                answer = (item.get("answer") or "").strip()
                
                if not question and text and text.endswith("?"):
                    question = text
                    text = ""
                
                if not question:
                    continue
                
                formatted_item = {
                    "_id": item.get("id", f"rule_qa_{idx}"),
                    "question": question,
                    "answer": answer,
                    "context": [("rule_definition", [text])] if text else [],
                    "context_type": "rules",
                }
                data.append(formatted_item)
    return data

def load_legalbench_qa_data(file_path: str) -> List[Dict[str, Any]]:
    """加载 LegalBench QA 数据"""
    data: List[Dict[str, Any]] = []
    path = Path(file_path)

    paths = []
    if path.is_dir():
        paths = list(path.glob("*.jsonl"))
    else:
        paths = [path]

    for path in paths:
        if path.suffix != ".jsonl":
            continue

        with jsonlines.open(path, "r") as reader:
            for item in reader:
                formatted_item = {
                    "_id": item.get("id", ""),
                    "question": (item.get("question") or "").strip(),
                    "answer": (item.get("answer") or "").strip(),
                    "context": [(path.stem, [item.get("text", "")])],
                    "context_type": "legal_document",
                }
                data.append(formatted_item)

    return data

def load_sara_entailment_data(file_path: str) -> List[Dict[str, Any]]:
    """加载 SARA Entailment 数据"""
    data: List[Dict[str, Any]] = []
    path = Path(file_path)
    paths = [path] if not path.is_dir() else list(path.glob("*.jsonl"))
    
    for path in paths:
        if path.suffix != ".jsonl":
            continue
        with jsonlines.open(path, "r") as reader:
            for item in reader:
                statute = (item.get("statute") or "").strip()
                description = (item.get("description") or "").strip()
                hypothesis = (item.get("question") or item.get("hypothesis") or "").strip()
                
                context_parts = [p for p in [statute, description] if p]
                context_text = "\n\n".join(context_parts) if context_parts else ""
                
                answer_raw = (item.get("answer") or "").strip()
                answer_map = {
                    "Entailment": "Entails",
                    "Entails": "Entails",
                    "Contradiction": "Contradicts",
                    "Contradicts": "Contradicts",
                    "Neutral": "Neutral",
                }
                answer = answer_map.get(answer_raw, answer_raw)
                
                formatted_item = {
                    "_id": item.get("id", item.get("case id", "")),
                    "question": hypothesis,
                    "answer": answer,
                    "context": [("legal_text", [context_text])],
                    "context_type": "legal_sara_entailment"
                }
                data.append(formatted_item)
    return data

def load_privacy_policy_entailment_data(file_path: str) -> List[Dict[str, Any]]:
    """加载隐私政策 Entailment 数据"""
    data: List[Dict[str, Any]] = []
    path = Path(file_path)
    paths = [path] if not path.is_dir() else list(path.glob("*.jsonl"))
    
    for path in paths:
        if path.suffix != ".jsonl":
            continue
        with jsonlines.open(path, "r") as reader:
            for item in reader:
                policy_text = (item.get("text") or "").strip()
                description = (item.get("description") or "").strip()
                answer = (item.get("answer") or "").strip()
                
                formatted_item = {
                    "_id": item.get("id", f"ppe_{item.get('index', '')}"),
                    "question": description,
                    "answer": answer,
                    "context": [("privacy_policy", [policy_text])],
                    "context_type": "privacy_policy_entailment",
                }
                data.append(formatted_item)
    return data

def load_legalbench_insurance_data(file_path: str) -> List[Dict[str, Any]]:
    """加载 LegalBench 保险数据"""
    data: List[Dict[str, Any]] = []
    path = Path(file_path)

    if path.is_dir():
        paths = list(path.glob("*.jsonl"))
    else:
        paths = [path]

    for path in paths:
        if path.suffix != ".jsonl":
            continue

        with jsonlines.open(path, "r") as reader:
            for item in reader:
                formatted_item = {
                    "_id": item.get("id", ""),
                    "question": (item.get("question") or "").strip(),
                    "answer": (item.get("answer") or "").strip(),
                    "context": [("insurance_policy", [item.get("text", "")])],
                    "context_type": "legal_insurance",
                }
                data.append(formatted_item)

    return data

def load_legalbench_corporate_lobbying_data(file_path: str) -> List[Dict[str, Any]]:
    """加载 LegalBench 企业游说数据"""
    data: List[Dict[str, Any]] = []
    path = Path(file_path)

    if path.is_dir():
        paths = list(path.glob("*.jsonl"))
    else:
        paths = [path]

    for path in paths:
        if path.suffix != ".jsonl":
            continue

        with jsonlines.open(path, "r") as reader:
            for item in reader:
                company_name = item.get("company_name", "Unknown Company")
                question = f"Is this bill relevant to {company_name}?"
                formatted_item = {
                    "_id": item.get("id", ""),
                    "question": question,
                    "answer": (item.get("answer") or "").strip(),
                    "company_name": company_name,
                    "context": [("bill", [item.get("text", "")])],
                    "context_type": "legal_corporate_lobbying",
                }
                data.append(formatted_item)

    return data

def load_legalbench_scalr_data(file_path: str) -> List[Dict[str, Any]]:
    """加载 LegalBench 案例法数据"""
    data: List[Dict[str, Any]] = []
    path = Path(file_path)
    if path.is_dir():
        paths = list(path.glob("*.jsonl"))
    else:
        paths = [path]
    
    for path in paths:
        if path.suffix != ".jsonl":
            continue
        with jsonlines.open(path, "r") as reader:
            for item in reader:
                options = []
                i = 0
                while f"choice_{i}" in item:
                    opt_text = item.get(f"choice_{i}", "")
                    if opt_text:
                        options.append(opt_text)
                    i += 1
                
                options_str = "\n".join([f"({chr(65+i)}) {opt}" for i, opt in enumerate(options)])
                question = (item.get("question") or "").strip()
                if options_str:
                    question = f"{question}\n\nOptions:\n{options_str}"
                
                answer_idx = item.get("answer")
                if answer_idx is None:
                    answer_label = ""
                elif isinstance(answer_idx, int) and 0 <= answer_idx < len(options):
                    answer_label = chr(65 + answer_idx)
                else:
                    answer_label = str(answer_idx).strip()
                
                formatted_item = {
                    "_id": item.get("id", item.get("index", "")),
                    "question": question,
                    "answer": answer_label,
                    "answer_index": answer_idx,
                    "options": options,
                    "context": [("supreme_court_case", [item.get("text", "")])],
                    "context_type": "legal_case",
                }
                data.append(formatted_item)
    return data

def load_data(file_path: str, task: str = "hotpotqa", use_supporting_only: bool = False) -> List[Dict[str, Any]]:
    """通用数据加载接口"""
    
    if task == "legalbench":
        legalbench_tasks = [
            (load_contract_qa_data, "QA/contract_qa.jsonl", "contract"),
            (load_consumer_contracts_qa_data, "QA/consumer_contracts_qa.jsonl", "tos"),
            (load_privacy_policy_qa_data, "QA/privacy_policy_qa.jsonl", "privacy_policy"),
            (load_rule_qa_data, "QA/rule_qa.jsonl", "rules"),
            (load_privacy_policy_entailment_data, "privacy_policy_entailment.jsonl", "legal_privacy_policy_entailment"),
            (load_sara_entailment_data, "sara_entailment.jsonl", "legal_sora_entailment"),
            (load_legalbench_insurance_data, "insurance_policy_interpretation.jsonl", "legal_insurance"),
            (load_legalbench_corporate_lobbying_data, "corporate_lobbying.jsonl", "legal_corporate_lobbying"),
            (load_legalbench_scalr_data, "scalr.jsonl", "legal_case"),
        ]
        
        all_items = []
        base_path = Path(file_path)
        for loader, subpath, ctx_type in legalbench_tasks:
            task_path = base_path / subpath
            if task_path.exists():
                try:
                    sub_data = loader(str(task_path))
                    for item in sub_data:
                        item["context_type"] = ctx_type
                    all_items.extend(sub_data)
                    print(f"  ✓ Loaded {len(sub_data)} from {subpath}")
                except Exception as e:
                    print(f"  ⚠️ Failed to load {subpath}: {e}")
        
        if not all_items:
            raise ValueError(f"No LegalBench data found at {file_path}")
        
        print(f"✓ Total LegalBench samples: {len(all_items)}")
        return all_items
    
    elif task == "qa/contract":
        return load_contract_qa_data(file_path)
    elif task == "qa/consumer":
        return load_consumer_contracts_qa_data(file_path)
    elif task == "qa/privacy":
        return load_privacy_policy_qa_data(file_path)
    elif task == "qa/rule":
        return load_rule_qa_data(file_path)
    elif task.startswith("legalbench/qa"):
        return load_legalbench_qa_data(file_path)
    elif task.startswith("legalbench/sara_entailment"):
        return load_sara_entailment_data(file_path)
    elif task.startswith("legalbench/privacy_policy_entailment"):
        return load_privacy_policy_entailment_data(file_path)
    elif task.startswith("legalbench/insurance"):
        return load_legalbench_insurance_data(file_path)
    elif task.startswith("legalbench/corporate_lobbying"):
        return load_legalbench_corporate_lobbying_data(file_path)
    elif task.startswith("legalbench/scalr"):
        return load_legalbench_scalr_data(file_path)
    elif task == "hotpotqa":
        return load_hotpotqa_data(file_path)
    elif task == "musique":
        return load_musique_data(file_path, use_supporting_only)
    elif task == "multihop":
        return load_multihop_data(file_path)
    elif task == "ARC":
        return load_arc_data(file_path)
    else:
        raise ValueError(f"Unsupported task: {task}")
