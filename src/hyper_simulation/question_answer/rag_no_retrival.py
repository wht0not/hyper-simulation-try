import sentencepiece
import json
import re
import jsonlines
import argparse
from tqdm import tqdm
from typing import List, Dict, Any
from pathlib import Path
import time

from hyper_simulation.query_instance import QueryInstance, build_query_instance_for_task
from hyper_simulation.utils.log import current_task
from hyper_simulation.question_answer.utils.load_data import load_data
from hyper_simulation.question_answer.utils.build_prompt import build_prompt
from hyper_simulation.question_answer.utils.post_answer import postprocess_answer, evaluate_answer

def run_rag_evaluation(
    data_path: str,
    model_name: str = "qwen3.5:9b",
    output_path: str = "",
    batch_size: int = 5,
    temperature: float = 0.2,
    task: str = "hotpotqa",
    method: str = "vanilla",
    build: bool = True,
    rebuild: bool = False,
    using_support_only: bool = False,
    save_interval: int = 100,
    save_prompts_only: bool = False,
    load_prompts: str = None
):
    """
    杩愯 RAG 璇勪及浠诲姟 (鏀寔鏂偣缁紶鍜屽閲忎繚瀛?
    娉ㄦ剰锛氫娇鐢?'question' 浣滀负鍞竴鏍囪瘑绗﹁繘琛屽幓閲嶅拰缁紶銆?
    """
    print(f"Loading data from {data_path}...")
    data: List[Dict[str, Any]] = load_data(data_path, task, using_support_only)
    print(f"Loaded {len(data)} samples")
    
    # 馃敼 1. 鍔犺浇宸叉湁缁撴灉锛岃幏鍙栧凡澶勭悊鐨勯棶棰橀泦鍚?
    processed_questions = set()
    existing_results = []
    
    if output_path:
        out_file = Path(output_path) / f"{task}.json"
        if out_file.exists():
            try:
                with open(out_file, 'r', encoding='utf-8') as f:
                    old_data = json.load(f)
                existing_results = old_data.get("results", [])
                
                # 鎻愬彇宸插鐞嗙殑闂鏂囨湰
                for r in existing_results:
                    q = r.get("question")
                    if q:
                        processed_questions.add(q)
                
                print(f"发现已有结果文件，已加载 {len(processed_questions)} 条已完成记录。将从断点处继续。")
            except Exception as e:
                print(f"读取已有结果文件失败：{e}。将重新开始。")
        else:
            print("未找到已有结果文件，将从头开始运行。")
            
    # 馃敼 2. 鍔犺浇宸叉湁 Prompts锛堝鏋滄寚瀹氾級
    prompts_data = []
    if load_prompts:
        print(f"馃搨 浠?{load_prompts} 鍔犺浇 Prompts...")
        if load_prompts.endswith('.jsonl'):
            with jsonlines.open(load_prompts, 'r') as reader:
                prompts_data = list(reader)
        else:
            with open(load_prompts, 'r', encoding='utf-8') as f:
                prompts_data = json.load(f)
        print(f"鉁?宸插姞杞?{len(prompts_data)} 鏉?Prompts")
        # 杩囨护宸插畬鎴愮殑
        prompts_data = [p for p in prompts_data if p.get('question') not in processed_questions]
        print(f"馃摑 鍓╀綑 {len(prompts_data)} 鏉″緟澶勭悊 Prompts")
    
    # 濡傛灉鏈夊姞杞界殑 prompts锛岀洿鎺ヤ娇鐢?
    if load_prompts:
        items_to_process = prompts_data
    else:
        items_to_process = data
    
    # 濡傛灉鎵€鏈変换鍔￠兘宸插畬鎴?
    if len(items_to_process) - len(processed_questions) <= 0 and not load_prompts:
        # 褰撲娇鐢?load_prompts 鏃讹紝items_to_process 宸茬粡鏄繃婊ゅ悗鐨勪簡锛屽鏋滀负绌哄垯鐩存帴缁撴潫
        pass
    if not items_to_process:
        print("所有任务已完成！无需重新运行。")
        # 閲嶆柊璁＄畻涓€涓嬫渶缁堟寚鏍囧苟杩斿洖
        all_metrics_tmp = {"exact_match": [], "f1": [], "match": []}
        for r in existing_results:
            m = r.get("metrics", {})
            for k in all_metrics_tmp:
                if k in m:
                    all_metrics_tmp[k].append(m[k])
        avg_metrics_tmp = {k: sum(v)/len(v) if v else 0 for k, v in all_metrics_tmp.items()}
        return existing_results, avg_metrics_tmp

    # 馃敼 3. 鍒濆鍖?LLM锛堝鏋滈渶瑕侊級
    if not load_prompts and not save_prompts_only:
        print(f"Initializing LLM: {model_name}")
        if build or method != "hyper_simulation":
            from langchain_ollama import ChatOllama
            from hyper_simulation.utils.chat_completion import get_generate
            model = ChatOllama(model=model_name, temperature=temperature, top_p=0.95, reasoning=False, timeout=300)
    elif load_prompts:
        print(f"Initializing LLM for pre-loaded prompts: {model_name}")
        from langchain_ollama import ChatOllama
        from hyper_simulation.utils.chat_completion import get_generate
        model = ChatOllama(model=model_name, temperature=temperature, top_p=0.95, reasoning=False, timeout=300)
    
    # 馃敼 4. 鍒濆鍖栫粨鏋滃鍣ㄥ拰鎸囨爣
    results = list(existing_results) 
    all_metrics = {"exact_match": [], "f1": [], "match": []}
    
    # 鎭㈠宸叉湁缁撴灉鐨勬寚鏍囩粺璁?
    for r in existing_results:
        m = r.get("metrics", {})
        for k in all_metrics:
            if k in m:
                all_metrics[k].append(m[k])

    print(f"Starting evaluation with batch_size={batch_size}... (Skip {len(processed_questions)} done)")
    current_task.set(task)
    
    # 馃敼 5. Prompt 淇濆瓨璺緞
    prompt_save_path = None
    if save_prompts_only and output_path:
        prompt_save_path = Path(output_path) / f"{method}" / f"{task}.jsonl"
        prompt_save_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Prompts 将保存到：{prompt_save_path}")
    
    config = {
        "model_name": model_name,
        "data_path": data_path,
        "batch_size": batch_size,
        "temperature": temperature,
        "method": method,
        "task": task,
        "total_samples_original": len(data)
    }

    new_results_buffer = [] 
    prompts_buffer = []  # 鐢ㄤ簬鎵归噺淇濆瓨 prompts
    if load_prompts:
        pbar_initial = 0
        pbar_total = len(items_to_process)
    else:
        pbar_initial = len(processed_questions)
        pbar_total = len(data)
    # 杩涘害鏉″垵濮嬪寲锛氬彧鏄剧ず鍓╀綑浠诲姟鏁伴噺
    pbar = tqdm(total=pbar_total, desc=f"Processing {task}_{method}", position=0, leave=True, initial=pbar_initial)

    for batch_start in range(0, len(items_to_process), batch_size):
        batch = items_to_process[batch_start:(batch_start + batch_size)]
        filtered_batch = []
        for item in batch:
            q_text = item.get('question', '').strip() if isinstance(item, dict) else getattr(item, 'query', '')
            if q_text in processed_questions and not load_prompts:
                # 褰?load_prompts 鏃讹紝涔嬪墠宸茬粡杩囨护杩囦簡
                continue
            filtered_batch.append(item)
        
        if not filtered_batch:
            continue

        # 馃敼 6. 鏋勫缓 QueryInstance (濡傛灉娌℃湁鍔犺浇 prompts)
        if not load_prompts:
            query_instances = [build_query_instance_for_task(item, task) for item in filtered_batch]
            if not query_instances:
                pbar.update(len(filtered_batch))
                continue
    
            # Method 澶勭悊
            method_start = time.time()
            if method == "vanilla":
                fixed_query_instances = query_instances
            elif method == "contradoc":
                from hyper_simulation.baselines.contradoc import query_fixup
                fixed_query_instances = [query_fixup(qi, model=model) for qi in query_instances]
            elif method == "sparsecl":
                from hyper_simulation.baselines.sparseCL import query_fixup
                fixed_query_instances = [query_fixup(qi, alpha=1.5) for qi in query_instances]
            elif method == "sentli":
                from hyper_simulation.baselines.sentLI import query_fixup
                fixed_query_instances = [ query_fixup(qi) for qi in query_instances]
            elif method == "cdit":
                from hyper_simulation.baselines.CDIT import query_fixup
                fixed_query_instances = [query_fixup(qi, model=model) for qi in query_instances]
            elif method == "bsim":
                from hyper_simulation.baselines.BSIM import run_bsim_for_query
                fixed_query_instances = [run_bsim_for_query(qi, task=task) for qi in query_instances]
            elif method == "hyper_simulation":
                if not build:
                    from hyper_simulation.component.build_hypergraph import build_hypergraph_batch_gpu
                    build_hypergraph_batch_gpu(query_instances, dataset_name=task, force_rebuild=rebuild, batch_size=128)
                    print("Hypergraph built. Please re-run with --build to evaluate.")
                    pbar.update(len(filtered_batch))
                    continue
                else:
                    from hyper_simulation.component.consistent import query_fixup
                    fixed_query_instances = [query_fixup(qi, task) for qi in query_instances]
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            method_time = time.time() - method_start
            
            # 鍑嗗 Prompts
            prompts = []
            for item in fixed_query_instances:
                context_text = "\n\n".join(item.fixed_data if item.fixed_data else item.data)
                context_type = getattr(item, 'context_type', None)
                score_guide = ""
                if method == "sparsecl":
                    score_guide = ( "Note: Documents prefixed with [SparseCL: X.XX] include contradiction scores.\nHigher scores indicate higher potential contradiction with the question. \nUse this as a reference when evaluating evidence reliability.\n\n")
                if method == "sentli":
                    score_guide = ( "Note: Documents are prefixed with [SENTLI: label] indicating logical relationship.\n[SENTLI: e] = Entailment (Supported/Correct)\n[SENTLI: c] = Contradiction (Incorrect)\n[SENTLI: n] = Neutral (Not mentioned)\nUse this label to help determine the final answer.\n\n")
                context_text = score_guide + context_text
                prompt = build_prompt(item.query, context_text, task=task, context_type=context_type)
                prompts.append(prompt)
        else:
            # 馃敼 7. 鐩存帴浠庡姞杞界殑 prompts 涓仮澶?
            fixed_query_instances = []
            prompts = []
            for item in filtered_batch:
                qi = QueryInstance(
                    query=item.get('question', ''),
                    data=[],
                    fixed_data=[],
                    answers=item.get('reference_answer', []),
                    ground_truth=[]
                )
                fixed_query_instances.append(qi)
                prompts.append(item.get('prompt', ''))
            method_time = 0
            
        # 馃敼 8. 淇濆瓨 Prompts锛堝鏋滃彧淇濆瓨 prompts锛?
        if save_prompts_only:
            for qi, prompt in zip(fixed_query_instances, prompts):
                prompt_entry = {
                    "question": qi.query,
                    "prompt": prompt,
                    "reference_answer": qi.answers,
                    "context_type": getattr(qi, 'context_type', None),
                    "method": method,
                    "task": task,
                }
                prompts_buffer.append(prompt_entry)
            
            # 鎵归噺淇濆瓨
            if len(prompts_buffer) >= save_interval:
                with jsonlines.open(prompt_save_path, 'a') as writer:  # 鉁?浣跨敤 jsonlines.open
                    for entry in prompts_buffer:
                        writer.write(entry)  # 鉁?浣跨敤 Writer 瀵硅薄鐨?write 鏂规硶
                print(f"馃捑 宸蹭繚瀛?{len(prompts_buffer)} 鏉?Prompts 鍒?{prompt_save_path}")
                prompts_buffer = []
            
            pbar.update(len(filtered_batch))
            continue  # 璺宠繃 LLM 璋冪敤

        # 馃敼 9. LLM 鐢熸垚
        gen_start = time.time()
        predictions = get_generate(prompts, model)
        gen_time = time.time() - gen_start
        
        n_samples = len(fixed_query_instances)
        batch_gen_time_per_item = gen_time / n_samples
        batch_method_time_per_item = method_time / n_samples
        
        # 鍚庡鐞嗗拰璇勪及
        for item, pred in zip(fixed_query_instances, predictions):
            processed_pred, parse_status, is_fallback = postprocess_answer(pred)
            # import sys
            # tqdm.write(f"[{pbar.n}] {processed_pred}", file=sys.stdout)
            metrics = evaluate_answer(processed_pred, item.answers)
            
            pbar.update(1)  # 琛ュ洖杩欒锛岀‘淇濆鐞嗚繃鐨勬瘡涓潯鐩兘鑳芥帹鍔ㄨ繘搴︽潯鍓嶈繘
            
            is_correct = metrics['exact_match'] > 0
            result = {
                "question": item.query,
                "prediction": processed_pred,
                "raw_prediction": pred,
                "reference_answer": item.answers,
                "is_correct": is_correct,                         
                "metrics": metrics,                               
                "status": "parsed_fallback" if is_fallback else "success",
                "parse_status": parse_status,                     
                "timing": {
                    "method_processing": batch_method_time_per_item,
                    "generation": batch_gen_time_per_item,
                    "total": batch_method_time_per_item + batch_gen_time_per_item
                },
                "context_type": getattr(item, 'context_type', None),
                "parse_fallback_used": is_fallback,
            }
            
            results.append(result)
            new_results_buffer.append(result)
            
            # 鏍囪涓哄凡澶勭悊
            processed_questions.add(item.query)
            
            # 绱Н鎸囨爣
            for metric_name, score in metrics.items():
                all_metrics[metric_name].append(score)
        
        # 馃敼 10. 妫€鏌ユ槸鍚﹂渶瑕佸閲忎繚瀛樼粨鏋?
        if len(new_results_buffer) >= save_interval:
            # 鍚堝苟鏃х粨鏋滃拰鏂扮紦鍐插尯缁撴灉
            full_results_to_save = existing_results + new_results_buffer
            
            avg_metrics_curr = {
                metric_name: sum(scores) / len(scores) if scores else 0
                for metric_name, scores in all_metrics.items()
            }
            
            output_data = {
                "config": config,
                "avg_metrics": avg_metrics_curr,
                "total_processed": len(full_results_to_save),
                "results": full_results_to_save
            }
            
            if output_path:
                out_file = Path(output_path) / f"{task}.json"
                out_file.parent.mkdir(parents=True, exist_ok=True)
                temp_file = out_file.with_suffix('.tmp')
                
                with open(temp_file, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, indent=2, ensure_ascii=False)
                temp_file.replace(out_file)
                
                print(f"已增量保存 {len(new_results_buffer)} 条新记录 (总计：{len(full_results_to_save)})")
            
            # 鏇存柊 existing_results 浠ヤ究涓嬫澧為噺淇濆瓨鍖呭惈涔嬪墠鐨勬暟鎹?
            existing_results.extend(new_results_buffer)
            new_results_buffer = []

    pbar.close()
    
    # 馃敼 11. 淇濆瓨鍓╀綑 Prompts
    if save_prompts_only and prompts_buffer:
        # 濡傛灉鏄?load_prompts锛岃鏄庤繖鍘熸湰鏄竴涓粠 .jsonl 鍔犺浇鐨勪换鍔★紙濡?sentli锛?
        # 姝ゆ椂 prompt_save_path 鏄?data/mid_result/{method}/{task}.jsonl
        # 鎴戜滑纭繚鐢?jsonlines 杩藉姞鍐欏叆
        if prompt_save_path:
            with jsonlines.open(prompt_save_path, 'a') as writer:
                for entry in prompts_buffer:
                    writer.write(entry)
            print(f"馃捑 宸蹭繚瀛樺墿浣?{len(prompts_buffer)} 鏉?Prompts 鍒?{prompt_save_path}")
        else:
            print("提示词缓存非空，但未设置 prompt_save_path！")
    
    # 馃敼 12. 淇濆瓨鍓╀綑缁撴灉
    if new_results_buffer:
        full_results_to_save = existing_results + new_results_buffer
        avg_metrics_curr = {
            metric_name: sum(scores) / len(scores) if scores else 0
            for metric_name, scores in all_metrics.items()
        }
        output_data = {
            "config": config,
            "avg_metrics": avg_metrics_curr,
            "total_processed": len(full_results_to_save),
            "results": full_results_to_save
        }
        
        if output_path:
            out_file = Path(output_path) / f"{task}.json"
            out_file.parent.mkdir(parents=True, exist_ok=True)
            temp_file = out_file.with_suffix('.tmp')
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            temp_file.replace(out_file)
            print(f"已保存剩余 {len(new_results_buffer)} 条记录 (总计：{len(full_results_to_save)})")

    # 璁＄畻鏈€缁堟寚鏍?
    avg_metrics = {
        metric_name: sum(scores) / len(scores) if scores else 0
        for metric_name, scores in all_metrics.items()
    }
    
    if not save_prompts_only:
        print("\n" + "="*60)
        print("Evaluation Finished!")
        print(f"Total samples processed this run: {len(results) - len(existing_results) + len(new_results_buffer)}")
        print(f"Exact Match: {avg_metrics['exact_match']:.4f}")
        print(f"F1 Score: {avg_metrics['f1']:.4f}")
        print(f"Match Score: {avg_metrics['match']:.4f}")
        print("="*60)
    
    return results, avg_metrics

def main():
    parser = argparse.ArgumentParser(description="RAG Evaluation on HotpotQA without Retrieval")
    
    parser.add_argument(
        '--data_path',
        type=str,
        required=True,
        help='Path to HotpotQA data file (json or jsonl)'
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default='qwen3.5:9b',
        help='LLM model name for Ollama'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default=None,
        help='Path to save evaluation results'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=100,
        help='Batch size for LLM inference'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.1,
        help='LLM temperature parameter'
    )
    
    parser.add_argument(
        '--method',
        type=str,
        default='vanilla',
        choices=['vanilla', 'contradoc', 'sparsecl', 'sentli', 'cdit', 'bsim', 'hyper_simulation'],
        help='Method to use: vanilla, contradoc, sparsecl, sentli, cdit, bsim, hyper_simulation'
    )
    
    parser.add_argument(
        '--task',
        type=str,
        default='hotpotqa',
        choices = ['hotpotqa','musique', 'multihop', 'ARC', 'legalbench', 'locomo'],
        help='Task type (default: hotpotqa)'
    )
    
    parser.add_argument(
        '--build',
        action='store_true',
        help='Whether to build hypergraph (default: False). Set to True to build hypergraph before evaluation.'
    )
    
    parser.add_argument(
        '--rebuild',
        action='store_true',
        help='Whether to rebuild hypergraph (default: False). Set to True to rebuild hypergraph before evaluation.'
    )

    parser.add_argument(
        '--using_support_only',
        action='store_true',
        help='Whether to use supporting paragraphs only (default: False). Set to True to use only supporting paragraphs.'
    )
    parser.add_argument(
        '--save_interval',
        type=int,
        default=100,
        help='Save results every N samples (default: 100)'
    )
    
    parser.add_argument(
        '--save_prompts_only',
        action='store_true',
        help='If set, only generate and save prompts without running LLM evaluation.'
    )
    
    parser.add_argument(
        '--load_prompts',
        type=str,
        default=None,
        help='Path to a saved prompts file (jsonl or json) to load and run LLM directly.'
    )

    args = parser.parse_args()
    build_flag = args.build == False
    rebuild_flag = args.rebuild == True
    using_support_only_flag = args.using_support_only == True

    # 杩愯璇勪及
    run_rag_evaluation(
        data_path=args.data_path,
        model_name=args.model_name,
        output_path=args.output_path,
        batch_size=args.batch_size,
        temperature=args.temperature,
        method=args.method,
        task=args.task,
        build=build_flag,
        rebuild=rebuild_flag,
        using_support_only=using_support_only_flag,
        save_interval=args.save_interval,
        save_prompts_only=args.save_prompts_only,
        load_prompts=args.load_prompts
    )


if __name__ == "__main__":
    main()

# 1. 鍏堝湪excel涓婅繘琛岃褰曡〃
# > 灏忔ā鍨
