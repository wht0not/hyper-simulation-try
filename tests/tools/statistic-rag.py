import os
import json

def parse_dir_params(dir_name):
    """解析目录名中的 top_k 和 chunk_size"""
    try:
        parts = dir_name.split('_')
        if len(parts) == 2:
            return int(parts[0]), int(parts[1])
    except (ValueError, IndexError):
        pass
    return None, None

def sum_prepared_elapsed_seconds(data, category):
    """统计指定 category 的 prepared_elapsed_seconds 总和"""
    total = 0.0
    for item in data.get("results", []):
        if item.get("category") != category:
            continue
        elapsed = item.get("prepared_elapsed_seconds")
        if isinstance(elapsed, (int, float)):
            total += float(elapsed)
    return round(total, 6)

def collect_data(base_dir, is_hyper):
    """从指定基础目录收集 category3 的数据"""
    results = []
    if not os.path.exists(base_dir):
        print(f"Warning: Directory not found: {base_dir}")
        return results

    for dir_name in os.listdir(base_dir):
        dir_path = os.path.join(base_dir, dir_name)
        if not os.path.isdir(dir_path):
            continue
            
        top_k, chunk_size = parse_dir_params(dir_name)
        if top_k is None:
            continue
            
        final_json_path = os.path.join(dir_path, "final.json")
        if not os.path.exists(final_json_path):
            continue
            
        with open(final_json_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                category3_data = data.get("summary", {}).get("by_category", {}).get("3", {})
                prepared_elapsed_seconds_sum = sum_prepared_elapsed_seconds(data, category=3)
                
                if category3_data:
                    record = {
                        "top_k": top_k,
                        "chunk_size": chunk_size,
                        "hyper": is_hyper,
                        "prepared_elapsed_seconds_sum": prepared_elapsed_seconds_sum,
                        **category3_data
                    }
                    results.append(record)
            except json.JSONDecodeError:
                print(f"Error decoding JSON from {final_json_path}")
    return results

def main():
    hyper_rag_dir = "/home/vincent/hyper-simulation-try/tests/tasks/memory/locomo/data/hyper_simulation/rag"
    normal_rag_dir = "/home/vincent/hyper-simulation-try/tests/tasks/memory/locomo/data/rag"
    output_file = "/home/vincent/hyper-simulation-try/tests/tools/category3_stats.jsonl"
    
    # 收集两部分数据
    all_results = []
    all_results.extend(collect_data(hyper_rag_dir, is_hyper=True))
    all_results.extend(collect_data(normal_rag_dir, is_hyper=False))
    
    # 按照 (top_k, chunk_size, hyper, 各项分数) 排序
    # 分数通常是越高越好，如果需要降序可以用 -x.get(...)
    all_results.sort(key=lambda x: (
        x["top_k"], 
        x["chunk_size"], 
        x["hyper"],
        x.get("cosine_similarity", 0),
        x.get("F1", 0),
        x.get("rouge_L", 0),
        x.get("BLEU", 0),
        x.get("LLM-as-judge_mean", 0)
    ))
    
    # 保存为 jsonl
    with open(output_file, 'w', encoding='utf-8') as f:
        for record in all_results:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
            
    print(f"Successfully saved {len(all_results)} records to {output_file}")

if __name__ == "__main__":
    main()
