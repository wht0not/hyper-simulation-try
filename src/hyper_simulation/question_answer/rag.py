import torch
import argparse
import os
import json
import jsonlines
from types import SimpleNamespace
from typing import List, Dict, Union
import glob
import time
from pathlib import Path
# ==============================================================================
# 1. 澶嶇敤鐜版湁鐨勬ā鍧?(Reusing Existing Interfaces)
# ==============================================================================

# 澶嶇敤妫€绱㈡ā鍧楃殑宸ュ叿鍑芥暟
#
from hyper_simulation.question_answer.vmdit.retrieval import (
    embed_queries, 
    add_passages, 
    add_hasanswer,
    index_encoded_data
)
# 澶嶇敤 Contriever 搴曞眰鎺ュ彛
#
import contrievers
import contrievers.index
import contrievers.data

# 澶嶇敤 LLM 璋冪敤鎺ュ彛
#
from hyper_simulation.utils.chat_completion import get_generate
from langchain_ollama import ChatOllama

# 澶嶇敤鏁版嵁澶勭悊鍜?Prompt 妯℃澘
#
from hyper_simulation.question_answer.vmdit.utils import (
    PROMPT_DICT, 
    TASK_INST, 
    postprocess_answers_closed,
    preprocess_input_data
)

# ==============================================================================
# 2. RAG 妗嗘灦瀹炵幇 (RAG Framework Implementation)
# ==============================================================================

class RAGPipeline:
    def __init__(self, 
                 retriever_model_path: str = "models/contriever-msmarco",
                 passages_path: str = "data/psgs_w100.tsv",
                 index_path: str = "index_hnsw/",
                 embedding_dir: str = "data/wikipedia_embeddings",
                 llm_model_name: str = "qwen3.5:9b",
                 device: str = "cuda"):
        """
        鍒濆鍖?RAG 娴佹按绾匡紝鍔犺浇鎵€鏈夊繀瑕佺殑妯″瀷鍜岀储寮曘€?
        """
        self.device = device
        
        # --- 鍒濆鍖栨绱㈠櫒 (Retrieval Setup) ---
        print(f"Loading Retriever from {retriever_model_path}...")
        # 鐩存帴澶嶇敤 contrievers.load_retriever
        self.retriever_model, self.retriever_tokenizer, _ = contrievers.load_retriever(retriever_model_path)
        self.retriever_model.eval()
        self.retriever_model.to(device)
        if device == "cuda":
            self.retriever_model.half()

        # 鍔犺浇绱㈠紩 (Index)
        # 澶嶇敤 contrievers.index.Indexer
        print(f"Loading Index from {index_path}...")
        self.index = contrievers.index.Indexer(vector_sz=768, n_subquantizers=0, n_bits=8, mode='hnsw')
        # 鏇挎崲鍘熸湁鐨?self.index.deserialize_from(index_path) 鍙婂叾鍛ㄨ竟浠ｇ爜
        index_dir = index_path.rstrip('/')
        index_file = os.path.join(index_dir, "index.faiss")
        meta_file = os.path.join(index_dir, "index_meta.faiss")
        if os.path.exists(index_file):
            import faiss
            import pickle
            print(f"鈿?Loading 65GB index via Memory-Mapped I/O (MMAP) to bypass RAM limit...")
            faiss_idx = faiss.read_index(index_file, faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY)
            target_attr = "index" if hasattr(self.index, "index") else "faiss_index"
            setattr(self.index, target_attr, faiss_idx)
            
            if os.path.exists(meta_file):
                print(f"Loading meta data from {meta_file}")
                with open(meta_file, "rb") as reader:
                    self.index.index_id_to_db_id = pickle.load(reader)
            else:
                print(f"鈿狅笍 Warning: Meta data not found at {meta_file}")
                
            print(f"鉁?Index mapped successfully. Physical RAM usage stable.")            
        else:
            print(f"Index not found at {index_path}. Building from embeddings in {embedding_dir}...")
            
            # 鑾峰彇鎵€鏈?embedding 鏂囦欢 (.pkl)
            input_paths = glob.glob(os.path.join(embedding_dir, "passages_*")) 
            input_paths = sorted(input_paths)
            
            if not input_paths:
                 raise FileNotFoundError(f"No embedding files found in {embedding_dir}. Please run generate_passage_embedding.py first.")

            # 鏋勫缓绱㈠紩
            start_time = time.time()
            index_encoded_data(self.index, input_paths, indexing_batch_size=1000000) #
            print(f"Indexing finished in {time.time()-start_time:.1f} s.")
            
            # 淇濆瓨绱㈠紩浠ヤ究涓嬫浣跨敤
            os.makedirs(os.path.dirname(index_path), exist_ok=True)
            self.index.serialize(index_path)
            print(f"Index saved to {index_path}")

        # 鍔犺浇鏂囨。搴?(Passages)
        # 澶嶇敤 contrievers.data.load_passages
        print(f"Loading Passages from {passages_path}...")
        self.passages = contrievers.data.load_passages(passages_path)
        self.passage_id_map = {x["id"]: x for x in self.passages}

        # --- 鍒濆鍖栫敓鎴愬櫒 (Generation Setup) ---
        print(f"Loading LLM {llm_model_name}...")
        # 澶嶇敤 ChatOllama
        self.llm = ChatOllama(model=llm_model_name, temperature=0.8, top_p=0.95)

    def retrieve(self, queries: List[str], top_k: int = 5) -> List[List[Dict]]:
        """
        鎵ц妫€绱㈡楠ゃ€?
        瀹屽叏澶嶇敤 vmdit/retrieval.py 涓殑閫昏緫銆?
        """
        # 鏋勯€?args 瀵硅薄浠ラ€傞厤 embed_queries 鍑芥暟鐨勭鍚?
        #
        args = SimpleNamespace(
            lowercase=False, 
            normalize_text=True, 
            per_gpu_batch_size=32, 
            question_maxlength=512
        )

        print("Embedding queries...")
        # 澶嶇敤 embed_queries
        query_embeddings = embed_queries(args, queries, self.retriever_model, self.retriever_tokenizer)

        print("Searching index...")
        # 澶嶇敤 index.search_knn
        top_ids_and_scores = self.index.search_knn(query_embeddings, top_k)

        # 鏋勯€犱复鏃舵暟鎹粨鏋勪互鍒╃敤 add_passages 鍑芥暟
        dummy_data = [{"question": q} for q in queries]
        
        # 澶嶇敤 add_passages 灏嗘绱㈢粨鏋滄敞鍏ユ暟鎹?
        add_passages(dummy_data, self.passage_id_map, top_ids_and_scores)
        
        # 杩斿洖姣忎釜 query 瀵瑰簲鐨?ctxs 鍒楄〃
        return [item["ctxs"] for item in dummy_data]

    def generate(self, items: List[Dict], task: str = "qa", top_n: int = 5, save_prompts_only: bool = False, prompt_save_path: str = None) -> List[str]:
        """
        鎵ц鐢熸垚姝ラ銆?
        澶嶇敤 base_line_lm.py 鍜?vmdit/utils.py 鐨勯€昏緫銆?
        """
        
        # 1. 鍑嗗 Prompts
        prompts = []
        for item in items:
            # 杩欓噷鐨?item 搴旇宸茬粡鍖呭惈 'ctxs' (鐢?retrieve 姝ラ浜х敓)
            
            # A. 鎷兼帴妫€绱㈠埌鐨勬钀?(Context Construction)
            # 閫昏緫鏉ユ簮:
            retrieval_result = item.get("ctxs", [])[:top_n]
            evidences = [
                "[{}] {}\n{}".format(i+1, ctx["title"], ctx["text"]) 
                for i, ctx in enumerate(retrieval_result)
            ]
            paragraph = "\n".join(evidences)

            # B. 澶勭悊鎸囦护鍜岄€夐」 (Instruction Formatting)
            # 閫昏緫鏉ユ簮:
            # 鎴戜滑鎵嬪姩鏋勫缓 preprocess_input_data 鐨勬晥鏋?
            instruction_text = TASK_INST.get(task, item.get("question", ""))
            
            # 澶勭悊 ARC/澶氶€夐鐨勯€夐」鏍煎紡鍖?
            choices_str = ""
            if task in ["arc_c", "arc_easy", "obqa"] and "choices" in item:
                # 绠€鍖栫殑閫夐」鏍煎紡鍖栭€昏緫锛屽弬鑰?utils.py
                choices = item["choices"]
                labels = choices.get("label", [])
                texts = choices.get("text", [])
                formatted = []
                map_key = {"1": "A", "2": "B", "3": "C", "4": "D"}
                for l, t in zip(labels, texts):
                    k = map_key.get(l, l)
                    formatted.append(f"{k}: {t}")
                if formatted:
                    choices_str = "\n" + "\n".join(formatted)
            
            full_instruction = f"{instruction_text}\n\n### Input:\n{item['question']}{choices_str}"
            
            # C. 搴旂敤妯℃澘
            # 澶嶇敤 PROMPT_DICT
            prompt = PROMPT_DICT["prompt_no_input_retrieval"].format(
                paragraph=paragraph,
                instruction=full_instruction
            )
            prompts.append(prompt)

        # 馃敼 濡傛灉鍙渶瑕佷繚瀛?prompts (鍦ㄨ繖閲岋紝鎴戜滑淇濆瓨鐨勬槸鍘熷鏁版嵁鍔犱笂 ctxs)
        if save_prompts_only and prompt_save_path:
            prompts_buffer = []
            for item in items:
                # 閲嶆柊鏋勯€犵被浼煎師濮?jsonl 鐨勭粨鏋勶紝浣嗗姞涓婁簡 retrieved 鐨?ctxs
                # 涓轰簡閫傞厤鍚庣画 rag_no_retrival.py 鐨?load_data 鑳藉璇诲彇锛屾垜浠妸 ctxs 閲岀殑 text 鎻愬彇鍑烘潵浣滀负鏀寔浜嬪疄
                
                # 鎻愬彇 retrieved 鏂囨湰浣滀负鏀寔鏂囨。 (杩欓噷妯℃嫙 hotpotqa/musique 鐨勬暟鎹粨鏋?
                retrieval_result = item.get("ctxs", [])[:top_n]
                paragraphs = []
                for ctx in retrieval_result:
                    title = ctx.get("title", "")
                    text = ctx.get("text", "")
                    # 妯℃嫙娈佃惤缁撴瀯锛屽鏋滄槸鏀寔鏂囨。灏卞姞涓?
                    paragraphs.append({
                        "title": title,
                        "text": text,
                        "is_supporting": True  # 鍋囪鎵€鏈夋绱㈠埌鐨勯兘瑙嗕负鍙敤鐨勬敮鎸佹枃妗?
                    })
                
                # 鏋勫缓杈撳嚭瀛楀吀
                prompt_entry = {
                    "question": item.get("question", ""),
                    "answerKey": item.get("answers", []), # ARC 涓撶敤鐨勭瓟妗堝瓧娈?
                    "choices": item.get("choices", {}),   # ARC 涓撶敤鐨勯€夐」瀛楁
                    "paragraphs": paragraphs              # 鏂板鐨勬绱㈠嚭鏉ョ殑涓婁笅鏂?
                }
                prompts_buffer.append(prompt_entry)
            
            # 鎵归噺淇濆瓨
            Path(prompt_save_path).parent.mkdir(parents=True, exist_ok=True)
            with jsonlines.open(prompt_save_path, 'a') as writer:
                for entry in prompts_buffer:
                    writer.write(entry)
            print(f"馃捑 宸蹭繚瀛?{len(prompts_buffer)} 鏉″甫 Retrieval Context 鐨勬暟鎹埌 {prompt_save_path}")
            return [""] * len(items)  # 鍗犱綅杩斿洖

        # 2. 鎵归噺鐢熸垚
        print(f"Generating responses for {len(prompts)} prompts...")
        # 澶嶇敤 get_generate
        raw_responses = get_generate(prompts, self.llm)
        print(f"Raw responses is {raw_responses}")

        # 3. 鍚庡鐞?
        final_results = []
        for resp in raw_responses:
            # 鍩虹娓呮礂
            cleaned = resp.split("\n\n")[0].replace("</s>", "").strip()
            
            # 閽堝鐗瑰畾浠诲姟鐨勬彁鍙?
            # 澶嶇敤 postprocess_answers_closed
            choices_arg = "A B C D" if task in ["arc_c", "arc_easy"] else None
            final_out = postprocess_answers_closed(cleaned, task, choices=choices_arg)
            final_results.append(final_out)

        return final_results

    def run_batch(self, input_data: List[Dict], task: str = "qa", top_n: int = 5, save_prompts_only: bool = False, prompt_save_path: str = None):
        """
        绔埌绔繍琛岋細杈撳叆鏁版嵁 -> 妫€绱?-> 鐢熸垚
        """
        # 1. 鎻愬彇 Query
        queries = [item["question"] for item in input_data]
        
        # 2. 妫€绱?
        print("--- Start Retrieval ---")
        ctxs_list = self.retrieve(queries, top_k=top_n)
        
        # 3. 灏嗘绱㈢粨鏋滃悎骞跺洖 input_data
        for item, ctxs in zip(input_data, ctxs_list):
            item["ctxs"] = ctxs
            
        # 4. 鐢熸垚 (鎴栧彧淇濆瓨 Prompt)
        print("--- Start Generation ---")
        answers = self.generate(input_data, task=task, top_n=top_n, save_prompts_only=save_prompts_only, prompt_save_path=prompt_save_path)
                
        # 5. 缁撴灉鍚堝苟
        for item, ans in zip(input_data, answers):
            item["output"] = ans
            
        return input_data

# ==============================================================================
# 3. 浣跨敤绀轰緥 (Usage Example)
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=None, help="Path to ARC data")
    parser.add_argument('--save_prompts_only', action='store_true', help="Only retrieve and save prompts")
    parser.add_argument('--prompt_save_path', type=str, default="/home/vincent/hyper-simulation/data/mid_result/arc/arc_retrieved.jsonl")
    args = parser.parse_args()

    # 鍒濆鍖?pipeline (璇风‘淇濊矾寰勬寚鍚戜綘瀹為檯鐨勬ā鍨嬫枃浠?
    rag = RAGPipeline(
        retriever_model_path="models/contriever-msmarco", # 闇€鏇挎崲涓哄疄闄呰矾寰?
        passages_path="data/psgs_w100.tsv",             # 闇€鏇挎崲涓哄疄闄呰矾寰?
        index_path="../index_hnsw/"                     # 闇€鏇挎崲涓哄疄闄呰矾寰?
    )

    if args.data_path:
        # 濡傛灉浼犲叆浜嗙湡瀹炴暟鎹矾寰勶紝鍒欏姞杞芥暟鎹?
        from hyper_simulation.question_answer.utils.load_data import load_data
        
        # ARC 鏁版嵁闇€瑕佷互鐗瑰畾鏂瑰紡鍔犺浇锛岃繖閲屽€熺敤宸叉湁鐨?load_data
        raw_data = load_data(args.data_path, "ARC")
        print(f"Loaded {len(raw_data)} samples from {args.data_path}")
        
        # 灏嗗師濮嬫暟鎹浆鎹负 rag.py 闇€瑕佺殑鏍煎紡
        test_data = []
        for item in raw_data:
            test_data.append({
                "question": item.get("question", ""),
                "choices": item.get("choices", {}),
                "answers": item.get("answerKey", [])  # 浠呬綔鍙傝€冧繚鐣?
            })
            
        # 鎵归噺澶勭悊
        # 濡傛灉鏁版嵁閲忓緢澶э紝寤鸿澶栧眰鍐嶅涓€涓?batch 寰幆锛岃繖閲岀洿鎺ヨ窇
        batch_size = 100
        for i in range(0, len(test_data), batch_size):
            batch = test_data[i:i+batch_size]
            print(f"Processing batch {i} to {i+len(batch)}...")
            rag.run_batch(
                batch, 
                task="arc_c", 
                top_n=5, 
                save_prompts_only=args.save_prompts_only,
                prompt_save_path=args.prompt_save_path
            )
        
        print("Done!")
    else:
        # 榛樿鐨勬ā鎷熸暟鎹祴璇?
        test_data = [
            {
                "id": 1,
                "question": "what is the capital of China?",
            },
            {
                "id": 2,
                "question": "Which material conducts heat best?",
                "choices": {"text": ["Wood", "Copper", "Plastic", "Glass"], "label": ["A", "B", "C", "D"]}
            }
        ]

        # 杩愯 PopQA 椋庢牸浠诲姟
        results = rag.run_batch(test_data[:1], task="qa")
        print(f"QA Result: {results[0]['output']}")

        # 杩愯 ARC 椋庢牸浠诲姟
        results_arc = rag.run_batch(test_data[1:], task="arc_c")
        print(f"ARC Result: {results_arc[0]['output']}")
