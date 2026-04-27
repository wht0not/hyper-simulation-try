import sys
from pathlib import Path
sys.path.insert(0, str(Path("/home/vincent/hyper-simulation-try/data/bench/locomo-main")))

import os, json
import argparse
import random
from tqdm import tqdm
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

from task_eval.evaluation import eval_question_answering
from task_eval.evaluation_stats import analyze_aggr_acc

MAX_LENGTH = 32000
ANS_TOKENS_PER_QUES = 50

QA_PROMPT = """
Based on the above conversations, write a short answer for the following question in a few words. Do not write complete and lengthy sentences. Answer with exact words from the conversations whenever possible.

Question: {}
"""

CONV_START_PROMPT = "Below is a conversation between two people: {} and {}. The conversation takes place over multiple days and the date of each conversation is wriiten at the beginning of the conversation.\n\n"

def get_input_context(data, encoding, max_len):
    speakers_names = list(set([d['speaker'] for d in data['session_1']]))
    start_prompt = CONV_START_PROMPT.format(speakers_names[0], speakers_names[1])
    start_tokens = len(encoding.encode(start_prompt))

    query_conv = ''
    total_tokens = 0
    stop = False
    session_nums = [int(k.split('_')[-1]) for k in data.keys() if 'session' in k and 'date_time' not in k]
    
    for i in range(min(session_nums), max(session_nums) + 1):
        if f'session_{i}' in data:
            for dialog in data[f'session_{i}'][::-1]:
                turn = dialog['speaker'] + ' said, \"' + dialog['text'] + '\"' + '\n'
                if "blip_caption" in dialog:
                    turn += ' and shared %s.' % dialog["blip_caption"]
                turn += '\n'

                new_tokens = len(encoding.encode(f"DATE: {data[f'session_{i}_date_time']}\nCONVERSATION:\n{turn}"))
                if (start_tokens + new_tokens + total_tokens) < max_len:
                    query_conv = turn + query_conv
                    total_tokens += len(encoding.encode(turn))
                else:
                    stop = True
                    break

            query_conv = f"\nDATE: {data[f'session_{i}_date_time']}\nCONVERSATION:\n" + query_conv
        if stop:
            break
    
    return start_prompt + query_conv

def get_qwen_answers(in_data, out_data, args, model, tokenizer):
    for batch_start_idx in tqdm(range(0, len(in_data['qa']))):
        qa = in_data['qa'][batch_start_idx]
        if f"{args.model_name}_prediction" in qa and not args.overwrite:
            print("Skipping -->", qa['question'])
            continue

        question = qa['question']
        cat_5 = False
        cat_5_answer = {}

        if qa['category'] == 2:
            question += ' Use DATE of CONVERSATION to answer with an approximate date.'
        elif qa['category'] == 5:
            cat_5 = True
            question += " (a) {} (b) {}. Select the correct answer by writing (a) or (b)."
            if random.random() < 0.5:
                question = question.format('No information available', qa['answer'])
                cat_5_answer = {'a': 'No information available', 'b': qa['answer']}
            else:
                question = question.format(qa['answer'], 'No information available')
                cat_5_answer = {'b': 'No information available', 'a': qa['answer']}

        # Get context
        question_prompt = QA_PROMPT.format(question)
        question_tokens = len(tokenizer.encode(question_prompt))
        max_context_len = MAX_LENGTH - question_tokens - ANS_TOKENS_PER_QUES
        
        query_conv = get_input_context(in_data['conversation'], tokenizer, max_context_len)
        full_query = query_conv + '\n\n' + question_prompt

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": full_query}
        ]
        
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=ANS_TOKENS_PER_QUES,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            answer = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # Post-process answer
        answer = answer.replace('\\"', "'").strip()
        answer = [w.strip() for w in answer.split('\n') if not w.strip().isspace()][0] if answer else ""
        
        if cat_5:
            answer = answer.lower().strip()
            if '(a)' in answer:
                answer = cat_5_answer['a']
            else:
                answer = cat_5_answer['b']
        else:
            answer = answer.lower().replace('(a)', '').replace('(b)', '').replace('a)', '').replace('b)', '').replace('answer:', '').strip()
            
        out_data['qa'][batch_start_idx][f"{args.model_name}_prediction"] = answer

    return out_data

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-file', type=str, required=True, help="Path to Locomo data file")
    parser.add_argument('--out-file', type=str, required=True, help="Path to save predictions")
    parser.add_argument('--model-path', type=str, required=True, help="Path to local Qwen3.5-9B model")
    parser.add_argument('--model-name', type=str, default="qwen3.5-9b", help="Name to use for prediction keys")
    parser.add_argument('--overwrite', action="store_true", help="Overwrite existing predictions")
    return parser.parse_args()

def main():
    args = parse_args()
    print(f"****************** Evaluating Model {args.model_name} ***************")

    print(f"Loading tokenizer and model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    model.eval()

    samples = json.load(open(args.data_file))
    prediction_key = f"{args.model_name}_prediction"
    model_key = args.model_name

    if os.path.exists(args.out_file):
        out_samples = {d['sample_id']: d for d in json.load(open(args.out_file))}
    else:
        out_samples = {}

    for data in samples:
        out_data = {'sample_id': data['sample_id']}
        if data['sample_id'] in out_samples:
            out_data['qa'] = out_samples[data['sample_id']]['qa'].copy()
        else:
            out_data['qa'] = data['qa'].copy()

        print(f"Processing sample: {data['sample_id']}")
        answers = get_qwen_answers(data, out_data, args, model, tokenizer)

        exact_matches, lengths, recall = eval_question_answering(answers['qa'], prediction_key)
        for i in range(0, len(answers['qa'])):
            answers['qa'][i][model_key + '_f1'] = round(exact_matches[i], 3)

        out_samples[data['sample_id']] = answers

    with open(args.out_file, 'w') as f:
        json.dump(list(out_samples.values()), f, indent=2)

    stats_file = args.out_file.replace('.json', '_stats.json')
    analyze_aggr_acc(args.data_file, args.out_file, stats_file, model_key, model_key + '_f1', rag=False)
    print(f"Evaluation complete. Results saved to {args.out_file} and stats to {stats_file}")

if __name__ == "__main__":
    main()
