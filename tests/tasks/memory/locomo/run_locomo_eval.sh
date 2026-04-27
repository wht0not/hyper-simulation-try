#!/bin/bash

# 1. 确保已生成合并后的数据集 locomo_merged.json
python /home/vincent/hyper-simulation-try/data/bench/locomo-main/data/merge_locomo_data.py

# 2. 运行评估脚本 (使用 vanilla 方法，您可以修改为 hyper_simulation 或其他 baseline)
export PYTHONPATH=/home/vincent/hyper-simulation-try/src:$PYTHONPATH
pixi run -e simulation /home/vincent/hyper-simulation-try/src/hyper_simulation/question_answer/rag_no_retrival.py \
    --data_path /home/vincent/hyper-simulation-try/data/bench/locomo-main/data/locomo_merged.json \
    --task locomo \
    --model_name qwen3.5:9b \
    --method vanilla \
    --output_path /home/vincent/hyper-simulation-try/tests/tasks/memory/locomo/results \
    --batch_size 1

