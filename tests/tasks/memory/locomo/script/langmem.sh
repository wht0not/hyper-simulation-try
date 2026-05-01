#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LOCOMO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
LOCOMO_DATA="$LOCOMO_ROOT/data/langmem"
LANGMEM_DATASET="$LOCOMO_ROOT/data/langmem/locomo10_rag.json"

MODEL_NAME="${MODEL_NAME:-qwen3.5:9b}"
ANSWER_BATCH_SIZE="${ANSWER_BATCH_SIZE:-5}"
JUDGE_MAX_WORKERS="${JUDGE_MAX_WORKERS:-4}"
LLM_JUDGE_REPEAT="${LLM_JUDGE_REPEAT:-5}"

echo "Running LoCoMo langmem pipeline..."
pixi run -e simulation python "$LOCOMO_ROOT/run_experiments.py" \
  --method langmem \
  --stage all \
  --dataset-path "$LANGMEM_DATASET" \
  --output-dir "$LOCOMO_DATA" \
  --model-name "$MODEL_NAME" \
  --answer-batch-size "$ANSWER_BATCH_SIZE" \
  --judge-max-workers "$JUDGE_MAX_WORKERS" \
  --llm-judge-repeat "$LLM_JUDGE_REPEAT"

# 如需分阶段跑，可手动运行：
# pixi run -e simulation python "$LOCOMO_ROOT/run_experiments.py" \
#   --method langmem \
#   --stage retrieve \
#   --dataset-path "$LANGMEM_DATASET" \
#   --output-dir "$LOCOMO_DATA" \
#   --model-name "$MODEL_NAME"
#
# pixi run -e simulation python "$LOCOMO_ROOT/run_experiments.py" \
#   --method langmem \
#   --stage compose \
#   --dataset-path "$LOCOMO_DATA/retrieved.json" \
#   --output-dir "$LOCOMO_DATA" \
#   --model-name "$MODEL_NAME"
#
# pixi run -e simulation python "$LOCOMO_ROOT/run_experiments.py" \
#   --method langmem \
#   --stage answer \
#   --prepared-path "$LOCOMO_DATA/prepared.json" \
#   --model-name "$MODEL_NAME" \
#   --answer-batch-size "$ANSWER_BATCH_SIZE"
#
# pixi run -e simulation python "$LOCOMO_ROOT/run_experiments.py" \
#   --method langmem \
#   --stage evaluate \
#   --answers-path "$LOCOMO_DATA/answers.json" \
#   --model-name "$MODEL_NAME" \
#   --judge-max-workers "$JUDGE_MAX_WORKERS" \
#   --llm-judge-repeat "$LLM_JUDGE_REPEAT"
