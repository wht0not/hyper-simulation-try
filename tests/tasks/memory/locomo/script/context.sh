#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LOCOMO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
LOCOMO_DATA="$LOCOMO_ROOT/data"
CONTEXT_DATASET="$LOCOMO_ROOT/data/context/locomo_context_raw.json"

ANSWER_BATCH_SIZE="${ANSWER_BATCH_SIZE:-5}"
JUDGE_MAX_WORKERS="${JUDGE_MAX_WORKERS:-4}"
LLM_JUDGE_REPEAT="${LLM_JUDGE_REPEAT:-5}"

echo "Running LoCoMo context pipeline..."
pixi run -e simulation python "$LOCOMO_ROOT/run_experiments.py" \
  --method context \
  --stage all \
  --dataset-path "$CONTEXT_DATASET" \
  --output-dir "$LOCOMO_DATA" \
  --answer-batch-size "$ANSWER_BATCH_SIZE" \
  --judge-max-workers "$JUDGE_MAX_WORKERS" \
  --llm-judge-repeat "$LLM_JUDGE_REPEAT"

# 如需分阶段跑，可手动运行：
# pixi run -e simulation python "$LOCOMO_ROOT/run_experiments.py" \
#   --method context \
#   --stage compose \
#   --dataset-path "$CONTEXT_DATASET" \
#   --output-dir "$LOCOMO_DATA"
#
# pixi run -e simulation python "$LOCOMO_ROOT/run_experiments.py" \
#   --method context \
#   --stage answer \
#   --prepared-path "$LOCOMO_DATA/locomo_context_raw_prepared.json" \
#   --answer-batch-size "$ANSWER_BATCH_SIZE"
#
# pixi run -e simulation python "$LOCOMO_ROOT/run_experiments.py" \
#   --method context \
#   --stage evaluate \
#   --answers-path "$LOCOMO_DATA/locomo_context_raw_answers.json" \
#   --judge-max-workers "$JUDGE_MAX_WORKERS" \
#   --llm-judge-repeat "$LLM_JUDGE_REPEAT"
