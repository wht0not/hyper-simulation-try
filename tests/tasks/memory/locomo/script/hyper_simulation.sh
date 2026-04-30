#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LOCOMO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd -- "$LOCOMO_ROOT/../../../.." && pwd)"
LOCOMO_DATA="$LOCOMO_ROOT/data/hyper_simulation"
CONTEXT_DATASET="$LOCOMO_ROOT/data/context/locomo_context_raw.json"
INSTANCES_CONTEXT="$PROJECT_ROOT/data/hypergraphs/locomo_context"

ANSWER_BATCH_SIZE="${ANSWER_BATCH_SIZE:-5}"
JUDGE_MAX_WORKERS="${JUDGE_MAX_WORKERS:-4}"
LLM_JUDGE_REPEAT="${LLM_JUDGE_REPEAT:-5}"
HYPERGRAPH_BATCH_SIZE="${HYPERGRAPH_BATCH_SIZE:-128}"

echo "Running LoCoMo hyper-simulation pipeline..."
pixi run -e simulation python "$LOCOMO_ROOT/run_experiments.py" \
  --method hyper_simulation \
  --stage all \
  --instances-root "$INSTANCES_CONTEXT" \
  --output-dir "$LOCOMO_DATA" \
  --answer-batch-size "$ANSWER_BATCH_SIZE" \
  --judge-max-workers "$JUDGE_MAX_WORKERS" \
  --llm-judge-repeat "$LLM_JUDGE_REPEAT"

/home/vincent/hyper-simulation-try/tests/tasks/memory/locomo/script/context.sh
# 如需先建图，再跑 all，可手动运行：
# pixi run -e hypergraph python "$LOCOMO_ROOT/run_experiments.py" \
#   --method hyper_simulation \
#   --stage build \
#   --dataset-path "$CONTEXT_DATASET" \
#   --instances-root "$INSTANCES_CONTEXT" \
#   --batch-size "$HYPERGRAPH_BATCH_SIZE"
#
# pixi run -e simulation python "$LOCOMO_ROOT/run_experiments.py" \
#   --method hyper_simulation \
#   --stage all \
#   --instances-root "$INSTANCES_CONTEXT" \
#   --output-dir "$LOCOMO_DATA" \
#   --answer-batch-size "$ANSWER_BATCH_SIZE" \
#   --judge-max-workers "$JUDGE_MAX_WORKERS" \
#   --llm-judge-repeat "$LLM_JUDGE_REPEAT"
#
# 如需分阶段跑，也可以手动运行：
# pixi run -e simulation python "$LOCOMO_ROOT/run_experiments.py" \
#   --method hyper_simulation \
#   --stage compose \
#   --instances-root "$INSTANCES_CONTEXT" \
#   --output-dir "$LOCOMO_DATA"
#
# pixi run -e simulation python "$LOCOMO_ROOT/run_experiments.py" \
#   --method hyper_simulation \
#   --stage answer \
#   --prepared-path "$LOCOMO_DATA/locomo_hyper_simulation_context_prepared.json" \
#   --answer-batch-size "$ANSWER_BATCH_SIZE"
#
# pixi run -e simulation python "$LOCOMO_ROOT/run_experiments.py" \
#   --method hyper_simulation \
#   --stage evaluate \
#   --answers-path "$LOCOMO_DATA/locomo_hyper_simulation_context_answers.json" \
#   --judge-max-workers "$JUDGE_MAX_WORKERS" \
#   --llm-judge-repeat "$LLM_JUDGE_REPEAT"
