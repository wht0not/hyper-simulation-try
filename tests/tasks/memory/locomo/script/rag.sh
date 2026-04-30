#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LOCOMO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd -- "$LOCOMO_ROOT/../../../.." && pwd)"
LOCOMO_DATA="$LOCOMO_ROOT/data"
RAG_SOURCE="$LOCOMO_ROOT/data/rag/locomo10_rag.json"

ANSWER_BATCH_SIZE="${ANSWER_BATCH_SIZE:-5}"
JUDGE_MAX_WORKERS="${JUDGE_MAX_WORKERS:-4}"
LLM_JUDGE_REPEAT="${LLM_JUDGE_REPEAT:-5}"
RAG_CHUNK_SIZE="${RAG_CHUNK_SIZE:-128}"
RAG_TOP_K="${RAG_TOP_K:-5}"

RAG_DATASET="$LOCOMO_DATA/rag/${RAG_TOP_K}_${RAG_CHUNK_SIZE}/locomo10_rag.json"
RAG_INSTANCES="$PROJECT_ROOT/data/hypergraphs/locomo-rag-${RAG_TOP_K}_${RAG_CHUNK_SIZE}"

echo "Preparing LoCoMo rag retrieval dataset..."
pixi run -e simulation python "$LOCOMO_ROOT/method/rag/retrieval.py" \
  --rag-source-path "$RAG_SOURCE" \
  --output-dir "$LOCOMO_DATA" \
  --chunk-size "$RAG_CHUNK_SIZE" \
  --top-k "$RAG_TOP_K"

# retrieval 之后，如需跑 rag -> context，可手动运行：
# pixi run -e simulation python "$LOCOMO_ROOT/run_experiments.py" \
#   --method context \
#   --stage all \
#   --dataset-path "$RAG_DATASET" \
#   --output-dir "$LOCOMO_DATA" \
#   --answer-batch-size "$ANSWER_BATCH_SIZE" \
#   --judge-max-workers "$JUDGE_MAX_WORKERS" \
#   --llm-judge-repeat "$LLM_JUDGE_REPEAT"
#
# 如需跑 rag -> hyper_simulation，先建图再跑：
# pixi run -e hypergraph python "$LOCOMO_ROOT/run_experiments.py" \
#   --method hyper_simulation \
#   --stage build \
#   --dataset-path "$RAG_DATASET" \
#   --instances-root "$RAG_INSTANCES"
#
# pixi run -e simulation python "$LOCOMO_ROOT/run_experiments.py" \
#   --method hyper_simulation \
#   --stage all \
#   --instances-root "$RAG_INSTANCES" \
#   --output-dir "$LOCOMO_DATA" \
#   --answer-batch-size "$ANSWER_BATCH_SIZE" \
#   --judge-max-workers "$JUDGE_MAX_WORKERS" \
#   --llm-judge-repeat "$LLM_JUDGE_REPEAT"
#
# 如需只做 retrieval，也可以直接修改：
#   RAG_CHUNK_SIZE
#   RAG_TOP_K
