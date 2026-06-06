#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LOCOMO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd -- "$LOCOMO_ROOT/../../../.." && pwd)"

DATA_ROOT="$LOCOMO_ROOT/data"
UTILITY_ROOT="$DATA_ROOT/data-utility"
HYPERGRAPH_ROOT="$PROJECT_ROOT/data/hypergraphs/locomo/data-utility"

RETRIEVAL_SOURCE_DATASET="$LOCOMO_ROOT/data/locomo10_rag.json"
RAG_RETRIEVED_DATASET="$UTILITY_ROOT/rag/locomo10_rag.json"

MODEL_NAME="${MODEL_NAME:-qwen3.5:9b}"
ANSWER_BATCH_SIZE="${ANSWER_BATCH_SIZE:-5}"
JUDGE_MAX_WORKERS="${JUDGE_MAX_WORKERS:-4}"
LLM_JUDGE_REPEAT="${LLM_JUDGE_REPEAT:-1}"
HYPERGRAPH_BATCH_SIZE="${HYPERGRAPH_BATCH_SIZE:-128}"
LIMIT="${LIMIT:-0}"
FORCE_REBUILD="${FORCE_REBUILD:-0}"
DRY_RUN="${DRY_RUN:-0}"

OPEN_DOMAIN_CATEGORY="${OPEN_DOMAIN_CATEGORY:-3}"
OPEN_DOMAIN_MAX_ROWS="${OPEN_DOMAIN_MAX_ROWS:-50}"
AMEM_RETRIEVE_K="${AMEM_RETRIEVE_K:-10}"
MEMORYBANK_RETRIEVE_K="${MEMORYBANK_RETRIEVE_K:-5}"

run_cmd() {
  echo "$" "$@" >&2
  if [[ "$DRY_RUN" == "1" ]]; then
    return 0
  fi
  "$@"
}

run_simulation_pipeline() {
  run_cmd env \
    LOCOMO_ALLOWED_CATEGORIES="$OPEN_DOMAIN_CATEGORY" \
    LOCOMO_MAX_ROWS="$OPEN_DOMAIN_MAX_ROWS" \
    pixi run -e simulation python "$LOCOMO_ROOT/run_experiments.py" "$@"
}

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "[ERROR] required file not found: $path" >&2
    exit 1
  fi
}

run_prepare_then_answer_evaluate() {
  local method="$1"
  local output_dir="$2"
  local dataset_path="$3"
  shift 3

  mkdir -p "$output_dir"

  run_simulation_pipeline \
    --method "$method" \
    --stage compose \
    --dataset-path "$dataset_path" \
    --output-dir "$output_dir" \
    --model-name "$MODEL_NAME" \
    --limit "$LIMIT" \
    "$@"

  run_simulation_pipeline \
    --method "$method" \
    --stage answer \
    --dataset-path "$dataset_path" \
    --prepared-path "$output_dir/prepared.json" \
    --output-dir "$output_dir" \
    --model-name "$MODEL_NAME" \
    --answer-batch-size "$ANSWER_BATCH_SIZE" \
    --limit "$LIMIT"

  run_simulation_pipeline \
    --method "$method" \
    --stage evaluate \
    --dataset-path "$dataset_path" \
    --answers-path "$output_dir/answers.json" \
    --output-dir "$output_dir" \
    --model-name "$MODEL_NAME" \
    --judge-max-workers "$JUDGE_MAX_WORKERS" \
    --llm-judge-repeat "$LLM_JUDGE_REPEAT" \
    --limit "$LIMIT"
}

run_hypersim_open_domain_qa() {
  local tag="$1"
  local dataset_path="$2"
  local output_dir="$3"
  local instances_root="$HYPERGRAPH_ROOT/$tag"
  local build_flag_args=()

  if [[ "$FORCE_REBUILD" == "1" ]]; then
    build_flag_args+=(--force-rebuild)
  fi

  mkdir -p "$output_dir"
  run_cmd env \
    HYPERSIM_ALLOWED_CATEGORIES="$OPEN_DOMAIN_CATEGORY" \
    LOCOMO_MAX_ROWS="$OPEN_DOMAIN_MAX_ROWS" \
    pixi run -e hypergraph python "$LOCOMO_ROOT/run_experiments.py" \
    --method hyper_simulation \
    --stage build \
    --dataset-path "$dataset_path" \
    --instances-root "$instances_root" \
    --batch-size "$HYPERGRAPH_BATCH_SIZE" \
    --limit "$LIMIT" \
    "${build_flag_args[@]}"

  run_cmd env \
    HYPERSIM_ALLOWED_CATEGORIES="$OPEN_DOMAIN_CATEGORY" \
    LOCOMO_MAX_ROWS="$OPEN_DOMAIN_MAX_ROWS" \
    pixi run -e simulation python "$LOCOMO_ROOT/run_experiments.py" \
    --method hyper_simulation \
    --stage compose \
    --instances-root "$instances_root" \
    --output-dir "$output_dir" \
    --model-name "$MODEL_NAME" \
    --limit "$LIMIT"

  run_cmd env \
    HYPERSIM_ALLOWED_CATEGORIES="$OPEN_DOMAIN_CATEGORY" \
    LOCOMO_MAX_ROWS="$OPEN_DOMAIN_MAX_ROWS" \
    pixi run -e simulation python "$LOCOMO_ROOT/run_experiments.py" \
    --method hyper_simulation \
    --stage answer \
    --instances-root "$instances_root" \
    --prepared-path "$output_dir/prepared.json" \
    --output-dir "$output_dir" \
    --model-name "$MODEL_NAME" \
    --answer-batch-size "$ANSWER_BATCH_SIZE" \
    --limit "$LIMIT"

  run_cmd env \
    HYPERSIM_ALLOWED_CATEGORIES="$OPEN_DOMAIN_CATEGORY" \
    LOCOMO_MAX_ROWS="$OPEN_DOMAIN_MAX_ROWS" \
    pixi run -e simulation python "$LOCOMO_ROOT/run_experiments.py" \
    --method hyper_simulation \
    --stage evaluate \
    --instances-root "$instances_root" \
    --answers-path "$output_dir/answers.json" \
    --output-dir "$output_dir" \
    --model-name "$MODEL_NAME" \
    --judge-max-workers "$JUDGE_MAX_WORKERS" \
    --llm-judge-repeat "$LLM_JUDGE_REPEAT" \
    --limit "$LIMIT"
}

echo "MODEL_NAME=$MODEL_NAME LIMIT=$LIMIT DRY_RUN=$DRY_RUN"
echo "OPEN_DOMAIN_CATEGORY=$OPEN_DOMAIN_CATEGORY OPEN_DOMAIN_MAX_ROWS=$OPEN_DOMAIN_MAX_ROWS ANSWER_BATCH_SIZE=$ANSWER_BATCH_SIZE"
echo "HYPERGRAPH_BATCH_SIZE=$HYPERGRAPH_BATCH_SIZE"
echo "Outputs will be saved under: $UTILITY_ROOT"

mkdir -p "$UTILITY_ROOT"

memorybank_output="$UTILITY_ROOT/memorybank"
amem_output="$UTILITY_ROOT/amem"
rag_output="$UTILITY_ROOT/rag"
hyper_memorybank_output="$UTILITY_ROOT/hyper_memorybank"
hyper_amem_output="$UTILITY_ROOT/hyper_amem"
hyper_rag_output="$UTILITY_ROOT/hyper_rag"

require_file "$RETRIEVAL_SOURCE_DATASET"
require_file "$RAG_RETRIEVED_DATASET"

echo "=== [1] memorybank open-domain qa ==="
run_prepare_then_answer_evaluate memorybank "$memorybank_output" "$RETRIEVAL_SOURCE_DATASET" --memorybank-retrieve-k "$MEMORYBANK_RETRIEVE_K"

echo "=== [2] amem open-domain qa ==="
run_prepare_then_answer_evaluate amem "$amem_output" "$RETRIEVAL_SOURCE_DATASET" --amem-retrieve-k "$AMEM_RETRIEVE_K"

echo "=== [3] rag 6_128 open-domain qa ==="
run_prepare_then_answer_evaluate context "$rag_output" "$RAG_RETRIEVED_DATASET"

echo "=== [4] hyper_memorybank open-domain qa ==="
run_hypersim_open_domain_qa memorybank "$memorybank_output/retrieved.json" "$hyper_memorybank_output"

echo "=== [5] hyper_amem open-domain qa ==="
run_hypersim_open_domain_qa amem "$amem_output/retrieved.json" "$hyper_amem_output"

echo "=== [6] hyper_rag open-domain qa ==="
run_hypersim_open_domain_qa rag_6_128 "$RAG_RETRIEVED_DATASET" "$hyper_rag_output"

echo "All data-utility pipelines finished."
