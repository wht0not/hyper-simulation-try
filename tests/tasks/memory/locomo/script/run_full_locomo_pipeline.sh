#!/bin/bash

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LOCOMO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd -- "$LOCOMO_ROOT/../../../.." && pwd)"

DATA_ROOT="$LOCOMO_ROOT/data"
HYPERGRAPH_ROOT="$PROJECT_ROOT/data/hypergraphs"

CONTEXT_DATASET="$LOCOMO_ROOT/data/context/locomo_context.json"
RETRIEVAL_SOURCE_DATASET="$LOCOMO_ROOT/data/locomo10_rag.json"

MODEL_NAME="${MODEL_NAME:-qwen3.5:9b}"
ANSWER_BATCH_SIZE="${ANSWER_BATCH_SIZE:-5}"
JUDGE_MAX_WORKERS="${JUDGE_MAX_WORKERS:-4}"
LLM_JUDGE_REPEAT="${LLM_JUDGE_REPEAT:-1}"
HYPERGRAPH_BATCH_SIZE="${HYPERGRAPH_BATCH_SIZE:-128}"
LIMIT="${LIMIT:-0}"
FORCE_REBUILD="${FORCE_REBUILD:-0}"

# 控制阶段
RUN_BASE_ALL="${RUN_BASE_ALL:-1}"
RUN_HYPER_FROM_RETRIEVAL="${RUN_HYPER_FROM_RETRIEVAL:-0}"

run_simulation_pipeline() {
  pixi run -e simulation python "$LOCOMO_ROOT/run_experiments.py" "$@"
}

run_hypergraph_build() {
  pixi run -e hypergraph python "$LOCOMO_ROOT/run_experiments.py" "$@"
}

build_flag_args=()
if [[ "$FORCE_REBUILD" == "1" ]]; then
  build_flag_args+=(--force-rebuild)
fi

pick_existing_file() {
  for file in "$@"; do
    if [[ -f "$file" ]]; then
      echo "$file"
      return 0
    fi
  done
  return 1
}

ensure_retrieved_file() {
  local method="$1"
  local output_dir="$2"
  local dataset_path="$3"
  local retrieved_file
  retrieved_file="$(pick_existing_file "$output_dir/retrieved.json")" || true
  if [[ -z "${retrieved_file:-}" ]]; then
    echo "[retrieve] missing for ${method}, running retrieve..."
    run_simulation_pipeline \
      --method "$method" \
      --stage retrieve \
      --dataset-path "$dataset_path" \
      --output-dir "$output_dir" \
      --model-name "$MODEL_NAME" \
      --limit "$LIMIT"
    retrieved_file="$(pick_existing_file "$output_dir/retrieved.json")" || true
  fi
  if [[ -z "${retrieved_file:-}" ]]; then
    echo "[WARN] cannot find retrieved file for ${method} in ${output_dir}, skip."
    return 1
  fi
  echo "$retrieved_file"
}

echo "MODEL_NAME=$MODEL_NAME LIMIT=$LIMIT RUN_BASE_ALL=$RUN_BASE_ALL RUN_HYPER_FROM_RETRIEVAL=$RUN_HYPER_FROM_RETRIEVAL"

if [[ "$RUN_BASE_ALL" == "1" ]]; then
  echo "=== [A] Run base method all ==="

  echo "[A1] context all -> data/context"
  run_simulation_pipeline \
    --method context \
    --stage all \
    --dataset-path "$CONTEXT_DATASET" \
    --output-dir "$DATA_ROOT/context" \
    --model-name "$MODEL_NAME" \
    --answer-batch-size "$ANSWER_BATCH_SIZE" \
    --judge-max-workers "$JUDGE_MAX_WORKERS" \
    --llm-judge-repeat "$LLM_JUDGE_REPEAT" \
    --limit "$LIMIT"

  echo "[A2] langmem all -> data/langmem"
  run_simulation_pipeline \
    --method langmem \
    --stage all \
    --dataset-path "$RETRIEVAL_SOURCE_DATASET" \
    --output-dir "$DATA_ROOT/langmem" \
    --model-name "$MODEL_NAME" \
    --answer-batch-size "$ANSWER_BATCH_SIZE" \
    --judge-max-workers "$JUDGE_MAX_WORKERS" \
    --llm-judge-repeat "$LLM_JUDGE_REPEAT" \
    --limit "$LIMIT"

  echo "[A3] amem all -> data/amem"
  run_simulation_pipeline \
    --method amem \
    --stage all \
    --dataset-path "$RETRIEVAL_SOURCE_DATASET" \
    --output-dir "$DATA_ROOT/amem" \
    --model-name "$MODEL_NAME" \
    --answer-batch-size "$ANSWER_BATCH_SIZE" \
    --judge-max-workers "$JUDGE_MAX_WORKERS" \
    --llm-judge-repeat "$LLM_JUDGE_REPEAT" \
    --limit "$LIMIT"

  echo "[A4] memorybank all -> data/memorybank"
  run_simulation_pipeline \
    --method memorybank \
    --stage all \
    --dataset-path "$RETRIEVAL_SOURCE_DATASET" \
    --output-dir "$DATA_ROOT/memorybank" \
    --model-name "$MODEL_NAME" \
    --answer-batch-size "$ANSWER_BATCH_SIZE" \
    --judge-max-workers "$JUDGE_MAX_WORKERS" \
    --llm-judge-repeat "$LLM_JUDGE_REPEAT" \
    --limit "$LIMIT"
  
  echo "[A5] context + hyper_simulation baseline all -> data/hyper_simulation/context"
  context_instances="$HYPERGRAPH_ROOT/locomo/context"
  run_hypergraph_build \
    --method hyper_simulation \
    --stage build \
    --dataset-path "$CONTEXT_DATASET" \
    --instances-root "$context_instances" \
    --batch-size "$HYPERGRAPH_BATCH_SIZE" \
    --limit "$LIMIT" \
    "${build_flag_args[@]}"

  run_simulation_pipeline \
    --method hyper_simulation \
    --stage all \
    --instances-root "$context_instances" \
    --output-dir "$DATA_ROOT/hyper_simulation/context" \
    --model-name "$MODEL_NAME" \
    --answer-batch-size "$ANSWER_BATCH_SIZE" \
    --judge-max-workers "$JUDGE_MAX_WORKERS" \
    --llm-judge-repeat "$LLM_JUDGE_REPEAT" \
    --limit "$LIMIT"

  echo "[A6] rag existing retrieved -> context all (resume-aware)"
  shopt -s nullglob
  rag_dirs=( "$DATA_ROOT"/rag/*_* )
  shopt -u nullglob
  for rag_output_dir in "${rag_dirs[@]}"; do
    if [[ ! -d "$rag_output_dir" ]]; then
      continue
    fi
    rag_dataset="$rag_output_dir/locomo10_rag.json"
    if [[ ! -f "$rag_dataset" ]]; then
      echo "[A5] skip $rag_output_dir (missing retrieved dataset locomo10_rag.json)"
      continue
    fi
    echo "[A6] context all for rag dir: $rag_output_dir"
    run_simulation_pipeline \
      --method context \
      --stage all \
      --dataset-path "$rag_dataset" \
      --output-dir "$rag_output_dir" \
      --model-name "$MODEL_NAME" \
      --answer-batch-size "$ANSWER_BATCH_SIZE" \
      --judge-max-workers "$JUDGE_MAX_WORKERS" \
      --llm-judge-repeat "$LLM_JUDGE_REPEAT" \
      --limit "$LIMIT"
  done

fi

if [[ "$RUN_HYPER_FROM_RETRIEVAL" == "1" ]]; then
  echo "=== [B] retrieval -> hypergraph -> hyper_simulation(all) ==="
# langmem amem
  for method in memorybank langmem amem; do
    echo "[B] ${method} retrieved -> hypergraph -> hypersim all"
    method_output_dir="$DATA_ROOT/$method"
    retrieved_file="$(ensure_retrieved_file "$method" "$method_output_dir" "$RETRIEVAL_SOURCE_DATASET")" || continue
    method_instances="$HYPERGRAPH_ROOT/locomo/$method"
    run_hypergraph_build \
      --method hyper_simulation \
      --stage build \
      --dataset-path "$retrieved_file" \
      --instances-root "$method_instances" \
      --batch-size "$HYPERGRAPH_BATCH_SIZE" \
      --limit "$LIMIT" \
      "${build_flag_args[@]}"

    run_simulation_pipeline \
      --method hyper_simulation \
      --stage all \
      --instances-root "$method_instances" \
      --output-dir "$DATA_ROOT/hyper_simulation/$method" \
      --model-name "$MODEL_NAME" \
      --answer-batch-size "$ANSWER_BATCH_SIZE" \
      --judge-max-workers "$JUDGE_MAX_WORKERS" \
      --llm-judge-repeat "$LLM_JUDGE_REPEAT" \
      --limit "$LIMIT"
  done

  echo "[B] rag existing retrieved -> hypergraph -> hypersim all"
  shopt -s nullglob
  rag_dirs=( "$DATA_ROOT"/rag/*_* )
  shopt -u nullglob
  for rag_dir in "${rag_dirs[@]}"; do
    if [[ ! -d "$rag_dir" ]]; then
      continue
    fi
    rag_retrieved="$rag_dir/locomo10_rag.json"
    if [[ ! -f "$rag_retrieved" ]]; then
      echo "[B] skip $rag_dir (missing retrieved dataset locomo10_rag.json)"
      continue
    fi
    combo_name="$(basename "$rag_dir")"
    rag_instances="$HYPERGRAPH_ROOT/locomo/rag/$combo_name"
    rag_hypersim_output="$DATA_ROOT/hyper_simulation/rag/$combo_name"

    run_hypergraph_build \
      --method hyper_simulation \
      --stage build \
      --dataset-path "$rag_retrieved" \
      --instances-root "$rag_instances" \
      --batch-size "$HYPERGRAPH_BATCH_SIZE" \
      --limit "$LIMIT" \
      "${build_flag_args[@]}"

    run_simulation_pipeline \
      --method hyper_simulation \
      --stage all \
      --instances-root "$rag_instances" \
      --output-dir "$rag_hypersim_output" \
      --model-name "$MODEL_NAME" \
      --answer-batch-size "$ANSWER_BATCH_SIZE" \
      --judge-max-workers "$JUDGE_MAX_WORKERS" \
      --llm-judge-repeat "$LLM_JUDGE_REPEAT" \
      --limit "$LIMIT"
  done
fi

echo "All requested pipelines finished."
