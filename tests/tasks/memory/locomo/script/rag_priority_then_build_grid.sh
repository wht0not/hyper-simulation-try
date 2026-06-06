#!/bin/bash

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LOCOMO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd -- "$LOCOMO_ROOT/../../../.." && pwd)"

DATA_ROOT="$LOCOMO_ROOT/data"
HYPERGRAPH_ROOT="$PROJECT_ROOT/data/hypergraphs/locomo/rag"

HYPERGRAPH_BATCH_SIZE="${HYPERGRAPH_BATCH_SIZE:-1280}"
FORCE_REBUILD="${FORCE_REBUILD:-0}"
RUN_FULL_GRID_BUILD="${RUN_FULL_GRID_BUILD:-1}"
DRY_RUN="${DRY_RUN:-0}"
MODEL_NAME="${MODEL_NAME:-qwen3.5:9b}"
ANSWER_BATCH_SIZE="${ANSWER_BATCH_SIZE:-1}"
JUDGE_MAX_WORKERS="${JUDGE_MAX_WORKERS:-4}"
LLM_JUDGE_REPEAT="${LLM_JUDGE_REPEAT:-1}"
STEP3_ALLOWED_CATEGORIES="${STEP3_ALLOWED_CATEGORIES:-3}"

CHUNK_SIZES_A=(128 256 512 1024 2048)
TOP_KS_A=(6 7 8 9 10)
CHUNK_SIZES_B=(4096 8192)
TOP_KS_B=(1 2 3)

run_cmd() {
  echo "$" "$@" >&2
  if [[ "$DRY_RUN" == "1" ]]; then
    return 0
  fi
  "$@"
}

run_build() {
  local dataset_path="$1"
  local instances_root="$2"
  local limit="${3:-0}"
  local allowed_categories="${4:-}"
  local force_args=()
  local limit_args=()
  local env_args=()
  if [[ "$FORCE_REBUILD" == "1" ]]; then
    force_args+=(--force-rebuild)
  fi
  if [[ "$limit" != "0" ]]; then
    limit_args+=(--limit "$limit")
  fi
  if [[ -n "$allowed_categories" ]]; then
    env_args+=(HYPERSIM_ALLOWED_CATEGORIES="$allowed_categories")
  fi

  echo "[build] dataset=$(basename "$dataset_path"), instances_root=$instances_root"
  run_cmd env "${env_args[@]}" pixi run -e hypergraph python "$LOCOMO_ROOT/run_experiments.py" \
    --method hyper_simulation \
    --stage build \
    --dataset-path "$dataset_path" \
    --instances-root "$instances_root" \
    --batch-size "$HYPERGRAPH_BATCH_SIZE" \
    "${limit_args[@]}" \
    "${force_args[@]}"
}

run_hypersim_all() {
  local instances_root="$1"
  local output_dir="$2"
  local limit="${3:-0}"
  local allowed_categories="${4:-}"
  local limit_args=()
  local env_args=()
  if [[ "$limit" != "0" ]]; then
    limit_args+=(--limit "$limit")
  fi
  if [[ -n "$allowed_categories" ]]; then
    env_args+=(HYPERSIM_ALLOWED_CATEGORIES="$allowed_categories")
  fi
  echo "[all] instances_root=$instances_root -> $output_dir"
  run_cmd env "${env_args[@]}" pixi run -e simulation python "$LOCOMO_ROOT/run_experiments.py" \
    --method hyper_simulation \
    --stage all \
    --instances-root "$instances_root" \
    --output-dir "$output_dir" \
    --model-name "$MODEL_NAME" \
    --answer-batch-size "$ANSWER_BATCH_SIZE" \
    --judge-max-workers "$JUDGE_MAX_WORKERS" \
    --llm-judge-repeat "$LLM_JUDGE_REPEAT" \
    "${limit_args[@]}"
}

combo_name() {
  local top_k="$1"
  local chunk_size="$2"
  echo "${top_k}_${chunk_size}"
}

retrieved_file_for_combo() {
  local top_k="$1"
  local chunk_size="$2"
  echo "$DATA_ROOT/rag/$(combo_name "$top_k" "$chunk_size")/locomo10_rag.json"
}

instances_root_for_combo() {
  local top_k="$1"
  local chunk_size="$2"
  echo "$HYPERGRAPH_ROOT/$(combo_name "$top_k" "$chunk_size")"
}

echo "HYPERGRAPH_BATCH_SIZE=$HYPERGRAPH_BATCH_SIZE FORCE_REBUILD=$FORCE_REBUILD"
echo "RUN_FULL_GRID_BUILD=$RUN_FULL_GRID_BUILD DRY_RUN=$DRY_RUN"
echo "MODEL_NAME=$MODEL_NAME ANSWER_BATCH_SIZE=$ANSWER_BATCH_SIZE JUDGE_MAX_WORKERS=$JUDGE_MAX_WORKERS LLM_JUDGE_REPEAT=$LLM_JUDGE_REPEAT"
echo "STEP3_ALLOWED_CATEGORIES=$STEP3_ALLOWED_CATEGORIES"

if [[ "$RUN_FULL_GRID_BUILD" == "1" ]]; then
  echo "=== [Step 3] grid-style hypergraph build loop (category: ${STEP3_ALLOWED_CATEGORIES}) ==="
  total_runs=$(( ${#CHUNK_SIZES_A[@]} * ${#TOP_KS_A[@]} + ${#CHUNK_SIZES_B[@]} * ${#TOP_KS_B[@]} ))
  current_run=0

  for chunk_size in "${CHUNK_SIZES_A[@]}"; do
    for top_k in "${TOP_KS_A[@]}"; do
      current_run=$((current_run + 1))
      echo "[${current_run}/${total_runs}] build combo $(combo_name "$top_k" "$chunk_size")"
      retrieved_file="$(retrieved_file_for_combo "$top_k" "$chunk_size")"
      if [[ ! -f "$retrieved_file" ]]; then
        echo "[WARN] skip missing retrieved file: $retrieved_file" >&2
        continue
      fi
      combo_instances="$(instances_root_for_combo "$top_k" "$chunk_size")"
      combo_output="$DATA_ROOT/hyper_simulation/rag/$(combo_name "$top_k" "$chunk_size")"
      run_build "$retrieved_file" "$combo_instances" 0 "$STEP3_ALLOWED_CATEGORIES"
      run_hypersim_all "$combo_instances" "$combo_output" 0 "$STEP3_ALLOWED_CATEGORIES"
    done
  done

  for chunk_size in "${CHUNK_SIZES_B[@]}"; do
    for top_k in "${TOP_KS_B[@]}"; do
      current_run=$((current_run + 1))
      echo "[${current_run}/${total_runs}] build combo $(combo_name "$top_k" "$chunk_size")"
      retrieved_file="$(retrieved_file_for_combo "$top_k" "$chunk_size")"
      if [[ ! -f "$retrieved_file" ]]; then
        echo "[WARN] skip missing retrieved file: $retrieved_file" >&2
        continue
      fi
      combo_instances="$(instances_root_for_combo "$top_k" "$chunk_size")"
      combo_output="$DATA_ROOT/hyper_simulation/rag/$(combo_name "$top_k" "$chunk_size")"
      run_build "$retrieved_file" "$combo_instances" 0 "$STEP3_ALLOWED_CATEGORIES"
      run_hypersim_all "$combo_instances" "$combo_output" 0 "$STEP3_ALLOWED_CATEGORIES"
    done
  done
fi

echo "Done."
