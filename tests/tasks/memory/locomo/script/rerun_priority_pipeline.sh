#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LOCOMO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd -- "$LOCOMO_ROOT/../../../.." && pwd)"

DATA_ROOT="$LOCOMO_ROOT/data"
HYPERGRAPH_ROOT="$PROJECT_ROOT/data/hypergraphs"

RETRIEVAL_SOURCE_DATASET="$LOCOMO_ROOT/data/locomo10_rag.json"
CONTEXT_DATASET="$LOCOMO_ROOT/data/context/locomo_context.json"

MODEL_NAME="${MODEL_NAME:-qwen3.5:9b}"
ANSWER_BATCH_SIZE="${ANSWER_BATCH_SIZE:-5}"
JUDGE_MAX_WORKERS="${JUDGE_MAX_WORKERS:-4}"
LLM_JUDGE_REPEAT="${LLM_JUDGE_REPEAT:-1}"
LIMIT="${LIMIT:-0}"
DRY_RUN="${DRY_RUN:-0}"

RUN_EVALUATE_ONLY="${RUN_EVALUATE_ONLY:-0}" # 没有断点续传，全量重跑
RUN_BASE_ALL="${RUN_BASE_ALL:-0}"
RUN_HYPER_CONTEXT="${RUN_HYPER_CONTEXT:-0}"
RUN_HYPER_FROM_RETRIEVAL="${RUN_HYPER_FROM_RETRIEVAL:-0}"
RUN_RAG_BASE="${RUN_RAG_BASE:-0}"
RUN_HYPER_RAG="${RUN_HYPER_RAG:-1}"

run_cmd() {
  echo "$" "$@" >&2
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  "$@"
}

run_simulation_pipeline() {
  run_cmd pixi run -e simulation python "$LOCOMO_ROOT/run_experiments.py" "$@"
}

run_hypergraph_build() {
  run_cmd pixi run -e hypergraph python "$LOCOMO_ROOT/run_experiments.py" "$@"
}

run_evaluate_only() {
  run_cmd pixi run -e simulation python "$SCRIPT_DIR/rerun_evaluate_only.py" --targets context,rag
}

run_from_prepare_dataset() {
  local method="$1"
  local dataset_path="$2"
  local output_dir="$3"
  run_simulation_pipeline \
    --method "$method" \
    --stage compose \
    --dataset-path "$dataset_path" \
    --output-dir "$output_dir" \
    --model-name "$MODEL_NAME" \
    --limit "$LIMIT"
  run_simulation_pipeline \
    --method "$method" \
    --stage answer \
    --dataset-path "$dataset_path" \
    --output-dir "$output_dir" \
    --model-name "$MODEL_NAME" \
    --answer-batch-size "$ANSWER_BATCH_SIZE" \
    --limit "$LIMIT"
  run_simulation_pipeline \
    --method "$method" \
    --stage evaluate \
    --dataset-path "$dataset_path" \
    --output-dir "$output_dir" \
    --model-name "$MODEL_NAME" \
    --judge-max-workers "$JUDGE_MAX_WORKERS" \
    --llm-judge-repeat "$LLM_JUDGE_REPEAT" \
    --limit "$LIMIT"
}

run_hypersim_from_prepare() {
  local instances_root="$1"
  local output_dir="$2"
  run_simulation_pipeline \
    --method hyper_simulation \
    --stage compose \
    --instances-root "$instances_root" \
    --output-dir "$output_dir" \
    --model-name "$MODEL_NAME" \
    --limit "$LIMIT"
  run_simulation_pipeline \
    --method hyper_simulation \
    --stage answer \
    --instances-root "$instances_root" \
    --output-dir "$output_dir" \
    --model-name "$MODEL_NAME" \
    --answer-batch-size "$ANSWER_BATCH_SIZE" \
    --limit "$LIMIT"
  run_simulation_pipeline \
    --method hyper_simulation \
    --stage evaluate \
    --instances-root "$instances_root" \
    --output-dir "$output_dir" \
    --model-name "$MODEL_NAME" \
    --judge-max-workers "$JUDGE_MAX_WORKERS" \
    --llm-judge-repeat "$LLM_JUDGE_REPEAT" \
    --limit "$LIMIT"
}

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
  local retrieved_file=""
  retrieved_file="$(pick_existing_file "$output_dir/retrieved.json")" || true
  if [[ -z "${retrieved_file:-}" ]]; then
    echo "[retrieve] missing for ${method}, running retrieve..." >&2
    run_simulation_pipeline \
      --method "$method" \
      --stage retrieve \
      --dataset-path "$dataset_path" \
      --output-dir "$output_dir" \
      --model-name "$MODEL_NAME" \
      --limit "$LIMIT"
    if [[ "${DRY_RUN}" == "1" ]]; then
      echo "$output_dir/retrieved.json"
      return 0
    fi
    retrieved_file="$(pick_existing_file "$output_dir/retrieved.json")" || true
  fi
  if [[ -z "${retrieved_file:-}" ]]; then
    echo "[ERROR] cannot find retrieved file for ${method} in ${output_dir}" >&2
    return 1
  fi
  echo "$retrieved_file"
}

echo "MODEL_NAME=$MODEL_NAME ANSWER_BATCH_SIZE=$ANSWER_BATCH_SIZE LIMIT=$LIMIT DRY_RUN=$DRY_RUN"
echo "RUN_EVALUATE_ONLY=$RUN_EVALUATE_ONLY RUN_BASE_ALL=$RUN_BASE_ALL RUN_HYPER_CONTEXT=$RUN_HYPER_CONTEXT RUN_HYPER_FROM_RETRIEVAL=$RUN_HYPER_FROM_RETRIEVAL RUN_RAG_BASE=$RUN_RAG_BASE RUN_HYPER_RAG=$RUN_HYPER_RAG"

if [[ "$RUN_EVALUATE_ONLY" == "1" ]]; then
  echo "=== [0] context/rag evaluate-only ==="
  run_evaluate_only
fi

if [[ "$RUN_BASE_ALL" == "1" ]]; then
  echo "=== [1] base methods from prepare ==="

  echo "[1.1] context compose+answer+evaluate -> data/context"
  run_from_prepare_dataset context "$CONTEXT_DATASET" "$DATA_ROOT/context"

  # echo "[1.2] amem compose+answer+evaluate -> data/amem"
  # run_from_prepare_dataset amem "$RETRIEVAL_SOURCE_DATASET" "$DATA_ROOT/amem"
fi

if [[ "$RUN_HYPER_CONTEXT" == "1" ]]; then
  echo "=== [2] hyper_context from prepare ==="
  context_instances="$HYPERGRAPH_ROOT/locomo/context"
  run_hypersim_from_prepare "$context_instances" "$DATA_ROOT/hyper_simulation/context"
fi

if [[ "$RUN_HYPER_FROM_RETRIEVAL" == "1" ]]; then
  echo "=== [3] hyper_amem + hyper_memorybank from prepare ==="
  for method in amem memorybank; do
    method_instances="$HYPERGRAPH_ROOT/locomo/$method"
    echo "[3] hyper_${method} compose+answer+evaluate"
    run_hypersim_from_prepare "$method_instances" "$DATA_ROOT/hyper_simulation/$method"
  done
fi

if [[ "$RUN_RAG_BASE" == "1" ]]; then
  echo "=== [4] rag base from prepare (context pipeline over rag retrieved sets) ==="
  shopt -s nullglob
  rag_dirs=( "$DATA_ROOT"/rag/*_* )
  shopt -u nullglob
  for rag_dir in "${rag_dirs[@]}"; do
    [[ -d "$rag_dir" ]] || continue
    rag_dataset="$(pick_existing_file "$rag_dir/locomo10_rag.json" "$rag_dir/retrieved.json")" || continue
    combo_name="$(basename "$rag_dir")"
    echo "[4] rag(base) compose+answer+evaluate: $combo_name"
    run_from_prepare_dataset context "$rag_dataset" "$rag_dir"
  done
fi

if [[ "$RUN_HYPER_RAG" == "1" ]]; then
  echo "=== [5] hyper_rag from prepare (priority combos first) ==="
  rag_priority_combos=( "10_128" "10_2048" "6_2048" )
  for combo_name in "${rag_priority_combos[@]}"; do
    rag_instances="$HYPERGRAPH_ROOT/locomo/rag/$combo_name"
    rag_hypersim_output="$DATA_ROOT/hyper_simulation/rag/$combo_name"
    if [[ ! -d "$rag_instances" ]]; then
      continue
    fi
    echo "[5.1] hyper_rag priority: $combo_name"
    run_hypersim_from_prepare "$rag_instances" "$rag_hypersim_output"
  done

  echo "[5.2] hyper_rag remaining combos"
  shopt -s nullglob
  rag_dirs=( "$DATA_ROOT"/rag/*_* )
  shopt -u nullglob
  for rag_dir in "${rag_dirs[@]}"; do
    if [[ ! -d "$rag_dir" ]]; then
      continue
    fi
    combo_name="$(basename "$rag_dir")"
    case " ${rag_priority_combos[*]} " in
      *" $combo_name "*) continue ;;
    esac
    rag_instances="$HYPERGRAPH_ROOT/locomo/rag/$combo_name"
    rag_hypersim_output="$DATA_ROOT/hyper_simulation/rag/$combo_name"
    [[ -d "$rag_instances" ]] || continue
    echo "[5.2] hyper_rag remaining: $combo_name"
    run_hypersim_from_prepare "$rag_instances" "$rag_hypersim_output"
  done
fi

echo "All requested reruns finished."
