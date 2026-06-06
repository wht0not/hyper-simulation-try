#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LOCOMO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd -- "$LOCOMO_ROOT/../../../.." && pwd)"

CONTEXT_DATASET="$LOCOMO_ROOT/data/context/locomo_context.json"
RESULT_ROOT="$LOCOMO_ROOT/data/hyper_simulation/sensitivity"
CONTEXT_INSTANCES_ROOT="${CONTEXT_INSTANCES_ROOT:-/home/vincent/hyper-simulation-try/data/hypergraphs/locomo/context}"
AMEM_INSTANCES_ROOT="${AMEM_INSTANCES_ROOT:-/home/vincent/hyper-simulation-try/data/hypergraphs/locomo/amem}"
MEMORYBANK_INSTANCES_ROOT="${MEMORYBANK_INSTANCES_ROOT:-/home/vincent/hyper-simulation-try/data/hypergraphs/locomo/memorybank}"
MEMORYBANK_OUTPUT_DIR="${MEMORYBANK_OUTPUT_DIR:-/home/vincent/hyper-simulation-try/tests/tasks/memory/locomo/data/memorybank}"
HYPER_MEMORYBANK_OUTPUT_DIR="${HYPER_MEMORYBANK_OUTPUT_DIR:-/home/vincent/hyper-simulation-try/tests/tasks/memory/locomo/data/hyper_simulation/memorybank}"
MEMORYBANK_RETRIEVED_DATASET="${MEMORYBANK_RETRIEVED_DATASET:-$MEMORYBANK_OUTPUT_DIR/retrieved.json}"

MODEL_NAME="${MODEL_NAME:-qwen3.5:9b}"
ANSWER_BATCH_SIZE="${ANSWER_BATCH_SIZE:-5}"
JUDGE_MAX_WORKERS="${JUDGE_MAX_WORKERS:-4}"
LLM_JUDGE_REPEAT="${LLM_JUDGE_REPEAT:-1}"
LIMIT="${LIMIT:-0}"
DRY_RUN="${DRY_RUN:-0}"
FORCE_RERUN="${FORCE_RERUN:-0}"

RUN_CONTEXT="${RUN_CONTEXT:-1}"
RUN_AMEM="${RUN_AMEM:-1}"
RUN_MEMORYBANK="${RUN_MEMORYBANK:-1}"
RUN_PRE_MEMORYBANK_REFRESH="${RUN_PRE_MEMORYBANK_REFRESH:-1}"
RUN_PRE_HYPER_MEMORYBANK_REFRESH="${RUN_PRE_HYPER_MEMORYBANK_REFRESH:-1}"

BASE_SIGMA="${BASE_SIGMA:-0.75}"
BASE_B="${BASE_B:-5}"
BASE_DELTA="${BASE_DELTA:-0.7}"

SIGMA_VALUES=(0.4 0.5 0.6 0.7 0.8 0.9 1.0)
B_VALUES=(1 2 3 4 5 6 7 8 9 10 11 12)
DELTA_VALUES=(0.4 0.5 0.6 0.7 0.8 0.9 1.0)

mkdir -p "$RESULT_ROOT"

log() {
  echo "[$(date '+%F %T')] $*"
}

run_cmd() {
  echo "$" "$@"
  if [[ "$DRY_RUN" == "1" ]]; then
    return 0
  fi
  "$@"
}

run_simulation() {
  (
    cd "$PROJECT_ROOT"
    run_cmd pixi run -e simulation python "$LOCOMO_ROOT/run_experiments.py" "$@"
  )
}

run_hypersim_stage() {
  local stage="$1"
  shift
  echo "$" "pixi run -e simulation python $LOCOMO_ROOT/run_experiments.py --stage $stage $*"
  if [[ "$DRY_RUN" == "1" ]]; then
    return 0
  fi
  (
    cd "$PROJECT_ROOT"
    pixi run -e simulation python "$LOCOMO_ROOT/run_experiments.py" --stage "$stage" "$@"
  )
}

run_hypersim_with_thresholds() {
  local sigma="$1"
  local b="$2"
  local delta="$3"
  shift 3
  echo "$" "HYPERSIM_SIGMA_THRESHOLD=$sigma" "HYPERSIM_B_THRESHOLD=$b" "HYPERSIM_DELTA_THRESHOLD=$delta" \
    "HYPERSIM_ALLOWED_CATEGORIES=1" \
    "pixi run -e simulation python $LOCOMO_ROOT/run_experiments.py $*"
  if [[ "$DRY_RUN" == "1" ]]; then
    return 0
  fi
  (
    cd "$PROJECT_ROOT"
    HYPERSIM_SIGMA_THRESHOLD="$sigma" \
    HYPERSIM_B_THRESHOLD="$b" \
    HYPERSIM_DELTA_THRESHOLD="$delta" \
    HYPERSIM_ALLOWED_CATEGORIES="1" \
    pixi run -e simulation python "$LOCOMO_ROOT/run_experiments.py" "$@"
  )
}

ensure_instances_root() {
  local label="$1"
  local path="$2"
  if [[ ! -d "$path" ]]; then
    echo "[ERROR] missing $label instances dir: $path" >&2
    exit 1
  fi
}

format_param_token() {
  local value="$1"
  value="${value//./p}"
  echo "$value"
}

ensure_file_exists() {
  local label="$1"
  local path="$2"
  if [[ ! -f "$path" ]]; then
    echo "[ERROR] missing $label file: $path" >&2
    exit 1
  fi
}

run_one_sweep_point() {
  local source_method="$1"
  local instances_root="$2"
  local sweep_name="$3"
  local sweep_value="$4"
  local sigma="$5"
  local b="$6"
  local delta="$7"

  local sigma_token
  local b_token
  local delta_token
  sigma_token="$(format_param_token "$sigma")"
  b_token="$(format_param_token "$b")"
  delta_token="$(format_param_token "$delta")"
  local output_dir="$RESULT_ROOT/${source_method}-sigma${sigma_token}-b${b_token}-delta${delta_token}"

  if [[ "$FORCE_RERUN" == "1" ]]; then
    rm -rf "$output_dir"
  fi

  if [[ -f "$output_dir/final.json" ]]; then
    log "skip existing final: $output_dir/final.json"
    return
  fi

  mkdir -p "$output_dir"
  log "run $source_method $sweep_name=$sweep_value (sigma=$sigma, b=$b, delta=$delta)"
  run_hypersim_with_thresholds \
    "$sigma" \
    "$b" \
    "$delta" \
    --method hyper_simulation \
    --stage all \
    --instances-root "$instances_root" \
    --output-dir "$output_dir" \
    --model-name "$MODEL_NAME" \
    --answer-batch-size "$ANSWER_BATCH_SIZE" \
    --judge-max-workers "$JUDGE_MAX_WORKERS" \
    --llm-judge-repeat "$LLM_JUDGE_REPEAT" \
    --limit "$LIMIT"
}

run_sensitivity_for_source() {
  local source_method="$1"
  local instances_root="$2"
  local value

  for value in "${SIGMA_VALUES[@]}"; do
    run_one_sweep_point "$source_method" "$instances_root" "sigma" "$value" "$value" "$BASE_B" "$BASE_DELTA"
  done

  for value in "${B_VALUES[@]}"; do
    run_one_sweep_point "$source_method" "$instances_root" "b" "$value" "$BASE_SIGMA" "$value" "$BASE_DELTA"
  done

  for value in "${DELTA_VALUES[@]}"; do
    run_one_sweep_point "$source_method" "$instances_root" "delta" "$value" "$BASE_SIGMA" "$BASE_B" "$value"
  done
}

log "MODEL_NAME=$MODEL_NAME LIMIT=$LIMIT DRY_RUN=$DRY_RUN"
log "BASE_SIGMA=$BASE_SIGMA BASE_B=$BASE_B BASE_DELTA=$BASE_DELTA"
log "RESULT_ROOT=$RESULT_ROOT"
log "RUN_PRE_MEMORYBANK_REFRESH=$RUN_PRE_MEMORYBANK_REFRESH RUN_PRE_HYPER_MEMORYBANK_REFRESH=$RUN_PRE_HYPER_MEMORYBANK_REFRESH"

if [[ "$RUN_CONTEXT" == "1" ]]; then
  ensure_instances_root "context" "$CONTEXT_INSTANCES_ROOT"
  run_sensitivity_for_source "context" "$CONTEXT_INSTANCES_ROOT"
fi

if [[ "$RUN_AMEM" == "1" ]]; then
  ensure_instances_root "amem" "$AMEM_INSTANCES_ROOT"
  run_sensitivity_for_source "amem" "$AMEM_INSTANCES_ROOT"
fi

if [[ "$RUN_MEMORYBANK" == "1" ]]; then
  ensure_instances_root "memorybank" "$MEMORYBANK_INSTANCES_ROOT"
  run_sensitivity_for_source "memorybank" "$MEMORYBANK_INSTANCES_ROOT"
fi

log "sensitivity sweep finished"
