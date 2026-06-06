#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LOCOMO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd -- "$LOCOMO_ROOT/../../../.." && pwd)"

RESULT_ROOT="/home/vincent/hyper-simulation-try/tests/tasks/memory/locomo/data/hyper_simulation/efficiency"
CONTEXT_INSTANCES_ROOT="${CONTEXT_INSTANCES_ROOT:-/home/vincent/hyper-simulation-try/data/hypergraphs/locomo/context}"

MODEL_NAME="${MODEL_NAME:-qwen3.5:9b}"
LIMIT="${LIMIT:-0}"
DRY_RUN="${DRY_RUN:-0}"
FORCE_RERUN="${FORCE_RERUN:-0}"

# 允许的类别设为 3
ALLOWED_CATEGORIES="3"

mkdir -p "$RESULT_ROOT"

log() {
  echo "[$(date '+%F %T')] $*"
}

run_hypersim_with_thresholds() {
  local sigma="$1"
  local b="$2"
  local delta="$3"
  shift 3
  
  log "HYPERSIM_SIGMA_THRESHOLD=$sigma HYPERSIM_B_THRESHOLD=$b HYPERSIM_DELTA_THRESHOLD=$delta"
  
  if [[ "$DRY_RUN" == "1" ]]; then
    return 0
  fi
  
  (
    cd "$PROJECT_ROOT"
    HYPERSIM_SIGMA_THRESHOLD="$sigma" \
    HYPERSIM_B_THRESHOLD="$b" \
    HYPERSIM_DELTA_THRESHOLD="$delta" \
    HYPERSIM_ALLOWED_CATEGORIES="$ALLOWED_CATEGORIES" \
    pixi run -e simulation python "$LOCOMO_ROOT/run_experiments.py" \
      --method hyper_simulation \
      --stage compose \
      "$@"
  )
}

format_param_token() {
  local value="$1"
  value="${value//./p}"
  echo "$value"
}

run_one_efficiency_point() {
  local label="$1"
  local instances_root="$2"
  local sigma="$3"
  local b="$4"
  local delta="$5"

  local sigma_token
  local b_token
  local delta_token
  sigma_token="$(format_param_token "$sigma")"
  b_token="$(format_param_token "$b")"
  delta_token="$(format_param_token "$delta")"
  
  local output_dir="$RESULT_ROOT/${label}-sigma${sigma_token}-b${b_token}-delta${delta_token}"

  if [[ "$FORCE_RERUN" == "1" ]]; then
    rm -rf "$output_dir"
  fi

  if [[ -f "$output_dir/prepared.json" ]]; then
    log "skip existing prepared: $output_dir/prepared.json"
    return
  fi

  mkdir -p "$output_dir"
  log "run efficiency point: sigma=$sigma, b=$b, delta=$delta -> $output_dir"
  
  run_hypersim_with_thresholds \
    "$sigma" \
    "$b" \
    "$delta" \
    --instances-root "$instances_root" \
    --output-dir "$output_dir" \
    --limit "$LIMIT"
}

# --- Sweep Definitions ---

# 1. Sweep sigma (0.3-0.8)
log "Starting sigma sweep (0.3-0.8)..."
SIGMA_SWEEP=(0.3 0.4 0.5 0.6 0.7 0.8)
for sigma in "${SIGMA_SWEEP[@]}"; do
  # Group 1: b=5, delta=0.65
  run_one_efficiency_point "hyper_context" "$CONTEXT_INSTANCES_ROOT" "$sigma" 5 0.65
  # Group 2: b=7, delta=0.75
  run_one_efficiency_point "hyper_context" "$CONTEXT_INSTANCES_ROOT" "$sigma" 7 0.75
  # Group 3: b=10, delta=0.85
  run_one_efficiency_point "hyper_context" "$CONTEXT_INSTANCES_ROOT" "$sigma" 10 0.85
done

# 2. Sweep b (1-9)
log "Starting b sweep (1-9)..."
B_SWEEP=(1 2 3 4 5 6 7 8 9)
for b in "${B_SWEEP[@]}"; do
  # Group 1: sigma=0.55, delta=0.5
  run_one_efficiency_point "hyper_context" "$CONTEXT_INSTANCES_ROOT" 0.55 "$b" 0.5
  # Group 2: sigma=0.65, delta=0.7
  run_one_efficiency_point "hyper_context" "$CONTEXT_INSTANCES_ROOT" 0.65 "$b" 0.7
  # Group 3: sigma=0.75, delta=0.8
  run_one_efficiency_point "hyper_context" "$CONTEXT_INSTANCES_ROOT" 0.75 "$b" 0.8
done

# 3. Sweep delta (0.35-0.85)
log "Starting delta sweep (0.35-0.85)..."
DELTA_SWEEP=(0.35 0.45 0.55 0.65 0.75 0.85)
for delta in "${DELTA_SWEEP[@]}"; do
  # Group 1: sigma=0.8, b=5
  run_one_efficiency_point "hyper_context" "$CONTEXT_INSTANCES_ROOT" 0.8 5 "$delta"
  # Group 2: sigma=0.7, b=7
  run_one_efficiency_point "hyper_context" "$CONTEXT_INSTANCES_ROOT" 0.7 7 "$delta"
  # Group 3: sigma=0.6, b=10
  run_one_efficiency_point "hyper_context" "$CONTEXT_INSTANCES_ROOT" 0.6 10 "$delta"
done

log "Efficiency sweep finished."
