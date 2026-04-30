#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LOCOMO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
LOCOMO_DATA="$LOCOMO_ROOT/data"
RAG_SOURCE="$LOCOMO_ROOT/data/rag/locomo10_rag.json"

CHUNK_SIZES_A=(128 256 512 1024 2048)
TOP_KS_A=(6 7 8 9 10)

CHUNK_SIZES_B=(4096 8192)
TOP_KS_B=(1 2 3)

run_retrieval() {
  local chunk_size="$1"
  local top_k="$2"
  local current="$3"
  local total="$4"

  echo "[${current}/${total}] Running rag retrieval: chunk_size=${chunk_size}, top_k=${top_k}"
  pixi run -e simulation python "$LOCOMO_ROOT/method/rag/retrieval.py" \
    --rag-source-path "$RAG_SOURCE" \
    --output-dir "$LOCOMO_DATA" \
    --chunk-size "$chunk_size" \
    --top-k "$top_k"
}

total_runs=$(( ${#CHUNK_SIZES_A[@]} * ${#TOP_KS_A[@]} + ${#CHUNK_SIZES_B[@]} * ${#TOP_KS_B[@]} ))
current_run=0

for chunk_size in "${CHUNK_SIZES_A[@]}"; do
  for top_k in "${TOP_KS_A[@]}"; do
    current_run=$((current_run + 1))
    run_retrieval "$chunk_size" "$top_k" "$current_run" "$total_runs"
  done
done

for chunk_size in "${CHUNK_SIZES_B[@]}"; do
  for top_k in "${TOP_KS_B[@]}"; do
    current_run=$((current_run + 1))
    run_retrieval "$chunk_size" "$top_k" "$current_run" "$total_runs"
  done
done
