#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Serve a trained loop head with vLLM, from the repo root: compose it into a
# checkpoint, then run serve_vllm.py (one-step answers, accuracy, mixed batch,
# chunked prefill, HF agreement). Env: RUN_DIR (results), COMPOSE_HEAD (the
# trained loop_head.safetensors), BASE_MODEL, READ_LAYER, EVAL_START, EVAL_N,
# HF_N (puzzles checked against HF), CHUNK (token budget of the chunked run).
set -euo pipefail
WORK=/tmp/work
mkdir -p "$WORK" "$RUN_DIR"
log() { echo "[$(date -u +%H:%M:%S)] $*"; }
py() { uv run --no-sync python "$@"; }
BASE_MODEL=${BASE_MODEL:-ibm-granite/granite-4.1-3b}

log "1/3 test split"
py - <<PYEOF
from huggingface_hub import hf_hub_download
import os, sys
sys.path.insert(0, "tutorials/scripts/loop_sudoku")
from sudoku_data import build_test
raw = os.path.dirname(hf_hub_download("sapientinc/sudoku-extreme", "test.csv", repo_type="dataset"))
print("test", build_test(raw, "$WORK/data"))
PYEOF

log "2/3 compose the trained head"
slot_args=(--head "$COMPOSE_HEAD")
[ -n "${READ_LAYER:-}" ] && slot_args+=(--read-layer "$READ_LAYER")
py tutorials/scripts/loop_sudoku/make_slot.py --base "$BASE_MODEL" --out "$WORK/slot" "${slot_args[@]}"
py -m granite_switch.composer.compose_granite_switch \
  --adapters "$WORK/slot/manifest.yaml" --base-model "$BASE_MODEL" \
  --output "$WORK/ckpt" > "$RUN_DIR/compose.log" 2>&1
tail -n 3 "$RUN_DIR/compose.log"

log "3/3 serve with vLLM"
env -u RANK -u WORLD_SIZE -u LOCAL_RANK uv run --no-sync python \
  tutorials/scripts/loop_sudoku/serve_vllm.py "$WORK/ckpt" "$WORK/data" \
  --first "${EVAL_START:-2000}" --count "${EVAL_N:-2000}" \
  --hf "${HF_N:-32}" --chunk "${CHUNK:-48}" --json "$RUN_DIR/serve_eval.json"
log "done"
