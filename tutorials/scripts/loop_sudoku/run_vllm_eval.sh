#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Serve the Sudoku loop slot with vLLM (answer burst) and score it, from the
# repo root. Env: RUN_DIR (results), COMPOSE_HEAD (trained loop_head.safetensors),
# BASE_MODEL, READ_LAYER, EVAL_START, EVAL_N, REFERENCE (puzzles re-solved on
# the HF backend from the same fixed start), TESTS (1: run the vLLM loop tests
# first).
set -euo pipefail
WORK=/tmp/work
mkdir -p "$WORK" "$RUN_DIR"
log() { echo "[$(date -u +%H:%M:%S)] $*"; }
py() { uv run --no-sync python "$@"; }
BASE_MODEL=${BASE_MODEL:-ibm-granite/granite-4.1-3b}

if [ "${TESTS:-1}" = 1 ]; then
  log "0/4 loop tests (CPU parts + HF solve_each)"
  uv run --no-sync python -m pytest tests/vllm/test_loop_burst.py tests/hf/test_loop.py -v -s --tb=short -x \
    2>&1 | tee "$RUN_DIR/tests.log" | tail -n 40
fi

log "1/4 test split"
py - <<PYEOF
from huggingface_hub import hf_hub_download
import os, sys
sys.path.insert(0, "tutorials/scripts/loop_sudoku")
from sudoku_data import build_test
raw = os.path.dirname(hf_hub_download("sapientinc/sudoku-extreme", "test.csv", repo_type="dataset"))
print("test", build_test(raw, "$WORK/data"))
PYEOF

log "2/4 compose the checkpoint with the trained loop head"
slot_args=(--head "$COMPOSE_HEAD")
[ -n "${READ_LAYER:-}" ] && slot_args+=(--read-layer "$READ_LAYER")
py tutorials/scripts/loop_sudoku/make_slot.py --base "$BASE_MODEL" --out "$WORK/slot" "${slot_args[@]}"
py -m granite_switch.composer.compose_granite_switch \
  --adapters "$WORK/slot/manifest.yaml" --base-model "$BASE_MODEL" \
  --output "$WORK/ckpt" > "$RUN_DIR/compose.log" 2>&1
tail -n 3 "$RUN_DIR/compose.log"

log "3/4 smoke: CUDA graphs vs eager, a mixed batch, the HF reference"
py tutorials/scripts/loop_sudoku/serve_vllm.py "$WORK/ckpt" "$WORK/data" \
  --first "${EVAL_START:-2000}" --count 64 --reference "${REFERENCE:-16}" --mix \
  --json "$RUN_DIR/vllm_smoke.json" 2>&1 | tee "$RUN_DIR/vllm_smoke.log" | grep -E "LOOP_VLLM|Error|error|Traceback" || true
py tutorials/scripts/loop_sudoku/serve_vllm.py "$WORK/ckpt" "$WORK/data" \
  --first "${EVAL_START:-2000}" --count 64 --eager \
  --json "$RUN_DIR/vllm_smoke_eager.json" 2>&1 | tee "$RUN_DIR/vllm_smoke_eager.log" | grep -E "LOOP_VLLM|Error|error|Traceback" || true

log "4/4 held-out test, served"
py tutorials/scripts/loop_sudoku/serve_vllm.py "$WORK/ckpt" "$WORK/data" \
  --first "${EVAL_START:-2000}" --count "${EVAL_N:-20000}" \
  --json "$RUN_DIR/vllm_eval.json" 2>&1 | tee "$RUN_DIR/vllm_eval.log" | grep -E "LOOP_VLLM|Error|error|Traceback" || true
log "done"
