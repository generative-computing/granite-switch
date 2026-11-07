#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Compose a trained generation adapter (LoRA, aLoRA or SR) into a Granite Switch
# checkpoint and score it on Sudoku through vLLM, from the repo root
# (`uv sync --extra vllm --extra compose` first).
#
#   ADAPTER_DIR=<the trained adapter> TECH=lora|alora|sr RUN_DIR=<results> \
#     bash tutorials/scripts/loop_sudoku/run_adapter_eval.sh
#
# Env: ADAPTER_DIR (adapter_config.json + adapter_model.safetensors), TECH,
# RUN_DIR, BASE_MODEL, DATA (a folder with test.npz; built when unset),
# EVAL_START / EVAL_N (the test slice, default the loop slot's held-out
# [2000, 22000)), BASELINE=1 to also score the base model without an adapter.
set -euo pipefail

WORK=/tmp/work
mkdir -p "$WORK" "$RUN_DIR"
log() { echo "[$(date -u +%H:%M:%S)] $*"; }
py() { uv run --no-sync python "$@"; }
BASE_MODEL=${BASE_MODEL:-ibm-granite/granite-4.1-3b}
HERE=tutorials/scripts/loop_sudoku

if [ -z "${DATA:-}" ]; then
  log "1/3 test split"
  DATA=$WORK/data
  py - <<PYEOF
from huggingface_hub import hf_hub_download
import os, sys
sys.path.insert(0, "$HERE")
from sudoku_data import build_test
raw = os.path.dirname(hf_hub_download("sapientinc/sudoku-extreme", "test.csv", repo_type="dataset"))
print("test", build_test(raw, "$DATA"))
PYEOF
fi

log "2/3 compose the adapter as slot 'sudoku' ($TECH)"
# One adapter folder in the library layout <name>/<model>/<technology>/: the
# composer names the slot after <name> and reads lora/alora from the folder; an
# SR adapter is told by its weights (its folder stays "lora", the layout's).
SLOT=$WORK/lib/sudoku/$(basename "$BASE_MODEL")/$([ "$TECH" = alora ] && echo alora || echo lora)
mkdir -p "$SLOT"
cp "$ADAPTER_DIR"/adapter_config.json "$ADAPTER_DIR"/adapter_model.safetensors "$SLOT"/
printf 'name: sudoku\nmodel: ~\n' > "$SLOT/io.yaml"
py -m granite_switch.composer.compose_granite_switch \
  --adapters "$SLOT" --base-model "$BASE_MODEL" --output "$WORK/ckpt" \
  > "$RUN_DIR/compose.log" 2>&1
tail -n 3 "$RUN_DIR/compose.log"

log "3/3 score it on test puzzles [${EVAL_START:-2000}, +${EVAL_N:-20000})"
py "$HERE/eval_adapter.py" "$WORK/ckpt" "$DATA" --adapter sudoku \
  --first "${EVAL_START:-2000}" --count "${EVAL_N:-20000}" --json "$RUN_DIR/adapter_eval.json"
if [ "${BASELINE:-0}" = 1 ]; then
  py "$HERE/eval_adapter.py" "$WORK/ckpt" "$DATA" --adapter none \
    --first "${EVAL_START:-2000}" --count "${EVAL_N:-20000}" --json "$RUN_DIR/base_eval.json" --tag BASE_EVAL
fi
log "done: results in $RUN_DIR"
