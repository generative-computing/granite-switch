#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Score saved loop heads on the test split through the served HF path, for
# several solver budgets, from the repo root. Env: RUN_DIR (results), HEADS
# (space-separated head files, or "composed"), BUDGETS ("max:min ..."), EVAL_N,
# EVAL_START, GPUS, BASE_MODEL, READ_LAYER, SAMPLES (answers kept per head),
# DUMP_INDEX (one test puzzle's full output, from the composed checkpoint). COMPOSE_HEAD: a head to compose into the
# checkpoint itself (then HEADS defaults to "composed": the checkpoint as served).
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

log "2/3 compose a checkpoint with a loop slot"
if [ -n "${COMPOSE_HEAD:-}" ]; then
  slot_args=(--head "$COMPOSE_HEAD"); compose_args=(); HEADS=${HEADS:-composed}
else
  slot_args=(); compose_args=(--untrained-loop-heads)
fi
# READ_LAYER: the layer the heads were trained to read (default: the final state).
[ -n "${READ_LAYER:-}" ] && slot_args+=(--read-layer "$READ_LAYER")
py tutorials/scripts/loop_sudoku/make_slot.py --base "$BASE_MODEL" --out "$WORK/slot" "${slot_args[@]}"
py -m granite_switch.composer.compose_granite_switch \
  --adapters "$WORK/slot/manifest.yaml" --base-model "$BASE_MODEL" \
  --output "$WORK/ckpt" "${compose_args[@]}" > "$RUN_DIR/compose.log" 2>&1
tail -n 3 "$RUN_DIR/compose.log"

log "3/3 score the heads on $GPUS GPU(s)"
run_py() {
  if [ "${GPUS:-1}" -gt 1 ]; then
    env -u RANK -u WORLD_SIZE -u MASTER_ADDR -u MASTER_PORT \
      uv run --no-sync torchrun --standalone --nproc_per_node="$GPUS" "$@"
  else
    env -u RANK -u WORLD_SIZE -u LOCAL_RANK uv run --no-sync python "$@"
  fi
}
# VARIANTS: space-separated eval_head.py precision options per pass, "_" for
# the default; e.g. "_ --solve-dtype=bfloat16 --outer-autocast".
for b in $BUDGETS; do
  for v in ${VARIANTS:-_}; do
    [ "$v" = _ ] && v=
    run_py tutorials/scripts/loop_sudoku/eval_head.py "$WORK/ckpt" "$WORK/data" $HEADS --count "$EVAL_N" --first "${EVAL_START:-0}" \
      --max-iter "${b%%:*}" --min-iter "${b##*:}" --samples "${SAMPLES:-0}" $v | tee -a "$RUN_DIR/head_eval.log"
  done
done
if [ -n "${DUMP_INDEX:-}" ]; then
  # One request's full output (the 81 x 9 score matrix), for the report.
  py tutorials/scripts/loop_sudoku/dump_output.py "$WORK/ckpt" "$WORK/data" \
    --index "$DUMP_INDEX" --out "$RUN_DIR/output_example.json" --seeds "${DUMP_SEEDS:-1}"
fi
log "done"
