#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# The whole Sudoku loop-slot pipeline on one GPU machine, from the repo root
# (`uv sync --extra hf --extra compose` first). Builds the Sudoku data, composes
# the training checkpoint (a kind: loop slot, head untrained), trains the loop
# head, then composes the final checkpoint with the best head and evaluates it
# through the served HF path on test puzzles no selection decision saw.
#
#   RUN_DIR=./runs/sudoku MODE=full GPUS=4 bash tutorials/scripts/loop_sudoku/run_pipeline.sh
#
# Env: RUN_DIR (results), MODE (smoke|full), GPUS, BASE_MODEL, READ_LAYER, plus
# the training knobs below. Everything heavy is written under /tmp/work first
# and the results are copied to RUN_DIR as they are produced.
set -euo pipefail

WORK=/tmp/work
mkdir -p "$WORK" "$RUN_DIR"
log() { echo "[$(date -u +%H:%M:%S)] $*"; }
py() { uv run --no-sync python "$@"; }

BASE_MODEL=${BASE_MODEL:-ibm-granite/granite-4.1-3b}
READ_LAYER=${READ_LAYER:-}
read_layer_args=()
[ -n "$READ_LAYER" ] && read_layer_args=(--read-layer "$READ_LAYER")

if [ "$MODE" = smoke ]; then
  SUBSAMPLE=${SUBSAMPLE:-50}; NUM_AUG=${NUM_AUG:-20}
  # One GPU's share of the full run's batch (768 / 4), so the step time is the real one.
  STEPS=${STEPS:-300}; BATCH=${BATCH:-192}; WARMUP=${WARMUP:-10}
  EVAL_EVERY=${EVAL_EVERY:-150}; EVAL_N=${EVAL_N:-256}; FINAL_EVAL_N=${FINAL_EVAL_N:-512}
  EVAL_MAX_ITER=${EVAL_MAX_ITER:-100}
else
  SUBSAMPLE=${SUBSAMPLE:-1000}; NUM_AUG=${NUM_AUG:-1000}
  STEPS=${STEPS:-78125}; BATCH=${BATCH:-768}; WARMUP=${WARMUP:-2000}
  EVAL_EVERY=${EVAL_EVERY:-2500}; EVAL_N=${EVAL_N:-2000}; FINAL_EVAL_N=${FINAL_EVAL_N:-10000}
  COMPOSED_EVAL_N=${COMPOSED_EVAL_N:-10000}
  EVAL_MAX_ITER=${EVAL_MAX_ITER:-35000}
fi

log "1/5 Sudoku-Extreme data (subsample $SUBSAMPLE x $NUM_AUG)"
py - <<EOF
from huggingface_hub import hf_hub_download
import os, sys
sys.path.insert(0, "tutorials/scripts/loop_sudoku")
from sudoku_data import build
raw = os.path.dirname(hf_hub_download("sapientinc/sudoku-extreme", "train.csv", repo_type="dataset"))
hf_hub_download("sapientinc/sudoku-extreme", "test.csv", repo_type="dataset")
n_train, n_test = build(raw, "$WORK/data", subsample=$SUBSAMPLE, num_aug=$NUM_AUG)
print("train", n_train, "test", n_test)
EOF

log "2/5 compose the training checkpoint (untrained loop head)"
py tutorials/scripts/loop_sudoku/make_slot.py --base "$BASE_MODEL" --out "$WORK/slot" "${read_layer_args[@]}"
py -m granite_switch.composer.compose_granite_switch \
  --adapters "$WORK/slot/manifest.yaml" --base-model "$BASE_MODEL" \
  --output "$WORK/train_ckpt" --untrained-loop-heads > "$RUN_DIR/compose_train.log" 2>&1
tail -n 5 "$RUN_DIR/compose_train.log"

if [ "${READ_PROBE:-1}" = 1 ]; then
  log "2b/5 which layer carries the grid? (linear read probe per layer)"
  py tutorials/scripts/loop_sudoku/read_probe.py "$WORK/train_ckpt" "$WORK/data" "$RUN_DIR/read_probe.json" || true
fi

log "3/5 train the loop head on $GPUS GPU(s)"
train_args=(--model "$WORK/train_ckpt" --data "$WORK/data" --out "$RUN_DIR/train"
  --steps "$STEPS" --batch "$BATCH" --warmup "$WARMUP" --eval-every "$EVAL_EVERY"
  --eval-n "$EVAL_N" --final-eval-n "$FINAL_EVAL_N" --eval-max-iter "$EVAL_MAX_ITER"
  --eval-min-iter "${EVAL_MIN_ITER:-0}" --patience "${PATIENCE:-0}"
  --weight-decay "${WEIGHT_DECAY:-0.001}" --read-dropout "${READ_DROPOUT:-0}")
# RESUME: an earlier run's train/resume.pt; training continues after its last eval.
[ -n "${RESUME:-}" ] && train_args+=(--resume "$RESUME")
# One pod: a standalone torchrun. Drop the job operator's multi-node env first.
run_py() {
  if [ "$GPUS" -gt 1 ]; then
    env -u RANK -u WORLD_SIZE -u MASTER_ADDR -u MASTER_PORT \
      uv run --no-sync torchrun --standalone --nproc_per_node="$GPUS" "$@"
  else
    env -u RANK -u WORLD_SIZE -u LOCAL_RANK uv run --no-sync python "$@"
  fi
}
run_py tutorials/scripts/loop_sudoku/train_sudoku.py "${train_args[@]}"

log "4/5 compose the final checkpoint with the trained head"
py tutorials/scripts/loop_sudoku/make_slot.py --base "$BASE_MODEL" --out "$WORK/slot_final" \
  --head "$RUN_DIR/train/loop_head.best.safetensors" "${read_layer_args[@]}"
cp -r "$WORK/slot_final/sudoku" "$RUN_DIR/slot"
py -m granite_switch.composer.compose_granite_switch \
  --adapters "$WORK/slot_final/manifest.yaml" --base-model "$BASE_MODEL" \
  --output "$WORK/final_ckpt" > "$RUN_DIR/compose_final.log" 2>&1
tail -n 5 "$RUN_DIR/compose_final.log"

log "5/5 evaluate the composed checkpoint as served (HF), and save it"
# The periodic evals (head selection) used test puzzles [0, EVAL_N); this slice
# starts after them, as the trainer's final test does.
n=${COMPOSED_EVAL_N:-20000}
if [ "${FINAL_EVAL_N:-0}" -gt 0 ] && [ "$FINAL_EVAL_N" -lt "$n" ]; then n=$FINAL_EVAL_N; fi
run_py tutorials/scripts/loop_sudoku/eval_head.py "$WORK/final_ckpt" "$WORK/data" composed --first "$EVAL_N" --count "$n" \
  --max-iter "$EVAL_MAX_ITER" --min-iter "${EVAL_MIN_ITER:-0}" --tag COMPOSED_EVAL --json "$RUN_DIR/composed_eval.json"
if [ "$MODE" = full ]; then
  cp -r "$WORK/final_ckpt" "$RUN_DIR/checkpoint"
fi
log "done: results in $RUN_DIR"
