#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# In-pod entry point for the adapter benchmark jobs rendered by render_job.py.
# Runs from the unpacked harness; everything it needs comes from environment
# variables set in the job (ADAPTER_BENCH_*).
#
#   discover  list adapter checkpoints and eval files under the source roots
#   stage     copy the selection shipped in extra/ into the bench root
#   bench     clone + install the commit, then run run_benchmark.py
#   reference install pinned torch / transformers / peft in their own
#             virtualenv, then run reference.py (no commit involved)
#   script    clone + install the commit, then run the Python file shipped as
#             extra/script.py (a one-off check, e.g. from scratch/)
#   rescore   score saved answers again (the cells in extra/targets.json), with
#             no commit and no generation
set -euo pipefail

HARNESS=$(cd "$(dirname "$0")/../../.." && pwd)
MODE=${ADAPTER_BENCH_MODE:?}
# The adapters.yaml model; empty means its first (default) model.
MODEL=${ADAPTER_BENCH_MODEL:-}
MODEL_ARGS=()
if [[ -n "$MODEL" ]]; then MODEL_ARGS=(--model "$MODEL"); fi
echo "[pod] mode=${MODE} model=${MODEL:-default} harness=${ADAPTER_BENCH_HARNESS_SHA:-?} dirty=${ADAPTER_BENCH_HARNESS_DIRTY:-?} run=${RUN_TS:-?}"
nvidia-smi -L 2>/dev/null || true

if ! command -v uv >/dev/null; then
    python3 -m pip install --quiet --user uv || curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# The harness's own tools only need Python and PyYAML.
harness_py() {
    (cd "$HARNESS" && uv run --quiet --no-project --python 3.12 --with pyyaml python "$@")
}

# Clone and install the benchmarked commit into $REPO.
install_commit() {
    SHA=${ADAPTER_BENCH_COMMIT:?}
    REPO=/workspace/granite-switch
    git clone --quiet --filter=blob:none https://github.com/generative-computing/granite-switch.git "$REPO"
    # A pull request's commit can be on no branch (a fork's): fetch it by sha.
    if ! git -C "$REPO" cat-file -e "$SHA^{commit}" 2>/dev/null; then
        git -C "$REPO" fetch --quiet origin "$SHA"
    fi
    git -C "$REPO" checkout --quiet --detach "$SHA"
    # shellcheck disable=SC2086  # the sync args are a word list on purpose
    (cd "$REPO" && uv sync --frozen ${ADAPTER_BENCH_UV_SYNC_ARGS:---group dev})
}

case "$MODE" in
discover)
    : "${ADAPTER_BENCH_ADAPTER_SOURCES:?set ADAPTER_SOURCE_ROOTS in local.env}"
    args=()
    for r in ${ADAPTER_BENCH_ADAPTER_SOURCES:-}; do args+=(--adapter-root "$r"); done
    for r in ${ADAPTER_BENCH_EVAL_SOURCES:-}; do args+=(--eval-root "$r"); done
    harness_py -m benchmarks.adapter_eval.stage discover "${args[@]}" ${MODEL_ARGS[@]+"${MODEL_ARGS[@]}"}
    ;;
stage)
    args=(--selection "$HARNESS/extra/selection.json" --bench-root "${ADAPTER_BENCH_ROOT:?}")
    if [[ -f "$HARNESS/extra/judge_prompt.txt" ]]; then
        args+=(--judge-prompt "$HARNESS/extra/judge_prompt.txt")
    fi
    if [[ -n "${ADAPTER_BENCH_STAGE_REPLACE:-}" ]]; then args+=(--replace); fi
    harness_py -m benchmarks.adapter_eval.stage apply "${args[@]}" ${MODEL_ARGS[@]+"${MODEL_ARGS[@]}"}
    ;;
bench)
    install_commit
    args=(
        --repo-dir "$REPO"
        --bench-root "${ADAPTER_BENCH_ROOT:?}"
        --work-dir "${ADAPTER_BENCH_WORK_ROOT:?}/${MODEL:-default}/${SHA:0:12}/${RUN_TS:-run}"
        --model-dir /workspace/models
        # The switching runs' synthetic adapters, built once per model.
        --switching-cache "${ADAPTER_BENCH_WORK_ROOT:?}/synthetic/${MODEL:-default}"
    )
    if [[ -n "$MODEL" ]]; then args+=(--model "$MODEL"); fi
    if [[ -n "${ADAPTER_BENCH_LIMIT:-}" ]]; then args+=(--limit "$ADAPTER_BENCH_LIMIT"); fi
    if [[ -n "${ADAPTER_BENCH_ONLY:-}" ]]; then args+=(--only "$ADAPTER_BENCH_ONLY"); fi
    if [[ -n "${ADAPTER_BENCH_BASE_MODEL:-}" ]]; then
        args+=(--base-model "$ADAPTER_BENCH_BASE_MODEL")
    fi
    # shellcheck disable=SC2086
    cd "$HARNESS" && "$REPO/.venv/bin/python" -m benchmarks.adapter_eval.run_benchmark \
        "${args[@]}" ${ADAPTER_BENCH_EXTRA_ARGS:-}
    ;;
reference)
    # Pinned, so the reference columns change only with reference_version
    # (adapters.yaml). Keep in step with reference_version.
    uv venv --quiet --python 3.12 /workspace/refenv
    uv pip install --quiet --python /workspace/refenv/bin/python \
        torch==2.10.0 transformers==5.8.1 peft==0.19.1 accelerate==1.13.0 pyyaml
    # The SR model code, shipped by render_job.py at a pinned commit.
    PYTHONPATH=$HARNESS
    if [[ -n "${ADAPTER_BENCH_SR_TGZ:-}" ]]; then
        mkdir -p /workspace/sr_ref
        printf %s "$ADAPTER_BENCH_SR_TGZ" | base64 -d | tar xzf - -C /workspace/sr_ref
        PYTHONPATH=$PYTHONPATH:/workspace/sr_ref/src
    fi
    export PYTHONPATH
    args=(
        --bench-root "${ADAPTER_BENCH_ROOT:?}"
        --work-dir "${ADAPTER_BENCH_WORK_ROOT:?}/reference/${MODEL:-default}/${RUN_TS:-run}"
        --model-dir /workspace/models
    )
    if [[ -n "$MODEL" ]]; then args+=(--model "$MODEL"); fi
    if [[ -n "${ADAPTER_BENCH_LIMIT:-}" ]]; then args+=(--limit "$ADAPTER_BENCH_LIMIT"); fi
    if [[ -n "${ADAPTER_BENCH_ONLY:-}" ]]; then args+=(--only "$ADAPTER_BENCH_ONLY"); fi
    if [[ -n "${ADAPTER_BENCH_BASE_MODEL:-}" ]]; then
        args+=(--base-model "$ADAPTER_BENCH_BASE_MODEL")
    fi
    if [[ -f "$HARNESS/extra/base_instructions.json" ]]; then
        args+=(--base-instructions "$HARNESS/extra/base_instructions.json")
    fi
    # shellcheck disable=SC2086
    cd "$HARNESS" && /workspace/refenv/bin/python -m benchmarks.adapter_eval.reference \
        "${args[@]}" ${ADAPTER_BENCH_EXTRA_ARGS:-}
    ;;
rescore)
    harness_py -m benchmarks.adapter_eval.rescore \
        --targets "$HARNESS/extra/targets.json" \
        --work-root "${ADAPTER_BENCH_WORK_ROOT:?}" \
        --bench-root "${ADAPTER_BENCH_ROOT:?}" ${MODEL_ARGS[@]+"${MODEL_ARGS[@]}"}
    ;;
script)
    install_commit
    export ADAPTER_BENCH_REPO=$REPO
    export ADAPTER_BENCH_WORK_DIR=${ADAPTER_BENCH_WORK_ROOT:?}/scripts/${SHA:0:12}/${RUN_TS:-run}
    export PYTHONPATH=$HARNESS${PYTHONPATH:+:$PYTHONPATH}
    # shellcheck disable=SC2086
    cd "$HARNESS" && "$REPO/.venv/bin/python" extra/script.py ${ADAPTER_BENCH_EXTRA_ARGS:-}
    ;;
*)
    echo "[pod] unknown mode $MODE" >&2
    exit 2
    ;;
esac
