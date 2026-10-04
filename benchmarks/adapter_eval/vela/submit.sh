#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Run the adapter benchmark on Vela from this machine (see docs/ADAPTER_BENCHMARK.md).
#
#   submit.sh discover             list adapter checkpoints and eval sets on the mount
#   submit.sh stage [--replace]    copy the selection's picks into the bench root
#   submit.sh bench <ref> [--limit N] [--only a,b] [--no-cache] [--no-publish]
#                         [--extra-args "..."]
#   submit.sh reference [--limit N] [--only a,b/sr] [--no-cache] [--no-publish]
#                       [--extra-args "..."]
#                                  the HF + PEFT and base-model columns, computed
#                                  once per benchmark version (4 GPUs)
#   submit.sh rescore [--only a,b]  score saved answers again after a scoring
#                                  change (score_version); no generation
#   submit.sh script <ref> <file.py> [--extra-args "..."]
#                                  run one local Python file on a GPU pod, with
#                                  the commit installed (one-off checks)
#   submit.sh fetch <job>          resume following a job and collect its output
#
# Any mode takes --model <id> (an adapters.yaml model; default: its first) and
# --dry-run (render the job and stop). Settings come from the gitignored
# local/local.env (start from local.env.example).
#
# Written for bash 3.2 (the macOS default).
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)
LOCAL_DIR=$HERE/local
RENDERED=$HERE/.rendered

usage() {
    sed -n '4,23p' "$0" | sed 's/^# \{0,1\}//'
    exit 2
}
say() { echo "[submit] $*"; }
die() {
    echo "[submit] $*" >&2
    exit 1
}

[[ -f "$LOCAL_DIR/local.env" ]] || die "missing $LOCAL_DIR/local.env (copy local.env.example)"
set -a
# shellcheck source=/dev/null
source "$LOCAL_DIR/local.env"
SELECTION_FILE=${SELECTION_FILE:-$LOCAL_DIR/selection.json}
if [[ -z "${JUDGE_PROMPT_FILE:-}" && -f "$LOCAL_DIR/judge_prompt.txt" ]]; then
    JUDGE_PROMPT_FILE=$LOCAL_DIR/judge_prompt.txt
fi
BASE_INSTRUCTIONS_FILE=${BASE_INSTRUCTIONS_FILE:-$LOCAL_DIR/base_instructions.json}
set +a
: "${NAMESPACE:?set NAMESPACE in local.env}"
JOB_PREFIX=${JOB_PREFIX:-adapter-bench}
CHART=${CHART:-mlbatch/pytorchjob-generator}

# The harness's local tools only need Python and PyYAML.
local_py() {
    # shellcheck disable=SC2086  # LOCAL_PYTHON is a command line on purpose
    (cd "$REPO" && ${LOCAL_PYTHON:-uv run --quiet --no-project --with pyyaml python} "$@")
}
publish() { local_py -m benchmarks.adapter_eval.publish "$@"; }

# --- arguments ---------------------------------------------------------------

MODE=${1:-}
[[ $# -gt 0 ]] && shift
SHA="" LIMIT="" ONLY="" EXTRA_ARGS="" NO_CACHE="" PUBLISH=1 REPLACE="" DRY_RUN="" JOB=""
MODEL=""
SCRIPT=""
case "$MODE" in
bench)
    [[ $# -gt 0 ]] || usage
    REF=$1
    shift
    ;;
script)
    [[ $# -gt 1 ]] || usage
    REF=$1
    SCRIPT=$2
    shift 2
    [[ -f "$SCRIPT" ]] || die "no such file: $SCRIPT"
    ;;
fetch)
    [[ $# -gt 0 ]] || usage
    JOB=$1
    shift
    ;;
discover | stage | reference | rescore) ;;
*) usage ;;
esac
while [[ $# -gt 0 ]]; do
    case "$1" in
    --limit) LIMIT=$2 && shift 2 ;;
    --only) ONLY=$2 && shift 2 ;;
    --extra-args) EXTRA_ARGS=$2 && shift 2 ;;
    --no-cache) NO_CACHE=1 && shift ;;
    --no-publish) PUBLISH="" && shift ;;
    --replace) REPLACE=1 && shift ;;
    --dry-run) DRY_RUN=1 && shift ;;
    --model) MODEL=$2 && shift 2 ;;
    *) usage ;;
    esac
done

# --- follow a submitted job ----------------------------------------------------

pod_name() {
    oc get pods -n "$NAMESPACE" -o name 2>/dev/null | grep "^pod/${JOB}-" | head -n 1 || true
}
pod_phase() {
    oc get "$1" -n "$NAMESPACE" -o jsonpath='{.status.phase}' 2>/dev/null || true
}

follow() {
    local pod="" phase="" last="" since="" waited=0
    say "waiting for $JOB (Ctrl-C is safe; resume with: submit.sh fetch $JOB)"
    while [[ -z "$pod" ]]; do
        pod=$(pod_name)
        if [[ -z "$pod" ]]; then
            if ((waited % 300 == 0)); then say "no pod yet after ${waited}s (queued?)"; fi
            sleep 30
            waited=$((waited + 30))
        fi
    done
    while :; do
        phase=$(pod_phase "$pod")
        if [[ "$phase" != "$last" ]]; then say "$pod: ${phase:-unknown}"; fi
        last=$phase
        case "$phase" in
        Pending | "") sleep 20 ;;
        Running)
            # A dropped stream is resumed; the full log is fetched again below.
            # shellcheck disable=SC2086
            oc logs -f $since "$pod" -n "$NAMESPACE" || true
            since="--since=20s"
            sleep 5
            ;;
        *) break ;;
        esac
    done
    mkdir -p "$LOCAL_DIR/logs"
    LOG=$LOCAL_DIR/logs/$JOB.log
    oc logs "$pod" -n "$NAMESPACE" >"$LOG"
    POD_PHASE=$phase
    say "pod $phase; log saved to $LOG"
}

# Merge a run's results file into the page data, unless it must not be published.
publish_run() {
    local merge_cmd=$1 out=$2 label=$3
    if [[ "$POD_PHASE" != Succeeded ]]; then
        say "pod did not succeed; not publishing"
    elif [[ -n "$LIMIT" ]]; then
        say "--limit run; not publishing"
    elif [[ -z "$PUBLISH" ]]; then
        say "--no-publish; results are in $out"
    else
        publish "$merge_cmd" "$out"
        publish render
        say "page updated. To record it:"
        say "  git add docs/benchmarks && git commit -s -m 'Adapter benchmark: $label'"
    fi
}

collect() {
    local out=$LOCAL_DIR/results/$JOB.json
    case "$MODE" in
    discover)
        local found=$LOCAL_DIR/discovery${SUFFIX:-}.json
        local draft=$LOCAL_DIR/selection${SUFFIX:-}.draft.json
        publish extract "$LOG" --kind discovery --out "$found"
        local_py -c 'import json, sys
d = json.load(open(sys.argv[1]))
json.dump(d["draft_selection"], open(sys.argv[2], "w"), indent=2, sort_keys=True)' \
            "$found" "$draft"
        say "draft selection: $draft (review, then save it without .draft)"
        ;;
    stage)
        publish extract "$LOG" --kind stage --out "$LOCAL_DIR/stage/$JOB.json"
        ;;
    bench)
        publish extract "$LOG" --out "$out"
        publish_run merge "$out" "${SHA:0:8}"
        ;;
    reference)
        publish extract "$LOG" --kind reference --out "$out"
        publish_run merge-reference "$out" reference
        ;;
    rescore)
        publish extract "$LOG" --kind rescore --out "$out"
        publish_run merge-rescore "$out" "rescore $MODEL"
        ;;
    esac
}

finish() {
    if [[ "$POD_PHASE" == Succeeded ]]; then
        oc delete -f "$K8S" -n "$NAMESPACE" >/dev/null && say "deleted $JOB"
    else
        say "kept $JOB for inspection (removed after 24h, or: oc delete -f $K8S)"
        exit 1
    fi
}

if [[ "$MODE" == fetch ]]; then
    [[ -f "$LOCAL_DIR/jobs/$JOB.env" ]] || die "no record of $JOB in $LOCAL_DIR/jobs"
    # shellcheck source=/dev/null
    source "$LOCAL_DIR/jobs/$JOB.env"
    follow
    collect
    finish
    exit 0
fi

# --- render and submit ---------------------------------------------------------

# The model's own settings. local.env names another model's with a suffix,
# e.g. BENCH_ROOT__granite_4_2_3b; the plain names are the default model's,
# and are never used for another model.
DEFAULT_MODEL=$(local_py -c 'from benchmarks.adapter_eval.common import load_spec
print(load_spec().model_id)')
MODEL=${MODEL:-$DEFAULT_MODEL}
local_py -c 'import sys
from benchmarks.adapter_eval.common import load_spec
load_spec(model=sys.argv[1])' "$MODEL" || die "unknown model $MODEL"
# Local files of another model get its id: selection.<model>.json, ...
SUFFIX=""
if [[ "$MODEL" != "$DEFAULT_MODEL" ]]; then
    SUFFIX=.$MODEL
    key=$(printf %s "$MODEL" | tr '.-' '__')
    SELECTION_FILE=$LOCAL_DIR/selection$SUFFIX.json
    for name in BENCH_ROOT BASE_MODEL_PATH SELECTION_FILE; do
        var=${name}__$key
        if [[ -n "${!var:-}" ]]; then
            export "$name=${!var}"
        elif [[ "$name" == BENCH_ROOT ]]; then
            die "set $var in local.env: the bench root of $MODEL"
        elif [[ "$name" == BASE_MODEL_PATH ]]; then
            unset BASE_MODEL_PATH
        fi
    done
    # Job names may have at most 50 characters (a chart limit), so the tag
    # leaves out a leading "granite-".
    MODEL_TAG=-$(printf %s "${MODEL#granite-}" | tr '.' '-')
else
    MODEL_TAG=""
fi
say "model $MODEL"

STAMP=$(date -u +%m%d%H%M)
RUN_TS=$(date -u +%Y%m%dT%H%M%SZ)
render_args=(--mode "$MODE" --run-ts "$RUN_TS" --model "$MODEL")
if [[ "$MODE" == bench ]]; then
    SHA=$(git -C "$REPO" rev-parse --verify "$REF^{commit}") || die "unknown ref $REF"
    if ! git -C "$REPO" branch -r --contains "$SHA" 2>/dev/null | grep -q .; then
        say "warning: ${SHA:0:12} is on no remote branch; the pod clones from GitHub"
    fi
    if [[ -z "$NO_CACHE$LIMIT$ONLY" ]] && publish check "$SHA" --model "$MODEL"; then
        say "cache hit for ${SHA:0:12}; nothing to run (--no-cache to re-run)"
        exit 0
    fi
    JOB="$JOB_PREFIX-bench-${SHA:0:8}$MODEL_TAG-$STAMP"
    render_args+=(--sha "$SHA")
    if [[ -n "$LIMIT" ]]; then render_args+=(--limit "$LIMIT"); fi
    if [[ -n "$ONLY" ]]; then render_args+=(--only "$ONLY"); fi
    if [[ -n "$EXTRA_ARGS" ]]; then render_args+=("--extra-args=$EXTRA_ARGS"); fi
elif [[ "$MODE" == reference ]]; then
    if [[ -z "$NO_CACHE$LIMIT$ONLY" ]] && publish check-reference --model "$MODEL"; then
        say "reference columns are cached; nothing to run (--no-cache to re-run)"
        exit 0
    fi
    JOB="$JOB_PREFIX-reference$MODEL_TAG-$STAMP"
    if [[ -n "$LIMIT" ]]; then render_args+=(--limit "$LIMIT"); fi
    if [[ -n "$ONLY" ]]; then render_args+=(--only "$ONLY"); fi
    if [[ -n "$EXTRA_ARGS" ]]; then render_args+=("--extra-args=$EXTRA_ARGS"); fi
elif [[ "$MODE" == rescore ]]; then
    mkdir -p "$LOCAL_DIR/rescore"
    TARGETS=$LOCAL_DIR/rescore/targets$SUFFIX.json
    targets_args=(--model "$MODEL" --out "$TARGETS")
    if [[ -n "$ONLY" ]]; then targets_args+=(--only "$ONLY"); fi
    status=0
    publish rescore-targets "${targets_args[@]}" || status=$?
    if ((status == 3)); then
        say "nothing to score again"
        exit 0
    fi
    ((status == 0)) || die "could not list the cells to score again"
    JOB="$JOB_PREFIX-rescore$MODEL_TAG-$STAMP"
    render_args+=(--targets "$TARGETS")
elif [[ "$MODE" == script ]]; then
    SHA=$(git -C "$REPO" rev-parse --verify "$REF^{commit}") || die "unknown ref $REF"
    JOB="$JOB_PREFIX-script-${SHA:0:8}$MODEL_TAG-$STAMP"
    render_args+=(--sha "$SHA" --script "$SCRIPT")
    if [[ -n "$EXTRA_ARGS" ]]; then render_args+=("--extra-args=$EXTRA_ARGS"); fi
else
    JOB="$JOB_PREFIX-$MODE$MODEL_TAG-$STAMP"
    if [[ -n "$REPLACE" ]]; then render_args+=(--replace); fi
fi
JOB=$(echo "$JOB" | tr '[:upper:]_' '[:lower:]-')
if ((${#JOB} > 50)); then die "job name $JOB is longer than the chart's 50 characters"; fi

VALUES=$RENDERED/$JOB.values.yaml
K8S=$RENDERED/$JOB.yaml
python3 "$HERE/render_job.py" "${render_args[@]}" --job-name "$JOB" --out "$VALUES"
helm template -f "$VALUES" "$CHART" >"$K8S"
say "rendered $K8S ($(wc -c <"$K8S" | tr -d ' ') bytes)"
if [[ -n "$DRY_RUN" ]]; then
    say "--dry-run; not submitting"
    exit 0
fi

mkdir -p "$LOCAL_DIR/jobs"
{
    printf 'MODE=%q\nSHA=%q\nLIMIT=%q\nONLY=%q\n' "$MODE" "$SHA" "$LIMIT" "$ONLY"
    printf 'PUBLISH=%q\nK8S=%q\nMODEL=%q\nSUFFIX=%q\n' "$PUBLISH" "$K8S" "$MODEL" "$SUFFIX"
} >"$LOCAL_DIR/jobs/$JOB.env"
oc create -n "$NAMESPACE" -f "$K8S"
say "submitted $JOB"
follow
collect
finish
