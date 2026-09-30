#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Run the adapter benchmark on Vela from this machine (see docs/ADAPTER_BENCHMARK.md).
#
#   submit.sh discover             list adapter checkpoints and eval sets on the mount
#   submit.sh stage [--replace]    copy local/selection.json's picks into the bench root
#   submit.sh bench <ref> [--limit N] [--only a,b] [--no-cache] [--no-publish]
#                         [--extra-args "..."]
#   submit.sh script <ref> <file.py> [--extra-args "..."]
#                                  run one local Python file on a GPU pod, with
#                                  the commit installed (one-off checks)
#   submit.sh fetch <job>          resume following a job and collect its output
#
# Any mode takes --dry-run: render the job and stop. Settings come from the
# gitignored local/local.env (start from local.env.example).
#
# Written for bash 3.2 (the macOS default).
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)
LOCAL_DIR=$HERE/local
RENDERED=$HERE/.rendered

usage() {
    sed -n '4,16p' "$0" | sed 's/^# \{0,1\}//'
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
discover | stage) ;;
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

collect() {
    case "$MODE" in
    discover)
        publish extract "$LOG" --kind discovery --out "$LOCAL_DIR/discovery.json"
        local_py -c 'import json, sys
d = json.load(open(sys.argv[1]))
json.dump(d["draft_selection"], open(sys.argv[2], "w"), indent=2, sort_keys=True)' \
            "$LOCAL_DIR/discovery.json" "$LOCAL_DIR/selection.draft.json"
        say "draft selection: $LOCAL_DIR/selection.draft.json (review, then save as selection.json)"
        ;;
    stage)
        publish extract "$LOG" --kind stage --out "$LOCAL_DIR/stage/$JOB.json"
        ;;
    bench)
        local out=$LOCAL_DIR/results/$JOB.json
        publish extract "$LOG" --out "$out"
        if [[ "$POD_PHASE" != Succeeded ]]; then
            say "pod did not succeed; not publishing"
        elif [[ -n "$LIMIT" ]]; then
            say "--limit run; not publishing"
        elif [[ -z "$PUBLISH" ]]; then
            say "--no-publish; results are in $out"
        else
            publish merge "$out"
            publish render
            say "page updated. To record it:"
            say "  git add docs/benchmarks && git commit -s -m 'Adapter benchmark: ${SHA:0:8}'"
        fi
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

STAMP=$(date -u +%m%d%H%M)
RUN_TS=$(date -u +%Y%m%dT%H%M%SZ)
render_args=(--mode "$MODE" --run-ts "$RUN_TS")
if [[ "$MODE" == bench ]]; then
    SHA=$(git -C "$REPO" rev-parse --verify "$REF^{commit}") || die "unknown ref $REF"
    if ! git -C "$REPO" branch -r --contains "$SHA" 2>/dev/null | grep -q .; then
        say "warning: ${SHA:0:12} is on no remote branch; the pod clones from GitHub"
    fi
    if [[ -z "$NO_CACHE$LIMIT$ONLY" ]] && publish check "$SHA"; then
        say "cache hit for ${SHA:0:12}; nothing to run (--no-cache to re-run)"
        exit 0
    fi
    JOB="$JOB_PREFIX-bench-${SHA:0:8}-$STAMP"
    render_args+=(--sha "$SHA")
    if [[ -n "$LIMIT" ]]; then render_args+=(--limit "$LIMIT"); fi
    if [[ -n "$ONLY" ]]; then render_args+=(--only "$ONLY"); fi
    if [[ -n "$EXTRA_ARGS" ]]; then render_args+=("--extra-args=$EXTRA_ARGS"); fi
elif [[ "$MODE" == script ]]; then
    SHA=$(git -C "$REPO" rev-parse --verify "$REF^{commit}") || die "unknown ref $REF"
    JOB="$JOB_PREFIX-script-${SHA:0:8}-$STAMP"
    render_args+=(--sha "$SHA" --script "$SCRIPT")
    if [[ -n "$EXTRA_ARGS" ]]; then render_args+=("--extra-args=$EXTRA_ARGS"); fi
else
    JOB="$JOB_PREFIX-$MODE-$STAMP"
    if [[ -n "$REPLACE" ]]; then render_args+=(--replace); fi
fi
JOB=$(echo "$JOB" | tr '[:upper:]_' '[:lower:]-')

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
    printf 'PUBLISH=%q\nK8S=%q\n' "$PUBLISH" "$K8S"
} >"$LOCAL_DIR/jobs/$JOB.env"
oc create -n "$NAMESPACE" -f "$K8S"
say "submitted $JOB"
follow
collect
finish
