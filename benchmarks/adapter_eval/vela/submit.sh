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
#   submit.sh stage --release <rev>      a Hugging Face release (releases in
#   submit.sh bench --release <rev> ...  adapters.yaml, named by its revision):
#   submit.sh reference --release <rev>  fetch it into its folder, run it as
#                                  published (served by its granite-switch
#                                  version from PyPI), run its library
#                                  adapters under HF + PEFT
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
    sed -n '4,29p' "$0" | sed 's/^# \{0,1\}//'
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
# Models kept off the public page (common.PRIVATE_MODELS_ENV): defined in a
# file of their own here, known to every job, never published.
if [[ -z "${ADAPTER_BENCH_PRIVATE_MODELS:-}" && -f "$LOCAL_DIR/models.private.yaml" ]]; then
    ADAPTER_BENCH_PRIVATE_MODELS=$LOCAL_DIR/models.private.yaml
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
MODEL=""
SCRIPT=""
RELEASE="" REF=""
case "$MODE" in
bench)
    # A release names its own serving commit: then there is no ref.
    [[ $# -gt 0 ]] || usage
    if [[ "$1" != --* ]]; then
        REF=$1
        shift
    fi
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
    --release) RELEASE=$2 && shift 2 ;;
    *) usage ;;
    esac
done
if [[ -n "$RELEASE" ]]; then
    case "$MODE" in
    stage | bench | reference) ;;
    *) die "--release is for stage, bench and reference" ;;
    esac
    [[ -z "$REF" ]] || die "a release is served by its PyPI version: give no ref"
elif [[ "$MODE" == bench && -z "$REF" ]]; then
    usage
fi

# --- the cluster login --------------------------------------------------------

# The jobs are AppWrappers (the chart renders one): the cluster oc is logged in
# to must run them, and the login may create them in NAMESPACE. Otherwise a
# job is refused ("no matches for kind AppWrapper"), or following one finds
# nothing. Asks only what any login may ask.
JOB_API=workload.codeflare.dev
check_login() {
    local server out
    server=$(oc whoami --show-server 2>/dev/null || true)
    [[ -n "$server" ]] || die "log in to the cluster: oc login (if its host does not resolve, connect the VPN)"
    # A cluster that does not tell its API groups (Forbidden) is left to the
    # rights check below.
    if ! out=$(oc get --raw "/apis/$JOB_API" --request-timeout=30s 2>&1 >/dev/null) &&
        [[ "$out" != *Forbidden* ]]; then
        if [[ "$out" == *NotFound* ]]; then
            die "oc is logged in to $server, which runs no AppWrappers: not the jobs' cluster. Switch to its login (oc config get-contexts, then oc config use-context <name>), or log in to it (oc login)"
        fi
        die "oc cannot reach $server (${out##*$'\n'}): its login may have expired, or the VPN is off. Log in again (oc login)"
    fi
    oc auth can-i create "appwrappers.$JOB_API" -n "$NAMESPACE" --request-timeout=30s >/dev/null 2>&1 ||
        die "the login to $server cannot create jobs in $NAMESPACE: log in to the cluster that holds it (oc login)"
}

# --- follow a submitted job ----------------------------------------------------

# Whether the job is still on the cluster: yes, no, or unknown (no answer).
job_exists() {
    local out
    if out=$(oc get appwrapper "$JOB" -n "$NAMESPACE" -o name 2>&1); then
        echo yes
    elif [[ "$out" == *NotFound* ]]; then
        echo no
    else
        echo unknown
    fi
}
# Stop: the job is no longer on the cluster, so there is nothing to follow. A
# succeeded job is deleted once its log is saved; a failed one is kept a day,
# or less.
gone() {
    local log=$LOCAL_DIR/logs/$JOB.log saved="no log of it is saved here"
    [[ -f "$log" ]] && saved="its log is saved: $log"
    die "$JOB is no longer on the cluster (it finished, or was deleted); $saved"
}

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
            if [[ "$(job_exists)" == no ]]; then gone; fi
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
        if [[ -n "${RELEASE:-}" ]]; then
            publish extract "$LOG" --kind release --out "$out"
            publish_run merge-release "$out" "release ${RELEASE:0:8}"
        else
            publish extract "$LOG" --out "$out"
            publish_run merge "$out" "${SHA:0:8}"
        fi
        ;;
    reference)
        if [[ -n "${RELEASE:-}" ]]; then
            publish extract "$LOG" --kind release-reference --out "$out"
            publish_run merge-release-reference "$out" "release ${RELEASE:0:8} reference"
        else
            publish extract "$LOG" --kind reference --out "$out"
            publish_run merge-reference "$out" reference
        fi
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
    check_login
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
# A release is of one model, and runs in a folder of its own, served by its
# granite-switch version from PyPI.
if [[ -n "$RELEASE" ]]; then
    RELEASE_INFO=$(local_py -c 'import sys
from benchmarks.adapter_eval.common import load_spec
r = load_spec().release(sys.argv[1])
print(r.model, r.revision, r.repo.rsplit("/", 1)[-1], r.serve_pypi,
      r.reference_from or "-")' "$RELEASE") ||
        die "unknown release $RELEASE"
    read -r REL_MODEL RELEASE REL_NAME REL_PYPI REL_REFERENCE <<<"$RELEASE_INFO"
    [[ -z "$MODEL" || "$MODEL" == "$REL_MODEL" ]] || die "that release is of $REL_MODEL"
    MODEL=$REL_MODEL
    # Its reference columns may be another release's, the same adapters on the
    # same base: those are never run for it.
    if [[ "$MODE" == reference && "$REL_REFERENCE" != - ]]; then
        say "release ${RELEASE:0:8} shows release ${REL_REFERENCE:0:8}'s reference columns" \
            "(reference_from in adapters.yaml): nothing to run"
        exit 0
    fi
fi
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
    # The source roots and SR_REF keep the default model's when not suffixed.
    for name in BENCH_ROOT BASE_MODEL_PATH SELECTION_FILE ADAPTER_SOURCE_ROOTS \
        EVAL_SOURCE_ROOTS SR_REF; do
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
if [[ -n "$RELEASE" ]]; then
    : "${RELEASE_ROOT:?set RELEASE_ROOT in local.env: where releases are staged}"
    # Staging copies the model's eval sets from its bench root.
    export EVAL_FROM=$BENCH_ROOT
    export BENCH_ROOT=$RELEASE_ROOT/$MODEL/$REL_NAME/$RELEASE
    say "release $REL_NAME@${RELEASE:0:8}, in $BENCH_ROOT"
fi
say "model $MODEL"
if local_py -c 'import sys
from benchmarks.adapter_eval.common import load_spec
sys.exit(0 if load_spec(model=sys.argv[1]).model.private else 1)' "$MODEL"; then
    PUBLISH=""
    PRIVATE_MODEL=1
    say "$MODEL is private: its results stay in $LOCAL_DIR/results, off the page"
fi

STAMP=$(date -u +%m%d%H%M)
RUN_TS=$(date -u +%Y%m%dT%H%M%SZ)
render_args=(--mode "$MODE" --run-ts "$RUN_TS" --model "$MODEL")
# The model's own pod size for this mode (resources in adapters.yaml).
RESOURCES_JSON=$(local_py -c 'import json, sys
from benchmarks.adapter_eval.common import load_spec
print(json.dumps(load_spec(model=sys.argv[1]).model.resources_for(sys.argv[2])))' "$MODEL" "$MODE")
if [[ "$RESOURCES_JSON" != "{}" ]]; then render_args+=(--resources "$RESOURCES_JSON"); fi
if [[ -n "${PRIVATE_MODEL:-}" ]]; then
    render_args+=(--private-models "$ADAPTER_BENCH_PRIVATE_MODELS")
fi
# Only jobs that score with the LLM judge get its endpoint and key.
case "$MODE" in
bench | reference | rescore)
    if local_py -c 'import sys
from benchmarks.adapter_eval.common import load_spec
sys.exit(0 if load_spec(model=sys.argv[1]).needs_judge else 1)' "$MODEL"; then
        render_args+=(--judge)
    fi
    ;;
esac
if [[ -n "$RELEASE" ]]; then
    render_args+=(--release "$RELEASE")
    JOB="$JOB_PREFIX-release-$MODE-${RELEASE:0:8}-$STAMP"
    # Served by its granite-switch version from PyPI, as its users install it.
    if [[ "$MODE" == bench ]]; then
        render_args+=(--pypi "$REL_PYPI")
        say "served by granite-switch $REL_PYPI from PyPI"
    fi
    if [[ "$MODE" != stage && -z "$NO_CACHE$LIMIT$ONLY" ]]; then
        check=check-release
        if [[ "$MODE" == reference ]]; then check=check-release-reference; fi
        if publish "$check" "$RELEASE"; then
            say "release ${RELEASE:0:8} is cached; nothing to run (--no-cache to re-run)"
            exit 0
        fi
    fi
    if [[ -n "$LIMIT" ]]; then render_args+=(--limit "$LIMIT"); fi
    if [[ -n "$ONLY" ]]; then render_args+=(--only "$ONLY"); fi
    if [[ -n "$EXTRA_ARGS" ]]; then render_args+=("--extra-args=$EXTRA_ARGS"); fi
    if [[ -n "$REPLACE" ]]; then render_args+=(--replace); fi
elif [[ "$MODE" == bench ]]; then
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

check_login
mkdir -p "$LOCAL_DIR/jobs"
{
    printf 'MODE=%q\nSHA=%q\nLIMIT=%q\nONLY=%q\n' "$MODE" "$SHA" "$LIMIT" "$ONLY"
    printf 'PUBLISH=%q\nK8S=%q\nMODEL=%q\nSUFFIX=%q\n' "$PUBLISH" "$K8S" "$MODEL" "$SUFFIX"
    printf 'RELEASE=%q\n' "$RELEASE"
} >"$LOCAL_DIR/jobs/$JOB.env"
oc create -n "$NAMESPACE" -f "$K8S"
say "submitted $JOB"
follow
collect
finish
