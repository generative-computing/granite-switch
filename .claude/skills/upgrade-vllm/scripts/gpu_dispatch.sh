#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Dispatch gpu-tests.yaml at a BRANCH ref, wait for it, and surface the failures.
#
# Why not just comment /gpu-test? Because gpu_test_command.sh dispatches with
# `--ref <default branch>`, so main's matrix decides which legs run. A PR that adds
# a vllm31 leg would still be tested as vllm26+vllm30. Dispatching with
# `--ref <this branch>` runs THIS branch's workflow file, so new legs take effect
# immediately. The commit still gets checked out from `sha` either way.
#
# no_cache is forced on. The result cache is keyed on sha+leg and the test pod pins
# vllm/vllm-openai:latest, so a cached pass answers "what did this commit do" when
# a version bump is asking "is this still true today".
set -uo pipefail

WORKFLOW=gpu-tests.yaml
REF=""
SUITE=full
SHA=""
PR=""
WAIT=0
POLL=60

usage() {
    cat <<'EOF'
usage: gpu_dispatch.sh --ref BRANCH [--suite full|short|dev|audio|multi|eval]
                       [--sha SHA] [--pr N] [--wait] [--poll SECONDS]

  --ref    branch whose workflow file AND commit to use (required)
  --suite  test scope (default: full)
  --sha    commit to check out on the cluster (default: origin/<ref> head)
  --pr     PR number to comment on (default: derived by the workflow)
  --wait   block until the run finishes, then report per-leg results
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --ref) REF="$2"; shift 2 ;;
        --suite) SUITE="$2"; shift 2 ;;
        --sha) SHA="$2"; shift 2 ;;
        --pr) PR="$2"; shift 2 ;;
        --wait) WAIT=1; shift ;;
        --poll) POLL="$2"; shift 2 ;;
        -h | --help) usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; usage; exit 2 ;;
    esac
done

[ -n "$REF" ] || { echo "--ref is required" >&2; usage; exit 2; }
command -v gh >/dev/null 2>&1 || { echo "gh is not installed" >&2; exit 1; }

REPO="$(gh repo view --json nameWithOwner -q .nameWithOwner)"

# The sha must be on the REMOTE: the cluster clones from GitHub, so a
# locally-committed-but-unpushed sha produces a pod that dies on checkout.
if [ -z "$SHA" ]; then
    git fetch -q origin "$REF" || { echo "branch $REF is not on origin -- push first" >&2; exit 1; }
    SHA="$(git rev-parse "origin/$REF")"
fi
if ! gh api "repos/$REPO/commits/$SHA" -q .sha >/dev/null 2>&1; then
    echo "sha $SHA is not on $REPO -- push the branch before dispatching" >&2
    exit 1
fi

# A suite with no `case` arm fails at "Resolve test scope" after the pod is already
# queued. Catch it here, against the file on the ref we are about to dispatch.
if ! git show "origin/$REF:.github/workflows/$WORKFLOW" 2>/dev/null |
    grep -q "options:.*\b$SUITE\b"; then
    echo "suite '$SUITE' is not in $WORKFLOW's options list on origin/$REF." >&2
    echo "Add it to the options line AND give it a case arm in 'Resolve test scope'." >&2
    exit 1
fi

echo "dispatching $WORKFLOW"
echo "  ref   : $REF   (this branch's workflow file -- new matrix legs apply)"
echo "  sha   : $SHA"
echo "  suite : $SUITE"
echo "  cache : disabled"

BEFORE="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
ARGS=(-f "sha=$SHA" -f "suite=$SUITE" -f "no_cache=true")
[ -n "$PR" ] && ARGS+=(-f "pr_number=$PR")

gh workflow run "$WORKFLOW" --ref "$REF" "${ARGS[@]}" || {
    echo "dispatch failed. If this says the workflow has no workflow_dispatch trigger," >&2
    echo "the file is missing on origin/$REF -- push it there first." >&2
    exit 1
}

# `gh workflow run` prints nothing useful, so find the run we just created: newest
# workflow_dispatch on this branch created at or after BEFORE.
RUN_ID=""
for _ in $(seq 1 30); do
    sleep 4
    RUN_ID="$(
        gh run list --workflow "$WORKFLOW" --branch "$REF" --event workflow_dispatch \
            --limit 10 --json databaseId,createdAt \
            -q "[.[] | select(.createdAt >= \"$BEFORE\")] | sort_by(.createdAt) | last | .databaseId"
    )"
    [ -n "$RUN_ID" ] && [ "$RUN_ID" != "null" ] && break
done

if [ -z "$RUN_ID" ] || [ "$RUN_ID" = "null" ]; then
    echo "dispatched, but could not identify the run id. Find it with:" >&2
    echo "  gh run list --workflow $WORKFLOW --branch $REF" >&2
    exit 1
fi

URL="$(gh run view "$RUN_ID" --json url -q .url)"
echo "run   : $URL"
[ "$WAIT" -eq 1 ] || exit 0

# Each leg caps at 420 minutes and the two legs run in parallel, so ~7.5h is the
# real ceiling for a `full` run.
echo "waiting (legs cap at 420 min each; polling every ${POLL}s)"
while :; do
    STATUS="$(gh run view "$RUN_ID" --json status -q .status)"
    [ "$STATUS" = "completed" ] && break
    printf '.'
    sleep "$POLL"
done
echo

CONCLUSION="$(gh run view "$RUN_ID" --json conclusion -q .conclusion)"
echo "=== run $RUN_ID: $CONCLUSION ==="
gh run view "$RUN_ID" --json jobs \
    -q '.jobs[] | "  \(.conclusion // .status)\t\(.name)"'

# A GREEN CONCLUSION IS NOT EVIDENCE THAT ANYTHING RAN. pytest exits 0 when every
# test skips, the pod prints "ALL GPU TESTS PASSED", and the workflow posts
# success. Observed on run 39299824463: the eval suite collected 4 items, skipped
# all 4 for want of staged checkpoints, and reported PASSED. An upgrade that
# stamps a per-version checkmark from that is claiming coverage it does not have,
# which is the exact failure this skill exists to avoid.
#
# So on success, read the log anyway and require a non-zero passed count.
TMP="$(mktemp -d)"
echo
echo "downloading log artifacts to $TMP"
gh run download "$RUN_ID" -D "$TMP" 2>/dev/null || echo "  (no artifacts -- the job may have died before the upload step)"

if [ "$CONCLUSION" = "success" ]; then
    shopt -s nullglob
    VACUOUS=0
    SEEN=0
    for log in "$TMP"/*/*.log; do
        SEEN=$((SEEN + 1))
        LEG_NAME=$(basename "$(dirname "$log")")
        # pytest's summary line, e.g. "4 skipped in 0.03s" or "12 passed, 1 skipped".
        if grep -qE "^=+ .*[0-9]+ passed" "$log"; then
            echo "  ok   $LEG_NAME -- tests actually ran"
        else
            SUMMARY=$(grep -oE "[0-9]+ (passed|failed|skipped|error)[^=]*" "$log" | tail -1)
            echo "  !!   $LEG_NAME -- GREEN BUT NOTHING PASSED: ${SUMMARY:-no pytest summary found}"
            VACUOUS=1
        fi
    done
    if [ "$SEEN" -eq 0 ]; then
        echo
        echo "run was green but published no log artifact -- cannot confirm tests ran."
        echo "Treat this leg as UNPROVEN, not as a pass."
        exit 1
    fi
    if [ "$VACUOUS" -ne 0 ]; then
        echo
        echo "VACUOUS PASS for suite '$SUITE': the run is green because tests were"
        echo "skipped, not because they succeeded. Do NOT record a checkmark for this."
        echo "For the eval suite this means the checkpoints are not staged on the pod:"
        echo "  GSW_EVAL_*_DIR must point at a staged checkpoint (see tests/eval/_eval_core.py)."
        exit 1
    fi
    echo
    echo "all legs green for suite '$SUITE', and every leg ran real tests."
    exit 0
fi

# Failure: the published log artifact is the readable one (clean_log.sh has already
# stripped the noise). Triage every leg's artifact.

shopt -s nullglob
for log in "$TMP"/*/*.log; do
    SIZE=$(wc -c <"$log" | tr -d ' ')
    echo
    echo "--- $(basename "$(dirname "$log")") (${SIZE} bytes) ---"
    # THE signature worth knowing: under ~1.1 KB means pytest never started. Nearly
    # always an unresolvable --dep-group, i.e. a dependency group this branch
    # deleted or never added -- NOT a test failure. Reading it as a test failure
    # sends you editing tests for hours.
    if [ "$SIZE" -lt 1100 ]; then
        echo "  !! suite never started (log < 1.1 KB)."
        echo "  !! Check the leg's dep_group exists in pyproject.toml on origin/$REF:"
        echo "  !!   git show origin/$REF:pyproject.toml | grep -n '^dev'"
        cat "$log"
        continue
    fi
    grep -E "^(FAILED|ERROR)|^E  |assert|Error:|== .* failed" "$log" | head -40
    echo "  (full log: $log)"
done

echo
echo "run $CONCLUSION -- $URL"
exit 1
