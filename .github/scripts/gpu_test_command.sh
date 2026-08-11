#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Handle a /gpu-test* PR comment: verify the commenter holds Maintain/Admin, work
# out which test scope was asked for, then dispatch gpu-tests.yaml against the PR's
# head commit. On rejection, react 👎 and reply naming the requirement.
#
# THIS SCRIPT DOES NOT KNOW THE TEST FAMILIES. It derives the suite name from the
# command instead: `/gpu-test` -> full, `/gpu-test-<x>` -> x. So the list lives in
# exactly one place, gpu-tests.yaml's `suite` input, and adding a family is a
# one-file change there rather than an edit here that is easy to forget -- the old
# shape had a hardcoded case, and forgetting it meant a family that existed in the
# workflow, the mapping and the docs was still told "not a command".
#
# `/gpu-test-<anything>` is NOT accepted. Two layers reject an unknown name:
#
#   1. This script reads the `options:` list out of the checked-out workflow (see
#      WORKFLOW_FILE) and declines locally -- in seconds, on a hosted runner, with
#      a reply listing the families READ FROM THAT LIST so it cannot go stale.
#   2. If that read fails -- someone reformats `options:` into a block list -- the
#      dispatch goes ahead and GitHub's own `type: choice` validation rejects it
#      with a 422, which is caught below. So a reformat costs the nice message,
#      never the enforcement.
#
# Deliberately NOT fully dynamic. With no `options:` the dispatch would succeed and
# the failure would land as a red GPU Tests run, having consumed a runner slot and a
# queue wait, for a typo. There is no upside either: a name with no `options:` entry
# has no mapping arm to run.
#
# The suite NAME is dispatched, never a path list: gpu-tests.yaml owns the mapping.
#
# Deployed to granite-switch as .github/scripts/gpu_test_command.sh.
# It runs on a GitHub-hosted runner (no /opt/gsw), which is why it is checked in
# rather than being baked into the runner image. See
# gpu-test-command.yaml for the full rationale.
#
# This check is fast-fail UX. The authoritative gate is /opt/gsw/check_role.sh
# inside gpu-tests.yaml, which also covers direct workflow_dispatch.
#
# All GitHub-controlled values arrive as positional args from quoted env in the
# workflow — never interpolated into this script — so a crafted login cannot
# inject shell.
#
# Usage: gpu_test_command.sh <actor-login> <pr-number> <comment-id> <comment-body>
# Env:   GH_TOKEN, GITHUB_REPOSITORY, DEFAULT_BRANCH, SCRIPT_DIR
#        WORKFLOW_FILE  optional path to the checked-out gpu-tests.yaml. Enables the
#                       local family check; without it layer 2 above still applies.
set -euo pipefail

ACTOR="${1:?usage: gpu_test_command.sh <actor-login> <pr-number> <comment-id> <comment-body>}"
PR_NUMBER="${2:?missing pr number}"
COMMENT_ID="${3:?missing comment id}"
# May legitimately be empty or multi-line, so no :? guard.
BODY="${4:-}"

REPO="${GITHUB_REPOSITORY:?}"
DEFAULT_BRANCH="${DEFAULT_BRANCH:?}"
SCRIPT_DIR="${SCRIPT_DIR:?}"
WORKFLOW_FILE="${WORKFLOW_FILE:-}"

react() {
  gh api -X POST "repos/${REPO}/issues/comments/${COMMENT_ID}/reactions" \
    -f content="$1" >/dev/null
}

reply() {
  gh api -X POST "repos/${REPO}/issues/${PR_NUMBER}/comments" -f body="$1" >/dev/null
}

# check_role.sh exits non-zero (and prints the role) when not authorized.
if ! ROLE_MSG="$("${SCRIPT_DIR}/check_role.sh" "$ACTOR" 2>&1)"; then
  react '-1'
  reply "@${ACTOR} the GPU test commands require the **Maintain** or **Admin** role. Not launching."
  echo "$ROLE_MSG" >&2
  exit 1
fi

# The families, read from the ONE place they are defined: the `options:` line of
# the `suite` input in the checked-out gpu-tests.yaml. Space-separated, or empty if
# the file is absent or the line is not in flow style -- in which case the dispatch
# below is left to GitHub to validate.
#
# One awk, no pipe: splitting on [ and ] puts the list body in $2, and `exit` stops
# at the first match. (A pipe into head would risk SIGPIPE under `pipefail`.)
FAMILIES=""
if [[ -n "$WORKFLOW_FILE" && -r "$WORKFLOW_FILE" ]]; then
  FAMILIES="$(awk -F'[][]' '
    /^[[:space:]]*options:[[:space:]]*\[/ { gsub(/[ ,]+/, " ", $2); print $2; exit }
  ' "$WORKFLOW_FILE" 2>/dev/null || true)"
fi

# Render the families back as the commands a human types: `full` is the bare
# /gpu-test, everything else is suffixed. Used only in the decline message.
usage_list() {
  local f out=""
  for f in $FAMILIES; do
    if [[ "$f" == "full" ]]; then out="${out}\`/gpu-test\` "; else out="${out}\`/gpu-test-${f}\` "; fi
  done
  printf '%s' "$out"
}

# Exit 0 throughout: a mistyped command is user error, not a broken workflow, and a
# red X on the launcher would send someone hunting a bug that isn't there.
decline() {
  local msg="@${ACTOR} $1"
  # Built in steps rather than as one ${FAMILIES:+...} expansion: $'\n' inside that
  # is honoured by bash but not by every shell, and a message that silently prints
  # a literal $'\n\n' is not worth the saved line.
  if [[ -n "$FAMILIES" ]]; then
    msg="$msg"$'\n\n'"Available: $(usage_list)"
  fi
  react 'confused'
  reply "$msg"
  echo "declined: $2" >&2
  exit 0
}

# Which scope? First whitespace-delimited token of the FIRST line, so
# "/gpu-test-dev please" works and a command followed by prose or a second
# paragraph still parses. \r is stripped because GitHub sends CRLF line endings.
CMD="$(printf '%s' "$BODY" | head -n1 | tr -d '\r' | awk '{print $1}')"

# Derive rather than look up. Note `/gpu-testing` does NOT match /gpu-test-* (the
# next character is `i`, not `-`), so the workflow's startsWith prefilter letting it
# through does not make it a command.
case "$CMD" in
  /gpu-test)     SUITE="full" ;;
  /gpu-test-*)   SUITE="${CMD#/gpu-test-}" ;;
  *)             decline "\`${CMD}\` is not a GPU test command." "not a command: '$CMD'" ;;
esac

# The derived name comes from an attacker-controlled comment and ends up in an API
# request, so it is constrained to a shape a family name could plausibly have before
# it is used for anything. Also stops an empty `/gpu-test-` from being dispatched.
if [[ ! "$SUITE" =~ ^[a-z0-9][a-z0-9-]{0,31}$ ]]; then
  decline "\`${CMD}\` is not a GPU test command." "malformed family name: '$SUITE'"
fi

# Layer 1: local check against the list, when it could be read.
if [[ -n "$FAMILIES" ]]; then
  case " $FAMILIES " in
    *" $SUITE "*) : ;;
    *)            decline "there is no \`${SUITE}\` test family." "unknown family: '$SUITE'" ;;
  esac
fi

SHA="$(gh api "repos/${REPO}/pulls/${PR_NUMBER}" --jq '.head.sha')"

# Layer 2: GitHub's own `type: choice` validation. Only reachable when FAMILIES
# could not be read, since layer 1 would have caught it otherwise.
#
# The reaction is posted AFTER a successful dispatch, not before: a 🚀 followed by
# "no such family" reads as though something launched and then broke.
if ! gh workflow run gpu-tests.yaml \
       --ref "$DEFAULT_BRANCH" \
       -f sha="$SHA" \
       -f pr_number="$PR_NUMBER" \
       -f suite="$SUITE" 2>/tmp/gh_dispatch_err; then
  echo "dispatch failed:" >&2
  cat /tmp/gh_dispatch_err >&2
  decline "could not launch \`${SUITE}\` — it is probably not a valid test family." \
          "dispatch rejected for suite='$SUITE'"
fi

react 'rocket'

echo "Dispatched gpu-tests.yaml (suite=${SUITE}) for PR #${PR_NUMBER} at ${SHA} (by ${ACTOR})"
