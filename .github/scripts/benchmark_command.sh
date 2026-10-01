#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Handle a /benchmark PR comment: verify the commenter holds Maintain/Admin, read
# its flags, then dispatch adapter-benchmark.yaml against the PR's head commit. On
# rejection, react and reply naming the requirement.
#
#   /benchmark                          the first model of the benchmark
#   /benchmark --model granite-4.2-3b   another model
#   /benchmark --no-cache               re-run a commit that already has its row
#
# Same conventions as gpu_test_command.sh, and for the same reasons:
#   - It runs on a GitHub-hosted runner, so it is checked in rather than baked into
#     the runner image. It is fast-fail UX only: the authoritative role gate is
#     /opt/gsw/check_role.sh, the first step of adapter-benchmark.yaml.
#   - GitHub-controlled values arrive as quoted positional args, never interpolated
#     into this script, so a crafted login or comment cannot inject shell.
#   - Only the comment's FIRST line is the command; an unknown flag is declined
#     rather than ignored (`--nocache` must not quietly return a cached result).
#
# The model id is checked for shape only. The list of models lives with the
# benchmark (adapters.yaml on its branch), and an unknown id fails that run with a
# reply here, rather than this script keeping a second list to go stale.
#
# Usage: benchmark_command.sh <actor-login> <pr-number> <comment-id> <comment-body>
# Env:   GH_TOKEN, GITHUB_REPOSITORY, DEFAULT_BRANCH, SCRIPT_DIR
set -euo pipefail

ACTOR="${1:?usage: benchmark_command.sh <actor-login> <pr-number> <comment-id> <comment-body>}"
PR_NUMBER="${2:?missing pr number}"
COMMENT_ID="${3:?missing comment id}"
# May legitimately be empty or multi-line, so no :? guard.
BODY="${4:-}"

REPO="${GITHUB_REPOSITORY:?}"
DEFAULT_BRANCH="${DEFAULT_BRANCH:?}"
SCRIPT_DIR="${SCRIPT_DIR:?}"

USAGE='Usage: `/benchmark [--model <id>] [--no-cache]`.'

react() {
  gh api -X POST "repos/${REPO}/issues/comments/${COMMENT_ID}/reactions" \
    -f content="$1" >/dev/null
}

reply() {
  gh api -X POST "repos/${REPO}/issues/${PR_NUMBER}/comments" -f body="$1" >/dev/null
}

if ! ROLE_MSG="$("${SCRIPT_DIR}/check_role.sh" "$ACTOR" 2>&1)"; then
  react '-1'
  reply "@${ACTOR} \`/benchmark\` requires the **Maintain** or **Admin** role. Not launching."
  echo "$ROLE_MSG" >&2
  exit 1
fi

# Exit 0: a mistyped command is user error, not a broken workflow.
decline() {
  react 'confused'
  reply "@${ACTOR} $1"$'\n\n'"${USAGE}"
  echo "declined: $2" >&2
  exit 0
}

# The first line only; \r stripped (GitHub sends CRLF). read -ra rather than an
# unquoted expansion, which would glob-expand an attacker's text.
FIRST_LINE="$(printf '%s' "$BODY" | head -n1 | tr -d '\r')"
read -ra TOKENS <<<"$FIRST_LINE"
CMD="${TOKENS[0]:-}"
# The workflow's startsWith prefilter also lets `/benchmarks` through.
[[ "$CMD" == "/benchmark" ]] || decline "\`${CMD}\` is not the benchmark command." \
  "not a command: '$CMD'"

MODEL=""
NO_CACHE=false
for ((i = 1; i < ${#TOKENS[@]}; i++)); do
  case "${TOKENS[i]}" in
    --no-cache) NO_CACHE=true ;;
    --model)
      i=$((i + 1))
      MODEL="${TOKENS[i]:-}"
      [[ -n "$MODEL" ]] || decline "\`--model\` needs a model id." "--model without a value"
      ;;
    --model=*) MODEL="${TOKENS[i]#--model=}" ;;
    -*) decline "\`${TOKENS[i]}\` is not a benchmark option." "unknown flag: '${TOKENS[i]}'" ;;
    *) : ;;  # prose after the command
  esac
done

# The id ends up in an API request and a cluster job name: constrain its shape.
if [[ -n "$MODEL" && ! "$MODEL" =~ ^[a-z0-9][a-z0-9.-]{0,63}$ ]]; then
  decline "\`${MODEL}\` is not a model id." "malformed model id: '$MODEL'"
fi

SHA="$(gh api "repos/${REPO}/pulls/${PR_NUMBER}" --jq '.head.sha')"
ERR="${RUNNER_TEMP:-${TMPDIR:-/tmp}}/gh_dispatch_err"

# The reaction comes AFTER a successful dispatch: a rocket followed by an error
# would read as though something launched and then broke.
if ! gh workflow run adapter-benchmark.yaml \
       --ref "$DEFAULT_BRANCH" \
       -f sha="$SHA" \
       -f pr_number="$PR_NUMBER" \
       -f model="$MODEL" \
       -f no_cache="$NO_CACHE" 2>"$ERR"; then
  echo "dispatch failed:" >&2
  cat "$ERR" >&2
  decline "could not launch the benchmark." "dispatch rejected"
fi

react 'rocket'

echo "Dispatched adapter-benchmark.yaml (model=${MODEL:-default}, no_cache=${NO_CACHE}) for PR #${PR_NUMBER} at ${SHA} (by ${ACTOR})"
