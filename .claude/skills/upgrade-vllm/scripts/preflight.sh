#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Preflight for the vLLM upgrade skill. Every check here is something that, if
# missing, makes the run fail HOURS later -- after a GPU dispatch, or at `gh pr
# create`. Failing in the first five seconds instead is the whole point.
#
# Exits 0 only if the run can complete unattended. Prints one actionable line per
# failure.
set -uo pipefail

FAIL=0
note() { printf '  %s\n' "$1"; }
ok() { printf '\033[32mok\033[0m   %s\n' "$1"; }
bad() {
    printf '\033[31mFAIL\033[0m %s\n' "$1"
    FAIL=1
}

# The uv-lock pre-commit hook's rev. A newer uv writes a different lockfile
# revision; the hook then rewrites uv.lock and fails with "files were modified by
# this hook". Read it from the config rather than hardcoding so this cannot drift.
REQUIRED_UV="$(
    awk '/astral-sh\/uv-pre-commit/{found=1} found && /rev:/{gsub(/[^0-9.]/,"",$2); print $2; exit}' \
        .pre-commit-config.yaml 2>/dev/null
)"
: "${REQUIRED_UV:=0.8.4}"

echo "== tooling =="

if command -v gh >/dev/null 2>&1; then
    ok "gh $(gh --version | head -1 | awk '{print $3}')"
    if gh auth status >/dev/null 2>&1; then
        ok "gh authenticated as $(gh api user -q .login 2>/dev/null)"
    else
        bad "gh is not authenticated"
        note "run: gh auth login"
    fi
else
    bad "gh is not installed -- the skill cannot open a PR or dispatch GPU tests"
    note "run: brew install gh && gh auth login"
fi

if command -v uv >/dev/null 2>&1; then
    HAVE_UV="$(uv --version | awk '{print $2}')"
    if [ "$HAVE_UV" = "$REQUIRED_UV" ]; then
        ok "uv $HAVE_UV (matches the uv-lock hook rev)"
    else
        # The local uv version does not matter, because every relock in this skill
        # goes through `uvx uv@$REQUIRED_UV`. What matters is that the pin RESOLVES
        # -- offline or behind a proxy it will not, and the skill would then relock
        # with the local uv, write a different lockfile revision, and fail the
        # uv-lock hook at commit time with "files were modified by this hook".
        # So: check the pin, not the local version.
        if uvx "uv@$REQUIRED_UV" --version >/dev/null 2>&1; then
            ok "uv $HAVE_UV locally, but uvx uv@$REQUIRED_UV resolves (relock uses the pin)"
        else
            bad "uv $HAVE_UV locally and uvx uv@$REQUIRED_UV does not resolve"
            note "the uv-lock hook needs $REQUIRED_UV; relocking with $HAVE_UV fails the hook"
            note "fix: restore network access to PyPI, or: uv self update $REQUIRED_UV"
        fi
    fi
else
    bad "uv is not installed"
    note "run: curl -LsSf https://astral.sh/uv/install.sh | sh"
fi

command -v git >/dev/null 2>&1 && ok "git" || bad "git is not installed"

echo "== repository =="

if [ -n "$(git status --porcelain)" ]; then
    bad "working tree is dirty -- commit or stash before an unattended run"
    git status --porcelain | sed 's/^/     /'
else
    ok "working tree clean"
fi

REPO="$(gh repo view --json nameWithOwner -q .nameWithOwner 2>/dev/null)"
if [ -n "$REPO" ]; then
    ok "repo $REPO"
else
    bad "could not resolve the GitHub repo (is origin a GitHub remote?)"
fi

# The authorization gate. gpu-tests.yaml runs /opt/gsw/check_role.sh and accepts
# only admin or maintain, so a `write` collaborator gets DECLINED after dispatch --
# the run starts, then fails on the role step. Check it here instead.
if [ -n "$REPO" ]; then
    ACTOR="$(gh api user -q .login 2>/dev/null)"
    ROLE="$(gh api "repos/$REPO/collaborators/$ACTOR/permission" -q .role_name 2>/dev/null)"
    case "$ROLE" in
        admin | maintain) ok "role '$ROLE' -- authorized to dispatch GPU tests" ;;
        "") bad "could not read your role on $REPO" ;;
        *)
            bad "role '$ROLE' is not enough -- gpu-tests.yaml accepts only admin or maintain"
            note "ask a repo admin to grant Maintain, or run the GPU legs via a maintainer's /gpu-test comment"
            ;;
    esac
fi

# SSH, not HTTPS: the push target is git@github.com and the keychain has no token
# for github.com on this machine.
if git ls-remote --exit-code origin HEAD >/dev/null 2>&1; then
    ok "origin is reachable (push credentials work)"
else
    bad "cannot reach origin -- check your SSH key"
    note "test with: ssh -T git@github.com"
fi

echo "== gpu suite plumbing =="

# A dispatch naming a suite with no case arm fails at "Resolve test scope" with
# `unknown suite`.
#
# CHECK THE DEFAULT BRANCH'S COPY, NOT THE LOCAL ONE. `workflow_dispatch` always
# executes the workflow file as it exists on the DEFAULT BRANCH, regardless of
# which ref or sha is being tested -- the same property pyproject's `vllm27`
# comment records. Grepping the local file therefore reports a suite as
# available when a dispatch of it would be rejected, which is exactly the
# false green this check exists to prevent: the `eval` suite is on the feature
# branch and not yet on main.
WF=.github/workflows/gpu-tests.yaml
DEFAULT_BRANCH=$(gh repo view --json defaultBranchRef -q .defaultBranchRef.name 2>/dev/null)
if [ -z "$DEFAULT_BRANCH" ]; then
    bad "cannot determine the default branch -- cannot verify suite availability"
else
    WF_REMOTE=$(git fetch -q origin "$DEFAULT_BRANCH" 2>/dev/null &&
        git show "origin/$DEFAULT_BRANCH:$WF" 2>/dev/null)
    if [ -z "$WF_REMOTE" ]; then
        bad "$WF is missing on origin/$DEFAULT_BRANCH -- no GPU workflow to dispatch"
    else
        for suite in full eval; do
            if printf '%s' "$WF_REMOTE" | grep -q "options:.*\b$suite\b"; then
                ok "suite '$suite' is dispatchable (present on origin/$DEFAULT_BRANCH)"
            else
                bad "suite '$suite' is not in $WF on origin/$DEFAULT_BRANCH -- a dispatch would be rejected with 'unknown suite'"
                note "workflow_dispatch runs the DEFAULT BRANCH's workflow file; merge the suite to $DEFAULT_BRANCH first"
            fi
        done
    fi
fi

# A dispatchable eval suite is still useless without recorded baselines. Tier-1
# compares this commit's adapter output against committed numbers; with nothing
# to compare to, every eval leg fails (or worse, vacuously passes) and the PR
# body's drift column has no source. README.md is the directory's own
# documentation, not a baseline, so it does not count.
BASELINES=tests/eval/baselines
if [ ! -d "$BASELINES" ]; then
    bad "$BASELINES does not exist -- the eval suite has nothing to compare against"
else
    n=$(find "$BASELINES" -type f ! -name 'README.md' | wc -l | tr -d ' ')
    if [ "$n" -gt 0 ]; then
        ok "$n tier-1 baseline file(s) recorded"
    else
        bad "no tier-1 baselines in $BASELINES (only README.md) -- eval legs cannot be green"
        note "record them once on a GPU pod with adapter checkpoints staged, and commit them"
    fi
fi

echo
if [ "$FAIL" -ne 0 ]; then
    echo "preflight FAILED -- fix the above before starting an unattended upgrade."
    exit 1
fi
echo "preflight passed."
