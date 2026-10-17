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
        # Not fatal: `uvx uv@<rev>` can relock with the right version. But the
        # skill must know to use it rather than plain `uv lock`.
        bad "uv $HAVE_UV != $REQUIRED_UV (the uv-lock pre-commit hook's rev)"
        note "relock with: uvx uv@$REQUIRED_UV lock"
        note "or pin the toolchain: uv self update $REQUIRED_UV"
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
# `unknown suite`. Cheap to verify, and the eval suite is new enough to be missing
# on an older checkout.
WF=.github/workflows/gpu-tests.yaml
for suite in full eval; do
    if grep -q "options:.*\b$suite\b" "$WF" 2>/dev/null; then
        ok "suite '$suite' is in the workflow's options list"
    else
        bad "suite '$suite' is not an option in $WF -- dispatch would be rejected"
    fi
done

echo
if [ "$FAIL" -ne 0 ]; then
    echo "preflight FAILED -- fix the above before starting an unattended upgrade."
    exit 1
fi
echo "preflight passed."
