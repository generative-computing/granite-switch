#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Gate: verify an actor holds the Maintain or Admin role on the repo under test.
#
# This is the single source of truth for "who may launch GPU tests". It runs in
# two places:
#   1. Baked into the runner image at /opt/gsw/check_role.sh, called as the
#      first step of workflow/gpu-tests.yaml. This is the AUTHORITATIVE gate: it
#      covers every entry point, including a direct workflow_dispatch from the
#      Actions tab (which needs only *write* access, so it would otherwise
#      bypass the /gpu-test comment check entirely).
#   2. Checked into the repository and used by the /gpu-test
#      comment workflow, which runs on a GitHub-hosted runner and therefore
#      cannot reach /opt/gsw. That copy is a fast-fail UX nicety only.
#
# Living in the image is what makes (1) trustworthy: the gate cannot be edited
# by a pull request, only by rebuilding and redeploying the runner image.
#
# The default GITHUB_TOKEN bot is allowed through: when gpu-tests.yaml is
# dispatched by the comment workflow, github.actor is github-actions[bot], and
# that path was already role-checked upstream.
#
# Usage: check_role.sh <actor-login>
# Env:   GH_TOKEN            token with repo read access
#        GITHUB_REPOSITORY   owner/repo to check the role against
# Exit:  0 authorized, 1 not authorized (reason on stderr).
set -euo pipefail

ACTOR="${1:?usage: check_role.sh <actor-login>}"
REPO="${GITHUB_REPOSITORY:?GITHUB_REPOSITORY must be set}"

if [[ "$ACTOR" == "github-actions[bot]" ]]; then
  echo "actor=$ACTOR is the workflow bot (already gated upstream) — authorized"
  exit 0
fi

# role_name is the granular role: admin / maintain / write / triage / read.
# author_association cannot distinguish maintain from write, so it is unusable
# for this check.
ROLE="$(gh api "repos/${REPO}/collaborators/${ACTOR}/permission" --jq '.role_name')"

if [[ "$ROLE" == "admin" || "$ROLE" == "maintain" ]]; then
  echo "actor=$ACTOR role=$ROLE — authorized"
  exit 0
fi

echo "actor=$ACTOR role=${ROLE:-none} — NOT authorized (requires maintain or admin)" >&2
exit 1
