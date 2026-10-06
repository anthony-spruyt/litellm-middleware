#!/usr/bin/env bash
# shellcheck disable=SC2034 # Variables used by sourcing script (lint.sh)
# This file is automatically updated - do not modify directly
# The image pin lives in repo-operator (src/groups.yaml, or src/repos.yaml for a per-repo flavor), where Renovate bumps it

MEGALINTER_IMAGE="ghcr.io/anthony-spruyt/megalinter-python:1.0.0@sha256:1bc5bf1a854a1addd7142a096b1a1c85c6521fe9367ef92d424a26b3d7aa93f8"

SKIP_BOT_COMMITS=false

# MegaLinter flavor (use "all" for custom images to bypass flavor validation)
MEGALINTER_FLAVOR="all"
