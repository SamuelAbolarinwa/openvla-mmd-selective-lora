#!/usr/bin/env bash
set -Eeuo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export VLA_SMOKE_ONLY=1
bash "$REPO_DIR/run_all.sh"
