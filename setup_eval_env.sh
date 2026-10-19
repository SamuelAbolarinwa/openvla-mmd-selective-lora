#!/usr/bin/env bash
set -Eeuo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="${1:?Usage: setup_eval_env.sh RUN_DIR}"
# One pinned environment includes both training and simulator dependencies.
bash "$REPO_DIR/setup_env.sh" "$RUN_DIR"
