#!/usr/bin/env bash
set -Eeuo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ID="${VLA_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${VLA_RUN_DIR:-$REPO_DIR/runs/$RUN_ID}"
mkdir -p "$RUN_DIR/logs"
RUN_DIR="$(cd "$RUN_DIR" && pwd)"
export VLA_RUN_DIR="$RUN_DIR"
exec 7> "$RUN_DIR/.run.lock"
flock -n 7 || { echo 'This complete run is already active.' >&2; exit 2; }

bash "$REPO_DIR/run_training.sh"
if [[ "${VLA_SMOKE_ONLY:-0}" == 1 ]]; then
  echo "UNIVERSITY SMOKE COMPLETE: $RUN_DIR"
  exit 0
fi
bash "$REPO_DIR/run_evaluation.sh" "$RUN_DIR"

echo
echo "================================================================"
echo "FULL OPENVLA EXPERIMENT: PASS"
echo "Run directory: $RUN_DIR"
echo "================================================================"
