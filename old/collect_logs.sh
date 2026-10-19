#!/usr/bin/env bash
set -Eeuo pipefail
RUN_DIR="${1:?Usage: bash collect_logs.sh RUN_DIR}"
RUN_DIR="$(cd "$RUN_DIR" && pwd)"
OUT="$RUN_DIR/diagnostics.tar.gz"
cd "$RUN_DIR"
find . -type f \( -path './logs/*' -o -name 'failure.json' -o -name 'status.json' -o -name 'completion.json' -o -name 'latest_checkpoint.json' -o -name 'resume_disclosure.json' -o -name 'config.json' -o -name 'assets.json' -o -name 'source_manifest.json' -o -name 'results.json' -o -name 'progress.json' -o -name 'FINAL_EXPERIMENT_SUMMARY.json' \) -print0 | tar --null -T - -czf "$OUT"
echo "Diagnostics: $OUT"
