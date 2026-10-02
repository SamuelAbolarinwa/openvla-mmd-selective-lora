#!/usr/bin/env bash
set -Eeuo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for file in "$REPO_DIR"/*.sh "$REPO_DIR"/slurm/*.example; do bash -n "$file"; done
PYTHON=${PYTHON_BIN:-python3}
"$PYTHON" - "$REPO_DIR" <<'PY'
import pathlib, sys
root=pathlib.Path(sys.argv[1])
for path in root.rglob('*.py'):
    if any(p in path.parts for p in ['third_party','tools','.venv','.venv-eval']): continue
    compile(path.read_text(),str(path),'exec')
print('Python/Bash syntax: PASS')
PY
"$PYTHON" "$REPO_DIR/tests/offline_validation.py"
echo 'OFFLINE PACKAGE VALIDATION: PASS (no real model, CUDA or simulator execution)'
