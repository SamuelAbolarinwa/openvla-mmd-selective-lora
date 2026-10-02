#!/usr/bin/env bash
set -Eeuo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="${1:?Usage: setup_eval_env.sh RUN_DIR}"
source "$REPO_DIR/common.sh"
driver_log setup_eval_env
exec 8> "$REPO_DIR/.setup.lock"
flock -n 8 || { echo 'Another environment setup is active in this repository.' >&2; exit 43; }
STAGE=python
choose_python
VENV="$REPO_DIR/.venv-eval"
if [[ ! -x "$VENV/bin/python" ]]; then "$VLA_PYTHON" -m venv "$VENV"; fi
source "$VENV/bin/activate"
python -c 'import sys; assert (3,10) <= sys.version_info[:2] <= (3,11); assert sys.prefix != sys.base_prefix'
STAGE=dependencies
python -m pip install pip==25.0.1 setuptools==75.8.0 wheel==0.45.1
torch_install
python -m pip install --retries 3 --timeout 60 -c "$REPO_DIR/constraints.txt" -r "$REPO_DIR/requirements-train.txt" -r "$REPO_DIR/requirements-eval.txt"
# Preserve robosuite code, replace only its OpenCV dependency declaration with
# the API-compatible headless distribution. wheel pack rebuilds RECORD hashes.
STAGE=robosuite_headless_metadata
mkdir -p "$REPO_DIR/tools"
WHEEL_DIR=$(mktemp -d "$REPO_DIR/tools/robosuite_XXXXXX")
python -m pip download --no-deps --only-binary=:all: --dest "$WHEEL_DIR" robosuite==1.4.1
python -m wheel unpack "$WHEEL_DIR"/*.whl --dest "$WHEEL_DIR/unpacked"
python - "$WHEEL_DIR" <<'PATCH'
from pathlib import Path
import sys
root=Path(sys.argv[1])
paths=list((root/'unpacked').glob('*/robosuite-*.dist-info/METADATA'))
assert len(paths)==1, paths
p=paths[0]; s=p.read_text()
assert 'Requires-Dist: opencv-python' in s
p.write_text(s.replace('Requires-Dist: opencv-python', 'Requires-Dist: opencv-python-headless'))
PATCH
mkdir -p "$WHEEL_DIR/patched"
python -m wheel pack "$WHEEL_DIR/unpacked"/* --dest-dir "$WHEEL_DIR/patched"
python -m pip install --no-deps --force-reinstall "$WHEEL_DIR/patched"/*.whl
attach_sources
STAGE=runtime_check
python -m pip check
python -u "$REPO_DIR/scripts/check_environment.py" --output "$RUN_DIR/logs/evaluation_environment.json"
python -m pip freeze > "$RUN_DIR/logs/pip-freeze-eval.txt"
STAGE=complete
echo 'EVALUATION ENVIRONMENT SETUP: PASS'
