#!/usr/bin/env bash
set -Eeuo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="${1:?Usage: setup_env.sh RUN_DIR}"
source "$REPO_DIR/common.sh"
driver_log setup_env
exec 8> "$REPO_DIR/.setup.lock"
flock -n 8 || { echo 'Another environment setup is active in this repository.' >&2; exit 33; }
STAGE=python
choose_python
"$VLA_PYTHON" --version
VENV="$REPO_DIR/.venv"
if [[ ! -x "$VENV/bin/python" ]]; then "$VLA_PYTHON" -m venv "$VENV"; fi
source "$VENV/bin/activate"
python -c 'import sys; assert (3,10) <= sys.version_info[:2] <= (3,11); assert sys.prefix != sys.base_prefix'
STAGE=dependencies
python -m pip install pip==25.0.1 setuptools==75.8.0 wheel==0.45.1
torch_install
python -m pip install --retries 3 --timeout 60 -c "$REPO_DIR/constraints.txt" -r "$REPO_DIR/requirements-train.txt"
STAGE=pinned_sources
mkdir -p "$REPO_DIR/third_party"
for name in openvla libero dlimp; do
  commit=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]+"_commit"])' "$REPO_DIR/config.json" "$name")
  case "$name" in
    openvla) url=https://github.com/openvla/openvla.git; dest=openvla;;
    libero) url=https://github.com/Lifelong-Robot-Learning/LIBERO.git; dest=LIBERO;;
    dlimp) url=https://github.com/moojink/dlimp_openvla.git; dest=dlimp;;
  esac
  clone_exact "$url" "$REPO_DIR/third_party/$dest" "$commit"
done
python - "$REPO_DIR/third_party/openvla" <<'PATCH'
from pathlib import Path
import sys
p = Path(sys.argv[1]) / 'prismatic/vla/datasets/rlds/oxe/utils/droid_utils.py'
s = p.read_text()
needle = 'import tensorflow_graphics.geometry.transformation as tfg'
if needle in s: p.write_text(s.replace(needle, '# ' + needle))
assert not any(line.strip().startswith('import tensorflow_graphics') for line in p.read_text().splitlines())
print('DROID-only import compatibility patch applied; DROID is not used.')
PATCH
attach_sources
STAGE=runtime_check
python -m pip check
python -u "$REPO_DIR/scripts/check_environment.py" --output "$RUN_DIR/logs/training_environment.json"
python -m pip freeze > "$RUN_DIR/logs/pip-freeze.txt"
STAGE=complete
echo 'TRAINING ENVIRONMENT SETUP: PASS'
