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
ensure_project_venv
source "$VENV/bin/activate"
python -c 'import sys; assert (3,10) <= sys.version_info[:2] <= (3,11); assert sys.prefix != sys.base_prefix'
STAGE=dependencies
STAMP=$(cat "$REPO_DIR/requirements-train.txt" "$REPO_DIR/requirements-eval.txt" "$REPO_DIR/constraints.txt" "$REPO_DIR/setup_env.sh" "$REPO_DIR/common.sh" | sha256sum | cut -d' ' -f1)
if [[ ! -f "$VENV/.dependencies-complete" ]] || [[ $(cat "$VENV/.dependencies-complete") != "$STAMP" ]] || ! python -m pip check; then
  python -m pip install pip==25.0.1 setuptools==75.8.0 wheel==0.45.1
  torch_install
  python -m pip install --retries 3 --timeout 60 -c "$REPO_DIR/constraints.txt" -r "$REPO_DIR/requirements-train.txt" -r "$REPO_DIR/requirements-eval.txt"
  STAGE=robosuite_headless_metadata
  WHEEL_DIR=$(mktemp -d "$TMPDIR/robosuite_XXXXXX")
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
  rm -rf -- "$WHEEL_DIR"
  python -m pip check
  printf '%s\n' "$STAMP" > "$VENV/.dependencies-complete"
else
  echo 'Pinned shared training/evaluation dependencies already installed.'
fi
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
