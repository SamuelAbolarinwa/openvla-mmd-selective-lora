#!/usr/bin/env bash
set -Eeuo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="${1:?Usage: check_renderer.sh RUN_DIR}"
source "$REPO_DIR/common.sh"
driver_log check_renderer
source "$REPO_DIR/.venv-eval/bin/activate"
export LIBERO_CONFIG_PATH
LIBERO_CONFIG_PATH=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["libero_config_dir"])' "$RUN_DIR/assets.json")
resolve_egl() {
# Existing administrator-provided EGL selection takes priority.
if [[ -z "${VLA_EGL_DEVICE_ID:-${MUJOCO_EGL_DEVICE_ID:-}}" ]]; then
  VLA_EGL_DEVICE_ID=$(python - <<'PY'
import os, subprocess
visible=os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')[0]
if visible.isdigit(): print(visible)
elif visible.startswith('GPU-'):
    lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True).splitlines()
    matches=[line.split(',')[0].strip() for line in lines if line.split(',')[1].strip().startswith(visible)]
    if len(matches)!=1: raise RuntimeError('Cannot map GPU UUID to EGL device; supply VLA_EGL_DEVICE_ID')
    print(matches[0])
elif visible.startswith('MIG-'): raise RuntimeError('MIG EGL rendering requires an administrator-verified VLA_EGL_DEVICE_ID; OSMesa can be used without this mapping')
else: print(0)
PY
) || return 1
else
  VLA_EGL_DEVICE_ID=${VLA_EGL_DEVICE_ID:-$MUJOCO_EGL_DEVICE_ID}
fi
[[ "$VLA_EGL_DEVICE_ID" =~ ^[0-9]+$ ]] || { echo 'EGL device ID must be a non-negative integer.' >&2; return 2; }
export VLA_EGL_DEVICE_ID
 }
TRIALS=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["evaluation"]["full_trials_per_task"])' "$RUN_DIR/config.json")
REQUESTED_EGL=${VLA_EGL_DEVICE_ID:-${MUJOCO_EGL_DEVICE_ID:-}}
CANDIDATES=(${VLA_RENDERER:-osmesa egl})
for renderer in "${CANDIDATES[@]}"; do
  [[ "$renderer" == osmesa || "$renderer" == egl ]] || { echo 'VLA_RENDERER must be osmesa or egl'; exit 2; }
  STAGE="probe_$renderer"
  if [[ "$renderer" == egl ]]; then
    if [[ -n "$REQUESTED_EGL" ]]; then export VLA_EGL_DEVICE_ID="$REQUESTED_EGL"; else unset VLA_EGL_DEVICE_ID; fi
    if ! resolve_egl; then
      echo 'Could not resolve EGL device; see allocator configuration.' >&2
      continue
    fi
  else
    VLA_EGL_DEVICE_ID=0
    export VLA_EGL_DEVICE_ID
  fi
  if VLA_RENDERER="$renderer" MUJOCO_GL="$renderer" PYOPENGL_PLATFORM="$renderer" MUJOCO_EGL_DEVICE_ID="$VLA_EGL_DEVICE_ID" \
    timeout --signal=TERM --kill-after=15s 600s python -u "$REPO_DIR/scripts/evaluate_libero.py" \
      --mode preflight --trials "$TRIALS" --env-seed 0 > "$RUN_DIR/logs/renderer_${renderer}.log" 2>&1; then
    printf 'export VLA_RENDERER=%s\nexport MUJOCO_GL=%s\nexport PYOPENGL_PLATFORM=%s\nexport VLA_EGL_DEVICE_ID=%s\nexport MUJOCO_EGL_DEVICE_ID=%s\n' \
      "$renderer" "$renderer" "$renderer" "$VLA_EGL_DEVICE_ID" "$VLA_EGL_DEVICE_ID" > "$RUN_DIR/renderer.env"
    cat "$RUN_DIR/logs/renderer_${renderer}.log"
    STAGE=complete
    echo "HEADLESS RENDERER CHECK: PASS ($renderer)"
    exit 0
  fi
  cat "$RUN_DIR/logs/renderer_${renderer}.log"
done
echo 'No tested rendering backend worked. See renderer logs. Required system OpenGL/OSMesa libraries may need administrator installation.' >&2
exit 50
