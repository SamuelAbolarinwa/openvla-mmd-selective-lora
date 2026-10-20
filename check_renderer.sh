#!/usr/bin/env bash
set -Eeuo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="${1:?Usage: check_renderer.sh RUN_DIR}"
source "$REPO_DIR/common.sh"
driver_log check_renderer
source "$REPO_DIR/.venv-cluster/bin/activate"
export LIBERO_CONFIG_PATH
LIBERO_CONFIG_PATH=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["libero_config_dir"])' "$RUN_DIR/assets.json")
resolve_egl() {
  # EGL enumeration is not guaranteed to match CUDA/nvidia-smi numbering.
  VLA_EGL_DEVICE_ID=${VLA_EGL_DEVICE_ID:-${MUJOCO_EGL_DEVICE_ID:-}}
  if [[ ! "$VLA_EGL_DEVICE_ID" =~ ^[0-9]+$ ]]; then
    echo 'EGL requires VLA_EGL_DEVICE_ID set to an administrator-verified EGL index for an allocated GPU. CUDA indices/UUIDs are not converted automatically. OSMesa needs no EGL mapping.' >&2
    return 2
  fi
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
