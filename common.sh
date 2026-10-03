#!/usr/bin/env bash
# Shared functions; called from Bash launchers, never from a notebook.
set -Eeuo pipefail
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export TF_CPP_MIN_LOG_LEVEL=2
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export PYTHONHASHSEED=7
umask 077

driver_log() {
  mkdir -p "$RUN_DIR/logs"
  exec > >(tee -a "$RUN_DIR/logs/$1.log") 2>&1
  STAGE=start
  trap 'rc=$?; printf "exit_code=%s\nstage=%s\n" "$rc" "$STAGE" > "$RUN_DIR/logs/'"$1"'_status.txt"; if ((rc)); then printf "FAILED stage=%s exit=%s; see logs in %s\n" "$STAGE" "$rc" "$RUN_DIR" >&2; fi' EXIT
  trap 'printf "Failed command at line %s, stage %s, exit %s\n" "$LINENO" "$STAGE" "$?" >&2' ERR
}

python_supports_venv() {
  local candidate=$1 probe rc=0
  "$candidate" -c 'import sys, venv, ensurepip; assert (3,10) <= sys.version_info[:2] <= (3,11)' >/dev/null 2>&1 || return 1
  probe=$(mktemp -d) || return 1
  "$candidate" -m venv "$probe/venv" >/dev/null 2>&1 || rc=$?
  if ((rc == 0)); then
    "$probe/venv/bin/python" -m pip --version >/dev/null 2>&1 || rc=$?
  fi
  rm -rf -- "$probe"
  return "$rc"
}

choose_python() {
  if [[ -n "${PYTHON_BIN:-}" ]]; then
    python_supports_venv "$PYTHON_BIN" || {
      echo 'PYTHON_BIN must be Python 3.10 or 3.11 with working venv and pip. Unset it to allow automatic selection.' >&2
      return 30
    }
    VLA_PYTHON="$PYTHON_BIN"
    return
  fi
  local candidate
  for candidate in python3.11 python3.10 python3; do
    if command -v "$candidate" >/dev/null && python_supports_venv "$candidate"; then
      VLA_PYTHON=$(command -v "$candidate")
      return
    fi
  done
  echo 'No usable system Python with venv/pip found; installing local Python.'
  # No system modifications. Bootstrap a pinned user-local interpreter.
  [[ $(uname -m) == x86_64 ]] || { echo 'This package supports x86_64 Linux; another architecture requires review.' >&2; return 30; }
  for candidate in curl tar sha256sum; do command -v "$candidate" >/dev/null || { echo "Missing bootstrap tool: $candidate" >&2; return 30; }; done
  local root="$REPO_DIR/tools" archive
  mkdir -p "$root"
  if [[ ! -x "$root/uv-x86_64-unknown-linux-gnu/uv" ]]; then
    archive=$(mktemp "$root/uv_XXXXXX.tar.gz")
    curl --fail --location --retry 3 --connect-timeout 20 --max-time 300 \
      https://github.com/astral-sh/uv/releases/download/0.8.22/uv-x86_64-unknown-linux-gnu.tar.gz -o "$archive"
    printf '%s  %s\n' 741ff1f5742c5a4a25d2f829e8395355e43f7a5ae2ebc6368e9ae2df0efb69cf "$archive" | sha256sum -c -
    tar -xzf "$archive" -C "$root"
    rm -- "$archive"
  fi
  export UV_PYTHON_INSTALL_DIR="$root/python"
  "$root/uv-x86_64-unknown-linux-gnu/uv" python install 3.11.13
  VLA_PYTHON=$("$root/uv-x86_64-unknown-linux-gnu/uv" python find --managed-python 3.11.13)
  python_supports_venv "$VLA_PYTHON" || { echo 'Local Python could not create a working virtual environment.' >&2; return 30; }
}

clone_exact() {
  local url=$1 dest=$2 commit=$3
  if [[ ! -d "$dest/.git" ]]; then
    [[ ! -e "$dest" ]] || { echo "Refusing to overwrite existing non-Git directory: $dest" >&2; return 31; }
    git clone --no-checkout "$url" "$dest"
  fi
  git -C "$dest" fetch origin "$commit"
  git -C "$dest" checkout --detach "$commit"
  [[ $(git -C "$dest" rev-parse HEAD) == "$commit" ]]
}

torch_install() {
  local runtime index
  runtime=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1)
  # cu121 works with modern H100 drivers; actual CUDA execution is checked below.
  index=${VLA_TORCH_INDEX:-https://download.pytorch.org/whl/cu121}
  echo "Driver: $runtime; PyTorch index: $index"
  python -m pip install --retries 3 --timeout 60 --index-url "$index" torch==2.5.1 torchvision==0.20.1
}

attach_sources() {
  python - "$REPO_DIR" <<'PY'
import pathlib, site, sys
root = pathlib.Path(sys.argv[1]).resolve()
names = ['openvla', 'dlimp', 'LIBERO']
for name in names:
    path = root / 'third_party' / name
    if not path.is_dir(): raise FileNotFoundError(path)
path = pathlib.Path(site.getsitepackages()[0]) / 'vla_sources.pth'
path.write_text(''.join(str(root / 'third_party' / n) + '\n' for n in names))
print('Pinned source paths attached:', path)
PY
}
