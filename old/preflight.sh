#!/usr/bin/env bash
set -Eeuo pipefail

RUN_DIR="${1:?Usage: preflight.sh RUN_DIR}"
RUN_DIR="$(mkdir -p "$RUN_DIR" && cd "$RUN_DIR" && pwd)"
mkdir -p "$RUN_DIR/logs"
OUT="$RUN_DIR/logs/preflight.txt"

{
  echo "================================================================"
  echo "OpenVLA institutional-compute preflight"
  echo "================================================================"
  date -Is || true
  echo "hostname: $(hostname 2>/dev/null || true)"
  echo "user: $(whoami 2>/dev/null || true)"
  echo "pwd: $PWD"
  echo

  echo "---- scheduler / allocation ----"
  echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
  echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-}"
  echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-}"
  echo "SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
  echo

  echo "---- nvidia-smi ----"
  command -v timeout >/dev/null || { echo "FATAL: timeout is unavailable"; exit 22; }
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "FATAL: nvidia-smi is not available on this node."
    exit 20
  fi
  timeout --kill-after=5s 30s nvidia-smi
  echo
  timeout --kill-after=5s 30s nvidia-smi -L
  echo
  timeout --kill-after=5s 30s nvidia-smi topo -m || true
  echo
  timeout --kill-after=5s 30s nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.free,driver_version --format=csv || true
  echo

  echo "---- CUDA compiler ----"
  if command -v nvcc >/dev/null 2>&1; then
    nvcc --version
  else
    echo "nvcc not in PATH (not fatal; PyTorch wheels carry their own CUDA runtime)."
  fi
  echo

  echo "---- Python candidates ----"
  for p in python3.11 python3.10 python3 python; do
    if command -v "$p" >/dev/null 2>&1; then
      echo "$p -> $(command -v "$p")"
      "$p" --version
    fi
  done
  echo

  echo "---- Git ----"
  git --version || true
  echo

  echo "---- storage ----"
  df -h "$RUN_DIR" || true
  df -h /tmp || true
  echo

  echo "---- memory ----"
  free -h || true
  echo

  echo "---- CPU ----"
  lscpu || true
  echo

  echo "---- limits ----"
  ulimit -a || true
  echo

  echo "---- module environment ----"
  if type module >/dev/null 2>&1; then
    module list 2>&1 || true
  else
    echo "Environment Modules command not available in this shell."
  fi
  echo

  echo "PREFLIGHT_CAPTURE_COMPLETE"
} 2>&1 | tee "$OUT"

grep -q "GPU " "$OUT" || {
  echo "FATAL: nvidia-smi did not report a GPU." | tee -a "$OUT"
  exit 21
}

for tool in bash git curl tar sha256sum flock timeout; do
  command -v "$tool" >/dev/null || { echo "Missing required tool: $tool" | tee -a "$OUT"; exit 22; }
done
[[ $(uname -m) == x86_64 ]] || { echo 'Only x86_64 Linux is supported by this pinned package.' | tee -a "$OUT"; exit 22; }
# Conservative free-space reserve, checked before large downloads.
# Operator may lower it only after confirming available storage/caches.
MIN_FREE_GIB=${VLA_MIN_FREE_GIB:-200}
[[ "$MIN_FREE_GIB" =~ ^[0-9]+$ ]] || { echo 'Invalid VLA_MIN_FREE_GIB'; exit 22; }
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for root in "$RUN_DIR" "${VLA_CACHE_DIR:-$REPO_DIR/cache}" "${VLA_DATA_ROOT:-$REPO_DIR/data/modified_libero_rlds}" "$REPO_DIR"; do
  mkdir -p "$root"
  bytes=$(df -PB1 "$root" | awk 'NR==2 {print $4}')
  if ((bytes < MIN_FREE_GIB * 1024 * 1024 * 1024)); then
    echo "Insufficient free space at $root: need reserve ${MIN_FREE_GIB} GiB. Choose a larger filesystem." | tee -a "$OUT"
    exit 23
  fi
done
echo 'PREFLIGHT RESOURCE CHECK: PASS' | tee -a "$OUT"
