#!/usr/bin/env bash
set -Eeuo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ID="${VLA_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${VLA_RUN_DIR:-$REPO_DIR/runs/$RUN_ID}"
mkdir -p "$RUN_DIR/logs"

RUN_DIR="$(cd "$RUN_DIR" && pwd)"
source "$REPO_DIR/common.sh"
driver_log run_training_driver
exec 9> "$RUN_DIR/.training.lock"
flock -n 9 || { echo 'Training is already running in this directory.' >&2; exit 2; }

echo "================================================================"
echo "OpenVLA university training pipeline"
echo "Run directory: $RUN_DIR"
echo "================================================================"

STAGE=preflight
bash "$REPO_DIR/preflight.sh" "$RUN_DIR"

cd "$REPO_DIR"
STAGE=training_environment
bash "$REPO_DIR/setup_env.sh" "$RUN_DIR"
source "$REPO_DIR/.venv/bin/activate"
python - "$REPO_DIR" "$RUN_DIR" <<'FREEZE'
from pathlib import Path
import hashlib, json, sys
repo, run = map(Path, sys.argv[1:])
config = json.loads((repo/'config.json').read_text())
target = run/'config.json'
if target.exists():
    assert json.loads(target.read_text()) == config, 'Run configuration changed; use original settings or a new run directory'
else: target.write_text(json.dumps(config, indent=2)+'\n')
files = sorted(list(repo.glob('*.sh')) + list((repo/'scripts').glob('*.py')) + list(repo.glob('requirements*.txt')) + [repo/'constraints.txt'])
manifest={str(f.relative_to(repo)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
target=run/'source_manifest.json'
if target.exists():
    assert json.loads(target.read_text()) == manifest, 'Package source changed since run began; retain the original package for resume'
else: target.write_text(json.dumps(manifest,indent=2)+'\n')
FREEZE

export HF_HOME="${VLA_CACHE_DIR:-$REPO_DIR/cache}/huggingface"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled

STAGE=assets
python -u "$REPO_DIR/scripts/prepare_assets.py" \
  --repo-dir "$REPO_DIR" \
  --run-dir "$RUN_DIR" \
  2>&1 | tee "$RUN_DIR/logs/prepare_assets.log"

STAGE=evaluation_environment
bash "$REPO_DIR/setup_eval_env.sh" "$RUN_DIR"
STAGE=renderer_preflight
bash "$REPO_DIR/check_renderer.sh" "$RUN_DIR"

GPU_COUNT="$(python - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)"
export VLA_VISIBLE_GPU_COUNT="$GPU_COUNT"

echo "Visible GPUs after setup: $GPU_COUNT"
echo
STAGE=model_smoke
echo "=== Production-profile smoke: BF16, no quantisation, real batch/accumulation ==="
python -u "$REPO_DIR/scripts/orchestrate_training.py" \
  --repo-dir "$REPO_DIR" \
  --run-dir "$RUN_DIR" \
  --phase smoke

STAGE=combined_model_renderer_smoke
source "$RUN_DIR/renderer.env"
export LIBERO_CONFIG_PATH
LIBERO_CONFIG_PATH=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["libero_config_dir"])' "$RUN_DIR/assets.json")
SMOKE_ROOT=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["output_root"])' "$RUN_DIR/SMOKE_COMPLETE.json")
SMOKE_BASE=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["base_model_overlay"])' "$RUN_DIR/assets.json")
for condition in uniform mmd_selective; do
  "$REPO_DIR/.venv-eval/bin/python" -u "$REPO_DIR/scripts/evaluate_libero.py" \
    --mode combined_smoke --checkpoint "$SMOKE_BASE" \
    --adapter "$SMOKE_ROOT/$condition/final_adapter" \
    --stats "$SMOKE_ROOT/$condition/dataset_statistics.json" \
    2>&1 | tee "$RUN_DIR/logs/smoke_${condition}_simulator.log"
done
if [[ "${VLA_SMOKE_ONLY:-0}" == 1 ]]; then
  STAGE=complete
  echo 'UNIVERSITY PRODUCTION-PROFILE SMOKE: PASS; long training was not started.'
  exit 0
fi

echo
STAGE=training
echo "=== Full 50,000-step training ==="
RESUME_FLAG=()
if [[ "${RESUME:-0}" == "1" ]]; then
  RESUME_FLAG=(--resume)
fi
python -u "$REPO_DIR/scripts/orchestrate_training.py" \
  --repo-dir "$REPO_DIR" \
  --run-dir "$RUN_DIR" \
  --phase train \
  "${RESUME_FLAG[@]}"

echo
STAGE=merge
echo "=== Merge each final adapter exactly once for evaluation ==="
ASSETS="$RUN_DIR/assets.json"
BASE="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["base_model_overlay"])' "$ASSETS")"
OPENVLA_SRC="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["openvla_source"])' "$ASSETS")"

for condition in uniform mmd_selective; do
  python -u "$REPO_DIR/scripts/merge_adapter.py" \
    --base "$BASE" \
    --adapter "$RUN_DIR/training/$condition/final_adapter" \
    --stats "$RUN_DIR/training/$condition/dataset_statistics.json" \
    --output "$RUN_DIR/merged/$condition" \
    --openvla-source "$OPENVLA_SRC" \
    2>&1 | tee "$RUN_DIR/logs/merge_${condition}.log"
done

cat > "$RUN_DIR/TRAINING_COMPLETE.txt" <<EOF
Training and final merge completed successfully.
Run directory: $RUN_DIR
Completed: $(date -Is)
EOF

echo
STAGE=complete
echo "TRAINING PIPELINE: PASS"
echo "Run directory: $RUN_DIR"
echo "Next: bash run_evaluation.sh \"$RUN_DIR\""
