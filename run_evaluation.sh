#!/usr/bin/env bash
set -Eeuo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="${1:-${VLA_RUN_DIR:-}}"
if [[ -z "$RUN_DIR" ]]; then
  echo "Usage: bash run_evaluation.sh /path/to/existing/run"
  exit 2
fi
RUN_DIR="$(cd "$RUN_DIR" && pwd)"
mkdir -p "$RUN_DIR/logs" "$RUN_DIR/evaluation"

source "$REPO_DIR/common.sh"
driver_log run_evaluation_driver
exec 9> "$RUN_DIR/.evaluation.lock"
flock -n 9 || { echo 'Evaluation is already running in this directory.' >&2; exit 2; }
[[ -f "$RUN_DIR/TRAINING_COMPLETE.txt" ]] || { echo 'Training/merge completion marker missing.' >&2; exit 2; }

cd "$REPO_DIR"
STAGE=evaluation_environment
bash "$REPO_DIR/setup_eval_env.sh" "$RUN_DIR"
source "$REPO_DIR/.venv-eval/bin/activate"
ASSETS="$RUN_DIR/assets.json"
LIBERO_CONFIG="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["libero_config_dir"])' "$ASSETS")"
export LIBERO_CONFIG_PATH="$LIBERO_CONFIG"
export TOKENIZERS_PARALLELISM=false
export TF_CPP_MIN_LOG_LEVEL=2

STAGE=renderer_preflight
bash "$REPO_DIR/check_renderer.sh" "$RUN_DIR"
source "$RUN_DIR/renderer.env"

export VLA_EVAL_CONFIG="$RUN_DIR/config.json"
TRIALS="$(python -c 'import json; print(json.load(open(__import__("os").environ["VLA_EVAL_CONFIG"]))["evaluation"]["full_trials_per_task"])')"
DRY="$(python -c 'import json; print(json.load(open(__import__("os").environ["VLA_EVAL_CONFIG"]))["evaluation"]["dry_trials_per_task"])')"
ENV_SEED="$(python -c 'import json; print(json.load(open(__import__("os").environ["VLA_EVAL_CONFIG"]))["evaluation"]["env_seed"])')"
SEED="$(python -c 'import json; print(json.load(open(__import__("os").environ["VLA_EVAL_CONFIG"]))["seed"])')"
MAX_STEPS="$(python -c 'import json; print(json.load(open(__import__("os").environ["VLA_EVAL_CONFIG"]))["evaluation"]["max_steps"])')"
SETTLE="$(python -c 'import json; print(json.load(open(__import__("os").environ["VLA_EVAL_CONFIG"]))["evaluation"]["settle_steps"])')"

STAGE=evaluation
for condition in uniform mmd_selective; do
  python -u "$REPO_DIR/scripts/evaluate_libero.py" \
    --mode combined_smoke \
    --checkpoint "$RUN_DIR/merged/$condition" \
    --device 0 \
    --env-seed "$ENV_SEED" \
    2>&1 | tee "$RUN_DIR/logs/eval_${condition}_combined_smoke.log"
done

STAGE=evaluation
for condition in uniform mmd_selective; do
  python -u "$REPO_DIR/scripts/evaluate_libero.py" \
    --mode evaluate \
    --checkpoint "$RUN_DIR/merged/$condition" \
    --output "$RUN_DIR/evaluation/${condition}_dry" \
    --trials "$DRY" \
    --seed "$SEED" \
    --env-seed "$ENV_SEED" \
    --device 0 \
    --max-steps "$MAX_STEPS" \
    --settle-steps "$SETTLE" \
    2>&1 | tee "$RUN_DIR/logs/eval_${condition}_dry.log"
done

STAGE=evaluation
for condition in uniform mmd_selective; do
  python -u "$REPO_DIR/scripts/evaluate_libero.py" \
    --mode evaluate \
    --checkpoint "$RUN_DIR/merged/$condition" \
    --output "$RUN_DIR/evaluation/${condition}_full" \
    --trials "$TRIALS" \
    --seed "$SEED" \
    --env-seed "$ENV_SEED" \
    --device 0 \
    --max-steps "$MAX_STEPS" \
    --settle-steps "$SETTLE" \
    2>&1 | tee "$RUN_DIR/logs/eval_${condition}_full.log"
done

python -u "$REPO_DIR/scripts/summarize_results.py" \
  --run-dir "$RUN_DIR" \
  2>&1 | tee "$RUN_DIR/logs/final_summary.log"

STAGE=complete
echo "EVALUATION PIPELINE: PASS"
echo "Final summary: $RUN_DIR/FINAL_EXPERIMENT_SUMMARY.json"
