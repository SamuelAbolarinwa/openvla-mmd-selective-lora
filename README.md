# OpenVLA MMD-Selective LoRA

Scripts for comparing uniform and MMD-selected LoRA fine-tuning of OpenVLA on LIBERO-Object with an equal trainable-parameter budget.

## Experiment

Both models start independently from the same pinned pretrained OpenVLA checkpoint.

| Setting | Uniform LoRA | MMD-selective LoRA |
| --- | --- | --- |
| Decoder layers | All 32 | Layers 9–23 and 25 |
| Decoder rank / alpha | 32 / 16 | 64 / 32 |
| Non-decoder adapters | Rank 32 / alpha 16 | Same modules and settings |
| Trainable parameters | 110,828,288 | 110,828,288 |
| Optimiser updates | 50,000 | 50,000 |

Shared training settings include BF16, FP32 adapters, batch size 16, gradient accumulation 8, effective batch size 128, learning rate `5e-4`, image augmentation and gradient checkpointing.

The MMD layer selection is fixed. These scripts train and evaluate the selected layout; they do not recompute MMD selection.

## Requirements

- x86_64 Linux.
- NVIDIA GPU with BF16 support and sufficient memory for the training configuration.
- Working NVIDIA driver and `nvidia-smi`.
- Bash, Git, curl, tar, sha256sum, timeout and flock.
- Internet access for package, source, model and dataset downloads.
- Working EGL or OSMesa libraries for headless rendering.

Preflight requires a conservative reserve of **200 GiB free** at the repository, run, cache and data locations.

Setup uses Python 3.10 or 3.11 when available. Otherwise, it downloads a verified uv binary and installs Python 3.11.13 locally. Training and evaluation use separate virtual environments.

## Run the complete experiment

Open a terminal in the repository folder:

```bash
bash run_all.sh
```

The pipeline:

1. Records hardware and environment diagnostics.
2. Installs dependencies and prepares pinned assets.
3. Checks headless rendering across all ten tasks.
4. Runs short training, adapter-save and fresh-process reload tests.
5. Tests each smoke adapter alongside the simulator.
6. Trains both conditions.
7. Merges the final adapters.
8. Runs dry and full evaluations.
9. Saves the comparison reports.

The long training runs start only after the preceding checks pass.

## Run only the smoke tests

```bash
bash run_smoke.sh
```

To start the complete experiment using that run directory:

```bash
VLA_RUN_DIR=/absolute/path/to/run bash run_all.sh
```

## GPU allocation

The scripts respect an existing `CUDA_VISIBLE_DEVICES` allocation.

- One visible GPU: conditions run sequentially.
- Two or more: conditions run independently on the first two visible GPUs.

To force sequential execution:

```bash
VLA_SEQUENTIAL=1 bash run_all.sh
```

The scripts do not allocate GPUs or distribute one model across multiple GPUs. Use only devices assigned to your run.

## Choose storage locations

```bash
VLA_RUN_DIR=/storage/openvla/run_01 \
VLA_CACHE_DIR=/storage/openvla/cache \
VLA_DATA_ROOT=/storage/openvla/data/modified_libero_rlds \
bash run_all.sh
```

Use a separate run directory for each experiment.

## Resume interrupted training

Checkpoints are saved every 5,000 optimiser updates.

```bash
VLA_RUN_DIR=/absolute/path/to/existing/run \
RESUME=1 \
bash run_all.sh
```

Keep the original configuration and scripts unchanged. Resume restores the adapter, optimiser and saved RNG states, but recreates the RLDS data stream rather than restoring its exact shuffle position.

If interruption occurs before the first checkpoint, start a new run directory.

## Evaluate without retraining

```bash
bash run_evaluation.sh /absolute/path/to/existing/run
```

Full evaluation runs 15 episodes per task across ten LIBERO-Object tasks, giving 150 episodes per model.

## Outputs

Each run contains:

| Path | Contents |
| --- | --- |
| `logs/` | Hardware diagnostics, dependency versions and stage logs |
| `smoke/` | Smoke-test outputs |
| `training/` | Metrics, checkpoints, final adapters and status records |
| `merged/` | Final merged models |
| `evaluation/` | Episode progress and evaluation results |
| `FINAL_EXPERIMENT_SUMMARY.json` | Overall comparison and training provenance |
| `FINAL_PER_TASK_RESULTS.csv` | Per-task success-rate comparison |

## Troubleshooting

Create a diagnostic bundle:

```bash
bash collect_logs.sh /absolute/path/to/run
```

This produces `diagnostics.tar.gz` containing logs and relevant JSON records, without model weights or optimiser tensors.

## Validation

Run the offline package checks:

```bash
bash validate_package.sh
```

These check syntax, configuration, orchestration and checkpoint handling. They do not execute real model training.

See `VALIDATION.json` for preparation checks and remaining runtime validation requirements. GPU memory fit and model execution are checked by the smoke tests on the target machine.

Generated environments, datasets, caches, checkpoints and model weights are excluded from Git through `.gitignore`.
