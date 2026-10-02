# OpenVLA MMD-Selective LoRA

Scripts for comparing uniform and MMD-selected LoRA fine-tuning of OpenVLA on LIBERO-Object with the same trainable-parameter budget.

## Requirements

- x86_64 Linux.
- A working NVIDIA driver and a GPU that supports BF16, with enough memory for training.
- Bash, Git, curl, tar, sha256sum, timeout and flock.
- Internet access to download dependencies, models and data.
- EGL or OSMesa libraries for headless simulation.
- At least 200 GiB of free space at the repository, run, cache and data locations.

Setup creates separate Python environments for training and evaluation. It uses Python 3.10 or 3.11 if available, or downloads a local Python installation.

Download or clone the repository, then open a terminal in its folder.

## Optional: run a smoke test first

To check setup, short training runs, adapter saving and reloading, and simulator execution:

```bash
bash run_smoke.sh
```

This stops after the checks and prints the run directory.

To continue with the full experiment using that directory:

```bash
VLA_RUN_DIR=/absolute/path/to/run bash run_all.sh
```

You can skip this separate smoke run and use the command below. The full pipeline includes the same checks before starting long training.

## Run the full experiment

```bash
bash run_all.sh
```

The script installs dependencies, downloads the required assets, runs the checks, trains both models and evaluates them. Logs and results are saved in the run directory printed in the terminal.

Each model trains for 50,000 optimiser updates. Full evaluation runs 150 episodes per model across ten LIBERO-Object tasks.

## GPU selection

The scripts use the GPUs available through `CUDA_VISIBLE_DEVICES`. Keep any allocation supplied by your environment.

With one GPU, the two models run sequentially. With two or more, they run independently on the first two visible GPUs.

To run sequentially even when multiple GPUs are visible:

```bash
VLA_SEQUENTIAL=1 bash run_all.sh
```

## Choose where files are stored

```bash
VLA_RUN_DIR=/storage/openvla/run_01 \
VLA_CACHE_DIR=/storage/openvla/cache \
VLA_DATA_ROOT=/storage/openvla/data/modified_libero_rlds \
bash run_all.sh
```

Use a different run directory for each experiment.

## Resume interrupted training

Checkpoints are saved every 5,000 updates. To resume from the latest saved checkpoint:

```bash
VLA_RUN_DIR=/absolute/path/to/existing/run \
RESUME=1 \
bash run_all.sh
```

Keep the scripts and configuration unchanged. Resume restores the model, optimiser and saved random states, but restarts the dataset shuffle stream.

If no checkpoint was saved, start a new run directory.

## Run evaluation separately

After training and merging have completed:

```bash
bash run_evaluation.sh /absolute/path/to/run
```

## Results

The run directory contains:

- `logs/`: setup, hardware and execution logs.
- `training/`: checkpoints, final adapters and training metrics.
- `merged/`: the merged models.
- `evaluation/`: episode records and evaluation results.
- `FINAL_EXPERIMENT_SUMMARY.json`: the overall comparison.
- `FINAL_PER_TASK_RESULTS.csv`: results for each task.

## If a run fails

Collect the logs and error records:

```bash
bash collect_logs.sh /absolute/path/to/run
```

This creates `diagnostics.tar.gz` without including model weights.

## Experiment settings

| Setting | Uniform LoRA | MMD-selective LoRA |
| --- | --- | --- |
| Decoder layers | All 32 | Layers 9–23 and 25 |
| Decoder rank / alpha | 32 / 16 | 64 / 32 |
| Non-decoder adapters | Rank 32 / alpha 16 | Same modules and settings |
| Trainable parameters | 110,828,288 | 110,828,288 |

Both models start independently from the same pretrained checkpoint. They use BF16, FP32 adapters, batch size 16, gradient accumulation 8, learning rate `5e-4`, image augmentation and gradient checkpointing.

The MMD layer selection is already fixed. The scripts do not recompute it.
