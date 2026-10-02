# OpenVLA uniform versus MMD-selective LoRA

A Bash-launched, headless Linux experiment. No notebook, desktop, tracking service or multi-GPU distributed training is required. This package is for the university machine, not the 4 GB laptop.

## Start here

1. Obtain access to the allocated GPU machine through the institution's normal procedure. Headless does not establish whether it uses a scheduler. Do not run training on a shared login node.
2. Copy this folder, or clone your GitHub repository, onto a writable filesystem with sufficient storage.
3. Keep the GPU allocation supplied by the operator or scheduler. Do not select GPUs merely because they appear in `nvidia-smi`.
4. Inside the folder, run:

```bash
bash run_all.sh
```

This records hardware diagnostics first, sets up both environments, prepares assets, checks all ten simulator tasks, runs both real training/save/reload smoke tests, checks model and simulator together, then trains, merges and evaluates both conditions. Every gate must pass before long training begins. A smoke pass checks execution; it does not require successful task completion by a barely trained model.

For an explicit university smoke without starting long training:

```bash
bash run_smoke.sh
```

To continue that same run into the full experiment, use the printed absolute run directory:

```bash
VLA_RUN_DIR=/absolute/path/to/run bash run_all.sh
```

The smoke runs again; it uses separate output folders and does not overwrite training. Do not run the university scripts on the laptop as a substitute for this hardware test.

## What the experiment does

Both conditions independently start from the same untouched `openvla/openvla-7b` pretrained base and train on the same LIBERO-Object RLDS data.

| Setting | Uniform | MMD-selective |
| --- | --- | --- |
| Decoder coverage | All 32 layers | Layers 9–23 and 25 |
| Decoder LoRA rank / alpha | 32 / 16 | 64 / 32 on the selected 16 layers |
| Non-decoder LoRA modules | Rank 32 / alpha 16 | Same modules, rank 32 / alpha 16 |
| Trainable parameters | 110,828,288 | 110,828,288 |
| Successful optimiser updates | 50,000 | 50,000 |

The MMD layer selection is already frozen. This package does not rerun selection or add an MMD loss/network layer. It tests how allocating the same adapter parameter budget to those selected layers affects performance. Equal adapter budget and update count do not establish equal total compute; offline MMD-selection cost and measured runtime must be disclosed.

Shared settings: BF16 frozen base, FP32 trainable adapters, no 4-bit quantisation, physical batch 16, accumulation 8, effective batch 128, learning rate 5e-4, AdamW, clipping at norm 1, image augmentation, shuffle buffer 100,000, seed 7 and adapter/state checkpoints every 5,000 updates.

Decoder gradient checkpointing with non-reentrant recomputation is enabled for both conditions to reduce activation memory. This is an explicit memory-management addition to the old university draft; it is not the 4-bit Kaggle path. Its runtime cost is recorded through training metrics. There is no automatic reduction of batch size, change of precision or quantisation fallback if the fixed configuration does not fit. The real production smoke must establish memory fit on the allocated hardware.

The university smoke performs two optimiser updates per condition with the same batch, accumulation, precision, augmentation, gradient checkpointing and shuffle-buffer size as production, followed by fresh-process adapter reload and finite seven-dimensional action inference. It then verifies those adapters alongside the simulator before long training begins.

## GPU selection and launch options

An existing `CUDA_VISIBLE_DEVICES` allocation is respected, including GPU UUIDs. Children receive the original allocation token for their assigned device, rather than assumed physical GPU 0/1. Inside a child, that assigned device is locally `cuda:0`.

- One visible suitable GPU: run conditions sequentially.
- Two or more: use the first two visible allocated GPUs for two independent single-GPU conditions.
- Force sequential execution without changing scientific settings:

```bash
VLA_SEQUENTIAL=1 bash run_all.sh
```

The package does not request GPUs, detect which unallocated GPUs are free, or distribute one condition across multiple GPUs. The operator must supply an authorised allocation. MIG device layouts require site-specific review and a working rendering configuration.

Use environment variables to choose storage locations:

```bash
VLA_RUN_DIR=/large-volume/vla/run_01 \
VLA_CACHE_DIR=/large-volume/vla/cache \
VLA_DATA_ROOT=/large-volume/vla/data/modified_libero_rlds \
bash run_all.sh
```

Paths containing spaces are supported. Avoid simultaneous experiments sharing this folder's environments and source directories; setup locks reject concurrent setup. Different completed experiments should have different run directories.

Preflight conservatively requires 200 GiB free at the repository, run, cache and data locations. This is a reserve, not a measured exact footprint. Completed caches/checkpoints can change the requirement. The operator may override `VLA_MIN_FREE_GIB` after checking storage, for example when reusing fully downloaded caches; do not lower it merely to bypass insufficient space. Keep checkpoints, environments and symlinked base-model caches available until the experiment is complete.

## Linux and Python setup

Supported target: x86_64 Linux with a working NVIDIA driver, BF16-capable allocated GPU(s), and enough GPU/host memory for the fixed configuration. An H100 is the intended target, but its actual allocation and memory are checked at runtime.

Required shell tools: Bash, Git, curl, tar, sha256sum, timeout and flock. GPU diagnostics require `nvidia-smi`. Required native renderer libraries must be supplied by the institution. The scripts do not modify system Python, install system packages with sudo, or replace NVIDIA drivers.

Python 3.11 or 3.10 is used. A specific interpreter can be selected with `PYTHON_BIN=/absolute/path/to/python3.11`. If neither is available, the package downloads a SHA256-verified uv 0.8.22 binary and uses it to install user-local Python 3.11.13. A pre-existing interpreter still needs `venv`/ensurepip support. If that is absent, use an administrator-provided compatible interpreter.

Training uses `.venv`; evaluation uses `.venv-eval`. TensorFlow CPU 2.15.1 handles RLDS data without reserving GPU memory. PyTorch 2.5.1 / torchvision 0.20.1 use CUDA 12.1 wheels by default. The script checks actual BF16 tensor execution, not just driver visibility. A site mirror can be supplied through standard pip settings or `VLA_TORCH_INDEX`, but the same versioned packages must remain available.

The first setup needs internet access to GitHub, Hugging Face, PyPI and download.pytorch.org; managed Python bootstrap also fetches its release assets through GitHub. A disconnected site needs assets and wheels prepared by the administrator; an arbitrary offline machine is not automatically supported.

## Pinned sources and compatibility adjustments

- OpenVLA commit: `c8f03f48af692657d3060c19588038c7220e9af9`.
- Base model revision: `cf0c8cbad1e427993a2e02da9c5182a91c33f869`.
- LIBERO commit: `8f1084e3132a39270c3a13ebe37270a43ece2a01`.
- Dataset revision: `6ce6aaaaabdbe590b1eef5cd29c0d33f14a08551`.
- dlimp commit: `040105d256bd28866cc6620621a3d5f7b6b91b46`.
- Key Python requirements and constraints are stored alongside this README. Resolved transitive package versions are saved in each environment's pip-freeze log; this is not a universal offline wheel lock.

Pinned OpenVLA, dlimp and LIBERO source folders are attached using a `.pth` file, rather than installing their obsolete dependency declarations. Required runtime packages are installed explicitly and `pip check` must pass. The unused DROID-only tensorflow_graphics import is disabled; this package is only for LIBERO-Object.

robosuite 1.4.1's wheel is downloaded and repacked with only its dependency declaration changed from `opencv-python` to `opencv-python-headless`. Its Python code is unchanged, and wheel RECORD hashes are rebuilt. This avoids installing two distributions that both own `cv2` or requiring a display for OpenCV. All other declared robosuite dependencies are installed explicitly.

No FlashAttention compilation is required; attention uses SDPA.

## Headless evaluation

Setup creates LIBERO's configuration file before import, preventing its interactive first-run prompt. Renderer probing tries OSMesa, then EGL, in separate processes and records the successful backend. Explicit `VLA_RENDERER=osmesa` or `VLA_RENDERER=egl` requests only that backend. Existing `MUJOCO_EGL_DEVICE_ID` or `VLA_EGL_DEVICE_ID` takes priority; UUID allocations are mapped through `nvidia-smi` when necessary. Site-specific GPU/container rendering still needs to pass the actual probe.

All ten task environments and required initial states are checked before training. Both smoke adapters are tested with the model CUDA context and simulator together. Evaluation later repeats the renderer check and the combined model/simulator check for the merged models.

Evaluation uses LIBERO-Object, centre-crop preprocessing, ten settling actions, up to 280 policy actions per episode, one dry-run episode per task and 15 full episodes per task: 150 episodes per model. Both conditions use the same initial-state indices and seeds. Task success is measured by simulator completion. Raw per-episode outcomes are saved incrementally, including policy steps and elapsed time. A failed rollout that raises an exception is recorded as a technical failure, not silently counted as a task failure.

## Checkpoints and recovery

Each checkpoint commits adapter weights, optimiser state and Python/NumPy/PyTorch RNG states together, then atomically updates `latest_checkpoint.json`. An interrupted save cannot point to a half-written new checkpoint. A completed condition is reused on resume rather than trained beyond 50,000 updates.

Resume an interrupted long run:

```bash
VLA_RUN_DIR=/absolute/path/to/existing/run RESUME=1 bash run_all.sh
```

The original package and configuration must remain unchanged. A run stores frozen configuration and source hashes and rejects mismatches. Metrics written after the latest committed checkpoint are removed before continuing, to avoid duplicate/uncommitted step records. If interruption occurred before the first checkpoint, start a new run directory.

RLDS does not expose a serialisable exact internal shuffle-stream position. Resume therefore restores model/optimiser/RNG state but recreates the data stream. `resume_disclosure.json` records this limitation; report any resumed production runs. Checkpoint commits protect against ordinary process interruption, not storage hardware failure or loss of the filesystem.

Final merging uses a temporary directory and a completion identity, so completed merges are reused on restart. Evaluation stores checkpoint identity and settings with episode progress and rejects incompatible resume settings.

Run evaluation again without retraining:

```bash
bash run_evaluation.sh /absolute/path/to/existing/run
```

## Outputs and diagnosis

Important outputs under the printed run directory:

| Path | Contents |
| --- | --- |
| `logs/` | Preflight, setup, runtime versions, stage logs, stage exit codes and training/evaluation output |
| `config.json`, `source_manifest.json`, `assets.json` | Frozen settings, code hashes and pinned asset paths |
| `smoke/` | Small university training/save/reload runs |
| `training/uniform/`, `training/mmd_selective/` | Metrics, committed checkpoints, final adapters, statistics and completion/failure records |
| `merged/uniform/`, `merged/mmd_selective/` | Final BF16 models and merge identities |
| `evaluation/` | Dry/full episode progress and per-task results |
| `FINAL_EXPERIMENT_SUMMARY.json` | Overall comparison and training provenance |
| `FINAL_PER_TASK_RESULTS.csv` | Task-level success-rate comparison |

If anything fails:

```bash
bash collect_logs.sh /absolute/path/to/run
```

Send the resulting `diagnostics.tar.gz`. It includes logs and relevant JSON diagnostics, without model weights or optimiser tensors. Setup/import failures may appear only in stage logs; training/evaluation exceptions include Python tracebacks and, once those modules have started, failure JSON.

## File inventory

`run_all.sh`, `run_smoke.sh`, `run_training.sh`, `run_evaluation.sh`, `preflight.sh`, `setup_env.sh`, `setup_eval_env.sh`, `check_renderer.sh`, `common.sh`, `collect_logs.sh`, `config.json`, requirement files, constraints, and the Python scripts under `scripts/`.

`validate_package.sh` runs syntax/configuration and offline orchestration/checkpoint tests without downloading models. `VALIDATION.json` records the checks performed during preparation and their limits. The optional `slurm/job.slurm.example` is a site-adapted template only; this package does not assume Slurm.

The successful Kaggle notebook established the 4-bit T4 training/save/reload path. The local laptop check established Bash/venv/logging. Neither establishes the university BF16 batch-16 path or final task performance. This package must pass its real on-machine gates. No first-run success guarantee is claimed without executing it on that machine.

## GitHub upload

Upload the contents of this folder to your existing VLE repository. Do not upload generated environments, tools, caches, datasets, runs or weights. They are excluded by `.gitignore`. This package itself has not been pushed to GitHub.

Preparation checks also installed the Python 3.11 dependency set, passed `pip check`, imported pinned OpenVLA/dlimp/TFDS, audited both actual model architectures using meta tensors, and passed the simulator-only EGL preflight for all ten tasks. `VALIDATION_DEPENDENCIES_CPU.txt` records that CPU validation environment; it is diagnostic provenance, not an installation requirements file. These checks do not establish CUDA memory fit or real model inference.
