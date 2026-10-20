# OpenVLA MMD-selective LoRA: cluster fixes, 20 October 2026

Complete Bash package for the independent, equal-parameter comparison on LIBERO-Object.
Start with the production-profile smoke test. Its results decide readiness and runtime.
This source release has passed offline checks; the revised single-GPU cluster pipeline
has not yet been executed on Glory’s Blackwell/Ada nodes.

## Hardware and allocation

Prefer `dmo-node1`: RTX PRO 6000 Blackwell, approximately 96 GB per GPU. The CUDA
12.8 PyTorch build supports this architecture. `seal-node1` has RTX 6000 Ada,
approximately 48 GB per GPU; the unchanged batch-16 profile must pass the smoke
memory test there before long training. No memory-fit claim is made for Ada.

Use x86_64 Linux with the existing NVIDIA driver, Internet, Bash, Git, curl, tar,
sha256sum, timeout, flock and system EGL or OSMesa libraries. Do not install or
replace the system CUDA Toolkit or driver. SLURM is not a dependency. Run directly
on a GPU node under the lab’s allocation policy; the optional `slurm/` file is only
a scheduler example.

**Set `CUDA_VISIBLE_DEVICES` to the GPU indices or UUIDs actually allocated to you.**
The script refuses to choose GPUs automatically on a shared node. Preserve any
allocation already supplied by the lab. It uses at most the first two visible
GPUs, one independent model on each. With one visible GPU, the models run
sequentially. They never share or pool model memory.

At least 64 GiB of available host memory is required for automatic parallel mode
because the original two 100,000-item shuffle buffers are retained. With less
available memory (including cgroup limits), the models run sequentially. This is
an operational safeguard, not proof that either memory profile fits. Use
`VLA_SEQUENTIAL=1` to require sequential mode. `VLA_PARALLEL=1` overrides the host
memory gate only when the allocation’s memory has been checked.

## Python and installation

The package uses its own `.venv-cluster` for both training and evaluation, with
PyTorch 2.7.1+cu128 and torchvision 0.22.1+cu128. Python 3.10/3.11 is required by
the pinned TensorFlow CPU/RLDS stack. A compatible system interpreter is used
when available; otherwise the pinned user-local Python 3.11.13 is downloaded.

A separately activated Python 3.12 venv and old `.venv`/`.venv-eval` folders are
not reused or deleted. If `.venv-cluster` itself is incompatible, it is preserved
under a backup name and rebuilt. No sudo is needed for Python. If `PYTHON_BIN`
is explicitly set to Python 3.12, unset it to permit automatic selection.
Dependency downloads disable pip’s wheel cache. Installation scratch is placed
under `tools/tmp`. Completed installations are reused when the dependency
fingerprint and `pip check` agree.

## First run: short smoke test

From the repository directory, after GPU allocation is set:

```bash
bash validate_package.sh
sha256sum -c SHA256SUMS
export VLA_RUN_DIR="$PWD/runs/cluster_01"
bash run_smoke.sh
```

This performs hardware/resource checks, pinned downloads, text/image processing,
headless rendering, three **complete optimiser updates** per condition, atomic
checkpoint saves, a process restart after update one, optimiser/RNG restoration
and two further updates, fresh-process adapter reload and action inference,
followed by model-plus-simulator checks. It also exercises the production CPU
merge, saves and reloads the merged model, and tests it with the simulator.
Temporary merged smoke models are removed after successful checks, one at a time. It stops before long training. These checks test
execution, not policy quality. Initial downloads and installation take extra time.

Collect the output to send back:

```bash
bash collect_logs.sh "$VLA_RUN_DIR"
```

Send `diagnostics.tar.gz` from that directory. It includes per-condition
`timing.json`, hardware/environment records, metrics and error logs, but no
weights. The smoke timing excludes the first update for its estimate; one timed
update after restart per condition is preliminary and must not be treated as a long-run benchmark.

## Full run after the smoke passes

Keep the same terminal environment, allocation, repository and run directory:

```bash
bash run_all.sh
```

Or set the printed smoke run directory explicitly in a later shell:

```bash
VLA_RUN_DIR=/absolute/path/to/runs/cluster_01 bash run_all.sh
```

Completed smoke training is reused only within that frozen source/configuration
run. Renderer and combined simulator checks are repeated. Both conditions start
full training independently from the original pretrained model, not the smoke
adapters. Full training is followed by one merge per model, dry evaluation and
150 full evaluation episodes per condition (10 tasks × 15 trials).

For a long direct Bash run, an existing `tmux` session can keep the process alive
when SSH disconnects. Follow the lab’s policy for shared GPU access and job duration.
Do not change this package in place after an experiment has started.

## Storage: the 108 GB case

The former blanket 200 GiB gate is replaced with a per-filesystem additional-space
budget in `logs/storage_budget.json`. For a fresh run with everything on one
filesystem, the estimate is about **91.05 GiB (97.8 decimal GB)** including a
10 GiB safety margin. Thus 108 decimal GB of actual free space fits the estimate.
It is not a measured end-to-end peak guarantee.

| Component | Planning allowance |
| --- | ---: |
| Shared environment | 16 GiB |
| Bootstrap/install scratch | 6 GiB |
| Pinned base weights and metadata | 14.05 GiB |
| Processing/source metadata | 0.25 GiB |
| LIBERO-Object RLDS | 2.62 GiB |
| Checkpoint saves, including atomic-save overlap | 7.43 GiB |
| Final adapters and smoke outputs | 6.61 GiB |
| Both merged models | 28.09 GiB |
| Safety margin | 10 GiB |

Only two committed full-training checkpoints per condition are retained. The new
save and pointer commit before older committed checkpoints are removed. Failed
partial files and previous runs are kept for diagnosis; they consume real space.
The check accounts for recognised existing assets. It checks each distinct
filesystem once, grouping repo/run/cache/data components on that filesystem.
Filesystem free space does not reveal all user quotas and can change while other
jobs run. Review the report; old installations and unrelated files are never
removed automatically. Keep only one active experiment within this storage budget.

To use a separate larger filesystem:

```bash
VLA_RUN_DIR=/storage/openvla/run_01 \
VLA_CACHE_DIR=/storage/openvla/cache \
VLA_DATA_ROOT=/storage/openvla/data/modified_libero_rlds \
bash run_smoke.sh
```

## Resume and evaluation-only runs

Checkpoints are saved every 500 successful optimiser updates. A forced stop can
lose work after the last committed checkpoint. Keep the complete original run and
source package. To resume full training:

```bash
VLA_RUN_DIR=/absolute/path/to/existing/run RESUME=1 bash run_all.sh
```

Resume restores the adapter, AdamW state, update count and Python/NumPy/PyTorch
random states. The RLDS shuffle/augmentation stream is recreated; this is
**not exact data-stream continuation**, and the disclosure is saved. If no
checkpoint committed, begin a new run. Evaluation progress is stored separately.
If training/merging completed but evaluation failed:

```bash
bash run_evaluation.sh /absolute/path/to/existing/run
```

## Scientific configuration

The scientific training and evaluation settings are retained. The checkpoint
interval is now 500 updates and smoke qualification uses three updates.
A separate processing-asset revision is included in `config.json`:
the original model revision contains no tokenizer/processor files. Model weights
remain pinned to `cf0c8cbad1e427993a2e02da9c5182a91c33f869`. Seven official processing
files are pinned separately to `47a0ec7fc4ec123775a391911046cf33cf9ed83f`.
Explicit pinned source classes load the processor and model in all stages.

| Setting | Uniform | MMD-selective |
| --- | --- | --- |
| Decoder layers | All 32 | Layers 9–23 and 25 |
| Decoder rank / alpha | 32 / 16 | 64 / 32 |
| Non-decoder adapters | Rank 32 / alpha 16 | Same modules/settings |
| Trainable parameters | 110,828,288 | 110,828,288 |

Both use unquantised BF16 base weights, FP32 adapters, physical batch 16,
accumulation 8, effective batch 128, learning rate 5e-4, image augmentation,
decoder gradient checkpointing and 50,000 optimiser updates. There is no DDP or
FSDP between conditions. The fixed MMD selection is not recomputed. Lowering the
batch automatically after an OOM is deliberately not implemented; the smoke must
qualify the actual profile.

## Results and validation

`training/` holds adapters, retained checkpoints, metrics and timings; `merged/`
holds evaluation models; `evaluation/` holds episode records. The final outputs
are `FINAL_EXPERIMENT_SUMMARY.json` and `FINAL_PER_TASK_RESULTS.csv`.

`VALIDATION.json` and `validation_logs/` describe this release’s local checks.
`validation_history/` contains historical reports and is not evidence that the
new release ran on a cluster. The successful Kaggle qualification established
unquantised FP16 FSDP training and fresh-process restoration for that different
trainer; it does not qualify the single-GPU BF16 cluster recipe. Blackwell/Ada
execution, full batch memory fit, host-memory fit, renderer availability and the
full experiment remain subject to the target-machine smoke and subsequent run.

## Changes in the 20 October replacement package

Use a fresh run directory after applying these source/configuration changes.
Do not replace source files while a run is active. Existing frozen runs require
their original source/configuration to resume; this update does not migrate them.

Host-memory checks resolve the process cgroup from `/proc/self/cgroup` and its
mount from `/proc/self/mountinfo`, including visible ancestor limits on cgroup v1
and v2. A 64 GiB parallel threshold is still a heuristic, not a measured peak
requirement. Dataset statistics initialisation is protected by an advisory file
lock shared by both training conditions on the dataset filesystem.

OSMesa is tried first. If it is unavailable, EGL requires an administrator-verified
EGL device index, supplied as `VLA_EGL_DEVICE_ID`. This is the EGL enumeration
index for an allocated GPU, which must not be assumed equal to a CUDA index.
For example, after the administrator supplies the correct index:

```bash
export VLA_RENDERER=egl
export VLA_EGL_DEVICE_ID=VERIFIED_INDEX
```

Replace `VERIFIED_INDEX` with that numeric index. The script does not install
system renderer libraries. If neither backend works, diagnostics explain the
failure and full training does not start.

Full training remains 50,000 updates per condition, physical batch 16,
accumulation 8, with the same adapters, optimiser and learning rate. Only the
checkpoint interval (500) and smoke qualification (three updates) change.
Smoke resume does not imply exact RLDS stream restoration; that limitation remains.
The merge check validates execution and finite actions, not policy quality or
numerical equivalence between merged and unmerged predictions.
