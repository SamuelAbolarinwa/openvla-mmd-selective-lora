import argparse
import gc
import json
import math
import os
import random
import sys
import time
import traceback
from pathlib import Path

RENDERER = os.environ.get("VLA_RENDERER", "osmesa").lower()
if RENDERER not in {"osmesa", "egl"}:
    raise RuntimeError(f"Unsupported VLA_RENDERER={RENDERER}")
os.environ.setdefault("MUJOCO_GL", RENDERER)
os.environ.setdefault("PYOPENGL_PLATFORM", RENDERER)
if RENDERER == "egl":
    os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", os.environ.get("VLA_EGL_DEVICE_ID", "0"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import tensorflow as tf
import torch
from PIL import Image
from peft import PeftModel
import hashlib
from transformers import AutoModelForVision2Seq, AutoProcessor

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

TASK_SUITE = "libero_object"
IMAGE_SIZE = 224
ENV_RESOLUTION = 256
DUMMY_ACTION = [0, 0, 0, 0, 0, 0, -1]


def atomic_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def suite():
    s = benchmark.get_benchmark_dict()[TASK_SUITE]()
    if s.n_tasks != 10:
        raise RuntimeError(f"Expected 10 LIBERO-Object tasks, got {s.n_tasks}")
    return s


def make_env(task, env_seed):
    bddl = os.path.join(
        get_libero_path("bddl_files"),
        task.problem_folder,
        task.bddl_file,
    )
    env = OffScreenRenderEnv(
        bddl_file_name=bddl,
        camera_heights=ENV_RESOLUTION,
        camera_widths=ENV_RESOLUTION,
    )
    env.seed(env_seed)
    return env


def resize_image(img):
    x = tf.convert_to_tensor(img, dtype=tf.uint8)
    x = tf.image.encode_jpeg(x)
    x = tf.io.decode_image(x, expand_animations=False, dtype=tf.uint8)
    x = tf.image.resize(x, (IMAGE_SIZE, IMAGE_SIZE), method="lanczos3", antialias=True)
    x = tf.cast(tf.clip_by_value(tf.round(x), 0, 255), tf.uint8)
    return x.numpy()


def get_image(obs):
    img = obs["agentview_image"]
    img = img[::-1, ::-1]
    return resize_image(img)


def center_crop(image):
    crop_scale = 0.9
    x = tf.convert_to_tensor(np.array(image))
    original_dtype = x.dtype
    x = tf.image.convert_image_dtype(x, tf.float32)
    h = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), (1,))
    w = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), (1,))
    y0 = (1 - h) / 2
    x0 = (1 - w) / 2
    boxes = tf.stack([y0, x0, y0 + h, x0 + w], axis=1)
    x = tf.image.crop_and_resize(
        tf.expand_dims(x, 0), boxes, tf.range(1), (IMAGE_SIZE, IMAGE_SIZE)
    )[0]
    x = tf.clip_by_value(x, 0, 1)
    x = tf.image.convert_image_dtype(x, original_dtype, saturate=True)
    return Image.fromarray(x.numpy()).convert("RGB")


def load_model(checkpoint, device_index, adapter=None, stats_path=None):
    stats = json.loads((stats_path or checkpoint / "dataset_statistics.json").read_text())
    key = (
        "libero_object_no_noops"
        if "libero_object_no_noops" in stats
        else "libero_object"
        if "libero_object" in stats
        else None
    )
    if key is None:
        raise KeyError(f"No LIBERO-Object stats in {checkpoint}")

    torch.cuda.set_device(device_index)
    processor = AutoProcessor.from_pretrained(
        checkpoint, trust_remote_code=True, local_files_only=True
    )
    model = AutoModelForVision2Seq.from_pretrained(
        checkpoint,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    ).to(f"cuda:{device_index}")
    if adapter is not None:
        model = PeftModel.from_pretrained(model, adapter, is_trainable=False).base_model.model
    model.eval()
    model.norm_stats = stats
    return model, processor, key


def predict(model, processor, image_np, instruction, key, device_index):
    image = center_crop(Image.fromarray(image_np).convert("RGB"))
    prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"
    inputs = processor(prompt, image)
    moved = {}
    for k, v in inputs.items():
        if torch.is_tensor(v):
            moved[k] = (
                v.to(f"cuda:{device_index}", dtype=torch.bfloat16)
                if torch.is_floating_point(v)
                else v.to(f"cuda:{device_index}")
            )
        else:
            moved[k] = v
    with torch.inference_mode():
        action = model.predict_action(**moved, unnorm_key=key, do_sample=False)
    action = np.asarray(action, dtype=np.float64)
    if action.shape != (7,) or not np.all(np.isfinite(action)):
        raise RuntimeError(f"Invalid action: shape={action.shape}, value={action}")
    action[-1] = -1.0 if action[-1] > 0.5 else 1.0
    return action


def renderer_preflight(trials, env_seed):
    print("Renderer:", RENDERER)
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("MUJOCO_EGL_DEVICE_ID:", os.environ.get("MUJOCO_EGL_DEVICE_ID"))
    print("GPU count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print("GPU", i, torch.cuda.get_device_name(i))

    s = suite()
    for task_id in range(s.n_tasks):
        task = s.get_task(task_id)
        states = s.get_task_init_states(task_id)
        if len(states) < trials:
            raise RuntimeError(
                f"Task {task_id}: only {len(states)} init states; need {trials}"
            )
        env = make_env(task, env_seed)
        try:
            env.reset()
            obs = env.set_init_state(states[0])
            if "agentview_image" not in obs:
                raise KeyError("agentview_image missing")
        finally:
            env.close()
    print("LIBERO RENDERER PREFLIGHT: PASS")


def combined_smoke(checkpoint, device_index, env_seed, adapter=None, stats_path=None):
    # This specifically checks that model CUDA context and simulator rendering coexist.
    model, processor, key = load_model(checkpoint, device_index, adapter, stats_path)
    s = suite()
    task = s.get_task(0)
    states = s.get_task_init_states(0)
    env = make_env(task, env_seed)
    try:
        env.seed(env_seed)
        env.reset()
        obs = env.set_init_state(states[0])
        for _ in range(10):
            obs, _, done, _ = env.step(DUMMY_ACTION)
            if done:
                raise RuntimeError("Task ended during settle steps")
        image_np = get_image(obs)
        # Base/merged checkpoint must contain LIBERO stats for actual action smoke.
        action = predict(model, processor, image_np, task.language, key, device_index)
        env.step(action.tolist())
        print("COMBINED MODEL + LIBERO SMOKE: PASS")
    finally:
        env.close()
        del model, processor
        gc.collect()
        torch.cuda.empty_cache()


def evaluation_identity(checkpoint, trials, seed, env_seed, max_steps, settle_steps):
    return {
        'checkpoint':str(checkpoint.resolve()), 'trials':trials, 'seed':seed,
        'env_seed':env_seed, 'max_steps':max_steps, 'settle_steps':settle_steps,
        'config_sha256':hashlib.sha256((checkpoint/'config.json').read_bytes()).hexdigest(),
        'evaluation_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'merge_identity':json.loads((checkpoint/'MERGE_COMPLETE.json').read_text()),
    }


def load_progress(progress_path, identity):
    if progress_path.exists():
        progress = json.loads(progress_path.read_text())
        if progress.get('identity') != identity:
            raise RuntimeError('Existing evaluation progress belongs to different code/model/settings; use the original package or a fresh output directory')
        return progress
    return {'identity': identity, 'episodes': {}}


def evaluate(checkpoint, out_dir, trials, seed, env_seed, device_index, max_steps, settle_steps):
    set_seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = out_dir / "progress.json"
    results_path = out_dir / "results.json"
    failure_path = out_dir / "failure.json"

    identity = evaluation_identity(checkpoint, trials, seed, env_seed, max_steps, settle_steps)
    progress = load_progress(progress_path, identity)
    if trials <= 0 or max_steps <= 0 or settle_steps < 0: raise ValueError('Invalid evaluation counts')

    try:
        model, processor, key = load_model(checkpoint, device_index)
        s = suite()
        total_successes = 0
        total_episodes = 0
        per_task = []

        for task_id in range(s.n_tasks):
            task = s.get_task(task_id)
            states = s.get_task_init_states(task_id)
            if len(states) < trials:
                raise RuntimeError(
                    f"Task {task_id}: only {len(states)} init states; need {trials}"
                )
            env = make_env(task, env_seed)
            task_successes = 0
            try:
                for ep in range(trials):
                    key_ep = f"{task_id}:{ep}"
                    if key_ep in progress["episodes"]:
                        success = bool(progress["episodes"][key_ep]["success"])
                        task_successes += int(success)
                        total_successes += int(success)
                        total_episodes += 1
                        print(f"[resume] task={task_id} ep={ep} success={success}")
                        continue

                    env.seed(env_seed)
                    env.reset()
                    obs = env.set_init_state(states[ep])
                    success = False
                    start = time.time()

                    for t in range(max_steps + settle_steps):
                        if t < settle_steps:
                            obs, _, done, _ = env.step(DUMMY_ACTION)
                            if done:
                                raise RuntimeError("Task ended during settle steps")
                            continue
                        image_np = get_image(obs)
                        action = predict(
                            model, processor, image_np, task.language, key, device_index
                        )
                        obs, _, done, _ = env.step(action.tolist())
                        if done:
                            success = True
                            break

                    rec = {
                        "task_id": task_id,
                        "task": task.language,
                        "episode_idx": ep,
                        "success": bool(success),
                        "elapsed_seconds": time.time() - start,
                        "policy_steps": max(0, t - settle_steps + 1),
                    }
                    progress["episodes"][key_ep] = rec
                    atomic_json(progress_path, progress)
                    task_successes += int(success)
                    total_successes += int(success)
                    total_episodes += 1
                    print(
                        f"task={task_id} ep={ep} success={success} "
                        f"overall={total_successes}/{total_episodes}",
                        flush=True,
                    )
            finally:
                env.close()

            per_task.append({
                "task_id": task_id,
                "task": task.language,
                "successes": task_successes,
                "episodes": trials,
                "success_rate": task_successes / trials,
            })

        result = {
            "checkpoint": str(checkpoint),
            "identity": identity,
            "center_crop": True,
            "renderer": RENDERER,
            "seed": seed,
            "env_seed": env_seed,
            "trials_per_task": trials,
            "total_successes": total_successes,
            "total_episodes": total_episodes,
            "success_rate": total_successes / total_episodes,
            "per_task": per_task,
        }
        atomic_json(results_path, result)
        if failure_path.exists():
            failure_path.unlink()
        print(
            f"FINAL: {total_successes}/{total_episodes} "
            f"= {100 * result['success_rate']:.2f}%"
        )
        print("Saved:", results_path)
    except Exception as exc:
        atomic_json(failure_path, {
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "completed_episodes": len(progress.get("episodes", {})),
        })
        raise
    finally:
        try:
            del model, processor
        except Exception:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["preflight", "combined_smoke", "evaluate"], required=True)
    ap.add_argument("--checkpoint", type=Path)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--adapter", type=Path)
    ap.add_argument("--stats", type=Path)
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--env-seed", type=int, default=0)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=280)
    ap.add_argument("--settle-steps", type=int, default=10)
    args = ap.parse_args()

    if args.mode == "preflight":
        renderer_preflight(args.trials, args.env_seed)
    elif args.mode == "combined_smoke":
        if args.checkpoint is None:
            ap.error("--checkpoint required")
        combined_smoke(args.checkpoint, args.device, args.env_seed, args.adapter, args.stats)
    else:
        if args.checkpoint is None or args.output is None:
            ap.error("--checkpoint and --output required")
        evaluate(
            args.checkpoint,
            args.output,
            args.trials,
            args.seed,
            args.env_seed,
            args.device,
            args.max_steps,
            args.settle_steps,
        )


if __name__ == "__main__":
    main()
