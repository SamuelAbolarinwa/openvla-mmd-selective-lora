import argparse
import csv
import gc
import json
import os
import random
import shutil
import signal
import re
import subprocess
import time
import traceback
from collections import deque
from pathlib import Path

import numpy as np
import tensorflow as tf
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModelForVision2Seq,
    AutoProcessor,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import (
    PrismaticImageProcessor,
    PrismaticProcessor,
)
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics


def atomic_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


def build_lora_config(condition, selected_layers):
    if condition == "uniform":
        return LoraConfig(
            r=32,
            lora_alpha=16,
            lora_dropout=0.0,
            target_modules="all-linear",
            init_lora_weights="gaussian",
        )

    lp = "|".join(map(str, selected_layers))
    llm_full = (
        rf"language_model\.model\.layers\.({lp})\."
        r"(self_attn\.(q_proj|k_proj|v_proj|o_proj)|"
        r"mlp\.(gate_proj|up_proj|down_proj))"
    )
    fixed_tail = (
        r"(vision_backbone|projector)\..*(fc1|fc2|fc3|proj|q|kv|qkv)"
        r"|language_model\.lm_head"
    )
    target_regex = rf"^({llm_full}|{fixed_tail})$"
    rank_key = (
        rf"layers\.({lp})\."
        r"(self_attn\.(q_proj|k_proj|v_proj|o_proj)|"
        r"mlp\.(gate_proj|up_proj|down_proj))"
    )
    return LoraConfig(
        r=32,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules=target_regex,
        rank_pattern={rank_key: 64},
        alpha_pattern={rank_key: 32},
        init_lora_weights="gaussian",
    )


def audit_peft(model, condition, selected_layers, expected_trainable):
    layer_re = re.compile(r"language_model\.model\.layers\.(\d+)\.")
    wrapped = {}
    for name, module in model.named_modules():
        if hasattr(module, "lora_A") and "default" in module.lora_A:
            wrapped[name] = {
                "rank": int(module.r["default"]),
                "alpha": int(module.lora_alpha["default"]),
            }

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable != expected_trainable:
        raise RuntimeError(
            f"Trainable parameter mismatch: {trainable:,} != {expected_trainable:,}"
        )

    observed_layers = sorted({
        int(layer_re.search(name).group(1))
        for name in wrapped
        if layer_re.search(name) is not None
    })
    expected_layers = list(range(32)) if condition == "uniform" else selected_layers
    if observed_layers != expected_layers:
        raise RuntimeError(
            f"{condition}: decoder-layer coverage mismatch: {observed_layers}"
        )

    for name, info in wrapped.items():
        lm = layer_re.search(name)
        if lm is None or condition == "uniform":
            expected = {"rank": 32, "alpha": 16}
        else:
            expected = {"rank": 64, "alpha": 32}
        if info != expected:
            raise RuntimeError(
                f"{condition}: wrong LoRA config for {name}: {info}; expected {expected}"
            )

    non_llm = sorted(name for name in wrapped if layer_re.search(name) is None)
    print(
        f"PEFT AUDIT PASS | condition={condition} | "
        f"trainable={trainable:,} | layers={observed_layers} | "
        f"non_llm_modules={len(non_llm)}",
        flush=True,
    )
    return {
        "trainable_params": trainable,
        "decoder_layers": observed_layers,
        "non_llm_module_count": len(non_llm),
        "non_llm_modules": non_llm,
    }


def save_checkpoint(
    model,
    processor,
    optimizer,
    run_dir,
    step,
    condition,
    config_snapshot,
    audit,
    metrics_tail,
):
    checkpoints = run_dir / 'checkpoints'
    checkpoints.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = checkpoints / f'step-{step:06d}-{time.time_ns()}'
    temporary = checkpoints / f'.step-{step:06d}-{os.getpid()}'
    if temporary.exists(): shutil.rmtree(temporary)
    adapter_dir = temporary / 'adapter' 
    adapter_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(
        adapter_dir,
        safe_serialization=True,
        save_embedding_layers=False,
    )
    processor.save_pretrained(run_dir / "processor")

    meta = {
        "condition": condition,
        "optimizer_step": step,
        "adapter_dir": str(checkpoint_dir.relative_to(run_dir) / 'adapter'),
        "audit": audit,
        "config": config_snapshot,
        "metrics_tail": metrics_tail,
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    atomic_json(temporary / "checkpoint.json", meta)

    state = {
        "optimizer_step": step,
        "optimizer_state": optimizer.state_dict(),
        "config": config_snapshot,
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
        "adapter_dir": str(checkpoint_dir.relative_to(run_dir) / 'adapter'),
    }
    torch.save(state, temporary / 'training_state.pt')
    if checkpoint_dir.exists():
        raise RuntimeError(f'Refusing to replace an existing checkpoint: {checkpoint_dir}')
    os.replace(temporary, checkpoint_dir)
    atomic_json(run_dir / 'latest_checkpoint.json', {
        'optimizer_step': step,
        'adapter_dir': str(checkpoint_dir.relative_to(run_dir) / 'adapter'),
        'state_file': str(checkpoint_dir.relative_to(run_dir) / 'training_state.pt'),
    })
    print(f"CHECKPOINT SAVED: step={step} -> {checkpoint_dir}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", choices=["uniform", "mmd_selective"], required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--assets", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--max-steps", type=int)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--grad-accum", type=int)
    ap.add_argument("--shuffle-buffer", type=int)
    ap.add_argument("--save-steps", type=int)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    cfg = json.loads(args.config.read_text())
    assets = json.loads(args.assets.read_text())
    train_cfg = cfg["training"]

    max_steps = args.max_steps or int(train_cfg["max_steps"])
    batch_size = args.batch_size or int(train_cfg["batch_size"])
    grad_accum = args.grad_accum or int(train_cfg["grad_accumulation_steps"])
    shuffle_buffer = args.shuffle_buffer or int(train_cfg["shuffle_buffer_size"])
    save_steps = args.save_steps or int(train_cfg["save_steps"])
    condition = args.condition
    seed = int(cfg["seed"])

    run_dir = args.output_root / condition
    run_dir.mkdir(parents=True, exist_ok=True)
    status_path = run_dir / "status.json"
    failure_path = run_dir / "failure.json"
    metrics_path = run_dir / "training_metrics.csv"

    run_config = {
        "condition": condition,
        "max_steps": max_steps,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "effective_batch": batch_size * grad_accum,
        "shuffle_buffer": shuffle_buffer,
        "save_steps": save_steps,
        "learning_rate": train_cfg["learning_rate"],
        "max_grad_norm": train_cfg["max_grad_norm"],
        "image_aug": train_cfg["image_aug"],
        "gradient_checkpointing": train_cfg["gradient_checkpointing"],
        "model_revision": assets["model_revision"],
        "dataset_revision": assets["dataset_revision"],
        "seed": seed,
        "smoke": args.smoke,
        "base_model_overlay": assets["base_model_overlay"],
        "dataset_root": assets["dataset_root"],
    }

    completion_path = run_dir / 'completion.json'
    if completion_path.exists() and args.resume:
        old = json.loads(completion_path.read_text())
        if old['config'] != run_config or old['optimizer_steps'] != max_steps:
            raise RuntimeError('Completed run configuration does not match requested resume')
        if not (run_dir / 'final_adapter' / 'adapter_model.safetensors').is_file():
            raise RuntimeError('Completed run is missing its adapter')
        print(f'Already complete: {condition}; skipping training', flush=True)
        return
    if not args.resume and not args.smoke and (metrics_path.exists() or completion_path.exists()):
        raise RuntimeError('Existing training output: use RESUME=1 or a new run directory')
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('Termination requested; resume from latest committed checkpoint')))
    atomic_json(status_path, {
        "state": "STARTING",
        "config": run_config,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    })

    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError(
                f"BF16 unsupported on GPU: {torch.cuda.get_device_name(0)}"
            )

        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass

        device = torch.device("cuda:0")
        print("GPU:", torch.cuda.get_device_name(0), flush=True)
        print(
            "VRAM GiB:",
            round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
            flush=True,
        )

        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

        base_path = Path(assets["base_model_overlay"])
        processor = AutoProcessor.from_pretrained(
            base_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        base = AutoModelForVision2Seq.from_pretrained(
            base_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            local_files_only=True,
            attn_implementation="sdpa",
        ).to(device)
        image_sizes = tuple(base.config.image_sizes)
        if train_cfg['gradient_checkpointing']:
            base.language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
            base.language_model.config.use_cache = False
            print('Decoder gradient checkpointing enabled for both conditions', flush=True)

        resume_state_path = None
        pointer = run_dir / 'latest_checkpoint.json'
        if args.resume and pointer.exists():
            pointer_data = json.loads(pointer.read_text())
            resume_state_path = run_dir / pointer_data['state_file']
            if not resume_state_path.is_file(): raise FileNotFoundError(resume_state_path)
        if args.resume and resume_state_path is None and metrics_path.exists():
            raise RuntimeError('No committed checkpoint exists for this interrupted run; start a new run directory')
        start_step = 0

        if args.resume and resume_state_path is not None:
            state = torch.load(resume_state_path, map_location="cpu", weights_only=False)
            if state['config'] != run_config: raise RuntimeError('Checkpoint configuration mismatch')
            adapter_dir = run_dir / state["adapter_dir"]
            if not adapter_dir.exists():
                raise FileNotFoundError(adapter_dir)
            model = PeftModel.from_pretrained(base, adapter_dir, is_trainable=True)
            start_step = int(state["optimizer_step"])
            if start_step > max_steps: raise RuntimeError('Checkpoint exceeds requested training steps')
            print(f"RESUME: loaded adapter from step {start_step}", flush=True)
        else:
            model = get_peft_model(
                base,
                build_lora_config(condition, cfg["mmd_selected_layers"]),
            )

        audit = audit_peft(
            model,
            condition,
            cfg["mmd_selected_layers"],
            int(cfg["expected_trainable_params"]),
        )

        # PEFT LoRA parameters are kept in FP32; the frozen base remains BF16.
        for p in model.parameters():
            if p.requires_grad and p.dtype != torch.float32:
                p.data = p.data.float()

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = AdamW(trainable_params, lr=float(train_cfg["learning_rate"]))

        if args.resume and resume_state_path is not None:
            state = torch.load(resume_state_path, map_location="cpu", weights_only=False)
            optimizer.load_state_dict(state["optimizer_state"])
            restore_state = state
            del state

        action_tokenizer = ActionTokenizer(processor.tokenizer)
        batch_transform = RLDSBatchTransform(
            action_tokenizer,
            processor.tokenizer,
            image_transform=processor.image_processor.apply_transform,
            prompt_builder_fn=PurePromptBuilder,
        )
        dataset = RLDSDataset(
            Path(assets["dataset_root"]),
            cfg["dataset_name"],
            batch_transform,
            resize_resolution=image_sizes,
            shuffle_buffer_size=shuffle_buffer,
            image_aug=bool(train_cfg["image_aug"]),
        )

        if not (run_dir / "dataset_statistics.json").exists():
            save_dataset_statistics(dataset.dataset_statistics, run_dir)

        collator = PaddedCollatorForActionPrediction(
            processor.tokenizer.model_max_length,
            processor.tokenizer.pad_token_id,
            padding_side="right",
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=None,
            collate_fn=collator,
            num_workers=0,
        )

        if args.resume and resume_state_path is not None:
            random.setstate(restore_state['python_random_state'])
            np.random.set_state(restore_state['numpy_random_state'])
            torch.set_rng_state(restore_state['torch_rng_state'])
            torch.cuda.set_rng_state_all(restore_state['cuda_rng_state_all'])
            del restore_state
            if metrics_path.exists():
                with metrics_path.open(newline='') as f:
                    rows = list(csv.DictReader(f))
                fields = list(rows[0]) if rows else []
                rows = [r for r in rows if int(r['optimizer_step']) <= start_step]
                if fields:
                    with metrics_path.open('w', newline='') as f:
                        w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
            atomic_json(run_dir / 'resume_disclosure.json', {
                'checkpoint_step':start_step,
                'exact_rlds_stream_resume':False,
                'note':'Model/optimiser/RNG restored; RLDS shuffle stream recreated',
            })
        mode = "a" if start_step > 0 and metrics_path.exists() else "w"
        with metrics_path.open(mode, newline="") as f:
            writer = csv.writer(f)
            if mode == "w":
                writer.writerow([
                    "optimizer_step",
                    "loss",
                    "action_accuracy",
                    "action_l1",
                    "grad_norm_preclip",
                    "lr",
                    "elapsed_seconds",
                    "gpu_allocated_gib",
                    "gpu_reserved_gib",
                ])

        recent_losses = deque(maxlen=grad_accum)
        recent_accs = deque(maxlen=grad_accum)
        recent_l1s = deque(maxlen=grad_accum)

        optimizer_step = start_step
        micro_step = 0
        start_time = time.time()
        optimizer.zero_grad(set_to_none=True)
        model.train()

        atomic_json(status_path, {
            "state": "RUNNING",
            "optimizer_step": optimizer_step,
            "config": run_config,
            "audit": audit,
        })

        for batch in (() if optimizer_step >= max_steps else loader):
            micro_step += 1

            with torch.autocast("cuda", dtype=torch.bfloat16):
                out: CausalLMOutputWithPast = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    pixel_values=batch["pixel_values"].to(
                        device, dtype=torch.bfloat16
                    ),
                    labels=batch["labels"].to(device),
                )
                loss = out.loss

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss at micro-step {micro_step}: {loss.item()}"
                )

            (loss / grad_accum).backward()

            num_patches = (
                model.base_model.model.vision_backbone.featurizer.patch_embed.num_patches
            )
            action_logits = out.logits[:, num_patches:-1]
            action_preds = action_logits.argmax(dim=2)
            action_gt = batch["labels"][:, 1:].to(action_preds.device)
            mask = action_gt > action_tokenizer.action_token_begin_idx
            if mask.sum().item() == 0:
                raise RuntimeError("No action tokens found in current batch")

            acc = (((action_preds == action_gt) & mask).sum().float() / mask.sum().float())
            pred_cont = torch.tensor(
                action_tokenizer.decode_token_ids_to_actions(
                    action_preds[mask].detach().cpu().numpy()
                )
            )
            gt_cont = torch.tensor(
                action_tokenizer.decode_token_ids_to_actions(
                    action_gt[mask].detach().cpu().numpy()
                )
            )
            l1 = torch.nn.functional.l1_loss(pred_cont, gt_cont)

            recent_losses.append(float(loss.item()))
            recent_accs.append(float(acc.item()))
            recent_l1s.append(float(l1.item()))
            del out, loss, action_logits, action_preds, action_gt, mask, pred_cont, gt_cont, l1, acc

            if micro_step % grad_accum != 0:
                continue

            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable_params,
                max_norm=float(train_cfg["max_grad_norm"]),
                error_if_nonfinite=False,
            )
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(
                    f"Non-finite gradient norm at optimizer step {optimizer_step + 1}: {grad_norm}"
                )

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_step += 1

            smooth_loss = sum(recent_losses) / len(recent_losses)
            smooth_acc = sum(recent_accs) / len(recent_accs)
            smooth_l1 = sum(recent_l1s) / len(recent_l1s)
            elapsed = time.time() - start_time
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3

            with metrics_path.open("a", newline="") as f:
                csv.writer(f).writerow([
                    optimizer_step,
                    smooth_loss,
                    smooth_acc,
                    smooth_l1,
                    float(grad_norm),
                    optimizer.param_groups[0]["lr"],
                    elapsed,
                    allocated,
                    reserved,
                ])

            if optimizer_step == 1 or optimizer_step % 10 == 0:
                print(
                    f"[{condition}] step={optimizer_step}/{max_steps} "
                    f"loss={smooth_loss:.4f} acc={smooth_acc:.4f} "
                    f"l1={smooth_l1:.4f} grad={float(grad_norm):.3f} "
                    f"gpu={allocated:.1f}/{reserved:.1f}GiB",
                    flush=True,
                )

            if optimizer_step == 1 or optimizer_step % 100 == 0:
                atomic_json(status_path, {
                    "state": "RUNNING",
                    "optimizer_step": optimizer_step,
                    "max_steps": max_steps,
                    "last_loss": smooth_loss,
                    "last_action_accuracy": smooth_acc,
                    "last_action_l1": smooth_l1,
                    "grad_norm": float(grad_norm),
                    "gpu_allocated_gib": allocated,
                    "gpu_reserved_gib": reserved,
                    "config": run_config,
                    "audit": audit,
                })

            if optimizer_step % save_steps == 0 or optimizer_step == max_steps:
                save_checkpoint(
                    model,
                    processor,
                    optimizer,
                    run_dir,
                    optimizer_step,
                    condition,
                    run_config,
                    audit,
                    {
                        "loss": smooth_loss,
                        "action_accuracy": smooth_acc,
                        "action_l1": smooth_l1,
                        "grad_norm": float(grad_norm),
                    },
                )

            if optimizer_step >= max_steps:
                break

        if optimizer_step != max_steps:
            raise RuntimeError(
                f"Training ended at optimizer step {optimizer_step}; expected {max_steps}"
            )

        final_adapter = run_dir / "final_adapter"
        if final_adapter.exists():
            import shutil
            shutil.rmtree(final_adapter)
        model.save_pretrained(
            final_adapter,
            safe_serialization=True,
            save_embedding_layers=False,
        )
        processor.save_pretrained(run_dir / "processor")

        completion = {
            "state": "PASS",
            "condition": condition,
            "optimizer_steps": optimizer_step,
            "micro_steps_this_process": micro_step,
            "final_adapter": str(final_adapter),
            "metrics_csv": str(metrics_path),
            "dataset_statistics": str(run_dir / "dataset_statistics.json"),
            "audit": audit,
            "config": run_config,
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        atomic_json(run_dir / "completion.json", completion)
        atomic_json(status_path, completion)
        if failure_path.exists():
            failure_path.unlink()

        print(
            f"TRAINING COMPLETE: {condition} {optimizer_step}/{max_steps} optimizer steps",
            flush=True,
        )

    except BaseException as exc:
        failure = {
            "state": "FAIL",
            "condition": condition,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "config": run_config,
        }
        atomic_json(failure_path, failure)
        atomic_json(status_path, failure)
        print("FAILURE JSON:", failure_path, flush=True)
        raise

    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
