import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--stats", type=Path, required=True)
    ap.add_argument("--condition", required=True)
    args = ap.parse_args()

    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    processor = AutoProcessor.from_pretrained(
        args.base, trust_remote_code=True, local_files_only=True
    )
    base = AutoModelForVision2Seq.from_pretrained(
        args.base,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        local_files_only=True,
        attn_implementation="sdpa",
    ).to("cuda:0")
    model = PeftModel.from_pretrained(base, args.adapter, is_trainable=False)
    model.eval()

    stats = json.loads(args.stats.read_text())
    key = (
        "libero_object_no_noops"
        if "libero_object_no_noops" in stats
        else "libero_object"
        if "libero_object" in stats
        else None
    )
    if key is None:
        raise KeyError(f"Unexpected dataset-stat keys: {list(stats)}")
    core = model.base_model.model
    core.norm_stats = stats

    img = np.zeros((224, 224, 3), dtype=np.uint8)
    img[:, :112, 0] = 255
    img[:, 112:, 2] = 255
    prompt = (
        "In: What action should the robot take to "
        "pick up the alphabet soup and place it in the basket?\nOut:"
    )
    inputs = processor(prompt, Image.fromarray(img).convert("RGB"))
    moved = {}
    for k, v in inputs.items():
        if torch.is_tensor(v):
            moved[k] = (
                v.to("cuda:0", dtype=torch.bfloat16)
                if torch.is_floating_point(v)
                else v.to("cuda:0")
            )
        else:
            moved[k] = v

    with torch.inference_mode():
        action = core.predict_action(
            **moved,
            unnorm_key=key,
            do_sample=False,
        )

    action = np.asarray(action, dtype=np.float64)
    if action.shape != (7,):
        raise RuntimeError(f"Expected 7-D action, got {action.shape}")
    if not np.all(np.isfinite(action)):
        raise RuntimeError(f"Non-finite action: {action}")

    print(f"FRESH ADAPTER RELOAD PASS: {args.condition}")
    print("Action:", action.tolist())


if __name__ == "__main__":
    main()
