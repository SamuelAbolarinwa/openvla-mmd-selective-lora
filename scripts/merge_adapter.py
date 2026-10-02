import argparse
import json
import shutil
import hashlib
import os
from pathlib import Path

import torch
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
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--openvla-source", type=Path, required=True)
    args = ap.parse_args()

    identity = {
        'base':str(args.base.resolve()),
        'adapter_sha256':hashlib.sha256((args.adapter/'adapter_model.safetensors').read_bytes()).hexdigest(),
        'adapter_config_sha256':hashlib.sha256((args.adapter/'adapter_config.json').read_bytes()).hexdigest(),
        'statistics_sha256':hashlib.sha256(args.stats.read_bytes()).hexdigest(),
    }
    marker=args.output/'MERGE_COMPLETE.json'
    if marker.exists():
        if json.loads(marker.read_text()) != identity: raise RuntimeError('Existing merge belongs to another adapter')
        print('Existing completed merge reused:',args.output); return
    output=args.output.with_name(args.output.name+'.partial')
    if output.exists(): shutil.rmtree(output)
    output.mkdir(parents=True)

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
    )
    model = PeftModel.from_pretrained(base, args.adapter, is_trainable=False)
    merged = model.merge_and_unload()
    merged.save_pretrained(output, safe_serialization=True)
    processor.save_pretrained(output)
    shutil.copy2(args.stats, output / "dataset_statistics.json")

    hf_src = args.openvla_source / "prismatic" / "extern" / "hf"
    for name in [
        "configuration_prismatic.py",
        "modeling_prismatic.py",
        "processing_prismatic.py",
    ]:
        shutil.copy2(hf_src / name, output / name)


    def replace_remote_refs(obj):
        if isinstance(obj, str):
            return obj.replace("openvla/openvla-7b--", "")
        if isinstance(obj, list):
            return [replace_remote_refs(x) for x in obj]
        if isinstance(obj, dict):
            return {k: replace_remote_refs(v) for k, v in obj.items()}
        return obj

    for jp in output.glob("*.json"):
        try:
            obj = json.loads(jp.read_text())
        except Exception:
            continue
        jp.write_text(json.dumps(replace_remote_refs(obj), indent=2))

    (output/'MERGE_COMPLETE.json').write_text(json.dumps(identity,indent=2)+'\n')
    if args.output.exists(): raise RuntimeError('Refusing to overwrite an unmarked output directory')
    os.replace(output,args.output)
    print("MERGE PASS:", args.output)


if __name__ == "__main__":
    main()
