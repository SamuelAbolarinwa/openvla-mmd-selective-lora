"""Check real tensor execution, not just nvidia-smi visibility."""
import argparse
import importlib.metadata as metadata
import json
from pathlib import Path
import torch
import tensorflow as tf
import tensorflow_datasets
import dlimp
from prismatic.vla.datasets import RLDSDataset
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError('PyTorch cannot execute on CUDA')
    devices = []
    for i in range(torch.cuda.device_count()):
        with torch.cuda.device(i):
            if not torch.cuda.is_bf16_supported(): raise RuntimeError(f'GPU {i} does not support BF16')
            x = torch.ones((16, 16), dtype=torch.bfloat16, device=f'cuda:{i}')
            y = x @ x
            torch.cuda.synchronize(i)
            assert torch.isfinite(y).all().item()
            prop = torch.cuda.get_device_properties(i)
            devices.append({'local_index': i, 'name': prop.name, 'total_memory_bytes': prop.total_memory})
    # TensorFlow is CPU-only so the RLDS pipeline cannot reserve training VRAM.
    assert not tf.config.list_physical_devices('GPU'), 'TensorFlow must be CPU-only'
    tf.debugging.assert_equal(tf.reduce_sum(tf.ones((2,2))).numpy(), 4)
    versions = {name: metadata.version(name) for name in
                ['torch','torchvision','transformers','peft','accelerate','tensorflow-cpu','tensorflow-datasets']}
    args.output.write_text(json.dumps({'status':'PASS', 'versions':versions, 'devices':devices}, indent=2)+'\n')
    print(args.output.read_text(), flush=True)

if __name__ == '__main__': main()
