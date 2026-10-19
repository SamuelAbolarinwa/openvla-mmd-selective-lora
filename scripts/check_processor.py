"""Check pinned processing assets without loading model weights or allocating a GPU."""
import argparse
import json
from pathlib import Path

def load_processor(path):
    from transformers import AutoTokenizer
    from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
    # Explicit classes avoid mixing dynamic-module and locally registered config types.
    image_processor = PrismaticImageProcessor.from_pretrained(path, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    return PrismaticProcessor(image_processor=image_processor, tokenizer=tokenizer)

def main():
    import torch
    from PIL import Image
    ap = argparse.ArgumentParser()
    ap.add_argument('--assets', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    assets = json.loads(Path(args.assets).read_text())
    path = assets['base_model_overlay']
    processor = load_processor(path)
    config = json.loads((Path(path) / 'config.json').read_text())
    batch = processor(text='In: What action should the robot take to pick up the object?\nOut:',
                      images=Image.new('RGB', (256, 256)), return_tensors='pt')
    pixels = batch['pixel_values']
    assert tuple(pixels.shape) == (1, 6, 224, 224), pixels.shape
    assert torch.isfinite(pixels).all()
    assert batch['input_ids'].ndim == 2 and batch['input_ids'].shape[0] == 1
    assert batch['input_ids'].numel() > 0
    assert int(batch['input_ids'].max()) < config['text_config']['vocab_size']
    assert processor.tokenizer.pad_token_id == config['pad_token_id']
    assert len(processor.tokenizer) <= config['text_config']['vocab_size']
    report = {'status': 'PASS', 'processor_revision': assets['processor_revision'],
              'model_revision': assets['model_revision'], 'processor': type(processor).__name__,
              'tokenizer': type(processor.tokenizer).__name__, 'vocab_size': len(processor.tokenizer),
              'pixel_shape': list(pixels.shape), 'token_shape': list(batch['input_ids'].shape),
              'scope': 'Processing only; model weights and training are not tested here.'}
    Path(args.output).write_text(json.dumps(report, indent=2))
    print('PROCESSOR CHECK: PASS', json.dumps(report), flush=True)

if __name__ == '__main__':
    main()
