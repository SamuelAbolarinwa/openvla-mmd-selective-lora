"""Optional CPU regression checks for the PyTorch upgrade, without model downloads."""
import ast
from pathlib import Path
import tempfile
import types

import numpy as np
import torch
import torchvision
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import LlamaConfig, LlamaForCausalLM

ROOT = Path(__file__).resolve().parents[1]
assert torch.__version__.split('+')[0] == '2.7.1'
assert torchvision.__version__.split('+')[0] == '0.22.1'
torch.manual_seed(7)
config = LlamaConfig(vocab_size=32, hidden_size=32, intermediate_size=64,
                     num_hidden_layers=1, num_attention_heads=4,
                     num_key_value_heads=4, max_position_embeddings=32)
base = LlamaForCausalLM(config)
initial = {key: value.clone() for key, value in base.state_dict().items()}
model = get_peft_model(base, LoraConfig(r=4, lora_alpha=2,
                       target_modules=['q_proj', 'v_proj'], lora_dropout=0.0))
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
tokens = torch.randint(0, 32, (2, 8))
optimizer = torch.optim.AdamW(p for p in model.parameters() if p.requires_grad)
model.train()
loss = model(input_ids=tokens, labels=tokens).loss
loss.backward()
assert torch.isfinite(loss)
assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters() if p.requires_grad)
optimizer.step()
model.eval()
with torch.no_grad(): expected = model(input_ids=tokens).logits
with tempfile.TemporaryDirectory() as tmp:
    folder = Path(tmp)
    model.save_pretrained(folder / 'adapter')
    fresh = LlamaForCausalLM(config); fresh.load_state_dict(initial)
    reloaded = PeftModel.from_pretrained(fresh, folder / 'adapter'); reloaded.eval()
    with torch.no_grad(): actual = reloaded(input_ids=tokens).logits
    torch.testing.assert_close(actual, expected)
    torch.save({'optimizer': optimizer.state_dict()}, folder / 'resume.pt')
    assert 'optimizer' in torch.load(folder / 'resume.pt', weights_only=False)
    states_file = folder / 'task.pruned_init'
    torch.save([np.arange(8, dtype=np.float64)], states_file)
    try:
        torch.load(states_file)
    except Exception as error:
        assert 'Weights only load failed' in str(error)
    else:
        raise AssertionError('Legacy NumPy states unexpectedly accepted by default loader')
    tree = ast.parse((ROOT / 'scripts/evaluate_libero.py').read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'load_init_states')
    namespace = {'torch': torch, 'Path': Path, 'get_libero_path': lambda kind: folder}
    exec(compile(ast.Module(body=[function], type_ignores=[]), '<state loader>', 'exec'), namespace)
    suite = types.SimpleNamespace(get_task=lambda i: types.SimpleNamespace(
        problem_folder='', init_states_file=states_file.name))
    states = namespace['load_init_states'](suite, 0)
    np.testing.assert_array_equal(states[0], np.arange(8, dtype=np.float64))
print(f'PASS: torch {torch.__version__}, torchvision {torchvision.__version__}')
print('PASS: tiny Llama LoRA backward, non-reentrant checkpointing, adapter save/fresh reload, optimizer load')
print('PASS: legacy NumPy simulator states load through the updated evaluation helper')
print('CPU regression only: no OpenVLA weights, dataset, simulator or CUDA execution')
