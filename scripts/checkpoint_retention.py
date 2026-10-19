"""Bound disk use without deleting a checkpoint before its successor commits."""
import json
from pathlib import Path
import shutil


def prune_checkpoints(run_dir, keep=2):
    if keep < 2:
        raise ValueError('Retain at least two committed checkpoints')
    run_dir = Path(run_dir)
    pointer = json.loads((run_dir / 'latest_checkpoint.json').read_text())
    latest = (run_dir / pointer['state_file']).resolve().parent
    parent = (run_dir / 'checkpoints').resolve()
    if latest.parent != parent or not (latest / 'training_state.pt').is_file():
        raise RuntimeError('Invalid latest checkpoint; pruning refused')
    candidates = []
    for path in parent.glob('step-*'):
        if path.is_symlink() or not path.is_dir():
            continue
        if not (path/'checkpoint.json').is_file() or not (path/'training_state.pt').is_file():
            continue
        step = json.loads((path/'checkpoint.json').read_text())['optimizer_step']
        candidates.append((int(step), path.name, path))
    ordered = [row[2] for row in sorted(candidates, reverse=True)]
    protected = {latest, *ordered[:keep]}
    for path in ordered:
        if path not in protected:
            shutil.rmtree(path)

