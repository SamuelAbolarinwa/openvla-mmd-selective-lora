"""Conservative additional-space budget per filesystem; no weights downloaded."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

GIB = 1024**3
# Actual published bytes for the pinned revisions, queried 19 October 2026.
MODEL_BYTES = 15082794851
DATASET_BYTES = 2817328769


def size(path):
    """Logical size, following snapshot file symlinks but not directory symlinks."""
    if not path.exists(): return 0
    if path.is_file(): return path.stat().st_size
    seen = set(); total = 0
    for directory, _, files in os.walk(path):
        for name in files:
            item = Path(directory)/name
            try:
                stat = item.stat()
            except FileNotFoundError:
                continue
            key = stat.st_dev, stat.st_ino
            if key not in seen:
                total += stat.st_size; seen.add(key)
    return total


def components(repo, run, cfg):
    cache = Path(os.environ.get('VLA_CACHE_DIR',repo/'cache')).resolve()
    data = Path(os.environ.get('VLA_DATA_ROOT',repo/'data/modified_libero_rlds')).resolve()
    # Per condition: FP32 adapter plus two FP32 Adam moments. While committing
    # a third save, the previous two must still fit. Final adapters/smoke extra.
    checkpoint_peak = cfg['expected_trainable_params'] * 12 * 3 * 2
    environment = repo/'.venv-cluster'
    # An incompatible environment is preserved rather than reused, so it cannot
    # count towards the space needed for the new installation.
    try:
        compatible = subprocess.run([str(environment/'bin/python'), '-c',
            'import sys; assert (3,10) <= sys.version_info[:2] <= (3,11); assert sys.prefix != sys.base_prefix'],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        compatible = False
    rows = [
        ('shared environment', repo, 16*GIB, environment if compatible else None),
        ('bootstrap and installation scratch', repo, 6*GIB, None),
        ('pinned base model', cache, MODEL_BYTES, cache/'huggingface/models--openvla--openvla-7b/snapshots'/cfg['model_revision']),
        ('processor and source metadata', cache, int(.25*GIB), None),
        ('LIBERO-Object RLDS', data, DATASET_BYTES, data/cfg['dataset_name']),
        ('two retained checkpoints plus atomic new saves', run, checkpoint_peak, None),
        ('final adapters and smoke outputs', run, cfg['expected_trainable_params']*4*8, run/'smoke'),
        ('two merged models', run, MODEL_BYTES*2, run/'merged'),
    ]
    return rows


def budget(rows, margin_gib=10, free_fn=None):
    groups = {}
    for label, root, expected, existing in rows:
        root.mkdir(parents=True, exist_ok=True)
        dev = root.stat().st_dev
        group = groups.setdefault(dev, {'path':str(root), 'additional_bytes':0, 'components':[]})
        credit = min(expected, size(existing)) if existing else 0
        extra = max(0, expected-credit)
        group['additional_bytes'] += extra
        group['components'].append({'name':label, 'planned_gib':round(expected/GIB,3),
                                    'credited_existing_gib':round(credit/GIB,3),
                                    'additional_gib':round(extra/GIB,3), 'path':str(root)})
    for group in groups.values():
        group['safety_margin_gib'] = margin_gib
        required = group['additional_bytes'] + margin_gib*GIB
        free = free_fn(group['path']) if free_fn else shutil.disk_usage(group['path']).free
        group['required_additional_gib'] = round(required/GIB,3)
        group['free_gib'] = round(free/GIB,3)
        group['fits_estimate'] = free >= required
    return list(groups.values())


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--repo-dir',type=Path,required=True)
    ap.add_argument('--run-dir',type=Path,required=True)
    args=ap.parse_args()
    repo=args.repo_dir.resolve(); run=args.run_dir.resolve()
    cfg=json.loads((run/'config.json').read_text())
    groups=budget(components(repo,run,cfg))
    report={'status':'PASS' if all(g['fits_estimate'] for g in groups) else 'FAIL',
            'filesystems':groups,
            'assumptions':['One shared environment; no pip wheel cache; two committed checkpoints per condition plus new-save headroom.',
                           'One experiment/run; smoke reruns reuse completed outputs.',
                           'Filesystem free-space check does not detect user quotas or space taken by other users after this check.']}
    (run/'logs').mkdir(parents=True,exist_ok=True)
    (run/'logs/storage_budget.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
    if report['status']!='PASS':
        raise RuntimeError('Storage budget does not fit. Choose a larger filesystem using VLA_RUN_DIR, VLA_CACHE_DIR or VLA_DATA_ROOT; see logs/storage_budget.json. Old environments/runs are never deleted automatically.')


if __name__=='__main__': main()
