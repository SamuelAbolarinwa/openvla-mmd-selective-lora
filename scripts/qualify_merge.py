"""Exercise the production CPU merge and merged-model simulator path before training."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo-dir', type=Path, required=True)
    ap.add_argument('--run-dir', type=Path, required=True)
    args = ap.parse_args()
    repo, run = args.repo_dir.resolve(), args.run_dir.resolve()
    assets = json.loads((run/'assets.json').read_text())
    smoke = json.loads((run/'SMOKE_COMPLETE.json').read_text())
    identity = {'smoke': smoke, 'source': json.loads((run/'source_manifest.json').read_text()),
                'config': json.loads((run/'config.json').read_text())}
    marker = run/'MERGE_SMOKE_COMPLETE.json'
    if marker.exists():
        if json.loads(marker.read_text()) != identity:
            raise RuntimeError('Merge qualification belongs to different smoke/code/configuration')
        print('Previously completed merged-model smoke reused', flush=True)
        return
    for condition in ('uniform', 'mmd_selective'):
        source = Path(smoke['output_root'])/condition
        output = run/'qualification_merge'/condition
        subprocess.run([sys.executable, '-u', str(repo/'scripts/merge_adapter.py'),
            '--base', assets['base_model_overlay'], '--adapter', str(source/'final_adapter'),
            '--stats', str(source/'dataset_statistics.json'), '--output', str(output),
            '--openvla-source', assets['openvla_source']], check=True)
        subprocess.run([sys.executable, '-u', str(repo/'scripts/evaluate_libero.py'),
            '--mode', 'combined_smoke', '--checkpoint', str(output)], check=True)
        # Only these dedicated qualification outputs are disposable. Keep failures for diagnosis.
        shutil.rmtree(output)
    temporary = marker.with_suffix('.tmp')
    temporary.write_text(json.dumps(identity, indent=2)+'\n')
    os.replace(temporary, marker)
    print('CPU MERGE + FRESH MERGED-MODEL SIMULATOR CHECK: PASS', flush=True)


if __name__ == '__main__':
    main()
