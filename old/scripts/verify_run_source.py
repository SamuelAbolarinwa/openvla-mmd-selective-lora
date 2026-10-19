"""Freeze or verify run provenance using only the Python standard library."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def source_manifest(repo):
    files = sorted(list(repo.glob('*.sh')) + list((repo / 'scripts').glob('*.py'))
                   + list(repo.glob('requirements*.txt')) + [repo / 'constraints.txt'])
    return {str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def check_record(path, expected, freeze):
    if path.exists():
        if json.loads(path.read_text()) != expected:
            raise RuntimeError(f'{path.name} differs from this package; use the original code/settings or a new run directory')
    elif freeze:
        temporary = path.with_suffix(path.suffix + '.tmp')
        temporary.write_text(json.dumps(expected, indent=2) + '\n')
        os.replace(temporary, path)
    else:
        raise RuntimeError(f'Missing run provenance: {path}')


def verify_run(repo, run, freeze=False):
    check_record(run / 'config.json', json.loads((repo / 'config.json').read_text()), freeze)
    check_record(run / 'source_manifest.json', source_manifest(repo), freeze)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo-dir', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--freeze', action='store_true')
    args = parser.parse_args()
    verify_run(args.repo_dir.resolve(), args.run_dir.resolve(), args.freeze)
    print('RUN SOURCE AND CONFIGURATION: PASS')


if __name__ == '__main__':
    main()
