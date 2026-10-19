"""Offline launcher/checkpoint contract tests. These do not test OpenVLA training."""
import ast
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('orchestrator', ROOT/'scripts/orchestrate_training.py')
orch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(orch)

class Contracts(unittest.TestCase):
    def test_python_bootstrap_when_system_venv_is_broken(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bin_dir = root / 'bin'; bin_dir.mkdir()
            broken = bin_dir / 'broken-python'
            broken.write_text('#!/bin/bash\nif [[ "$1" == -c ]]; then exec "$AUDIT_REAL_PYTHON" -c "import venv, ensurepip"; fi\nexit 9\n')
            broken.chmod(0o755)
            for name in ['python3.11', 'python3.10', 'python3']:
                (bin_dir / name).symlink_to(broken)
            # Simulate the bootstrap's compatible interpreter while exercising
            # real venv creation, including when tests run under Python 3.12+.
            usable = root / 'usable-python'
            usable.write_text('#!/bin/bash\nif [[ "$1" == -c ]]; then exec "$AUDIT_REAL_PYTHON" -c "import venv, ensurepip"; fi\nexec "$AUDIT_REAL_PYTHON" "$@"\n')
            usable.chmod(0o755)
            repo = root / 'repo'
            uv = repo / 'tools/uv-x86_64-unknown-linux-gnu/uv'
            uv.parent.mkdir(parents=True)
            uv.write_text('#!/bin/bash\nprintf "%s\\n" "$*" >> "$AUDIT_UV_CALLS"\nif [[ "$2" == find ]]; then printf "%s\\n" "$AUDIT_BOOTSTRAP_PYTHON"; fi\n')
            uv.chmod(0o755)
            env = os.environ.copy(); env.pop('PYTHON_BIN', None)
            env.update(PATH=str(bin_dir) + ':' + env['PATH'],
                       AUDIT_REAL_PYTHON=str(Path(sys.executable).resolve()),
                       AUDIT_UV_CALLS=str(root / 'uv_calls'),
                       AUDIT_BOOTSTRAP_PYTHON=str(usable))
            result = subprocess.run(['bash', '-c',
                'source "$1"; REPO_DIR="$2"; choose_python; [[ "$VLA_PYTHON" == "$AUDIT_BOOTSTRAP_PYTHON" ]]',
                'audit', str(ROOT / 'common.sh'), str(repo)], env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('python install 3.11.13', (root / 'uv_calls').read_text())
            self.assertIn('python find --managed-python 3.11.13', (root / 'uv_calls').read_text())

    def test_explicit_broken_python_has_actionable_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            broken = Path(tmp) / 'python'
            broken.write_text('#!/bin/bash\nexit 1\n'); broken.chmod(0o755)
            env = dict(os.environ, PYTHON_BIN=str(broken))
            result = subprocess.run(['bash', '-c', 'source "$1"; choose_python',
                                     'audit', str(ROOT / 'common.sh')], env=env,
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 30)
            self.assertIn('Unset it to allow automatic selection', result.stderr)

    def test_run_provenance_rejects_changed_or_missing_records(self):
        spec = importlib.util.spec_from_file_location('provenance', ROOT / 'scripts/verify_run_source.py')
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / 'repo'; shutil.copytree(ROOT, repo, ignore=shutil.ignore_patterns('.git', '__pycache__'))
            run = Path(tmp) / 'run'; run.mkdir()
            with self.assertRaises(RuntimeError): module.verify_run(repo, run)
            module.verify_run(repo, run, freeze=True)
            module.verify_run(repo, run)
            evaluator = repo / 'scripts/evaluate_libero.py'
            original = evaluator.read_text(); evaluator.write_text(original + '\n# changed evaluator\n')
            with self.assertRaises(RuntimeError): module.verify_run(repo, run)
            evaluator.write_text(original)
            config = repo / 'config.json'; cfg = json.loads(config.read_text()); cfg['seed'] += 1
            config.write_text(json.dumps(cfg))
            with self.assertRaises(RuntimeError): module.verify_run(repo, run, freeze=True)

    def test_evaluation_resume_rejects_changed_evaluator(self):
        import hashlib
        source = ROOT / 'scripts/evaluate_libero.py'
        tree = ast.parse(source.read_text())
        funcs = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in {'evaluation_identity', 'load_progress'}]
        ns = {'Path': Path, 'json': json, 'hashlib': hashlib, '__file__': str(source)}
        exec(compile(ast.Module(body=funcs, type_ignores=[]), '<evaluation identity>', 'exec'), ns)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); checkpoint = root / 'checkpoint'; checkpoint.mkdir()
            (checkpoint / 'config.json').write_text('{}')
            (checkpoint / 'MERGE_COMPLETE.json').write_text('{}')
            identity = ns['evaluation_identity'](checkpoint, 15, 7, 0, 280, 10)
            progress = root / 'progress.json'
            record = {'identity': identity, 'episodes': {'0:0': {'success': True}}}
            progress.write_text(json.dumps(record))
            self.assertEqual(ns['load_progress'](progress, identity), record)
            changed = root / 'evaluate.py'; changed.write_text(source.read_text() + '\n# changed evaluator\n')
            ns['__file__'] = str(changed)
            new_identity = ns['evaluation_identity'](checkpoint, 15, 7, 0, 280, 10)
            with self.assertRaises(RuntimeError): ns['load_progress'](progress, new_identity)

    def test_scientific_configuration(self):
        cfg=json.loads((ROOT/'config.json').read_text())
        self.assertEqual(cfg['mmd_selected_layers'], list(range(9,24))+[25])
        self.assertEqual(cfg['expected_trainable_params'],110828288)
        self.assertEqual(cfg['training']['batch_size']*cfg['training']['grad_accumulation_steps'],128)
        self.assertEqual(cfg['training']['smoke_shuffle_buffer_size'],cfg['training']['shuffle_buffer_size'])
        self.assertEqual(cfg['training']['max_steps'],50000)

    def exercise_orchestrator(self, allocation, count, sequential=False, failure=False):
        with tempfile.TemporaryDirectory(prefix='vla contracts ') as tmp:
            repo=Path(tmp)/'repo'; (repo/'scripts').mkdir(parents=True)
            run=Path(tmp)/'run'; run.mkdir()
            shutil.copy2(ROOT/'config.json',run/'config.json')
            (run/'assets.json').write_text(json.dumps({'base_model_overlay':'unused'}))
            record=run/'calls.jsonl'
            fake = '''import json,os,sys,time
from pathlib import Path
condition=sys.argv[sys.argv.index('--condition')+1]
kind=Path(__file__).stem
with open(os.environ['TEST_RECORD'],'a') as f:
 f.write(json.dumps({'condition':condition,'kind':kind,'gpu':os.environ['CUDA_VISIBLE_DEVICES']})+'\\n')
if kind=='train_condition' and os.environ.get('TEST_FAIL')=='1':
 if condition=='uniform': raise SystemExit(7)
 time.sleep(8)
if kind=='train_condition':
 root=Path(sys.argv[sys.argv.index('--output-root')+1])/condition
 root.mkdir(parents=True,exist_ok=True)
 (root/'completion.json').write_text(json.dumps({'audit':{'non_llm_modules':['same_module']}}))
print('mock child complete')
'''
            for name in ['train_condition','verify_adapter']:
                (repo/'scripts'/f'{name}.py').write_text(fake)
            env={'VLA_VISIBLE_GPU_COUNT':str(count),'VLA_SEQUENTIAL':str(int(sequential)),
                 'TEST_RECORD':str(record),'TEST_FAIL':str(int(failure))}
            if allocation is not None: env['CUDA_VISIBLE_DEVICES']=allocation
            with patch.dict(os.environ,env):
                if allocation is None: os.environ.pop('CUDA_VISIBLE_DEVICES',None)
                start=time.monotonic()
                with contextlib.redirect_stdout(io.StringIO()):
                    if failure:
                        with self.assertRaises(RuntimeError): orch.launch_pair(repo,run,'smoke')
                        self.assertLess(time.monotonic()-start,5, 'Other child was not stopped promptly')
                    else: orch.launch_pair(repo,run,'smoke')
            return [json.loads(x) for x in record.read_text().splitlines()]

    def test_numeric_gpu_allocation(self):
        rows=self.exercise_orchestrator('4,7',2)
        training={x['condition']:x['gpu'] for x in rows if x['kind']=='train_condition'}
        self.assertEqual(training,{'uniform':'4','mmd_selective':'7'})
        self.assertEqual({x['gpu'] for x in rows if x['kind']=='verify_adapter'},{'4'})

    def test_uuid_gpu_allocation(self):
        rows=self.exercise_orchestrator('GPU-alpha,GPU-beta',2)
        self.assertEqual({x['gpu'] for x in rows if x['kind']=='train_condition'},{'GPU-alpha','GPU-beta'})

    def test_single_gpu_sequential(self):
        rows=self.exercise_orchestrator('5',1)
        self.assertEqual({x['gpu'] for x in rows},{'5'})
        self.assertEqual([x['condition'] for x in rows[:2]],['uniform','mmd_selective'])

    def test_force_sequential(self):
        rows=self.exercise_orchestrator('4,7',2,True)
        self.assertEqual({x['gpu'] for x in rows},{'4'})

    def test_stop_other_condition_on_failure(self):
        self.exercise_orchestrator('4,7',2,failure=True)

    def test_checkpoint_commit_survives_failed_next_save(self):
        tree=ast.parse((ROOT/'scripts/train_condition.py').read_text())
        funcs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in {'atomic_json','save_checkpoint'}]
        def save(state,path): Path(path).write_text('mock optimiser state')
        torch=types.SimpleNamespace(save=save,get_rng_state=lambda:'rng',
                                   cuda=types.SimpleNamespace(get_rng_state_all=lambda:['rng']))
        ns={'Path':Path,'json':json,'os':os,'shutil':shutil,'time':time,'random':random,
            'torch':torch,'np':types.SimpleNamespace(random=types.SimpleNamespace(get_state=lambda:'rng'))}
        exec(compile(ast.Module(body=funcs,type_ignores=[]),'<checkpoint functions>','exec'),ns)
        class Model:
            def save_pretrained(self,path,**kwargs):
                Path(path).mkdir(parents=True,exist_ok=True)
                (Path(path)/'adapter_model.safetensors').write_text('mock adapter')
        processor=types.SimpleNamespace(save_pretrained=lambda path:None)
        optim=types.SimpleNamespace(state_dict=lambda:{})
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp)
            with contextlib.redirect_stdout(io.StringIO()):
                ns['save_checkpoint'](Model(),processor,optim,run,5000,'uniform',{}, {}, {})
            pointer=(run/'latest_checkpoint.json').read_text()
            committed=json.loads(pointer)
            self.assertTrue((run/committed['state_file']).is_file())
            self.assertTrue((run/committed['adapter_dir']/'adapter_model.safetensors').is_file())
            def fail(*args): raise OSError('simulated disk-write interruption')
            torch.save=fail
            with self.assertRaises(OSError):
                ns['save_checkpoint'](Model(),processor,optim,run,10000,'uniform',{}, {}, {})
            self.assertEqual((run/'latest_checkpoint.json').read_text(),pointer)

    def test_missing_gpu_keeps_preflight_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp)/'run'
            result=subprocess.run(['bash',str(ROOT/'preflight.sh'),str(run)],capture_output=True,text=True)
            # On GPU hosts this may fail the storage reserve instead; either
            # way diagnostic logging must exist before any large installation.
            self.assertTrue((run/'logs/preflight.txt').is_file())

    def test_log_bundle_excludes_weights(self):
        import tarfile
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp); (run/'logs').mkdir()
            (run/'logs/preflight.txt').write_text('diagnostic')
            (run/'failure.json').write_text('{}')
            (run/'weights.pt').write_text('not diagnostics')
            subprocess.run(['bash',str(ROOT/'collect_logs.sh'),str(run)],check=True,capture_output=True)
            with tarfile.open(run/'diagnostics.tar.gz') as archive:
                names=archive.getnames()
                self.assertIn('./logs/preflight.txt',names)
                self.assertIn('./failure.json',names)
                self.assertNotIn('./weights.pt',names)

if __name__=='__main__': unittest.main(verbosity=2)
