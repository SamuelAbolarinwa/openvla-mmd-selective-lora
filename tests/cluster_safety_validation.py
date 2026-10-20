"""CPU-only regression coverage for the 20 October cluster fixes."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from host_resources import available_memory_bytes
from dataset_lock import dataset_statistics_lock
import qualify_merge
GIB = 1024**3


class SafetyChecks(unittest.TestCase):
    def memory_fixture(self, version=2, mount_root='/', membership='/slurm/job/step'):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        p = Path(tmp.name)
        mount = p/'mount'; mount.mkdir()
        leaf = mount/'slurm/job/step'; leaf.mkdir(parents=True)
        (p/'meminfo').write_text('MemAvailable: 104857600 kB\n')
        (p/'cgroup').write_text(('0::' if version == 2 else '5:memory:') + membership+'\n')
        fs = 'cgroup2 none rw' if version == 2 else 'cgroup memory rw,memory'
        (p/'mountinfo').write_text(f'23 1 0:22 {mount_root} {mount} rw - {fs}\n')
        return p, mount, leaf

    def measure(self, p):
        return available_memory_bytes(p/'meminfo', p/'unused', p/'cgroup', p/'mountinfo')

    def test_nested_v2_and_parent_budget(self):
        p, mount, leaf = self.memory_fixture()
        (mount/'memory.max').write_text('max'); (mount/'memory.current').write_text('0')
        (leaf/'memory.max').write_text(str(32*GIB)); (leaf/'memory.current').write_text(str(4*GIB))
        self.assertEqual(self.measure(p), 28*GIB)
        (leaf.parent/'memory.max').write_text(str(40*GIB))
        (leaf.parent/'memory.current').write_text(str(30*GIB))
        self.assertEqual(self.measure(p), 10*GIB)

    def test_nested_v1(self):
        p, mount, leaf = self.memory_fixture(version=1)
        (leaf/'memory.limit_in_bytes').write_text(str(24*GIB))
        (leaf/'memory.usage_in_bytes').write_text(str(6*GIB))
        self.assertEqual(self.measure(p), 18*GIB)

    def test_nonroot_mount(self):
        p, mount, leaf = self.memory_fixture(mount_root='/tenant', membership='/tenant/slurm/job/step')
        (leaf/'memory.max').write_text(str(32*GIB)); (leaf/'memory.current').write_text(str(4*GIB))
        self.assertEqual(self.measure(p), 28*GIB)

    def test_namespaced_mount_root(self):
        p, mount, leaf = self.memory_fixture(mount_root='/tenant/job', membership='/')
        (mount/'memory.max').write_text(str(8*GIB)); (mount/'memory.current').write_text(str(2*GIB))
        self.assertEqual(self.measure(p), 6*GIB)

    def test_unlimited_cgroup_uses_host_available(self):
        p, mount, leaf = self.memory_fixture()
        (leaf/'memory.max').write_text('max')
        self.assertEqual(self.measure(p), 100*GIB)

    def test_statistics_lock_blocks_another_process_then_releases(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            ready = root/'ready'; acquired = root/'acquired'
            code = ('import sys; from pathlib import Path; '
                    'sys.path.insert(0,sys.argv[1]); from dataset_lock import dataset_statistics_lock\n'
                    'Path(sys.argv[3]).touch()\n'
                    'with dataset_statistics_lock(sys.argv[2]): Path(sys.argv[4]).touch()\n')
            with dataset_statistics_lock(root):
                child = subprocess.Popen([sys.executable, '-c', code, str(ROOT/'scripts'), d, str(ready), str(acquired)], stdout=subprocess.DEVNULL)
                try:
                    import time
                    deadline = time.monotonic()+5
                    while not ready.exists() and time.monotonic()<deadline:
                        time.sleep(.01)
                    self.assertTrue(ready.exists())
                    self.assertFalse(acquired.exists())
                except BaseException:
                    child.kill(); child.wait(); raise
            child.wait(timeout=5)
            self.assertEqual(child.returncode,0)
            self.assertTrue(acquired.exists())

    def test_orchestrator_requires_a_real_resume_stage(self):
        import orchestrate_training as orch
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); repo=root/'repo'; (repo/'scripts').mkdir(parents=True)
            run=root/'run'; run.mkdir()
            (run/'config.json').write_text((ROOT/'config.json').read_text())
            (run/'assets.json').write_text('{"base_model_overlay":"unused"}')
            child = """import json,os,sys
from pathlib import Path
args=sys.argv
condition=args[args.index('--condition')+1]
root=Path(args[args.index('--output-root')+1])/condition
root.mkdir(parents=True,exist_ok=True)
checkpoint=root/'checkpoint'
if '--stop-after-step' in args:
 assert '--resume' not in args
 assert args[args.index('--stop-after-step')+1]=='1'
 checkpoint.write_text(str(os.getpid()))
else:
 assert '--resume' in args
 assert checkpoint.is_file()
 assert int(checkpoint.read_text())!=os.getpid()
 (root/'completion.json').write_text(json.dumps({'audit':{'non_llm_modules':[]},'resumed_from_step':1,'optimizer_steps':3}))
"""
            (repo/'scripts/train_condition.py').write_text(child)
            (repo/'scripts/verify_adapter.py').write_text('print("mock reload")')
            with patch.dict(os.environ,{'VLA_VISIBLE_GPU_COUNT':'1','CUDA_VISIBLE_DEVICES':'4','VLA_SEQUENTIAL':'1'}):
                orch.launch_pair(repo,run,'smoke')
            marker=json.loads((run/'SMOKE_COMPLETE.json').read_text())
            self.assertEqual(marker['qualification_version'],2)
            self.assertEqual(marker['status'],'PASS')

    def merge_fixture(self, root):
        (root/'assets.json').write_text(json.dumps({'base_model_overlay':'base','openvla_source':'source'}))
        (root/'SMOKE_COMPLETE.json').write_text(json.dumps({'output_root':str(root/'smoke'),'qualification_version':2}))
        (root/'source_manifest.json').write_text('{}'); (root/'config.json').write_text('{}')

    def test_merge_smoke_orders_reload_and_cleanup_and_reuses_marker(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); self.merge_fixture(root); calls=[]
            def run(cmd,check):
                self.assertTrue(check); calls.append(cmd)
                if '--output' in cmd:
                    Path(cmd[cmd.index('--output')+1]).mkdir(parents=True)
                else:
                    self.assertTrue(Path(cmd[cmd.index('--checkpoint')+1]).is_dir())
            with patch.object(sys,'argv',['qualify','--repo-dir',str(ROOT),'--run-dir',d]), patch.object(qualify_merge.subprocess,'run',side_effect=run):
                qualify_merge.main(); qualify_merge.main()
            self.assertEqual(len(calls),4)
            self.assertTrue((root/'MERGE_SMOKE_COMPLETE.json').exists())
            self.assertFalse((root/'qualification_merge/uniform').exists())
            self.assertFalse((root/'qualification_merge/mmd_selective').exists())

    def test_failed_merged_inference_retains_evidence_without_pass_marker(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); self.merge_fixture(root)
            def run(cmd,check):
                if '--output' in cmd: Path(cmd[cmd.index('--output')+1]).mkdir(parents=True)
                else: raise subprocess.CalledProcessError(1,cmd)
            with patch.object(sys,'argv',['qualify','--repo-dir',str(ROOT),'--run-dir',d]), patch.object(qualify_merge.subprocess,'run',side_effect=run):
                with self.assertRaises(subprocess.CalledProcessError): qualify_merge.main()
            self.assertFalse((root/'MERGE_SMOKE_COMPLETE.json').exists())
            self.assertTrue((root/'qualification_merge/uniform').exists())


if __name__ == '__main__': unittest.main(verbosity=2)
