"""Regression checks for cluster setup/storage fixes, without GPUs or downloads."""
import ast
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from check_storage import GIB, components, budget
from checkpoint_retention import prune_checkpoints
from host_resources import available_memory_bytes


class ClusterFixes(unittest.TestCase):
    def test_cgroup_limit_takes_priority_over_host_free_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)
            (path/'meminfo').write_text('MemAvailable: 104857600 kB\n')
            (path/'memory.max').write_text(str(32*GIB))
            (path/'memory.current').write_text(str(4*GIB))
            self.assertEqual(available_memory_bytes(path/'meminfo',path,path/'no-proc',path/'no-mountinfo'),28*GIB)

    def test_clean_108_decimal_gb_fits_budget_but_70_gib_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo=Path(tmp)/'repo'; repo.mkdir(); run=repo/'runs/test'; run.mkdir(parents=True)
            cfg=json.loads((ROOT/'config.json').read_text())
            with patch.dict(os.environ,{},clear=True):
                rows=components(repo,run,cfg)
            groups=budget(rows,free_fn=lambda _:108_000_000_000)
            self.assertEqual(len(groups),1)
            self.assertTrue(groups[0]['fits_estimate'],groups)
            self.assertGreater(groups[0]['required_additional_gib'],80)
            self.assertFalse(budget(rows,free_fn=lambda _:70*GIB)[0]['fits_estimate'])

    def test_existing_files_credit_is_capped_and_margin_remains(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); file=root/'existing'; file.write_bytes(b'x'*200)
            result=budget([('part',root,100,file)],free_fn=lambda _:11*GIB)[0]
            self.assertEqual(result['additional_bytes'],0)
            self.assertEqual(result['required_additional_gib'],10)

    def test_retention_keeps_latest_previous_and_incomplete_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp); parent=run/'checkpoints'; parent.mkdir()
            for step in [5,10,15]:
                path=parent/f'step-{step:06d}-1'; path.mkdir()
                (path/'training_state.pt').write_bytes(b'state')
                (path/'checkpoint.json').write_text(json.dumps({'optimizer_step':step}))
            (parent/'.step-000020-1').mkdir()
            (parent/'step-000025-incomplete').mkdir()
            (run/'latest_checkpoint.json').write_text(json.dumps({'state_file':'checkpoints/step-000015-1/training_state.pt'}))
            prune_checkpoints(run)
            self.assertFalse((parent/'step-000005-1').exists())
            self.assertTrue((parent/'step-000010-1').is_dir())
            self.assertTrue((parent/'step-000015-1').is_dir())
            self.assertTrue((parent/'.step-000020-1').is_dir())
            self.assertTrue((parent/'step-000025-incomplete').is_dir())

    def test_foreign_latest_pointer_refuses_pruning(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp); (run/'other').mkdir(); (run/'other/state.pt').write_bytes(b'x')
            (run/'latest_checkpoint.json').write_text(json.dumps({'state_file':'other/state.pt'}))
            with self.assertRaises(RuntimeError): prune_checkpoints(run)

    def test_wrong_project_python_is_preserved_and_rebuilt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); venv=root/'.venv-cluster'; (venv/'bin').mkdir(parents=True)
            py=venv/'bin/python'; py.write_text('#!/bin/bash\nexit 9\n');py.chmod(0o755)
            script='source "$1"; REPO_DIR="$2"; VLA_PYTHON="$3"; ensure_project_venv; "$VENV/bin/python" -c "import sys; assert sys.prefix != sys.base_prefix"'
            result=subprocess.run(['bash','-c',script,'audit',str(ROOT/'common.sh'),str(root),sys.executable],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            backups=list(root.glob('.venv-cluster.incompatible.*'))
            self.assertEqual(len(backups),1)
            self.assertEqual((backups[0]/'bin/python').read_text(),'#!/bin/bash\nexit 9\n')

    def test_existing_operator_environment_is_not_used(self):
        for path in ROOT.glob('*.sh'):
            source=path.read_text()
            self.assertNotIn('source "$REPO_DIR/.venv/bin/activate"',source)
            self.assertNotIn('source "$REPO_DIR/.venv-eval/bin/activate"',source)
        self.assertIn('PIP_NO_CACHE_DIR=1',(ROOT/'common.sh').read_text())
        self.assertIn('CUDA_VISIBLE_DEVICES',(ROOT/'preflight.sh').read_text())

    def test_asset_overlay_fills_missing_processor_without_replacing_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);repo=base/'repo';repo.mkdir()
            cfg=json.loads((ROOT/'config.json').read_text());(repo/'config.json').write_text(json.dumps(cfg))
            weights=base/'weights';weights.mkdir();processing=base/'processing';processing.mkdir()
            (weights/'config.json').write_text('{"original_weights":true}')
            (weights/'model.safetensors.index.json').write_text('{"weight_map":{"x":"model.safetensors"}}')
            (weights/'model.safetensors').write_bytes(b'original')
            names=['preprocessor_config.json','processor_config.json','tokenizer.json','tokenizer.model',
                   'tokenizer_config.json','special_tokens_map.json','added_tokens.json']
            for name in names:(processing/name).write_text('{}')
            hf=repo/'third_party/openvla/prismatic/extern/hf';hf.mkdir(parents=True)
            for name in ['configuration_prismatic.py','modeling_prismatic.py','processing_prismatic.py']:(hf/name).write_text('# pinned')
            for name in ['bddl_files','init_files','assets']:(repo/'third_party/LIBERO/libero/libero'/name).mkdir(parents=True)
            calls=[]
            def download(**kw):
                calls.append(kw)
                if kw.get('repo_type')=='dataset':
                    (Path(kw['local_dir'])/cfg['dataset_name']).mkdir(parents=True)
                    return kw['local_dir']
                return str(processing if 'allow_patterns' in kw else weights)
            hub=types.ModuleType('huggingface_hub');hub.snapshot_download=download
            yaml=types.ModuleType('yaml');yaml.safe_dump=json.dumps
            spec=importlib.util.spec_from_file_location('test_prepare',ROOT/'scripts/prepare_assets.py')
            module=importlib.util.module_from_spec(spec)
            with patch.dict(sys.modules,{'huggingface_hub':hub,'yaml':yaml}),patch.dict(os.environ,{},clear=True),patch.object(sys,'argv',['prepare','--repo-dir',str(repo),'--run-dir',str(repo)]):
                spec.loader.exec_module(module);module.main()
            overlay=repo/'base_model_overlay'
            self.assertTrue((overlay/'model.safetensors').is_symlink())
            self.assertEqual((overlay/'model.safetensors').read_bytes(),b'original')
            self.assertEqual(json.loads((overlay/'config.json').read_text()),{'original_weights':True})
            self.assertTrue(all((overlay/name).is_file() for name in names))
            self.assertEqual(calls[1]['revision'],cfg['processor_revision'])

    def test_all_runtime_model_loaders_use_explicit_source_classes(self):
        for name in ['train_condition.py','verify_adapter.py','merge_adapter.py','evaluate_libero.py']:
            text=(ROOT/'scripts'/name).read_text()
            self.assertNotIn('AutoProcessor.from_pretrained(',text)
            self.assertNotIn('AutoModelForVision2Seq.from_pretrained(',text)
            self.assertIn('load_base(',text)
            self.assertIn('load_processor(',text)


if __name__=='__main__':unittest.main(verbosity=2)
