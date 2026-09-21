"""Geometry-only provenance, matched physical target and lifecycle; no physics."""
import copy
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import run_geometry_four_body_growth as runner
from analyze_mobile_posterior_pilot import (FOUR_BODY_SCHEMA, GEOMETRY_FOUR_BODY_SCHEMA,
    campaign_body_count, validate_campaign_jobs)


class GeometryFourBody(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.example=runner.read(runner.positive.EXAMPLE)
        self.starts=runner.read(runner.positive.DESIGN/'recommended-poses.json')
        self.model_sha=runner.read(runner.PACKAGE/'manifest.json')['model_sha256']

    def make(self,start='triangle_free',replicate=0,index=0):
        return runner.configured(self.example,self.starts,self.root/'geometry',start,replicate,index,self.model_sha)

    def test_same_physics_and_starts_as_positive_controls_with_fresh_streams(self):
        original=copy.deepcopy((self.example,self.starts))
        seeds=[]
        for i,(start,replicate)in enumerate((s,r)for s in runner.positive.STARTS for r in (0,1)):
            cfg=self.make(start,replicate,i)
            old=runner.positive.configured(self.example,self.starts,self.root/'geometry','original',start,replicate,i)
            omit={'seed','metadata'}
            self.assertEqual({k:v for k,v in cfg.items()if k not in omit},{k:v for k,v in old.items()if k not in omit})
            self.assertFalse(cfg['metadata']['native_informed_proposal'])
            self.assertTrue(cfg['metadata']['native_initial_scaffold'])
            self.assertEqual(cfg['metadata']['model_sha256'],self.model_sha)
            self.assertEqual(cfg['fixed_body_indices'],[]);self.assertEqual(cfg['seed_labels'],[])
            self.assertEqual(cfg['initial_poses'][:3],self.starts['triangle_plus_free'][:3])
            seeds.append(cfg['seed'])
        self.assertEqual(seeds,[129101010,129102019,129103028,129104037])
        self.assertTrue(set(seeds).isdisjoint({127101010+1009*i for i in range(8)}))
        self.assertEqual((self.example,self.starts),original)

    def test_geometry_observer_grid_requires_truthful_provenance_and_keeps_old_variants(self):
        jobs=[dict(id=f'{s}-{r}',start=s,replicate=r,mode='c09')for s in runner.positive.STARTS for r in (0,1)]
        manifest=dict(schema=GEOMETRY_FOUR_BODY_SCHEMA,atlas_variant='geometry',body_count=4,tracked_body_index=3,
            scaffold_body_indices=[0,1,2],native_informed_proposal=False,native_initial_scaffold=True,jobs=jobs)
        status=dict(complete=True,running=False,jobs=[dict(id=j['id'],status='complete',exit_code=0)for j in jobs])
        self.assertEqual(validate_campaign_jobs(manifest,status),jobs)
        self.assertEqual(campaign_body_count(manifest),4)
        for key,value in [('native_informed_proposal',True),('native_informed_proposal',0),('native_initial_scaffold',False),
                          ('native_initial_scaffold',1),('atlas_variant','coverage'),('body_count',3)]:
            bad=copy.deepcopy(manifest);bad[key]=value
            with self.subTest(key=key,value=value),self.assertRaises(AssertionError):validate_campaign_jobs(bad,status)
        for key in ('native_informed_proposal','native_initial_scaffold'):
            bad=copy.deepcopy(manifest);del bad[key]
            with self.assertRaises(KeyError):validate_campaign_jobs(bad,status)
        bad=copy.deepcopy(manifest);bad['schema']=FOUR_BODY_SCHEMA
        with self.assertRaises(AssertionError):validate_campaign_jobs(bad,status)
        for variant in ('original','coverage'):
            bad['atlas_variant']=variant
            self.assertEqual(validate_campaign_jobs(bad,status),jobs)
        bad=copy.deepcopy(manifest);bad['jobs'][0]['mode']='c0'
        with self.assertRaises(AssertionError):validate_campaign_jobs(bad,status)

    def test_exact_package_preserves_all96_geometry_components_and_generator(self):
        package=runner.package_check(runner.PACKAGE)
        model=runner.read(runner.PACKAGE/'model.json')
        self.assertEqual(model['base_model'],runner.read(runner.PACKAGE/'provenance/source-model.json'))
        self.assertEqual(model['reciprocal_components'],[True]*96)
        self.assertEqual(package['generator_source_sha256'],runner.GENERATOR_SHA)
        copied=self.root/'package';shutil.copytree(runner.PACKAGE,copied)
        model['base_model']['means'][0][0]+=.01;runner.write(copied/'model.json',model)
        with self.assertRaisesRegex(ValueError,'package changed'):runner.package_check(copied)
        # Even an internally re-hashed package cannot silently alter its base.
        m=runner.read(copied/'manifest.json');m['model_sha256']=runner.sha(copied/'model.json');runner.write(copied/'manifest.json',m)
        frozen=runner.read(copied/'freeze.json')
        for name in ('model.json','manifest.json'):frozen['files'][name]=runner.sha(copied/name)
        runner.write(copied/'freeze.json',frozen)
        with self.assertRaisesRegex(ValueError,'changes geometry atlas'):runner.package_check(copied)

    def freeze(self):
        out=self.root/'campaign';runner.freeze(out);return out

    def test_inert_freeze_full_source_reference_binding_and_no_overwrite(self):
        out=self.freeze();protocol=runner.check(out);m=runner.read(out/'geometry/manifest.json')
        self.assertEqual(protocol['total_jobs'],4);self.assertEqual(protocol['maximum_physical_workers'],4)
        self.assertEqual(protocol['audit_workers'],4)
        self.assertEqual(len(m['rust_sources']),31)
        self.assertEqual(len([n for n in m['input_sha256']if n.startswith('reference/')]),8)
        self.assertEqual(m['model_sha256'],self.model_sha)
        self.assertFalse((out/'status.json').exists())
        for j in m['jobs']:
            self.assertEqual(j['command'],runner.positive.command(out/'geometry',j))
            self.assertIn('--no-gsd',j['command']);self.assertNotIn('--region',j['command'])
        with self.assertRaisesRegex(ValueError,'Fresh'):runner.freeze(out)
        (out/'geometry/logs/existing.log').write_text('preserve')
        with self.assertRaisesRegex(ValueError,'Existing'):runner.check(out)
        (out/'geometry/logs/existing.log').unlink()
        (out/'geometry/provenance/model.json').write_text('altered')
        with self.assertRaisesRegex(ValueError,'Frozen'):runner.validate(out)

    def test_four_worker_batch_drains_on_launch_failure_without_retry(self):
        jobs=[dict(id=str(i),directory=str(self.root/f'out{i}'),log=str(self.root/f'{i}.log'),command=[str(i)],status='pending')for i in range(4)]
        launched=[];drained=[]
        class Child:
            def __init__(self,i):self.i=i;self.pid=i+100
            def wait(self):drained.append(self.i);return 0
        def spawn(command,**kwargs):
            i=int(command[0])
            if i==2:raise OSError('mock launch failure')
            launched.append(i);return Child(i)
        with self.assertRaises(OSError):runner.execute_batches(jobs,lambda:None,popen=spawn)
        self.assertEqual(launched,drained);self.assertEqual(drained,[0,1])
        self.assertTrue(all(j['status']=='not_started'for j in jobs[2:]))

    def test_physical_failure_skips_audit_and_preserves_terminal_failure(self):
        out=self.freeze()
        def fail(jobs,snapshot):
            for j in jobs:j.update(status='failed',returncode=1)
            snapshot();raise RuntimeError('mock physical failure')
        with patch.object(runner,'execute_batches',side_effect=fail),patch.object(runner.subprocess,'run')as audit:
            with self.assertRaises(RuntimeError):runner.run(out)
            audit.assert_not_called()
        state=runner.read(out/'status.json');self.assertEqual(state['phase'],'physical_failed');self.assertFalse(state['complete'])
        self.assertFalse(runner.read(out/'geometry/status.json')['complete'])
        with self.assertRaisesRegex(ValueError,'retry'):runner.run(out)

    def test_audit_failure_occurs_only_after_all_four_outputs_validate(self):
        out=self.freeze();sequence=[]
        def complete(jobs,snapshot):
            for j in jobs:j.update(status='complete',returncode=0)
            sequence.append('drained');snapshot()
        def output(*args):sequence.append('validation');return dict(files={})
        def audit(argv,**kwargs):
            sequence.append('audit');self.assertEqual(argv[-2:],['--workers','4']);return subprocess.CompletedProcess(argv,1)
        with patch.object(runner,'execute_batches',side_effect=complete),patch.object(runner.positive,'verify_output',side_effect=output),patch.object(runner.subprocess,'run',side_effect=audit):
            with self.assertRaises(subprocess.CalledProcessError):runner.run(out)
        self.assertEqual(sequence,['drained']+['validation']*4+['audit'])
        state=runner.read(out/'status.json');self.assertEqual(state['phase'],'audit_failed');self.assertFalse(state['complete'])
        self.assertEqual(state['audits']['geometry']['returncode'],1)


if __name__=='__main__':unittest.main()
