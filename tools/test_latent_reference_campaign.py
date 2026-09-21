"""Synthetic freeze/lifecycle controls; no physical jobs or production freeze."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import run_latent_reference_campaign as runner


def fixture(root):
    write=runner.write;source=root/'inputs';source.mkdir()
    p=lambda x:dict(position=[x,0.,0.],orientation=[1.,0.,0.,0.])
    shape=dict(atoms=[dict(center=[0.,0.,0.],radius=.5)])
    write(source/'shape.json',shape);shape_sha=runner.sha(source/'shape.json')
    cfg=dict(shape=str(source/'shape.json'),fixed_poses=[p(0),p(10)],capture_center=[2.,0.,0.],capture_radius=50.,
        reservoir_density=.035,depletant_radius=1.5,poisson_lambda_ratio=64.,metadata=dict(physical_sphere_radius_A=100.))
    chart=dict(schema='weighted-pose-mixture-v1',coordinate_convention='anchor-body-relative',angular_length=1.,
        shape_sha256=shape_sha,weights=[1.],anchors=[dict(position=[3.,0.,0.],rotation=np.eye(3).tolist())],
        means=[[0.]*6],covariances=[(np.eye(6)*.04).tolist()])
    region=dict(fixed_neighbor=p(10),physical_fixed_neighbors=cfg['fixed_poses'],capture_center=cfg['capture_center'],
        capture_radius=50.,activity=.035,depletant_radius=1.5,physical_metric=cfg['metadata'],shape_sha256=shape_sha,
        gaussian_chart=chart,minimum_mahalanobis_radius=0.,mahalanobis_radius=4.,
        minimum_original_q=0.,minimum_original_q_inclusive=True)
    write(source/'config.json',cfg);write(source/'region.json',region)
    review=root/'review';(review/'source/src').mkdir(parents=True);(review/'python').mkdir()
    rust='// synthetic source\n';(review/'source/src/test.rs').write_text(rust)
    digest=runner.sha(review/'source/src/test.rs')
    write(review/'source-bundle.json',dict(files={'src/test.rs':dict(text=rust,sha256=digest)}))
    (review/'binary').write_bytes(b'synthetic executable header'+(review/'source-bundle.json').read_bytes())
    names=['launcher.py','run_latent_region_campaign.py','analyze_latent_region.py','prepare_smc_normalizer_atlas.py',
        'prepare_deep_far_normalizer_atlas.py','analyze_basin_normalizers.py','analyze_native_region_reference.py','analyze_latent_region_shells.py']
    for name in names:(review/'python'/name).write_text('# synthetic frozen dependency '+name+'\n')
    write(review/'python-closure-complete.json',dict(source_sha256={name:runner.sha(review/'python'/name)for name in names}))
    write(review/'validation.json',dict(binary=str(review/'binary'),source_bundle=str(review/'source-bundle.json'),
        source_sha256={'src/test.rs':digest}))
    constants=dict(BINARY_SHA=runner.sha(review/'binary'),BUNDLE_SHA=runner.sha(review/'source-bundle.json'),
        REVIEW_SHA=runner.sha(review/'validation.json'),PYTHON_REVIEW_SHA=runner.sha(review/'python-closure-complete.json'))
    return source,review,cfg,region,shape,constants


class LatentReferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.source,self.review,self.cfg,self.region,self.shape,self.constants=fixture(self.root)
        self.patch=patch.multiple(runner,**self.constants);self.patch.start();self.addCleanup(self.patch.stop)

    def freeze(self,**kwargs):
        out=self.root/'campaign'
        runner.freeze(out,self.source/'config.json',self.source/'region.json',review=self.review,**kwargs)
        return out

    def test_second_neighbor_anchor_and_shifted_capture_preserve_exact_target(self):
        before=copy.deepcopy((self.cfg,self.region,self.shape))
        result=runner.validate_region(self.region,self.cfg,self.shape,self.region['shape_sha256'])
        self.assertEqual(result['chart_anchor_index'],1)
        self.assertAlmostEqual(result['chart_center'][0],13.)
        self.assertAlmostEqual(result['capture_center_norm_upper_A'],11.8)
        self.assertEqual((self.cfg,self.region,self.shape),before)
        out=self.freeze();self.assertEqual(runner.check(out)['chart_anchor_index'],1)
        self.assertEqual((out/'provenance/region.json').read_bytes(),(self.source/'region.json').read_bytes())
        cfg=runner.read(out/'provenance/config.json');cfg['shape']=self.cfg['shape'];self.assertEqual(cfg,self.cfg)

    def test_target_mismatch_bad_anchor_and_uncertified_wall_fail_closed(self):
        for mode in ('scaffold','anchor','metric','shape','q','wall','covariance'):
            region=copy.deepcopy(self.region)
            if mode=='scaffold':region['physical_fixed_neighbors']=region['physical_fixed_neighbors'][:1]
            elif mode=='anchor':region['fixed_neighbor']['position']=[99.,0.,0.]
            elif mode=='metric':region['physical_metric']={}
            elif mode=='shape':region['shape_sha256']='wrong'
            elif mode=='q':region['minimum_original_q_inclusive']=1
            elif mode=='wall':region['gaussian_chart']['anchors'][0]['position']=[1000.,0.,0.]
            else:region['gaussian_chart']['covariances'][0][0][0]=-1
            with self.subTest(mode=mode),self.assertRaises((ValueError,np.linalg.LinAlgError)):
                runner.validate_region(region,self.cfg,self.shape,self.region['shape_sha256'])

    def test_independent_allocation_and_frozen_command_are_exact(self):
        out=self.freeze(samples=12,replicates=4,seed=128101010)
        protocol=runner.validate(out);manifest=runner.read(out/'manifest.json')
        self.assertEqual(protocol['seeds'],[128101010,128102019,128103028,128104037])
        for job in manifest['jobs']:
            self.assertEqual(job['command'],runner.command(out/'provenance',job,64.))
            self.assertNotIn('--importance-guide',job['command'])
        for values in ((0,4,1),(1,True,1),(1,4,2**64-1),(1,1,-1)):
            with self.assertRaises(ValueError):runner.allocation(*values)

    def test_changed_dependency_and_existing_outputs_prevent_launch(self):
        out=self.freeze();(out/'logs/existing.log').write_text('not ours')
        with self.assertRaisesRegex(ValueError,'Existing'):runner.check(out)
        (out/'logs/existing.log').unlink()
        (out/'provenance/analyze_latent_region.py').write_text('changed')
        with self.assertRaisesRegex(ValueError,'Frozen file changed'):runner.validate(out)
        with self.assertRaisesRegex(ValueError,'fresh'):self.freeze()

    def test_reference_package_identity_is_bound_but_never_a_sampler_filter(self):
        package=self.root/'reference';package.mkdir()
        for name in ('config.json','region.json'):(package/name).write_bytes((self.source/name).read_bytes())
        runner.write(package/'plan.json',dict(config='config.json',region='region.json',
            config_sha256=runner.sha(package/'config.json'),region_sha256=runner.sha(package/'region.json'),
            native_definition='provenance/native/definition.json',native_definition_sha256='diagnostic-only'))
        runner.write(package/'freeze.json',dict(files={p.name:runner.sha(p)for p in package.iterdir()}))
        out=self.freeze(reference_package=package)
        p=runner.validate(out);self.assertIsNotNone(p['reference_package'])
        self.assertEqual(runner.read(out/'provenance/region.json'),self.region)
        for job in runner.read(out/'manifest.json')['jobs']:
            self.assertFalse(any('native' in arg for arg in job['command']))

    def batch(self,count=12,fail_exit=None,fail_launch=None):
        jobs=[dict(id=str(i),directory=str(self.root/f'out{i}'),log=str(self.root/f'{i}.log'),command=[str(i)],status='pending')for i in range(count)]
        launched=[];waited=[];peak=[0]
        class Child:
            def __init__(self,i):self.i=i;self.pid=100+i
            def wait(self):waited.append(self.i);return 1 if self.i==fail_exit else 0
        def spawn(argv,**kwargs):
            i=int(argv[0])
            if i==fail_launch:raise OSError('mock launch failure')
            launched.append(i);peak[0]=max(peak[0],len(launched)-len(waited));return Child(i)
        return jobs,launched,waited,peak,spawn

    def test_worker_cap_draining_and_failure_prevents_later_batch(self):
        jobs,started,waited,peak,spawn=self.batch(fail_exit=1)
        with self.assertRaises(RuntimeError):runner.execute_batches(jobs,lambda:None,popen=spawn)
        self.assertEqual(started,waited);self.assertEqual(peak[0],8)
        self.assertEqual(started,list(range(8)));self.assertTrue(all(j['status']=='not_started'for j in jobs[8:]))

    def test_launch_failure_drains_and_does_not_retry(self):
        jobs,started,waited,_,spawn=self.batch(fail_launch=2)
        with self.assertRaises(OSError):runner.execute_batches(jobs,lambda:None,popen=spawn)
        self.assertEqual(started,waited);self.assertEqual(waited,[0,1])
        self.assertTrue(all(j['status']=='not_started'for j in jobs[2:]))

    def test_physical_failure_preserves_terminal_failure_and_skips_audit(self):
        out=self.freeze()
        def fail(jobs,snapshot):
            for j in jobs:j.update(status='failed',returncode=1)
            snapshot();raise RuntimeError('mock physical failure')
        with patch.object(runner,'execute_batches',side_effect=fail),patch.object(runner.subprocess,'run')as audit:
            with self.assertRaisesRegex(RuntimeError,'mock physical'):runner.run(out)
            audit.assert_not_called()
        state=runner.read(out/'status.json');self.assertFalse(state['complete']);self.assertEqual(state['phase'],'physical_failed')
        with self.assertRaisesRegex(ValueError,'retry'):runner.run(out)

    def test_audit_failure_is_terminal_after_all_physics_validate(self):
        out=self.freeze();sequence=[]
        def success(jobs,snapshot):
            for j in jobs:j.update(status='complete',returncode=0)
            sequence.append('all_drained');snapshot()
        def output(*args):sequence.append('validate');return dict(samples_sha256='saved')
        def audit(argv,**kwargs):sequence.append('audit');return subprocess.CompletedProcess(argv,1)
        with patch.object(runner,'execute_batches',side_effect=success),patch.object(runner,'verify_output',side_effect=output),patch.object(runner.subprocess,'run',side_effect=audit):
            with self.assertRaises(subprocess.CalledProcessError):runner.run(out)
        self.assertEqual(sequence,['all_drained']+['validate']*4+['audit'])
        state=runner.read(out/'status.json');self.assertFalse(state['complete']);self.assertEqual(state['phase'],'audit_failed')
        self.assertEqual(state['audit']['returncode'],1)


if __name__=='__main__':unittest.main()
