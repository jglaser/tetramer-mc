"""Controller-only fixed allocation and failure tests; no physical execution.

Preparation internals are independently tested by their own module. These tests
replace its validation boundary with a synthetic frozen package, while retaining
the controller's actual target/pin/seed/archive checks and shared executor.
"""
import copy
import functools
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import run_protected_guide_validation as runner
from test_contact_confirmation import fixture as base_fixture, refreeze


def synthetic_preparation(root, package, binary, bundle):
    root.mkdir(); (root/'proposal-reference').mkdir()
    old=runner.read(package/'guide-bank.json'); guide=copy.deepcopy(old)
    for c in guide['gaussian_components']: c['weight'] *= .5
    guide['gaussian_components'] += [dict(weight=.125,mean=[float(i)]*6,covariance=np.eye(6).tolist()) for i in range(4)]
    runner.write(root/'guide-protected.json',guide)
    runner.write(root/'proposal-reference/declaration.json',dict(seeds=[148001010+1009*i for i in range(8)]))
    runner.write(root/'proposal-reference/validation.json',dict(complete=True,physical_draws=0))
    refreeze(root/'proposal-reference')
    runner.write(root/'sphere-reference.json',dict(schema='conditional-ray-sphere-crosslanguage-v1',complete=True,
        binary_sha256=runner.sha(binary),source_bundle_sha256=runner.sha(bundle),
        results=[dict(activity=z,physical_returncode=0,audit_returncode=0,checks=[dict(passed=True)]) for z in (0.,2.)]))
    plan=dict(schema='synthetic-protected-preparation',complete=True,training_seeds=[71000,71001],
        guide_sha256=dict(bank=runner.sha(package/'guide-bank.json'),protected=runner.sha(root/'guide-protected.json')),
        **{key:runner.sha(package/name) for key,name in [('region_sha256','region.json'),('shape_sha256','shape.json'),
            ('config_sha256','config.json'),('reference_region_sha256','old-r5-region.json')]},
        proposal_reference='proposal-reference',sphere_reference='sphere-reference.json',
        sphere_reference_sha256=runner.sha(root/'sphere-reference.json'),binary_sha256=runner.sha(binary),source_bundle_sha256=runner.sha(bundle))
    runner.write(root/'plan.json',plan);runner.write(root/'preparation.json',plan);refreeze(root)


def checked_synthetic_preparation(root):
    root=Path(root)
    for name,digest in runner.read(root/'freeze.json')['files'].items():
        runner.require(runner.sha(root/name) == digest,'Synthetic preparation changed')
    plan=runner.read(root/'plan.json');runner.require(plan['complete'] is True,'Incomplete preparation')
    runner.require(plan['guide_sha256']['protected'] == runner.sha(root/'guide-protected.json'),'Protected guide binding changed')
    return plan


class ProtectedControllerTests(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);self.root=Path(tmp.name)
        self.repo,self.binary,self.bundle,self.package,_=base_fixture(self.root)
        self.preparation=self.root/'protected-preparation'
        synthetic_preparation(self.preparation,self.package,self.binary,self.bundle)
        changes=[patch.multiple(runner,ROOT=self.repo,BINARY_SHA256=runner.sha(self.binary),
            BUNDLE_SHA256=runner.sha(self.bundle),OLD_GUIDE_SHA256=runner.sha(self.package/'guide-bank.json'),
            SPHERE_REFERENCE_SHA256=runner.sha(self.preparation/'sphere-reference.json')),
            patch.object(runner,'validate_base_package',side_effect=lambda p:(runner.read(Path(p)/'plan.json'),{'synthetic_geometry':True})),
            patch.object(runner,'validate_preparation',side_effect=checked_synthetic_preparation)]
        for change in changes:change.start();self.addCleanup(change.stop)

    def freeze(self):
        out=self.root/'campaign'
        runner.freeze(out,self.binary,self.bundle,self.package,self.preparation,self.repo)
        return out

    def test_fixed_budget_all_controls_full_closure_and_no_outputs(self):
        out=self.freeze();p=runner.preflight(out)
        self.assertEqual([a['id'] for a in p['arms']],['bank','protected','small','intensity256'])
        self.assertEqual([a['samples'] for a in p['arms']],[1048576,1048576,262144,262144])
        self.assertEqual(sum(j['samples'] for j in p['jobs']),10485760)
        self.assertEqual([j['seed'] for j in p['jobs']],[148101010+1009*i for i in range(16)])
        self.assertEqual(p['decision_arms'],['bank','protected','intensity256'])
        self.assertEqual(p['comparisons'],[['bank','protected'],['protected','small'],['protected','intensity256'],['small','intensity256']])
        self.assertEqual((p['maximum_physical_workers'],p['maximum_total_workers'],p['maximum_classification_workers']),(8,32,16))
        self.assertEqual(p['convergence'],runner.CONVERGENCE);self.assertEqual(p['strata'],runner.STRATA)
        self.assertIn('run_full_vessel_comparison.py',p['python_sources'])
        for name in (runner.RUNNER_NAME,runner.ANALYZER_NAME,runner.WORKFLOW_NAME,'prepare_protected_guide_validation.py'):
            self.assertIn(name,p['python_sources'])
        for arm in p['arms']:
            manifest=runner.read(out/arm['id']/'manifest.json')
            self.assertEqual(manifest['lambda_ratio'],arm['lambda_ratio'])
            for job in manifest['jobs']:
                self.assertEqual(job['command'][-4:],['--cloud-replicates','2','--lambda-ratio',str(arm['lambda_ratio'])])
                self.assertNotIn('--target-region',job['command'])
        self.assertFalse((out/'status.json').exists())
        self.assertTrue(all(not Path(j['directory']).exists() for j in p['jobs']))

    def test_seed_inventory_catches_recursive_protocol_training_and_reference(self):
        previous=self.repo/'runs/nested/archive';previous.mkdir(parents=True)
        runner.write(previous/'protocol.json',dict(nested={'jobs':[dict(seed=runner.SEEDS[7])]}))
        with self.assertRaisesRegex(ValueError,'seed collides'):self.freeze()
        self.assertFalse((self.root/'campaign').exists())
        (previous/'protocol.json').unlink()
        path=self.preparation/'plan.json';initial=runner.read(path)
        changed=copy.deepcopy(initial);changed['training_seeds'].append(runner.SEEDS[2]);runner.write(path,changed);refreeze(self.preparation)
        with self.assertRaisesRegex(ValueError,'seed collides'):self.freeze()
        runner.write(path,initial)
        runner.write(self.preparation/'proposal-reference/declaration.json',dict(seeds=[runner.SEEDS[3]]))
        refreeze(self.preparation/'proposal-reference');refreeze(self.preparation)
        with self.assertRaisesRegex(ValueError,'seed collides'):self.freeze()

    def test_archived_seed_declarations_remain_independently_checkable(self):
        previous=self.repo/'runs/old';previous.mkdir(parents=True)
        path=previous/'protocol.json';runner.write(path,dict(jobs=[dict(seed=553)]))
        digest=runner.sha(path);out=self.freeze();path.unlink()
        runner.validate(out)  # Future external deletion does not erase provenance.
        archived=out/'common/seed-protocols'/(digest+'.json')
        self.assertEqual(runner.read(archived)['jobs'][0]['seed'],553)
        runner.write(archived,dict(jobs=[dict(seed=runner.SEEDS[0])]))
        with self.assertRaisesRegex(ValueError,'Frozen file changed'):runner.validate(out)

    def test_refrozen_law_mutations_cannot_change_protocol(self):
        out=self.freeze();original=runner.read(out/'protocol.json')
        changes=[('total_unconditional_draws',1),('maximum_classification_workers',32),
                 ('comparisons',runner.COMPARISONS[:3]),('decision_arms',['protected'])]
        for key,value in changes:
            altered=copy.deepcopy(original);altered[key]=value;runner.write(out/'protocol.json',altered);refreeze(out)
            with self.subTest(key=key),self.assertRaises(ValueError):runner.validate(out)
        runner.write(out/'protocol.json',original);refreeze(out);runner.validate(out)

    def test_preparation_target_binary_and_runtime_fail_closed(self):
        planpath=self.preparation/'plan.json';plan=runner.read(planpath)
        for key in ('region_sha256','shape_sha256','config_sha256','reference_region_sha256','binary_sha256'):
            changed=copy.deepcopy(plan);changed[key]='0'*64;runner.write(planpath,changed);refreeze(self.preparation)
            with self.subTest(key=key),self.assertRaises(ValueError):self.freeze()
            self.assertFalse((self.root/'campaign').exists())
        runner.write(planpath,plan);refreeze(self.preparation);out=self.freeze()
        with patch.object(runner,'runtime',return_value={'changed':True}):
            with self.assertRaisesRegex(ValueError,'runtime'):runner.preflight(out)

    def test_no_retry_overwrite_or_partial_output_discard(self):
        out=self.freeze();existing=out/'protected/logs/preserve';existing.write_text('retain partial output')
        with self.assertRaisesRegex(ValueError,'Existing outputs'):runner.preflight(out)
        self.assertEqual(existing.read_text(),'retain partial output');existing.unlink()
        runner.write(out/'status.json',dict(complete=False))
        with self.assertRaisesRegex(ValueError,'already launched'):runner.preflight(out)
        with self.assertRaisesRegex(ValueError,'Fresh validation'):self.freeze()

    def population(self,out,p,index):
        job=p['jobs'][index];arm=next(a for a in p['arms'] if a['id']==job['arm']);n=job['samples']
        d=Path(job['directory']);a=out/arm['id']/'provenance';(d/'provenance').mkdir(parents=True)
        manifest=runner.read(out/arm['id']/'manifest.json')
        pm=dict(schema=runner.POPULATION_SCHEMA,samples=n,seed=job['seed'],cloud_replicates=2,activity=.035,
            lambda_ratio=arm['lambda_ratio'],importance_uniform_probability=.5,importance_component_count=arm['component_count'],
            proposal_density_measure=runner.DENSITY_MEASURE,executable_sha256=p['binary_sha256'],source_bundle_sha256=p['source_bundle_sha256'],
            minimum_latent_radius=0.,latent_radius=4.,minimum_original_q=0.,maximum_original_q=None,
            minimum_original_q_inclusive=True,maximum_original_q_inclusive=True,
            physical_fixed_neighbors=runner.read(a/'config.json')['fixed_poses'],chart_anchor=runner.read(a/'region.json')['fixed_neighbor'])
        pm['lambda']=.035*arm['lambda_ratio']
        for key in ('region_sha256','shape_sha256','config_sha256','importance_guide_sha256'):pm[key]=manifest[key]
        for dest,source in [('input-config.json','config.json'),('region.json','region.json'),('shape.json','shape.json'),
            ('importance-guide.json','importance-guide.json'),('source-bundle.json','source-bundle.json')]:shutil.copy2(a/source,d/'provenance'/dest)
        (d/'samples.jsonl').write_text('synthetic binding fixture, not physical rows\n')
        summary=dict(complete=True,manifest=pm,samples=n,shell_rejected=12,samples_sha256=runner.sha(d/'samples.jsonl'),
            sampler_cpu_seconds=1.,estimates={name:dict(draws=n) for name in ('region','hard_region')})
        runner.write(d/'manifest.json',pm);runner.write(d/'summary.json',summary)
        return job,d,pm,summary

    def test_all_arm_laws_keep_unconditional_zeros_and_cloud_intensity(self):
        out=self.freeze();p=runner.validate(out)
        for index in (0,4,8,12):
            job,d,pm,summary=self.population(out,p,index)
            self.assertEqual(runner.verify_output(out,p,job)['shell_rejected'],12)
            altered=copy.deepcopy(summary);altered['estimates']['region']['draws']-=12;runner.write(d/'summary.json',altered)
            with self.assertRaisesRegex(ValueError,'denominator'):runner.verify_output(out,p,job)
            runner.write(d/'summary.json',summary)
            altered=copy.deepcopy(pm);altered['lambda']=123.;changed=copy.deepcopy(summary);changed['manifest']=altered
            runner.write(d/'manifest.json',altered);runner.write(d/'summary.json',changed)
            with self.assertRaisesRegex(ValueError,'Population law'):runner.verify_output(out,p,job)

    def test_failure_drains_without_audit_or_refill_and_records_all_jobs(self):
        out=self.freeze();children=[]
        class Child:
            pid=9100
            def poll(self):return 7
            def wait(self):return 7
        def popen(*args,**kwargs):
            self.assertEqual(kwargs['env']['PYTHONOPTIMIZE'],'0')
            self.assertTrue(all(kwargs['env'][k]=='1' for k in runner.THREAD_ENV))
            child=Child();children.append(child);return child
        helper=functools.partial(runner.execute_group,popen=popen,pause=lambda _:None,
            capacity=lambda _:dict(workers=0,physical_pids=[]),birth=lambda _:123)
        with patch.object(runner,'execute_group',helper):
            with self.assertRaisesRegex(RuntimeError,'Child failed'):runner.run(out)
        self.assertEqual(len(children),1)
        state=runner.read(out/'status.json');self.assertEqual(state['phase'],'physical_failed')
        self.assertEqual(len(state['jobs']),16);self.assertEqual(state['jobs'][0]['status'],'failed')
        self.assertTrue(all(j['status']=='not_started' for j in state['jobs'][1:]));self.assertEqual(state['audits'],{})
        with self.assertRaisesRegex(ValueError,'already launched'):runner.preflight(out)


if __name__=='__main__':unittest.main()
