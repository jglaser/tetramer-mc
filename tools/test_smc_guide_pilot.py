"""No-physics tests of the fresh SMC-guide pilot's target and lifecycle contract."""
import copy
import functools
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import run_smc_guide_pilot as runner
from test_contact_confirmation import fixture as old_fixture, refreeze


def fixture(root):
    repo,binary,bundle,package,pins=old_fixture(root)
    geometry=root/'geometry';geometry.mkdir()
    old=runner.read(package/'guide-bank.json');guide=copy.deepcopy(old)
    for c in guide['gaussian_components']:c['weight']*=.5
    guide['gaussian_components'] += [dict(weight=.125,mean=[float(i)]*6,covariance=np.eye(6).tolist()) for i in range(4)]
    runner.write(geometry/'guide-r5-cov1.json',guide)
    declaration=dict(schema='synthetic-declaration');runner.write(geometry/'declaration.json',declaration)
    guides={'r5-cov1':dict(sha256=runner.sha(geometry/'guide-r5-cov1.json'))}
    info=dict(schema='smc-terminal-geometry-guide-preparation-v1',complete=True,all_terminal_slots_retained=True,
              native_classifier_calls=0,physical_draws_launched=0,audits_replayed=0,guides=guides)
    plan=dict(info,fit_population_ids=['r00','r01'],heldout_population_ids=['r02','r03'],populations=[dict(seed=4000+i) for i in range(4)],
        old_guide=str(package/'guide-bank.json'),current_region=str(package/'region.json'),historical_r5_region=str(package/'old-r5-region.json'),
        input_sha256={str(package/name):runner.sha(package/name) for name in ('guide-bank.json','region.json','old-r5-region.json')})
    runner.write(geometry/'plan.json',plan);runner.write(geometry/'preparation.json',info);refreeze(geometry)
    reference=root/'proposal-reference';reference.mkdir()
    decl=dict(schema='smc-guide-analytic-volume-reference-v1',samples_per_population=16384,populations_per_arm=4,
        arms=['bank','smc'],alpha=.5,expected_mean=1.,weight_bounds=[0.,2.],family_failure_probability=1e-6,
        seeds=list(range(8000,8008)),source_sha256={str(p):runner.sha(p) for p in
            [geometry/'guide-r5-cov1.json',package/'guide-bank.json',package/'region.json',geometry/'plan.json',geometry/'freeze.json']},code_sha256={})
    runner.write(reference/'declaration.json',decl)
    runner.write(reference/'validation.json',dict(schema=decl['schema'],complete=True,physical_draws=0,classifiers_rerun=0,
        audits_replayed=0,total_proposal_only_draws=131072,declaration_sha256=runner.sha(reference/'declaration.json'),
        arms={name:dict(passed=True) for name in ('bank','smc')}));refreeze(reference)
    sphere=root/'sphere-reference.json';runner.write(sphere,dict(schema='conditional-ray-sphere-crosslanguage-v1',complete=True,
        binary_sha256=runner.sha(binary),source_bundle_sha256=runner.sha(bundle),
        results=[dict(activity=z,physical_returncode=0,audit_returncode=0,checks=[dict(passed=True)]) for z in (0.,2.)]))
    return repo,binary,bundle,package,pins,geometry,reference,sphere


class PilotTests(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup);self.root=Path(temporary.name)
        self.repo,self.binary,self.bundle,self.package,pins,self.geometry,self.reference,self.sphere=fixture(self.root)
        changes=[patch.multiple(runner,ROOT=self.repo,GUIDE_SHA256=runner.sha(self.geometry/'guide-r5-cov1.json'),
            OLD_GUIDE_SHA256=runner.sha(self.package/'guide-bank.json'),SPHERE_REFERENCE_SHA256=runner.sha(self.sphere)),
            patch.multiple(runner.confirmation,PINS=pins,RETAINED_GUIDE_SHA256=runner.sha(self.package/'retained-guide.json'))]
        for change in changes:change.start();self.addCleanup(change.stop)

    def freeze(self):
        out=self.root/'pilot'
        runner.freeze(out,self.binary,self.bundle,self.package,self.geometry,self.repo,self.reference,self.sphere)
        return out

    def test_exact_allocation_full_target_sources_and_reference_bindings(self):
        before=runner.file_hashes(self.package)
        with patch.object(runner.sys, 'dont_write_bytecode', False):
            out=self.freeze();p=runner.validate(out)
            self.assertFalse(runner.sys.dont_write_bytecode)
        self.assertEqual(p['total_unconditional_draws'],524288)
        self.assertEqual([j['seed'] for j in p['jobs']],[137101010+1009*i for i in range(8)])
        self.assertEqual([a['component_count'] for a in p['arms']],[80,84])
        self.assertEqual([j['samples'] for j in p['jobs']],[65536]*8)
        self.assertEqual(p['convergence'],runner.CONVERGENCE)
        self.assertEqual(p['decision_arms'],['bank','smc'])
        self.assertEqual(p['maximum_physical_workers'],8);self.assertEqual(p['maximum_total_workers'],32)
        for job in p['jobs']:
            self.assertNotIn('--target-region',job['command'])
            self.assertEqual(job['command'][-4:],['--cloud-replicates','2','--lambda-ratio','128.0'])
        self.assertIn('run_full_vessel_comparison.py',p['python_sources'])
        self.assertIn('analyze_smc_guide_pilot.py',p['python_sources'])
        self.assertIn('run_smc_guide_workflow.py',p['python_sources'])
        self.assertEqual(p['references']['sphere_validation_sha256'],runner.sha(self.sphere))
        self.assertEqual(runner.file_hashes(self.package),before)
        runner.preflight(out)

    def test_geometry_must_preserve_all_slots_and_retained_gaussian_law(self):
        path=self.geometry/'plan.json';original=runner.read(path)
        for key,value in [('all_terminal_slots_retained',False),('native_classifier_calls',1),('fit_population_ids',['r00','r02'])]:
            changed=copy.deepcopy(original);changed[key]=value;runner.write(path,changed);refreeze(self.geometry)
            with self.assertRaises(ValueError):self.freeze()
            self.assertFalse((self.root/'pilot').exists())
        runner.write(path,original);refreeze(self.geometry)
        guide=runner.read(self.geometry/'guide-r5-cov1.json');guide['gaussian_components'][0]['mean'][0]+=.1
        runner.write(self.geometry/'guide-r5-cov1.json',guide);refreeze(self.geometry)
        with self.assertRaisesRegex(ValueError,'Selected SMC guide'):self.freeze()

    def test_seed_collisions_rejected_before_creation(self):
        campaign=self.repo/'runs/prior';campaign.mkdir(parents=True)
        runner.write(campaign/'protocol.json',dict(jobs=[dict(seed=runner.SEEDS[3])]))
        with self.assertRaisesRegex(ValueError,'seed collides'):self.freeze()
        self.assertFalse((self.root/'pilot').exists())

    def test_runtime_source_and_reference_fail_closed(self):
        (self.repo/'src/synthetic.rs').write_text('stale binary source')
        with self.assertRaisesRegex(ValueError,'Reviewed source differs'):self.freeze()
        self.assertFalse((self.root/'pilot').exists())
        source=runner.read(self.bundle)['files']['src/synthetic.rs']['text'];(self.repo/'src/synthetic.rs').write_text(source)
        p=self.reference/'validation.json';data=runner.read(p);data['arms']['smc']['passed']=False;runner.write(p,data);refreeze(self.reference)
        with self.assertRaisesRegex(ValueError,'reference failed'):self.freeze()
        self.assertFalse((self.root/'pilot').exists())

    def test_frozen_output_and_no_retry(self):
        out=self.freeze()
        with patch.object(runner,'runtime',return_value=dict(changed=True)):
            with self.assertRaisesRegex(ValueError,'runtime'):runner.preflight(out)
        (out/'bank/logs/preserve').write_text('preserve')
        with self.assertRaisesRegex(ValueError,'Existing outputs'):runner.preflight(out)
        (out/'bank/logs/preserve').unlink();runner.write(out/'status.json',dict(complete=False))
        with self.assertRaisesRegex(ValueError,'already launched'):runner.preflight(out)
        with self.assertRaisesRegex(ValueError,'Fresh pilot'):self.freeze()

    def test_population_full_unconditional_denominator_and_arm_count(self):
        out=self.freeze();p=runner.validate(out)
        for index in (0,4):
            job=p['jobs'][index];arm=p['arms'][index//4];d=Path(job['directory']);a=out/arm['id']/'provenance'
            (d/'provenance').mkdir(parents=True);manifest=runner.read(out/arm['id']/'manifest.json')
            pm=dict(schema=runner.POPULATION_SCHEMA,samples=65536,seed=job['seed'],cloud_replicates=2,activity=.035,
                lambda_ratio=128.,importance_uniform_probability=.5,importance_component_count=arm['component_count'],
                proposal_density_measure=runner.DENSITY_MEASURE,executable_sha256=p['binary_sha256'],source_bundle_sha256=p['source_bundle_sha256'],
                minimum_latent_radius=0.,latent_radius=4.,minimum_original_q=0.,maximum_original_q=None,
                minimum_original_q_inclusive=True,maximum_original_q_inclusive=True,
                physical_fixed_neighbors=runner.read(a/'config.json')['fixed_poses'],chart_anchor=runner.read(a/'region.json')['fixed_neighbor'])
            pm['lambda']=.035*128
            for key in ('region_sha256','shape_sha256','config_sha256','importance_guide_sha256'):pm[key]=manifest[key]
            for dest,source in [('input-config.json','config.json'),('region.json','region.json'),('shape.json','shape.json'),
                ('importance-guide.json','importance-guide.json'),('source-bundle.json','source-bundle.json')]:shutil.copy2(a/source,d/'provenance'/dest)
            (d/'samples.jsonl').write_text('synthetic fixture; not physical rows\n')
            summary=dict(complete=True,manifest=pm,samples=65536,shell_rejected=12,samples_sha256=runner.sha(d/'samples.jsonl'),
                sampler_cpu_seconds=1.,estimates={name:dict(draws=65536) for name in ('region','hard_region')})
            runner.write(d/'manifest.json',pm);runner.write(d/'summary.json',summary)
            self.assertEqual(runner.verify_output(out,p,job)['shell_rejected'],12)
            wrong=copy.deepcopy(summary);wrong['estimates']['region']['draws']-=12;runner.write(d/'summary.json',wrong)
            with self.assertRaisesRegex(ValueError,'denominator'):runner.verify_output(out,p,job)
            runner.write(d/'summary.json',summary);pm['importance_component_count']+=1;summary['manifest']=pm
            runner.write(d/'manifest.json',pm);runner.write(d/'summary.json',summary)
            with self.assertRaisesRegex(ValueError,'Population law'):runner.verify_output(out,p,job)

    def test_failure_draining_helper_is_used_without_starting_audits_or_refilling(self):
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
        self.assertEqual(state['jobs'][0]['process_birth'],123)
        self.assertEqual(state['jobs'][0]['status'],'failed')
        self.assertTrue(all(j['status']=='not_started' for j in state['jobs'][1:]));self.assertEqual(state['audits'],{})
        with self.assertRaisesRegex(ValueError,'already launched'):runner.preflight(out)


if __name__=='__main__':unittest.main()
