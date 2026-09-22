"""No-physics tests of the frozen 6D pilot's allocation, audit and lifecycle."""
import copy
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import run_contact_bank_pilot as runner
from test_entry_shell_reference_campaign import fixture as old_fixture


def fixture(root):
    repo, binary, bundle, package, _ = old_fixture(root)
    shutil.copy2(package/'inputs/tetramer-shape.json', package/'shape.json')
    native = package/'native-region'; (native/'inputs/source').mkdir(parents=True)
    shutil.copy2(package/'config.json', native/'inputs/physical-config.json')
    runtime = native/'inputs/source/native_contact_regions.py'
    runtime.write_text('import json\nclass NativeContactRegions:\n def __init__(self,path):\n  self.definition=json.loads(path.read_text())\n')
    definition = dict(criteria={}, scope='Synthetic portable classifier; no physical classifications',
        shape_sha256=runner.sha(package/'shape.json'), fixed_poses=runner.read(package/'config.json')['fixed_poses'],
        physical_config_sha256=runner.sha(native/'inputs/physical-config.json'),
        input_sha256={'physical-config.json':runner.sha(native/'inputs/physical-config.json'),
                      'source/native_contact_regions.py':runner.sha(runtime)})
    runner.write(native/'definition.json',definition)
    shutil.copy2(native/'definition.json', package/'native-definition.json')
    covariance = .04*np.eye(6); covariance[0, 4] = covariance[4, 0] = .002
    components = [dict(weight=(1/96 if i%2 == 0 else 1/48) if i < 48 else 1/32,
        mean=[.01*i, 0., 0., 0., 0., 0.], covariance=covariance.tolist()) for i in range(56)]
    guide = dict(schema=runner.GUIDE_SCHEMA, region_sha256=runner.sha(package/'region.json'),
                 defensive_uniform_shell_probability=.5, gaussian_components=components)
    runner.write(package/'guide-bank.json', guide)
    wide = copy.deepcopy(guide)
    for c in wide['gaussian_components']: c['covariance'] = (4*np.asarray(c['covariance'])).tolist()
    runner.write(package/'guide-wide.json', wide)
    plan = dict(schema=runner.PACKAGE_SCHEMA, complete=True, production_launched=False, native_definition='native-region/definition.json',
        **{name+'_sha256': runner.sha(package/(name+'.json')) for name in ('config', 'region', 'shape')},
        native_definition_sha256=runner.sha(package/'native-definition.json'),
        guide_sha256={name:runner.sha(package/f'guide-{name}.json') for name in ('bank', 'wide')},
        covariance_multipliers=dict(bank=1, wide=4),
        anchor_metadata=[dict(component_index=i, **{'class': 'native' if i >= 48 or i%2 == 0 else 'no-entry'},
            source_group='conditional-ray' if i < 48 else 'R5-reference', arm='source' if i < 48 else 'reference',
            population_id=f'p{i//2:02d}' if i < 48 else f'r{i-48:02d}', seed=1000+i//2 if i < 48 else 2000+i-48, draw=i,
            source_samples=16384, original_log_importance_weight=float(i)) for i in range(56)])
    runner.write(package/'plan.json', plan); refreeze(package)
    return repo, binary, bundle, package, {name:runner.sha(package/name) for name in runner.PINS}


def refreeze(package):
    (package/'freeze.json').unlink(missing_ok=True)
    runner.write(package/'freeze.json', dict(files=runner.file_hashes(package)))


class ContactBankPilotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup); self.root = Path(self.temp.name)
        self.repo, self.binary, self.bundle, self.package, pins = fixture(self.root)
        patches = patch.multiple(runner, ROOT=self.repo, PINS=pins); patches.start(); self.addCleanup(patches.stop)

    def freeze(self):
        out = self.root/'pilot'; runner.freeze(out, self.binary, self.bundle, self.package); return out

    def test_fixed_allocation_and_full_covariance_guides_are_archived(self):
        before = runner.file_hashes(self.package); out = self.freeze(); p = runner.validate(out)
        self.assertEqual(p['total_unconditional_draws'], 131072)
        self.assertEqual(sum(j['samples'] for j in p['jobs']), 131072)
        self.assertEqual([j['seed'] for j in p['jobs']], [132101010+1009*i for i in range(8)])
        self.assertEqual([a['id'] for a in p['arms']], ['bank', 'wide'])
        self.assertEqual(p['maximum_physical_workers'], 8)
        self.assertEqual(p['component_groups'], runner.COMPONENT_GROUPS)
        self.assertEqual(sum(g['count']*g['component_weight'] for g in p['component_groups']), 1.)
        self.assertEqual(set(p['thread_environment'].values()), {'1'})
        self.assertEqual(len(p['jobs']), 8)
        for arm in p['arms']:
            base = out/arm['id']; archive = base/'provenance'; m = runner.read(base/'manifest.json')
            self.assertEqual(arm['component_count'], 56); self.assertEqual(arm['alpha'], .5)
            self.assertEqual(m['schema'], runner.CAMPAIGN_SCHEMA)
            self.assertEqual((archive/'importance-guide.json').read_bytes(), (self.package/f"guide-{arm['id']}.json").read_bytes())
            self.assertEqual((archive/'region.json').read_bytes(), (self.package/'region.json').read_bytes())
            cfg = runner.read(archive/'config.json'); old = runner.read(self.package/'config.json')
            self.assertEqual(cfg.pop('shape'), str(archive/'shape.json')); old.pop('shape'); self.assertEqual(cfg, old)
            for job in m['jobs']:
                self.assertEqual(job['samples'], 16384); self.assertEqual(job['command'], runner.command(out, arm, job))
                self.assertEqual(job['command'][-4:], ['--cloud-replicates','2','--lambda-ratio','128.0'])
                self.assertNotIn('--target-region', job['command'])
        self.assertEqual(runner.file_hashes(self.package), before)
        self.assertIn('run_conditional_ray_campaign.py', p['python_sources'])
        self.assertIn('analyze_latent_region.py', p['python_sources'])
        self.assertIn('analyze_contact_bank_pilot.py', p['python_sources'])
        self.assertEqual(p['native_definition'], 'common/reference-package/native-region/definition.json')
        self.assertEqual(m['allocation']['component_count'], 56)
        with patch.object(runner, '__file__', str(out/'common/controller.py')): runner.preflight(out)

    def test_portable_classifier_is_loaded_and_assertions_cannot_be_disabled(self):
        with patch.object(runner, 'load_classifier', side_effect=ValueError('portable classifier rejected')):
            with self.assertRaisesRegex(ValueError, 'portable classifier'): self.freeze()
        self.assertFalse((self.root/'pilot').exists())
        with patch.dict(runner.os.environ, {'PYTHONOPTIMIZE': '2'}):
            self.assertEqual(runner.worker_environment()['PYTHONOPTIMIZE'], '0')

    def test_different_physical_shape_and_stale_source_fail_before_output_creation(self):
        (self.package/'shape.json').write_text('{"changed":true}')
        with self.assertRaisesRegex(ValueError, 'Preparation dependency'): self.freeze()
        self.assertFalse((self.root/'pilot').exists())
        refreeze(self.package)
        with self.assertRaisesRegex(ValueError, 'physical target'): self.freeze()
        self.assertFalse((self.root/'pilot').exists())

    def test_source_closure_is_checked_before_freeze_and_archived_afterwards(self):
        source = self.repo/'src/synthetic.rs'; original = source.read_text(); source.write_text('stale source')
        with self.assertRaisesRegex(ValueError, 'Reviewed source differs'): self.freeze()
        self.assertFalse((self.root/'pilot').exists()); source.write_text(original)
        out = self.freeze(); source.write_text('later development'); runner.validate(out)
        (out/'common/run_conditional_ray_campaign.py').write_text('changed imported job helper')
        with self.assertRaisesRegex(ValueError, 'Frozen file changed'): runner.validate(out)

    def test_anchor_classes_training_seeds_and_width_cannot_change(self):
        path = self.package/'plan.json'; original = runner.read(path)
        variants = []
        x=copy.deepcopy(original); x['anchor_metadata'][0]['class']='no-entry'; variants.append(x)
        x=copy.deepcopy(original); x['anchor_metadata'][0]['seed']=runner.SEEDS[0]; variants.append(x)
        x=copy.deepcopy(original); x['covariance_multipliers']['wide']=16; variants.append(x)
        x=copy.deepcopy(original); x['anchor_metadata'][48]['source_group']='conditional-ray'; variants.append(x)
        x=copy.deepcopy(original); x['anchor_metadata'][0]['component_index']=1; variants.append(x)
        for changed in variants:
            runner.write(path, changed); refreeze(self.package)
            with self.assertRaises(ValueError): self.freeze()
            self.assertFalse((self.root/'pilot').exists())
        runner.write(path,original)
        guidepath=self.package/'guide-wide.json'; wide=runner.read(guidepath)
        wide['gaussian_components'][0]['covariance'][0][4]=0.; wide['gaussian_components'][0]['covariance'][4][0]=0.
        runner.write(guidepath,wide); original['guide_sha256']['wide']=runner.sha(guidepath)
        runner.write(path,original);refreeze(self.package)
        with self.assertRaisesRegex(ValueError,'full covariance'):self.freeze()

    def test_reference_group_allocation_cannot_be_replaced_by_equal_weights(self):
        plan=runner.read(self.package/'plan.json')
        for arm in ('bank','wide'):
            path=self.package/f'guide-{arm}.json';guide=runner.read(path)
            for component in guide['gaussian_components']:component['weight']=1/56
            runner.write(path,guide);plan['guide_sha256'][arm]=runner.sha(path)
        runner.write(self.package/'plan.json',plan);refreeze(self.package)
        with self.assertRaisesRegex(ValueError,'Source-group component weights'):self.freeze()
        self.assertFalse((self.root/'pilot').exists())

    def test_preflight_no_overwrite_retry_and_runtime_binding(self):
        out=self.freeze();runner.preflight(out)
        with patch.object(runner,'runtime',return_value={'different':True}):
            with self.assertRaisesRegex(ValueError,'runtime'):runner.preflight(out)
        path=out/'bank/logs/existing.log';path.write_text('preserve')
        with self.assertRaisesRegex(ValueError,'Existing outputs'):runner.preflight(out)
        self.assertEqual(path.read_text(),'preserve');path.unlink()
        runner.write(out/'status.json',dict(complete=False))
        with self.assertRaisesRegex(ValueError,'no retries'):runner.preflight(out)
        with self.assertRaisesRegex(ValueError,'Fresh campaign'):self.freeze()

    def population(self,out,index=0):
        protocol=runner.read(out/'protocol.json');job=protocol['jobs'][index];d=Path(job['directory'])
        if d.exists():return protocol,job,d,runner.read(d/'manifest.json'),runner.read(d/'summary.json')
        archive=out/job['arm']/'provenance';m=runner.read(out/job['arm']/'manifest.json');(d/'provenance').mkdir(parents=True)
        pm=dict(schema=runner.POPULATION_SCHEMA,samples=job['samples'],seed=job['seed'],cloud_replicates=2,
            activity=.035,**{'lambda':.035*128},lambda_ratio=128.,importance_uniform_probability=.5,importance_component_count=56,
            proposal_density_measure=runner.DENSITY_MEASURE,executable_sha256=protocol['binary_sha256'],
            source_bundle_sha256=protocol['source_bundle_sha256'],minimum_latent_radius=0.,latent_radius=4.,
            minimum_original_q=0.,maximum_original_q=None,minimum_original_q_inclusive=True,maximum_original_q_inclusive=True,
            physical_fixed_neighbors=runner.read(archive/'config.json')['fixed_poses'],chart_anchor=runner.read(archive/'region.json')['fixed_neighbor'])
        for key in ('region_sha256','shape_sha256','config_sha256','importance_guide_sha256'):pm[key]=m[key]
        for target,source in [('input-config.json','config.json'),('region.json','region.json'),('shape.json','shape.json'),
                              ('importance-guide.json','importance-guide.json'),('source-bundle.json','source-bundle.json')]:
            shutil.copy2(archive/source,d/'provenance'/target)
        (d/'samples.jsonl').write_text('synthetic controller fixture; no physical sampling\n')
        summary=dict(complete=True,manifest=pm,samples=job['samples'],shell_rejected=12,
            samples_sha256=runner.sha(d/'samples.jsonl'),sampler_cpu_seconds=1.,
            estimates={name:dict(draws=job['samples']) for name in ('region','hard_region')})
        runner.write(d/'manifest.json',pm);runner.write(d/'summary.json',summary)
        return protocol,job,d,pm,summary

    def test_gaussian_exterior_draws_allowed_but_denominator_and_metadata_are_fixed(self):
        out=self.freeze();protocol,job,d,pm,summary=self.population(out)
        self.assertEqual(runner.verify_output(out,protocol,job)['shell_rejected'],12)
        for key in ('region','hard_region'):
            x=copy.deepcopy(summary);x['estimates'][key]['draws']-=12;runner.write(d/'summary.json',x)
            with self.assertRaisesRegex(ValueError,'denominator'):runner.verify_output(out,protocol,job)
        for value in (-1,True,16385):
            x=copy.deepcopy(summary);x['shell_rejected']=value;runner.write(d/'summary.json',x)
            with self.assertRaisesRegex(ValueError,'exterior'):runner.verify_output(out,protocol,job)
        for key,value in [('schema','importance-latent-region-normalizer-v3'),('importance_component_count',47),
                          ('activity',.04),('lambda',1.),('cloud_replicates',1),('shape_sha256','0'*64),
                          ('seed',job['seed']+1),('minimum_original_q_inclusive',False),('chart_anchor',pm['physical_fixed_neighbors'][0])]:
            current=copy.deepcopy(pm);current[key]=value;x=copy.deepcopy(summary);x['manifest']=current
            runner.write(d/'manifest.json',current);runner.write(d/'summary.json',x)
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'Population law'):runner.verify_output(out,protocol,job)
        runner.write(d/'manifest.json',pm);runner.write(d/'summary.json',summary);(d/'samples.jsonl').write_text('changed')
        with self.assertRaisesRegex(ValueError,'Rows changed'):runner.verify_output(out,protocol,job)

    def assessment(self,out,arm_index=0):
        protocol=runner.read(out/'protocol.json');arm=protocol['arms'][arm_index];jobs=copy.deepcopy(protocol['jobs']);populations=[]
        for i in range(4*arm_index,4*arm_index+4):
            _,job,_,_,summary=self.population(out,i);jobs[i]['output']=runner.verify_output(out,protocol,job)
            n=job['samples'];audit=dict(draws=n,shell_rejected=12,branch_counts={'uniform-shell':n-100,'gaussian':100},
                gaussian_component_counts=[100]+[0]*55,maximum_log_proposal_density_error=0.,
                maximum_latent_vector_reconstruction_error=0.,maximum_log_jacobian_error=0.)
            populations.append(dict(id=job['id'],seed=job['seed'],estimate=dict(draws=n),hard_region=dict(draws=n),
                samples_sha256=summary['samples_sha256'],importance_sampling_audit=audit))
        total=4*arm['samples'];m=runner.read(out/arm['id']/'manifest.json')
        analysis=dict(region_sha256=protocol['region_sha256'],physical_fixed_neighbors=runner.read(out/arm['id']/'provenance/config.json')['fixed_poses'],
            estimate=dict(draws=total),hard_region=dict(draws=total),independently_reconstructed_poses=total,populations=populations,
            importance_sampling=dict(guide_sha256=m['importance_guide_sha256'],gaussian_component_count=56,
                uniform_shell_probability=.5,density_measure=runner.DENSITY_MEASURE,draws=total,shell_rejected=48,
                branch_counts={'uniform-shell':total-400,'gaussian':400}))
        return protocol,arm,analysis,jobs

    def test_audit_exterior_zeros_and_every_population_are_bound(self):
        out=self.freeze();p,a,analysis,jobs=self.assessment(out);runner.verify_assessment(out,p,a,analysis,jobs)
        variants=[]
        x=copy.deepcopy(analysis);x['populations'].pop();variants.append(x)
        x=copy.deepcopy(analysis);x['hard_region']['draws']-=48;variants.append(x)
        x=copy.deepcopy(analysis);x['importance_sampling']['shell_rejected']=0;variants.append(x)
        x=copy.deepcopy(analysis);x['importance_sampling']['branch_counts']['gaussian']-=1;variants.append(x)
        x=copy.deepcopy(analysis);x['importance_sampling']['guide_sha256']='0'*64;variants.append(x)
        x=copy.deepcopy(analysis);x['populations'][0]['importance_sampling_audit']['shell_rejected']=13;variants.append(x)
        x=copy.deepcopy(analysis);x['populations'][0]['importance_sampling_audit']['gaussian_component_counts'][0]=-1;variants.append(x)
        x=copy.deepcopy(analysis);x['populations'][0]['importance_sampling_audit']['maximum_log_jacobian_error']=1e-6;variants.append(x)
        x=copy.deepcopy(analysis);x['populations'][0]['importance_sampling_audit']['branch_counts']['uniform-shell']=True;variants.append(x)
        x=copy.deepcopy(analysis);x['populations'][0]['id']='r01';variants.append(x)
        for i,x in enumerate(variants):
            with self.subTest(variant=i),self.assertRaises(ValueError):runner.verify_assessment(out,p,a,x,jobs)
        Path(jobs[0]['directory'],'samples.jsonl').write_text('changed after raw audit')
        with self.assertRaisesRegex(ValueError,'Output changed'):runner.verify_assessment(out,p,a,analysis,jobs)

    def test_all_eight_physical_jobs_precede_audits_and_completed_validation(self):
        out=self.freeze();events=[];original_verify=runner.verify_output
        def physical(jobs,snapshot,**kwargs):
            self.assertEqual(kwargs['workers'],8)
            for i,job in enumerate(jobs):self.population(out,i);job.update(status='complete',returncode=0)
            events.append('all-eight-drained');snapshot()
        def verify(*args):events.append('raw');return original_verify(*args)
        def audit(argv,**kwargs):
            self.assertEqual(events[:9],['all-eight-drained']+['raw']*8)
            base=Path(argv[-1]);index=0 if base.name=='bank' else 1;_,_,data,_=self.assessment(out,index)
            (base/'assessment').mkdir();runner.write(base/'assessment/analysis.json',data)
            self.assertTrue(all(kwargs['env'][key]=='1' for key in runner.THREAD_ENV))
            events.append(base.name);return subprocess.CompletedProcess(argv,0)
        with patch.object(runner,'execute_jobs',side_effect=physical),patch.object(runner,'verify_output',side_effect=verify),patch.object(runner.subprocess,'run',side_effect=audit):
            state=runner.run(out)
        self.assertTrue(state['complete']);self.assertEqual([x for x in events if x in ('bank','wide')],['bank','wide'])
        protocol,terminal=runner.validate_completed(out);self.assertEqual(terminal,state);self.assertEqual(len(protocol['jobs']),8)
        with self.assertRaisesRegex(ValueError,'no retries'):runner.run(out)
        state['jobs'][0]['output']['shell_rejected']=0;runner.write(out/'status.json',state)
        with self.assertRaisesRegex(ValueError,'Terminal output'):runner.validate_completed(out)

    def test_physical_validation_failure_never_launches_audit(self):
        out=self.freeze()
        def physical(jobs,snapshot,**kwargs):
            for job in jobs:job.update(status='complete',returncode=0)
        with (patch.object(runner,'execute_jobs',side_effect=physical),patch.object(runner,'verify_output',side_effect=ValueError('wrong shape')),
              patch.object(runner.subprocess,'run') as audit):
            with self.assertRaisesRegex(ValueError,'wrong shape'):runner.run(out)
        audit.assert_not_called();state=runner.read(out/'status.json');self.assertFalse(state['complete'])
        self.assertEqual(state['phase'],'physical_validation_failed')
        with self.assertRaisesRegex(ValueError,'no retries'):runner.preflight(out)

    def test_first_audit_failure_preserves_populations_and_does_not_launch_second(self):
        out=self.freeze()
        def physical(jobs,snapshot,**kwargs):
            for i,job in enumerate(jobs):self.population(out,i);job.update(status='complete',returncode=0)
        with patch.object(runner,'execute_jobs',side_effect=physical),patch.object(runner.subprocess,'run',return_value=subprocess.CompletedProcess(['audit'],9)) as audit:
            with self.assertRaises(subprocess.CalledProcessError):runner.run(out)
        self.assertEqual(audit.call_count,1);state=runner.read(out/'status.json')
        self.assertEqual(state['phase'],'audit_failed');self.assertFalse(state['complete'])
        self.assertTrue(all(j['status']=='complete' and Path(j['directory'],'samples.jsonl').exists() for j in state['jobs']))


class ReusedDrainTests(unittest.TestCase):
    def test_failed_spawn_drains_started_children_and_preserves_attempts(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);jobs=[dict(id=str(i),directory=str(root/f'population{i}'),log=str(root/f'{i}.log'),
                                         command=[str(i)],status='pending') for i in range(8)]
            started=[];waited=[]
            class Child:
                def __init__(self,i):self.i=i;self.pid=1000+i
                def poll(self):return None
                def wait(self):waited.append(self.i);return 7 if self.i==2 else 0
            def spawn(argv,**kwargs):
                i=int(argv[0])
                if i==3:raise OSError('synthetic launch failure')
                started.append(i);return Child(i)
            with self.assertRaisesRegex(OSError,'synthetic'):
                runner.execute_jobs(jobs,lambda:None,popen=spawn,pause=lambda _:None)
            self.assertEqual(started,[0,1,2]);self.assertEqual(waited,started)
            self.assertEqual(jobs[2]['status'],'failed');self.assertEqual(jobs[2]['returncode'],7)
            self.assertTrue(all(j['status']=='not_started' for j in jobs[3:]));self.assertTrue((root/'3.log').exists())


if __name__=='__main__':unittest.main()
