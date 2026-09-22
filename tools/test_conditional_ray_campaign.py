"""Synthetic campaign contracts and draining checks; never run physical sampling."""
import copy
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import run_conditional_ray_campaign as runner
import run_entry_shell_reference_campaign as entry
from test_entry_shell_reference_campaign import fixture as base_fixture


def fixture(root):
    repo, binary, bundle, package, _ = base_fixture(root)
    entries = []
    for group in range(8):
        for wi, width in enumerate((.02, .1, .5)):
            entries.append(dict(anchor_index=group//4, motif_id=7 if group < 4 else 4,
                moving_member_index=group % 4, width_A=width, entry=3*group+wi))
    runner.write(package/'construction.json', dict(entries=entries))
    (package/'freeze.json').unlink()
    runner.write(package/'freeze.json', dict(files=runner.file_hashes(package)))
    pins = {name: runner.sha(package/name) for name in entry.PINS}
    return repo, binary, bundle, package, pins


class ConditionalRayBindings(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo, self.binary, self.bundle, self.package, pins = fixture(self.root)
        for change in (patch.object(runner, 'ROOT', self.repo), patch.object(entry, 'PINS', pins)):
            change.start(); self.addCleanup(change.stop)

    def freeze(self):
        out = self.root/'campaign'
        runner.freeze(out, self.binary, self.bundle, self.package)
        return out

    def test_allocation_exact_guides_physical_region_and_archived_controller(self):
        before = runner.file_hashes(self.package)
        out = self.freeze(); protocol = runner.validate(out)
        self.assertEqual(protocol['total_unconditional_draws'], 786432)
        self.assertEqual(len(protocol['jobs']), 24)
        self.assertEqual([j['seed'] for j in protocol['jobs']], [131101010+1009*i for i in range(24)])
        self.assertEqual(sum(j['samples'] for j in protocol['jobs']), 786432)
        self.assertEqual(protocol['maximum_physical_workers'], 8)
        for spec, arm in zip(runner.ARMS, protocol['arms']):
            self.assertEqual({k:arm[k] for k in spec},spec)
            m = runner.read(out/arm['id']/'manifest.json'); a = out/arm['id']/'provenance'
            self.assertEqual(m['schema'], 'importance-latent-region-campaign-v3')
            self.assertEqual([j['samples'] for j in m['jobs']], [arm['samples']]*4)
            self.assertEqual((a/'region.json').read_bytes(), (self.package/'region.json').read_bytes())
            cfg = runner.read(a/'config.json'); original = runner.read(self.package/'config.json')
            self.assertEqual(cfg.pop('shape'), str(a/'shape.json')); original.pop('shape')
            self.assertEqual(cfg, original)
            guide = runner.read(a/'importance-guide.json')
            self.assertEqual(guide['defensive_uniform_shell_probability'], arm['alpha'])
            self.assertEqual(guide['widths'], [.02,.1,.5]); self.assertEqual(guide['inner_radius'], 2.)
            self.assertEqual([len(i['moving_members']) for i in guide['interfaces']], [4,4])
            for job in m['jobs']:
                self.assertEqual(job['command'], runner.command(out, arm, job))
                self.assertNotIn('--target-region', job['command'])
                self.assertEqual(job['command'][job['command'].index('--cloud-replicates')+1], '2')
        self.assertEqual(runner.file_hashes(self.package), before)
        self.assertIn('conditional_ray_proposal.py', protocol['python_sources'])
        self.assertIn('analyze_latent_region.py', protocol['python_sources'])
        with patch.object(runner, '__file__', str(out/'common/controller.py')):
            runner.validate(out); runner.preflight(out)

    def test_changed_source_package_or_binary_prevents_freeze(self):
        self.binary.write_bytes(b'not an embedded source bundle')
        with self.assertRaisesRegex(ValueError, 'embedded'): self.freeze()
        self.assertFalse((self.root/'campaign').exists())
        self.binary.write_bytes(self.bundle.read_bytes())
        (self.repo/'src/synthetic.rs').write_text('new code after binary was built')
        with self.assertRaisesRegex(ValueError, 'Reviewed source differs'): self.freeze()
        self.assertFalse((self.root/'campaign').exists())

    def test_old_package_geometry_cannot_be_silently_redefined(self):
        config = self.package/'config.json'; config.write_text(config.read_text()+' ')
        with self.assertRaisesRegex(ValueError, 'Selected preparation changed'): self.freeze()
        self.assertFalse((self.root/'campaign').exists())

    def test_current_source_mutations_are_irrelevant_archived_sources_are_bound(self):
        out=self.freeze(); (self.repo/'src/synthetic.rs').write_text('later development')
        runner.validate(out)
        (out/'common/conditional_ray_proposal.py').write_text('mutated observer')
        with self.assertRaisesRegex(ValueError, 'Frozen file changed'): runner.validate(out)

    def test_preflight_rejects_prior_launch_and_existing_artifacts(self):
        out=self.freeze(); runner.preflight(out)
        (out/'pilot_ray/logs/existing.log').write_text('preserve me')
        with self.assertRaisesRegex(ValueError, 'Existing outputs'): runner.preflight(out)
        (out/'pilot_ray/logs/existing.log').unlink()
        runner.write(out/'status.json', dict(complete=False, phase='physical_failed'))
        with self.assertRaisesRegex(ValueError, 'already launched'): runner.preflight(out)
        with self.assertRaisesRegex(ValueError, 'Fresh campaign'): self.freeze()

    def population(self, out, index=0):
        protocol = runner.read(out/'protocol.json'); job=protocol['jobs'][index]
        arm=next(a for a in protocol['arms'] if a['id']==job['arm'])
        m = runner.read(out/arm['id']/'manifest.json'); archive=out/arm['id']/'provenance'
        d=Path(job['directory']); (d/'provenance').mkdir(parents=True)
        pm=dict(schema=runner.POPULATION_SCHEMA, samples=job['samples'], seed=job['seed'], cloud_replicates=2,
            activity=.035, **{'lambda':.035*arm['lambda_ratio']}, lambda_ratio=arm['lambda_ratio'],
            guide_schema=runner.GUIDE_SCHEMA, proposal_kind=runner.PROPOSAL_KIND,
            importance_uniform_probability=arm['alpha'], importance_component_count=3,
            proposal_density_measure='Lebesgue measure in the original six-dimensional whitened region chart',
            executable_sha256=protocol['binary_sha256'], source_bundle_sha256=protocol['source_bundle_sha256'],
            minimum_latent_radius=0., latent_radius=4., minimum_original_q=0., maximum_original_q=None,
            minimum_original_q_inclusive=True, maximum_original_q_inclusive=True,
            physical_fixed_neighbors=runner.read(archive/'config.json')['fixed_poses'],
            chart_anchor=runner.read(archive/'region.json')['fixed_neighbor'])
        for key in ('region_sha256','shape_sha256','config_sha256','importance_guide_sha256'): pm[key]=m[key]
        for target, source in [('input-config.json','config.json'),('region.json','region.json'),('shape.json','shape.json'),
            ('importance-guide.json','importance-guide.json'),('source-bundle.json','source-bundle.json')]:
            shutil.copy2(archive/source,d/'provenance'/target)
        (d/'samples.jsonl').write_text('synthetic raw rows; no physical sampler in this test\n')
        summary=dict(complete=True,manifest=pm,samples=job['samples'],shell_rejected=0,
            samples_sha256=runner.sha(d/'samples.jsonl'),sampler_cpu_seconds=1.,
            estimates={name:dict(draws=job['samples']) for name in ('region','hard_region')})
        runner.write(d/'manifest.json',pm);runner.write(d/'summary.json',summary)
        return protocol, job, d, pm, summary

    def test_physical_population_metadata_is_bound_independently_of_raw_audit(self):
        out=self.freeze();protocol,job,d,pm,summary=self.population(out)
        result=runner.verify_output(out,protocol,job)
        self.assertEqual(result['samples_sha256'],runner.sha(d/'samples.jsonl'))
        variants=[('schema','importance-latent-region-normalizer-v2'),('seed',job['seed']+1),
            ('shape_sha256','0'*64),('cloud_replicates',1),('activity',.04),('lambda',.1),
            ('importance_uniform_probability',.5),('importance_component_count',2),
            ('proposal_density_measure','Haar'),('minimum_original_q_inclusive',False),
            ('maximum_original_q_inclusive',False),('minimum_original_q',1.),('chart_anchor',pm['physical_fixed_neighbors'][0])]
        for key,value in variants:
            changed=copy.deepcopy(pm); changed[key]=value; current=copy.deepcopy(summary);current['manifest']=changed
            runner.write(d/'manifest.json',changed);runner.write(d/'summary.json',current)
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'Population law'):
                runner.verify_output(out,protocol,job)

    def test_zero_draws_preserved_and_output_tampering_rejected(self):
        out=self.freeze();protocol,job,d,pm,summary=self.population(out)
        for key in ('region','hard_region'):
            current=copy.deepcopy(summary);current['estimates'][key]['draws']-=1
            runner.write(d/'summary.json',current)
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'denominator'):
                runner.verify_output(out,protocol,job)
        current=copy.deepcopy(summary);current['shell_rejected']=1;runner.write(d/'summary.json',current)
        with self.assertRaisesRegex(ValueError,'left R4'):runner.verify_output(out,protocol,job)
        runner.write(d/'summary.json',summary);(d/'samples.jsonl').write_text('mutated')
        with self.assertRaisesRegex(ValueError,'Rows changed'):runner.verify_output(out,protocol,job)

    def assessment(self, out, arm_index=1):
        protocol=runner.read(out/'protocol.json');arm=protocol['arms'][arm_index];total=4*arm['samples']
        jobs=copy.deepcopy(protocol['jobs']);populations=[];uniform=conditional=fallback=0
        for index in range(4*arm_index,4*arm_index+4):
            _,job,d,_,_=self.population(out,index);jobs[index]['output']=runner.verify_output(out,protocol,job)
            n=job['samples'];u=n if arm['alpha']==1. else 100;c=n-u;f=c//2
            populations.append(dict(id=job['id'],seed=job['seed'],estimate=dict(draws=n),hard_region=dict(draws=n),
                samples_sha256=jobs[index]['output']['samples_sha256'],importance_sampling_audit=dict(draws=n,
                    shell_rejected=0,branch_counts={'uniform-shell':u,'conditional-ray':c},
                    ray_component_counts=[c,0,0],ray_fallback_counts=[f,0,0])))
            uniform+=u;conditional+=c;fallback+=f
        manifest=runner.read(out/arm['id']/'manifest.json')
        analysis=dict(region_sha256=protocol['region_sha256'],
            physical_fixed_neighbors=runner.read(out/arm['id']/'provenance/config.json')['fixed_poses'],
            estimate=dict(draws=total),hard_region=dict(draws=total),independently_reconstructed_poses=total,
            populations=populations,importance_sampling=dict(guide_sha256=manifest['importance_guide_sha256'],
                conditional_ray_component_count=3,uniform_shell_probability=arm['alpha'],guide_schema=runner.GUIDE_SCHEMA,
                proposal_kind=runner.PROPOSAL_KIND,density_measure=runner.DENSITY_MEASURE,shell_rejected=0,draws=total,
                branch_counts={'uniform-shell':uniform,'conditional-ray':conditional},ray_fallback_counts=[fallback,0,0]))
        return protocol,arm,analysis,jobs

    def test_independent_audit_all_rows_components_and_identity_required(self):
        out=self.freeze();protocol,arm,analysis,jobs=self.assessment(out)
        runner.verify_assessment(out,protocol,arm,analysis,jobs)
        for mode in ('region','neighbor','guide_sha','density','hard_draws','duplicate','fallback','branch_sum','negative','boolean'):
            changed=copy.deepcopy(analysis)
            if mode=='region':changed['region_sha256']='0'*64
            elif mode=='neighbor':changed['physical_fixed_neighbors']=changed['physical_fixed_neighbors'][:1]
            elif mode=='guide_sha':changed['importance_sampling']['guide_sha256']='0'*64
            elif mode=='density':changed['importance_sampling']['density_measure']='Haar'
            elif mode=='hard_draws':changed['populations'][0]['hard_region']['draws']-=1
            elif mode=='duplicate':changed['populations'][0]=changed['populations'][1]
            elif mode=='fallback':changed['importance_sampling']['ray_fallback_counts'][0]-=1
            elif mode=='branch_sum':
                changed['importance_sampling']['branch_counts']['uniform-shell']+=1
                changed['importance_sampling']['branch_counts']['conditional-ray']-=1
            elif mode=='negative':
                audit=changed['populations'][0]['importance_sampling_audit'];n=arm['samples']
                audit['branch_counts']={'uniform-shell':-1,'conditional-ray':n+1};audit['ray_component_counts']=[n+1,0,0]
            elif mode=='boolean':
                audit=changed['populations'][0]['importance_sampling_audit'];n=arm['samples']
                audit['branch_counts']={'uniform-shell':True,'conditional-ray':n-1};audit['ray_component_counts']=[n-1,0,0]
            with self.subTest(mode=mode),self.assertRaises(ValueError):
                runner.verify_assessment(out,protocol,arm,changed,jobs)
        first=next(j for j in jobs if j['arm']==arm['id'])
        Path(first['directory'],'samples.jsonl').write_text('mutated after raw audit')
        with self.assertRaisesRegex(ValueError,'Output changed'):runner.verify_assessment(out,protocol,arm,analysis,jobs)

    def test_physical_failure_is_terminal_drained_and_skips_all_audits(self):
        out=self.freeze()
        def fail(jobs,snapshot):
            for i,j in enumerate(jobs):j.update(status='failed' if i==1 else 'complete' if i<8 else 'not_started')
            snapshot();raise RuntimeError('physical failed after draining')
        with patch.object(runner,'execute_jobs',side_effect=fail),patch.object(runner.subprocess,'run') as audit:
            with self.assertRaisesRegex(RuntimeError,'after draining'):runner.run(out)
            audit.assert_not_called()
        state=runner.read(out/'status.json');self.assertFalse(state['complete']);self.assertEqual(state['phase'],'physical_failed')
        self.assertTrue(all(j['status']=='not_started' for j in state['jobs'][8:]))
        with self.assertRaisesRegex(ValueError,'already launched'):runner.run(out)

    def test_all_physics_validate_before_audits_and_audit_failure_is_terminal(self):
        out=self.freeze();order=[]
        def complete(jobs,snapshot):
            for j in jobs:j.update(status='complete',returncode=0)
            order.append('drained');snapshot()
        def verify(*args):order.append('validated');return dict(samples_sha256='rows')
        def audit(argv,**kwargs):order.append('audit');return subprocess.CompletedProcess(argv,3)
        with patch.object(runner,'execute_jobs',side_effect=complete),patch.object(runner,'verify_output',side_effect=verify),\
             patch.object(runner.subprocess,'run',side_effect=audit):
            with self.assertRaises(subprocess.CalledProcessError):runner.run(out)
        self.assertEqual(order,['drained']+['validated']*24+['audit'])
        state=runner.read(out/'status.json');self.assertFalse(state['complete']);self.assertEqual(state['phase'],'audit_failed')
        self.assertEqual(state['audits']['pilot_uniform']['returncode'],3)
        self.assertTrue(all(j['status']=='complete' for j in state['jobs']))

    def test_complete_only_after_six_distinct_audits_and_assessments(self):
        out=self.freeze();order=[]
        def complete(jobs,snapshot):
            for j in jobs:j.update(status='complete',returncode=0)
            order.append('drained');snapshot()
        def verify(*args):order.append('validated');return dict(samples_sha256='rows')
        def audit(argv,**kwargs):
            base=Path(argv[argv.index('--root')+1]);order.append(('audit',base.name))
            (base/'assessment').mkdir();runner.write(base/'assessment/analysis.json',{})
            return subprocess.CompletedProcess(argv,0)
        def assess(out,protocol,arm,analysis,jobs):order.append(('assessed',arm['id']))
        with patch.object(runner,'execute_jobs',side_effect=complete),patch.object(runner,'verify_output',side_effect=verify),\
             patch.object(runner.subprocess,'run',side_effect=audit),patch.object(runner,'verify_assessment',side_effect=assess):
            state=runner.run(out)
        self.assertTrue(state['complete']);self.assertEqual(state['phase'],'complete')
        self.assertEqual(order[:25],['drained']+['validated']*24)
        self.assertEqual(order[25:],[event for arm in runner.ARMS for event in [('audit',arm['id']),('assessed',arm['id'])]])
        self.assertEqual(len(state['audits']),6)


class ConditionalRayLifecycle(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)

    def jobs(self,n=24):
        return [dict(id=str(i),directory=str(self.root/f'population{i}'),log=str(self.root/f'{i}.log'),
            command=[str(i)],status='pending') for i in range(n)]

    def child_factory(self,failed_exit=None,failed_launch=None,blocked=(),wait_interrupt=None):
        started=[];waited=[];peak=[0];interrupts=[]
        class Child:
            def __init__(self,i):self.i=i;self.pid=1000+i
            def poll(self):return None if self.i in blocked else (7 if self.i==failed_exit else 0)
            def wait(self):
                if self.i==wait_interrupt and self.i not in interrupts:
                    interrupts.append(self.i);raise KeyboardInterrupt('drain interrupted')
                waited.append(self.i);return 7 if self.i==failed_exit else 0
        def spawn(argv,**kwargs):
            i=int(argv[0])
            if i==failed_launch:raise OSError('launch failed')
            started.append(i);peak[0]=max(peak[0],len(started)-len(waited));return Child(i)
        return spawn,started,waited,peak

    def test_worker_cap_success_and_every_started_child_waited(self):
        jobs=self.jobs();spawn,started,waited,peak=self.child_factory();snapshots=[]
        runner.execute_jobs(jobs,lambda:snapshots.append(copy.deepcopy(jobs)),popen=spawn,pause=lambda _:None)
        self.assertEqual(started,waited);self.assertEqual(len(started),24);self.assertEqual(peak[0],8)
        self.assertTrue(all(j['status']=='complete' and j['returncode']==0 for j in jobs))
        for count in (0,9,True,2.5):
            with self.assertRaises(ValueError):runner.execute_jobs([],lambda:None,workers=count,popen=spawn)

    def test_nonzero_exit_drains_active_children_and_does_not_launch_next_arm(self):
        jobs=self.jobs();spawn,started,waited,_=self.child_factory(failed_exit=1,blocked=(2,3,4,5,6,7))
        with self.assertRaisesRegex(RuntimeError,'Physical failure'):
            runner.execute_jobs(jobs,lambda:None,popen=spawn,pause=lambda _:None)
        self.assertEqual(started,list(range(8)));self.assertEqual(set(started),set(waited))
        self.assertEqual(jobs[1]['status'],'failed');self.assertEqual(jobs[1]['returncode'],7)
        self.assertTrue(all(j['status']=='not_started' for j in jobs[8:]))

    def test_launch_failure_drains_then_marks_every_unlaunched_population(self):
        jobs=self.jobs();spawn,started,waited,_=self.child_factory(failed_launch=3,failed_exit=2,blocked=(0,1,2))
        with self.assertRaisesRegex(OSError,'launch failed'):
            runner.execute_jobs(jobs,lambda:None,popen=spawn,pause=lambda _:None)
        self.assertEqual(started,[0,1,2]);self.assertEqual(started,waited)
        self.assertEqual(jobs[2]['returncode'],7);self.assertEqual(jobs[2]['status'],'failed')
        self.assertTrue(all(j['status']=='not_started' for j in jobs[3:]))
        self.assertTrue(Path(jobs[3]['log']).exists(),'preserve attempted-launch log')

    def test_keyboard_interrupt_drains_even_when_wait_is_interrupted(self):
        jobs=self.jobs();spawn,started,waited,_=self.child_factory(blocked=tuple(range(8)),wait_interrupt=3)
        def stop(_):raise KeyboardInterrupt('observation interrupted')
        with self.assertRaises(KeyboardInterrupt):runner.execute_jobs(jobs,lambda:None,popen=spawn,pause=stop)
        self.assertEqual(started,list(range(8)));self.assertEqual(started,waited)
        self.assertTrue(all(j['status']=='complete' for j in jobs[:8]))
        self.assertTrue(all(j['status']=='not_started' for j in jobs[8:]))

    def test_existing_population_never_overwritten_and_started_child_drains(self):
        jobs=self.jobs();Path(jobs[2]['directory']).mkdir();spawn,started,waited,_=self.child_factory()
        with self.assertRaisesRegex(ValueError,'Existing population'):
            runner.execute_jobs(jobs,lambda:None,popen=spawn,pause=lambda _:None)
        self.assertEqual(started,[0,1]);self.assertEqual(waited,started)
        self.assertTrue(Path(jobs[2]['directory']).is_dir())
        self.assertTrue(all(j['status']=='not_started' for j in jobs[2:]))


if __name__=='__main__':unittest.main()
