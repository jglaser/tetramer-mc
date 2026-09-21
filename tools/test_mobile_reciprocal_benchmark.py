"""Mocked eight-job lifecycle controls; never execute a physical binary."""
import subprocess
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import run_mobile_reciprocal_benchmark as controller


class Child:
    def __init__(self,index,code=0):self.pid,self.code,self.waits=9000+index,code,0
    def poll(self):return self.code
    def wait(self):self.waits+=1;return self.code


class MatchedControllerTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='matched-controller-test-');self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.benchmark=self.root/'benchmark';self.benchmark.mkdir()
        self.journal=self.root/'journal';self.assessment=self.root/'assessment'
        self.children=[];self.protocol=dict(campaigns=[],total_jobs=8)
        self.binary=self.root/'fake-binary';self.binary.write_text('Never execute')
        for variant in ('legacy','reciprocal'):
            folder=self.benchmark/variant
            for name in ('configs','runs','logs','provenance'):(folder/name).mkdir(parents=True)
            model=folder/'provenance/model.json';controller.write(model,{'synthetic':variant})
            jobs=[]
            for index in range(4):
                cfg=folder/'configs'/f'job{index}.json';controller.write(cfg,{'synthetic':index})
                jobs.append(dict(id=f'job{index}',mode='c0' if index%2==0 else 'c09',replicate=index//2,start='competing',
                    seed=index+(0 if variant=='legacy' else 4),config=str(cfg),config_sha256=controller.sha(cfg),
                    directory=str(folder/'runs'/f'job{index}'),log=str(folder/'logs'/f'job{index}.log'),command=[str(self.binary),'synthetic']))
            controller.write(folder/'manifest.json',dict(jobs=jobs,binary_sha256=controller.sha(self.binary),
                model=str(model),model_sha256=controller.sha(model),observer_sha256='synthetic-observer',reference=str(folder/'provenance/reference')))
            self.protocol['campaigns'].append(dict(atlas=variant,path=str(folder)))
        controller.write(self.benchmark/'protocol.json',self.protocol)
        self.addCleanup(patch.stopall)
        patch.object(controller,'print',create=True).start()
        patch.object(controller,'check',return_value=self.protocol).start()
        patch.object(controller,'validate',return_value=self.protocol).start()

    def start(self,argv,**kwargs):
        child=Child(len(self.children));self.children.append(child);return child

    def audit(self,argv,**kwargs):
        self.assertEqual(len(self.children),8)
        self.assertTrue(all(child.waits==1 for child in self.children))
        self.assertIn('-B',argv)
        campaign=Path(argv[argv.index('--campaign')+1]);destination=Path(argv[argv.index('--out')+1]);destination.mkdir()
        manifest=controller.read(campaign/'manifest.json')
        rows=[dict(passed=True,**{key:job[key] for key in ('id','mode','start','replicate','seed')}) for job in manifest['jobs']]
        controller.write(destination/'analysis.json',dict(complete=True,runs=rows,
            manifest_sha256=controller.sha(campaign/'manifest.json'),terminal_status_sha256=controller.sha(campaign/'status.json'),
            analyzer_sha256=manifest['observer_sha256']))
        return subprocess.CompletedProcess(argv,0)

    def test_all_eight_children_finish_before_either_audit(self):
        with patch.object(controller.subprocess,'Popen',side_effect=self.start) as launches,patch.object(controller.subprocess,'run',side_effect=self.audit) as audits:
            result=controller.run(self.benchmark,self.journal,self.assessment)
        self.assertTrue(result['complete']);self.assertEqual((launches.call_count,audits.call_count),(8,2))
        for variant in ('legacy','reciprocal'):
            status=controller.read(self.benchmark/variant/'status.json');self.assertTrue(status['complete']);self.assertFalse(status['running'])

    def test_nonzero_child_drains_every_started_child_and_skips_audits(self):
        def start(*args,**kwargs):
            child=self.start(*args,**kwargs)
            if len(self.children)==3:child.code=7
            return child
        with patch.object(controller.subprocess,'Popen',side_effect=start),patch.object(controller.subprocess,'run') as audits:
            with self.assertRaisesRegex(RuntimeError,'physical job failed'):controller.run(self.benchmark,self.journal,self.assessment)
        self.assertEqual(len(self.children),8);self.assertTrue(all(child.waits==1 for child in self.children));audits.assert_not_called()
        self.assertEqual(controller.read(self.journal/'status.json')['phase'],'physical_failed')
        self.assertFalse(controller.read(self.benchmark/'legacy/status.json')['complete'])
        self.assertTrue(controller.read(self.benchmark/'reciprocal/status.json')['complete'])

    def test_launch_failure_drains_started_children_without_retry(self):
        attempts=[]
        def start(*args,**kwargs):
            attempts.append(1)
            if len(attempts)==3:raise OSError('synthetic launch failure')
            return self.start(*args,**kwargs)
        with patch.object(controller.subprocess,'Popen',side_effect=start),patch.object(controller.subprocess,'run') as audits:
            with self.assertRaisesRegex(OSError,'synthetic launch failure'):controller.run(self.benchmark,self.journal,self.assessment)
        self.assertEqual(len(attempts),3);self.assertEqual(len(self.children),2)
        self.assertTrue(all(child.waits==1 for child in self.children));audits.assert_not_called()
        rows=controller.read(self.journal/'status.json')['jobs']
        self.assertEqual(sum(r['status']=='launch_failed' for r in rows),1)
        self.assertEqual(sum(r['status']=='not_started' for r in rows),5)

    def test_audit_failure_preserves_terminal_physical_success(self):
        with patch.object(controller.subprocess,'Popen',side_effect=self.start),patch.object(controller.subprocess,'run',return_value=subprocess.CompletedProcess(['fake'],3)) as audits:
            with self.assertRaises(subprocess.CalledProcessError):controller.run(self.benchmark,self.journal,self.assessment)
        self.assertEqual(audits.call_count,1)
        self.assertTrue(all(child.waits==1 for child in self.children))
        for variant in ('legacy','reciprocal'):self.assertTrue(controller.read(self.benchmark/variant/'status.json')['complete'])
        self.assertEqual(controller.read(self.journal/'status.json')['phase'],'audit_failed')

    def test_reservation_failure_launches_nothing(self):
        # Simulate a status file appearing after preflight but before reservation.
        controller.write(self.benchmark/'reciprocal/status.json',{'owner':'other-controller'})
        with patch.object(controller.subprocess,'Popen') as launches,patch.object(controller.subprocess,'run') as audits:
            with self.assertRaises(FileExistsError):controller.run(self.benchmark,self.journal,self.assessment)
        launches.assert_not_called();audits.assert_not_called()
        self.assertEqual(controller.read(self.benchmark/'reciprocal/status.json'),{'owner':'other-controller'})
        self.assertFalse(controller.read(self.benchmark/'legacy/status.json')['running'])
        self.assertEqual(controller.read(self.journal/'status.json')['phase'],'reservation_failed')




class ReciprocalPreparationTests(unittest.TestCase):
    def test_envelope_retains_unchanged_base_and_exact_density(self):
        from prepare_mobile_reciprocal_benchmark import reciprocal_envelope, density_preflight
        from test_reciprocal_pose_density import fixture
        import copy
        base=fixture();unchanged=copy.deepcopy(base);wrapped=reciprocal_envelope(base)
        poses=[dict(position=[i,.1*i,-.2*i],orientation=[1.,0.,0.,0.]) for i in range(3)]
        result=density_preflight(base,wrapped,poses)
        self.assertEqual(base,unchanged);self.assertEqual(wrapped['base_model'],base)
        self.assertEqual(wrapped['reciprocal_components'],[True]*3)
        self.assertTrue(result['passed']);self.assertEqual(len(result['physical_pair_proposal_densities']),6)
        self.assertLess(result['maximum_log_density_identity_error'],1e-10)
        self.assertEqual((result['physical_updates'],result['bath_clouds']),(0,0))

    def test_preparation_configs_preserve_physics_and_use_fresh_seeds(self):
        from prepare_mobile_reciprocal_benchmark import configured
        source=dict(seed=1,shape='a',monomer_shape='b',initial_poses=[{'synthetic':1}],
            boundary=dict(kind='spherical',radius=3),fixed_body_indices=[],seed_labels=[],
            depletant_radius=1.5,reservoir_density=.035,global_probability=.5,
            learned_uniform_weight=.1,gca_probability=1.,center_shift_probability=1.,
            frozen_posterior=dict(probability=.5,correlation=0.),metadata={})
        seeds=[]
        for variant in ('legacy','reciprocal'):
            for mode in ('c0','c09'):
                cfg=configured(source,variant,mode,Path('/unused/provenance'))
                unchanged=lambda d:{k:v for k,v in d.items() if k not in ('seed','shape','monomer_shape','metadata')}
                self.assertEqual(unchanged(cfg),unchanged(source))
                self.assertEqual(cfg['metadata']['added_contact_charts'],0)
                seeds.extend([cfg['seed'],cfg['seed']+4036])
        self.assertEqual(set(seeds),{119201010+1009*j for j in range(8)})
        self.assertEqual(source['metadata'],{})

    def test_invalid_or_nested_base_is_rejected(self):
        from prepare_mobile_reciprocal_benchmark import reciprocal_envelope
        from test_reciprocal_pose_density import fixture
        base=fixture();base['coordinate_convention']='laboratory'
        with self.assertRaises(ValueError):reciprocal_envelope(base)

    def test_observer_accepts_only_declared_reciprocal_grid(self):
        from analyze_mobile_posterior_pilot import validate_campaign_jobs
        import copy
        jobs=[dict(id=f'{mode}-{rep}',start='competing',mode=mode,replicate=rep)
            for mode in ('c0','c09') for rep in (0,1)]
        status=dict(complete=True,running=False,jobs=[dict(id=j['id'],status='complete',exit_code=0) for j in jobs])
        for variant in ('legacy','reciprocal'):
            manifest=dict(schema='mobile-reciprocal-atlas-benchmark-v1',atlas_variant=variant,jobs=jobs)
            self.assertEqual(validate_campaign_jobs(manifest,status),jobs)
            for changed in ('variant','schema','grid','status'):
                m,s=copy.deepcopy(manifest),copy.deepcopy(status)
                if changed=='variant':m['atlas_variant']='augmented'
                elif changed=='schema':m['schema']='unknown'
                elif changed=='grid':m['jobs'][0]['mode']='capture_only'
                else:s['jobs'][0]['exit_code']=1
                with self.assertRaises(AssertionError):validate_campaign_jobs(m,s)
        old=dict(schema='mobile-competing-atlas-benchmark-v1',atlas_variant='augmented',jobs=jobs)
        self.assertEqual(validate_campaign_jobs(old,status),jobs)
        old['atlas_variant']='reciprocal'
        with self.assertRaises(AssertionError):validate_campaign_jobs(old,status)


if __name__=='__main__':unittest.main()
