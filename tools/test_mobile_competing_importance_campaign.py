"""Fixed-design and mocked lifecycle controls; never launch a physical kernel."""
import copy
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import prepare_mobile_competing_importance_campaign as preparation
import run_mobile_competing_importance_campaign as controller


class Child:
    def __init__(self,index,code=0):self.pid,self.code,self.waits=10000+index,code,0
    def poll(self):return self.code
    def wait(self):self.waits+=1;return self.code


class ImportanceControllerTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory(prefix='guided-controller-test-');self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name);self.campaign=self.root/'campaign';self.campaign.mkdir()
        self.journal=self.root/'journal';self.children=[];self.checked=[]
        self.protocol=dict(campaigns=[])
        for shell_index,shell in enumerate(preparation.SHELLS):
            folder=self.campaign/shell;archive=folder/'provenance'
            for name in ('runs','logs','provenance'):(folder/name).mkdir(parents=True)
            for name in ('latent-region-normalizer','config.json','region.json','importance-guide.json','shape.json',preparation.ANALYZER_NAME):
                (archive/name).write_text('not an executable; mocks only')
            jobs=preparation.jobs_for(folder,shell_index)
            manifest=dict(jobs=jobs,archive_sha256={p.name:preparation.sha(p) for p in archive.iterdir()},
                observer_sha256=preparation.sha(archive/preparation.ANALYZER_NAME))
            preparation.write(folder/'manifest.json',manifest)
            self.protocol['campaigns'].append(dict(region=shell,path=str(folder),manifest_sha256=preparation.sha(folder/'manifest.json')))
        preparation.write(self.campaign/'protocol.json',self.protocol)
        self.addCleanup(patch.stopall)
        patch.object(controller,'print',create=True).start()
        patch.object(controller,'check',return_value=self.protocol).start()
        patch.object(controller,'validate',return_value=self.protocol).start()
        patch.object(controller,'verify_population',side_effect=self.verify_population).start()
        patch.object(controller,'verify_assessment').start()

    def verify_population(self,folder,manifest,job):
        self.checked.append(job['id'])
        return dict(samples_sha256='synthetic',summary_sha256='synthetic',manifest_sha256='synthetic',sampler_cpu_seconds=1.)

    def start(self,*args,**kwargs):
        child=Child(len(self.children));self.children.append(child);return child

    def audit(self,argv,**kwargs):
        self.assertEqual(len(self.children),8);self.assertEqual(len(self.checked),8)
        self.assertTrue(all(child.waits==1 for child in self.children))
        folder=Path(argv[argv.index('--root')+1]);(folder/'assessment').mkdir()
        preparation.write(folder/'assessment/analysis.json',dict(synthetic=True))
        return subprocess.CompletedProcess(argv,0)

    def test_all_physics_drains_and_validates_before_single_audit_per_shell(self):
        with patch.object(controller.subprocess,'Popen',side_effect=self.start) as launches,patch.object(controller.subprocess,'run',side_effect=self.audit) as audits:
            result=controller.run(self.campaign,self.journal)
        self.assertTrue(result['complete']);self.assertEqual(result['phase'],'complete')
        self.assertEqual((launches.call_count,audits.call_count),(8,2))
        self.assertTrue(all(a['returncode']==0 and len(a['analysis_sha256'])==64 for a in result['audits'].values()))
        for entry in self.protocol['campaigns']:
            folder=Path(entry['path'])
            for replicate in range(4):
                terminal=preparation.read(folder/f'r{replicate:02}-status.json')
                self.assertEqual(terminal['returncode'],0);self.assertEqual(terminal['samples'],8192)

    def test_nonzero_exit_drains_children_preserves_failure_and_skips_audits(self):
        def start(*args,**kwargs):
            child=self.start()
            if len(self.children)==2:child.code=7
            return child
        with patch.object(controller.subprocess,'Popen',side_effect=start),patch.object(controller.subprocess,'run') as audits:
            with self.assertRaisesRegex(RuntimeError,'physical job failed'):controller.run(self.campaign,self.journal)
        self.assertEqual(len(self.children),8);self.assertTrue(all(c.waits==1 for c in self.children))
        audits.assert_not_called();self.assertFalse(self.checked)
        state=preparation.read(self.campaign/'status.json');self.assertFalse(state['complete'])
        self.assertEqual(state['phase'],'physical_failed');self.assertEqual(state['jobs'][1]['exit_code'],7)

    def test_launch_failure_drains_started_children_without_retry(self):
        attempts=[]
        def start(*args,**kwargs):
            attempts.append(1)
            if len(attempts)==3:raise OSError('synthetic launch error')
            return self.start()
        with patch.object(controller.subprocess,'Popen',side_effect=start),patch.object(controller.subprocess,'run') as audits:
            with self.assertRaisesRegex(OSError,'synthetic launch error'):controller.run(self.campaign,self.journal)
        self.assertEqual(len(attempts),3);self.assertEqual(len(self.children),2)
        self.assertTrue(all(c.waits==1 for c in self.children));audits.assert_not_called()
        jobs=preparation.read(self.campaign/'status.json')['jobs']
        self.assertEqual(sum(j['status']=='launch_failed' for j in jobs),1)
        self.assertEqual(sum(j['status']=='not_started' for j in jobs),5)

    def test_physical_output_validation_failure_forbids_any_audit(self):
        with (patch.object(controller.subprocess,'Popen',side_effect=self.start),
              patch.object(controller,'verify_population',side_effect=ValueError('wrong guide')),
              patch.object(controller.subprocess,'run') as audits):
            with self.assertRaisesRegex(ValueError,'wrong guide'):controller.run(self.campaign,self.journal)
        self.assertTrue(all(c.waits==1 for c in self.children));audits.assert_not_called()
        self.assertEqual(preparation.read(self.campaign/'status.json')['phase'],'physical_validation_failed')

    def test_audit_failure_preserves_physical_success_without_retry(self):
        with patch.object(controller.subprocess,'Popen',side_effect=self.start),patch.object(controller.subprocess,'run',return_value=subprocess.CompletedProcess(['fake'],4)) as audits:
            with self.assertRaises(subprocess.CalledProcessError):controller.run(self.campaign,self.journal)
        self.assertEqual(audits.call_count,1);self.assertTrue(all(c.waits==1 for c in self.children))
        state=preparation.read(self.campaign/'status.json');self.assertEqual(state['phase'],'audit_failed')
        self.assertTrue(all(j['status']=='complete' for j in state['jobs']))
        self.assertEqual(next(iter(state['audits'].values()))['returncode'],4)

    def test_reservation_collision_does_not_launch_or_overwrite(self):
        other=self.campaign/'competitor-shell-8-12/status.json';preparation.write(other,dict(owner='other'))
        with patch.object(controller.subprocess,'Popen') as launches,patch.object(controller.subprocess,'run') as audits:
            with self.assertRaises(FileExistsError):controller.run(self.campaign,self.journal)
        launches.assert_not_called();audits.assert_not_called();self.assertEqual(preparation.read(other),dict(owner='other'))
        self.assertEqual(preparation.read(self.campaign/'status.json')['phase'],'reservation_failed')


class ImportanceDesignTests(unittest.TestCase):
    def test_exact_eight_fresh_streams_and_cli_proposal_binding(self):
        jobs=[j for index,shell in enumerate(preparation.SHELLS) for j in preparation.jobs_for(Path('/unused')/shell,index)]
        self.assertEqual([j['seed'] for j in jobs],[120501010+1009*j for j in range(8)])
        for job in jobs:
            argv=job['command'];self.assertEqual(job['samples'],8192)
            self.assertEqual(argv[argv.index('--samples')+1],'8192')
            self.assertEqual(argv[argv.index('--lambda-ratio')+1],'64.0')
            self.assertEqual(argv[argv.index('--cloud-replicates')+1],'2')
            self.assertTrue(argv[argv.index('--importance-guide')+1].endswith('/provenance/importance-guide.json'))
            self.assertEqual(argv[argv.index('--out')+1],job['directory'])

    def test_relocation_changes_only_shape_path(self):
        original=dict(shape='/source/shape.json',fixed_poses=[{'position':[1.,2.,3.]}],capture_radius=170.,reservoir_density=.035,
            depletant_radius=1.5,metadata={'frozen':'scaffold'},seed=19)
        changed=preparation.config_for(original,Path('/archive'))
        self.assertEqual(changed['shape'],'/archive/shape.json');changed['shape']=original['shape']
        self.assertEqual(changed,original);self.assertEqual(original['shape'],'/source/shape.json')

    def test_scaffold_shell_and_guide_law_cannot_change(self):
        shell='competitor-shell-5-8';identity=dict(position=[0.,0.,0.],orientation=[1.,0.,0.,0.])
        config=dict(fixed_poses=[identity],metadata={},reservoir_density=.035,depletant_radius=1.5,capture_center=[0.,0.,0.],capture_radius=170.)
        region=dict(minimum_mahalanobis_radius=5.,mahalanobis_radius=8.,minimum_original_q=1.,minimum_original_q_inclusive=False,
            physical_fixed_neighbors=[identity],fixed_neighbor=identity,physical_metric={},activity=.035,depletant_radius=1.5,
            capture_center=[0.,0.,0.],capture_radius=170.,shape_sha256=preparation.SHAPE_SHA,gaussian_chart={'shape_sha256':preparation.SHAPE_SHA})
        component=dict(weight=1.,mean=[0.]*6,covariance=np.eye(6).tolist())
        guide=dict(schema='defensive-latent-shell-guide-v1',region_sha256=preparation.SHELLS[shell]['region_sha256'],
            defensive_uniform_shell_probability=.5,gaussian_components=[copy.deepcopy(component) for _ in range(32)])
        preparation.check_physics(config,region,guide,shell)
        for field,value in [('mahalanobis_radius',9.),('minimum_original_q_inclusive',True),('physical_fixed_neighbors',[]),('capture_radius',171.)]:
            bad=copy.deepcopy(region);bad[field]=value
            with self.assertRaises((ValueError,AssertionError)):preparation.check_physics(config,bad,guide,shell)
        bad=copy.deepcopy(guide);bad['defensive_uniform_shell_probability']=.4
        with self.assertRaises(ValueError):preparation.check_physics(config,region,bad,shell)


if __name__=='__main__':unittest.main()
