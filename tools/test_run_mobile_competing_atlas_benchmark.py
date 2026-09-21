"""Mocked eight-job lifecycle controls; never execute a physical binary."""
import subprocess
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import run_mobile_competing_atlas_benchmark as controller


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
        for variant in ('legacy','augmented'):
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
        for variant in ('legacy','augmented'):
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
        self.assertTrue(controller.read(self.benchmark/'augmented/status.json')['complete'])

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
        for variant in ('legacy','augmented'):self.assertTrue(controller.read(self.benchmark/variant/'status.json')['complete'])
        self.assertEqual(controller.read(self.journal/'status.json')['phase'],'audit_failed')

    def test_reservation_failure_launches_nothing(self):
        # Simulate a status file appearing after preflight but before reservation.
        controller.write(self.benchmark/'augmented/status.json',{'owner':'other-controller'})
        with patch.object(controller.subprocess,'Popen') as launches,patch.object(controller.subprocess,'run') as audits:
            with self.assertRaises(FileExistsError):controller.run(self.benchmark,self.journal,self.assessment)
        launches.assert_not_called();audits.assert_not_called()
        self.assertEqual(controller.read(self.benchmark/'augmented/status.json'),{'owner':'other-controller'})
        self.assertFalse(controller.read(self.benchmark/'legacy/status.json')['running'])
        self.assertEqual(controller.read(self.journal/'status.json')['phase'],'reservation_failed')


if __name__=='__main__':unittest.main()
