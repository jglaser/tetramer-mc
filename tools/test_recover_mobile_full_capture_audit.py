import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import recover_mobile_full_capture_audit as recovery


class RecoveryBindings(unittest.TestCase):
    def fixture(self,root):
        protocol=dict(campaigns=[]);status=dict(phase='audit_failed',complete=False,jobs=[],audits={'epsilon-0p1':dict(returncode=1)})
        for a in ('epsilon-0p1','epsilon-0p5'):
            folder=root/a;folder.mkdir();jobs=[]
            for i in range(4):
                directory=folder/'runs'/f'r{i:02}';directory.mkdir(parents=True)
                pm=dict(samples=8192,seed=10*len(protocol['campaigns'])+i)
                summary=dict(complete=True,manifest=pm)
                recovery.write(directory/'manifest.json',pm);recovery.write(directory/'summary.json',summary)
                (directory/'samples.jsonl').write_text('immutable saved rows\n')
                job=dict(id=f'r{i:02}',seed=pm['seed'],samples=8192,directory=str(directory));jobs.append(job)
                status['jobs'].append(dict(arm=a,id=job['id'],status='complete',exit_code=0,output={
                    key:recovery.sha(directory/name)for name,key in [('samples.jsonl','samples_sha256'),('summary.json','summary_sha256'),('manifest.json','manifest_sha256')]}))
            recovery.write(folder/'manifest.json',dict(jobs=jobs))
            protocol['campaigns'].append(dict(arm=a,path=str(folder),manifest_sha256=recovery.sha(folder/'manifest.json')))
        recovery.write(root/'protocol.json',protocol);digest=recovery.sha(root/'protocol.json');status['protocol_sha256']=digest
        recovery.write(root/'status.json',status)
        recovery.write(root/'freeze.json',dict(files={'protocol.json':digest}))
        return digest,protocol,status

    def test_original_outputs_and_failed_audit_are_bound_without_rewriting(self):
        with tempfile.TemporaryDirectory()as directory:
            root=Path(directory);digest,p,s=self.fixture(root)
            before=(root/'status.json').read_bytes()
            with patch.object(recovery,'PROTOCOL_SHA',digest):
                result=recovery.original_snapshot(root)
                self.assertEqual(len(result['physical_output_sha256']),24)
                self.assertEqual(before,(root/'status.json').read_bytes())
                (root/'epsilon-0p1/runs/r00/samples.jsonl').write_text('changed')
                with self.assertRaisesRegex(ValueError,'Physical output changed'):recovery.original_snapshot(root)

    def test_incomplete_physical_or_different_observer_history_refuses_recovery(self):
        with tempfile.TemporaryDirectory()as directory:
            root=Path(directory);digest,p,s=self.fixture(root)
            with patch.object(recovery,'PROTOCOL_SHA',digest):
                bad=copy.deepcopy(s);bad['jobs'][0]['exit_code']=1;recovery.write(root/'status.json',bad)
                with self.assertRaisesRegex(ValueError,'terminal and successful'):recovery.original_snapshot(root)
                bad=copy.deepcopy(s);bad['audits']['epsilon-0p5']=dict(returncode=0);recovery.write(root/'status.json',bad)
                with self.assertRaisesRegex(ValueError,'failure history'):recovery.original_snapshot(root)

    def test_recovered_audit_requires_full_density_check_and_factor_certificate(self):
        with tempfile.TemporaryDirectory()as directory:
            root=Path(directory);_,protocol,_=self.fixture(root);entry=protocol['campaigns'][0]
            folder=Path(entry['path']);manifest=recovery.read(folder/'manifest.json')
            record=dict(observer_path='/frozen/observer.py',observer_sha256='abc')
            analysis=dict(pending=[],populations=[],provenance={str(folder/'manifest.json'):entry['manifest_sha256'],record['observer_path']:'abc'})
            for job in manifest['jobs']:
                analysis['populations'].append(dict(job=job,proposal_audit=dict(checked_actual_poses=8192,
                    proposal_anchor_indices=[0,1],maximum_log_density_error=1e-10,factor_audit=dict(valid=True))))
                for name in ('samples.jsonl','summary.json','manifest.json'):
                    path=Path(job['directory'])/name;analysis['provenance'][str(path)]=recovery.sha(path)
            recovery.verify_audit(root,entry,record,analysis)
            for key,value in [('checked_actual_poses',8191),('maximum_log_density_error',3e-8),('factor_audit',None)]:
                bad=copy.deepcopy(analysis);bad['populations'][0]['proposal_audit'][key]=value
                with self.subTest(key=key),self.assertRaises(ValueError):recovery.verify_audit(root,entry,record,bad)


if __name__=='__main__':unittest.main()
