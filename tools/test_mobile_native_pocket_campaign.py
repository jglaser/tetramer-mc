"""Failure/draining and physical region identity controls; no protein MC."""
import copy
import tempfile
import unittest
from pathlib import Path
from mobile_native_pocket_campaign import execute_all,execute_batches,make_regions,read,MODEL,REFERENCE,DEFINITION,DESIGNS

class Campaign(unittest.TestCase):
    def test_physical_region_preserved(self):
        cfg=read(REFERENCE/'config.json');base=read(MODEL)['base_model']
        motif=next(m for m in read(DEFINITION.parent/'inputs/native-pair-motifs.json')['motifs']if m['id']==7)
        original=copy.deepcopy((cfg,base,motif));regions=make_regions(cfg,base,motif)
        self.assertEqual((cfg,base,motif),original)
        for r in regions.values():
            self.assertEqual(r['physical_fixed_neighbors'],cfg['fixed_poses'])
            self.assertEqual(r['physical_metric'],cfg['metadata'])
            self.assertEqual(r['gaussian_chart']['covariances'],[base['covariances'][19]])
            self.assertEqual(r['gaussian_chart']['means'],[[0.]*6])
            self.assertEqual(r['minimum_original_q'],1.)
            self.assertIs(r['minimum_original_q_inclusive'],False)
            self.assertNotIn('maximum_original_q',r)
        self.assertEqual(regions['r5']['gaussian_chart'],regions['shell5to8']['gaussian_chart'])

    def test_coverage_allocation_and_same_core(self):
        cfg=read(REFERENCE/'config.json');base=read(MODEL)['base_model']
        motif=next(m for m in read(DEFINITION.parent/'inputs/native-pair-motifs.json')['motifs']if m['id']==7)
        old=make_regions(cfg,base,motif);new=make_regions(cfg,base,motif,'coverage')
        self.assertEqual(old['r5'],new['r5repeat'])
        specifications=DESIGNS['coverage']['arms']
        self.assertEqual(sum(s['replicates']for s in specifications.values()),24)
        self.assertEqual(sum(s['replicates']*s['samples']for s in specifications.values()),262144)
        for name in tuple(new)[1:]:
            self.assertEqual(new[name]['gaussian_chart'],old['r5']['gaussian_chart'])
            self.assertEqual(new[name]['minimum_mahalanobis_radius'],specifications[name]['inner'])

    def batch(self,failing_launch=None,failing_exit=None,count=4):
        directory=tempfile.TemporaryDirectory();self.addCleanup(directory.cleanup)
        root=Path(directory.name);jobs=[dict(status='pending',directory=str(root/f'out{i}'),log=str(root/f'{i}.log'),command=[str(i)])for i in range(count)]
        launched=[];waited=[];snapshots=[]
        class Child:
            def __init__(self,index):self.pid=100+index;self.index=index
            def wait(self):waited.append(self.index);return 1 if self.index==failing_exit else 0
        def spawn(argv,**kwargs):
            index=int(argv[0])
            if index==failing_launch:raise OSError('launch failed')
            launched.append(index);return Child(index)
        return jobs,launched,waited,snapshots,spawn

    def test_launch_failure_drains_started_children(self):
        jobs,launched,waited,snapshots,spawn=self.batch(failing_launch=2)
        with self.assertRaises(OSError):execute_all(jobs,lambda:snapshots.append(copy.deepcopy(jobs)),spawn)
        self.assertEqual(launched,waited);self.assertEqual(waited,[0,1])
        self.assertEqual([j['status']for j in jobs],['complete','complete','not_started','not_started'])

    def test_exit_failure_drains_entire_batch(self):
        jobs,launched,waited,_,spawn=self.batch(failing_exit=1)
        with self.assertRaises(RuntimeError):execute_all(jobs,lambda:None,spawn)
        self.assertEqual(launched,waited);self.assertEqual(waited,list(range(4)))
        self.assertEqual([j['returncode']for j in jobs],[0,1,0,0])

    def test_complete_and_worker_cap(self):
        jobs,launched,waited,_,spawn=self.batch()
        execute_all(jobs,lambda:None,spawn)
        self.assertEqual(launched,waited);self.assertTrue(all(j['status']=='complete'for j in jobs))
        with self.assertRaisesRegex(ValueError,'Worker cap'):execute_all([{}]*9,lambda:None,spawn)

    def test_failed_batch_prevents_later_launches(self):
        jobs,launched,waited,_,spawn=self.batch(failing_exit=2,count=24)
        with self.assertRaises(RuntimeError):execute_batches(jobs,lambda:None,spawn)
        self.assertEqual(launched,waited);self.assertEqual(waited,list(range(8)))
        self.assertTrue(all(j['status']=='not_started'for j in jobs[8:]))

    def test_three_batches_drain_before_next_launch(self):
        jobs,launched,waited,_,spawn=self.batch(count=24);peak=0
        def checked_spawn(*args,**kwargs):
            nonlocal peak
            child=spawn(*args,**kwargs);peak=max(peak,len(launched)-len(waited))
            return child
        execute_batches(jobs,lambda:None,checked_spawn)
        self.assertEqual(launched,waited);self.assertEqual(peak,8)
        self.assertTrue(all(j['returncode']==0 for j in jobs))

if __name__=='__main__':unittest.main()
