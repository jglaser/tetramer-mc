import copy
from pathlib import Path
import tempfile
import unittest

from prepare_context_broadened_physical import panel_entries,caps,seed,repository_root,executable_resolutions
from empty_physical_stratum import empty_summary

def fixture(valid=(False,True,False,True)):
    entry=dict(id='baseline-p0-full',draws=len(valid),comparison_arm='baseline',population_index=0,component='full')
    rows=[];cs=[]
    for i,v in enumerate(valid):
        rows.append(dict(complete=True,input=dict(ordinal=i,proposed_pose=dict(position=[i,0,0],orientation=[1,0,0,0])),
            actual=dict(physical_valid=v,wall_valid=v,core_valid=v),physical_zero=not v,region='unbound' if v else None,
            clouds=[],log_physical_contribution=None,physical_weight_status='not_estimated'))
        cs.append(dict(stratum_id=entry['id'],ordinal=i,comparison_arm='baseline',population_index=0,component='full',
            physical_valid=v,region='unbound' if v else 'hard_invalid',log_q_arm=-4.,
            proposed_pose=copy.deepcopy(rows[-1]['input']['proposed_pose'])))
    return entry,rows,cs

class PreparationTests(unittest.TestCase):
    def test_all_valid_and_only_valid_poses_are_included(self):
        entry,rows,cs=fixture();saved=copy.deepcopy(rows)
        panel=panel_entries(entry,rows,cs,dict(path='/held/rows',sha256='a'*64))
        self.assertEqual([e['ordinal'] for e in panel],[1,3])
        self.assertEqual([e['metadata']['log_q_balanced'] for e in panel],[-4.,-4.])
        self.assertEqual(panel[1]['metadata']['original_row'],rows[3])
        self.assertEqual(rows,saved)

    def test_missing_or_changed_attempt_is_rejected(self):
        entry,rows,cs=fixture()
        with self.assertRaises(ValueError):panel_entries(entry,rows[:-1],cs,{})
        for key,value in [('ordinal',2),('stratum_id','other'),('physical_valid',True),('region','unbound')]:
            bad=copy.deepcopy(cs);bad[0][key]=value
            with self.assertRaises(ValueError):panel_entries(entry,rows,bad,{})
        bad=copy.deepcopy(cs);bad[1]['proposed_pose']['position'][0]+=1
        with self.assertRaises(ValueError):panel_entries(entry,rows,bad,{})
        bad=copy.deepcopy(cs);bad[1]['log_q_arm']=float('nan')
        with self.assertRaises(ValueError):panel_entries(entry,rows,bad,{})

    def test_empty_stratum_has_a_real_zero_receipt(self):
        entry,rows,cs=fixture((False,False))
        self.assertEqual(panel_entries(entry,rows,cs,{}),[])
        panel=dict(entries=[],every_valid_pose_included=True,attempts=2,hard_invalid_count=2,
                   stratum_id=entry['id'],source_rows=dict(path='/held/rows',sha256='b'*64))
        result=empty_summary({},panel,rows)
        self.assertEqual(result['attempts'],2);self.assertEqual(result['clouds_completed'],0)
        rows[0]['actual']['physical_valid']=True
        with self.assertRaises(ValueError):empty_summary({},panel,rows)

    def test_count_caps_preserve_original_attempt_denominator(self):
        policy=dict(cloud_limits=dict(raw_per_cloud=1000000,processed_per_cloud=1000000,
                    raw_per_pose=2000000,processed_per_pose=2000000,total_per_attempt=2000000,callback_interval=8192))
        result=caps(policy,1536)
        self.assertEqual(result['raw_total'],3072000000)
        self.assertEqual(result['processed_total'],3072000000)
        self.assertEqual(result['raw_per_cloud'],1000000)
        with self.assertRaises(ValueError):caps(policy,0)
        bad=copy.deepcopy(policy);bad['cloud_limits']['raw_per_cloud']=500000
        with self.assertRaises(ValueError):caps(bad,1536)

    def test_fresh_role_seeds_are_deterministic_and_distinct(self):
        a=seed('a'*64,'baseline-p0-full')
        self.assertEqual(a,seed('a'*64,'baseline-p0-full'))
        self.assertNotEqual(a,seed('a'*64,'broadened-p0-full'))
        self.assertNotEqual(a,seed('b'*64,'baseline-p0-full'))

    def test_repository_resolution_survives_frozen_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name in ('Cargo.toml','src/overlap_weight.rs','examples/fixed_saved_pose_overlap.rs'):
                path=root/name;path.parent.mkdir(exist_ok=True);path.touch()
            self.assertEqual(repository_root(root/'tools/prepare.py'),root)
            self.assertEqual(repository_root(root/'results/campaign/code/prepare.py'),root)
            with self.assertRaises(ValueError):repository_root(root.parent/'unrelated/prepare.py')

    def test_lane_executable_inventory_matches_actual_job_types(self):
        scoring=dict(argv=['/synthetic/scorer','--config','ignored'])
        empty=dict(argv=['/synthetic/python','-B','empty.py'])
        self.assertEqual(executable_resolutions([scoring,scoring]),{'/synthetic/scorer':'/synthetic/scorer'})
        self.assertEqual(executable_resolutions([empty]),{'/synthetic/python':'/synthetic/python'})
        self.assertEqual(set(executable_resolutions([scoring,empty])),{'/synthetic/scorer','/synthetic/python'})

if __name__=='__main__':unittest.main()
