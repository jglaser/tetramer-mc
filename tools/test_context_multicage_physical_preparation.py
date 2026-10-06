"""Prospective metadata controls; no materialized campaign or cloud work."""
import copy
import unittest

from prepare_context_multicage_physical import seed,validate_policy,panel_entries,caps,executable_resolutions
from prepare_context_multicage_guides import CLOUD_RULE
from test_context_broadened_physical_preparation import fixture


def policy():
    return dict(schema='context-multicage-physical-policy-v1',all_attempts=32768,strata=32,population_denominator=4096,
        clouds_per_valid_pose=2,depletant_radius=1.5,activity=.035,lambda_intensity=2.24,estimator=CLOUD_RULE,
        cpu_seconds_per_job=600,wall_seconds_per_job=1200,memory_bytes_per_job=4*1024**3,
        cloud_limits=dict(raw_per_cloud=1000000,processed_per_cloud=1000000,raw_per_pose=2000000,
            processed_per_pose=2000000,total_per_attempt=2000000,callback_interval=8192))


class PreparationTests(unittest.TestCase):
    def test_fixed_policy_requires_prospective_RB_and_all_attempts(self):
        value=policy();validate_policy(value)
        for key,bad_value in [('strata',20),('population_denominator',2048),('clouds_per_valid_pose',1),
            ('activity',.04),('wall_seconds_per_job',600),('estimator',dict(CLOUD_RULE,primary_estimator='arithmetic'))]:
            bad=copy.deepcopy(value);bad[key]=bad_value
            with self.assertRaises(ValueError):validate_policy(bad)

    def test_all_valid_poses_no_selection_and_no_changed_input(self):
        e,rows,contributions=fixture();e['comparison_arm']='multicage';e['component']='cage0'
        for c in contributions:c.update(comparison_arm='multicage',component='cage0')
        before=copy.deepcopy(rows);entries=panel_entries(e,rows,contributions,dict(path='/held/rows',sha256='a'*64))
        self.assertEqual([x['ordinal'] for x in entries],[1,3]);self.assertEqual(rows,before)
        contributions[1]['log_q_arm']=float('nan')
        with self.assertRaises(ValueError):panel_entries(e,rows,contributions,{})

    def test_empty_stratum_and_small_cage_stratum_caps(self):
        e,rows,cs=fixture((False,False));self.assertEqual(panel_entries(e,rows,cs,{}),[])
        self.assertEqual(caps(policy(),512)['raw_total'],1024000000)
        self.assertEqual(caps(policy(),1024)['processed_total'],2048000000)

    def test_fresh_seed_domain_and_population_component_roles(self):
        values=[seed('a'*64,f'{a}-pop{p}-{c}') for a in ('baseline','multicage') for p in range(4) for c in ('full','cage0','cage1')]
        self.assertEqual(len(set(values)),24)
        self.assertEqual(seed('a'*64,'baseline-pop0-full'),seed('a'*64,'baseline-pop0-full'))
        self.assertNotEqual(seed('a'*64,'baseline-pop0-full'),seed('b'*64,'baseline-pop0-full'))

    def test_empty_and_scoring_executable_inventory_is_exact(self):
        scorer=dict(argv=['/synthetic/scorer']);empty=dict(argv=['/synthetic/python','-B','empty.py'])
        self.assertEqual(set(executable_resolutions([scorer,scorer])),{'/synthetic/scorer'})
        self.assertEqual(set(executable_resolutions([empty])),{'/synthetic/python'})
        self.assertEqual(set(executable_resolutions([empty,scorer])),{'/synthetic/python','/synthetic/scorer'})


if __name__=='__main__':unittest.main()
