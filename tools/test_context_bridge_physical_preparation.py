"""Exploratory saved-bank metadata controls; no materialized campaign or cloud work."""
import copy
import unittest

from prepare_context_bridge_physical import seed,validate_policy,panel_entries,caps,executable_resolutions,validate_drained_identity_set
from analyze_context_bridge_physical import CLOUD_RULE,EXTENSION_SCOPE,ALLOCATIONS
from test_context_broadened_physical_preparation import fixture


def policy():
    return dict(schema='context-bridge-physical-policy-v1',all_attempts=16384,strata=72,population_denominator=2048,
        extension_scope=EXTENSION_SCOPE,expected_valid_poses=3727,expected_clouds=7454,maximum_workers=4,threads_per_job=1,
        clouds_per_valid_pose=2,depletant_radius=1.5,activity=.035,lambda_intensity=2.24,estimator=CLOUD_RULE,
        cpu_seconds_per_job=600,wall_seconds_per_job=1200,memory_bytes_per_job=4*1024**3,
        cloud_limits=dict(raw_per_cloud=1000000,processed_per_cloud=1000000,raw_per_pose=2000000,
            processed_per_pose=2000000,total_per_attempt=2000000,callback_interval=8192))


class PreparationTests(unittest.TestCase):
    def test_fixed_policy_requires_exploratory_RB_and_all_attempts(self):
        value=policy();validate_policy(value)
        for key,bad_value in [('strata',20),('population_denominator',4096),('clouds_per_valid_pose',1),
            ('activity',.04),('wall_seconds_per_job',600),('expected_valid_poses',557),('expected_clouds',1114),
            ('extension_scope',dict(EXTENSION_SCOPE,fresh_confirmation=True)),('maximum_workers',8),('estimator',dict(CLOUD_RULE,primary_estimator='arithmetic'))]:
            bad=copy.deepcopy(value);bad[key]=bad_value
            with self.assertRaises(ValueError):validate_policy(bad)

    def test_all_valid_poses_no_selection_and_no_changed_input(self):
        e,rows,contributions=fixture();e['comparison_arm']='bridge';e['component']='cage0'
        for c in contributions:c.update(comparison_arm='bridge',component='cage0')
        before=copy.deepcopy(rows);entries=panel_entries(e,rows,contributions,dict(path='/held/rows',sha256='a'*64))
        self.assertEqual([x['ordinal'] for x in entries],[1,3]);self.assertEqual(rows,before)
        contributions[1]['log_q_arm']=float('nan')
        with self.assertRaises(ValueError):panel_entries(e,rows,contributions,{})

    def test_empty_stratum_and_small_bridge_stratum_caps(self):
        e,rows,cs=fixture((False,False));self.assertEqual(panel_entries(e,rows,cs,{}),[])
        self.assertEqual(caps(policy(),128)['raw_total'],256000000)
        self.assertEqual(caps(policy(),512)['processed_total'],1024000000)

    def test_fresh_seed_domain_and_population_component_roles(self):
        values=[seed('a'*64,f'{a}-pop{p}-{c}') for a,components in ALLOCATIONS.items() for p in range(4) for c,n in components.items() if n]
        self.assertEqual(len(set(values)),72)
        self.assertEqual(seed('a'*64,'baseline-pop0-full'),seed('a'*64,'baseline-pop0-full'))
        self.assertNotEqual(seed('a'*64,'baseline-pop0-full'),seed('b'*64,'baseline-pop0-full'))

    def test_exact77_drain_identities_cannot_be_replaced_by_equal_sized_set(self):
        identities=[(i,100+i) for i in range(77)]
        drain=dict(owned_groups=[dict(pid=p,birth_ticks=b) for p,b in identities])
        validate_drained_identity_set(identities,drain)
        bad=copy.deepcopy(drain);bad['owned_groups'][0]['birth_ticks']+=1
        with self.assertRaises(ValueError):validate_drained_identity_set(identities,bad)
        bad=copy.deepcopy(drain);bad['owned_groups'][1]=bad['owned_groups'][0]
        with self.assertRaises(ValueError):validate_drained_identity_set(identities,bad)

    def test_zero_volume_valid_and_non_AT_regions_are_not_filtered(self):
        entry,rows,contributions=fixture((True,True,True))
        for i,region in enumerate(('A_patch_complete','B','unbound')):
            rows[i]['region']=contributions[i]['region']=region
            rows[i]['envelope']=dict(lower_volume=0.,uncertain_volume=0.)
        panel=panel_entries(entry,rows,contributions,{})
        self.assertEqual(len(panel),3)
        self.assertEqual([p['metadata']['original_row']['region'] for p in panel],['A_patch_complete','B','unbound'])

    def test_empty_and_scoring_executable_inventory_is_exact(self):
        scorer=dict(argv=['/synthetic/scorer']);empty=dict(argv=['/synthetic/python','-B','empty.py'])
        self.assertEqual(set(executable_resolutions([scorer,scorer])),{'/synthetic/scorer'})
        self.assertEqual(set(executable_resolutions([empty])),{'/synthetic/python'})
        self.assertEqual(set(executable_resolutions([empty,scorer])),{'/synthetic/python','/synthetic/scorer'})


if __name__=='__main__':unittest.main()
