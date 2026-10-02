import copy
import json
from pathlib import Path
import unittest
import numpy as np
import analyze_dimer_one_step_stationarity as base
import analyze_dimer_matched_stationarity as matched
from test_analyze_dimer_one_step_stationarity import identity_row

CONFIG = json.loads((Path(__file__).resolve().parents[1]/'examples/dimer-matched-stationarity.json').read_text())


def fixture(branch='uniform'):
    spec = CONFIG['populations'][0]
    _, common = identity_row()
    common['population_id'] = spec['id']; common['source_seed'] = base.expected_seed(spec['seed'], 0, 'source')
    proposal = common['proposal']; common['proposal'] = None; common['defensive_proposal'] = proposal
    proposal['null_reason'] = None
    old = common['old_state']
    relative = [old[0], {'position': [0., 1.1, 0.], 'orientation': [1., 0., 0., 0.]}]
    proposal['edges'] = []
    for pose in relative:
        if branch == 'uniform':
            trace = {'branch_uniform': .25, 'translation_uniforms': [(x/2.5+1)/2 for x in pose['position']],
                     'quaternion_normals': [1., 0., 0., 0.]}
        else:
            trace = {'branch_uniform': .75, 'target_label': {'kind': 'single', 'member': 0, 'anchor': 0, 'branch': 0},
                     'target_latent': pose['position']+[0., 0., 0.]}
        proposal['edges'].append({'old_relative_pose': copy.deepcopy(pose), 'proposed_relative_pose': copy.deepcopy(pose),
                                  'branch': branch, 'trace': trace, 'null_reason': None})
    endpoints = []
    for arm in matched.ARMS:
        _, row = identity_row(arm)
        endpoints.append({'arm': arm, **{k:copy.deepcopy(row[k]) for k in matched.ENDPOINT_KEYS-{'arm'}}})
    common.update(state=old, d=common['old_d'], accepted=False, gate=None, path_gate=None, log_ratio=None)
    return spec, {'common': common, 'outcomes': endpoints}


class MatchedTests(unittest.TestCase):
    def test_frozen_allocation_and_seeds(self):
        matched.validate_config(CONFIG)
        for edit in [lambda c:c.__setitem__('draws_per_population', 16384),
                     lambda c:c['populations'][1].__setitem__('seed', c['populations'][0]['seed']),
                     lambda c:c.__setitem__('activity', 1.4), lambda c:c['populations'].pop()]:
            bad=copy.deepcopy(CONFIG);edit(bad)
            with self.assertRaises(ValueError):matched.validate_config(bad)

    def test_both_generation_branches_and_complete_matched_rows(self):
        for branch in ['uniform','learned']:
            spec, row=fixture(branch)
            result=matched.audit_pair(row,CONFIG,spec)
            self.assertLess(max(result.values()),1e-12)
            json.dumps(result,allow_nan=False)

    def test_generation_tampering_is_rejected(self):
        for branch in ['uniform','learned']:
            spec,row=fixture(branch)
            for edit in [lambda r:r['common']['defensive_proposal']['edges'][0]['trace'].__setitem__('branch_uniform', .75 if branch=='uniform' else .25),
                         lambda r:r['common']['defensive_proposal']['edges'][1]['proposed_relative_pose']['position'].__setitem__(0,.2),
                         lambda r:r['common']['defensive_proposal']['edges'].pop()]:
                bad=copy.deepcopy(row);edit(bad)
                with self.assertRaises(ValueError):matched.audit_pair(bad,CONFIG,spec)
        spec,row=fixture('learned');row['common']['defensive_proposal']['edges'][0]['trace']['target_latent'][0]+=.1
        with self.assertRaises(ValueError):matched.audit_pair(row,CONFIG,spec)

    def test_matching_scope_and_numerical_tampering_is_rejected(self):
        spec,row=fixture()
        for edit in [lambda r:r['outcomes'].pop(),
                     lambda r:r['outcomes'][1].__setitem__('arm','analytic'),
                     lambda r:r['outcomes'][1].__setitem__('log_uniform',-20.),
                     lambda r:r['common'].__setitem__('accepted',True),
                     lambda r:r['common'].__setitem__('q_correction',.1),
                     lambda r:r['outcomes'][1]['gate'].__setitem__('gained',1),
                     lambda r:r['outcomes'][0].__setitem__('accepted',False),
                     lambda r:r['common'].__setitem__('orientation_refresh_count',2)]:
            bad=copy.deepcopy(row);edit(bad)
            with self.assertRaises(ValueError):matched.audit_pair(bad,CONFIG,spec)

    def test_five_signed_primary_tests_include_zeros_and_serialize(self):
        self.assertEqual(len(matched.GROUPS),5)
        for data,pos,neg in [(np.zeros(1000),0,0),(np.array([1.]*50+[-1.]*50+[0.]*900),50,50),(np.array([-1.]*100+[0.]*900),0,100)]:
            result=base.primary_contact(base.moments(data),pos,neg)
            self.assertEqual(result,json.loads(json.dumps(result,allow_nan=False)))
            self.assertEqual(result['primary_reject_stationarity'],bool(neg==100))
            self.assertGreater(result['simultaneous_hoeffding_interval'][1]-result['simultaneous_hoeffding_interval'][0],0.)


if __name__=='__main__':unittest.main()
