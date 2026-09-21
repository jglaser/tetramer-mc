"""Focused full-density, prefix, finite-calibration and child-lifecycle checks."""
import copy
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from analyze_shoulder_contact_atlas import (CENTERS, INNER_WINDOW, KEYS, SHOULDER_WINDOW,
    PrefixMoments, calibrations, ensure_density_audits, finish, full_weights, labeled_comparison, validate_plan)
from analyze_expanded_contact_atlas import nested_prefix_comparison
from prepare_cayley_rms_cover import read, write
from shoulder_atlas_moments import ShoulderAtlasMoments


def proposal_row(q=1.05, density=.2, cloud=(2.,6.)):
    return dict(q=q, log_proposal_density=math.log(density), log_hard_weight=-math.log(density),
        cloud_log_weights=[math.log(v) for v in cloud],
        log_importance_weight=math.log(sum(cloud)/len(cloud)/density))


def minimal_plan():
    return dict(q_window=SHOULDER_WINDOW, inner_q_window=INNER_WINDOW,
        analysis=dict(priority_order=list(CENTERS), radii=[.25,.5], centers=[dict(name=n) for n in CENTERS]),
        arms=[dict(name=n, samples_per_population=8, seeds=s) for n,s in [('narrow',[11,12]),('broad',[21,22])]],
        prefix_counts=[2,4,8], lambda_ratio=64, cloud_replicates=2, proposal_anchor_index=0,
        hybrid=dict(model_weight=.99, uniform_probability=.05, cover_scales=[1.]))


class FullShoulderAtlasTests(unittest.TestCase):
    def test_outer_rows_preserve_original_full_density_and_clouds(self):
        for q in [math.nextafter(1.,math.inf),1.05,1.1,1.95]:
            p,h,pair=full_weights(proposal_row(q))
            self.assertAlmostEqual(math.exp(p),20.)
            self.assertAlmostEqual(math.exp(h),5.)
            np.testing.assert_allclose(np.exp(pair),[10.,30.])
        self.assertEqual(full_weights(dict(zero='hard',q=1.04)),(None,None,None))
        with self.assertRaises(ValueError):full_weights(dict(zero='capture',log_importance_weight=0.))
        for q in [1.,2.]:
            with self.assertRaises(ValueError):full_weights(proposal_row(q))
        row=proposal_row();row['log_hard_weight']+=.01
        with self.assertRaises(AssertionError):full_weights(row)

    def test_window_and_stream_controls_reject_changed_targets(self):
        plan=minimal_plan();validate_plan(plan)
        changes=[('q_window',dict(SHOULDER_WINDOW,lower_inclusive=True)),
                 ('inner_q_window',dict(INNER_WINDOW,upper_inclusive=True)),
                 ('prefix_counts',[2,8,4]),('cloud_replicates',1),('lambda_ratio',32),
                 ('hybrid',dict(model_weight=.99,uniform_probability=.1,cover_scales=[1.]))]
        for key,value in changes:
            modified=copy.deepcopy(plan);modified[key]=value
            with self.assertRaises(ValueError):validate_plan(modified)
        modified=copy.deepcopy(plan);modified['arms'][1]['seeds'][0]=11
        with self.assertRaises(ValueError):validate_plan(modified)

    def test_fixed_prefixes_share_rows_and_keep_zero_denominators(self):
        prefix=PrefixMoments(8,[2,4,8]);weights=np.array([1.,0.,2.,3.,0.,4.,5.,6.])
        for i,w in enumerate(weights):
            if w:prefix.add(i,1.05,(.1,.8,.8),math.log(w),0.,(math.log(w),math.log(w)))
            else:prefix.add(i)
        prefix.complete(); reports={n:m.report() for n,m in prefix.prefixes.items()}
        for n,result in reports.items():
            self.assertEqual(result['physical']['full']['draws'],n)
            self.assertAlmostEqual(math.exp(result['physical']['full']['logQ']),weights[:n].mean())
        comparison=nested_prefix_comparison(reports[4]['physical']['full'],reports[8]['physical']['full'])
        expected=np.var(weights,ddof=1)*(1/4-1/8)
        self.assertAlmostEqual(math.exp(comparison['estimated_log_variance_of_difference']),expected)
        with self.assertRaises(ValueError):prefix.add(8)
        with self.assertRaises(ValueError):PrefixMoments(8,[4,8]).add(1)
        with self.assertRaises(ValueError):PrefixMoments(8,[4,8]).complete()

    def test_population_means_and_concentration_use_all_original_n(self):
        aggregate=ShoulderAtlasMoments();populations=[]
        for seed,weights in [(11,[0.,0.,1.,1.]),(12,[0.,0.,3.,3.])]:
            local=ShoulderAtlasMoments()
            for w in weights:
                if w:local.add(1.05,(.1,.8,.8),math.log(w),0.,(math.log(w),math.log(w)))
                else:local.add()
            aggregate.merge(local);populations.append(dict(seed=seed,samples=4,**local.report()))
        result=finish(aggregate,populations)
        full=result['physical']['full']
        self.assertEqual(full['draws'],8)
        self.assertAlmostEqual(math.exp(full['logQ']),1.)
        np.testing.assert_allclose(full['population_mass_fractions'],[.25,.75])
        self.assertAlmostEqual(full['independent_population_RSE'],.5)
        self.assertIsNone(result['physical']['outer']['maximum_population_fraction'])

    def test_calibration_selects_only_matching_inner_regions(self):
        def summary(value):return dict(logQ=math.log(value),log_variance_of_mean=math.log(.01),row_RSE=.1/value,independent_population_RSE=.1/value,draws=100)
        campaign={kind:{key:summary(2.) for key in KEYS} for kind in ('physical','hard')}
        campaign['physical']['full']=summary(200.)
        reference=dict(original_q_window=INNER_WINDOW,campaigns=[dict(owner=n,radius_A=r,**{k:dict(full=summary(2.)) for k in ('physical','hard')}) for n in ('mixture','geometry') for r in (.25,.5)],
            independent_assigned_region_sums={k:{n:summary(2.) for n in CENTERS} for k in ('physical','hard')},
            independent_union_sum={k:summary(2.) for k in ('physical','hard')})
        direct=dict(original_q_window=INNER_WINDOW,campaigns=[dict(radius_A=r,**{k:dict(full=summary(2.)) for k in ('physical','hard')}) for r in (.25,.5)])
        result=calibrations(campaign,reference,direct)
        self.assertTrue(all(k.startswith('inner_') for k in result['regions']))
        self.assertNotIn('full',result['regions']);self.assertNotIn('inner',result['regions'])
        for row in result['regions'].values():
            self.assertEqual(row['physical']['left_to_right_ratio'],1.)
            self.assertEqual(row['physical']['left_label'],'campaign')
            self.assertEqual(row['physical']['right_label'],'finite_reference')
        # A nonunit ratio identifies the direction, unlike equality alone.
        reference['campaigns'][0]['physical']['full']=summary(4.)
        result=calibrations(campaign,reference,direct)
        comparison=result['regions']['inner_mixture_ball0p25']['physical']
        self.assertAlmostEqual(comparison['left_to_right_ratio'],.5)
        self.assertAlmostEqual(comparison['left_minus_right_logQ'],-math.log(2.))
        self.assertLess(comparison['linear_left_minus_right_in_combined_row_SE'],0)
        reference['original_q_window']=SHOULDER_WINDOW
        with self.assertRaises(ValueError):calibrations(campaign,reference,direct)

    def test_named_comparison_has_explicit_ratio_and_difference_direction(self):
        left=dict(logQ=math.log(2.),row_RSE=.1,independent_population_RSE=.2)
        right=dict(logQ=math.log(8.),row_RSE=.25,independent_population_RSE=.3)
        value=labeled_comparison(left,right,'narrow','broad','Neutral fixed-arm diagnostic.')
        self.assertEqual((value['left_label'],value['right_label']),('narrow','broad'))
        self.assertEqual((value['left'],value['right']),(left,right))
        self.assertAlmostEqual(value['left_to_right_ratio'],.25)
        self.assertAlmostEqual(value['left_minus_right_logQ'],-math.log(4.))
        self.assertAlmostEqual(value['linear_left_minus_right_in_combined_row_SE'],
                               (2.-8.)/math.hypot(2.*.1,8.*.25))
        self.assertAlmostEqual(value['linear_left_minus_right_in_combined_population_SE'],
                               (2.-8.)/math.hypot(2.*.2,8.*.3))
        self.assertNotIn('historical',value);self.assertNotIn('fresh',value)
        zero=dict(logQ=None,row_RSE=None,independent_population_RSE=None)
        self.assertIsNone(labeled_comparison(zero,right,'new','reference','scope')['left_to_right_ratio'])

    def test_auditors_all_drained_before_failure_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);requests=[(dict(name=n),out/n,None,None) for n in ('narrow','broad')]
            for _,p,*_ in requests:p.mkdir()
            children=[Mock(pid=17),Mock(pid=18)];children[0].wait.return_value=1;children[1].wait.return_value=0
            with patch('analyze_shoulder_contact_atlas.subprocess.Popen',side_effect=children):
                with self.assertRaises(ValueError):ensure_density_audits(requests,out,True,16)
            self.assertTrue(all(c.wait.call_count==1 for c in children))
            state=read(out/'runner-state.json')
            self.assertEqual(state['phase'],'density_audits_failed')
            self.assertEqual([p['exit_code'] for p in state['processes']],[1,0])
            self.assertTrue(all(p['terminal'] for p in state['processes']))

    def test_later_launch_failure_drains_earlier_auditor_without_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);requests=[(dict(name=n),out/n,None,None) for n in ('narrow','broad')]
            for _,p,*_ in requests:p.mkdir()
            child=Mock(pid=17);child.wait.return_value=0
            with patch('analyze_shoulder_contact_atlas.subprocess.Popen',side_effect=[child,OSError('launch failure')]) as launch:
                with self.assertRaises(OSError):ensure_density_audits(requests,out,True,16)
            self.assertEqual(launch.call_count,2);self.assertEqual(child.wait.call_count,1)
            self.assertTrue(read(out/'runner-state.json')['processes'][0]['terminal'])
            # Existing outputs are reused without executing another child.
            for _,p,*_ in requests:write(p/'assessment-streaming.json',dict(complete=True))
            with patch('analyze_shoulder_contact_atlas.subprocess.Popen') as launch:
                result=ensure_density_audits(requests,out,False,16)
            launch.assert_not_called();self.assertTrue(all(p['reused'] for p in result['processes']))


if __name__=='__main__':unittest.main()
