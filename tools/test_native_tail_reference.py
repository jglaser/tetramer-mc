"""Exact cover, native endpoints, and full-N direct tail-mask controls."""
import json
import copy
import math
import unittest

import numpy as np

from analyze_latent_region import original_q_contains
from analyze_native_tail_reference import NativeTailMoments, validated_cloud_log_weight, validate_terminal_status
from prepare_cayley_rms_cover import derive_model
from prepare_native_tail_reference import MASKS, WINDOW, exact_cover_check, old_radius_bin


class NativeTailReferenceTests(unittest.TestCase):
    def fixture(self):
        # Rationally rotated anisotropic box: use adjacent antipodal pairs so
        # both floating-point and emitted-decimal centroids are exactly zero.
        positions=[]
        for y in (-9.,9.):
            for z in (-20.,20.):
                point=[(45+12*z)/13,y,(-108+5*z)/13]
                positions.extend([point,[-v for v in point]])
        metric=dict(member_error_scale=2.,angle_error_scale_deg=15.,
                    native_poses=[dict(position=[0.,0.,0.],orientation=[1.,0.,0.,0.])],
                    rigid_members=[dict(position=p,orientation=[1.,0.,0.,0.]) for p in positions])
        cfg=dict(metadata=metric,fixed_poses=[dict(position=[12.,-4.,8.],orientation=[0.,1.,0.,0.])])
        model,_=derive_model(metric,cfg['fixed_poses'][0],'fixture',q_max=1.,ell=17.)
        return cfg,model

    def test_exact_emitted_cover_precision_and_failure_of_narrowed_cover(self):
        cfg,model=self.fixture()
        report=exact_cover_check(json.dumps(cfg),model)
        self.assertTrue(report['complete'])
        self.assertTrue(all(p>0 for p in report['certified_precision_gap_positive_LDL_pivots_approx']))
        for i in range(3,6):
            for j in range(3,6): model['covariances'][0][i][j]*=.8
        with self.assertRaisesRegex(ValueError,'positive definite'):
            exact_cover_check(json.dumps(cfg),model)

    def test_native_endpoints_and_tail_boundary_are_exact(self):
        self.assertTrue(original_q_contains(0.,WINDOW)); self.assertTrue(original_q_contains(1.,WINDOW))
        self.assertFalse(original_q_contains(np.nextafter(1.,np.inf),WINDOW))
        self.assertEqual([old_radius_bin(x) for x in (4.,5.,8.,12.)],list(MASKS[:4]))
        self.assertEqual(old_radius_bin(np.nextafter(12.,np.inf)),'old_r_gt_12')
        self.assertEqual(old_radius_bin(53.),'old_r_gt_12')  # No artificial certificate cutoff.

    def test_disjoint_masks_keep_full_N_and_covariance(self):
        moments=NativeTailMoments(); weights=[0.,2.,3.,5.,7.,11.]
        moments.add()
        for radius,w in zip((4.,5.,8.,12.,15.),weights[1:]):
            moments.add(radius,math.log(w),0.,[math.log(w)]*2)
        report=moments.report(); n=len(weights)
        self.assertEqual(report['physical']['full']['draws'],n)
        self.assertAlmostEqual(math.exp(report['physical']['full']['logQ']),np.mean(weights))
        self.assertAlmostEqual(math.exp(report['physical']['full']['log_variance_of_mean']),np.var(weights,ddof=1)/n)
        for key,w in zip(MASKS,weights[1:]):
            self.assertEqual(report['physical'][key]['draws'],n)
            self.assertAlmostEqual(math.exp(report['physical'][key]['logQ']),w/n)
        self.assertTrue(report['partition_checks']['physical']['complete_row_partition'])

    def cloud(self):
        return dict(lower_volume=3.,upper_volume=7.,uncertain_volume=4.,raw_points=20,overlap_points=5,
                    retained_cells=3,created_cells=5,certified_cells=1,
                    log_weight=.035*3+5*math.log1p(1/64))

    def test_poisson_counters_require_nonnegative_integers_and_thinning(self):
        cloud=self.cloud()
        self.assertAlmostEqual(validated_cloud_log_weight(cloud,.035,.035*64),cloud['log_weight'])
        for key,bad in (('raw_points',-1),('raw_points',20.),('overlap_points',True),
                        ('overlap_points',21),('retained_cells',6)):
            changed=copy.deepcopy(cloud);changed[key]=bad
            with self.subTest(key=key,bad=bad),self.assertRaises(ValueError):
                validated_cloud_log_weight(changed,.035,.035*64)

    def test_poisson_envelope_volumes_must_be_finite_and_consistent(self):
        for key,bad in (('lower_volume',-.1),('upper_volume',float('inf')),
                        ('upper_volume',2.),('uncertain_volume',5.),('uncertain_volume',float('nan'))):
            cloud=self.cloud();cloud[key]=bad
            with self.subTest(key=key,bad=bad),self.assertRaises((ValueError,AssertionError)):
                validated_cloud_log_weight(cloud,.035,.035*64)
        cloud=self.cloud();cloud.update(upper_volume=3.,uncertain_volume=0.)
        with self.assertRaisesRegex(ValueError,'zero-intensity'):
            validated_cloud_log_weight(cloud,.035,.035*64)
        cloud.update(raw_points=0,overlap_points=0,log_weight=.035*3)
        self.assertAlmostEqual(validated_cloud_log_weight(cloud,.035,.035*64),.105)

    def test_only_terminal_success_matches_the_population(self):
        job=dict(id='r02')
        validate_terminal_status(dict(id='r02',returncode=0,completed_unix=1.),job)
        for change in (dict(returncode=1),dict(returncode=False),dict(id='r01'),dict(completed_unix=float('nan'))):
            status=dict(id='r02',returncode=0,completed_unix=1.);status.update(change)
            with self.subTest(change=change),self.assertRaises(ValueError): validate_terminal_status(status,job)


if __name__=='__main__': unittest.main()
