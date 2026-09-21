"""Support, Haar, epsilon-one and analytic-integration controls."""
import copy
from decimal import Decimal, localcontext
import math
import unittest

from analyze_native_region_reference import GaussianGuide, hybrid_log_density
from prepare_far_capture_control import (BETA, POSE, ball, capture_proof, dormant_model,
    haar_cap, norm_upper, sphere_exact, sphere_overlap)


class FarCaptureControls(unittest.TestCase):
    def config(self):
        return dict(capture_center=[0.]*3,capture_radius=18.,fixed_poses=[POSE],metadata=dict(
            native_poses=[POSE],rigid_members=[dict(POSE,position=[-27.,0.,0.]),dict(POSE,position=[27.,0.,0.])],
            member_error_scale=2.,angle_error_scale_deg=15.))

    def test_bound_is_saturated_by_halfturn_at_capture_edge(self):
        p=capture_proof(self.config())
        self.assertEqual(p['exact_kinematic_supremum_evaluated_FP64'],36.)
        self.assertAlmostEqual(p['saturating_pose']['original_q'],36.)
        self.assertGreater(p['guarded_q_upper'],36.)
        self.assertLess(p['guarded_q_upper'],37.)

    def test_norm_upper_bounds_exact_binary_inputs(self):
        for v in ([.1,.2,.3],[-27.21763845354943,1e-20,4.],[0.,0.,0.]):
            upper=norm_upper(v)
            with localcontext() as ctx:
                ctx.prec=100
                exact_squared=sum(Decimal.from_float(x)**2 for x in v)
                self.assertGreaterEqual(Decimal.from_float(upper)**2,exact_squared)

    def test_incomplete_qmax_and_changed_reference_rejected(self):
        c=self.config();c['capture_radius']=21.
        with self.assertRaisesRegex(ValueError,'qmax'):capture_proof(c)
        c=self.config();c['capture_center'][0]=1.
        with self.assertRaisesRegex(ValueError,'coincident'):capture_proof(c)

    def test_zero_integral_has_both_radial_and_angular_cutoffs(self):
        answer=sphere_exact(0.)
        expected=ball(3)-ball(1)-haar_cap(math.radians(75))*(ball(1.25)-ball(1))
        self.assertAlmostEqual(answer['Q'],expected,places=12)
        self.assertAlmostEqual(answer['hard_exact'],expected,places=12)
        self.assertGreater(answer['Q'],(1-haar_cap(math.radians(75)))*(ball(3)-ball(1)))
        self.assertLess(answer['Q'],ball(3)-ball(1))

    def test_positive_lens_integral_and_noncontact_piece(self):
        a,b=sphere_exact(0.),sphere_exact(.4)
        self.assertGreater(b['Q'],a['Q'])
        self.assertAlmostEqual(a['radial_contributions'][-1],b['radial_contributions'][-1],places=12)
        self.assertEqual(sphere_overlap(2.),0.)
        self.assertAlmostEqual(sphere_overlap(1.),5*math.pi/12)

    def test_epsilon_one_density_at_haar_seam_and_cube_boundaries(self):
        cfg=self.config();raw=dormant_model('a'*64)
        description=dict(weight=BETA,uniform_probability=1.,anchor_index=0,anchor_pose=POSE,
                         cube_lengths=[36.]*3,capture_center=[0.]*3)
        guide=GaussianGuide(description,raw,cfg,'a'*64)
        cover=dict(reference=POSE,centroid=[0.]*3,ball_radius=74.,angle_cap=math.pi,volume=ball(74.))
        model=dict(covers=[cover],weights=[1.],scales=[1.])
        for x,inside in [(0.,True),(-18.,True),(18.,False),(40.,False)]:
            pose=dict(position=[x,0.,0.],orientation=[0.,1.,0.,0.])
            g=hybrid_log_density(model,guide,pose)[0]
            self.assertAlmostEqual(g,math.log((BETA/36**3 if inside else 0.)+(1-BETA)/ball(74.)),places=12)
            self.assertEqual(guide.log_density(pose)[1],inside)

    def test_dormant_gaussian_parameters_do_not_change_density(self):
        cfg=self.config();raw=dormant_model('a'*64)
        desc=dict(weight=BETA,uniform_probability=1.,anchor_index=0,anchor_pose=POSE,cube_lengths=[36.]*3,capture_center=[0.]*3)
        one=GaussianGuide(desc,raw,cfg,'a'*64)
        raw=copy.deepcopy(raw);raw['means'][0]=[100.]*6
        two=GaussianGuide(desc,raw,cfg,'a'*64)
        for p in [POSE,dict(position=[40.,0.,0.],orientation=[1.,0.,0.,0.])]:
            self.assertEqual(one.log_density(p)[0],two.log_density(p)[0])


if __name__=='__main__':unittest.main()
