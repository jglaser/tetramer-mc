"""Deterministic exterior-tail controls; no sampled poses or physical audits.

The mixed-component fixture isolates active-set arithmetic. The near-seam
fixtures use the actual chart and hard-free-line reconstructor for a sphere.
Only this module's tests run when invoked directly.
"""
import copy
import math
import unittest

import numpy as np

import conditioned_density_audit as split
from test_conditioned_density_audit import SyntheticReference, fixture
from test_physical_hard_free_line_vessel import setup, trace


class MixedReference(SyntheticReference):
    """One fallback-active component and one excluded, inactive component."""

    def __init__(self):
        super().__init__(means=(30., 0.), sigmas=(1., 1.))

    def gaussian_logs(self, u):
        return np.array([math.log(self.weights[0])-3*math.log(2*math.pi)
                         -.5*np.dot(u, u), -math.inf])


def density_record(guide, density):
    return dict(latent=copy.deepcopy(density.latent),
                in_reference_ball=density.in_reference_ball,
                log_latent_density=density.log_latent_density,
                log_physical_jacobian=density.log_physical_jacobian,
                log_physical_density=density.log_physical_density,
                structural_zero=density.structural_zero,
                hard_free_line_density=trace(guide, density))


def near_seam(guide, scalar=1e-10):
    return guide.evaluate(dict(position=[3., 0., 0.],
                               orientation=[scalar, math.sqrt(1-scalar*scalar), 0., 0.]))


class FullVesselConditionedExtremeTests(unittest.TestCase):
    def test_mixed_finite_active_and_negative_infinite_inactive(self):
        r = MixedReference()
        args = fixture([split.line.interval(-1., 1.)], r=r,
                       u=[5., 0., 0., 0., 0., 0.], full=True)
        record, density, guide = args
        result = split.audit_conditioned_density(*args, full_vessel=True)
        self.assertTrue(result['complete'])
        self.assertFalse(density.structural_zero)
        self.assertEqual(result['axes'][0]['component_fallbacks'], [True, False])
        self.assertFalse(result['axes'][0]['allowed'])
        # Query lies outside every interval. Only component zero falls back;
        # the inactive component's -inf must not censor the positive density.
        expected = math.log1p(-r.alpha)+r.gaussian_logs(density.latent)[0]
        self.assertEqual(result['log_latent_density_saved_intervals'], expected)
        self.assertEqual(record['log_latent_density'], expected)
        self.assertEqual(result['maxima']['log_envelope_width'], 0.)
        with self.assertRaisesRegex(ValueError, 'Invalid Gaussian'):
            split.audit_conditioned_density(*args)

    def test_mixed_component_becoming_active_rejects_negative_infinity(self):
        # A positive unconditioned fraction activates the formerly excluded
        # component. Finite total q cannot hide that component's log overflow.
        r = MixedReference()
        r.beta = .5
        args = fixture([split.line.interval(-1., 1.)], r=r,
                       u=[5., 0., 0., 0., 0., 0.], full=True)
        self.assertTrue(math.isfinite(args[1].log_latent_density))
        with self.assertRaisesRegex(ValueError, 'Unrepresentable positive Gaussian'):
            split.audit_conditioned_density(*args, full_vessel=True)

    def test_large_finite_active_near_seam_has_real_log_density(self):
        guide, *_ = setup(alpha=.5, beta=1.)
        for scalar in (1e-6, 1e-10, 1e-50):
            with self.subTest(quaternion_scalar=scalar):
                density = near_seam(guide, scalar)
                u = np.asarray(density.latent)
                self.assertGreater(np.max(abs(u)), 1e6)
                self.assertFalse(density.in_reference_ball)
                self.assertFalse(density.structural_zero)
                self.assertTrue(math.isfinite(density.log_latent_density))
                self.assertEqual(math.exp(density.log_physical_density), 0.)
                self.assertFalse(density.reconstruction.get('conditioning_disabled', False))
                self.assertTrue(all(a['component_fallbacks'] == [True]
                                    for a in density.reconstruction['axes']))
                expected = math.log(.5)-3*math.log(2*math.pi)-.5*float(u@u)
                self.assertEqual(density.log_latent_density, expected)
                self.assertAlmostEqual(density.log_physical_jacobian,
                                       -3*math.log(2.)-2*math.log(math.pi)
                                       +4*math.log(scalar), places=10)
                result = split.audit_conditioned_density(
                    density_record(guide, density), density, guide, full_vessel=True)
                self.assertTrue(result['complete'])
                self.assertEqual(result['log_latent_density_saved_intervals'], expected)

    def test_near_seam_coordinate_and_jacobian_tampering_rejected(self):
        guide, *_ = setup(alpha=.5, beta=1.)
        density = near_seam(guide)
        original = density_record(guide, density)
        bad = copy.deepcopy(original)
        k = int(np.argmax(np.abs(density.latent)))
        # This perturbation PASSES the unchanged full-vessel coordinate
        # tolerance but changes the active Gaussian beyond its log tolerance.
        bad['latent'][k] *= 1.+1.5e-11
        self.assertTrue(np.allclose(bad['latent'], density.latent,
                                    rtol=split.LOG_RTOL, atol=split.LOG_ATOL))
        bad['log_latent_density'] = (math.log(.5)-3*math.log(2*math.pi)
                                     -.5*float(np.dot(bad['latent'], bad['latent'])))
        bad['log_physical_density'] = bad['log_latent_density']-bad['log_physical_jacobian']
        for axis in bad['hard_free_line_density']['axes']:
            axis['axis_log_proposal_density'] = bad['log_latent_density']
        with self.assertRaisesRegex(ValueError, 'coordinate sensitivity'):
            split.audit_conditioned_density(bad, density, guide, full_vessel=True)
        bad = copy.deepcopy(original)
        bad['log_physical_jacobian'] += 1e-4
        with self.assertRaisesRegex(ValueError, 'Physical Jacobian differs'):
            split.audit_conditioned_density(bad, density, guide, full_vessel=True)

    def test_actual_active_gaussian_log_overflow_is_not_a_structural_zero(self):
        # Disable line geometry only for this overflow control: all Gaussian
        # components are then necessarily active. The finite near-seam tests
        # above keep conditioning enabled and traverse its fallback branches.
        guide, *_ = setup(alpha=.5, beta=0.)
        with self.assertRaisesRegex(ValueError, 'Unrepresentable positive Gaussian'):
            near_seam(guide, 1e-160)


if __name__ == '__main__':
    unittest.main()
