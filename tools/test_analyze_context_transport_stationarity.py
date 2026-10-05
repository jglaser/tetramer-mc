"""Deterministic reconstruction controls; no statistical/protein draws."""
import math
import unittest
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from scipy.spatial.transform import Rotation
from scipy.stats import multivariate_normal

from analyze_context_transport_stationarity import Chart, Reference, observations, verify_compiled_chart_source
from dimer_destination_density import inverse_pose, pose_arrays


def fixture():
    def parameter(index):
        lower = np.diag([.7, .8, .6, .4, .5, .45])
        lower[3, 0] = .16
        lower[4, 1] = -.11
        lower[5, 2] = .09
        lower *= 1+.1*index
        return dict(anchor_position=[-1.+2.*index, .2, -.1],
                    anchor_rotation=Rotation.from_rotvec([.2, -.3+.4*index, .15]).as_matrix().tolist(),
                    mean=[0.]*6, covariance=(lower @ lower.T).tolist(), weight=1.)
    original = [parameter(0), parameter(1)]
    unadjusted = []
    for k, inverted in [(0, False), (0, True), (1, False)]:
        center = Chart(original[k], 1.3).decode([0.]*6)
        if inverted:
            center = inverse_pose(center)
        t, r = pose_arrays(center)
        unadjusted.append(dict(anchor_position=t.tolist(), anchor_rotation=r.tolist(), mean=[0.]*6,
                               covariance=original[k]['covariance'], weight=1.))
    adjusted = []
    for i, p in enumerate(unadjusted):
        adjusted.append(dict(anchor_position=(np.asarray(p['anchor_position'])+[.5*(i+1), -.3*i, .2]).tolist(),
                             anchor_rotation=(Rotation.from_rotvec([.1*i, .2, -.15]).as_matrix() @ p['anchor_rotation']).tolist(),
                             mean=[0.]*6, covariance=(np.asarray(p['covariance'])*[.7, 1.2, 1.1][i]**2).tolist(), weight=1.))
    return dict(correlation=.65, angular_length=1.3, original_parameters=original,
                adjusted_parameters=adjusted, unadjusted_parameters=unadjusted,
                branches=[dict(component_index=0, inverted=False, prior=.275),
                          dict(component_index=0, inverted=True, prior=.275),
                          dict(component_index=1, inverted=False, prior=.45)])


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        self.data = fixture()
        self.ref = Reference(self.data)
        self.z = np.array([.4, -.2, .1, .5, -.3, .2])
        self.noise = np.array([.1, .3, -.4, .8, .2, -.1])

    def assert_pose_close(self, a, b):
        at, ar = pose_arrays(a)
        bt, br = pose_arrays(b)
        np.testing.assert_allclose(at, bt, atol=2e-12, rtol=2e-12)
        np.testing.assert_allclose(ar, br, atol=2e-12, rtol=2e-12)

    def test_noncommuting_chart_roundtrip_and_physical_density(self):
        chart = self.ref.original[0]
        pose = chart.decode(self.z)
        np.testing.assert_allclose(chart.encode(pose), self.z, atol=2e-12, rtol=2e-12)
        coordinates = chart.coordinates(self.z)
        c = coordinates[3:]/chart.ell
        log_orientation_measure = -3*math.log(chart.ell)-2*math.log(math.pi)-2*math.log1p(float(c @ c))
        expected = multivariate_normal.logpdf(coordinates, chart.mean, self.data['original_parameters'][0]['covariance'])-log_orientation_measure
        self.assertAlmostEqual(chart.log_density(pose), expected, places=11)

    def test_original_inverse_branch_is_exact_pose_inverse(self):
        direct = self.ref.source(0, self.z)
        inverse = self.ref.source(1, self.z)
        self.assert_pose_close(inverse, inverse_pose(direct))
        logs = self.ref.logs(inverse)
        self.assertAlmostEqual(logs[1], math.log(.275)+self.ref.original[0].log_density(direct), places=11)
        ordinary_child = self.ref.unadjusted[1].log_density(inverse)
        self.assertGreater(abs(ordinary_child-(logs[1]-math.log(.275))), 1e-4)

    def test_selected_chart_involution_and_full_extended_flow(self):
        old = self.ref.source(1, self.z)
        step = self.ref.step('adjusted_full', old, 1, 2, self.noise)
        back = self.ref.step('adjusted_full', step['pose'], 2, 1, step['inverse_noise'])
        self.assert_pose_close(back['pose'], old)
        np.testing.assert_allclose(back['inverse_noise'], self.noise, atol=2e-12, rtol=2e-12)
        self.assertAlmostEqual(step['log_reverse_forward'], -back['log_reverse_forward'], places=10)
        forward = step['old_g']+step['log_forward_label_probability']-.5*float(self.noise @ self.noise)
        reverse = step['new_g']+step['log_reverse_label_probability']-.5*float(step['inverse_noise'] @ step['inverse_noise'])+step['log_extended_jacobian']
        ratio = step['new_g']-step['old_g']+step['log_reverse_forward']
        self.assertAlmostEqual(forward+min(0., ratio), reverse+min(0., -ratio), places=10)

    def test_old_shortcut_omits_a_nonzero_required_factor(self):
        old = self.ref.source(0, self.z)
        step = self.ref.step('adjusted_full', old, 0, 2, self.noise)
        wrong = step['old_g']-step['new_g']
        self.assertGreater(abs(step['log_reverse_forward']-wrong), .01)
        forward = step['old_g']+step['log_forward_label_probability']-.5*float(self.noise @ self.noise)
        reverse = step['new_g']+step['log_reverse_label_probability']-.5*float(step['inverse_noise'] @ step['inverse_noise'])+step['log_extended_jacobian']
        self.assertGreater(abs(forward-reverse), .01)

    def test_jacobian_and_noise_are_both_nontrivial(self):
        step = self.ref.step('adjusted_full', self.ref.source(0, self.z), 0, 2, self.noise)
        self.assertGreater(abs(step['log_extended_jacobian']), .01)
        self.assertGreater(abs(step['log_auxiliary_ratio']), .01)
        weights = np.exp(step['old_logs']-logsumexp(step['old_logs']))
        self.assertAlmostEqual(float(weights.sum()), 1., places=14)
        self.assertAlmostEqual(step['log_forward_label_probability'], math.log(weights[0])+math.log(.45), places=12)

    def test_observables_ignore_quaternion_double_cover(self):
        pose = self.ref.source(1, self.z)
        opposite = dict(position=pose['position'], orientation=[-v for v in pose['orientation']])
        self.assertEqual(observations(pose), observations(opposite))
        self.assertEqual(len(observations(pose)), 12)

    def test_compiled_source_guard_accepts_reviewed_support_check_and_rejects_tampering(self):
        text = (Path(__file__).resolve().parents[1]/'src/basin_involution.rs').read_text()
        verify_compiled_chart_source(text)
        start = text.index('    pub fn encode(')
        modified = text[:start]+text[start:].replace('pub fn encode(', 'pub fn encode(/* changed */', 1)
        with self.assertRaisesRegex(ValueError, 'Unknown compiled'):
            verify_compiled_chart_source(modified)


if __name__ == '__main__':
    unittest.main()
