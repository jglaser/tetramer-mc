import copy
import math
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from conditional_ray_proposal import ConditionalRayGuide, union, difference, cubic_mass
from test_entry_shell_proposal import fixture as old_fixture, DIGEST
from analyze_latent_region import audit_importance_rows
from prepare_smc_normalizer_atlas import Density


def fixture(alpha=.5):
    _, region = old_fixture(alpha)
    guide = dict(schema='defensive-conditional-ray-guide-v1', region_sha256=DIGEST,
        defensive_uniform_shell_probability=alpha, inner_radius=.5, widths=[.02, .1, .5],
        interfaces=[dict(moving_members=[[.2, 0, 0], [-.2, 0, 0]],
                         target_world_members=[[.8, 0, 0], [.4, 0, 0]])])
    return guide, region


class ConditionalRayTests(unittest.TestCase):
    def test_interval_algebra_and_cubic_volume(self):
        self.assertEqual(union([(0, 1), (.5, 2), (3, 4), (4, 4)]), [(0, 2), (3, 4)])
        self.assertEqual(difference([(0, 4)], [(1, 2), (3, 5)]), [(0, 1), (2, 3)])
        self.assertEqual(difference([(0, 1)], [(0, 1)]), [])
        self.assertAlmostEqual(cubic_mass([(0, 1), (2, 3)]), 20.)

    def test_ray_membership_matches_direct_geometry(self):
        raw, region = fixture(); g = ConditionalRayGuide(raw, region, DIGEST)
        u, _, _ = g.draw_for_validation(np.random.default_rng(57), 120)
        info = g.diagnostics(u); ray = info['ray']
        for i in range(len(u)):
            radii = np.linspace(0., ray['s'][i], 137)
            positions = ray['origins'][i]+radii[:, None]*ray['slopes'][i]
            errors = []
            for moving, target in g.interfaces:
                members = np.einsum('ij,kj->ki', ray['rotations'][i], moving)
                errors.append(np.linalg.norm(positions[:, None, :]+members-target, axis=2).max(axis=1))
            errors = np.asarray(errors).min(axis=0)
            for k, width in enumerate(g.widths):
                observed = np.array([any(a <= r <= b for a, b in info['intervals'][i][k]) for r in radii])
                expected = (errors > g.inner) & (errors <= g.inner+width)
                np.testing.assert_array_equal(observed, expected)

    def test_correlated_factorization_and_uniform_volume(self):
        raw, region = fixture(); g = ConditionalRayGuide(raw, region, DIGEST)
        u, _, _ = g.draw_for_validation(np.random.default_rng(834), 24000)
        info = g.diagnostics(u); ray = info['ray']; x = g.coordinates(u)
        np.testing.assert_allclose(ray['mu']+(ray['direction']*ray['radius'][:, None])@g.conditional_lower.T, x[:, :3], atol=2e-14)
        self.assertTrue(np.all(np.linalg.norm(u, axis=1) <= g.radius))
        logq = g.log_density(u, np.ones(len(u), bool), g.log_volume)
        ratio = np.exp(-logq-g.log_volume)
        self.assertLessEqual(ratio.max(), 1/g.alpha+1e-12)
        self.assertLess(abs(ratio.mean()-1), max(.025, 6*ratio.std(ddof=1)/len(u)**.5))
        expected = g.radius**2*np.asarray(region['gaussian_chart']['covariances'][0])[3:, 3:]/8
        np.testing.assert_allclose(np.cov(x[:, 3:].T), expected, atol=.035)

    def test_empty_rays_keep_full_uniform_law(self):
        raw, region = fixture(.2)
        raw['interfaces'][0]['target_world_members'] = [[100, 0, 0], [101, 0, 0]]
        g = ConditionalRayGuide(raw, region, DIGEST)
        u, chosen, fallback = g.draw_for_validation(np.random.default_rng(124), 6000)
        self.assertTrue(all(fallback[i] for i in np.flatnonzero(chosen >= 0)))
        np.testing.assert_array_equal(g.log_density(u, np.ones(len(u), bool), g.log_volume), np.full(len(u), -g.log_volume))
        radial_cdf = (np.linalg.norm(u, axis=1)/g.radius)**6
        self.assertLess(abs(radial_cdf.mean()-.5), .015)

    def test_interface_duplicate_and_global_isometry_neutral(self):
        raw, region = fixture(); g = ConditionalRayGuide(raw, region, DIGEST)
        u, _, _ = g.draw_for_validation(np.random.default_rng(14), 300)
        reference = g.log_density(u, np.ones(len(u), bool), g.log_volume)
        duplicate = copy.deepcopy(raw); duplicate['interfaces'] *= 2
        h = ConditionalRayGuide(duplicate, region, DIGEST)
        np.testing.assert_allclose(h.log_density(u, np.ones(len(u), bool), h.log_volume), reference, atol=2e-13)
        rot = Rotation.from_rotvec([.5, .3, -.2]); shift = np.array([12., 33., -7.])
        transformed = copy.deepcopy(region)
        transformed['fixed_neighbor'] = dict(position=shift.tolist(), orientation=rot.as_quat()[[3, 0, 1, 2]].tolist())
        moved = copy.deepcopy(raw)
        for item in moved['interfaces']:
            item['target_world_members'] = (np.asarray(item['target_world_members'])@rot.as_matrix().T+shift).tolist()
        h = ConditionalRayGuide(moved, transformed, DIGEST)
        np.testing.assert_allclose(h.log_density(u, np.ones(len(u), bool), h.log_volume), reference, atol=2e-11)

    def test_uniform_limit_and_strict_schema(self):
        raw, region = fixture(1.); g = ConditionalRayGuide(raw, region, DIGEST)
        u, selected, fallback = g.draw_for_validation(np.random.default_rng(81), 100)
        self.assertTrue(np.all(selected == -1)); self.assertTrue(all(x is None for x in fallback))
        np.testing.assert_array_equal(g.log_density(u, np.ones(len(u), bool), g.log_volume), np.full(len(u), -g.log_volume))
        for patch in [dict(widths=[]), dict(widths=[0]), dict(interfaces=[]), dict(inner_radius=-1),
                      dict(defensive_uniform_shell_probability=0), dict(extra=1)]:
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                ConditionalRayGuide(dict(raw, **patch), region, DIGEST)
        bad = copy.deepcopy(region); bad['minimum_mahalanobis_radius'] = 1.
        with self.assertRaises(ValueError): ConditionalRayGuide(raw, bad, DIGEST)

    def test_independent_raw_audit_and_corrupt_fallback(self):
        raw, region = fixture(); g = ConditionalRayGuide(raw, region, DIGEST)
        u, k, fallback = g.draw_for_validation(np.random.default_rng(7), 80)
        t, r, x = g.world(u); q = g.log_density(u, np.ones(len(u), bool), g.log_volume)
        j = g.log_det-3*math.log(g.ell)-2*math.log(math.pi)-2*np.log1p(np.sum((x[:, 3:]/g.ell)**2, axis=1))
        quats = Rotation.from_matrix(r).as_quat()[:, [3, 0, 1, 2]]
        rows = []
        for i in range(len(u)):
            radius = float(np.linalg.norm(u[i]))
            rows.append(dict(latent=u[i].tolist(), latent_radius=radius, backmapped_latent=u[i].tolist(), backmapped_radius=radius,
                pose=dict(position=t[i].tolist(), orientation=quats[i].tolist()), log_physical_jacobian=float(j[i]),
                log_proposal_density=float(q[i]), shell_valid=True, hard_valid=False, capture_valid=True, region_valid=True,
                clouds=[], log_importance_weight=None, log_hard_weight=None,
                proposal_branch='uniform-shell' if k[i] < 0 else 'conditional-ray',
                proposal_component=None if k[i] < 0 else int(k[i]), selected_ray_fallback=fallback[i]))
        result = audit_importance_rows(rows, region, g, Density(region['gaussian_chart']))
        self.assertEqual(sum(result['branch_counts'].values()), 80)
        index = int(np.flatnonzero(k >= 0)[0]); changed = copy.deepcopy(rows)
        changed[index]['selected_ray_fallback'] = not changed[index]['selected_ray_fallback']
        with self.assertRaises(AssertionError): audit_importance_rows(changed, region, g, Density(region['gaussian_chart']))
        changed = copy.deepcopy(rows); changed[0]['log_proposal_density'] += .1
        with self.assertRaises(AssertionError): audit_importance_rows(changed, region, g, Density(region['gaussian_chart']))


if __name__ == '__main__': unittest.main()
