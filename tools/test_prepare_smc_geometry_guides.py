"""Focused geometry fitting, proposal normalization, and authenticated slot tests."""
import copy
import json
import math
from pathlib import Path
import tempfile
import unittest
import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from prepare_smc_geometry_guides import (CANDIDATES, FIT_IDS, HELDOUT_IDS,
    arithmetic_geometry_fit, augment_guide, fit_population_geometry,
    load_authenticated_terminal_geometry, log_proposal, sha, validate_guide)


class GeometryFits(unittest.TestCase):
    def setUp(self):
        self.floor = np.eye(6) * .01

    @staticmethod
    def population(identifier, coordinates, inside):
        u = np.zeros((len(coordinates), 6))
        u[:, 0] = coordinates
        return {'id': identifier, 'u': u, 'old_r5_inside': np.asarray(inside, dtype=bool)}

    def test_repeated_descendants_retain_arithmetic_multiplicity(self):
        population = self.population('r00', [0., 0., 3.], [True] * 3)
        components, metadata = fit_population_geometry([population], self.floor)
        self.assertEqual(components[0]['mean'][0], 1.)
        self.assertAlmostEqual(components[0]['covariance'][0][0], 2.01)
        self.assertEqual(metadata[0]['fitting_slots'], [0, 1, 2])
        self.assertEqual(metadata[0]['population_slot_count'], 3)

    def test_collapsed_geometry_stays_positive_and_empty_group_kept(self):
        population = self.population('r00', [2., 2.], [True, True])
        components, metadata = fit_population_geometry([population], self.floor, split_sign=True)
        self.assertEqual(len(components), 4)
        self.assertEqual(sum(m['empty_group_full_population_fallback'] for m in metadata), 3)
        for component, entry in zip(components, metadata):
            np.testing.assert_allclose(component['covariance'], self.floor)
            self.assertGreater(np.linalg.eigvalsh(component['covariance']).min(), 0)
            self.assertEqual(entry['fitting_slots'], [0, 1])
        self.assertEqual(sum(c['weight'] for c in components), 1.)

    def test_population_and_group_masses_do_not_follow_slot_counts(self):
        populations = [self.population('r00', [-1., 1.], [True, False]),
                       self.population('r01', [-1.] * 7 + [1.], [False] * 7 + [True])]
        components, metadata = fit_population_geometry(populations, self.floor)
        self.assertEqual([c['weight'] for c in components], [.25] * 4)
        for identifier in ('r00', 'r01'):
            self.assertEqual(sum(c['weight'] for c, m in zip(components, metadata)
                                 if m['population_id'] == identifier), .5)

    def test_partition_covers_every_slot_once_and_zero_is_nonnegative(self):
        population = self.population('r00', [-2., -1., 0., 1.], [True, False, True, False])
        _, metadata = fit_population_geometry([population], self.floor, split_sign=True)
        self.assertEqual(sorted(i for m in metadata for i in m['group_slots']), list(range(4)))
        zero_group = next(m for m in metadata if 2 in m['group_slots'])
        self.assertIs(zero_group['original_u0_nonnegative'], True)
        self.assertIs(zero_group['inside_unfiltered_historical_R5_ball'], True)

    def test_labels_and_ancestry_never_enter_fit(self):
        population = self.population('r00', [-1., 2.], [True, False])
        expected = fit_population_geometry([population], self.floor)
        population.update(native=[True, False], class_id=[42, 9], log_weight=[1e99, -1e99],
                          initial_ancestor=[0, 0])
        self.assertEqual(fit_population_geometry([population], self.floor), expected)

    def test_invalid_geometry_or_floor_fails_instead_of_dropping_slots(self):
        for u, floor in [(np.empty((0, 6)), self.floor), (np.full((2, 6), np.nan), self.floor),
                         (np.zeros((2, 6)), np.zeros((6, 6)))]:
            with self.subTest(shape=u.shape), self.assertRaises((ValueError, np.linalg.LinAlgError)):
                arithmetic_geometry_fit(u, floor)


class DefensiveMixture(unittest.TestCase):
    def setUp(self):
        self.old = {'schema': 'defensive-latent-shell-guide-v1', 'region_sha256': 'synthetic',
            'defensive_uniform_shell_probability': .5,
            'gaussian_components': [{'weight': .3, 'mean': [0.] * 6, 'covariance': np.eye(6).tolist()},
                {'weight': .7, 'mean': [2.] * 6, 'covariance': (np.eye(6) * 2).tolist()}]}
        self.smc = [{'weight': 1., 'mean': [-1.] * 6, 'covariance': (np.eye(6) * .2).tolist()}]

    def test_exact_mixture_matches_independent_normalized_density(self):
        guide = augment_guide(self.old, self.smc, 4.)
        points = np.array([[0.] * 6, [1.] * 6, [4.] * 6, [-2.] * 6])
        log_uniform = np.where(np.linalg.norm(points, axis=1) <= 4,
            -math.log(math.pi ** 3 / 6 * 4 ** 6), -np.inf)
        old_gaussian = logsumexp([math.log(c['weight']) +
            multivariate_normal.logpdf(points, mean=c['mean'], cov=c['covariance'])
            for c in self.old['gaussian_components']], axis=0)
        smc_gaussian = multivariate_normal.logpdf(points, mean=[-1.] * 6, cov=np.eye(6) * .8)
        expected = logsumexp(np.array([math.log(.5) + log_uniform,
            math.log(.25) + old_gaussian, math.log(.25) + smc_gaussian]), axis=0)
        np.testing.assert_allclose(log_proposal(points, guide), expected, rtol=0, atol=1e-12)

    def test_old_components_unchanged_and_only_new_covariance_widened(self):
        original = copy.deepcopy(self.old)
        guide = augment_guide(self.old, self.smc, 4.)
        self.assertEqual(original, self.old)
        for old, new in zip(self.old['gaussian_components'], guide['gaussian_components']):
            self.assertEqual(new['mean'], old['mean'])
            self.assertEqual(new['covariance'], old['covariance'])
            self.assertEqual(new['weight'], .5 * old['weight'])
        np.testing.assert_allclose(guide['gaussian_components'][-1]['covariance'], np.eye(6) * .8)
        validate_guide(guide)

    def test_pointwise_half_old_bound_inside_and_outside_ball(self):
        guide = augment_guide(self.old, self.smc)
        points = np.random.default_rng(7801).normal(size=(1000, 6)) * 5
        difference = log_proposal(points, guide) - log_proposal(points, self.old)
        self.assertGreaterEqual(difference.min(), math.log(.5) - 1e-12)
        self.assertTrue(np.isfinite(log_proposal(points, guide)).all())

    def test_fixed_candidates_and_population_split(self):
        self.assertEqual(FIT_IDS, ('r00', 'r01'))
        self.assertEqual(HELDOUT_IDS, ('r02', 'r03'))
        self.assertEqual([(c['split_sign'], c['covariance_multiplier']) for c in CANDIDATES],
                         [(False, 1.), (False, 4.), (True, 1.), (True, 4.)])
        with self.assertRaises(ValueError):
            augment_guide(self.old, self.smc, 2.)


class AuthenticatedMapping(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        region = {'mahalanobis_radius': 4., 'fixed_neighbor': {'position': [0.] * 3,
            'orientation': [1., 0., 0., 0.]}, 'gaussian_chart': {'angular_length': 1.,
            'means': [[0.] * 6], 'covariances': [np.eye(6).tolist()],
            'anchors': [{'position': [0.] * 3, 'rotation': np.eye(3).tolist()}]}}
        self.current = self.root / 'current.json'
        self.current.write_text(json.dumps(region))
        region['mahalanobis_radius'] = 5.
        # An impossible native-q filter is deliberately ignored by Chart.
        region['minimum_original_q'] = 1e100
        self.historical = self.root / 'historical.json'
        self.historical.write_text(json.dumps(region))
        self.plan = {'current_region': str(self.current), 'historical_r5_region': str(self.historical),
            'input_sha256': {str(p): sha(p) for p in (self.current, self.historical)}, 'populations': []}
        for index, identifier in enumerate(FIT_IDS):
            path = self.root / (identifier + '.json')
            summary = {'complete': True, 'completed_stage': 128, 'manifest_sha256': 'manifest',
                'terminal_particles': [{'initial_ancestor': 0, 'latent': [float(index)] + [0.] * 5,
                    'pose': {'position': [float(index), 0., 0.], 'orientation': [1., 0., 0., 0.]}}] * 2}
            path.write_text(json.dumps(summary))
            self.plan['populations'].append({'id': identifier, 'seed': index, 'summary': str(path),
                'summary_sha256': sha(path), 'manifest_sha256': 'manifest', 'completed_stage': 128,
                'expected_slots': 2})

    def test_validated_slots_include_duplicates_and_ignore_old_filters(self):
        populations, validations = load_authenticated_terminal_geometry(self.plan, FIT_IDS)
        self.assertEqual([len(p['u']) for p in populations], [2, 2])
        self.assertTrue(all(p['old_r5_inside'].all() for p in populations))
        self.assertEqual([v['distinct_latent_coordinates'] for v in validations], [1, 1])
        self.assertTrue(all(v['all_slots_retained'] for v in validations))

    def test_changed_summary_hash_fails(self):
        Path(self.plan['populations'][0]['summary']).write_text('{}')
        with self.assertRaisesRegex(ValueError, 'Hash mismatch'):
            load_authenticated_terminal_geometry(self.plan, FIT_IDS)

    def test_authenticated_but_mismapped_pose_fails(self):
        identity = self.plan['populations'][0]
        path = Path(identity['summary'])
        summary = json.loads(path.read_text())
        summary['terminal_particles'][0]['pose']['position'][0] = 1.
        path.write_text(json.dumps(summary))
        identity['summary_sha256'] = sha(path)
        with self.assertRaisesRegex(ValueError, 'mapping mismatch'):
            load_authenticated_terminal_geometry(self.plan, FIT_IDS)

    def test_missing_population_fails_instead_of_reducing_fit_allocation(self):
        self.plan['populations'].pop()
        with self.assertRaisesRegex(ValueError, 'Missing or reordered'):
            load_authenticated_terminal_geometry(self.plan, FIT_IDS)

    def test_undeclared_population_selection_fails(self):
        with self.assertRaisesRegex(ValueError, 'declared fit or held-out'):
            load_authenticated_terminal_geometry(self.plan, ('r00', 'r02'))


if __name__ == '__main__':
    unittest.main()
