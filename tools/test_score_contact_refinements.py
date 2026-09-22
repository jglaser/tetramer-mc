"""Regression tests for leakage, exact mixture defense, and physical scoring."""
import copy
import math
import unittest
import numpy as np
from scipy.special import logsumexp
from score_contact_refinements import (CANDIDATES, STRATA, capped_weights,
    fit_candidates, guide_from_components, log_proposal, mixture_guide, score_rows)


def synthetic_data():
    rng = np.random.default_rng(1357)
    populations = [{'arm': 'bank' if i < 4 else 'wide', 'id': f'r{i}',
        'seed': 70 + i, 'samples': 1000 * (i + 1)} for i in range(8)]
    data = {key: [] for key in ('u', 'population', 'draw', 'class_id', 'log_weight', 'log_q', 'pairs')}
    for population in range(8):
        for class_id in range(3):
            points = rng.normal(size=(40, 6)) * .025
            points[:, 0] += class_id * .3
            points[:, 1] += .7 * points[:, 0]
            logs = rng.normal(size=40)
            logs[population] += 8
            for key, values in [('u', points), ('population', np.full(40, population)),
                ('draw', np.arange(40) + class_id * 40), ('class_id', np.full(40, class_id)),
                ('log_weight', logs), ('log_q', np.full(40, -2)), ('pairs', np.column_stack((logs, logs)))]:
                data[key].append(values)
    return {key: np.concatenate(values) for key, values in data.items()}, populations


class ContactRefinementTests(unittest.TestCase):
    def test_grid_declares_exact_twelve_training_hyperparameters(self):
        self.assertEqual(len(CANDIDATES), 12)
        self.assertEqual({s['floor_multiplier'] for s in CANDIDATES}, {1., .25, .0625})
        self.assertEqual({s['neighbors'] for s in CANDIDATES}, {64, 128})
        self.assertEqual({s['cap'] for s in CANDIDATES}, {1 / 16})

    def test_whole_population_holdout_resists_extreme_poisoning(self):
        data, populations = synthetic_data()
        spec = CANDIDATES[0]
        components, metadata = fit_candidates(data, populations, np.eye(6) * .002, spec, 3)
        self.assertEqual(len(components), 21)
        self.assertEqual({m['class'] for m in metadata}, set(STRATA))
        for item in metadata:
            self.assertNotEqual(item['source_population_index'], 3)
            self.assertTrue(all(n['source_population_index'] != 3 for n in item['neighbors']))
        altered = {key: value.copy() for key, value in data.items()}
        altered['u'][altered['population'] == 3] += 1e6
        altered['log_weight'][altered['population'] == 3] += 1e9
        revised, revised_metadata = fit_candidates(altered, populations, np.eye(6) * .002, spec, 3)
        self.assertEqual(components, revised)
        self.assertEqual(metadata, revised_metadata)

    def test_local_covariance_uses_original_weight_over_source_N_and_cap(self):
        data, populations = synthetic_data()
        components, metadata = fit_candidates(data, populations, np.eye(6) * .002, CANDIDATES[0])
        lookup = {(int(p), int(d)): i for i, (p, d) in enumerate(zip(data['population'], data['draw']))}
        for component, item in zip(components, metadata):
            indices = [lookup[(p['source_population_index'], p['draw'])] for p in item['neighbors']]
            ns = [populations[int(data['population'][i])]['samples'] for i in indices]
            weights, _ = capped_weights(data['log_weight'][indices] - np.log(ns), 1 / 16)
            np.testing.assert_allclose(weights, item['fitting_weights'], rtol=0, atol=1e-15)
            self.assertLessEqual(max(weights), 1 / 16 + 1e-14)
            self.assertGreaterEqual(1 / (weights @ weights), 16 - 1e-10)
            self.assertTrue(np.all(data['class_id'][indices] == item['class_id']))
            offsets = data['u'][indices] - component['mean']
            empirical = offsets.T @ (weights[:, None] * offsets)
            np.testing.assert_allclose(component['covariance'], empirical + np.eye(6) * .002, atol=1e-15)
            np.testing.assert_allclose(component['mean'], data['u'][lookup[(item['source_population_index'], item['draw'])]], atol=0)

    def test_floor_factor_changes_only_floor_not_learned_full_covariance(self):
        data, populations = synthetic_data()
        floor = np.eye(6) * .002
        first, metadata = fit_candidates(data, populations, floor, CANDIDATES[0])
        second, _ = fit_candidates(data, populations, floor, CANDIDATES[1])
        for a, b in zip(first, second):
            self.assertEqual(a['mean'], b['mean'])
            np.testing.assert_allclose(np.asarray(a['covariance']) - b['covariance'], .75 * floor, atol=1e-15)
        self.assertTrue(any(abs(np.asarray(m['empirical_covariance'])[0, 1]) > 1e-6 for m in metadata))

    def test_mean_option_centers_at_capped_neighbor_mean(self):
        data, populations = synthetic_data()
        components, metadata = fit_candidates(data, populations, np.eye(6) * .002, CANDIDATES[6])
        for component, item in zip(components, metadata):
            self.assertEqual(component['mean'], item['weighted_neighbor_mean'])
        self.assertTrue(any(m['mean'] != m['anchor'] for m in metadata))
        self.assertEqual(len(components), 24)
        for name in STRATA:
            self.assertAlmostEqual(sum(c['weight'] for c, m in zip(components, metadata) if m['class'] == name), 1 / 3)

    def test_exact_mixture_preserves_geometry_and_defends_exterior_support(self):
        old = guide_from_components([{'weight': 1., 'mean': [0.] * 6,
            'covariance': (np.eye(6) * 3).tolist()}], 'region')
        candidate = [{'weight': 1., 'mean': [.2] * 6, 'covariance': (np.eye(6) * .01).tolist()}]
        before = copy.deepcopy(old)
        combined = mixture_guide(old, candidate)
        points = np.vstack((np.zeros((1, 6)), np.eye(6) * 5, np.ones((1, 6)) * 100))
        qold = log_proposal(points, old)
        qcand = log_proposal(points, guide_from_components(candidate, 'region'))
        qnew = log_proposal(points, combined)
        np.testing.assert_allclose(qnew, np.logaddexp(qold, qcand) - math.log(2), atol=1e-11)
        self.assertTrue(np.all(qnew >= qold - math.log(2) - 1e-10))
        self.assertEqual(old, before)
        self.assertEqual(combined['gaussian_components'][0]['mean'], old['gaussian_components'][0]['mean'])
        self.assertEqual(combined['gaussian_components'][0]['covariance'], old['gaussian_components'][0]['covariance'])
        self.assertEqual(combined['defensive_uniform_shell_probability'], .5)
        self.assertAlmostEqual(sum(c['weight'] for c in combined['gaussian_components']), 1.)

    def test_moment_uses_cloud_product_actual_source_q_and_all_attempts(self):
        pair = np.log([[2., 8.], [3., 12.]])
        data = {'log_weight': logsumexp(pair, axis=1) - math.log(2), 'pairs': pair,
            'log_q': np.log([.2, .8])}
        target = np.log([.1, .4]); old = np.log([.2, .2])
        score = score_rows(data, np.array([True, True]), 100, {'new': target}, old)
        expected_moment = (2 * 8 * .2 / .1 + 3 * 12 * .8 / .4) / 100
        self.assertAlmostEqual(score['guides']['new']['paired_cloud_log_secondmoment'], math.log(expected_moment))
        self.assertAlmostEqual(score['log_Qz'], math.log((5 + 7.5) / 100))
        self.assertAlmostEqual(score['guides']['new']['weighted_log_q'], (.4 * target[0] + .6 * target[1]))
        expected_old = (2 * 8 * .2 / .2 + 3 * 12 * .8 / .2) / 100
        self.assertAlmostEqual(score['guides']['new']['paired_cloud_secondmoment_ratio_to_old_bank'], expected_moment / expected_old)
        self.assertNotAlmostEqual(expected_moment, (25 * .2 / .1 + 7.5 ** 2 * .8 / .4) / 100)
        doubled = score_rows(data, np.array([True, True]), 200, {'new': target}, old)
        self.assertAlmostEqual(doubled['log_Qz'], score['log_Qz'] - math.log(2))
        self.assertAlmostEqual(doubled['guides']['new']['paired_cloud_log_secondmoment'], math.log(expected_moment / 2))

    def test_scoring_never_caps_extreme_physical_weights(self):
        data = {'log_weight': np.array([0., 100.]), 'pairs': np.array([[0., 0.], [100., 100.]]), 'log_q': np.zeros(2)}
        score = score_rows(data, np.ones(2, bool), 300, {'new': np.array([0., 4.])}, np.zeros(2))
        self.assertAlmostEqual(score['original_weight_ESS'], 1.)
        self.assertAlmostEqual(score['original_largest_weight'], 1.)
        self.assertAlmostEqual(score['guides']['new']['weighted_log_q'], 4.)
        self.assertEqual(data['log_weight'][1], 100.)

    def test_unobserved_stratum_is_missing_evidence_and_retains_denominator(self):
        data = {'log_weight': np.zeros(2), 'pairs': np.zeros((2, 2)), 'log_q': np.zeros(2)}
        score = score_rows(data, np.zeros(2, bool), 400, {'new': np.zeros(2)}, np.zeros(2))
        self.assertIsNone(score['log_Qz'])
        self.assertEqual(score['source_unconditional_samples'], 400)
        self.assertEqual(score['guides'], {})


if __name__ == '__main__':
    unittest.main()
