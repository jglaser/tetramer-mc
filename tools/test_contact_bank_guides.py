"""Focused correctness tests for frozen defensive contact-bank construction."""
import copy
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

from prepare_contact_bank_guides import (
    capped_weights, chart_log_jacobian, check_hash, fit_bank, geometric_floor, guide_from_components,
    latent_geometric_floor, log_proposal, nearest_distinct, sha,
)
from scipy.spatial import cKDTree


def synthetic():
    rng = np.random.default_rng(7701)
    populations = [{'arm': 'synthetic', 'id': f'r{i}', 'seed': 100 + i, 'samples': 1000 * (i + 1)} for i in range(3)]
    batch = []
    for population in range(3):
        for class_id in range(2):
            u = rng.normal(size=(40, 6)) * .12
            u[:, 0] += 1.0 * class_id + .3 * population
            logw = rng.normal(size=40)
            logw[3 + population] = 1000 + population
            batch.append({'u': u, 'log_weight': logw,
                'population': np.full(40, population), 'class_id': np.full(40, class_id),
                'draw': np.arange(40) + 40 * class_id})
    return {key: np.concatenate([b[key] for b in batch]) for key in batch[0]}, populations


class ContactBankGuideTests(unittest.TestCase):
    def test_waterfill_extreme_weights_cannot_collapse(self):
        logs = np.r_[10000., np.linspace(0., -2000., 63)]
        weights, report = capped_weights(logs)
        self.assertAlmostEqual(float(sum(weights)), 1, places=13)
        self.assertLessEqual(float(weights.max()), 1 / 16 + 1e-13)
        self.assertGreaterEqual(report['capped_weight_ESS'], 16 - 1e-10)
        self.assertAlmostEqual(report['raw_weight_ESS'], 1)
        self.assertTrue(np.isfinite(weights).all())

    def test_waterfill_equal_weights_and_shift_invariance(self):
        weights, _ = capped_weights(np.zeros(64))
        np.testing.assert_allclose(weights, np.full(64, 1 / 64), atol=1e-15)
        logs = np.linspace(-8, 5, 64)
        np.testing.assert_allclose(capped_weights(logs)[0], capped_weights(logs + 800)[0], atol=3e-15)
        with self.assertRaises(ValueError):
            capped_weights(np.zeros(15))
        with self.assertRaises(ValueError):
            capped_weights(np.array([float('nan')] * 64))

    def test_geometric_floor_is_transformed_from_raw_coordinates(self):
        lower = np.diag([.2, .3, .4, .5, .6, .7])
        lower[5, 0] = .31
        chart = {'angular_length': 55.02283113084892, 'covariances': [(lower @ lower.T).tolist()]}
        floor, raw = latent_geometric_floor(chart)
        np.testing.assert_allclose(lower @ floor @ lower.T, raw, rtol=1e-13, atol=1e-15)
        np.testing.assert_allclose(raw, geometric_floor(), rtol=1e-14)
        self.assertGreater(float(np.linalg.eigvalsh(floor).min()), 0)
        self.assertFalse(np.allclose(floor, raw))

    def test_every_population_class_anchor_survives_ess_one(self):
        data, populations = synthetic()
        floor = np.eye(6) * .001
        components, metadata = fit_bank(data, populations, floor)
        self.assertEqual(len(components), 6)
        for component, meta in zip(components, metadata):
            p = meta['source_population_index']
            class_id = 0 if meta['class'] == 'native' else 1
            rows = np.flatnonzero((data['population'] == p) & (data['class_id'] == class_id))
            expected = rows[np.argmax(data['log_weight'][rows])]
            np.testing.assert_array_equal(component['mean'], data['u'][expected])
            self.assertEqual(meta['draw'], int(data['draw'][expected]))
            self.assertEqual(meta['neighbor_count'], 64)
            self.assertEqual(meta['distinct_neighbor_count'], 64)
            self.assertGreaterEqual(meta['weight_cap_diagnostic']['capped_weight_ESS'], 16 - 1e-10)
            self.assertGreaterEqual(np.linalg.eigvalsh(np.asarray(component['covariance']) - floor).min(), -1e-12)
        for name in ('native', 'no-entry'):
            self.assertAlmostEqual(sum(c['weight'] for c, m in zip(components, metadata) if m['class'] == name), .5)

    def test_heldout_population_absent_from_anchors_and_neighbors(self):
        data, populations = synthetic()
        for held in range(3):
            components, metadata = fit_bank(data, populations, np.eye(6) * .001, held)
            self.assertEqual(len(components), 4)
            for meta in metadata:
                self.assertNotEqual(meta['source_population_index'], held)
                self.assertTrue(all(n['source_population_index'] != held for n in meta['neighbors']))
            altered = {key: value.copy() for key, value in data.items()}
            altered['u'][altered['population'] == held] += 100
            altered['log_weight'][altered['population'] == held] += 100000
            other, _ = fit_bank(altered, populations, np.eye(6) * .001, held)
            self.assertEqual(components, other)

    def test_neighborhood_excludes_exact_duplicates_and_breaks_ties(self):
        points = np.zeros((21, 6))
        points[1:, 0] = np.repeat(np.arange(1, 11), 2)
        data = {'u': points}
        eligible = np.arange(21)
        chosen = nearest_distinct(data, eligible, cKDTree(points), points[0], 10)
        np.testing.assert_array_equal(chosen, [0, 1, 3, 5, 7, 9, 11, 13, 15, 17])
        with self.assertRaises(ValueError):
            nearest_distinct(data, eligible, cKDTree(points), points[0], 12)

    def test_fitting_weights_include_population_size(self):
        data, populations = synthetic()
        _, metadata = fit_bank(data, populations, np.eye(6) * .001)
        meta = metadata[0]
        lookup = {(int(p), int(d)): i for i, (p, d) in enumerate(zip(data['population'], data['draw']))}
        rows = [lookup[(n['source_population_index'], n['draw'])] for n in meta['neighbors']]
        n = np.asarray([populations[int(data['population'][i])]['samples'] for i in rows])
        expected, _ = capped_weights(data['log_weight'][rows] - np.log(n))
        np.testing.assert_allclose(meta['fitting_weights'], expected, rtol=0, atol=1e-15)
        offset = data['u'][rows] - np.asarray(meta['mean'])
        expected_cov = offset.T @ (expected[:, None] * offset) + np.eye(6) * .001
        np.testing.assert_allclose(meta['anchored_covariance'], expected_cov, rtol=1e-12, atol=1e-13)

    def test_full_density_includes_all_components_and_untruncated_tails(self):
        components = [{'weight': .25, 'mean': [0.] * 6, 'covariance': np.eye(6).tolist()},
                      {'weight': .75, 'mean': [.5] * 6, 'covariance': (np.eye(6) * 2).tolist()}]
        guide = guide_from_components(components, 'test')
        u = np.array([[0.] * 6, [1.] * 6, [5., 0, 0, 0, 0, 0]])
        volume = math.pi ** 3 * 4 ** 6 / 6
        expected = .5 / volume * (np.linalg.norm(u, axis=1) <= 4)
        for c in components:
            expected += .5 * c['weight'] * multivariate_normal.pdf(u, mean=c['mean'], cov=c['covariance'])
        np.testing.assert_allclose(np.exp(log_proposal(u, guide)), expected, rtol=1e-13)
        self.assertGreater(float(np.exp(log_proposal(u[-1:], guide)[0])), 0)
        self.assertGreaterEqual(float(log_proposal(u[:1], guide)[0]), math.log(.5 / volume))

    def test_wide_is_exactly_four_times_covariance_and_duplicate_split_neutral(self):
        components = [{'weight': 1., 'mean': [.1] * 6, 'covariance': np.eye(6).tolist()}]
        base = guide_from_components(components, 'test')
        wide = guide_from_components(components, 'test', 4)
        self.assertEqual(base['gaussian_components'][0]['mean'], wide['gaussian_components'][0]['mean'])
        np.testing.assert_array_equal(np.asarray(wide['gaussian_components'][0]['covariance']), 4 * np.eye(6))
        split = copy.deepcopy(base)
        split['gaussian_components'] = [dict(components[0], weight=.5), dict(components[0], weight=.5)]
        u = np.random.default_rng(3).normal(size=(30, 6))
        np.testing.assert_allclose(log_proposal(u, base), log_proposal(u, split), atol=2e-14)

    def test_missing_class_is_a_construction_failure(self):
        data, populations = synthetic()
        data['class_id'][(data['population'] == 0) & (data['class_id'] == 1)] = 0
        with self.assertRaisesRegex(ValueError, 'Missing no-entry anchor'):
            fit_bank(data, populations, np.eye(6) * .001)

    def test_source_hash_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'source.json'
            path.write_text('{}\n')
            original = sha(path)
            self.assertEqual(check_hash(path, original), original)
            path.write_text('{"changed": true}\n')
            with self.assertRaisesRegex(ValueError, 'Source hash mismatch'):
                check_hash(path, original)

    def test_chart_jacobian_and_reference_density_conversion(self):
        region = {'gaussian_chart': {'covariances': [np.eye(6).tolist()],
                  'means': [[0.] * 6], 'angular_length': 2.}}
        u = np.asarray([[0.] * 6, [0., 0, 0, 2, 0, 0]])
        log_j = chart_log_jacobian(u, region)
        np.testing.assert_allclose(np.exp(log_j), [1 / (8 * math.pi ** 2), 1 / (32 * math.pi ** 2)])
        # Under a linear chart rescaling old u = 2 new u, q gains det(2I).
        old_log_j = log_j - 6 * math.log(2)
        old_log_q = np.full(2, -7.)
        old_log_hard = old_log_j - old_log_q
        current_log_q = log_j - old_log_hard
        np.testing.assert_allclose(current_log_q, old_log_q + 6 * math.log(2))
        np.testing.assert_allclose(log_j - current_log_q, old_log_hard)

    def test_paired_cloud_secondmoment_formula_and_zero_denominator(self):
        # Exact finite discrete analogue: q_old != q_new and region omits a
        # nonzero state. Product of independent estimators uses original N.
        old = np.array([.2, .3, .5])
        new = np.array([.5, .2, .3])
        physical = np.array([3., 2., 4.])
        counts = np.array([2, 3, 5])
        region = np.array([True, True, False])
        log_pairs = np.column_stack((np.log(physical / old), np.log(physical / old)))
        terms = log_pairs.sum(axis=1) + np.log(old) - np.log(new)
        estimate = np.exp(logsumexp(terms[region] + np.log(counts[region])) - math.log(counts.sum()))
        self.assertAlmostEqual(estimate, float(sum(physical[region] ** 2 / new[region])), places=11)


if __name__ == '__main__':
    unittest.main()
