"""Tests of fixed-bank reference augmentation and population-level exclusion."""
import copy
import math
import unittest

import numpy as np

from prepare_reference_contact_bank import (
    augmented_components, fit_reference, guide_from_components, log_proposal,
    predictive_report, validate_reference,
)


def synthetic_reference():
    rng = np.random.default_rng(831)
    populations = [{'arm': 'independent_R5', 'id': f'r{i:02d}', 'seed': 900 + i, 'samples': 16384} for i in range(8)]
    data = {'u': [], 'population': [], 'draw': [], 'log_weight': [], 'log_q': [], 'pairs': []}
    for population in range(8):
        u = rng.normal(scale=.01, size=(24, 6))
        u[:, 0] += .5 + population * .002
        logw = rng.normal(size=24)
        logw[population] = 20 + population
        data['u'].append(u)
        data['population'].append(np.full(24, population))
        data['draw'].append(np.arange(24))
        data['log_weight'].append(logw)
        data['log_q'].append(np.full(24, -3.))
        data['pairs'].append(np.column_stack((logw, logw)))
    return {key: np.concatenate(value) for key, value in data.items()}, populations


def old_bank():
    components, metadata = [], []
    for index in range(48):
        components.append({'weight': 1 / 48, 'mean': [index * .01] + [0.] * 5,
                           'covariance': (np.eye(6) * .2).tolist()})
        metadata.append({'component_index': index, 'class': 'native' if index % 2 == 0 else 'no-entry',
            'arm': 'original', 'population_id': f'r{index // 2}', 'seed': index // 2,
            'draw': index, 'source_population_index': index // 2})
    return components, metadata


class ReferenceContactBankTests(unittest.TestCase):
    def test_mandatory_observed_anchors_and_local_ess_floor(self):
        data, populations = synthetic_reference()
        components, metadata = fit_reference(data, populations, np.eye(6) * 1e-5)
        self.assertEqual(len(components), 8)
        for index, (component, meta) in enumerate(zip(components, metadata)):
            rows = np.flatnonzero(data['population'] == index)
            anchor = rows[np.argmax(data['log_weight'][rows])]
            np.testing.assert_array_equal(component['mean'], data['u'][anchor])
            self.assertEqual(meta['draw'], index)
            self.assertEqual(meta['source_group'], 'R5-reference')
            self.assertEqual(meta['class'], 'native')
            self.assertEqual(meta['neighbor_count'], 64)
            self.assertEqual(meta['distinct_neighbor_count'], 64)
            self.assertGreaterEqual(meta['weight_cap_diagnostic']['capped_weight_ESS'], 16 - 1e-10)
            self.assertAlmostEqual(component['weight'], 1 / 32)

    def test_floor_is_added_exactly_once_to_anchored_moment(self):
        data, populations = synthetic_reference()
        floor = np.eye(6) * .001
        _, metadata = fit_reference(data, populations, floor)
        lookup = {(int(p), int(d)): i for i, (p, d) in enumerate(zip(data['population'], data['draw']))}
        for meta in metadata:
            rows = [lookup[(m['source_population_index'], m['draw'])] for m in meta['neighbors']]
            offset = data['u'][rows] - np.asarray(meta['mean'])
            empirical = offset.T @ (np.asarray(meta['fitting_weights'])[:, None] * offset)
            np.testing.assert_allclose(meta['anchored_empirical_covariance'], empirical, atol=1e-15)
            np.testing.assert_allclose(meta['anchored_covariance'], empirical + floor, atol=1e-15)

    def test_holdout_removes_entire_reference_population(self):
        data, populations = synthetic_reference()
        for heldout in range(8):
            components, metadata = fit_reference(data, populations, np.eye(6) * 1e-5, heldout)
            self.assertEqual(len(components), 7)
            for component, item in zip(components, metadata):
                self.assertNotEqual(item['source_population_index'], heldout)
                self.assertTrue(all(n['source_population_index'] != heldout for n in item['neighbors']))
                self.assertAlmostEqual(component['weight'], 1 / 28)
            altered = {key: value.copy() for key, value in data.items()}
            altered['u'][altered['population'] == heldout] += 100
            altered['log_weight'][altered['population'] == heldout] += 100000
            alternative, _ = fit_reference(altered, populations, np.eye(6) * 1e-5, heldout)
            self.assertEqual(components, alternative)

    def test_exact_56_component_group_allocation_preserves_old_geometry(self):
        data, populations = synthetic_reference()
        old, old_metadata = old_bank()
        saved = copy.deepcopy(old)
        reference, reference_metadata = fit_reference(data, populations, np.eye(6) * 1e-5)
        components, metadata = augmented_components(old, old_metadata, reference, reference_metadata)
        self.assertEqual(len(components), 56)
        self.assertEqual(old, saved)
        self.assertEqual([m['component_index'] for m in metadata], list(range(56)))
        for before, after in zip(old, components[:48]):
            self.assertEqual(before['mean'], after['mean'])
            self.assertEqual(before['covariance'], after['covariance'])
        weights = {}
        for component, item in zip(components, metadata):
            key = (item['source_group'], item['class'])
            weights[key] = weights.get(key, 0) + component['weight'] * .5
        self.assertAlmostEqual(weights[('conditional-ray', 'native')], .125)
        self.assertAlmostEqual(weights[('conditional-ray', 'no-entry')], .25)
        self.assertAlmostEqual(weights[('R5-reference', 'native')], .125)
        self.assertAlmostEqual(sum(c['weight'] for c in components), 1)

    def test_augmented_density_retains_pointwise_half_old_density(self):
        data, populations = synthetic_reference()
        old, old_metadata = old_bank()
        reference, reference_metadata = fit_reference(data, populations, np.eye(6) * 1e-5)
        components, _ = augmented_components(old, old_metadata, reference, reference_metadata)
        u = np.random.default_rng(55).normal(size=(200, 6))
        for width in (1, 4):
            before = log_proposal(u, guide_from_components(old, 'test', width))
            after = log_proposal(u, guide_from_components(components, 'test', width))
            self.assertGreaterEqual(float((after - before).min()), -math.log(2) - 1e-12)
        base = guide_from_components(components, 'test')
        wide = guide_from_components(components, 'test', 4)
        for a, b in zip(base['gaussian_components'], wide['gaussian_components']):
            self.assertEqual(a['mean'], b['mean'])
            self.assertEqual(a['weight'], b['weight'])
            np.testing.assert_array_equal(np.asarray(b['covariance']), 4 * np.asarray(a['covariance']))

    def test_reference_loo_keeps_old_bank_fixed_and_scores_all_original_N(self):
        data, populations = synthetic_reference()
        old, metadata = old_bank()
        result = validate_reference(data, populations, np.eye(6) * 1e-5, old, metadata, 'test')
        self.assertEqual(len(result['folds']), 8)
        self.assertEqual(len(result['reports']), 9)
        self.assertEqual(result['reports'][-1]['source_unconditional_samples'], 8 * 16384)
        for fold in result['folds']:
            self.assertEqual(fold['old_components_preserved'], 48)
            self.assertEqual(fold['reference_components'], 7)
        self.assertEqual(result['selection'].split(':')[0], 'None')

    def test_paired_moment_uses_unconditional_population_denominator(self):
        data = {'u': np.zeros((2, 6)), 'population': np.zeros(2, dtype=int),
            'log_weight': np.log([2., 2.]), 'log_q': np.log([.5, .5]),
            'pairs': np.log([[2., 2.], [2., 2.]])}
        populations = [{'id': 'r0', 'samples': 10}]
        q = {'bank': np.log([.25, .25])}
        report = predictive_report(data, populations, q, 'test')['reports'][0]
        self.assertAlmostEqual(math.exp(report['log_Q_intersection_native']), .4)
        # Two contributions 4*.5/.25 =8; eight original zero rows retained.
        self.assertAlmostEqual(math.exp(report['guides']['bank']['paired_cloud_log_secondmoment_estimate']), 1.6)

    def test_missing_reference_population_fails_closed(self):
        data, populations = synthetic_reference()
        selected = data['population'] != 0
        missing = {key: value[selected] for key, value in data.items()}
        with self.assertRaisesRegex(ValueError, 'Missing mandatory reference anchor'):
            fit_reference(missing, populations, np.eye(6) * 1e-5)


if __name__ == '__main__':
    unittest.main()
