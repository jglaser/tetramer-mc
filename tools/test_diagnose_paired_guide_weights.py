"""Controller accounting and freeze-order checks; no protein fitting/sampling."""
import copy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
from scipy.special import logsumexp

import diagnose_paired_guide_weights as driver


def guide():
    return dict(defensive_uniform_shell_probability=.5, gaussian_components=[
        dict(weight=.3, mean=[0.]*6, covariance=np.eye(6).tolist()),
        dict(weight=.7, mean=[.7]*6, covariance=(2*np.eye(6)).tolist())])


def rows():
    n = 8; u = np.zeros((n, 6)); u[:, 0] = np.arange(n)/10
    q = driver.log_proposal(u, guide()); j = np.full(n, -.7)
    h = j-q; h[-1] = -np.inf
    pairs = np.column_stack([h, h+math.log(3)])
    classes = np.array([0, 0, 1, 1, 2, 2, -1, -2], np.int8)
    return dict(u=u, log_q=q, log_physical_jacobian=j, h=h, pairs=pairs,
        z=logsumexp(pairs, axis=1)-math.log(2), draw=np.arange(n),
        source_n=np.full(n, n), class_id=classes, support=np.ones(n, bool),
        native=np.isin(classes, [0, 1]), contact=classes >= 0,
        bin_radial=np.arange(n)%3, bin_angular=np.arange(n)%3, bin_orthant=np.arange(n))


class PairedControllerTests(unittest.TestCase):
    def test_component_densities_keep_untruncated_gaussians_and_original_ball(self):
        u = np.zeros((4, 6)); u[:, 0] = [0., 2., 4., 4.001]
        g = guide(); logs, uniform = driver.component_logs(u, g)
        expected = -3*math.log(2*math.pi)-.5*np.sum(u*u, axis=1)
        np.testing.assert_allclose(logs[:, 0], expected, rtol=0, atol=1e-14)
        self.assertTrue(np.isfinite(logs).all()); self.assertTrue(np.isneginf(uniform[-1]))
        self.assertTrue(np.isfinite(uniform[:-1]).all())
        direct = np.logaddexp(math.log(.5)+uniform,
            math.log(.5)+logsumexp(logs+np.log([.3, .7]), axis=1))
        np.testing.assert_allclose(direct, driver.log_proposal(u, g), rtol=0, atol=1e-14)

    def test_weight_change_keeps_all_other_fields_and_rejects_unnormalized_vectors(self):
        g = guide(); g['metadata'] = {'retain': [1, 2]}; before = copy.deepcopy(g)
        changed = driver.with_weights(g, [.8, .2])
        self.assertEqual(g, before); self.assertEqual(driver.geometry(g), driver.geometry(changed))
        self.assertEqual(changed['metadata'], before['metadata'])
        self.assertEqual(changed['defensive_uniform_shell_probability'], .5)
        for weights in ([.8, .3], [0., 1.], [.5], [math.nan, .5], [-.1, 1.1]):
            with self.assertRaises(ValueError): driver.with_weights(g, weights)

    def test_protected_groups_are_exact_not_reselected_from_evaluation(self):
        a = rows(); groups = [dict(name='native', class_id=0, orthant=None),
            dict(name='noentry5', class_id=2, orthant=5)]
        masks = driver.protected_masks(a, groups)
        np.testing.assert_array_equal(np.flatnonzero(masks['native']), [0, 1])
        np.testing.assert_array_equal(np.flatnonzero(masks['noentry5']), [5])
        with self.assertRaises(ValueError): driver.protected_masks(a, groups+groups[:1])
        with self.assertRaises(ValueError): driver.protected_masks(a, [dict(name='absent', class_id=2, orthant=63)])

    def test_one_fit_freezes_weights_before_holdouts_and_preserves_all_attempts(self):
        import fit_paired_guide_weights as core
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); (root/'source').mkdir()
            records = [dict(arm=arm, id=f'r{i:02d}', seed=10*j+i, samples=8,
                role='training' if i < 2 else 'heldout', records='unused', records_sha256='unused')
                for j, arm in enumerate(('bank', 'smc')) for i in range(4)]
            groups = [dict(name=str(cid), class_id=cid, orthant=None) for cid in range(3)]
            plan = dict(datasets=records, training_attempts=32, heldout_attempts=32,
                reference_weights=[.3, .7], historical_noisy_weights=[.3, .7],
                protected_groups=groups, source_and_input_sha256={})
            driver.write_new(root/'plan.json', plan)
            driver.write_new(root/'bank.json', guide()); driver.write_new(root/'dictionary.json', guide())
            events = []
            class Fit:
                weights = (.6, .4)
                def to_dict(self): return dict(success=True, weights=list(self.weights))
            class Problem:
                def metadata(self): return dict(total_attempts=32, supplied_rows=28)
            def load(record):
                if record['role'] == 'heldout':
                    self.assertTrue((root/'model-freeze.json').exists()); self.assertEqual(events.count('fit'), 1)
                events.append(record['role']); return rows()
            def build(**kwargs):
                self.assertNotIn('heldout', events)
                self.assertEqual(kwargs['moment'], 'paired'); self.assertEqual(kwargs['total_attempts'], 32)
                self.assertEqual(kwargs['log_cloud_importance'].shape, (28, 2))
                self.assertEqual(set(kwargs['groups']), {'0', '1', '2'})
                return Problem()
            def fit(problem, **options):
                self.assertNotIn('heldout', events); events.append('fit'); return Fit()
            with mock.patch.object(driver, '__file__', str(root/'source'/Path(driver.__file__).name)), \
                 mock.patch.object(driver, 'validate', return_value=plan), \
                 mock.patch.object(driver, 'load_rows', side_effect=load), \
                 mock.patch.object(core, 'build_problem', side_effect=build), \
                 mock.patch.object(core, 'fit_weights', side_effect=fit):
                result = driver.run(root, 'synthetic')
                with self.assertRaises(FileExistsError): driver.run(root, 'synthetic')
            self.assertEqual(events.count('fit'), 1); self.assertEqual(events.count('heldout'), 4)
            self.assertEqual(result['original_attempts_preserved'], 64)
            self.assertFalse(result['historical_fit_repeated']); self.assertFalse(result['gate_promotion'])
            self.assertEqual(result['protected_validation_rows_read'], 0)
            self.assertEqual(result['fit']['training']['invalid_zero_rows'], 4)
            self.assertEqual(len(result['populations']), 8)
            for pop in result['populations']:
                self.assertEqual(len(pop['groups']), 5*71)
                with np.load(root/pop['rows'], allow_pickle=False) as saved:
                    np.testing.assert_array_equal(saved['draw'], np.arange(8))
                    self.assertTrue(np.isneginf(saved['z'][-1]))

    def test_failed_optimizer_preserves_failure_and_never_opens_holdouts(self):
        import fit_paired_guide_weights as core
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); (root/'source').mkdir()
            record = dict(role='training', id='r00', arm='bank', samples=8)
            plan = dict(datasets=[record, dict(record, role='heldout')], training_attempts=8,
                reference_weights=[.3, .7], protected_groups=[dict(name='native', class_id=0, orthant=None)])
            driver.write_new(root/'dictionary.json', guide()); driver.write_new(root/'bank.json', guide())
            def load(record):
                self.assertEqual(record['role'], 'training'); return rows()
            with mock.patch.object(driver, '__file__', str(root/'source'/Path(driver.__file__).name)), \
                 mock.patch.object(driver, 'validate', return_value=plan), \
                 mock.patch.object(driver, 'load_rows', side_effect=load), \
                 mock.patch.object(core, 'build_problem', return_value=object()), \
                 mock.patch.object(core, 'fit_weights', side_effect=ValueError('failed optimizer')):
                with self.assertRaisesRegex(ValueError, 'failed optimizer'): driver.run(root, 'synthetic')
            self.assertFalse((root/'candidate-weights.json').exists())
            self.assertFalse((root/'model-freeze.json').exists())
            self.assertTrue((root/'failure-freeze.json').exists())
            self.assertEqual(driver.read(root/'status.json')['phase'], 'failed')

    def test_fresh_directory_required_before_input_reads(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); (root/'keep').write_text('preserve')
            with self.assertRaisesRegex(ValueError, 'Fresh'): driver.freeze(root, '/missing')
            self.assertEqual((root/'keep').read_text(), 'preserve')


if __name__ == '__main__': unittest.main()
