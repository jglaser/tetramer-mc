"""Pure scoring/accounting tests; no protein fitting, sampling or classification."""
import copy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
from scipy.special import logsumexp

import diagnose_fitted_kernel_shear as driver


def arrays(scale=1.):
    n = 8; u = np.zeros((n, 6)); u[:, 0] = np.arange(n)/10
    classes = np.array([0, 0, 1, 1, 2, 2, -1, -2], np.int8)
    logq = np.linspace(-2., -1., n); logj = np.full(n, -.7)
    h = logj-logq; h[-1] = -np.inf
    pairs = np.column_stack([h+math.log(scale), h+math.log(3*scale)])
    z = logsumexp(pairs, axis=1)-math.log(2)
    return dict(u=u, z=z, h=h, pairs=pairs, log_q=logq, log_physical_jacobian=logj,
        draw=np.arange(n), source_n=np.full(n, n), class_id=classes, support=np.ones(n, bool),
        native=np.isin(classes, [0, 1]), contact=classes >= 0,
        bin_radial=np.arange(n)%3, bin_angular=np.arange(n)%3, bin_orthant=np.arange(n)%64)


def join(*pieces):
    result = {k: np.concatenate([p[k] for p in pieces]) for k in pieces[0]}
    n = len(result['draw']); result['draw'] = np.arange(n); result['source_n'][:] = n
    return result


class DiagnosticTests(unittest.TestCase):
    def test_equal_class_training_uses_linear_physical_weights_only(self):
        a = arrays(); b = arrays(11.)
        u, logw, report = driver.training_data([a, b])
        self.assertEqual(len(u), 12); self.assertEqual(report['attempts'], 16)
        self.assertEqual(report['invalid_zero_rows'], 2); self.assertEqual(report['residual_valid_rows'], 2)
        self.assertAlmostEqual(float(logsumexp(logw)), 0.)
        classes = np.r_[a['class_id'], b['class_id']]; z = np.r_[a['z'], b['z']]
        selected = np.isfinite(z) & (classes >= 0)
        for cid in (0, 1, 2):
            self.assertAlmostEqual(float(np.exp(logsumexp(logw[classes[selected] == cid]))), 1/3)
        self.assertAlmostEqual(math.exp(logw[6]-logw[0]), 11.)
        bad = arrays(); bad['class_id'][bad['class_id'] == 2] = -1
        with self.assertRaisesRegex(ValueError, 'unobserved'): driver.training_data([bad])

    def test_likelihood_gain_and_both_second_moments_keep_all_N(self):
        a = arrays(); base = a['log_q']+.4; warp = base+math.log(2)
        mask = a['class_id'] == 2
        result = driver.evaluate_group(a, base, warp, mask)
        self.assertEqual(result['attempts'], 8); self.assertEqual(result['contributing_rows'], 2)
        self.assertAlmostEqual(result['likelihood']['mean_log_density_gain'], math.log(2))
        for kind, terms in [('two_cloud_noisy', 2*a['z'][mask]), ('paired_physical', a['pairs'][mask].sum(axis=1))]:
            expected = float(logsumexp(terms+a['log_q'][mask]-base[mask])-math.log(8))
            self.assertAlmostEqual(result['second_moments'][kind]['baseline']['log_M2'], expected)
            self.assertAlmostEqual(result['second_moments'][kind]['log_warp_to_baseline_ratio'], -math.log(2))
        # W1=1,W2=3: squared mean=4, cross product=3; never replace either silently.
        self.assertAlmostEqual(result['second_moments']['two_cloud_noisy']['baseline']['log_M2']-
                               result['second_moments']['paired_physical']['baseline']['log_M2'], math.log(4/3))

    def test_all_strata_residuals_and_empty_classes_are_preserved(self):
        a = arrays(); groups = dict(driver.group_masks(a))
        self.assertEqual(len(groups), 5*(1+3+3+64))
        self.assertEqual(int(groups['residual_valid'].sum()), 1)
        self.assertEqual(int(groups['all_valid'].sum()), 7)
        for name in list(driver.CLASSES)+['residual_valid', 'all_valid']:
            for family, size in driver.STRATA.items():
                np.testing.assert_array_equal(sum(groups[f'{name}:{family}:{i}'].astype(int)
                                                   for i in range(size)), groups[name].astype(int))
        empty = driver.evaluate_group(a, a['log_q'], a['log_q'], groups['contact_no_native_entry:orthant:63'])
        self.assertEqual(empty['attempts'], 8)
        self.assertIsNone(empty['likelihood']['mean_log_density_gain'])
        self.assertIsNone(empty['second_moments']['paired_physical']['warped']['log_M2'])
        combined = driver.combine_group([empty, empty])
        self.assertEqual(combined['attempts'], 16)
        self.assertIsNone(combined['second_moments']['paired_physical']['log_warp_to_baseline_ratio'])

    def test_population_summaries_combine_as_sums_not_log_means(self):
        a, b = arrays(), arrays(13.)
        a0, b0 = a['log_q']+.4, b['log_q']-.2
        a1, b1 = a0+.5, b0-.1
        left = driver.evaluate_group(a, a0, a1, a['class_id'] == 2)
        right = driver.evaluate_group(b, b0, b1, b['class_id'] == 2)
        joined = join(a, b)
        direct = driver.evaluate_group(joined, np.r_[a0, b0], np.r_[a1, b1], joined['class_id'] == 2)
        combined = driver.combine_group([left, right])
        for key in direct['likelihood']:
            self.assertAlmostEqual(combined['likelihood'][key], direct['likelihood'][key])
        for kind in direct['second_moments']:
            for name in ('baseline', 'warped'):
                for key in direct['second_moments'][kind][name]:
                    self.assertAlmostEqual(combined['second_moments'][kind][name][key], direct['second_moments'][kind][name][key])
        self.assertAlmostEqual(combined['likelihood']['mean_log_density_gain'], (.5-13*.1)/14)

    def test_saved_cache_zeros_density_and_cloud_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'cache.npz'; a = arrays()
            np.savez_compressed(path, **a)
            record = dict(records=str(path), records_sha256=driver.sha(path), samples=8)
            restored = driver.load_rows(record)
            self.assertEqual(len(restored['draw']), 8); self.assertTrue(np.isneginf(restored['z'][-1]))
            changes = [lambda p: p['source_n'].__setitem__(0, 7),
                       lambda p: p['draw'].__setitem__(0, 1),
                       lambda p: p['h'].__setitem__(0, 0.),
                       lambda p: p['pairs'].__setitem__((0, 1), 42.),
                       lambda p: p['class_id'].__setitem__(-1, 2),
                       lambda p: p['bin_orthant'].__setitem__(0, 64),
                       lambda p: p['z'].__setitem__(-1, 0.)]
            for change in changes:
                bad = copy.deepcopy(a); change(bad); np.savez_compressed(path, **bad)
                record['records_sha256'] = driver.sha(path)
                with self.assertRaises(ValueError): driver.load_rows(record)
            np.savez_compressed(path, **a)
            with self.assertRaisesRegex(ValueError, 'records changed'): driver.load_rows(record)

    def test_angular_first_transform_is_row_permuted_current_cholesky(self):
        lower = np.tril(np.arange(36).reshape(6, 6)/100)+np.eye(6)
        covariance = lower@lower.T
        region = dict(gaussian_chart=dict(covariances=[covariance.tolist()]))
        transform = driver.transform_for(region)
        np.testing.assert_allclose(transform, lower[[3, 4, 5, 0, 1, 2]])
        self.assertAlmostEqual(abs(np.linalg.det(transform)), abs(np.linalg.det(lower)))

    def test_fit_model_is_frozen_before_any_holdout_is_opened(self):
        # Mock only the fitting kernel and external provenance validation. This
        # checks driver scheduling without launching an actual protein fit.
        import fit_kernel_shear
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); (root/'source').mkdir()
            a = arrays(); records = []
            for arm in ('bank', 'smc'):
                for i in range(4):
                    records.append(dict(arm=arm, id=f'r{i:02d}', samples=8, seed=len(records)+1,
                        role='training' if i < 2 else 'heldout', records='unused', records_sha256='unused'))
            plan = dict(datasets=records, fit=driver.FIT, transform=np.eye(6).tolist(),
                training_attempts=32, heldout_attempts=32, input_and_source_sha256={})
            driver.write_new(root/'plan.json', plan); driver.write_new(root/'guide.json', {})
            events = []
            class Model:
                def __init__(self, fitted=False): self.fitted = fitted
                @classmethod
                def from_guide(cls, guide, transform, radius): return cls()
                @classmethod
                def from_dict(cls, value): return cls(value['fitted'])
                def to_dict(self): return dict(fitted=self.fitted)
                def log_density(self, u): return np.full(len(u), .2 if self.fitted else 0.)
            def load(record):
                if record['role'] == 'heldout':
                    self.assertTrue((root/'model-freeze.json').exists())
                    self.assertEqual(events.count('fit'), 1)
                    events.append('heldout')
                else: events.append('training')
                return copy.deepcopy(a)
            def fit(model, u, weights, **options):
                self.assertNotIn('heldout', events); events.append('fit')
                self.assertEqual(options, driver.FIT)
                return Model(True), dict(test_only=True)
            def plot(path, populations): Path(path).write_text('synthetic plot placeholder')
            with mock.patch.object(driver, '__file__', str(root/'source'/Path(driver.__file__).name)), \
                 mock.patch.object(driver, 'validate', return_value=plan), \
                 mock.patch.object(driver, 'load_rows', side_effect=load), \
                 mock.patch.object(driver, 'log_proposal', side_effect=lambda u, guide: np.zeros(len(u))), \
                 mock.patch.object(driver, 'draw_plot', side_effect=plot), \
                 mock.patch.object(fit_kernel_shear, 'KernelMixture', Model), \
                 mock.patch.object(fit_kernel_shear, 'fit_shears', side_effect=fit):
                result = driver.run(root, 'synthetic')
            self.assertEqual(events.count('fit'), 1); self.assertEqual(events.count('heldout'), 4)
            self.assertEqual(result['all_unconditional_draws_preserved'], 64)
            self.assertEqual(result['protected_validation_rows_read'], 0)
            self.assertEqual(result['fit']['holdout_rows_read'], 0)
            self.assertEqual(len(result['populations']), 8)
            self.assertEqual(set(result['aggregates']), {'training_bank', 'training_smc', 'heldout_bank', 'heldout_smc'})

    def test_fresh_output_refusal_precedes_any_source_reads(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); (root/'keep').write_text('retain')
            with self.assertRaisesRegex(ValueError, 'Fresh'): driver.freeze(root, '/nonexistent')
            self.assertEqual((root/'keep').read_text(), 'retain')


if __name__ == '__main__': unittest.main()
