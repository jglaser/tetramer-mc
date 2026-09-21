"""Independent dense-array controls for full-N inner-shoulder masks."""
import math
import unittest

import numpy as np

from shoulder_peak_moments import ATOMS, MASKS, ShoulderMoments, inner_contains, selected_keys


def signed_covariance(item):
    return item['sign']*math.exp(item['log_absolute_covariance']) if item['sign'] else 0.


class ShoulderPeakMomentsTests(unittest.TestCase):
    def test_original_q_window_is_strict(self):
        for q in (-math.inf, 0., 1., 1.1, 2., math.inf, math.nan):
            self.assertFalse(inner_contains(q))
        self.assertTrue(inner_contains(np.nextafter(1., math.inf)))
        self.assertTrue(inner_contains(np.nextafter(1.1, -math.inf)))

    def test_radial_edges_and_infinite_cayley_seam(self):
        self.assertEqual(selected_keys(0.), ('full', 'ball0p25', 'ball0p5', 'radial_0'))
        self.assertEqual(selected_keys(.25), selected_keys(0.))
        self.assertEqual(selected_keys(np.nextafter(.25, math.inf)),
                         ('full', 'ball0p5', 'radial_1', 'outside0p25'))
        self.assertEqual(selected_keys(.5), selected_keys(.3))
        self.assertEqual(selected_keys(np.nextafter(.5, math.inf)),
                         ('full', 'outside0p25', 'outside0p5'))
        self.assertEqual(selected_keys(math.inf), selected_keys(1.))
        for radius in (None, -.1, -math.inf, math.nan):
            with self.assertRaises(ValueError):
                selected_keys(radius)

    def test_outer_q_and_invalid_rows_keep_full_n_even_with_tiny_radius(self):
        qs = [1., np.nextafter(1., math.inf), 1.05, 1.1, 2., 1.06]
        valid = [True, True, True, True, True, False]
        physical = [1e6, 2., 4., 1e6, 1e6, 1e6]
        hard = [1e5, 3., 5., 1e5, 1e5, 1e5]
        moments = ShoulderMoments()
        for q, hard_valid, weight, volume in zip(qs, valid, physical, hard):
            if hard_valid and inner_contains(q):
                log_weight = math.log(weight)
                moments.add(.1, log_weight, math.log(volume), (log_weight, log_weight))
            else:
                moments.add(.1)
        report = moments.report()
        for kind, weights in (('physical', [0., 2., 4., 0., 0., 0.]),
                              ('hard', [0., 3., 5., 0., 0., 0.])):
            row = report[kind]['full']
            self.assertEqual(row['draws'], 6)
            self.assertEqual(row['nonzero'], 2)
            self.assertAlmostEqual(math.exp(row['logQ']), np.mean(weights))
            self.assertAlmostEqual(math.exp(row['log_variance_of_mean']), np.var(weights, ddof=1)/6)
            self.assertAlmostEqual(row['row_RSE'], np.std(weights, ddof=1)/math.sqrt(6)/np.mean(weights))
            self.assertEqual(row['logQ'], report[kind]['ball0p25']['logQ'])
            self.assertTrue(all(item['draws'] == 6 for item in report[kind].values()))

    def test_nested_and_disjoint_covariance_matches_numpy_for_both_weights(self):
        radii = np.array([.1, .25, .3, .5, .8, math.inf, .05, .4, .7])
        physical = np.array([2., 4., 3., 5., 7., 1., 0., 0., 0.])
        hard = np.array([3., 1., 2., 8., 6., 4., 0., 0., 0.])
        moments = ShoulderMoments()
        for radius, weight, volume in zip(radii, physical, hard):
            if weight:
                value = math.log(weight)
                moments.add(radius, value, math.log(volume), (value, value))
            else:
                moments.add(radius)
        report = moments.report()
        selections = {
            'full': np.ones(len(radii), dtype=bool),
            'ball0p25': radii <= .25,
            'ball0p5': radii <= .5,
            'radial_0': radii <= .25,
            'radial_1': (radii > .25) & (radii <= .5),
            'outside0p25': radii > .25,
            'outside0p5': radii > .5,
        }
        self.assertEqual(set(selections), set(MASKS))
        for kind, weights in (('physical', physical), ('hard', hard)):
            arrays = {key: weights*mask for key, mask in selections.items()}
            for left, a in arrays.items():
                self.assertAlmostEqual(math.exp(report[kind][left]['logQ']), a.mean())
                for right, b in arrays.items():
                    actual = signed_covariance(report['same_row_covariances'][kind][left][right])
                    expected = np.cov(a, b, ddof=1)[0, 1]/len(weights)
                    self.assertAlmostEqual(actual, expected, places=13)
            matrix = report['same_row_covariances'][kind]
            reconstructed = sum(signed_covariance(matrix[a][b]) for a in ATOMS for b in ATOMS)
            self.assertAlmostEqual(reconstructed, np.var(weights, ddof=1)/len(weights), places=13)
            for check in report['partition_checks'][kind].values():
                self.assertTrue(check['complete_row_partition'])
                self.assertLess(check['mass_error'], 1e-12)
                self.assertLess(check['scaled_variance_error'], 1e-12)
        self.assertEqual(report['same_row_covariances']['physical']['ball0p25']['ball0p5']['sign'], 1)
        self.assertEqual(report['same_row_covariances']['physical']['ball0p5']['outside0p5']['sign'], -1)

    def test_nonuniform_original_proposal_is_not_renormalized_by_mask(self):
        probabilities = [.1, .2, .3, .4]
        targets = [2., 3., 0., 5.]
        volumes = [1., 2., 0., 4.]
        radii = [.1, .3, .4, .7]
        moments = ShoulderMoments()
        for probability, target, volume, radius in zip(probabilities, targets, volumes, radii):
            for _ in range(round(10*probability)):
                if target:
                    value = math.log(target/probability)
                    moments.add(radius, value, math.log(volume/probability), (value, value))
                else:
                    moments.add(radius)
        report = moments.report()
        for kind, masses in (('physical', targets), ('hard', volumes)):
            self.assertTrue(all(row['draws'] == 10 for row in report[kind].values()))
            self.assertAlmostEqual(math.exp(report[kind]['full']['logQ']), sum(masses))
            self.assertAlmostEqual(math.exp(report[kind]['ball0p25']['logQ']), masses[0])
            self.assertAlmostEqual(math.exp(report[kind]['ball0p5']['logQ']), sum(masses[:2]))
            self.assertAlmostEqual(math.exp(report[kind]['outside0p5']['logQ']), masses[-1])

    def test_merge_matches_one_stream_and_preserves_cloud_noise(self):
        complete = ShoulderMoments()
        batches = [ShoulderMoments(), ShoulderMoments()]
        weights = [2., 0., 8., 4., 0., 6.]
        radii = [.1, .2, .3, .5, .7, 1.]
        for i, (radius, weight) in enumerate(zip(radii, weights)):
            arguments = (radius, math.log(weight), math.log(weight/2),
                         (math.log(weight/2), math.log(1.5*weight))) if weight else (radius,)
            complete.add(*arguments)
            batches[i % 2].add(*arguments)
        merged = ShoulderMoments().merge(batches[0]).merge(batches[1])
        for kind in ('physical', 'hard'):
            for key in MASKS:
                left, right = complete.values[kind][key], merged.values[kind][key]
                for field in ('count', 'nonzero', 'total', 'square', 'maximum', 'noise'):
                    a, b = getattr(left, field), getattr(right, field)
                    self.assertEqual(a, b) if a == -math.inf else self.assertAlmostEqual(a, b)
        report = merged.report()
        self.assertGreater(report['physical']['full']['paired_cloud_variance_fraction'], 0.)
        self.assertIsNone(report['hard']['full']['paired_cloud_variance_fraction'])

    def test_empty_masks_report_observed_zero_without_mass_bound(self):
        moments = ShoulderMoments()
        moments.add()
        moments.add(.1)
        report = moments.report()
        for kind in ('physical', 'hard'):
            for row in report[kind].values():
                self.assertEqual(row['draws'], 2)
                self.assertIsNone(row['logQ'])
                self.assertIsNone(row['row_RSE'])
                self.assertIn('not a mass upper bound', row['coverage'])
            for row in report['same_row_covariances'][kind].values():
                self.assertTrue(all(item == dict(sign=0, log_absolute_covariance=None) for item in row.values()))

    def test_large_log_weights_remain_finite(self):
        moments = ShoulderMoments()
        for radius, value in [(.1, 1000.), (.3, 1001.), (.8, 999.)]:
            moments.add(radius, value, value-100., (value, value))
        moments.add()
        report = moments.report()
        self.assertTrue(math.isfinite(report['physical']['full']['logQ']))
        self.assertTrue(math.isfinite(report['physical']['full']['row_RSE']))
        self.assertLess(report['partition_checks']['physical']['full']['scaled_variance_error'], 2e-10)

    def test_malformed_rows_fail_before_advancing_n(self):
        moments = ShoulderMoments()
        bad_arguments = [dict(radius=-1.), dict(radius=math.nan), dict(physical=0., hard=0., cloud_pair=(0., 0.)),
                         dict(radius=.1, physical=0.), dict(radius=.1, physical=0., hard=0.),
                         dict(radius=.1, physical=math.inf, hard=0., cloud_pair=(0., 0.)),
                         dict(radius=.1, physical=0., hard=0., cloud_pair=(math.nan, 0.)),
                         dict(cloud_pair=(0., 0.))]
        for arguments in bad_arguments:
            with self.assertRaises(ValueError):
                moments.add(**arguments)
        self.assertEqual(moments.values['physical']['full'].count, 0)
        with self.assertRaises(ValueError):
            moments.report()


if __name__ == '__main__':
    unittest.main()
