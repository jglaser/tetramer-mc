"""Finite reference laws for full-N overlapping masks and covariance."""
import math
import unittest

import numpy as np

from analyze_contact_atlas import AtlasMoments, KEYS, OLD_PARTITION, PEAK_PARTITION, selected_keys


class ContactAtlasAuditTests(unittest.TestCase):
    def test_boundaries_cover_whole_space_once_per_partition(self):
        for peak, old in ((0., 0.), (.5, 4.), (1., 8.), (2., 16.), (3., 32.), (math.inf, math.inf)):
            selected = selected_keys(peak, old)
            self.assertEqual(sum(key in selected for key in PEAK_PARTITION), 1)
            self.assertEqual(sum(key in selected for key in OLD_PARTITION), 1)
        self.assertIn('peak_r_le_0p5', selected_keys(.5, 4.))
        self.assertIn('peak_0p5_lt_r_le_1', selected_keys(math.nextafter(.5, 1.), 4.))
        self.assertIn('old_r_gt_32', selected_keys(3., math.nextafter(32., math.inf)))
        with self.assertRaises(ValueError): selected_keys(math.nan, 0.)
        with self.assertRaises(ValueError): selected_keys(0., -1.)

    def test_direct_masked_means_variances_and_paired_clouds(self):
        radius = [.2, .5, .8, 1., 1.5, 2., 3., math.inf, None]
        old = [2., 5., 10., 20., 40., 32., 4., math.inf, None]
        pair = np.array([[1., 3.], [4., 2.], [2., 8.], [8., 4.], [5., 7.], [2., 2.], [6., 4.], [1., 1.], [0., 0.]])
        weights = pair.mean(axis=1); hard = np.array([1., 2., 2., 1., 3., 5., 1., 2., 0.])
        moments = AtlasMoments(); n = len(radius)
        for rad, previous, w, h, clouds in zip(radius, old, weights, hard, pair):
            if rad is None: moments.add()
            else: moments.add(rad, previous, math.log(w), math.log(h), np.log(clouds).tolist())
        result = moments.report()
        for key in KEYS:
            mask = np.array([r is not None and key in selected_keys(r, o) for r, o in zip(radius, old)])
            w = np.where(mask, weights, 0.); h = np.where(mask, hard, 0.)
            measured = result['physical'][key]
            self.assertEqual(measured['draws'], n)
            self.assertAlmostEqual(math.exp(measured['logQ']), w.mean())
            self.assertAlmostEqual(math.exp(measured['log_variance_of_mean']), w.var(ddof=1)/n)
            self.assertAlmostEqual(measured['row_RSE'], w.std(ddof=1)/math.sqrt(n)/w.mean())
            self.assertAlmostEqual(math.exp(result['hard'][key]['logQ']), h.mean())
            covariance = np.cov(w, h, ddof=1)[0, 1]/n/(w.mean()*h.mean())
            self.assertAlmostEqual(result['physical_hard_paired_statistics'][key]['relative_covariance_of_means'], covariance)
            noise = np.where(mask, (pair[:, 0]-pair[:, 1])**2/4, 0.).sum()/n**2
            self.assertAlmostEqual(measured['paired_cloud_variance_fraction'] or 0., noise/(w.var(ddof=1)/n))
        for kind in ('physical', 'hard'):
            for partition in (PEAK_PARTITION, OLD_PARTITION):
                self.assertAlmostEqual(sum(math.exp(result[kind][k]['logQ']) for k in partition),
                                       math.exp(result[kind]['full']['logQ']))
        self.assertLess(result['partition_checks']['physical']['old_weighted_chart']['relative_variance_error'], 1e-12)

    def test_nonuniform_proposal_uses_original_density_and_full_denominator(self):
        probability = np.array([.1, .2, .3, .4]); target = np.array([3., 2., 7., 1.])
        radii, old = [.2, .8, 1.5, 4.], [2., 6., 10., 40.]
        moments = AtlasMoments()
        for i, count in enumerate(np.rint(10*probability).astype(int)):
            for _ in range(count):
                value = math.log(target[i]/probability[i])
                moments.add(radii[i], old[i], value, -math.log(probability[i]), [value]*2)
        result = moments.report()
        self.assertAlmostEqual(math.exp(result['physical']['full']['logQ']), target.sum())
        self.assertAlmostEqual(math.exp(result['physical']['peak_r_le_0p5']['logQ']), target[0])
        self.assertEqual(result['physical']['peak_r_le_0p5']['draws'], 10)
        self.assertEqual(result['physical']['peak_r_le_0p5']['nonzero'], 1)
        self.assertAlmostEqual(math.exp(result['hard']['full']['logQ']), 4.)

    def test_empty_supported_masks_remain_unresolved(self):
        moments = AtlasMoments()
        moments.add(); moments.add()
        result = moments.report()
        for key in KEYS:
            self.assertIsNone(result['physical'][key]['logQ'])
            self.assertIsNone(result['physical'][key]['row_RSE'])
            self.assertIn('not a mass upper bound', result['physical'][key]['coverage'])
        self.assertIsNone(result['physical_hard_paired_statistics']['full']['relative_covariance_of_means'])

    def test_merging_population_moments_preserves_shared_row_variance(self):
        whole, first, second = AtlasMoments(), AtlasMoments(), AtlasMoments()
        for i, value in enumerate((1., 3., 2., 5., 4., 8.)):
            args = (i/2, 2.**i, math.log(value), 0., [math.log(value)]*2)
            whole.add(*args); (first if i < 3 else second).add(*args)
        first.merge(second)
        direct, merged = whole.report(), first.report()
        for key in KEYS:
            for field in ('logQ', 'row_RSE', 'weight_ESS'):
                a, b = direct['physical'][key][field], merged['physical'][key][field]
                if a is None: self.assertIsNone(b)
                else: self.assertAlmostEqual(a, b)


if __name__ == '__main__': unittest.main()
