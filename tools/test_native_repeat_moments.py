"""Population-prefix reconstruction against independent dense raw arrays."""
import copy
import math
import unittest

import numpy as np

from analyze_native_tail_reference import NativeTailMoments, KEYS
from analyze_expanded_contact_atlas import nested_prefix_comparison
from native_repeat_moments import recover, pooled_populations


class NativeRepeatMomentTests(unittest.TestCase):
    def population(self, seed, weights):
        one = NativeTailMoments()
        for i, w in enumerate(weights):
            if w:
                # Unequal clouds preserve the original paired-noise moment.
                one.add([3.,4.5,7.,10.,13.,20.][i%6], math.log(w), 0.,
                        [math.log(.5*w),math.log(1.5*w)])
            else: one.add()
        return dict(seed=seed,samples=len(weights),**one.report())

    def test_recovery_and_merge_preserve_raw_weights_zeros_and_noise(self):
        arrays = [np.array([0.,0.,0.,0.,0.,0.]),np.array([1.,3.,0.,9.,2.,5.]),
                  np.array([100.,0.,2.,0.,8.,4.]),np.array([3.,1.,7.,2.,0.,6.])]
        pops = [self.population(i+100,a) for i,a in enumerate(arrays)]
        result = pooled_populations(pops)
        raw = np.concatenate(arrays); row = result['physical']['full']
        self.assertEqual(row['draws'],24)
        self.assertAlmostEqual(math.exp(row['logQ']),raw.mean())
        self.assertAlmostEqual(math.exp(row['log_variance_of_mean']),raw.var(ddof=1)/len(raw))
        self.assertAlmostEqual(row['weight_ESS'],raw.sum()**2/np.sum(raw**2))
        self.assertAlmostEqual(row['maximum_fraction'],raw.max()/raw.sum())
        noise=np.sum(raw**2/4)
        self.assertAlmostEqual(row['paired_cloud_variance_fraction'],noise/(raw.var(ddof=1)*len(raw)))
        for kind in ('physical','hard'):
            self.assertTrue(result['partition_checks'][kind]['complete_row_partition'])
            for key in KEYS:self.assertEqual(result[kind][key]['draws'],24)
        expected=np.std([a.mean() for a in arrays],ddof=1)/math.sqrt(4)/raw.mean()
        self.assertAlmostEqual(row['independent_population_RSE'],expected)

    def test_prefix_difference_uses_shared_row_covariance(self):
        pops=[self.population(i,[0.,1.+i,3.,0.,7.,4.]) for i in range(4)]
        prefix=pooled_populations(pops[:2])['physical']['full']
        full=pooled_populations(pops)['physical']['full']
        contrast=nested_prefix_comparison(prefix,full)
        self.assertAlmostEqual(contrast['estimated_log_covariance_of_means'],full['log_variance_of_mean'])
        self.assertAlmostEqual(contrast['observed_SE_difference_relative_to_full_mean'],full['row_RSE'])

    def test_empty_population_and_invalid_denominators(self):
        p=self.population(1,[0.]*6)
        self.assertEqual(recover(p['physical']['full']).count,6)
        self.assertIsNone(pooled_populations([p])['physical']['full']['logQ'])
        with self.assertRaises(ValueError):pooled_populations([p,p])
        changed=copy.deepcopy(p);changed['physical']['old_r_gt_12']['draws']=5
        with self.assertRaises(ValueError):pooled_populations([changed])


if __name__ == '__main__':unittest.main()
