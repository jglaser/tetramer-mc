#!/usr/bin/env python3
"""Small numerical controls for the factorial population covariance calculation."""
import json
import math
from pathlib import Path
import tempfile
import unittest

from analyze_native_factorial_screen import summed_standard_error, target


class FactorialAnalysisTests(unittest.TestCase):
    def analyze(self, weights, counts):
        mean=sum(weights)/len(weights)
        data={
            'all_rows_and_hashes_validated':True,
            'regions':{'native':{'log_normalizer':math.log(mean) if mean else None}},
            'populations':[
                {'samples':10,'cover_volume':1.0,'counts':{'valid':count},
                 'regions':{'native':{'log_normalizer':math.log(weight) if weight else None}}}
                for weight,count in zip(weights,counts)],
            'cpu_seconds':0,'max_population_wall_seconds':0,
            'raw_cloud_points':0,'sample_bytes':0}
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)
            (path/'assessment-streaming.json').write_text(json.dumps(data))
            return target(path)

    def test_proportional_physical_and_hard_estimates_cancel(self):
        result=self.analyze([2.,4.],[2,4])
        self.assertAlmostEqual(result['variance_logQz'],1/9)
        self.assertAlmostEqual(result['variance_logQhard'],1/9)
        self.assertAlmostEqual(result['covariance_logQz_logQhard'],1/9)
        self.assertAlmostEqual(result['variance_log_depletion_enhancement'],0.)

    def test_anticorrelation_is_retained(self):
        result=self.analyze([2.,4.],[4,2])
        self.assertAlmostEqual(result['covariance_logQz_logQhard'],-1/9)
        self.assertAlmostEqual(result['variance_log_depletion_enhancement'],4/9)

    def test_zero_mass_has_no_log_or_uncertainty(self):
        result=self.analyze([0.,0.],[0,0])
        self.assertFalse(result['resolved'])
        self.assertEqual(result['hard_volume'],0.)

    def test_single_population_uncertainty_is_not_estimable(self):
        result=self.analyze([2.],[2])
        self.assertTrue(result['resolved'])
        self.assertIsNone(result['variance_logQz'])
        self.assertIsNone(summed_standard_error([result['variance_logQz'],1.]))
        self.assertAlmostEqual(summed_standard_error([1.,3.]),2.)


if __name__=='__main__':unittest.main()
