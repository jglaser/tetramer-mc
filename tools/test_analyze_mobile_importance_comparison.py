"""Paired-zero accounting, unchanged target and observed cost controls."""
import copy
import math
import unittest
from analyze_mobile_importance_comparison import paired_moments,physical_identity,work_precision


class ImportanceComparisonTests(unittest.TestCase):
    def test_paired_ratio_uses_covariance_and_retains_invalid_zeros(self):
        result=paired_moments([math.log(2.),math.log(6.),-math.inf,-math.inf],
                              [0.,math.log(3.),-math.inf,-math.inf])
        self.assertEqual(result['draws'],4);self.assertEqual(result['nonzero'],2)
        self.assertAlmostEqual(result['log_Qz'],math.log(2.))
        self.assertAlmostEqual(result['log_Q0'],0.)
        self.assertAlmostEqual(result['log_enhancement'],math.log(2.))
        self.assertAlmostEqual(result['log_enhancement_SE'],0.)
        self.assertGreater(result['Qz_relative_SE'],0.)

    def test_allzero_is_unresolved_and_support_mismatch_rejected(self):
        result=paired_moments([-math.inf]*4,[-math.inf]*4)
        self.assertIsNone(result['log_Qz']);self.assertIsNone(result['Qz_relative_SE'])
        with self.assertRaisesRegex(ValueError,'zero supports'):
            paired_moments([0.,-math.inf],[0.,0.])

    def test_target_identity_allows_only_shape_path_relocation(self):
        a=dict(shape='/old/shape.json',activity=.035,scaffold=[1,2])
        b=dict(a,shape='/new/shape.json');region=dict(inner=5,outer=8)
        physical_identity(a,b,region,copy.deepcopy(region))
        b['activity']=.034
        with self.assertRaisesRegex(ValueError,'Physical configuration'):
            physical_identity(a,b,region,region)
        with self.assertRaisesRegex(ValueError,'physical shell'):
            physical_identity(a,a,region,dict(inner=5,outer=9))

    def test_work_precision_uses_relative_variance_not_accepted_counts(self):
        result=work_precision(10.,dict(Qz_relative_SE=.2),dict(Qz_relative_SE=.3))
        self.assertAlmostEqual(result['CPU_times_Qz_row_relative_variance'],.4)
        self.assertAlmostEqual(result['CPU_times_Qz_population_relative_variance'],.9)
        empty=work_precision(10.,dict(Qz_relative_SE=None),dict(Qz_relative_SE=None))
        self.assertIsNone(empty['CPU_times_Qz_row_relative_variance'])


if __name__=='__main__':unittest.main()
