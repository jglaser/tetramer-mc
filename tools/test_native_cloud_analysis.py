#!/usr/bin/env python3
import math
import unittest

from analyze_native_cloud_noise import accumulate, describe, empty


class PairedCloudNoiseTests(unittest.TestCase):
    def test_known_paired_variance_including_unconditional_zero(self):
        moment = empty()
        accumulate(moment, math.log(1), math.log(3))
        accumulate(moment, math.log(3), math.log(1))
        result = describe(moment, 3)  # Third proposal has zero target weight.
        self.assertAlmostEqual(math.exp(result["mean"]["log_normalizer"]), 4/3)
        self.assertAlmostEqual(result["mean"]["relative_SE"], .5)
        self.assertAlmostEqual(result["cloud_only_relative_SE"], math.sqrt(2)/4)
        self.assertAlmostEqual(result["observed_cloud_variance_fraction"], .5)

    def test_equal_clouds_have_no_measured_conditional_noise(self):
        moment = empty()
        for value in [1, 2, 3]:
            accumulate(moment, math.log(value), math.log(value))
        result = describe(moment, 4)
        self.assertEqual(result["cloud_only_relative_SE"], 0)
        self.assertEqual(result["observed_cloud_variance_fraction"], 0)
        self.assertIsNone(describe(empty(), 4)["observed_cloud_variance_fraction"])

    def test_large_common_log_scale_and_near_equal_clouds(self):
        baseline, scaled = empty(), empty()
        for a, b in [(1., 2.), (3., 1.), (2., 2.+1e-10)]:
            accumulate(baseline, math.log(a), math.log(b))
            accumulate(scaled, 1000+math.log(a), 1000+math.log(b))
        first, second = describe(baseline, 5), describe(scaled, 5)
        self.assertAlmostEqual(first["cloud_only_relative_SE"], second["cloud_only_relative_SE"], places=10)
        self.assertAlmostEqual(first["observed_cloud_variance_fraction"], second["observed_cloud_variance_fraction"], places=10)


if __name__ == "__main__":
    unittest.main()
