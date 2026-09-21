"""Reject invalid finite-stratum additions without launching physical calculations."""
import copy
import unittest
from analyze_mobile_competing_outer_weights import check_partition


class OuterComparisonTests(unittest.TestCase):
    def fixture(self):
        base = dict(gaussian_chart={'same': 'chart'}, fixed_neighbor={'same': 'anchor'},
                    physical_fixed_neighbors=['A', 'B'], capture_center=[0., 0., 0.],
                    capture_radius=170., shape_sha256='same', activity=.035, depletant_radius=1.5,
                    physical_metric={'same': 'metric'}, minimum_original_q=1.,
                    minimum_original_q_inclusive=False, maximum_original_q_inclusive=True)
        return [dict(copy.deepcopy(base), minimum_mahalanobis_radius=a, mahalanobis_radius=b)
                for a, b in ((0., 3.), (3., 5.), (5., 8.), (8., 12.))]

    def test_disjoint_sum_rejects_gap_or_double_count(self):
        regions = self.fixture()
        check_partition(regions)
        regions[-1]['minimum_mahalanobis_radius'] = 7.
        with self.assertRaisesRegex(ValueError, 'disjoint'):
            check_partition(regions)

    def test_disjoint_sum_requires_same_chart_and_physical_domain(self):
        for field, value in [('gaussian_chart', {'different': 'chart'}),
                             ('physical_fixed_neighbors', ['A', 'C']), ('activity', .03),
                             ('minimum_original_q_inclusive', True)]:
            with self.subTest(field=field):
                regions = self.fixture()
                regions[-1][field] = value
                with self.assertRaisesRegex(ValueError, 'changed'):
                    check_partition(regions)
        regions = self.fixture()
        regions[-1]['maximum_original_q'] = 100.
        with self.assertRaisesRegex(ValueError, 'upper q'):
            check_partition(regions)


if __name__ == '__main__':
    unittest.main()
