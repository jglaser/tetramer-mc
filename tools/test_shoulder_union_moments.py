"""Independent dense-array controls for overlapping three-center masks."""
import itertools
import math
import unittest

import numpy as np

from shoulder_peak_moments import inner_contains
from shoulder_union_moments import CENTERS, MASKS, UnionMoments, selected_keys


def oracle_masks(radii):
    """Direct Boolean masks, independent of the accumulator's atomic states."""
    radii = np.asarray(radii)
    masks = {'full': np.ones(len(radii), dtype=bool)}
    earlier_outside = np.ones(len(radii), dtype=bool)
    for index, center in enumerate(('direct', 'mixture', 'geometry')):
        inner = radii[:, index] <= .25
        ball = radii[:, index] <= .5
        shell = (radii[:, index] > .25) & ball
        masks[f'{center}_ball0p25'] = inner
        masks[f'{center}_ball0p5'] = ball
        masks[f'{center}_radial_1'] = shell
        masks[f'{center}_assigned_ball0p25'] = inner & earlier_outside
        masks[f'{center}_assigned_ball0p5'] = ball & earlier_outside
        masks[f'{center}_assigned_radial_1'] = shell & earlier_outside
        earlier_outside &= ~ball
    masks['union'] = np.any(radii <= .5, axis=1)
    masks['outside_union'] = np.all(radii > .5, axis=1)
    return masks


def covariance_matrix(report, kind):
    return np.array([[item['sign']*math.exp(item['log_absolute_covariance']) if item['sign'] else 0.
                      for item in report['same_row_covariances'][kind][left].values()] for left in MASKS])


def add_weights(moments, radii, physical, hard):
    for radii_row, weight, volume in zip(radii, physical, hard):
        if weight:
            value = math.log(weight)
            moments.add(radii_row, value, math.log(volume), (value, value))
        else:
            moments.add(radii_row)


class ShoulderUnionMomentsTests(unittest.TestCase):
    def test_closed_inner_and_outer_edges_and_infinite_seams(self):
        outside = math.nextafter(.5, math.inf)
        for index, center in enumerate(CENTERS):
            radii = [math.inf]*3
            radii[index] = .25
            keys = selected_keys(radii)
            self.assertIn(f'{center}_ball0p25', keys)
            self.assertIn(f'{center}_assigned_ball0p25', keys)
            self.assertNotIn(f'{center}_radial_1', keys)
            radii[index] = math.nextafter(.25, math.inf)
            keys = selected_keys(radii)
            self.assertNotIn(f'{center}_ball0p25', keys)
            self.assertIn(f'{center}_assigned_radial_1', keys)
            radii[index] = .5
            self.assertIn(f'{center}_assigned_ball0p5', selected_keys(radii))
            radii[index] = outside
            self.assertEqual(selected_keys(radii), ('full', 'outside_union'))
        self.assertEqual(selected_keys((math.inf, math.inf, math.inf)), ('full', 'outside_union'))

    def test_earlier_outer_ball_excludes_later_inner_assignment(self):
        keys = selected_keys((.49, .1, .1))
        self.assertIn('mixture_ball0p25', keys)
        self.assertIn('geometry_ball0p25', keys)
        self.assertIn('direct_assigned_radial_1', keys)
        self.assertNotIn('mixture_assigned_ball0p25', keys)
        self.assertNotIn('geometry_assigned_ball0p25', keys)
        keys = selected_keys((.51, .5, .1))
        self.assertIn('mixture_assigned_radial_1', keys)
        self.assertNotIn('geometry_assigned_ball0p25', keys)

    def test_all_boundary_combinations_match_disjoint_priority_oracle(self):
        grid = (0., .25, math.nextafter(.25, math.inf), .5, math.nextafter(.5, math.inf), math.inf)
        radii = list(itertools.product(grid, repeat=3))
        masks = oracle_masks(radii)
        self.assertEqual(tuple(masks), tuple(MASKS))
        for index, row in enumerate(radii):
            keys = selected_keys(row)
            self.assertEqual(set(keys), {key for key, flags in masks.items() if flags[index]})
            assigned = sum(f'{center}_assigned_ball0p5' in keys for center in CENTERS)
            self.assertEqual(assigned, int('union' in keys))
            self.assertEqual(int('union' in keys)+int('outside_union' in keys), 1)

    def test_every_cross_center_covariance_and_partition_matches_numpy(self):
        radii = list(itertools.product((.1, .4, .8), repeat=3))
        radii += [(.25, .5, math.inf), (math.inf, .25, .5), (.5, math.inf, .25), (.1, .1, .1), (.8, .8, .8)]
        physical = np.array([1.+(i*7 % 13) for i in range(len(radii))])
        hard = np.array([1.+(i*11 % 17)/3 for i in range(len(radii))])
        physical[-2:] = hard[-2:] = 0.
        moments = UnionMoments()
        add_weights(moments, radii, physical, hard)
        report = moments.report()
        masks = oracle_masks(radii)
        for kind, weights in (('physical', physical), ('hard', hard)):
            samples = np.column_stack([weights*masks[key] for key in MASKS])
            expected = np.cov(samples, rowvar=False, ddof=1)/len(weights)
            actual = covariance_matrix(report, kind)
            np.testing.assert_allclose(actual, expected, rtol=3e-12, atol=2e-13)
            for index, key in enumerate(MASKS):
                row = report[kind][key]
                self.assertEqual(row['draws'], len(weights))
                self.assertAlmostEqual(math.exp(row['logQ']), samples[:, index].mean())
                self.assertAlmostEqual(math.exp(row['log_variance_of_mean']), expected[index, index])
                self.assertAlmostEqual(row['row_RSE'], math.sqrt(expected[index, index])/samples[:, index].mean())
            for check in report['partition_checks'][kind].values():
                self.assertTrue(check['complete_row_partition'])
                self.assertLess(check['mass_error'], 1e-12)
                self.assertLess(check['scaled_variance_error'], 1e-12)
            assigned = [list(MASKS).index(f'{center}_assigned_ball0p5') for center in CENTERS]
            union = list(MASKS).index('union')
            self.assertAlmostEqual(actual[np.ix_(assigned, assigned)].sum(), expected[union, union])

    def test_outer_q_and_invalid_zeros_keep_full_original_denominator(self):
        qs = [1., 1.1, 1.05, 1.07, 2., math.nextafter(1., math.inf)]
        valid = [True, True, True, False, True, True]
        moments = UnionMoments()
        expected = []
        for index, (q, hard_valid) in enumerate(zip(qs, valid)):
            weight = index+1.
            if inner_contains(q) and hard_valid:
                value = math.log(weight)
                moments.add((.1, .1, .1), value, 0., (value, value))
                expected.append(weight)
            else:
                moments.add((.1, .1, .1))
                expected.append(0.)
        report = moments.report()
        self.assertTrue(all(row['draws'] == len(qs) for kind in ('physical', 'hard') for row in report[kind].values()))
        self.assertEqual(report['physical']['full']['nonzero'], 2)
        self.assertAlmostEqual(math.exp(report['physical']['full']['logQ']), np.mean(expected))
        self.assertAlmostEqual(math.exp(report['physical']['union']['log_variance_of_mean']), np.var(expected, ddof=1)/len(qs))
        self.assertEqual(report['physical']['direct_assigned_ball0p5']['logQ'], report['physical']['union']['logQ'])
        self.assertIsNone(report['physical']['mixture_assigned_ball0p5']['logQ'])

    def test_nonuniform_law_keeps_original_density_in_overlaps_and_complement(self):
        probabilities = [.1, .2, .3, .4]
        physical = [1., 2., 0., 4.]
        hard = [2., 1., 0., 3.]
        radii = [(.1, .1, math.inf), (.6, .3, .1), (.8, .8, .8), (.8, .8, .6)]
        moments = UnionMoments()
        for probability, weight, volume, row in zip(probabilities, physical, hard, radii):
            for _ in range(round(10*probability)):
                if weight:
                    value = math.log(weight/probability)
                    moments.add(row, value, math.log(volume/probability), (value, value))
                else:
                    moments.add(row)
        report = moments.report()
        for kind, masses in (('physical', physical), ('hard', hard)):
            self.assertAlmostEqual(math.exp(report[kind]['full']['logQ']), sum(masses))
            self.assertAlmostEqual(math.exp(report[kind]['union']['logQ']), sum(masses[:2]))
            self.assertAlmostEqual(math.exp(report[kind]['outside_union']['logQ']), masses[3])
            self.assertAlmostEqual(math.exp(report[kind]['mixture_ball0p5']['logQ']), sum(masses[:2]))
            self.assertAlmostEqual(math.exp(report[kind]['mixture_assigned_ball0p5']['logQ']), masses[1])
            self.assertIsNone(report[kind]['geometry_assigned_ball0p5']['logQ'])
            self.assertTrue(all(row['draws'] == 10 for row in report[kind].values()))

    def test_merge_preserves_intersection_squares_and_original_cloud_noise(self):
        radii = list(itertools.product((.1, .4, .8), repeat=3))
        complete = UnionMoments()
        batches = [UnionMoments(), UnionMoments()]
        for index, row in enumerate(radii):
            weight = index+1.
            arguments = (row, math.log(weight), math.log(weight/2),
                         (math.log(weight/2), math.log(1.5*weight)))
            complete.add(*arguments)
            batches[index % 2].add(*arguments)
        complete.add()
        batches[0].add()
        merged = UnionMoments().merge(batches[0]).merge(batches[1])
        direct, combined = complete.report(), merged.report()
        for kind in ('physical', 'hard'):
            np.testing.assert_allclose(covariance_matrix(combined, kind), covariance_matrix(direct, kind), rtol=3e-12, atol=1e-12)
            for key in MASKS:
                for field in ('count', 'nonzero', 'total', 'square', 'maximum', 'noise'):
                    a = getattr(complete.values[kind][key], field)
                    b = getattr(merged.values[kind][key], field)
                    self.assertEqual(a, b) if a == -math.inf else self.assertAlmostEqual(a, b)
        self.assertGreater(combined['physical']['union']['paired_cloud_variance_fraction'], 0.)
        self.assertIsNone(combined['hard']['union']['paired_cloud_variance_fraction'])

    def test_zero_observations_and_large_log_contributions(self):
        empty = UnionMoments()
        empty.add()
        empty.add((0., 0., 0.))
        report = empty.report()
        for kind in ('physical', 'hard'):
            self.assertTrue(all(row['draws'] == 2 and row['logQ'] is None and row['row_RSE'] is None for row in report[kind].values()))
            np.testing.assert_array_equal(covariance_matrix(report, kind), np.zeros((len(MASKS), len(MASKS))))
        large = UnionMoments()
        for index, radii in enumerate(((.1, .4, .8), (.8, .1, .4), (.4, .8, .1), (.8, .8, .8))):
            value = 1000.+index
            large.add(radii, value, value-100., (value, value))
        result = large.report()
        self.assertTrue(math.isfinite(result['physical']['full']['logQ']))
        self.assertTrue(math.isfinite(result['physical']['union']['row_RSE']))

    def test_invalid_inputs_fail_before_advancing_counts(self):
        moments = UnionMoments()
        for index in range(3):
            for invalid in (-.1, -math.inf, math.nan):
                radii = [0., 0., 0.]
                radii[index] = invalid
                with self.assertRaises(ValueError):
                    selected_keys(radii)
                with self.assertRaises(ValueError):
                    moments.add(radii)
        for radii in ((), (.1, .2), (.1, .2, .3, .4)):
            with self.assertRaises(ValueError):
                moments.add(radii)
        for arguments in (dict(physical=0., hard=0., cloud_pair=(0., 0.)),
                          dict(radii=(0., 0., 0.), physical=0.),
                          dict(radii=(0., 0., 0.), physical=0., hard=0.),
                          dict(radii=(0., 0., 0.), physical=math.inf, hard=0., cloud_pair=(0., 0.)),
                          dict(radii=(0., 0., 0.), physical=0., hard=0., cloud_pair=(0., math.nan)),
                          dict(cloud_pair=(0., 0.))):
            with self.assertRaises(ValueError):
                moments.add(**arguments)
        self.assertEqual(moments.values['physical']['full'].count, 0)
        with self.assertRaises(ValueError):
            moments.report()


if __name__ == '__main__':
    unittest.main()
