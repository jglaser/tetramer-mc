"""Independent geometric checks for finite-system starting configurations.

These tests exercise small analytic shapes. They do not sample a physical
ensemble, authenticate protein assets, or decide assembly stability.
"""
import math
import unittest

import numpy as np

import prepare_finite_assembly_starts as starts
import validate_finite_assembly_starts as validation


def independent_ray_intervals(fixed, fr, moving, mr, direction):
    """Exhaustive closed-ball intersections using a transverse basis.

    This scalar oracle deliberately uses neither the generator's spatial tree
    nor its candidate-pair selection. The returned intervals include only the
    nonnegative ray and are merged independently.
    """
    d = np.asarray(direction, dtype=float)
    d /= np.linalg.norm(d)
    basis = np.eye(3)[np.argmin(np.abs(d))]
    a = np.cross(d, basis)
    a /= np.linalg.norm(a)
    b = np.cross(d, a)
    intervals = []
    for x, rx in zip(fixed, fr):
        for y, ry in zip(moving, mr):
            delta = np.asarray(x) - np.asarray(y)
            transverse = math.hypot(float(delta @ a), float(delta @ b))
            radius = float(rx + ry)
            if transverse > radius:
                continue
            half_width = math.sqrt(max(0., (radius-transverse)*(radius+transverse)))
            center = float(delta @ d)
            left, right = center-half_width, center+half_width
            if right >= 0.:
                intervals.append((max(0., left), right))
    merged = []
    for left, right in sorted(intervals):
        if merged and left <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], right))
        else:
            merged.append((left, right))
    return merged


def minimum_core_gap(fixed, fr, moving, mr):
    """Brute pair distances, independent of any ray solver."""
    return min(float(np.linalg.norm(x-y)-rx-ry)
               for x, rx in zip(fixed, fr) for y, ry in zip(moving, mr))


def proper_rotation():
    """A fixed proper rotation, calculated directly from Rodrigues' formula."""
    axis = np.asarray([1., -2., 3.])
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    cross = np.asarray([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    angle = .713
    return np.eye(3)*math.cos(angle) + (1.-math.cos(angle))*np.outer(axis, axis) + math.sin(angle)*cross


class OutermostRayTests(unittest.TestCase):
    def result(self, fixed, fr, moving, mr, direction=(1., 0., 0.)):
        return starts.last_core_exit(np.asarray(fixed, dtype=float), np.asarray(fr, dtype=float),
                                    np.asarray(moving, dtype=float), np.asarray(mr, dtype=float),
                                    np.asarray(direction, dtype=float))

    def test_single_sphere_analytic_exit_and_direction_normalization(self):
        for scale in (.01, 1., 17.):
            with self.subTest(scale=scale):
                # Lateral distance 3 and sum of radii 5 gives half-width 4.
                self.assertAlmostEqual(self.result([[9., 3., 0.]], [2.], [[0., 0., 0.]], [3.],
                                                   (scale, 0., 0.)), 13., places=12)

    def test_moving_member_origin_and_translated_obstacles(self):
        # Translation acts on every moving atom; its centers need not be at 0.
        self.assertAlmostEqual(self.result([[12., 5., -3.]], [1.5], [[2., 5., -3.]], [.5]), 12., places=12)
        self.assertAlmostEqual(self.result([[7., 5., -3.]], [1.5], [[2., 5., -3.]], [.5]), 7., places=12)

    def test_disjoint_intervals_choose_outermost_not_first_collision(self):
        self.assertAlmostEqual(self.result([[0., 0., 0.], [7., 0., 0.], [23., 0., 0.]],
                                          [1., 1., 1.], [[0., 0., 0.]], [1.]), 25., places=12)

    def test_overlapping_union_intervals_and_unequal_radii(self):
        fixed = np.asarray([[0., 0., 0.], [3., 0., 0.], [5., 1., 0.], [14., 9., 0.]])
        fr = np.asarray([2., .5, 3., 1.])
        moving = np.asarray([[0., 0., 0.], [1., .25, 0.]])
        mr = np.asarray([.7, 1.2])
        intervals = independent_ray_intervals(fixed, fr, moving, mr, [1., 0., 0.])
        self.assertEqual(len(intervals), 1)
        self.assertAlmostEqual(self.result(fixed, fr, moving, mr), intervals[-1][1], places=12)

    def test_tangency_is_retained_but_a_missing_ray_is_zero(self):
        self.assertEqual(self.result([[7., 2., 0.]], [1.], [[0., 0., 0.]], [1.]), 7.)
        self.assertEqual(self.result([[7., 2.000001, 0.]], [1.], [[0., 0., 0.]], [1.]), 0.)
        self.assertEqual(self.result([[0., 2., 0.]], [1.], [[0., 0., 0.]], [1.]), 0.)

    def test_all_intersections_behind_origin_and_initial_overlap(self):
        self.assertEqual(self.result([[-7., 0., 0.], [-5., 0., 0.]], [1., 1.], [[0., 0., 0.]], [1.]), 0.)
        self.assertEqual(self.result([[-1., 0., 0.]], [1.], [[0., 0., 0.]], [1.]), 1.)
        self.assertEqual(self.result([[0., 0., 0.]], [.7], [[0., 0., 0.]], [1.3]), 2.)

    def test_longitudinal_distance_does_not_erase_small_transverse_gap(self):
        self.assertAlmostEqual(self.result([[1.e7, 1., 0.]], [.6], [[0., 0., 0.]], [.4]), 1.e7, places=7)
        self.assertEqual(self.result([[1.e7, 1.0001, 0.]], [.6], [[0., 0., 0.]], [.4]), 0.)

    def test_common_translation_and_proper_rotation_preserve_exit(self):
        fixed = np.asarray([[0., 0., 0.], [3., 1., -1.], [9., 2., .5]])
        moving = np.asarray([[.1, -.4, 0.], [-.7, .2, .6]])
        fr, mr = np.asarray([.8, 1.4, .9]), np.asarray([.7, 1.1])
        direction = np.asarray([1., .2, -.15])
        expected = self.result(fixed, fr, moving, mr, direction)
        q = proper_rotation()
        self.assertAlmostEqual(np.linalg.det(q), 1., places=14)
        shift = np.asarray([19., -13., 5.])
        self.assertAlmostEqual(self.result(fixed+shift, fr, moving+shift, mr, direction), expected, places=11)
        self.assertAlmostEqual(self.result(fixed@q.T+shift, fr, moving@q.T+shift, mr, q@direction), expected, places=11)

    def test_random_small_unions_match_exhaustive_interval_union(self):
        rng = np.random.default_rng(774093)
        for case in range(48):
            fixed = rng.normal(size=(5, 3))*3.
            moving = rng.normal(size=(4, 3))
            fr, mr = rng.uniform(.2, 1.6, size=5), rng.uniform(.2, 1.6, size=4)
            direction = rng.normal(size=3)
            intervals = independent_ray_intervals(fixed, fr, moving, mr, direction)
            expected = intervals[-1][1] if intervals else 0.
            with self.subTest(case=case):
                self.assertAlmostEqual(self.result(fixed, fr, moving, mr, direction), expected, places=11)

    def test_returned_exit_is_contact_and_outward_gap_clears_all_cores(self):
        fixed = np.asarray([[0., 0., 0.], [2.1, .2, -.1], [4.2, -.1, .2]])
        fr = np.asarray([1., .8, 1.2])
        moving = np.asarray([[0., 0., 0.], [.3, .1, -.2]])
        mr = np.asarray([.6, .9])
        d = np.asarray([1., .04, -.03]); d /= np.linalg.norm(d)
        distance = self.result(fixed, fr, moving, mr, d)
        self.assertGreater(distance, 0.)
        self.assertAlmostEqual(minimum_core_gap(fixed, fr, moving+distance*d, mr), 0., places=11)
        self.assertLess(minimum_core_gap(fixed, fr, moving+(distance-.001)*d, mr), 0.)
        # The construction's 0.1-A safety offset stays within the 3-A
        # exclusion-contact shell for 1.5-A depletants by the triangle inequality.
        gap = minimum_core_gap(fixed, fr, moving+(distance+.1)*d, mr)
        self.assertGreater(gap, 0.)
        self.assertLessEqual(gap, .10000000001)
        self.assertLess(gap, 2.*1.5)

    def test_invalid_geometry_fails_closed(self):
        fixed, fr, moving, mr, d = [[0., 0., 0.]], [1.], [[0., 0., 0.]], [1.], [1., 0., 0.]
        cases = [([], [], moving, mr, d), (fixed, fr, [], [], d),
                 (fixed, [0.], moving, mr, d), (fixed, fr, moving, [-1.], d),
                 (fixed, [float('nan')], moving, mr, d),
                 ([[float('inf'), 0., 0.]], fr, moving, mr, d),
                 (fixed, fr, moving, mr, [0., 0., 0.]),
                 (fixed, fr, moving, mr, [float('nan'), 0., 0.]),
                 (fixed, [1., 1.], moving, mr, d)]
        for index, args in enumerate(cases):
            with self.subTest(case=index), self.assertRaises(ValueError):
                self.result(*args)


def pose(position, orientation=(1., 0., 0., 0.)):
    return dict(position=list(position), orientation=list(orientation))


class EmptyNative:
    motifs = []
    definition_sha256 = 'a'*64

    def __init__(self):
        self.calls = 0

    def classify_pair(self, anchor, moving):
        self.calls += 1
        return []


class ChainNative(EmptyNative):
    motifs = [dict(id=0, relative_position=[2., 0., 0.], relative_orientation=[1., 0., 0., 0.])]

    def classify_pair(self, anchor, moving):
        self.calls += 1
        delta = np.asarray(moving['position'])-np.asarray(anchor['position'])
        return [dict(motif_id=0)] if np.linalg.norm(delta-[2., 0., 0.]) < 1e-10 else []


class StaticValidationTests(unittest.TestCase):
    sphere = dict(atoms=[dict(center=[0., 0., 0.], radius=.5)])

    def validate(self, poses, preparation='dispersed', native=None, shape=None,
                 boundary=None, lengths=(100., 100., 100.)):
        return validation.validate_start(self.sphere if shape is None else shape, poses,
            dict(kind='spherical', radius=50.) if boundary is None else boundary,
            lengths, EmptyNative() if native is None else native, preparation)

    def test_dispersed_has_all_pairs_no_contacts_and_no_equilibrium_claim(self):
        poses = [pose([10.*i-10., 0., 0.]) for i in range(3)]
        native = EmptyNative()
        report = self.validate(poses, native=native)
        self.assertTrue(report['passed'])
        self.assertEqual(report['exclusion']['components'], [[0], [1], [2]])
        self.assertEqual(report['native']['keys'], [])
        self.assertEqual(native.calls, 3)
        self.assertEqual(len(report['geometry']['pair_checks']), 3)
        self.assertEqual(report['physical_draws'], 0)
        self.assertNotIn('equilibrium_passed', report)

    def test_strict_core_and_exclusion_thresholds(self):
        for distance, hard, contact in [(1.-1e-7, False, True), (1., True, True),
                                        (4.-1e-7, True, True), (4., True, False),
                                        (4.+1e-7, True, False)]:
            native = EmptyNative()
            report = self.validate([pose([0., 0., 0.]), pose([distance, 0., 0.])], native=native)
            with self.subTest(distance=distance):
                self.assertEqual(report['geometry']['hard_valid'], hard)
                self.assertEqual(bool(report['exclusion']['edges']), contact)
                self.assertEqual(report['native']['evaluated'], hard)
                self.assertEqual(native.calls, int(hard))
                self.assertEqual(report['passed'], hard and not contact)

    def test_atomic_wall_uses_rotated_atoms_not_only_body_center(self):
        shape = dict(atoms=[dict(center=[2., 0., 0.], radius=.5),
                            dict(center=[-1., 0., 0.], radius=.3)])
        poses = [pose([48., 0., 0.]), pose([-10., 0., 0.])]
        native = EmptyNative()
        report = self.validate(poses, native=native, shape=shape)
        self.assertFalse(report['geometry']['wall_valid'])
        self.assertAlmostEqual(report['geometry']['minimum_atomic_wall_clearance_A'], -.5)
        self.assertFalse(report['native']['evaluated'])
        self.assertEqual(native.calls, 0)
        self.assertIn('strict_atomic_wall_violation', report['failure_reasons'])
        # A pi rotation reverses the protruding end without changing the center.
        poses[0]['orientation'] = [0., 0., 0., 1.]
        self.assertTrue(self.validate(poses, shape=shape)['passed'])

    def test_wall_equality_is_valid_but_small_violation_is_not(self):
        other = pose([-10., 0., 0.])
        for x, valid in [(49.5, True), (49.5000001, False)]:
            report = self.validate([pose([x, 0., 0.]), other])
            with self.subTest(x=x):
                self.assertEqual(report['geometry']['wall_valid'], valid)
                self.assertEqual(report['passed'], valid)

    def test_periodic_images_expose_contacts_and_core_overlaps(self):
        boundary = dict(kind='periodic')
        for right, hard in [(19., True), (20.6, False)]:
            # Both raw center distances are large; their minimum-image distance
            # is 2 or 0.4, respectively.
            report = self.validate([pose([-19., 0., 0.]), pose([right, 0., 0.])],
                preparation='competing-aggregate', boundary=boundary, lengths=(40.,)*3)
            with self.subTest(right=right):
                self.assertEqual(report['geometry']['hard_valid'], hard)
                self.assertEqual(report['exclusion']['edges'], [[0, 1]])
                self.assertEqual(report['passed'], hard)

    def test_periodic_shape_bound_prevents_unchecked_multiple_images(self):
        with self.assertRaisesRegex(ValueError, 'minimum-image'):
            self.validate([pose([0., 0., 0.]), pose([3., 0., 0.])],
                          boundary=dict(kind='periodic'), lengths=(8., 20., 20.))

    def test_competing_requires_one_complete_component_and_no_native_edges(self):
        poses = [pose([2.*i, 0., 0.]) for i in range(3)]
        self.assertTrue(self.validate(poses, preparation='competing-aggregate')['passed'])
        native_report = self.validate(poses, preparation='competing-aggregate', native=ChainNative())
        self.assertFalse(native_report['passed'])
        self.assertIn('competing_has_native_entry', native_report['failure_reasons'])
        poses[-1] = pose([12., 0., 0.])
        split = self.validate(poses, preparation='competing-aggregate')
        self.assertFalse(split['passed'])
        self.assertIn('competing_exclusion_graph_not_connected', split['failure_reasons'])

    def test_exact_eight_body_native_seed_and_free_remainder(self):
        poses = [pose([2.*i-7., 0., 0.]) for i in range(8)]
        poses += [pose([0., 15., 0.]), pose([0., -15., 0.]),
                  pose([0., 0., 15.]), pose([0., 0., -15.])]
        report = self.validate(poses, preparation='native-seeded', native=ChainNative())
        self.assertTrue(report['passed'])
        self.assertEqual(report['native']['cycle']['components'], [list(range(8)), [8], [9], [10], [11]])
        self.assertEqual(report['native']['largest_component_size'], 8)
        poses[8] = pose([0., 2., 0.])
        attached = self.validate(poses, preparation='native-seeded', native=ChainNative())
        self.assertFalse(attached['passed'])
        self.assertIn('native_seed_remainder_has_exclusion_contacts', attached['failure_reasons'])
        poses[8] = pose([9., 0., 0.])
        nine = self.validate(poses, preparation='native-seeded', native=ChainNative())
        self.assertFalse(nine['passed'])
        self.assertIn('native_seed_is_not_exact_eight_body_component', nine['failure_reasons'])

    def test_inconsistent_native_cycle_is_not_counted_as_registered_cluster(self):
        class ConflictedNative(ChainNative):
            def classify_pair(self, anchor, moving):
                return [dict(motif_id=0)]

        triangle = [pose([0., 0., 0.]), pose([2., 0., 0.]), pose([1., math.sqrt(3.), 0.])]
        report = self.validate(triangle, preparation='competing-aggregate', native=ConflictedNative())
        self.assertFalse(report['native']['cycle']['consistent'])
        self.assertIn('native_catalogue_cycle_inconsistent', report['failure_reasons'])
        self.assertFalse(report['passed'])

    def test_periodic_native_cycle_requires_ordinary_space_lift(self):
        positions = np.asarray([[0., 0., 0.], [4., 0., 0.], [8., 0., 0.]])
        lengths = np.asarray([12., 12., 12.])
        self.assertTrue(validation.periodic_lift(positions, lengths, [(0, 1), (1, 2)]))
        self.assertFalse(validation.periodic_lift(positions, lengths, [(0, 1), (1, 2), (0, 2)]))
        positions[2, 0] = 5.
        self.assertTrue(validation.periodic_lift(positions, lengths, [(0, 1), (1, 2), (0, 2)]))

    def test_unequal_atomic_radii_contacts_match_brute_geometry(self):
        shape = dict(atoms=[dict(center=[-1., 0., 0.], radius=.3),
                            dict(center=[1., .2, 0.], radius=.8)])
        poses = [pose([0., 0., 0.]), pose([4., 0., 0.]), pose([0., 8., 0.])]
        report = self.validate(poses, shape=shape)
        points = np.asarray([a['center'] for a in shape['atoms']])
        radii = np.asarray([a['radius'] for a in shape['atoms']])
        expected = []
        for i in range(len(poses)):
            for j in range(i+1, len(poses)):
                left, right = points+poses[i]['position'], points+poses[j]['position']
                self.assertGreaterEqual(minimum_core_gap(left, radii, right, radii), 0.)
                if minimum_core_gap(left, radii+1.5, right, radii+1.5) < 0.:
                    expected.append([i, j])
        self.assertEqual(report['exclusion']['edges'], expected)
        self.assertTrue(report['geometry']['hard_valid'])


def quaternion_matrix(q):
    """Independent scalar wxyz formula, without the generator's helper."""
    w, x, y, z = q
    return np.asarray([[1.-2.*(y*y+z*z), 2.*(x*y-w*z), 2.*(x*z+w*y)],
                       [2.*(x*y+w*z), 1.-2.*(x*x+z*z), 2.*(y*z-w*x)],
                       [2.*(x*z-w*y), 2.*(y*z+w*x), 1.-2.*(x*x+y*y)]])


class PreparationConstructionTests(unittest.TestCase):
    shape = dict(atoms=[dict(center=[0., 0., 0.], radius=.5)])
    geometry = dict(sphere_radius=15., box_lengths=[40., 40., 40.])

    def test_disconnected_and_connected_preparations_are_deterministic_and_record_every_trial(self):
        for kind in ('dispersed', 'competing-aggregate'):
            records = []
            poses, events = starts.prepare_family(self.shape, None, kind, 12,
                np.random.default_rng(706615), self.geometry, record=records.append)
            other_poses, other_events = starts.prepare_family(self.shape, None, kind, 12,
                np.random.default_rng(706615), self.geometry)
            with self.subTest(kind=kind):
                self.assertEqual(poses, other_poses)
                self.assertEqual(events, other_events)
                self.assertEqual(events, records)
                self.assertEqual([e['index'] for e in events], list(range(len(events))))
                if kind == 'dispersed':
                    self.assertEqual(sum(e['accepted'] for e in events), 12)
                    self.assertGreater(sum(not e['accepted'] for e in events), 0)
                else:
                    self.assertEqual(sum(e['kind'] == 'aggregate-ray' and e['accepted'] for e in events), 11)
                    self.assertEqual(sum(e['kind'] == 'proper-fragment-isometry' for e in events), 1)
                for boundary in (dict(kind='spherical', radius=15.), dict(kind='periodic')):
                    report = validation.validate_start(self.shape, poses, boundary,
                        self.geometry['box_lengths'], EmptyNative(), kind)
                    self.assertTrue(report['passed'], report['failure_reasons'])
                    self.assertEqual(report['exclusion']['largest_component_size'], 1 if kind == 'dispersed' else 12)

    def test_native_fragment_preserves_all_relative_poses_under_one_proper_isometry(self):
        fragment = []
        rng = np.random.default_rng(49934)
        for x in (-1.1, 1.1):
            for y in (-1.1, 1.1):
                for z in (-1.1, 1.1):
                    q = rng.normal(size=4); q /= np.linalg.norm(q)
                    fragment.append(pose([x+7., y-3., z+11.], q))
        before = [(np.asarray(p['position']), quaternion_matrix(p['orientation'])) for p in fragment]
        poses, events = starts.prepare_family(self.shape, fragment, 'native-seeded', 12,
            np.random.default_rng(201556), self.geometry)
        after = [(np.asarray(p['position']), quaternion_matrix(p['orientation'])) for p in poses[:8]]
        self.assertEqual(len(poses), 12)
        self.assertEqual(sum(e['kind'] == 'proper-fragment-isometry' for e in events), 1)
        for (p, r), original in zip(before, fragment):
            np.testing.assert_array_equal(p, original['position'])
            np.testing.assert_allclose(r.T@r, np.eye(3), atol=2e-15)
        for i in range(8):
            self.assertAlmostEqual(np.linalg.det(after[i][1]), 1., places=13)
            for j in range(i+1, 8):
                relative_before = before[i][1].T@(before[j][0]-before[i][0])
                relative_after = after[i][1].T@(after[j][0]-after[i][0])
                np.testing.assert_allclose(relative_after, relative_before, atol=1e-13, rtol=0.)
                np.testing.assert_allclose(after[i][1].T@after[j][1], before[i][1].T@before[j][1], atol=1e-13, rtol=0.)
        report = validation.validate_start(self.shape, poses, dict(kind='spherical', radius=15.),
            self.geometry['box_lengths'], EmptyNative(), 'native-seeded')
        # The toy fragment has no registered catalogue. Its hard geometry and
        # isolated extra particles are still independently verifiable.
        self.assertTrue(report['geometry']['hard_valid'])
        self.assertTrue(report['geometry']['wall_valid'])
        self.assertFalse(any(i >= 8 or j >= 8 for i, j in report['exclusion']['edges']))

    def test_haar_rotation_sampler_has_correct_quaternion_and_direction_moments(self):
        rng = np.random.default_rng(502967)
        quaternions = np.asarray([starts.quaternion(rng) for _ in range(4096)])
        np.testing.assert_allclose(np.sum(quaternions**2, axis=1), 1., atol=8e-16, rtol=0.)
        self.assertLess(float(np.max(np.abs(np.mean(quaternions, axis=0)))), .03)
        np.testing.assert_allclose(np.mean(quaternions**2, axis=0), .25, atol=.02, rtol=0.)
        directions = np.asarray([quaternion_matrix(q)[:, 0] for q in quaternions])
        self.assertLess(float(np.max(np.abs(directions.mean(axis=0)))), .04)
        np.testing.assert_allclose(directions.T@directions/len(directions), np.eye(3)/3., atol=.03, rtol=0.)


if __name__ == '__main__':
    unittest.main()
