"""Synthetic-only centroid broad-phase checks: no protein inputs or classifiers."""
import copy
import itertools
import math
import unittest
from unittest import mock

import numpy as np
from scipy.spatial.transform import Rotation

import native_pair_candidates as candidate


def pose(position, rotvec=(0., 0., 0.)):
    q = Rotation.from_rotvec(rotvec).as_quat()
    return dict(position=list(position), orientation=q[[3, 0, 1, 2]].tolist())


def independent_rotation(p):
    q = np.asarray(p['orientation'], float)
    w, x, y, z = q/math.sqrt(float(sum(v*v for v in q)))
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def compose(a, b):
    ra, rb = independent_rotation(a), independent_rotation(b)
    q = Rotation.from_matrix(ra@rb).as_quat()
    return dict(position=(np.asarray(a['position'])+ra@b['position']).tolist(),
                orientation=q[[3, 0, 1, 2]].tolist())


def oracle(members, motifs, poses, tolerance):
    """Unpruned independent all-member gate; no centroid/tree calculation."""
    members = np.asarray(members)
    result = set()
    for i, j in itertools.combinations(range(len(poses)), 2):
        a, b = poses[i], poses[j]
        ra, rb = independent_rotation(a), independent_rotation(b)
        d = ra.T@(np.asarray(b['position'])-a['position'])
        observed = members@(ra.T@rb).T+d
        for motif in motifs:
            expected = members@independent_rotation(motif).T+motif['position']
            if all(np.linalg.norm(row) <= tolerance for row in observed-expected):
                result.add((i, j))
                break
    return result


class NativePairCandidateTests(unittest.TestCase):
    def setUp(self):
        self.members = np.array([[9., -3., 4.], [11., 2., 4.], [8., 0., 7.], [13., -1., 2.]])
        self.motifs = [pose([15., -2., 4.], [.3, -.2, .6]), pose([-8., 3., 7.], [-.4, .1, -.2])]

    def test_noncentered_rotated_members_and_directed_catalogue(self):
        model = candidate.NativePairCandidates(self.members, self.motifs[:1], 2.)
        a = pose([17., -31., 20.], [.8, .3, -.1])
        b = compose(a, self.motifs[0])
        self.assertEqual(oracle(self.members, self.motifs[:1], [a, b], 2.), {(0, 1)})
        self.assertEqual(model.candidate_pairs([a, b]), [(0, 1)])
        self.assertEqual(oracle(self.members, self.motifs[:1], [b, a], 2.), set())
        self.assertEqual(model.candidate_pairs([b, a]), [])

    def test_random_synthetic_unpruned_reference_has_no_false_negatives(self):
        rng = np.random.default_rng(73482)
        model = candidate.NativePairCandidates(self.members, self.motifs, 2.)
        for _ in range(30):
            poses = [pose(rng.uniform(-25, 25, 3), rng.normal(0, .8, 3)) for _ in range(8)]
            for motif in self.motifs:
                moving = compose(poses[0], motif)
                moving['position'] = (np.asarray(moving['position'])+rng.uniform(-.5, .5, 3)).tolist()
                poses.append(moving)
            expected = oracle(self.members, self.motifs, poses, 2.)
            self.assertTrue(expected)
            self.assertLessEqual(expected, set(model.candidate_pairs(poses)))

    def test_global_proper_isometry_and_quaternion_sign(self):
        model = candidate.NativePairCandidates(self.members, self.motifs, 2.)
        initial = [pose([0, 0, 0]), *copy.deepcopy(self.motifs), pose([100, 0, 0])]
        transform = pose([1200., -600., 240.], [.7, -.3, .9])
        changed = [compose(transform, p) for p in initial]
        changed[1]['orientation'] = (-np.asarray(changed[1]['orientation'])).tolist()
        self.assertEqual(model.candidate_pairs(initial), model.candidate_pairs(changed))
        self.assertLessEqual(oracle(self.members, self.motifs, changed, 2.), set(model.candidate_pairs(changed)))

    def test_relabeling_with_inverse_closed_catalogue(self):
        motifs = [pose([4., 0, 0]), pose([-4., 0, 0])]
        poses = [pose([0, 0, 0]), pose([4., 0, 0]), pose([40., 0, 0]), pose([8., 0, 0])]
        model = candidate.NativePairCandidates(self.members, motifs, 2.)
        order = [3, 0, 2, 1]
        relabeled = {tuple(sorted((order[i], order[j]))) for i, j in model.candidate_pairs([poses[k] for k in order])}
        self.assertEqual(relabeled, set(model.candidate_pairs(poses)))

    def test_centroid_match_can_be_false_positive(self):
        members = [[-10., 0, 0], [10., 0, 0]]
        motifs = [pose([20., 0, 0])]
        poses = [pose([0, 0, 0]), pose([20., 0, 0], [0, 0, math.pi])]
        model = candidate.NativePairCandidates(members, motifs, 2.)
        self.assertEqual(oracle(members, motifs, poses, 2.), set())
        self.assertEqual(model.candidate_pairs(poses), [(0, 1)])

    def test_closed_tangencies_and_rounding_neighborhood(self):
        members = [[0., 0, 0]]
        model = candidate.NativePairCandidates(members, [pose([0, 0, 0])], 2.)
        for distance in (np.nextafter(2., -np.inf), 2., np.nextafter(2., np.inf)):
            self.assertEqual(model.candidate_pairs([pose([0, 0, 0]), pose([distance, 0, 0])]), [(0, 1)])
        self.assertEqual(model.candidate_pairs([pose([0, 0, 0]), pose([2.+1e-7, 0, 0])]), [])
        exact = candidate.NativePairCandidates(members, [pose([0, 0, 0])], 0.)
        self.assertEqual(exact.candidate_pairs([pose([0, 0, 0])]*2), [(0, 1)])

    def test_rotated_member_boundary_survives_body_zero_recentering(self):
        # A world-x displacement places the centroid near a face of the tree's
        # infinity-norm neighborhood, while the mandatory predicate below uses
        # all four noncentered members in the anchor frame, without a centroid.
        model = candidate.NativePairCandidates(self.members, self.motifs, 2.)
        origins = ([350., -200., 70.],
                   [2.**35+.125, -2.**34+.375, 2.**33+.625],
                   [-2.**35+.375, 2.**34+.125, -2.**33+.625])
        steps = (-128, -16, -1, 0, 1, 16, 128)
        for rotation in ([.8, .3, -.1], [-.2, .7, .6]):
            anchor = pose([17., -31., 20.], rotation)
            anchor['orientation'] = (np.asarray(anchor['orientation'])*(1.+5e-9)).tolist()
            boundary = compose(anchor, self.motifs[0])
            boundary['position'][0] += 2.
            boundary['orientation'] = (-np.asarray(boundary['orientation'])*(1.-5e-9)).tolist()
            observed_sides, represented_positions = set(), set()
            for ulps in steps:
                moving = copy.deepcopy(boundary)
                for _ in range(abs(ulps)):
                    moving['position'][0] = float(np.nextafter(moving['position'][0],
                        math.inf if ulps > 0 else -math.inf))
                represented_positions.add(moving['position'][0])
                near_pair_passes = (0, 1) in oracle(self.members, self.motifs, [anchor, moving], 2.)
                observed_sides.add(near_pair_passes)
                for origin in origins:
                    with self.subTest(rotation=rotation, ulps=ulps, body_zero=origin):
                        # Body 0 and body 2 are mobile. Moving body 0 changes the
                        # numerical tree origin but not the tested (1, 2) pair.
                        state = [pose(origin, [.1, -.3, .2]), anchor, moving,
                                 pose([600., -400., 700.], [.4, .1, -.2])]
                        expected = oracle(self.members, self.motifs, state, 2.)
                        self.assertEqual((1, 2) in expected, near_pair_passes)
                        full = set(model.candidate_pairs(state))
                        mobile = set(model.candidate_pairs(state, mobile_labels=[0, 2]))
                        self.assertLessEqual(expected, full)
                        self.assertLessEqual({p for p in expected if {0, 2}.intersection(p)}, mobile)
                        # The large centering offset must exercise usable finite
                        # arithmetic, not make an all-pairs fallback pass vacuously.
                        self.assertNotIn((2, 3), full)
                        self.assertNotIn((2, 3), mobile)
            self.assertEqual(len(represented_positions), len(steps))
            self.assertEqual(observed_sides, {False, True})

    def test_empty_duplicate_and_copied_catalogues(self):
        empty = candidate.NativePairCandidates(self.members, [], 2.)
        self.assertEqual(empty.candidate_pairs([pose([0, 0, 0])]*2), [])
        model = candidate.NativePairCandidates(self.members, self.motifs*2, 2.)
        old = copy.deepcopy(self.motifs)
        self.motifs[0]['position'][0] += 1000
        self.members[:] = 1000
        self.assertEqual(model.candidate_pairs([pose([0, 0, 0]), old[0]]), [(0, 1)])
        self.assertEqual(model.candidate_pairs([]), [])
        self.assertEqual(model.candidate_pairs([pose([0, 0, 0])]), [])

    def test_mobile_filter_matches_full_candidate_union(self):
        model = candidate.NativePairCandidates(self.members, self.motifs, 2.)
        poses = [pose([0, 0, 0]), *copy.deepcopy(self.motifs), pose([100, 0, 0])]
        full = model.candidate_pairs(poses)
        for labels in ([], [0], [1], [0, 3], [0, 1, 2, 3]):
            self.assertEqual(model.candidate_pairs(poses, mobile_labels=labels),
                             [p for p in full if any(i in labels for i in p)])
        for labels in ([0, 0], [-1], [4], [True]):
            with self.assertRaises(ValueError):
                model.candidate_pairs(poses, mobile_labels=labels)

    def test_unsafe_arithmetic_and_tree_failure_fall_back(self):
        model = candidate.NativePairCandidates(self.members, self.motifs, 2.)
        largest = np.finfo(float).max
        poses = [pose([largest, 0, 0]), pose([-largest, 0, 0]), pose([0, 0, 0])]
        self.assertEqual(model.candidate_pairs(poses), [(0, 1), (0, 2), (1, 2)])
        self.assertEqual(model.candidate_pairs(poses, mobile_labels=[1]), [(0, 1), (1, 2)])
        with mock.patch.object(candidate, 'cKDTree', side_effect=ValueError('unsupported scale')):
            self.assertEqual(model.candidate_pairs([pose([0, 0, 0]), pose([100, 0, 0])]), [(0, 1)])

    def test_large_common_offset_retains_reference_matches(self):
        model = candidate.NativePairCandidates(self.members, self.motifs, 2.)
        transform = pose([1e14, -1e14, 1e14], [.2, -.1, .3])
        poses = [transform, compose(transform, self.motifs[0]), compose(transform, self.motifs[1])]
        self.assertLessEqual(oracle(self.members, self.motifs, poses, 2.), set(model.candidate_pairs(poses)))

    def test_invalid_inputs_and_periodic_request_rejected(self):
        for members in ([], [[1., 2.]], [[np.nan, 0, 0]]):
            with self.assertRaises(ValueError):
                candidate.NativePairCandidates(members, self.motifs, 2.)
        for tolerance in (-1., np.nan, np.inf):
            with self.assertRaises(ValueError):
                candidate.NativePairCandidates(self.members, self.motifs, tolerance)
        with self.assertRaisesRegex(ValueError, 'periodic'):
            candidate.NativePairCandidates(self.members, self.motifs, 2., boundary='periodic')
        invalid = dict(position=[0., 0, 0], orientation=[2., 0, 0, 0])
        with self.assertRaises(ValueError):
            candidate.NativePairCandidates(self.members, [invalid], 2.)
        model = candidate.NativePairCandidates(self.members, [], 2.)
        for bad in (invalid, dict(position=[np.inf, 0, 0], orientation=[1., 0, 0, 0])):
            with self.assertRaises(ValueError):
                model.candidate_pairs([bad])

    def test_exact_tree_search_is_single_threaded_and_conservative(self):
        real = candidate.cKDTree
        calls = []

        class InspectTree:
            def __init__(self, points, **kwargs):
                self.tree = real(points, **kwargs)

            def query_ball_point(self, points, radius, **kwargs):
                calls.append(kwargs)
                return self.tree.query_ball_point(points, radius, **kwargs)

        model = candidate.NativePairCandidates(self.members, self.motifs, 2., boundary='spherical')
        with mock.patch.object(candidate, 'cKDTree', InspectTree):
            result = model.candidate_pairs([pose([0, 0, 0]), self.motifs[0]])
        self.assertEqual(result, [(0, 1)])
        self.assertTrue(calls)
        self.assertTrue(all(c == dict(p=np.inf, eps=0, workers=1, return_sorted=True) for c in calls))


if __name__ == '__main__':
    unittest.main()
