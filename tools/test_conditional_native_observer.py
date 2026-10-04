"""Synthetic registry and exact-cache checks; no protein or bath calculations."""
from __future__ import annotations

import copy
import itertools
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from conditional_native_observer import ConditionalNativeObserver
from native_contact_regions import pose_arrays
from native_graph_consistency import NativeGraphConsistency


def pose(x, q=(1., 0., 0., 0.)):
    return dict(position=[float(x), 0., 0.], orientation=list(q))


class ToyNative:
    """An independent all-member predicate plus a proper-orientation condition."""
    def __init__(self, spacing=4., tolerance=.1):
        self.member_positions = np.asarray([[1., 0., 0.], [1., 1., 0.],
                                           [1., 0., 1.], [2., 0., 0.]])
        self.definition = dict(criteria=dict(body_member_position_entry_A=tolerance))
        self.motifs = [dict(id=7, relative_position=[spacing, 0., 0.],
                            relative_orientation=[1., 0., 0., 0.])]
        self.calls = []

    def classify_pair(self, anchor, moving):
        self.calls.append(copy.deepcopy([anchor, moving]))
        ta, ra = pose_arrays(anchor); tb, rb = pose_arrays(moving)
        d, rotation = ra.T@(tb-ta), ra.T@rb
        if Rotation.from_matrix(rotation).magnitude() > .3:
            return []
        observed = self.member_positions@rotation.T+d
        expected = self.member_positions+np.asarray(self.motifs[0]['relative_position'])
        if np.max(np.linalg.norm(observed-expected, axis=1)) <= self.definition['criteria']['body_member_position_entry_A']:
            return [dict(motif_id=7)]
        return []


def brute(native, poses):
    keys = [(i, j, match['motif_id']) for i, j in itertools.combinations(range(len(poses)), 2)
            for match in native.classify_pair(poses[i], poses[j])]
    graph = NativeGraphConsistency(native.motifs, len(poses)).check(keys)
    return sorted(keys), graph


class ConditionalNativeTests(unittest.TestCase):
    def test_all_matching_motif_alternatives_survive(self):
        native = ToyNative()
        native.motifs.append(dict(native.motifs[0], id=8))
        original = native.classify_pair
        def alternatives(anchor, moving):
            matches = original(anchor, moving)
            return matches+[dict(motif_id=8)] if matches else []
        native.classify_pair = alternatives
        source = [pose(0), pose(4)]
        result = ConditionalNativeObserver(native, source, [0, 1]).classify(source)
        self.assertEqual(result['instantaneous_native_keys'], [(0, 1, 7), (0, 1, 8)])
        self.assertEqual(result['internal_native_keys'], [(0, 1, 7), (0, 1, 8)])
        self.assertEqual(result['full_native_graph']['independent_cycles'], 0)

    def test_isolated_bodies_are_resolved_without_implying_native_attachment(self):
        native = ToyNative(); source = [pose(x) for x in (0, 40, 80)]
        result = ConditionalNativeObserver(native, source, [0, 2]).classify([source[0], source[2]])
        self.assertEqual(result['instantaneous_native_keys'], [])
        self.assertEqual(result['mobile_related_native_keys'], [])
        self.assertTrue(result['mobile_registry_resolved'])
        self.assertEqual([c['bodies'] for c in result['mobile_native_components']], [[0], [2]])
        self.assertEqual(native.calls, [])

    def test_full_labels_match_unpruned_observer_and_fixed_graph_enters_cycles(self):
        source = [pose(x) for x in (0, 4, 8, 30)]
        native = ToyNative(); observer = ConditionalNativeObserver(native, source, [0, 3])
        for selected in ([pose(.05), pose(12)], [pose(.04), pose(40)], [pose(.04), pose(12)]):
            state = copy.deepcopy(source); state[0], state[3] = selected
            expected, graph = brute(ToyNative(), state)
            result = observer.classify(selected)
            self.assertEqual(result['instantaneous_native_keys'], expected)
            self.assertEqual(result['full_native_graph'], graph)
            self.assertEqual(result['fixed_native_keys'], [(1, 2, 7)])
            self.assertTrue(result['mobile_registry_resolved'])
        self.assertEqual(observer.counts['fixed_pair_calls'], 1)
        self.assertEqual(result['mobile_native_components'][0]['bodies'], [0, 1, 2, 3])

    def test_rejected_residence_retained_and_exact_pair_cache_only(self):
        source = [pose(x) for x in (0, 4, 8, 30)]
        native = ToyNative(); observer = ConditionalNativeObserver(native, source, [0, 3])
        selected = [pose(0), pose(12)]
        first = observer.classify(selected); before = len(native.calls)
        repeated = observer.classify(copy.deepcopy(selected))
        self.assertEqual(len(native.calls), before)
        self.assertEqual(repeated['instantaneous_native_keys'], first['instantaneous_native_keys'])
        self.assertEqual(repeated['observer_counts']['observations'], 2)
        self.assertEqual(repeated['observer_counts']['whole_endpoint_cache_hits'], 1)
        self.assertEqual(repeated['observer_counts']['candidate_searches'],
                         first['observer_counts']['candidate_searches'])
        observer.classify([pose(.01), pose(12)])
        self.assertEqual(len(native.calls), before+1)
        self.assertEqual(observer.counts['mobile_pair_cache_hits'], 1)

    def test_member_order_is_not_assumed_to_be_canonical_pair_order(self):
        source = [pose(x) for x in (0, 4, 8, 30)]
        left = ConditionalNativeObserver(ToyNative(), source, [0, 3]).classify([pose(0), pose(12)])
        right = ConditionalNativeObserver(ToyNative(), source, [3, 0]).classify([pose(12), pose(0)])
        self.assertEqual(left['instantaneous_native_keys'], right['instantaneous_native_keys'])
        self.assertEqual(left['external_native_keys'], [(0, 1, 7), (2, 3, 7)])

    def test_internal_pair_retained_and_centroid_candidates_are_not_labels(self):
        native = ToyNative(); source = [pose(0), pose(4), pose(100)]
        observer = ConditionalNativeObserver(native, source, [0, 1])
        registered = observer.classify([pose(0), pose(4)])
        self.assertEqual(registered['internal_native_keys'], [(0, 1, 7)])
        self.assertEqual(registered['external_native_keys'], [])
        # Rotate about the member centroid and translate to preserve that
        # centroid exactly. The filter passes, the full predicate rejects.
        q = Rotation.from_euler('z', 90, degrees=True).as_quat()[[3, 0, 1, 2]]
        rotation = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
        center = native.member_positions.mean(axis=0)
        moving = dict(position=(np.asarray([4., 0., 0.])+center-rotation@center).tolist(), orientation=q.tolist())
        before = observer.counts['mobile_pair_calls']
        rejected = observer.classify([pose(0), moving])
        self.assertGreater(observer.counts['mobile_pair_calls'], before)
        self.assertEqual(rejected['instantaneous_native_keys'], [])

    def test_frustration_preserved_without_poisoning_disconnected_mobile_component(self):
        native = ToyNative(spacing=1., tolerance=1.1)
        source = [pose(x) for x in (0, 1, 2, 100, 101)]
        observer = ConditionalNativeObserver(native, source, [0, 4], graph_cache_entries=1)
        result = observer.classify([source[0], source[4]])
        self.assertFalse(result['full_native_graph']['consistent'])
        self.assertFalse(result['mobile_registry_resolved'])
        components = {tuple(c['bodies']): c for c in result['mobile_native_components']}
        self.assertFalse(components[0, 1, 2]['catalogue_consistent'])
        self.assertTrue(components[3, 4]['catalogue_consistent'])
        self.assertEqual(components[0, 1, 2]['independent_cycles'], 1)
        self.assertLessEqual(len(observer.checker.cache), 1)
        self.assertEqual(result['instantaneous_native_keys'], brute(ToyNative(1., 1.1), source)[0])

    def test_full_classifier_errors_and_unknown_labels_are_never_empty_matches(self):
        for mode in ('failure', 'unknown'):
            with self.subTest(mode=mode):
                native = ToyNative(); source = [pose(0), pose(4)]
                def invalid(*_):
                    if mode == 'failure':
                        raise RuntimeError('classifier failure')
                    return [dict(motif_id=900)]
                native.classify_pair = invalid
                observer = ConditionalNativeObserver(native, source, [0, 1])
                with self.assertRaisesRegex((ValueError, RuntimeError), 'classifier failure|Unknown native'):
                    observer.classify(source)
                self.assertEqual(observer.counts['mobile_pair_calls'], 1)
                self.assertEqual(observer.counts['observations'], 0)

    def test_invalid_poses_boundaries_labels_and_limits_fail_before_pair_queries(self):
        for mode in ('periodic', 'duplicate', 'noninteger', 'cache', 'bad_pose'):
            with self.subTest(mode=mode):
                native = ToyNative(); source = [pose(0), pose(4)]
                kwargs = {}; members = [0, 1]
                if mode == 'periodic': kwargs['boundary'] = 'periodic'
                elif mode == 'duplicate': members = [0, 0]
                elif mode == 'noninteger': members = [False, 1]
                elif mode == 'cache': kwargs['graph_cache_entries'] = 0
                elif mode == 'bad_pose': source[0]['orientation'] = [2., 0., 0., 0.]
                with self.assertRaises(ValueError): ConditionalNativeObserver(native, source, members, **kwargs)
                self.assertEqual(native.calls, [])
        native = ToyNative(); source = [pose(0), pose(4)]
        observer = ConditionalNativeObserver(native, source, [0, 1])
        with self.assertRaises(ValueError): observer.classify([pose(0)])
        self.assertEqual(native.calls, [])


if __name__ == '__main__':
    unittest.main()
