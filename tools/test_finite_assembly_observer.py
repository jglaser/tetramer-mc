"""Synthetic instantaneous-graph and retained-series observer tests.

No protein geometry, trajectory generation, or physical normalizer is evaluated.
Graph examples are derived directly from elementary rigid transformations.
"""
import copy
import json
import math
import unittest

import numpy as np

import finite_assembly_observer as observer


def motif(label, translation, quaternion=(1., 0., 0., 0.)):
    return dict(id=label, relative_position=list(translation), relative_orientation=list(quaternion))


LINE_MOTIFS = [motif(0, [1., 0., 0.]), motif(1, [-1., 0., 0.]),
               motif(2, [2., 0., 0.]), motif(3, [-2., 0., 0.])]


def chain_keys(body_ids):
    return [(min(a, b), max(a, b), 0 if a < b else 1)
            for a, b in zip(body_ids[:-1], body_ids[1:])]


def keys_to_edges(keys):
    return sorted({(a, b) for a, b, _ in keys})


class InstantaneousGraphTests(unittest.TestCase):
    def summary(self, n, native=(), exclusion=None, motifs=None, positions=None, lengths=None):
        native = list(native)
        return observer.graph_summary(n, keys_to_edges(native) if exclusion is None else exclusion,
            native, LINE_MOTIFS if motifs is None else motifs,
            np.column_stack((np.arange(n), np.zeros((n, 2)))) if positions is None else positions,
            box_lengths=lengths)

    def test_isolates_and_competing_contacts_partition_without_native_labels(self):
        self.assertEqual(observer.REGIONS, ('dispersed', 'contact_no_entry', 'registered_small',
                                          'registered_eight', 'registered_growth', 'remaining'))
        dispersed = self.summary(4)
        self.assertEqual(dispersed['environment'], 'dispersed')
        self.assertEqual(dispersed['exclusion_components'], [[0], [1], [2], [3]])
        self.assertEqual(dispersed['largest_exclusion'], 1)
        self.assertEqual(dispersed['largest_native_raw'], 1)
        self.assertEqual(dispersed['largest_native_certified'], 1)
        self.assertTrue(dispersed['native_registry_resolved'])
        competing = self.summary(4, exclusion=[(0, 1), (1, 2)])
        self.assertEqual(competing['environment'], 'contact_no_entry')
        self.assertEqual(competing['largest_exclusion'], 3)
        self.assertEqual(competing['largest_native_certified'], 1)

    def test_tree_and_consistent_cycle_are_certified(self):
        keys = [(0, 1, 0), (1, 2, 0)]
        tree = self.summary(4, keys)
        cycle = self.summary(4, keys+[(0, 2, 2)])
        for report, count in [(tree, 0), (cycle, 1)]:
            with self.subTest(cycles=count):
                self.assertEqual(report['environment'], 'registered_small')
                self.assertEqual(report['largest_native_certified'], 3)
                self.assertTrue(report['native_registry_resolved'])
                component = next(c for c in report['native_components'] if len(c['bodies']) == 3)
                self.assertTrue(component['catalogue_consistent'])
                self.assertTrue(component['ordinary_space_lift'])
                self.assertTrue(component['certified'])
                self.assertEqual(component['independent_cycles'], count)

    def test_frustrated_cycle_does_not_erase_separate_good_component(self):
        # Three +1 edges cannot close a triangle; the disconnected four-body
        # chain is nevertheless independently certifiable.
        bad = [(0, 1, 0), (1, 2, 0), (0, 2, 0)]
        good = chain_keys([3, 4, 5, 6])
        report = self.summary(8, bad+good)
        components = {tuple(c['bodies']): c for c in report['native_components']}
        self.assertFalse(components[(0, 1, 2)]['catalogue_consistent'])
        self.assertFalse(components[(0, 1, 2)]['certified'])
        self.assertTrue(components[(3, 4, 5, 6)]['certified'])
        self.assertEqual(report['largest_native_raw'], 4)
        self.assertEqual(report['largest_native_certified'], 4)
        self.assertFalse(report['native_registry_resolved'])
        self.assertEqual(report['environment'], 'remaining')

    def test_alternative_labels_are_alternatives_not_extra_constraints(self):
        report = self.summary(3, [(0, 1, 0), (1, 2, 0), (0, 2, 0), (0, 2, 2)])
        self.assertTrue(report['native_registry_resolved'])
        self.assertEqual(report['largest_native_certified'], 3)
        self.assertEqual(report['native_components'][0]['independent_cycles'], 1)
        self.assertEqual(report['environment'], 'registered_small')

    def test_periodic_boundary_crossing_is_not_itself_winding(self):
        positions = np.asarray([[9.5, 0., 0.], [.5, 0., 0.], [1.5, 0., 0.]])
        report = self.summary(3, [(0, 1, 0), (1, 2, 0), (0, 2, 2)],
                              positions=positions, lengths=[10., 10., 10.])
        self.assertTrue(report['native_components'][0]['ordinary_space_lift'])
        self.assertTrue(report['native_components'][0]['certified'])
        self.assertEqual(report['environment'], 'registered_small')

    def test_periodic_winding_is_unresolved_even_when_catalogue_cycle_closes(self):
        # Measured minimum-image edges wind once around x. The assigned ideal
        # catalogue labels are algebraically consistent, so the image check
        # supplies an independent reason not to certify this as a finite cluster.
        positions = np.asarray([[0., 0., 0.], [4., 0., 0.], [8., 0., 0.]])
        report = self.summary(3, [(0, 1, 0), (1, 2, 0), (0, 2, 2)],
                              positions=positions, lengths=[12., 12., 12.])
        component = report['native_components'][0]
        self.assertIsNone(component['catalogue_consistent'])
        self.assertFalse(component['ordinary_space_lift'])
        self.assertFalse(component['certified'])
        self.assertFalse(report['native_cycle_frustrated'])
        self.assertFalse(report['native_registry_resolved'])
        self.assertEqual(report['environment'], 'remaining')

    def test_body_relabeling_with_reciprocal_rigid_motifs_preserves_registration(self):
        c = math.sqrt(.5)
        # A four-step square: Rz(pi/2), t=(1,0), with inverse
        # Rz(-pi/2), t=-R^T(1,0)=(0,1). Negating t alone is wrong.
        motifs = [motif(0, [1., 0., 0.], [c, 0., 0., c]),
                  motif(1, [0., 1., 0.], [c, 0., 0., -c])]
        keys = [(0, 1, 0), (1, 2, 0), (2, 3, 0), (0, 3, 1)]
        positions = np.asarray([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]])
        original = self.summary(4, keys, motifs=motifs, positions=positions)
        permutation = [2, 0, 3, 1]
        relabeled = []
        permuted_positions = np.empty_like(positions)
        for i, j in enumerate(permutation):
            permuted_positions[j] = positions[i]
        for a, b, label in keys:
            i, j = permutation[a], permutation[b]
            relabeled.append((min(i, j), max(i, j), label if i < j else 1-label))
        changed = self.summary(4, relabeled, motifs=motifs, positions=permuted_positions)
        for report in (original, changed):
            self.assertTrue(report['native_registry_resolved'])
            self.assertEqual(report['largest_native_certified'], 4)
            self.assertEqual(report['environment'], 'registered_small')
            self.assertEqual(report['native_components'][0]['independent_cycles'], 1)

    def test_reference_size_is_instantaneous_size_not_seed_body_identity(self):
        bodies = [2, 4, 6, 8, 10, 11, 0, 1, 3]
        positions = np.zeros((12, 3))
        for x, body in enumerate(bodies):
            positions[body, 0] = x
        for size, region in [(7, 'registered_small'), (8, 'registered_eight'), (9, 'registered_growth')]:
            report = self.summary(12, chain_keys(bodies[:size]), positions=positions)
            with self.subTest(size=size):
                self.assertEqual(report['environment'], region)
                self.assertEqual(report['largest_native_certified'], size)

    def test_duplicate_records_do_not_manufacture_cycles_or_growth(self):
        keys = [(0, 1, 0), (1, 2, 0)]
        report = self.summary(4, keys+keys, exclusion=[(0, 1), (0, 1), (1, 2)])
        self.assertEqual(report['largest_native_raw'], 3)
        self.assertEqual(report['native_components'][0]['independent_cycles'], 0)

    def test_bad_body_indices_and_unknown_motifs_fail_closed(self):
        cases = [dict(native=[(-1, 1, 0)]), dict(native=[(0, 4, 0)]),
                 dict(native=[(0, 0, 0)]), dict(native=[(0, 1, 99)]),
                 dict(native=[(False, 1, 0)]), dict(exclusion=[(-1, 2)]),
                 dict(exclusion=[(1, 4)]), dict(exclusion=[(1, 1)])]
        for case in cases:
            with self.subTest(case=case), self.assertRaises(ValueError):
                self.summary(4, **case)


def make_observation(n, native=(), exclusion=None):
    edges = keys_to_edges(native) if exclusion is None else exclusion
    positions = np.column_stack((np.arange(n), np.zeros((n, 2))))
    return dict(tokens=[(a, b, 'surface', 'surface') for a, b in edges],
                **observer.graph_summary(n, edges, native, LINE_MOTIFS, positions))


def retained_fixture():
    dispersed = make_observation(12)
    competing = make_observation(12, exclusion=[(0, 1)])
    remaining = make_observation(12, native=[(0, 1, 0), (1, 2, 0), (0, 2, 0)])
    registered = make_observation(12, native=chain_keys(list(range(8))))
    observations = [dispersed, dispersed, competing, competing, remaining,
                    registered, registered, competing, dispersed]
    frames = [dict(sweep=i, sampler_cpu_seconds=.1+.5*i) for i in range(len(observations))]
    window = dict(indices=list(range(9)), sample_indices=list(range(1, 9)), cpu_seconds=4., cadence_sweeps=1)
    return observations, frames, window


class RetainedSeriesTests(unittest.TestCase):
    def test_every_retained_endpoint_including_repeats_and_remaining_contributes(self):
        observations, frames, window = retained_fixture()
        result = observer.summarize_series(observations, frames, window)
        self.assertEqual(result['fingerprint']['samples'], 8)
        self.assertEqual(result['labels_by_frame'], [o['environment'] for o in observations])
        expected = {'dispersed': 2, 'contact_no_entry': 3, 'registered_small': 0,
                    'registered_eight': 2, 'registered_growth': 0, 'remaining': 1}
        for region, count in expected.items():
            self.assertEqual(result['environment_occupancies'][region]['observations'], count)
            self.assertEqual(result['environment_occupancies'][region]['fraction'], count/8.)
        joint = result['size_pair_occupancies']
        self.assertEqual(sum(row['observations'] for row in joint), 8)
        self.assertAlmostEqual(sum(row['fraction'] for row in joint), 1.)
        counts = {(row['largest_exclusion'], row['largest_native_raw'], row['largest_native_certified'],
                   row['registry_resolved']): row['observations'] for row in joint}
        self.assertEqual(counts, {(1, 1, 1, True): 2, (2, 1, 1, True): 3,
                                  (3, 3, 1, False): 1, (8, 8, 8, True): 2})
        self.assertEqual(result['fingerprint']['sampling_CPU_seconds'], 4.)
        self.assertAlmostEqual(result['fingerprint']['apparent_ess_per_sampling_CPU_second'],
                               result['fingerprint']['apparent_ess']/4.)
        json.dumps(result, allow_nan=False)

    def test_cpu_changes_efficiency_but_not_occupancy_or_effective_count(self):
        observations, frames, window = retained_fixture()
        first = observer.summarize_series(observations, frames, window)
        second = observer.summarize_series(observations, frames, dict(window, cpu_seconds=12.))
        self.assertEqual(first['size_pair_occupancies'], second['size_pair_occupancies'])
        self.assertEqual(first['fingerprint']['apparent_ess'], second['fingerprint']['apparent_ess'])
        self.assertAlmostEqual(first['fingerprint']['apparent_ess_per_sampling_CPU_second'],
                               3.*second['fingerprint']['apparent_ess_per_sampling_CPU_second'])

    def test_component_number_histograms_keep_all_body_mass_and_unresolved_deficit(self):
        result = observer.summarize_series(*retained_fixture())
        histograms = result['component_number_statistics']
        for name in ('exclusion', 'native_raw'):
            row = histograms[name]
            self.assertEqual(row['sizes'], list(range(1, 13)))
            self.assertAlmostEqual(sum(n*c for n, c in zip(row['sizes'], row['mean_component_counts'])), 12.)
        certified = histograms['native_certified']
        # One of eight endpoints has a frustrated three-body component. Its
        # bodies stay in the raw distribution and are absent only from the
        # explicitly certified lower-bound distribution.
        self.assertAlmostEqual(sum(n*c for n, c in zip(certified['sizes'], certified['mean_component_counts'])), 12.-3./8.)

    def test_native_fingerprint_detects_registry_change_at_unchanged_contact_and_size(self):
        observations = [make_observation(3, native=[(0, 1, 0 if i % 2 else 2)]) for i in range(9)]
        frames = [dict(sweep=i) for i in range(9)]
        window = dict(indices=list(range(9)), sample_indices=list(range(1, 9)), cpu_seconds=4., cadence_sweeps=1)
        result = observer.summarize_series(observations, frames, window)
        self.assertIsNone(result['fingerprint']['apparent_ess'])
        self.assertIsNotNone(result['native_fingerprint']['apparent_ess'])
        self.assertEqual(result['native_fingerprint']['observed_token_dictionary'], [[0, 1, 0], [0, 1, 2]])
        self.assertIsNone(result['size_statistics']['largest_native_certified']['apparent_ess']['apparent_ess'])

    def test_completed_exchanges_allow_remaining_and_preserve_unassessed_equilibrium(self):
        result = observer.summarize_series(*retained_fixture())
        exchanges = result['environment_exchanges']
        pair = next(p for p in exchanges['pairs'] if set(p['regions']) == {'contact_no_entry', 'registered_eight'})
        self.assertEqual(pair['forward'], 1)
        self.assertEqual(pair['reverse'], 1)
        self.assertEqual(pair['completed_nonoverlapping_roundtrips'], 1)
        self.assertEqual(pair['completed_passages_per_sampler_cpu_second'], .5)
        self.assertEqual(pair['equilibrium_interpretation']['status'], 'unassessed_equilibrium_weights')
        self.assertTrue(any(e['intervening_saved_labels'] == ['remaining'] for e in exchanges['events']))

    def test_constant_contacts_have_null_ess_even_when_many_frames_are_saved(self):
        observations, frames, window = retained_fixture()
        observations = [copy.deepcopy(observations[2]) for _ in observations]
        result = observer.summarize_series(observations, frames, window)
        self.assertEqual(result['fingerprint']['samples'], 8)
        self.assertIsNone(result['fingerprint']['apparent_ess'])
        self.assertIsNone(result['fingerprint']['apparent_ess_per_sampling_CPU_second'])
        self.assertIsNone(result['native_fingerprint']['apparent_ess'])
        self.assertEqual(result['environment_occupancies']['contact_no_entry']['fraction'], 1.)
        self.assertIsNone(result['environment_occupancies']['contact_no_entry']['apparent_ess']['apparent_ess'])
        self.assertEqual(sum(r['observations'] for r in result['size_pair_occupancies']), 8)

    def test_empty_contact_dictionary_is_valid_but_does_not_claim_independence(self):
        observations, frames, window = retained_fixture()
        observations = [copy.deepcopy(observations[0]) for _ in observations]
        result = observer.summarize_series(observations, frames, window)
        self.assertEqual(result['fingerprint']['observed_token_dictionary'], [])
        self.assertIsNone(result['fingerprint']['apparent_ess'])
        self.assertEqual(result['environment_occupancies']['dispersed']['fraction'], 1.)

    def test_window_baseline_is_excluded_from_occupancy_but_retained_for_passages(self):
        observations, frames, window = retained_fixture()
        window.update(indices=list(range(4, 9)), sample_indices=list(range(5, 9)), cpu_seconds=2.)
        result = observer.summarize_series(observations, frames, window)
        self.assertEqual(result['fingerprint']['samples'], 4)
        self.assertEqual(result['environment_occupancies']['remaining']['observations'], 0)
        self.assertEqual(result['environment_occupancies']['registered_eight']['fraction'], .5)
        self.assertEqual(sum(r['observations'] for r in result['size_pair_occupancies']), 4)

    def test_missing_inventory_skipped_endpoints_irregular_cadence_or_cpu_fail(self):
        for mode in ('inventory', 'skip', 'irregular', 'cpu', 'unknown-region'):
            observations, frames, window = retained_fixture()
            if mode == 'inventory': observations.pop()
            elif mode == 'skip': window['sample_indices'].remove(3)
            elif mode == 'irregular': frames[3]['sweep'] += 1
            elif mode == 'cpu': window['cpu_seconds'] = 0.
            else: observations[4] = dict(observations[4], environment='unknown')
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                observer.summarize_series(observations, frames, window)


def pose(position):
    return dict(position=list(position), orientation=[1., 0., 0., 0.])


class ToyNative:
    motifs = LINE_MOTIFS

    def __init__(self):
        self.calls = []

    def classify_pair(self, anchor, moving):
        self.calls.append((anchor, moving))
        # Prescribed catalogue choices are sufficient for testing composition;
        # this mock is deliberately not a protein-native geometry predicate.
        return [dict(motif_id=0)]


class ComposedObserverTests(unittest.TestCase):
    shape = dict(atoms=[dict(center=[0., 0., 0.], radius=.2)])

    def configured(self, poses, periodic=False, radius=1.5):
        config = dict(initial_poses=poses, boundary=dict(kind='periodic') if periodic else dict(kind='spherical', radius=20.),
                      box_lengths=[12., 12., 12.] if periodic else [40., 40., 40.], depletant_radius=radius)
        native = ToyNative()
        return observer.FiniteAssemblyObserver(self.shape, ['surface'], config, native), native

    def test_hard_invalid_state_still_fails_before_any_native_calls(self):
        poses = [pose([0., 0., 0.]), pose([.3, 0., 0.])]
        combined, native = self.configured(poses)
        with self.assertRaisesRegex(ValueError, 'hard overlap'):
            combined.classify(poses)
        self.assertEqual(native.calls, [])

    def test_inventory_cannot_change_between_observations(self):
        poses = [pose([0., 0., 0.]), pose([2., 0., 0.])]
        combined, native = self.configured(poses)
        with self.assertRaisesRegex(ValueError, 'inventory'):
            combined.classify(poses[:1])
        self.assertEqual(native.calls, [])

    def test_periodic_winding_is_retained_as_remaining_without_legacy_exception(self):
        poses = [pose([0., 0., 0.]), pose([4., 0., 0.]), pose([8., 0., 0.])]
        combined, native = self.configured(poses, periodic=True, radius=2.)
        result = combined.classify(poses)
        self.assertEqual(result['environment'], 'remaining')
        self.assertTrue(result['native_periodic_winding'])
        self.assertFalse(result['native_cycle_frustrated'])
        self.assertEqual(len(result['instantaneous_native_keys']), 3)
        self.assertEqual(len(result['tokens']), 3)
        self.assertEqual(len(native.calls), 3)
        # Pair 0--2 is sent to the frozen classifier in its minimum image.
        self.assertEqual(native.calls[1][1]['position'], [-4., 0., 0.])


if __name__ == '__main__':
    unittest.main()
