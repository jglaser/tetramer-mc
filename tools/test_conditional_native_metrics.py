"""Synthetic decoded journals and native keys only; no physical inputs/queries."""
from __future__ import annotations

import copy
import json
import math
import unittest
from unittest.mock import patch

from conditional_native_metrics import iter_authenticated_endpoints, native_metrics


JOB = dict(id=7, context_index=0, arm='singleton_two_neighbor', initialization='source', stream=1)
MEMBERS = [0, 1]


def pose(x=0.):
    return dict(position=[float(x), 0., 0.], orientation=[1., 0., 0., 0.])


def journal(blocks=4, warmup=1):
    initial = [pose(), pose(4)]
    rows = [dict(kind='initial', block=0, job=copy.deepcopy(JOB), selected=initial,
                 conditional_target=True, sampler_cpu_seconds=.1)]
    for block in range(1, blocks+1):
        # Deliberately incomplete proposal fields: the prior authenticated
        # receipt, not this extraction adapter, owns elementary replay.
        rows.append(dict(kind='local', block=block, accepted='not replayed here'))
        rows.append(dict(kind='two_neighbor_singleton', block=block))
        rows.append(dict(kind='retained_block', block=block, production=block > warmup,
                         selected=copy.deepcopy(initial), sampler_cpu_seconds=block+.1))
    return rows


def observation(block, external=(), internal=(), *, resolved=True, warmup=0, fixed=()):
    external, internal, fixed = map(list, (external, internal, fixed))
    mobile = external+internal
    # A declared synthetic component is enough for pure reduction. No graph
    # classifier is called or reconstructed by these tests.
    bodies = sorted({0, 1} | {i for key in mobile for i in key[:2]})
    component = dict(bodies=bodies, catalogue_consistent=resolved,
                     independent_cycles=0 if resolved else 1)
    return dict(block=block, production=block > warmup, sampler_cpu_seconds=float(block),
                external_native_keys=external, internal_native_keys=internal,
                mobile_related_native_keys=mobile, fixed_native_keys=fixed,
                instantaneous_native_keys=fixed+mobile, mobile_registry_resolved=resolved,
                mobile_native_components=[component])


def metrics(rows, warmup=0, cpu=100.):
    return native_metrics(rows, members=MEMBERS, full_sampler_cpu_seconds=cpu,
                          blocks=len(rows)-1, warmup=warmup)


class EndpointExtraction(unittest.TestCase):
    def extract(self, rows, blocks=4, warmup=1):
        return list(iter_authenticated_endpoints(iter(rows), job=JOB, initial=[pose(), pose(4)],
                                                  blocks=blocks, warmup=warmup))

    def test_retains_baseline_rejected_residence_and_skips_arithmetic(self):
        rows = journal()
        endpoints = self.extract(rows)
        self.assertEqual([row['block'] for row in endpoints], list(range(5)))
        self.assertEqual([row['production'] for row in endpoints], [False, False, True, True, True])
        self.assertTrue(all(row['selected'] == [pose(), pose(4)] for row in endpoints))
        endpoints[0]['selected'][0]['position'][0] = 20
        self.assertEqual(rows[0]['selected'][0]['position'][0], 0)

    def test_default_allocation_requires_all_4609_endpoints(self):
        rows = journal(blocks=4608, warmup=512)
        endpoints = list(iter_authenticated_endpoints(iter(rows), job=JOB, initial=[pose(), pose(4)]))
        self.assertEqual(len(endpoints), 4609)
        self.assertEqual(sum(row['production'] for row in endpoints), 4096)

    def test_missing_duplicate_reordered_unknown_and_tail_records_fail(self):
        for mode in ('missing', 'duplicate', 'reordered', 'unknown', 'tail', 'elementary_first', 'empty'):
            with self.subTest(mode=mode):
                rows = journal()
                if mode == 'missing': rows.pop()
                elif mode == 'duplicate': rows.insert(4, copy.deepcopy(rows[3]))
                elif mode == 'reordered': rows[3]['block'] = 2
                elif mode == 'unknown': rows[1]['kind'] = 'unreviewed_move'
                elif mode == 'tail': rows.append(dict(kind='local'))
                elif mode == 'elementary_first': rows.insert(0, dict(kind='local'))
                elif mode == 'empty': rows = []
                with self.assertRaises(ValueError): self.extract(rows)

    def test_initial_identity_conditional_target_and_production_flags_fail_closed(self):
        for mode in ('job', 'initial_pose', 'target', 'initial_production', 'retained_production', 'retained_job'):
            with self.subTest(mode=mode):
                rows = journal()
                if mode == 'job': rows[0]['job']['stream'] = 3
                elif mode == 'initial_pose': rows[0]['selected'][0] = pose(5)
                elif mode == 'target': rows[0]['conditional_target'] = False
                elif mode == 'initial_production': rows[0]['production'] = True
                elif mode == 'retained_production': rows[3]['production'] = 0
                elif mode == 'retained_job': rows[3]['job'] = dict(JOB, stream=3)
                with self.assertRaises(ValueError): self.extract(rows)

    def test_pose_cpu_and_allocation_validation(self):
        for mode in ('nan_pose', 'bad_quaternion', 'few_poses', 'nan_cpu', 'infinite_cpu', 'boolean_cpu', 'cpu_decrease', 'bool_block'):
            with self.subTest(mode=mode):
                rows = journal()
                if mode == 'nan_pose': rows[3]['selected'][0]['position'][0] = math.nan
                elif mode == 'bad_quaternion': rows[3]['selected'][0]['orientation'] = [2., 0., 0., 0.]
                elif mode == 'few_poses': rows[3]['selected'].pop()
                elif mode == 'nan_cpu': rows[3]['sampler_cpu_seconds'] = math.nan
                elif mode == 'infinite_cpu': rows[3]['sampler_cpu_seconds'] = math.inf
                elif mode == 'boolean_cpu': rows[3]['sampler_cpu_seconds'] = True
                elif mode == 'cpu_decrease': rows[3]['sampler_cpu_seconds'] = 0
                elif mode == 'bool_block': rows[3]['block'] = True
                with self.assertRaises(ValueError): self.extract(rows)
        for blocks, warmup in [(True, 0), (0, 0), (2, 2), (2, -1)]:
            with self.subTest(blocks=blocks, warmup=warmup), self.assertRaises(ValueError):
                self.extract([], blocks, warmup)

    def test_truncated_generator_must_be_exhausted_to_validate_completion(self):
        iterator = iter_authenticated_endpoints(iter(journal()[:-1]), job=JOB,
                                                initial=[pose(), pose(4)], blocks=4, warmup=1)
        self.assertEqual(next(iterator)['block'], 0)
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            list(iterator)


class NativeMetrics(unittest.TestCase):
    def test_empty_consistent_registry_is_not_attachment_and_constant_ess_is_null(self):
        rows = [observation(i) for i in range(5)]
        result = metrics(rows)
        self.assertEqual(result['any_native_attachment_fraction'], 0)
        self.assertEqual(result['resolved_native_attachment_fraction'], 0)
        self.assertEqual(result['mobile_registry_resolved_fraction'], 1)
        self.assertEqual(result['empty_native_registry_fraction'], 1)
        self.assertEqual(result['mobile_native_isolate_fraction_by_member'], {'0': 1., '1': 1.})
        for name in ('external_partners', 'external_motifs', 'internal_motifs'):
            self.assertIsNone(result[name]['presence_ess']['apparent_ess'])
            self.assertEqual(result[name]['counts']['completed_passages'], 0)

    def test_motif_change_with_same_partner_and_internal_only_changes_are_distinct(self):
        rows = [observation(0, [(0, 2, 7)]), observation(1, [(0, 2, 8)]),
                observation(2, [(0, 2, 8)], [(0, 1, 9)]),
                observation(3, [(0, 2, 8)], [(0, 1, 10)])]
        result = metrics(rows)
        self.assertEqual(result['external_partners']['counts']['completed_passages'], 0)
        self.assertEqual(result['external_motifs']['counts']['direct_nonempty_resolved_changes'], 1)
        self.assertEqual(result['internal_motifs']['counts']['enter_nonempty'], 1)
        self.assertEqual(result['internal_motifs']['counts']['direct_nonempty_resolved_changes'], 1)
        self.assertEqual(result['internal_native_attachment_fraction'], 2/3)
        self.assertIsNone(result['external_partners']['presence_ess']['apparent_ess'])

    def test_baseline_does_not_enter_occupancy_and_every_residence_remains(self):
        rows = [observation(0, [(0, 8, 1)], warmup=1), observation(1, [(0, 2, 7)], warmup=1),
                observation(2, [(0, 3, 7)], warmup=1), observation(3, [(0, 3, 7)], warmup=1),
                observation(4, warmup=1)]
        result = metrics(rows, warmup=1, cpu=20)
        self.assertEqual(result['production_samples'], 3)
        self.assertEqual(result['initial_baseline']['external'], ((0, 8, 1),))
        self.assertEqual(result['production_boundary_baseline']['external'], ((0, 2, 7),))
        self.assertEqual(result['external_native_attachment_fraction'], 2/3)
        self.assertEqual(result['external_partners']['counts']['direct_nonempty_resolved_changes'], 1)
        self.assertEqual(result['external_partners']['counts']['leave_nonempty'], 1)
        self.assertEqual(sorted(result['external_partners']['environments']['occupancy'].values()), [1/3, 2/3])
        self.assertEqual(result['external_partners']['rates_per_full_sampler_cpu_second']['completed_passages'], .1)
        self.assertEqual(result['production_window_cpu_seconds'], 3)

    def test_returns_through_empty_are_explicit_not_direct_native_exchange(self):
        a, b = [(0, 2, 7)], [(0, 3, 8)]
        rows = [observation(0, a), observation(1), observation(2, b), observation(3, a)]
        result = metrics(rows)['external_motifs']
        self.assertEqual(result['counts']['leave_nonempty'], 1)
        self.assertEqual(result['counts']['enter_nonempty'], 1)
        self.assertEqual(result['counts']['direct_nonempty_resolved_changes'], 1)
        event, = result['environments']['completed_returns']
        self.assertEqual((event['departure_block'], event['return_block']), (1, 3))
        self.assertTrue(event['passed_through_empty'])
        self.assertTrue(event['returned_environment_nonempty'])
        self.assertTrue(event['return_endpoints_resolved'])
        self.assertFalse(event['passed_through_unresolved'])

    def test_unresolved_and_empty_returns_remain_distinct(self):
        a, b = [(0, 2, 7)], [(0, 3, 8)]
        rows = [observation(0), observation(1, a), observation(2, b, resolved=False),
                observation(3, a), observation(4)]
        result = metrics(rows)
        env = result['external_motifs']
        self.assertEqual(env['counts']['direct_nonempty_resolved_changes'], 0)
        self.assertEqual(env['counts']['direct_nonempty_unresolved_changes'], 2)
        self.assertEqual(env['counts']['completed_returns'], 2)
        self.assertEqual(env['counts']['nonempty_resolved_returns'], 1)
        self.assertTrue(all(e['passed_through_unresolved'] for e in env['environments']['completed_returns']))
        self.assertEqual(result['mobile_cycle_frustration_fraction'], .25)
        self.assertEqual(result['resolved_native_attachment_fraction'], .5)
        self.assertEqual(result['any_native_attachment_fraction'], .75)

    def test_all_alternative_labels_are_preserved_without_double_counting_attachment(self):
        keys = [(0, 2, 7), (0, 2, 8), (1, 2, 7)]
        rows = [observation(i, keys) for i in range(4)]
        before = copy.deepcopy(rows)
        result = metrics(rows)
        self.assertEqual(len(result['external_motifs']['marginal_occupancy']), 3)
        self.assertEqual(len(result['external_partners']['marginal_occupancy']), 2)
        self.assertEqual(result['any_native_attachment_fraction'], 1)
        self.assertEqual(rows, before)
        json.dumps(result, allow_nan=False)

    def test_only_observed_transitions_are_enumerated(self):
        rows = [observation(i, [(0, i+2, i)]) for i in range(81)]
        with patch('analyze_evolving_dimer_benchmark.itertools.combinations',
                   side_effect=AssertionError('No all-pairs environment scan')):
            result = metrics(rows)
        self.assertEqual(result['external_motifs']['counts']['completed_passages'], 80)
        self.assertEqual(len(result['external_motifs']['environments']['transitions']), 80)
        self.assertEqual(result['external_motifs']['counts']['completed_returns'], 0)

    def test_incomplete_concatenated_changed_fixed_or_malformed_native_records_fail(self):
        for mode in ('short', 'concat', 'production', 'cpu', 'fixed', 'partition', 'duplicate', 'internal', 'resolved', 'components'):
            with self.subTest(mode=mode):
                rows = [observation(i, [(0, 2, 7)]) for i in range(4)]
                if mode == 'short': rows.pop()
                elif mode == 'concat': rows[2]['block'] = 0
                elif mode == 'production': rows[2]['production'] = False
                elif mode == 'cpu': rows[2]['sampler_cpu_seconds'] = -1
                elif mode == 'fixed': rows[2]['fixed_native_keys'] = [(3, 4, 8)]; rows[2]['instantaneous_native_keys'].append((3, 4, 8))
                elif mode == 'partition': rows[2]['mobile_related_native_keys'] = []
                elif mode == 'duplicate': rows[2]['external_native_keys'].append((0, 2, 7))
                elif mode == 'internal': rows[2]['internal_native_keys'] = [(0, 4, 8)]
                elif mode == 'resolved': rows[2]['mobile_registry_resolved'] = False
                elif mode == 'components': rows[2]['mobile_native_components'][0]['bodies'] = [0, 2]
                with self.assertRaises(ValueError):
                    native_metrics(rows, members=MEMBERS, full_sampler_cpu_seconds=100, blocks=3, warmup=0)
        rows = [observation(i) for i in range(4)]
        for cpu in (0, True, math.nan, math.inf, 2):
            with self.subTest(cpu=cpu), self.assertRaises(ValueError): metrics(rows, cpu=cpu)


if __name__ == '__main__':
    unittest.main()
