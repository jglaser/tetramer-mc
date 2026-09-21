"""Synthetic graph histories only; no simulator, archived states or audits."""
from __future__ import annotations

import copy
import json
import unittest

from mobile_posterior_metrics import apparent_effective_count, summarize_graph_history


def history(states):
    """Rows are (sweep, native, nonspecific[, source])."""
    return [dict(serial=index-1, sweep=row[0], native_edges=row[1], nonspecific_edges=row[2],
                 source='initial' if index == 0 else row[3] if len(row) > 3 else 'local:local')
            for index, row in enumerate(states)]


class MobilePosteriorMetricsTests(unittest.TestCase):
    def summarize(self, rows, burn=0, cpu=4.):
        return summarize_graph_history(rows, 3, burn, rows[-1]['sweep'], cpu)

    def test_initial_bonds_and_registry_only_changes_are_not_formations(self):
        rows = history([(0, [[0, 1]], [[0, 1], [1, 2]]),
                        (1, [[0, 1]], [[0, 1], [1, 2]]),
                        (2, [[0, 1]], [[0, 1], [1, 2]])])
        rows[1]['registered_keys'] = [[0, 1, 'different-motif']]
        original = copy.deepcopy(rows)
        result = self.summarize(rows)
        self.assertEqual(rows, original)
        native = result['graphs']['native']
        self.assertEqual(native['events'], [])
        self.assertEqual(native['partner_exchanges'], [])
        self.assertEqual(native['event_rates']['full']['counts'], dict(formed_edges=0, detached_edges=0))
        self.assertEqual(native['occupancy']['postburn']['edge_occupancy'][0]['fraction'], 1.)
        episode = native['bodies'][0]['episodes']['full']['episodes'][0]
        self.assertTrue(episode['left_censored'] and episode['right_censored'])
        self.assertEqual(episode['attempted_update_dwell'], 2)
        self.assertEqual(episode['sweep_dwell'], 2)
        self.assertEqual(len(result['individual_edge_apparent_ess']), 6)
        self.assertTrue(all(row['apparent_ess'] is None for row in result['individual_edge_apparent_ess']))
        self.assertIsNone(result['joint_edge_apparent_ess']['apparent_ess'])
        json.dumps(result, allow_nan=False)

    def test_every_attempt_counts_and_residence_preserves_repeats_and_censoring(self):
        rows = history([(0, [[0, 1]], [[0, 1]]),
                        (1, [[0, 1]], [[0, 1]], 'global:uniform'),
                        (1, [], [], 'local:local'),
                        (1, [], [], 'gca:gca'),
                        (2, [[0, 1]], [[0, 1]], 'global:gaussian'),
                        (2, [[0, 1]], [[0, 1]], 'center_shift:center_shift')])
        result = self.summarize(rows)
        native = result['graphs']['native']
        rates = native['event_rates']['postburn']
        self.assertEqual(rates['attempted_updates'], 5)
        self.assertEqual(rates['counts'], dict(formed_edges=1, detached_edges=1))
        self.assertEqual(rates['per_attempted_update'], dict(formed_edges=.2, detached_edges=.2))
        self.assertEqual(rates['per_sampling_CPU_second'], dict(formed_edges=.25, detached_edges=.25))
        self.assertEqual(rates['by_source']['gca:gca']['attempted_updates'], 1)
        self.assertEqual(rates['by_source']['gca:gca']['counts'], dict(formed_edges=0, detached_edges=0))
        body = native['bodies'][0]
        self.assertEqual(body['neighbors_by_observation'], [[1], [1], [], [], [1], [1]])
        episodes = body['episodes']['full']['episodes']
        self.assertEqual([e['attempted_update_dwell'] for e in episodes], [2, 2, 1])
        self.assertEqual([e['sweep_dwell'] for e in episodes], [1, 1, 0])
        self.assertEqual([e['left_censored'] for e in episodes], [True, False, False])
        self.assertEqual([e['right_censored'] for e in episodes], [False, False, True])
        self.assertEqual(body['episodes']['full']['observed_environment_returns'], 1)
        self.assertEqual(native['partner_exchanges'], [])  # Same partner returned.

    def test_sequential_exchange_retains_partner_identity_and_intervening_states(self):
        rows = history([(0, [[0, 1]], [[0, 1]]),
                        (1, [], [], 'local:local'),
                        (1, [], [], 'center_shift:center_shift'),
                        (2, [[0, 2]], [[0, 2]], 'global:involution'),
                        (3, [[0, 2]], [[0, 2]])])
        result = self.summarize(rows, burn=1)
        native = result['graphs']['native']
        self.assertEqual(len(native['partner_exchanges']), 1)
        event = native['partner_exchanges'][0]
        self.assertEqual((event['body'], event['lost_partner'], event['gained_partner']), (0, 1, 2))
        self.assertEqual((event['loss_serial'], event['gain_serial']), (0, 2))
        self.assertEqual(event['attempted_updates_between_loss_and_gain'], 2)
        self.assertEqual([row['serial'] for row in event['intervening_neighbor_states']], [0, 1, 2])
        self.assertEqual([row['neighbors'] for row in event['intervening_neighbor_states']], [[], [], [2]])
        self.assertTrue(event['loss_precedes_postburn_window'])
        self.assertEqual(native['exchange_rates']['postburn']['counts']['completed_exchanges'], 1)
        self.assertEqual(native['exchange_rates']['completed_postburn_with_preburn_loss'], 1)
        self.assertEqual(native['event_rates']['postburn']['counts'], dict(formed_edges=1, detached_edges=0))
        episode = native['bodies'][0]['episodes']['postburn']['episodes'][0]
        self.assertTrue(episode['left_censored'])
        self.assertEqual(episode['first_observed_serial'], 1)  # Last endpoint of burn sweep.
        self.assertEqual(episode['attempted_update_dwell'], 1)

    def test_same_update_replacement_and_reattachment_cancel_do_not_count(self):
        rows = history([(0, [[0, 1]], [[0, 1]]),
                        (1, [[0, 2]], [[0, 2]]),
                        (2, [[0, 1]], [[0, 1]]),
                        (3, [[0, 1], [0, 2]], [[0, 1], [0, 2]])])
        result = self.summarize(rows)
        self.assertEqual(result['graphs']['native']['partner_exchanges'], [])
        self.assertEqual(result['graphs']['native']['pending_partner_losses_at_end'], [])
        replacements = result['graphs']['native']['same_attempt_partner_replacements']
        self.assertEqual([(row['serial'], row['body'], row['lost_partner'], row['gained_partner'])
                          for row in replacements], [(0, 0, 1, 2), (1, 0, 2, 1)])
        self.assertEqual(replacements[0]['neighbors_before'], [1])
        self.assertEqual(replacements[0]['neighbors_after'], [2])
        rates = result['graphs']['native']['replacement_rates']['postburn']
        self.assertEqual(rates['counts']['same_attempt_replacements'], 2)
        self.assertEqual(rates['per_attempted_update']['same_attempt_replacements'], 2/3)
        self.assertEqual(rates['per_sampling_CPU_second']['same_attempt_replacements'], .5)

    def test_replacement_counts_are_separate_and_rejections_cannot_change_graph(self):
        rows = history([(0, [[0, 1]], [[0, 1]]),
                        (1, [[0, 2]], [[0, 2]], 'frozen-posterior:involution'),
                        (2, [[0, 2]], [[0, 2]], 'local:local')])
        rows[1]['accepted'] = True
        rows[2]['accepted'] = False
        result = self.summarize(rows, burn=1)
        native = result['graphs']['native']
        self.assertEqual(native['replacement_rates']['full']['counts']['same_attempt_replacements'], 1)
        self.assertEqual(native['replacement_rates']['postburn']['counts']['same_attempt_replacements'], 0)
        self.assertEqual(native['partner_exchanges'], [])
        rows[1]['accepted'] = False
        with self.assertRaisesRegex(ValueError, 'Rejected physical update'):
            self.summarize(rows)

    def test_apparent_ess_validates_cpu_even_for_constants_or_short_records(self):
        for cpu in (0., -1., float('inf'), float('nan')):
            for values in ([1, 1, 1], [1], []):
                with self.subTest(cpu=cpu, values=values), self.assertRaisesRegex(ValueError, 'CPU'):
                    apparent_effective_count(values, cpu)

    def test_replacement_loss_is_never_reused_by_a_later_sequential_exchange(self):
        rows = history([(0, [[0, 1]], [[0, 1]]),
                        (1, [[0, 2]], [[0, 2]]),
                        (2, [[0, 2], [0, 3]], [[0, 2], [0, 3]])])
        result = summarize_graph_history(rows, 4, 0, 2, 2.)
        native = result['graphs']['native']
        self.assertEqual(native['partner_exchanges'], [])
        self.assertEqual([(event['body'], event['lost_partner'], event['gained_partner'])
                          for event in native['same_attempt_partner_replacements']], [(0, 1, 2)])
        self.assertFalse(any(event['body'] == 0 for event in native['pending_partner_losses_at_end']))

    def test_existing_pending_loss_can_resolve_during_a_later_replacement(self):
        rows = history([(0, [[0, 1], [0, 2]], [[0, 1], [0, 2]]),
                        (1, [[0, 2]], [[0, 2]]),
                        (2, [[0, 3]], [[0, 3]]),
                        (3, [[0, 1], [0, 3]], [[0, 1], [0, 3]])])
        result = summarize_graph_history(rows, 4, 0, 3, 3.)
        native = result['graphs']['native']
        self.assertEqual([(event['body'], event['lost_partner'], event['gained_partner'])
                          for event in native['partner_exchanges']], [(0, 1, 3)])
        self.assertEqual([(event['body'], event['lost_partner'], event['gained_partner'])
                          for event in native['same_attempt_partner_replacements']], [(0, 2, 3)])

    def test_endpoint_halves_and_six_edge_ess_columns_use_full_postburn_cpu(self):
        rows = history([(0, [], []),
                        (1, [[0, 1]], [[0, 1]]),
                        (1, [], [], 'center_shift:center_shift'),
                        (2, [], [[1, 2]]),
                        (3, [[0, 1]], [[0, 1], [1, 2]]),
                        (4, [[0, 1]], [[0, 1], [1, 2]])])
        result = self.summarize(rows, cpu=10.)
        native = result['graphs']['native']
        self.assertEqual(result['postburn_sampling']['endpoint_observation_indices'], [2, 3, 4, 5])
        self.assertEqual(native['occupancy']['postburn']['edge_occupancy'][0]['fraction'], .5)
        self.assertEqual(native['occupancy']['first_half']['edge_occupancy'][0]['fraction'], 0.)
        self.assertEqual(native['occupancy']['second_half']['edge_occupancy'][0]['fraction'], 1.)
        columns = result['individual_edge_apparent_ess']
        self.assertEqual([(row['kind'], row['bodies']) for row in columns],
                         [(kind, edge) for kind in ('native', 'nonspecific') for edge in ([0, 1], [0, 2], [1, 2])])
        varying = columns[0]
        self.assertEqual(varying['samples'], 4)
        self.assertIsNotNone(varying['apparent_ess'])
        self.assertEqual(varying['apparent_ess_per_sampling_CPU_second'], varying['apparent_ess']/10.)
        self.assertIsNone(columns[1]['apparent_ess'])
        self.assertEqual(native['graph_indicator_apparent_ess']['any_edge']['samples'], 4)
        self.assertIsNone(native['graph_indicator_apparent_ess']['connected']['apparent_ess'])

    def test_single_endpoint_and_invalid_histories_are_explicit(self):
        rows = history([(0, [], []), (1, [[0, 1]], [[0, 1]])])
        result = self.summarize(rows)
        first = result['graphs']['native']['occupancy']['first_half']
        self.assertEqual(first['samples'], 0)
        self.assertIsNone(first['edge_occupancy'][0]['fraction'])
        self.assertIsNone(result['joint_edge_apparent_ess']['apparent_ess'])
        self.assertIsNone(apparent_effective_count([], 1.)['apparent_ess'])
        for mutation in ('serial', 'missing_sweep', 'duplicate_edge'):
            bad = copy.deepcopy(rows)
            if mutation == 'serial': bad[-1]['serial'] = 1
            elif mutation == 'missing_sweep': bad[-1]['sweep'] = 2
            else: bad[-1]['native_edges'] = [[0, 1], [1, 0]]
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.summarize(bad)
        with self.assertRaises(ValueError):
            self.summarize(rows, cpu=0.)

    def test_ten_thousand_self_loops_remain_ten_thousand_attempts(self):
        rows = history([(0, [[0, 1]], [[0, 1]])]+[(1+index//5, [[0, 1]], [[0, 1]]) for index in range(10000)])
        result = self.summarize(rows, burn=400, cpu=100.)
        self.assertEqual(len(result['observation_axis']), 10001)
        self.assertEqual(result['postburn_sampling']['physical_attempts'], 8000)
        self.assertEqual(result['joint_edge_apparent_ess']['samples'], 1600)
        self.assertIsNone(result['joint_edge_apparent_ess']['apparent_ess'])
        self.assertEqual(result['graphs']['native']['events'], [])
        self.assertEqual(len(result['graphs']['native']['bodies'][0]['neighbors_by_observation']), 10001)


if __name__ == '__main__':
    unittest.main()
