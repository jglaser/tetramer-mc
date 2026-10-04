"""Synthetic saved-summary fixtures; no native classification or trajectory IO."""
import copy
import itertools
import json
import unittest

import conditional_native_comparisons as compare


def environment(values=(), *, cpu=10., ess=None, count=0):
    label = compare.key(values)
    return dict(environments=dict(occupancy={label: 1.}), environment_values={label: values},
        marginal_occupancy=[dict(key=k, fraction=1.) for k in values],
        presence_ess=dict(samples=4096, sampling_CPU_seconds=cpu, apparent_ess=ess,
            apparent_ess_per_sampling_CPU_second=None if ess is None else ess/cpu,
            reason='Constant synthetic descriptor' if ess is None else 'synthetic'),
        counts={name: count for name in compare.COUNTS},
        rates_per_full_sampler_cpu_second={name: count/cpu for name in compare.COUNTS})


def fixture():
    rows = []
    for context, arm, start, stream in itertools.product(range(4), compare.ARMS, compare.STARTS, range(4)):
        ident = dict(context_index=context, arm=arm, initialization=start, stream=stream)
        fractions = {name: 0. for name in compare.FRACTIONS}
        fractions.update(empty_native_registry_fraction=1., mobile_registry_resolved_fraction=1.)
        rows.append(dict(chain_id=f'{context}-{arm}-{start}-{stream}', identity=ident,
            job=dict(id=len(rows), **ident), metrics=dict(schema='conditional-native-metrics-v1',
            production_samples=4096, retained_endpoints=4609, full_sampler_cpu_seconds=10.,
            mobile_native_isolate_fraction_by_member={str(2*context): 1., str(2*context+1): 1.},
            **fractions, **{name: environment() for name in compare.DESCRIPTORS})))
    return rows


def vary(row, *, cpu=20., ess=64., count=8):
    row['metrics']['full_sampler_cpu_seconds'] = cpu
    for name in compare.DESCRIPTORS:
        empty, full = (), ((0, 4),) if name == 'external_partners' else ((0, 4, 7),)
        item = environment(full, cpu=cpu, ess=ess, count=count)
        item['environments']['occupancy'] = {compare.key(empty): .5, compare.key(full): .5}
        item['environment_values'][compare.key(empty)] = empty
        item['marginal_occupancy'][0]['fraction'] = .5
        row['metrics'][name] = item


class NativeComparisonTests(unittest.TestCase):
    def test_complete_inventory_determinism_and_no_input_mutation(self):
        rows = fixture(); saved = copy.deepcopy(rows)
        result = compare.reduce_native_comparisons(rows)
        self.assertEqual((len(result['groups']), len(result['arm_contrasts']), len(result['initialization_agreement'])), (32, 192, 16))
        self.assertEqual(rows, saved)
        self.assertEqual(result, compare.reduce_native_comparisons(list(reversed(rows))))
        self.assertFalse(result['context_pooling'] or result['ess_reestimated'] or result['equilibrium_established'])
        json.dumps(result, allow_nan=False)
        for contrast in result['arm_contrasts']:
            self.assertTrue(contrast['local_rng_roles_paired'])
            self.assertEqual(contrast['collective_rng_roles_paired'],
                contrast['left_arm'] == 'singleton_two_neighbor' and contrast['right_arm'] == 'singleton_two_neighbor_unfused')

    def test_all_empty_agreement_keeps_undefined_ess_and_no_mixing_claim(self):
        result = compare.reduce_native_comparisons(fixture())
        for group in result['groups']:
            self.assertEqual(group['presence_ess_defined_streams'], dict.fromkeys(compare.DESCRIPTORS, 0))
            self.assertEqual(group['presence_ess_undefined_streams'], dict.fromkeys(compare.DESCRIPTORS, 4))
            self.assertEqual(len(group['independent_stream_metrics']), 4)
        for init in result['initialization_agreement']:
            self.assertTrue(init['both_starts_all_empty'])
            self.assertFalse(init['initialization_agreement_establishes_mixing'])
            self.assertFalse(init['pairwise_distances_are_independent'])
            for descriptor in init['descriptors'].values():
                self.assertTrue(descriptor['both_starts_all_constant'] and descriptor['no_nonempty_resolved_returns'])
                self.assertEqual(descriptor['mean_environment_total_variation'], 0.)
                self.assertEqual(tuple(len(descriptor[k]) for k in ('cross_start_comparisons',
                    'within_source_comparisons', 'within_proposal_prepared_comparisons')), (16, 6, 6))

    def test_matched_contrasts_use_full_cpu_and_propagate_nulls(self):
        rows = fixture()
        for row in rows:
            if row['identity']['arm'] == 'singleton_two_neighbor': vary(row, cpu=20., ess=80., count=8)
            elif row['identity']['arm'] == 'singleton_two_neighbor_unfused': vary(row, cpu=10., ess=80., count=2)
        result = compare.reduce_native_comparisons(rows)
        for contrast in result['arm_contrasts']:
            for descriptor in contrast['descriptors'].values():
                delta = descriptor['presence_ess_per_full_sampler_cpu_second_difference']
                if contrast['left_arm'] == 'singleton_two_neighbor':
                    self.assertEqual(delta, 4.)
                    self.assertAlmostEqual(descriptor['rates_per_full_sampler_cpu_second_difference']['completed_returns'], -.2)
                else: self.assertIsNone(delta)
        fused = next(g for g in result['groups'] if g['arm'] == 'singleton_two_neighbor')
        self.assertEqual(fused['presence_ess_defined_streams']['external_motifs'], 4)
        self.assertEqual(fused['independent_stream_metrics'][0]['full_sampler_cpu_seconds'], 20.)

    def test_equal_mixture_means_do_not_hide_stuck_stream_disagreement(self):
        rows = fixture()
        for row in rows:
            motif = ((0, 4, 7 if row['identity']['stream'] < 2 else 8),)
            row['metrics']['external_motifs'] = environment(motif)
            row['metrics']['any_native_attachment_fraction'] = 1.
        init = compare.reduce_native_comparisons(rows)['initialization_agreement'][0]
        desc = init['descriptors']['external_motifs']
        self.assertEqual(desc['mean_environment_total_variation'], 0.)
        self.assertEqual({p['environment_total_variation'] for p in desc['cross_start_comparisons']}, {0., 1.})
        self.assertEqual({p['environment_total_variation'] for p in desc['within_source_comparisons']}, {0., 1.})
        self.assertTrue(desc['both_starts_all_constant'] and desc['no_nonempty_resolved_returns'])
        self.assertFalse(init['both_starts_all_empty'] or init['initialization_agreement_establishes_mixing'])

    def test_baseline_only_values_and_overlapping_motif_marginals_are_not_probability_masses(self):
        rows = fixture()
        for row in rows:
            if row['identity']['initialization'] == 'source':
                value = ((0, 4, 7), (0, 4, 8))
                row['metrics']['external_motifs'] = environment(value)
            else:
                row['metrics']['external_motifs']['environment_values'][compare.key(((0, 4, 99),))] = ((0, 4, 99),)
        desc = compare.reduce_native_comparisons(rows)['initialization_agreement'][0]['descriptors']['external_motifs']
        self.assertEqual(desc['mean_environment_total_variation'], 1.)
        self.assertEqual(desc['mean_marginal_occupancy_max_difference'], 1.)
        self.assertEqual(desc['proposal_prepared_mean_occupancy'], {compare.key(()): 1.})

    def test_inventory_missing_duplicate_foreign_and_mismatched_job_are_rejected(self):
        for mode in ('missing', 'duplicate', 'foreign', 'job', 'chain_id'):
            with self.subTest(mode=mode):
                rows = fixture()
                if mode == 'missing': rows.pop()
                elif mode == 'duplicate': rows[-1] = copy.deepcopy(rows[0])
                elif mode == 'foreign': rows[0]['identity']['context_index'] = 4
                elif mode == 'job': rows[0]['job']['stream'] = 3
                else: rows[-1]['chain_id'] = rows[0]['chain_id']
                with self.assertRaises(ValueError): compare.reduce_native_comparisons(rows)

    def test_allocation_cpu_ess_and_event_denominators_are_checked(self):
        for mode in ('samples', 'endpoints', 'cpu', 'ess_cpu', 'ess_samples', 'ess_rate', 'rate', 'missing_reason'):
            with self.subTest(mode=mode):
                rows = fixture(); metrics = rows[0]['metrics']; descriptor = metrics['external_motifs']
                if mode == 'samples': metrics['production_samples'] = 4095
                elif mode == 'endpoints': metrics['retained_endpoints'] = 4608
                elif mode == 'cpu': metrics['full_sampler_cpu_seconds'] = float('nan')
                elif mode == 'ess_cpu': descriptor['presence_ess']['sampling_CPU_seconds'] = 9.
                elif mode == 'ess_samples': descriptor['presence_ess']['samples'] = 4095
                elif mode == 'ess_rate':
                    vary(rows[0]); rows[0]['metrics']['external_motifs']['presence_ess']['apparent_ess_per_sampling_CPU_second'] *= 2.
                elif mode == 'rate': descriptor['rates_per_full_sampler_cpu_second']['enter_nonempty'] = 1.
                else: descriptor['presence_ess'].pop('reason')
                with self.assertRaises(ValueError): compare.reduce_native_comparisons(rows)

    def test_categorical_mass_and_finite_marginal_requirements(self):
        for mode in ('mass', 'negative', 'nonfinite', 'duplicate', 'label'):
            with self.subTest(mode=mode):
                rows = fixture(); descriptor = rows[0]['metrics']['external_motifs']
                if mode == 'mass': descriptor['environments']['occupancy'][compare.key(())] = .9
                elif mode == 'negative': descriptor['environments']['occupancy'][compare.key(())] = -.1
                elif mode == 'nonfinite': descriptor['marginal_occupancy'] = [dict(key=(0, 4, 7), fraction=float('inf'))]
                elif mode == 'duplicate': descriptor['marginal_occupancy'] = [dict(key=(0, 4, 7), fraction=.5)]*2
                else: descriptor['environment_values'][compare.key(())] = ((0, 4, 7),)
                with self.assertRaises(ValueError): compare.reduce_native_comparisons(rows)


if __name__ == '__main__': unittest.main()
