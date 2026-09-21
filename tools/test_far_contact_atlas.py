"""Fixed-prefix bookkeeping, covariance and independent width errors."""
import math
import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np

from analyze_far_contact_atlas import (PrefixMoments, compare_widths, finish, wait_for_auditors,
    command_options, compare_baseline_repeat)
from analyze_far_peak_reference import WINDOW
from compare_far_local_history import MASKS
from analyze_expanded_contact_atlas import nested_prefix_comparison
from compare_far_local_history import HistoryMoments


def add_value(moment, value, radius=.2):
    if value:
        logw = math.log(value); moment.add(radius, logw, 0., [logw, logw])
    else: moment.add()


class FarContactAtlasTests(unittest.TestCase):
    def test_frozen_command_vector_preserves_flags_and_rejects_duplicates(self):
        command = ['python', '/frozen/runner.py', '--samples', '65536', '--q-upper-open', '--model', '/original/model.json']
        self.assertEqual(command_options(command), {'--samples': '65536', '--q-upper-open': True, '--model': '/original/model.json'})
        with self.assertRaises(ValueError): command_options(command+['--samples', '32768'])
        with self.assertRaises(ValueError): command_options(['python', 'runner.py', 'unparsed'])

    def test_baseline_repeat_comparison_keeps_means_and_denominators_separate(self):
        def campaign(name, q, n, seed):
            estimate = dict(logQ=math.log(q), row_RSE=.1, independent_population_RSE=.2, draws=n)
            return dict(arm='narrow', root=name, populations=[dict(seed=seed)],
                physical={k: estimate.copy() for k in MASKS}, hard={k: estimate.copy() for k in MASKS})
        baseline = dict(complete=True, original_q_window=WINDOW, campaigns=[campaign('old', 2., 100, 1)])
        repeated = [campaign('new', 3., 400, 2)]; before = copy.deepcopy([baseline, repeated])
        result = compare_baseline_repeat(baseline, repeated, dict(q_window=WINDOW))
        value = result['arms'][0]['comparisons']['physical']['full']
        self.assertFalse(result['pooled'])
        self.assertEqual(result['arms'][0]['baseline_unconditional_draws'], 100)
        self.assertEqual(result['arms'][0]['repeat_unconditional_draws'], 400)
        self.assertAlmostEqual(math.exp(value['baseline']['logQ']), 2.)
        self.assertAlmostEqual(math.exp(value['repeat']['logQ']), 3.)
        self.assertAlmostEqual(value['linear_difference_in_combined_row_SE'], -1/math.hypot(.2, .3))
        self.assertEqual([baseline, repeated], before)
        repeated[0]['populations'][0]['seed'] = 1
        with self.assertRaises(ValueError): compare_baseline_repeat(baseline, repeated, dict(q_window=WINDOW))
        with self.assertRaises(ValueError): compare_baseline_repeat(dict(baseline, original_q_window=dict(WINDOW, maximum=5.)), [], dict(q_window=WINDOW))

    def test_failed_auditor_does_not_skip_other_live_child(self):
        calls = []
        class Child:
            def __init__(self, name, code): self.name, self.code = name, code
            def wait(self): calls.append(self.name); return self.code
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary); state = dict(phase='original_density_audits', processes=[]); live = []
            for name, code in [('first', 1), ('second', 0)]:
                path = directory/f'{name}.log'; log = path.open('w'); record = dict(arm=name, terminal=False, log=str(path))
                state['processes'].append(record); live.append((Child(name, code), log, record))
            with self.assertRaisesRegex(ValueError, 'first exit1'): wait_for_auditors(live, state, directory/'state.json')
            self.assertEqual(calls, ['first', 'second'])
            self.assertEqual([r['exit_code'] for r in state['processes']], [1, 0])
            self.assertTrue(all(r['terminal'] for r in state['processes']))
            self.assertTrue(all(log.closed for _, log, _ in live))
            self.assertEqual(state['phase'], 'density_audits_failed')

    def test_prefixes_keep_all_original_zeros_and_outside_mass(self):
        moments = PrefixMoments(8, [2, 4, 8])
        weights = [2., 0., 3., 0., 0., 7., 0., 4.]
        for draw, value in enumerate(weights):
            if value:
                logw = math.log(value); moments.add(draw, 3. if draw == 5 else .2, logw, 0., [logw, logw])
            else: moments.add(draw)
        moments.complete()
        for count, item in moments.prefixes.items():
            result = item.report()['physical']
            self.assertEqual(result['full']['draws'], count)
            self.assertAlmostEqual(math.exp(result['full']['logQ']), sum(weights[:count])/count)
        self.assertIsNone(moments.prefixes[4].report()['physical']['outside2']['logQ'])
        self.assertAlmostEqual(math.exp(moments.prefixes[8].report()['physical']['outside2']['logQ']), 7/8)

    def test_shared_prefix_variance_matches_linear_coefficients(self):
        values = np.array([2., 0., 3., 0., 0., 7., 0., 4.]); m, n = 4, 8
        prefix, full = HistoryMoments(), HistoryMoments()
        for i, value in enumerate(values):
            add_value(full, value)
            if i < m: add_value(prefix, value)
        small, large = prefix.report()['physical']['full'], full.report()['physical']['full']
        result = nested_prefix_comparison(small, large)
        a = np.r_[np.full(m, 1/m), np.zeros(n-m)]; b = np.full(n, 1/n)
        sigma2 = values.var(ddof=1)
        self.assertAlmostEqual(math.exp(result['estimated_log_covariance_of_means']), sigma2*np.dot(a, b))
        self.assertAlmostEqual(math.exp(result['estimated_log_variance_of_difference']), sigma2*np.dot(a-b, a-b))
        self.assertAlmostEqual(result['relative_prefix_minus_full'], values[:m].mean()/values.mean()-1)
        self.assertLess(sigma2*np.dot(a-b, a-b), sigma2*(np.dot(a, a)+np.dot(b, b)))

    def test_no_observations_are_unresolved_in_prefixes(self):
        moments = PrefixMoments(4, [2, 4])
        for i in range(4): moments.add(i)
        moments.complete(); small = moments.prefixes[2].report()['physical']['outside2']; large = moments.prefixes[4].report()['physical']['outside2']
        result = nested_prefix_comparison(small, large)
        self.assertIsNone(result['estimated_log_variance_of_difference'])
        self.assertIsNone(result['relative_prefix_minus_full'])
        self.assertIn('unseen tails', result['qualification'])

    def test_prefix_schedule_and_missing_rows_rejected(self):
        for counts in ([4, 2, 8], [2, 2, 8], [2, 4], [2, 8.]):
            with self.assertRaises(ValueError): PrefixMoments(8, counts)
        moments = PrefixMoments(4, [2, 4])
        with self.assertRaises(ValueError): moments.add(1)
        moments.add(0)
        with self.assertRaises(ValueError): moments.add(0)
        with self.assertRaises(ValueError): moments.complete()

    def test_population_concentration_includes_zero_population(self):
        aggregate = HistoryMoments(); populations = []
        for index, values in enumerate(([2., 0.], [6., 0.], [0., 0.])):
            local = HistoryMoments()
            for value in values: add_value(local, value)
            aggregate.merge(local); populations.append(dict(id=index, samples=2, **local.report()))
        result = finish(aggregate, populations)['physical']['full']
        np.testing.assert_allclose(result['population_mass_fractions'], [.25, .75, 0.])
        self.assertAlmostEqual(result['maximum_population_fraction'], .75)
        self.assertAlmostEqual(result['population_weight_ESS'], 1/(.25**2+.75**2))
        self.assertEqual(result['draws'], 6)

    def test_width_difference_is_linear_and_independent(self):
        narrow = dict(logQ=math.log(4.), row_RSE=.25, independent_population_RSE=.5)
        broad = dict(logQ=math.log(2.), row_RSE=.5, independent_population_RSE=.2)
        result = compare_widths(narrow, broad)
        self.assertAlmostEqual(result['linear_difference_in_combined_row_SE'], 2/math.sqrt(2))
        self.assertAlmostEqual(result['linear_difference_in_combined_population_SE'], 2/math.hypot(2., .4))
        self.assertAlmostEqual(result['narrow_to_broad_ratio'], 2.)
        self.assertIn('separately retained proposal denominators', result['scope'])


if __name__ == '__main__': unittest.main()
