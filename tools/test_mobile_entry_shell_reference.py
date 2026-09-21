"""Unconditional importance weights, target partition, and descriptive controls."""
import copy
import math
import unittest
import numpy as np
from scipy.special import logsumexp
from analyze_mobile_entry_shell_reference import (
    PRIMARY, SAMPLES, SEEDS, read_weights, validate_allocation, partition_masks,
    summarize_regions, branch_contributions, compare_estimates)


def row(index=0, branch='entry-shell', **changes):
    result = dict(draw=index, pose=dict(position=[0., 0., 0.], orientation=[1., 0., 0., 0.]),
        hard_valid=True, capture_valid=True, region_valid=True, shell_valid=True,
        q=1., latent_radius=3., log_physical_jacobian=1., log_proposal_density=-1.,
        proposal_branch=branch, proposal_component=0 if branch == 'entry-shell' else None,
        log_hard_weight=2., clouds=[dict(log_weight=math.log(2)), dict(log_weight=math.log(4))],
        log_importance_weight=2.+math.log(3))
    result.update(changes)
    return result


def invalid(index=1, **changes):
    return row(index, hard_valid=False, log_hard_weight=None, log_importance_weight=None, clouds=[], **changes)


def masked_populations():
    # Equal full-q contributions from two branch labels; zeros and all classes.
    rows = [row(0), row(1, branch='uniform-shell'),
            invalid(2, shell_valid=False, latent_radius=5.), row(3)]
    arrays = read_weights(rows, 4, dict(mahalanobis_radius=4.))
    masks = partition_masks([True, True, False, True], [True, True, False, False],
                            [True, False, False, False], [1, 0, 0, 0], [False]*4, [1.]*4, [5.]*4)
    for branch in ('entry-shell', 'uniform-shell'):
        selected = np.asarray([r['proposal_branch'] == branch for r in rows])
        for name in ('total',)+PRIMARY:
            masks['proposal_branch:'+branch+':'+name] = selected & masks[name]
    return [dict(id=f'r{i:02d}', seed=SEEDS[i], masks=masks, **arrays) for i in range(4)]


class EntryShellReferenceTests(unittest.TestCase):
    def test_exterior_and_core_zeros_keep_original_N(self):
        exterior = row(1, shell_valid=False, latent_radius=5., log_hard_weight=None,
                       log_importance_weight=None, clouds=[])
        arrays = read_weights([row(), exterior, invalid(2)], 3, dict(mahalanobis_radius=4.))
        self.assertEqual(len(arrays['z']), 3)
        np.testing.assert_array_equal(np.isfinite(arrays['z']), [True, False, False])
        self.assertAlmostEqual(logsumexp(arrays['z'])-math.log(3), 2.)
        self.assertAlmostEqual(arrays['h'][0], 2.)
        np.testing.assert_allclose(arrays['pairs'][0], [2.+math.log(2), 2.+math.log(4)])
        # A finite shell radius outside the R4 is valid proposal support, not an error.
        self.assertTrue(exterior['hard_valid'])

    def test_exterior_assigned_weight_or_wrong_shell_flag_rejected(self):
        exterior = row(shell_valid=False, latent_radius=5.)
        with self.assertRaisesRegex(ValueError, 'retain zero'):
            read_weights([exterior], 1, dict(mahalanobis_radius=4.))
        with self.assertRaisesRegex(ValueError, 'R4 support'):
            read_weights([row(latent_radius=5.)], 1, dict(mahalanobis_radius=4.))

    def test_full_J_over_q_and_paired_cloud_mean_required(self):
        with self.assertRaisesRegex(ValueError, 'J/q'):
            read_weights([row(log_hard_weight=1.)], 1, dict(mahalanobis_radius=4.))
        with self.assertRaisesRegex(ValueError, 'cloud mean'):
            read_weights([row(log_importance_weight=2.)], 1, dict(mahalanobis_radius=4.))
        changed = invalid(); changed['log_hard_weight'] = 0.
        with self.assertRaisesRegex(ValueError, 'retain zero'):
            read_weights([row(), changed], 2, dict(mahalanobis_radius=4.))

    def test_branch_and_capture_constraints_do_not_condition_target(self):
        with self.assertRaisesRegex(ValueError, 'uniform branch'):
            read_weights([invalid(0, branch='uniform-shell', latent_radius=5., shell_valid=False)], 1, dict(mahalanobis_radius=4.))
        with self.assertRaisesRegex(ValueError, 'component'):
            read_weights([row(proposal_component=24)], 1, dict(mahalanobis_radius=4.))
        with self.assertRaisesRegex(ValueError, 'q filter'):
            read_weights([row(region_valid=False)], 1, dict(mahalanobis_radius=4.))
        with self.assertRaisesRegex(ValueError, 'capture enclosure'):
            read_weights([invalid(0, capture_valid=False)], 1, dict(mahalanobis_radius=4.))
        outside = invalid(0, capture_valid=False, shell_valid=False, latent_radius=500.)
        self.assertTrue(np.isneginf(read_weights([outside], 1, dict(mahalanobis_radius=4.))['z'][0]))

    def test_missing_duplicate_and_failed_populations_rejected(self):
        jobs = [dict(id=f'r{i:02d}', samples=SAMPLES, seed=SEEDS[i]) for i in range(4)]
        terminal = [dict(j, status='complete', returncode=0) for j in jobs]
        validate_allocation(jobs, terminal)
        for mutation in ('duplicate', 'old-seed', 'failed'):
            a, b = copy.deepcopy(jobs), copy.deepcopy(terminal)
            if mutation == 'duplicate': b[1] = b[0]
            if mutation == 'old-seed': a[0]['seed'] -= 2000000
            if mutation == 'failed': b[0]['returncode'] = 1
            with self.assertRaises(ValueError): validate_allocation(a, b)
        with self.assertRaisesRegex(ValueError, 'missing/repeated'):
            read_weights([row(), row()], 2, dict(mahalanobis_radius=4.))

    def test_disjoint_class_and_branch_sums_keep_full_denominator(self):
        estimates, _, _ = summarize_regions(masked_populations())
        parts = branch_contributions(estimates)
        for kind in ('Q0', 'Qz'):
            total = estimates['total']['row_uncertainty']['log_'+kind]
            self.assertAlmostEqual(logsumexp([estimates[n]['row_uncertainty']['log_'+kind] for n in PRIMARY]), total)
            fractions = [parts['total'][b][kind]['observed_weight_fraction'] for b in ('entry-shell', 'uniform-shell')]
            np.testing.assert_allclose(fractions, [2/3, 1/3])
            self.assertAlmostEqual(sum(fractions), 1.)
        for name in ('total',)+PRIMARY:
            for branch in ('entry-shell', 'uniform-shell'):
                self.assertEqual(parts[name][branch]['draws'], 16)
        self.assertAlmostEqual(estimates['total']['row_uncertainty']['log_enhancement_SE'], 0., places=14)

    def test_cost_comparison_keeps_estimates_separate(self):
        estimates, _, _ = summarize_regions(masked_populations())
        fresh = dict(estimates=estimates, sampler_cpu_seconds=10.)
        uniform = dict(estimates=copy.deepcopy(estimates), sampler_cpu_seconds=30.)
        before = copy.deepcopy(uniform)
        result = compare_estimates(fresh, uniform)
        self.assertEqual(uniform, before)
        for name in ('total',)+PRIMARY:
            for kind in ('Q0', 'Qz'):
                item = result[name][kind]['row_uncertainty']
                self.assertAlmostEqual(item['log_fresh_minus_uniform'], 0.)
                self.assertAlmostEqual(item['uniform_over_fresh_cost_variance'], 3.)

    def test_unobserved_class_has_no_ratio_or_upper_bound(self):
        populations = masked_populations()
        for pop in populations:
            pop['masks'][PRIMARY[2]] = np.zeros(4, bool)
            pop['masks']['total'][-1] = False
            for branch in ('entry-shell', 'uniform-shell'):
                pop['masks']['proposal_branch:'+branch+':'+PRIMARY[2]] = np.zeros(4, bool)
                pop['masks']['proposal_branch:'+branch+':total'][-1] = False
        estimates, _, _ = summarize_regions(populations)
        source = dict(estimates=estimates, sampler_cpu_seconds=10.)
        result = compare_estimates(source, source)
        self.assertFalse(result[PRIMARY[2]]['Qz']['row_uncertainty']['observed'])
        self.assertIsNone(branch_contributions(estimates)[PRIMARY[2]]['entry-shell']['Qz']['observed_weight_fraction'])


if __name__ == '__main__': unittest.main()
