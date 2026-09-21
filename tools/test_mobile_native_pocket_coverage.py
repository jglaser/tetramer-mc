"""Coverage follow-up budgets and independent disjoint sums, without simulation."""
import copy
import math
import unittest
import numpy as np

from analyze_mobile_native_pocket import (
    COVERAGE_ALLOCATION, COVERAGE_SEED, coverage_totals, protocol_design,
    summarize, validate_disjoint_strata, validate_jobs)


def constant_population(index, seed, value, draws):
    z = np.full(draws, math.log(value));h = np.zeros(draws)
    return dict(id=f'r{index:02d}', seed=seed, z=z, h=h, pairs=np.column_stack((z, z)))


def strata_fixture():
    names = ['r5repeat', 'shell5to8', 'shell8to12', 'shell12to16', 'shell16to24', 'shell24to32']
    output = []
    for j, name in enumerate(names):
        count, draws = (8, 16) if j == 0 else (4, 8)
        pops = [constant_population(i, 10000*j+i, (j+1)*(i+1), draws) for i in range(count)]
        output.append(summarize(pops, name))
    return output


class CoverageFollowupTests(unittest.TestCase):
    def test_mixed_frozen_budget_and_all_twentyfour_seeds(self):
        protocol = dict(design='coverage', allocation=copy.deepcopy(COVERAGE_ALLOCATION), seed_base=COVERAGE_SEED,
            samples_per_population=None, populations_per_region=None, total_unconditional_draws=262144)
        allocation, seed = protocol_design(protocol)
        offset = 0;all_seeds = []
        for arm, design in allocation.items():
            jobs = [dict(id=f'r{i:02d}', seed=seed+1009*(offset+i), samples=design['samples'])
                    for i in range(design['replicates'])]
            terminal = [dict(job, arm=arm, returncode=0) for job in jobs]
            validate_jobs(jobs, terminal, arm, allocation, seed)
            all_seeds.extend(j['seed'] for j in jobs);offset += len(jobs)
        self.assertEqual(all_seeds, [COVERAGE_SEED+1009*i for i in range(24)])
        self.assertEqual(sum(d['replicates']*d['samples'] for d in allocation.values()), 262144)
        bad = copy.deepcopy(protocol);bad['allocation']['r5repeat']['replicates'] = 4
        with self.assertRaisesRegex(ValueError, 'allocation'):protocol_design(bad)
        bad = copy.deepcopy(protocol);bad['samples_per_population'] = 8192
        with self.assertRaisesRegex(ValueError, 'Mixed-budget'):protocol_design(bad)

    def test_repeat_requires_eight_fullsize_populations(self):
        jobs = [dict(id=f'r{i:02d}', seed=COVERAGE_SEED+1009*i, samples=16384) for i in range(8)]
        terminal = [dict(job, arm='r5repeat', returncode=0) for job in jobs]
        for n in (4, 7):
            with self.assertRaisesRegex(ValueError, 'populations'):
                validate_jobs(jobs[:n], terminal[:n], 'r5repeat', COVERAGE_ALLOCATION, COVERAGE_SEED)
        bad = copy.deepcopy(jobs);bad[7]['samples'] = 8192
        with self.assertRaisesRegex(ValueError, 'sample count'):
            validate_jobs(bad, terminal, 'r5repeat', COVERAGE_ALLOCATION, COVERAGE_SEED)

    def test_unequal_population_counts_add_integrals_not_rows_or_population_means(self):
        strata = strata_fixture();result = coverage_totals(strata)
        total = result['cumulative_r32']
        masses = np.array([math.exp(r['row_uncertainty']['log_Qz']) for r in strata])
        wanted = masses.sum()
        self.assertAlmostEqual(math.exp(total['row_uncertainty']['log_Qz']), wanted)
        self.assertEqual(total['population_counts_by_region']['r5repeat'], 8)
        self.assertEqual(total['population_counts_by_region']['shell5to8'], 4)
        self.assertEqual(total['attempted_draws_by_region']['r5repeat'], 128)
        self.assertEqual(total['attempted_draws_by_region']['shell5to8'], 32)
        fractions = masses/wanted
        expected_variance = sum(f*f*r['population_uncertainty']['Qz_relative_SE']**2 for r, f in zip(strata, fractions))
        self.assertAlmostEqual(total['population_uncertainty']['Qz_relative_SE']**2, expected_variance)
        self.assertAlmostEqual(sum(v['row_uncertainty']['Qz']['observed_fraction']
            for v in result['stratum_fractions_of_r32'].values()), 1.)
        outside = result['outside_r8_fraction_of_r32']['row_uncertainty']['Qz']['observed_fraction']
        self.assertAlmostEqual(outside, masses[2:].sum()/wanted)
        self.assertIn('8 population means', strata[0]['uncertainty_scope'])

    def test_initial_core_cannot_be_added_or_substituted_and_duplicate_seeds_fail(self):
        strata = strata_fixture()
        bad = copy.deepcopy(strata);bad[0]['name'] = 'initial-r5'
        with self.assertRaisesRegex(ValueError, 'new core'):coverage_totals(bad)
        with self.assertRaisesRegex(ValueError, 'new core'):coverage_totals([strata[0]]+strata)
        bad = copy.deepcopy(strata);bad[1]['populations'][0]['seed'] = bad[0]['populations'][0]['seed']
        with self.assertRaisesRegex(ValueError, 'independent'):coverage_totals(bad)

    def test_shell_boundary_gap_overlap_or_different_chart_is_rejected(self):
        base = dict(gaussian_chart={'covariance':'fixed'}, fixed_neighbor={'pose':'A'}, physical_fixed_neighbors=['A','B'],
            capture_center=[0,0,0], capture_radius=170., shape_sha256='shape', activity=.035, depletant_radius=1.5,
            physical_metric={'q':'original'}, minimum_original_q=1., minimum_original_q_inclusive=False)
        ends = [0.,5.,8.,12.,16.,24.,32.]
        regions = [dict(base, minimum_mahalanobis_radius=a, mahalanobis_radius=b) for a,b in zip(ends,ends[1:])]
        self.assertEqual(validate_disjoint_strata(regions), 32.)
        for lower in (7., 9.):
            bad = copy.deepcopy(regions);bad[2]['minimum_mahalanobis_radius'] = lower
            with self.assertRaisesRegex(ValueError, 'overlap or leave a gap'):validate_disjoint_strata(bad)
        bad = copy.deepcopy(regions);bad[-1]['gaussian_chart'] = {'covariance':'changed'}
        with self.assertRaisesRegex(ValueError, 'different physical'):validate_disjoint_strata(bad)


if __name__ == '__main__':unittest.main()
