#!/usr/bin/env python3
"""Fixed-allocation and command checks; no binary or proposal draws required."""
import copy
from pathlib import Path
import unittest

from prepare_contact_line_passive import SEEDS,COVERAGE_SEED,WIDTHS
from run_contact_line_passive import validate_design,command


def protocol():
    jobs=[dict(arm='baseline92' if i<4 else 'line92',id=f'r{i%4:02}',seed=s,
               fresh_proposal_draws=64,archived_probe_queries=78 if i%4==0 else 0,
               directory=f'/tmp/passive/{i}') for i,s in enumerate(SEEDS)]
    return dict(fresh_draws=512,archived_probe_queries=156,maximum_CPU_workers=1,new_Poisson_clouds=0,
        axes=[0],widths_A=WIDTHS,conditional_probability=.5,minimum_conditional_mass=1e-12,jobs=jobs,
        additional_coverage_queries=128,coverage_job=dict(arm='line92',id='coverage',seed=COVERAGE_SEED,
            fresh_proposal_draws=0,archived_probe_queries=128,probe_file='coverage-probes.jsonl',directory='/tmp/passive/coverage'))


class FixedPassiveTests(unittest.TestCase):
    def test_exact_design(self):
        validate_design(protocol())

    def test_modified_budget_and_physics_rejected(self):
        for key,value in [('fresh_draws',513),('maximum_CPU_workers',2),('new_Poisson_clouds',1),
                          ('axes',[0,1]),('additional_coverage_queries',129)]:
            p=protocol();p[key]=value
            with self.assertRaises((AssertionError,ValueError)):validate_design(p)

    def test_repeated_seed_and_unplanned_probes_rejected(self):
        p=protocol();p['jobs'][1]['seed']=p['jobs'][0]['seed']
        with self.assertRaises((AssertionError,ValueError)):validate_design(p)
        p=protocol();p['jobs'][1]['archived_probe_queries']=78
        with self.assertRaises((AssertionError,ValueError)):validate_design(p)

    def test_coverage_is_zero_draws_and_has_distinct_probe_file(self):
        c=command(Path('/tmp/frozen'),protocol()['coverage_job'])
        self.assertEqual(c[c.index('--samples')+1],'0')
        self.assertEqual(c[c.index('--probes')+1],'/tmp/frozen/common/coverage-probes.jsonl')

    def test_only_first_fresh_population_scores_critical_probes(self):
        p=protocol()
        self.assertIn('--probes',command(Path('/tmp/frozen'),p['jobs'][0]))
        self.assertNotIn('--probes',command(Path('/tmp/frozen'),p['jobs'][1]))


if __name__=='__main__':unittest.main()
