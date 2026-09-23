"""Nonphysical fixed-allocation and three-way independent-evidence checks."""
import copy
from pathlib import Path
import unittest
from prepare_full_vessel_comparison import jobs_for,diagnostic_state,evidence_gate,TOTAL,STAGES

PRIMARY=('total','registered_native_entry','old_R5_intersection_native','remaining_R4_native')
CHECKS=('main_regional_quality','main_paired_free_energy_precision','every_control_regional_agreement',
    'every_control_direct_free_energy_agreement','significant_original_strata_agreement','classifier_contact_consistency','native_partition_sum')


def reports():
    c=dict(schema='contact-confirmation-comparison-v1',complete=True,arms={'bank':{},'wide':{}},
        convergence=dict(confirmation_passed=True,passed=False,full_wall_coverage_established=False,checks={k:True for k in CHECKS}))
    item=dict(log_mass_difference_SMC_minus_importance=.1,combined_population_SE=.1)
    arm=dict(masses={kind:{p:copy.deepcopy(item) for p in PRIMARY} for kind in ('Qz','Q0')},
        strata={'radial':[{p:dict(item,decision_relevant=True) for p in PRIMARY}],
                'orthant':[{p:dict(unresolved='not observed',decision_relevant=False) for p in PRIMARY}]})
    s=dict(schema='r4-smc-importance-comparison-v1',complete=True,primary_regions=list(PRIMARY),arms={k:copy.deepcopy(arm) for k in c['arms']})
    return c,[copy.deepcopy(s),copy.deepcopy(s)]


class PreparationTests(unittest.TestCase):
    def test_exact_two_fresh_stages_no_overlap_or_added_baseline_guide(self):
        jobs=jobs_for(Path('/tmp/inert-vessel'))
        self.assertEqual(len(jobs),16);self.assertEqual(sum(j['samples'] for j in jobs),TOTAL)
        self.assertEqual(len({j['seed'] for j in jobs}),16);self.assertEqual(len({j['directory'] for j in jobs}),16)
        for stage,n in STAGES:
            for arm in ('vessel','half_mixture'):
                selected=[j for j in jobs if j['stage']==stage and j['arm']==arm]
                self.assertEqual([j['samples'] for j in selected],[n]*4)
                self.assertTrue(all(('--latent-guide' in j['command'])==(arm=='half_mixture') for j in selected))
                self.assertTrue(all('--proposal-anchor-index' not in j['command'] for j in selected))
        for a,b in zip(jobs[::2],jobs[1::2]):
            self.assertEqual(a['samples'],b['samples']);self.assertNotEqual(a['seed'],b['seed'])
        self.assertEqual(jobs,jobs_for(Path('/tmp/inert-vessel')))

    def test_three_way_decision_and_missing_mass(self):
        for d,se,expected in ((.1,.1,'corroborated'),(.3,.2,'unresolved'),(.1,.001,'unresolved'),(.5,.1,'material_contradiction')):
            self.assertEqual(diagnostic_state(dict(log_mass_difference_SMC_minus_importance=d,combined_population_SE=se)),expected)
        self.assertEqual(diagnostic_state(dict(unresolved='no hits')),'unresolved')
        with self.assertRaises(ValueError):diagnostic_state(dict(passed=True))

    def test_regional_gate_does_not_require_downstream_coverage_or_equal_occupancy(self):
        c,s=reports();result=evidence_gate(c,s)
        self.assertTrue(result['regional_checks_passed']);self.assertEqual(result['independent_status'],'corroborated')
        # No native/noentry contrast, noentry hits or descendant ESS are needed.
        s[0]['arms']['bank']['masses']['Qz']['total']={'unresolved':'imprecise total'}
        self.assertEqual(evidence_gate(c,s)['independent_status'],'unresolved')

    def test_failed_regional_checks_and_material_primary_conflict_stop(self):
        c,s=reports();c['convergence']['checks'][CHECKS[0]]=False
        with self.assertRaisesRegex(ValueError,'Regional convergence'):evidence_gate(c,s)
        c,s=reports();s[1]['arms']['wide']['masses']['Qz']['remaining_R4_native']=dict(log_mass_difference_SMC_minus_importance=.7,combined_population_SE=.1)
        with self.assertRaisesRegex(ValueError,'Material matching'):evidence_gate(c,s)

    def test_significant_stratum_conflict_stops_but_unobserved_bins_do_not(self):
        c,s=reports();s[1]['arms']['bank']['strata']['radial'][0]['total'].update(log_mass_difference_SMC_minus_importance=1.,combined_population_SE=.1)
        with self.assertRaisesRegex(ValueError,'Material matching'):evidence_gate(c,s)
        s[1]['arms']['bank']['strata']['radial'][0]['total']=dict(unresolved='unobserved',decision_relevant=True)
        self.assertEqual(evidence_gate(c,s)['independent_status'],'unresolved')

    def test_missing_control_arm_and_contract_rejected(self):
        c,s=reports()
        with self.assertRaises(ValueError):evidence_gate(c,s[:1])
        del s[1]['arms']['wide']
        with self.assertRaises(ValueError):evidence_gate(c,s)
        c,s=reports();c['convergence']['checks'].pop(CHECKS[0])
        with self.assertRaises(ValueError):evidence_gate(c,s)


if __name__=='__main__':unittest.main()
