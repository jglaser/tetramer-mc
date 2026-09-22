"""Nonphysical independent-population estimator and identity controls."""
import copy
import math
import unittest
from compare_r4_smc_importance import (CLASSES, compare, compare_mass, identity, importance_rows, smc_rows)
from analyze_r4_smc_control import population_statistics, STRATA
from compare_r4_smc_importance import STRATA_DEFINITION, validate_strata


def fixture():
    cfg=dict(shape='/shape',fixed_poses=[{'id':0},{'id':1}],capture_center=[0,0,0],capture_radius=170.,
        reservoir_density=.035,depletant_radius=1.5,metadata={'metric':'fixed'},translation_steps=[.2,2.],rotation_steps_deg=[1.5,15.])
    target=dict(shape_sha256='shape',region_sha256='region',measure='d3t in Angstrom cubed times normalized proper SO(3) Haar',
        fixed_poses=cfg['fixed_poses'],capture_center=cfg['capture_center'],capture_radius=170.,activity=.035,depletant_radius=1.5)
    protocol=dict(physical_target=target,proposal={'reference_region_sha256':'old'},
        allocation=dict(independent_populations=4,unconditional_initialization_draws_each=1000,particles_each=100),
        jobs=[dict(id=f'r{i}',seed=100+i) for i in range(4)])
    counts=dict(zip(CLASSES,[100,50,20,30,40,10]));pops=[];estimators={c:dict(populations=[]) for c in CLASSES}
    records=[dict(id=f'r{i}',seed=200+i,samples=2000) for i in range(4)]
    for i,z in enumerate((8.,10.,12.,14.)):
        masses={c:math.log(z*counts[c]/100) for c in CLASSES}
        hard={c:math.log(2*counts[c]/100) for c in CLASSES}
        terminal=dict(particles=100,counts=counts.copy(),log_masses=masses,
            log_strata={family:{c:[masses[c]]+[None]*(size-1) for c in CLASSES} for family,size in STRATA.items()})
        pops.append(dict(id=f'r{i}',seed=100+i,initial_draws=1000,zero_estimate=False,terminal=terminal,
            initial_physical_hard_log_masses=hard,initial_bridge_log_masses={c:0. for c in CLASSES}))
        for c in CLASSES:estimators[c]['populations'].append(dict(id=f'r{i}',seed=200+i,draws=2000,log_Qz=masses[c],log_Q0=hard[c]))
    smc=dict(schema='smc-r4-control-analysis-v1',complete=True,physical_config=cfg,protocol_snapshot=protocol,
        native_definition_sha256='native',populations=pops)
    arm=dict(populations=records,allocation=dict(samples=2000),estimates=estimators,
             strata={family:{c:[estimators[c]]+[dict(populations=[dict(v,log_Qz=None,log_Q0=None) for v in estimators[c]['populations']]) for _ in range(size-1)] for c in CLASSES} for family,size in STRATA.items()})
    imp=dict(schema='contact-confirmation-comparison-v1',complete=True,region_sha256='region',shape_sha256='shape',
        native_definition={'definition_sha256':'native'},reference_region_sha256='old',arms={'bank':arm},strata_definition=copy.deepcopy(STRATA_DEFINITION))
    return smc,imp,cfg


class CrossMethodTests(unittest.TestCase):
    def test_matching_linear_masses_and_covariance(self):
        smc,imp,cfg=fixture()
        result=compare(smc,imp,cfg)
        self.assertTrue(result['arms']['bank']['primary_Qz_mass_agreement'])
        self.assertTrue(result['arms']['bank']['direct_free_energy_contrast']['passed'])
        self.assertEqual(result['assembly_conclusion'],'unresolved')
        self.assertAlmostEqual(result['smc_population_statistics']['Qz']['estimates']['total']['log_Q'],math.log(11.))
        self.assertAlmostEqual(result['smc_population_statistics']['Q0']['estimates']['total']['log_Q'],math.log(2.))

    def test_arithmetic_not_mean_log_or_endpoint_fraction(self):
        smc,_,_=fixture();s=population_statistics(smc_rows(smc,'Qz'))
        self.assertAlmostEqual(s['estimates']['registered_native_entry']['log_Q'],math.log(5.5))
        self.assertNotAlmostEqual(s['estimates']['registered_native_entry']['log_Q'],math.log(.5))
        smc['populations'][0]['terminal']['log_masses']['registered_native_entry']=math.log(.5)
        with self.assertRaisesRegex(ValueError,'normalizer times indicator'):smc_rows(smc,'Qz')

    def test_initial_bridge_is_not_physical_hard_mass(self):
        smc,_,_=fixture()
        smc['populations'][0]['initial_bridge_log_masses']['total']=999.
        self.assertAlmostEqual(smc_rows(smc,'Q0')[0]['total'],math.log(2.))
        with self.assertRaisesRegex(ValueError,'No initial hard-only stratum'):smc_rows(smc,'Q0','radial',0)

    def test_zero_population_retained_unobserved_region_not_bounded(self):
        smc,_,_=fixture();p=smc['populations'][0];p['zero_estimate']=True;p['terminal']['particles']=0
        p['terminal']['counts']={c:0 for c in CLASSES};p['terminal']['log_masses']={c:None for c in CLASSES}
        rows=smc_rows(smc,'Qz');self.assertEqual(len(rows),4)
        s=population_statistics(rows)
        self.assertAlmostEqual(s['estimates']['total']['log_Q'],math.log(9.))
        self.assertEqual(s['estimates']['total']['nonzero_populations'],3)
        zero=population_statistics([{c:None for c in CLASSES}]*4)
        self.assertFalse(compare_mass(zero['estimates']['total'],s['estimates']['total'])['passed'])

    def test_frozen_domain_and_classifier_and_seed_checks(self):
        smc,imp,cfg=fixture()
        for field in ('region_sha256','shape_sha256','reference_region_sha256'):
            bad=copy.deepcopy(imp);bad[field]='changed'
            with self.assertRaises(ValueError):identity(smc,bad,cfg)
        bad=copy.deepcopy(imp);bad['native_definition']['definition_sha256']='changed'
        with self.assertRaises(ValueError):identity(smc,bad,cfg)
        bad=copy.deepcopy(imp);bad['arms']['bank']['populations'][0]['seed']=100
        with self.assertRaisesRegex(ValueError,'share population seeds'):identity(smc,bad,cfg)
        bad=copy.deepcopy(cfg);bad['capture_radius']=169.
        with self.assertRaises(ValueError):identity(smc,imp,bad)

    def test_missing_population_and_valid_only_denominator_rejected(self):
        smc,imp,_=fixture();arm=imp['arms']['bank']
        arm['estimates']['total']['populations'][0]['draws']=50
        with self.assertRaisesRegex(ValueError,'unconditional'):importance_rows(arm,'Qz')
        smc['populations'].pop()
        with self.assertRaisesRegex(ValueError,'Dropped independent SMC'):smc_rows(smc,'Qz')


    def test_stratum_definition_lengths_and_conservation(self):
        smc,imp,_=fixture();validate_strata(smc,imp)
        bad=copy.deepcopy(imp);bad['strata_definition']['radial_edges'][1]=1.9
        with self.assertRaisesRegex(ValueError,'stratum geometry'):validate_strata(smc,bad)
        bad=copy.deepcopy(smc);bad['populations'][0]['terminal']['log_strata']['orthant']['total'].pop()
        with self.assertRaisesRegex(ValueError,'array length'):validate_strata(bad,imp)
        bad=copy.deepcopy(smc);bad['populations'][0]['terminal']['log_strata']['angular']['total'][0]+=.1
        with self.assertRaisesRegex(ValueError,'do not sum'):validate_strata(bad,imp)

    def test_two_agreement_thresholds_and_direct_contrast(self):
        base=dict(log_Q=5.,population_relative_SE=.01)
        self.assertFalse(compare_mass(base,dict(base,log_Q=5.1))['passed'])
        self.assertFalse(compare_mass(dict(base,population_relative_SE=2.),dict(base,log_Q=5.3,population_relative_SE=2.))['passed'])
        self.assertTrue(compare_mass(base,dict(base))['passed'])
        smc,imp,cfg=fixture()
        for c in CLASSES:
            shift=.15 if c in ('registered_native_entry','old_R5_intersection_native','remaining_R4_native') else -.15 if c=='contact_no_native_entry' else 0.
            for p in imp['arms']['bank']['estimates'][c]['populations']:p['log_Qz']+=shift
        es=imp['arms']['bank']['estimates']
        for i,pop in enumerate(es['total']['populations']):
            pop['log_Qz']=math.log(sum(math.exp(es[c]['populations'][i]['log_Qz']) for c in ('registered_native_entry','contact_no_native_entry','unbound_no_native_entry')))
        r=compare(smc,imp,cfg)
        self.assertFalse(r['arms']['bank']['direct_free_energy_contrast']['passed'])


if __name__=='__main__':unittest.main()
