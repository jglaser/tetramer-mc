"""Synthetic mass-table controls; no physical sampler or native search."""
import copy
import math
import unittest
import numpy as np

import compare_streaming_vessel_statistics as stage
from partition_vessel_streaming import classify, REGIONAL, BIN_COUNTS, STRATA
from vessel_contact_partition import Mass, PRIMARY, SUPPORTS


def fixture(scale=1., n=65536, outside_native_scale=1.):
    pair = lambda: {k:Mass() for k in ('Qz','Q0')}
    global_names = classify(False,False,False,dict.fromkeys(SUPPORTS,False))[0]
    global_m = {k:pair() for k in global_names}; regional = {k:pair() for k in REGIONAL}
    strata = {f:{k:[pair() for _ in range(size)] for k in REGIONAL} for f,size in BIN_COUNTS.items()}
    for i in range(1200):
        native = i%3 == 0; contact = i%3 != 2
        supports = dict(zip(SUPPORTS,[(i//3)%2 == 0, (i//3)%4 == 0, (i//6)%2 == 0, (i//6)%4 == 0]))
        classes,local = classify(True,contact,native,supports)
        log_weight = math.log(scale*(1+i%3)*(outside_native_scale if native and not supports['current_R4'] else 1.))
        def add(values):
            values['Qz'].add(log_weight); values['Q0'].add(math.log(1+i%3))
        for name,selected in classes.items():
            if selected: add(global_m[name])
        for name,selected in local.items():
            if selected:
                add(regional[name])
                for family,size in BIN_COUNTS.items(): add(strata[family][name][(i//6)%size])
    finish = lambda p: {k:v.result(n) for k,v in p.items()}
    return dict(schema=stage.PARTITION_SCHEMA,complete=True,samples=n,invalid_draws=n-1200,
        estimates={k:finish(p) for k,p in global_m.items()},regional_estimates={k:finish(p) for k,p in regional.items()},
        strata={f:{k:[finish(p) for p in bins] for k,bins in by_class.items()} for f,by_class in strata.items()},
        stratum_definition=copy.deepcopy(STRATA),new_native_classifier_calls=1200,native_unbound_anomalies=0)


def populations(stage_name='standard', scales=(1.,2.,3.,4.), outside_native_scale=1.):
    n = dict(stage.previous.STAGES)[stage_name]; values = []
    for ai,arm in enumerate(stage.previous.ARMS):
        for i,scale in enumerate(scales):
            data = fixture(scale,n,outside_native_scale)
            values.append(dict(id=f'{stage_name}-{arm}-r{i}',stage=stage_name,arm=arm,population=i,
                seed=1000+ai*4+i,samples=n,estimates=data['estimates'],partition_data=data))
    return values


class StatisticsTests(unittest.TestCase):
    def test_complete_partition_and_fixed_stratum_contract(self):
        value = fixture(); stage.validate_partition(value,65536)
        for mutation in ('missing_bin','different_parent','dropped_zero','changed_edge','wrong_calls'):
            bad = copy.deepcopy(value)
            if mutation == 'missing_bin': bad['strata']['orthant'][REGIONAL[0]].pop()
            elif mutation == 'different_parent': bad['regional_estimates'][REGIONAL[0]] = copy.deepcopy(bad['estimates'][REGIONAL[0]])
            elif mutation == 'dropped_zero': bad['strata']['radial'][REGIONAL[0]][0]['Qz']['draws'] = 1200
            elif mutation == 'changed_edge': bad['stratum_definition']['radial_edges'][1] = 2.1
            else: bad['new_native_classifier_calls'] -= 1
            with self.subTest(mutation=mutation), self.assertRaises(ValueError): stage.validate_partition(bad,65536)

    def test_linear_not_logarithmic_SE_comparison_and_no_absolute_overflow(self):
        a = dict(log_Q=.19,population_relative_SE=.06); b = dict(log_Q=0.,population_relative_SE=0.)
        self.assertTrue(stage.linear_agreement(a,b)['passed'])
        self.assertFalse(stage.previous.agreement(a,b)['passed'])
        offset = stage.linear_agreement(dict(a,log_Q=1000.19),dict(b,log_Q=1000.))
        self.assertTrue(offset['passed'])
        self.assertAlmostEqual(offset['scaled_linear_difference'],stage.linear_agreement(a,b)['scaled_linear_difference'])
        self.assertFalse(stage.linear_agreement(dict(log_Q=None),b)['observed'])
        self.assertFalse(stage.linear_agreement(dict(a,log_Q=.3),dict(b,population_relative_SE=1.))['passed'])

    def test_stage_retains_whole_population_uncertainty_and_all_bins(self):
        result = stage.compare(populations(),'standard')
        native = result['regional_estimates']['vessel']['Qz'][REGIONAL[0]]
        means = np.arange(1.,5.)*200/65536
        self.assertAlmostEqual(native['log_Q'],math.log(means.mean()))
        self.assertAlmostEqual(native['population_relative_SE'],means.std(ddof=1)/2/means.mean())
        self.assertEqual(len(result['stratum_comparisons']),sum(BIN_COUNTS.values())*len(REGIONAL))
        self.assertFalse(result['stages_pooled']); self.assertFalse(result['input_authentication_performed'])
        self.assertFalse(result['assembly_stability_established']); self.assertFalse(result['full_vessel_unseen_modes_certified'])
        self.assertAlmostEqual(result['arms']['vessel']['Qz']['contrasts']['native_vs_contact_noentry']['beta_F_native_minus_noentry'],math.log(2))
        self.assertAlmostEqual(result['arms']['vessel']['Qz']['contrasts']['native_vs_contact_noentry']['population_SE'],0.)
        self.assertFalse(result['diagnostics']['declared_observed_diagnostics_passed'])

    def test_material_strata_use_regional_parent_even_with_huge_outside_native_weight(self):
        result = stage.compare(populations(scales=(1.,1.,1.,1.),outside_native_scale=1e6),'standard')
        native_bins = [b for b in result['stratum_comparisons'] if b['region'] == REGIONAL[0] and b['family'] == 'radial']
        self.assertTrue(all(b['material'] for b in native_bins))
        self.assertTrue(all(.3 < b['observed_regional_fractions']['vessel'] < .4 for b in native_bins))
        remainder = next(x for x in result['remainder_diagnostics'] if x['region'] == PRIMARY[0]+':outside_current_R4')
        self.assertGreater(remainder['observed_primary_fractions']['vessel'],.999)
        self.assertFalse(result['assembly_stability_established'])

    def test_mixed_stage_or_repeated_stream_rejected(self):
        ps = populations()
        with self.assertRaisesRegex(ValueError,'Mixed stages'): stage.compare(ps,'large')
        ps[4]['seed'] = ps[0]['seed']
        with self.assertRaisesRegex(ValueError,'independent populations'): stage.compare(ps,'standard')


if __name__ == '__main__': unittest.main()
