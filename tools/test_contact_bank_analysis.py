"""Guard unconditional Gaussian tails, full mixture weights and screening scope."""
import copy
import math
from pathlib import Path
import tempfile
import unittest
import numpy as np
from scipy.special import logsumexp
from analyze_contact_bank_pilot import (read_weights, stratify_with_exterior, summarize_arm,
    evaluate, plot, PRIMARY, CLASSES, sha, compare_free_energy_intervals)
from analyze_mobile_competing_reference import paired_moments
from test_analyze_conditional_ray_campaign import region, row as ray_row, arm as fixture_arm, GATES, STRATA, estimate


def row(index=0, **changes):
    item = ray_row(index, proposal_branch='gaussian', proposal_component=0, selected_ray_fallback=None)
    item.update(changes)
    return item


class ContactBankAnalysisTests(unittest.TestCase):
    def test_outside_gaussian_kept_as_unconditional_zero(self):
        rows = [row(), row(1, latent=[5.,0.,0.,0.,0.,0.], latent_radius=5., shell_valid=False,
            log_importance_weight=None, log_hard_weight=None, clouds=[]),
            row(2, hard_valid=False, log_importance_weight=None, log_hard_weight=None, clouds=[])]
        result = read_weights(rows, 3, region(), dict(alpha=.5, component_count=2))
        np.testing.assert_array_equal(result['support'], [True,False,True])
        self.assertTrue(np.isneginf(result['z'][1:]).all())
        self.assertAlmostEqual(logsumexp(result['z'])-math.log(3), 2.)
        rows[1]['log_hard_weight']=0.
        with self.assertRaisesRegex(ValueError, 'zero weight'): read_weights(rows,3,region(),dict(alpha=.5,component_count=2))

    def test_mixture_weight_and_saved_two_cloud_mean(self):
        with self.assertRaisesRegex(ValueError,'J/q'):
            read_weights([row(log_hard_weight=1.)],1,region(),dict(alpha=.5,component_count=2))
        with self.assertRaisesRegex(ValueError,'Two-cloud'):
            read_weights([row(log_importance_weight=2.)],1,region(),dict(alpha=.5,component_count=2))
        with self.assertRaisesRegex(ValueError,'Missing/repeated'):
            read_weights([row(),row()],2,region(),dict(alpha=.5,component_count=2))

    def test_branch_and_uniform_exterior_rejected(self):
        for changes in (dict(proposal_component=2),dict(proposal_component=True),dict(proposal_branch='conditional-ray')):
            with self.assertRaisesRegex(ValueError,'Gaussian branch'):
                read_weights([row(**changes)],1,region(),dict(alpha=.5,component_count=2))
        with self.assertRaisesRegex(ValueError,'uniform branch'):
            read_weights([row(proposal_branch='uniform-shell',proposal_component=None,latent=[5.,0.,0.,0.,0.,0.],latent_radius=5.,shell_valid=False)],
                         1,region(),dict(alpha=.5,component_count=2))

    def test_fixed_strata_with_explicit_exterior(self):
        u=np.zeros((4,6));u[:,0]=[0.,2.,4.,5.]
        bins=stratify_with_exterior(u,region(),STRATA,np.array([1,1,1,0],bool))
        np.testing.assert_array_equal(bins['radial'],[0,1,2,-1])
        np.testing.assert_array_equal(bins['angular'],[0,0,0,-1])
        np.testing.assert_array_equal(bins['orthant'],[63,63,63,-1])

    def test_summary_exterior_never_changes_denominator_or_class_partition(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory);records=[];all_z=[];all_h=[]
            for i in range(4):
                z=np.array([math.log(2),math.log(3),0.,-np.inf,-np.inf]);h=np.array([0.,0.,0.,-np.inf,-np.inf]);path=out/f'r{i:02d}.npz'
                np.savez_compressed(path,z=z,h=h,pairs=np.column_stack([z,z]),native=np.array([1,0,0,0,0]),contact=np.array([1,1,0,0,0]),
                    anchors=np.array([1,0,0,0,0]),triangle=np.zeros(5,bool),support=np.array([1,1,1,0,1],bool),
                    branch=np.array([0,1,1,1,0]),component=np.array([-1,0,1,1,-1]),
                    bin_radial=np.array([0,1,2,-1,0]),bin_angular=np.array([0,1,2,-1,0]),bin_orthant=np.array([0,1,63,-1,0]))
                records.append(dict(id=f'r{i:02d}',arm='bank',seed=i,samples=5,records=path.name,records_sha256=sha(path),sampler_cpu_seconds=2.,
                    native_entry_unbound_anomalies=[],near_zero_negative_core_gaps=0));all_z.extend(z);all_h.extend(h)
            m=paired_moments(all_z,all_h)
            assessment=dict(estimate=dict(logQ=m['log_Qz'],draws=20),hard_region=dict(logQ=m['log_Q0'],draws=20),importance_sampling={})
            source=summarize_arm(out,dict(id='bank',samples=5,alpha=.5,component_count=2),records,STRATA,assessment)
            for name in CLASSES:self.assertEqual(source['estimates'][name]['row_uncertainty']['draws'],20)
            self.assertAlmostEqual(source['estimates']['total']['row_uncertainty']['log_Qz'],math.log(6/5))
            self.assertAlmostEqual(source['estimates'][PRIMARY[0]]['row_uncertainty']['log_Qz'],math.log(2/5))
            self.assertEqual(source['branch_dispositions'][0]['exterior_attempts'],1)
            self.assertEqual(source['branch_dispositions'][0]['branches']['gaussian']['attempted'],3)
            self.assertEqual(source['branch_dispositions'][0]['strata_counts']['radial']['attempted'],[2,1,1])
            for family,count in [('radial',3),('angular',3),('orthant',64)]:
                self.assertEqual(len(source['strata'][family][PRIMARY[0]]),count)
                self.assertAlmostEqual(sum(v['observed_class_fraction']['Qz'] for v in source['strata'][family][PRIMARY[0]]),1.)

    def test_passing_screen_does_not_promote_production(self):
        arms={name:fixture_arm() for name in ('bank','wide')}
        result=evaluate(arms,GATES)
        self.assertTrue(result['screening_passed']);self.assertFalse(result['passed'])
        self.assertFalse(result['full_wall_coverage_established']);self.assertFalse(result['assembly_stability_established'])
        self.assertIn('fresh larger populations',result['missing_confirmatory_checks'])
        child=arms['wide']['strata']['angular'][PRIMARY[1]][1]
        child.update(estimate(math.log(.02)));child['observed_class_fraction']=dict(Qz=.02,Q0=.02)
        result=evaluate(arms,GATES)
        self.assertFalse(result['screening_passed']);self.assertTrue(result['significant_stratum_disagreements'])

    def test_contrast_agreement_catches_opposing_region_mass_shifts(self):
        arms={name:fixture_arm() for name in ('bank','wide')}
        native,other=PRIMARY[:2]
        for name,delta in [(native,.14),(other,-.14)]:
            for level in ('row_uncertainty','population_uncertainty'):
                arms['bank']['estimates'][name][level]['log_Qz'] += delta
                arms['bank']['estimates'][name][level]['Qz_relative_SE'] = .1
        arms['bank']['primary_ratios'][native+'/'+other]['population']['log_ratio'] += .28
        result=evaluate(arms,GATES)
        self.assertTrue(result['checks']['width_agreement'])
        self.assertFalse(result['checks']['width_free_energy_contrast_agreement'])
        self.assertAlmostEqual(result['free_energy_contrast_comparison']['beta_deltaF_left_minus_right'],-.28)
        self.assertFalse(compare_free_energy_intervals(dict(observed=False),dict(observed=False),GATES)['passed'])

    def test_plot_scope_is_unresolved_even_with_good_pilot(self):
        arms={name:fixture_arm() for name in ('bank','wide')};result=dict(arms=arms,convergence=evaluate(arms,GATES))
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory);plot(result,out)
            self.assertGreater((out/'contact-bank-pilot.png').stat().st_size,1000)
            self.assertIn('assembly remains unresolved',(out/'contact-bank-pilot.svg').read_text())


if __name__=='__main__':unittest.main()
