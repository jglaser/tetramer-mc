"""Unconditional statistics, strict comparison gates, and immutable audit bindings."""
import copy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from scipy.special import logsumexp
from analyze_conditional_ray_campaign import (
    PRIMARY, CLASSES, COMPARISONS, read_weights, stratify, primary_masks, contribution_fraction,
    compare_mass, quality_gate, free_energy_interval, evaluate, unbound_volume_bound,
    summarize_arm, validate_terminal, sha, write, plot)
from analyze_mobile_competing_reference import paired_moments
from analyze_mobile_full_capture import paired_ratio

GATES = dict(population_relative_SE_max=.1, importance_ESS_min=200., largest_draw_max=.02,
             log_agreement_absolute_max=.2, log_agreement_combined_SE_max=3., deltaF_95_halfwidth_max=.5,
             significant_stratum_mass_fraction=.01, stratum_log_agreement_absolute_max=.2)
STRATA = dict(radial_edges=[0., 2., 3., 4.], angular_projection_squared_edges=[0., 4., 9., 16.],
              latent_orthants='six signs, zero assigned positive; all 64 bins retained')


def region(correlated=False):
    lower = np.eye(6)
    if correlated: lower[3, 0] = 3.
    return dict(mahalanobis_radius=4., gaussian_chart=dict(covariances=[(lower@lower.T).tolist()],
                means=[[9., 8., 7., 6., 5., 4.]], angular_length=1.))


def row(index=0, **changes):
    item = dict(draw=index, pose=dict(position=[1., 0., 0.], orientation=[1., 0., 0., 0.]),
                hard_valid=True, capture_valid=True, region_valid=True, shell_valid=True, latent=[1., 0., 0., 0., 0., 0.],
                latent_radius=1., q=1., log_physical_jacobian=1., log_proposal_density=-1., log_hard_weight=2.,
                log_importance_weight=2.+math.log(3), clouds=[dict(log_weight=math.log(2)), dict(log_weight=math.log(4))],
                proposal_branch='conditional-ray', proposal_component=1, selected_ray_fallback=False)
    item.update(changes); return item


def zero(index):
    return row(index, hard_valid=False, clouds=[], log_importance_weight=None, log_hard_weight=None)


def estimate(logmass=0., rse=.02, ess=1000., largest=.005):
    if logmass is None:
        value=dict(log_Qz=None, log_Q0=None, draws=100, nonzero=0)
    else:
        value=dict(log_Qz=logmass, log_Q0=logmass, Qz_relative_SE=rse, Q0_relative_SE=rse,
                   Qz_ESS=ess, Q0_ESS=ess, largest_Qz_fraction=largest, largest_Q0_fraction=largest, draws=100, nonzero=100)
    return dict(row_uncertainty=copy.deepcopy(value), population_uncertainty=copy.deepcopy(value), population_count=4)


def arm():
    estimates={name:estimate(math.log(2) if name==PRIMARY[0] else 0.) for name in CLASSES}
    estimates[PRIMARY[2]]=estimate(None)
    for item in estimates.values():
        item['populations']=[dict(id=f'r{i:02d}',log_Qz=item['row_uncertainty']['log_Qz'])for i in range(4)]
        item['paired_cloud_noise']=dict(cloud_fraction=.2)if item['row_uncertainty']['log_Qz']is not None else None
    strata={}
    for family,count in [('radial',3),('angular',3),('orthant',64)]:
        strata[family]={}
        for name in CLASSES:
            items=[]
            for i in range(count):
                item=copy.deepcopy(estimates[name]) if i==0 else estimate(None)
                item.update(bin=i, observed_class_fraction=dict(Qz=1. if i==0 else 0.,Q0=1. if i==0 else 0.))
                items.append(item)
            strata[family][name]=items
    return dict(estimates=estimates,strata=strata,sampler_cpu_seconds=10.,native_entry_unbound_anomaly_count=0,
                primary_ratios={PRIMARY[0]+'/'+PRIMARY[1]:{'population':dict(log_ratio=math.log(2),log_ratio_SE=.03)}})


class ConditionalRayAnalysisTests(unittest.TestCase):
    def test_unconditional_zeros_and_complete_J_over_q(self):
        rows=[row(),zero(1),row(2,proposal_branch='uniform-shell',proposal_component=None,selected_ray_fallback=None)]
        arrays=read_weights(rows,3,region(),dict(alpha=.5))
        self.assertEqual(len(arrays['z']),3);self.assertTrue(np.isneginf(arrays['z'][1]))
        self.assertAlmostEqual(logsumexp(arrays['z'])-math.log(3),2.+math.log(2))
        np.testing.assert_array_equal(arrays['branch'],[1,1,0]);np.testing.assert_array_equal(arrays['fallback'],[0,0,-1])
        with self.assertRaisesRegex(ValueError,'J/q'):read_weights([row(log_hard_weight=1.)],1,region(),dict(alpha=.5))
        with self.assertRaisesRegex(ValueError,'Two-cloud'):read_weights([row(log_importance_weight=2.)],1,region(),dict(alpha=.5))
        changed=zero(0);changed['log_hard_weight']=0.
        with self.assertRaisesRegex(ValueError,'retain a zero'):read_weights([changed],1,region(),dict(alpha=.5))

    def test_missing_draw_support_and_branch_changes_rejected(self):
        with self.assertRaisesRegex(ValueError,'Missing/repeated'):read_weights([row(),row()],2,region(),dict(alpha=.5))
        for key in ('shell_valid','capture_valid','region_valid'):
            with self.assertRaisesRegex(ValueError,'left R4'):read_weights([row(**{key:False})],1,region(),dict(alpha=.5))
        for change in [dict(proposal_component=None),dict(selected_ray_fallback=None),dict(proposal_component=True),dict(proposal_branch='entry-shell')]:
            with self.assertRaisesRegex(ValueError,'branch metadata'):read_weights([row(**change)],1,region(),dict(alpha=.5))
        with self.assertRaisesRegex(ValueError,'branch metadata'):read_weights([row()],1,region(),dict(alpha=1.))
        with self.assertRaisesRegex(ValueError,'uniform branch'):read_weights([row(proposal_branch='uniform-shell')],1,region(),dict(alpha=.5))

    def test_strata_cover_boundaries_and_use_full_angular_covariance(self):
        points=np.zeros((6,6));points[1,0]=2.;points[2,0]=3.;points[3,3]=2.;points[4,3]=3.;points[5]=-.1
        bins=stratify(points,region(),STRATA)
        np.testing.assert_array_equal(bins['radial'],[0,1,2,1,2,0]);np.testing.assert_array_equal(bins['angular'],[0,0,0,1,2,0])
        self.assertEqual(bins['orthant'][0],63);self.assertEqual(bins['orthant'][-1],0)
        correlated=stratify(np.array([[3.,0.,0.,0.,0.,0.]]),region(True),STRATA)
        self.assertEqual(correlated['angular'][0],1)  # a²=81/10, not 81 from conditional-only covariance.
        with self.assertRaisesRegex(ValueError,'support'):stratify(np.array([[5.,0.,0.,0.,0.,0.]]),region(),STRATA)

    def test_native_precedence_preserves_exhaustive_partition(self):
        masks=primary_masks([True,True,True,False],[False,True,False,True],[True,False,False,False],[1,0,0,0],[False]*4)
        np.testing.assert_array_equal(sum(masks[k].astype(int)for k in PRIMARY),[1,1,1,0])
        self.assertTrue(masks[PRIMARY[0]][0]);self.assertFalse(masks[PRIMARY[2]][0])
        with self.assertRaisesRegex(ValueError,'anchors'):primary_masks([True],[True],[True],[0],[False])

    def test_agreement_requires_absolute_AND_population_error_limits(self):
        a,b=estimate(.3,rse=1.),estimate(0.,rse=1.)
        result=compare_mass(a,b,'Qz',GATES)
        self.assertFalse(result['passed']);self.assertFalse(result['absolute_passed']);self.assertTrue(result['SE_passed'])
        result=compare_mass(estimate(.1,rse=.001),estimate(0.,rse=.001),'Qz',GATES)
        self.assertFalse(result['passed']);self.assertTrue(result['absolute_passed']);self.assertFalse(result['SE_passed'])
        self.assertTrue(compare_mass(a,b,'Q0',GATES,absolute=False)['passed'])
        self.assertFalse(compare_mass(estimate(None),b,'Qz',GATES)['observed'])
        self.assertIsNone(contribution_fraction(estimate(None),estimate(None)))

    def test_quality_uses_population_variance_not_only_large_row_ESS(self):
        sample=estimate();sample['population_uncertainty']['Qz_relative_SE']=.4
        gate=quality_gate(sample,GATES);self.assertFalse(gate['passed']);self.assertTrue(gate['checks']['importance_ESS'])
        self.assertFalse(quality_gate(estimate(ess=199.),GATES)['passed'])
        self.assertFalse(quality_gate(estimate(largest=.021),GATES)['passed'])
        self.assertFalse(quality_gate(estimate(None),GATES)['passed'])

    def test_paired_free_energy_interval_retains_population_covariance(self):
        source=arm();source['primary_ratios'][PRIMARY[0]+'/'+PRIMARY[1]]['population']=paired_ratio(np.log([2.,4.,8.,16.]),np.log([1.,2.,4.,8.]))
        interval=free_energy_interval(source,GATES)
        self.assertAlmostEqual(interval['beta_F_native_minus_noentry'],-math.log(2));self.assertAlmostEqual(interval['halfwidth_95'],0.,places=14)
        self.assertEqual(interval['degrees_of_freedom'],3);self.assertAlmostEqual(interval['Student_t_quantile'],3.182446305284263,places=10)
        source['primary_ratios'][PRIMARY[0]+'/'+PRIMARY[1]]['population']=dict(log_ratio=None,log_ratio_SE=None)
        self.assertFalse(free_energy_interval(source,GATES)['observed'])

    def test_significant_unobserved_stratum_fails_even_when_totals_match(self):
        names={x for pair in COMPARISONS for x in pair};arms={name:arm()for name in names}
        protocol=dict(convergence=GATES)
        good=evaluate(arms,protocol);self.assertTrue(good['passed']);self.assertFalse(good['assembly_stability_established'])
        child=arms['large_ray']['strata']['angular'][PRIMARY[1]][1]
        child.update(estimate(math.log(.02)));child['observed_class_fraction']=dict(Qz=.02,Q0=.02)
        result=evaluate(arms,protocol)
        self.assertFalse(result['passed']);self.assertTrue(result['checks']['specified_mass_comparisons'])
        self.assertTrue(any(v['family']=='angular' and v['bin']==1 and not v['observed'] for v in result['significant_stratum_disagreements']))
        self.assertIn('unresolved',result['verdict'])

    def test_unbound_bound_is_geometric_and_does_not_use_observation_count(self):
        r=region();r['mahalanobis_radius']=1.
        bound=unbound_volume_bound(r)
        self.assertAlmostEqual(math.exp(bound['log_Qz_upper']),math.pi/6.)
        self.assertEqual(bound['log_Qz_upper'],bound['log_Q0_upper'])

    def test_arm_summary_preserves_every_stratum_and_original_denominator(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory);records=[];all_z=[];all_h=[]
            for index in range(4):
                z=np.log(np.array([2.,3.,1.,1.]));z[-1]=-np.inf;h=np.array([0.,0.,0.,-np.inf]);pairs=np.column_stack([z,z])
                p=out/f'r{index:02d}.npz'
                np.savez_compressed(p,z=z,h=h,pairs=pairs,native=np.array([1,0,0,0]),contact=np.array([1,1,0,0]),anchors=np.array([1,0,0,0]),triangle=np.zeros(4,bool),
                                    branch=np.array([0,1,1,1]),fallback=np.array([-1,0,1,0]),component=np.array([-1,0,1,2]),
                                    bin_radial=np.array([0,1,2,0]),bin_angular=np.array([0,1,2,0]),bin_orthant=np.array([0,1,63,0]))
                records.append(dict(id=f'r{index:02d}',arm='fixture',seed=30+index,samples=4,records=p.name,records_sha256=sha(p),sampler_cpu_seconds=2.,
                                    native_entry_unbound_anomalies=[],near_zero_negative_core_gaps=0))
                all_z.extend(z);all_h.extend(h)
            moments=paired_moments(all_z,all_h)
            assessment=dict(estimate=dict(logQ=moments['log_Qz'],draws=16),hard_region=dict(logQ=moments['log_Q0'],draws=16),importance_sampling={})
            source=summarize_arm(out,dict(id='fixture',samples=4,alpha=.5),records,STRATA,assessment)
            for name in CLASSES:self.assertEqual(source['estimates'][name]['row_uncertainty']['draws'],16)
            for family,count in [('radial',3),('angular',3),('orthant',64)]:
                self.assertEqual(len(source['strata'][family][PRIMARY[0]]),count)
                self.assertAlmostEqual(sum(v['observed_class_fraction']['Qz']for v in source['strata'][family][PRIMARY[0]]),1.)
            self.assertEqual(source['branch_dispositions'][0]['branches']['ray-fallback']['attempted'],1)
            self.assertEqual(source['branch_dispositions'][0]['strata_counts']['radial']['attempted'],[2,1,1])
            self.assertEqual(source['branch_dispositions'][0]['strata_counts']['radial']['contributing'],[1,1,1])
            self.assertAlmostEqual(source['branch_weight_contributions']['conditional-ray'][PRIMARY[1]]['observed_class_fraction']['Qz'],1.)
            self.assertAlmostEqual(source['estimates']['total']['row_uncertainty']['log_Qz'],math.log(6/4))

    def test_diagnostic_plot_renders_separate_populations_and_unresolved_title(self):
        names=['pilot_uniform','pilot_ray','large_uniform','large_ray','alpha02','lambda128']
        result=dict(arms={name:arm()for name in names},convergence=dict(passed=False),protocol_convergence=GATES)
        with tempfile.TemporaryDirectory()as directory:
            out=Path(directory);plot(result,out)
            for name in ('conditional-ray-reference','conditional-ray-strata'):
                self.assertGreater((out/(name+'.png')).stat().st_size,1000)
                svg=(out/(name+'.svg')).read_text()
                self.assertIn('Convergence unresolved',svg)
            self.assertIn('Small dots: independent populations',(out/'conditional-ray-reference.svg').read_text())

    def test_terminal_requires_completed_physics_and_successful_hash_bound_audit(self):
        with tempfile.TemporaryDirectory()as directory:
            root=Path(directory);(root/'a/assessment').mkdir(parents=True);(root/'a/provenance').mkdir()
            write(root/'protocol.json',{});write(root/'a/provenance/config.json',{});write(root/'definition.json',{})
            output=dict(samples_sha256='rows')
            jobs=[dict(arm='a',id=f'r{i:02d}',seed=i,samples=2)for i in range(4)]
            protocol=dict(jobs=jobs,arms=[dict(id='a',alpha=.5)],region_sha256='region',shape_sha256='shape',
                          native_definition='definition.json',native_definition_sha256=sha(root/'definition.json'))
            assessment=dict(region_sha256='region',estimate=dict(draws=8),hard_region=dict(draws=8),independently_reconstructed_poses=8,
                            populations=[dict(id=j['id'],seed=j['seed'],samples_sha256='rows',estimate=dict(draws=2),hard_region=dict(draws=2))for j in jobs],
                            importance_sampling=dict(uniform_shell_probability=.5,conditional_ray_component_count=3,guide_schema='defensive-conditional-ray-guide-v1',shell_rejected=0))
            write(root/'a/assessment/analysis.json',assessment)
            status=dict(schema='conditional-ray-reference-status-v1',complete=True,phase='complete',protocol_sha256=sha(root/'protocol.json'),
                        jobs=[dict(j,status='complete',returncode=0,output=output)for j in jobs],audits={'a':dict(returncode=0,analysis_sha256=sha(root/'a/assessment/analysis.json'))})
            write(root/'status.json',status)
            with patch('analyze_conditional_ray_campaign.validate_frozen',return_value=protocol),patch('analyze_conditional_ray_campaign.verify_output',return_value=output),patch('analyze_conditional_ray_campaign.validate_classifier_target'):
                validate_terminal(root)
                status['audits']['a']['analysis_sha256']='wrong';write(root/'status.json',status)
                with self.assertRaisesRegex(ValueError,'Raw audit'):validate_terminal(root)
                status['complete']=False;write(root/'status.json',status)
                with self.assertRaisesRegex(ValueError,'has not completed'):validate_terminal(root)


if __name__=='__main__':unittest.main()
