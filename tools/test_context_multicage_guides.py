"""Deterministic arithmetic/metadata tests; no sampled poses or geometry."""
import copy
import math
import unittest

import numpy as np

import prepare_context_multicage_guides as p


def synthetic_cage():
    ref=dict(source_frame_assets={'source_state':{'path':'original','sha256':'abc'}},
        source_inputs={'patch_map':{'path':'patches','sha256':'def'},'regions':{'a_neighbors':[16,217]}})
    guide=dict(schema='competing-cage-frozen-guide-v1',cage_id=0,heldout_fit_samples=0,normalized_gaussian=True,
        original_source_state=copy.deepcopy(ref['source_frame_assets']['source_state']),
        patch_reference=dict(unchanged_during_training=True,patch_map=copy.deepcopy(ref['source_inputs']['patch_map']),
            definitions=copy.deepcopy(ref['source_inputs']['regions'])),
        fit_rule=dict(training_streams=[0,1],heldout_streams=[2,3],phase='production',bandwidth_multiplier=1.,
            ridge_absolute_scaled=1e-10,ridge_trace_factor=1e-6,gaussian_components_per_declared_cage=1,heldout_parameter_updates=0),
        source_chart=dict(angular_length=3.,covariance=np.eye(6).tolist(),
            explicit_gaussian=dict(schema='source-gaussian-v1',mean=[0.]*6,provenance='training only'),
            chart_center=dict(schema='source-chart-center-v1',frame='saved-spherical-center',
                pose=dict(position=[7.,8.,9.],orientation=[1.,0.,0.,0.]),provenance='auxiliary center')))
    return guide,ref


class MulticageControls(unittest.TestCase):
    def test_exact_population_and_stratum_allocation(self):
        populations=p.future_populations()
        self.assertEqual(len(populations),8)
        self.assertEqual(sum(x['total_attempts'] for x in populations),32768)
        self.assertEqual(sum(len(x['strata']) for x in populations),32)
        for x in populations:
            self.assertEqual(x['denominator'],4096)
            self.assertEqual(sum(s['draws'] for s in x['strata']),4096)
        self.assertEqual(p.ALLOCATIONS['multicage']['cage0']*.25,256)

    def test_normalization_on_nonuniform_discrete_measure(self):
        # Seven independently normalized component densities, unrelated supports.
        cell_measure=np.array([.2,1.,2.,.5])
        masses=np.array([[1,1,1,1],[1,0,0,0],[0,1,0,0],[0,0,1,0],
            [0,0,0,1],[1,2,3,4],[4,1,2,8]],float)
        masses/=masses.sum(axis=1)[:,None];densities=masses/cell_measure
        for arm in p.ALLOCATIONS:
            logs=np.full(densities.shape,-math.inf);np.log(densities,out=logs,where=densities>0)
            q=np.array([math.exp(p.log_mixture(arm,logs[:,j])) for j in range(4)])
            self.assertAlmostEqual(float(q@cell_measure),1.)

    def test_nested_runner_law_equals_complete_arm_density(self):
        values=np.array([.11,.23,.17,.19,.31,.41,.53])
        for arm,counts in p.ALLOCATIONS.items():
            direct=sum(n/4096*(.5*values[0]+.25*values[1]+.25*values[2+i])
                       for i,n in enumerate(counts.values()))
            self.assertAlmostEqual(math.exp(p.log_mixture(arm,np.log(values))),direct)
            self.assertEqual(p.density_coefficients(arm)[0],.5)

    def test_pointwise_bound_on_every_extreme_ray(self):
        old=p.density_coefficients('baseline');new=p.density_coefficients('multicage')
        np.testing.assert_array_equal(new-.5*old,[.25,.125,0.,0.,0.,.0625,.0625])
        for j in range(7):
            logs=np.full(7,-math.inf);logs[j]=0.
            q=math.exp(p.log_mixture('baseline',logs));qnew=math.exp(p.log_mixture('multicage',logs))
            self.assertGreaterEqual(qnew,.5*q)
            if q>0:self.assertLessEqual(q/qnew,2.)

    def test_log_density_handles_disjoint_support_and_extreme_scales(self):
        for logs in [[-10000.]*7,[-math.inf]*7,[700.,-700.,-9000.,-1000.,-math.inf,400.,600.]]:
            q=p.log_mixture('baseline',logs);qn=p.log_mixture('multicage',logs)
            self.assertGreaterEqual(qn,q-math.log(2)-1e-11)
        with self.assertRaises(ValueError):p.log_mixture('multicage',[math.nan]*7)
        with self.assertRaises(ValueError):p.log_mixture('multicage',[math.inf]*7)

    def test_complete_arm_denominator_and_invalid_zeros_are_essential(self):
        # Direct expectation of all fixed strata, including a hard-invalid cell.
        component=np.array([[.2,.3,.5],[.8,.1,.1],[.1,.8,.1],[.7,.2,.1],
            [.1,.1,.8],[.2,.7,.1],[.6,.2,.2]])
        h=np.array([0.,3.,5.])
        for arm,counts in p.ALLOCATIONS.items():
            q=p.density_coefficients(arm)@component
            estimate=sum(n/4096*np.sum((.5*component[0]+.25*component[1]+.25*component[2+i])*h/q)
                         for i,n in enumerate(counts.values()))
            self.assertAlmostEqual(estimate,float(h.sum()))
            # Conditioning each stratum on validity changes its normalization.
            wrong=sum(n/4096*np.sum((.5*component[0]+.25*component[1]+.25*component[2+i])[1:]*h[1:]/q[1:])
                /(1-(.5*component[0]+.25*component[1]+.25*component[2+i])[0])
                for i,n in enumerate(counts.values()))
            self.assertGreater(wrong,estimate)

    def test_chart_center_preserves_original_reference_and_is_not_recentered(self):
        guide,ref=synthetic_cage();before=copy.deepcopy(guide)
        self.assertEqual(p.validate_cage(guide,0,ref)['pose']['position'],[7.,8.,9.])
        self.assertEqual(guide,before)
        guide['original_source_state']={'path':'center-substituted','sha256':'ghi'}
        with self.assertRaises(ValueError):p.validate_cage(guide,0,ref)

    def test_changed_region_patch_or_fit_rule_rejected(self):
        for mutate in [lambda g:g['patch_reference']['definitions'].update(a_neighbors=[16,56]),
            lambda g:g['patch_reference']['patch_map'].update(sha256='other'),
            lambda g:g['fit_rule'].update(bandwidth_multiplier=2.),
            lambda g:g['fit_rule'].update(heldout_parameter_updates=1),
            lambda g:g.update(heldout_fit_samples=1)]:
            guide,ref=synthetic_cage();mutate(guide)
            with self.assertRaises(ValueError):p.validate_cage(guide,0,ref)

    def test_heldout_diagnostics_do_not_select_or_modify_guides(self):
        guide,ref=synthetic_cage()
        for coverage in (0.,.99):
            guide['heldout_diagnostic_only']={'coverage':coverage,'loss':1e100}
            p.validate_cage(guide,0,ref)
        self.assertEqual(guide['source_chart']['covariance'],np.eye(6).tolist())

    def test_poisson_estimator_allocation_is_predeclared(self):
        recipe=dict(schema='context-multicage-source-recipe-v1',arms=list(p.ALLOCATIONS),
            components=list(p.COMPONENTS),component_draws=copy.deepcopy(p.ALLOCATIONS),
            populations_per_arm=4,attempts_per_population=4096,runner_mixture=list(p.RUNNER_MIXTURE),
            physical_conditions=copy.deepcopy(p.PHYSICAL),physical_cloud_allocation=copy.deepcopy(p.CLOUD_RULE),
            reuse_old_draws=False,seeds_materialized=False,heldout_tuning_allowed=False)
        p.validate_recipe(recipe)
        recipe['physical_cloud_allocation']['primary_estimator']='arithmetic-pair-positive-weights'
        with self.assertRaises(ValueError):p.validate_recipe(recipe)

    def test_successful_but_unrelated_fit_drain_is_rejected(self):
        recipe={k:dict(path=k,sha256=k+'-hash') for k in
            ('fit_report','fit_protocol','fit_execution_plan','fit_execution_summary')}
        summary=dict(complete=True,passed=True,unstarted=[],active=None,failure=None,
            plan_sha256=recipe['fit_execution_plan']['sha256'],
            completed=[dict(success=True,child_drained=True,returncode=0),
                dict(success=True,child_drained=True,returncode=0,terminal=recipe['fit_report'])])
        drain=dict(complete=True,passed=True,report_sha256=recipe['fit_report']['sha256'],
            protocol_sha256=recipe['fit_protocol']['sha256'],execution_plan_sha256=recipe['fit_execution_plan']['sha256'],
            execution_summary_sha256=recipe['fit_execution_summary']['sha256'])
        p.validate_fit_completion(recipe,summary,drain)
        with self.assertRaises(ValueError):p.validate_fit_completion(recipe,summary,dict(drain,report_sha256='unrelated'))
        summary['completed'][0]['child_drained']=False
        with self.assertRaises(ValueError):p.validate_fit_completion(recipe,summary,drain)


if __name__=='__main__':unittest.main()
