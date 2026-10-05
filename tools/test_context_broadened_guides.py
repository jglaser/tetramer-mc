"""Deterministic metadata/density tests; no physical geometry, RNG or fitting."""
import copy
import math
import unittest

import numpy as np

from prepare_context_broadened_guides import (
    ALLOCATIONS, ATTEMPTS, COMPONENTS, broaden, density_coefficients, log_mixture,
    stratified_allocation, validate_controls, validate_recipe,
)


def controls():
    lower = np.diag([.2,.3,.4,.1,.15,.25])
    lower[3,0], lower[4,1], lower[5,2] = .04,-.05,.08
    covariance = lower@lower.T
    result = []
    for arm, matrix in [('full', covariance), ('diagonal', np.diag(np.diag(covariance)))]:
        result.append(dict(schema='context-covariance-frozen-guide-v1', arm=arm,
            training_streams=[0,1],heldout_streams=[2,3],reduction_sha256='a'*64,fit_plan_sha256='b'*64,
            source_chart=dict(angular_length=3.,covariance=matrix.tolist(),explicit_gaussian=dict(
                schema='source-gaussian-v1',mean=[.1,-.2,.3,.02,-.03,.04],provenance='synthetic frozen control')),
            regularization=dict(scaled_ridge=1e-9)))
    return result


class BroadenedGuideTests(unittest.TestCase):
    def test_covariance_scale_preserves_mean_frame_and_original_assets(self):
        full, diagonal = controls(); original = copy.deepcopy(full)
        validate_controls(full,diagonal)
        broad = broaden(full,dict(path='/held/full.json',sha256='c'*64))
        np.testing.assert_array_equal(broad['source_chart']['covariance'],4*np.asarray(full['source_chart']['covariance']))
        self.assertEqual(broad['source_chart']['explicit_gaussian']['mean'],full['source_chart']['explicit_gaussian']['mean'])
        self.assertEqual(broad['source_chart']['angular_length'],full['source_chart']['angular_length'])
        self.assertEqual(broad['inherited_regularization'],full['regularization'])
        self.assertFalse(broad['additional_regularization']);self.assertEqual(full,original)
        self.assertEqual(broad['new_fit_samples'],0)

    def test_chart_density_scale_ratio_is_exact_at_fixed_physical_pose(self):
        full,_ = controls(); broad = broaden(full,dict(path='/held/full.json',sha256='c'*64))
        covariance = np.asarray(full['source_chart']['covariance'])
        wider = np.asarray(broad['source_chart']['covariance'])
        mean = np.asarray(full['source_chart']['explicit_gaussian']['mean'])
        lower = np.linalg.cholesky(covariance)
        for latent in [np.zeros(6),np.array([1.,-2.,3.,.4,-.5,.6])]:
            coordinate = mean+lower@latent
            delta = coordinate-mean
            def log_gaussian(matrix):
                return -.5*(delta@np.linalg.solve(matrix,delta)+np.linalg.slogdet(matrix)[1]+6*math.log(2*math.pi))
            # Both charts evaluate the same physical pose in the same Cayley
            # coordinates, so the unchanged physical Jacobian cancels here.
            actual = log_gaussian(wider)-log_gaussian(covariance)
            expected = -6*math.log(2)+3*float(latent@latent)/8
            self.assertAlmostEqual(actual,expected,places=12)

    def test_normalization_density_bound_and_equal_fixed_stratification(self):
        densities = np.array([[.1,.2,.7],[.4,.4,.2],[.8,.2,0],[0,.3,.7],[.3,.1,.6]])
        self.assertTrue(np.allclose(densities.sum(axis=1),1))
        logs = np.full_like(densities,-np.inf)
        np.log(densities,out=logs,where=densities>0)
        q = {a:np.array([math.exp(log_mixture(a,logs[:,i])) for i in range(3)]) for a in ALLOCATIONS}
        for arm in ALLOCATIONS:
            self.assertAlmostEqual(q[arm].sum(),1.)
            stratified = np.zeros(3)
            for i,component in enumerate(COMPONENTS):
                runner = .5*densities[0]+.25*densities[1]+.25*densities[2+i]
                stratified += ALLOCATIONS[arm][component]/ATTEMPTS*runner
            np.testing.assert_allclose(stratified,q[arm],rtol=2e-15)
            target = np.array([1.,0.,2.])  # Hard-invalid middle state stays zero.
            self.assertAlmostEqual(np.sum(stratified*target/q[arm]),target.sum())
        self.assertTrue(np.all(q['broadened'] >= .75*q['baseline']-1e-15))
        np.testing.assert_array_equal(density_coefficients('broadened')-.75*density_coefficients('baseline'),[.125,.0625,0,0,.0625])
        # A point supported only by F attains the .75 lower bound exactly.
        only_full=[-math.inf,-math.inf,0.,-math.inf,-math.inf]
        self.assertAlmostEqual(math.exp(log_mixture('broadened',only_full)-log_mixture('baseline',only_full)),.75)

    def test_allocations_have_eight_independent_populations_and_no_seeds(self):
        populations = stratified_allocation()
        self.assertEqual(len(populations),8)
        self.assertEqual(len({(p['arm'],p['population']) for p in populations}),8)
        self.assertEqual(sum(len(p['strata']) for p in populations),20)
        self.assertEqual(sum(p['total_attempts'] for p in populations),32768)
        for p in populations:
            self.assertEqual(sum(s['draws'] for s in p['strata']),4096)
            self.assertEqual(sum(s['deterministic_fraction'] for s in p['strata']),1.)
            self.assertEqual(p['denominator'],4096)
            self.assertNotIn('seed',p)

    def test_changed_controls_and_unsupported_recipe_are_rejected(self):
        full,diagonal = controls()
        for mutation in ('mean','diagonal','singular'):
            bad = copy.deepcopy(diagonal)
            if mutation == 'mean':bad['source_chart']['explicit_gaussian']['mean'][0] += .01
            elif mutation == 'diagonal':bad['source_chart']['covariance'][0][1] = .001
            else:bad['source_chart']['covariance'] = np.zeros((6,6)).tolist()
            with self.assertRaises(ValueError):validate_controls(full,bad)
        recipe=dict(schema='context-broadened-source-recipe-v1',arms=['baseline','broadened'],
            components=list(COMPONENTS),populations_per_arm=4,attempts_per_population=4096,
            component_draws=copy.deepcopy(ALLOCATIONS),covariance_scale=4.,runner_mixture=[.5,.25,.25],
            physical_conditions=dict(depletant_radius=1.5,activity=.035,**{'lambda':2.24}),
            seeds_materialized=False,physical_cloud_allocation=None,reuse_old_draws=False)
        validate_recipe(recipe)
        for name,value in [('covariance_scale',9.),('seeds_materialized',True),('reuse_old_draws',True)]:
            bad=copy.deepcopy(recipe);bad[name]=value
            with self.assertRaises(ValueError):validate_recipe(bad)
        with self.assertRaises(ValueError):log_mixture('baseline',[0,0,float('nan'),0,0])
        with self.assertRaises(ValueError):log_mixture('baseline',[0,0,float('inf'),0,0])


if __name__ == '__main__':unittest.main()
