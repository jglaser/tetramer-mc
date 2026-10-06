"""Deterministic bridge-coordinate controls; no protein queries or pose draws."""
import copy
import math
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

import prepare_context_bridge_guides as p
from source_guide_reference import SourceDensity, cayley, pose, relative, rotation


def fixture():
    anchor = pose([3.,-4.,7.],Rotation.from_rotvec([.6,-.2,.4]).as_matrix())
    source = pose([-2.,5.,1.],Rotation.from_rotvec([-.3,.5,.2]).as_matrix())
    center = pose([1.,8.,-2.],Rotation.from_rotvec([.1,-.4,.7]).as_matrix())
    lower = np.diag([.12,.15,.08,.11,.09,.14])
    lower[3,0]=.07;lower[4,1]=-.05;lower[5,2]=.03;lower[5,0]=.04
    covariance = lower@lower.T
    def spec(mean,scale,origin=None):
        result = dict(angular_length=3.5,covariance=(scale*covariance).tolist(),
            explicit_gaussian=dict(schema='source-gaussian-v1',mean=mean,provenance='synthetic fixture'))
        if origin is not None:
            result['chart_center'] = dict(schema='source-chart-center-v1',frame='saved-spherical-center',
                pose=origin,provenance='synthetic noncommuting center')
        return result
    return source,anchor,spec([.2,-.1,.3,.7,-.4,.2],1.),spec([-.2,.3,.1,-.4,.5,.6],2.,center)


def encode_relative_rotation(matrix):
    q = Rotation.from_matrix(matrix).as_quat()
    return q[:3]/q[3]


class BridgeTests(unittest.TestCase):
    def test_recenter_derivative_matches_finite_difference_noncommuting_frame(self):
        source,anchor,spec,_ = fixture()
        d = SourceDensity(spec,source,anchor);mean=d.mean;ell=d.ell
        center = d.decode(np.zeros(6));rcenter=rotation(relative(center,anchor))
        def recentered(x):
            value_rotation=cayley(x[3:]/ell)@d.center_rotation
            return np.r_[x[:3]-mean[:3],ell*encode_relative_rotation(value_rotation@rcenter.T)]
        step=1e-6
        numerical=np.column_stack([(recentered(mean+step*np.eye(6)[j])-recentered(mean-step*np.eye(6)[j]))/(2*step) for j in range(6)])
        np.testing.assert_allclose(p.recenter_jacobian(mean,ell),numerical,rtol=2e-8,atol=8e-10)
        np.testing.assert_allclose(recentered(mean),np.zeros(6),atol=2e-15)

    def test_exact_finite_recenter_identity_fixes_cross_product_sign(self):
        u=np.array([.3,-.7,.2]);delta=np.array([.02,-.04,.01])
        actual=encode_relative_rotation(cayley(u+delta)@cayley(u).T)
        expected=(np.eye(3)+p.skew(u))@delta/(1+u@u+delta@u)
        np.testing.assert_allclose(actual,expected,rtol=2e-13,atol=2e-16)
        self.assertGreater(np.linalg.norm(actual-(np.eye(3)-p.skew(u))@delta/(1+u@u+delta@u)),1e-3)

    def test_covariance_recentering_keeps_translation_rotation_cross_blocks(self):
        source,anchor,spec,_=fixture();e=p.recenter_endpoint(spec,source,anchor)
        j=p.recenter_jacobian(spec['explicit_gaussian']['mean'],spec['angular_length'])
        np.testing.assert_allclose(e['covariance'],j@np.array(spec['covariance'])@j.T,rtol=1e-14,atol=1e-17)
        self.assertGreater(np.linalg.norm(e['covariance'][:3,3:]),.001)
        self.assertGreater(np.linalg.eigvalsh(e['covariance']).min(),0.)

    def test_bridge_endpoints_shortest_arc_and_anchor_frame_noncommute(self):
        source,anchor,f,c=fixture();guides,meta=p.bridge_specs(f,c,source,anchor,'frozen')
        left,right=[relative(x,anchor) for x in meta['endpoint_centers']]
        r0,r1=rotation(left),rotation(right);omega=np.array(meta['anchor_frame_rotation_vector'])
        self.assertGreater(np.linalg.norm(r0@r1-r1@r0),.01)
        np.testing.assert_allclose(Rotation.from_rotvec(omega).as_matrix()@r0,r1,atol=6e-16)
        for i,a in enumerate(p.ALPHAS):
            actual=relative(guides[f'bridge_a{i}_b1']['source_chart']['chart_center']['pose'],anchor)
            np.testing.assert_allclose(actual['position'],(1-a)*np.array(left['position'])+a*np.array(right['position']),atol=2e-15)
            np.testing.assert_allclose(rotation(actual),Rotation.from_rotvec(a*omega).as_matrix()@r0,atol=8e-16)
        self.assertLess(np.linalg.norm(omega),math.pi)

    def test_quaternion_sign_changes_do_not_change_bridge_law(self):
        source,anchor,f,c=fixture();g,_=p.bridge_specs(f,c,source,anchor,'frozen')
        source2=copy.deepcopy(source);anchor2=copy.deepcopy(anchor);c2=copy.deepcopy(c)
        for value in (source2,anchor2,c2['chart_center']['pose']):
            value['orientation']=[-x for x in value['orientation']]
        h,_=p.bridge_specs(f,c2,source2,anchor2,'frozen')
        for name in p.BRIDGES:
            a,b=g[name]['source_chart'],h[name]['source_chart']
            np.testing.assert_allclose(a['covariance'],b['covariance'],rtol=0,atol=2e-16)
            np.testing.assert_allclose(rotation(a['chart_center']['pose']),rotation(b['chart_center']['pose']),atol=8e-16)
            x=SourceDensity(a,source,anchor).decode(np.arange(6)/7)
            self.assertAlmostEqual(SourceDensity(a,source,anchor).evaluate(x),SourceDensity(b,source2,anchor2).evaluate(x),places=11)

    def test_half_turn_ambiguity_rejected(self):
        source=dict(position=[0.,0.,0.],orientation=[1.,0.,0.,0.]);anchor=copy.deepcopy(source)
        f=dict(angular_length=1.,covariance=np.eye(6).tolist(),explicit_gaussian=dict(schema='source-gaussian-v1',mean=[0.]*6,provenance='fixture'))
        c=copy.deepcopy(f);c['chart_center']=dict(schema='source-chart-center-v1',frame='saved-spherical-center',
            pose=dict(position=[1.,0.,0.],orientation=[0.,1.,0.,0.]),provenance='half-turn')
        with self.assertRaisesRegex(ValueError,'half-turn'):p.bridge_specs(f,c,source,anchor,'frozen')

    def test_b4_is_covariance_factor16_before_unchanged_along_term(self):
        source,anchor,f,c=fixture();g,meta=p.bridge_specs(f,c,source,anchor,'frozen')
        v=np.array(meta['tangent_anchor_coordinates']);along=p.ALONG_SIGMA**2*np.outer(v,v)
        for i,a in enumerate(p.ALPHAS):
            s1=np.array(g[f'bridge_a{i}_b1']['source_chart']['covariance'])
            s4=np.array(g[f'bridge_a{i}_b4']['source_chart']['covariance'])
            np.testing.assert_allclose(s4-along,16*(s1-along),rtol=2e-14,atol=5e-16)
            self.assertGreater(np.linalg.eigvalsh(s1).min(),0.);self.assertGreater(np.linalg.eigvalsh(s4).min(),0.)
        # Endpoint bridge covariance includes the deliberate along-path term.
        self.assertGreater(np.linalg.norm(np.array(g['bridge_a0_b1']['source_chart']['covariance'])-meta['endpoint_recentered_covariances'][0]),.01)

    def test_bridge_normal_density_jacobian_at_fixed_latents(self):
        source,anchor,f,c=fixture();guides,_=p.bridge_specs(f,c,source,anchor,'frozen')
        for g in guides.values():
            spec=g['source_chart'];d=SourceDensity(spec,source,anchor)
            for z in (np.zeros(6),np.array([.4,-.7,.2,-.8,.3,.5])):
                x=d.mean+d.lower@z;u=x[3:]/d.ell
                log_j=-3*math.log(d.ell)-2*math.log(math.pi)-2*math.log1p(u@u)
                expected=-.5*(z@z)-3*math.log(2*math.pi)-d.logdet-log_j
                self.assertAlmostEqual(d.evaluate(d.decode(z)),expected,places=11)

    def test_inputs_old_mean_and_old_covariance_remain_unchanged(self):
        source,anchor,f,c=fixture();before=copy.deepcopy((source,anchor,f,c))
        guides,_=p.bridge_specs(f,c,source,anchor,'frozen')
        self.assertEqual((source,anchor,f,c),before)
        self.assertTrue(all(g['source_chart']['explicit_gaussian']['mean']==[0.]*6 for g in guides.values()))

    def test_exact72_strata_16384_attempts_and_fixed_fractions(self):
        populations=p.future_populations()
        self.assertEqual(len(populations),8);self.assertEqual(sum(x['total_attempts'] for x in populations),16384)
        self.assertEqual(sum(len(x['strata']) for x in populations),72)
        for x in populations:
            self.assertEqual(sum(j['draws'] for j in x['strata']),2048)
            self.assertEqual(sum(j['deterministic_fraction'] for j in x['strata']),1.)
        self.assertEqual(p.ALLOCATIONS['bridge']['bridge_a0_b1'],128)

    def test_normalized_full_mixture_on_nonuniform_discrete_measure(self):
        measure=np.array([.2,1.,2.,.5]);mass=np.arange(1,61,dtype=float).reshape(15,4)
        mass[0]=[1.,0.,0.,0.];mass[1]=[0.,0.,1.,0.];mass/=mass.sum(axis=1)[:,None]
        density=mass/measure;logs=np.full(density.shape,-math.inf);np.log(density,out=logs,where=density>0)
        for arm in p.ALLOCATIONS:
            q=np.array([math.exp(p.log_mixture(arm,logs[:,i])) for i in range(4)])
            self.assertAlmostEqual(q@measure,1.,places=14)

    def test_nested_runner_density_and_pointwise_defensive_bound(self):
        values=np.arange(1,16,dtype=float)/17
        for arm,counts in p.ALLOCATIONS.items():
            expected=sum(n/2048*(.5*values[0]+.25*values[1]+.25*values[2+i]) for i,n in enumerate(counts.values()))
            self.assertAlmostEqual(math.exp(p.log_mixture(arm,np.log(values))),expected,places=14)
            self.assertEqual(p.density_coefficients(arm)[0],.5)
        self.assertTrue(np.all(p.density_coefficients('bridge')-.5*p.density_coefficients('baseline')>=0))
        for i in range(15):
            logs=np.full(15,-math.inf);logs[i]=0.
            self.assertGreaterEqual(math.exp(p.log_mixture('bridge',logs)),.5*math.exp(p.log_mixture('baseline',logs)))

    def test_mixture_preserves_hard_invalid_zeros_and_unconditional_denominator(self):
        mass=np.arange(1,46,dtype=float).reshape(15,3);mass/=mass.sum(axis=1)[:,None]
        physical=np.array([0.,3.,5.])
        for arm,counts in p.ALLOCATIONS.items():
            q=p.density_coefficients(arm)@mass
            expected=sum(n/2048*np.sum((.5*mass[0]+.25*mass[1]+.25*mass[2+i])*physical/q) for i,n in enumerate(counts.values()))
            self.assertAlmostEqual(expected,8.,places=13)

    def test_recipe_locks_geometry_only_and_all_diagnostics(self):
        recipe=dict(schema='context-bridge-source-recipe-v1',components=list(p.COMPONENTS),component_draws=copy.deepcopy(p.ALLOCATIONS),
            arms=list(p.ALLOCATIONS),populations_per_arm=4,attempts_per_population=2048,runner_mixture=list(p.RUNNER_MIXTURE),
            alphas=list(p.ALPHAS),standard_deviation_widths=list(p.WIDTHS),along_parameter_sigma=p.ALONG_SIGMA,
            physical_conditions=copy.deepcopy(p.PHYSICAL),mode='geometry',cloud_draws=0,diagnostics=copy.deepcopy(p.DIAGNOSTICS),
            reuse_old_draws=False,seeds_materialized=False,heldout_tuning_allowed=False)
        p.validate_recipe(recipe)
        for key,value in [('mode','clouds'),('cloud_draws',1),('standard_deviation_widths',[1.,2.]),('reuse_old_draws',True)]:
            bad=copy.deepcopy(recipe);bad[key]=value
            with self.assertRaises(ValueError):p.validate_recipe(bad)

    def test_baseline_and_original_source_classifier_are_preserved(self):
        old=dict(schema='context-multicage-source-mixture-v1',complete=True,passed=True,runner_mixture=list(p.RUNNER_MIXTURE),
            physical_conditions=copy.deepcopy(p.PHYSICAL),guides={name:{} for name in p.OLD_COMPONENTS},
            effective_density_coefficients={'multicage':p.density_coefficients('baseline')[:7].tolist()},
            source_inputs={'regions':dict(a_neighbors=[16,217],b_neighbors=[16,56],secondary_label=217,
                source_secondary_tokens=list(range(16)),inclusion_boundaries=[0.,.25,.5,.75,1.])})
        before=copy.deepcopy(old);p.validate_baseline(old);self.assertEqual(old,before)
        old['source_inputs']['regions']['a_neighbors']=[16,56]
        with self.assertRaises(ValueError):p.validate_baseline(old)


if __name__=='__main__':
    unittest.main()
