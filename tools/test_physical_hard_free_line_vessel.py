"""Bounded synthetic controls; no protein poses, target clouds or classifiers."""
import copy
import math
import unittest
import numpy as np
from scipy.special import logsumexp
from scipy.integrate import quad
import physical_hard_free_line_vessel as reference
from test_physical_latent_guide import fixture,independent_pose,SHAPE,REGION


def setup(alpha=.5,beta=1.,scale=1.,axes=(0,1,2)):
    region,guide,lower=fixture(alpha);region['capture_radius']=170.
    lower[0,0]=scale;region['gaussian_chart']['covariances']=[(lower@lower.T).tolist()]
    guide.update(schema='defensive-hard-free-line-guide-v1',conditional_probability=beta,minimum_conditional_mass=1e-12,raw_translation_axes=list(axes))
    shape=dict(atoms=[dict(center=[0.,0.,0.],radius=1.)])
    config=dict(fixed_poses=[region['fixed_neighbor']],capture_center=[0.,0.,0.],capture_radius=273.,
        initial_pose=dict(position=[3.,0.,0.],orientation=[1.,0.,0.,0.]),depletant_radius=.2,
        metadata=dict(native_poses=[region['fixed_neighbor']],rigid_members=[dict(position=[0.,0.,0.])],member_error_scale=1.,angle_error_scale_deg=30.))
    value=reference.PhysicalHardFreeLineGuide(region,guide,config,shape,region_sha256=REGION,expected_shape_sha256=SHAPE)
    return value,region,guide,config,shape,lower


def trace(value,density):
    r=value.recon;result=density.reconstruction
    if result is None:return None
    if result.get('conditioning_disabled'):return dict(conditioning_disabled=True)
    raw=r.raw(density.latent);g=r.gaussian_logs(density.latent)
    answer=dict(raw_coordinates=raw.tolist(),baseline_log_density=result['baseline_log_density'],
        component_branches=result['component_branches'],fallback_component_branches=result['fallback_component_branches'],axes=[])
    uniform=math.log(r.alpha)-r.logvolume if density.in_reference_ball else -math.inf
    for a in result['axes']:
        item={k:copy.deepcopy(v) for k,v in a.items() if k not in ('conditional_means','conditional_sigmas','conditional_masses','component_fallbacks')}
        item['hard_free_intervals']=item.pop('intervals');mass=np.array(a['conditional_masses']);fallback=np.array(a['component_fallbacks'])
        multipliers=np.ones(len(g));multipliers[~fallback]=1/mass[~fallback] if reference.line.contains(a['intervals'],raw[a['axis']]) else 0.
        c=1-r.beta+r.beta*multipliers;positive=c>0
        logq=float(np.logaddexp(uniform,math.log1p(-r.alpha)+logsumexp(g[positive]+np.log(c[positive]))))
        item['axis_log_proposal_density']=logq if math.isfinite(logq) else None;answer['axes'].append(item)
    return answer


def rows_fixture():
    value,region,guide,cfg,shape,L=setup()
    manifest=dict(schema=6,outer_mixture_schema=reference.SCHEMA,latent_guide_schema=guide['schema'],
        samples=4,shape_sha256=SHAPE,pose_proposal_schema=1,covariance_scale=1.,proposal_anchor_index=None,
        physical_fixed_neighbor_count=1,uniform_probability=.5,latent_gaussian_component_count=1,
        latent_defensive_uniform_probability=.5,cloud_replicates=2,activity=0.,**{'lambda':1.})
    vessel=reference.vessel_reference.VesselDensity(cfg,manifest,region['gaussian_chart'])
    us=[[3.,0.,0.,0.,0.,0.],[2.5,2.5,2.5,0.,0.,0.],[5.,0.,0.,0.,0.,0.],[0.]*6]
    poses=[independent_pose(u,region,L)[0] for u in us];vl,_=vessel.evaluate(poses);densities=value.evaluate_many(poses)
    mixture=reference.original.half_mixture_log_density(vl,[d.log_physical_density for d in densities]);rows=[]
    for i,(u,p,d) in enumerate(zip(us,poses,densities)):
        valid=i!=3;clouds=[dict(log_weight=0.,overlap_points=0,lower_volume=0.,uncertain_volume=0.) for _ in range(2)] if valid else [];latent=i==0
        row=dict(draw=i,pose=p,outer_branch='latent' if latent else 'vessel',proposal=None if latent else {},
            latent_proposal=dict(latent=u,latent_radius=float(np.linalg.norm(u)),gaussian_component=None,
                hard_free_line_draw=dict(conditional=False,original_latent=u)) if latent else None,
            latent_density=dict(latent=d.latent,in_reference_ball=d.in_reference_ball,coordinate_chart_seam=False,
                log_latent_density=d.log_latent_density if math.isfinite(d.log_latent_density) else None,
                log_physical_jacobian=d.log_physical_jacobian,structural_zero=d.structural_zero,hard_free_line_density=trace(value,d)),
            log_vessel_proposal_density=float(vl[i]),log_latent_physical_density=d.log_physical_density if math.isfinite(d.log_physical_density) else None,
            log_proposal_density=float(mixture[i]),capture_valid=True,wall_valid=True,hard_valid=valid,
            log_hard_weight=-float(mixture[i]) if valid else None,log_importance_weight=-float(mixture[i]) if valid else None,
            q=reference.regional.native.native_q(cfg['metadata'],p) if valid else None,region=None,depletion_contact=None,clouds=clouds)
        rows.append(row)
    return cfg,manifest,rows,vessel,value


class PhysicalHardFreeVesselTests(unittest.TestCase):
    def test_source_capture170_is_separate_from_vessel273(self):
        value,region,guide,cfg,shape,L=setup(scale=100.,axes=(0,))
        pose,_=independent_pose([1.8,0.,0.,0.,0.,0.],region,L);a=value.evaluate(pose)
        self.assertGreater(pose['position'][0],170.);self.assertLess(pose['position'][0],273.)
        self.assertEqual(value.source_config['capture_radius'],170.)
        changed=copy.deepcopy(cfg);changed['capture_radius']=500.
        other=reference.PhysicalHardFreeLineGuide(region,guide,changed,shape,region_sha256=REGION,expected_shape_sha256=SHAPE)
        self.assertEqual(a.log_physical_density,other.evaluate(pose).log_physical_density)
        wrong=copy.deepcopy(region);wrong['capture_radius']=273.
        different=reference.PhysicalHardFreeLineGuide(wrong,guide,cfg,shape,region_sha256=REGION,expected_shape_sha256=SHAPE)
        self.assertGreater(different.evaluate(pose).log_physical_density,a.log_physical_density)

    def test_genuine_exterior_zero_and_exterior_fallback_both_retained(self):
        value,region,_,_,_,L=setup()
        zero=value.evaluate(independent_pose([2.5,2.5,2.5,0,0,0],region,L)[0])
        tail=value.evaluate(independent_pose([5.,0,0,0,0,0],region,L)[0])
        self.assertTrue(zero.structural_zero);self.assertEqual(zero.log_physical_density,-math.inf)
        self.assertFalse(tail.in_reference_ball);self.assertFalse(tail.structural_zero);self.assertTrue(math.isfinite(tail.log_physical_density))

    def test_underflow_is_not_structural_zero_or_component_censoring(self):
        value,region,_,_,_,L=setup(beta=0.)
        result=value.evaluate(independent_pose([100.,0,0,0,0,0],region,L)[0])
        self.assertFalse(result.structural_zero);self.assertTrue(math.isfinite(result.log_physical_density))
        self.assertEqual(math.exp(result.log_physical_density),0.)
        with self.assertRaisesRegex(ValueError,'Unrepresentable positive Gaussian'):
            value.evaluate(dict(position=[1e160,0.,0.],orientation=[1.,0.,0.,0.]))

    def test_exact_seam_and_uniform_extreme_exterior(self):
        value,*_=setup(alpha=1.)
        seam=value.evaluate(dict(position=[3.,0.,0.],orientation=[0.,1.,0.,0.]))
        self.assertIsNone(seam.latent);self.assertTrue(seam.structural_zero)
        exterior=value.evaluate(dict(position=[1e160,0.,0.],orientation=[1.,0.,0.,0.]))
        self.assertIsNotNone(exterior.latent);self.assertTrue(exterior.structural_zero);self.assertEqual(exterior.log_physical_density,-math.inf)

    def test_full_mixture_outside_R4_jacobian_once_and_invalid_zeros(self):
        cfg,manifest,rows,vessel,value=rows_fixture();result=reference.check_rows(cfg,manifest,rows,vessel,value)
        self.assertEqual(result['checked_attempts'],4);self.assertEqual(result['valid_outside_R4'],2)
        self.assertEqual(result['structural_zero_queries'],1)
        self.assertAlmostEqual(rows[1]['log_proposal_density'],rows[1]['log_vessel_proposal_density']-math.log(2.),places=13)
        bad=copy.deepcopy(rows);bad[0]['log_hard_weight']+=bad[0]['latent_density']['log_physical_jacobian']
        with self.assertRaisesRegex(ValueError,'no extra J'):reference.check_rows(cfg,manifest,bad,vessel,value)
        bad=copy.deepcopy(rows);bad[3]['log_importance_weight']=0.
        with self.assertRaisesRegex(ValueError,'explicit zero'):reference.check_rows(cfg,manifest,bad,vessel,value)

    def test_tampered_structural_zero_missing_axis_and_source_interval_rejected(self):
        cfg,manifest,rows,vessel,value=rows_fixture()
        for change,message in [('structural','Structural-zero'),('axis','axis sequence'),('interval','hard-free interval')]:
            bad=copy.deepcopy(rows)
            if change=='structural':bad[0]['latent_density']['structural_zero']=True
            if change=='axis':bad[0]['latent_density']['hard_free_line_density']['axes'].pop()
            if change=='interval':bad[0]['latent_density']['hard_free_line_density']['axes'][0]['hard_free_intervals'][0]['lower']+=.1
            with self.assertRaisesRegex(ValueError,message):reference.check_rows(cfg,manifest,bad,vessel,value)

    def test_uniform_and_beta_zero_match_original_normalized_guide(self):
        for alpha,beta in [(1.,1.),(.5,0.)]:
            value,region,guide,_,_,L=setup(alpha=alpha,beta=beta)
            g=dict(schema='defensive-latent-shell-guide-v1',region_sha256=REGION,
                defensive_uniform_shell_probability=alpha,gaussian_components=guide['gaussian_components'])
            old=reference.original.PhysicalLatentGuide(region,g,region_sha256=REGION,expected_shape_sha256=SHAPE)
            for u in ([2.2,0,0,0,0,0],[5.,0,0,.1,.2,.3]):
                p,_=independent_pose(u,region,L);self.assertAlmostEqual(value.evaluate(p).log_physical_density,old.evaluate(p).log_physical_density,places=12)

    def test_conditional_normalization_after_physical_measure_conversion(self):
        value,region,_,_,_,L=setup(axes=(0,))
        def integrand(s):
            pose,jac=independent_pose([s,0,0,0,0,0],region,L)
            d=value.evaluate(pose)
            return math.exp(d.log_physical_density+jac)
        # Integrate one raw translation while fixing the five retained
        # coordinates. The Gaussian conditional mass must cancel exactly;
        # multiplying physical q by J restores the original latent measure.
        integral=sum(quad(integrand,a,b,epsabs=1e-13)[0] for a,b in [(-4.,-2.),(-2.,2.),(2.,4.)])
        expected=.5*8/math.exp(value.log_volume)+.5/(2*math.pi)**2.5
        self.assertAlmostEqual(integral,expected,places=12)

    def test_all_attempt_denominator_and_support_checks_are_not_filtered(self):
        cfg,manifest,rows,vessel,value=rows_fixture()
        with self.assertRaisesRegex(ValueError,'Missing attempted draw'):
            reference.check_rows(cfg,manifest,rows[:3],vessel,value)
        weights=[r['log_importance_weight'] if r['hard_valid'] else -math.inf for r in rows]
        estimate=reference.moments(weights)
        self.assertEqual(estimate['draws'],4);self.assertEqual(estimate['nonzero'],3)


if __name__=='__main__':unittest.main()
