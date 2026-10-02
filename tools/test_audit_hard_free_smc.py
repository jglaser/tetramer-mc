"""Adversarial audit controls without new physical or SMC sampling."""
import copy
import math
import unittest
from unittest.mock import patch
import numpy as np
import audit_hard_free_smc as audit
from test_physical_hard_free_line_vessel import setup, trace, independent_pose, REGION, SHAPE


class GuidedSmcAuditTests(unittest.TestCase):
    def test_options_require_complete_bridge_and_valid_unconditional_budgets(self):
        opts = dict(bridge='proposal_density', initial_reference_region=None,
            initial_current_probability=1., initial_draws=1024, population=32,
            cloud_replicates=2, sweeps_per_stage=4, seed=42, lambda_ratio=128.,
            schedule=[0., .25, .5, 1.])
        audit.validate_options(opts)
        for key, value in [('schedule', [0., .25, .5]), ('schedule', [.1, 1.]),
            ('schedule', [0., .5, .5, 1.]), ('schedule', [0., float('nan'), 1.]),
            ('initial_draws', 0), ('initial_draws', True), ('population', -1),
            ('cloud_replicates', 0), ('sweeps_per_stage', -1), ('seed', 2**64),
            ('lambda_ratio', 0.), ('bridge', 'physical_activity'),
            ('initial_reference_region', 'unfiltered-reference.json')]:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                audit.validate_options(dict(opts, **{key:value}))

    def test_impossible_generation_labels_fail(self):
        guide = dict(defensive_uniform_shell_probability=.5, conditional_probability=1., gaussian_components=[{}])
        audit.validate_generation_branch(None, True, dict(conditional=False), guide)
        audit.validate_generation_branch(0, False, dict(conditional=True), guide)
        for component, inside, conditional, change in [(None,False,False,{}), (None,True,True,{}),
            (0,True,False,{}), (0,True,True,dict(defensive_uniform_shell_probability=1.)),
            (0,True,True,dict(conditional_probability=0.)), (1,True,True,{})]:
            with self.subTest(component=component, change=change), self.assertRaises(ValueError):
                audit.validate_generation_branch(component, inside, dict(conditional=conditional), dict(guide, **change))

    def test_zero_hit_summary_preserves_zero_and_empty_genealogy(self):
        ancestry = dict(initial_ancestor_counts={}, distinct_initial_ancestors=0,
            initial_family_ESS=0., largest_initial_family_fraction=None)
        summary = dict(zero_estimate=True, log_Z=None, Z=0., terminal_particles=[], ancestry=ancestry)
        first = dict(zero_estimate=True, log_Z=None, Z=0., particles=[], parents=[], initial_draws=256,
            initial_hits=0, activity=0.)
        audit.validate_zero_summary(summary, first, 256)
        for where, key, value in [('summary','Z',1.), ('first','parents',[0]),
            ('first','initial_draws',255), ('first','log_Z',0.), ('first','activity',1.)]:
            a,b = copy.deepcopy(summary),copy.deepcopy(first)
            (a if where=='summary' else b)[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): audit.validate_zero_summary(a,b,256)

    def test_positive_summary_cannot_change_linear_mass_or_initial_ess(self):
        base = dict(log_Z=2., initial_weight_ESS=51, linear_Z_representable=True, Z=math.exp(2.))
        audit.validate_positive_summary(base,2.,51)
        for key, value in [('Z',1.), ('Z',None), ('linear_Z_representable',False), ('initial_weight_ESS',50)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                audit.validate_positive_summary(dict(base,**{key:value}),2.,51)
        for log_z in [-1000.,1000.]:
            audit.validate_positive_summary(dict(base, log_Z=log_z, linear_Z_representable=False, Z=None),log_z,51)

    def test_independent_density_and_stale_pose_cache(self):
        value, region, guide, config, shape, lower = setup()
        config['capture_radius'] = region['capture_radius']
        proposal = audit.Proposal(region, guide, config, shape, REGION, SHAPE)
        pose, _ = independent_pose([3.,0.,0.,0.,0.,0.], region, lower)
        current, density, _, hard = proposal.details(pose)
        self.assertTrue(hard)
        particle = dict(pose=pose, latent=current['latent'].tolist(), initial_ancestor=0,
            guide_density_cache=dict(pose=copy.deepcopy(pose), log_g=density.log_physical_density))
        proposal.check_cache(particle)
        self.assertEqual(proposal.evaluations,1)
        for kind in ['pose','density','latent']:
            bad = copy.deepcopy(particle)
            if kind=='pose': bad['guide_density_cache']['pose']['position'][0] += .1
            elif kind=='density': bad['guide_density_cache']['log_g'] += .1
            else: bad['latent'][0] += .1
            with self.subTest(kind=kind), self.assertRaises(ValueError): proposal.check_cache(bad)
        record = dict(latent=density.latent, in_reference_ball=density.in_reference_ball,
            structural_zero=False, log_latent_density=density.log_latent_density,
            log_physical_jacobian=density.log_physical_jacobian,
            log_physical_density=density.log_physical_density, hard_free_line_density=trace(value,density))
        proposal.check_density_record(pose,record)
        bad = copy.deepcopy(record); bad['log_physical_density'] += density.log_physical_jacobian
        with self.assertRaises(ValueError): proposal.check_density_record(pose,bad)
        bad = copy.deepcopy(record); bad['hard_free_line_density']['axes'].pop()
        with self.assertRaises(ValueError): proposal.check_density_record(pose,bad)

    def test_sphere_cloud_bounds_reject_impossible_certificates(self):
        _, region, guide, config, shape, _ = setup()
        proposal = audit.Proposal(region,guide,config,shape,REGION,SHAPE)
        pose = copy.deepcopy(config['fixed_poses'][0])
        pose['position'][0] += 2.1  # core separation >2; exclusion separation <2.4
        proposal.check_sphere_cloud_bounds(pose,[dict(lower_volume=0.,upper_volume=1.)],.1)
        self.assertEqual(proposal.analytic_sphere_cloud_bounds_checked,1)
        for cloud in [dict(lower_volume=1.,upper_volume=1.),dict(lower_volume=0.,upper_volume=0.)]:
            with self.assertRaisesRegex(ValueError,'analytic sphere lens'):
                proposal.check_sphere_cloud_bounds(pose,[cloud],.1)
        proposal.check_sphere_cloud_bounds(pose,[dict(lower_volume=0.,upper_volume=0.)],0.)

    def test_split_mode_requires_certificates_and_prunes_only_after_history(self):
        value, region, guide, config, shape, lower = setup()
        config['capture_radius'] = region['capture_radius']
        proposal = audit.Proposal(region,guide,config,shape,REGION,SHAPE,conditioned_density=True)
        pose,_ = independent_pose([3.,0.,0.,0.,0.,0.],region,lower)
        current,density,_,hard = proposal.details(pose)
        self.assertTrue(hard)
        with self.assertRaisesRegex(ValueError,'without its complete numerical audit'):
            proposal.evaluate(pose)
        record = dict(latent=density.latent,in_reference_ball=density.in_reference_ball,
            structural_zero=False,log_latent_density=density.log_latent_density,
            log_physical_jacobian=density.log_physical_jacobian,
            log_physical_density=density.log_physical_density,hard_free_line_density=trace(value,density))
        checked = proposal.check_density_record(pose,record)
        self.assertAlmostEqual(proposal.evaluate(pose)[2],checked.log_physical_density,places=12)
        self.assertEqual(proposal.conditioning_queries,1)
        with patch.object(proposal,'details',return_value=(dict(current,inside=False),density,True,True)):
            with self.assertRaisesRegex(ValueError,'Density support certificate differs'):
                proposal.check_density_record(pose,record)
        particle = dict(pose=pose,latent=current['latent'].tolist(),
            guide_density_cache=dict(pose=copy.deepcopy(pose),log_g=density.log_physical_density))
        proposal.cache.clear()  # Recomputed geometry does not lose the checked scalar.
        proposal.check_cache(particle)
        proposal.retain_certificates([particle,particle])
        self.assertEqual(len(proposal.certified_log_g),1)
        key = audit.old.pose_key(pose)
        proposal.certified_log_g[key] += .001
        with self.assertRaisesRegex(ValueError,'Duplicate pose has inconsistent certified density'):
            proposal.check_density_record(pose,record)
        proposal.certified_log_g[key] = checked.log_physical_density
        bad = copy.deepcopy(particle);bad['guide_density_cache']['log_g'] += .001
        with self.assertRaisesRegex(ValueError,'Retained density cache differs'): proposal.check_cache(bad)
        proposal.retain_certificates([])
        with self.assertRaisesRegex(ValueError,'lacks its complete numerical audit'):proposal.check_cache(particle)

    def test_failure_context_is_retained_without_changing_exception(self):
        def fail(*args,context,**kwargs):
            context.update(phase='initialization',draw=17)
            raise ValueError('diagnostic failure')
        with patch.object(audit,'_audit',side_effect=fail):
            with self.assertRaisesRegex(ValueError,'diagnostic failure') as raised:
                audit.audit('unused','unused')
        self.assertIn('"draw": 17',raised.exception.__notes__[0])


if __name__ == '__main__': unittest.main()
