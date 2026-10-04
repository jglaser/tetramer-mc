"""Deterministic toy world-density controls; no physical or random draws."""
import copy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.special import logsumexp

import physical_native_class_line_vessel as reference
from prepare_native_class_line_physical_toy import prepare
from test_native_class_line_physical_reference import compact
from test_physical_latent_guide import fixture as chart_fixture, independent_pose


def save(path, value):
    Path(path).write_text(json.dumps(value, allow_nan=False)+'\n')


class PhysicalNativeClassVesselTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        # This helper writes analytic toy inputs only. No producer is invoked.
        plan = prepare(self.root/'inputs')
        job = next(j for j in plan['jobs'] if j['activity'] == .3 and j['method'] == 'class')
        self.config = reference.read(job['config'])
        self.region = reference.read(job['region'])
        self.guide = reference.read(job['guide'])
        self.shape = reference.read(self.root/'inputs/common/shape.json')
        self.compiled = reference.read(self.root/'inputs/common/compiled-native.json')
        self.shape_hash = self.region['shape_sha256']
        self.region['capture_radius'] = self.guide['capture_radius'] = 170.
        self.config['capture_radius'] = 273.
        self.region_hash = 'b'*64
        self.guide['region_sha256'] = self.region_hash
        self.lower = np.eye(6)

    def build(self):
        observer = reference.line.observer_from_compiled_for_synthetic(self.compiled)
        return reference.PhysicalNativeClassLineGuide(self.region, self.guide, self.config,
            self.shape, observer, region_sha256=self.region_hash,
            expected_shape_sha256=self.shape_hash)

    def at(self, guide, u):
        return guide.evaluate(independent_pose(u, self.region, self.lower)[0])

    def frozen_inputs(self):
        paths = [self.root/name for name in ('region.json', 'guide.json', 'compiled.json')]
        save(paths[0], self.region); save(paths[2], self.compiled)
        self.guide['region_sha256'] = reference.sha(paths[0])
        self.guide['compiled_native'] = dict(path=str(paths[2]), sha256=reference.sha(paths[2]))
        save(paths[1], self.guide)
        return paths

    def from_files(self, paths):
        return reference.PhysicalNativeClassLineGuide.from_files(paths[0], paths[1],
            compiled_path=paths[2],
            observer=reference.line.observer_from_compiled_for_synthetic(self.compiled),
            vessel_config=self.config, shape=self.shape, expected_shape_sha256=self.shape_hash)

    def rows(self):
        guide = self.build()
        manifest = dict(schema=7, outer_mixture_schema=reference.SCHEMA,
            latent_guide_schema=reference.line.SCHEMA, samples=5, shape_sha256=self.shape_hash,
            pose_proposal_schema=1, covariance_scale=1., proposal_anchor_index=None,
            physical_fixed_neighbor_count=1, uniform_probability=.5,
            latent_gaussian_component_count=2, latent_defensive_uniform_probability=.5,
            cloud_replicates=2, activity=0., **{'lambda': 1.})
        vessel = reference.vessel_reference.VesselDensity(self.config, manifest,
                                                         self.region['gaussian_chart'])
        us = [[2.5, 0., 0., 0., 0., 0.], [2.5, 2.5, 2.5, 0., 0., 0.],
              [5., 5., 0., 0., 0., 0.], [0.]*6]
        poses = [independent_pose(u, self.region, self.lower)[0] for u in us]
        poses.append(dict(position=[3., 0., 0.], orientation=[0., 1., 0., 0.]))
        densities = guide.evaluate_many(poses)
        vessel_logs, _ = vessel.evaluate(poses)
        mixed = reference.original.half_mixture_log_density(
            vessel_logs, [d.log_physical_density for d in densities])
        qs = reference.vessel_reference.registration(poses, self.config['metadata'])
        rows = []
        for i, (pose, density) in enumerate(zip(poses, densities)):
            valid, latent = i != 3, i == 0
            trace = compact(guide.recon, np.asarray(density.latent)) if density.latent is not None else None
            encode = lambda v: v if v is None or math.isfinite(v) else None
            rows.append(dict(draw=i, pose=pose, outer_branch='latent' if latent else 'vessel',
                proposal=None if latent else {},
                latent_proposal=dict(latent=us[i], latent_radius=float(np.linalg.norm(us[i])),
                    gaussian_component=None,
                    native_class_line_draw=dict(conditional=False, original_latent=us[i])) if latent else None,
                latent_density=dict(latent=density.latent, in_reference_ball=density.in_reference_ball,
                    coordinate_chart_seam=density.latent is None,
                    log_latent_density=encode(density.log_latent_density),
                    log_physical_jacobian=density.log_physical_jacobian,
                    structural_zero=density.structural_zero, native_class_line_density=trace),
                log_vessel_proposal_density=float(vessel_logs[i]),
                log_latent_physical_density=encode(density.log_physical_density),
                log_proposal_density=float(mixed[i]), capture_valid=True, wall_valid=True, hard_valid=valid,
                log_hard_weight=-float(mixed[i]) if valid else None,
                log_importance_weight=-float(mixed[i]) if valid else None,
                q=float(qs[i]) if valid else None, region=None, depletion_contact=None,
                clouds=[dict(log_weight=0., overlap_points=0, lower_volume=0., uncertain_volume=0.)
                        for _ in range(2)] if valid else []))
        return guide, manifest, rows, vessel

    def test_world_roundtrip_full_jacobian_and_quaternion_sign(self):
        chart, _, self.lower = chart_fixture(coupled=True)
        self.region['gaussian_chart'] = chart['gaussian_chart']
        self.region['gaussian_chart']['shape_sha256'] = self.shape_hash
        self.region['fixed_neighbor'] = chart['fixed_neighbor']
        fixed = [chart['fixed_neighbor']]
        self.region['physical_fixed_neighbors'] = copy.deepcopy(fixed)
        for value in (self.config, self.guide, self.compiled):
            value['fixed_poses'] = copy.deepcopy(fixed)
        self.guide['conditional_probability'] = 0.
        guide = self.build()
        for u in ([.2, -.7, 1.2, .3, -.6, .8], [4.3, .2, .1, .3, -.2, .1]):
            pose, logj = independent_pose(u, self.region, self.lower)
            density = guide.evaluate(pose)
            np.testing.assert_allclose(density.latent, u, atol=2e-13, rtol=0)
            self.assertAlmostEqual(density.log_physical_jacobian, logj, places=12)
            self.assertAlmostEqual(density.log_physical_density+logj, density.log_latent_density, places=12)
            gaussian = logsumexp(guide.recon.gaussian_logs(np.asarray(u)))
            uniform = math.log(.5)-guide.recon.logvolume if np.linalg.norm(u) <= 4 else -math.inf
            expected = np.logaddexp(uniform, math.log(.5)+gaussian)
            self.assertAlmostEqual(density.log_physical_density, float(expected)-logj, places=12)
            opposite = copy.deepcopy(pose); opposite['orientation'] = [-q for q in pose['orientation']]
            self.assertAlmostEqual(guide.evaluate(opposite).log_physical_density, density.log_physical_density, places=12)

    def test_source_capture_conditions_guide_without_restricting_vessel(self):
        self.lower[0, 0] = 100.
        self.region['gaussian_chart']['covariances'] = [(self.lower@self.lower.T).tolist()]
        self.guide['raw_translation_axes'] = [0]
        guide = self.build(); u = [1.8, 0., 0., 0., 0., 0.]
        pose, _ = independent_pose(u, self.region, self.lower)
        self.assertGreater(pose['position'][0], 170.)
        self.assertLess(pose['position'][0], 273.)
        density = guide.evaluate(pose)
        self.assertTrue(math.isfinite(density.log_physical_density))
        self.assertEqual(guide.source_config['capture_radius'], 170.)
        self.config['capture_radius'] = 500.
        self.assertEqual(self.build().evaluate(pose).log_physical_density, density.log_physical_density)
        self.region['capture_radius'] = self.guide['capture_radius'] = 273.
        self.assertGreater(self.build().evaluate(pose).log_physical_density, density.log_physical_density)

    def test_class_hard_free_and_unconditional_fallbacks_are_retained(self):
        guide = self.build()
        interior = self.at(guide, [2.5, 0., 0., 0., 0., 0.])
        branches = [b for a in interior.reconstruction['axes'] for c in a['components'] for b in c['channels']]
        self.assertIn('class', {b['fallback'] for b in branches})
        self.assertIn('hard_free', {b['fallback'] for b in branches})
        exterior = self.at(guide, [5., 5., 0., 0., 0., 0.])
        tails = [b for a in exterior.reconstruction['axes'] for c in a['components'] for b in c['channels']]
        self.assertEqual({b['fallback'] for b in tails}, {'unconditional'})
        self.assertTrue(all(b['query_coordinate_allowed'] and b['multiplier'] == 1. for b in tails))
        self.assertFalse(exterior.structural_zero)
        self.assertAlmostEqual(exterior.log_latent_density,
            math.log(.5)+float(logsumexp(guide.recon.gaussian_logs(np.asarray(exterior.latent)))), places=12)

    def test_exterior_zero_and_unconditional_positive_tail_are_distinct(self):
        guide = self.build()
        zero = self.at(guide, [2.5, 2.5, 2.5, 0., 0., 0.])
        tail = self.at(guide, [5., 5., 0., 0., 0., 0.])
        self.assertFalse(zero.in_reference_ball); self.assertTrue(zero.structural_zero)
        self.assertEqual(zero.log_physical_density, -math.inf)
        self.assertFalse(reference.positive_components(guide.recon, zero.reconstruction).any())
        self.assertFalse(tail.in_reference_ball); self.assertFalse(tail.structural_zero)
        self.assertTrue(reference.positive_components(guide.recon, tail.reconstruction).all())

    def test_exact_seam_and_near_seam_pure_uniform_have_different_coordinates(self):
        self.guide['defensive_uniform_shell_probability'] = 1.
        guide = self.build()
        seam = guide.evaluate(dict(position=[3., 0., 0.], orientation=[0., 1., 0., 0.]))
        self.assertIsNone(seam.latent); self.assertIsNone(seam.log_physical_jacobian)
        self.assertTrue(seam.structural_zero)
        for scalar in (1e-12, 1e-100):
            density = guide.evaluate(dict(position=[3., 0., 0.], orientation=[scalar, 1., 0., 0.]))
            self.assertIsNotNone(density.latent)
            self.assertTrue(math.isfinite(density.log_physical_jacobian))
            self.assertTrue(density.structural_zero)
            self.assertEqual(density.log_physical_density, -math.inf)

    def test_underflow_keeps_positive_log_density_and_active_overflow_fails(self):
        self.guide['conditional_probability'] = 0.
        guide = self.build()
        tail = self.at(guide, [100., 0., 0., 0., 0., 0.])
        self.assertFalse(tail.structural_zero); self.assertTrue(math.isfinite(tail.log_physical_density))
        self.assertEqual(math.exp(tail.log_physical_density), 0.)
        with self.assertRaisesRegex(ValueError, 'Unrepresentable positive Gaussian'):
            guide.evaluate(dict(position=[1e160, 0., 0.], orientation=[1., 0., 0., 0.]))
        # A finite defensive uniform term cannot hide one invalid active component.
        original = guide.recon.gaussian_logs
        def corrupted(u):
            values = original(u).copy(); values[0] = -math.inf
            return values
        with patch.object(guide.recon, 'gaussian_logs', side_effect=corrupted):
            with self.assertRaisesRegex(ValueError, 'Unrepresentable positive Gaussian'):
                self.at(guide, [2.5, 0., 0., 0., 0., 0.])

    def test_complete_compact_trace_rejects_axis_channel_and_density_mutations(self):
        guide = self.build(); density = self.at(guide, [2.5, 0., 0., 0., 0., 0.])
        actual = compact(guide.recon, np.asarray(density.latent))
        self.assertLess(reference.audit_trace(actual, density, guide), 1e-10)
        for mutation in ('axis', 'channel', 'multiplier', 'baseline', 'interval'):
            with self.subTest(mutation=mutation):
                bad = copy.deepcopy(actual)
                if mutation == 'axis': bad['axes'].pop()
                elif mutation == 'channel': bad['axes'][0]['channels'][1]['probability'] += .1
                elif mutation == 'multiplier': bad['component_mixture_multipliers'][0] += 1.
                elif mutation == 'baseline': bad['baseline_log_density'] += 1.
                else: bad['axes'][0]['hard_free_intervals'][0]['lower'] += .1
                with self.assertRaises(ValueError): reference.audit_trace(bad, density, guide)

    def test_constructor_rejects_source_shape_scaffold_and_physical_drift(self):
        for container, key, replacement in (
                ('guide', 'shape_sha256', 'f'*64), ('guide', 'region_sha256', 'f'*64),
                ('guide', 'capture_radius', 273.), ('guide', 'depletant_radius', .6),
                ('config', 'fixed_poses', [dict(position=[1., 0., 0.], orientation=[1., 0., 0., 0.])]),
                ('compiled', 'fixed_poses', [])):
            with self.subTest(container=container, key=key):
                target = getattr(self, container); old = target[key]; target[key] = replacement
                try:
                    with self.assertRaises(ValueError): self.build()
                finally: target[key] = old

    def test_bound_compiled_hash_shape_and_scaffold_mutations_fail(self):
        paths = self.frozen_inputs(); self.assertEqual(len(self.from_files(paths).sources), 3)
        original = copy.deepcopy(self.compiled)
        original_hash = reference.sha(paths[2])
        for mutation in ('bytes', 'shape', 'scaffold'):
            with self.subTest(mutation=mutation):
                changed = copy.deepcopy(original)
                if mutation == 'bytes': changed['source_definition_sha256'] = 'e'*64
                elif mutation == 'shape': changed['source_input_sha256']['tetramer-shape.json'] = 'e'*64
                else: changed['fixed_poses'] = []
                save(paths[2], changed)
                self.guide['compiled_native']['sha256'] = reference.sha(paths[2]) if mutation != 'bytes' else original_hash
                save(paths[1], self.guide)
                with self.assertRaises(ValueError): self.from_files(paths)
        save(paths[2], original)

    def test_full_outer_mixture_jacobian_once_seam_and_invalid_zero(self):
        guide, manifest, rows, vessel = self.rows()
        result = reference.check_rows(self.config, manifest, rows, vessel, guide)
        self.assertEqual(result['checked_attempts'], 5)
        self.assertEqual(result['structural_zero_queries'], 2)
        self.assertEqual(result['exact_chart_seams'], 1)
        self.assertEqual(result['valid_outside_R4'], 3)
        for index in (1, 4):
            self.assertAlmostEqual(rows[index]['log_proposal_density'],
                rows[index]['log_vessel_proposal_density']-math.log(2.), places=12)
        bad = copy.deepcopy(rows); bad[0]['log_hard_weight'] += bad[0]['latent_density']['log_physical_jacobian']
        with self.assertRaisesRegex(ValueError, 'no extra J'):
            reference.check_rows(self.config, manifest, bad, vessel, guide)
        bad = copy.deepcopy(rows); bad[3]['log_importance_weight'] = 0.
        with self.assertRaisesRegex(ValueError, 'explicit zero'):
            reference.check_rows(self.config, manifest, bad, vessel, guide)
        bad = copy.deepcopy(rows); bad[1]['latent_density']['structural_zero'] = False
        with self.assertRaisesRegex(ValueError, 'Structural-zero'):
            reference.check_rows(self.config, manifest, bad, vessel, guide)

    def test_tiny_positive_channel_cannot_be_lost_in_categorical_draw(self):
        self.guide['class_channels']=[dict(**{'class':'hard_free'},probability=1.),
                                     dict(**{'class':'native'},probability=1e-20)]
        with self.assertRaisesRegex(ValueError,'categorical increment'):
            self.build()
        self.guide['class_channels'].reverse()
        with self.assertRaisesRegex(ValueError,'categorical increment'):
            self.build()

    def test_uniform_floor_does_not_mask_lost_positive_component(self):
        guide=self.build();u=np.array([2.5,0.,0.,0.,0.,0.])
        result=guide.recon.density(u)
        active=reference.positive_components(guide.recon,result)
        self.assertTrue(active.any())
        result['component_multipliers'][int(np.flatnonzero(active)[0])]=0.
        with patch.object(guide.recon,'density',return_value=result):
            with self.assertRaisesRegex(ValueError,'multiplier disagrees'):
                self.at(guide,u)

    def test_all_attempt_inventory_and_outer_density_tampering_fail(self):
        guide, manifest, rows, vessel = self.rows()
        with self.assertRaisesRegex(ValueError, 'Missing attempted draw'):
            reference.check_rows(self.config, manifest, rows[:-1], vessel, guide)
        for field in ('log_proposal_density', 'log_latent_physical_density'):
            bad = copy.deepcopy(rows); bad[0][field] += .1
            with self.subTest(field=field), self.assertRaises(ValueError):
                reference.check_rows(self.config, manifest, bad, vessel, guide)
        estimate = reference.moments([r['log_importance_weight'] if r['hard_valid'] else -math.inf for r in rows])
        self.assertEqual(estimate['draws'], 5); self.assertEqual(estimate['nonzero'], 4)


if __name__ == '__main__':
    unittest.main()
