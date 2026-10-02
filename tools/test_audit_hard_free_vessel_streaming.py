"""Deterministic batch/accounting controls; no normalizer or Poisson jobs."""
import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import audit_hard_free_vessel_streaming as stream
from analyze_basin_normalizers import audit_wall_domain
from test_physical_hard_free_line_vessel import setup, trace
from scipy.special import logsumexp


class AttemptTests(unittest.TestCase):
    def test_exact_once_order_and_bounded_batches(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); rows = root/'rows'; attempts = root/'attempts'
            values = [dict(draw=i, payload=i*i) for i in range(11)]
            rows.write_text(''.join(json.dumps(r)+'\n' for r in values))
            journal = [dict(draw=i, state='begin') for i in range(11)]
            attempts.write_text(''.join(json.dumps(r)+'\n' for r in journal))
            for batch_size in [1, 4, 50]:
                batches = list(stream.attempt_batches(rows, attempts, 11, batch_size))
                self.assertEqual([r for batch in batches for r in batch], values)
                self.assertLessEqual(max(map(len, batches)), batch_size)
            for invalid in [journal[:-1], journal+[dict(draw=11, state='begin')],
                            [dict(draw=0, state='done')]+journal[1:],
                            [dict(draw=False, state='begin')]+journal[1:]]:
                attempts.write_text(''.join(json.dumps(r)+'\n' for r in invalid))
                with self.assertRaises(ValueError): list(stream.attempt_batches(rows, attempts, 11, 4))
            attempts.write_text(''.join(json.dumps(r)+'\n' for r in journal))
            for invalid in [values[:-1], values+[dict(draw=11)], values[1:]+values[:1],
                            [dict(draw=False)]+values[1:], [values[0]]*11]:
                rows.write_text(''.join(json.dumps(r)+'\n' for r in invalid))
                with self.assertRaises(ValueError): list(stream.attempt_batches(rows, attempts, 11, 4))


def synthetic_rows():
    return [dict(draw=i, capture_valid=i % 4 != 0, wall_valid=i % 3 != 0,
                 hard_valid=i % 4 != 0 and i % 3 != 0,
                 outer_branch='vessel' if i % 2 else 'latent',
                 clouds=[dict(raw_points=i), dict(raw_points=i+1)] if i % 4 != 0 and i % 3 != 0 else [],
                 nested=dict(original=i)) for i in range(13)]


class BatchTests(unittest.TestCase):
    def test_original_helpers_receive_all_unchanged_poses_and_global_counts_are_checked(self):
        rows = synthetic_rows(); saved = copy.deepcopy(rows)
        manifest = dict(samples=len(rows)); summary = dict(samples=len(rows), **stream.row_counts(rows))
        observed = []
        def density(config, local, batch, vessel, guide):
            self.assertEqual([r['draw'] for r in batch], list(range(local['samples'])))
            observed.extend(r['nested']['original'] for r in batch)
            return dict(checked_attempts=len(batch), valid_outside_R4=1, structural_zero_queries=0,
                        exact_chart_seams=0, valid_outside_source_capture=1, maximum_log_density_error=.1,
                        maximum_interval_endpoint_error=.2, maximum_inverse_CDF_error=.3,
                        source_capture=2., vessel_capture=4., scope='fixed scope',
                        outer_branches=dict(stream.Counter(r['outer_branch'] for r in batch)))
        def generation(config, local, batch, vessel):
            return dict(checked_vessel_generation_rows=sum(r['outer_branch'] == 'vessel' for r in batch))
        def primitive(local, summary, batch):
            self.assertEqual(local['samples'], len(batch))
            self.assertEqual(summary, dict(samples=len(batch), **stream.row_counts(batch)))
            return stream.row_counts(batch)
        checks = stream.BatchChecks()
        with patch.object(stream.reference, 'check_rows', side_effect=density), \
             patch.object(stream.reference.vessel_reference, 'check_generation_metadata', side_effect=generation), \
             patch.object(stream.reference.vessel_reference, 'check_cloud_envelopes_and_counts', side_effect=primitive):
            for start in range(0, len(rows), 4): checks.check({}, manifest, rows[start:start+4], None, None)
        self.assertEqual(observed, list(range(13))); self.assertEqual(rows, saved)
        checks.finish(manifest, summary)
        self.assertEqual(checks.batches, 4); self.assertEqual(checks.peak_rows, 4)
        self.assertEqual(checks.density['maximum_log_density_error'], .1)
        for key in stream.row_counts(rows):
            with self.subTest(key=key), self.assertRaises(ValueError):
                checks.finish(manifest, dict(summary, **{key: summary[key]+1}))
        with self.assertRaises(ValueError): checks.finish(dict(samples=14), dict(summary, samples=14))


def wall_fixture(root):
    (root/'provenance').mkdir()
    shape = dict(atoms=[dict(center=[-.2, 0., 0.], radius=.3), dict(center=[.5, 0., 0.], radius=.2)])
    fixed = dict(position=[1., -1., .4], orientation=[1., 0., 0., 0.])
    config = dict(capture_center=[0., 0., 0.], capture_radius=6., fixed_poses=[fixed])
    text = 'synthetic wall arithmetic fixture\n'
    bundle = dict(files={'fixture.txt': dict(text=text, sha256=hashlib.sha256(text.encode()).hexdigest())})
    stream.write(root/'config.json', config)
    stream.write(root/'provenance/shape.json', shape)
    stream.write(root/'provenance/source-bundle.json', bundle)
    manifest = dict(schema=4, pose_proposal_schema=1, bath_wall_permeable=True,
                    source_bundle_sha256=stream.sha(root/'provenance/source-bundle.json'),
                    shape_bound=.7, atomic_wall=dict(center=fixed['position'], radius=3.))
    return shape, config, bundle, manifest


class WallAndPartitionTests(unittest.TestCase):
    def test_nonconstant_sphere_union_wall_matches_original_direct_oracle(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); shape, config, bundle, manifest = wall_fixture(root)
            oracle = stream.WallOracle(config, manifest, shape, bundle)
            rows = []
            for dx, orientation, valid in [(0., [1., 0., 0., 0.], True),
                                           (2.2, [1., 0., 0., 0.], True),
                                           (2.31, [1., 0., 0., 0.], False),
                                           (2.6, [math.sqrt(.5), 0., 0., math.sqrt(.5)], True),
                                           (2.9, [math.sqrt(.5), 0., 0., math.sqrt(.5)], False)]:
                pose = dict(position=[1.+dx, -1., .4], orientation=orientation)
                row = dict(pose=pose, wall_valid=valid, capture_valid=True, hard_valid=valid,
                           clouds=[], log_importance_weight=0. if valid else None,
                           log_hard_weight=0. if valid else None, q=1. if valid else None,
                           region='arbitrary' if valid else None, depletion_contact=False if valid else None)
                oracle.check(row); rows.append(row)
            old = audit_wall_domain(root, manifest, rows, dict(wall_rejected=2))
            result = oracle.result()
            for name in ['checked_poses', 'wall_rejected_inside_capture', 'atomic_wall', 'bath_wall_permeable']:
                self.assertEqual(old[name], result[name])
            with self.assertRaises(ValueError): oracle.check(dict(rows[0], wall_valid=False))
            with self.assertRaises(ValueError): oracle.check(dict(rows[0], capture_valid=False))
            with self.assertRaises(ValueError): oracle.check(dict(rows[-1], log_hard_weight=0.))
            with self.assertRaises(ValueError):
                stream.WallOracle(dict(config, capture_radius=3.), manifest, shape, bundle)

    def test_exterior_mass_and_explicit_invalid_zeros_partition_every_pose(self):
        for valid, contact, inside, zero in [(True, True, True, False), (True, False, False, True),
                                             (True, True, False, False), (False, None, True, False)]:
            row = dict(hard_valid=valid, capture_valid=valid, wall_valid=valid,
                       depletion_contact=contact, latent_density=dict(in_reference_ball=inside, structural_zero=zero))
            geometry = dict(core_disjoint=valid if valid else None, exclusion_contact=contact)
            classes = stream.memberships(row, geometry)
            self.assertEqual(set(classes), set(stream.CLASSES))
            self.assertEqual(classes['total'], valid)
            self.assertEqual(classes['inside_R4']+classes['outside_R4'], int(valid))
            self.assertEqual(classes['exclusion_contact']+classes['unbound'], int(valid))
            if not valid: self.assertFalse(any(classes.values()))
        with self.assertRaises(ValueError):
            stream.memberships(dict(row, capture_valid=True, wall_valid=True, hard_valid=True),
                               dict(core_disjoint=False, exclusion_contact=True))


def complete_fixture(root):
    """Hand-authored poses/counts, with the existing independent guide scorer.

    The binary is an inert byte fixture used only to exercise hash/bundle checks;
    it is never executed. This is not a new physical or random reference run.
    """
    root.mkdir(); provenance = root/'provenance'; provenance.mkdir()
    _, region, guide, config, shape, _ = setup()
    stream.write(provenance/'shape.json', shape); shape_hash = stream.sha(provenance/'shape.json')
    region['shape_sha256'] = region['gaussian_chart']['shape_sha256'] = shape_hash
    stream.write(provenance/'latent-region.json', region)
    guide['region_sha256'] = stream.sha(provenance/'latent-region.json')
    stream.write(provenance/'latent-guide.json', guide)
    config.update(shape=str(provenance/'shape.json'), poisson_lambda_ratio=8., reservoir_density=0.)
    stream.write(root/'config.json', config); stream.write(provenance/'input-config.json', config)
    stream.write(provenance/'model.json', region['gaussian_chart'])
    text = 'deterministic synthetic audit fixture; never executed\n'
    bundle = dict(files={'fixture.txt': dict(text=text, sha256=hashlib.sha256(text.encode()).hexdigest())})
    stream.write(provenance/'source-bundle.json', bundle)
    binary = root/'inert-binary'; binary.write_bytes(b'not executable\n'+(provenance/'source-bundle.json').read_bytes())
    manifest = dict(schema=6, outer_mixture_schema=stream.reference.SCHEMA,
        outer_vessel_probability=.5, latent_guide_schema=guide['schema'], samples=7,
        shape_sha256=shape_hash, pose_proposal_schema=1, covariance_scale=1., proposal_anchor_index=None,
        physical_fixed_neighbor_count=1, uniform_probability=.5, latent_gaussian_component_count=1,
        latent_defensive_uniform_probability=.5, cloud_replicates=2, activity=0., **{'lambda':1.},
        shape_bound=1., atomic_wall=dict(center=[0.,0.,0.], radius=10.), bath_wall_permeable=True,
        density_measure='Lebesgue center volume times normalized SO(3) Haar measure',
        latent_reference_ball_is_target_restriction=False,
        latent_source_capture=dict(center=region['capture_center'], radius=region['capture_radius'], restricts_target=False),
        executable_sha256=stream.sha(binary))
    for name, key in [('input-config.json','config_sha256'), ('model.json','model_sha256'),
                      ('source-bundle.json','source_bundle_sha256'), ('latent-region.json','latent_region_sha256'),
                      ('latent-guide.json','latent_guide_sha256')]: manifest[key] = stream.sha(provenance/name)
    physical = stream.reference.PhysicalHardFreeLineGuide.from_files(provenance/'latent-region.json',
        provenance/'latent-guide.json', vessel_config=config, shape=shape, expected_shape_sha256=shape_hash)
    vessel = stream.reference.vessel_reference.VesselDensity(config, manifest, region['gaussian_chart'])
    positions = [[3.,0.,0.], [2.5,2.5,2.5], [5.,0.,0.], [0.,0.,0.], [2.2,0.,0.], [20.,0.,0.], [300.,0.,0.]]
    poses = [dict(position=p, orientation=[1.,0.,0.,0.]) for p in positions]
    vessel_logs, vg = vessel.evaluate(poses); initial = vessel.evaluate([config['initial_pose']])[1]
    densities = physical.evaluate_many(poses); rows = []
    for i, (pose, d) in enumerate(zip(poses, densities)):
        radius = math.hypot(*pose['position']); capture = radius <= config['capture_radius']
        wall = radius <= 9.; valid = wall and radius >= 2.; contact = radius < 2.4
        logq = float(np.logaddexp(vessel_logs[i], d.log_physical_density)-math.log(2))
        q = stream.reference.regional.native.native_q(config['metadata'], pose) if valid else None
        band = ('native_core' if q <= .8 else 'native_shell' if q <= 1 else 'shoulder' if q < 2
                else 'intermediate' if q < 5 else 'distant') if valid else None
        overlap = math.pi*(4.8+radius)*(2.4-radius)**2/12 if valid and contact else 0.
        cloud = dict(log_weight=0., raw_points=0, overlap_points=0, lower_volume=overlap, upper_volume=overlap,
                     uncertain_volume=0., retained_cells=0, created_cells=int(overlap > 0), certified_cells=int(overlap > 0))
        is_latent = i == 0
        proposal = None if is_latent else dict(moving_index=0, anchor_index=1, null_reason=None,
            candidate=copy.deepcopy(pose), component_index=None if capture else 0,
            branch='uniform' if capture else 'learned', new_log_density=float(vg['anchor_log_densities'][0,i]),
            old_log_density=float(initial['anchor_log_densities'][0,0]),
            log_reverse_forward=float(initial['anchor_log_densities'][0,0]-vg['anchor_log_densities'][0,i]))
        rows.append(dict(draw=i, pose=pose, outer_branch='latent' if is_latent else 'vessel', proposal=proposal,
            latent_proposal=dict(latent=pose['position']+[0.,0.,0.], latent_radius=radius, gaussian_component=None,
                hard_free_line_draw=dict(conditional=False, original_latent=pose['position']+[0.,0.,0.])) if is_latent else None,
            latent_density=dict(latent=d.latent, in_reference_ball=d.in_reference_ball, coordinate_chart_seam=False,
                log_latent_density=d.log_latent_density if math.isfinite(d.log_latent_density) else None,
                log_physical_jacobian=d.log_physical_jacobian, structural_zero=d.structural_zero,
                hard_free_line_density=trace(physical,d)),
            log_vessel_proposal_density=float(vessel_logs[i]),
            log_latent_physical_density=d.log_physical_density if math.isfinite(d.log_physical_density) else None,
            log_proposal_density=logq, capture_valid=capture, wall_valid=wall, hard_valid=valid,
            log_hard_weight=-logq if valid else None, log_importance_weight=-logq if valid else None,
            q=q, region=band+('_bound' if contact else '_unbound') if valid else None,
            depletion_contact=contact if valid else None, clouds=[copy.deepcopy(cloud), copy.deepcopy(cloud)] if valid else []))
    stream.write(root/'manifest.json', manifest)
    (root/'samples.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (root/'attempts.jsonl').write_text(''.join(json.dumps(dict(draw=i,state='begin'))+'\n' for i in range(len(rows))))
    total = float(logsumexp([r['log_importance_weight'] for r in rows if r['hard_valid']])-math.log(len(rows)))
    stream.write(root/'summary.json', dict(complete=True, manifest=manifest, samples=len(rows), numerical_nulls=0,
        samples_sha256=stream.sha(root/'samples.jsonl'), attempts_sha256=stream.sha(root/'attempts.jsonl'),
        estimates={k:dict(log_normalizer=total) for k in ['total','hard_total']}, **stream.row_counts(rows)))
    return binary


class EntryPointTests(unittest.TestCase):
    def test_complete_entry_point_preserves_results_across_batch_sizes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); population = root/'population'; binary = complete_fixture(population)
            outputs = [stream.audit(population, root/f'batch-{b}', binary, b) for b in [1,3,64]]
            for result in outputs:
                self.assertTrue(result['complete']); self.assertEqual(result['new_pose_draws'],0)
                self.assertEqual(result['density_audit']['checked_attempts'],7)
                self.assertEqual(result['estimates']['total']['Qz']['nonzero'],4)
                self.assertEqual(result['primitive_count_audit']['wall_rejected'],1)
                self.assertEqual(result['primitive_count_audit']['capture_rejected'],1)
                self.assertEqual(result['primitive_count_audit']['hard_rejected'],1)
                self.assertEqual(result['geometry_sha256'],outputs[0]['geometry_sha256'])
                self.assertEqual(result['estimates'],outputs[0]['estimates'])
            self.assertEqual([o['batching']['peak_rows'] for o in outputs],[1,3,7])
            with self.assertRaises(ValueError): stream.audit(population,root/'batch-1',binary)

    def test_density_corruption_fails_and_retains_finished_batches(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); population = root/'population'; binary = complete_fixture(population)
            rows = [json.loads(s) for s in (population/'samples.jsonl').read_text().splitlines()]
            rows[4]['log_hard_weight'] += .25
            (population/'samples.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            summary = stream.read(population/'summary.json'); summary['samples_sha256'] = stream.sha(population/'samples.jsonl')
            stream.write(population/'summary.json',summary)
            with self.assertRaisesRegex(ValueError,'Physical hard weight'):
                stream.audit(population,root/'failed',binary,3)
            self.assertEqual(len((root/'failed/geometry.jsonl').read_text().splitlines()),3)
            self.assertFalse((root/'failed/analysis.json').exists())
            self.assertEqual(stream.read(root/'failed/status.json')['phase'],'failed')


if __name__ == '__main__': unittest.main()
