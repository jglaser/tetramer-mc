"""Exhaustive wall partition and covariance controls using synthetic saved rows."""
import copy
import math
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

from analyze_mobile_wall_contacts import (
    PRIMARY, WALL, SEEDS, SAMPLES, native_unbound_diagnostic, partition_masks,
    read_weights, reference_target, summarize_regions, validate_jobs,
    validate_model_manifest, wall_target, validate_parent, sha)


def log(values):
    with np.errstate(divide='ignore'):
        return np.log(np.asarray(values, float))


def masks_for(valid, contact, native, anchors, triangle=None, d170=None, q=None, rn=None, ra=None):
    n = len(valid)
    return partition_masks(valid, contact, native, anchors,
        [False]*n if triangle is None else triangle,
        [True]*n if d170 is None else d170, [2.]*n if q is None else q,
        [0.]*n if rn is None else rn, [0.]*n if ra is None else ra)


class WallContactTests(unittest.TestCase):
    def test_exhaustive_native_precedence_and_native_unbound_anomaly(self):
        m = masks_for([True]*4+[False], [True, True, False, False, True],
            [True, False, False, True, False], [2, 0, 0, 1, 0], [True, False, False, False, False])
        np.testing.assert_array_equal(sum(m[k].astype(int) for k in PRIMARY), [1, 1, 1, 1, 0])
        self.assertEqual(np.flatnonzero(m['registered_native_entry']).tolist(), [0, 3])
        self.assertEqual(np.flatnonzero(m['contact_no_native_entry']).tolist(), [1])
        self.assertEqual(np.flatnonzero(m['unbound_no_native_entry']).tolist(), [2])
        self.assertEqual(np.flatnonzero(m['native_entry_unbound']).tolist(), [3])
        self.assertFalse(m['unbound_no_native_entry'][3])

    def test_q_is_historical_not_native_classifier_and_finite_masks_are_exact(self):
        m = masks_for([True]*6, [True]*6, [False, True, True, True, True, True], [0, 1, 2, 2, 2, 2],
            d170=[True, True, True, True, False, True], q=[1., 1., 2., 2., 2., 2.],
            rn=[4., np.nextafter(4., np.inf), 0., 0., 0., 0.],
            ra=[0., 0., 5., 32., 2., np.nextafter(32., np.inf)])
        self.assertTrue(m['original_q_le_1'][0]);self.assertFalse(m['registered_native_entry'][0])
        self.assertTrue(m['registered_native_entry'][2]);self.assertFalse(m['original_q_le_1'][2])
        self.assertEqual(np.flatnonzero(m['original_native_r4']).tolist(), [0])
        self.assertEqual(np.flatnonzero(m['alternative_native_r5']).tolist(), [2])
        self.assertEqual(np.flatnonzero(m['alternative_native_r32']).tolist(), [2, 3])
        self.assertTrue(m['registered_native_entry_outside_d170'][4])

    def test_shared_partition_covariance_and_fixed_n_zeros(self):
        m = masks_for([True, True, True, False], [True, True, False, False], [True, False, False, False], [1, 0, 0, 0])
        h = np.array([1., 1., 1., 0.]);z = h*3
        populations = [dict(id=f'r{i:02d}', seed=SEEDS[i], z=log(z), h=log(h),
            pairs=log(np.column_stack((z, z))), masks=m) for i in range(4)]
        estimates, covariance, ratios = summarize_regions(populations)
        total = estimates['total']['row_uncertainty']
        self.assertEqual(total['draws'], 16)
        self.assertAlmostEqual(math.exp(total['log_Qz']), 9/4)
        self.assertAlmostEqual(sum(math.exp(estimates[k]['row_uncertainty']['log_Qz']) for k in PRIMARY), 9/4)
        self.assertAlmostEqual(total['log_enhancement_SE'], 0., places=14)
        a = covariance['row']['regions'].index(PRIMARY[0]+':Qz')
        b = covariance['row']['regions'].index(PRIMARY[1]+':Qz')
        self.assertLess(covariance['row']['covariance_relative'][a][b], 0.)
        pair = ratios[PRIMARY[0]+'/'+PRIMARY[1]]['row']
        self.assertAlmostEqual(pair['log_ratio'], 0.)
        self.assertGreater(pair['log_ratio_SE'], 0.)

    def test_invalid_zeros_and_two_cloud_saved_weight_identity(self):
        valid = dict(draw=0, pose={'position':[0,0,0]}, hard_valid=True, wall_valid=True, capture_valid=True,
            q=2., depletion_contact=True, log_proposal_density=-2., log_hard_weight=2.,
            log_importance_weight=2.+math.log(3), clouds=[dict(log_weight=math.log(2)), dict(log_weight=math.log(4))])
        invalid = dict(valid, draw=1, hard_valid=False, wall_valid=False, q=None, depletion_contact=None,
                       log_hard_weight=None, log_importance_weight=None, clouds=[])
        result = read_weights([valid, invalid], 2)
        self.assertTrue(np.isneginf(result['z'][1]))
        bad = copy.deepcopy(valid);bad['wall_valid'] = False
        with self.assertRaisesRegex(ValueError, 'full-wall'):read_weights([bad, invalid], 2)
        bad = copy.deepcopy(invalid);bad['log_hard_weight'] = 0.
        with self.assertRaisesRegex(ValueError, 'zeros'):read_weights([valid, bad], 2)
        bad = copy.deepcopy(valid);bad['log_proposal_density'] -= .1
        with self.assertRaisesRegex(ValueError, 'full proposal'):read_weights([bad, invalid], 2)

    def test_atomic_wall_enclosure_not_d170_or_center_only_target(self):
        cfg = dict(capture_center=[0.,0.,0.], capture_radius=273., depletant_radius=1.5,
                   reservoir_density=.035, fixed_poses=['A','B'])
        shape = dict(atoms=[dict(center=[49.,0.,0.], radius=.4)])
        result = wall_target(cfg, shape, WALL)
        self.assertGreater(result['required_capture_radius_A'], WALL['radius'])
        bad = copy.deepcopy(cfg);bad['capture_radius'] = 170.
        with self.assertRaises(ValueError):wall_target(bad, shape, WALL)
        bad = dict(atoms=[dict(center=[50.,0.,0.], radius=1.)])
        with self.assertRaisesRegex(ValueError, 'enclose'):wall_target(cfg, bad, WALL)
        changed = dict(WALL, radius=220.)
        with self.assertRaisesRegex(ValueError, 'wall changed'):wall_target(cfg, shape, changed)

    def test_partial_reciprocal_manifest_supports_dynamic_count_and_rejects_wrong_wall(self):
        model = dict(schema='reciprocal-pose-mixture-v1', base_model=dict(weights=[.4,.6]), reciprocal_components=[True,False])
        pm = dict(schema=4, pose_proposal_schema=3, proposal_model_kind=model['schema'], base_component_count=2,
            virtual_component_count=3, reciprocal_components=[True,False], atomic_wall=WALL, bath_wall_permeable=True,
            covariance_scale=1., uniform_probability=.1, proposal_anchor_index=None, physical_fixed_neighbor_count=2)
        validate_model_manifest(model, pm)
        for key,value in [('pose_proposal_schema',1),('virtual_component_count',4),('bath_wall_permeable',False),('uniform_probability',.5)]:
            bad = copy.deepcopy(pm);bad[key] = value
            with self.assertRaises(ValueError):validate_model_manifest(model,bad)

    def test_failed_or_missing_population_and_reused_seed_rejected(self):
        jobs = [dict(id=f'r{i:02d}', seed=SEEDS[i], samples=SAMPLES, covariance_std_scale=1., cloud_replicates=2,
                     proposal_anchor_index=None) for i in range(4)]
        terminal = [dict(job, status='complete', returncode=0) for job in jobs]
        validate_jobs(jobs, terminal, 'original')
        with self.assertRaises(ValueError):validate_jobs(jobs[:-1], terminal, 'original')
        bad = copy.deepcopy(jobs);bad[1]['seed'] = bad[0]['seed']
        with self.assertRaises(ValueError):validate_jobs(bad, terminal, 'original')
        bad = copy.deepcopy(terminal);bad[0]['returncode'] = 1
        with self.assertRaises(ValueError):validate_jobs(jobs, bad, 'original')

    def test_native_unbound_diagnostic_preserves_supporting_motifs_and_gaps(self):
        row = dict(draw=7, pose={}, q=41., depletion_contact=False)
        label = dict(matches=[dict(motif_id=7, supporting_member_bonds=[dict(minimum_gap_A=1.9)])])
        result = native_unbound_diagnostic(row,label,'r02')
        self.assertEqual(result['supporting_entry_gaps_A'],[1.9])
        self.assertEqual(result['classification']['matches'][0]['motif_id'],7)
        self.assertIn('retained',result['interpretation'])

    def test_finite_reference_requires_same_scaffold_bath_and_containment(self):
        identity = dict(position=[0.,0.,0.], orientation=[1.,0.,0.,0.])
        cfg = dict(fixed_poses=[identity,identity], metadata={'metric':'original'}, reservoir_density=.035, depletant_radius=1.5)
        region = dict(shape_sha256='shape', physical_fixed_neighbors=cfg['fixed_poses'], physical_metric=cfg['metadata'],
            activity=.035, depletant_radius=1.5, capture_center=[0.,0.,0.], capture_radius=170., fixed_neighbor=identity,
            gaussian_chart=dict(weights=[1.], anchors=[dict(position=[0.,0.,0.], rotation=np.eye(3).tolist())],
                means=[[0.]*6], covariances=[(np.eye(6)*.01).tolist()]))
        self.assertAlmostEqual(reference_target(cfg,region,'shape',32.),3.2)
        changed = copy.deepcopy(region);changed['activity'] = .04
        with self.assertRaisesRegex(ValueError,'target differs'):reference_target(cfg,changed,'shape',32.)
        changed = copy.deepcopy(region);changed['gaussian_chart']['anchors'][0]['position'] = [169.,0.,0.]
        with self.assertRaisesRegex(ValueError,'extends beyond'):reference_target(cfg,changed,'shape',32.)

    def test_frozen_input_and_terminal_protocol_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary);(root/'provenance').mkdir();(root/'provenance/source-config.json').write_text('{}')
            entries = []
            for arm in ('original','coverage'):
                folder = root/arm;folder.mkdir()
                (folder/'manifest.json').write_text(json.dumps(dict(archive_sha256={'basin-normalizer':'binary','source-bundle.json':'bundle'})))
                entries.append(dict(arm=arm,path=str(folder)))
            protocol = dict(schema='mobile-wall-contact-controller-v1',campaigns=entries,
                populations_per_arm=4,samples_per_population=SAMPLES,seed_base=SEEDS[0],total_jobs=8,total_unconditional_draws=8*SAMPLES,
                cloud_replicates=2,lambda_ratio=64.,uniform_probability=.1,physical_wall=WALL,capture_radius=273.,
                original_config_sha256=sha(root/'provenance/source-config.json'),physical_executable_sha256='binary',source_bundle_sha256='bundle',
                native_definition='provenance/native-region/definition.json')
            (root/'protocol.json').write_text(json.dumps(protocol))
            status = dict(complete=True,phase='complete',protocol_sha256=sha(root/'protocol.json'),
                jobs=[dict(arm=a,id=f'r{i:02d}')for a in ('original','coverage')for i in range(4)],audits={'original':{},'coverage':{}})
            (root/'status.json').write_text(json.dumps(status))
            (root/'freeze.json').write_text(json.dumps(dict(files={'provenance/source-config.json':sha(root/'provenance/source-config.json')})))
            validate_parent(root)
            (root/'provenance/source-config.json').write_text('changed')
            with self.assertRaisesRegex(ValueError,'Frozen wall input'):validate_parent(root)
            (root/'provenance/source-config.json').write_text('{}');status['protocol_sha256'] = 'wrong'
            (root/'status.json').write_text(json.dumps(status))
            with self.assertRaisesRegex(ValueError,'Terminal protocol'):validate_parent(root)


if __name__ == '__main__':unittest.main()
