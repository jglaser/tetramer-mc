"""Hand-constructed records only: no sampler, Poisson draws, or protein audit."""
import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest
import numpy as np
from scipy.special import logsumexp

from audit_full_vessel_baseline import audit, check_rows, estimate_labels
from audit_full_vessel_latent import VesselDensity, check_generation_metadata, check_cloud_envelopes_and_counts
from analyze_basin_normalizers import moments
from physical_latent_guide import PhysicalLatentGuide
from prepare_deep_far_normalizer_atlas import registration
from test_audit_full_vessel_latent import data
from test_physical_latent_guide import fixture, independent_pose


def baseline_data(alpha=.5, reciprocal=False, selected=None):
    config, manifest, rows, vessel, reporting = data(alpha, reciprocal, selected)
    manifest['schema'] = 4
    for key in ('latent_gaussian_component_count', 'latent_defensive_uniform_probability'):
        manifest.pop(key)
    logs, geometry = vessel.evaluate([r['pose'] for r in rows])
    _, initial = vessel.evaluate([config['initial_pose']])
    for i, row in enumerate(rows):
        for key in ('outer_branch', 'latent_proposal', 'latent_density', 'log_vessel_proposal_density', 'log_latent_physical_density'):
            row.pop(key)
        row['log_proposal_density'] = float(logs[i])
        if row['hard_valid']:
            row['log_hard_weight'] = float(-logs[i])
            row['log_importance_weight'] = float(logsumexp([c['log_weight'] for c in row['clouds']]) - math.log(2) - logs[i])
        new, old = float(geometry['anchor_log_densities'][0, i]), float(initial['anchor_log_densities'][0, 0])
        row['proposal'] = dict(moving_index=0, anchor_index=1, branch='uniform', component_index=None,
            null_reason=None, candidate=dict(position=(np.asarray(row['pose']['position']) - config['capture_center']).tolist(),
                orientation=row['pose']['orientation']), old_log_density=old, new_log_density=new, log_reverse_forward=old-new)
    return config, manifest, rows, vessel, reporting


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def artifact(root):
    """A hard-only sphere geometry record; all four poses are prescribed by hand."""
    root.mkdir(); provenance = root / 'provenance'; provenance.mkdir()
    shape = dict(atoms=[dict(center=[0., 0., 0.], radius=.1)])
    write(provenance / 'shape.json', shape); shape_hash = sha(provenance / 'shape.json')
    region, guide, lower = fixture(.5, coupled=True)
    region['shape_sha256'] = region['gaussian_chart']['shape_sha256'] = shape_hash
    fixed = region['fixed_neighbor']; other = copy.deepcopy(fixed); other['position'][0] += 2.
    region['physical_fixed_neighbors'] = [fixed, other]
    region_path, guide_path = root / 'reporting-region.json', root / 'reporting-guide.json'
    write(region_path, region); guide['region_sha256'] = sha(region_path); write(guide_path, guide)
    model = region['gaussian_chart']; write(provenance / 'model.json', model)
    poses = [independent_pose(u, region, lower)[0] for u in ([.2,-.1,.3,.1,.2,.4], [5.,0.,0.,.3,0.,0.])]
    poses += [copy.deepcopy(fixed), dict(position=[400.,-2.,1.], orientation=[1.,0.,0.,0.])]
    config = dict(shape=str(provenance / 'shape.json'), fixed_poses=[fixed, other], initial_pose=poses[0],
        capture_center=[4.,-2.,1.], capture_radius=100., depletant_radius=.2, reservoir_density=0., poisson_lambda_ratio=64.,
        metadata=dict(native_poses=[fixed], rigid_members=[dict(position=[0.,0.,0.], orientation=[1.,0.,0.,0.])],
            member_error_scale=3., angle_error_scale_deg=90.))
    write(root / 'config.json', config); write(provenance / 'input-config.json', config)
    text = 'hand-constructed test records; no executable or RNG was invoked\n'
    bundle = dict(files={'fixture-description': dict(text=text, sha256=hashlib.sha256(text.encode()).hexdigest())})
    write(provenance / 'source-bundle.json', bundle)
    manifest = dict(schema=4, samples=4, shape_sha256=shape_hash, shape_bound=.1,
        pose_proposal_schema=1, covariance_scale=1.3, proposal_anchor_index=None, physical_fixed_neighbor_count=2,
        uniform_probability=.4, cloud_replicates=2, activity=0., **{'lambda':1.},
        atomic_wall=dict(center=[0.,0.,0.], radius=50.), bath_wall_permeable=True,
        executable_sha256='f'*64, config_sha256=sha(provenance / 'input-config.json'),
        model_sha256=sha(provenance / 'model.json'), source_bundle_sha256=sha(provenance / 'source-bundle.json'))
    vessel = VesselDensity(config, manifest, model)
    logs, geometry = vessel.evaluate(poses); qs = registration(poses, config['metadata'])
    _, initial = vessel.evaluate([poses[0]]); rows=[]
    for i, pose in enumerate(poses):
        capture = bool(np.linalg.norm(np.asarray(pose['position']) - config['capture_center']) <= 100.)
        wall = bool(np.linalg.norm(pose['position']) <= 49.9)
        gaps = [float(np.linalg.norm(np.asarray(pose['position']) - p['position']) - .2) for p in config['fixed_poses']]
        valid = capture and wall and min(gaps) >= 0
        contact = min(gaps) < .4
        cloud = dict(lower_volume=0., uncertain_volume=0., upper_volume=0., overlap_points=0, raw_points=0,
                     created_cells=0, retained_cells=0, certified_cells=0, log_weight=0.)
        label = 'native_core' if qs[i] <= .8 else 'native_shell' if qs[i] <= 1 else 'shoulder' if qs[i] < 2 else 'intermediate' if qs[i] < 5 else 'distant'
        new, old = float(geometry['anchor_log_densities'][0,i]), float(initial['anchor_log_densities'][0,0])
        uniform = bool(geometry['cube'][i])
        proposal = dict(moving_index=0, anchor_index=1, branch='uniform' if uniform else 'learned',
            component_index=None if uniform else 0, null_reason=None,
            candidate=dict(position=(np.asarray(pose['position']) - config['capture_center']).tolist(), orientation=pose['orientation']),
            old_log_density=old, new_log_density=new, log_reverse_forward=old-new)
        rows.append(dict(draw=i, pose=pose, proposal=proposal, capture_valid=capture, wall_valid=wall, hard_valid=valid,
            log_proposal_density=float(logs[i]), log_hard_weight=float(-logs[i]) if valid else None,
            log_importance_weight=float(-logs[i]) if valid else None, q=float(qs[i]) if valid else None,
            region=label+('_bound' if contact else '_unbound') if valid else None,
            depletion_contact=contact if valid else None, clouds=[copy.deepcopy(cloud), copy.deepcopy(cloud)] if valid else []))
    weights = np.array([r['log_importance_weight'] if r['hard_valid'] else -np.inf for r in rows])
    summary = dict(complete=True, manifest=manifest, numerical_nulls=0, samples=4,
        hard_valid=sum(r['hard_valid'] for r in rows), capture_rejected=sum(not r['capture_valid'] for r in rows),
        wall_rejected=sum(r['capture_valid'] and not r['wall_valid'] for r in rows),
        hard_rejected=sum(r['capture_valid'] and r['wall_valid'] and not r['hard_valid'] for r in rows), raw_points=0,
        estimates={k:dict(log_normalizer=moments(weights)['logQ']) for k in ('total','hard_total')})
    write(root / 'manifest.json', manifest); write(root / 'summary.json', summary)
    (root / 'samples.jsonl').write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in rows))
    return region_path, guide_path, rows


class BaselineAuditTests(unittest.TestCase):
    def test_original_law_all_and_selected_anchors_legacy_and_reciprocal(self):
        for reciprocal in (False, True):
            for selected in (None, 1):
                args = baseline_data(reciprocal=reciprocal, selected=selected)
                original = copy.deepcopy(args[2]); result, coords = check_rows(*args)
                self.assertEqual(result['checked_attempts'], 3)
                self.assertEqual(result['valid_outside_R4'], 1)
                self.assertEqual(args[2], original)
                self.assertFalse(coords[1].in_reference_ball)
                generation = check_generation_metadata(args[0], args[1], [dict(r, outer_branch='vessel') for r in args[2]], args[3])
                self.assertEqual(generation['checked_vessel_generation_rows'], 3)

    def test_reporting_guide_never_changes_weights_or_exterior_support(self):
        a = baseline_data(alpha=.5); b = baseline_data(alpha=1.)
        self.assertEqual(a[2], b[2])
        for args in (a,b):
            result, coords = check_rows(*args)
            self.assertEqual(result['valid_outside_R4'], 1)
            self.assertFalse(coords[1].in_reference_ball)
        self.assertEqual(b[4].evaluate(b[2][1]['pose']).log_physical_density, -math.inf)

    def test_missing_attempt_invalid_zero_and_outer_mixture_are_rejected(self):
        cfg,m,rows,v,g = baseline_data()
        for bad, message in [(rows[:2], 'Missing attempted draw'),
                             ([dict(rows[0],outer_branch='vessel')]+rows[1:], 'outer-mixture')]:
            with self.assertRaisesRegex(ValueError, message): check_rows(cfg,m,bad,v,g)
        bad = copy.deepcopy(rows); bad[2]['log_hard_weight'] = 0.
        with self.assertRaisesRegex(ValueError, 'explicit zero'): check_rows(cfg,m,bad,v,g)
        with self.assertRaisesRegex(ValueError, 'schema-4'): check_rows(cfg,dict(m,schema=5),rows,v,g)

    def test_wrong_density_extra_jacobian_and_cloud_geometric_mean_fail(self):
        cfg,m,rows,v,g = baseline_data()
        bad=copy.deepcopy(rows);bad[0]['log_proposal_density'] += math.log(2)
        with self.assertRaisesRegex(ValueError, 'original vessel density'):check_rows(cfg,m,bad,v,g)
        bad=copy.deepcopy(rows);bad[0]['log_hard_weight'] += g.evaluate(rows[0]['pose']).log_physical_jacobian
        with self.assertRaisesRegex(ValueError, 'no mixture or extra J'):check_rows(cfg,m,bad,v,g)
        bad=copy.deepcopy(rows);bad[0]['log_importance_weight'] = sum(c['log_weight'] for c in rows[0]['clouds'])/2-rows[0]['log_proposal_density']
        with self.assertRaisesRegex(ValueError, 'Arithmetic cloud-mean'):check_rows(cfg,m,bad,v,g)

    def test_mass_estimator_keeps_original_n_and_unobserved_zero_class(self):
        _,m,rows,_,_ = baseline_data()
        labels=[dict(classes=dict(total=r['hard_valid'],unobserved=False)) for r in rows]
        estimates=estimate_labels(rows,labels)
        expected=logsumexp([r['log_importance_weight'] for r in rows if r['hard_valid']])-math.log(3)
        self.assertAlmostEqual(estimates['total']['Qz']['logQ'],expected)
        self.assertEqual(estimates['total']['Qz']['draws'],3)
        self.assertIsNone(estimates['unobserved']['Qz']['logQ'])
        self.assertEqual(estimates['unobserved']['Qz']['draws'],3)
        self.assertIn('not a physical zero',estimates['unobserved']['Qz']['coverage'])
        counts=check_cloud_envelopes_and_counts(m,dict(samples=3,hard_valid=2,capture_rejected=0,wall_rejected=0,hard_rejected=1,raw_points=24),rows)
        self.assertEqual(counts['hard_rejected'],1)

    def test_end_to_end_hand_constructed_records_bind_reporting_and_keep_all_rows(self):
        with tempfile.TemporaryDirectory() as temp:
            parent=Path(temp);root=parent/'population';region,guide,rows=artifact(root)
            result=audit(root,parent/'audit',region,guide)
            self.assertTrue(result['complete']);self.assertEqual(result['density_audit']['checked_attempts'],4)
            self.assertEqual(result['primitive_count_audit']['hard_valid'],2)
            self.assertEqual(result['primitive_count_audit']['capture_rejected'],1)
            self.assertEqual(result['geometry_audit']['domain_invalid_pose_skips'],1)
            self.assertEqual(result['reporting_guide_binding']['region_sha256'],sha(region))
            self.assertEqual(result['reporting_guide_binding']['guide_sha256'],sha(guide))
            self.assertFalse(result['reporting_guide_binding']['affects_proposal'])
            self.assertFalse(result['executable_binding']['artifact_verified'])
            labels=[json.loads(s) for s in (parent/'audit/labels.jsonl').read_text().splitlines()]
            self.assertEqual([r['draw'] for r in labels],list(range(4)))
            self.assertEqual(sum(r['classes']['outside_R4'] for r in labels),1)
            self.assertEqual(result['estimates']['total']['Qz']['draws'],4)
            self.assertEqual(rows,[json.loads(s) for s in (root/'samples.jsonl').read_text().splitlines()])
            with self.assertRaisesRegex(ValueError,'Fresh audit'):audit(root,parent/'audit',region,guide)
            changed=json.loads(guide.read_text());changed['region_sha256']='0'*64;write(guide,changed)
            with self.assertRaisesRegex(AssertionError, 'another frozen region'):audit(root,parent/'bad-audit',region,guide)
            self.assertFalse((parent/'bad-audit').exists())


if __name__ == '__main__': unittest.main()
