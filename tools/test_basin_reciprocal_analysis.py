"""Independent finite-fixture controls for reciprocal whole-capture auditing."""
import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

from analyze_basin_normalizers import audit_pose_proposal, population, sha
from prepare_deep_far_normalizer_atlas import registration
import test_basin_selected_anchor_analysis as legacy_tests


def save(path, value):
    path.write_text(json.dumps(value))


def pose(position, rotation):
    return dict(position=list(position), orientation=rotation.as_quat()[[3, 0, 1, 2]].tolist())


def matrix(p):
    return Rotation.from_quat(np.asarray(p['orientation'])[[1, 2, 3, 0]]).as_matrix()


def oracle_anchor_density(p, fixed, model, epsilon, center, radius):
    """Direct scipy normal PDF times analytic Cayley/Haar factor; no Density use."""
    rotation = matrix(fixed)
    t = rotation.T @ (np.asarray(p['position'])-fixed['position'])
    r = rotation.T @ matrix(p)
    terms = []
    base = model['base_model']
    for k, flag in enumerate(model['reciprocal_components']):
        for inverted in ([False, True] if flag else [False]):
            tx, rx = (-r.T@t, r.T) if inverted else (t, r)
            a = base['anchors'][k]
            delta = Rotation.from_matrix(rx@np.asarray(a['rotation']).T).as_quat()
            c = delta[:3]/delta[3]
            x = np.r_[tx-a['position'], base['angular_length']*c]
            jacobian = math.pi**2*base['angular_length']**3*(1+np.dot(c, c))**2
            terms.append(math.log(base['weights'][k]/(2 if flag else 1)) +
                         multivariate_normal.logpdf(x, base['means'][k], base['covariances'][k]) + math.log(jacobian))
    displacement = np.asarray(p['position'])-center
    uniform = math.log(epsilon)-3*math.log(2*radius) if np.all((displacement >= -radius)&(displacement < radius)) else -np.inf
    return float(np.logaddexp(uniform, math.log1p(-epsilon)+logsumexp(terms)))


class ReciprocalBasinAudit(unittest.TestCase):
    def fixture(self, root, selected=None):
        (root/'provenance').mkdir()
        center = np.array([7., -4., 3.])
        identity = Rotation.identity()
        fixed = [pose(center+[.4, .2, -.3], Rotation.from_rotvec([.2, -.3, .4])),
                 pose(center+[-1.3, .6, .8], Rotation.from_rotvec([-.4, .1, -.2]))]
        config = dict(fixed_poses=fixed, capture_center=center.tolist(), capture_radius=4.,
            initial_pose=pose(center+[1., .7, -.2], Rotation.from_rotvec([.1, -.1, .2])),
            reservoir_density=.4, depletant_radius=.7,
            metadata=dict(native_poses=[pose(center, identity)], rigid_members=[pose([0, 0, 0], identity)],
                          member_error_scale=2., angle_error_scale_deg=30.))
        save(root/'config.json', config);save(root/'provenance/input-config.json', config)
        save(root/'provenance/shape.json', dict(atoms=[dict(center=[0, 0, 0], radius=.1)]))
        lower = np.diag([1., 1.2, .8, .5, .7, .9]);lower[3, 0] = .3;lower[4, 1] = -.2
        base = dict(shape_sha256=sha(root/'provenance/shape.json'), coordinate_convention='anchor-body-relative',
            angular_length=1.3, weights=[.3, .7],
            anchors=[dict(position=[.5, -.2, .4], rotation=Rotation.from_rotvec([.3, .1, -.2]).as_matrix().tolist()),
                     dict(position=[-1., .6, -.1], rotation=Rotation.from_rotvec([-.2, .4, .3]).as_matrix().tolist())],
            means=[[.2, -.1, .3, .1, -.2, .05], [-.1, .2, -.3, -.05, .2, .1]],
            covariances=[(lower@lower.T).tolist(), (np.diag([1.5, .7, 1.1, .6, .9, .8])**2).tolist()])
        model = dict(schema='reciprocal-pose-mixture-v1', base_model=base, reciprocal_components=[True, False])
        save(root/'provenance/model.json', model)
        source = (Path(__file__).resolve().parents[1]/'src/proposal.rs').read_text()
        save(root/'provenance/source-bundle.json', dict(files={'src/proposal.rs':dict(text=source, sha256=hashlib.sha256(source.encode()).hexdigest())}))
        manifest = dict(schema=3, proposal_anchor_index=selected, physical_fixed_neighbor_count=2,
            proposal_model_kind='reciprocal-pose-mixture-v1', base_component_count=2, virtual_component_count=3,
            reciprocal_components=[True, False], covariance_scale=1., uniform_probability=.2, activity=.4,
            samples=7, cloud_replicates=2, seed=9123, **{'lambda':6.4})
        for key, name in [('config', 'input-config.json'), ('model', 'model.json'), ('shape', 'shape.json'), ('source_bundle', 'source-bundle.json')]:
            manifest[key+'_sha256'] = sha(root/'provenance'/name)
        poses = [fixed[0], pose(center+[1., 0, 0], Rotation.from_rotvec([.4, -.3, .2])),
                 pose(center+[4., 0, 0], identity), pose(center+[-4., 0, 0], identity),
                 pose(center+[4.5, 0, 0], Rotation.from_rotvec([-.2, .3, .1])),
                 pose(center+[0, 1.8, .2], Rotation.from_rotvec([.5, .2, -.3])),
                 pose(center+[.3, -.7, .9], Rotation.from_rotvec([-.3, -.4, .2]))]
        anchors = [selected] if selected is not None else [0, 1]
        qs = registration(poses, config['metadata'])
        rows = []
        for n, p in enumerate(poses):
            local = n % len(anchors)
            anchor_values = [oracle_anchor_density(p, fixed[i], model, .2, center, 4.) for i in anchors]
            logq = float(logsumexp(anchor_values)-math.log(len(anchors)))
            old = oracle_anchor_density(config['initial_pose'], fixed[anchors[local]], model, .2, center, 4.)
            uniform = n in (3, 6)
            component, inverted = (0, True) if n % 2 == 0 else (1, False)
            proposal = dict(moving_index=0, anchor_index=local+1, branch='uniform' if uniform else 'learned',
                component_index=None if uniform else component, candidate=dict(position=(np.asarray(p['position'])-center).tolist(), orientation=p['orientation']),
                null_reason=None, old_log_density=old, new_log_density=anchor_values[local], log_reverse_forward=old-anchor_values[local])
            if not uniform:proposal['component_inverted'] = inverted
            capture = np.linalg.norm(np.asarray(p['position'])-center) <= 4.
            valid = bool(capture and all(np.linalg.norm(np.asarray(p['position'])-f['position']) >= .2 for f in fixed))
            q = float(qs[n]) if valid else None
            region = ('native_core' if q <= .8 else 'native_shell' if q <= 1 else 'shoulder' if q < 2 else 'intermediate' if q < 5 else 'distant')+'_bound' if valid else None
            clouds = [dict(lower_volume=.1, overlap_points=k, uncertain_volume=.2, log_weight=.04+k*math.log1p(.4/6.4)) for k in (0, 1)] if valid else []
            rows.append(dict(draw=n, pose=p, proposal=proposal, log_proposal_density=logq,
                capture_valid=bool(capture), hard_valid=valid, q=q, region=region, depletion_contact=True if valid else None,
                log_hard_weight=-logq if valid else None,
                log_importance_weight=float(logsumexp([c['log_weight'] for c in clouds])-math.log(2)-logq) if valid else None, clouds=clouds))
        job = dict(directory=str(root), samples=7, seed=9123, covariance_std_scale=1., proposal_anchor_index=selected)
        return manifest, rows, job

    def test_full_reciprocal_density_for_rotated_selected_and_marginal_anchors(self):
        for selected in (None, 0, 1):
            with self.subTest(selected=selected), tempfile.TemporaryDirectory() as directory:
                root=Path(directory);m,rows,job=self.fixture(root, selected)
                result=audit_pose_proposal(root,job,m,rows)
                self.assertLess(result['maximum_log_density_error'], 1e-12)
                self.assertEqual(result['checked_actual_poses'], 7)
                self.assertEqual(result['checked_proposal_labels'], 7)
                self.assertEqual(result['virtual_component_count'], 3)

    def test_sampled_anchor_and_unsymmetrized_denominators_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);m,rows,job=self.fixture(root)
            bad=copy.deepcopy(rows);bad[1]['log_proposal_density']=bad[1]['proposal']['new_log_density']
            self.assertGreater(abs(bad[1]['log_proposal_density']-rows[1]['log_proposal_density']), .01)
            with self.assertRaises(AssertionError):audit_pose_proposal(root,job,m,bad)
            model=json.loads((root/'provenance/model.json').read_text());model['reciprocal_components']=[False,False]
            cfg=json.loads((root/'config.json').read_text())
            wrong=[oracle_anchor_density(rows[1]['pose'],f,model,.2,np.asarray(cfg['capture_center']),4.) for f in cfg['fixed_poses']]
            bad=copy.deepcopy(rows);bad[1]['log_proposal_density']=float(logsumexp(wrong)-math.log(2))
            self.assertGreater(abs(bad[1]['log_proposal_density']-rows[1]['log_proposal_density']), .001)
            with self.assertRaises(AssertionError):audit_pose_proposal(root,job,m,bad)

    def test_strict_manifest_and_scale(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);m,rows,job=self.fixture(root)
            for key, value in [('schema',2),('base_component_count',3),('virtual_component_count',2),
                               ('reciprocal_components',[True,True]),('reciprocal_components',[1,False]),
                               ('proposal_model_kind','gaussian'),('covariance_scale',1.01),('source_bundle_sha256','0'*64)]:
                with self.subTest(key=key,value=value):
                    bad=dict(m);bad[key]=value
                    with self.assertRaises(AssertionError):audit_pose_proposal(root,job,bad,rows)

    def test_proposal_labels_cube_and_centered_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);m,rows,job=self.fixture(root)
            corruptions=[(1,'component_inverted',True),(1,'component_index',2),(1,'component_index',True),
                         (1,'anchor_index',0),(1,'anchor_index',3),(1,'moving_index',1),
                         (3,'component_inverted',False),(3,'component_index',0),(1,'branch','posterior'),
                         (1,'new_log_density',0.),(1,'old_log_density',0.),(1,'log_reverse_forward',0.)]
            for row,key,value in corruptions:
                with self.subTest(row=row,key=key,value=value):
                    bad=copy.deepcopy(rows);bad[row]['proposal'][key]=value
                    with self.assertRaises(AssertionError):audit_pose_proposal(root,job,m,bad)
            bad=copy.deepcopy(rows);bad[2]['proposal'].update(branch='uniform',component_index=None);bad[2]['proposal'].pop('component_inverted')
            with self.assertRaisesRegex(AssertionError,'half-open'):audit_pose_proposal(root,job,m,bad)
            bad=copy.deepcopy(rows);bad[1]['proposal']['candidate']['position'][0]+=.1
            with self.assertRaises(AssertionError):audit_pose_proposal(root,job,m,bad)

    def test_no_nulls_invalid_zeros_and_hard_denominator(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);m,rows,job=self.fixture(root)
            for row,key,value in [(0,'depletion_contact',False),(0,'log_importance_weight',0.),
                                  (4,'log_hard_weight',0.),(1,'log_hard_weight',0.),(1,'pose',None)]:
                with self.subTest(row=row,key=key):
                    bad=copy.deepcopy(rows);bad[row][key]=value
                    with self.assertRaises(AssertionError):audit_pose_proposal(root,job,m,bad)

    def test_source_bundle_internal_hash_and_physical_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);m,rows,job=self.fixture(root)
            source=root/'provenance/source-bundle.json';old=source.read_text();bad=json.loads(old)
            bad['files']['src/proposal.rs']['text']+='changed';save(source,bad)
            altered=dict(m,source_bundle_sha256=sha(source))
            with self.assertRaises(AssertionError):audit_pose_proposal(root,job,altered,rows)
            source.write_text(old)
            cfg=json.loads((root/'config.json').read_text());cfg['fixed_poses'][1]['position'][0]+=.1;save(root/'config.json',cfg)
            with self.assertRaises(AssertionError):audit_pose_proposal(root,job,m,rows)

    def test_population_retains_all_draws_and_rejects_missing_rows_or_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);m,rows,job=self.fixture(root)
            save(root/'manifest.json',m)
            summary=dict(complete=True,manifest=m,numerical_nulls=0,estimates={},sampler_cpu_seconds=1.)
            save(root/'summary.json',summary)
            raw=''.join(json.dumps(row)+'\n' for row in rows);(root/'samples.jsonl').write_text(raw)
            result=population(job)
            self.assertEqual(result['estimates']['total']['draws'],7)
            self.assertEqual(result['estimates']['total']['nonzero'],5)
            (root/'samples.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows if row['hard_valid']))
            with self.assertRaises(AssertionError):population(job)
            (root/'samples.jsonl').write_text(raw)
            m['schema']=1;save(root/'manifest.json',m);summary['manifest']=m;save(root/'summary.json',summary)
            with self.assertRaisesRegex(AssertionError,'schema disagree'):population(job)

    def test_all_false_envelope_matches_scaled_legacy_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);m,rows=legacy_tests.SelectedAnchorAudit().fixture(root);job=dict(proposal_anchor_index=0)
            before=audit_pose_proposal(root,job,m,rows)
            path=root/'provenance/model.json';model=json.loads(path.read_text())
            save(path,dict(schema='reciprocal-pose-mixture-v1',base_model=model,reciprocal_components=[False]))
            m['model_sha256']=sha(path)
            self.assertEqual(before,audit_pose_proposal(root,job,m,rows))


if __name__ == '__main__':
    unittest.main()
