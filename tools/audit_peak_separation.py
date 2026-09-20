#!/usr/bin/env python3
"""Audit two frozen geometric charts without sampling or changing their inputs."""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import shutil

import numpy as np
from scipy.linalg import eigvalsh
from scipy.spatial.transform import Rotation


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def rotation(pose):
    q = np.asarray(pose['orientation'], dtype=float)
    require(q.shape == (4,) and np.isfinite(q).all() and np.linalg.norm(q) > 0,
            'Invalid quaternion')
    return Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()


def skew(v):
    x, y, z = v
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def q_value(metric, position, matrix):
    members = np.asarray([m['position'] for m in metric['rigid_members']])
    values = []
    for reference in metric['native_poses']:
        rr = rotation(reference)
        displacement = members @ (matrix-rr).T + position-reference['position']
        angle = Rotation.from_matrix(matrix @ rr.T).magnitude()
        values.append(max(np.linalg.norm(displacement, axis=1).max()/metric['member_error_scale'],
                          angle/math.radians(metric['angle_error_scale_deg'])))
    return float(min(values))


def geometry_summary(config, fixed, selected, model, radii):
    metric = config['metadata']
    members = np.asarray([m['position'] for m in metric['rigid_members']], dtype=float)
    require(len(members) > 0 and np.array_equal(members.mean(axis=0), np.zeros(3)),
            'Exactly centered members required')
    # Cross-product actions reconstruct A independently of the preparer.
    a = sum(skew(p).T @ skew(p) for p in members)/len(members)
    amin = float(eigvalsh(a).min())
    require(amin > 0 and math.isfinite(amin), 'Positive rotational moment required')
    fr, pr = rotation(fixed), rotation(selected)
    ft, pt = np.asarray(fixed['position']), np.asarray(selected['position'])
    relative = fr.T @ pr
    aa = relative @ a @ relative.T
    ell = model['angular_length']
    require(math.isfinite(ell) and ell > 0 and model['weights'] == [1.]
            and model['means'] == [[0.]*6] and len(model['anchors']) == 1,
            'One centered geometric chart required')
    expected = np.zeros((6, 6)); expected[:3, :3] = np.eye(3)
    expected[3:, 3:] = ell**2/4*np.linalg.inv(aa)
    covariance = np.asarray(model['covariances'][0])
    require(np.allclose(covariance, expected, rtol=2e-11, atol=2e-11), 'Chart covariance is not geometric')
    anchor = model['anchors'][0]
    require(np.allclose(anchor['position'], fr.T @ (pt-ft), atol=2e-11, rtol=0.)
            and np.allclose(anchor['rotation'], relative, atol=2e-11, rtol=0.), 'Chart center mismatch')
    lambdas = [float(eigvalsh(skew(p).T @ skew(p), a).max()) for p in members]
    # These margins accommodate ordinary roundoff; they are not interval arithmetic.
    lam_guard = max(lambdas)+1e-10*max(1., max(lambdas))
    amin_guard = amin*(1-1e-10)
    member_lipschitz = math.sqrt(1+lam_guard)
    q0 = q_value(metric, pt, pr)
    distance = float(np.linalg.norm(pt-config['capture_center']))
    bounds = []
    for radius in radii:
        angle = 2*math.atan(radius/(2*math.sqrt(amin_guard)))
        dq = max(member_lipschitz*radius/metric['member_error_scale'],
                 angle/math.radians(metric['angle_error_scale_deg']))
        bounds.append(dict(radius=radius, q_lower=max(0., q0-dq), q_upper=q0+dq,
                           capture_distance_upper_A=distance+radius,
                           capture_margin_A=config['capture_radius']-distance-radius,
                           rotation_angle_upper_deg=math.degrees(angle)))
    # Fixed directions exercise noncommuting rotations and mixed translation;
    # this is deterministic algebra validation, not a Monte Carlo draw.
    directions = list(np.eye(6))+list(-np.eye(6))+list(itertools.product((-1., 1.), repeat=6))
    chol = np.linalg.cholesky(covariance)
    errors, containments, q_excess = [], [], []
    for direction in directions:
        u = max(radii)*np.asarray(direction)/np.linalg.norm(direction)
        x = chol @ u; c = x[3:]/ell; cs = skew(c)
        increment = np.linalg.solve((np.eye(3)-cs).T, (np.eye(3)+cs).T).T
        moved_rotation = fr @ increment @ relative
        moved_position = pt+fr @ x[:3]
        delta = members @ (moved_rotation-pr).T+moved_position-pt
        actual = float(np.mean(np.sum(delta*delta, axis=1)))
        predicted = float(x[:3] @ x[:3]+4*c @ aa @ c/(1+c @ c))
        errors.append(abs(actual-predicted)); containments.append(actual-u @ u)
        q_excess.append(abs(q_value(metric, moved_position, moved_rotation)-q0)
                        -max(q0-bounds[-1]['q_lower'], bounds[-1]['q_upper']-q0))
    require(max(errors) < 2e-10 and max(containments) < 2e-10 and max(q_excess) < 2e-10,
            'Deterministic containment or metric check failed')
    return dict(center_q=q0, center_capture_distance_A=distance, A_A2=a.tolist(),
                member_generalized_max_eigenvalues=lambdas,
                guarded_member_displacement_per_chart_radius=member_lipschitz, bounds=bounds,
                deterministic_checks=dict(count=len(directions), RMS_squared_identity_error=max(errors),
                                          RMS_squared_minus_radius_squared=max(containments), q_bound_excess=max(q_excess))), members @ pr.T+pt


def load_preparation(path):
    freeze = read(path/'freeze.json')
    for name, digest in freeze.items():
        require(sha(path/name) == digest, 'Changed frozen input: '+name)
    protocol = read(path/'protocol.json')
    for name, digest in protocol['archived_sha256'].items():
        require(sha(path/'provenance'/name) == digest, 'Changed archived input: '+name)
    config, model, selected = (read(path/name) for name in ('config.json', 'model.json', 'selected-pose.json'))
    source = read(path/'provenance/source-region.json')
    require(source['physical_metric'] == config['metadata'] and source['physical_fixed_neighbors'] == config['fixed_poses'],
            'Changed physical geometry')
    require(sha(Path(config['shape'])) == source['shape_sha256'] == model['shape_sha256'], 'Changed shape')
    require(source['fixed_neighbor'] in config['fixed_poses'], 'Nonphysical chart frame')
    radii = [campaign['radius'] for campaign in protocol['campaigns']]
    require(radii == sorted(radii) and all(math.isfinite(r) and r > 0 for r in radii), 'Invalid radii')
    for campaign in protocol['campaigns']:
        region = read(Path(campaign['region']))
        require(sha(Path(campaign['region'])) == campaign['region_sha256'], 'Changed region')
        require(region['gaussian_chart'] == model and region['mahalanobis_radius'] == campaign['radius'], 'Region/model mismatch')
        for key in ('physical_metric', 'physical_fixed_neighbors', 'fixed_neighbor', 'capture_center', 'capture_radius',
                    'minimum_original_q', 'maximum_original_q', 'minimum_original_q_inclusive', 'maximum_original_q_inclusive'):
            require(region[key] == source[key], 'Changed region field: '+key)
    report, world = geometry_summary(config, source['fixed_neighbor'], selected['pose'], model, radii)
    return config, source, report, world


def audit(first, second, out):
    require(not out.exists(), 'Use a fresh output directory')
    a, sa, ra, wa = load_preparation(first); b, sb, rb, wb = load_preparation(second)
    for key in ('metadata', 'fixed_poses', 'capture_center', 'capture_radius', 'reservoir_density', 'depletant_radius'):
        require(a[key] == b[key], 'Different physical targets: '+key)
    require(sa == sb, 'Different original region definitions')
    distance = float(np.sqrt(np.mean(np.sum((wa-wb)**2, axis=1))))
    separations = [dict(first_radius=x['radius'], second_radius=y['radius'],
                       member_RMS_gap_A=distance-x['radius']-y['radius'],
                       disjoint_by_triangle=distance > x['radius']+y['radius']+1e-9)
                   for x in ra['bounds'] for y in rb['bounds']]
    inputs = {str(path/name):sha(path/name) for path in (first, second)
              for name in ('freeze.json','protocol.json','config.json','model.json','selected-pose.json','provenance/source-region.json')}
    archive = out/'provenance'; archive.mkdir(parents=True)
    shutil.copy2(Path(__file__), archive/Path(__file__).name)
    result = dict(complete=True, no_new_random_draws=True, first=str(first), second=str(second),
                  first_geometry=ra, second_geometry=rb, center_member_RMS_distance_A=distance,
                  pairwise_ball_separations=separations, input_sha256=inputs,
                  archived_sha256={Path(__file__).name:sha(archive/Path(__file__).name)},
                  identities=dict(chart_radius_squared='|dt|^2+4*c^T*A_anchor*c',
                                  member_RMS_squared='|dt|^2+4*c^T*A_anchor*c/(1+|c|^2)',
                                  member_bound='|delta member_i| <= sqrt(1+lambda_max(B_i,A))*rho; B_i=[r_i]_cross^T[r_i]_cross',
                                  q_bound='|q-q_center| <= max(max_member_bound/member_error_scale, rotation_bound/angle_error_scale)',
                                  separation='Member-RMS is a Euclidean metric on the labeled member configuration. Disjoint enclosing RMS balls imply disjoint chart balls.'),
                  scope='Analytic geometric inequalities checked with floating-point linear algebra and deterministic probes; no rigorous interval certificate, hard-validity claim, statistical weight, or convergence result. Original physical masks remain operative.')
    (out/'analysis.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    for path, digest in inputs.items():
        require(sha(Path(path)) == digest, 'Input changed during audit')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--first', type=Path, required=True)
    parser.add_argument('--second', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.first.resolve(), args.second.resolve(), args.out.resolve())
    print(json.dumps({key:result[key] for key in ('center_member_RMS_distance_A','pairwise_ball_separations')}, indent=2))
