#!/usr/bin/env python3
"""Independent SciPy reference for Rust frozen relative-pose proposals.

This generator does not import the research proposal implementation. SciPy
Cholesky/triangular solves evaluate normalized Gaussians directly; explicit
Cayley/Haar formulas and ordinary quaternion geometry supply pose densities.
The generated JSON is self-contained for cargo tests (Python is not needed
when running the Rust tests).
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.special import logsumexp
from scipy.spatial.transform import Rotation
from scipy.stats import norm


def serialize(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def write(path, data):
    path.write_text(json.dumps(data, indent=2, default=serialize, allow_nan=False)+'\n')


def pose(position, matrix):
    q = Rotation.from_matrix(matrix).as_quat()
    return dict(position=np.asarray(position), orientation=q[[3, 0, 1, 2]])


def matrix(p):
    q = np.asarray(p['orientation'])
    return Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()


def cayley(c):
    q = np.r_[np.asarray(c), 1.]
    q /= np.linalg.norm(q)
    return Rotation.from_quat(q).as_matrix()


def inverse_cayley(R):
    # Quaternion extraction is independent of the Rust's direct matrix branch.
    q = Rotation.from_matrix(R).as_quat()
    if q[3] == 0:
        return None
    c = q[:3]/q[3]
    return c if np.all(np.isfinite(c)) else None


def relative_log_density(t, R, data):
    ell = data['angular_length']; terms = []
    for weight, anchor, mean, covariance in zip(data['weights'], data['anchors'], data['means'], data['covariances']):
        c = inverse_cayley(R@np.asarray(anchor['rotation']).T)
        if c is None:
            terms.append(-math.inf); continue
        latent = np.r_[t-np.asarray(anchor['position']), ell*c]
        L = cholesky(covariance, lower=True)
        y = solve_triangular(L, latent-np.asarray(mean), lower=True)
        gaussian = -3*math.log(2*math.pi)-np.log(np.diag(L)).sum()-.5*float(y@y)
        log_j = -2*math.log(math.pi)-4*math.log(math.hypot(1., *c))
        terms.append(math.log(weight)+gaussian+3*math.log(ell)-log_j)
    return float(logsumexp(terms))


def log_density(p, anchor, lengths, eta, data):
    Rt = matrix(anchor).T
    delta = np.asarray(p['position'])-np.asarray(anchor['position'])
    delta -= lengths*np.floor(delta/lengths+.5)
    t, R = Rt@delta, Rt@matrix(p)
    learned = relative_log_density(t, R, data)
    total = float(logsumexp([math.log(eta)-np.log(lengths).sum(), math.log1p(-eta)+learned]))
    return total, learned, t, R


def synthetic_model():
    rng = np.random.default_rng(725940)
    root = np.tril(rng.normal(scale=.23, size=(6, 6)))
    root[np.diag_indices(6)] = [.7, 1.1, .9, .4, .5, .6]
    reference = Rotation.from_rotvec([.3, -.2, .4]).as_matrix()
    return dict(schema='weighted-pose-mixture-v1', angular_length=2.1,
        means=[[.2, -.3, .4, .3, -.2, .1]], covariances=[root@root.T],
        anchors=[dict(position=[.3, -.2, .5], rotation=reference)], weights=[1.],
        shape_sha256='0'*64, coordinate_convention='anchor-body-relative')


def make_cases(data, seed):
    rng = np.random.default_rng(seed)
    lengths = np.array([571.4643787085516]*3); eta = .1
    pairs = []
    # Means and nearby full-covariance draws test narrow, coupled modes under
    # independently rotated and translated anchor particles.
    for k in range(len(data['weights'])):
        mean = np.array(data['means'][k]); covariance = np.asarray(data['covariances'][k])
        for centered in (True, False):
            u = mean if centered else mean+cholesky(covariance, lower=True)@rng.normal(size=6)*.4
            relative_t = np.array(data['anchors'][k]['position'])+u[:3]
            relative_R = cayley(u[3:]/data['angular_length'])@np.array(data['anchors'][k]['rotation'])
            anchor_R = Rotation.random(random_state=rng).as_matrix()
            anchor_t = rng.uniform(0., lengths)
            anchor = pose(anchor_t, anchor_R)
            old = pose(np.mod(anchor_t+anchor_R@relative_t, lengths), anchor_R@relative_R)
            candidate = pose(rng.uniform(0., lengths), Rotation.random(random_state=rng).as_matrix())
            pairs.append((f'component_{k}_'+('mean' if centered else 'nearby'), old, candidate, anchor))
    for axis in np.eye(3):
        for distance in (1e-4, 1e-8, 0.):
            relR = Rotation.from_rotvec(axis*(math.pi-distance)).as_matrix()
            anchor_R = Rotation.random(random_state=rng).as_matrix()
            anchor_t = rng.uniform(0., lengths)
            anchor = pose(anchor_t, anchor_R)
            old = pose(np.mod(anchor_t+anchor_R@np.array([8., -13., 21.]), lengths), anchor_R@relR)
            candidate = pose(np.mod(np.asarray(old['position'])+[.3, -.8, .5], lengths), anchor_R@relR)
            pairs.append((f'near_pi_{axis.tolist()}_{distance:g}', old, candidate, anchor))
    for k in range(12):
        anchor = pose(rng.uniform(0., lengths), Rotation.random(random_state=rng).as_matrix())
        old = pose(rng.uniform(0., lengths), Rotation.random(random_state=rng).as_matrix())
        candidate = pose(rng.uniform(0., lengths), Rotation.random(random_state=rng).as_matrix())
        pairs.append((f'random_{k}', old, candidate, anchor))
    cases = []
    for label, old, candidate, anchor in pairs:
        old_log, learned, relative_t, relative_R = log_density(old, anchor, lengths, eta, data)
        new_log = log_density(candidate, anchor, lengths, eta, data)[0]
        image_old = dict(old, position=np.asarray(old['position'])+lengths*np.array([2, -3, 1]))
        image_anchor = dict(anchor, position=np.asarray(anchor['position'])+lengths*np.array([-1, 2, 3]))
        cases.append(dict(label=label, old=old, candidate=candidate, anchor=anchor,
            expected_old_log_density=old_log, expected_new_log_density=new_log,
            expected_log_reverse_forward=old_log-new_log,
            relative_position=relative_t, relative_rotation=relative_R,
            expected_relative_log_density=learned,
            periodic_old=image_old, periodic_anchor=image_anchor))
    return dict(box_lengths=lengths, uniform_weight=eta, cases=cases)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=Path('/home/xvg/protein-nucleation/results/learned-tetramer-assembly/fit/frozen-relative-mixture.json'))
    parser.add_argument('--out-dir', type=Path, default=Path(__file__).resolve().parents[1]/'tests/fixtures')
    args = parser.parse_args(); args.out_dir.mkdir(parents=True, exist_ok=True)
    raw = json.loads(args.model.read_text())
    fields = ('schema', 'angular_length', 'anchors', 'means', 'covariances', 'weights', 'shape_sha256', 'coordinate_convention')
    model = {k: raw[k] for k in fields}
    model['source_model_sha256'] = hashlib.sha256(args.model.read_bytes()).hexdigest()
    synthetic = synthetic_model()
    write(args.out_dir/'proposal_model.json', model)
    write(args.out_dir/'proposal_synthetic_model.json', synthetic)
    reference = make_cases(model, 769111)
    reference['reference_generator_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    reference['source_model_sha256'] = model['source_model_sha256']
    reference['shape_sha256'] = model['shape_sha256']
    reference['independence'] = 'SciPy Cholesky/solve_triangular and quaternion geometry; no research proposal imports.'
    reference['synthetic_sampling'] = dict(mean=synthetic['means'][0], covariance=synthetic['covariances'][0],
        angular_length=synthetic['angular_length'], anchor=synthetic['anchors'][0], samples=40000,
        box_lengths=[1000., 1000., 1000.], uniform_weight=.1)
    cube_mass = float((norm.cdf(1.)-norm.cdf(-1.))**3)
    reference['unit_sphere_gaussian_sampling'] = dict(samples=40000, box_lengths=[2., 2., 2.],
        uniform_weight=.1, learned_cube_mass=cube_mass, expected_null_mass=.9*(1-cube_mass))
    write(args.out_dir/'proposal_reference.json', reference)
    print(json.dumps(dict(cases=len(reference['cases']), source_model_sha256=model['source_model_sha256'],
        out_dir=str(args.out_dir))))


if __name__ == '__main__':
    main()
