#!/usr/bin/env python3
"""Independent spherical-wall/AO stationary references and kernel checks.

The wall confines hard particles only; ideal depletants occupy homogeneous R^3.
Exact independent starts avoid mistaking long autocorrelation for a target-law
test. These controls establish finite-sample stationarity, not mixing speed.
"""
from __future__ import annotations
import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.integrate import quad
from scipy.spatial.transform import Rotation
from scipy.stats import chi2
from scipy.special import logsumexp
from scipy.stats import multivariate_normal


def overlap(radius, separation):
    """Intersection of two equal balls; independent analytical expression."""
    d = np.asarray(separation, float)
    value = math.pi*(4*radius+d)*np.maximum(2*radius-d, 0)**2/12
    return np.where(d < 2*radius, value, 0.)


def lens_second_moment(radius, separation):
    """Integral |midpoint|^2 over B(-d/2,R) intersection B(+d/2,R)."""
    a = np.asarray(separation, float)/2
    # Integrate z²*pi*b(z)+pi*b(z)²/2 over both lens halves,
    # b(z)=R²-(|z|+d/2)². The origin is the lens midpoint.
    value = 2*math.pi*((2/5)*radius**5-a*radius**4+(2/3)*a*a*radius**3-a**5/15)
    return np.where(a < radius, np.maximum(value, 0.), 0.)


def pair_reference(a=1., wall=4., rd=.5, z=1.5):
    c, exclusion = wall-a, a+rd
    assert 0 < a < c and rd >= 0 and z >= 0
    points = [x for x in (2*exclusion,) if 2*a < x < 2*c]
    # A constant energy offset improves numerical conditioning.
    maximum = z*float(overlap(exclusion, 2*a))
    tilt = lambda d: math.exp(z*float(overlap(exclusion, d))-maximum)
    weight = lambda d: d*d*float(overlap(c, d))*tilt(d)
    integral = lambda f, upper=2*c: quad(f, 2*a, upper,
        points=[p for p in points if p < upper], epsabs=1e-10, epsrel=1e-11)[0]
    normalizer = integral(weight)
    d1 = integral(lambda d: d*weight(d))/normalizer
    d2 = integral(lambda d: d*d*weight(d))/normalizer
    com2 = integral(lambda d: d*d*float(lens_second_moment(c, d))*tilt(d))/normalizer
    contact = integral(weight, min(2*exclusion, 2*c))/normalizer if rd else 0.
    return dict(a=a, wall=wall, rd=rd, z=z, center_radius=c,
        expected=dict(distance=d1, distance_squared=d2, exclusion_contact_probability=contact,
            midpoint_squared=com2, mean_center_squared=com2+d2/4,
            midpoint_x=0., midpoint_y=0., midpoint_z=0., mean_rotation_trace=0.),
        density='p(d) proportional to d² overlap(C,d) exp[z overlap(a+rd,d)]; C=R-a, 2a<=d<=2C',
        wall_description='Protein-only wall; no solvent-wall exclusion or periodic images.')


def uniform_ball(rng, count, radius):
    direction = rng.normal(size=(count, 3))
    direction /= np.linalg.norm(direction, axis=1)[:, None]
    return radius*direction*rng.random(count)[:, None]**(1/3)


def exact_pair_starts(rng, count, a=1., wall=4., rd=.5, z=1.5):
    """Independent rejection draws from the exact continuous physical target."""
    selected = []; accepted = attempts = 0; c = wall-a
    cap = float(overlap(a+rd, 2*a))
    while accepted < count:
        batch = max(2048, 4*(count-accepted))
        p = uniform_ball(rng, 2*batch, c).reshape(batch, 2, 3)
        d = np.linalg.norm(p[:, 1]-p[:, 0], axis=1)
        keep = (d >= 2*a)&(np.log(rng.random(batch)) <= z*(overlap(a+rd, d)-cap))
        selected.append(p[keep]); accepted += int(keep.sum()); attempts += batch
    p = np.concatenate(selected)[:count]
    rotations = Rotation.random(2*count, random_state=rng).as_matrix().reshape(count, 2, 3, 3)
    return p, rotations, dict(independent_draws=count, proposals=attempts,
        method='Uniform independent centers in accessible balls, exact AO exponential rejection, independent Haar orientations')


def features(positions, matrices, rd=.5, a=1.):
    d = np.linalg.norm(positions[:, 1]-positions[:, 0], axis=1)
    mid = positions.mean(axis=1)
    return dict(distance=d, distance_squared=d*d,
        exclusion_contact_probability=(d < 2*(a+rd)).astype(float),
        midpoint_squared=np.sum(mid*mid, axis=1),
        mean_center_squared=np.mean(np.sum(positions*positions, axis=2), axis=1),
        midpoint_x=mid[:, 0], midpoint_y=mid[:, 1], midpoint_z=mid[:, 2],
        mean_rotation_trace=np.trace(matrices, axis1=2, axis2=3).mean(axis=1))


def moment_checks(values, expected, multiplier=6.):
    checks = {}
    for name, target in expected.items():
        value = np.asarray(values[name]); mean = float(value.mean())
        se = float(value.std(ddof=1)/math.sqrt(len(value)))
        error = mean-target
        passed = abs(error) <= multiplier*se+2e-10
        checks[name] = dict(mean=mean, exact=target, standard_error=se,
            standardized_error=error/se if se else None, passed=passed)
    return dict(passed=all(c['passed'] for c in checks.values()), checks=checks)


def collinear_union(radius, centers):
    """Exact union volume for arbitrary equal spheres on a common line.

    Every overlap with an earlier nonadjacent sphere is covered by the preceding
    sphere. This geometrical fact cancels all higher inclusion/exclusion terms.
    Pair additivity is NOT being assumed.
    """
    centers = np.sort(np.asarray(centers))
    return float(len(centers)*4*math.pi*radius**3/3-np.sum(overlap(radius, np.diff(centers))))


def three_sphere_orbit(z=.4):
    magnitudes = np.array([.6, 1.2, 1.8]); exclusion = 1.25
    signs = np.asarray(list(itertools.product((-1, 1), repeat=3)))
    centers = signs*magnitudes
    unions = np.array([collinear_union(exclusion, x) for x in centers])
    pair_only = np.array([3*4*math.pi*exclusion**3/3-sum(float(overlap(exclusion, abs(x[i]-x[j])))
        for i, j in itertools.combinations(range(3), 2)) for x in centers])
    probability = np.exp(-z*(unions-unions.min())); probability /= probability.sum()
    wrong = np.exp(-z*(pair_only-pair_only.min())); wrong /= wrong.sum()
    return dict(a=.25, rd=1., wall=4., z=z, signs=signs.tolist(), magnitudes=magnitudes.tolist(),
        centers=centers.tolist(), exact_union_volumes=unions.tolist(), probability=probability.tolist(),
        incorrect_pair_additive_probability=wrong.tolist(),
        pair_additive_pearson_divergence=float(np.sum((wrong-probability)**2/probability)))


def world_chord(world, radii, wall, direction):
    u = np.asarray(direction)/np.linalg.norm(direction)
    b = np.asarray(world)@u
    discriminant = b*b+(wall-np.asarray(radii))**2-np.sum(np.asarray(world)**2, axis=1)
    assert np.min(discriminant) >= -1e-10
    roots = np.sqrt(np.maximum(discriminant, 0))
    return float(np.max(-b-roots)), float(np.min(-b+roots))


def reference_checks(rng, count=12000):
    c = 3.
    assert math.isclose(float(lens_second_moment(c, 0))/float(overlap(c, 0)), 3*c*c/5, rel_tol=1e-13)
    for d in np.linspace(0, 5.99, 37):
        h = c-d/2
        integral = 2*math.pi*quad(lambda z: z*z*(c*c-(z+d/2)**2)+.5*(c*c-(z+d/2)**2)**2, 0, h,
            epsabs=1e-12, epsrel=1e-12)[0]
        assert abs(integral-float(lens_second_moment(c, d))) < 3e-11
    reference = pair_reference()
    p, r, sampling = exact_pair_starts(rng, count)
    checks = moment_checks(features(p, r), reference['expected'])
    assert checks['passed'], checks
    orbit = three_sphere_orbit()
    # Integrate perpendicular disk unions directly: at each x all cross-section
    # disks are concentric, so the largest radius gives the exact union area.
    for centers, volume in zip(orbit['centers'], orbit['exact_union_volumes']):
        radius = 1.25; low, high = min(centers)-radius, max(centers)+radius
        knots = sorted({*(x-radius for x in centers), *(x+radius for x in centers),
            *((a+b)/2 for a, b in itertools.combinations(centers, 2))})
        integral = quad(lambda x: math.pi*max(0., *(radius*radius-(x-c)**2 for c in centers)),
            low, high, points=[x for x in knots if low < x < high], epsabs=1e-10, epsrel=1e-11)[0]
        assert abs(integral-volume) < 1e-9
    assert 6000*orbit['pair_additive_pearson_divergence'] > 50
    return dict(passed=True, exact_pair_reference=reference, iid_reference=checks,
        iid_sampling=sampling, collinear_triple_reference=orbit,
        lens_integral_checks=37, independent_cross_section_union_integrals=8,
        negative_control='Pair-additive eight-state law differs by a Pearson divergence large enough to detect at the planned budget.')


def shift_geometry(module, rng, repetitions=300):
    body = np.array([[.3, 0, 0], [-.2, .2, 0], [0, 0, .4]])
    radii = np.array([.3, .2, .2]); wall = 4.
    system = module.SphericalSystem(body, radii, wall, .7, .4)
    p = np.array([[-1.2, 0, 0], [1.2, 0, 0], [0, 1.5, 0.]])
    matrices = Rotation.random(3, random_state=rng).as_matrix()
    assert system.valid(p, matrices)
    worst = 0.
    for _ in range(repetitions):
        direction = rng.normal(size=3); direction /= np.linalg.norm(direction)
        world = (np.einsum('bij,aj->bai', matrices, body)+p[:, None]).reshape(-1, 3)
        lo, hi = world_chord(world, np.tile(radii, len(p)), wall, direction)
        new, newr, _ = system.shift(rng, p.copy(), matrices.copy(), direction=direction)
        change = new-p; distance = float(change[0]@direction)
        assert np.max(np.abs(change-distance*direction)) < 2e-12
        assert np.array_equal(newr, matrices)
        assert lo-1e-10 <= distance <= hi+1e-10
        reverse = world_chord(world+distance*direction, np.tile(radii, len(p)), wall, direction)
        error = max(abs(reverse[0]-(lo-distance)), abs(reverse[1]-(hi-distance)))
        assert error < 2e-11; worst = max(worst, error)
        assert system.valid(new, newr)
        p = new
    return dict(passed=True, steps=repetitions, maximum_reverse_chord_error=worst,
        internal_relative_poses_unchanged=True, atomic_wall_checked=True)


def synthetic_proposal(module):
    """Two overlapping full-covariance charts, not a fitted physical density."""
    lower = np.diag([1., 1.3, .9, .8, 1.1, .7])
    lower[1, 0] = .25; lower[3, 1] = -.3; lower[5, 2] = .2
    data = dict(schema='weighted-pose-mixture-v1', angular_length=1.3,
        weights=[.4, .6], means=[[0., 0., 0., 0., 0., 0.], [.2, -.1, 0., .1, 0., -.15]],
        covariances=[(lower@lower.T).tolist(), (1.4*lower@lower.T).tolist()],
        anchors=[dict(position=[2.3, 0., 0.], rotation=Rotation.from_rotvec([.2, -.1, .1]).as_matrix().tolist()),
                 dict(position=[-2.3, 0., 0.], rotation=Rotation.from_rotvec([-.1, .3, -.2]).as_matrix().tolist())])
    return module.SphericalRelativeProposal(data, 4., .2, support_radius=5.)


def independent_pose_logpdf(proposal, t, matrix, anchor_t, anchor_r):
    """SciPy Gaussian plus normalized-Haar Jacobian; no FrozenMixture methods."""
    t = np.asarray(t)
    if np.any(t < -proposal.support_radius) or np.any(t >= proposal.support_radius): return -math.inf
    relative = anchor_r.T@(t-anchor_t); orientation = anchor_r.T@matrix
    data = proposal.data; length = float(data['angular_length']); terms = []
    for weight, anchor, mean, cov in zip(data['weights'], data['anchors'], data['means'], data['covariances']):
        vector = Rotation.from_matrix(orientation@np.asarray(anchor['rotation']).T).as_rotvec()
        angle = np.linalg.norm(vector)
        cayley = vector*(math.tan(angle/2)/angle) if angle else np.zeros(3)
        latent = np.r_[relative-anchor['position'], length*cayley]
        log_jacobian_to_haar = 3*math.log(length)+2*math.log(math.pi)+2*math.log1p(float(cayley@cayley))
        terms.append(math.log(weight)+multivariate_normal.logpdf(latent, mean=mean, cov=cov)+log_jacobian_to_haar)
    return float(logsumexp([math.log(proposal.uniform_weight)-3*math.log(2*proposal.support_radius),
        math.log1p(-proposal.uniform_weight)+logsumexp(terms)]))


def auxiliary_balance(module, rng, trials=220):
    proposal = synthetic_proposal(module)
    auxiliary = module.AuxiliaryMeans(proposal, gain=1., noise=.35, cutoff=6., clip=4., shrinkage=.3)
    system = module.SphericalSystem(np.zeros((1, 3)), np.ones(1), 4., .5, 0.)
    positions, rotations, _ = exact_pair_starts(rng, trials, z=0.)
    maximum_density_error = maximum_cancel = maximum_flux = maximum_step_error = 0.
    wrong_reverse_error = nonzero_log_jacobian = 0.; compared = hard_valid = 0
    for p, r in zip(positions, rotations):
        eta = rng.normal(size=(auxiliary.count, 6)); i = int(rng.integers(2))
        _, _, stats = system.step(rng, p.copy(), r.copy(), i, proposal, auxiliary, eta, kind='global')
        if 'proposed_pose' not in stats: continue
        compared += 1; j = stats['proposal']['anchor_index']
        y = p.copy(); ry = r.copy(); y[i] = stats['proposed_pose']['position']
        ry[i] = Rotation.from_quat(np.asarray(stats['proposed_pose']['orientation'])[[1, 2, 3, 0]]).as_matrix()
        forward, reverse = auxiliary.model(p, r, eta), auxiliary.model(y, ry, eta)
        qxy = independent_pose_logpdf(forward, y[i], ry[i], p[j], r[j])
        qyx = independent_pose_logpdf(reverse, p[i], r[i], p[j], r[j])
        correct = qyx-qxy
        reported = stats['log_reverse_forward']
        maximum_step_error = max(maximum_step_error, abs(correct-reported))
        maximum_density_error = max(maximum_density_error,
            abs(qxy-forward.logpdf(y[i], ry[i], p[j], r[j])),
            abs(qyx-reverse.logpdf(p[i], r[i], p[j], r[j])))
        wrong = independent_pose_logpdf(forward, p[i], r[i], p[j], r[j])-qxy
        wrong_reverse_error = max(wrong_reverse_error, abs(wrong-correct))
        ax, ay = auxiliary.coordinates(p, r, eta), auxiliary.coordinates(y, ry, eta)
        fx, sx, _ = auxiliary.fit(p, r); fy, sy, _ = auxiliary.fit(y, ry)
        reverse_ax = fx+(sx/sy)[:, None]*(ay-fy)
        assert np.max(np.abs(reverse_ax-ax)) < 3e-13
        logjac = 6*float(np.log(sy/sx).sum())
        cancel = auxiliary.log_density(y, ry, ay)-auxiliary.log_density(p, r, ax)+logjac
        maximum_cancel = max(maximum_cancel, abs(cancel)); nonzero_log_jacobian = max(nonzero_log_jacobian, abs(logjac))
        if system.valid(y, ry):
            hard_valid += 1
            assert abs(stats['gate']['log_weight']) < 1e-15
            assert abs(stats['log_acceptance']-min(0., correct)) < 3e-10
            # At z=0 both valid endpoint physical densities are the same.
            flux_error = abs((qxy+min(0., reported))-(qyx+min(0., -reported)))
            maximum_flux = max(maximum_flux, flux_error)
    assert compared >= trials/3 and hard_valid > 10
    assert max(maximum_step_error, maximum_density_error, maximum_flux) < 3e-10
    assert maximum_cancel < 3e-11 and nonzero_log_jacobian > .5
    assert wrong_reverse_error > .05, 'Stale reverse-model negative control was uninformative'
    return dict(passed=True, attempted=trials, nonnull_proposals=compared, hard_valid_endpoints=hard_valid,
        independent_log_density_maximum_error=maximum_density_error,
        actual_step_hastings_maximum_error=maximum_step_error,
        accepted_log_flux_maximum_error=maximum_flux,
        conditional_density_plus_transport_jacobian_maximum_error=maximum_cancel,
        maximum_nonzero_transport_log_jacobian=nonzero_log_jacobian,
        stale_reverse_model_negative_control_log_error=wrong_reverse_error,
        scope='Actual AuxiliaryMeans and SphericalSystem.step; full state-dependent reverse model checked independently using SciPy Gaussian densities and Haar Jacobian. Normalized conditional density cancels its nonunit affine transport Jacobian.')


def write_auxiliary_fixture(module, path):
    """Actual conditional-fit outputs plus independent q values for Rust tests."""
    rng = np.random.default_rng(80192732)
    base = synthetic_proposal(module)
    settings = dict(gain=1., noise=.35, cutoff=6., clip=4., shrinkage=.3)
    auxiliary = module.AuxiliaryMeans(base, **settings)
    system = module.SphericalSystem(np.zeros((1, 3)), np.ones(1), 4., .5, 0.)
    positions, rotations, _ = exact_pair_starts(rng, 500, z=0.)
    def pose(t, r):
        return dict(position=t.tolist(), orientation=Rotation.from_matrix(r).as_quat()[[3, 0, 1, 2]].tolist())
    cases = []
    for p, r in zip(positions, rotations):
        eta = rng.normal(size=(auxiliary.count, 6)); i = int(rng.integers(2))
        _, _, stats = system.step(rng, p.copy(), r.copy(), i, base, auxiliary, eta, kind='global')
        if 'proposed_pose' not in stats: continue
        y, yr = p.copy(), r.copy()
        y[i] = stats['proposed_pose']['position']
        yr[i] = Rotation.from_quat(np.asarray(stats['proposed_pose']['orientation'])[[1, 2, 3, 0]]).as_matrix()
        if not system.valid(y, yr): continue
        j = stats['proposal']['anchor_index']
        fx, sx, nx = auxiliary.fit(p, r); fy, sy, ny = auxiliary.fit(y, yr)
        ax, ay = auxiliary.coordinates(p, r, eta), auxiliary.coordinates(y, yr, eta)
        mx, my = auxiliary.model(p, r, eta), auxiliary.model(y, yr, eta)
        forward = independent_pose_logpdf(mx, y[i], yr[i], p[j], r[j])
        reverse = independent_pose_logpdf(my, p[i], r[i], p[j], r[j])
        stale = independent_pose_logpdf(mx, p[i], r[i], p[j], r[j])
        assert abs(stats['log_reverse_forward']-(reverse-forward)) < 3e-11
        cases.append(dict(moving_index=i, anchor_index=j, eta=eta.tolist(),
            poses_x=[pose(t, R) for t, R in zip(p, r)], poses_y=[pose(t, R) for t, R in zip(y, yr)],
            fit_x=dict(f=fx.tolist(), s=sx.tolist(), counts=nx.tolist()),
            fit_y=dict(f=fy.tolist(), s=sy.tolist(), counts=ny.tolist()),
            whitened_coordinates_x=ax.tolist(), whitened_coordinates_y=ay.tolist(),
            model_means_x=mx.data['means'], model_means_y=my.data['means'],
            forward_logpdf=forward, reverse_logpdf=reverse, stale_reverse_logpdf=stale,
            log_reverse_forward=reverse-forward,
            conditional_logdensity_x=auxiliary.log_density(p, r, ax),
            conditional_logdensity_y=auxiliary.log_density(y, yr, ay),
            transport_log_jacobian=6*float(np.log(sy/sx).sum())))
        if len(cases) == 12: break
    assert len(cases) == 12 and max(abs(c['transport_log_jacobian']) for c in cases) > .5
    shape = dict(name='unit analytic sphere', atoms=[dict(center=[0., 0., 0.], radius=1.)], volume=4*math.pi/3)
    shape_bytes = (json.dumps(shape, indent=2)+'\n').encode()
    shape_hash = hashlib.sha256(shape_bytes).hexdigest()
    model = dict(base.data, shape_sha256=shape_hash, coordinate_convention='anchor-body-relative')
    result = dict(schema='spherical-auxiliary-python-reference-v1', model_data=model,
        shape=shape, shape_sha256=shape_hash, shape_hash=shape_hash,
        shape_hash_encoding='UTF8 json.dumps(shape,indent=2) plus newline',
        wall_radius=4., body_bound=1., support_radius=5., cube_lengths=[10., 10., 10.],
        uniform_weight=.2, auxiliary_settings=settings, cases=cases,
        source_sha256={str(Path(__file__).resolve()):hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            str(Path(module.__file__).resolve()):hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()},
        density_reference='Independent SciPy multivariate_normal and normalized SO3 Haar Jacobian; actual AuxiliaryMeans supplies deterministic f/s/counts. Every X/Y is hard- and wall-valid.')
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return dict(path=str(path), cases=len(cases), shape_sha256=shape_hash,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def pair_stationarity(module, rng, count, method, z=1.5):
    began = time.process_time(); reference = pair_reference(z=z)
    p, rotations, starts = exact_pair_starts(rng, count, z=z)
    system = module.SphericalSystem(np.zeros((1, 3)), np.ones(1), 4., .5, z)
    proposal = synthetic_proposal(module) if method in ('global-frozen', 'global-auxiliary', 'auxiliary-gca-shift') else None
    auxiliary = (module.AuxiliaryMeans(proposal, gain=1., noise=.35, cutoff=6., clip=4., shrinkage=.3)
                 if method in ('global-auxiliary', 'auxiliary-gca-shift') else None)
    resultp, resultr = [], []
    for old, oldr in zip(p, rotations):
        new, newr = old.copy(), oldr.copy()
        if method in ('global-uniform', 'global-frozen', 'global-auxiliary', 'auxiliary-gca-shift'):
            i = int(rng.integers(2)); eta = rng.normal(size=(auxiliary.count, 6)) if auxiliary else None
            new, newr, _ = system.step(rng, new, newr, i, proposal, auxiliary, eta, kind='global')
        if method in ('local', 'local-gca-shift'):
            i = int(rng.integers(2))
            new, newr, _ = system.step(rng, new, newr, i, kind='local', translation_std=.8, angle_std_deg=15.)
        if method in ('gca', 'gca-shift', 'local-gca-shift', 'auxiliary-gca-shift'):
            new, newr, _ = system.gca(rng, new, newr)
        if method in ('shift', 'gca-shift', 'local-gca-shift', 'auxiliary-gca-shift'):
            new, newr, _ = system.shift(rng, new, newr)
        assert np.min(np.linalg.norm(new[1]-new[0])) >= 2-1e-10
        assert np.max(np.linalg.norm(new, axis=1)) <= 3+1e-10
        resultp.append(new); resultr.append(newr)
    observed = features(np.asarray(resultp), np.asarray(resultr))
    checked = moment_checks(observed, reference['expected'])
    before = features(p, rotations)
    paired = moment_checks({key:observed[key]-before[key] for key in before}, {key:0. for key in before})
    assert checked['passed'] and paired['passed'], (method, checked, paired)
    return dict(method=method, z=z, independent_starts=count, passed=True, moments=checked,
        paired_change=paired, cpu_seconds=time.process_time()-began, sampling=starts,
        scope='One actual kernel composition per exact independent equilibrium start; stationarity, not an ergodicity or convergence claim.')


def triple_stationarity(module, rng, count, z=.4):
    began = time.process_time(); reference = three_sphere_orbit(z)
    centers = np.asarray(reference['centers']); probability = np.asarray(reference['probability'])
    system = module.SphericalSystem(np.zeros((1, 3)), np.array([.25]), 4., 1., z)
    starts = rng.choice(8, size=count, p=probability); counts = np.zeros(8, int)
    transition = np.zeros((8, 8), int)
    for index in starts:
        p = np.zeros((3, 3)); p[:, 0] = centers[index]
        moved, _, _ = system.gca(rng, p, np.repeat(np.eye(3)[None], 3, axis=0), axis=np.array([0., 0., 1.]))
        distances = np.max(np.abs(centers-moved[:, 0]), axis=1)
        target = int(np.argmin(distances))
        assert distances[target] < 1e-10 and np.max(np.abs(moved[:, 1:])) < 1e-10
        counts[target] += 1; transition[index, target] += 1
    statistic = float(np.sum((counts-count*probability)**2/(count*probability)))
    pvalue = float(chi2.sf(statistic, 7))
    assert pvalue > 1e-6, (reference, counts, pvalue)
    # With stationary independent inputs, forward/reverse event counts have
    # equal expectations for this involutive fixed-axis GCA kernel.
    flux = []
    for i, j in itertools.combinations(range(8), 2):
        a, b = int(transition[i, j]), int(transition[j, i])
        if a+b >= 30:
            standardized = (a-b)/math.sqrt(a+b)
            assert abs(standardized) <= 6
            flux.append(dict(states=[i, j], forward=a, reverse=b, standardized_difference=standardized))
    return dict(passed=True, z=z, independent_starts=count, observed_counts=counts.tolist(),
        exact_probabilities=probability.tolist(), chi_squared=statistic, p_value=pvalue,
        transition_counts=transition.tolist(), flux_checks=flux,
        cpu_seconds=time.process_time()-began,
        scope='True triple-overlap exact collinear sphere orbit; all8 states hard/wall-valid, fixed proper half-turn axis. No pair-additive approximation.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path(__file__).resolve().parent/'spherical-validation.json')
    parser.add_argument('--reference-only', action='store_true')
    parser.add_argument('--pair-samples', type=int, default=3000)
    parser.add_argument('--triple-samples', type=int, default=6000)
    parser.add_argument('--fixture-out', type=Path)
    parser.add_argument('--methods', nargs='+', choices=['shift', 'gca', 'local', 'gca-shift',
        'local-gca-shift', 'global-uniform', 'global-frozen', 'global-auxiliary', 'auxiliary-gca-shift'],
        default=['shift', 'gca', 'local', 'gca-shift', 'global-frozen', 'global-auxiliary', 'auxiliary-gca-shift'])
    args = parser.parse_args(); rng = np.random.default_rng(202609201)
    result = dict(reference=reference_checks(rng), source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if not args.reference_only:
        import spherical_ensemble as module
        result['auxiliary_balance'] = auxiliary_balance(module, rng)
        if args.fixture_out:
            result['auxiliary_fixture'] = write_auxiliary_fixture(module, args.fixture_out)
        result['shift_geometry'] = shift_geometry(module, rng)
        result['pair_stationarity'] = []
        for method in args.methods:
            case = pair_stationarity(module, rng, args.pair_samples, method)
            result['pair_stationarity'].append(case)
            print(json.dumps(dict(method=method, passed=True, cpu_seconds=case['cpu_seconds'])), flush=True)
        result['triple_stationarity'] = [triple_stationarity(module, rng, args.triple_samples, z) for z in (0., .4)]
        result['implementation_sha256'] = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
    result['passed'] = True
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(passed=True, out=str(args.out))))


if __name__ == '__main__': main()
