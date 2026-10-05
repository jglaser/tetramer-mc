#!/usr/bin/env python3
"""Independent, coefficient-aware audit of the fixed Gaussian one-step control.

The physical toy target is the ORIGINAL map-generated normalized mixture on
R^3 x SO(3). No protein geometry, new random draws, fitting, or MC is performed.
All IID source and retained endpoint records are used, including MH self-loops.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.linalg import solve_triangular
from scipy.spatial.transform import Rotation
from scipy.special import logsumexp
from scipy.stats import binomtest

from dimer_destination_density import map_cholesky, ordered_sum, inverse_pose, pose_arrays, _pose


ARMS = ('adjusted_full', 'unadjusted_full', 'adjusted_old_shortcut')
VALID_ARMS = ARMS[:2]
OBSERVABLES = (
    'tx<0', 'ty<0', 'tz<0',
    'R00>0', 'R11>0', 'R22>0',
    'R01>0', 'R12>0', 'R20>0',
    'tx<0_and_R00>0', 'ty<0_and_R11>0', 'tz<0_and_R22>0',
)
ABSOLUTE_TOLERANCE = 2e-8
RELATIVE_TOLERANCE = 2e-10
FAMILY_ALPHA = .05
PRIMARY_TESTS = len(VALID_ARMS) * len(OBSERVABLES)
SOURCE_EXCERPTS = (
    ('fn cholesky(', 'fn chart_parameters_valid(',
     'e06c2385578d028b4a68cb912f64011c63b62ea8f441910dc13945731bb6c238'),
    ('    pub fn encode(', '\n}\n\n/// The correlated latent map',
     'e6354e442aa659e280a96d591992bbfbb5300c730e63b2a0be67197c35c9bc05'),
)


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def observations(pose):
    t, r = pose_arrays(pose)
    negative = [bool(t[i] < 0) for i in range(3)]
    diagonal = [bool(r[i, i] > 0) for i in range(3)]
    return negative + diagonal + [bool(r[0, 1] > 0), bool(r[1, 2] > 0), bool(r[2, 0] > 0)] + [
        negative[i] and diagonal[i] for i in range(3)]


class Chart:
    """Ordinary chart, factoring the covariance passed to the Rust MAP once."""
    def __init__(self, parameter, ell):
        self.parameter = parameter
        self.ell = float(ell)
        self.lower = map_cholesky(parameter['covariance'])
        self.mean = np.asarray(parameter['mean'], float)
        self.translation = np.asarray(parameter['anchor_position'], float)
        self.rotation = np.asarray(parameter['anchor_rotation'], float)
        self.logdet = ordered_sum(math.log(float(self.lower[i, i])) for i in range(6))
        require(self.mean.shape == (6,) and self.translation.shape == (3,) and self.rotation.shape == (3, 3), 'Bad chart dimensions')

    def coordinates(self, latent):
        z = np.asarray(latent, float)
        require(z.shape == (6,) and np.isfinite(z).all(), 'Bad latent')
        return np.asarray([float(self.mean[i]) + ordered_sum(float(self.lower[i, j])*float(z[j])
                           for j in range(i+1)) for i in range(6)])

    def encode(self, pose):
        t, r = pose_arrays(pose)
        q = Rotation.from_matrix(r @ self.rotation.T).as_quat()
        require(q[3] != 0., 'Exact Cayley seam in independent audit')
        x = np.r_[t-self.translation, self.ell*q[:3]/q[3]]
        require(np.isfinite(x).all(), 'Nonfinite Cayley coordinates')
        z = solve_triangular(self.lower, x-self.mean, lower=True, check_finite=False)
        require(np.isfinite(z).all(), 'Nonfinite latent')
        return z

    def decode(self, latent):
        x = self.coordinates(latent)
        c = x[3:]/self.ell
        length = math.hypot(1., *c)
        r = Rotation.from_quat(np.r_[c/length, 1./length]).as_matrix()
        return _pose(x[:3]+self.translation, r @ self.rotation)

    def log_volume(self, latent):
        c = self.coordinates(latent)[3:]/self.ell
        return self.logdet - 3*math.log(self.ell) - 2*math.log(math.pi) - 4*math.log(math.hypot(1., *c))

    def log_density(self, pose):
        z = self.encode(pose)
        return -3*math.log(2*math.pi) - .5*float(z @ z) - self.log_volume(z)


class Reference:
    def __init__(self, fixture):
        self.fixture = fixture
        self.ell = float(fixture['angular_length'])
        self.rho = float(fixture['correlation'])
        require(self.rho == .65, 'Changed fixed correlation')
        self.sine = math.sqrt((1-self.rho)*(1+self.rho))
        self.original = [Chart(p, self.ell) for p in fixture['original_parameters']]
        self.adjusted = [Chart(p, self.ell) for p in fixture['adjusted_parameters']]
        self.unadjusted = [Chart(p, self.ell) for p in fixture['unadjusted_parameters']]
        self.branches = fixture['branches']
        self.priors = np.asarray([b['prior'] for b in self.branches], float)
        require(len(self.branches) == len(self.adjusted) == len(self.unadjusted) == 3,
                'Changed virtual branch inventory')
        require(np.isfinite(self.priors).all() and np.all(self.priors > 0) and
                abs(math.fsum(self.priors)-1.) <= 1e-14, 'Invalid normalized branch priors')
        require([b['inverted'] for b in self.branches] == [False, True, False], 'Changed reciprocal control')
        self.logpriors = np.log(self.priors)

    def source(self, label, latent):
        b = self.branches[label]
        p = self.original[b['component_index']].decode(latent)
        return inverse_pose(p) if b['inverted'] else p

    def logs(self, pose):
        inverse = inverse_pose(pose)
        return np.asarray([self.logpriors[i] + self.original[b['component_index']].log_density(
                           inverse if b['inverted'] else pose) for i, b in enumerate(self.branches)])

    def step(self, arm, old, source, target, noise):
        charts = self.unadjusted if arm == 'unadjusted_full' else self.adjusted
        a, b = charts[source], charts[target]
        z = a.encode(old)
        noise = np.asarray(noise, float)
        next_z = self.rho*z+self.sine*noise
        inverse_noise = self.sine*z-self.rho*noise
        proposed = b.decode(next_z)
        source_volume, target_volume = a.log_volume(z), b.log_volume(next_z)
        old_noise_log = -.5*float(noise @ noise)
        new_noise_log = -.5*float(inverse_noise @ inverse_noise)
        log_jacobian = target_volume-source_volume
        log_noise = new_noise_log-old_noise_log
        old_logs, new_logs = self.logs(old), self.logs(proposed)
        old_g, new_g = float(logsumexp(old_logs)), float(logsumexp(new_logs))
        forward = old_logs[source]-old_g+self.logpriors[target]
        reverse = new_logs[target]-new_g+self.logpriors[source]
        selection = reverse-forward
        correction = log_jacobian+log_noise+selection
        scales = dict(log_extended_jacobian=1+abs(source_volume)+abs(target_volume),
                      log_auxiliary_ratio=1+abs(old_noise_log)+abs(new_noise_log),
                      log_forward_label_probability=1+abs(old_logs[source])+abs(old_g)+abs(self.logpriors[target]),
                      log_reverse_label_probability=1+abs(new_logs[target])+abs(new_g)+abs(self.logpriors[source]),
                      pi_ratio=1+abs(new_g)+abs(old_g))
        scales['log_map_correction'] = scales['log_extended_jacobian']+scales['log_auxiliary_ratio']
        scales['log_selection_reverse_forward'] = scales['log_forward_label_probability']+scales['log_reverse_label_probability']
        scales['log_reverse_forward'] = scales['log_map_correction']+scales['log_selection_reverse_forward']
        return dict(pose=proposed, source_latent=z, target_latent=next_z, inverse_noise=inverse_noise,
                    log_extended_jacobian=log_jacobian, log_auxiliary_ratio=log_noise,
                    log_map_correction=log_jacobian+log_noise,
                    old_logs=old_logs, new_logs=new_logs, old_g=old_g, new_g=new_g,
                    log_forward_label_probability=forward, log_reverse_label_probability=reverse,
                    log_selection_reverse_forward=selection, log_reverse_forward=correction, scales=scales)


def audit(fixture, events_path, summary):
    require(fixture['draws_per_population'] == 4096 and len(fixture['seeds']) == 4,
            'Changed fixed IID allocation')
    require(tuple(fixture['observables']) == OBSERVABLES, 'Changed predeclared observables')
    require(tuple(fixture['arms']) == ARMS and fixture['source_points'] == 16384
            and fixture['arm_attempts'] == 49152 and fixture['event_count'] == 131072,
            'Changed fixed counts/arm inventory')
    require(fixture['seeds'] == [202610040501, 202610040502, 202610040503, 202610040504], 'Changed IID seeds')
    require(summary['complete'] and summary['passed'] and summary['source_points'] == 16384
            and summary['arm_attempts'] == 49152 and summary['events'] == 131072
            and summary['populations'] == 4 and summary['draws_per_population'] == 4096
            and tuple(summary['arms']) == ARMS and summary['geometry_queries'] == summary['protein_queries'] == 0,
            'Incomplete or inconsistent Rust terminal')
    reference = Reference(fixture)
    maxima = {}
    maximum_derived_scales = {}
    errors = []
    checked = 0

    def check(condition, message):
        if not condition:
            errors.append(message)

    def near(actual, expected, name, identity, cancellation_scale=None):
        x, y = np.asarray(actual, float), np.asarray(expected, float)
        require(x.shape == y.shape and np.isfinite(x).all() and np.isfinite(y).all(), 'Invalid numerical field '+identity+': '+name)
        difference = np.abs(x-y)
        scale = np.abs(y) if cancellation_scale is None else np.asarray(cancellation_scale)
        if cancellation_scale is not None:
            maximum_derived_scales[name] = max(maximum_derived_scales.get(name, 0.), float(np.max(scale)))
        allowance = ABSOLUTE_TOLERANCE+RELATIVE_TOLERANCE*scale
        maxima[name] = max(maxima.get(name, 0.), float(np.max(difference)))
        check(bool(np.all(difference <= allowance)), identity+': '+name)

    def pose_near(actual, expected, name, identity):
        at, ar = pose_arrays(actual)
        et, er = pose_arrays(expected)
        near(at, et, name+'_translation', identity)
        near(ar, er, name+'_rotation', identity)

    populations = []
    missing_factor_max = 0.
    missing_factor_counterexamples = 0
    source_branch_counts = np.zeros(3, dtype=np.int64)
    source_latent_sum = np.zeros(6)
    source_latent_square_sum = np.zeros(6)
    with events_path.open() as stream:
        def event(kind, population, draw):
            nonlocal checked
            line = next(stream, None)
            require(line is not None, 'Incomplete event prefix')
            row = json.loads(line)
            require(row['kind'] == kind and row['population'] == population and row['draw'] == draw,
                    'Changed event enumeration')
            checked += 1
            return row

        for population in range(4):
            stats = {arm: dict(n=0, accepted=0, up=np.zeros(12, dtype=np.int64), down=np.zeros(12, dtype=np.int64),
                              old_sum=np.zeros(12, dtype=np.int64), new_sum=np.zeros(12, dtype=np.int64),
                              maximum_recovery_error=0., maximum_flow_residual=0., maximum_antisymmetry_error=0.,
                              maximum_recovery_ratio=0., maximum_flow_ratio=0., maximum_antisymmetry_ratio=0.) for arm in ARMS}
            for draw in range(4096):
                identity = f'{population}:{draw}'
                event('source_begun', population, draw)
                row = event('source', population, draw)
                label = row['generating_branch']
                require(type(label) is int and 0 <= label < 3, 'Invalid generating branch')
                z = np.asarray(row['generating_latent'], float)
                expected_old = reference.source(label, z)
                old = row['old']
                pose_near(old, expected_old, 'source_pose', identity)
                source_branch_counts[label] += 1
                source_latent_sum += z
                source_latent_square_sum += z*z
                old_logs = reference.logs(old)
                near(row['old_weighted_logs'], old_logs, 'old_weighted_logs', identity)
                old_obs = observations(old)
                check(row['observations_old'] == old_obs, identity+': source observations')
                trace = row['trace']
                i, j = trace['source'], trace['target']
                require(type(i) is int and type(j) is int and 0 <= i < 3 and 0 <= j < 3, 'Invalid selected labels')
                uniform = row['acceptance_uniform']
                require(math.isfinite(uniform) and 0 < uniform < 1, 'Invalid MH uniform')
                references = {}
                for arm in ARMS:
                    begun = event('arm_begun', population, draw)
                    require(begun['arm'] == arm, 'Changed arm order')
                    result = event('arm_result', population, draw)
                    require(result['arm'] == arm, 'Changed arm identity')
                    key = 'unadjusted' if arm == 'unadjusted_full' else 'adjusted'
                    if key not in references:
                        references[key] = reference.step(arm, old, i, j, trace['noise'])
                    expected = references[key]
                    step = result['step']
                    require(step['forward_trace'] == trace, 'Shared forward trace changed')
                    mapped = step['map_step']
                    require(mapped['inverse_trace']['source'] == j and mapped['inverse_trace']['target'] == i, 'Wrong reverse labels')
                    pose_near(mapped['pose'], expected['pose'], 'candidate_pose', identity+':'+arm)
                    for field in ('source_latent', 'target_latent', 'log_extended_jacobian', 'log_auxiliary_ratio'):
                        near(mapped[field], expected[field], field, identity+':'+arm, expected['scales'].get(field))
                    near(mapped['inverse_trace']['noise'], expected['inverse_noise'], 'inverse_noise', identity+':'+arm)
                    near(mapped['log_correction'], expected['log_map_correction'], 'map_correction', identity+':'+arm, expected['scales']['log_map_correction'])
                    correction = result['correction']
                    require(correction['status'] == 'finite', 'Unexpected zero reverse support in full Gaussian toy')
                    for field in ('log_forward_label_probability', 'log_reverse_label_probability',
                                  'log_selection_reverse_forward', 'log_map_correction', 'log_reverse_forward'):
                        near(correction[field], expected[field], field, identity+':'+arm, expected['scales'][field])
                    near(result['new_weighted_logs'], expected['new_logs'], 'new_weighted_logs', identity+':'+arm)
                    pi_ratio = expected['new_g']-expected['old_g']
                    used = -pi_ratio if arm == 'adjusted_old_shortcut' else expected['log_reverse_forward']
                    used_scale = expected['scales']['pi_ratio'] if arm == 'adjusted_old_shortcut' else expected['scales']['log_reverse_forward']
                    acceptance_scale = expected['scales']['pi_ratio']+used_scale
                    near(result['log_pi_ratio'], pi_ratio, 'pi_ratio', identity+':'+arm, expected['scales']['pi_ratio'])
                    near(result['used_log_correction'], used, 'used_correction', identity+':'+arm, used_scale)
                    near(result['raw_log_acceptance'], pi_ratio+used, 'raw_log_acceptance', identity+':'+arm, acceptance_scale)
                    near(result['log_acceptance'], min(0., pi_ratio+used), 'log_acceptance', identity+':'+arm, acceptance_scale)
                    check(result['log_acceptance'] == min(0., result['raw_log_acceptance']), identity+':'+arm+': clipped log ratio')
                    accepted = math.log(uniform) < min(0., result['log_acceptance'])
                    check(result['accepted'] is accepted, identity+':'+arm+': MH decision')
                    check(result['retained'] == (mapped['pose'] if accepted else old), identity+':'+arm+': retained self-loop')
                    retained_obs = observations(result['retained'])
                    check(result['observations_retained'] == retained_obs, identity+':'+arm+': retained observations')
                    pose_near(result['reverse_recovered_pose'], old, 'reverse_pose', identity+':'+arm)
                    near(result['reverse_recovered_noise'], trace['noise'], 'reverse_noise', identity+':'+arm)
                    near(result['reverse_log_correction'], -expected['log_reverse_forward'], 'reverse_correction', identity+':'+arm, expected['scales']['log_reverse_forward'])
                    # Full extended-flow symmetry is checked independently of whether MH accepted.
                    full_log_ratio = pi_ratio+expected['log_reverse_forward']
                    reverse_log_ratio = -pi_ratio+result['reverse_log_correction']
                    log_phi_old = -.5*float(np.asarray(trace['noise']) @ np.asarray(trace['noise']))
                    log_phi_new = -.5*float(expected['inverse_noise'] @ expected['inverse_noise'])
                    flow_scale = 1 + math.fsum(abs(v) for v in [expected['old_g'], expected['new_g'],
                        expected['log_forward_label_probability'], expected['log_reverse_label_probability'],
                        log_phi_old, log_phi_new, expected['log_extended_jacobian'],
                        expected['log_reverse_forward'], result['reverse_log_correction']])
                    old_t, old_r = pose_arrays(old)
                    recovered_t, recovered_r = pose_arrays(result['reverse_recovered_pose'])
                    recovery_scale = 1 + max(abs(float(v)) for group in [old_t, expected['pose']['position'], recovered_t,
                        expected['source_latent'], expected['target_latent'], trace['noise'], expected['inverse_noise'],
                        result['reverse_recovered_noise']] for v in group)
                    near(result['recovery_scale'], recovery_scale, 'recovery_scale', identity+':'+arm)
                    near(result['flow_scale'], flow_scale, 'flow_scale', identity+':'+arm)
                    require(fixture['relative_tolerance'] == 2e-11 and fixture['pose_tolerance'] == 2e-10
                            and fixture['flow_tolerance'] == 2e-9, 'Changed pre-data Rust consistency criteria')
                    near(result['recovery_tolerance'], 2e-10+2e-11*recovery_scale, 'recovery_tolerance', identity+':'+arm)
                    near(result['flow_tolerance'], 2e-9+2e-11*flow_scale, 'flow_tolerance', identity+':'+arm)
                    pose_error = max(float(np.max(abs(old_t-recovered_t))), float(np.max(abs(old_r-recovered_r))))
                    noise_error = float(np.max(abs(np.asarray(trace['noise'])-result['reverse_recovered_noise'])))
                    antisymmetry = abs(correction['log_reverse_forward']+result['reverse_log_correction'])
                    near(result['pose_recovery_error'], pose_error, 'pose_recovery_error', identity+':'+arm)
                    near(result['noise_recovery_error'], noise_error, 'noise_recovery_error', identity+':'+arm)
                    near(result['correction_antisymmetry_error'], antisymmetry, 'antisymmetry_error', identity+':'+arm, flow_scale)
                    check(max(result['pose_recovery_error'], result['noise_recovery_error']) <= result['recovery_tolerance'], identity+':'+arm+': producer recovery criterion')
                    check(result['correction_antisymmetry_error'] <= result['flow_tolerance'], identity+':'+arm+': producer antisymmetry criterion')
                    near(full_log_ratio+reverse_log_ratio, 0., 'forward_reverse_flux', identity+':'+arm, flow_scale)
                    forward_log_flow = expected['old_g']+expected['log_forward_label_probability']+log_phi_old
                    reverse_log_flow = expected['new_g']+expected['log_reverse_label_probability']+log_phi_new+expected['log_extended_jacobian']
                    flow_residual = abs(forward_log_flow + min(0., pi_ratio+used)
                                        - reverse_log_flow - min(0., -pi_ratio-used))
                    near(result['pointwise_flow_residual'], flow_residual, 'pointwise_flow_residual', identity+':'+arm, flow_scale)
                    if arm in VALID_ARMS:
                        check(result['pointwise_flow_residual'] <= result['flow_tolerance'], identity+':'+arm+': producer flow criterion')
                    if arm == 'adjusted_old_shortcut':
                        mismatch = abs(expected['log_reverse_forward']-used)
                        missing_factor_max = max(missing_factor_max, mismatch)
                        missing_factor_counterexamples += int(mismatch > ABSOLUTE_TOLERANCE+RELATIVE_TOLERANCE*(expected['scales']['log_reverse_forward']+used_scale))
                    s = stats[arm]
                    before, after = np.asarray(old_obs, int), np.asarray(retained_obs, int)
                    s['n'] += 1
                    s['accepted'] += accepted
                    s['up'] += (before == 0) & (after == 1)
                    s['down'] += (before == 1) & (after == 0)
                    s['old_sum'] += before
                    s['new_sum'] += after
                    s['maximum_recovery_error'] = max(s['maximum_recovery_error'], result['pose_recovery_error'], result['noise_recovery_error'])
                    s['maximum_flow_residual'] = max(s['maximum_flow_residual'], result['pointwise_flow_residual'])
                    s['maximum_antisymmetry_error'] = max(s['maximum_antisymmetry_error'], result['correction_antisymmetry_error'])
                    s['maximum_recovery_ratio'] = max(s['maximum_recovery_ratio'], max(result['pose_recovery_error'], result['noise_recovery_error'])/result['recovery_tolerance'])
                    s['maximum_flow_ratio'] = max(s['maximum_flow_ratio'], result['pointwise_flow_residual']/result['flow_tolerance'])
                    s['maximum_antisymmetry_ratio'] = max(s['maximum_antisymmetry_ratio'], result['correction_antisymmetry_error']/result['flow_tolerance'])
            for arm_index, arm in enumerate(ARMS):
                rust = summary['counts_by_population'][population][arm_index]
                for local, exported in [('n', 'attempted'), ('accepted', 'accepted'), ('up', 'up'), ('down', 'down'),
                                        ('old_sum', 'old_true'), ('new_sum', 'retained_true'),
                                        ('maximum_recovery_error', 'maximum_recovery_error'),
                                        ('maximum_flow_residual', 'maximum_flow_residual'),
                                        ('maximum_antisymmetry_error', 'maximum_antisymmetry_error'),
                                        ('maximum_recovery_ratio', 'maximum_recovery_ratio'),
                                        ('maximum_flow_ratio', 'maximum_flow_ratio'),
                                        ('maximum_antisymmetry_ratio', 'maximum_antisymmetry_ratio')]:
                    value = stats[arm][local]
                    value = value.tolist() if isinstance(value, np.ndarray) else value
                    check(value == rust[exported], f'{population}:{arm}: Rust summary '+exported)
            populations.append({arm: {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in s.items()}
                                for arm, s in stats.items()})
        require(next(stream, None) is None, 'Unexpected trailing events')
    require(checked == 131072, 'Wrong all-attempt event count')
    pooled = {}
    for arm in ARMS:
        totals = {key: np.asarray([p[arm][key] for p in populations]).sum(axis=0) for key in ('up', 'down', 'old_sum', 'new_sum')}
        tests = []
        for k, name in enumerate(OBSERVABLES):
            up, down = int(totals['up'][k]), int(totals['down'][k])
            probability = float(binomtest(up, up+down, .5).pvalue) if up+down else 1.
            tests.append(dict(name=name, up=up, down=down, old_count=int(totals['old_sum'][k]),
                              retained_count=int(totals['new_sum'][k]), paired_mean_change=(up-down)/16384,
                              exact_conditional_binomial_p=probability,
                              rejects_at_declared_threshold=probability <= FAMILY_ALPHA/PRIMARY_TESTS))
        pooled[arm] = dict(n=16384, accepted=sum(p[arm]['accepted'] for p in populations), tests=tests,
                           any_declared_rejection=any(t['rejects_at_declared_threshold'] for t in tests))
    correct_rejected = any(pooled[arm]['any_declared_rejection'] for arm in VALID_ARMS)
    primary_passed = not errors and not correct_rejected
    negative_sensitive = missing_factor_counterexamples > 0 and pooled['adjusted_old_shortcut']['any_declared_rejection']
    return dict(complete=True, passed=primary_passed, primary_checks_passed=primary_passed,
                negative_control_sensitivity_demonstrated=negative_sensitive,
                validation_status=('primary_checks_failed' if not primary_passed else
                    ('primary_checks_passed_negative_control_detected' if negative_sensitive else
                     'primary_checks_passed_negative_control_sensitivity_unresolved')),
                numerical_audit_passed=not errors, numerical_discrepancies=errors,
                maximum_absolute_numerical_differences=maxima, maximum_derived_term_scales=maximum_derived_scales,
                source_count=16384, transition_count=49152,
                events_checked=checked, population_results=populations, pooled=pooled,
                source_branch_counts=source_branch_counts.tolist(), source_latent_means=(source_latent_sum/16384).tolist(),
                source_latent_second_moments=(source_latent_square_sum/16384).tolist(),
                family_alpha=FAMILY_ALPHA, primary_tests=PRIMARY_TESTS, per_test_threshold=FAMILY_ALPHA/PRIMARY_TESTS,
                negative_control=dict(pointwise_missing_factor_max=missing_factor_max,
                                      pointwise_counterexamples=missing_factor_counterexamples,
                                      stationary_diagnostic_detected=pooled['adjusted_old_shortcut']['any_declared_rejection'],
                                      interpretation='A weak statistical negative control is reported without tuning; pointwise missing-factor disagreement remains separately visible.'),
                arithmetic_tolerances=dict(absolute=ABSOLUTE_TOLERANCE, relative=RELATIVE_TOLERANCE,
                    individual_fields='Original weighted branch log densities, latent/noise vectors, pose coordinates and positive scale/allowance fields use the absolute reference value.',
                    derived_scale_rules={
                        'chart_jacobian': '1+abs(source log volume)+abs(target log volume)',
                        'noise_ratio': '1+abs(old Gaussian log without constant)+abs(new Gaussian log without constant)',
                        'map_correction': 'chart Jacobian scale + noise-ratio scale',
                        'each_normalized_label_log': '1+abs(selected weighted original log)+abs(original mixture log)+abs(destination prior log)',
                        'selection_ratio': 'forward-label scale + reverse-label scale',
                        'full_or_reverse_correction': 'map-correction scale + selection-ratio scale',
                        'physical_target_ratio': '1+abs(old original-mixture log)+abs(new original-mixture log)',
                        'used_correction': 'full-correction scale for valid arms; target-ratio scale for old shortcut',
                        'raw_and_clipped_acceptance': 'target-ratio scale + used-correction scale',
                        'zero_and_pointwise_flow_residuals': '1+sum absolute old/new mixture, forward/reverse label, old/new Gaussian, Jacobian and full/reverse correction logs'},
                    scope='All scales predeclared algebraically before data. Numerical diagnostics only, not certified floating-point error bounds.'),
                geometry_queries=0, new_random_draws=0,
                scope='IID one-step normalized Gaussian-target control only. Nonrejection is not a proof of invariance or protein sampling efficacy. Original reciprocal branch density is exact inversion of its ordinary map chart, not a Gaussian fitted at the inverse mean.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('fixture', 'events', 'summary', 'source-bundle', 'producer-source', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    require(not args.out.exists(), 'Fresh output required')
    fixture = read(args.fixture)
    # The declared execution plan must bind the compiled source bundle, fixture,
    # this analyzer and all its imported arithmetic helpers before toy execution.
    inputs = {name: getattr(args, name) for name in ('fixture', 'events', 'summary', 'source_bundle', 'producer_source')}
    hashes = {name: sha(path) for name, path in inputs.items()}
    bundle = read(args.source_bundle)
    for name in ('src/basin_involution.rs', 'src/context_transport.rs'):
        entry = bundle['files'][name]
        require(hashlib.sha256(entry['text'].encode()).hexdigest() == entry['sha256'], 'Corrupt compiled source entry')
    require(bundle['files']['src/context_transport.rs']['sha256'] ==
            '16c218e4b4326b456da916d159da91acf0c71f81f007024c135ed21010ac67f8',
            'Unreviewed compiled selected-chart utility')
    require(hashes['producer_source'] == 'cc11fb80b3f1af48158b5cf307206565477e5aa1acc74d590fd9b7148760b469',
            'Unreviewed archived producer example')
    text = bundle['files']['src/basin_involution.rs']['text']
    for start, end, digest in SOURCE_EXCERPTS:
        first = text.index(start)
        require(hashlib.sha256(text[first:text.index(end, first)].encode()).hexdigest() == digest,
                'Unknown compiled factor/chart arithmetic')
    report = audit(fixture, args.events, read(args.summary))
    require(all(sha(path) == hashes[name] for name, path in inputs.items()), 'Audit input changed')
    report.update(schema='context-transport-independent-stationarity-v1', input_sha256=hashes,
                  reducer_sha256=sha(__file__), compiled_source_sha256={name: bundle['files'][name]['sha256'] for name in
                  ('src/basin_involution.rs', 'src/context_transport.rs')},
                  producer_source_sha256=hashes['producer_source'],
                  producer_source_binding='Example source is not embedded by build.rs; archived source and binary are jointly bound by the frozen build receipt and execution plan.')
    with args.out.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
