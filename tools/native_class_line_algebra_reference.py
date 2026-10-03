"""Proposal algebra conditional on saved base line intervals; no atom geometry.

This module does not instantiate the native observer, read shapes, build atom
trees, or certify saved hard/native/exclusion unions. It reconstructs the chart,
line frames, class Boolean operations, Normal conditionals and complete proposal
law. Agreement is conditional on the supplied base unions being geometrically
correct. Physical validity, thinning, row inventory and estimator checks remain
separate obligations. There is no CLI or sampling entry point.
"""
from __future__ import annotations

import copy
import math
import numpy as np
from scipy.linalg import solve, solve_triangular
from scipy.special import logsumexp
from scipy.spatial.transform import Rotation

import native_class_line_reference as line

normal = line.hard.normal
require, close = normal.require, normal.close
CLASS_SCOPE = 'complete native; optional original latent orthants; no old-R5 restriction'
AUDIT_SCOPE = 'proposal algebra conditional on saved base H/native/exclusion unions; geometry is not certified'
FALLBACKS = ('class', 'hard_free', 'unconditional')


def finite_array(value, shape, label):
    result = np.asarray(value, float)
    require(result.shape == shape and np.isfinite(result).all(), 'Invalid '+label)
    return result


def proper_rotation(value, label):
    matrix = finite_array(value, (3, 3), label)
    require(np.allclose(matrix.T@matrix, np.eye(3), atol=1e-10, rtol=0)
            and abs(np.linalg.det(matrix)-1.) < 1e-10, 'Invalid proper rotation '+label)
    return matrix


class AlgebraLaw:
    """Geometry-free initializer and compact-trace proposal reconstruction."""

    def __init__(self, region, guide, config):
        self.region, self.guide, self.config = map(copy.deepcopy, (region, guide, config))
        require(guide['schema'] == line.SCHEMA, 'Wrong class-guide schema')
        require(region.get('minimum_mahalanobis_radius', 0.) == 0., 'Only complete latent balls are supported')
        chart = region['gaussian_chart']
        require(len(chart['means']) == len(chart['anchors']) == len(chart['covariances']) == 1,
                'Expected one frozen coordinate chart')
        self.m0 = finite_array(chart['means'][0], (6,), 'chart mean')
        self.L0, self.chart_factor_validation = line.chart_fp64_factor(chart['covariances'][0])
        self.ell, self.radius = chart['angular_length'], region['mahalanobis_radius']
        require(math.isfinite(self.ell) and self.ell > 0 and math.isfinite(self.radius) and self.radius > 0,
                'Invalid chart angular length or radius')
        self.anchor_position = finite_array(chart['anchors'][0]['position'], (3,), 'chart anchor position')
        self.anchor_rotation = proper_rotation(chart['anchors'][0]['rotation'], 'chart anchor')
        self.fixed_position, self.Rf = line.pose_arrays(region['fixed_neighbor'])
        self.capture_center = finite_array(config['capture_center'], (3,), 'capture center')
        self.capture_radius = config['capture_radius']
        require(math.isfinite(self.capture_radius) and self.capture_radius > 0, 'Invalid capture radius')
        self.logvolume = 3*math.log(math.pi)+6*math.log(self.radius)-math.log(6)
        self.alpha, self.beta = guide['defensive_uniform_shell_probability'], guide['conditional_probability']
        self.floor, self.axes = guide['minimum_conditional_mass'], list(guide['raw_translation_axes'])
        require(0 < self.alpha <= 1 and 0 <= self.beta <= 1 and 0 < self.floor < 1, 'Invalid mixture controls')
        require(self.axes and len(set(self.axes)) == len(self.axes)
                and all(type(a) is int and a in (0, 1, 2) for a in self.axes), 'Invalid translation axes')
        components = guide['gaussian_components']
        require(bool(components) or self.alpha == 1., 'Missing Gaussian mixture')
        weights = np.asarray([c['weight'] for c in components], float)
        require(np.isfinite(weights).all() and (weights > 0).all()
                and (not len(weights) or math.isfinite(weights.sum())), 'Invalid Gaussian weights')
        self.weights = weights/weights.sum() if len(weights) else weights
        n = len(components)
        self.means = finite_array(np.asarray([c['mean'] for c in components]).reshape((n, 6)), (n, 6), 'Gaussian means')
        covariance = finite_array(np.asarray([c['covariance'] for c in components]).reshape((n, 6, 6)),
                                  (n, 6, 6), 'Gaussian covariances')
        require(np.allclose(covariance, covariance.transpose(0, 2, 1), atol=1e-12, rtol=1e-12), 'Asymmetric Gaussian covariance')
        self.lowers = np.linalg.cholesky(covariance)
        self.normalizers = -3*math.log(2*math.pi)-np.log(np.diagonal(self.lowers, axis1=1, axis2=2)).sum(axis=1)
        self.rawmeans = self.m0+self.means@self.L0.T
        rawcovariance = np.einsum('ij,kjl,ml->kim', self.L0, covariance, self.L0)
        self.conditionals = {}
        for axis in self.axes:
            others = [i for i in range(6) if i != axis]
            coefficients, sigmas = [], []
            for matrix in rawcovariance:
                coefficient = solve(matrix[np.ix_(others, others)], matrix[others, axis], assume_a='pos')
                variance = matrix[axis, axis]-matrix[axis, others]@coefficient
                require(math.isfinite(variance) and variance > 0, 'Nonpositive conditional variance')
                coefficients.append(coefficient); sigmas.append(math.sqrt(variance))
            self.conditionals[axis] = (others, np.asarray(coefficients).reshape((n, 5)), np.asarray(sigmas))
        self.channels = copy.deepcopy(guide['class_channels'])
        require(bool(self.channels), 'Missing class channels')
        total = 0.
        for channel in self.channels:
            require(channel['class'] in ('hard_free', 'native', 'contact_without_native'), 'Unknown class')
            require(math.isfinite(channel['probability']) and channel['probability'] > 0, 'Invalid channel probability')
            if channel.get('orthant') is not None:
                require(type(channel['orthant']) is int and 0 <= channel['orthant'] < 64, 'Invalid original orthant')
            total += channel['probability']
        require(abs(total-1.) <= 1e-12, 'Channel probabilities must sum to one')
        for channel in self.channels: channel['probability'] /= total

    def raw(self, u):
        u = finite_array(u, (6,), 'latent coordinate')
        # Preserve the defined scalar chart convention, including exact zeros.
        result = np.zeros(6)
        for i in range(6):
            total = 0.
            for j in range(i+1): total += float(self.L0[i, j])*float(u[j])
            result[i] = float(self.m0[i])+total
        require(np.isfinite(result).all(), 'Unrepresentable raw coordinate')
        return result

    def inverse_raw(self, raw):
        return line.scalar_chart_solve(self.L0, finite_array(raw, (6,), 'raw coordinate')-self.m0)

    def decode(self, u):
        raw = self.raw(u); c = raw[3:]/self.ell
        squared = c@c; scale = 1+squared
        require(math.isfinite(scale), 'Unrepresentable Cayley rotation')
        relative = Rotation.from_quat(np.r_[c, 1.]/math.sqrt(scale)).as_matrix()
        rotation = self.Rf@relative@self.anchor_rotation
        position = self.fixed_position+self.Rf@(self.anchor_position+raw[:3])
        logj = np.log(np.diag(self.L0)).sum()-3*math.log(self.ell)-2*math.log(math.pi)-2*math.log1p(squared)
        require(np.isfinite(position).all() and math.isfinite(logj), 'Unrepresentable decoded pose/Jacobian')
        return raw, position, rotation, float(logj)

    def inverse_pose(self, position, rotation):
        position = finite_array(position, (3,), 'pose position')
        rotation = proper_rotation(rotation, 'pose')
        relative = self.Rf.T@rotation@self.anchor_rotation.T
        q = Rotation.from_matrix(relative).as_quat()
        require(q[3] != 0., 'Cayley chart excludes half-turn singularity')
        raw = np.r_[self.Rf.T@(position-self.fixed_position)-self.anchor_position, self.ell*q[:3]/q[3]]
        return self.inverse_raw(raw)

    def gaussian_logs(self, u):
        u = finite_array(u, (6,), 'latent coordinate')
        if not len(self.weights): return np.empty(0)
        z = np.asarray([solve_triangular(L, u-mean, lower=True) for L, mean in zip(self.lowers, self.means)])
        logs = np.log(self.weights)+self.normalizers-np.sum(z*z, axis=1)/2
        require(np.isfinite(logs).all(), 'Unrepresentable Gaussian log density')
        return logs

    def conditional(self, raw, axis):
        others, coefficient, sigma = self.conditionals[axis]
        return self.rawmeans[:, axis]+np.einsum('ki,ki->k', coefficient,
            np.asarray(raw)[others]-self.rawmeans[:, others]), sigma

    def line_frame(self, u, axis):
        require(axis in self.axes, 'Unknown translation axis')
        raw = self.raw(u); raw[axis] = 0.
        u0 = self.inverse_raw(raw); du = line.scalar_chart_solve(self.L0, np.eye(6)[axis])
        _, position, rotation, _ = self.decode(u0)
        from native_contact_regions import make_pose
        frame = dict(axis=axis, latent_line_origin=u0.tolist(), latent_line_direction=du.tolist(),
            origin=make_pose(position, rotation), direction=self.Rf[:, axis].tolist())
        r4 = line.closed_ball_chord(u0, du, self.radius)
        if r4 is None: frame['empty_reason'] = 'no_R4_chord'
        else:
            capture = line.closed_ball_chord(position-self.capture_center, self.Rf[:, axis], self.capture_radius)
            if capture is None: frame['empty_reason'] = 'no_capture_chord'
            else:
                segment = [max(r4[0], capture[0]), min(r4[1], capture[1])]
                if segment[0] > segment[1]: frame['empty_reason'] = 'disjoint_chords'
                else: frame['segment'] = segment
        return frame

    def _axis(self, u, saved):
        """Rebuild every channel from saved base unions, never saved channels."""
        require(type(saved['axis']) is int, 'Invalid saved translation-axis label')
        frame = self.line_frame(u, saved['axis'])
        require(saved.get('empty_reason') == frame.get('empty_reason'), 'Changed empty-line reason')
        for key in ('latent_line_origin', 'latent_line_direction', 'direction'):
            close(saved[key], frame[key], 'Changed '+key)
        close(saved['origin']['position'], frame['origin']['position'], 'Changed line origin')
        close(line.pose_arrays(saved['origin'])[1], line.pose_arrays(frame['origin'])[1], 'Changed line rotation')
        require(('segment' in saved) == ('segment' in frame), 'Unexpected/missing line segment')
        if 'segment' in frame: close(saved['segment'], frame['segment'], 'Changed R4/capture segment')
        bases = []
        for key in ('hard_free_intervals', 'native_intervals', 'exclusion_contact_intervals'):
            require(type(saved[key]) is list, 'Base interval union must be a list')
            normal.validate_intervals(saved[key], frame.get('segment'))
            if 'segment' not in frame: require(not saved[key], 'Empty line has nonempty base geometry')
            bases.append(copy.deepcopy(saved[key])); frame[key] = bases[-1]
        hard, native, exclusion = bases
        classes = dict(hard_free=hard, native=line.intersection(hard, native),
                       contact_without_native=line.difference(line.intersection(hard, exclusion), native))
        require(len(saved['channels']) == len(self.channels), 'Missing class channels')
        channels = []; endpoint_error = 0.
        for index, (channel, actual) in enumerate(zip(self.channels, saved['channels'])):
            require(type(actual['channel']) is int and actual['channel'] == index and all(actual.get(k) == channel.get(k)
                    for k in ('class', 'probability', 'orthant')), 'Changed channel identity')
            orthant = None
            if channel.get('orthant') is not None and 'segment' in frame:
                orthant = line.orthant_intervals(frame['latent_line_origin'], frame['latent_line_direction'],
                                               channel['orthant'], frame['segment'])
            intervals = classes[channel['class']]
            if orthant is not None: intervals = line.intersection(intervals, orthant)
            if orthant is None: require(actual.get('orthant_intervals') is None, 'Unexpected orthant interval union')
            else: endpoint_error = max(endpoint_error, line.compare_intervals(actual['orthant_intervals'], orthant, 'Original orthant'))
            endpoint_error = max(endpoint_error, line.compare_intervals(actual['intervals'], intervals, 'Rebuilt class'))
            channels.append(dict(channel, channel=index, intervals=intervals, orthant_intervals=orthant))
        return dict(frame, intervals=hard, channels=channels, interval_error=endpoint_error)

    def evaluate_saved(self, u, compact_trace):
        """Return vectorized complete q, conditioned on validated saved unions."""
        u = finite_array(u, (6,), 'latent coordinate')
        raw, position, rotation, logj = self.decode(u)
        logs = self.gaussian_logs(u)
        uniform = math.log(self.alpha)-self.logvolume if np.linalg.norm(u) <= self.radius else -math.inf
        baseline = float(np.logaddexp(uniform, math.log1p(-self.alpha)+logsumexp(logs))) if self.alpha < 1 else uniform
        common = dict(raw_coordinates=raw, position=position, rotation=rotation, log_physical_jacobian=logj,
                      baseline_log_density=baseline, geometry_certified=False, audit_scope=AUDIT_SCOPE,
                      latent=u.copy())
        if self.beta == 0 or self.alpha == 1:
            require(compact_trace == {'conditioning_disabled': True}, 'Changed disabled trace')
            return dict(common, log_density=baseline, conditioning_disabled=True, axes=[], interval_error=0.,
                        fallback_component_branches=0)
        require(compact_trace.get('trace_format') == 'class-line-compact-v1'
                and compact_trace.get('class_scope') == CLASS_SCOPE, 'Changed compact trace contract/scope')
        close(compact_trace['raw_coordinates'], raw, 'Changed saved raw coordinates')
        line.log_close(compact_trace['baseline_log_density'], baseline, 'Changed baseline density')
        require([a['axis'] for a in compact_trace['axes']] == self.axes, 'Missing/reordered axes')
        factors = np.zeros(len(logs)); axes = []; fallback_count = 0; endpoint_error = 0.
        for saved in compact_trace['axes']:
            require('components' not in saved, 'Compact trace contains component records')
            geometry = self._axis(u, saved); axis = geometry['axis']
            means, sigmas = self.conditional(raw, axis)
            hard = geometry['hard_free_intervals']
            hard_mass = line.interval_masses(hard, means, sigmas)
            hard_allowed = line.contains(hard, raw[axis])
            axis_factor = np.zeros(len(logs)); cache = {}
            def key(intervals):
                return tuple((i['lower'], i['upper'], i['lower_closed'], i['upper_closed']) for i in intervals)
            cache[key(hard)] = hard_mass
            for channel in geometry['channels']:
                intervals = channel['intervals']; fingerprint = key(intervals)
                if fingerprint not in cache: cache[fingerprint] = line.interval_masses(intervals, means, sigmas)
                mass = cache[fingerprint]
                require(np.all(mass <= hard_mass+2e-12), 'Class mass exceeds hard-free mass')
                fallback = np.where(mass > self.floor, 0, np.where(hard_mass > self.floor, 1, 2))
                effective = np.where(fallback == 0, mass, np.where(fallback == 1, hard_mass, 1.))
                allowed = np.where(fallback == 0, line.contains(intervals, raw[axis]),
                                   np.where(fallback == 1, hard_allowed, True))
                multiplier = np.where(allowed, 1./effective, 0.)
                require(np.isfinite(multiplier).all(), 'Unrepresentable conditional multiplier')
                channel.update(class_masses=mass, fallback_indices=fallback, effective_masses=effective,
                               query_coordinate_allowed=allowed, multipliers=multiplier)
                axis_factor += channel['probability']*multiplier
                fallback_count += int(np.count_nonzero(fallback))
            factors += axis_factor/len(self.axes)
            geometry.update(conditional_means=means, conditional_sigmas=sigmas, hard_free_masses=hard_mass,
                            channel_mixture_multipliers=axis_factor, distinct_mass_unions=len(cache))
            axes.append(geometry); endpoint_error = max(endpoint_error, geometry['interval_error'])
        close(compact_trace['component_mixture_multipliers'], factors,
              'Changed complete component mixture factors', atol=1e-12, rtol=3e-7)
        correction = 1-self.beta+self.beta*factors
        positive = correction > 0
        density = float(np.logaddexp(uniform, math.log1p(-self.alpha)+logsumexp(logs[positive]+np.log(correction[positive]))))
        return dict(common, log_density=density, axes=axes, interval_error=endpoint_error,
                    component_mixture_factors=factors, component_multipliers=correction,
                    fallback_component_branches=fallback_count,
                    component_branches=len(logs)*len(self.axes)*len(self.channels))

    def verify_draw(self, draw, u, evaluation):
        """Verify saved selection and inverse CDF using already rebuilt axes."""
        u = finite_array(u, (6,), 'latent coordinate')
        close(evaluation['latent'], u, 'Evaluation belongs to another latent')
        require(type(draw['conditional']) is bool, 'Invalid conditional draw flag')
        if not draw['conditional']:
            close(draw['original_latent'], u, 'Unconditioned draw changed')
            return dict(inverse_error=0., endpoint_error=0., geometry_certified=False)
        require(self.alpha < 1 and self.beta > 0, 'Conditional draw under disabled law')
        axis, component, channel_id = draw['axis'], draw['component'], draw['channel']
        require(type(axis) is int and axis in self.axes and type(component) is int
                and 0 <= component < len(self.weights) and type(channel_id) is int
                and 0 <= channel_id < len(self.channels), 'Invalid selected labels')
        channel = self.channels[channel_id]
        require(all(draw['class'].get(k) == channel.get(k) for k in ('class', 'probability', 'orthant')),
                'Selected channel differs')
        choice = draw['uniform_channel_selection']; require(0 < choice < 1, 'Invalid channel draw')
        cumulative = 0.; chosen = len(self.channels)-1
        for index, candidate in enumerate(self.channels):
            cumulative += candidate['probability']
            if choice < cumulative: chosen = index; break
        require(chosen == channel_id, 'Wrong channel categorical selection')
        raw, old = self.raw(u), self.raw(draw['original_latent'])
        others = [i for i in range(6) if i != axis]
        close(old[others], raw[others], 'Conditional draw changed retained coordinates')
        geometry = next(a for a in evaluation['axes'] if a['axis'] == axis)
        observed = self._axis(u, draw['geometry'])
        require(observed['axis'] == axis, 'Selected geometry axis differs')
        endpoint_error = observed['interval_error']
        for name in ('hard_free_intervals', 'native_intervals', 'exclusion_contact_intervals'):
            endpoint_error = max(endpoint_error, line.compare_intervals(observed[name], geometry[name], 'Draw/density '+name))
        for actual, expected in zip(observed['channels'], geometry['channels']):
            endpoint_error = max(endpoint_error, line.compare_intervals(actual['intervals'], expected['intervals'], 'Draw/density class'))
        branch = geometry['channels'][channel_id]
        mean, sigma = geometry['conditional_means'][component], geometry['conditional_sigmas'][component]
        mass = branch['class_masses'][component]; hard_mass = geometry['hard_free_masses'][component]
        effective = branch['effective_masses'][component]; fallback = FALLBACKS[branch['fallback_indices'][component]]
        for name, expected in [('conditional_mean', mean), ('conditional_sigma', sigma), ('class_mass', mass),
                               ('hard_free_mass', hard_mass), ('effective_mass', effective)]:
            close(draw[name], expected, 'Changed draw '+name, atol=2e-15 if 'mass' in name else 2e-8, rtol=3e-7)
        require(type(draw['fallback']) is bool and draw['fallback_target'] == fallback
                and draw['fallback'] == (fallback != 'class'), 'Changed fallback')
        if fallback == 'unconditional':
            close(draw['original_latent'], u, 'Unconditional fallback redrew coordinates')
            require(not any(k in draw for k in ('selected_interval', 'selected_interval_mass',
                'uniform_interval_selection', 'uniform_within_interval', 'returned_raw_coordinate')),
                'Unconditional fallback contains interval selection')
            return dict(inverse_error=0., endpoint_error=endpoint_error, geometry_certified=False)
        intervals = branch['intervals'] if fallback == 'class' else geometry['hard_free_intervals']
        require(line.contains(intervals, raw[axis]), 'Returned coordinate outside selected conditional law')
        selected = draw['selected_interval']
        source_intervals = draw['geometry']['channels'][channel_id]['intervals'] if fallback == 'class' else draw['geometry']['hard_free_intervals']
        # Rust stores the selected interval verbatim. Prefer that identity, so
        # distinct tiny intervals closer than the endpoint tolerance remain
        # distinguishable. Geometry/channel topology was checked above.
        matches = [i for i, v in enumerate(source_intervals) if v == selected]
        if not matches:
            matches = [i for i, v in enumerate(source_intervals) if abs(v['lower']-selected['lower']) < 1e-9
                       and abs(v['upper']-selected['upper']) < 1e-9]
        require(len(matches) == 1, 'Selected interval is missing or ambiguous')
        selected_index = matches[0]
        endpoint_error = max(endpoint_error, line.compare_intervals([selected], [intervals[selected_index]], 'Selected interval'))
        masses = np.asarray([float(line.interval_masses([v], mean, sigma)) for v in intervals])
        selected_mass = masses[selected_index]
        require(selected_mass > 0, 'Selected a zero-mass interval')
        close(draw['selected_interval_mass'], selected_mass, 'Changed selected interval mass', atol=2e-15, rtol=3e-7)
        choice = draw['uniform_interval_selection']; require(0 < choice < 1, 'Invalid interval draw')
        target = choice*effective
        cumulative = np.r_[0., np.cumsum(masses)]
        # Independent Normal integration need not round exactly as Rust's CDF.
        # Retain the established audit tolerance and report ambiguous bins.
        selection_tolerance = 2e-14
        possible = [i for i, value in enumerate(masses) if value > 0
                    and cumulative[i]-selection_tolerance <= target <= cumulative[i+1]+selection_tolerance]
        require(selected_index in possible, 'Wrong interval categorical selection')
        probability = float(normal.normal_masses((selected['lower']-mean)/sigma, (raw[axis]-mean)/sigma))/selected_mass
        error = abs(probability-draw['uniform_within_interval'])
        require(0 < draw['uniform_within_interval'] < 1 and error < 2e-6, 'Inverse conditional CDF differs')
        close(draw['returned_raw_coordinate'], raw[axis], 'Changed returned raw coordinate')
        return dict(inverse_error=error, endpoint_error=endpoint_error, geometry_certified=False,
                    interval_selection_ambiguous=len(possible) > 1,
                    interval_selection_cdf_tolerance=selection_tolerance)
