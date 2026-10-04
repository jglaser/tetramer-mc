"""Explicit v6 hard-free physical input, conditional on saved H intervals.

No atom trees, overlap queries, native observer, pose draws or clouds are created
here. The producer schema and hard-free trace remain v6 throughout. This bridge
checks all-attempt accounting and the complete proposal algebra; it neither
certifies the saved H intervals nor supplies independent physical labels.

The caller must stream every original row and attempt-journal entry exactly
once, retain zero rows, and recheck bound files after reading. ``provenance``
authenticates files, but deliberately does not load the sample/journal streams.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np
from scipy.linalg import solve
from scipy.special import logsumexp
from scipy.spatial.transform import Rotation

import hard_free_line_physical_reference as physical
import native_class_line_algebra_reference as numeric
from native_class_line_weight_row import _finite, _integer, stream_key

SCHEMA, GUIDE_SCHEMA = physical.SCHEMA, physical.GUIDE_SCHEMA
PROPOSAL_KIND, MEASURE = physical.PROPOSAL_KIND, physical.MEASURE
ATTEMPT_JOURNAL = 'attempts.jsonl; begin record before each draw; no retries'
AUDIT_SCOPE = 'v6 proposal algebra conditional on saved hard-free intervals; no geometry or physical labels certified'
require, close = physical.require, physical.close
line, normal = physical.line, physical.line.normal
_V7_MANIFEST = ('compiled_native', 'density_trace_contract', 'random_stream_contract', 'class_scope', 'failure_trace_contract')
_CLASS_TRACE = ('native_class_line_draw', 'native_class_line_density')


def validate_manifest(manifest, region):
    """Validate the real v6 contract, without inserting v7 metadata."""
    require(manifest['schema'] == SCHEMA and manifest['guide_schema'] == GUIDE_SCHEMA
            and manifest['proposal_kind'] == PROPOSAL_KIND
            and manifest['proposal_density_measure'] == MEASURE, 'Wrong v6 hard-free physical contract')
    require(not any(key in manifest for key in _V7_MANIFEST), 'v6 manifest acquired v7 contracts')
    _integer(manifest['samples'], 'samples', 1)
    require(manifest['samples'] <= 2**64, 'Allocation exceeds u64 draw range')
    require(_integer(manifest['cloud_replicates'], 'cloud_replicates') == 2, 'Exactly two cloud replicas required')
    require(_integer(manifest['seed'], 'seed') < 2**64, 'Seed exceeds u64 range')
    alpha = _finite(manifest['importance_uniform_probability'], 'uniform probability')
    require(0 < alpha <= 1, 'Uniform defensive probability must be in (0,1]')
    count = _integer(manifest['importance_component_count'], 'component count')
    require(alpha == 1 or count > 0, 'Gaussian branch lacks components')
    activity = _finite(manifest['activity'], 'activity', 0.)
    require(activity == _finite(region['activity'], 'region activity', 0.), 'Physical activity differs')
    intensity, ratio = (_finite(manifest[key], key) for key in ('lambda', 'lambda_ratio'))
    require(intensity > 0 and ratio > 0, 'Auxiliary intensity and ratio must be positive')
    expected = activity*ratio if activity > 0 else 1.
    require(math.isfinite(expected), 'Unrepresentable auxiliary intensity')
    close(intensity, expected, 'Auxiliary intensity differs')
    radius = _finite(region['mahalanobis_radius'], 'outer latent radius')
    inner = _finite(region.get('minimum_mahalanobis_radius', 0.), 'inner latent radius', 0.)
    require(radius > inner and inner == 0., 'Hard-free line guide requires a complete latent ball')
    close(_finite(manifest['latent_radius'], 'manifest latent radius'), radius, 'Latent radius differs')
    require(_finite(manifest['minimum_latent_radius'], 'manifest inner radius') == inner, 'Inner latent radius differs')
    lo = _finite(region['minimum_original_q'], 'minimum original q', 0.)
    hi = region.get('maximum_original_q')
    require(hi is None or _finite(hi, 'maximum original q', 0.) >= lo, 'Original-q bounds are reversed')
    for key, default in [('minimum_original_q', None), ('maximum_original_q', None),
                         ('minimum_original_q_inclusive', True), ('maximum_original_q_inclusive', True)]:
        require(manifest.get(key, default) == region.get(key, default), 'Changed original-q contract '+key)
    for key in ('minimum_original_q_inclusive', 'maximum_original_q_inclusive'):
        require(type(region.get(key, True)) is bool and type(manifest.get(key, True)) is bool,
                'Original-q endpoint inclusion must be Boolean')


def validate_row(row, *, expected_draw, manifest, region):
    """Same accounting outputs as v7, with genuine v6 branch/trace admission."""
    validate_manifest(manifest, region)
    _integer(expected_draw, 'expected_draw'); _integer(row['draw'], 'draw')
    require(expected_draw < manifest['samples'] and row['draw'] == expected_draw,
            'Missing, reordered or out-of-allocation original draw ID')
    require(row['draw'] < 2**64, 'Draw exceeds u64 stream range')
    require(not any(key in row for key in _CLASS_TRACE), 'v6 row acquired class-line traces')
    for key in ('hard_valid', 'capture_valid', 'region_valid', 'shell_valid'):
        require(type(row[key]) is bool, 'Missing Boolean support flag '+key)
    require(not row['hard_valid'] or row['capture_valid'], 'Hard-valid flag outside capture')
    latent = row['latent']
    require(isinstance(latent, (list, tuple)) and len(latent) == 6, 'Six latent coordinates required')
    radius = _finite(row['latent_radius'], 'latent radius', 0.)
    close(radius, math.hypot(*[_finite(v, 'latent coordinate') for v in latent]), 'Saved latent radius differs')
    support = physical.shell_contains(radius, region)
    require(row['shell_valid'] == support, 'Saved shell predicate differs')
    q_region = dict(region)
    if q_region.get('maximum_original_q') is None: q_region.pop('maximum_original_q', None)
    require(row['region_valid'] == physical.q_contains(_finite(row['q'], 'original q', 0.), q_region),
            'Saved original-q predicate differs')
    log_j = _finite(row['log_physical_jacobian'], 'log Jacobian')
    log_q = _finite(row['log_proposal_density'], 'log proposal density')
    label, component = row['proposal_branch'], row['proposal_component']
    if label == 'uniform-shell':
        require(support and component is None, 'Invalid uniform branch')
        branch, component_index = 0, -1
    else:
        require(label == 'hard-free-line', 'Unknown v6 proposal branch')
        require(manifest['importance_uniform_probability'] < 1 and type(component) is int
                and 0 <= component < manifest['importance_component_count'], 'Invalid hard-free Gaussian component')
        branch, component_index = 1, component
    require(row.get('selected_ray_fallback') is None, 'Unexpected ray metadata')
    draw = row['hard_free_line_draw']
    require(type(draw) is dict and type(draw['conditional']) is bool, 'Missing Boolean conditioner flag')
    require(type(row['hard_free_line_density']) is dict, 'Missing hard-free density trace')
    conditional, fallback = draw['conditional'], draw.get('fallback', False)
    require(type(fallback) is bool, 'Fallback flag must be Boolean')
    require(not any(key in draw for key in ('channel', 'class', 'orthant', 'width_index', 'width_A')),
            'Hard-free draw acquired class/contact conditioning')
    if conditional:
        require(branch == 1 and type(draw['component']) is int and draw['component'] == component,
                'Conditional draw lineage differs')
    else:
        require(not fallback, 'Unconditioned draw cannot be a conditional fallback')
    role_keys = [stream_key(manifest['seed'], expected_draw, replica, role)
                 for replica, role in ((0, 'latent'), (0, 'cloud'), (1, 'cloud'))]
    require(len(set(role_keys)) == 3, 'Within-row proposal/cloud role key collision')
    valid = row['hard_valid'] and row['capture_valid'] and row['region_valid'] and support
    require(type(row['clouds']) is list, 'Cloud records must be a list')
    h = z = -math.inf; pairs = [-math.inf, -math.inf]; raw_points = 0
    maxima = dict(log_hard_weight=0., log_importance_weight=0., cloud_log_weight=0.)
    if not valid:
        require(row['log_hard_weight'] is None and row['log_importance_weight'] is None and not row['clouds'],
                'Invalid/exterior attempted draw must retain zero weight and no clouds')
    else:
        require(len(row['clouds']) == 2, 'Exactly two cloud records required per valid pose')
        h = _finite(row['log_hard_weight'], 'log hard weight'); z = _finite(row['log_importance_weight'], 'log physical weight')
        expected_h = log_j-log_q
        require(math.isfinite(expected_h), 'Unrepresentable hard importance weight')
        close(h, expected_h, 'Hard weight is not full J/q'); maxima['log_hard_weight'] = abs(h-expected_h)
        cloud_logs = []
        for cloud in row['clouds']:
            recorded = _finite(cloud['log_weight'], 'cloud log weight')
            expected = physical.cloud_log_weight(cloud, manifest['activity'], manifest['lambda'])
            cloud_logs.append(recorded); raw_points += cloud['raw_points']
            maxima['cloud_log_weight'] = max(maxima['cloud_log_weight'], abs(recorded-expected))
        for key in ('lower_volume', 'upper_volume', 'uncertain_volume', 'retained_cells', 'created_cells', 'certified_cells'):
            require(row['clouds'][0][key] == row['clouds'][1][key], 'Cloud replica envelope differs: '+key)
        pairs = [h+value for value in cloud_logs]
        require(all(math.isfinite(v) for v in pairs), 'Unrepresentable weighted cloud log')
        expected_z = float(logsumexp(pairs)-math.log(2))
        close(z, expected_z, 'Two-cloud arithmetic mean differs'); maxima['log_importance_weight'] = abs(z-expected_z)
    counters = dict(attempted=1, capture_rejected=int(not row['capture_valid']),
        hard_rejected=int(row['capture_valid'] and not row['hard_valid']),
        region_rejected=int(row['hard_valid'] and not row['region_valid']), shell_rejected=int(not support),
        contributing=int(valid), conditioned_draws=int(conditional), fallback_draws=int(fallback))
    return dict(draw=expected_draw, z=z, h=h, pairs=pairs, counters=counters, raw_points=raw_points,
                role_keys=role_keys, branch=branch, component=component_index, support=support, maxima=maxima)


class AlgebraLaw:
    """Only chart, Gaussian and saved-interval algebra; no shape is accepted.

    The six pure chart/Normal methods below are shared numerical helpers, not a
    v7 initializer or schema adapter. No class channels are constructed.
    """
    raw = numeric.AlgebraLaw.raw
    inverse_raw = numeric.AlgebraLaw.inverse_raw
    decode = numeric.AlgebraLaw.decode
    inverse_pose = numeric.AlgebraLaw.inverse_pose
    gaussian_logs = numeric.AlgebraLaw.gaussian_logs
    conditional = numeric.AlgebraLaw.conditional

    def __init__(self, region, guide, config):
        self.region, self.guide, self.config = map(copy.deepcopy, (region, guide, config))
        require(guide['schema'] == GUIDE_SCHEMA, 'Wrong hard-free guide schema')
        require(not any(k in guide for k in ('contact_widths_A', 'contact_neighbor_indices', 'class_channels',
                                            'compiled_native', 'class_scope')), 'Hard-free guide acquired contact/class constraints')
        require(region.get('minimum_mahalanobis_radius', 0.) == 0., 'Only complete latent balls are supported')
        chart = region['gaussian_chart']
        require(len(chart['means']) == len(chart['anchors']) == len(chart['covariances']) == 1,
                'Expected one frozen coordinate chart')
        self.m0 = numeric.finite_array(chart['means'][0], (6,), 'chart mean')
        self.L0, self.chart_factor_validation = numeric.line.chart_fp64_factor(chart['covariances'][0])
        self.ell = _finite(chart['angular_length'], 'angular length')
        self.radius = _finite(region['mahalanobis_radius'], 'latent radius')
        require(self.ell > 0 and self.radius > 0, 'Invalid chart angular length or radius')
        self.anchor, self.fixed = copy.deepcopy(chart['anchors'][0]), copy.deepcopy(region['fixed_neighbor'])
        self.anchor_position = numeric.finite_array(self.anchor['position'], (3,), 'chart anchor position')
        self.anchor_rotation = numeric.proper_rotation(self.anchor['rotation'], 'chart anchor')
        self.fixed_position, self.Rf = numeric.line.pose_arrays(self.fixed)
        self.capture_center = numeric.finite_array(config['capture_center'], (3,), 'capture center')
        self.capture_radius = _finite(config['capture_radius'], 'capture radius')
        require(self.capture_radius > 0, 'Invalid capture radius')
        self.logvolume = 3*math.log(math.pi)+6*math.log(self.radius)-math.log(6)
        self.alpha = _finite(guide['defensive_uniform_shell_probability'], 'uniform probability')
        self.beta = _finite(guide['conditional_probability'], 'conditional probability')
        self.floor = _finite(guide['minimum_conditional_mass'], 'conditional mass floor')
        self.axes = list(guide['raw_translation_axes'])
        require(0 < self.alpha <= 1 and 0 <= self.beta <= 1 and 0 < self.floor < 1, 'Invalid mixture controls')
        require(self.axes and len(set(self.axes)) == len(self.axes)
                and all(type(a) is int and a in (0, 1, 2) for a in self.axes), 'Invalid translation axes')
        components = guide['gaussian_components']; n = len(components)
        require(n > 0 or self.alpha == 1., 'Missing Gaussian mixture')
        weights = np.asarray([_finite(c['weight'], 'Gaussian weight') for c in components])
        require((weights > 0).all() and (not n or math.isfinite(weights.sum())), 'Invalid Gaussian weights')
        self.weights = weights/weights.sum() if n else weights
        self.means = numeric.finite_array(np.asarray([c['mean'] for c in components]).reshape((n, 6)), (n, 6), 'Gaussian means')
        covariance = numeric.finite_array(np.asarray([c['covariance'] for c in components]).reshape((n, 6, 6)),
                                          (n, 6, 6), 'Gaussian covariances')
        require(np.allclose(covariance, covariance.transpose(0, 2, 1), atol=1e-12, rtol=1e-12), 'Asymmetric Gaussian covariance')
        self.lowers = np.linalg.cholesky(covariance)
        self.normalizers = -3*math.log(2*math.pi)-np.log(np.diagonal(self.lowers, axis1=1, axis2=2)).sum(axis=1)
        self.rawmeans = self.m0+self.means@self.L0.T
        rawcov = np.einsum('ij,kjl,ml->kim', self.L0, covariance, self.L0)
        require(np.isfinite(self.rawmeans).all() and np.isfinite(rawcov).all(), 'Unrepresentable raw Gaussian law')
        self.conditionals = {}
        for axis in self.axes:
            others = [j for j in range(6) if j != axis]; coefficients, sigmas = [], []
            for matrix in rawcov:
                coefficient = solve(matrix[np.ix_(others, others)], matrix[others, axis], assume_a='pos')
                variance = matrix[axis, axis]-matrix[axis, others]@coefficient
                require(np.isfinite(coefficient).all() and math.isfinite(variance) and variance > 0, 'Invalid conditional variance')
                coefficients.append(coefficient); sigmas.append(math.sqrt(variance))
            self.conditionals[axis] = (others, np.asarray(coefficients).reshape((n, 5)), np.asarray(sigmas))

    @staticmethod
    def chord(origin, direction, radius):
        # v6 keeps only positive-length chords; a tangent falls back.
        aa = direction@direction
        require(math.isfinite(aa) and aa > 0, 'Invalid chord direction')
        center = -(origin@direction)/aa; nearest = origin+center*direction
        residual = radius*radius-nearest@nearest
        require(math.isfinite(center) and math.isfinite(residual), 'Unrepresentable chord')
        if residual <= 0: return None
        half = math.sqrt(residual/aa); result = [center-half, center+half]
        require(np.isfinite(result).all(), 'Unrepresentable chord endpoints')
        return result

    def line_frame(self, u, axis):
        require(type(axis) is int and axis in self.axes, 'Invalid saved translation axis')
        raw = self.raw(u); raw[axis] = 0.
        u0 = self.inverse_raw(raw); du = numeric.line.scalar_chart_solve(self.L0, np.eye(6)[axis])
        _, position, rotation, _ = self.decode(u0); direction = self.Rf[:, axis]
        r4 = self.chord(u0, du, self.radius)
        if r4 is None: return dict(axis=axis, empty_reason='no_R4_chord')
        capture = self.chord(position-self.capture_center, direction, self.capture_radius)
        if capture is None: return dict(axis=axis, empty_reason='no_capture_chord')
        segment = [max(r4[0], capture[0]), min(r4[1], capture[1])]
        if segment[0] >= segment[1]: return dict(axis=axis, empty_reason='disjoint_chords')
        return dict(axis=axis, segment=segment, direction=direction.tolist(), origin=dict(position=position.tolist(),
            orientation=Rotation.from_matrix(rotation).as_quat()[[3, 0, 1, 2]].tolist()))

    def _saved_axis(self, u, saved):
        require(type(saved) is dict, 'Saved hard-free axis must be an object')
        require(not any(k in saved for k in ('channels', 'native_intervals', 'exclusion_contact_intervals', 'widths')),
                'Hard-free axis acquired contact/class intervals')
        frame = self.line_frame(u, saved['axis'])
        intervals = saved['hard_free_intervals']
        require(type(intervals) is list, 'Hard-free interval union must be a list')
        for interval in intervals:
            _finite(interval['lower'], 'interval lower'); _finite(interval['upper'], 'interval upper')
        normal.validate_intervals(intervals, frame.get('segment'))
        require(('segment' in saved) == ('segment' in frame), 'Missing/unexpected R4/capture segment')
        if 'segment' not in frame:
            require(saved.get('empty_reason') == frame['empty_reason'] and not intervals,
                    'Changed empty-line reason or nonempty intervals outside chords')
            require('origin' not in saved and 'direction' not in saved, 'Unexpected empty-line frame')
        else:
            close(saved['segment'], frame['segment'], 'Changed R4/capture segment')
            close(saved['direction'], frame['direction'], 'Changed line direction')
            p, r = numeric.line.pose_arrays(saved['origin']); ep, er = numeric.line.pose_arrays(frame['origin'])
            close(p, ep, 'Changed line origin'); close(r, er, 'Changed line orientation')
            reason = saved.get('empty_reason')
            require(reason in (None, 'no_positive_hard_free_length'), 'Changed empty-line reason')
            require((reason == 'no_positive_hard_free_length') == (not intervals), 'Inconsistent zero-length hard-free trace')
            require(not intervals or any(i['lower'] < i['upper'] for i in intervals), 'Zero-measure hard-free union must fall back')
        return frame

    def evaluate_saved(self, u, details):
        require(type(details) is dict and not any(k in details for k in ('trace_format', 'class_scope', 'component_mixture_multipliers')),
                'Invalid v6 density trace')
        raw, position, rotation, jac = self.decode(u)
        if self.beta == 0 or self.alpha == 1:
            require(details == {'conditioning_disabled': True}, 'Changed disabled-conditioning trace')
        else:
            require(type(details['axes']) is list and [a['axis'] for a in details['axes']] == self.axes,
                    'Missing/reordered density axes')
            _integer(details['component_branches'], 'component branches')
            _integer(details['fallback_component_branches'], 'fallback component branches')
            _finite(details['baseline_log_density'], 'baseline log density')
            for saved in details['axes']:
                self._saved_axis(u, saved)
                if saved['axis_log_proposal_density'] is not None: _finite(saved['axis_log_proposal_density'], 'axis log density')
        result = physical.compact_density(self, u, details, geometry='saved-intervals')
        result.update(raw_coordinates=raw, position=position, rotation=rotation, log_physical_jacobian=jac,
            geometry_certified=False, audit_scope=AUDIT_SCOPE, component_branches=details.get('component_branches', 0),
            fallback_component_branches=details.get('fallback_component_branches', 0))
        return result

    def verify_draw(self, row, result):
        draw = row['hard_free_line_draw']; u = row['latent']
        numeric.finite_array(draw['original_latent'], (6,), 'original latent coordinate')
        if draw['conditional']:
            require(self.beta > 0 and self.alpha < 1, 'Conditional draw on disabled proposal')
            self._saved_axis(u, draw['geometry'])
            require(type(draw['axis']) is int and draw['axis'] == draw['geometry']['axis'], 'Selected geometry axis differs')
            for key in ('conditional_mean', 'conditional_sigma', 'conditional_mass'): _finite(draw[key], key)
            require(draw['conditional_sigma'] > 0 and 0 <= draw['conditional_mass'] <= 1+1e-12,
                    'Invalid selected conditional law')
            if not draw['fallback']:
                selected = draw['selected_interval']; normal.validate_intervals([selected])
                axis = next(a for a in result['axes'] if a['axis'] == draw['axis'])
                matching = [a for a in axis['intervals'] if all(abs(selected[k]-a[k]) < 1e-9 for k in ('lower', 'upper'))]
                require(matching and any(all(selected[k] == a[k] for k in ('lower_closed', 'upper_closed')) for a in matching),
                        'Selected interval endpoint inclusion differs')
                for key in ('selected_interval_mass', 'uniform_interval_selection', 'uniform_within_interval', 'returned_raw_coordinate'):
                    _finite(draw[key], key)
        else:
            require(not any(k in draw for k in ('geometry', 'axis', 'component', 'selected_interval')),
                    'Unconditioned draw acquired conditional metadata')
        return physical.audit_draw(row, self, result)


def _read(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'Duplicate JSON key '+key); result[key] = value
        return result
    def constant(value): raise ValueError('Nonfinite JSON constant '+value)
    return json.loads(Path(path).read_bytes(), object_pairs_hook=pairs, parse_constant=constant)


def _digest(value, label):
    require(type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value), 'Invalid SHA256 '+label)


def provenance(root, bind):
    """Authenticate completed v6 inputs; no native witness is implied or made."""
    root = Path(root)
    manifest, summary = (_read(bind(root/name)) for name in ('manifest.json', 'summary.json'))
    require(summary['complete'] is True and summary['manifest'] == manifest and not (root/'failure.json').exists(),
            'Incomplete or failed physical population')
    require(manifest.get('resume_supported') is False and manifest.get('attempt_journal') == ATTEMPT_JOURNAL,
            'Unsupported v6 continuation/attempt-journal contract')
    _digest(manifest['executable_sha256'], 'producer executable')
    for name, key in [('input-config.json', 'config_sha256'), ('region.json', 'region_sha256'), ('shape.json', 'shape_sha256'),
                      ('importance-guide.json', 'importance_guide_sha256'), ('source-bundle.json', 'source_bundle_sha256')]:
        _digest(manifest[key], key); bind(root/'provenance'/name, manifest[key])
    region, guide, config = [_read(root/'provenance'/name) for name in ('region.json', 'importance-guide.json', 'input-config.json')]
    validate_manifest(manifest, region)
    require(guide['region_sha256'] == manifest['region_sha256'] and region['shape_sha256'] == manifest['shape_sha256']
            and region['gaussian_chart']['shape_sha256'] == manifest['shape_sha256'], 'Guide/chart/region/shape binding differs')
    fixed = region.get('physical_fixed_neighbors', [region['fixed_neighbor']])
    require(fixed == config['fixed_poses'] == manifest['physical_fixed_neighbors']
            and region['fixed_neighbor'] == manifest['chart_anchor'], 'Changed physical scaffold/chart anchor')
    for rk, ck in [('capture_center', 'capture_center'), ('capture_radius', 'capture_radius'), ('depletant_radius', 'depletant_radius'),
                   ('activity', 'reservoir_density'), ('physical_metric', 'metadata')]:
        require(region[rk] == config[ck], 'Changed physical field '+rk)
    law = AlgebraLaw(region, guide, config)
    require(manifest['importance_uniform_probability'] == law.alpha and manifest['importance_component_count'] == len(law.weights),
            'Changed guide mixture')
    close(_finite(manifest['log_latent_ball_volume'], 'ball log volume'), law.logvolume, 'Ball volume differs')
    close(_finite(manifest['log_latent_shell_volume'], 'shell log volume'), law.logvolume, 'Shell volume differs')
    require(type(summary['samples']) is int and summary['samples'] == manifest['samples'], 'Changed unconditional denominator')
    for name in ('samples', 'attempts'):
        _digest(summary[name+'_sha256'], name); bind(root/(name+'.jsonl'), summary[name+'_sha256'])
    return manifest, summary, region, config, law, None


def check_row(row, index, manifest, region, config, law):
    accounting = validate_row(row, expected_draw=index, manifest=manifest, region=region)
    u = numeric.finite_array(row['latent'], (6,), 'latent coordinate')
    result = law.evaluate_saved(u, row['hard_free_line_density'])
    position, rotation, jac = result['position'], result['rotation'], result['log_physical_jacobian']
    saved_position, saved_rotation = numeric.line.pose_arrays(row['pose'])
    close(saved_position, position, 'Physical position differs'); close(saved_rotation, rotation, 'Physical orientation differs')
    backmap = law.inverse_pose(saved_position, saved_rotation)
    close(backmap, u, 'Independent inverse chart differs'); close(row['backmapped_latent'], backmap, 'Saved inverse chart differs')
    close(_finite(row['backmapped_radius'], 'inverse radius', 0.), np.linalg.norm(backmap), 'Inverse radius differs')
    close(row['log_physical_jacobian'], jac, 'Physical Jacobian differs')
    close(_finite(row['physical_jacobian'], 'physical Jacobian', 0.), math.exp(jac), 'Exponentiated Jacobian differs')
    require(math.isfinite(result['log_density']), 'Generated row has zero/nonfinite proposal density')
    close(row['log_proposal_density'], result['log_density'], 'Complete proposal density differs', atol=2e-7, rtol=1e-11)
    if row['proposal_branch'] == 'hard-free-line' and law.beta == 1.:
        require(row['hard_free_line_draw']['conditional'] is True, 'Gaussian beta=1 draw must record conditioning including fallback')
    inverse_error = law.verify_draw(row, result)
    capture = bool(np.linalg.norm(position-law.capture_center) <= law.capture_radius)
    require(row['capture_valid'] == capture, 'Capture predicate differs')
    original_q = physical.native.native_q(region['physical_metric'], row['pose'])
    close(row['q'], original_q, 'Original physical q differs')
    q_region = dict(region)
    if q_region.get('maximum_original_q') is None: q_region.pop('maximum_original_q', None)
    require(row['region_valid'] == physical.q_contains(original_q, q_region), 'Original q-window differs')
    require(row['shell_valid'] == physical.shell_contains(float(np.linalg.norm(u)), region), 'Independent shell predicate differs')
    hard_memberships = []; indeterminate_zero_measure_axes = 0
    for axis in result['axes']:
        coordinate = result['raw_coordinates'][axis['axis']]
        if 'segment' in axis and axis['segment'][0] <= coordinate <= axis['segment'][1]:
            # Genuine v6 emits [] when H has zero total length, discarding
            # isolated feasible tangencies from the proposal trace. Physical
            # validity at those points belongs to the independent endpoint
            # audit; an empty density union does not prove hard invalidity.
            if axis.get('empty_reason') == 'no_positive_hard_free_length':
                indeterminate_zero_measure_axes += 1
                continue
            hard_memberships.append(line.contains(axis['intervals'], coordinate))
    if hard_memberships:
        require(all(value == hard_memberships[0] for value in hard_memberships), 'Saved H membership differs across axes')
        require(hard_memberships[0] == row['hard_valid'], 'Saved H membership differs from recorded physical hard flag')
    errors = dict(accounting['maxima'], inverse_cdf=inverse_error, interval_endpoints=result['interval_error'],
        log_proposal_density=abs(row['log_proposal_density']-result['log_density']),
        log_physical_jacobian=abs(row['log_physical_jacobian']-jac), latent_backmap=float(np.max(abs(backmap-u))),
        original_q=abs(row['q']-original_q))
    return accounting, errors, dict(hard_membership_axes=len(hard_memberships), interval_selection_ambiguous=0,
        indeterminate_zero_measure_axes=indeterminate_zero_measure_axes,
        component_branches=result['component_branches'], fallback_component_branches=result['fallback_component_branches'])
