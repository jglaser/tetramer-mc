"""Single attempted-row weight audit for the v7 native-class-line normalizer.

No geometry, files, samples or Poisson clouds are evaluated here. The caller
must independently check physical pose/J, proposal q, original-q and capture,
and separately establish physical hard validity and final region classification.
This module verifies saved scalar predicates and estimator algebra, conditional
on that physical evidence. It never uses guide intervals as physical labels.

``validate_row(row, expected_draw=..., manifest=..., region=...)`` retains the
original draw ID. A streaming caller must visit every declared index once and
validate the original attempt journal; this function cannot establish file
completeness from one row. Returned z/h/pairs use -inf for zero contributions,
ready for the unchanged array statistics. Every row contributes one attempted
denominator, including hard-invalid, capture, original-q and shell rejections.

Derived role keys verify the declared key function/role separation, not actual
RNG consumption, stochastic independence, exact thinning, spatial Poisson
sampling or geometric correctness of recorded envelope bounds.
"""
from __future__ import annotations

import math
from typing import Any, Mapping

from scipy.special import logsumexp

import hard_free_line_physical_reference as physical
from native_class_line_physical_reference import (
    GUIDE_SCHEMA, MEASURE, PROPOSAL_KIND, SCHEMA, stream_key, validate_stream_contract,
)

require, close = physical.require, physical.close


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    require(type(value) is int and value >= minimum, name + ' must be an integer >= ' + str(minimum))
    return value


def _finite(value: Any, name: str, minimum: float | None = None) -> float:
    require(type(value) in (int, float), name + ' must be a number, not a Boolean')
    try:
        number = float(value)
    except OverflowError as error:
        raise ValueError('Unrepresentable ' + name) from error
    require(math.isfinite(number) and (minimum is None or number >= minimum), 'Invalid ' + name)
    return number


def validate_manifest(manifest: Mapping[str, Any], region: Mapping[str, Any]) -> None:
    """The subset of the immutable physical/stream contract used by row algebra."""
    require(manifest['schema'] == SCHEMA and manifest['guide_schema'] == GUIDE_SCHEMA
            and manifest['proposal_kind'] == PROPOSAL_KIND
            and manifest['proposal_density_measure'] == MEASURE, 'Wrong v7 physical class-guide contract')
    _integer(manifest['samples'], 'samples', 1)
    require(_integer(manifest['cloud_replicates'], 'cloud_replicates') == 2, 'Exactly two cloud replicas required')
    seed = _integer(manifest['seed'], 'seed')
    require(seed < 2**64, 'Seed exceeds u64 range')
    alpha = _finite(manifest['importance_uniform_probability'], 'uniform probability')
    require(0 < alpha <= 1, 'Uniform defensive probability must be in (0,1]')
    count = _integer(manifest['importance_component_count'], 'component count')
    require(alpha == 1 or count > 0, 'Gaussian branch lacks components')
    activity = _finite(manifest['activity'], 'activity', 0.)
    require(activity == _finite(region['activity'], 'region activity', 0.), 'Physical activity differs')
    intensity = _finite(manifest['lambda'], 'auxiliary intensity')
    ratio = _finite(manifest['lambda_ratio'], 'intensity ratio')
    require(intensity > 0 and ratio > 0, 'Auxiliary intensity and ratio must be positive')
    expected = activity * ratio if activity > 0 else 1.
    require(math.isfinite(expected), 'Unrepresentable auxiliary intensity')
    close(intensity, expected, 'Auxiliary intensity differs')
    radius = _finite(region['mahalanobis_radius'], 'outer latent radius')
    inner = _finite(region.get('minimum_mahalanobis_radius', 0.), 'inner latent radius', 0.)
    require(radius > inner, 'Empty latent shell')
    # This guide version is defined on the complete ball; do not silently
    # reinterpret its normalized uniform component as an annular proposal.
    require(inner == 0., 'Class-line guide requires complete latent ball')
    close(_finite(manifest['latent_radius'], 'manifest latent radius'), radius, 'Latent radius differs')
    require(_finite(manifest['minimum_latent_radius'], 'manifest inner radius') == inner,
            'Inner latent radius differs')
    lo = _finite(region['minimum_original_q'], 'minimum original q', 0.)
    hi = region.get('maximum_original_q')
    require(hi is None or _finite(hi, 'maximum original q', 0.) >= lo, 'Original-q bounds are reversed')
    for key, default in [('minimum_original_q', None), ('maximum_original_q', None),
                         ('minimum_original_q_inclusive', True), ('maximum_original_q_inclusive', True)]:
        require(manifest.get(key, default) == region.get(key, default), 'Changed original-q contract ' + key)
    for key in ('minimum_original_q_inclusive', 'maximum_original_q_inclusive'):
        require(type(region.get(key, True)) is bool and type(manifest.get(key, True)) is bool,
                'Original-q endpoint inclusion must be Boolean')
    validate_stream_contract(manifest)


def validate_row(row: Mapping[str, Any], *, expected_draw: int,
                 manifest: Mapping[str, Any], region: Mapping[str, Any]) -> dict[str, Any]:
    """Check one original attempted index; never renumber or drop invalid rows."""
    validate_manifest(manifest, region)
    _integer(expected_draw, 'expected_draw')
    _integer(row['draw'], 'draw')
    require(expected_draw < manifest['samples'] and row['draw'] == expected_draw,
            'Missing, reordered or out-of-allocation original draw ID')
    require(row['draw'] < 2**64, 'Draw exceeds u64 stream range')
    for key in ('hard_valid', 'capture_valid', 'region_valid', 'shell_valid'):
        require(type(row[key]) is bool, 'Missing Boolean support flag ' + key)
    require(not row['hard_valid'] or row['capture_valid'], 'Hard-valid flag outside capture')
    latent = row['latent']
    require(isinstance(latent, (list, tuple)) and len(latent) == 6, 'Six latent coordinates required')
    values = [_finite(v, 'latent coordinate') for v in latent]
    radius = _finite(row['latent_radius'], 'latent radius', 0.)
    close(radius, math.hypot(*values), 'Saved latent radius differs')
    support = physical.shell_contains(radius, region)
    require(row['shell_valid'] == support, 'Saved shell predicate differs')
    q = _finite(row['q'], 'original q', 0.)
    # The JSON schema represents an absent upper bound as either omitted or
    # null. Canonicalize that metadata only, never the row ID or sampled pose.
    q_region = dict(region)
    if q_region.get('maximum_original_q') is None:
        q_region.pop('maximum_original_q', None)
    require(row['region_valid'] == physical.q_contains(q, q_region), 'Saved original-q predicate differs')
    log_j = _finite(row['log_physical_jacobian'], 'log Jacobian')
    log_q = _finite(row['log_proposal_density'], 'log proposal density')
    label, component = row['proposal_branch'], row['proposal_component']
    if label == 'uniform-shell':
        require(support and component is None, 'Invalid uniform branch')
        branch, component_index = 0, -1
    else:
        require(label == 'native-class-line', 'Unknown v7 proposal branch')
        require(manifest['importance_uniform_probability'] < 1 and type(component) is int
                and 0 <= component < manifest['importance_component_count'], 'Invalid class Gaussian component')
        branch, component_index = 1, component
    require(row.get('selected_ray_fallback') is None, 'Unexpected ray metadata')
    draw = row['native_class_line_draw']
    require(type(draw['conditional']) is bool, 'Missing Boolean conditioner flag')
    conditional = draw['conditional']
    fallback = draw.get('fallback', False)
    require(type(fallback) is bool, 'Fallback flag must be Boolean')
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
    h = z = -math.inf
    pairs = [-math.inf, -math.inf]
    raw_points = 0
    maxima = dict(log_hard_weight=0., log_importance_weight=0., cloud_log_weight=0.)
    if not valid:
        require(row['log_hard_weight'] is None and row['log_importance_weight'] is None and not row['clouds'],
                'Invalid/exterior attempted draw must retain zero weight and no clouds')
    else:
        require(len(row['clouds']) == 2, 'Exactly two cloud records required per valid pose')
        h = _finite(row['log_hard_weight'], 'log hard weight')
        z = _finite(row['log_importance_weight'], 'log physical weight')
        expected_h = log_j - log_q
        require(math.isfinite(expected_h), 'Unrepresentable hard importance weight')
        close(h, expected_h, 'Hard weight is not full J/q')
        maxima['log_hard_weight'] = abs(h - expected_h)
        cloud_logs = []
        for cloud in row['clouds']:
            recorded = _finite(cloud['log_weight'], 'cloud log weight')
            expected = physical.cloud_log_weight(cloud, manifest['activity'], manifest['lambda'])
            cloud_logs.append(recorded)
            raw_points += cloud['raw_points']
            maxima['cloud_log_weight'] = max(maxima['cloud_log_weight'], abs(recorded - expected))
        for key in ('lower_volume', 'upper_volume', 'uncertain_volume',
                    'retained_cells', 'created_cells', 'certified_cells'):
            require(row['clouds'][0][key] == row['clouds'][1][key], 'Cloud replica envelope differs: ' + key)
        pairs = [h + value for value in cloud_logs]
        require(all(math.isfinite(v) for v in pairs), 'Unrepresentable weighted cloud log')
        expected_z = float(logsumexp(pairs) - math.log(2))
        close(z, expected_z, 'Two-cloud arithmetic mean differs')
        maxima['log_importance_weight'] = abs(z - expected_z)
    counters = dict(
        attempted=1, capture_rejected=int(not row['capture_valid']),
        hard_rejected=int(row['capture_valid'] and not row['hard_valid']),
        region_rejected=int(row['hard_valid'] and not row['region_valid']),
        shell_rejected=int(not support), contributing=int(valid),
        conditioned_draws=int(conditional), fallback_draws=int(fallback))
    return dict(draw=expected_draw, z=z, h=h, pairs=pairs, counters=counters,
                raw_points=raw_points, role_keys=role_keys, branch=branch,
                component=component_index, support=support, maxima=maxima)
