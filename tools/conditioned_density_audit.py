#!/usr/bin/env python3
"""Split numerical audit of a frozen conditional-line proposal density.

This module draws no poses/clouds and changes no sampler or physical estimator.
``audit_conditioned_density(record, expected, guide)`` takes the complete saved
Rust ``guide_density`` record, an independently evaluated PhysicalGuideDensity,
and its PhysicalHardFreeLineGuide. The caller must already bind their pose,
shape, guide, sources, and executable by the existing provenance checks.

Three separate claims are checked:
(1) Same-input density arithmetic on the SAVED interval law, at the original
    2e-8 absolute + 2e-11 relative log tolerance; J and coordinates stay checked.
(2) Independently rebuilt interval topology, endpoint flags, geometry and the
    existing 1e-9 Angstrom absolute endpoint tolerance.
(3) Conditioning sensitivity: intersection/hull unions enclose the two supplied
    interval families' Normal masses. With fixed support/fallback branches,
    positivity and monotonicity of 1/M bound their complete mixture densities.
    A 2e-7 absolute cap on every log-density-envelope width is an ADDITIONAL
    guard, inherited from the proposal-only physical-integration reference.

The envelope bounds differences between two supplied floating-point geometries;
it is NOT a certified enclosure of real-arithmetic geometry or a global error
bound on normalization, equilibrium, physical weights, or unseen poses. Normal
mass arithmetic reuses the independent stable quadrature/log-CDF helper. The
small padding below covers arithmetic comparison, not uncertain geometry.

Compact production traces do not store individual component fallback flags.
Their per-component decisions are independently reconstructed for BOTH interval
families and checked, and the stored aggregate count is checked separately.
When a full trace supplies component records those are also checked directly.
Generation/scoring consistency and inverse-CDF tests remain caller obligations;
this helper never replaces or relaxes them.
"""
from __future__ import annotations

import math
from numbers import Real
from collections.abc import Mapping
import numpy as np
from scipy.special import logsumexp
import analyze_contact_line_audit as normal
import hard_free_line_reference as line

LOG_ATOL = 2e-8
LOG_RTOL = 2e-11
ENDPOINT_ATOL = 1e-9
LOG_ENVELOPE_CAP = 2e-7
ENVELOPE_ROUNDING = 2e-11
SCHEMA = 'split-conditioned-density-numerical-audit-v1'
SCOPE = ('Same-saved-law arithmetic, independent floating-point interval geometry, '
         'and propagated conditioning checks. No global floating-point, '
         'normalization, sampling-convergence or physical certificate.')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def _log(value):
    if value is None:
        return -math.inf
    require(isinstance(value, Real) and not isinstance(value, (bool, np.bool_))
            and (math.isfinite(value) or value == -math.inf),
            'Invalid recorded logarithm')
    return float(value)


def _log_error(actual, expected, label, *, atol=LOG_ATOL):
    actual, expected = _log(actual), _log(expected)
    if actual == expected == -math.inf:
        return 0.
    require(math.isfinite(actual) and math.isfinite(expected), label + ': support differs')
    error = abs(actual-expected)
    require(error <= atol+LOG_RTOL*max(abs(actual), abs(expected)), label)
    return error


def _vector(value, shape, label):
    value = np.asarray(value, dtype=float)
    require(value.shape == shape and np.isfinite(value).all(), label)
    return value


def _inside(recon, u):
    return bool(np.all(abs(u) <= recon.radius) and u@u <= recon.radius**2)


def _uniform(recon, inside):
    return math.log(recon.alpha)-recon.logvolume if inside else -math.inf


def _compose(recon, gaussian_logs, uniform, factors):
    if recon.alpha == 1.:
        return uniform
    factors = np.asarray(factors, float)
    correction = 1-recon.beta+recon.beta*factors
    require(np.isfinite(correction).all() and np.all(correction >= 0),
            'Invalid mixture correction')
    active = correction > 0
    require(np.isfinite(gaussian_logs[active]).all(),
            'Unrepresentable positive Gaussian component; not a structural zero')
    return float(np.logaddexp(uniform, math.log1p(-recon.alpha)
                 + logsumexp(gaussian_logs[active]+np.log(correction[active]))))


def _multipliers(masses, fallback, allowed):
    answer = np.ones(len(masses))
    answer[~fallback] = 1/masses[~fallback] if allowed else 0.
    return answer


def interval_envelope(saved, independent):
    """Enclose matched interval sets without double-counting intersecting hulls."""
    require(len(saved) == len(independent), 'Different interval topology')
    inner, outer = [], []
    for a, b in zip(saved, independent):
        require(a['lower_closed'] == b['lower_closed'] and
                a['upper_closed'] == b['upper_closed'], 'Different endpoint flags')
        require(max(abs(a['lower']-b['lower']), abs(a['upper']-b['upper'])) <= ENDPOINT_ATOL,
                'Different hard-free interval endpoint')
        lo, hi = max(a['lower'], b['lower']), min(a['upper'], b['upper'])
        if lo < hi:
            inner.append(line.interval(lo, hi, a['lower_closed'], a['upper_closed']))
        elif lo == hi and a['lower_closed'] and a['upper_closed']:
            inner.append(line.interval(lo, hi))
        outer.append(line.interval(min(a['lower'], b['lower']), max(a['upper'], b['upper']),
                                   a['lower_closed'], a['upper_closed']))
    return line.union(inner), line.union(outer)


def _enclosure(low, high, values, label):
    low, high = _log(low), _log(high)
    if high == -math.inf:
        require(low == -math.inf and all(_log(v) == -math.inf for v in values),
                label + ': unresolved structural support')
        return 0.
    require(math.isfinite(low) and math.isfinite(high), label + ': nonfinite bound')
    width = high-low
    require(width >= -ENVELOPE_ROUNDING, label + ': inverted bounds')
    require(width <= LOG_ENVELOPE_CAP, label + ': log-envelope cap exceeded')
    for value in values:
        value = _log(value)
        require(math.isfinite(value) and low-ENVELOPE_ROUNDING <= value <= high+ENVELOPE_ROUNDING,
                label + ': reconstructed density outside envelope')
    return max(0., width)


def audit_conditioned_density(record, expected, guide, *, full_vessel=False):
    """Audit one already-reconstructed pose; return separate q values and maxima.

    ``record`` must be the complete Rust guide_density object, not a bare trace.
    Returned log_physical_density_saved_intervals uses the independently checked
    Jacobian. It remains the same saved-law q used by production, not a newly
    substituted independent-geometry importance weight.
    """
    require(isinstance(record, Mapping), 'Missing complete density record')
    keys = {'latent', 'in_reference_ball', 'log_latent_density', 'log_physical_jacobian',
            'log_physical_density', 'structural_zero', 'hard_free_line_density'}
    require(keys <= record.keys(), 'Incomplete density/J/input record')
    for key in ['structural_zero', 'in_reference_ball']:
        require(type(record[key]) is bool and record[key] == getattr(expected, key),
                'Density support certificate differs: '+key)
    require(type(full_vessel) is bool, 'Invalid audit-domain option')
    maxima = dict(same_input_log_error=0., coordinate_log_error=0., interval_endpoint_error=0.,
                  independent_geometry_log_difference=0., log_envelope_width=0.,
                  latent_coordinate_error=0., raw_coordinate_error=0., jacobian_error=0.)
    result = dict(schema=SCHEMA, complete=False, scope=SCOPE, axes=[], maxima=maxima,
                  per_component_decisions='independent reconstruction; direct stored flags when available')
    if expected.latent is None:
        require(expected.structural_zero and record['latent'] is None
                and record['hard_free_line_density'] is None
                and record['log_latent_density'] is None
                and record['log_physical_jacobian'] is None, 'Exact seam trace differs')
        _log_error(record['log_physical_density'], -math.inf, 'Exact seam density differs')
        result.update(complete=True, log_latent_density_saved_intervals=None,
                      log_latent_density_independent_geometry=None,
                      log_physical_density_saved_intervals=None,
                      log_physical_density_independent_geometry=None)
        return result
    recon = guide.recon
    u = _vector(expected.latent, (6,), 'Invalid independently reconstructed coordinates')
    saved_u = _vector(record['latent'], (6,), 'Invalid saved coordinates')
    maxima['latent_coordinate_error'] = float(np.max(abs(u-saved_u)))
    if full_vessel:
        # Match the existing all-pose vessel coordinate check. The strict
        # density/J checks below still reject amplified coordinate errors.
        require(np.allclose(saved_u, u, rtol=2e-11, atol=LOG_ATOL),
                'Density latent coordinates differ')
    else:
        require(maxima['latent_coordinate_error'] < LOG_ATOL, 'Density latent coordinates differ')
    require(_inside(recon, u) == expected.in_reference_ball
            and _inside(recon, saved_u) == expected.in_reference_ball, 'Latent ball support differs')
    maxima['jacobian_error'] = _log_error(record['log_physical_jacobian'], expected.log_physical_jacobian,
                                         'Physical Jacobian differs')
    _log_error(record['log_physical_density'], _log(record['log_latent_density'])
               -record['log_physical_jacobian'], 'Recorded q/J arithmetic differs')
    trace, rebuilt = record['hard_free_line_density'], expected.reconstruction
    require(isinstance(trace, Mapping) and isinstance(rebuilt, Mapping), 'Missing density reconstruction')
    if rebuilt.get('conditioning_disabled'):
        require(trace.get('conditioning_disabled') is True
                and (recon.alpha == 1. or recon.beta == 0.), 'Wrong disabled-conditioning branch')
        error = _log_error(record['log_latent_density'], expected.log_latent_density,
                           'Disabled guide density differs')
        _log_error(record['log_physical_density'], expected.log_physical_density,
                   'Disabled physical guide density differs')
        maxima['same_input_log_error'] = error
        result.update(complete=True, log_latent_density_saved_intervals=expected.log_latent_density,
                      log_latent_density_independent_geometry=expected.log_latent_density,
                      log_physical_density_saved_intervals=expected.log_physical_density,
                      log_physical_density_independent_geometry=expected.log_physical_density)
        return result
    require(not trace.get('conditioning_disabled', False), 'Conditioning unexpectedly disabled')
    axes = list(recon.axes)
    require(axes and [a['axis'] for a in trace['axes']] == axes
            and [a['axis'] for a in rebuilt['axes']] == axes, 'Density axis sequence differs')
    raw = _vector(recon.raw(u), (6,), 'Invalid reconstructed raw coordinates')
    saved_raw = _vector(recon.raw(saved_u), (6,), 'Invalid saved-input raw coordinates')
    trace_raw = _vector(trace['raw_coordinates'], (6,), 'Invalid recorded raw coordinates')
    normal.close(trace_raw, raw, 'Density raw coordinates differ')
    normal.close(trace_raw, saved_raw, 'Saved-input raw coordinates differ')
    maxima['raw_coordinate_error'] = float(max(np.max(abs(trace_raw-raw)), np.max(abs(trace_raw-saved_raw))))
    def gaussian_vector(coordinates, label):
        values = np.asarray(recon.gaussian_logs(coordinates), dtype=float)
        allowed = np.isfinite(values) | (np.isneginf(values) if full_vessel else False)
        require(values.shape == (len(recon.weights),) and allowed.all(), label)
        # Negative infinity is permitted only for an inactive component.
        # Every _compose call checks the actual active set independently.
        return values
    g = gaussian_vector(u, 'Invalid Gaussian density')
    saved_g = gaussian_vector(saved_u, 'Invalid saved-input Gaussian density')
    uniform = _uniform(recon, expected.in_reference_ball)
    baseline = float(np.logaddexp(uniform, math.log1p(-recon.alpha)+logsumexp(g)))
    _log_error(trace['baseline_log_density'], baseline, 'Baseline density differs')
    _log_error(rebuilt['baseline_log_density'], baseline, 'Independent baseline differs')
    all_factors = {key: [] for key in ['saved', 'pinned', 'independent', 'lower', 'upper']}
    fallback_count = 0
    for actual, ref in zip(trace['axes'], rebuilt['axes']):
        maximum = line.compare_axis(actual, ref)
        maxima['interval_endpoint_error'] = max(maxima['interval_endpoint_error'], maximum)
        saved_set, independent_set = actual['hard_free_intervals'], ref['intervals']
        inner, outer = interval_envelope(saved_set, independent_set)
        axis = actual['axis']
        decisions = [line.contains(s, x) for s in [saved_set, independent_set, inner, outer]
                     for x in [trace_raw[axis], saved_raw[axis], raw[axis]]]
        require(all(v == decisions[0] for v in decisions), 'Query support changes within geometry/coordinate envelope')
        allowed = decisions[0]
        means, sigmas = recon.conditional(raw, axis)
        saved_means, saved_sigmas = recon.conditional(saved_raw, axis)
        means = _vector(means, g.shape, 'Invalid conditional means')
        sigmas = _vector(sigmas, g.shape, 'Invalid conditional sigmas')
        require(np.all(sigmas > 0), 'Nonpositive conditional sigma')
        saved_mass = line.interval_masses(saved_set, saved_means, saved_sigmas)
        pinned_mass = line.interval_masses(saved_set, means, sigmas)
        independent_mass = line.interval_masses(independent_set, means, sigmas)
        lower_mass = line.interval_masses(inner, means, sigmas)
        upper_mass = line.interval_masses(outer, means, sigmas)
        normal.close(independent_mass, ref['conditional_masses'], 'Independent conditional masses differ', atol=2e-15, rtol=2e-7)
        normal.close(means, ref['conditional_means'], 'Independent conditional means differ')
        normal.close(sigmas, ref['conditional_sigmas'], 'Independent conditional sigmas differ')
        fallback = independent_mass <= recon.floor
        require(np.array_equal(fallback, np.asarray(ref['component_fallbacks'], bool)), 'Independent component fallback differs')
        require(np.array_equal(saved_mass <= recon.floor, fallback)
                and np.array_equal(pinned_mass <= recon.floor, fallback), 'Per-component saved/rebuilt fallback differs')
        # Fail rather than interpolate over the discontinuous fallback rule.
        require(np.all(upper_mass[fallback] <= recon.floor)
                and np.all(lower_mass[~fallback] > recon.floor), 'Conditional mass envelope crosses fallback floor')
        require(np.all(lower_mass <= upper_mass+2e-15), 'Inverted Normal mass envelope')
        fallback_count += int(fallback.sum())
        factors = dict(saved=_multipliers(saved_mass, fallback, allowed),
                       pinned=_multipliers(pinned_mass, fallback, allowed),
                       independent=_multipliers(independent_mass, fallback, allowed),
                       lower=_multipliers(upper_mass, fallback, allowed),
                       upper=_multipliers(lower_mass, fallback, allowed))
        logs = {key: _compose(recon, saved_g if key == 'saved' else g, uniform, values)
                for key, values in factors.items()}
        same_error = _log_error(actual['axis_log_proposal_density'], logs['saved'], 'Single-axis saved-input density differs')
        coordinate_error = _log_error(logs['saved'], logs['pinned'], 'Single-axis coordinate sensitivity exceeds arithmetic tolerance')
        _log_error(actual['axis_log_proposal_density'], logs['pinned'], 'Single-axis pinned-interval density differs')
        width = _enclosure(logs['lower'], logs['upper'], [logs['pinned'], logs['independent']], 'Single-axis geometry')
        if 'components' in actual:
            require(len(actual['components']) == len(g), 'Incomplete full component trace')
            for k, component in enumerate(actual['components']):
                require(component['component'] == k and type(component['fallback']) is bool
                        and component['fallback'] == bool(fallback[k])
                        and type(component['query_coordinate_allowed']) is bool
                        and component['query_coordinate_allowed'] == allowed, 'Recorded component support/fallback differs')
                normal.close(component['conditional_mean'], saved_means[k], 'Recorded conditional mean differs')
                normal.close(component['conditional_sigma'], saved_sigmas[k], 'Recorded conditional sigma differs')
                normal.close(component['conditional_mass'], saved_mass[k], 'Recorded conditional mass differs', atol=2e-15, rtol=2e-7)
                _log_error(component['gaussian_log_density'], saved_g[k]-math.log(recon.weights[k]), 'Recorded Gaussian density differs')
        difference = abs(logs['pinned']-logs['independent']) if logs['pinned'] != -math.inf else 0.
        maxima['same_input_log_error'] = max(maxima['same_input_log_error'], same_error)
        maxima['coordinate_log_error'] = max(maxima['coordinate_log_error'], coordinate_error)
        maxima['independent_geometry_log_difference'] = max(maxima['independent_geometry_log_difference'], difference)
        maxima['log_envelope_width'] = max(maxima['log_envelope_width'], width)
        for key, values in factors.items(): all_factors[key].append(values)
        result['axes'].append(dict(axis=axis, allowed=allowed, component_fallbacks=fallback.tolist(),
            conditional_masses_saved=saved_mass.tolist(), conditional_masses_independent=independent_mass.tolist(),
            conditional_mass_lower=lower_mass.tolist(), conditional_mass_upper=upper_mass.tolist(),
            log_density_saved_intervals=logs['saved'], log_density_independent_geometry=logs['independent'],
            log_density_lower=logs['lower'], log_density_upper=logs['upper'], log_envelope_width=width,
            same_input_log_error=same_error, coordinate_log_error=coordinate_error))
    require(trace['component_branches'] == rebuilt['component_branches'] == len(g)*len(axes), 'Incomplete component/axis denominator')
    require(trace['fallback_component_branches'] == rebuilt['fallback_component_branches'] == fallback_count,
            'Aggregate component fallback count differs')
    full = {key: _compose(recon, saved_g if key == 'saved' else g, uniform, np.mean(values, axis=0))
            for key, values in all_factors.items()}
    _log_error(full['independent'], expected.log_latent_density, 'Independent complete density inconsistent')
    _log_error(full['independent'], rebuilt['log_density'], 'Independent reconstruction density inconsistent')
    _log_error(expected.log_physical_density, full['independent']-expected.log_physical_jacobian,
               'Independent q/J density inconsistent')
    maxima['same_input_log_error'] = max(maxima['same_input_log_error'],
        _log_error(record['log_latent_density'], full['saved'], 'Complete saved-input density differs'),
        _log_error(record['log_physical_density'], full['saved']-expected.log_physical_jacobian, 'Complete saved-input q/J differs'))
    maxima['coordinate_log_error'] = max(maxima['coordinate_log_error'],
        _log_error(full['saved'], full['pinned'], 'Complete coordinate sensitivity exceeds arithmetic tolerance'))
    _log_error(record['log_latent_density'], full['pinned'], 'Complete pinned-interval density differs')
    width = _enclosure(full['lower'], full['upper'], [full['pinned'], full['independent']], 'Complete geometry')
    maxima['log_envelope_width'] = max(maxima['log_envelope_width'], width)
    maxima['independent_geometry_log_difference'] = max(maxima['independent_geometry_log_difference'],
        abs(full['pinned']-full['independent']) if full['pinned'] != -math.inf else 0.)
    require((full['saved'] == -math.inf) == expected.structural_zero, 'Complete saved-law support differs')
    result.update(complete=True, log_latent_density_saved_intervals=full['saved'],
        log_latent_density_independent_geometry=full['independent'], log_latent_density_pinned_intervals=full['pinned'],
        log_physical_density_saved_intervals=full['saved']-expected.log_physical_jacobian,
        log_physical_density_independent_geometry=expected.log_physical_density,
        log_density_lower=full['lower'], log_density_upper=full['upper'])
    return result
