"""Independent-population contact-mass diagnostics, without geometry or row data.

Public functions consume a frozen declaration and already audited population
records. Declaration fields are ``arm_id``, ``population_count``,
``draws_per_population``, ``total_unconditional_draws``, ``regions`` (ordered),
``target_and_regions_sha256`` (a lowercase 64-hex audited scope identity),
and ``populations`` (each ``{id, seed}``). Records contain matching ``id``,
``seed``, ``draws``, ``unconditional_denominator``, and ``log_masses`` keyed by
region. Each log mass is the logarithm of a *linear population mean*, with all
attempted draws in its denominator; None or -inf denotes an observed zero.

The caller must bind that scope identity to the exact shape, scaffold, domain,
physical measure, activity, depletant radius and complete region definitions;
this module does not open those files or establish equivalence of targets.
The metadata checks cannot prove how saved masses were calculated: the upstream
row audit remains responsible for the denominator and estimator. Independent
seeds are necessary, but cannot prove stochastic independence. No row ESS,
maximum row contribution, tail bound, or physical zero follows from these data.
Nothing here pools stages, changes archived analyses, or launches calculations.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import re
import sys
from typing import Any, Mapping, Sequence

from scipy.stats import t as student_t


SCOPE = ('Arithmetic mean of independent linear population masses, retaining '
         'observed zero estimates and paired regional covariance. Delta-method '
         'log ratios and Student-t intervals are approximate diagnostics, not '
         'unbiased log estimates or guarantees against unseen contributions.')


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    _require(type(value) is int and value >= minimum, name + ' must be an integer >= ' + str(minimum))
    return value


def _name(value: Any, label: str) -> str:
    _require(isinstance(value, str) and bool(value), label + ' must be a nonempty string')
    return value


def _log(value: Any) -> float | None:
    if value is None:
        return None
    _require(not isinstance(value, bool) and isinstance(value, (int, float)),
             'Log masses must be numeric or null')
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError('Unrepresentable population log mass') from error
    _require(not math.isnan(result) and result != math.inf, 'NaN/+inf population log mass')
    return None if result == -math.inf else result


@dataclass(frozen=True)
class _Batch:
    arm_id: str
    target_and_regions_sha256: str
    regions: tuple[str, ...]
    ids: tuple[str, ...]
    seeds: tuple[int, ...]
    draws: int
    total: int
    logs: tuple[tuple[float | None, ...], ...]

    @property
    def count(self) -> int:
        return len(self.ids)


def _validate(declaration: Mapping[str, Any], records: Sequence[Mapping[str, Any]]) -> _Batch:
    arm = _name(declaration['arm_id'], 'arm_id')
    scope = declaration['target_and_regions_sha256']
    _require(isinstance(scope, str) and re.fullmatch('[0-9a-f]{64}', scope) is not None,
             'target_and_regions_sha256 must be a lowercase SHA256 identity')
    count = _integer(declaration['population_count'], 'population_count', 2)
    draws = _integer(declaration['draws_per_population'], 'draws_per_population', 1)
    total = _integer(declaration['total_unconditional_draws'], 'total_unconditional_draws', 1)
    _require(total == count * draws, 'Declared unconditional total is not population_count * draws_per_population')
    regions = tuple(_name(v, 'region') for v in declaration['regions'])
    _require(bool(regions) and len(set(regions)) == len(regions), 'Regions must be nonempty and unique')
    slots = declaration['populations']
    _require(len(slots) == count and len(records) == count, 'Missing or extra declared populations')
    ids = tuple(_name(p['id'], 'population id') for p in slots)
    seeds = tuple(_integer(p['seed'], 'seed') for p in slots)
    _require(all(seed < 2**64 for seed in seeds), 'Seed exceeds u64 range')
    _require(len(set(ids)) == count and len(set(seeds)) == count, 'Population IDs and seeds must be distinct')
    by_id = {}
    for record in records:
        key = _name(record['id'], 'record id')
        _require(key not in by_id, 'Duplicate population record')
        by_id[key] = record
    _require(set(by_id) == set(ids), 'Population identities differ from declaration')
    logs = []
    for key, seed in zip(ids, seeds):
        record = by_id[key]
        _require(_integer(record['seed'], 'record seed') == seed, 'Population seed differs from declaration')
        _require(_integer(record['draws'], 'record draws', 1) == draws,
                 'Attempted draws differ from frozen population allocation')
        _require(_integer(record['unconditional_denominator'], 'unconditional_denominator', 1) == draws,
                 'Lost unconditional attempted-draw denominator')
        _require(set(record['log_masses']) == set(regions), 'Missing or extra regional masses')
        logs.append(tuple(_log(record['log_masses'][name]) for name in regions))
    return _Batch(arm, scope, regions, ids, seeds, draws, total, tuple(logs))


def _linear_display(log_value: float | None, zero: bool = False) -> dict[str, Any]:
    """A missing display of a positive value never silently becomes zero."""
    if zero:
        return dict(value=0., status='observed_zero')
    if log_value is None:
        return dict(value=None, status='unresolved')
    try:
        value = math.exp(log_value)
    except OverflowError:
        return dict(value=None, status='overflow_use_log_or_scaled_value')
    if value == 0.:
        return dict(value=None, status='underflow_use_log_or_scaled_value')
    return dict(value=value, status='finite')


def _moments(logs: Sequence[float | None]) -> dict[str, Any]:
    count = len(logs)
    positive = [v for v in logs if v is not None]
    offset = max(positive) if positive else 0.
    x = [0. if v is None else math.exp(v - offset) for v in logs]
    total = math.fsum(x)
    mean = total / count
    variance = math.fsum((v - mean)**2 for v in x) / (count * (count - 1))
    se = math.sqrt(variance)
    log_mean = offset + math.log(mean) if positive else None
    log_se = offset + math.log(se) if se > 0 else None
    shares = [v / total for v in x] if positive else None
    relative_se = se / mean if positive else None
    return dict(log_scale=offset, scaled_linear_mean=mean, scaled_population_SE=se if positive else None,
                log_linear_mean=log_mean, log_population_SE=log_se,
                linear_mean=_linear_display(log_mean, zero=not positive),
                linear_population_SE=_linear_display(log_se, zero=bool(positive) and se == 0),
                population_relative_SE=relative_se,
                observed_positive=bool(positive), nonzero_populations=len(positive),
                observed_zero_populations=count - len(positive),
                positive_population_scaling_underflows=sum(v is not None and w == 0 for v, w in zip(logs, x)),
                population_log_masses=list(logs), zero_observed_dispersion=se == 0,
                uncertainty_resolved=bool(positive),
                unresolved=None if positive else 'All observed population estimates are zero; no physical zero, precision claim or upper bound follows.',
                _scaled_values=x, _shares=shares)


def _public_moments(value: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in value.items() if not k.startswith('_')}


def _summarize(batch: _Batch, native_region: str, competing_region: str) -> dict[str, Any]:
    _require(native_region in batch.regions and competing_region in batch.regions and native_region != competing_region,
             'Distinct native and competing regions must be declared')
    count = batch.count
    columns = list(zip(*batch.logs))
    moments = [_moments(column) for column in columns]
    covariance = []
    relative = []
    for a in moments:
        covariance.append([math.fsum((x - a['scaled_linear_mean']) * (y - b['scaled_linear_mean'])
                           for x, y in zip(a['_scaled_values'], b['_scaled_values'])) /
                           (count * (count - 1)) for b in moments])
        relative.append([None if a['_shares'] is None or b['_shares'] is None else
                         count / (count - 1) * math.fsum((x - 1 / count) * (y - 1 / count)
                         for x, y in zip(a['_shares'], b['_shares'])) for b in moments])
    ni, ci = batch.regions.index(native_region), batch.regions.index(competing_region)
    native, competing = moments[ni], moments[ci]
    contrast = dict(observed=False, native_region=native_region, competing_region=competing_region,
                    beta_F_native_minus_competing=None, population_SE=None,
                    degrees_of_freedom=count - 1, Student_t_quantile=None,
                    halfwidth_95=None, interval_95=None,
                    reason='Native or competing mass unobserved; no finite contrast or upper bound.')
    if native['observed_positive'] and competing['observed_positive']:
        difference = competing['log_linear_mean'] - native['log_linear_mean']
        _require(math.isfinite(difference), 'Unrepresentable free-energy log contrast')
        # This equivalent covariance expression avoids subtracting large,
        # nearly equal marginal terms when the paired ratio is well determined.
        variance = count / (count - 1) * math.fsum((x - y)**2 for x, y in
                   zip(native['_shares'], competing['_shares']))
        se = math.sqrt(variance)
        quantile = float(student_t.ppf(.975, count - 1))
        half = quantile * se
        contrast.update(observed=True, beta_F_native_minus_competing=difference,
                        population_SE=se, Student_t_quantile=quantile,
                        halfwidth_95=half, interval_95=[difference - half, difference + half],
                        reason=None, paired_relative_covariance=relative[ni][ci],
                        paired_influence_variance=variance)
    return dict(schema='contact-independent-population-statistics-v1', arm_id=batch.arm_id,
                target_and_regions_sha256=batch.target_and_regions_sha256,
                population_count=count, draws_per_population=batch.draws,
                total_unconditional_draws=batch.total,
                populations=[dict(id=key, seed=seed, draws=batch.draws) for key, seed in zip(batch.ids, batch.seeds)],
                region_order=list(batch.regions),
                estimates={key: _public_moments(value) for key, value in zip(batch.regions, moments)},
                covariance_of_population_means=dict(
                    region_order=list(batch.regions),
                    per_region_log_scales=[v['log_scale'] for v in moments],
                    scaled_matrix=covariance, relative_matrix=relative,
                    convention='Entry i,j multiplied by exp(log_scale[i]+log_scale[j]) gives covariance of the linear means.'),
                free_energy_contrast=contrast, scope=SCOPE)


def summarize_populations(declaration: Mapping[str, Any], records: Sequence[Mapping[str, Any]], *,
                          native_region: str = 'native', competing_region: str = 'competing') -> dict[str, Any]:
    """Validate a declared P>=2 allocation and summarize paired linear masses."""
    return _summarize(_validate(declaration, records), native_region, competing_region)


def compare_population_masses(left_declaration: Mapping[str, Any], left_records: Sequence[Mapping[str, Any]],
                              right_declaration: Mapping[str, Any], right_records: Sequence[Mapping[str, Any]],
                              region: str, *, se_multiplier: float = 3., log_absolute_limit: float = .2) -> dict[str, Any]:
    """Compare independent arms in linear Q, allowing unequal population counts.

    Both positive means are required. Agreement means abs(Q_left-Q_right) <=
    se_multiplier * sqrt(SE_left^2+SE_right^2), with a declared 32-epsilon
    floating-point tolerance, AND abs(log(Q_left/Q_right)) <= log_absolute_limit.
    The delta-method log SE is descriptive, never substituted for the linear test.
    """
    for value in (se_multiplier, log_absolute_limit):
        _require(not isinstance(value, bool) and isinstance(value, (int, float)) and
                 math.isfinite(value) and value >= 0, 'Invalid comparison threshold')
    left, right = _validate(left_declaration, left_records), _validate(right_declaration, right_records)
    _require(left.target_and_regions_sha256 == right.target_and_regions_sha256,
             'Physical target or complete region definitions differ between arms')
    _require(set(left.seeds).isdisjoint(right.seeds), 'Cross-arm seeds must be disjoint for independent comparison')
    _require(region in left.regions and region in right.regions, 'Compared region must be declared in both arms')
    a = _moments([row[left.regions.index(region)] for row in left.logs])
    b = _moments([row[right.regions.index(region)] for row in right.logs])
    result = dict(schema='contact-independent-population-comparison-v1', region=region,
                  target_and_regions_sha256=left.target_and_regions_sha256,
                  left_arm=left.arm_id, right_arm=right.arm_id,
                  left_population_count=left.count, right_population_count=right.count,
                  left=_public_moments(a), right=_public_moments(b),
                  observed=False, passed=False, SE_passed=None, absolute_passed=None,
                  SE_multiplier=float(se_multiplier), absolute_limit=float(log_absolute_limit),
                  reason='An arm has unobserved mass; this is neither physical zero nor an upper bound.',
                  scope=SCOPE)
    if not (a['observed_positive'] and b['observed_positive']):
        return result
    log_difference = a['log_linear_mean'] - b['log_linear_mean']
    _require(math.isfinite(log_difference), 'Unrepresentable cross-arm log contrast')
    scale = max(a['log_scale'], b['log_scale'])
    fa, fb = math.exp(a['log_scale'] - scale), math.exp(b['log_scale'] - scale)
    ma, mb = a['scaled_linear_mean'] * fa, b['scaled_linear_mean'] * fb
    sa, sb = a['scaled_population_SE'] * fa, b['scaled_population_SE'] * fb
    difference = ma - mb
    combined = math.hypot(sa, sb)
    tolerance = 32 * sys.float_info.epsilon * max(ma, mb)
    statistical = abs(difference) <= se_multiplier * combined + tolerance
    absolute = abs(log_difference) <= log_absolute_limit
    standard_errors = difference / combined if combined else 0. if difference == 0 else None
    standard_error_ratio_overflow = standard_errors is not None and not math.isfinite(standard_errors)
    if standard_error_ratio_overflow:
        standard_errors = None
    result.update(observed=True, passed=statistical and absolute, SE_passed=statistical,
                  absolute_passed=absolute, reason=None, log_left_minus_right=log_difference,
                  log_scale=scale, scaled_left_linear_mean=ma, scaled_right_linear_mean=mb,
                  scaled_left_population_SE=sa, scaled_right_population_SE=sb,
                  scaled_linear_difference=difference, scaled_independent_difference_SE=combined,
                  scaled_comparison_roundoff_tolerance=tolerance,
                  difference_in_combined_SE=standard_errors,
                  standard_error_ratio_overflow=standard_error_ratio_overflow,
                  zero_variance_disagreement=a['scaled_population_SE'] == b['scaled_population_SE'] == 0 and difference != 0,
                  shared_display_scale_underflow=dict(left=ma == 0., right=mb == 0.),
                  shared_standard_error_underflow=dict(
                      left=a['scaled_population_SE'] > 0 and sa == 0.,
                      right=b['scaled_population_SE'] > 0 and sb == 0.),
                  combined_population_log_delta_SE=math.hypot(a['population_relative_SE'], b['population_relative_SE']),
                  SE_test='Independent linear mean difference, not a difference of population log means or a log-SE test.')
    return result
