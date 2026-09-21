"""Full-denominator moments for masks around a frozen inner-shoulder peak.

The caller audits geometry, the strict original-q window, hard validity, and
the original proposal density. Pass original log importance contributions for
included rows and call ``add()`` for every excluded row. No density changes,
conditioning, physical sampling, or independent-shell pooling occur here.
"""
from __future__ import annotations

import math

from scipy.special import logsumexp

from shoulder_mis import LogMoments, quota_summary, require


ATOMS = ('radial_0', 'radial_1', 'outside0p5')
MASKS = {
    'full': ATOMS,
    'ball0p25': ATOMS[:1],
    'ball0p5': ATOMS[:2],
    'radial_0': ATOMS[:1],
    'radial_1': ATOMS[1:2],
    'outside0p25': ATOMS[1:],
    'outside0p5': ATOMS[2:],
}


def inner_contains(q):
    """The original-q inner shoulder is strict at both 1 and 1.1."""
    return bool(1. < q < 1.1)


def selected_keys(radius):
    """Closed ball edges; a Cayley-seam radius of infinity is outside."""
    require(radius is not None and radius >= 0 and not math.isnan(radius),
            'Invalid geometric radius')
    atom = 'radial_0' if radius <= .25 else 'radial_1' if radius <= .5 else 'outside0p5'
    return tuple(key for key, atoms in MASKS.items() if atom in atoms)


def covariance_of_means(a, b, log_intersection_square):
    """Signed log covariance of two masks applied to the same weighted rows.

Cov(mean A, mean B) = (sum AB - sum A sum B/N) / (N(N-1)).
The intersection square sum comes from disjoint atomic masks. This retains
positive overlap terms for nested balls and negative disjoint-bin covariance.
"""
    require(a.count == b.count and a.count >= 2,
            'Covariance requires matching original N >= 2')
    # Identical masks must have a nonnegative diagonal, including when FP64
    # cancellation makes a constant row's centered sum round slightly below 0.
    if a.total == b.total and a.square == b.square == log_intersection_square:
        centered = a.centered_square()
        return dict(sign=0, log_absolute_covariance=None) if centered == -math.inf else dict(
            sign=1, log_absolute_covariance=centered-math.log(a.count)-math.log(a.count-1))
    product = a.total+b.total-math.log(a.count)
    if log_intersection_square == product:
        return dict(sign=0, log_absolute_covariance=None)
    sign = 1 if log_intersection_square > product else -1
    high = max(log_intersection_square, product)
    low = min(log_intersection_square, product)
    difference = high+(math.log(-math.expm1(low-high)) if low != -math.inf else 0.)
    return dict(sign=sign,
                log_absolute_covariance=difference-math.log(a.count)-math.log(a.count-1))


def partition_check(values, keys):
    """Reconstruct a total's mass and variance from a complete disjoint split."""
    full = values['full']
    parts = [values[key] for key in keys]
    require(full.count >= 2 and all(part.count == full.count for part in parts),
            'Partition requires matching original N >= 2')
    require(sum(part.nonzero for part in parts) == full.nonzero,
            'Masks fail disjoint row coverage')
    covariances = [dict(left=left, right=right,
                        **covariance_of_means(values[left], values[right], -math.inf))
                   for i, left in enumerate(keys) for right in keys[i+1:]]
    if not full.nonzero:
        return dict(complete_row_partition=True, logQ=None, log_variance_of_mean=None,
                    mass_error=0., scaled_variance_error=0., relative_variance_error=None,
                    covariances=covariances)
    log_total = float(logsumexp([part.total for part in parts]))
    log_square = float(logsumexp([part.square for part in parts]))
    mass_error = max(abs(log_total-full.total), abs(log_square-full.square))
    require(mass_error < 2e-10, 'Partition mass or square sum changed')
    # Work relative to the total squared-weight sum to avoid overflow. The
    # subtraction includes every disjoint same-row covariance exactly once.
    offset = full.square
    diagonals = math.fsum(math.exp(part.centered_square()-offset) for part in parts)
    cross = math.fsum(2*math.exp(a.total+b.total-math.log(full.count)-offset)
                      for i, a in enumerate(parts) for b in parts[i+1:]
                      if a.nonzero and b.nonzero)
    direct = math.exp(full.centered_square()-offset)
    variance_error = abs((diagonals-cross)-direct)
    require(variance_error < 2e-10, 'Partition covariance reconstruction failed')
    summary = quota_summary([full])
    return dict(complete_row_partition=True, logQ=summary['logQ'],
                log_variance_of_mean=summary['log_variance_of_mean'],
                mass_error=mass_error, scaled_variance_error=variance_error,
                relative_variance_error=variance_error/direct if direct else None,
                covariances=covariances,
                variance_rule='Sum bin variances plus twice all same-row covariances; full original N')


class ShoulderMoments:
    """Original-density physical/hard moments on one unconditional row stream.

``physical`` and ``hard`` are finite log contributions on the same valid row;
``cloud_pair`` contains the two original log physical cloud contributions.
When both weights are ``None``, all masks receive zero, even if a radius is
available. The q window is deliberately audited by the caller using
``inner_contains``; a geometric radius never establishes q-window membership.
``merge`` pools batches from the same proposal law, not independent shells
with different supports or fixed-quota proposal families.
"""

    def __init__(self):
        self.keys = tuple(MASKS)
        self.values = {kind: {key: LogMoments() for key in self.keys}
                       for kind in ('physical', 'hard')}

    def add(self, radius=None, physical=None, hard=None, cloud_pair=None):
        require((physical is None) == (hard is None), 'Physical/hard zero masks differ')
        keys = selected_keys(radius) if radius is not None else ()
        if physical is not None:
            require(radius is not None, 'Nonzero contributions require a geometric radius')
            require(math.isfinite(physical) and math.isfinite(hard),
                    'Finite original log weights required')
            require(cloud_pair is not None and len(cloud_pair) == 2 and
                    all(math.isfinite(value) for value in cloud_pair),
                    'Two finite original cloud contributions required')
        else:
            require(cloud_pair is None, 'Zero rows cannot carry nonzero cloud diagnostics')
            keys = ()
        for kind, value in (('physical', physical), ('hard', hard)):
            for key in self.keys:
                include = key in keys
                self.values[kind][key].add(value if include else -math.inf,
                    cloud_pair if include and kind == 'physical' else None)

    def merge(self, other):
        require(isinstance(other, ShoulderMoments) and self.keys == other.keys,
                'Cannot merge different mask definitions')
        for kind in self.values:
            for key in self.keys:
                self.values[kind][key].merge(other.values[kind][key])
        return self

    def report(self):
        result = {}
        covariance = {}
        checks = {}
        partitions = {
            'full': ('full', ATOMS),
            'ball0p25': ('ball0p25', ATOMS[:1]),
            'ball0p5': ('ball0p5', ATOMS[:2]),
            'outside0p25': ('outside0p25', ATOMS[1:]),
            'ball0p25_vs_outside': ('full', ('ball0p25', 'outside0p25')),
            'ball0p5_vs_outside': ('full', ('ball0p5', 'outside0p5')),
        }
        for kind, values in self.values.items():
            result[kind] = {}
            for key, value in values.items():
                row = quota_summary([value])
                row['row_RSE'] = row.pop('stratified_RSE')
                row['variance_rule'] = ('Original-density IID row variance / full unconditional N; '
                                        'all out-of-window and hard-invalid zeros retained')
                result[kind][key] = row
            covariance[kind] = {}
            for left in self.keys:
                covariance[kind][left] = {}
                for right in self.keys:
                    shared = tuple(atom for atom in ATOMS
                                   if atom in MASKS[left] and atom in MASKS[right])
                    log_square = float(logsumexp([values[key].square for key in shared])) if shared else -math.inf
                    covariance[kind][left][right] = covariance_of_means(values[left], values[right], log_square)
            checks[kind] = {
                name: partition_check(dict(full=values[total], **{key: values[key] for key in keys}), keys)
                for name, (total, keys) in partitions.items()
            }
        return dict(**result, same_row_covariances=covariance, partition_checks=checks)
