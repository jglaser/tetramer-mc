"""Full-N positive-mask moments for three frozen inner-shoulder charts.

The caller audits original q, hard validity, and original proposal weights.
Every excluded row must still call ``add()``. Center balls overlap; assignment
uses direct, mixture, geometry priority and never changes any row's density.
"""
from __future__ import annotations

import itertools
import math

from scipy.special import logsumexp

from shoulder_mis import LogMoments, logadd, quota_summary, require
from shoulder_peak_moments import covariance_of_means, partition_check


CENTERS = ('direct', 'mixture', 'geometry')
# 0: [0,.25], 1: (.25,.5], 2: (.5,infinity]. These 27 disjoint
# geometric states resolve all mask intersections without storing raw rows.
_ATOMS = tuple(itertools.product(range(3), repeat=3))
MASKS = {'full': _ATOMS}
for _index, _center in enumerate(CENTERS):
    for _suffix, _states in (('ball0p25', (0,)), ('ball0p5', (0, 1)), ('radial_1', (1,))):
        MASKS[f'{_center}_{_suffix}'] = tuple(atom for atom in _ATOMS if atom[_index] in _states)
    for _suffix, _states in (('ball0p25', (0,)), ('ball0p5', (0, 1)), ('radial_1', (1,))):
        MASKS[f'{_center}_assigned_{_suffix}'] = tuple(
            atom for atom in _ATOMS if atom[_index] in _states and all(state == 2 for state in atom[:_index]))
MASKS['union'] = tuple(atom for atom in _ATOMS if any(state < 2 for state in atom))
MASKS['outside_union'] = ((2, 2, 2),)
_SELECTED = {atom: tuple(key for key, atoms in MASKS.items() if atom in atoms) for atom in _ATOMS}
_INTERSECTIONS = {(left, right): tuple(atom for atom in MASKS[left] if atom in MASKS[right])
                  for left in MASKS for right in MASKS}


def _radial_atom(radii):
    require(radii is not None and len(radii) == len(CENTERS) and
            all(radius >= 0 and not math.isnan(radius) for radius in radii),
            'Need three nonnegative geometric radii in direct/mixture/geometry order')
    return tuple(0 if radius <= .25 else 1 if radius <= .5 else 2 for radius in radii)


def selected_keys(radii):
    """Closed ball edges; every earlier radius must exceed .5 for assignment."""
    return _SELECTED[_radial_atom(radii)]


class UnionMoments:
    """Original-density physical/hard moments on one unconditional stream.

    Nonzero rows need three radii, matching finite physical/hard log weights,
    and two finite original physical-cloud log weights. ``None`` weights mean
    zero in all masks even when radii are available. The caller applies the
    strict original window 1 < q < 1.1 before passing nonzero contributions.
    Merge only batches from the same original proposal law; independent
    support or fixed-quota estimators need a separate combination rule.
    """

    def __init__(self):
        self.keys = tuple(MASKS)
        self.values = {kind: {key: LogMoments() for key in self.keys}
                       for kind in ('physical', 'hard')}
        self._atom_squares = {kind: {atom: -math.inf for atom in _ATOMS} for kind in self.values}

    def add(self, radii=None, physical=None, hard=None, cloud_pair=None):
        require((physical is None) == (hard is None), 'Physical/hard zero masks differ')
        atom = _radial_atom(radii) if radii is not None else None
        if physical is not None:
            require(atom is not None, 'Nonzero contributions require three geometric radii')
            require(math.isfinite(physical) and math.isfinite(hard), 'Finite original log weights required')
            require(cloud_pair is not None and len(cloud_pair) == 2 and
                    all(math.isfinite(value) for value in cloud_pair),
                    'Two finite original cloud contributions required')
            keys = _SELECTED[atom]
        else:
            require(cloud_pair is None, 'Zero rows cannot carry cloud diagnostics')
            keys = ()
        for kind, value in (('physical', physical), ('hard', hard)):
            for key in self.keys:
                include = key in keys
                self.values[kind][key].add(value if include else -math.inf,
                    cloud_pair if include and kind == 'physical' else None)
            if value is not None:
                self._atom_squares[kind][atom] = logadd(self._atom_squares[kind][atom], 2*value)

    def merge(self, other):
        require(isinstance(other, UnionMoments) and self.keys == other.keys,
                'Cannot merge different union mask definitions')
        for kind in self.values:
            for key in self.keys:
                self.values[kind][key].merge(other.values[kind][key])
            for atom in _ATOMS:
                self._atom_squares[kind][atom] = logadd(self._atom_squares[kind][atom], other._atom_squares[kind][atom])
        return self

    def report(self):
        result = {}
        covariance = {}
        checks = {}
        partitions = {'full': ('full', ('union', 'outside_union')),
                      'union': ('union', tuple(f'{center}_assigned_ball0p5' for center in CENTERS))}
        for center in CENTERS:
            for prefix in (center, f'{center}_assigned'):
                partitions[f'{prefix}_ball0p5'] = (
                    f'{prefix}_ball0p5', (f'{prefix}_ball0p25', f'{prefix}_radial_1'))
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
                    shared = _INTERSECTIONS[left, right]
                    # A contained mask gives the intersection directly and
                    # keeps diagonal variances consistent with quota_summary.
                    if shared == MASKS[left]:
                        square = values[left].square
                    elif shared == MASKS[right]:
                        square = values[right].square
                    else:
                        square = float(logsumexp([self._atom_squares[kind][atom] for atom in shared])) if shared else -math.inf
                    covariance[kind][left][right] = covariance_of_means(values[left], values[right], square)
            checks[kind] = {
                name: partition_check(dict(full=values[total], **{key: values[key] for key in keys}), keys)
                for name, (total, keys) in partitions.items()
            }
        return dict(**result, same_row_covariances=covariance, partition_checks=checks)
