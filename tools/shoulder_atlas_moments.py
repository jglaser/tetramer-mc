"""Full-N full-shoulder moments with the unchanged three-chart inner masks."""
from __future__ import annotations

import math

from shoulder_mis import LogMoments, quota_summary, require
from shoulder_peak_moments import covariance_of_means, partition_check
from shoulder_union_moments import MASKS as INNER_MASKS, UnionMoments


def inner_key(key):
    return 'inner' if key == 'full' else f'inner_{key}'


KEYS = ('full', 'outer', *(inner_key(key) for key in INNER_MASKS))


class ShoulderAtlasMoments:
    """One fixed original proposal law, including every rejected raw draw.

    Nonzero rows must satisfy 1 < q < 2. The inner window is strict at 1.1;
    q == 1.1 belongs to outer. Only inner rows enter geometric masks. All
    masks retain the same unconditional denominator and original weights.
    """

    def __init__(self):
        self.inner = UnionMoments()
        self.keys = KEYS
        self.values = {
            kind: dict(full=LogMoments(), outer=LogMoments(),
                       **{inner_key(key): value for key, value in values.items()})
            for kind, values in self.inner.values.items()
        }

    def add(self, q=None, radii=None, physical=None, hard=None, cloud_pair=None):
        require((physical is None) == (hard is None), 'Physical/hard zero masks differ')
        if physical is not None:
            require(q is not None and 1 < q < 2, 'Positive row outside full shoulder')
            require(math.isfinite(physical) and math.isfinite(hard), 'Finite original log weights required')
            require(cloud_pair is not None and len(cloud_pair) == 2 and
                    all(math.isfinite(value) for value in cloud_pair), 'Need two finite original clouds')
            pair_mean = max(cloud_pair)+math.log1p(math.exp(-abs(cloud_pair[0]-cloud_pair[1])))-math.log(2)
            require(abs(pair_mean-physical) < 2e-9,
                    'Cloud pair does not reproduce original physical weight')
        else:
            require(cloud_pair is None, 'Zero row cannot carry cloud diagnostics')
        is_inner = physical is not None and q < 1.1
        self.inner.add(radii if is_inner else None,
                       physical if is_inner else None, hard if is_inner else None,
                       cloud_pair if is_inner else None)
        for kind, value in (('physical', physical), ('hard', hard)):
            for key, include in (('full', physical is not None), ('outer', physical is not None and not is_inner)):
                self.values[kind][key].add(value if include else -math.inf,
                    cloud_pair if include and kind == 'physical' else None)

    def merge(self, other):
        require(isinstance(other, ShoulderAtlasMoments), 'Different full-shoulder masks')
        self.inner.merge(other.inner)
        for kind in self.values:
            for key in ('full', 'outer'):
                self.values[kind][key].merge(other.values[kind][key])
        return self

    def report(self):
        inner = self.inner.report()
        result = dict(same_row_covariances={}, partition_checks={})
        for kind, values in self.values.items():
            result[kind] = {}
            for key in self.keys:
                if key in ('full', 'outer'):
                    row = quota_summary([values[key]])
                    row['row_RSE'] = row.pop('stratified_RSE')
                    row['variance_rule'] = 'Original-density IID row variance / full unconditional N; rejected zeros retained'
                else:
                    old = 'full' if key == 'inner' else key[len('inner_'):]
                    row = inner[kind][old]
                result[kind][key] = row
            covariance = {}
            for left in self.keys:
                covariance[left] = {}
                for right in self.keys:
                    if left not in ('full', 'outer') and right not in ('full', 'outer'):
                        a = 'full' if left == 'inner' else left[len('inner_'):]
                        b = 'full' if right == 'inner' else right[len('inner_'):]
                        value = inner['same_row_covariances'][kind][a][b]
                    else:
                        if left == 'full': square = values[right].square
                        elif right == 'full' or left == right: square = values[left].square
                        else: square = -math.inf  # outer is disjoint from every inner mask
                        value = covariance_of_means(values[left], values[right], square)
                    covariance[left][right] = value
            result['same_row_covariances'][kind] = covariance
            result['partition_checks'][kind] = {
                'full': partition_check(values, ('inner', 'outer')),
                'full_threeway': partition_check(values, ('inner_union', 'inner_outside_union', 'outer')),
                **{f'inner_{key}': value for key, value in inner['partition_checks'][kind].items()}
            }
        return result
