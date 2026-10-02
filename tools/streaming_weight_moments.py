"""Unconditional importance moments without retaining every weight or trace.

Only the largest ceil(N/100) scalar log weights are retained for the existing
top-one-percent diagnostic. Cloud variance uses a rescaled online central moment;
zeros contribute to its denominator, and negative residual pose variance is kept.
"""
from __future__ import annotations

import heapq
import math
from numbers import Real

from analyze_r4_smc_control import require


def optional_log(value):
    require(value is None or isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value),
            'A weight must be an explicit zero or a finite log weight')
    return None if value is None else float(value)


class WeightMoments:
    def __init__(self, draws):
        require(type(draws) is int and draws > 0, 'Positive unconditional allocation required')
        self.draws = draws
        self.seen = self.nonzero = 0
        self.offset = -math.inf
        self.sum1 = self.sum2 = 0.
        self.top_count = max(1, (draws + 99)//100)
        self.top = []

    def add(self, log_weight):
        value = optional_log(log_weight)
        require(self.seen < self.draws, 'Attempt count exceeds the allocation')
        self.seen += 1
        if value is None:
            return
        self.nonzero += 1
        if value > self.offset:
            scale = math.exp(self.offset-value)
            self.sum1 *= scale
            self.sum2 *= scale*scale
            self.offset = value
        weight = math.exp(value-self.offset)
        self.sum1 += weight
        self.sum2 += weight*weight
        if len(self.top) < self.top_count:
            heapq.heappush(self.top, value)
        elif value > self.top[0]:
            heapq.heapreplace(self.top, value)

    def result(self):
        require(self.seen == self.draws, 'Missing unconditional attempts')
        if not self.nonzero:
            return dict(draws=self.draws, nonzero=0, logQ=None, ess=0., relative_se=None,
                        max_fraction=None, top_one_percent_fraction=None,
                        coverage='Unresolved: no nonzero observations, not a physical zero or upper bound')
        ess = self.sum1*self.sum1/self.sum2
        return dict(draws=self.draws, nonzero=self.nonzero,
                    logQ=self.offset+math.log(self.sum1/self.draws), ess=ess,
                    relative_se=math.sqrt(max(0., (self.draws/ess-1)/(self.draws-1)))
                        if self.draws > 1 else None,
                    max_fraction=1./self.sum1,
                    top_one_percent_fraction=math.fsum(math.exp(v-self.offset) for v in self.top)/self.sum1,
                    top_one_percent_draw_count=self.top_count,
                    coverage='Observed-sample uncertainty; unseen high-weight regions remain possible')


class PairedNoise:
    def __init__(self, draws):
        require(type(draws) is int and draws > 0, 'Positive unconditional allocation required')
        self.draws = draws
        self.seen = 0
        self.offset = -math.inf
        self.mean = self.m2 = self.sum_difference2 = self.sum_product = 0.

    def add(self, first, second):
        first, second = optional_log(first), optional_log(second)
        require((first is None) == (second is None), 'A selected pose needs both independent cloud weights')
        require(self.seen < self.draws, 'Attempt count exceeds the allocation')
        self.seen += 1
        if first is None:
            a = b = 0.
        else:
            new_offset = max(first, second)
            if new_offset > self.offset:
                scale = math.exp(self.offset-new_offset)
                self.mean *= scale
                self.m2 *= scale*scale
                self.sum_difference2 *= scale*scale
                self.sum_product *= scale*scale
                self.offset = new_offset
            a, b = math.exp(first-self.offset), math.exp(second-self.offset)
        y = (a+b)/2
        delta = y-self.mean
        self.mean += delta/self.seen
        self.m2 += delta*(y-self.mean)
        self.sum_difference2 += (a-b)**2
        self.sum_product += a*b

    def result(self):
        require(self.seen == self.draws, 'Missing unconditional attempts')
        if self.draws < 2 or self.offset == -math.inf:
            return None
        total = self.m2/(self.draws-1)
        cloud = self.sum_difference2/(4*self.draws)
        residual = total-cloud
        return dict(log_weight_offset=self.offset, scaled_total_variance=total,
                    scaled_paired_cloud_variance=cloud, scaled_residual_pose_variance=residual,
                    paired_cloud_variance_fraction=cloud/total if total > 0 else None,
                    relative_variance_of_mean_cloud=cloud/self.draws/self.mean**2,
                    relative_variance_of_mean_pose=residual/self.draws/self.mean**2,
                    scaled_mean_W1W2=self.sum_product/self.draws,
                    scaled_mean_squared_cloud_difference=self.sum_difference2/self.draws,
                    scope='Paired independent clouds at each pose, with zero weights for invalid draws. '
                          'Residual pose variance may be negative due to finite sampling; it is not clipped. '
                          'This diagnostic cannot certify unseen tails.')


class RegionMoments:
    """One selected region, with the same N for physical and hard weights."""
    def __init__(self, draws):
        self.physical = WeightMoments(draws)
        self.hard = WeightMoments(draws)
        self.pairs = PairedNoise(draws)

    def add(self, row, selected):
        require(type(selected) is bool, 'Region membership must be Boolean')
        if selected:
            require(row['hard_valid'] is True and len(row['clouds']) == 2,
                    'Only physical-valid poses with two clouds contribute')
            require(row['log_importance_weight'] is not None and row['log_hard_weight'] is not None,
                    'A selected physical-valid pose must have positive finite weights')
            self.physical.add(row['log_importance_weight'])
            self.hard.add(row['log_hard_weight'])
            self.pairs.add(*(c['log_weight']-row['log_proposal_density'] for c in row['clouds']))
        else:
            self.physical.add(None)
            self.hard.add(None)
            self.pairs.add(None, None)

    def result(self):
        return dict(Qz=self.physical.result(), Q0=self.hard.result(), paired_noise=self.pairs.result())
