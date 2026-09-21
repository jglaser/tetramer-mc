"""Recover original raw-weight moments from audited population summaries."""
import math

from analyze_native_tail_reference import KEYS, NativeTailMoments
from audit_shoulder_mis_independently import near
from compare_intermediate_reference import population_rse
from prepare_cayley_rms_cover import require
from shoulder_mis import LogMoments, quota_summary


def recover(row):
    """Invert reporting identities; never renormalize individual populations."""
    n, nonzero = row['draws'], row['nonzero']
    require(type(n) is int and n >= 2 and type(nonzero) is int and 0 <= nonzero <= n, 'Invalid full N')
    if nonzero == 0:
        require(row['logQ'] is None and row['weight_ESS'] == 0, 'Inconsistent empty population')
        return LogMoments(count=n)
    require(math.isfinite(row['logQ']) and 0 < row['weight_ESS'] <= n*(1+1e-12), 'Invalid weight moments')
    total = row['logQ']+math.log(n)
    noise_fraction = row['paired_cloud_variance_fraction']
    variance = row['log_variance_of_mean']
    noise = (variance+2*math.log(n)+math.log(noise_fraction)
             if noise_fraction is not None and noise_fraction > 0 and variance is not None else -math.inf)
    moment = LogMoments(count=n, nonzero=nonzero, total=total,
        square=2*total-math.log(row['weight_ESS']),
        maximum=total+math.log(row['maximum_fraction']), noise=noise)
    reconstructed = quota_summary([moment])
    near(reconstructed['logQ'], row['logQ'])
    near(reconstructed['stratified_RSE'], row['row_RSE'])
    return moment


def pooled_populations(populations):
    """Merge IID raw moments, including zero-only populations in full N."""
    require(bool(populations), 'At least one population required')
    require(len({p['seed'] for p in populations}) == len(populations), 'Repeated seed')
    require(len({p['samples'] for p in populations}) == 1, 'Fixed equal-size populations required')
    aggregate = NativeTailMoments()
    for population in populations:
        one = NativeTailMoments()
        for kind in ('physical', 'hard'):
            for key in KEYS:
                require(population[kind][key]['draws'] == population['samples'], 'Masked denominator differs')
                one.values[kind][key] = recover(population[kind][key])
        # This also checks the disjoint same-row partition before merging it.
        one.report()
        aggregate.merge(one)
    result = aggregate.report()
    for kind in ('physical', 'hard'):
        for key in KEYS:
            row = result[kind][key]
            values = [p[kind][key]['logQ'] for p in populations]
            row['independent_population_RSE'] = population_rse(values)
            row['population_logQ_values'] = values
            finite = [v for v in values if v is not None]
            if finite:
                offset = max(finite)
                mass = [0. if v is None else math.exp(v-offset) for v in values]
                row['maximum_population_fraction'] = max(mass)/sum(mass)
            else:
                row['maximum_population_fraction'] = None
    result['population_count'] = len(populations)
    return result
