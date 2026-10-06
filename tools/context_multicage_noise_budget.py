"""Saved-count decomposition of the pooled Poisson importance second moment.

At a fixed valid pose let M=O-L, S~Poisson(2 lambda M), a=z/(2 lambda),
Y=exp(zL)(1+a)^S/q, and P=exp(2zL)(1+2a)^S/q^2. Then
E[Y]=exp(zO)/q, E[P]=(exp(zO)/q)^2, and E[Y^2-P]=Var(Y|pose).
The subtraction is nonnegative for every count, not only in expectation.

This diagnoses the already reported estimator; it draws no points, changes no
weights, and does not replace primary mass estimates. Fixed generating strata
are retained when estimating variance of the normalizer. Noise-free ESS ratios
are noisy plug-ins, not actual ESS, rigorous ceilings, or CPU-speed forecasts.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time

from scipy.stats import t as student_t

ARMS = ('baseline', 'multicage')
GROUPS = ('full_domain', 'A_T', 'T_any', 'T_outside_A', 'remaining_without_A_T',
          'contact_without_A_T', 'A_partial', 'B', 'other_contact', 'unbound')
REGIONS = ('A_patch_0_0.25', 'A_patch_0.25_0.5', 'A_patch_0.5_0.75',
           'A_patch_0.75_1', 'A_patch_complete', 'B', 'other_contact', 'unbound', 'hard_invalid')
CHARTS = ('full', 'diagonal', 'broad_full', 'cage0', 'cage1')
BINS = (0., 6., 12., 24., 48., 96., math.inf)
INTENSITY_MULTIPLIERS = {'1':1., '2':2., '4':4., 'infinity':math.inf}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def close(actual, expected, name):
    require(math.isfinite(actual) and math.isfinite(expected)
            and abs(actual-expected) <= 3e-9+3e-11*max(abs(actual), abs(expected)),
            'Reconstruction differs: '+name)


def logsum(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    high = max(values)
    return high+math.log(math.fsum(math.exp(v-high) for v in values))


def linear(log_value):
    return None if log_value is None or log_value > math.log(sys.float_info.max) else math.exp(log_value)


def signed_value(value, log_scale=0.):
    if value == 0:
        return dict(sign=0, log_abs=None, value=0.)
    log_abs = math.log(abs(value))+log_scale
    absolute = linear(log_abs)
    return dict(sign=1 if value > 0 else -1, log_abs=log_abs,
                value=None if absolute is None else math.copysign(absolute, value))


def add_signed(values):
    active = [v for v in values if v['sign']]
    if not active:
        return signed_value(0.)
    scale = max(v['log_abs'] for v in active)
    return signed_value(math.fsum(v['sign']*math.exp(v['log_abs']-scale) for v in active), scale)


def moment_terms(lower, count, intensity, activity, log_q):
    """All logs include the SAME complete arm density, once or twice."""
    require(type(count) is int and count >= 0
            and all(math.isfinite(v) for v in (lower, intensity, activity, log_q))
            and lower >= 0 and intensity > 0 and activity >= 0, 'Invalid Poisson moment inputs')
    a = activity/(2*intensity)
    # Stable even when the two direct logarithms agree to machine precision.
    delta = count*math.log1p(a*a/(1+2*a))
    log_y = activity*lower+count*math.log1p(a)-log_q
    log_p = 2*activity*lower+count*math.log1p(2*a)-2*log_q
    log_y2 = 2*log_y
    close(log_y2-log_p, delta, 'nonnegative log moment gap')
    fraction = -math.expm1(-delta)
    log_v = log_y2+math.log(fraction) if fraction > 0 else None
    require(all(math.isfinite(v) for v in (log_y, log_p, log_y2, delta)), 'Unrepresentable moment logarithm')
    forecasts = {label:log_p+count*math.log1p(a*a/(m*(1+2*a)))
                 for label,m in INTENSITY_MULTIPLIERS.items()}
    close(forecasts['1'],log_y2,'current-intensity forecast')
    forecasts['1'] = log_y2  # Exact identity, also avoid roundoff in its self-ratio.
    return dict(log_y=log_y, log_y2=log_y2, log_pose_second=log_p,
                log_forecast_second=forecasts,
                log_auxiliary_variance=log_v, realized_auxiliary_fraction=fraction)


def memberships(row):
    valid = row['physical_valid']; region = row['region']; any_t = row['source_T_complete_all_regions']
    require(type(valid) is bool and type(any_t) is bool and region in REGIONS
            and valid == (region != 'hard_invalid') and (not any_t or valid), 'Invalid region partition')
    target = valid and region == 'A_patch_complete'
    require(not target or any_t, 'A_T is not contained in T-any')
    return dict(full_domain=valid, A_T=target, T_any=any_t, T_outside_A=any_t and not target,
                remaining_without_A_T=valid and not target,
                contact_without_A_T=valid and region != 'unbound' and not target,
                A_partial=valid and region.startswith('A_patch_') and not target,
                B=valid and region == 'B', other_contact=valid and region == 'other_contact',
                unbound=valid and region == 'unbound')


def radial_index(value):
    if value is None:
        return 5
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0, 'Invalid saved radius')
    return next((i for i in range(6) if value < BINS[i+1]), 5)


def summarize(terms, denominator):
    require(type(denominator) is int and denominator > 0 and len(terms) <= denominator,
            'Lost unconditional attempt denominator')
    if not terms:
        return dict(hits=0, denominator=denominator, log_mass=None, observed_importance_ess=0.,
                    log_second_moment=None, log_pose_second_moment=None,
                    log_auxiliary_second_moment=None, auxiliary_fraction=None,
                    noise_free_ess_plugin=None, noise_free_ess_plugin_exceeds_attempts=False,
                    second_moment_gain_plugin=None, exact_fraction_bounds=[0., 1.],
                    intensity_forecasts={label:dict(log_second_moment=None,importance_ess_plugin=None,
                        importance_ess_plugin_exceeds_attempts=False,second_moment_gain_plugin=None)
                        for label in INTENSITY_MULTIPLIERS})
    first = logsum([r['log_y'] for r in terms]); second = logsum([r['log_y2'] for r in terms])
    pose = logsum([r['log_pose_second'] for r in terms]); auxiliary = logsum([r['log_auxiliary_variance'] for r in terms])
    fraction = math.exp(auxiliary-second) if auxiliary is not None else 0.
    close(fraction+math.exp(pose-second), 1., 'second moment partition')
    ess = math.exp(2*first-second); proxy = math.exp(2*first-pose)
    forecasts = {}
    for label in INTENSITY_MULTIPLIERS:
        value = logsum([r['log_forecast_second'][label] for r in terms])
        proxy_m = math.exp(2*first-value)
        forecasts[label] = dict(log_second_moment=value-math.log(denominator),
            importance_ess_plugin=proxy_m,importance_ess_plugin_exceeds_attempts=proxy_m > denominator,
            second_moment_gain_plugin=math.exp(second-value))
    return dict(hits=len(terms), denominator=denominator, log_mass=first-math.log(denominator),
                observed_importance_ess=ess, log_second_moment=second-math.log(denominator),
                log_pose_second_moment=pose-math.log(denominator),
                log_auxiliary_second_moment=auxiliary-math.log(denominator) if auxiliary is not None else None,
                auxiliary_fraction=fraction, exact_fraction_bounds=[0., 1.],
                noise_free_ess_plugin=proxy, noise_free_ess_plugin_exceeds_attempts=proxy > denominator,
                second_moment_gain_plugin=math.exp(second-pose),intensity_forecasts=forecasts)


def stratum_variance(terms, stratum_attempts, population_attempts):
    """Unbiased signed pose and nonnegative auxiliary normalizer-variance estimates.

    For n_j draws and N total attempts, auxiliary = sum_i V_i/N^2.
    Pose = [sum_i P_i - sum_(i!=k) Y_i Y_k/(n_j-1)]/N^2.
    Their sum is n_j times ordinary sample variance of Y divided by N^2.
    A negative pose estimate is retained; it is possible at finite cloud noise.
    """
    require(type(stratum_attempts) is int and stratum_attempts > 1
            and type(population_attempts) is int and population_attempts >= stratum_attempts
            and len(terms) <= stratum_attempts, 'Invalid fixed stratum denominator')
    if not terms:
        return {k:signed_value(0.) for k in ('total', 'pose', 'auxiliary')}
    scale = max(r['log_y'] for r in terms)
    weights = [math.exp(r['log_y']-scale) for r in terms]
    mean = math.fsum(weights)/stratum_attempts
    centered = math.fsum((w-mean)**2 for w in weights)+(stratum_attempts-len(weights))*mean*mean
    total = stratum_attempts*centered/(stratum_attempts-1)
    auxiliary = math.fsum(math.exp(r['log_auxiliary_variance']-2*scale)
                          for r in terms if r['log_auxiliary_variance'] is not None)
    factor = 2*scale-2*math.log(population_attempts)
    return dict(total=signed_value(total, factor), pose=signed_value(total-auxiliary, factor),
                auxiliary=signed_value(auxiliary, factor))


def fieller_ratio(numerators, denominators, physical_bounds):
    """Approximate 95% Fieller set from independent paired population summaries.

    An unbounded set remains unbounded. Intersect only with algebraically known
    parameter bounds. Four populations and heavy tails invalidate guarantees.
    """
    require(len(numerators) == len(denominators) >= 3, 'Too few independent populations')
    require(all(math.isfinite(v) for v in numerators+denominators)
            and all(v >= 0 for v in numerators+denominators), 'Invalid second-moment population means')
    n = len(numerators); x = statistics.mean(numerators); y = statistics.mean(denominators)
    base = dict(populations=n, confidence=.95, degrees_of_freedom=n-1,
                physical_bounds=list(physical_bounds), method='paired-population Fieller Student-t approximation',
                guarantee=False)
    if y == 0:
        return dict(base, available=False, point_estimate=None, confidence_set=[],
                    reason='No positive denominator observed; no ratio inference.')
    critical = float(student_t.ppf(.975, n-1))
    vx = statistics.variance(numerators)/n; vy = statistics.variance(denominators)/n
    covariance = math.fsum((a-x)*(b-y) for a,b in zip(numerators, denominators))/(n*(n-1))
    aa = y*y-critical*critical*vy
    bb = -2*(x*y-critical*critical*covariance)
    cc = x*x-critical*critical*vx
    discriminant = bb*bb-4*aa*cc
    if aa > 0:
        intervals = [] if discriminant < 0 else [[(-bb-math.sqrt(discriminant))/(2*aa), (-bb+math.sqrt(discriminant))/(2*aa)]]
    elif aa < 0:
        if discriminant <= 0:
            intervals = [[-math.inf, math.inf]]
        else:
            roots = sorted([(-bb-math.sqrt(discriminant))/(2*aa), (-bb+math.sqrt(discriminant))/(2*aa)])
            intervals = [[-math.inf, roots[0]], [roots[1], math.inf]]
    elif bb > 0:
        intervals = [[-math.inf, -cc/bb]]
    elif bb < 0:
        intervals = [[-cc/bb, math.inf]]
    else:
        intervals = [[-math.inf, math.inf]] if cc <= 0 else []
    lower, upper = physical_bounds; upper = math.inf if upper is None else upper
    clipped = [[max(lo, lower), min(hi, upper)] for lo,hi in intervals if max(lo, lower) <= min(hi, upper)]
    return dict(base, available=True, point_estimate=x/y,
                denominator_separated_from_zero=aa > 0,
                confidence_set=[[lo, hi if math.isfinite(hi) else None] for lo,hi in clipped],
                upper_unbounded=any(math.isinf(hi) for _,hi in clipped),
                informative=bool(clipped) and clipped != [[lower, upper]])


def ratio_uncertainty(populations):
    logs = [p['log_second_moment'] for p in populations if p['log_second_moment'] is not None]
    scale = max(logs) if logs else 0.
    values = lambda key:[math.exp(p[key]-scale) if p[key] is not None else 0. for p in populations]
    second = values('log_second_moment'); pose = values('log_pose_second_moment'); auxiliary = values('log_auxiliary_second_moment')
    forecasts = {label:fieller_ratio(second,
        [math.exp(p['intensity_forecasts'][label]['log_second_moment']-scale)
         if p['intensity_forecasts'][label]['log_second_moment'] is not None else 0. for p in populations],
        (1.,None)) for label in INTENSITY_MULTIPLIERS}
    forecasts['1'] = dict(available=bool(logs),populations=len(populations),
        point_estimate=1. if logs else None,confidence_set=[[1.,1.]] if logs else [],
        physical_bounds=[1.,1.],upper_unbounded=False,informative=bool(logs),
        guarantee=bool(logs),method='Exact identity Q_1=Y^2; no estimated confidence interval.',
        reason=None if logs else 'No positive denominator observed; ratio not reported.')
    return dict(auxiliary_fraction=fieller_ratio(auxiliary, second, (0.,1.)),
                noise_free_second_moment_gain=fieller_ratio(second, pose, (1.,None)),
                intensity_forecast_gain=forecasts,
                caveat='Approximate confidence sets for ratios of expected second moments, not guarantees, empirical ESS intervals, or CPU speedup forecasts. Heavy tails and four populations can hide important modes. No bounded estimate is inferred from an unbounded Fieller set.')


def require_partition(whole, parts):
    for key in ('log_second_moment', 'log_pose_second_moment', 'log_auxiliary_second_moment'):
        total = logsum([p[key] for p in parts]); expected = whole[key]
        require((total is None) == (expected is None), 'Empty/nonempty second moment partition differs')
        if total is not None:
            close(total, expected, 'region/radial '+key)
    for label in INTENSITY_MULTIPLIERS:
        total = logsum([p['intensity_forecasts'][label]['log_second_moment'] for p in parts])
        expected = whole['intensity_forecasts'][label]['log_second_moment']
        require((total is None) == (expected is None), 'Forecast partition empty status differs')
        if total is not None:
            close(total,expected,'intensity forecast partition')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True); parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); started = time.process_time(); protocol = read(args.protocol)
    require(protocol['schema'] == 'context-multicage-noise-budget-v1'
            and protocol['all_attempts'] == 32768 and protocol['population_denominator'] == 4096
            and protocol['intensity'] == 2.24 and protocol['activity'] == .035
            and protocol['bins'] == [0,6,12,24,48,96,None]
            and protocol['intensity_multipliers'] == ['1','2','4','infinity'], 'Changed frozen diagnostic allocation')
    bindings = protocol['input_sha256']
    for path,digest in bindings.items():
        require(sha(path) == digest, 'Changed input: '+path)
    def bound(path):
        path = str(Path(path).resolve()); require(path in bindings, 'Unbound input: '+path); return read(path)
    report = bound(protocol['physical_report']); primary_protocol = bound(protocol['physical_protocol'])
    require(report['schema'] == 'context-multicage-physical-report-v1' and report['complete'] and report['passed']
            and report['all_attempts'] == 32768 and report['finite_stability_conclusion'] is False
            and report['protocol_sha256'] == bindings[str(Path(protocol['physical_protocol']).resolve())], 'Incomplete or unrelated primary report')
    require(primary_protocol['physical_conditions'] == dict(depletant_radius=1.5,activity=.035,**{'lambda':2.24})
            and primary_protocol['estimator'] == report['estimator'], 'Changed admitted physical law or estimator')
    for path,digest in report['input_sha256'].items():
        require(bindings.get(path) == digest, 'Missing primary audit binding: '+path)
    path = Path(protocol['physical_contributions']).resolve()
    require(bindings.get(str(path)) == report['contributions_sha256'], 'Wrong all-attempt primary contributions')
    rows = []; seen = set(); jobs = primary_protocol['jobs']; inventory = {j['stratum_id']:j for j in jobs}
    require(len(jobs) == len(inventory) == 32, 'Changed fixed stratum inventory')
    for job in jobs:
        require(bindings.get(str(Path(job['config']['path']).resolve())) == job['config']['sha256'], 'Changed stratum configuration binding')
        config = bound(job['config']['path'])
        require(config['lambda'] == protocol['intensity'] and config['activity'] == protocol['activity'],
                'Cloud intensity or activity differs from actual scored stratum')
    with path.open() as stream:
        for line in stream:
            require(line.endswith('\n'), 'Truncated contribution row')
            row = json.loads(line); key = (row['stratum_id'],row['ordinal'])
            require(key not in seen and row['stratum_id'] in inventory, 'Duplicate or unknown attempted draw'); seen.add(key)
            job = inventory[row['stratum_id']]
            require((row['comparison_arm'],row['population_index'],row['component']) ==
                    (job['comparison_arm'],job['population_index'],job['component']), 'Changed attempt identity')
            memberships(row)
            if row['physical_valid']:
                lower = row['envelope']['lower_volume']; count = row['cloud_count_sum']
                require(row['envelope']['uncertain_volume'] > 0 or count == 0, 'Hits in empty envelope')
                terms = moment_terms(lower, count, protocol['intensity'], protocol['activity'], row['log_q_arm'])
                close(terms['log_y'], row['log_physical_weight'], 'primary pooled weight')
                row['moments'] = terms
            else:
                require(row['log_physical_weight'] is None and row['cloud_count_sum'] is None, 'Hard-invalid row lost zero status')
                row['moments'] = None
            rows.append(row)
    require(len(rows) == 32768 and sum(r['physical_valid'] for r in rows) == protocol['valid_poses'] == 7984,
            'Incomplete saved attempt allocation')
    strata = {}; stratum_reports = []
    for sid,job in inventory.items():
        selected = [r for r in rows if r['stratum_id'] == sid]; strata[sid] = selected
        require({r['ordinal'] for r in selected} == set(range(job['attempts'])), 'Missing unconditional stratum attempts')
        groups = {}
        for group in GROUPS:
            terms = [r['moments'] for r in selected if memberships(r)[group]]
            groups[group] = dict(summary=summarize(terms, job['attempts']),
                                normalizer_variance=stratum_variance(terms, job['attempts'], 4096))
        stratum_reports.append(dict(stratum_id=sid, comparison_arm=job['comparison_arm'],
            population_index=job['population_index'], component=job['component'], attempts=job['attempts'], groups=groups))
    populations = []
    for arm in ARMS:
        for population in range(4):
            selected = [r for r in rows if r['comparison_arm'] == arm and r['population_index'] == population]
            require(len(selected) == 4096, 'Changed unconditional population denominator')
            groups = {g:summarize([r['moments'] for r in selected if memberships(r)[g]],4096) for g in GROUPS}
            regions = {g:summarize([r['moments'] for r in selected if r['physical_valid'] and r['region'] == g],4096) for g in REGIONS}
            require_partition(groups['full_domain'], list(regions.values()))
            original = next(p for p in report['populations'] if (p['comparison_arm'],p['population_index']) == (arm,population))
            for g,value in groups.items():
                close(value['observed_importance_ess'], original['groups'][g]['physical']['importance_ess'], 'unchanged primary ESS '+g)
                require((value['log_mass'] is None) == (original['groups'][g]['physical']['log_mass'] is None), 'Changed empty mass')
                if value['log_mass'] is not None:
                    close(value['log_mass'], original['groups'][g]['physical']['log_mass'], 'unchanged primary mass '+g)
                for label,multiplier in INTENSITY_MULTIPLIERS.items():
                    forecast = value['intensity_forecasts'][label]
                    cost = original['geometry_cpu_seconds']+multiplier*original['physical_cpu_seconds']
                    forecast['linear_scoring_cost_cpu_seconds'] = cost if math.isfinite(cost) else None
                    forecast['linear_cost_ess_per_cpu_gain_plugin'] = (
                        forecast['second_moment_gain_plugin']*original['total_cpu_seconds']/cost
                        if math.isfinite(cost) and forecast['second_moment_gain_plugin'] is not None else None)
            radial = {}
            for chart in CHARTS:
                radial[chart] = {}
                for group in ('full_domain','A_T','remaining_without_A_T'):
                    bins = [summarize([r['moments'] for r in selected if memberships(r)[group]
                        and radial_index(r['mahalanobis_squared'][chart]) == i],4096) for i in range(6)]
                    require_partition(groups[group], bins); radial[chart][group] = bins
            relevant = [r for r in stratum_reports if (r['comparison_arm'],r['population_index']) == (arm,population)]
            variances = {g:{k:add_signed([r['groups'][g]['normalizer_variance'][k] for r in relevant])
                          for k in ('total','pose','auxiliary')} for g in GROUPS}
            populations.append(dict(comparison_arm=arm,population_index=population,groups=groups,regions=regions,
                radial_partitions=radial,normalizer_variance=variances,
                primary_geometry_and_scoring_cpu_seconds=original['total_cpu_seconds']))
    uncertainties = {arm:{g:ratio_uncertainty([p['groups'][g] for p in populations if p['comparison_arm'] == arm])
                         for g in GROUPS} for arm in ARMS}
    for path,digest in bindings.items():
        require(sha(path) == digest, 'Input changed during saved-count diagnostic: '+path)
    output = dict(schema='context-multicage-noise-budget-report-v1',complete=True,passed=True,
        protocol_sha256=sha(args.protocol),source_sha256=sha(__file__),input_sha256=bindings,
        all_attempts=32768,valid_poses=7984,audited_clouds=15968,populations=populations,
        strata=stratum_reports,ratio_uncertainty=uncertainties,primary_results_changed=False,
        new_clouds=0,new_geometry_queries=0,new_poses=0,finite_stability_conclusion=False,
        cpu_seconds=time.process_time()-started,
        identity='Y=exp(zL)(1+a)^S/q; P=exp(2zL)(1+2a)^S/q^2; V=Y^2-P>=0; conditional E[P]=true_weight^2/q^2 and E[V]=Var(Y|pose). S=K0+K1, a=z/(2lambda).',
        fixed_stratum_variance='For each generating stratum j with n_j draws: aux=sum V/N^2; pose=[sum P-sum_(i!=k)YiYk/(n_j-1)]/N^2. Sum over strata. Negative finite-sample pose estimates are retained.',
        intensity_forecast='Q_m=exp(2zL)*(1+2a+a^2/m)^S/q^2 is an unbiased estimator of the conditional second moment at m times the original pooled intensity. Q_1=Y^2 and Q_infinity=P. Projections retain the noisy original first-moment estimate. CPU projection assumes geometry_cost+m*scoring_cost; scoring includes overhead, so this is a crude unbenchmarked assumption. Larger-intensity realizations were not generated and would require separately authorized work caps.',
        scope='Saved counts at the historical fixed500uM environment; A_T is source contact pattern, not a native registry classifier. No primary weights, masses, allocations or conclusions changed.',
        limitations='The moment estimators are unbiased under the uncapped Poisson law, but their ratios and noise-free ESS proxies are not. Per-population ESS proxies can exceed attempt counts. Approximate Fieller sets from four populations are not coverage guarantees under heavy tails and cannot bound unseen contact modes. Removing all auxiliary noise is neither free nor a promise of CPU speedup; geometry coverage and the expense of drawing additional points remain.')
    require(not args.out.exists(), 'Fresh diagnostic output required')
    with args.out.open('x') as stream:
        json.dump(output,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(complete=True,passed=True,attempts=32768,cpu_seconds=output['cpu_seconds'],primary_results_changed=False)))


if __name__ == '__main__':
    main()
