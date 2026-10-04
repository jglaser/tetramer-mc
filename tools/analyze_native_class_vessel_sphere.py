#!/usr/bin/env python3
"""Analyze completed sphere rows under the frozen allocation; never draw poses."""
from __future__ import annotations
import argparse
import math
import os
from pathlib import Path
import statistics
import sys

from analyze_r4_smc_control import Ledger
import run_native_class_physical_campaign as driver
from prepare_native_class_vessel_sphere import (
    SCHEMA, ACTIVITIES, CENTER, CORE, RD, WALL, SOURCE_CAPTURE, POPULATIONS, DRAWS,
    TOTAL, MAXIMUM_CLOUDS, REGIONS, CRITERIA, jobs_for, overlap, read, require, sha, write,
)


def mean_stats(values):
    require(len(values) >= 2 and all(math.isfinite(v) for v in values), 'Finite repeated values required')
    mean = statistics.fmean(values)
    variance = math.fsum((v-mean)**2 for v in values)/(len(values)-1)
    return dict(mean=mean, sample_variance=variance, standard_error=math.sqrt(variance/len(values)), count=len(values))


def reference_check(values, reference):
    require(len(values) == POPULATIONS, 'Four independent population means required')
    s = mean_stats(values); q = reference['mass']; error = reference['quadrature_error_bound']
    fp = CRITERIA['floating_point_relative_allowance']*q
    tolerance = CRITERIA['population_SE_multiplier']*s['standard_error']+error+fp
    discrepancy = s['mean']-q
    log_error = abs(math.log(s['mean']/q)) if s['mean'] > 0 else None
    return dict(**s, per_population=values, analytic=q, quadrature_error_bound=error,
        floating_point_allowance=fp, discrepancy=discrepancy, linear_tolerance=tolerance,
        absolute_log_discrepancy=log_error, statistical_check=abs(discrepancy) <= tolerance,
        magnitude_check=log_error is not None and log_error <= CRITERIA['maximum_absolute_log_discrepancy'],
        passed=abs(discrepancy) <= tolerance and log_error is not None and log_error <= CRITERIA['maximum_absolute_log_discrepancy'])


def close(a, b, label, tolerance=5e-10):
    require(math.isfinite(a) and math.isfinite(b) and abs(a-b) <= tolerance*(1+abs(a)+abs(b)), label)


def masks(radius, valid):
    return dict(total=valid, contact=valid and radius < 1.6, unbound=valid and radius >= 1.6,
                inside_source=valid and radius <= SOURCE_CAPTURE, outside_source=valid and radius > SOURCE_CAPTURE)


def snapshot_gate_status(live_path, snapshot_path, ledger):
    """Bind exact immutable gate bytes; the driver owns the changing live file."""
    live_path, snapshot_path = Path(live_path).resolve(), Path(snapshot_path).resolve()
    require(live_path != snapshot_path, 'Gate snapshot must be separate from live status')
    raw = live_path.read_bytes()
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    with snapshot_path.open('xb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    value = read(ledger.bind(snapshot_path))
    def verify_live_unchanged():
        require(live_path.read_bytes() == raw, 'Live controller status changed during statistics gate')
    verify_live_unchanged()
    return value, verify_live_unchanged


def authenticated_predecessors(root, output, ledger):
    """Require all 16 audited predecessors drained, and this live final worker."""
    # Gate before any population/scientific output is opened.
    plan_path = root/'execution-plan.json'; plan = read(ledger.bind(plan_path))
    status, verify_live_status = snapshot_gate_status(root/'execution/status.json',
        output.parent/'gate-status.json', ledger)
    claim = read(ledger.bind(root/'execution/claim.json'))
    require(plan['schema'] == driver.SCHEMA and plan['root'] == str(root)
        and plan['maximum_workers'] == plan['threads'] == 1 and len(plan['jobs']) == 17,
        'Wrong sequential sphere execution contract')
    require(status['failure'] is None and status['unstarted'] == [] and len(status['completed']) == 16
        and status['complete'] is False and status['active']['ordinal'] == 16
        and status['active']['id'] == plan['jobs'][16]['id'] and status['active']['phase'] == 'statistics'
        and status['plan_sha256'] == claim['plan_sha256'] == sha(plan_path)
        and claim['pid'] == os.getppid(), 'Not the active final statistics worker after drained predecessors')
    stat = Path(f'/proc/{claim["pid"]}/stat').read_text().rsplit(')', 1)[1].split()
    require(stat[0] not in ('Z','X') and int(stat[19]) == claim['birth_ticks'], 'Controller process identity changed')
    last = plan['jobs'][16]
    require(last['terminal'] == dict(path=str(output), success_contract='complete_and_passed')
        and last['argv'] == [sys.executable, '-B', str(Path(__file__).resolve()), '--root', str(root), '--out', str(output)],
        'Statistics command/terminal differs')
    driver.verify_plan(plan_path, plan, sha(plan_path))
    terminals = []
    for done, job in zip(status['completed'], plan['jobs'][:16]):
        terminal = driver.completed_terminal(root, plan, job['id'])
        require(done['terminal'] == terminal and done['child_drained'] is True, 'Predecessor terminal changed')
        ledger.bind(terminal['path'], terminal['sha256']); terminals.append(terminal)
    verify_live_status()
    return plan, terminals


def check_preparation(root, plan, execution, ledger):
    ledger.frozen(root)
    require(plan['schema'] == SCHEMA and plan['root'] == str(root) and plan['launched'] is False
        and plan['preparation_only'] is True and plan['criteria'] == CRITERIA
        and plan['total_attempts'] == TOTAL and plan['maximum_clouds'] == MAXIMUM_CLOUDS
        and plan['populations_per_activity'] == POPULATIONS and plan['samples_per_population'] == DRAWS
        and plan['activities'] == list(ACTIVITIES) and plan['retries'] == plan['replacements'] == plan['extensions'] == 0,
        'Changed sphere allocation/criteria')
    require(plan['jobs'] == jobs_for(root, plan['python']), 'Changed fixed commands/seeds')
    require(plan['python'] == sys.executable and plan['python_sha256'] == sha(Path(sys.executable).resolve()), 'Python runtime changed')
    for value in (plan['executable'], plan['source_bundle'], plan['analytic_reference']): ledger.bind(value['path'], value['sha256'])
    expected = []
    for job in plan['jobs']:
        expected.extend([(job['argv'],job['producer_terminal'],'producer'),(job['audit_argv'],job['audit_terminal'],'geometry')])
    for (argv, terminal, phase), actual in zip(expected, execution['jobs'][:16]):
        require(actual['argv'] == argv and actual['terminal'] == terminal and actual['phase'] == phase,
                'Execution order/commands differ from frozen sphere job')
    for relative, digest in plan['files_sha256'].items():
        path = (root/relative).resolve(); require(path.is_relative_to(root), 'Unsafe preparation path'); ledger.bind(path,digest)


def population(job, plan, audit, ledger, companion_path):
    directory = Path(job['directory']); summary = read(ledger.bind(directory/'summary.json'))
    manifest = read(ledger.bind(directory/'manifest.json'))
    require(summary['complete'] is True and summary['manifest'] == manifest and summary['numerical_nulls'] == 0
        and manifest['samples'] == summary['samples'] == job['samples'] == DRAWS
        and manifest['seed'] == job['seed'] and manifest['activity'] == job['activity']
        and manifest['cloud_replicates'] == 2 and manifest['schema'] == 7
        and manifest['executable_sha256'] == plan['executable']['sha256']
        and manifest['source_bundle_sha256'] == plan['source_bundle']['sha256'], 'Wrong sphere population identity')
    require(audit['schema'] == 'full-vessel-native-class-line-independent-audit-v1' and audit['complete'] is True
        and audit['population'] == str(directory) and audit['manifest'] == manifest
        and audit['new_pose_draws'] == audit['new_Poisson_clouds'] == 0, 'Missing matching independent row audit')
    for path, digest in audit['source_sha256'].items(): ledger.bind(path,digest)
    sample_path = ledger.bind(directory/'samples.jsonl', summary['samples_sha256'])
    ledger.bind(directory/'attempts.jsonl', summary['attempts_sha256'])
    require(audit['samples_sha256'] == summary['samples_sha256'] and audit['attempts_sha256'] == summary['attempts_sha256'], 'Audit rows differ')
    import json
    rows = [json.loads(line) for line in sample_path.read_text().splitlines()]
    attempts = [json.loads(line) for line in (directory/'attempts.jsonl').read_text().splitlines()]
    require(len(rows) == DRAWS and attempts == [dict(draw=i,state='begin') for i in range(DRAWS)], 'Attempted denominator lost')
    values = {name:{kind:[] for kind in ('exact','noisy','hard','residual','replica1','replica2','conditional_variance','empirical_cloud_variance')}
              for name in REGIONS}
    negatives = dict(extra_jacobian=[], source_censored=[])
    counts = dict(valid=0,invalid_zeros=0,inside_R4=0,outside_R4=0,clouds=0)
    cloud = {name:[0.,0.] for name in ('raw_count','overlap_count','replica_difference','centered_product')}
    max_slack = 0.
    with companion_path.open('x') as out:
        for i,row in enumerate(rows):
            require(row['draw'] == i, 'Reordered attempted pose')
            radius = math.dist(row['pose']['position'], CENTER)
            valid = 2*CORE <= radius <= WALL-CORE
            require(row['hard_valid'] == valid, 'Analytic hard-wall predicate differs')
            logq = row['log_proposal_density']; require(type(logq) in (int,float) and math.isfinite(logq), 'Nonfinite full proposal density')
            selected = masks(radius, valid); volume = overlap(radius)
            inverse = math.exp(-logq) if valid else 0.
            exact = math.exp(job['activity']*volume-logq) if valid else 0.
            noisy = variance = empirical = 0.; replicas = [0.,0.]
            if valid:
                counts['valid'] += 1; counts['clouds'] += 2
                counts['inside_R4' if row['latent_density']['in_reference_ball'] else 'outside_R4'] += 1
                close(row['log_hard_weight'], -logq, 'Hard weight differs')
                clouds = row['clouds']; require(len(clouds) == 2, 'Two independent clouds required')
                fields = ('lower_volume','upper_volume','uncertain_volume','created_cells','certified_cells','retained_cells')
                require(all(clouds[0].get(k) == clouds[1].get(k) for k in fields), 'Replica envelopes differ')
                lower, upper, uncertain = (clouds[0][k] for k in fields[:3])
                require(all(math.isfinite(v) for v in (lower,upper,uncertain)) and lower >= 0 and uncertain >= 0, 'Invalid envelope')
                slack = max(lower-volume, volume-upper, 0.); max_slack=max(max_slack,slack)
                require(slack <= 5e-11*(1+volume), 'Analytic sphere overlap outside saved envelope')
                close(upper-lower,uncertain,'Envelope decomposition differs')
                remainder = max(0.,volume-lower)  # tolerated FP slack only, reported above
                z, lam = job['activity'], manifest['lambda']
                variance = exact*exact*.5*math.expm1(z*z*remainder/lam)
                replicas = [math.exp(c['log_weight']-logq) for c in clouds]
                noisy = math.fsum(replicas)/2
                empirical = (replicas[0]-replicas[1])**2/4
                close(math.log(noisy),row['log_importance_weight'],'Arithmetic cloud average/full-q identity differs')
                for c in clouds:
                    expected_log = z*lower+c['overlap_points']*math.log1p(z/lam)
                    close(c['log_weight'],expected_log,'Poisson PGF weight identity differs')
                    require(type(c['raw_points']) is int and type(c['overlap_points']) is int
                        and 0 <= c['overlap_points'] <= c['raw_points'], 'Malformed cloud count')
                    if z > 0:
                        cloud['raw_count'][0] += c['raw_points']-lam*uncertain
                        cloud['raw_count'][1] += lam*uncertain
                        cloud['overlap_count'][0] += c['overlap_points']-lam*remainder
                        cloud['overlap_count'][1] += lam*remainder
                    else: require(c['raw_points'] == c['overlap_points'] == 0, 'Hard-only cloud must be empty')
                cloud['replica_difference'][0] += replicas[0]-replicas[1]
                cloud['replica_difference'][1] += 4*variance
                cloud['centered_product'][0] += (replicas[0]-exact)*(replicas[1]-exact)
                cloud['centered_product'][1] += (2*variance)**2
                if z == 0: close(noisy,exact,'Hard-only exact/noisy identity differs')
            else:
                counts['invalid_zeros'] += 1
                require(row['log_importance_weight'] is None and row['log_hard_weight'] is None and row['clouds'] == [], 'Invalid draw was not a retained zero')
            for name,keep in selected.items():
                for kind,value in dict(exact=exact,noisy=noisy,hard=inverse,residual=noisy-exact,
                    replica1=replicas[0],replica2=replicas[1],conditional_variance=variance,empirical_cloud_variance=empirical).items():
                    values[name][kind].append(value if keep else 0.)
            log_jacobian = row['latent_density']['log_physical_jacobian']
            if valid and log_jacobian is None:
                require(row['latent_density']['coordinate_chart_seam'] is True
                    and row['latent_density']['latent'] is None and row['latent_density']['structural_zero'] is True,
                    'Missing nonseam Jacobian')
            jacobian = math.exp(log_jacobian) if valid and log_jacobian is not None else 0.
            negatives['extra_jacobian'].append(exact*jacobian)
            negatives['source_censored'].append(exact if selected['inside_source'] else 0.)
            out.write(json.dumps(dict(draw=i,radius=radius,overlap_volume=volume,hard_valid=valid,regions=selected,
                log_full_proposal=logq,exact_log_importance=job['activity']*volume-logq if valid else None,
                exact_importance=exact,noisy_importance=noisy,hard_importance=inverse,
                poisson_residual=noisy-exact,conditional_residual_variance=variance,
                empirical_cloud_variance=empirical,replica_importance=replicas,
                extra_jacobian_negative=negatives['extra_jacobian'][-1],source_censored_negative=negatives['source_censored'][-1]),
                separators=(',',':'),allow_nan=False)+'\n')
    require(counts['valid'] == summary['hard_valid'], 'Valid count differs')
    statistics_by_region = {}
    for name, series in values.items():
        s = {kind:mean_stats(v) for kind,v in series.items() if kind not in ('conditional_variance','empirical_cloud_variance')}
        s['known_conditional_residual_SE'] = math.sqrt(math.fsum(series['conditional_variance']))/DRAWS
        s['mean_conditional_cloud_variance'] = statistics.fmean(series['conditional_variance'])
        s['mean_empirical_cloud_variance'] = statistics.fmean(series['empirical_cloud_variance'])
        statistics_by_region[name] = s
    for key,kind in [('total','noisy'),('hard_total','hard')]:
        close(summary['estimates'][key]['log_normalizer'],math.log(statistics_by_region['total'][kind]['mean']), 'Summary all-attempt mean differs')
    return dict(id=job['id'],activity=job['activity'],population=job['population'],samples=DRAWS,seed=job['seed'],
        regions=statistics_by_region,negative_controls={k:mean_stats(v) for k,v in negatives.items()},counts=counts,
        cloud_diagnostics={name:dict(residual=v[0],known_variance=v[1],z=v[0]/math.sqrt(v[1]) if v[1]>0 else None,
            within_fixed_five_sigma=abs(v[0])<=CRITERIA['cloud_diagnostic_z_limit']*math.sqrt(v[1])+1e-10,
            decision_role='diagnostic only; primary PGF gate uses conditional variance of paired residuals') for name,v in cloud.items()},
        maximum_envelope_slack=max_slack,sampler_cpu_seconds=summary['sampler_cpu_seconds'],
        samples_sha256=summary['samples_sha256'],companion=dict(path=str(companion_path),sha256=sha(companion_path)))


def aggregate(populations, reference):
    checks=[]; groups={}
    for z in ACTIVITIES:
        selected=[p for p in populations if p['activity']==z]
        require(len(selected)==POPULATIONS and {p['population'] for p in selected}==set(range(POPULATIONS)), 'Incomplete independent populations')
        groups[str(z)]={}
        for name in REGIONS:
            group={}
            for kind in ('exact','noisy','hard'):
                target=reference['references'][str(0. if kind=='hard' else z)][name]
                values=[p['regions'][name][kind]['mean'] for p in selected]
                result=reference_check(values,target);group[kind]=result
                checks.append(dict(name=f'z{z:g}:{name}:{kind}:reference',passed=result['passed']))
            residuals=[p['regions'][name]['residual']['mean'] for p in selected]
            known=math.sqrt(math.fsum(p['regions'][name]['known_conditional_residual_SE']**2 for p in selected))/POPULATIONS
            residual=mean_stats(residuals);mean=residual['mean'];fp=CRITERIA['floating_point_relative_allowance']*reference['references'][str(z)][name]['mass']
            sigma=CRITERIA['paired_residual_sigma_multiplier']
            residual.update(known_conditional_SE=known,passed=abs(mean)<=sigma*known+fp,
                empirical_paired_SE=residual['standard_error'],conditional_sigma_multiplier=sigma,floating_point_allowance=fp)
            group['poisson_minus_exact']=residual
            checks.append(dict(name=f'z{z:g}:{name}:known-Poisson-residual',passed=residual['passed']))
            groups[str(z)][name]=group
        negative={}
        for kind in ('extra_jacobian','source_censored'):
            values=[p['negative_controls'][kind]['mean'] for p in selected]
            test=reference_check(values,reference['references'][str(z)]['total'])
            negative[kind]=test
            checks.append(dict(name=f'z{z:g}:{kind}:must-fail-full-target',passed=not test['passed']))
        censored=[p['negative_controls']['source_censored']['mean'] for p in selected]
        inside=[p['regions']['inside_source']['exact']['mean'] for p in selected]
        require(censored==inside,'Source censoring changed the unconditional denominator')
        negative['source_censored_inside_target']=reference_check(censored,reference['references'][str(z)]['inside_source'])
        checks.append(dict(name=f'z{z:g}:source-censored:matches-inside',passed=negative['source_censored_inside_target']['passed']))
        groups[str(z)]['negative_controls']=negative
    return groups,checks


def analyze(root, output):
    root,output=Path(root).resolve(),Path(output).resolve();require(not output.exists(),'Fresh statistics output required')
    ledger=Ledger();execution,terminals=authenticated_predecessors(root,output,ledger)
    plan=read(ledger.bind(root/'plan.json'));check_preparation(root,plan,execution,ledger)
    reference=read(ledger.bind(plan['analytic_reference']['path'],plan['analytic_reference']['sha256']))
    output.parent.mkdir(parents=True,exist_ok=True);companions=output.parent/'exact-weights';companions.mkdir()
    populations=[]
    for i,job in enumerate(plan['jobs']):
        require(terminals[2*i]['path']==job['producer_terminal']['path'] and terminals[2*i+1]['path']==job['audit'], 'Predecessor order differs')
        audit=read(ledger.bind(job['audit'],terminals[2*i+1]['sha256']))
        populations.append(population(job,plan,audit,ledger,companions/(job['id']+'.jsonl')))
    require(sum(p['samples'] for p in populations)==TOTAL and sum(p['counts']['clouds'] for p in populations)<=MAXIMUM_CLOUDS, 'Allocation exceeded')
    groups,checks=aggregate(populations,reference)
    exercised=all(sum(p['counts'][k] for p in populations)>0 for k in ('invalid_zeros','outside_R4'))
    checks.append(dict(name='invalid-and-exterior-accounting-exercised',passed=exercised))
    ledger.recheck()
    result=dict(schema='native-class-vessel-sphere-analysis-v1',complete=True,passed=all(c['passed'] for c in checks),
        plan_sha256=sha(root/'plan.json'),execution_plan_sha256=sha(root/'execution-plan.json'),
        populations=populations,groups=groups,checks=checks,failed_checks=[c for c in checks if not c['passed']],
        attempts=TOTAL,maximum_clouds=MAXIMUM_CLOUDS,actual_clouds=sum(p['counts']['clouds'] for p in populations),
        input_sha256=ledger.files,new_pose_draws=0,new_clouds=0,new_geometry_queries=0,
        scope='Fixed analytic sphere implementation references; population SE is from four independent linear means. Correlated checks are not a global confidence statement or protein convergence. R4 occupancy is descriptive only. Failed checks remain unresolved and authorize no replacement or extension.')
    write(output,result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();r=analyze(args.root,args.out)
    import json
    print(json.dumps(dict(complete=True,passed=r['passed'],attempts=r['attempts'],failed_checks=r['failed_checks'])))
