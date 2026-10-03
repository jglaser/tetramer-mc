#!/usr/bin/env python3
"""Frozen, scalar-only rejection diagnostic for completed evolving-dimer chains.

No coordinates are interpreted, geometry is constructed, or clouds are drawn.
Factor-removal diagnostics reuse recorded candidates and uniforms. They overlap
and are not alternative reversible kernels or predicted equilibrium rates.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import resource
import statistics
import time


def require(test, message):
    if not test: raise ValueError(message)


def read(path): return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def write(path, value):
    with Path(path).open('x') as target:
        target.write(json.dumps(value, indent=2, allow_nan=False)+'\n')


def finite_or_minus(value):
    if value == '-inf': return -math.inf
    require(type(value) in (int, float) and math.isfinite(value), 'Invalid saved log factor')
    return float(value)


def close(a, b, label):
    require(a == b or (math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-8)), label)


def probability(log_ratio): return math.exp(min(0., log_ratio))


def prepare(base, out):
    base, out = Path(base).resolve(), Path(out).resolve()
    require(not out.exists(), 'Fresh diagnostic directory required')
    cfg = read(base/'config.json'); require(len(cfg['jobs']) == 96, 'Expected all96 chains')
    bindings = {str(p): sha(p) for p in [base/'config.json', base/'run-binding.json', Path(__file__).resolve()]}
    source = base/'common/source/src'
    for name in ['evolving_dimer.rs', 'factorized_dimer.rs', 'bounded_singleton_path.rs',
                 'defensive_dimer_proposal.rs', 'capped_dimer.rs']:
        bindings[str(source/name)] = sha(source/name)
    jobs = []
    for job in cfg['jobs']:
        terminal_path = base/f"execution/job-{job['id']:03d}/terminal.json"
        terminal = read(terminal_path)
        require(terminal['complete'] is True and terminal['job'] == job, 'Incomplete/mismatched job')
        require(terminal['config_sha256'] == bindings[str(base/'config.json')]
                and terminal['binding_sha256'] == bindings[str(base/'run-binding.json')], 'Changed completion bindings')
        bindings[str(terminal_path)] = sha(terminal_path)
        trajectory = Path(terminal['trajectory']['path']).resolve()
        require(trajectory == base/f"execution/job-{job['id']:03d}/trajectory.jsonl", 'Unexpected trajectory path')
        jobs.append(dict(job=job, context=cfg['contexts'][job['context_index']]['name'],
            terminal=str(terminal_path), trajectory=str(trajectory), trajectory_sha256=terminal['trajectory']['sha256'],
            trajectory_bytes=trajectory.stat().st_size, counts=terminal['counts']))
    plan = dict(schema='evolving-dimer-rejection-diagnostic-v1', base=str(base), jobs=jobs,
        files=bindings, maximum_workers=1, cpu_limit_seconds=900, wall_limit_seconds=1200,
        rows='Every elementary attempt in all96 completion-bound trajectories, including rejected and capped/self-loop attempts.',
        windows=dict(warmup='blocks1..512', production='blocks513..4608'),
        stratification=['context', 'initialization', 'arm', 'stream', 'window'],
        metrics=dict(disjoint_dispositions=['source_outside_domain', 'root_cap_exhausted', 'internal_cap_exhausted',
                     'final_rejected', 'candidate_rejected', 'candidate_accepted'],
            proposal_factor='f=full log F(old)-log F(new)',
            auxiliary_factor='a=m*(log1p(Kold)-log1p(Knew)); unguided a=0',
            bath_factor='b=log1p(z/lambda)*(gained-lost), summed over the two sequential bath legs',
            decision='accepted iff saved log_u < min(0,f+a+b)',
            factor_removal='For every candidate: alpha and decision at unchanged candidate/uniform with one factor set to zero. Self-loop attempts contribute zero. Overlapping veto counts, not exclusive causal effects or valid alternative kernels.',
            summaries='All candidate log-factor quantiles, negative shares, and gain/loss counts; no accepted-only selections.',
            local='All local hard-rejection/physical-gate/acceptance counters; same step schedule across arms.'),
        expected_attempts=dict(local=cfg['allocation']['local_attempts'], dimer=cfg['allocation']['dimer_attempts']),
        new_geometry_queries=0, new_Poisson_clouds=0, new_sampling_attempts=0,
        frozen_before_scalar_event_reads=True,
        limitations='Conditional2-mobile benchmark at rd1.4,z.0275; fixed spectators. No equilibrium/native/finite-system assembly inference.')
    out.mkdir(); write(out/'plan.json', plan)
    return plan


def accumulator(): return dict(counts=Counter(), values=defaultdict(list), expected=Counter(), removed=Counter())


def add(target, row, cfg):
    c = target['counts']; c['attempted'] += 1
    accepted = row['accepted']; require(type(accepted) is bool, 'Invalid acceptance flag')
    c['accepted'] += accepted
    if row['kind'] == 'local':
        c['local_attempted'] += 1; c['local_accepted'] += accepted
        require(row['status'] in ('hard_rejected', 'completed'), 'Incomplete local event')
        if row['status'] == 'hard_rejected':
            require(not accepted, 'Hard rejection accepted'); c['local_hard_rejected'] += 1; return
        c['local_bath_candidates'] += 1
        b = finite_or_minus(row['bath']['log_weight']); close(b, finite_or_minus(row['log_acceptance_ratio']), 'Local ratio differs')
        require(accepted == (row['log_u'] < min(0., b)), 'Local decision differs')
        target['values']['local_bath_log'].append(b)
        target['expected']['local_acceptances'] += probability(b)
        c['local_bath_rejected'] += not accepted
        return
    c['dimer_attempted'] += 1; c['dimer_accepted'] += accepted
    p = row['proposal']; status = row['status']
    if status == 'proposal_self_loop':
        require(not accepted and p['candidate'] is None, 'Invalid self-loop')
        c['dimer_self_loop'] += 1
        if p['status'] == 'source_outside_domain': c['source_outside_domain'] += 1
        else:
            require(p['status'] == 'cap_exhausted' and len(p['attempts']) == 1, 'Unexpected capped law')
            disposition = p['attempts'][0]['status']
            require(disposition in ('root_cap_exhausted', 'internal_cap_exhausted', 'final_rejected'), 'Unexpected cap disposition')
            c[disposition] += 1
    else:
        require(status == 'completed' and p['status'] == 'candidate', 'Incomplete dimer event')
        c['dimer_candidates'] += 1; c['candidate_accepted' if accepted else 'candidate_rejected'] += 1
        f = finite_or_minus(p['candidate']['diagnostics']['log_reverse_forward'])
        guide = p.get('guidance'); a = 0.
        if guide is not None:
            a = guide['m']*(math.log1p(guide['old_count'])-math.log1p(guide['new_count']))
            close(a, guide['aux_log_correction'], 'Auxiliary correction differs')
            require(guide['threshold'] <= min(guide['old_count'], guide['new_count']), 'Threshold support differs')
            for key in ('old_count', 'new_count', 'threshold'): target['values']['guidance_'+key].append(guide[key])
        close(f+a, finite_or_minus(row['complete_log_correction']), 'Complete proposal correction differs')
        bath = row['bath']['aggregate']; b = math.log1p(1/cfg['physical']['lambda_ratio'])*(bath['gained']-bath['lost'])
        close(b, bath['log_weight'], 'Bath gained/lost correction differs')
        ratio = f+a+b; close(ratio, finite_or_minus(row['log_acceptance_ratio']), 'Total acceptance ratio differs')
        log_u = row['log_u']; require(math.isfinite(log_u) and log_u < 0 and accepted == (log_u < min(0., ratio)), 'Dimer decision differs')
        for name, value in [('proposal', f), ('threshold', a), ('bath', b), ('total', ratio)]:
            target['values'][name+'_log'].append(value)
        for name in ('gained', 'lost'): target['values']['bath_'+name].append(bath[name])
        target['expected']['full_acceptances'] += probability(ratio)
        for name, without in [('proposal', a+b), ('threshold', f+b), ('bath', f+a)]:
            target['expected']['without_'+name] += probability(without)
            hypothetical = log_u < min(0., without)
            target['removed']['without_'+name+'_acceptances'] += hypothetical
            target['removed'][name+'_decisive_veto'] += not accepted and hypothetical
            target['removed'][name+'_positive_rescue'] += accepted and not hypothetical
    # Count all raw edge filters, including each rejected trial; no atom rechecks.
    for attempt in p['attempts']:
        for stage in ('root', 'internal'):
            for trial in attempt[stage+'_draws']:
                c[stage+'_raw_trials'] += 1
                geom = trial['feasibility']; require(geom is not None, 'Missing trial feasibility')
                if stage == 'root':
                    good = not geom['spectator_core_collisions'] and geom['wall_valid']
                    c['root_spectator_collision_trials'] += bool(geom['spectator_core_collisions'])
                    c['root_wall_invalid_trials'] += not geom['wall_valid']
                else:
                    good = not geom['internal_core_overlap'] and geom['internal_exclusion_contact']
                    c['internal_core_overlap_trials'] += geom['internal_core_overlap']
                    c['internal_no_contact_trials'] += not geom['internal_exclusion_contact']
                    if good and p.get('guidance') is not None:
                        c['internal_threshold_tested'] += 1
                        threshold_good = trial['guidance_count'] >= p['guidance']['threshold']
                        c['internal_threshold_rejected_trials'] += not threshold_good
                c[stage+'_geometry_pass_trials'] += good
        if attempt['status'] == 'final_rejected':
            final = attempt['final_feasibility']
            c['final_spectator_collision'] += any(final['spectator_core_collisions'])
            c['final_wall_invalid'] += not all(final['wall_valid'])
            c['final_internal_core_overlap'] += final['internal_core_overlap']
            c['final_internal_no_contact'] += not final['internal_exclusion_contact']


def describe(values):
    finite = sorted(v for v in values if math.isfinite(v)); n = len(values)
    def quantile(p):
        if not finite: return None
        x = (len(finite)-1)*p; i = int(x); return finite[i] if i+1 == len(finite) else finite[i]+(x-i)*(finite[i+1]-finite[i])
    return dict(count=n, negative=sum(v < 0 for v in values), minus_infinite=n-len(finite),
        finite_mean=statistics.fmean(finite) if finite else None,
        finite_quantiles={str(p): quantile(p) for p in (0., .05, .25, .5, .75, .95, 1.)})


def present(value):
    n = value['counts']['dimer_attempted']
    return dict(counts=dict(value['counts']), distributions={k: describe(v) for k, v in value['values'].items()},
        expected_acceptance_counts=dict(value['expected']),
        expected_dimer_acceptance_per_attempt={k: v/n for k, v in value['expected'].items() if n and k != 'local_acceptances'},
        same_candidate_uniform_diagnostics=dict(value['removed']))


def run(out):
    out = Path(out).resolve(); plan = read(out/'plan.json'); plan_sha = sha(out/'plan.json')
    require(not (out/'claim.json').exists(), 'No retry or extension')
    for path, digest in plan['files'].items(): require(sha(path) == digest, 'Changed metadata/source binding '+path)
    write(out/'claim.json', dict(plan_sha256=plan_sha, started=time.time()))
    resource.setrlimit(resource.RLIMIT_CPU, (plan['cpu_limit_seconds'], plan['cpu_limit_seconds']))
    cfg = read(Path(plan['base'])/'config.json'); start = time.monotonic(); chains = []; groups = defaultdict(accumulator)
    progress = (out/'progress.jsonl').open('x')
    try:
        for item in plan['jobs']:
            job = item['job']; chain = {phase: accumulator() for phase in ('warmup', 'production')}
            counts = Counter(); digest = hashlib.sha256(); events = 0; blocks = 0; initial = 0
            progress.write(json.dumps(dict(state='begin', job=job['id']))+'\n'); progress.flush()
            with Path(item['trajectory']).open('rb') as source:
                for raw in source:
                    require(time.monotonic()-start < plan['wall_limit_seconds'], 'Diagnostic wall budget exceeded')
                    digest.update(raw); require(raw.endswith(b'\n'), 'Partial trajectory line')
                    row = json.loads(raw); kind = row['kind']
                    if kind == 'initial': initial += 1; continue
                    if kind == 'retained_block': blocks += 1; continue
                    require(kind in ('local', 'factorized_dimer'), 'Unknown event kind '+kind)
                    phase = 'warmup' if row['block'] <= cfg['allocation']['warmup_blocks'] else 'production'
                    add(chain[phase], row, cfg)
                    key = (item['context'], job['initialization'], job['arm'], phase)
                    add(groups[key], row, cfg)
                    events += 1
                    prefix = 'local' if kind == 'local' else 'dimer'
                    counts[prefix+'_attempted'] += 1; counts[prefix+'_accepted'] += row['accepted']
                    if kind == 'factorized_dimer': counts['dimer_self_loop'] += row['status'] == 'proposal_self_loop'
            require(digest.hexdigest() == item['trajectory_sha256'], 'Trajectory digest differs')
            require(initial == 1 and blocks == 4608, 'Missing initial/block endpoints')
            require(all(counts[k] == v for k, v in item['counts'].items()), 'Terminal scalar counters differ')
            require(sum(chain[p]['counts']['local_attempted'] for p in chain) == 18432, 'Missing local attempts')
            require(sum(chain[p]['counts']['dimer_attempted'] for p in chain) == (0 if job['arm'] == 'local' else 4608), 'Missing collective attempts')
            chains.append(dict(job=job, context=item['context'], windows={p: present(v) for p, v in chain.items()},
                               trajectory_sha256=digest.hexdigest(), trajectory_events=events))
            progress.write(json.dumps(dict(state='complete', job=job['id'], scalar_events=events))+'\n'); progress.flush()
        require(len(chains) == 96, 'Missing chain')
        for path, digest in plan['files'].items(): require(sha(path) == digest, 'Metadata/source changed during diagnostic')
        require(sha(out/'plan.json') == plan_sha, 'Plan changed')
        result = dict(schema='evolving-dimer-rejection-diagnostic-results-v1', complete=True, plan_sha256=plan_sha,
            chains=chains, groups=[dict(context=k[0], initialization=k[1], arm=k[2], window=k[3], **present(v)) for k, v in sorted(groups.items())],
            wall_seconds=time.monotonic()-start, new_geometry_queries=0, new_Poisson_clouds=0,
            scope=plan['limitations'], counterfactual_warning=plan['metrics']['factor_removal'])
        write(out/'analysis.json', result)
        write(out/'summary.json', dict(complete=True, chains=96, plan_sha256=plan_sha,
            analysis_sha256=sha(out/'analysis.json'), new_geometry_queries=0, new_Poisson_clouds=0))
        return result
    except BaseException as error:
        write(out/'failure.json', dict(complete=False, error=repr(error), completed_chains=len(chains),
            current_job=item['job']['id'] if 'item' in locals() else None, partial_outputs_retained=True))
        raise
    finally: progress.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--base', type=Path)
    p.add_argument('--out', required=True, type=Path); p.add_argument('--run', action='store_true')
    args = p.parse_args()
    value = run(args.out) if args.run else prepare(args.base, args.out)
    print(json.dumps(dict(complete=True, chains=len(value.get('jobs', value.get('chains', []))), new_geometry_queries=0)))
