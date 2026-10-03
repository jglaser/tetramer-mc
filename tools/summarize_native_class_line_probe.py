#!/usr/bin/env python3
"""Summarize audited proposal-only rows, without geometry or physical weights."""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import statistics
import os
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[_name] = '1'
import numpy as np
from run_native_class_line_probe import read, require, sha, verify, validate_terminal


def rows(path):
    with Path(path).open() as stream:
        for line in stream: yield json.loads(line)


def label(row):
    if not row['shell_valid'] or not row['capture_valid']: return 'outside_region_or_capture'
    if not row['hard_valid']: return 'hard_invalid'
    require(row['native_decision'] is not None, 'Missing valid-pose native label')
    if row['native_decision']['native_any']: return 'native'
    if any(row['exclusion_contact_by_anchor']): return 'contact_without_native'
    return 'unbound'


def hard_free_log_density(row, guide, radius=4.):
    """Old H law reconstructed algebraically from the already audited trace."""
    alpha = guide['defensive_uniform_shell_probability']; beta = guide['conditional_probability']; terms = []
    if row['shell_valid']:
        terms.append(math.log(alpha)-(3*math.log(math.pi)+6*math.log(radius)-math.log(6)))
    if alpha == 1.: return terms[0] if terms else None
    u = np.asarray(row['latent']); x = row['raw_coordinates']
    axes = row['density_details'].get('axes', [])
    require(beta == 0. or len(axes) == len(guide['raw_translation_axes']), 'Missing hard-free axes')
    total_weight = sum(c['weight'] for c in guide['gaussian_components'])
    for i, component in enumerate(guide['gaussian_components']):
        covariance = np.asarray(component['covariance']); lower = np.linalg.cholesky(covariance)
        z = np.linalg.solve(lower, u-np.asarray(component['mean']))
        gaussian = -3*math.log(2*math.pi)-float(np.log(np.diag(lower)).sum())-float(z@z)/2
        factors = []
        for axis in axes:
            value = x[axis['axis']]
            allowed = any((value > r['lower'] or value == r['lower'] and r['lower_closed']) and
                          (value < r['upper'] or value == r['upper'] and r['upper_closed'])
                          for r in axis['hard_free_intervals'])
            mass = axis['components'][i]['channels'][0]['hard_free_mass']
            factors.append(1. if mass <= guide['minimum_conditional_mass'] else (1/mass if allowed else 0.))
        factor = (1-beta)+beta*(sum(factors)/len(factors) if factors else 0.)
        if factor: terms.append(math.log(1-alpha)+math.log(component['weight']/total_weight)+gaussian+math.log(factor))
    if not terms: return None
    maximum = max(terms)
    return maximum+math.log(math.fsum(math.exp(v-maximum) for v in terms))


def summarize(root):
    root = Path(root).absolute()
    plan = read(root/'execution-plan.json'); verify(root, plan)
    plan_hash = sha(root/'execution-plan.json')
    require(read(root/'execution/claim.json')['plan_sha256'] == plan_hash, 'Execution plan changed after claim')
    done = read(root/'execution/summary.json')
    require(done['complete'] is True and done['passed'] is True and len(done['completed']) == 18, 'Incomplete diagnostic')
    require(done['plan_sha256'] == plan_hash and done['fresh_draws'] == 1024 and
            done['saved_development_probes'] == 40 and done['new_physical_clouds'] == 0, 'Execution scope/plan changed')
    require(not (root/'execution/failure.json').exists(), 'Execution has a failure receipt')
    for job, r in zip(plan['jobs'], done['completed']):
        require(r['id'] == job['id'] and r['terminal'] == job['terminal'], 'Completed job order/identity changed')
        require(sha(r['terminal']) == r['sha256'], 'Changed terminal result')
        validate_terminal(job, read(r['terminal']), plan)
    audit_jobs = {j['query_id']: j for j in plan['jobs'] if j['phase'] == 'audit'}
    allocation = read(root/'allocation.json'); populations = defaultdict(list)
    saved = []; audited_bindings = {}
    saved_groups = {v['id']: v['group'] for v in read(root/'selected-probes.json')}
    def mass_summary(values):
        return {channel: {field: dict(count=len(xs), minimum=min(xs), median=statistics.median(xs), maximum=max(xs))
                          for field, xs in fields.items() if xs}
                for channel, fields in values.items()}
    def record_masses(target, channel, record):
        for field in ('class_mass', 'hard_free_mass', 'effective_mass'):
            value = record[field]
            require(math.isfinite(value) and value >= 0., 'Invalid audited conditional mass')
            target[str(channel)][field].append(value)
    for job in plan['jobs']:
        if job['phase'] != 'query': continue
        result = read(job['terminal']); directory = Path(job['terminal']).parent
        audit = read(audit_jobs[job['id']]['terminal'])
        require(audit['complete'] is True and audit['passed'] is True and audit['synthetic'] is False,
                'Missing independent protein audit')
        require(audit['samples'] == job['samples'] and audit['probes'] == job['probes'], 'Audit attempt allocation differs')
        for path, digest in audit['input_sha256'].items():
            require(sha(path) == digest, 'Changed independently audited input: '+path)
            require(path not in audited_bindings or audited_bindings[path] == digest, 'Conflicting independent input bindings')
            audited_bindings[path] = digest
        for path in (directory/'summary.json', directory/'provenance/importance-guide.json',
                     directory/'samples.jsonl', directory/'probes.jsonl', directory/'attempts.jsonl'):
            require(audit['input_sha256'].get(str(path.resolve())) == sha(path), 'Missing independent file binding: '+str(path))
        require(audit['executable_sha256'] == result['manifest']['executable_sha256'], 'Audit binary differs')
        sample_path = directory/('samples.jsonl' if job['samples'] else 'probes.jsonl')
        guide_path = directory/'provenance/importance-guide.json'
        require(sha(guide_path) == result['manifest']['guide_sha256'], 'Changed audited guide')
        guide = read(guide_path)
        require(sha(sample_path) == result['samples_sha256' if job['samples'] else 'probes_sha256'], 'Changed rows')
        counts, orthants, selected, fallback, scored = Counter(), Counter(), Counter(), Counter(), Counter()
        cpu = Counter(); n = 0
        selected_masses = defaultdict(lambda: defaultdict(list))
        scored_masses = defaultdict(lambda: defaultdict(list))
        for row in rows(sample_path):
            n += 1; kind = label(row); counts[kind] += 1
            orthant = sum((1 << i) for i, u in enumerate(row['latent']) if u >= 0)
            orthants[f'{kind}:{orthant}'] += 1
            for key in ('draw_cpu_seconds', 'density_cpu_seconds', 'observer_cpu_seconds'): cpu[key] += row.get(key, 0.)
            draw = row['draw']
            if draw:
                if draw['conditional']:
                    key = str(draw['channel']); selected[key] += 1
                    fallback[key+':'+draw['fallback_target']] += 1
                    record_masses(selected_masses, key, draw)
                    if draw['class_mass'] <= allocation['minimum_conditional_mass']:
                        intervals = draw['geometry']['channels'][draw['channel']]['intervals']
                        fallback[key+(':empty_class' if not intervals else ':class_mass_below_floor')] += 1
                else: selected['uniform_or_unconditioned'] += 1
            for axis in row['density_details'].get('axes', []):
                for component in axis['components']:
                    for channel in component['channels']:
                        scored[f'{channel["channel"]}:{channel["fallback_target"]}'] += 1
                        record_masses(scored_masses, channel['channel'], channel)
            if not job['samples']:
                old_logq = hard_free_log_density(row, guide, result['manifest']['latent_radius'])
                require(row['id'] in saved_groups, 'Unknown saved development identity')
                saved.append(dict(id=row['id'], group=saved_groups[row['id']], label=kind, orthant=orthant,
                    log_proposal_density=row['log_proposal_density'],
                    reconstructed_hard_free_log_density=old_logq,
                    class_minus_hard_free_log_density=(row['log_proposal_density']-old_logq
                        if old_logq is not None and row['log_proposal_density'] is not None else None),
                    log_physical_jacobian=row['log_physical_jacobian']))
        require(n == job['samples']+job['probes'], 'Attempt count differs')
        report = dict(id=job['id'], attempted=n, counts=dict(counts), class_orthants=dict(orthants),
                      selected_channels=dict(selected), selected_fallbacks=dict(fallback),
                      scored_component_axis_channels=dict(scored),
                      selected_conditional_masses=mass_summary(selected_masses),
                      scored_conditional_masses=mass_summary(scored_masses), cpu_seconds=dict(cpu),
                      independent_audit_cpu_seconds=audit['analysis_cpu_seconds'])
        if job['samples']: populations[job['arm']].append(report)
    arms = {}
    for arm, items in populations.items():
        require(len(items) == 4, 'Missing independent proposal population')
        categories = sorted({k for p in items for k in p['counts']})
        fractions = {}
        for kind in categories:
            values = [p['counts'].get(kind, 0)/p['attempted'] for p in items]
            fractions[kind] = dict(mean=statistics.mean(values), population_SE=statistics.stdev(values)/2,
                                   per_population=values)
        arms[arm] = dict(populations=items, unconditional_proposal_fractions=fractions)
    verify(root, plan)
    for path, digest in audited_bindings.items(): require(sha(path) == digest, 'Audited input changed during summary: '+path)
    for item in done['completed']: require(sha(item['terminal']) == item['sha256'], 'Terminal changed during summary')
    return dict(schema='native-class-line-proposal-diagnostic-summary-v1', complete=True,
        execution_plan_sha256=plan_hash,
        arms=arms, saved_development_probes=saved, new_Poisson_clouds=0,
        execution_summary_sha256=sha(root/'execution/summary.json'),
        scope='Proposal fractions and arithmetic coverage only. Saved queries are development data. These are not equilibrium occupancies, physical region masses or mixing efficiencies.',
        CPU_limit='Both arms use complete class-geometry instrumentation; H-only timings are not an optimized legacy-hard-free baseline and cannot establish a speedup.',
        full_vessel_gate_open=False, assembly_gate_open=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); require(not args.out.exists(), 'Output already exists')
    value = summarize(args.root)
    args.out.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
