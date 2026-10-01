#!/usr/bin/env python3
"""Retained-index arithmetic after the entire regional pilot has completed.

No sampling, geometry, classifiers, interval reconstruction or proposal change.
The full-mixture results remain authoritative; this is a paired retrospective
comparison using the same saved draws, cloud realizations and region labels.
"""
from __future__ import annotations
import os
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '1'
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import shutil
import numpy as np
from scipy.special import logsumexp
from scipy.stats import t as student_t
from analyze_contact_bank_reference_partition import HashLedger
from analyze_mobile_native_pocket import require, read, sha, inside, local_sources
from analyze_mobile_wall_contacts import summarize_regions, PRIMARY
from analyze_mobile_competing_reference import paired_moments, check_estimate
from hard_free_line_physical_reference import read_weights, close

PARTS = ('old_R5_intersection_native', 'remaining_R4_native')
REGIONS = ('total',) + PRIMARY + PARTS
ARMS = ('baseline', 'conditioned')
SCOPE = ('Exploratory retained-index estimates on the same completed regional draws and '
    'independent cloud pairs. All original attempts and zeros remain in every denominator. '
    'Full and indexed estimates are paired, not independent evidence or a replacement for '
    'the frozen full-mixture analysis. No full-vessel support claim, equilibrium mixing '
    'claim or assembly conclusion. Importance ESS ratios and observed second moments '
    'are noisy diagnostics, not guaranteed variance or runtime improvements.')


def index_log_factor(row, guide):
    """log(W_index/W_full), with the actual primitive label retained.

    The old-responsibility target gives W_index=F*Zeff/Qold. Conditioning
    success uses its selected Normal interval mass; every other label has
    Zeff=1. This requires physical support contained in every conditioned
    line's feasible set (regional R4 only, never the unchanged full vessel).
    """
    require(guide['schema'] == 'defensive-hard-free-line-guide-v1', 'Unsupported guide')
    beta = guide['conditional_probability']; alpha = guide['defensive_uniform_shell_probability']
    require(0 <= beta <= 1 and 0 < alpha <= 1, 'Invalid guide probabilities')
    detail, trace = row['hard_free_line_density'], row['hard_free_line_draw']
    require(type(trace['conditional']) is bool, 'Missing primitive draw label')
    full = row['log_proposal_density']; require(math.isfinite(full), 'Nonfinite complete density')
    uniform = row['proposal_branch'] == 'uniform-shell'
    require(uniform or row['proposal_branch'] == 'hard-free-line', 'Changed proposal lineage')
    if uniform:
        require(row['proposal_component'] is None and not trace['conditional'], 'Uniform draw conditioned')
    else:
        k = row['proposal_component']
        require(type(k) is int and 0 <= k < len(guide['gaussian_components']) and alpha < 1,
                'Invalid Gaussian primitive label')
    if beta == 0 or alpha == 1:
        require(detail.get('conditioning_disabled') is True and not trace['conditional'],
                'Disabled conditioner generated a conditional draw')
        return 0., 'uniform' if uniform else 'unconditioned_gaussian'
    require(detail.get('conditioning_disabled') is not True, 'Missing active mixture density')
    old = detail['baseline_log_density']; require(math.isfinite(old), 'Missing finite Qold')
    logz = 0.
    if trace['conditional']:
        require(not uniform and trace['component'] == row['proposal_component']
                and trace['axis'] in guide['raw_translation_axes'], 'Conditional label mismatch')
        mass = trace['conditional_mass']; floor = guide['minimum_conditional_mass']
        # Match the production sum-of-interval-CDF tolerance. Preserve its
        # actual saved mass rather than silently clamp or rescore it.
        require(math.isfinite(mass) and 0 <= mass <= 1+1e-12 and type(trace['fallback']) is bool
                and trace['fallback'] == (mass <= floor), 'Conditional mass/fallback mismatch')
        label = 'fallback_gaussian' if trace['fallback'] else 'conditioned_gaussian'
        if not trace['fallback']:logz = math.log(mass)
    else:
        require(uniform or beta < 1, 'Unconditioned Gaussian with beta=1')
        label = 'uniform' if uniform else 'unconditioned_gaussian'
    return full-old+logz, label


def transform(rows, arrays, guide):
    n = len(rows)
    require(np.array_equal(arrays['draw'], np.arange(n)) and np.all(arrays['source_n'] == n),
            'Lost attempted-draw identities')
    require([r['draw'] for r in rows] == list(range(n)), 'Raw draws missing or repeated')
    delta = np.empty(n); labels = Counter(); conditional = 0
    for i, row in enumerate(rows):
        close(row['log_proposal_density'], arrays['log_q'][i], 'Archived complete density differs')
        close(row['log_physical_jacobian'], arrays['log_physical_jacobian'][i], 'Archived Jacobian differs')
        require(bool(np.isfinite(arrays['z'][i])) == all(row[k] for k in
            ('shell_valid','capture_valid','hard_valid','region_valid')), 'Physical support changed')
        delta[i], label = index_log_factor(row, guide); labels[label] += 1
        conditional += int(row['hard_free_line_draw']['conditional'])
    require(np.isfinite(delta).all(), 'Unrepresentable index factor')
    indexed = dict(arrays)
    for field in ('z', 'h'):indexed[field] = arrays[field]+delta
    indexed['pairs'] = arrays['pairs']+delta[:, None]
    for field in ('z','h','pairs'):
        require(np.array_equal(np.isfinite(arrays[field]), np.isfinite(indexed[field])), 'Index changed zero support')
    if guide['conditional_probability'] == 0 or guide['defensive_uniform_shell_probability'] == 1:
        require(np.all(delta == 0), 'Uniform/disabled-control identity failed')
    active = guide['conditional_probability'] > 0 and guide['defensive_uniform_shell_probability'] < 1
    return indexed, delta, dict(labels=labels, selected_draw_geometry_requests=conditional,
        removable_full_density_geometry_requests=n*len(guide['raw_translation_axes']) if active else 0,
        counter_scope='Exact geometry-request counts from source and draw labels, not measured durations. '
                      'The indexed sampler would retain selected-draw geometry and cheap Qold.')


def masks(arrays):
    valid = np.isfinite(arrays['z']); native = arrays['native']; contact = arrays['contact']
    require(native.dtype == contact.dtype == arrays['support'].dtype == bool, 'Changed saved boolean labels')
    require(not np.any(valid & ~arrays['support']) and not np.any((native | contact) & ~valid),
            'Invalid/exterior row acquired a physical label')
    result = dict(total=valid, registered_native_entry=valid & native,
        contact_no_native_entry=valid & contact & ~native,
        unbound_no_native_entry=valid & ~contact & ~native)
    for part in PARTS:
        result[part] = arrays[part]
        require(result[part].dtype == bool and not np.any(result[part] & ~result[PRIMARY[0]]),
                'Saved native partition differs')
    require(not np.any(result[PARTS[0]] & result[PARTS[1]])
        and np.array_equal(result[PARTS[0]] | result[PARTS[1]], result[PRIMARY[0]]),
        'Saved native partition not exhaustive')
    return result


def paired_difference(index, full):
    """Paired linear mean and its SE, with a shared explicit log scale."""
    index, full = np.asarray(index, float), np.asarray(full, float)
    require(index.shape == full.shape and index.ndim == 1 and len(full) >= 2, 'Missing paired denominators')
    require(np.array_equal(np.isfinite(index), np.isfinite(full)), 'Paired support differs')
    if not np.isfinite(full).any():return dict(observed=False, draws=len(full), reason='Unobserved, not zero mass.')
    scale = float(max(index.max(), full.max())); a, b = np.exp(index-scale), np.exp(full-scale)
    difference = a-b; se = float(np.std(difference, ddof=1)/math.sqrt(len(full)))
    mean = float(difference.mean()); half = float(student_t.ppf(.975,len(full)-1)*se)
    return dict(observed=True, draws=len(full), log_scale=scale,
        scaled_mean_index=float(a.mean()), scaled_mean_full=float(b.mean()), scaled_difference=mean,
        scaled_difference_SE=se, scaled_difference_interval_95=[mean-half,mean+half],
        difference_over_observed_full_mean=mean/float(b.mean()),
        scope='Paired linear difference and sample SE; multiply scaled values by exp(log_scale). '
              'Student-t interval is approximate; the difference is unbiased, its relative form is a ratio diagnostic.')


def load_completed(root, ledger):
    root = Path(root).resolve(); state = read(ledger.bind(root/'status.json'))
    require(state['complete'] is True and state['phase'] == 'complete',
            'Wait for all physical draws, independent audits and classification to complete')
    require(len(state['jobs']) == len(state['audits']) == 8 and all(j['status'] == 'complete'
        and j['returncode'] == 0 for j in state['jobs']+state['audits']), 'Incomplete population/audit')
    protocol = read(ledger.bind(root/'protocol.json', state['protocol_sha256']))
    data = read(ledger.bind(root/'comparison/analysis.json', state['comparison_sha256']))
    require(protocol['schema'] == 'hard-free-line-physical-pilot-v1' and data['complete']
        and data['schema'] == 'hard-free-line-physical-comparison-v1'
        and data['protocol_sha256'] == state['protocol_sha256'], 'Not the completed regional pilot')
    require(data['total_unconditional_draws'] == protocol['total_unconditional_draws']
        and set(data['arms']) == set(ARMS) and len(protocol['jobs']) == 8, 'Changed campaign allocation')
    require(not data['diagnostics']['full_vessel_gate_open'] and not data['diagnostics']['assembly_gate_open'],
            'Regional index support proof does not extend to the full vessel')
    return protocol, data


def population(root, record, job, arm, ledger):
    require(all(record[k] == job[k] for k in ('id','arm','samples','seed')), 'Population identity differs')
    directory = Path(job['directory']).resolve(); require(directory.is_relative_to(root), 'Population escapes execution root')
    audit_path = root/'audits'/arm['id']/(job['id']+'.json')
    audit = read(ledger.bind(audit_path,record['independent_audit_sha256']))
    require(audit['schema'] == 'independent-hard-free-line-physical-audit-v1' and audit['complete']
        and audit['geometry_mode'] == 'full' and audit['samples'] == job['samples'], 'Missing full independent audit')
    def audited(name):
        p = directory/name
        return ledger.bind(p,audit['input_sha256'][str(p)])
    summary = read(audited('summary.json')); manifest = read(audited('manifest.json'))
    require(summary['complete'] and manifest['schema'] == 'importance-latent-region-normalizer-v6'
        and manifest['guide_schema'] == 'defensive-hard-free-line-guide-v1'
        and manifest['seed'] == record['seed'] and manifest['samples'] == record['samples'],
        'Unexpected regional population')
    require(not (directory/'failure.json').exists(), 'Population contains failure record')
    rows_path = audited('samples.jsonl')
    require(sha(rows_path) == record['samples_sha256'] == summary['samples_sha256'], 'Raw population changed')
    journal = audited('attempts.jsonl')
    require(sha(journal) == summary['attempts_sha256'], 'Attempt journal changed')
    rows = [json.loads(s) for s in rows_path.read_text().splitlines()]
    guide = read(audited('provenance/importance-guide.json')); region = read(audited('provenance/region.json'))
    require(region['mahalanobis_radius'] == 4 and region.get('minimum_mahalanobis_radius',0) == 0,
            'Retained-index support proof is limited to the original R4 target')
    require(guide['conditional_probability'] == arm['beta']
        and guide['defensive_uniform_shell_probability'] == arm['alpha'], 'Guide differs from frozen arm')
    archive = ledger.bind(inside(root/'comparison',record['records']),record['records_sha256'])
    ledger.bind(inside(root/'comparison',record['labels']),record['labels_sha256'])
    with np.load(archive,allow_pickle=False) as stored: arrays = {k:stored[k] for k in stored.files}
    require(len(rows) == len(arrays['z']) == job['samples'], 'Lost attempted denominator')
    original = read_weights(rows,job['samples'],region,arm)
    for field in ('z','h','pairs','branch','component','support'):
        require(np.array_equal(original[field],arrays[field]), 'Raw/archive mismatch: '+field)
    require(np.array_equal(original['latents'],arrays['u']), 'Saved chart coordinates changed')
    check_estimate(paired_moments(arrays['z'],arrays['h']),audit['estimate'],audit['hard_region'])
    cpu = summary['sampler_cpu_seconds']
    require(math.isfinite(cpu) and cpu > 0 and cpu == record['sampler_cpu_seconds'], 'Changed sampler CPU')
    indexed, delta, counts = transform(rows,arrays,guide); partition = masks(arrays)
    def pack(value):return dict(id=job['id'],seed=job['seed'],masks=partition,**{k:value[k]for k in ('z','h','pairs')})
    return pack(arrays),pack(indexed),dict(id=job['id'],seed=job['seed'],samples=job['samples'],
        sampler_cpu_seconds=cpu,index_factor_range=[float(delta.min()),float(delta.max())],**counts)


def paired_regions(full, indexed, estimates):
    comparisons = {}
    for region in REGIONS:
        a = np.concatenate([np.where(p['masks'][region],p['z'],-np.inf) for p in indexed])
        b = np.concatenate([np.where(p['masks'][region],p['z'],-np.inf) for p in full])
        pop = [[-math.inf if p['log_Qz'] is None else p['log_Qz'] for p in
            estimates[mode][region]['populations']] for mode in ('index','full')]
        observed = np.isfinite(a).any()
        noisy = float(logsumexp(2*a)-logsumexp(2*b)) if observed else None
        cross = []
        for populations in (indexed,full):
            cross.append(np.concatenate([np.where(p['masks'][region],p['pairs'].sum(axis=1),-np.inf) for p in populations]))
        comparisons[region] = dict(row=paired_difference(a,b), population=paired_difference(*pop),
            observed_log_noisy_second_moment_index_over_full=noisy,
            observed_log_two_cloud_cross_product_index_over_full=float(logsumexp(cross[0])-logsumexp(cross[1])) if observed else None,
            scope='Full weights Rao-Blackwellize the retained index in exact arithmetic. Their theoretical '
                'second moment cannot increase, although finite-sample ratios can reverse. The two-cloud '
                'product estimates the physical squared-integrand contribution without cloud-variance bias; '
                'ratios of observed sums remain biased diagnostics. No new cloud is drawn.')
    return comparisons


def run(execution,out):
    root,out = Path(execution).resolve(),Path(out).resolve(); require(not out.exists(),'Fresh retrospective destination required')
    ledger = HashLedger(); protocol,data = load_completed(root,ledger); results = {}
    for name in ARMS:
        arm = next(a for a in protocol['arms'] if a['id'] == name)
        records = data['arms'][name]['populations']
        require(len(records) == 4 and len({r['seed'] for r in records}) == 4, 'Need four independent populations')
        full,indexed,counts = [],[],[]
        for record in records:
            jobs = [j for j in protocol['jobs'] if j['arm'] == name and j['id'] == record['id']]
            require(len(jobs) == 1, 'Duplicate/missing physical population')
            f,i,c = population(root,record,jobs[0],arm,ledger);full.append(f);indexed.append(i);counts.append(c)
        estimates,ratios = {},{}
        for mode,pops in (('full',full),('index',indexed)):
            estimates[mode],_,ratios[mode] = summarize_regions(pops)
            cpu_by_id = {c['id']:c['sampler_cpu_seconds'] for c in counts}
            for estimate in estimates[mode].values():
                for p in estimate['populations']:
                    p['sampler_cpu_seconds'] = cpu_by_id[p['id']]
                    p['observed_importance_ESS_per_actual_full_sampler_CPU'] = (
                        p['Qz_ESS']/cpu_by_id[p['id']] if p['log_Qz'] is not None else None)
        for region in REGIONS:
            for kind in ('log_Qz','log_Q0'):
                actual = estimates['full'][region]['row_uncertainty'][kind]
                expected = data['arms'][name]['estimates'][region]['row_uncertainty'][kind]
                require(actual == expected or actual is not None and expected is not None
                    and abs(actual-expected) < 2e-10, 'Full reference estimate changed')
        cpu = sum(c['sampler_cpu_seconds'] for c in counts)
        results[name] = dict(populations=counts,estimates=estimates,primary_ratios=ratios,
            paired_full_vs_index=paired_regions(full,indexed,estimates),sampler_cpu_seconds=cpu,
            observed_importance_ESS_per_actual_full_sampler_CPU={mode:{r:e['row_uncertainty'].get('Qz_ESS',0)/cpu
                if e['row_uncertainty']['log_Qz'] is not None else None for r,e in regions.items()}for mode,regions in estimates.items()},
            runtime=dict(index_only_sampler_cpu_seconds=None,stage_timing_available=False,
                full_density_geometry_requests=sum(c['removable_full_density_geometry_requests'] for c in counts),
                selected_draw_geometry_requests=sum(c['selected_draw_geometry_requests'] for c in counts),
                scope='Production saved total sampler CPU only: no per-stage density/cloud timers. '
                      'Both retrospective ESS/CPU values use the same measured full-scorer cost. '
                      'Geometry-request counts do not determine a CPU saving or index-sampler speedup.'))
    require(sum(c['samples']for arm in results.values()for c in arm['populations']) == protocol['total_unconditional_draws'],
            'Aggregate attempted denominator differs')
    sources = local_sources(__file__)
    for path in sources.values():ledger.bind(path)
    result = dict(schema='hard-free-line-physical-index-retrospective-v1',complete=True,arms=results,
        total_unconditional_draws=protocol['total_unconditional_draws'],scope=SCOPE,input_sha256=ledger.files,
        new_pose_draws=0,new_Poisson_clouds=0,new_geometry_queries=0,new_native_classifier_calls=0,
        actual_index_sampler_run=False,frozen_physical_weights_replaced=False,full_vessel_resolved=False,assembly_resolved=False)
    ledger.recheck();out.mkdir();(out/'source').mkdir()
    for name,path in sources.items():shutil.copy2(path,out/'source'/name)
    (out/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    lines=['# Retained-index regional retrospective','',SCOPE,'',
        'The retained draw label gives `W_index = W_full × exp(log Qnew − log Qold) × Zeff`. '
        '`Zeff` is the selected conditional mass after successful conditioning and 1 otherwise. '
        'The same factor applies to both saved cloud weights. The beta=0 control is exactly unchanged.','',
        '| Arm | Region | Full ESS | Index ESS | Paired population relative difference |',
        '|---|---|---:|---:|---:|']
    for arm,record in results.items():
        for region in REGIONS:
            e = record['estimates']; d = record['paired_full_vs_index'][region]['population']
            display = lambda v:'unobserved' if v is None else f'{v:.6g}'
            lines.append(f"| {arm} | {region} | {display(e['full'][region]['row_uncertainty'].get('Qz_ESS'))} | {display(e['index'][region]['row_uncertainty'].get('Qz_ESS'))} | {display(d.get('difference_over_observed_full_mean'))} |")
    lines+=['','Population and paired-row uncertainty, all original counts, two-cloud diagnostics and individual population estimates are in `analysis.json`. '
        'This analysis does not change the saved physical campaign or its convergence gates. '
        'The same measured full-sampler CPU is used for both weight summaries; index-only speedup remains unmeasured.']
    (out/'report.md').write_text('\n'.join(lines)+'\n');ledger.recheck()
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__);p.add_argument('--execution',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    print(json.dumps(dict(complete=run(a.execution,a.out)['complete'],out=str(a.out))))
