#!/usr/bin/env python3
"""Audit complete shoulder proposals; calibrate only their strict inner masks."""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[key] = '1'
import argparse
import hashlib
import heapq
import math
from pathlib import Path
import subprocess
import sys

import numpy as np

from analyze_expanded_contact_atlas import nested_prefix_comparison, validate_prefix_counts
from analyze_far_contact_atlas import command_options
from analyze_far_peak_reference import add_population_errors
from analyze_peak_neighborhood import geometry_model_audit
from analyze_shoulder_peak_reference import WINDOW as INNER_WINDOW, SHOULDER_WINDOW, compare_physical, geometry_batch
from audit_shoulder_mis_independently import near
from compare_far_local_history import independent_comparison, validate_runtime_metric
from compare_intermediate_local_reference import read_batches
from prepare_cayley_rms_cover import read, write, sha, require
from prepare_native_confirmation_atlas import AtomUnionAudit
from run_shoulder_mis_campaign import local_dependencies
from shoulder_atlas_moments import KEYS, ShoulderAtlasMoments
from shoulder_union_moments import CENTERS

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREPARATION = ROOT/'runs/ab-shoulder-contact-atlas-preparation-20260921'
CALIBRATION_SCOPE = ('Identical strict-inner finite masks. The reference data informed centers, widths or fit weights; '
    'fresh streams are independent conditional on that preparation, but these observed-error contrasts are calibration, '
    'not held-out significance tests or unseen-tail bounds. No finite reference is substituted for the full normalizer.')
WIDTH_SCOPE = ('Distinct random streams for two frozen complete-proposal widths targeting the same physical integral. '
    'The difference is left minus right. Observed row/population errors do not bound unseen tails or establish convergence.')


def labeled_comparison(left, right, left_label, right_label, scope):
    """Reuse the arithmetic without reversing chronological source labels."""
    legacy = independent_comparison(left, right)
    return dict(left_label=left_label, right_label=right_label, left=left, right=right,
        left_minus_right_logQ=legacy['historical_minus_fresh_logQ'],
        left_to_right_ratio=legacy['historical_to_fresh_ratio'],
        linear_left_minus_right_in_combined_row_SE=legacy['linear_difference_in_combined_row_SE'],
        linear_left_minus_right_in_combined_population_SE=legacy['linear_difference_in_combined_population_SE'],
        scope=scope)


class PrefixMoments:
    def __init__(self, samples, prefixes):
        validate_prefix_counts(samples, prefixes)
        self.samples = samples; self.count = 0
        self.prefixes = {n: ShoulderAtlasMoments() for n in prefixes}

    def add(self, draw, *args):
        require(draw == self.count and draw < self.samples, 'Missing, duplicate or extra original row')
        for n, moments in self.prefixes.items():
            if draw < n: moments.add(*args)
        self.count += 1

    def complete(self): require(self.count == self.samples, 'Incomplete original population')


def finish(aggregate, populations):
    result = add_population_errors(aggregate.report(), populations, KEYS)
    result['populations'] = populations
    for kind in ('physical', 'hard'):
        full = result[kind]['full']['logQ']; inner = result[kind]['inner']['logQ']
        for key, row in result[kind].items():
            finite = [v for v in row['population_logQ_values'] if v is not None]
            offset = max(finite) if finite else 0.
            mass = np.array([0. if v is None else math.exp(v-offset) for v in row['population_logQ_values']])
            fractions = mass/mass.sum() if mass.sum() else None
            row['population_mass_fractions'] = None if fractions is None else fractions.tolist()
            row['maximum_population_fraction'] = None if fractions is None else float(fractions.max())
            row['population_weight_ESS'] = None if fractions is None else float(1/np.sum(fractions**2))
            row['fraction_of_observed_full_shoulder'] = math.exp(row['logQ']-full) if row['logQ'] is not None else None
            row['fraction_of_observed_inner'] = (math.exp(row['logQ']-inner)
                if key.startswith('inner') and row['logQ'] is not None else None)
    return result


def full_weights(row):
    """Keep the archived full mixture density, including outer shoulder rows."""
    if row.get('zero') is not None:
        require(row.get('log_importance_weight') is None, 'Rejected row has nonzero weight')
        return None, None, None
    require(1 < row['q'] < 2, 'Positive row outside original shoulder')
    physical, hard = row['log_importance_weight'], row['log_hard_weight']
    near(hard, -row['log_proposal_density'])
    pair = [hard+v for v in row['cloud_log_weights']]
    require(len(pair) == 2 and all(math.isfinite(v) for v in pair), 'Missing original cloud pair')
    near(float(np.logaddexp(*pair))-math.log(2), physical)
    return physical, hard, pair


def validate_plan(protocol):
    require(protocol['q_window'] == SHOULDER_WINDOW and protocol['inner_q_window'] == INNER_WINDOW, 'Original shoulder windows changed')
    analysis = protocol['analysis']
    require(analysis['priority_order'] == list(CENTERS) and analysis['radii'] == [.25, .5], 'Frozen inner region allocation changed')
    require([c['name'] for c in analysis['centers']] == list(CENTERS), 'Frozen chart order changed')
    require([a['name'] for a in protocol['arms']] == ['narrow', 'broad'], 'Frozen width arms changed')
    require(protocol['lambda_ratio'] == 64 and protocol['cloud_replicates'] == 2, 'Frozen bath estimator changed')
    require(protocol['hybrid'] == dict(model_weight=.99, uniform_probability=.05, cover_scales=[1.])
            and protocol['proposal_anchor_index'] == 0, 'Frozen complete proposal law differs')
    for arm in protocol['arms']: validate_prefix_counts(arm['samples_per_population'], protocol['prefix_counts'])
    seeds = [seed for arm in protocol['arms'] for seed in arm['seeds']]
    require(len(seeds) == len(set(seeds)), 'Width arms reuse production streams')


def validate_arm(prep, protocol, arm, commands):
    root = Path(arm['output']); master = read(root/'manifest.json'); cfg = read(root/'provenance/config.json')
    compare_physical(cfg, protocol['physical'])
    require(master['q_window'] == protocol['q_window'], 'Original physical window differs')
    require(master['total_unconditional_draws'] == arm['populations']*arm['samples_per_population'] and len(master['jobs']) == arm['populations'], 'Frozen unconditional budget differs')
    require([j['seed'] for j in master['jobs']] == arm['seeds'], 'Frozen production seeds differ')
    for name, digest in master['archive_sha256'].items(): require(sha(root/'provenance'/name) == digest, 'Physical archive changed')
    model = Path(arm['model_path']); model_sha = arm['model_sha256']
    require(sha(model) == sha(root/'provenance/guide-model.json') == model_sha, 'Original proposal model bytes differ')
    require(len(read(model)['weights']) == arm['components'], 'Original proposal component count differs')
    require(master['binary_sha256'] == protocol['executable_sha256'] and master['shape_sha256'] == protocol['shape_sha256'], 'Physical executable or shape differs')
    require(master['archive_sha256']['analyze_native_region_reference.py'] == protocol['archived_sha256']['analyze_native_region_reference.py'], 'Original density auditor source differs')
    command = commands[arm['name']]; runner = Path(command[1]); opts = command_options(command)
    require(runner == prep/'provenance/run_native_region_reference.py' and
            sha(runner) == protocol['archived_sha256']['run_native_region_reference.py'], 'Frozen launcher source differs')
    require(sha(runner) == master['runner_sha256'] and master['max_workers'] == arm['workers'], 'Actual runner or worker allocation differs')
    expected = {'--root': str(root), '--config': str(prep/'config.json'), '--binary': protocol['executable'],
        '--expected-binary-sha256': protocol['executable_sha256'], '--replicates': str(arm['populations']),
        '--samples': str(arm['samples_per_population']), '--workers': str(arm['workers']), '--seed-base': str(arm['seeds'][0]),
        '--q-min': '1', '--q-max': '2', '--q-lower-open': True, '--q-upper-open': True, '--cover-scales': '1',
        '--model': str(model), '--model-weight': str(protocol['hybrid']['model_weight']),
        '--model-uniform-probability': str(protocol['hybrid']['uniform_probability']), '--model-anchor-index': str(protocol['proposal_anchor_index'])}
    require(opts == expected, 'Frozen launch command differs from protocol')
    require(master['model_source_path'] == str(model) and master['source_config_path'] == str(prep/'config.json'), 'Actual input paths differ from frozen command')
    status = read(root/'runner-status.json')
    require(status['complete'] and len(status['jobs']) == arm['populations'], 'Physical runner not terminal')
    for job in master['jobs']:
        path = Path(job['output']); run = read(path/'manifest.json'); summary = read(path/'summary.json')
        child = status['jobs'][path.name]
        require(child['status'] == 'complete' and child['exit_code'] == 0 and child['seed'] == job['seed'], 'Physical child not terminal success')
        require(summary['complete'] and run['samples'] == summary['samples'] == arm['samples_per_population'], 'Original sample count differs')
        require(run['seed'] == job['seed'] and run['q_window'] == protocol['q_window'] and run['schema'] == 4, 'Runtime stream/window/schema differs')
        require(run['lambda_ratio'] == protocol['lambda_ratio'] and run['cloud_replicates'] == protocol['cloud_replicates'], 'Runtime cloud law differs')
        require(run['activity'] == cfg['reservoir_density'] and run['lambda'] == run['activity']*run['lambda_ratio'] and run['depletant_radius'] == cfg['depletant_radius'], 'Runtime bath differs')
        require(run['config_sha256'] == master['config_sha256'] == sha(root/'provenance/config.json') and run['shape_sha256'] == master['shape_sha256'], 'Runtime config/shape provenance differs')
        require(run['executable_sha256'] == protocol['executable_sha256'], 'Runtime physical executable differs')
        guide = run['guide']; hybrid = protocol['hybrid']
        require(guide['model_sha256'] == model_sha and guide['weight'] == hybrid['model_weight'] and guide['uniform_probability'] == hybrid['uniform_probability'], 'Runtime full hybrid denominator differs')
        require(guide['anchor_index'] == protocol['proposal_anchor_index'] == 0 and guide['anchor_pose'] == cfg['fixed_poses'][0], 'Runtime A-relative guide frame differs')
        require(run['cover_mixture']['scales'] == hybrid['cover_scales'], 'Complete cover mixture differs')
        validate_runtime_metric(run['metric'], cfg['metadata'])
    return root, master, cfg


def ensure_density_audits(requests, out, allow_run, workers):
    """Drain all launched children, including after a later launch failure."""
    require(type(workers) is int and workers > 0, 'Positive audit worker count required')
    for arm, root, *_ in requests:
        require((root/'assessment-streaming.json').exists() or allow_run, 'Original audit missing; await root authorization')
    state = dict(phase='original_density_audits', processes=[]); live = []; state_path = out/'runner-state.json'
    try:
        for arm, root, *_ in requests:
            target = root/'assessment-streaming.json'
            if target.exists():
                state['processes'].append(dict(arm=arm['name'], reused=True, path=str(target), sha256=sha(target))); continue
            command = [sys.executable, str(root/'provenance/analyze_native_region_reference.py'), str(root), '--workers', str(workers)]
            logpath = out/f"density-audit-{arm['name']}.log"; log = logpath.open('x')
            try: process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            except BaseException: log.close(); raise
            record = dict(arm=arm['name'], pid=process.pid, command=command, log=str(logpath), output=str(target), terminal=False)
            live.append((process, log, record)); state['processes'].append(record); write(state_path, state)
            print(f"{arm['name']}: archived full-density audit PID{process.pid}", flush=True)
    finally:
        for process, log, record in live:
            record.update(terminal=True, exit_code=process.wait()); log.close(); write(state_path, state)
    failures = [p for p in state['processes'] if p.get('exit_code', 0)]
    state['phase'] = 'density_audits_failed' if failures else 'density_audits_complete'; write(state_path, state)
    require(not failures, 'Density audit failed; diagnose before any replay')
    return state


def remask_arm(arm, root, master, cfg, protocol, models, geometries):
    audit_path = root/'assessment-streaming.json'; original = read(audit_path)
    require(original['complete'] and original['all_rows_and_hashes_validated'] and original['independent_complete_cover_validated'], 'Original full-density audit incomplete')
    require(original['q_window'] == SHOULDER_WINDOW and original['target_key'] == 'region', 'Wrong original audited target')
    require(original['samples'] == original['independent_q_and_density_rows'] == master['total_unconditional_draws'], 'Audited original N differs')
    require(original['analysis_script_sha256'] == master['archive_sha256']['analyze_native_region_reference.py'], 'Original density audit source differs')
    counts = protocol['prefix_counts']; aggregates = {n: ShoulderAtlasMoments() for n in counts}; populations = {n: [] for n in counts}
    sources = {str(p): sha(p) for p in (root/'manifest.json', root/'runner-status.json', audit_path)}
    top = []; errors = {name: {} for name in CENTERS}
    for pi, job in enumerate(master['jobs']):
        path = Path(job['output']); sample = path/'samples.jsonl'; summary = read(path/'summary.json'); run = read(path/'manifest.json')
        prior = next(p for p in original['populations'] if p['replicate'] == path.name)
        require(sha(path/'summary.json') == prior['summary_sha256'] and prior['samples'] == arm['samples_per_population'], 'Original audited summary/count differs')
        require(run['guide'] == original['guide'] and run['cover_mixture'] == original['cover_mixture'], 'Original full density changes between populations')
        local = PrefixMoments(arm['samples_per_population'], counts); digest = hashlib.sha256()
        for lines, rows in read_batches(sample):
            for line in lines: digest.update(line)
            radii = []
            for name in CENTERS:
                r, _, current = geometry_batch(rows, geometries[name], models[name], cfg); radii.append(r)
                for key, value in current.items(): errors[name][key] = max(errors[name].get(key, 0.), value)
            for i, row in enumerate(rows):
                weights = full_weights(row); triple = tuple(float(r[i]) for r in radii)
                local.add(row['draw'], row.get('q'), triple, *weights)
                if weights[0] is not None:
                    record = dict(population=path.name, seed=job['seed'], draw=row['draw'], pose=row['pose'], q=row['q'],
                        geometric_radii_A=dict(zip(CENTERS, triple)), original_log_importance_weight=weights[0],
                        original_row=row, source_samples_path=str(sample))
                    entry = (weights[0], -pi, -row['draw'], record)
                    if len(top) < 8: heapq.heappush(top, entry)
                    elif entry[:3] > top[0][:3]: heapq.heapreplace(top, entry)
        local.complete()
        require(digest.hexdigest() == prior['sample_sha256'] == summary['samples_sha256'], 'Original samples changed after full-density audit')
        sources[str(sample)] = digest.hexdigest()
        for filename in ('manifest.json', 'summary.json'): sources[str(path/filename)] = sha(path/filename)
        for n, moments in local.prefixes.items():
            aggregates[n].merge(moments); populations[n].append(dict(id=path.name, seed=job['seed'], samples=n, **moments.report()))
    reports = {n: finish(aggregates[n], populations[n]) for n in counts}; result = reports[counts[-1]]
    for kind, key in (('physical', 'regions'), ('hard', 'hard_regions')):
        source, current = original[key]['region'], result[kind]['full']
        if current['logQ'] is None: require(source['log_normalizer'] is None, 'Full zero estimate differs')
        else: near(current['logQ'], source['log_normalizer']); near(current['row_RSE'], source['relative_SE'])
        require(current['draws'] == source['samples'] and current['nonzero'] == source['nonzero'], 'Original full denominator or mask differs')
    prefix_comparisons = [dict(earlier_samples_per_population=m, later_samples_per_population=n,
        **{kind: {k: nested_prefix_comparison(reports[m][kind][k], reports[n][kind][k]) for k in KEYS} for kind in ('physical', 'hard')})
        for i,m in enumerate(counts) for n in counts[i+1:]]
    atom = AtomUnionAudit(read(root/'provenance/shape.json'), cfg['fixed_poses']); extrema = []
    for _, _, _, row in sorted(top, key=lambda entry: entry[:3], reverse=True):
        gaps = atom.gaps(row['pose']); require(min(gaps) >= 0, 'Extreme positive row clashes in independent AB atomic check')
        row['minimum_AB_atomic_gaps_A'] = gaps; row['source_samples_sha256'] = sources[row['source_samples_path']]
        row['fraction_of_full_weight'] = math.exp(row['original_log_importance_weight']-aggregates[counts[-1]].values['physical']['full'].total)
        extrema.append(row)
    print(f"{arm['name']}: logQ={result['physical']['full']['logQ']}; rowRSE={result['physical']['full']['row_RSE']}", flush=True)
    return dict(arm=arm['name'], root=str(root), **result, original_unconditional_draws=master['total_unconditional_draws'],
        prefixes=[dict(samples_per_population=n, **reports[n]) for n in counts], prefix_comparisons=prefix_comparisons,
        top8_atomic_checks=extrema, input_sha256=sources, geometry_reconstruction_errors=errors,
        full_density_audit_path=str(audit_path), full_density_audit_sha256=sha(audit_path),
        independently_audited_rows=original['independent_q_and_density_rows'], CPU_seconds=original['cpu_seconds'],
        full_density_audit_seconds=original['analysis_wall_seconds'])


def calibrations(campaign, reference, direct):
    """Only identical strict-inner regions; no old raw rows are read here."""
    require(reference['original_q_window'] == direct['original_q_window'] == INNER_WINDOW, 'Finite reference target differs')
    rows = {}
    sources = [*reference['campaigns'], *[dict(c, owner='direct') for c in direct['campaigns']]]
    for center in CENTERS:
        for radius, suffix in ((.25, 'ball0p25'), (.5, 'ball0p5')):
            source = next(c for c in sources if c['owner'] == center and c['radius_A'] == radius)
            mask = f'inner_{center}_{suffix}'
            rows[mask] = {kind: labeled_comparison(campaign[kind][mask], source[kind]['full'],
                campaign.get('arm', 'campaign'), 'finite_reference', CALIBRATION_SCOPE) for kind in ('physical', 'hard')}
        mask = f'inner_{center}_assigned_ball0p5'
        rows[mask] = {kind: labeled_comparison(campaign[kind][mask], reference['independent_assigned_region_sums'][kind][center],
            campaign.get('arm', 'campaign'), 'finite_reference', CALIBRATION_SCOPE) for kind in ('physical', 'hard')}
    rows['inner_union'] = {kind: labeled_comparison(campaign[kind]['inner_union'], reference['independent_union_sum'][kind],
        campaign.get('arm', 'campaign'), 'finite_reference', CALIBRATION_SCOPE) for kind in ('physical', 'hard')}
    return dict(regions=rows, scope=CALIBRATION_SCOPE)


def analyze(preparation, out, run_audits=False, audit_workers=16):
    prep = Path(preparation).resolve(); out = Path(out).resolve(); require(not out.exists(), 'Use a fresh analysis output directory')
    protocol = read(prep/'protocol.json'); freeze = read(prep/'freeze.json'); cfg = read(prep/'config.json')
    require(sha(prep/'protocol.json') == freeze['protocol_sha256'] and sha(prep/'config.json') == protocol['config_sha256'] == freeze['config_sha256'], 'Frozen protocol/config changed')
    require(sha(prep/'commands.json') == protocol['commands_sha256'] == freeze['commands_sha256'], 'Frozen commands changed')
    require(freeze['archived_sha256'] == protocol['archived_sha256'], 'Preparation archive seal differs')
    for arm in protocol['arms']: require(sha(arm['model_path']) == arm['model_sha256'] == freeze['model_sha256'][arm['name']], 'Frozen model seal differs')
    require(sha(protocol['executable']) == protocol['executable_sha256'] and sha(prep/'provenance/shape.json') == protocol['shape_sha256'], 'Frozen executable/shape changed')
    for name, digest in protocol['archived_sha256'].items(): require(sha(prep/'provenance'/name) == digest, 'Preparation archive changed')
    for path, digest in protocol.get('input_sha256', {}).items(): require(sha(path) == digest, 'Frozen source input changed')
    validate_plan(protocol); compare_physical(cfg, protocol['physical']); commands = read(prep/'commands.json')
    spec = protocol['analysis']; reference_path = Path(spec['finite_reference_path']); direct_path = Path(spec['direct_reference_path'])
    require(sha(reference_path) == spec['finite_reference_sha256'] and sha(direct_path) == spec['direct_reference_sha256'], 'Finite reference analysis changed')
    reference, direct = read(reference_path), read(direct_path)
    require(reference['complete'] and direct['complete'] and reference['original_q_window'] == direct['original_q_window'] == INNER_WINDOW, 'Finite references incomplete or wrong window')
    require(reference['direct_reference']['sha256'] == sha(direct_path), 'Direct reference lineage differs')
    for source, path in ((reference, reference_path), (direct, direct_path)):
        for name, digest in source['archived_sha256'].items(): require(sha(path.parent/'provenance'/name) == digest, 'Finite reference archive changed')
        # The older direct analyzer archived its protocol, but no config copy.
        # Both archived protocol physical fields and the newer config are
        # already bound by the completed reference's immutable source hashes.
        compare_physical(read(path.parent/'provenance/protocol.json')['physical'], cfg)
    models = {}; geometries = {}; geometric_audits = {}
    for center in spec['centers']:
        name = center['name']; path = Path(center['model_path'])
        require(sha(path) == center['model_sha256'] == reference['archived_sha256'][f'model-{name}.json'], 'Frozen finite-mask chart differs')
        models[name] = read(path); require(models[name]['shape_sha256'] == protocol['shape_sha256'], 'Finite-mask shape differs')
        geometric_audits[name], geometries[name] = geometry_model_audit(models[name], dict(fixed_neighbor=cfg['fixed_poses'][0]), cfg, center['pose'])
    old_seeds = {p['seed'] for source in (reference, direct) for c in source['campaigns']+source.get('histories', []) for p in c['populations']}
    requests = []
    for arm in protocol['arms']:
        require(not old_seeds.intersection(arm['seeds']), 'Fresh and reference streams overlap')
        root, master, current_cfg = validate_arm(prep, protocol, arm, commands); requests.append((arm, root, master, current_cfg))
    out.mkdir(parents=True); archive = out/'provenance'; archive.mkdir()
    executing = local_dependencies([Path(__file__)])
    static = {'protocol.json': prep/'protocol.json', 'freeze.json': prep/'freeze.json', 'commands.json': prep/'commands.json',
        'config.json': prep/'config.json', 'finite-reference.json': reference_path, 'direct-reference.json': direct_path,
        **{f'chart-{c["name"]}.json': Path(c['model_path']) for c in spec['centers']},
        **{f'model-{a["name"]}.json': Path(a['model_path']) for a in protocol['arms']}}
    for name, path in {**executing, **static}.items(): (archive/name).write_bytes(path.read_bytes())
    execution_hashes = {str(path): sha(path) for path in executing.values()}; write(out/'execution-source-freeze.json', execution_hashes)
    sources = {str(path): sha(path) for path in [*static.values(), Path(protocol['executable'])]}
    state = ensure_density_audits(requests, out, run_audits, audit_workers)
    campaigns = [remask_arm(arm, root, master, current_cfg, protocol, models, geometries) for arm,root,master,current_cfg in requests]
    for campaign in campaigns: campaign['inner_finite_calibration'] = calibrations(campaign, reference, direct)
    width = {kind: {k: labeled_comparison(campaigns[0][kind][k], campaigns[1][kind][k],
        campaigns[0]['arm'], campaigns[1]['arm'], WIDTH_SCOPE) for k in KEYS} for kind in ('physical', 'hard')}
    for campaign in campaigns: sources.update(campaign['input_sha256'])
    for path, digest in {**sources, **execution_hashes}.items(): require(sha(path) == digest, 'Input or executing source changed during audit')
    scope = ('Complete original strict 1<q<2 shoulder with both AB neighbors, original full mixture density and all unconditional zeros. '
        'Full=inner+outer and inner=finite union+outside use the same rows and exact mask covariance; q=1.1 belongs to outer. '
        'Width arms remain separate. Prefixes share rows and retain covariance under their fixed IID law. '
        'Previous finite references calibrate only strict-inner masks; no historical remainder is stitched to fresh finite masses. '
        'Observed row/population errors, weight ESS and cloud diagnostics do not establish unseen-tail bounds, MCMC mixing or assembly.')
    result = dict(complete=True, original_q_window=SHOULDER_WINDOW, inner_q_window=INNER_WINDOW,
        center_order=list(CENTERS), prefix_counts=protocol['prefix_counts'], campaigns=campaigns, width_comparisons=width,
        geometric_model_audits=geometric_audits, finite_reference=dict(path=str(reference_path), sha256=sha(reference_path),
            independent_assigned_region_sums=reference['independent_assigned_region_sums'], independent_union_sum=reference['independent_union_sum']),
        direct_reference=dict(path=str(direct_path), sha256=sha(direct_path)), source_sha256=sources,
        executing_source_sha256=execution_hashes, archived_sha256={p.name: sha(p) for p in archive.iterdir()}, scope=scope)
    write(out/'analysis.json', result); state.update(phase='complete', analysis_sha256=sha(out/'analysis.json')); write(out/'runner-state.json', state)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preparation', type=Path, default=DEFAULT_PREPARATION); parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--run-density-audits', action='store_true'); parser.add_argument('--audit-workers', type=int, default=16)
    args = parser.parse_args(); analyze(args.preparation, args.out, args.run_density_audits, args.audit_workers)
    print(dict(complete=True, output=str(args.out/'analysis.json')))
