#!/usr/bin/env python3
"""Audit frozen complete-far atlases, local calibration and shared prefixes."""
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
import time

import numpy as np
from scipy.spatial.transform import Rotation

from analyze_expanded_contact_atlas import nested_prefix_comparison, validate_prefix_counts
from analyze_far_peak_reference import PHYSICAL, WINDOW, add_population_errors
from analyze_global_fixed_region import region_latent
from analyze_peak_neighborhood import geometry_model_audit, geometry_rows
from audit_shoulder_mis_independently import near
from compare_far_local_history import HistoryMoments, MASKS, independent_comparison, validate_runtime_metric
from compare_intermediate_local_reference import read_batches
from prepare_cayley_rms_cover import read, write, sha, require
from prepare_native_confirmation_atlas import AtomUnionAudit
from prepare_far_atlas_repeat import validate_repeat
from run_shoulder_mis_campaign import local_dependencies

ROOT = Path(__file__).resolve().parents[1]
PREPARATION = ROOT/'runs/ab-far-contact-atlas-preparation-20260921'


def command_options(command):
    """Read the generated launcher argument vector without executing it."""
    require(isinstance(command, list) and len(command) >= 2, 'Invalid frozen command vector')
    result = {}; i = 2
    while i < len(command):
        key = command[i]; require(key.startswith('--') and key not in result, 'Repeated or positional generated option')
        if i+1 == len(command) or command[i+1].startswith('--'):
            result[key] = True; i += 1
        else:
            result[key] = command[i+1]; i += 2
    return result


def validate_repeat_lineage(prep, protocol, freeze):
    source = protocol.get('repeat_source')
    if source is None: return None
    path = Path(source['path']); original = read(path); old_freeze_path = Path(source['freeze_path']); old_freeze = read(old_freeze_path)
    require(sha(path) == source['sha256'] == old_freeze['protocol_sha256'], 'Original repeat protocol changed')
    require(sha(old_freeze_path) == source['freeze_sha256'], 'Original model seal changed')
    baseline_path = Path(source['completed_baseline_analysis_path']); baseline = read(baseline_path)
    require(sha(baseline_path) == source['completed_baseline_analysis_sha256'] and baseline['complete'], 'Completed baseline analysis changed')
    require(baseline['source_sha256'][str(path)] == source['sha256'] and baseline['original_q_window'] == protocol['q_window'], 'Baseline target or lineage differs')
    require(source['model_sha256'] == old_freeze['model_sha256'], 'Original repeat model hashes changed')
    for name, digest in old_freeze['archived_sha256'].items(): require(sha(path.parent/'provenance'/name) == digest, 'Original preparation archive changed')
    for name, digest in baseline['archived_sha256'].items(): require(sha(baseline_path.parent/'provenance'/name) == digest, 'Original executed audit source changed')
    for input_path, digest in baseline['source_sha256'].items(): require(sha(Path(input_path)) == digest, 'Baseline audited source changed')
    prior_seeds = {p['seed'] for c in baseline['campaigns'] for p in c['populations']}
    validate_repeat(original, protocol, old_freeze['model_sha256'], freeze['model_sha256'], prior_seeds)
    for arm in protocol['arms']:
        name = arm['name']; original_model = path.parent/f'model-{name}.json'
        require(sha(original_model) == old_freeze['model_sha256'][name] == sha(prep/f'model-{name}.json'), 'Repeat model bytes differ from original')
        require(read(original_model)['proposal_provenance']['protocol_sha256'] == source['sha256'], 'Original embedded model lineage changed')
        require(arm['model_source_path'] == str(original_model), 'Repeat model source path changed')
    command_path = prep/'commands.json'
    require(sha(command_path) == protocol['commands_sha256'] == freeze['commands_sha256'], 'Frozen repeat commands changed')
    commands = read(command_path); require(set(commands) == {a['name'] for a in protocol['arms']}, 'Frozen command arms differ')
    for arm in protocol['arms']:
        command = commands[arm['name']]; runner = prep/'provenance/repeat-runner.py'
        require(command[1] == str(runner) and sha(runner) == protocol['archived_sha256']['repeat-runner.py'], 'Frozen launcher changed')
        opts = command_options(command)
        expected = {'--root': arm['output'], '--config': str(prep/'config.json'), '--binary': protocol['executable'],
            '--expected-binary-sha256': protocol['executable_sha256'], '--replicates': str(arm['populations']),
            '--samples': str(arm['samples_per_population']), '--workers': str(arm['workers']), '--seed-base': str(arm['seeds'][0]),
            '--q-min': '5', '--q-max': '37', '--q-upper-open': True, '--cover-scales': '1', '--model': arm['model_source_path'],
            '--model-weight': str(protocol['hybrid']['model_weight']), '--model-uniform-probability': str(protocol['hybrid']['uniform_probability']), '--model-anchor-index': '0'}
        require(opts == expected, 'Frozen repeat command parameters differ')
        master = read(Path(arm['output'])/'manifest.json')
        require(master['model_source_path'] == arm['model_source_path'] and master['source_config_path'] == expected['--config'], 'Actual repeat source paths differ from frozen command')
        require(master['runner_sha256'] == sha(runner) and master['max_workers'] == arm['workers'], 'Actual repeat launcher/workers differ')
    return dict(baseline_path=str(baseline_path), baseline_sha256=sha(baseline_path), original_protocol_sha256=source['sha256'],
        original_model_sha256=old_freeze['model_sha256'], commands_sha256=sha(command_path), byte_identical_models=True,
        original_provenance_preserved=True, scope='Only allocation, fresh streams and prefixes changed; baseline and repeat remain separate estimates.')


def compare_baseline_repeat(baseline, repeated, protocol):
    """Separate estimates and same-domain differences, never pooled means."""
    require(baseline['complete'] and baseline['original_q_window'] == protocol['q_window'] == WINDOW, 'Baseline/repeat targets differ')
    rows = []
    for repeat in repeated:
        old = next(c for c in baseline['campaigns'] if c['arm'] == repeat['arm'])
        old_seeds = {p['seed'] for p in old['populations']}; new_seeds = {p['seed'] for p in repeat['populations']}
        require(not old_seeds.intersection(new_seeds), 'Baseline/repeat streams overlap')
        comparisons = {}
        for kind in ('physical', 'hard'):
            comparisons[kind] = {}
            for key in MASKS:
                a, b = old[kind][key], repeat[kind][key]; value = independent_comparison(a, b)
                comparisons[kind][key] = dict(baseline=a, repeat=b, baseline_minus_repeat_logQ=value['historical_minus_fresh_logQ'],
                    baseline_to_repeat_ratio=value['historical_to_fresh_ratio'],
                    linear_difference_in_combined_row_SE=value['linear_difference_in_combined_row_SE'],
                    linear_difference_in_combined_population_SE=value['linear_difference_in_combined_population_SE'])
        rows.append(dict(arm=repeat['arm'], baseline_root=old['root'], repeat_root=repeat['root'], comparisons=comparisons,
            baseline_unconditional_draws=old['physical']['full']['draws'], repeat_unconditional_draws=repeat['physical']['full']['draws']))
    return dict(arms=rows, pooled=False,
        scope='Separate independent original-density estimates on identical frozen masks. Larger repeat allocation was chosen after the baseline; comparisons use observed errors, not nominal unseen-tail guarantees. No baseline/repeat pooling.')


class PrefixMoments:
    """Every raw draw, including rejected zeros, advances each fixed prefix."""
    def __init__(self, samples, prefixes):
        validate_prefix_counts(samples, prefixes)
        self.samples = samples; self.count = 0
        self.prefixes = {n: HistoryMoments() for n in prefixes}

    def add(self, draw, *args):
        require(draw == self.count and draw < self.samples, 'Missing, duplicate or extra raw prefix row')
        for n, moments in self.prefixes.items():
            if draw < n: moments.add(*args)
        self.count += 1

    def complete(self): require(self.count == self.samples, 'Incomplete original population')


def finish(aggregate, populations):
    result = add_population_errors(aggregate.report(), populations, aggregate.keys)
    result['populations'] = populations
    for kind in ('physical', 'hard'):
        total = result[kind]['full']['logQ']
        for key, row in result[kind].items():
            values = row['population_logQ_values']; finite = [v for v in values if v is not None]
            offset = max(finite) if finite else 0.
            weights = np.array([0. if v is None else math.exp(v-offset) for v in values])
            fractions = weights/weights.sum() if weights.sum() else None
            row['population_mass_fractions'] = fractions.tolist() if fractions is not None else None
            row['maximum_population_fraction'] = float(fractions.max()) if fractions is not None else None
            row['population_weight_ESS'] = float(1/np.sum(fractions**2)) if fractions is not None else None
            row['fraction_of_observed_full_far'] = math.exp(row['logQ']-total) if row['logQ'] is not None and total is not None else None
    return result


def compare_widths(narrow, broad):
    result = independent_comparison(narrow, broad)
    result['scope'] = ('Narrow minus broad on linear physical mass, with independent fixed production streams and separately retained proposal denominators. '
        'Observed row/population errors are diagnostics, not a certificate for unseen tails.')
    return dict(narrow=narrow, broad=broad, narrow_minus_broad_logQ=result['historical_minus_fresh_logQ'],
        narrow_to_broad_ratio=result['historical_to_fresh_ratio'],
        linear_difference_in_combined_row_SE=result['linear_difference_in_combined_row_SE'],
        linear_difference_in_combined_population_SE=result['linear_difference_in_combined_population_SE'], scope=result['scope'])


def validate_arm(root, arm, prep, protocol, freeze):
    root = Path(root); master = read(root/'manifest.json'); cfg = read(root/'provenance/config.json')
    require(str(root) == arm['output'] and all(cfg[k] == protocol['physical'][k] for k in PHYSICAL), 'Frozen arm or physical target changed')
    require(master['q_window'] == protocol['q_window'] == WINDOW, 'Original far target changed')
    require(set(protocol['analysis']['masks']) == set(MASKS), 'Frozen radial mask set changed')
    validate_prefix_counts(arm['samples_per_population'], protocol['prefix_counts'])
    require(master['total_unconditional_draws'] == arm['populations']*arm['samples_per_population'] and len(master['jobs']) == arm['populations'], 'Fixed original budget changed')
    require([j['seed'] for j in master['jobs']] == arm['seeds'], 'Frozen random streams changed')
    for name, digest in master['archive_sha256'].items(): require(sha(root/'provenance'/name) == digest, f'Arm archive changed: {name}')
    require(sha(root/'provenance/guide-model.json') == freeze['model_sha256'][arm['name']] == sha(prep/f"model-{arm['name']}.json"), 'Frozen proposal model changed')
    require(len(read(root/'provenance/guide-model.json')['weights']) == arm['components'], 'Frozen model component count changed')
    require(master['binary_sha256'] == protocol['executable_sha256'], 'Frozen physical executable changed')
    require(master['archive_sha256']['analyze_native_region_reference.py'] == protocol['archived_sha256']['analyze_native_region_reference.py'], 'Archived density auditor differs from preparation')
    for job in master['jobs']:
        path = Path(job['output']); run = read(path/'manifest.json'); summary = read(path/'summary.json')
        require(summary['complete'] and run['samples'] == summary['samples'] == arm['samples_per_population'], 'Physical population incomplete or budget changed')
        require(run['seed'] == job['seed'] and run['q_window'] == WINDOW and run['schema'] == 4, 'Original window or stream differs')
        require(run['cloud_replicates'] == protocol['cloud_replicates'] == 2 and run['lambda_ratio'] == protocol['lambda_ratio'] == 64., 'Frozen Poisson law changed')
        require(run['activity'] == cfg['reservoir_density'] and run['lambda'] == run['activity']*run['lambda_ratio'], 'Physical activity differs')
        require(run['executable_sha256'] == protocol['executable_sha256'] and run['config_sha256'] == master['config_sha256'] and run['shape_sha256'] == master['shape_sha256'], 'Population physical provenance differs')
        guide = run['guide']; hybrid = protocol['hybrid']
        require(guide['model_sha256'] == freeze['model_sha256'][arm['name']] and guide['weight'] == hybrid['model_weight'] and guide['uniform_probability'] == hybrid['uniform_probability'], 'Frozen full hybrid denominator differs')
        require(guide['anchor_index'] == protocol['proposal_anchor_index'] == 0 and guide['anchor_pose'] == cfg['fixed_poses'][0], 'Physical A proposal frame differs')
        require(run['cover_mixture']['scales'] == hybrid['cover_scales'], 'Complete proposal cover scales differ')
        validate_runtime_metric(run['metric'], cfg['metadata'])
    return master, cfg


def wait_for_auditors(live, state, state_path):
    """Record every launched child's outcome before reporting any failure."""
    failures = []
    for process, log, record in live:
        code = process.wait(); log.close(); record.update(terminal=True, exit_code=code)
        write(state_path, state)
        if code != 0: failures.append(f"{record['arm']} exit{code}; see {record['log']}")
    if failures:
        state['phase'] = 'density_audits_failed'; state['failures'] = failures; write(state_path, state)
        raise ValueError('Archived density audit failures: '+'; '.join(failures))


def ensure_density_audits(requests, out, allow_run, workers):
    """One archived auditor per arm; valid existing outputs are reused."""
    require(type(workers) is int and workers > 0, 'Positive density worker count required')
    state = dict(phase='original_density_audits', processes=[]); live = []
    state_path = out/'runner-state.json'
    try:
        for root, arm, *_ in requests:
            audit_path = root/'assessment-streaming.json'
            if audit_path.exists():
                state['processes'].append(dict(arm=arm['name'], reused=True, path=str(audit_path), sha256=sha(audit_path)))
                continue
            require(allow_run, 'A density audit is missing; enable --run-density-audits only after physical completion authorization')
            command = [sys.executable, str(root/'provenance/analyze_native_region_reference.py'), str(root), '--workers', str(workers)]
            log_path = out/f"density-audit-{arm['name']}.log"; log = log_path.open('x')
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            record = dict(arm=arm['name'], pid=process.pid, command=command, log=str(log_path), output=str(audit_path), terminal=False)
            state['processes'].append(record); live.append((process, log, record)); write(state_path, state)
            print(f"{arm['name']}: archived all-row density audit PID {process.pid}", flush=True)
        wait_for_auditors(live, state, state_path)
        state['phase'] = 'density_audits_complete'; write(state_path, state)
    finally:
        for process, log, record in live:
            if process.poll() is not None: log.close()
    return state


def remask_arm(root, arm, master, cfg, protocol, geometry, geometric_model):
    audit_path = root/'assessment-streaming.json'; original = read(audit_path)
    require(original['complete'] and original['all_rows_and_hashes_validated'] and original['independent_complete_cover_validated'], 'Incomplete full-density audit')
    require(original['q_window'] == WINDOW and original['target_key'] == 'region' and original['samples'] == master['total_unconditional_draws'] == original['independent_q_and_density_rows'], 'Wrong audited original target or full N')
    require(original['analysis_script_sha256'] == master['archive_sha256']['analyze_native_region_reference.py'], 'Full-density audit used different source')
    counts = protocol['prefix_counts']; aggregate = {n: HistoryMoments() for n in counts}; populations = {n: [] for n in counts}
    sources = {str(root/'manifest.json'): sha(root/'manifest.json'), str(audit_path): sha(audit_path)}; top = []; errors = dict(member_rms2=0., geometric_radius_relative=0., matrix_radius_relative=0.)
    for pi, job in enumerate(master['jobs']):
        path = Path(job['output']); sample = path/'samples.jsonl'; local = PrefixMoments(arm['samples_per_population'], counts)
        audited = next(p for p in original['populations'] if p['replicate'] == path.name)
        require(sha(path/'summary.json') == audited['summary_sha256'], 'Physical summary changed after density audit')
        require(audited['samples'] == arm['samples_per_population'], 'Audited population budget changed')
        digest = hashlib.sha256()
        for lines, rows in read_batches(sample):
            for line in lines: digest.update(line)
            poses = [r['pose'] for r in rows]; positions = np.asarray([p['position'] for p in poses]); quats = np.asarray([p['orientation'] for p in poses])
            near(np.sum(quats*quats, axis=1), np.ones(len(rows)))
            rotations = Rotation.from_quat(quats[:, [1, 2, 3, 0]]).as_matrix()
            radii, _, current, _ = geometry_rows(positions, rotations, geometry)
            _, check_radii, diagnostic = region_latent(poses, dict(gaussian_chart=geometric_model, fixed_neighbor=cfg['fixed_poses'][0]))
            finite = np.isfinite(radii) & np.isfinite(check_radii)
            require(np.array_equal(np.isfinite(radii), np.isfinite(check_radii)), 'Analysis Cayley seam mismatch')
            radius_error = float(np.max(abs(radii[finite]-check_radii[finite])/(1+check_radii[finite]))) if finite.any() else 0.
            require(radius_error < 2e-7, 'Independent geometric radius mismatch')
            errors['member_rms2'] = max(errors['member_rms2'], current['member_rms2']); errors['geometric_radius_relative'] = max(errors['geometric_radius_relative'], radius_error)
            errors['matrix_radius_relative'] = max(errors['matrix_radius_relative'], diagnostic['matrix_radius_relative_error'])
            for boundary in (.5, 1., 2.): require(np.array_equal(radii <= boundary, check_radii <= boundary), 'Frozen local masks differ between reconstructions')
            for i, row in enumerate(rows):
                if row.get('zero') is not None:
                    require(row.get('log_importance_weight') is None, 'Rejected row has nonzero weight'); local.add(row['draw']); continue
                require(5 <= row['q'] < 37, 'Positive row outside original far target')
                physical, hard = row['log_importance_weight'], row['log_hard_weight']; pair = [v+hard for v in row['cloud_log_weights']]
                near(hard, -row['log_proposal_density']); require(len(pair) == 2, 'Missing original cloud pair')
                near(float(np.logaddexp(*pair))-math.log(2), physical)
                local.add(row['draw'], float(radii[i]), physical, hard, pair)
                record = dict(population=path.name, seed=job['seed'], draw=row['draw'], pose=row['pose'], q=row['q'], geometric_radius_A=float(radii[i]),
                    original_log_importance_weight=physical, original_row=row, source_samples_path=str(sample))
                entry = (physical, -pi, -row['draw'], record)
                if len(top) < 8: heapq.heappush(top, entry)
                elif entry[:3] > top[0][:3]: heapq.heapreplace(top, entry)
        local.complete(); require(digest.hexdigest() == audited['sample_sha256'], 'Original rows changed after density audit')
        sources[str(sample)] = digest.hexdigest()
        for name in ('manifest.json', 'summary.json'): sources[str(path/name)] = sha(path/name)
        for n in counts:
            current = local.prefixes[n]; aggregate[n].merge(current)
            populations[n].append(dict(id=path.name, seed=job['seed'], samples=n, **current.report()))
    reports = {n: finish(aggregate[n], populations[n]) for n in counts}; result = reports[counts[-1]]
    for kind, original_key in (('physical', 'regions'), ('hard', 'hard_regions')):
        source = original[original_key]['region']; actual = result[kind]['full']
        if actual['logQ'] is None: require(source['log_normalizer'] is None, 'Full zero estimate changed')
        else:
            near(actual['logQ'], source['log_normalizer']); near(actual['row_RSE'], source['relative_SE'])
            require(actual['draws'] == source['samples'] and actual['nonzero'] == source['nonzero'], 'Original support/count changed')
    comparisons = [dict(earlier_samples_per_population=m, later_samples_per_population=n,
        physical={k: nested_prefix_comparison(reports[m]['physical'][k], reports[n]['physical'][k]) for k in MASKS},
        hard={k: nested_prefix_comparison(reports[m]['hard'][k], reports[n]['hard'][k]) for k in MASKS})
        for i, m in enumerate(counts) for n in counts[i+1:]]
    atom = AtomUnionAudit(read(root/'provenance/shape.json'), cfg['fixed_poses']); extrema = []
    for _, _, _, point in sorted(top, key=lambda t: t[:3], reverse=True):
        gaps = atom.gaps(point['pose']); require(min(gaps) >= 0, 'Extreme pose fails independent atomic AB check')
        point['minimum_AB_atomic_gaps_A'] = gaps; point['source_samples_sha256'] = sources[point['source_samples_path']]
        point['fraction_of_full_weight'] = math.exp(point['original_log_importance_weight']-aggregate[counts[-1]].values['physical']['full'].total)
        extrema.append(point)
    print(f"{arm['name']}: logQ={result['physical']['full']['logQ']}; row RSE={result['physical']['full']['row_RSE']}", flush=True)
    return dict(root=str(root), arm=arm['name'], **result, prefixes=[dict(samples_per_population=n, **reports[n]) for n in counts],
        prefix_comparisons=comparisons, top8_atomic_checks=extrema, input_sha256=sources, geometry_reconstruction_errors=errors,
        full_density_audit_path=str(audit_path), full_density_audit_sha256=sha(audit_path), independently_audited_rows=original['independent_q_and_density_rows'],
        CPU_seconds=original['cpu_seconds'], full_density_audit_seconds=original['analysis_wall_seconds'],
        prefix_scope='Predeclared first-m rows in every independent population. Prefixes share rows; covariance is estimated under the fixed IID law. No independent-confirmation or optional-stopping interpretation.')


def analyze(prep, out, run_audits=False, audit_workers=8):
    prep = Path(prep).resolve(); out = Path(out).resolve(); require(not out.exists(), 'Fresh derived output directory required')
    protocol = read(prep/'protocol.json'); freeze = read(prep/'freeze.json'); cfg = read(prep/'config.json')
    require(sha(prep/'protocol.json') == freeze['protocol_sha256'] and sha(prep/'config.json') == freeze['config_sha256'], 'Frozen preparation differs')
    for name, digest in protocol['archived_sha256'].items(): require(sha(prep/'provenance'/name) == digest, 'Preparation archive changed')
    require(protocol['q_window'] == WINDOW and all(cfg[k] == protocol['physical'][k] for k in PHYSICAL), 'Frozen far physical target differs')
    lineage = validate_repeat_lineage(prep, protocol, freeze)
    local_model_path = Path(protocol['analysis']['local_model_path']); ref_path = Path(protocol['analysis']['local_reference_path'])
    require(sha(local_model_path) == protocol['analysis']['local_model_sha256'] and sha(ref_path) == protocol['analysis']['local_reference_sha256'], 'Frozen calibration definitions changed')
    model = read(local_model_path); reference = read(ref_path); require(reference['complete'] and reference['original_q_window'] == WINDOW, 'Calibration reference wrong or incomplete')
    center_selection = read(Path(protocol['center_selection']['path']))
    require(sha(Path(protocol['center_selection']['path'])) == protocol['center_selection']['sha256'], 'Frozen centers differ')
    peak = center_selection['centers'][0]['pose']; geometry_audit, geometry = geometry_model_audit(model, dict(fixed_neighbor=cfg['fixed_poses'][0]), cfg, peak)
    require(model['shape_sha256'] == sha(prep/'provenance/shape.json'), 'Analysis shape differs')
    requests = []; seeds = {p['seed'] for c in reference['campaigns'] for p in c['populations']}
    for arm in protocol['arms']:
        root = Path(arm['output']); master, current_cfg = validate_arm(root, arm, prep, protocol, freeze)
        require(not seeds.intersection(arm['seeds']), 'Production/calibration random streams overlap'); seeds.update(arm['seeds'])
        requests.append((root, arm, master, current_cfg))
    require([r[1]['name'] for r in requests] == ['narrow', 'broad'], 'Frozen width arms differ')
    out.mkdir(parents=True); state = ensure_density_audits(requests, out, run_audits, audit_workers)
    campaigns = [remask_arm(root, arm, master, current_cfg, protocol, geometry, model) for root, arm, master, current_cfg in requests]
    for campaign in campaigns:
        calibration = {}
        for i, radius in enumerate((.5, 1., 2.)):
            source = next(c for c in reference['campaigns'] if c['radius_A'] == radius)
            for path, digest in source['sample_sha256'].items(): require(sha(Path(path)) == digest, 'Calibration sample changed')
            for key, ref_key in [(('ball0p5', 'ball1', 'ball2')[i], 'full'), (f'radial_{i}', f'radial_{i}')]:
                calibration[key] = {kind: independent_comparison(campaign[kind][key], source[kind][ref_key]) for kind in ('physical', 'hard')}
                for value in calibration[key].values(): value['scope'] = 'Calibration against local draws used in proposal learning/center discovery. Fresh production streams are independent conditional on the fit, but this is not held-out validation or a tail certificate.'
        calibration['ball2_independent_shell_sum'] = {kind: independent_comparison(campaign[kind]['ball2'], reference['independent_shell_sum'][kind]['full']) for kind in ('physical', 'hard')}
        for values in calibration.values():
            for value in values.values(): value['scope'] = 'Calibration against local draws used in proposal learning/center discovery. Fresh production streams are independent conditional on the fit, but this is not held-out validation or a tail certificate.'
        campaign['local_calibration'] = calibration
    width_comparisons = {kind: {key: compare_widths(campaigns[0][kind][key], campaigns[1][kind][key]) for key in MASKS} for kind in ('physical', 'hard')}
    baseline_comparison = compare_baseline_repeat(read(Path(lineage['baseline_path'])), campaigns, protocol) if lineage else None
    sources = {str(p): sha(p) for p in [prep/'protocol.json', prep/'freeze.json', prep/'config.json', local_model_path, ref_path]}
    if lineage:
        for path in (prep/'commands.json', Path(lineage['baseline_path']), Path(protocol['repeat_source']['path'])): sources[str(path)] = sha(path)
    for campaign in campaigns:
        sources.update(campaign['input_sha256'])
        for path, digest in campaign['input_sha256'].items(): require(sha(Path(path)) == digest, 'Arm input changed during final audit')
    archive = out/'provenance'; archive.mkdir()
    extra = {'atlas-protocol.json': prep/'protocol.json', 'atlas-freeze.json': prep/'freeze.json', 'physical-config.json': prep/'config.json',
        'geometric-model.json': local_model_path, 'local-reference.json': ref_path, 'model-narrow.json': prep/'model-narrow.json', 'model-broad.json': prep/'model-broad.json'}
    if lineage: extra.update({'repeat-commands.json': prep/'commands.json', 'baseline-analysis.json': Path(lineage['baseline_path']), 'original-protocol.json': Path(protocol['repeat_source']['path'])})
    for name, path in {**local_dependencies([Path(__file__)]), **extra}.items(): (archive/name).write_bytes(path.read_bytes())
    scope = ('Complete original 5<=q<37 far target with capture and both AB hard neighbors. Every estimate retains its original hybrid density and unconditional zeros. '
        'Disjoint radial shells plus outside2 partition full far mass with same-row covariance; nested balls are not summed. '
        'Widths remain independent, separate estimates. Prefixes share rows and use shared-stream covariance. Local-reference comparisons are calibration because these data informed the proposal. '
        'Importance ESS, population concentration and observed errors do not bound unseen tails, equilibrium mixing or assembly. No physical draw, Poisson cloud or fit is generated by this analysis.')
    result = dict(complete=True, original_q_window=WINDOW, prefix_counts=protocol['prefix_counts'], campaigns=campaigns,
        width_comparisons=width_comparisons, geometric_model_audit=geometry_audit, source_sha256=sources,
        archived_sha256={p.name: sha(p) for p in archive.iterdir()}, scope=scope)
    if lineage: result.update(repeat_lineage=lineage, baseline_repeat_comparison=baseline_comparison)
    write(out/'analysis.json', result); state.update(phase='complete', analysis_sha256=sha(out/'analysis.json')); write(out/'runner-state.json', state)
    def num(x): return 'unresolved' if x is None else f'{x:.6g}'
    lines = ['# Complete-far contact atlas audit', '', '| Arm | Mask | N | Positive | log Q | Row / population RSE | Max row | Max population |', '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for campaign in campaigns:
        for key, row in campaign['physical'].items():
            lines.append(f"| {campaign['arm']} | {key} | {row['draws']} | {row['nonzero']} | {num(row['logQ'])} | {num(row['row_RSE'])} / {num(row['independent_population_RSE'])} | {num(row['maximum_fraction'])} | {num(row['maximum_population_fraction'])} |")
    if baseline_comparison:
        lines += ['', '| Arm | Mask | Baseline log Q | Repeat log Q | Difference / combined row SE |', '| --- | --- | ---: | ---: | ---: |']
        for arm in baseline_comparison['arms']:
            for key, comparison in arm['comparisons']['physical'].items():
                lines.append(f"| {arm['arm']} | {key} | {num(comparison['baseline']['logQ'])} | {num(comparison['repeat']['logQ'])} | {num(comparison['linear_difference_in_combined_row_SE'])} |")
        lines += ['', baseline_comparison['scope']]
    lines += ['', scope, '', 'Full hard masses, cloud variance, covariance matrices, population values, prefix differences, calibrated reference comparisons, extremes and source hashes are in analysis.json.', '']
    (out/'report.md').write_text('\n'.join(lines)); return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--preparation', type=Path, default=PREPARATION)
    parser.add_argument('--out', type=Path, required=True); parser.add_argument('--run-density-audits', action='store_true')
    parser.add_argument('--audit-workers', type=int, default=8, help='Workers per independent arm auditor; two arms may audit concurrently')
    args = parser.parse_args(); result = analyze(args.preparation, args.out, args.run_density_audits, args.audit_workers)
    print(dict(complete=result['complete'], output=str(args.out/'analysis.json')))
