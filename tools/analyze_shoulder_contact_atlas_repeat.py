#!/usr/bin/env python3
"""Run the unchanged frozen shoulder audit and compare independent repeats."""
from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path
import subprocess
import sys

from analyze_shoulder_contact_atlas import KEYS, labeled_comparison
from prepare_cayley_rms_cover import read, write, sha, require
from prepare_shoulder_contact_atlas_repeat import validate_repeat_target

SCOPE = ('Independent original/repeat streams under byte-identical frozen proposals. Left is repeat and right is original. '
    'All unconditional zeros and full mixture weights remain included. Observed errors and ESS are descriptive, '
    'not unseen-tail bounds or stationary MCMC efficiency; no pooling or threshold-based stopping.')


def independent_repeat_comparison(original, repeat):
    result = labeled_comparison(repeat, original, 'repeat', 'original', SCOPE)
    result['unconditional_draw_count_ratio'] = repeat['draws']/original['draws']
    a, b = original['log_variance_of_mean'], repeat['log_variance_of_mean']
    finite = [v for v in (a, b) if v is not None]
    result['estimated_log_variance_of_difference'] = (None if not finite else
        max(finite)+math.log(sum(math.exp(v-max(finite)) for v in finite)))
    result['N_scaled_observed_variance_ratio'] = (None if a is None or b is None else
        math.exp(b-a)*result['unconditional_draw_count_ratio'])
    return result


def compare_campaigns(original, repeat):
    require(original['arm'] == repeat['arm'], 'Mismatched width controls')
    before = {p['seed'] for p in original['populations']}; after = {p['seed'] for p in repeat['populations']}
    require(not before.intersection(after), 'Repeated original/repeat random stream')
    result = dict(arm=repeat['arm'], original_CPU_seconds=original['CPU_seconds'], repeat_CPU_seconds=repeat['CPU_seconds'],
        original_prefix_comparisons=original['prefix_comparisons'], repeat_prefix_comparisons=repeat['prefix_comparisons'])
    for kind in ('physical', 'hard'):
        require(set(original[kind]) == set(repeat[kind]) == set(KEYS), 'Original/repeat mask set differs')
        result[kind] = {key: independent_repeat_comparison(original[kind][key], repeat[kind][key]) for key in KEYS}
    full = result['physical']['full']; a, b = original['physical']['full'], repeat['physical']['full']
    full['original_weight_ESS_per_CPU_second'] = a['weight_ESS']/original['CPU_seconds']
    full['repeat_weight_ESS_per_CPU_second'] = b['weight_ESS']/repeat['CPU_seconds']
    full['weight_ESS_per_CPU_second_ratio'] = (full['repeat_weight_ESS_per_CPU_second']/full['original_weight_ESS_per_CPU_second']
        if full['original_weight_ESS_per_CPU_second'] else None)
    full['observed_variance_times_CPU_ratio'] = (None if a['log_variance_of_mean'] is None or b['log_variance_of_mean'] is None else
        math.exp(b['log_variance_of_mean']-a['log_variance_of_mean'])*repeat['CPU_seconds']/original['CPU_seconds'])
    result['repeat_prefix_against_original_final'] = [dict(samples_per_population=prefix['samples_per_population'],
        **{kind: {key: independent_repeat_comparison(original[kind][key], prefix[kind][key]) for key in KEYS}
           for kind in ('physical', 'hard')}) for prefix in repeat['prefixes']]
    result['prefix_scope'] = ('Each repeat prefix is independent of the original final campaign, but these contrasts share '
        'the same original and nested repeat rows. They are correlated diagnostics, not multiple independent replications. '
        'Within-campaign prefix differences retain the source fixed-IID covariance calculation.')
    return result


def verify_preparation(prep):
    protocol, freeze = read(prep/'protocol.json'), read(prep/'freeze.json')
    require(sha(prep/'protocol.json') == freeze['protocol_sha256'], 'Repeat protocol changed')
    require(sha(prep/'config.json') == protocol['config_sha256'] == freeze['config_sha256'], 'Repeat config bytes changed')
    require(protocol['archived_sha256'] == freeze['archived_sha256'], 'Repeat archive seal differs')
    for name, digest in protocol['archived_sha256'].items(): require(sha(prep/'provenance'/name) == digest, 'Repeat archive changed')
    for name, digest in protocol['input_sha256'].items(): require(sha(name) == digest, 'Repeat preparation source changed')
    for name in ('commands', 'analysis-command'):
        key = name.replace('-', '_')+'_sha256'
        require(sha(prep/f'{name}.json') == protocol[key] == freeze[key], 'Repeat command changed')
    source = protocol['repeat_source']; original_path = Path(source['path']); original = read(original_path)
    require(sha(original_path) == source['sha256'], 'Original protocol lineage changed')
    old_freeze = read(original_path.parent/'freeze.json')
    require(old_freeze['protocol_sha256'] == source['sha256'] and old_freeze['model_sha256'] == source['model_sha256'], 'Original model seal changed')
    require(sha(original_path.parent/'config.json') == source['config_sha256'] == freeze['config_sha256'], 'Original/repeat config bytes differ')
    models = {}
    for arm in protocol['arms']:
        name = arm['name']; models[name] = sha(arm['model_path'])
        require(models[name] == freeze['model_sha256'][name] == sha(original_path.parent/f'model-{name}.json'), 'Original/repeat model bytes differ')
    validate_repeat_target(original, protocol, old_freeze['model_sha256'], models)
    original_analysis = Path(source['analysis_path'])
    require(sha(original_analysis) == source['analysis_sha256'], 'Original completed result changed')
    prior = read(original_analysis); require(prior['complete'], 'Original result incomplete')
    for name, digest in prior['archived_sha256'].items(): require(sha(original_analysis.parent/'provenance'/name) == digest, 'Original audited archive changed')
    require(prior['original_q_window'] == protocol['q_window'] and prior['inner_q_window'] == protocol['inner_q_window'], 'Original audited masks changed')
    return protocol, prior


def analyze(preparation, out, run_audits=False, audit_workers=16):
    prep, out = Path(preparation).resolve(), Path(out).resolve()
    require(not out.exists(), 'Use fresh repeat assessment; preserve failed outputs')
    protocol, prior = verify_preparation(prep)
    require(type(audit_workers) is int and 1 <= audit_workers <= 16, 'At most 16 workers per simultaneous width audit')
    out.mkdir(parents=True)
    current = out/'full-shoulder-audit'; source = prep/'provenance/analyze_shoulder_contact_atlas.py'
    command = [sys.executable, str(source), '--preparation', str(prep), '--out', str(current), '--audit-workers', str(audit_workers)]
    if run_audits: command.append('--run-density-audits')
    write(out/'runner-state.json', dict(phase='frozen_shoulder_audit', command=command, complete=False))
    with (out/'full-shoulder-audit.log').open('x') as log:
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    require(completed.returncode == 0, 'Frozen shoulder audit failed; preserve output and diagnose, do not replay automatically')
    fresh = read(current/'analysis.json'); require(fresh['complete'], 'Repeat audit incomplete')
    require(fresh['original_q_window'] == prior['original_q_window'] and fresh['inner_q_window'] == prior['inner_q_window']
            and fresh['center_order'] == prior['center_order'], 'Completed original/repeat targets differ')
    old_seeds = {p['seed'] for c in prior['campaigns'] for p in c['populations']}
    new_seeds = {p['seed'] for c in fresh['campaigns'] for p in c['populations']}
    require(not old_seeds.intersection(new_seeds), 'Original and repeat streams overlap across widths')
    comparisons = []
    for campaign in fresh['campaigns']:
        old = next(c for c in prior['campaigns'] if c['arm'] == campaign['arm'])
        comparisons.append(compare_campaigns(old, campaign))
    archive = out/'provenance'; archive.mkdir()
    # Keep the original analyzer's complete archive layout for downstream readers.
    for path in (current/'provenance').iterdir():
        (archive/path.name).write_bytes(path.read_bytes())
    files = {'protocol.json': prep/'protocol.json', 'freeze.json': prep/'freeze.json', 'commands.json': prep/'commands.json',
        'analysis-command.json': prep/'analysis-command.json', 'original-analysis.json': Path(protocol['repeat_source']['analysis_path']),
        'repeat-analysis.json': current/'analysis.json', 'analyze_shoulder_contact_atlas_repeat.py': Path(__file__),
        'prepare_shoulder_contact_atlas_repeat.py': prep/'provenance/prepare_shoulder_contact_atlas_repeat.py'}
    for name, path in files.items(): (archive/name).write_bytes(path.read_bytes())
    verify_preparation(prep)
    result = copy.deepcopy(fresh)
    result['base_audit'] = dict(path=str(current/'analysis.json'), sha256=sha(current/'analysis.json'),
        archived_sha256=fresh['archived_sha256'])
    result['repeat_source_verification'] = dict(models_unchanged=True, config_bytes_unchanged=True,
        disjoint_population_seeds=True, physical_and_masks_unchanged=True, original=protocol['repeat_source'])
    result['original_repeat_comparisons'] = comparisons
    result['original_width_comparisons'] = prior['width_comparisons']
    result['original_prefix_counts'] = prior['prefix_counts']
    result['no_pooling'] = True; result['global_convergence_established'] = False
    result['scope'] = fresh['scope']+' '+SCOPE
    result['source_sha256'].update({str(path): sha(path) for path in files.values()})
    result['archived_sha256'] = {p.name: sha(p) for p in archive.iterdir()}
    write(out/'analysis.json', result)
    write(out/'runner-state.json', dict(phase='complete', complete=True, analysis_sha256=sha(out/'analysis.json'), command=command))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preparation', type=Path, required=True); parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--run-density-audits', action='store_true'); parser.add_argument('--audit-workers', type=int, default=16)
    args = parser.parse_args(); analyze(args.preparation, args.out, args.run_density_audits, args.audit_workers)
