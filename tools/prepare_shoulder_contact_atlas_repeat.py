#!/usr/bin/env python3
"""Freeze a larger independent shoulder repeat without changing any proposal."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
from pathlib import Path
import shutil
import subprocess
import sys

from analyze_shoulder_contact_atlas import validate_plan
from prepare_cayley_rms_cover import read, write, sha, require
from prepare_far_atlas_repeat import validate_command_options

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT/'runs/ab-shoulder-contact-atlas-preparation-20260921'
ORIGINAL_ANALYSIS = ROOT/'runs/ab-shoulder-contact-atlas-assessment-20260921/analysis.json'
ORIGINAL_PROTOCOL_SHA = '1a846e78ff27b4af3572d81461ed1c58ac7db37ce0aa3bfe9c3c92b773b1308d'
ORIGINAL_ANALYSIS_SHA = '24e101b214af665dcf67697ea83d4d757d21703fd8e55cc7f5eb79a02c86d7df'
SEEDS = dict(narrow=114101010, broad=114201010)
POPULATIONS, SAMPLES, WORKERS = 16, 65536, 16
PREFIXES = [16384, 32768, 65536]
UNCHANGED = ('schema', 'physical', 'config_sha256', 'shape_sha256', 'q_window',
    'inner_q_window', 'executable', 'executable_sha256', 'lambda_ratio',
    'cloud_replicates', 'proposal_anchor_index', 'hybrid', 'atlas', 'analysis',
    'cover_proof_sha256', 'maximum_concurrent_workers')


def validate_repeat_target(original, repeat, original_models, repeat_models):
    """Reject target/proposal changes even if a new protocol is internally valid."""
    validate_plan(original); validate_plan(repeat)
    for key in UNCHANGED:
        require(original[key] == repeat[key], f'Repeat changed frozen target/proposal: {key}')
    require(original_models == repeat_models, 'Repeat proposal bytes differ')
    require(repeat['prefix_counts'] == PREFIXES, 'Repeat fixed prefixes differ')
    require(repeat['total_unconditional_draws'] == 2*POPULATIONS*SAMPLES, 'Repeat total N differs')
    old_seeds = {s for arm in original['arms'] for s in arm['seeds']}
    for old, new in zip(original['arms'], repeat['arms']):
        name = new['name']
        require(new['components'] == old['components'] == 5 and
                new['new_covariance_scale'] == old['new_covariance_scale'], 'Repeat atlas changed')
        require(new['populations'] == POPULATIONS and new['samples_per_population'] == SAMPLES
                and new['workers'] == WORKERS, 'Repeat allocation differs')
        require(new['seeds'] == [SEEDS[name]+1009*i for i in range(POPULATIONS)], 'Repeat fixed seeds differ')
        require(not old_seeds.intersection(new['seeds']), 'Repeat reuses original random streams')
        require(new['model_sha256'] == old['model_sha256'] == original_models[name], 'Repeat model lineage differs')


def checked_copy(source, target):
    if target.exists():
        require(sha(source) == sha(target), f'Frozen source collision: {target.name}')
    else:
        shutil.copy2(source, target)


def prepare(out):
    out = Path(out).resolve(); require(not out.exists(), 'Fresh repeat preparation required')
    source, seal, previous = read(ORIGINAL/'protocol.json'), read(ORIGINAL/'freeze.json'), read(ORIGINAL_ANALYSIS)
    require(sha(ORIGINAL/'protocol.json') == seal['protocol_sha256'] == ORIGINAL_PROTOCOL_SHA, 'Original protocol changed')
    require(sha(ORIGINAL_ANALYSIS) == ORIGINAL_ANALYSIS_SHA and previous['complete'], 'Completed original analysis changed')
    require(sha(ORIGINAL/'config.json') == source['config_sha256'] == seal['config_sha256'], 'Original config bytes changed')
    require(sha(ORIGINAL/'commands.json') == source['commands_sha256'] == seal['commands_sha256'], 'Original commands changed')
    require(seal['archived_sha256'] == source['archived_sha256'], 'Original archive seal differs')
    for directory, hashes in ((ORIGINAL/'provenance', seal['archived_sha256']),
                              (ORIGINAL_ANALYSIS.parent/'provenance', previous['archived_sha256'])):
        for name, digest in hashes.items(): require(sha(directory/name) == digest, 'Original small archive changed')
    require(sha(source['executable']) == source['executable_sha256'], 'Reviewed executable changed')
    cfg = read(ORIGINAL/'config.json')
    require(sha(cfg['shape']) == source['shape_sha256'], 'Original physical shape changed')
    for arm in source['arms']:
        require(sha(arm['model_path']) == arm['model_sha256'] == seal['model_sha256'][arm['name']], 'Original model bytes changed')
        campaign = next(c for c in previous['campaigns'] if c['arm'] == arm['name'])
        require([p['seed'] for p in campaign['populations']] == arm['seeds'] and
                campaign['physical']['full']['draws'] == arm['populations']*arm['samples_per_population'], 'Wrong original complete streams')
    archive = out/'provenance'; archive.mkdir(parents=True)
    inputs = {}
    for name in source['archived_sha256']:
        path = ORIGINAL/'provenance'/name; checked_copy(path, archive/name); inputs[str(path)] = sha(path)
    for name in previous['archived_sha256']:
        path = ORIGINAL_ANALYSIS.parent/'provenance'/name
        if path.suffix == '.py': checked_copy(path, archive/name); inputs[str(path)] = sha(path)
    extra = {'original-protocol.json': ORIGINAL/'protocol.json', 'original-freeze.json': ORIGINAL/'freeze.json',
        'original-analysis.json': ORIGINAL_ANALYSIS, 'original-commands.json': ORIGINAL/'commands.json',
        'prepare_shoulder_contact_atlas_repeat.py': Path(__file__),
        'analyze_shoulder_contact_atlas_repeat.py': ROOT/'tools/analyze_shoulder_contact_atlas_repeat.py'}
    for name, path in extra.items(): checked_copy(path, archive/name); inputs[str(path)] = sha(path)
    # Keep the original absolute shape path: even config bytes remain identical.
    for name in ('config.json', 'model-narrow.json', 'model-broad.json', 'cover-proof.json'):
        checked_copy(ORIGINAL/name, out/name); inputs[str(ORIGINAL/name)] = sha(ORIGINAL/name)
    for center in source['analysis']['centers']:
        path = Path(center['model_path']); require(sha(path) == center['model_sha256'], 'Original diagnostic chart changed')
        checked_copy(path, out/path.name); inputs[str(path)] = sha(path)
    protocol = copy.deepcopy(source)
    protocol['created_utc'] = datetime.now(timezone.utc).isoformat()
    protocol['prefix_counts'] = PREFIXES
    protocol['total_unconditional_draws'] = 2*POPULATIONS*SAMPLES
    protocol['probes'] = dict(new_draws=0, original_probes_reused=True,
        scope='No geometry or physical proposal probes; byte-identical frozen models and masks.')
    protocol['repeat_source'] = dict(path=str(ORIGINAL/'protocol.json'), sha256=ORIGINAL_PROTOCOL_SHA,
        analysis_path=str(ORIGINAL_ANALYSIS), analysis_sha256=ORIGINAL_ANALYSIS_SHA,
        config_sha256=seal['config_sha256'], model_sha256=seal['model_sha256'],
        rule='Independent fixed-N repeat; exact model/config bytes, full hybrid density, AB target and all analysis masks unchanged.')
    protocol['selection_scope'] = ('Repeat size selected after the original results. Every new budget and prefix is fixed before drawing; '
        'no refit, adaptive stopping, weight clipping, row replacement or historical/fresh pooling.')
    old_commands = read(ORIGINAL/'commands.json'); commands = {}
    for arm in protocol['arms']:
        name = arm['name']; arm.update(populations=POPULATIONS, samples_per_population=SAMPLES, workers=WORKERS,
            seeds=[SEEDS[name]+1009*i for i in range(POPULATIONS)], model_path=str(out/f'model-{name}.json'),
            output=str(ROOT/f'runs/ab-shoulder-contact-atlas-repeat-{name}-16x65536-l64-20260921'))
        require(not Path(arm['output']).exists(), 'Preserve existing campaign output')
        command = list(old_commands[name]); command[0] = sys.executable; command[1] = str(archive/'run_native_region_reference.py')
        for option, value in {'--root': arm['output'], '--config': str(out/'config.json'), '--model': arm['model_path'],
                              '--samples': str(SAMPLES), '--replicates': str(POPULATIONS), '--workers': str(WORKERS),
                              '--seed-base': str(arm['seeds'][0])}.items(): command[command.index(option)+1] = value
        commands[name] = command
    reference = read(source['analysis']['finite_reference_path']); direct = read(source['analysis']['direct_reference_path'])
    historical_seeds = {p['seed'] for data in (reference, direct) for c in data['campaigns']+data.get('histories', []) for p in c['populations']}
    require(not historical_seeds.intersection(s for a in protocol['arms'] for s in a['seeds']), 'Repeat overlaps calibration streams')
    models = {name: sha(out/f'model-{name}.json') for name in SEEDS}
    validate_repeat_target(source, protocol, seal['model_sha256'], models)
    help_text = subprocess.run([sys.executable, str(archive/'run_native_region_reference.py'), '--help'],
                              check=True, text=True, capture_output=True).stdout
    for command in commands.values(): validate_command_options(command, help_text)
    (archive/'repeat-runner-help.txt').write_text(help_text); write(out/'commands.json', commands)
    analysis_command = [sys.executable, str(archive/'analyze_shoulder_contact_atlas_repeat.py'), '--preparation', str(out),
        '--out', str(ROOT/'runs/ab-shoulder-contact-atlas-repeat-assessment-20260921'), '--run-density-audits', '--audit-workers', '16']
    write(out/'analysis-command.json', analysis_command)
    protocol.update(commands_sha256=sha(out/'commands.json'), analysis_command_sha256=sha(out/'analysis-command.json'),
        input_sha256=inputs, archived_sha256={p.name: sha(p) for p in archive.iterdir()})
    write(out/'protocol.json', protocol)
    write(out/'freeze.json', dict(protocol_sha256=sha(out/'protocol.json'), config_sha256=sha(out/'config.json'),
        model_sha256=models, commands_sha256=sha(out/'commands.json'), analysis_command_sha256=sha(out/'analysis-command.json'),
        archived_sha256=protocol['archived_sha256']))
    write(out/'report.json', dict(complete=True, prepared_only=True, no_new_geometry_or_physical_draws=True,
        original_protocol_sha256=ORIGINAL_PROTOCOL_SHA, original_analysis_sha256=ORIGINAL_ANALYSIS_SHA,
        protocol_sha256=sha(out/'protocol.json'), freeze_sha256=sha(out/'freeze.json'), identical_model_sha256=models,
        identical_config_sha256=sha(out/'config.json'), total_unconditional_draws=protocol['total_unconditional_draws'],
        prefix_counts=PREFIXES, original_streams_disjoint=True, calibration_streams_disjoint=True,
        old_data_audits_executed=False, scope=protocol['selection_scope']))
    print(dict(complete=True, prepared_only=True, protocol_sha256=sha(out/'protocol.json'), total_draws=protocol['total_unconditional_draws']))
    return protocol


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--out', type=Path, required=True)
    prepare(parser.parse_args().out)
