#!/usr/bin/env python3
"""Freeze a larger independent repeat with byte-identical far atlas models."""
from __future__ import annotations
import argparse
import copy
from datetime import datetime, timezone
from pathlib import Path
import re
import shutil
import subprocess
import sys

from prepare_cayley_rms_cover import read, write, sha, require
from analyze_far_peak_reference import WINDOW

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT/'runs/ab-far-contact-atlas-preparation-20260921'
BASELINE = ROOT/'runs/ab-far-contact-atlas-assessment-20260921/analysis.json'
SEEDS = dict(narrow=108101010, broad=108201010)
POPULATIONS, SAMPLES = 16, 65536
PREFIXES = [16384, 32768, 65536]
UNCHANGED = ('physical', 'q_window', 'executable', 'executable_sha256', 'lambda_ratio', 'cloud_replicates',
    'proposal_anchor_index', 'atlas', 'hybrid', 'center_selection', 'local_fit', 'analysis', 'capture_proof_sha256')


def validate_command_options(command, help_text):
    supported = set(re.findall(r'--[a-z][a-z0-9-]*', help_text))
    require(all(item in supported for item in command if item.startswith('--')), 'Generated command has an unsupported launcher option')


def validate_repeat(original, repeated, original_model_hashes, repeated_model_hashes, prior_seeds):
    for key in UNCHANGED: require(repeated[key] == original[key], f'Repeat changed frozen {key}')
    require(repeated_model_hashes == original_model_hashes, 'Repeat model bytes changed')
    require(repeated['prefix_counts'] == PREFIXES, 'Repeat fixed prefixes changed')
    require([a['name'] for a in repeated['arms']] == [a['name'] for a in original['arms']] == ['narrow', 'broad'], 'Wrong repeat arms')
    all_seeds = []
    for old, new in zip(original['arms'], repeated['arms']):
        for key in ('name', 'geometric_latent_SD_A', 'components'): require(new[key] == old[key], f'Repeat altered atlas {key}')
        require(new['populations'] == POPULATIONS and new['samples_per_population'] == SAMPLES, 'Repeat allocation differs')
        require(new['seeds'] == [SEEDS[new['name']]+1009*i for i in range(POPULATIONS)], 'Repeat streams differ')
        require(new['output'] != old['output'], 'Cannot reuse original output')
        all_seeds.extend(new['seeds'])
    require(len(set(all_seeds)) == len(all_seeds) and not set(all_seeds).intersection(prior_seeds), 'Repeat reuses a known stream')


def prepare(out):
    out = Path(out).resolve(); require(not out.exists(), 'Use a fresh repeat preparation directory')
    source = read(ORIGINAL/'protocol.json'); seal = read(ORIGINAL/'freeze.json'); prior = read(BASELINE)
    require(source['schema'] == 'complete-far-calibrated-contact-atlas-v1' and prior['complete'], 'Wrong or incomplete baseline stage')
    require(source['q_window'] == WINDOW and source['hybrid']['cover_scales'] == [1.] and source['proposal_anchor_index'] == 0, 'Require the frozen complete far/A-frame law')
    require(source['lambda_ratio'] == 64. and source['cloud_replicates'] == 2, 'Frozen launcher fixes the64/two-cloud law')
    require(sha(ORIGINAL/'protocol.json') == seal['protocol_sha256'] and sha(ORIGINAL/'config.json') == seal['config_sha256'], 'Original protocol/config changed')
    require(prior['source_sha256'][str(ORIGINAL/'protocol.json')] == seal['protocol_sha256'] and prior['original_q_window'] == source['q_window'], 'Baseline audit not tied to original protocol')
    for name, digest in seal['archived_sha256'].items(): require(sha(ORIGINAL/'provenance'/name) == digest, 'Original archive changed')
    for name, digest in prior['archived_sha256'].items(): require(sha(BASELINE.parent/'provenance'/name) == digest, 'Baseline audit source changed')
    for path, digest in prior['source_sha256'].items(): require(sha(Path(path)) == digest, 'Baseline audited source changed')
    for name, digest in seal['model_sha256'].items(): require(sha(ORIGINAL/f'model-{name}.json') == digest, 'Original model changed')
    require(sha(Path(source['executable'])) == source['executable_sha256'] and sha(ORIGINAL/'capture-proof.json') == source['capture_proof_sha256'], 'Physical executable/capture proof changed')
    prior_seeds = {seed for a in source['arms'] for seed in a['seeds']}
    reference = read(Path(source['analysis']['local_reference_path']))
    require(sha(Path(source['analysis']['local_reference_path'])) == source['analysis']['local_reference_sha256'], 'Local reference changed')
    prior_seeds.update(p['seed'] for c in reference['campaigns'] for p in c['populations'])
    centers = read(Path(source['center_selection']['path'])); require(sha(Path(source['center_selection']['path'])) == source['center_selection']['sha256'], 'Center selection changed')
    prior_seeds.update(c['seed'] for c in centers['candidates'])
    for old, campaign in zip(source['arms'], prior['campaigns']):
        require(campaign['arm'] == old['name'] and campaign['root'] == old['output'], 'Baseline arm mismatch')
        require([p['seed'] for p in campaign['populations']] == old['seeds'] and all(p['samples'] == old['samples_per_population'] for p in campaign['populations']), 'Baseline allocation mismatch')
        require(campaign['independently_audited_rows'] == old['populations']*old['samples_per_population'], 'Incomplete baseline density audit')
    archive = out/'provenance'; archive.mkdir(parents=True)
    # Keep original dependency bytes, especially the archived density auditor.
    for name in seal['archived_sha256']: shutil.copy2(ORIGINAL/'provenance'/name, archive/name)
    inputs = {'original-protocol.json': ORIGINAL/'protocol.json', 'original-freeze.json': ORIGINAL/'freeze.json',
        'original-report.json': ORIGINAL/'report.json', 'baseline-analysis.json': BASELINE,
        'prepare_far_atlas_repeat.py': Path(__file__), 'repeat-runner.py': ROOT/'tools/run_native_region_reference.py'}
    for name, path in inputs.items(): shutil.copy2(path, archive/name)
    shutil.copy2(ORIGINAL/'config.json', out/'config.json')
    shutil.copy2(ORIGINAL/'capture-proof.json', out/'capture-proof.json')
    for name in SEEDS: shutil.copy2(ORIGINAL/f'model-{name}.json', out/f'model-{name}.json')
    protocol = copy.deepcopy(source); protocol['created_utc'] = datetime.now(timezone.utc).isoformat()
    protocol['repeat_source'] = dict(path=str(ORIGINAL/'protocol.json'), sha256=seal['protocol_sha256'],
        freeze_path=str(ORIGINAL/'freeze.json'), freeze_sha256=sha(ORIGINAL/'freeze.json'),
        completed_baseline_analysis_path=str(BASELINE), completed_baseline_analysis_sha256=sha(BASELINE),
        model_sha256=seal['model_sha256'], rule='Only independent production streams, fixed population/draw counts and prefixes change. Model bytes and embedded original provenance are untouched.')
    protocol['prefix_counts'] = PREFIXES
    protocol['prefix_rule'] = 'First16384,32768,65536 unconditional rows per population; all same-stream covariance retained. No optional stop, refit or population replacement.'
    protocol['selection_scope'] = 'Larger fixed budget chosen after the completed baseline. Original models and local reference definitions remain unchanged. Separate baseline/repeat estimates; no pooling or new physical/geometry probes.'
    commands = {}
    for arm in protocol['arms']:
        name = arm['name']; arm.update(populations=POPULATIONS, samples_per_population=SAMPLES,
            seeds=[SEEDS[name]+1009*i for i in range(POPULATIONS)],
            output=str(ROOT/f'runs/ab-far-contact-atlas-{name}-16x65536-l64-20260921'),
            model_source_path=str(ORIGINAL/f'model-{name}.json'), workers=16)
        require(not Path(arm['output']).exists(), 'Repeat production directory exists')
        # Original model source path preserves interpretation of embedded
        # proposal_provenance.protocol_sha256; the repeat also holds exact copies.
        commands[name] = [sys.executable, str(archive/'repeat-runner.py'), '--root', arm['output'], '--config', str(out/'config.json'),
            '--binary', source['executable'], '--expected-binary-sha256', source['executable_sha256'],
            '--replicates', str(POPULATIONS), '--samples', str(SAMPLES), '--workers', '16', '--seed-base', str(SEEDS[name]),
            '--q-min', '5', '--q-max', '37', '--q-upper-open', '--cover-scales', '1',
            '--model', arm['model_source_path'], '--model-weight', str(source['hybrid']['model_weight']),
            '--model-uniform-probability', str(source['hybrid']['uniform_probability']), '--model-anchor-index', '0']
    help_text = subprocess.run([sys.executable, str(archive/'repeat-runner.py'), '--help'], check=True, capture_output=True, text=True).stdout
    for command in commands.values(): validate_command_options(command, help_text)
    (archive/'runner-help.txt').write_text(help_text)
    write(out/'commands.json', commands); protocol['commands_sha256'] = sha(out/'commands.json')
    protocol['input_sha256'] = {**source['input_sha256'], **{str(p): sha(p) for p in inputs.values()}}
    protocol['archived_sha256'] = {p.name: sha(p) for p in archive.iterdir()}
    model_hashes = {name: sha(out/f'model-{name}.json') for name in SEEDS}
    validate_repeat(source, protocol, seal['model_sha256'], model_hashes, prior_seeds)
    write(out/'protocol.json', protocol)
    frozen = dict(protocol_sha256=sha(out/'protocol.json'), config_sha256=sha(out/'config.json'), model_sha256=model_hashes,
        commands_sha256=sha(out/'commands.json'), archived_sha256=protocol['archived_sha256'])
    require(frozen['config_sha256'] == seal['config_sha256'], 'Exact original config bytes changed')
    write(out/'freeze.json', frozen)
    write(out/'report.json', dict(complete=True, prepared_only=True, original_protocol_sha256=seal['protocol_sha256'],
        original_baseline_analysis_sha256=sha(BASELINE), repeat_protocol_sha256=frozen['protocol_sha256'],
        identical_model_sha256=model_hashes, identical_config_sha256=frozen['config_sha256'],
        commands_sha256=frozen['commands_sha256'], freeze_sha256=sha(out/'freeze.json'),
        independent_new_seeds=[seed for arm in protocol['arms'] for seed in arm['seeds']],
        total_unconditional_draws=2*POPULATIONS*SAMPLES, maximum_concurrent_workers=32,
        no_new_geometry_physical_draws_or_fits=True,
        scope='Prepared commands only. Complete far and outside2 remain explicit. Original model metadata is byte-identical; no new training or probe results are implied.'))
    print(dict(complete=True, prepared_only=True, protocol=str(out/'protocol.json'), protocol_sha256=frozen['protocol_sha256'], commands=str(out/'commands.json')))
    return protocol


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--out', type=Path, required=True); prepare(parser.parse_args().out)
