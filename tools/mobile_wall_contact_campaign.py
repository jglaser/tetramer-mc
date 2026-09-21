#!/usr/bin/env python3
"""Two frozen proposal laws for the same complete atomic-wall contact integral."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from mobile_native_pocket_campaign import (
    CONFIG_SHA, DEFINITION, DEFINITION_SHA, MODEL, MODEL_SHA, REFERENCE,
    ROOT, SHAPE_SHA, exclusive, execute_batches, read, require, sha, write,
)

SCHEMA = 'mobile-wall-contact-controller-v1'
WALL = dict(center=[0., 0., 0.], radius=223.32617672378387)
CAPTURE_RADIUS = 273.
BINARY = ROOT/'target/wall-normalizer-review/release/basin-normalizer'
BUNDLE = ROOT/'target/wall-normalizer-review/release/build/tetramer-mc-356a8e0b913327fe/out/source-bundle.json'
BINARY_SHA = '54f5b63b866264d32d75a23b21fa09fa385e92c6e959fcf48cd70d2282b82d94'
BUNDLE_SHA = '2adfcc4ec6ad884c68d2c25bf595906b480bc550810f8a1596a943fa797e96e6'
COMPARISON = ROOT/'runs/mobile-native-pocket-coverage-comparison-20260921/analysis.json'
COMPARISON_SHA = 'e252600e2498b1fc437406b417d7b1053089833c10a2cd23202a6d5cbbcc0aa1'
SAMPLES, REPLICATES, SEED_BASE = 16384, 4, 126101010
ARMS = ('original', 'coverage')


def physical_config(original, shape):
    require('target_region' not in original, 'Unexpected regional target')
    require(original['capture_center'] == WALL['center'], 'Capture center differs')
    require(original['metadata']['physical_sphere_radius_A'] == WALL['radius'], 'Physical wall differs')
    require(original['depletant_radius'] == 1.5 and original['reservoir_density'] == .035,
            'Physical bath differs')
    require(original['poisson_lambda_ratio'] == 64., 'Auxiliary intensity differs')
    result = copy.deepcopy(original)
    result.update(shape=str(shape), capture_radius=CAPTURE_RADIUS)
    return result


def model_counts(path):
    from prepare_smc_normalizer_atlas import unwrap_proposal_model
    base, flags = unwrap_proposal_model(read(path))
    require(any(flags) and base['shape_sha256'] == SHAPE_SHA, 'Expected active reciprocal shape model')
    require(len(base['weights']) == len(flags), 'Model labels differ')
    return dict(base_component_count=len(flags), virtual_component_count=len(flags)+sum(flags),
                reciprocal_components=flags)


def command(folder, job):
    archive = folder/'provenance'
    return [str(archive/'basin-normalizer'), '--config', str(archive/'config.json'),
            '--model', str(archive/'model.json'), '--out', job['directory'],
            '--samples', str(job['samples']), '--seed', str(job['seed']),
            '--cloud-replicates', '2', '--covariance-scale', '1',
            '--uniform-probability', '0.1', '--wall-radius', str(WALL['radius']),
            '--wall-center', '0', '0', '0']


def freeze(out, coverage):
    from prepare_shoulder_docking_benchmark import local_dependencies
    out, coverage = Path(out).resolve(), Path(coverage).resolve()
    require(not out.exists(), 'Fresh campaign required')
    require(sha(REFERENCE/'config.json') == CONFIG_SHA and sha(MODEL) == MODEL_SHA,
            'Physical source or original model changed')
    require(sha(DEFINITION) == DEFINITION_SHA and sha(COMPARISON) == COMPARISON_SHA,
            'Frozen classifier or reference comparison changed')
    require(sha(BINARY) == BINARY_SHA and sha(BUNDLE) == BUNDLE_SHA, 'Reviewed executable changed')
    require(BUNDLE.read_bytes() in BINARY.read_bytes(), 'Embedded source bundle differs')
    for name, entry in read(BUNDLE)['files'].items():
        require(sha(ROOT/name) == entry['sha256'] and (ROOT/name).read_text() == entry['text'],
                'Current Rust source differs: '+name)
    for name, digest in read(DEFINITION)['input_sha256'].items():
        require(sha(DEFINITION.parent/'inputs'/name) == digest, 'Classifier input changed')
    original = read(REFERENCE/'config.json'); shape = Path(original['shape'])
    require(sha(shape) == SHAPE_SHA, 'Shape changed')
    preparation = read(coverage.parent/'manifest.json')
    require(preparation['schema'] == 'mobile-wall-proposal-preparation-v1' and preparation['complete']
            and not preparation['physical_simulations_launched'], 'Coverage preparation incomplete')
    for name, digest in read(coverage.parent/'freeze.json').items():
        require(sha(coverage.parent/name) == digest, 'Coverage preparation changed: '+name)
    require(preparation['arms']['coverage']['sha256'] == sha(coverage)
            and preparation['arms']['original']['sha256'] == MODEL_SHA, 'Coverage proposal identity differs')
    require(preparation['physical_config_sha256'] == CONFIG_SHA, 'Coverage scaffold differs')
    require(preparation['requested_runtime']['atomic_wall'] == WALL
            and preparation['requested_runtime']['capture_radius'] == CAPTURE_RADIUS, 'Coverage support differs')
    counts = {arm: model_counts(path) for arm, path in zip(ARMS, (MODEL, coverage))}
    require(counts['original']['base_component_count'] == 150 and
            counts['original']['virtual_component_count'] == 300, 'Original atlas differs')
    require(counts['coverage']['reciprocal_components'][:150] == [True]*150 and
            not any(counts['coverage']['reciprocal_components'][150:]), 'Coverage reciprocal flags differ')
    require(counts['coverage']['base_component_count'] > 150, 'Coverage model adds no guides')
    sources = local_dependencies([Path(__file__), ROOT/'tools/analyze_basin_normalizers.py',
                                 ROOT/'tools/test_mobile_wall_contact_campaign.py'])
    top = out/'provenance'; top.mkdir(parents=True)
    for name, path in sources.items(): shutil.copy2(path, top/name)
    shutil.copytree(DEFINITION.parent, top/'native-region')
    shutil.copytree(coverage.parent, top/'coverage-preparation')
    for name, path in {'source-config.json': REFERENCE/'config.json',
                       'finite-reference.json': COMPARISON,
                       'wall-validation.json': ROOT/'runs/wall-normalizer-validation-20260921/validation.json',
                       'cross-language-validation.json': ROOT/'runs/wall-normalizer-cross-language-20260921/validation.json'}.items():
        shutil.copy2(path, top/name)
    (top/'regions').mkdir()
    shutil.copy2(REFERENCE/'region-native-r4.json', top/'regions/region-native-r4.json')
    r5 = read(ROOT/'runs/mobile-native-pocket-coverage-campaign-20260921/r5repeat/provenance/region.json')
    write(top/'regions/region-alternative-r5.json', r5)
    r32 = copy.deepcopy(r5); r32['mahalanobis_radius'] = 32.
    r32['definition'] = 'Full 0<=rho<=32 witness region in the same ideal motif7/body2 chart, original q>1, original capture170. Reporting only.'
    write(top/'regions/region-alternative-r32.json', r32)
    campaigns = []; index = 0
    for arm, model in zip(ARMS, (MODEL, coverage)):
        folder = out/arm; archive = folder/'provenance'; archive.mkdir(parents=True)
        for name, path in sources.items(): shutil.copy2(path, archive/name)
        for name, path in {'basin-normalizer': BINARY, 'source-bundle.json': BUNDLE,
                           'shape.json': shape, 'model.json': model}.items(): shutil.copy2(path, archive/name)
        cfg = physical_config(original, archive/'shape.json'); write(archive/'config.json', cfg)
        for name in ('runs', 'logs'): (folder/name).mkdir()
        jobs = []
        for replicate in range(REPLICATES):
            ident = f'r{replicate:02d}'
            job = dict(id=ident, seed=SEED_BASE+1009*index, samples=SAMPLES, covariance_std_scale=1.,
                       cloud_replicates=2, proposal_anchor_index=None, directory=str(folder/'runs'/ident),
                       log=str(folder/'logs'/f'{ident}.log'))
            index += 1; job['command'] = command(folder, job); jobs.append(job)
        physical = dict(depletant_radius=1.5, activity=.035, capture_center=cfg['capture_center'],
                        capture_radius=CAPTURE_RADIUS, fixed_poses=cfg['fixed_poses'], fixed_neighbor_count=2,
                        lambda_ratio=64., uniform_probability=.1, atomic_wall=copy.deepcopy(WALL), bath_wall_permeable=True)
        manifest = dict(schema=1, experiment_schema='mobile-wall-contact-arm-v1', arm=arm, frozen=True,
                        physical=physical, jobs=jobs, workers=8, model_components=counts[arm]['base_component_count'],
                        proposal_anchor_index=None, model_counts=counts[arm],
                        archive_sha256={p.relative_to(archive).as_posix(): sha(p) for p in archive.rglob('*') if p.is_file()})
        write(folder/'manifest.json', manifest)
        campaigns.append(dict(arm=arm, path=str(folder), manifest_sha256=sha(folder/'manifest.json'),
                              model_sha256=sha(archive/'model.json')))
    protocol = dict(schema=SCHEMA, campaigns=campaigns, native_definition='provenance/native-region/definition.json',
                    references=dict(comparison='provenance/finite-reference.json', comparison_sha256=COMPARISON_SHA,
                        native_r4='provenance/regions/region-native-r4.json', alternative_r5='provenance/regions/region-alternative-r5.json',
                        alternative_r32='provenance/regions/region-alternative-r32.json'),
                    controller_sha256=sha(__file__), physical_executable_sha256=BINARY_SHA, source_bundle_sha256=BUNDLE_SHA,
                    coverage_preparation_sha256=sha(coverage.parent/'manifest.json'),
                    original_config_sha256=CONFIG_SHA, physical_wall=WALL, capture_radius=CAPTURE_RADIUS,
                    populations_per_arm=REPLICATES, samples_per_population=SAMPLES, seed_base=SEED_BASE,
                    total_jobs=index, total_unconditional_draws=index*SAMPLES, maximum_physical_workers=8,
                    cloud_replicates=2, lambda_ratio=64., uniform_probability=.1,
                    stopping='Exactly the prespecified independent populations; no retries or extensions. Drain children on failure, audit each arm once.',
                    estimands='Same full physical wall and permeable ideal bath in both arms. All registered sites, unregistered contact, unbound; inside/outside D170 diagnostics. Keep proposal arms separate; invalid draws contribute zero.',
                    scope='Importance integration conditional on two fixed neighbors. Not mobile assembly or template-free discovery.')
    write(out/'protocol.json', protocol)
    write(out/'freeze.json', dict(files={p.relative_to(out).as_posix(): sha(p) for p in out.rglob('*') if p.is_file()}))
    validate(out)
    return protocol


def validate(out):
    out = Path(out).resolve(); p = read(out/'protocol.json')
    require(p['schema'] == SCHEMA and p['controller_sha256'] == sha(out/'provenance'/Path(__file__).name), 'Controller identity differs')
    for name, digest in read(out/'freeze.json')['files'].items():
        require(sha(out/name) == digest, 'Frozen input changed: '+name)
    require([c['arm'] for c in p['campaigns']] == list(ARMS), 'Proposal arms differ')
    original = read(out/'provenance/source-config.json'); require(sha(out/'provenance/source-config.json') == CONFIG_SHA, 'Physical source changed')
    seeds = []
    for c in p['campaigns']:
        folder = Path(c['path']); require(folder == out/c['arm'], 'Campaign moved')
        require(sha(folder/'manifest.json') == c['manifest_sha256'], 'Manifest changed')
        m = read(folder/'manifest.json'); cfg = read(folder/'provenance/config.json')
        require(cfg == physical_config(original, folder/'provenance/shape.json'), 'Physical config changed')
        require(m['physical']['atomic_wall'] == WALL and m['physical']['bath_wall_permeable'] is True, 'Wall or bath changed')
        require(m['model_counts'] == model_counts(folder/'provenance/model.json'), 'Model counts changed')
        require(sha(folder/'provenance/model.json') == c['model_sha256'], 'Model changed')
        require(len(m['jobs']) == REPLICATES and m['physical']['uniform_probability'] == .1, 'Allocation differs')
        for j in m['jobs']:
            require(j['samples'] == SAMPLES and j['command'] == command(folder, j), 'Physical command differs')
            require(j['proposal_anchor_index'] is None and j['covariance_std_scale'] == 1., 'Proposal law differs')
            require(Path(j['directory']) == folder/'runs'/j['id'], 'Output path differs')
            seeds.append(j['seed'])
    require(seeds == [SEED_BASE+1009*i for i in range(8)], 'Independent streams differ')
    require(p['maximum_physical_workers'] == p['total_jobs'] == 8, 'Worker allocation differs')
    return p


def verify_output(folder, manifest, job):
    d = Path(job['directory']); summary = read(d/'summary.json'); pm = read(d/'manifest.json')
    require(summary['complete'] and summary['manifest'] == pm and summary['numerical_nulls'] == 0, 'Incomplete or invalid numerical output')
    wanted = dict(schema=4, pose_proposal_schema=3, samples=job['samples'], seed=job['seed'], cloud_replicates=2,
                  activity=.035, uniform_probability=.1, covariance_scale=1., atomic_wall=WALL, bath_wall_permeable=True,
                  source_bundle_sha256=BUNDLE_SHA, executable_sha256=BINARY_SHA, shape_sha256=SHAPE_SHA,
                  config_sha256=manifest['archive_sha256']['config.json'], model_sha256=manifest['archive_sha256']['model.json'],
                  proposal_anchor_index=None, physical_fixed_neighbor_count=2, **manifest['model_counts'])
    require(all(pm.get(k) == v for k, v in wanted.items()), 'Physical output identity differs')
    require(summary['estimates']['total']['unconditional_draws'] == job['samples'], 'Invalid zeros excluded')
    require(summary['samples'] == job['samples'], 'Incomplete physical sample budget')
    return dict(samples_sha256=sha(d/'samples.jsonl'), manifest_sha256=sha(d/'manifest.json'),
                summary_sha256=sha(d/'summary.json'), sampler_cpu_seconds=summary['sampler_cpu_seconds'])


def run(out):
    out = Path(out).resolve(); p = validate(out)
    require(sha(__file__) == p['controller_sha256'], 'Run the archived controller')
    require(not (out/'status.json').exists(), 'No retry or overwrite')
    jobs = []; manifests = {}
    for c in p['campaigns']:
        folder = Path(c['path']); m = read(folder/'manifest.json'); manifests[c['arm']] = m
        require(not (folder/'assessment').exists() and not any((folder/'runs').iterdir()), 'Existing outputs')
        jobs.extend(dict(j, arm=c['arm'], status='pending') for j in m['jobs'])
    state = dict(schema='mobile-wall-contact-status-v1', complete=False, phase='physical',
                 protocol_sha256=sha(out/'protocol.json'), jobs=jobs, audits={}, started=time.time())
    exclusive(out/'status.json', state)
    def snapshot(): write(out/'status.json', state)
    try:
        execute_batches(jobs, snapshot); validate(out); state['phase'] = 'physical_validation'; snapshot()
        for j in jobs: j['output'] = verify_output(out/j['arm'], manifests[j['arm']], j)
        state['phase'] = 'audit'; snapshot()
        for c in p['campaigns']:
            folder = Path(c['path']); argv = [sys.executable, '-B', str(folder/'provenance/analyze_basin_normalizers.py'), '--root', str(folder)]
            a = dict(argv=argv, started=time.time()); state['audits'][c['arm']] = a; snapshot()
            with (out/(c['arm']+'-audit.log')).open('xb') as log:
                result = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT)
            a.update(returncode=result.returncode, finished=time.time()); snapshot(); result.check_returncode()
            assessment = read(folder/'assessment/analysis.json'); pops = assessment['populations']
            require(not assessment['pending'] and len(pops) == REPLICATES, 'Incomplete audit')
            require(all(v['proposal_audit']['checked_actual_poses'] == SAMPLES and
                        v['proposal_audit']['wall_domain']['checked_poses'] == SAMPLES for v in pops), 'Audit omitted poses')
            a['analysis_sha256'] = sha(folder/'assessment/analysis.json'); snapshot()
        state.update(complete=True, phase='complete', finished=time.time()); snapshot(); return state
    except BaseException as error:
        state.update(phase=state['phase']+'_failed', exception=repr(error), finished=time.time()); snapshot(); raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('freeze', 'validate', 'run'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--coverage-model', type=Path)
    args = parser.parse_args()
    if args.action == 'freeze' and args.coverage_model is None: parser.error('--coverage-model required to freeze')
    result = freeze(args.out, args.coverage_model) if args.action == 'freeze' else globals()[args.action](args.out)
    print(json.dumps(dict(action=args.action, complete=result.get('complete'), out=str(args.out))))
