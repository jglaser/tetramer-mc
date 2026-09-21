#!/usr/bin/env python3
"""Freeze and run matched four-mobile-body attachment and retention controls."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from mobile_native_pocket_campaign import execute_batches
from prepare_shoulder_docking_benchmark import local_dependencies
from run_mobile_posterior_pilot import (
    COORDINATE_ARCHIVE, exclusive_write, read, require, sha,
    validate_residue_coordinates, verify_bundle, write,
)

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'mobile-four-body-growth-campaign-v1'
CONTROLLER_SCHEMA = 'matched-mobile-four-body-growth-controller-v1'
DESIGN = ROOT/'runs/mobile-four-body-design-20260921'
DESIGN_POSES_SHA = '994d860f28d0fe1e94fefff5ee6b818d524210153d0cae6ebc4f492ecdc53dd2'
EXAMPLE = ROOT/'examples/spherical-reciprocal-free.json'
EXAMPLE_SHA = '2f8be1b8ab0c788049d82b4f7af4b424259b8d65d97a7b131f19411ddc4c72d9'
PREVIOUS = ROOT/'runs/mobile-reciprocal-atlas-benchmark-20260921/reciprocal'
COVERAGE = ROOT/'runs/mobile-wall-proposal-preparation-20260921'
MODELS = dict(original=ROOT/'examples/frozen-reciprocal-mixture.json', coverage=COVERAGE/'coverage-reciprocal.json')
MODEL_SHA = dict(original='dc9218c9706e1691af9336ef1c0e87a75a86994933dfd41cc6e1d1cd3dc52aa3',
                 coverage='feb4011c622c3104bbe909a29685bd7f077e28f0c847f630c69d8fa87939c20e')
BINARY = ROOT/'target/wall-normalizer-review/release/tetramer-mc'
BINARY_SHA = '1304fe2eadb2dddc7a89c5757c202eb6338569c9a5f3bbb6481d57feafa292ec'
BUNDLE = ROOT/'target/wall-normalizer-review/release/build/tetramer-mc-356a8e0b913327fe/out/source-bundle.json'
BUNDLE_SHA = '2adfcc4ec6ad884c68d2c25bf595906b480bc550810f8a1596a943fa797e96e6'
SHAPE_SHA = 'c7442034e7ffa4627b67c57a81117f0e9bc2d389ecf956a83dac2a5e915554c9'
ANALYZER = ROOT/'tools/analyze_mobile_posterior_pilot.py'
STARTS = dict(triangle_free='triangle_plus_free', retained_motif8='bound_retention')
ARMS = ('original', 'coverage')
SWEEPS, BURN, SEED_BASE = 2000, 400, 127101010
RADIUS = 223.32617672378387
SCOPE = ('Four mobile rigid tetramers, same original protein wall and unbounded permeable ideal bath, rd1.5Å/z.035Å^-3. '
    'Native-informed positive control of growth beyond the confirmed triangle. Two nonequilibrium starts, two independent streams '
    'per proposal/start; no fitting, repairs, fixed bodies, optional extension or physical-kinetics/equilibrium claim.')


def configured(example, starts, folder, arm, start, replicate, index):
    cfg = copy.deepcopy(example); archive = folder/'provenance'
    cfg.update(shape=str(archive/'tetramer-shape.json'), monomer_shape=str(archive/'monomer-shape.json'),
               initial_poses=copy.deepcopy(starts[STARTS[start]]), box_lengths=[2*RADIUS]*3,
               boundary=dict(kind='spherical', radius=RADIUS), seed=SEED_BASE+1009*index,
               sweeps=SWEEPS, sample_every=1, fixed_body_indices=[], seed_labels=[])
    cfg['metadata'] = dict(start=start, replicate=replicate, proposal_mode='frozen-posterior-c09',
        atlas_variant=arm, native_pair_motifs=str(archive/'native-pair-motifs.json'),
        native_informed_proposal=True, model_sha256=MODEL_SHA[arm],
        body_labels=starts['body_labels'], initial_geometry_sha256=DESIGN_POSES_SHA,
        initial_fragments=[[0, 1, 2], [3]] if start == 'triangle_free' else [[0, 1, 2, 3]],
        preparation_equilibrated=False, training_feedback=False, scope=SCOPE)
    require(len(cfg['initial_poses']) == 4 and cfg['initial_poses'][:3] == starts['triangle_plus_free'][:3], 'Triangle changed')
    require(cfg['frozen_posterior'] == dict(probability=.5, correlation=.9), 'Transport control changed')
    return cfg


def command(folder, job):
    return [str(folder/'provenance/tetramer-mc'), 'run', '--config', job['config'],
            '--model', str(folder/'provenance/model.json'), '--method', 'learned', '--out', job['directory'],
            '--sweeps', str(SWEEPS), '--sample-every', '1', '--no-gsd']


def freeze(out):
    out = Path(out).resolve(); require(not out.exists(), 'Fresh campaign required')
    require(sha(DESIGN/'recommended-poses.json') == DESIGN_POSES_SHA and sha(EXAMPLE) == EXAMPLE_SHA, 'Source preparation changed')
    for name, digest in read(DESIGN/'freeze.json').items(): require(sha(DESIGN/name) == digest, 'Geometry design changed')
    for name, digest in read(COVERAGE/'freeze.json').items(): require(sha(COVERAGE/name) == digest, 'Coverage preparation changed')
    for arm, path in MODELS.items(): require(sha(path) == MODEL_SHA[arm], 'Proposal model changed')
    require(sha(BINARY) == BINARY_SHA and sha(BUNDLE) == BUNDLE_SHA, 'Reviewed executable changed')
    bundle, rust_sources = verify_bundle(BINARY, BUNDLE, ROOT)
    previous = read(PREVIOUS/'manifest.json')
    names = [name for name in previous['input_sha256'] if name.startswith('reference/')]
    names += ['tetramer-shape.json', 'monomer-shape.json', 'native-pair-motifs.json']
    inputs = {name: PREVIOUS/'provenance'/name for name in names}
    for name, path in inputs.items(): require(sha(path) == previous['input_sha256'][name], 'Observer/shape source changed')
    require(sha(inputs['tetramer-shape.json']) == SHAPE_SHA and len(names) == 11, 'Unexpected physical/reference closure')
    residue = validate_residue_coordinates(inputs['monomer-shape.json'], inputs[COORDINATE_ARCHIVE])
    sources = local_dependencies([Path(__file__), ANALYZER, ROOT/'tools/test_mobile_four_body_growth_campaign.py',
                                  ROOT/'tools/test_mobile_four_body_observer.py'])
    top = out/'provenance'; top.mkdir(parents=True)
    for name, path in sources.items(): shutil.copy2(path, top/name)
    shutil.copytree(DESIGN, top/'geometry-design')
    shutil.copy2(EXAMPLE, top/'source-example.json')
    shutil.copy2(COVERAGE/'manifest.json', top/'coverage-manifest.json')
    starts = read(DESIGN/'recommended-poses.json'); example = read(EXAMPLE)
    campaigns = []; index = 0
    for arm in ARMS:
        folder = out/arm; archive = folder/'provenance'; archive.mkdir(parents=True)
        arm_sources = dict(inputs, **sources)
        arm_sources.update({'tetramer-mc': BINARY, 'source-bundle.json': BUNDLE, 'model.json': MODELS[arm]})
        for name, path in arm_sources.items():
            target = archive/name; target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, target)
        for name, entry in bundle['files'].items():
            target = archive/'source'/name; target.parent.mkdir(parents=True, exist_ok=True); target.write_text(entry['text'])
        for name in ('configs', 'runs', 'logs'): (folder/name).mkdir()
        jobs = []
        for start in STARTS:
            for replicate in (0, 1):
                ident = f'{start}-r{replicate:02d}-c09'; path = folder/'configs'/f'{ident}.json'
                cfg = configured(example, starts, folder, arm, start, replicate, index); write(path, cfg)
                job = dict(id=ident, index=index, start=start, replicate=replicate, mode='c09', seed=cfg['seed'],
                           config=str(path), config_sha256=sha(path), directory=str(folder/'runs'/ident), log=str(folder/'logs'/f'{ident}.log'))
                job['command'] = command(folder, job); jobs.append(job); index += 1
        hashes = {p.relative_to(archive).as_posix(): sha(p) for p in archive.rglob('*') if p.is_file()}
        manifest = dict(schema=SCHEMA, atlas_variant=arm, frozen=True, binary_supplied=True,
            binary_sha256=BINARY_SHA, source_bundle_sha256=BUNDLE_SHA, rust_sources=rust_sources,
            sweeps=SWEEPS, burn_sweeps=BURN, sample_every=1, workers=4, bodies=4, body_count=4,
            tracked_body_index=3, scaffold_body_indices=[0, 1, 2], jobs=jobs, seeds=[j['seed'] for j in jobs],
            total_attempts=4*6*SWEEPS, total_single_body_attempts=4*4*SWEEPS, total_collective_attempts=4*2*SWEEPS,
            model=str(archive/'model.json'), model_sha256=MODEL_SHA[arm], shape_sha256=SHAPE_SHA,
            observer_sha256=sha(ANALYZER), reference=str(archive/'reference'), residue_coordinate_validation=residue,
            input_sha256=hashes, scope=SCOPE)
        write(folder/'manifest.json', manifest)
        campaigns.append(dict(arm=arm, path=str(folder), manifest_sha256=sha(folder/'manifest.json')))
    protocol = dict(schema=CONTROLLER_SCHEMA, campaigns=campaigns, controller_sha256=sha(__file__),
        physical_executable_sha256=BINARY_SHA, source_bundle_sha256=BUNDLE_SHA, observer_sha256=sha(ANALYZER),
        total_jobs=8, maximum_physical_workers=8, sweeps=SWEEPS, burn_sweeps=BURN, seed_base=SEED_BASE,
        total_attempts=8*6*SWEEPS, body_count=4, tracked_body_index=3, scaffold_body_indices=[0, 1, 2],
        first_endpoint='D joins a native-connected four-body component; a complete clique is not required.',
        paired_endpoint='Retention and loss/return of native/exclusion contacts; ABC survival, competing attachments, initial-state dependence, all kernels and failed runs retained.',
        stopping='Exactly eight declared runs, 2000 sweeps each. Drain launched children on failure; no retry, extension or observer after physical failure.', scope=SCOPE)
    write(out/'protocol.json', protocol)
    write(out/'freeze.json', dict(files={p.relative_to(out).as_posix(): sha(p) for p in out.rglob('*') if p.is_file()}))
    validate(out); return protocol


def validate(out):
    out = Path(out).resolve(); p = read(out/'protocol.json')
    require(p['schema'] == CONTROLLER_SCHEMA and p['controller_sha256'] == sha(out/'provenance'/Path(__file__).name), 'Controller identity differs')
    for name, digest in read(out/'freeze.json')['files'].items(): require(sha(out/name) == digest, 'Frozen input changed: '+name)
    require([c['arm'] for c in p['campaigns']] == list(ARMS) and p['total_jobs'] == p['maximum_physical_workers'] == 8, 'Allocation differs')
    starts = read(out/'provenance/geometry-design/recommended-poses.json'); example = read(out/'provenance/source-example.json')
    require(sha(out/'provenance/source-example.json') == EXAMPLE_SHA and sha(out/'provenance/geometry-design/recommended-poses.json') == DESIGN_POSES_SHA, 'Physical preparation differs')
    index = 0
    for c in p['campaigns']:
        folder = out/c['arm']; require(c['path'] == str(folder) and sha(folder/'manifest.json') == c['manifest_sha256'], 'Arm binding differs')
        m = read(folder/'manifest.json'); require(m['schema'] == SCHEMA and m['atlas_variant'] == c['arm'] and m['body_count'] == 4, 'Observer domain differs')
        require((m['sweeps'], m['burn_sweeps'], m['sample_every']) == (SWEEPS, BURN, 1), 'Run allocation differs')
        require(m['model_sha256'] == MODEL_SHA[c['arm']] and len(m['jobs']) == 4, 'Proposal/allocation differs')
        require(m['binary_sha256'] == BINARY_SHA and m['source_bundle_sha256'] == BUNDLE_SHA, 'Physical code differs')
        for j, (start, replicate) in zip(m['jobs'], [(s, r) for s in STARTS for r in (0, 1)]):
            require((j['start'], j['replicate'], j['mode'], j['index'], j['seed']) == (start, replicate, 'c09', index, SEED_BASE+1009*index), 'Independent stream allocation differs')
            cfg = configured(example, starts, folder, c['arm'], start, replicate, index)
            require(read(j['config']) == cfg and sha(j['config']) == j['config_sha256'], 'Configuration differs')
            require(j['directory'] == str(folder/'runs'/j['id']) and j['command'] == command(folder, j), 'Output or physical command differs')
            index += 1
    return p


def verify_output(manifest, job):
    d = Path(job['directory']); summary = read(d/'summary.json'); provenance = read(d/'manifest.json')
    require(summary['complete'] and summary['completed_sweeps'] == SWEEPS, 'Incomplete physical run')
    for key, expected in [('config_sha256', job['config_sha256']), ('model_sha256', manifest['model_sha256']),
                          ('shape_sha256', SHAPE_SHA), ('executable_sha256', BINARY_SHA), ('source_bundle_sha256', BUNDLE_SHA)]:
        require(provenance[key] == expected, 'Physical output identity differs: '+key)
    require(len(read(d/'checkpoint.json')['poses']) == 4, 'Physical body count differs')
    return dict(files={name: sha(d/name) for name in ('summary.json', 'manifest.json', 'config.json', 'checkpoint.json', 'moves.jsonl', 'trajectory.jsonl')},
                sampler_cpu_seconds=summary['sampler_cpu_seconds'])


def run(out):
    out = Path(out).resolve(); p = validate(out)
    require(sha(__file__) == p['controller_sha256'], 'Use the exact archived controller')
    require(not (out/'status.json').exists(), 'No retry or overwrite')
    jobs = []; manifests = {}
    for c in p['campaigns']:
        folder = out/c['arm']; m = read(folder/'manifest.json'); manifests[c['arm']] = m
        require(not (folder/'status.json').exists() and not (folder/'assessment').exists() and not any((folder/'runs').iterdir()), 'Existing output')
        jobs.extend(dict(j, arm=c['arm'], status='pending') for j in m['jobs'])
    state = dict(schema='mobile-four-body-growth-status-v1', complete=False, phase='physical', protocol_sha256=sha(out/'protocol.json'), jobs=jobs, audits={}, started=time.time())
    exclusive_write(out/'status.json', state)
    def snapshot():
        write(out/'status.json', state)
        for arm in ARMS:
            selected = [dict(j, exit_code=j.get('returncode')) for j in jobs if j['arm'] == arm]
            write(out/arm/'status.json', dict(running=any(j['status'] == 'running' for j in selected),
                complete=all(j['status'] == 'complete' and j['exit_code'] == 0 for j in selected), jobs=selected))
    snapshot()
    try:
        execute_batches(jobs, snapshot); validate(out); state['phase'] = 'physical_validation'; snapshot()
        for j in jobs: j['output'] = verify_output(manifests[j['arm']], j)
        state['phase'] = 'audit'; snapshot()
        for c in p['campaigns']:
            folder = out/c['arm']; argv = [sys.executable, '-B', str(folder/'provenance/analyze_mobile_posterior_pilot.py'), '--campaign', str(folder), '--out', str(folder/'assessment'), '--workers', '4']
            a = dict(argv=argv, started=time.time()); state['audits'][c['arm']] = a; snapshot()
            with (out/(c['arm']+'-audit.log')).open('xb') as log: result = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT)
            a.update(returncode=result.returncode, finished=time.time()); snapshot(); result.check_returncode()
            assessment = read(folder/'assessment/analysis.json')
            require(assessment['complete'] and len(assessment['runs']) == 4 and all(r['passed'] for r in assessment['runs']), 'Incomplete observer')
            a['analysis_sha256'] = sha(folder/'assessment/analysis.json'); snapshot()
        state.update(complete=True, phase='complete', finished=time.time()); snapshot(); return state
    except BaseException as error:
        state.update(phase=state['phase']+'_failed', exception=repr(error), finished=time.time()); snapshot(); raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('freeze', 'validate', 'run')); parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); result = globals()[args.action](args.out)
    print(json.dumps(dict(action=args.action, complete=result.get('complete'), out=str(args.out))))
