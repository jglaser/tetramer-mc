#!/usr/bin/env python3
"""Freeze an explicit mobile-pilot executable, preflight, or run exactly once.

Freeze and preflight never execute a binary. Run drains every launched child,
preserves failures, and invokes only the archived observer after all 12 succeed.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from prepare_mobile_posterior_pilot import (
    ROOT, BURN, MODES, SEED_BASE, SEED_STRIDE, STARTS, SWEEPS, WORKERS, validate_configs,
)
from prepare_shoulder_docking_benchmark import local_dependencies
import numpy as np

PREPARATION = ROOT/'runs/mobile-posterior-pilot-preparation-20260921'
PREPARATION_SHA256 = 'e7c53e81d479a2adc72c92fca696c52d8b848de369b8a3d8ea5fcec8ccf3ee08'
ANALYZER = ROOT/'tools/analyze_mobile_posterior_pilot.py'
SCHEMA = 'mobile-frozen-posterior-pilot-campaign-v1'
AUTHORITATIVE_COORDINATES = Path('/home/xvg/protein-nucleation/results/native-geometry-repair/rebuilt-hydrogens/heavy-coordinates.json')
COORDINATE_ARCHIVE = 'reference/results/native-geometry-repair/rebuilt-hydrogens/heavy-coordinates.json'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def exclusive_write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def inside(path, parent):
    return path == parent or parent in path.parents


def safe_relative(name):
    path = Path(name)
    require(not path.is_absolute() and '..' not in path.parts and str(path) == name and name not in ('', '.'),
            'Unsafe archived path: '+name)
    return path


def validate_residue_coordinates(shape_path, coordinate_path):
    """Check the native observer's complete data contract without replaying MC."""
    shape_path, coordinate_path = Path(shape_path), Path(coordinate_path)
    shape = read(shape_path)
    wanted = shape.get('source_coordinates_sha256')
    require(isinstance(wanted, str) and len(wanted) == 64, 'Monomer shape lacks an authoritative coordinate-record hash')
    require(coordinate_path.is_file() and sha(coordinate_path) == wanted,
            'Authoritative coordinate records unavailable or hash mismatch')
    coordinates = read(coordinate_path)
    positions = np.asarray(coordinates['positions'], dtype=float)
    records = coordinates['atom_records']
    centers = np.asarray([atom['center'] for atom in shape['atoms']], dtype=float)
    require(len(centers) > 0 and centers.shape == positions.shape == (len(records), 3)
            and np.isfinite(centers).all() and np.isfinite(positions).all(),
            'Authoritative residue-label record count/dimensions differ from shape')
    discrepancy = float(np.max(np.abs(positions-positions.mean(axis=0)-centers)))
    require(discrepancy <= 1e-10, 'Authoritative coordinate rows do not match shape atom ordering')
    require(all(isinstance(record, str) and len(record) >= 27 and record[22:27].strip()
                and record[17:20].strip() for record in records), 'Invalid authoritative atom residue records')
    keys = [(line[21], line[22:27].strip(), line[17:20].strip()) for line in records]
    residues = list(dict.fromkeys(keys))
    lookup = {key: index for index, key in enumerate(residues)}
    return dict(shape_sha256=sha(shape_path), source_sha256=wanted, atoms=len(records), residues=len(residues),
                maximum_coordinate_discrepancy_angstrom=discrepancy,
                atom_to_residue=[lookup[key] for key in keys],
                validation='Hash-bound full-precision coordinates recentered in original row order match every monomer shape atom; native labels use the same authoritative PDB atom records.')


def locate_residue_coordinates(shape_path, original_shape_paths=(), fallback=None):
    """Resolve the observer's sibling/fallback dependency before freezing it."""
    shape_path = Path(shape_path)
    wanted = read(shape_path).get('source_coordinates_sha256')
    require(isinstance(wanted, str) and len(wanted) == 64, 'Monomer shape lacks an authoritative coordinate-record hash')
    candidates = [shape_path.parent/'heavy-coordinates.json']
    candidates.extend(Path(path).parent/'heavy-coordinates.json' for path in original_shape_paths if path)
    candidates.append(Path(fallback) if fallback is not None else AUTHORITATIVE_COORDINATES)
    for candidate in dict.fromkeys(candidates):
        if candidate.is_file() and sha(candidate) == wanted:
            return candidate, validate_residue_coordinates(shape_path, candidate)
    raise ValueError('Authoritative shape-matched heavy-coordinate records unavailable or hash mismatch; cannot freeze native observer data closure')


def verify_bundle(binary, bundle_path, source_root):
    """Verify exact include_str! bytes, each hash/text, and the whole source tree."""
    binary, bundle_path, source_root = map(Path, (binary, bundle_path, source_root))
    require(binary.is_file() and os.access(binary, os.X_OK), 'Explicit executable binary required')
    raw = bundle_path.read_bytes()
    bundle = json.loads(raw)
    require(bundle['schema'] == 1 and bundle['files'], 'Unsupported or empty source bundle')
    expected = {'Cargo.toml', 'Cargo.lock', 'build.rs', 'vendor/README.md'}
    expected.update(str(path.relative_to(source_root)) for path in (source_root/'src').rglob('*.rs'))
    require(any(name.startswith('src/') for name in expected), 'Missing reviewed Rust source tree')
    require(set(bundle['files']) == expected, 'Embedded source membership differs from reviewed source tree')
    require(raw in binary.read_bytes(), 'Exact supplied source-bundle bytes are not embedded in this executable')
    hashes = {}
    for name, entry in bundle['files'].items():
        path = source_root/safe_relative(name)
        text_bytes = entry['text'].encode('utf-8')
        digest = hashlib.sha256(text_bytes).hexdigest()
        require(entry['sha256'] == digest, 'Source bundle text/hash mismatch: '+name)
        require(path.read_bytes() == text_bytes and sha(path) == digest, 'Reviewed source differs from embedded executable: '+name)
        hashes[name] = digest
    return bundle, hashes


def verify_preparation(preparation, expected_sha256):
    require(sha(preparation/'plan.json') == expected_sha256, 'Frozen preparation plan digest differs')
    plan = read(preparation/'plan.json')
    require(plan['preparation_only'] is True and plan['launched'] is False and plan['binary_supplied'] is False,
            'Require the inert binary-pending preparation')
    require(read(preparation/'manifest.json') == plan, 'Preparation plan/manifest disagree')
    frozen = read(preparation/'freeze.json')
    for name, digest in frozen['files'].items():
        require(sha(preparation/safe_relative(name)) == digest, 'Frozen preparation file changed: '+name)
    require((plan['sweeps'], plan['burn_sweeps'], plan['sample_every'], plan['workers']) == (SWEEPS, BURN, 1, WORKERS),
            'Preparation budget changed')
    require(plan['model_sha256'] == sha(preparation/'model.json'), 'Preparation model changed')
    require(not any((preparation/'runs').iterdir()) and not any((preparation/'logs').iterdir()), 'Preparation already contains job outputs')
    validate_configs(plan['jobs'])
    return plan


def relocated_config(original, campaign):
    config = copy.deepcopy(original)
    archive = campaign/'provenance'
    config['shape'] = str(archive/'tetramer-shape.json')
    config['monomer_shape'] = str(archive/'monomer-shape.json')
    config['metadata']['native_pair_motifs'] = str(archive/'native-pair-motifs.json')
    return config


def command_for(campaign, job):
    return [str(campaign/'provenance/tetramer-mc'), 'run', '--config', job['config'],
            '--model', str(campaign/'provenance/model.json'), '--method', 'learned',
            '--out', job['directory'], '--sweeps', str(SWEEPS), '--sample-every', '1', '--no-gsd']


def freeze(preparation, campaign, binary, source_bundle, *, preparation_sha256=PREPARATION_SHA256,
           reviewed_root=ROOT, analyzer=ANALYZER):
    preparation, campaign, binary, source_bundle, reviewed_root, analyzer = [Path(p).resolve() for p in
        (preparation, campaign, binary, source_bundle, reviewed_root, analyzer)]
    require(not campaign.exists(), 'Fresh campaign directory required')
    require(not inside(campaign, preparation), 'Production campaign must be separate from frozen preparation')
    plan = verify_preparation(preparation, preparation_sha256)
    bundle, rust_sources = verify_bundle(binary, source_bundle, reviewed_root)
    dependencies = local_dependencies([Path(__file__).resolve(), analyzer])
    require(analyzer.name == 'analyze_mobile_posterior_pilot.py' and analyzer.is_file(), 'Require the completed mobile observer')
    # Validate everything before creating the new immutable campaign.
    sources = {name: preparation/'provenance'/safe_relative(name) for name in plan['input_sha256']}
    for name, path in sources.items():
        require(sha(path) == plan['input_sha256'][name], 'Preparation input changed: '+name)
    coordinate_path, residue_validation = locate_residue_coordinates(sources['monomer-shape.json'],
        [plan.get('source_paths', {}).get('monomer-shape.json')])
    sources[COORDINATE_ARCHIVE] = coordinate_path
    for name, path in dependencies.items():
        if name in sources:
            require(sha(sources[name]) == sha(path), 'Observer/controller dependency differs from frozen preparation: '+name)
        sources[name] = path
    sources.update({'tetramer-mc': binary, 'source-bundle.json': source_bundle,
        'model.json': preparation/'model.json', 'initial-states.json': preparation/'initial-states.json',
        'analysis-plan.json': preparation/'analysis-plan.json', 'density-identity.json': preparation/'density-identity.json',
        'preparation-plan.json': preparation/'plan.json', 'preparation-freeze.json': preparation/'freeze.json'})
    for job in plan['jobs']:
        sources['sourceconfigs/'+job['id']+'.json'] = Path(job['config'])
    # Recreate the external observer's conventional reference tree without any
    # dependency on mutable live reference scripts or datasets at audit time.
    for name, path in list(sources.items()):
        if name.startswith('native-observer/'):
            filename = Path(name).name
            if filename.endswith('.py'):
                sources['reference/scripts/'+filename] = path
            elif filename == 'motifs.json':
                sources['reference/results/c1c3-scaffold/motifs.json'] = path
            elif filename == 'classification.json':
                sources['reference/results/native-neighbor-classes/classification.json'] = path
    source_hashes = {name: sha(path) for name, path in sources.items()}
    archive = campaign/'provenance'
    archive.mkdir(parents=True)
    for name, path in sources.items():
        target = archive/safe_relative(name)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        require(sha(target) == source_hashes[name], 'Source changed during campaign freeze: '+name)
    for name, entry in bundle['files'].items():
        target = archive/'source'/safe_relative(name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(entry['text'].encode('utf-8'))
        require(sha(target) == rust_sources[name], 'Archived Rust source changed: '+name)
        source_hashes['source/'+name] = rust_sources[name]
    for name in ('configs', 'runs', 'logs'):
        (campaign/name).mkdir()
    jobs = []
    for source_job in plan['jobs']:
        identity = source_job['id']
        config_path = campaign/'configs'/(identity+'.json')
        config = relocated_config(read(archive/'sourceconfigs'/(identity+'.json')), campaign)
        write(config_path, config)
        job = {key: source_job[key] for key in ('id', 'index', 'start', 'replicate', 'mode', 'seed')}
        job.update(config=str(config_path), config_sha256=sha(config_path),
                   directory=str(campaign/'runs'/identity), log=str(campaign/'logs'/(identity+'.log')))
        job['command'] = command_for(campaign, job)
        jobs.append(job)
    manifest = dict(schema=SCHEMA, frozen=True, launched=False, binary_supplied=True,
        binary_sha256=source_hashes['tetramer-mc'], source_bundle_sha256=source_hashes['source-bundle.json'],
        reviewed_source_root=str(reviewed_root), rust_sources=rust_sources,
        preparation_path=str(preparation), preparation_plan_sha256=preparation_sha256,
        sweeps=SWEEPS, burn_sweeps=BURN, sample_every=1, workers=WORKERS, bodies=3,
        single_body_attempts_per_run=3*SWEEPS, collective_attempts_per_run=2*SWEEPS,
        total_attempts=12*SWEEPS*5, total_single_body_attempts=12*SWEEPS*3, total_collective_attempts=12*SWEEPS*2,
        model=str(archive/'model.json'), model_sha256=source_hashes['model.json'], shape_sha256=plan['shape_sha256'],
        analysis_plan_sha256=source_hashes['analysis-plan.json'], observer_sha256=source_hashes[analyzer.name],
        reference=str(archive/'reference'), jobs=jobs, seeds=[j['seed'] for j in jobs],
        residue_coordinate_validation=residue_validation,
        input_sha256=source_hashes, source_paths={name: str(path) for name, path in sources.items()},
        scope=plan['scope'], proposal_groups=plan['proposal_groups'],
        unconditional_single_body_kernel_probabilities=plan['unconditional_single_body_kernel_probabilities'])
    write(campaign/'manifest.json', manifest)
    write(campaign/'commands.json', dict(jobs=jobs, workers=WORKERS, actual_kernel_executions=0))
    write(campaign/'freeze.json', dict(schema='immutable-mobile-posterior-campaign-v1',
        files={str(path.relative_to(campaign)): sha(path) for path in sorted(campaign.rglob('*')) if path.is_file()}))
    validate_immutable(campaign)
    return manifest


def validate_immutable(campaign):
    campaign = Path(campaign).resolve()
    manifest = read(campaign/'manifest.json')
    require(manifest['schema'] == SCHEMA and manifest['frozen'] is True and manifest['binary_supplied'] is True,
            'Unknown or unfrozen mobile production campaign')
    for name, digest in read(campaign/'freeze.json')['files'].items():
        require(sha(campaign/safe_relative(name)) == digest, 'Frozen campaign file changed: '+name)
    archive = campaign/'provenance'
    for name, digest in manifest['input_sha256'].items():
        require(sha(archive/safe_relative(name)) == digest, 'Archived campaign input changed: '+name)
    require(sha(archive/'tetramer-mc') == manifest['binary_sha256'] and
            sha(archive/'source-bundle.json') == manifest['source_bundle_sha256'], 'Executable/source identity changed')
    _, hashes = verify_bundle(archive/'tetramer-mc', archive/'source-bundle.json', archive/'source')
    require(hashes == manifest['rust_sources'], 'Frozen embedded source map changed')
    require(manifest['model'] == str(archive/'model.json') and sha(manifest['model']) == manifest['model_sha256'], 'Frozen model changed')
    require(sha(archive/'tetramer-shape.json') == manifest['shape_sha256'], 'Frozen shape changed')
    require(sha(archive/'analysis-plan.json') == manifest['analysis_plan_sha256'], 'Analysis plan changed')
    require(sha(archive/'analyze_mobile_posterior_pilot.py') == manifest['observer_sha256'], 'Archived observer changed')
    require(manifest['reference'] == str(archive/'reference'), 'Observer reference must be archived')
    require(COORDINATE_ARCHIVE in manifest['input_sha256'], 'Native observer authoritative coordinate data is not archived')
    residue_validation = validate_residue_coordinates(archive/'monomer-shape.json', archive/COORDINATE_ARCHIVE)
    require(residue_validation == manifest['residue_coordinate_validation'], 'Frozen native residue mapping changed')
    require((manifest['sweeps'], manifest['burn_sweeps'], manifest['sample_every'], manifest['workers'], manifest['bodies'])
            == (SWEEPS, BURN, 1, WORKERS, 3), 'Frozen campaign allocation changed')
    require(manifest['total_attempts'] == 120000 and manifest['total_single_body_attempts'] == 72000
            and manifest['total_collective_attempts'] == 48000, 'Frozen total attempt budget changed')
    jobs = manifest['jobs']
    require(len(jobs) == len({j['id'] for j in jobs}) == 12, 'Need exactly all 12 distinct jobs')
    source_plan = read(archive/'preparation-plan.json')
    require(sha(archive/'preparation-plan.json') == manifest['preparation_plan_sha256'], 'Source preparation identity changed')
    require(len(source_plan['jobs']) == 12, 'Missing source design jobs')
    expected = [(start, replica, mode) for start in STARTS for replica in (0, 1) for mode in MODES]
    for index, (job, source_job, design) in enumerate(zip(jobs, source_plan['jobs'], expected)):
        require(tuple(job[key] for key in ('start', 'replicate', 'mode')) == design and job['index'] == index,
                'Frozen factorial allocation changed')
        require(job['id'] == source_job['id'] and job['seed'] == source_job['seed'] == SEED_BASE+SEED_STRIDE*index,
                'Job identity or independent seed stream changed')
        require(job['config'] == str(campaign/'configs'/(job['id']+'.json'))
                and job['directory'] == str(campaign/'runs'/job['id'])
                and job['log'] == str(campaign/'logs'/(job['id']+'.log')), 'Frozen job paths changed')
        original = archive/'sourceconfigs'/(job['id']+'.json')
        require(sha(original) == source_job['config_sha256'], 'Archived source configuration changed')
        require(read(job['config']) == relocated_config(read(original), campaign), 'Physical configuration changed during relocation')
        require(sha(job['config']) == job['config_sha256'], 'Frozen configuration changed')
        require(job['command'] == command_for(campaign, job), 'Exact frozen command changed')
    require(manifest['seeds'] == [job['seed'] for job in jobs], 'Manifest seeds differ')
    return manifest


def check(campaign, journal, assessment):
    campaign, journal, assessment = [Path(p).resolve() for p in (campaign, journal, assessment)]
    paths = (campaign, journal, assessment)
    require(all(not inside(a, b) and not inside(b, a) for a, b in ((paths[0], paths[1]), (paths[0], paths[2]), (paths[1], paths[2]))),
            'Campaign, journal and assessment must be separate nonoverlapping directories')
    manifest = validate_immutable(campaign)
    require(sha(__file__) == manifest['input_sha256']['run_mobile_posterior_pilot.py'],
            'Run/preflight must use the frozen controller source')
    require(not journal.exists() and not assessment.exists() and not (campaign/'status.json').exists(), 'No restart or overwrite')
    require(not any((campaign/'runs').iterdir()) and not any((campaign/'logs').iterdir()), 'No replay of existing job outputs or logs')
    for job in manifest['jobs']:
        require(not Path(job['directory']).exists() and not Path(job['log']).exists(), 'Existing physical job output')
    return manifest


def finish_child(status, index, child):
    result = child.wait()
    status['jobs'][index].update(status='complete' if result == 0 else 'failed', exit_code=result, finished=time.time())
    return result


def run(campaign, journal, assessment):
    campaign, journal, assessment = [Path(p).resolve() for p in (campaign, journal, assessment)]
    manifest = check(campaign, journal, assessment)
    journal.mkdir(parents=True)
    shutil.copy2(__file__, journal/'runner.py')
    state = dict(complete=False, phase='physical', started=time.time(),
                 manifest_sha256=sha(campaign/'manifest.json'), runner_sha256=sha(__file__))
    status = dict(running=True, complete=False,
                  jobs=[dict(id=job['id'], status='pending', exit_code=None) for job in manifest['jobs']])
    # Exclusive reservation prevents two separately named journals racing to
    # launch the same frozen campaign after both passed read-only preflight.
    exclusive_write(campaign/'status.json', status)
    write(journal/'status.json', state)
    active, index, failed, exception = {}, 0, False, None
    try:
        while active or (index < len(manifest['jobs']) and not failed):
            while not failed and index < len(manifest['jobs']) and len(active) < manifest['workers']:
                slot, job = index, manifest['jobs'][index]
                require(sha(job['command'][0]) == manifest['binary_sha256'], 'Executable changed at launch')
                require(sha(job['config']) == job['config_sha256'] and sha(manifest['model']) == manifest['model_sha256'], 'Job input changed at launch')
                require(not Path(job['directory']).exists(), 'Job output appeared before launch')
                index += 1
                status['jobs'][slot].update(status='launching', started=time.time(), argv=job['command'], log=job['log'])
                write(campaign/'status.json', status)
                try:
                    with Path(job['log']).open('xb') as log:
                        child = subprocess.Popen(job['command'], stdout=log, stderr=subprocess.STDOUT)
                except BaseException as error:
                    status['jobs'][slot].update(status='launch_failed', exception=repr(error), finished=time.time())
                    raise
                active[slot] = child
                status['jobs'][slot].update(status='running', pid=child.pid)
                write(campaign/'status.json', status)
                print('Started', job['id'], 'PID', child.pid, flush=True)
            for slot, child in list(active.items()):
                if child.poll() is None:
                    continue
                result = finish_child(status, slot, child)
                failed = failed or result != 0
                del active[slot]
                write(campaign/'status.json', status)
                print('Finished', status['jobs'][slot]['id'], 'exit', result, flush=True)
            if active:
                time.sleep(.5)
    except BaseException as error:
        failed, exception = True, error
        state['exception'] = repr(error)
    finally:
        # Never leave a started child unobserved or restart it after a failure.
        for slot, child in active.items():
            result = finish_child(status, slot, child)
            failed = failed or result != 0
        for entry in status['jobs']:
            if entry['status'] == 'pending':
                entry.update(status='not_started', reason='Campaign stopped after failure; no retry.')
        status.update(running=False, complete=not failed and index == len(manifest['jobs'])
                      and all(entry['status'] == 'complete' and entry['exit_code'] == 0 for entry in status['jobs']))
        write(campaign/'status.json', status)
    if not status['complete']:
        state.update(phase='physical_failed', finished=time.time(), physical_status_sha256=sha(campaign/'status.json'))
        write(journal/'status.json', state)
        if exception is not None:
            raise exception
        raise RuntimeError('A frozen physical job failed; all children drained, outputs preserved, no retry or observer')
    try:
        validate_immutable(campaign)
        require(not assessment.exists(), 'Assessment appeared while physical jobs ran; no overwrite')
        command = [sys.executable, str(campaign/'provenance/analyze_mobile_posterior_pilot.py'),
                   '--campaign', str(campaign), '--out', str(assessment), '--reference', manifest['reference'], '--workers', str(WORKERS)]
        state.update(phase='audit', physical_status_sha256=sha(campaign/'status.json'), analysis_command=command)
        write(journal/'status.json', state)
        with (journal/'analysis.log').open('xb') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        state['analysis_exit_code'] = result.returncode
        result.check_returncode()
        assessment_summary = read(assessment/'analysis.json')
        require(assessment_summary['complete'] is True and len(assessment_summary['runs']) == 12
                and assessment_summary['manifest_sha256'] == sha(campaign/'manifest.json')
                and assessment_summary['terminal_status_sha256'] == sha(campaign/'status.json')
                and assessment_summary['analyzer_sha256'] == manifest['observer_sha256'], 'Observer did not produce the matching completed assessment')
        results = {entry['id']: entry for entry in assessment_summary['runs']}
        require(len(results) == 12 and set(results) == {job['id'] for job in manifest['jobs']},
                'Observer result identities do not contain exactly all 12 frozen jobs')
        require(all(results[job['id']]['passed'] is True and
                    all(results[job['id']][key] == job[key] for key in ('mode', 'start', 'replicate', 'seed'))
                    for job in manifest['jobs']), 'Observer omitted or failed a frozen run identity')
    except BaseException as error:
        state.update(complete=False, phase='audit_failed', exception=repr(error), finished=time.time())
        write(journal/'status.json', state)
        raise
    state.update(complete=True, phase='complete', finished=time.time(), assessment_sha256=sha(assessment/'analysis.json'))
    write(journal/'status.json', state)
    print('Observer complete:', assessment/'analysis.json', flush=True)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest='action', required=True)
    frozen = subcommands.add_parser('freeze', help='Archive an explicit reviewed binary and its exact embedded source; never execute')
    frozen.add_argument('--preparation', type=Path, default=PREPARATION)
    frozen.add_argument('--preparation-sha256', default=PREPARATION_SHA256)
    frozen.add_argument('--campaign', type=Path, required=True)
    frozen.add_argument('--binary', type=Path, required=True)
    frozen.add_argument('--source-bundle', type=Path, required=True)
    for action in ('preflight', 'run'):
        subparser = subcommands.add_parser(action)
        subparser.add_argument('--campaign', type=Path, required=True)
        subparser.add_argument('--journal', type=Path, required=True)
        subparser.add_argument('--assessment', type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'freeze':
        manifest = freeze(args.preparation, args.campaign, args.binary, args.source_bundle,
                          preparation_sha256=args.preparation_sha256)
        print(json.dumps(dict(frozen=True, jobs=len(manifest['jobs']), total_attempts=manifest['total_attempts'], actual_kernel_executions=0)))
    elif args.action == 'preflight':
        manifest = check(args.campaign, args.journal, args.assessment)
        print(json.dumps(dict(preflight=True, jobs=len(manifest['jobs']), workers=manifest['workers'], actual_kernel_executions=0)))
    else:
        run(args.campaign, args.journal, args.assessment)


if __name__ == '__main__':
    main()
