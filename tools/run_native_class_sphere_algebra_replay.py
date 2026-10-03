#!/usr/bin/env python3
"""Freeze metadata/bytes, then replay only saved sphere-control algebra once.

Preparation never parses sample/attempt rows, constructs atom geometry or draws
random points. Execution requires a separate root review and uses the tested
owned_child launcher; all 16 original populations are visited sequentially with
fixed limits. This is a compatibility gate, not new physical evidence.
"""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
import traceback

import native_class_line_physical_algebra_audit as audit
import run_evolving_dimer_analysis as launcher

require, read, sha = audit.require, audit.read, audit.sha
SCHEMA = 'native-class-sphere-algebra-replay-v1'
IDS = [f'z{z}-{arm}-p{p}' for z in ('0', '0.3') for arm in ('hard-free', 'class') for p in range(4)]
CPU, WALL, MEMORY = 120, 240, 4*1024**3
SOURCE_NAMES = tuple(audit.SOURCE_NAMES) + ('run_evolving_dimer_analysis.py', 'run_native_class_sphere_algebra_replay.py')


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')
        stream.flush(); os.fsync(stream.fileno())


def bind(files, path, expected=None):
    path = Path(path).resolve(); actual = sha(path)
    require(expected is None or actual == expected, 'Changed bound file '+str(path))
    require(str(path) not in files or files[str(path)] == actual, 'File changed during preparation '+str(path))
    files[str(path)] = actual
    return path


def prepare(base, root):
    base, root = Path(base).resolve(), Path(root).resolve()
    require(not root.exists(), 'Fresh replay directory required')
    files = {}
    allocation = read(bind(files, base/'allocation.json'))
    completion = read(bind(files, base/'completion.json'))
    execution = read(bind(files, base/'execution-summary.json', completion['execution_summary_sha256']))
    require(completion['complete'] is True and completion['passed'] is True
            and execution['complete'] is True and execution['passed'] is True
            and execution['failure'] is None and execution['remaining_jobs_not_started'] == 0
            and execution['retries'] == 0 and not (base/'failure.json').exists(), 'Original sphere campaign is incomplete/failed')
    bind(files, base/'allocation.json', execution['allocation_sha256'])
    binding = read(bind(files, base/'execution-binding.json', execution['binding_sha256']))
    require(binding['allocation_sha256'] == execution['allocation_sha256'], 'Original allocation binding differs')
    bind(files, binding['executable'], binding['executable_sha256'])
    reference = read(bind(files, base/'analytic-validation.json', completion['analysis']['terminal_sha256']))
    require(reference['complete'] is True and reference['passed'] is True and reference['attempts'] == 32768,
            'Original analytic sphere validation did not pass')
    require([j['id'] for j in allocation['jobs']] == IDS and allocation['total_attempts'] == 32768
            and allocation['draws_per_population'] == 2048, 'Changed original 16-population inventory')
    records = {(v['id'], v['phase']): v for v in execution['completed']}
    require(len(records) == len(execution['completed']) == 32
            and set(records) == {(i, p) for i in IDS for p in ('physical', 'audit')}, 'Original terminal inventory differs')
    jobs = []
    for old in allocation['jobs']:
        name = old['id']; population = (base/'populations'/name).resolve()
        require(Path(old['out']).resolve() == population and old['samples'] == 2048, 'Original population path/count differs')
        manifest = read(bind(files, population/'manifest.json'))
        for phase, expected in [('physical', population/'summary.json'), ('audit', base/'audits'/(name+'.json'))]:
            terminal = records[name, phase]
            require(terminal['returncode'] == 0 and terminal['timeout'] is False
                    and Path(terminal['terminal']).resolve() == expected, 'Original terminal failed or path differs')
            bind(files, expected, terminal['terminal_sha256'])
        summary = read(population/'summary.json'); prior = read(base/'audits'/(name+'.json'))
        require(summary['complete'] is True and summary['manifest'] == manifest
                and manifest['schema'] == audit.full.SCHEMA and manifest['samples'] == summary['samples'] == 2048
                and manifest['seed'] == old['seed'] and manifest['activity'] == old['activity']
                and not (population/'failure.json').exists(), 'Incomplete/changed original v7 population')
        require(prior['complete'] is True and prior['passed'] is True
                and prior['samples'] == prior['independently_reconstructed_geometry_rows'] == 2048
                and prior['samples_sha256'] == summary['samples_sha256']
                and prior['attempts_sha256'] == summary['attempts_sha256'], 'Prior full audit does not bind the same rows')
        require(manifest['executable_sha256'] == execution['executable_sha256'] == binding['executable_sha256'],
                'Original producer identity differs')
        for old_key, hash_key in [('config', 'config_sha256'), ('region', 'region_sha256'), ('guide', 'importance_guide_sha256')]:
            bind(files, old[old_key], manifest[hash_key])
        for filename, key in [('input-config.json', 'config_sha256'), ('region.json', 'region_sha256'),
                              ('shape.json', 'shape_sha256'), ('importance-guide.json', 'importance_guide_sha256'),
                              ('source-bundle.json', 'source_bundle_sha256')]:
            bind(files, population/'provenance'/filename, manifest[key])
        bind(files, population/'provenance/compiled-native.json', manifest['compiled_native']['compiled_sha256'])
        # Raw chunked hashes only: no row parsing or numerical reconstruction.
        bind(files, population/'samples.jsonl', summary['samples_sha256'])
        bind(files, population/'attempts.jsonl', summary['attempts_sha256'])
        jobs.append(dict(id=name, population=str(population), samples=2048, seed=old['seed'],
                         activity=old['activity'], method=old['method'],
                         prior_audit=str((base/'audits'/(name+'.json')).resolve()),
                         output=str(root/'audits'/(name+'.json'))))
    root.mkdir(); (root/'code').mkdir(); (root/'audits').mkdir(); (root/'children').mkdir()
    original_sources = {}
    try:
        source_root = Path(__file__).resolve().parent
        for name in SOURCE_NAMES:
            source = source_root/name; raw = source.read_bytes(); digest = sha(source)
            target = root/'code'/name; target.write_bytes(raw)
            require(sha(target) == digest, 'Source changed during archival')
            bind(files, target, digest); original_sources[str(source)] = digest
        runtime = audit.runtime_identity()
        for path, digest in runtime['file_sha256'].items(): bind(files, path, digest)
        python = str(Path(sys.executable).absolute())
        bind(files, python)
        validation = source_root.parent/'results/native-class-physical-algebra-validation-20261003/wrapper-tests02.json'
        tested = read(bind(files, validation))
        require(tested['complete'] is True and tested['passed'] is True
                and tested['source_closure_before'] == tested['source_closure_after'], 'Auditor validation is incomplete')
        tested_by_name = {Path(path).name: digest for path, digest in tested['source_closure_before'].items()}
        for name in audit.SOURCE_NAMES:
            require(tested_by_name.get(name) == files[str(root/'code'/name)], 'Auditor source differs from validated closure '+name)
        bind(files, tested['log_path'], tested['log_sha256'])
        for job in jobs:
            job['argv'] = [python, str(root/'code/native_class_line_physical_algebra_audit.py'),
                           '--root', job['population'], '--out', job['output']]
        plan = dict(schema=SCHEMA, execution_ready=False, review_required=True, base=str(base), root=str(root),
                    jobs=jobs, expected_populations=16, expected_saved_attempts=32768, maximum_workers=1,
                    cpu_limit_per_population_seconds=CPU, wall_limit_per_population_seconds=WALL,
                    address_space_limit_bytes=MEMORY, maximum_child_cpu_soft_seconds=16*CPU,
                    maximum_child_wall_seconds=16*WALL, source_names=list(SOURCE_NAMES),
                    files=files, original_source_sha256=original_sources, python=python, runtime=runtime,
                    original_analytic_validation=str(base/'analytic-validation.json'),
                    new_pose_draws=0, new_Poisson_clouds=0, new_atom_geometry_queries=0, retries=0,
                    scope='Compatibility replay of existing sphere-control algebra only; old full geometry/analytic evidence is linked, never rerun.',
                    comparison='New unconditional moments, paired-noise and counters must match prior full-audit receipts on identical saved rows; this is not a speedup comparison.')
        for path, digest in files.items(): require(sha(path) == digest, 'Frozen input changed during preparation')
        write(root/'execution-plan.json', plan)
        return plan
    except BaseException as exc:
        write(root/'preparation-failure.json', dict(complete=False, passed=False, error=repr(exc),
                                                   new_pose_draws=0, new_Poisson_clouds=0))
        raise


def verify(root, plan, plan_sha):
    require(sha(root/'execution-plan.json') == plan_sha and plan['schema'] == SCHEMA
            and plan['execution_ready'] is False and plan['review_required'] is True, 'Changed reviewed plan')
    require(plan['root'] == str(root) and plan['expected_populations'] == 16
            and plan['expected_saved_attempts'] == 32768 and plan['maximum_workers'] == 1
            and plan['cpu_limit_per_population_seconds'] == CPU and plan['wall_limit_per_population_seconds'] == WALL
            and plan['address_space_limit_bytes'] == MEMORY, 'Changed replay allocation or limits')
    require([j['id'] for j in plan['jobs']] == IDS and all(j['samples'] == 2048 for j in plan['jobs'])
            and all(plan[k] == 0 for k in ('new_pose_draws', 'new_Poisson_clouds', 'new_atom_geometry_queries', 'retries')),
            'Changed replay scope')
    require(Path(__file__).resolve() == root/'code'/Path(__file__).name
            and Path(launcher.__file__).resolve() == root/'code/run_evolving_dimer_analysis.py', 'Run archived driver and launcher only')
    for name in SOURCE_NAMES:
        require(str(root/'code'/name) in plan['files'], 'Missing frozen source '+name)
    for job in plan['jobs']:
        require(job['population'] == str(Path(plan['base'])/'populations'/job['id'])
                and job['output'] == str(root/'audits'/(job['id']+'.json'))
                and job['prior_audit'] == str(Path(plan['base'])/'audits'/(job['id']+'.json'))
                and job['argv'] == [plan['python'], str(root/'code/native_class_line_physical_algebra_audit.py'),
                                   '--root', job['population'], '--out', job['output']], 'Changed audit command')
        population = Path(job['population'])
        for name in ('manifest.json', 'summary.json', 'samples.jsonl', 'attempts.jsonl',
                     'provenance/input-config.json', 'provenance/region.json', 'provenance/shape.json',
                     'provenance/importance-guide.json', 'provenance/source-bundle.json', 'provenance/compiled-native.json'):
            require(str(population/name) in plan['files'], 'Missing frozen population input '+name)
        require(job['prior_audit'] in plan['files'], 'Missing frozen prior audit')
        require(not (population/'failure.json').exists(), 'Producer failure appeared')
    for path, digest in plan['files'].items(): require(sha(path) == digest, 'Frozen replay input changed '+path)
    require(audit.runtime_identity() == plan['runtime'], 'Python runtime identity changed')


def same_statistics(actual, expected):
    """Identical ordered log arrays, with explicit roundoff tolerance only."""
    if isinstance(expected, dict):
        require(type(actual) is dict and set(actual) == set(expected), 'Changed statistical fields')
        for key in expected: same_statistics(actual[key], expected[key])
    elif type(expected) in (int, float):
        require(type(actual) in (int, float) and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12),
                'Saved-row statistic differs')
    else:
        require(actual == expected, 'Saved-row statistic metadata differs')


def run(root):
    root = Path(root).resolve(); plan_sha = sha(root/'execution-plan.json'); plan = read(root/'execution-plan.json')
    verify(root, plan, plan_sha)
    review_path = root/'review.json'; review_sha = sha(review_path); review = read(review_path)
    require(review.get('complete') is True and review.get('passed') is True
            and review.get('plan_sha256') == plan_sha and review.get('maximum_workers') == 1
            and review.get('expected_saved_attempts') == 32768, 'Matching root review required')
    write(root/'claim.json', dict(pid=os.getpid(), started=time.time(), plan_sha256=plan_sha, review_sha256=review_sha, retries=0))
    completed = []; current = None
    previous = signal.signal(signal.SIGTERM, launcher.terminate_requested)
    try:
        for job in plan['jobs']:
            current = job['id']; verify(root, plan, plan_sha)
            require(sha(review_path) == review_sha, 'Replay review changed')
            destination = root/'children'/job['id']; destination.mkdir()
            write(destination/'begin.json', dict(id=current, argv=job['argv'], samples=2048,
                                                plan_sha256=plan_sha, review_sha256=review_sha))
            launcher.owned_child(job['argv'], root, destination, CPU, WALL, MEMORY)
            result = read(job['output']); prior = read(job['prior_audit'])
            require(result['complete'] is True and result['passed'] is True and result['all_rows_algebra'] == 2048
                    and result['independently_reconstructed_geometry_rows'] == 0 and result['geometry_certified'] is False,
                    'Incomplete or wrong-scope algebra receipt')
            for key in ('estimate', 'hard_region', 'paired_noise', 'counts'):
                same_statistics(result[key], prior[key])
            for key in ('samples_sha256', 'attempts_sha256'):
                require(result[key] == prior[key], 'Compatibility comparison uses different saved rows')
            require(result['new_pose_draws'] == result['new_Poisson_clouds'] == 0, 'Unexpected new draws')
            record = dict(id=current, rows=2048, receipt=dict(path=job['output'], sha256=sha(job['output'])),
                          prior_audit=dict(path=job['prior_audit'], sha256=sha(job['prior_audit'])),
                          algebra_cpu_seconds=result['analysis_cpu_seconds'],
                          prior_full_audit_cpu_seconds=prior['analysis_cpu_seconds'])
            write(destination/'comparison.json', dict(complete=True, passed=True, **record)); completed.append(record)
            print(json.dumps(dict(id=current, complete=True, rows=2048)), flush=True)
            current = None
        verify(root, plan, plan_sha); require(sha(review_path) == review_sha, 'Replay review changed')
        write(root/'completion.json', dict(complete=True, passed=True, plan_sha256=plan_sha,
            review_sha256=review_sha, populations=completed, all_rows_algebra=32768,
            independently_reconstructed_geometry_rows=0, geometry_certified=False,
            new_pose_draws=0, new_Poisson_clouds=0, new_atom_geometry_queries=0, retries=0,
            scope=plan['scope'], timing_note='Different work scopes; timings are reported separately without speedup inference.'))
    except BaseException as exc:
        # owned_child has already killed/reaped any active group on failure.
        write(root/'failure.json', dict(complete=False, passed=False, current=current, completed=completed,
            error=repr(exc), traceback=traceback.format_exc(), retries=0, remaining_jobs_not_started=16-len(completed)-(current is not None),
            partial_outputs_retained=True, new_pose_draws=0, new_Poisson_clouds=0))
        raise
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if args.run:
        run(args.root)
    else:
        if args.base is None: parser.error('--base is required for metadata-only preparation')
        plan = prepare(args.base, args.root)
        print(json.dumps(dict(execution_ready=False, populations=16, saved_attempts=32768,
                              bound_files=len(plan['files']), new_pose_draws=0, new_Poisson_clouds=0)))
