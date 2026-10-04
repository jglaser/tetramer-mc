#!/usr/bin/env python3
"""Freeze a completed 64-chain campaign, then observe exactly once.

Preparation hashes metadata and saved bytes only. The separately invoked run
uses the campaign's frozen analyzer for 64 new trajectories and 64 cached
local/m4 controls. It creates no new physical draws and performs no old geometry.
"""
from __future__ import annotations
import argparse
import ast
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time
import traceback

import analyze_two_neighbor_singleton_benchmark as analyzer
import run_evolving_dimer_analysis as launcher
import run_two_neighbor_singleton_benchmark as campaign

require, read, sha = launcher.require, launcher.read, launcher.sha
NAME = 'run_two_neighbor_singleton_analysis.py'
ANALYZER = 'analyze_two_neighbor_singleton_benchmark.py'
SCHEMA = 'two-neighbor-singleton-observer-execution-v1'
SCOPE = dict(maximum_workers=1, cpu_limit_seconds=3600, wall_limit_seconds=7200,
    address_space_limit_bytes=16*1024**3, new_chains=64, reused_control_chains=64,
    expected_new_retained_endpoints=294976, expected_new_production_endpoints=262144,
    maximum_new_pair_classifications=154862400, new_physical_draws=0,
    old_geometry_queries=0, native_observer=False, partial_analysis_allowed=False,
    retries=0, replacements=0)


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')
        stream.flush(); os.fsync(stream.fileno())


def record(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha(path))


def runtime():
    return dict(python=sys.version, executable=record(sys.executable),
        packages={name: importlib.metadata.version(name) for name in ('numpy', 'scipy')},
        threads={name: os.environ.get(name) for name in
                 ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS')})


def source_closure():
    """Archive the wrapper, its test and every local metadata helper import."""
    pending = [Path(__file__).resolve(), Path(__file__).resolve().parent/('test_'+NAME)]
    found = {}
    while pending:
        path = pending.pop().resolve()
        if path.name in found:
            require(found[path.name] == path, 'Ambiguous wrapper dependency name'); continue
        require(path.is_file(), 'Missing wrapper source '+str(path)); found[path.name] = path
        for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
            names = ([v.name for v in node.names] if isinstance(node, ast.Import) else
                     [node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
            for name in names:
                candidate = path.parent/(name.split('.')[0]+'.py')
                if candidate.is_file(): pending.append(candidate)
    return found


def observer_contract(base):
    value = read(Path(base)/'analysis-plan.json')
    require(value == analyzer.analysis_plan(), 'Frozen observer allocation or source contract differs')
    require(value['new_chains'] == 64 and value['reused_control_chains'] == 64
        and value['new_retained_initial_observations'] == 294976
        and value['new_production_observations'] == 262144
        and value['maximum_new_pair_classifications'] == 154862400
        and value['native_observer'] is False and value['new_physical_draws'] == 0
        and value['old_geometry_queries'] == 0 and value['complete_inventory_required'] is True,
        'Observer exceeds fixed scope')
    return value


def admit(base):
    """Authenticate complete authority without parsing trajectories or poses."""
    base = Path(base).resolve()
    status = read(base/'dispatch/status.json')
    require(status['complete'] is True and status['passed'] is True and status['failure'] is None
        and status['failure_draining'] is False and status['active'] == [] and status['unstarted'] == [],
        'All 64 completed and drained chains required before observer preparation')
    dispatch_path = base/'dispatch/plan.json'; dispatch = read(dispatch_path)
    require(sha(dispatch_path) == status['plan_sha256'], 'Dispatch plan changed')
    review_path = campaign.checked(dispatch['review'])
    config = campaign.review_inputs(base, review_path)
    # Existing metadata-only binding validates all 64 terminals and control
    # receipts/caches. It deliberately does not parse either kind of journal.
    analysis_inputs, items, controls = analyzer.bind_complete_inputs(base, config)
    files = dict(analysis_inputs)
    def bind(path, expected=None):
        value = record(path)
        require(expected is None or value['sha256'] == expected, 'Changed execution input '+str(path))
        require(value['path'] not in files or files[value['path']] == value['sha256'], 'Input changed during admission')
        files[value['path']] = value['sha256']; return Path(value['path'])
    def nested(value):
        if isinstance(value, dict):
            if set(value) == {'path', 'sha256'}: bind(value['path'], value['sha256'])
            else:
                for child in value.values(): nested(child)
        elif isinstance(value, list):
            for child in value: nested(child)
    frozen = read(bind(base/'freeze.json'))
    require(frozen['schema'] == 'two-neighbor-singleton-freeze-v1' and frozen['complete'] is True,
            'Missing campaign freeze')
    for name, digest in frozen['files'].items():
        path = (base/name).resolve(); require(path.is_relative_to(base), 'Frozen source escaped campaign')
        bind(path, digest)
    for path, digest in frozen['input_sha256'].items(): bind(path, digest)
    protocol = read(bind(base/'protocol.json'))
    for name, digest in protocol['source_files'].items():
        path = (base/'common/source'/name).resolve()
        require(path.is_relative_to(base/'common/source'), 'Execution source escaped archive')
        bind(path, digest)
    # Actual metadata helpers must match the previously frozen implementation.
    # New wrapper/test files are archived separately and do not change that code.
    required = {ANALYZER, 'run_two_neighbor_singleton_benchmark.py', 'run_evolving_dimer_analysis.py'}
    sources = source_closure()
    require(required <= set(sources), 'Missing executed metadata helper')
    for name, path in sources.items():
        key = 'tools/'+name
        if key in protocol['source_files']:
            require(sha(path) == protocol['source_files'][key], 'Metadata helper differs from frozen campaign: '+name)
        else:
            require(name not in required, 'Metadata helper lacks pre-production binding: '+name)
    for value in (config, dispatch): nested(value)
    for name in ('binding.json', 'run-binding.json'):
        value = read(bind(base/name)); nested(value)
        for path, digest in value.get('prepared_files', {}).items(): bind(path, digest)
    binding = read(base/'binding.json')
    for key, name in [('config_sha256', 'config.json'), ('protocol_sha256', 'protocol.json'),
            ('freeze_sha256', 'freeze.json'), ('executable_sha256', 'common/evolving_dimer_benchmark'),
            ('compiled_source_bundle_sha256', 'common/source-bundle.json'),
            ('example_source_sha256', 'common/source/examples/evolving_dimer_benchmark.rs')]:
        bind(base/name, binding[key])
    review = read(bind(review_path, dispatch['review']['sha256']))
    require(review['complete'] is True and review['passed'] is True, 'Incomplete campaign review')
    for value in (review, dispatch):
        for path, digest in value['input_sha256'].items(): bind(path, digest)
    observer_contract(base)
    for path, digest in files.items(): require(sha(path) == digest, 'Admission input changed '+path)
    return dict(files=files, required_analysis_inputs=analysis_inputs,
                new_jobs=[job for job, _, _ in items], control_jobs=[c['job'] for c, _ in controls])


def prepare(base, root):
    base, root = Path(base).resolve(), Path(root).resolve()
    require(not root.exists(), 'Fresh postrun observer directory required; no retry')
    admitted = admit(base)  # The directory does not exist until all 64 complete.
    sources = source_closure(); files = dict(admitted['files'])
    root.mkdir(); (root/'code').mkdir()
    try:
        source_hashes = {}
        for name, path in sources.items():
            digest = sha(path); target = root/'code'/name
            shutil.copyfile(path, target); require(sha(target) == digest, 'Source changed while archiving')
            files[str(target)] = digest; source_hashes[name] = digest
        python = str(Path(sys.executable).absolute()); files[python] = sha(python)
        plan = dict(schema=SCHEMA, base=str(base), root=str(root),
            argv=[python, '-B', str(base/'common/source/tools'/ANALYZER), '--base', str(base), '--output', str(root/'analysis')],
            files=files, source_sha256=source_hashes, runtime=runtime(),
            new_jobs=admitted['new_jobs'], control_jobs=admitted['control_jobs'],
            required_analysis_inputs=admitted['required_analysis_inputs'], **SCOPE)
        for path, digest in files.items(): require(sha(path) == digest, 'Input changed during preparation '+path)
        write(root/'execution-plan.json', plan)
        return plan
    except BaseException as error:
        write(root/'preparation-failure.json', dict(complete=False, passed=False, error=repr(error),
            new_geometry_queries=0, partial_outputs_retained=True, retries=0))
        raise


def verify_plan(root, plan):
    root = Path(root).resolve(); base = Path(plan['base']).resolve()
    require(plan['schema'] == SCHEMA and plan['root'] == str(root)
        and all(plan.get(key) == value for key, value in SCOPE.items()), 'Changed observer scope or budget')
    require(plan['runtime'] == runtime(), 'Changed observer runtime')
    python = str(Path(sys.executable).absolute())
    require(plan['argv'] == [python, '-B', str(base/'common/source/tools'/ANALYZER),
        '--base', str(base), '--output', str(root/'analysis')], 'Changed frozen observer command')
    current = admit(base)
    require(current['new_jobs'] == plan['new_jobs'] and current['control_jobs'] == plan['control_jobs']
        and current['required_analysis_inputs'] == plan['required_analysis_inputs']
        and all(plan['files'].get(path) == digest for path, digest in current['files'].items()),
        'Omitted or changed observer admission input')
    sources = source_closure()
    require(set(plan['source_sha256']) == set(sources), 'Changed wrapper source inventory')
    for name, source in sources.items():
        archived = root/'code'/name
        require(plan['files'].get(str(archived)) == plan['source_sha256'][name] == sha(archived) == sha(source),
                'Changed archived wrapper source '+name)
    require(plan['files'].get(python) == sha(python), 'Unbound Python executable')
    for path, digest in plan['files'].items(): require(sha(path) == digest, 'Changed execution input '+path)


def verify_output(base, root, plan):
    result_path = root/'analysis/analysis.json'; result = read(result_path)
    require(result['schema'] == 'two-neighbor-singleton-analysis-v1' and result['complete'] is True
        and result['new_chains'] == 64 and result['reused_control_chains'] == 64 and len(result['chains']) == 128
        and result['new_geometry_endpoints'] == 294976 and result['new_physical_draws'] == 0
        and result['old_geometry_queries'] == 0 and result['native_observer'] is False,
        'Incomplete or out-of-scope observer result')
    new = [c for c in result['chains'] if c['reused_control'] is False]
    old = [c for c in result['chains'] if c['reused_control'] is True]
    require([c['job'] for c in new] == plan['new_jobs'] and [c['job'] for c in old] == plan['control_jobs'],
            'Observer changed exact new/control IDs or preparations')
    require(sum(c['metrics']['production_samples'] for c in new) == 262144
        and all(type(c['metrics']['production_samples']) is int and c['metrics']['production_samples'] == 4096 for c in new+old)
        and all(type(c['new_geometry_queries']) is int and c['new_geometry_queries'] == 0 for c in old),
        'Lost production samples or repeated control geometry')
    require(all(type(c['pair_classifications']) is int and 0 <= c['pair_classifications'] <= 4609*525 for c in new)
        and sum(c['pair_classifications'] for c in new) <= 154862400, 'Observer geometry allocation exceeded')
    require(result['analysis_plan'] == read(base/'analysis-plan.json'), 'Changed scientific observer plan')
    for path, digest in plan['required_analysis_inputs'].items():
        require(result['input_sha256'].get(path) == digest, 'Observer omitted bound analysis input '+path)
    for path, digest in result['input_sha256'].items():
        require(plan['files'].get(path) == digest, 'Observer used an unbound input '+path)
    output_binding = read(root/'analysis/input-binding.json')
    require(output_binding['input_sha256'] == result['input_sha256']
        and output_binding['plan'] == result['analysis_plan'], 'Observer input-binding/result differs')
    manifest_path = root/'analysis/manifest.json'; manifest = read(manifest_path)
    expected = {'analysis.json', 'input-binding.json'}|{f"job-{j['id']:03}-observations.jsonl" for j in plan['new_jobs']}
    require(manifest['complete'] is True and set(manifest['files']) == expected, 'Observer artifact inventory differs')
    for name, digest in manifest['files'].items():
        require(sha(root/'analysis'/name) == digest, 'Changed observer artifact '+name)
    return record(result_path), record(manifest_path)


def run(root):
    root = Path(root).resolve(); plan = read(root/'execution-plan.json'); digest = sha(root/'execution-plan.json')
    verify_plan(root, plan)
    require(sha(root/'execution-plan.json') == digest, 'Observer plan changed during admission')
    write(root/'claim.json', dict(pid=os.getpid(), started=time.time(), plan_sha256=digest, retries=0))
    handlers = {s: signal.signal(s, launcher.terminate_requested) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        require(not any((root/name).exists() for name in ('analysis', 'failure.json', 'preparation-failure.json')),
                'Prior observer work exists; no retry')
        write(root/'begin.json', dict(argv=plan['argv'], plan_sha256=digest, started=time.time(), **SCOPE))
        launcher.owned_child(plan['argv'], str(Path(plan['base'])/'common/source'), root, 3600, 7200, 16*1024**3)
        child = read(root/'exit.json')
        require(child['child_started'] is True and child['child_drained'] is True
            and child['returncode'] == 0 and child['error'] is None, 'Observer child did not exit cleanly')
        result, manifest = verify_output(Path(plan['base']), root, plan)
        verify_plan(root, plan); require(sha(root/'execution-plan.json') == digest, 'Plan changed during observation')
        write(root/'summary.json', dict(schema='two-neighbor-singleton-observer-completion-v1',
            complete=True, passed=True, base=plan['base'], analysis=result, manifest=manifest,
            plan_sha256=digest, assembly_gate_open=False, **SCOPE))
    except BaseException as error:
        write(root/'failure.json', dict(complete=False, passed=False, error=repr(error),
            traceback=traceback.format_exc(), plan_sha256=digest, retries=0, partial_outputs_retained=True))
        raise
    finally:
        for sig, handler in handlers.items(): signal.signal(sig, handler)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--base', type=Path); p.add_argument('--run', action='store_true'); args = p.parse_args()
    if args.run: run(args.root)
    else:
        if args.base is None: p.error('--base required for metadata-only preparation')
        plan = prepare(args.base, args.root)
        print(json.dumps(dict(prepared=True, files=len(plan['files']), launched=False, new_geometry_queries=0)))
