#!/usr/bin/env python3
"""Bind all 96 completed chains, then run the frozen observer once.

Preparation only reads metadata and hashes files. Geometry is evaluated solely
by the existing, frozen analyzer, after every chain has completed successfully.
Interrupted/failed analysis is retained and cannot be silently restarted.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
import traceback


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def complete_jobs(config, status):
    """Do not interpret a partial, failed, duplicated or modified job inventory."""
    jobs = config['jobs']
    require(len(jobs) == 96 and len({j['id'] for j in jobs}) == 96,
            'Expected the complete 96-chain allocation')
    families = {(j['context_index'], j['arm'], j['initialization'], j['stream']) for j in jobs}
    require(families == {(c, a, i, s) for c in range(4)
        for a in ('local', 'unguided', 'm4') for i in ('source', 'proposal_prepared') for s in range(4)},
        'Changed context/arm/initialization/stream allocation')
    require(status.get('complete') is True and status.get('passed') is True
            and status.get('failure_draining') is False and status.get('active') == []
            and status.get('unstarted') == [], 'All 96 successful terminals are required')
    completed = status['completed']
    require(len(completed) == 96 and len({r['job']['id'] for r in completed}) == 96,
            'Missing or repeated completed job')
    by_id = {j['id']: j for j in jobs}
    for record in completed:
        require(record['job'] == by_id.get(record['job']['id']) and record.get('success') is True
                and record.get('returncode') == 0 and record.get('terminal_error') is None,
                'Failed or changed completed job')
    return sorted(completed, key=lambda r: r['job']['id'])


def campaign_bindings(base):
    """Metadata-only closure check; no imports of scientific observers."""
    base = Path(base).resolve()
    bindings = {}

    def bind(path, digest=None):
        path = Path(path).resolve()
        actual = sha(path)
        require(digest is None or actual == digest, 'Changed bound file: '+str(path))
        require(str(path) not in bindings or bindings[str(path)] == actual,
                'Input changed during binding: '+str(path))
        bindings[str(path)] = actual
        return path

    def nested(value):
        if isinstance(value, dict):
            if 'path' in value and 'sha256' in value:
                bind(value['path'], value['sha256'])
            for child in value.values():
                nested(child)
        elif isinstance(value, list):
            for child in value:
                nested(child)

    status_path = bind(base/'dispatch/status.json')
    status = read(status_path)
    config_path = bind(base/'config.json')
    config = read(config_path)
    completed = complete_jobs(config, status)
    bind(base/'dispatch/plan.json', status['plan_sha256'])
    dispatch = read(base/'dispatch/plan.json')
    require(dispatch['jobs'] == config['jobs'], 'Dispatch allocation changed')
    bind(config_path, dispatch['config_sha256'])
    bind(base/'run-binding.json', dispatch['binding_sha256'])
    bind(base/'common/evolving_dimer_benchmark', dispatch['executable_sha256'])
    bind(base/'dispatch/run.py', dispatch['source_sha256'])
    nested(dispatch['review'])
    review = read(dispatch['review']['path'])
    require(review['complete'] is True and review['passed'] is True, 'Prelaunch review failed')
    for path, digest in review['input_sha256'].items():
        bind(path, digest)
    frozen = read(bind(base/'freeze.json'))
    require(frozen['complete'] is True and frozen['scientific_execution_started'] is False,
            'Invalid preparation freeze')
    for name, digest in frozen['files'].items():
        path = (base/name).resolve()
        require(path.is_relative_to(base), 'Frozen file outside campaign')
        bind(path, digest)
    nested(frozen['original_input_bindings'])
    nested(config)
    binding_path = bind(base/'run-binding.json')
    binding = read(binding_path)
    require(binding['config_sha256'] == sha(config_path), 'Run/config mismatch')
    bind(base/'protocol.json', binding['protocol_sha256'])
    nested(binding)
    for path, digest in binding['prepared_files'].items():
        bind(path, digest)
    prepared = read(binding['prepared_manifest']['path'])
    require(prepared['complete'] is True and prepared['passed'] is True
            and prepared['all_attempts_retained'] is True, 'Incomplete preparation')
    nested(prepared)
    protocol = read(base/'protocol.json')
    for name, digest in protocol['source_files'].items():
        bind(base/'common/source'/name, digest)
    analyzer = base/'common/source/tools/analyze_evolving_dimer_benchmark.py'
    require(str(analyzer) in bindings, 'Frozen analyzer not bound')
    run_output = Path(config['output']).resolve()
    for record in completed:
        directory = run_output/f"job-{record['job']['id']:03}"
        terminal_path = bind(directory/'terminal.json', record['terminal_sha256'])
        terminal = read(terminal_path)
        require(terminal['complete'] is True and terminal['job'] == record['job']
                and terminal['conditional_target'] is True
                and terminal['blocks'] == config['allocation']['blocks_per_chain']
                and terminal['config_sha256'] == sha(config_path)
                and terminal['binding_sha256'] == sha(binding_path), 'Terminal binding differs')
        require(not (directory/'failure.json').exists(), 'Failed chain has a terminal')
        trajectory = bind(terminal['trajectory']['path'], terminal['trajectory']['sha256'])
        require(trajectory == directory/'trajectory.jsonl', 'Unexpected trajectory path')
    require(sha(status_path) == bindings[str(status_path)], 'Dispatch changed during binding')
    return bindings


def prepare(base, root, cpu_seconds=86400, wall_seconds=172800):
    base, root = Path(base).resolve(), Path(root).resolve()
    require(not root.exists(), 'Fresh analysis execution directory required')
    require(type(cpu_seconds) is int and 0 < cpu_seconds <= 86400
            and type(wall_seconds) is int and cpu_seconds <= wall_seconds <= 172800,
            'Analysis budget exceeds one CPU-day/two wall-days')
    files = campaign_bindings(base)
    python = str(Path(sys.executable).absolute())
    files[python] = sha(python)
    root.mkdir()
    runner = root/'runner.py'
    runner.write_bytes(Path(__file__).read_bytes())
    files[str(runner)] = sha(runner)
    plan = dict(schema='evolving-dimer-analysis-execution-v1', base=str(base), root=str(root),
        argv=[python, '-B', str(base/'common/source/tools/analyze_evolving_dimer_benchmark.py'),
              '--base', str(base), '--output', str(root/'analysis')],
        files=files, maximum_workers=1, cpu_limit_seconds=cpu_seconds,
        wall_limit_seconds=wall_seconds, address_space_limit_bytes=16*1024**3,
        python_version=sys.version,
        packages={name:importlib.metadata.version(name) for name in ('numpy', 'scipy')},
        expected_chains=96, expected_retained_endpoints=442464,
        expected_production_endpoints=393216, new_sampling_attempts=0,
        restart=False, partial_analysis_allowed=False,
        failure='Kill and reap owned child process group; retain partial outputs; no retry.',
        scope='Frozen conditional contact-efficiency observer; no native or finite-system conclusion.')
    write(root/'execution-plan.json', plan)
    return plan


def verify_plan(root, plan):
    require(plan['schema'] == 'evolving-dimer-analysis-execution-v1' and plan['root'] == str(root)
            and plan['maximum_workers'] == 1 and plan['expected_chains'] == 96
            and plan['new_sampling_attempts'] == 0 and plan['restart'] is False
            and plan['partial_analysis_allowed'] is False
            and plan['expected_retained_endpoints'] == 442464
            and plan['expected_production_endpoints'] == 393216,
            'Changed analysis execution scope')
    base = Path(plan['base'])
    require(plan['argv'][0] == str(Path(sys.executable).absolute()), 'Changed Python interpreter')
    require(plan['argv'][1:] == ['-B', str(base/'common/source/tools/analyze_evolving_dimer_benchmark.py'),
        '--base', str(base), '--output', str(root/'analysis')], 'Changed analysis command')
    require(0 < plan['cpu_limit_seconds'] <= 86400
            and plan['cpu_limit_seconds'] <= plan['wall_limit_seconds'] <= 172800, 'Changed runtime budget')
    require(plan['address_space_limit_bytes'] == 16*1024**3 and plan['python_version'] == sys.version
            and plan['packages'] == {name:importlib.metadata.version(name) for name in ('numpy', 'scipy')},
            'Changed memory budget or Python environment')
    current = campaign_bindings(base)
    require(all(plan['files'].get(path) == digest for path, digest in current.items()),
            'Frozen analysis plan omits/changes campaign input')
    for path, digest in plan['files'].items():
        if path not in current:
            require(sha(path) == digest, 'Execution input changed: '+path)
    require(str(root/'runner.py') in plan['files'] and plan['argv'][0] in plan['files'],
            'Unbound wrapper or interpreter')
    complete_jobs(read(base/'config.json'), read(base/'dispatch/status.json'))


def terminate_requested(signum, _frame):
    raise SystemExit('Analysis controller received signal '+str(signum))


def drain(child):
    handlers = {s: signal.signal(s, signal.SIG_IGN) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        while True:
            try:
                return child.wait()
            except KeyboardInterrupt:
                continue
    finally:
        for signum, handler in handlers.items():
            signal.signal(signum, handler)


def owned_child(argv, cwd, directory, cpu_seconds, wall_seconds, address_space_bytes=16*1024**3):
    """Record and drain exactly one child, including timeouts and stop signals."""
    env = dict(os.environ)
    for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS'):
        env[key] = '1'
    prior_mask = None
    def limits():
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds+1))
        resource.setrlimit(resource.RLIMIT_AS, (address_space_bytes, address_space_bytes))
        os.nice(10)
        # Popen inherits the parent's short launch mask. Restore it before exec.
        signal.pthread_sigmask(signal.SIG_SETMASK, prior_mask)
    child = None
    error = None
    started = time.monotonic()
    try:
        with (directory/'output.log').open('x') as log:
            # A pending stop must arrive only after the child handle is owned.
            # Otherwise an exception between fork and assignment can orphan it.
            prior_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM})
            try:
                child = subprocess.Popen(argv, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT,
                    preexec_fn=limits, start_new_session=True)
            finally:
                signal.pthread_sigmask(signal.SIG_SETMASK, prior_mask)
            try:
                birth = Path(f'/proc/{child.pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
            except FileNotFoundError:
                birth = None
            write(directory/'child.json', dict(pid=child.pid, birth_ticks=birth, started=time.time()))
            child.wait(timeout=wall_seconds)
        require(child.returncode == 0, 'Analysis child failed with code '+str(child.returncode))
    except BaseException as caught:
        error = caught
        if child is not None:
            drain(child)
        raise
    finally:
        write(directory/'exit.json', dict(returncode=None if child is None else child.returncode,
            child_started=child is not None, child_drained=child is None or child.returncode is not None,
            error=None if error is None else repr(error), wall_seconds=time.monotonic()-started))


def run(root):
    root = Path(root).resolve()
    plan_digest = sha(root/'execution-plan.json')
    plan = read(root/'execution-plan.json')
    verify_plan(root, plan)
    require(sha(root/'execution-plan.json') == plan_digest, 'Plan changed during verification')
    write(root/'claim.json', dict(pid=os.getpid(), started=time.time(),
        plan_sha256=plan_digest, retry=False))
    previous = signal.signal(signal.SIGTERM, terminate_requested)
    try:
        require(not (root/'analysis').exists(), 'Analysis already started; no replay')
        owned_child(plan['argv'], plan['base'], root, plan['cpu_limit_seconds'], plan['wall_limit_seconds'],
                    plan['address_space_limit_bytes'])
        result_path = root/'analysis/analysis.json'
        result = read(result_path)
        require(result['complete'] is True and len(result['chains']) == plan['expected_chains'],
                'Incomplete analysis terminal')
        require([c['job'] for c in result['chains']] == read(Path(plan['base'])/'config.json')['jobs'],
                'Analysis job inventory differs')
        require(sum(c['metrics']['production_samples'] for c in result['chains']) ==
                plan['expected_production_endpoints'], 'Analysis lost production endpoints')
        manifest = read(root/'analysis/manifest.json')
        expected_files = {'analysis.json', 'input-binding.json'} | {
            f"job-{c['job']['id']:03}-observations.jsonl" for c in result['chains']}
        require(manifest['complete'] is True and set(manifest['files']) == expected_files,
                'Incomplete analysis manifest')
        for name, digest in manifest['files'].items():
            path = (root/'analysis'/name).resolve()
            require(path.is_relative_to(root/'analysis') and sha(path) == digest,
                    'Analysis result hash differs')
        verify_plan(root, plan)
        require(sha(root/'execution-plan.json') == plan_digest, 'Plan changed during execution')
        write(root/'summary.json', dict(complete=True, passed=True, chains=96,
            analysis=dict(path=str(result_path), sha256=sha(result_path)),
            manifest_sha256=sha(root/'analysis/manifest.json'), plan_sha256=plan_digest,
            new_sampling_attempts=0, native_observer=False, assembly_gate_open=False))
    except BaseException as error:
        write(root/'failure.json', dict(complete=False, passed=False, error=repr(error),
            traceback=traceback.format_exc(), retries=0, partial_outputs_retained=True))
        raise
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--base', type=Path)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if args.run:
        run(args.root)
    else:
        if args.base is None:
            parser.error('--base is required for metadata-only preparation')
        plan = prepare(args.base, args.root)
        print(json.dumps(dict(ready=True, bound_files=len(plan['files']), analysis_started=False)))
