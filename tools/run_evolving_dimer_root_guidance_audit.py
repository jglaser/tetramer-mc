#!/usr/bin/env python3
"""Freeze all32 completed journals, then launch one bounded guidance auditor.

Preparation hashes metadata and saved bytes only. The separately reviewed run
imports the campaign's archived auditor, reconstructs its predeclared first
event per chain, and checks every other event algebraically. No replacements,
new point clouds, new poses, or automatic continuation are permitted.
"""
from __future__ import annotations
import argparse
import ast
from collections import Counter
import hashlib
import importlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import resource
import signal
import sys
import time
import traceback

SCHEMA = 'evolving-dimer-root-guidance-audit-execution-v1'
NAME = 'run_evolving_dimer_root_guidance_audit.py'
AUDITOR = 'audit_evolving_dimer_root_guidance.py'
LAUNCHER = 'run_evolving_dimer_analysis.py'
CPU, WALL, MEMORY = 1800, 3600, 16*1024**3


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')
        stream.flush(); os.fsync(stream.fileno())


def event(stream, value):
    stream.write(json.dumps(value, allow_nan=False)+'\n'); stream.flush(); os.fsync(stream.fileno())


def runtime():
    return dict(python=sys.version, executable=str(Path(sys.executable).resolve()),
                packages={name: version(name) for name in ('numpy', 'scipy')})


def check_protocol(p):
    require(p['schema'] == 'evolving-dimer-root-guidance-audit-protocol-v1'
        and p['chains'] == 32 and p['arm'] == 'root_m4'
        and p['geometric_event']['kind'] == 'factorized_dimer'
        and p['geometric_event']['block'] == 1 and p['geometric_event']['after_local_attempts'] == 4
        and p['geometric_event']['replace_skipped'] is False
        and p['maximum_count_queries'] == 2560 and p['maximum_point_membership_tests'] == 41943040
        and p['maximum_raw_root_trials'] == p['maximum_raw_internal_trials'] == 1024
        and p['maximum_source_frames'] == p['maximum_endpoint_frames'] == 32
        and p['maximum_full_fingerprint_evaluations'] == 128
        and p['maximum_root_hard_predicate_evaluations'] == p['maximum_internal_predicate_evaluations'] == 1088
        and p['scalar_dimer_attempts'] == 147456 and p['scalar_local_attempts'] == 589824
        and p['workers'] == 1 and p['cpu_seconds'] == CPU and p['wall_seconds'] == WALL
        and p['memory_bytes'] == MEMORY
        and all(p[k] == 0 for k in ('new_poses', 'new_clouds', 'repeated_cloud_thinning')),
        'Changed frozen guidance audit allocation')


def campaign_bindings(base):
    """No row parsing or scientific imports: authenticate complete authority."""
    base = Path(base).resolve(); files = {}
    def bind(path, expected=None):
        path = Path(path).resolve(); digest = sha(path)
        require(expected is None or digest == expected, 'Changed bound input '+str(path))
        require(str(path) not in files or files[str(path)] == digest, 'Input changed during binding')
        files[str(path)] = digest
        return path
    def bound(value):
        require(set(value) == {'path', 'sha256'}, 'Invalid BoundFile')
        return bind(value['path'], value['sha256'])
    def nested(value):
        if isinstance(value, dict):
            if set(value) == {'path', 'sha256'}:
                bound(value)
            else:
                for child in value.values(): nested(child)
        elif isinstance(value, list):
            for child in value: nested(child)
    config = read(bind(base/'config.json')); protocol = read(bind(base/'protocol.json'))
    require(config['schema'] == 'evolving-dimer-benchmark-v1'
        and protocol['schema'] == 'evolving-dimer-root-extension-protocol-v1', 'Wrong campaign schema')
    frozen = read(bind(base/'freeze.json'))
    require(frozen['complete'] is True and frozen['scientific_execution_started'] is False
            and frozen['schema'] == 'evolving-dimer-root-extension-freeze-v1', 'Wrong preparation freeze')
    for name, digest in frozen['files'].items():
        path = (base/name).resolve()
        require(path.is_relative_to(base), 'Frozen source escaped campaign')
        bind(path, digest)
    for path, digest in frozen['input_sha256'].items(): bind(path, digest)
    nested(config)
    jobs = config['jobs']
    require(len(jobs) == 32 and len({j['id'] for j in jobs}) == 32
        and all(j['arm'] == 'root_m4' for j in jobs)
        and {(j['context_index'], j['initialization'], j['stream']) for j in jobs}
        == {(c, i, s) for c in range(4) for i in ('source', 'proposal_prepared') for s in range(4)}, 'Changed32-chain inventory')
    require(config['allocation']['warmup_blocks'] == 512 and config['allocation']['production_blocks'] == 4096
            and config['allocation']['local_attempts'] == 589824
            and config['allocation']['dimer_attempts'] == 147456, 'Changed production schedule')
    require(config['physical'] == dict(depletant_radius=1.4, activity=.0275, lambda_ratio=64.,
        wall_radius=593.742500239952), 'Changed physical target')
    audit_protocol = read(bound(protocol['root_guidance_audit_plan']))
    require(protocol['root_guidance_audit_plan']['path'] == str(base/'root-guidance-audit-plan.json'), 'Unexpected audit plan path')
    check_protocol(audit_protocol)
    for name, digest in protocol['source_files'].items():
        path = (base/'common/source'/name).resolve()
        require(path.is_relative_to(base/'common/source'), 'Source escaped archive')
        bind(path, digest)
    require(str(base/'common/source/tools'/AUDITOR) in files, 'Auditor was not frozen before production')
    binding = read(bind(base/'binding.json')); run_binding = read(bind(base/'run-binding.json'))
    for key, name in [('config_sha256', 'config.json'), ('protocol_sha256', 'protocol.json'),
        ('freeze_sha256', 'freeze.json'), ('executable_sha256', 'common/evolving_dimer_benchmark'),
        ('compiled_source_bundle_sha256', 'common/source-bundle.json'),
        ('example_source_sha256', 'common/source/examples/evolving_dimer_benchmark.rs')]:
        bind(base/name, binding[key])
    require(run_binding['complete'] is True and run_binding['config_sha256'] == files[str(base/'config.json')]
        and run_binding['protocol_sha256'] == files[str(base/'protocol.json')]
        and run_binding['prelaunch_binding'] == dict(path=str(base/'binding.json'), sha256=files[str(base/'binding.json')]), 'Run binding differs')
    nested(run_binding)
    authority = config['inherited_campaign']
    require(run_binding['prepared_manifest'] == authority['prepared_manifest'], 'Changed inherited preparation')
    prepared = read(bound(authority['prepared_manifest'])); old_audit = read(bound(authority['preparation_audit']))
    require(prepared['complete'] is True and prepared['passed'] is True and prepared['all_attempts_retained'] is True
        and prepared['config_sha256'] == authority['config']['sha256']
        and prepared['binding_sha256'] == authority['binding']['sha256'], 'Original prepared identity differs')
    require(old_audit['schema'] == 'evolving-dimer-preparation-independent-audit-v1'
        and old_audit['complete'] is True and old_audit['passed'] is True
        and old_audit['physical'] == config['physical'] and old_audit['source_contexts'] == config['contexts'], 'Original preparation audit is incomplete or mismatched')
    for key in ('config', 'binding', 'prepared_manifest'):
        value = authority[key]
        require(old_audit['input_sha256'].get(value['path']) == value['sha256'], 'Original audited authority differs')
    bound(old_audit['protocol'])
    inventory = {f['path']: f['sha256'] for f in prepared['files']}
    require(len(inventory) == len(prepared['files']) == 96 and inventory == run_binding['prepared_files'], 'Prepared inventory differs')
    for path, digest in inventory.items():
        require(old_audit['input_sha256'].get(path) == digest, 'Cloud/start file lacks original independent audit')
        bind(path, digest)
    banks = {(b['context_index'], b['initialization'], b['stream']): b for b in prepared['cloud_banks']}
    require(len(prepared['cloud_banks']) == len(banks) == 32
        and set(banks) == {(j['context_index'], j['initialization'], j['stream']) for j in jobs}, 'Prepared bank family differs')
    starts = {(s['context_index'], s['stream']): s for s in prepared['alternative_starts']}
    require(len(prepared['alternative_starts']) == len(starts) == 16
        and set(starts) == {(c, s) for c in range(4) for s in range(4)}, 'Prepared start family differs')
    for group, names in [(banks.values(), ('raw', 'metadata')), (starts.values(), ('record', 'ledger'))]:
        for item in group:
            for name in names:
                f = item[name]
                require(inventory.get(f['path']) == f['sha256'], 'Nested prepared file escaped inventory')
    status = read(bind(base/'dispatch/status.json'))
    require(status['complete'] is True and status['passed'] is True and status['failure'] is None
        and status['failure_draining'] is False and status['active'] == [] and status['unstarted'] == [], 'All32 completed chains required before preparation')
    dispatch = read(bind(base/'dispatch/plan.json', status['plan_sha256']))
    require(dispatch['jobs'] == jobs and dispatch['retries'] is False and dispatch['replacements'] is False, 'Dispatch inventory differs')
    nested(dispatch)
    for path, digest in dispatch['input_sha256'].items(): bind(path, digest)
    review = read(bound(dispatch['review']))
    require(review['complete'] is True and review['passed'] is True, 'Prelaunch review failed')
    for path, digest in review['input_sha256'].items(): bind(path, digest)
    completed = {c['job']['id']: c for c in status['completed']}
    require(len(status['completed']) == len(completed) == 32 and set(completed) == {j['id'] for j in jobs}, 'Missing/repeated completed chain')
    require(Path(config['output']).resolve() == base/'execution', 'Unexpected execution output')
    output = []
    for job in jobs:
        done = completed[job['id']]
        require(done['job'] == job and done['success'] is True and done['returncode'] == 0
            and done['error'] is None and done['child_drained'] is True, 'Failed/mismatched terminal chain')
        directory = base/'execution'/f"job-{job['id']:03}"
        terminal_path = bind(directory/'terminal.json', done['terminal_sha256']); terminal = read(terminal_path)
        require(terminal['complete'] is True and terminal['conditional_target'] is True and terminal['job'] == job
            and terminal['blocks'] == 4608 and terminal['config_sha256'] == files[str(base/'config.json')]
            and terminal['binding_sha256'] == files[str(base/'run-binding.json')]
            and terminal['counts']['local_attempted'] == 18432 and terminal['counts']['dimer_attempted'] == 4608
            and not (directory/'failure.json').exists(), 'Terminal provenance/schedule differs')
        trajectory = bound(terminal['trajectory'])
        require(trajectory == directory/'trajectory.jsonl', 'Trajectory escaped own chain')
        bank = banks[job['context_index'], job['initialization'], job['stream']]
        start = starts[job['context_index'], job['stream']] if job['initialization'] == 'proposal_prepared' else None
        output.append(dict(job=job, terminal=dict(path=str(terminal_path), sha256=files[str(terminal_path)]),
            trajectory=terminal['trajectory'], bank=bank, start=start))
    return dict(files=files, jobs=output, audit_protocol=audit_protocol)


def source_closure(source_root, names):
    found = {}; pending = [source_root/name for name in names]
    while pending:
        p = pending.pop().resolve()
        if p.name in found:
            require(found[p.name] == p, 'Ambiguous source name'); continue
        require(p.is_file(), 'Missing local source '+str(p)); found[p.name] = p
        for node in ast.walk(ast.parse(p.read_text(), filename=str(p))):
            names = ([v.name for v in node.names] if isinstance(node, ast.Import) else
                     [node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
            for name in names:
                q = p.parent/(name.split('.')[0]+'.py')
                if q.is_file(): pending.append(q)
    return found


def prepare(base, root):
    base, root = Path(base).resolve(), Path(root).resolve()
    require(not root.exists(), 'Fresh audit output required')
    frozen = campaign_bindings(base)  # No source or geometry imports.
    campaign_source = base/'common/source/tools'
    sources = source_closure(campaign_source, [AUDITOR, 'test_audit_evolving_dimer_root_guidance.py', LAUNCHER])
    for name in (NAME, 'test_run_evolving_dimer_root_guidance_audit.py'):
        require(name not in sources, 'Unexpected wrapper in old archive')
        sources[name] = Path(__file__).resolve().parent/name
    root.mkdir(); (root/'code').mkdir(); (root/'child').mkdir()
    files = dict(frozen['files']); source_hashes = {}
    try:
        for name, path in sources.items():
            digest = sha(path); target = root/'code'/name
            with target.open('xb') as f:
                f.write(path.read_bytes()); f.flush(); os.fsync(f.fileno())
            require(sha(target) == digest, 'Source changed during archival')
            files[str(target)] = digest; source_hashes[name] = digest
        python = str(Path(sys.executable).absolute()); files[python] = sha(python)
        plan = dict(schema=SCHEMA, base=str(base), root=str(root), jobs=frozen['jobs'], files=files,
            source_sha256=source_hashes, audit_protocol=frozen['audit_protocol'], runtime=runtime(),
            argv=[python, '-B', str(root/'code'/NAME), '--root', str(root), '--worker'],
            maximum_workers=1, cpu_seconds=CPU, wall_seconds=WALL, memory_bytes=MEMORY,
            require_review=True, retries=0, replacements=0, new_poses=0, new_clouds=0,
            original_preparation_reused=True, completed_inventory_required=True,
            scope='All32 completed root_m4 journals. Independent geometry only for block1 collective events; other rows scalar-only. No assembly/equilibrium claim.')
        for path, digest in files.items(): require(sha(path) == digest, 'Input changed while preparing')
        write(root/'execution-plan.json', plan)
        return plan
    except BaseException as error:
        write(root/'preparation-failure.json', dict(complete=False, error=repr(error), new_geometry_queries=0))
        raise


def verify_plan(root, plan, digest=None):
    """Metadata-only admission, also reusable by the later observer wrapper."""
    root = Path(root).resolve()
    digest = sha(root/'execution-plan.json') if digest is None else digest
    require(sha(root/'execution-plan.json') == digest and plan['schema'] == SCHEMA
        and plan['root'] == str(root) and plan['runtime'] == runtime(), 'Changed audit plan/runtime')
    require(plan['maximum_workers'] == 1 and plan['cpu_seconds'] == CPU and plan['wall_seconds'] == WALL
        and plan['memory_bytes'] == MEMORY and plan['require_review'] is True
        and all(plan[k] == 0 for k in ('retries', 'replacements', 'new_poses', 'new_clouds')),
        'Changed audit execution limits')
    check_protocol(plan['audit_protocol'])
    python = str(Path(sys.executable).absolute())
    require(plan['argv'] == [python, '-B', str(root/'code'/NAME), '--root', str(root), '--worker'], 'Changed worker command')
    require(len(plan['jobs']) == 32 and len({j['job']['id'] for j in plan['jobs']}) == 32, 'Wrong audit job allocation')
    current = campaign_bindings(Path(plan['base']))
    require(current['jobs'] == plan['jobs'] and current['audit_protocol'] == plan['audit_protocol']
        and all(plan['files'].get(p) == h for p, h in current['files'].items()), 'Audit input closure incomplete')
    for name in (AUDITOR, LAUNCHER, NAME): require(name in plan['source_sha256'], 'Missing execution source')
    for name, expected in plan['source_sha256'].items():
        path = root/'code'/name
        require(path.resolve().parent == root/'code' and plan['files'].get(str(path)) == expected, 'Incomplete archived source binding')
    for path, expected in plan['files'].items(): require(sha(path) == expected, 'Changed audit input '+path)


def verify(root, plan, digest):
    require(Path(__file__).resolve() == root/'code'/NAME, 'Execute frozen wrapper only')
    verify_plan(root, plan, digest)


def checked_import(name, root, plan):
    module = importlib.import_module(name)
    expected = root/'code'/(name+'.py')
    require(Path(module.__file__).resolve() == expected and sha(expected) == plan['source_sha256'][expected.name], 'Imported unbound source '+name)
    return module


def terminate(signum, _frame):
    raise SystemExit('Audit interrupted by signal '+str(signum))


def run(root):
    root = Path(root).resolve(); digest = sha(root/'execution-plan.json'); plan = read(root/'execution-plan.json')
    verify(root, plan, digest)
    review = read(root/'review.json')
    require(review.get('complete') is True and review.get('passed') is True
        and review.get('plan_sha256') == digest and review.get('chains') == 32
        and review.get('maximum_count_queries') == 2560, 'Matching independent launch review required')
    write(root/'claim.json', dict(pid=os.getpid(), plan_sha256=digest,
        review_sha256=sha(root/'review.json'), started=time.time(), retries=0))
    handlers = {s: signal.signal(s, terminate) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        launcher = checked_import('run_evolving_dimer_analysis', root, plan)
        write(root/'child/begin.json', dict(argv=plan['argv'], plan_sha256=digest,
            cpu_seconds=CPU, wall_seconds=WALL, memory_bytes=MEMORY))
        launcher.owned_child(plan['argv'], str(root/'code'), root/'child', CPU, WALL, MEMORY)
        result = read(root/'audit.json')
        require(result['complete'] is True and result['passed'] is True and result['plan_sha256'] == digest
            and [r['job'] for r in result['chains']] == [j['job'] for j in plan['jobs']]
            and result['scalar_dimer_attempts'] == 147456 and result['scalar_local_attempts'] == 589824
            and result['independent_geometry_events'] == 32, 'Incomplete audit result')
        require(result['count_queries_completed'] == result['count_queries_started'] <= 2560
            and result['point_membership_tests_started'] <= 41943040, 'Count allocation exceeded')
        require(result['journal_sha256'] == sha(root/'attempts.jsonl'), 'Audit journal changed')
        verify(root, plan, digest)
        write(root/'summary.json', dict(schema='evolving-dimer-root-guidance-audit-summary-v1',
            complete=True, passed=True, base=plan['base'], plan_sha256=digest, chains=32,
            result=dict(path=str(root/'audit.json'), sha256=sha(root/'audit.json')),
            audit_sha256=sha(root/'audit.json'), journal_sha256=result['journal_sha256'],
            scalar_dimer_attempts=147456, scalar_local_attempts=589824,
            geometric_events=32, independent_geometry_events=32,
            count_queries=result['count_queries_completed'], new_poses=0, new_clouds=0,
            assembly_gate_open=False))
    except BaseException as error:
        write(root/'failure.json', dict(complete=False, passed=False, error=repr(error),
            traceback=traceback.format_exc(), partial_outputs_retained=True, retries=0))
        raise
    finally:
        for s, handler in handlers.items(): signal.signal(s, handler)


class CountBudget:
    """Reserve every complete point-array query before entering the reference."""
    def __init__(self, oracle, on_event=None):
        self.oracle = oracle; self.started = self.completed = self.point_tests = 0
        self.on_event = on_event

    def _call(self, name, points, pose):
        require(self.started < 2560 and self.point_tests+len(points) <= 41943040, 'Independent count budget exhausted')
        self.started += 1; self.point_tests += len(points)
        if self.on_event is not None:
            self.on_event(dict(state='count_begin', ordinal=self.started, operation=name,
                points=len(points), point_membership_tests_started=self.point_tests))
        value = getattr(self.oracle, name)(points, pose)
        self.completed += 1
        if self.on_event is not None:
            self.on_event(dict(state='count_complete', ordinal=self.started, operation=name,
                count=value, count_queries_completed=self.completed))
        return value

    def relative(self, points, pose): return self._call('relative', points, pose)
    def world(self, points, members): return self._call('world', points, members)


def worker(root):
    root = Path(root).resolve(); digest = sha(root/'execution-plan.json'); plan = read(root/'execution-plan.json')
    verify(root, plan, digest)
    claim = read(root/'claim.json')
    require(claim['plan_sha256'] == digest and claim['review_sha256'] == sha(root/'review.json')
            and claim['pid'] == os.getppid(), 'Worker lacks owning controller claim')
    require(resource.getrlimit(resource.RLIMIT_CPU) == (CPU, CPU+1)
            and resource.getrlimit(resource.RLIMIT_AS) == (MEMORY, MEMORY), 'Worker resource limits differ')
    write(root/'worker-claim.json', dict(pid=os.getpid(), plan_sha256=digest, started=time.time(), retries=0))
    handlers = {s: signal.signal(s, terminate) for s in (signal.SIGINT, signal.SIGTERM)}
    current = None; budget = None; reports = []; totals = Counter(); started = time.process_time()
    with (root/'attempts.jsonl').open('x') as journal:
        try:
            module = checked_import('audit_evolving_dimer_root_guidance', root, plan)
            require(module.audit_protocol() == plan['audit_protocol'], 'Imported auditor protocol differs')
            # Every local module must actually resolve to the bound flat archive.
            for name in plan['source_sha256']:
                loaded = sys.modules.get(name.removesuffix('.py'))
                if loaded is not None and hasattr(loaded, '__file__'):
                    require(Path(loaded.__file__).resolve() == root/'code'/name, 'Imported closure escaped archive')
            base = Path(plan['base']); config = read(base/'config.json')
            shape = read(config['shape']['path']); source = read(config['source_frame']['path'])['poses']
            event(journal, dict(state='setup_begin', geometry_objects=1, count_oracles=1, plan_sha256=digest))
            geometry = module.DimerGeometry(shape, source, config['physical']['depletant_radius'], config['physical']['wall_radius'])
            budget = CountBudget(module.CountOracle(geometry.centers, geometry.radii+geometry.rd),
                on_event=lambda value: event(journal, dict(value, job_id=current)))
            event(journal, dict(state='setup_complete', geometry_objects=1, count_oracles=1, count_queries=0))
            for item in plan['jobs']:
                current = item['job']['id']; terminal = read(item['terminal']['path'])
                case = config['contexts'][item['job']['context_index']]
                initial = [source[case[k]] for k in ('root', 'child')]
                if item['start'] is not None: initial = read(item['start']['record']['path'])['selected']
                event(journal, dict(state='begin', job=item['job'], trajectory=item['trajectory'],
                    selected_geometric_block=1, count_queries_started=budget.started))
                points = module.retained_points(item['bank']); checks = module.Checks()
                row_digest = hashlib.sha256(); row_count = 0
                def rows(stream):
                    nonlocal row_count
                    for raw in stream:
                        row_digest.update(raw); row_count += 1
                        require(raw.endswith(b'\n'), 'Partial saved trajectory line')
                        yield json.loads(raw, parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON '+value)))
                with Path(item['trajectory']['path']).open('rb') as stream:
                    report = module.audit_chain(config, item['job'], initial, terminal, rows(stream),
                        bank=item['bank'], counter=budget, geometry=geometry, points=points, checks=checks)
                require(row_digest.hexdigest() == item['trajectory']['sha256'] and row_count == 1+4608*6, 'Saved trajectory bytes/row count differ')
                require(not checks.failures, 'Independent checks failed')
                totals.update(report['first_event_geometry']); reports.append(report)
                require(totals['count_queries'] == budget.completed, 'Independent query accounting differs')
                event(journal, dict(state='complete', job=item['job'], rows=row_count,
                    trajectory_sha256=row_digest.hexdigest(), report=report, checks=checks.count,
                    maximum_absolute_errors=dict(checks.maximum_absolute_errors)))
            require(len(reports) == 32 and totals['source_frames'] == 32 and totals['endpoint_frames'] <= 32
                and totals['root_raw_trials'] <= 1024 and totals['internal_raw_trials'] <= 1024,
                'Geometry event allocation differs')
            require(budget.started == budget.completed == totals['count_queries'] <= 2560
                and budget.point_tests == totals['point_membership_tests'] <= 41943040, 'Point-query accounting differs')
            scalar_dimer = sum(r['scalar_counts']['scalar_dimer_attempts'] for r in reports)
            scalar_local = sum(r['scalar_counts']['scalar_local_attempts'] for r in reports)
            require(scalar_dimer == 147456 and scalar_local == 589824, 'Scalar attempt allocation differs')
            verify(root, plan, digest)
            event(journal, dict(state='all_complete', chains=32, count_queries=budget.completed))
            write(root/'audit.json', dict(complete=True, passed=True, plan_sha256=digest, chains=reports,
                scalar_dimer_attempts=scalar_dimer, scalar_local_attempts=scalar_local,
                independent_geometry_events=32, geometry_totals=dict(totals),
                count_queries_started=budget.started, count_queries_completed=budget.completed,
                point_membership_tests_started=budget.point_tests, journal_sha256=sha(root/'attempts.jsonl'),
                cpu_seconds=time.process_time()-started, new_poses=0, new_clouds=0,
                trust_boundary=plan['audit_protocol']['trust_boundary']))
        except BaseException as error:
            failure = dict(state='failure', complete=False, passed=False, plan_sha256=digest,
                error=repr(error), current_job=current, completed_chains=len(reports),
                count_queries_started=0 if budget is None else budget.started,
                count_queries_completed=0 if budget is None else budget.completed,
                point_membership_tests_started=0 if budget is None else budget.point_tests,
                partial_outputs_retained=True, retries=0)
            event(journal, failure); write(root/'worker-failure.json', failure)
            raise
        finally:
            for s, handler in handlers.items(): signal.signal(s, handler)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--base', type=Path); modes = p.add_mutually_exclusive_group()
    modes.add_argument('--run', action='store_true'); modes.add_argument('--worker', action='store_true')
    args = p.parse_args()
    if args.worker: worker(args.root)
    elif args.run: run(args.root)
    else:
        if args.base is None: p.error('--base required for metadata-only preparation')
        plan = prepare(args.base, args.root)
        print(json.dumps(dict(prepared=True, chains=len(plan['jobs']), new_geometry_queries=0, launched=False)))
