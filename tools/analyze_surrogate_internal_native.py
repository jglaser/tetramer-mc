#!/usr/bin/env python3
"""Bounded internal-native labels for completed rigid/flexible conditional chains.

Plan schema surrogate-internal-native-plan-v1:
  root, output; contact and rigid_contact each {execution_plan,analysis,manifest,
  input_binding} BoundFiles; config BoundFile (flexible config); native_audit
  {execution_plan,summary}; chains[48] {id,job,trajectory,terminal,initial}, where
  initial is {kind:'source'} or {kind:'proposal_prepared',record:BoundFile};
  allocation, limits, input_sha256, source_sha256, runtime.
admit_inputs() validates metadata and returns the exact consumed closure for a
future separate preparer. It never constructs a classifier or decodes a journal.

The completed contact audits authorize skipping elementary arithmetic. Only the
48 new native-label journals are read; the16 native controls use saved summaries.
No contact-edge screening, external/native-cycle inference, proposals or baths.
"""
from __future__ import annotations

import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
from collections import Counter
import copy
import json
import math
from pathlib import Path
import signal
import sys
import time

import run_native_class_physical_campaign as driver
import bind_conditional_native_audit as inherited
import audit_native_pair_candidates as initial_audit
from analyze_mobile_native_pocket import local_sources
from conditional_native_metrics import _environment_metrics
from native_contact_regions import pose_arrays
from native_class_physical_labels import load_bounded_classifier, observer_setup_inventory
from run_conditional_native_audit import Budget as NativeBudget, append

read, require, sha, write = driver.read, driver.require, driver.sha, driver.write
PLAN_SCHEMA = 'surrogate-internal-native-plan-v1'
SCHEMA = 'surrogate-internal-native-v1'
NATIVE_SHA = '07f25356cff250001bbc64fe3eef5b800581883fca9da6ec45a8a366136b8d68'
FLEXIBLE = ('flexible_m1', 'flexible_m8', 'flexible_flat8')
RIGID = ('rigid_surrogate_1', 'rigid_surrogate_8', 'rigid_surrogate_flat8')
STARTS = ('source', 'proposal_prepared')
MEMBERS = [27, 132]
ALLOCATION = dict(new_chains=48, cached_chains=16, blocks=4608, warmup=512,
    retained_endpoints=221232, production_endpoints=196608,
    main_pair_query_cap=221232, checkpoint_pair_query_cap=2, fixture_queries=2,
    external_pair_queries=0, contact_geometry_queries=0, arithmetic_replays=0, new_draws=0)
LIMITS = dict(cpu_seconds=3600, wall_seconds=7200, address_space_bytes=16*1024**3,
    threads=1, max_record_bytes=16*1024**2)
SCOPE = ('Instantaneous internal pair27/132 native registry, all matching motifs and all retained residence. '
    'No external native labels, cycle assessment, contact-edge screening, new sampling or arithmetic replay. '
    'Entry thresholds can flicker; no basin, equilibrium, assembly, physical-kinetic or unseen-coverage claim. '
    'Four streams remain separate per arm/start; shared RNG roles do not create independent arm replicates.')


def identity(job):
    return tuple(job[k] for k in ('context_index', 'arm', 'initialization', 'stream'))


def expected(arms):
    return {(0, a, i, s) for a in arms for i in STARTS for s in range(4)}


def source_paths():
    return local_sources(__file__)


def source_closure():
    return {name: sha(path) for name, path in source_paths().items()}


def completed_report(refs, b, *, native=False):
    """Metadata-only historical lifecycle join; do not rehash old raw journals.

    Published audit input validation is inherited. Rehash report/source metadata
    and only the actual new inputs consumed by this worker. This is deliberately
    not driver.verify_plan(), which would reread all historical raw inputs.
    """
    plan = b.load(refs['execution_plan']); root = Path(plan['root'])
    require(refs['execution_plan']['path'] == str(root/'execution-plan.json')
        and plan['schema'] == driver.SCHEMA and plan['maximum_workers'] == plan['threads'] == 1
        and len(plan['jobs']) == 1, 'Wrong completed analysis controller')
    def load(path): return b.load(b.bind(path))
    claim = load(root/'execution/claim.json'); status = load(root/'execution/status.json')
    summary = load(root/'execution/summary.json'); job = plan['jobs'][0]
    keys = ('complete', 'passed', 'failure', 'active', 'unstarted', 'completed', 'plan_sha256')
    require(all(status[k] == summary[k] for k in keys) and summary['complete'] is True
        and summary['passed'] is True and summary['failure'] is None and summary['active'] is None
        and summary['unstarted'] == [] and len(summary['completed']) == 1
        and summary['plan_sha256'] == refs['execution_plan']['sha256']
        and claim['plan_sha256'] == summary['plan_sha256'] and claim['schema'] == driver.SCHEMA
        and claim['maximum_workers'] == claim['threads'] == 1
        and claim['retries'] == claim['replacements'] == 0
        and not (root/'execution/failure.json').exists()
        and not (root/'preparation-failure.json').exists(), 'Incomplete/failed analysis controller')
    directory = driver.job_directory(root, 0, job)
    done = load(directory/'success.json'); process = load(directory/'process.json')
    require(done == load(directory/'exit.json') == summary['completed'][0]
        and load(directory/'attempt.json')['job'] == job
        and all(done[k] == job[k] for k in ('id', 'population', 'phase', 'argv'))
        and done['success'] is True and done['child_started'] is True and done['child_drained'] is True
        and type(done['returncode']) is int and done['returncode'] == 0
        and done['error'] is None and done['timeout'] is False
        and done['retries'] == done['replacements'] == 0
        and all(process[k] == done[k] for k in ('id', 'pid', 'birth_ticks', 'argv')),
        'Analysis lifecycle mismatch')
    terminal = refs['summary' if native else 'analysis']
    require(done['terminal'] == terminal and job['terminal']['path'] == terminal['path']
        and done['success_contract'] == job['terminal']['success_contract'], 'Unbound analysis terminal')
    result = b.load(terminal)
    require(result['complete'] is True and (not native or result['passed'] is True)
        and not (Path(terminal['path']).parent/'failure.json').exists(), 'Incomplete native/contact report')
    for name, digest in result['source_sha256'].items():
        matches = [(p, h) for p, h in plan['files'].items() if p.endswith('/'+name)]
        require(matches and all(h == digest for _, h in matches), 'Historical source not execution-bound: '+name)
        for p, h in matches: b.bind(p, h)
    if not native:
        manifest = b.load(refs['manifest']); inputs = b.load(refs['input_binding'])
        require(manifest['complete'] is True and manifest['files'][Path(terminal['path']).name] == terminal['sha256']
            and manifest['files'][Path(refs['input_binding']['path']).name] == refs['input_binding']['sha256']
            and inputs['input_sha256'] == result['input_sha256'], 'Contact manifest/input closure mismatch')
    result['_execution_files'] = plan['files']
    return result


def native_identity(audit, b):
    require(audit['schema'] == 'conditional-native-audit-v1' and len(audit['chains']) == 128,
            'Wrong inherited native inventory')
    # The pinned completed report authenticates its original audit plan and all
    # frozen native inputs; no saved atom witness is recomputed geometrically.
    paths = [(p, h) for p, h in audit['input_sha256'].items() if p.endswith('/audit-plan.json')]
    plans = [b.load(dict(path=p, sha256=h)) for p, h in paths]
    plans = [p for p in plans if p.get('schema') == initial_audit.PLAN_SCHEMA]
    require(len(plans) == 1, 'Missing original native input authority')
    p = plans[0]; refs = {k: p[k] for k in ('definition', 'compiled_native', 'witness', 'shape', 'source_frame')}
    definition, compiled, witness, shape, source = (b.load(refs[k]) for k in refs)
    require(refs['definition']['sha256'] == initial_audit.DEFINITION_SHA
        and refs['witness']['sha256'] == initial_audit.WITNESS_SHA
        and refs['shape']['sha256'] == initial_audit.SHAPE_SHA == definition['shape_sha256']
        and len(shape['atoms']) == 4004 and len(source['poses']) == 264
        and source['boundary'] == 'spherical', 'Native geometry identity differs')
    require(compiled['source_definition_sha256'] == refs['definition']['sha256']
        and compiled['source_input_sha256'] == definition['input_sha256']
        and compiled['criteria'] == definition['criteria']
        and witness['native_definition_sha256'] == refs['definition']['sha256']
        and witness['shape_sha256'] == refs['shape']['sha256']
        and witness['compiled_native_sha256'] == refs['compiled_native']['sha256'], 'Native witness binding differs')
    compatibility = witness['native_shape_compatibility']
    require(compatibility['compatible'] is True and compatibility['hard_valid_implication_within_tolerance'] is True
        and compatibility['native_atoms'] == compatibility['physical_atoms'] == compatibility['matched_atoms'] == 4004
        and compatibility['physical_index_by_native_atom'] == list(range(4004))
        and compatibility['matched_max_center_error_a'] == compatibility['matched_max_radius_error_a'] == 0.
        and compatibility['pair_overlap_slack_bound_a'] == 0., 'Incomplete native atom identity witness')
    require(len(definition['input_sha256']) == 15, 'Incomplete classifier input closure')
    for name, digest in definition['input_sha256'].items():
        part = Path(name); require(not part.is_absolute() and '..' not in part.parts, 'Unsafe native input path')
        b.bind(Path(refs['definition']['path']).parent/'inputs'/part, digest)
    setup = observer_setup_inventory(refs['definition']['path'])
    require(setup == p['setup_inventory'], 'Changed native setup allocation')
    return dict(refs=refs, source=source['poses'], setup=setup)


def admit_inputs(plan):
    """Metadata/hash admission only; returns context and consumed BoundFile map."""
    require(plan['schema'] == PLAN_SCHEMA and plan['allocation'] == ALLOCATION
        and plan['limits'] == LIMITS, 'Changed frozen scope/limits')
    b = inherited.Bindings()
    contact = completed_report(plan['contact'], b)
    rigid = completed_report(plan['rigid_contact'], b)
    require(contact['schema'] == 'flexible-surrogate-analysis-v1' and contact['new_chains'] == 24
        and contact['reused_control_chains'] == 40 and len(contact['chains']) == 64
        and contact['native_observer'] is False and contact['old_geometry_queries'] == contact['new_physical_draws'] == 0,
        'Complete64-chain flexible contact audit required')
    require(rigid['schema'] == 'rigid-surrogate-analysis-v1' and len(rigid['chains']) == 40
        and contact['input_sha256'].get(plan['rigid_contact']['analysis']['path']) == plan['rigid_contact']['analysis']['sha256'],
        'Rigid controls not inherited from authenticated audit')
    require(plan['native_audit']['summary']['sha256'] == NATIVE_SHA, 'Wrong completed native128 reference')
    audit = completed_report(plan['native_audit'], b, native=True)
    native = native_identity(audit, b); config = b.load(plan['config'])
    require(contact['input_sha256'].get(plan['config']['path']) == plan['config']['sha256']
        and config['source_frame'] == native['refs']['source_frame'] and config['shape'] == native['refs']['shape']
        and [config['contexts'][0][k] for k in ('root', 'child')] == MEMBERS
        and config['physical']['depletant_radius'] == 1.4 and config['physical']['activity'] == .0275,
        'Conditional target/shape/source differs')
    index = {identity(c['job']): c for c in contact['chains']}
    require(len(index) == 64 and set(index) == expected(FLEXIBLE+RIGID+('local', 'm4')), 'Wrong64 identity product')
    require(len(plan['chains']) == 48 and len({c['id'] for c in plan['chains']}) == 48
        and {identity(c['job']) for c in plan['chains']} == expected(FLEXIBLE+RIGID), 'Wrong48 native-label allocation')
    chains = []
    for chain in plan['chains']:
        require(chain['id'] and all(c.isalnum() or c in '-_' for c in chain['id']), 'Unsafe chain ID')
        prior = index[identity(chain['job'])]; origin = contact if chain['job']['arm'] in FLEXIBLE else rigid
        same = [c for c in origin['chains'] if identity(c['job']) == identity(chain['job'])]
        require(len(same) == 1 and same[0]['job'] == prior['job'] == chain['job']
            and chain['trajectory'] == prior['trajectory'] == same[0]['trajectory'], 'Chain/journal identity differs')
        for ref in (chain['terminal'], chain['trajectory']):
            require(origin['input_sha256'].get(ref['path']) == origin['_execution_files'].get(ref['path']) == ref['sha256'],
                    'Journal/terminal lacks frozen completed arithmetic audit')
            b.bind(ref['path'], ref['sha256'])
        terminal = b.load(chain['terminal']); cpu = prior['metrics']['full_sampler_cpu_seconds']
        require(terminal['complete'] is True and terminal['conditional_target'] is True
            and terminal['job'] == chain['job'] and terminal['trajectory'] == chain['trajectory']
            and terminal['blocks'] == 4608 and terminal['counts'] == prior['counts']
            and terminal['cpu_seconds'] == cpu and math.isfinite(cpu) and cpu > 0
            and not (Path(chain['terminal']['path']).parent/'failure.json').exists(), 'Incomplete/different sampler terminal')
        start = chain['initial']; require(start['kind'] == chain['job']['initialization'], 'Changed initialization')
        if start['kind'] == 'source':
            require(set(start) == {'kind'}, 'Unexpected source initial fields')
            poses = [native['source'][i] for i in MEMBERS]
        else:
            value = b.load(start['record'])
            require(origin['input_sha256'].get(start['record']['path']) == start['record']['sha256']
                and value['status'] == 'prepared' and value['context_index'] == 0
                and value['stream'] == chain['job']['stream'], 'Unbound/changed prepared start')
            poses = value['selected']
        chains.append(dict(chain, initial_poses=poses, full_sampler_cpu_seconds=cpu))
    controls = []
    for c in audit['chains']:
        if identity(c['job']) not in expected(('local', 'm4')): continue
        prior = index[identity(c['job'])]; metrics = c['metrics']
        require(c['job'] == prior['job'] and c['trajectory'] == prior['trajectory']
            and metrics['full_sampler_cpu_seconds'] == prior['metrics']['full_sampler_cpu_seconds']
            and metrics['production_samples'] == 4096 and metrics['retained_endpoints'] == 4609
            and metrics['mobile_registry_resolved_fraction'] == 1
            and metrics['mobile_cycle_frustration_fraction'] == 0,
            'Cached native control identity/metric scope differs')
        controls.append(dict(job=c['job'], reused_native_control=True, trajectory=c['trajectory'],
            metrics=control_metrics(metrics)))
    require(len(controls) == 16 and {identity(c['job']) for c in controls} == expected(('local', 'm4')),
            'Missing/repeated native controls')
    return dict(native=native, chains=chains, controls=controls), b.files


def internal_descriptor(descriptor):
    """Keep existing internal events, removing unobserved graph-resolution claims."""
    result = copy.deepcopy(descriptor); counts = result['counts']
    require(counts.pop('direct_nonempty_unresolved_changes') == 0, 'Unresolved old control cannot be projected')
    counts['direct_nonempty_changes'] = counts.pop('direct_nonempty_resolved_changes')
    counts['nonempty_returns'] = counts.pop('nonempty_resolved_returns')
    rates = result['rates_per_full_sampler_cpu_second']
    require(rates.pop('direct_nonempty_unresolved_changes') == 0, 'Unresolved rate')
    rates['direct_nonempty_changes'] = rates.pop('direct_nonempty_resolved_changes')
    rates['nonempty_returns'] = rates.pop('nonempty_resolved_returns')
    for event in result['environments']['completed_returns']:
        require(event.pop('return_endpoints_resolved') is True and event.pop('passed_through_unresolved') is False,
                'Unresolved historical return cannot be projected')
    result['event_scope'] = ('Internal motif-set entries/exits, adjacent distinct nonempty changes and completed returns; '
        'returns retain through-empty flags. Last warmup is transition baseline only. No external/cycle assessment.')
    return result


def control_metrics(m):
    keys = ('production_samples', 'retained_endpoints', 'full_sampler_cpu_seconds', 'production_window_cpu_seconds',
            'internal_native_attachment_fraction')
    return dict(schema='internal-native-metrics-v1', **{k: m[k] for k in keys},
        internal_motifs=internal_descriptor(m['internal_motifs']), scope=SCOPE)


def reduce_trace(trace, cpu, *, blocks=4608, warmup=512):
    require(len(trace) == blocks+1 and 0 <= warmup < blocks and math.isfinite(cpu) and cpu > 0,
            'Incomplete internal trace or CPU')
    previous = -1.
    for i, row in enumerate(trace):
        require(row['block'] == i and row['production'] is (i > warmup)
            and previous <= row['sampler_cpu_seconds'] <= cpu, 'Trace cadence/CPU differs')
        previous = row['sampler_cpu_seconds']
    values = [r['internal_native_keys'] for r in trace[warmup:]]
    descriptor = _environment_metrics(values, [True]*len(values), list(range(warmup, blocks+1)), cpu)
    return dict(schema='internal-native-metrics-v1', production_samples=blocks-warmup, retained_endpoints=blocks+1,
        full_sampler_cpu_seconds=cpu, production_window_cpu_seconds=trace[-1]['sampler_cpu_seconds']-trace[warmup]['sampler_cpu_seconds'],
        internal_native_attachment_fraction=sum(bool(v) for v in values[1:])/(blocks-warmup),
        internal_motifs=internal_descriptor(descriptor), scope=SCOPE)


def endpoints(rows, chain, *, blocks=4608, warmup=512):
    """Exact7-row adapter for a prior byte-authenticated arithmetic audit only."""
    rows = iter(rows); cpu = -1.
    family = 'flexible' if chain['job']['arm'] in FLEXIBLE else 'rigid'
    require(chain['job']['arm'] in FLEXIBLE+RIGID, 'Unknown surrogate arm')
    def take(kind, block):
        nonlocal cpu
        row = next(rows, None)
        require(type(row) is dict and row.get('kind') == kind and type(row.get('block')) is int
            and row['block'] == block and not any(k in row for k in ('fatal_error', 'error', 'bath_failure')),
            'Missing/out-of-order/fatal admitted journal row')
        now = row['sampler_cpu_seconds']
        require(type(now) in (int, float) and math.isfinite(now) and 0 <= now
            and cpu <= now <= chain['full_sampler_cpu_seconds'],
                'Nonmonotone/nonfinite journal CPU')
        cpu = now; return row
    first = take('initial', 0)
    require(first['job'] == chain['job'] and first['conditional_target'] is True
        and first['selected'] == chain['initial_poses'] and first.get('production', False) is False,
        'Initial identity/pose differs')
    def point(row, block):
        require(type(row['selected']) is list and len(row['selected']) == 2, 'Expected two retained poses')
        for pose in row['selected']: pose_arrays(pose)
        return dict(block=block, production=block > warmup, selected=row['selected'], sampler_cpu_seconds=cpu)
    yield point(first, 0)
    for block in range(1, blocks+1):
        for attempt, member in enumerate((27, 132, 27, 132)):
            local = take('local', block)
            require(local['attempt'] == attempt and local['member'] == member
                and local['status'] in ('hard_rejected', 'completed'), 'Local sequence/status differs')
        begun = take(family+'_surrogate_attempt_begun', block)
        outcome = take(family+'_surrogate_chain', block)
        require(begun['status'] == 'begun' and begun['members'] == outcome['members'] == MEMBERS
            and outcome['status'] in ('completed', 'identity_self_loop'), 'Incomplete collective attempt')
        row = take('retained_block', block)
        require(row['production'] is (block > warmup), 'Production flag differs')
        yield point(row, block)
    require(next(rows, None) is None, 'Unexpected journal tail')


class PairObserver:
    """One exact absolute-pose cache entry; no approximate pose/contact pruning."""
    def __init__(self, native, budget):
        self.native, self.budget = native, budget
        self.labels = {m['id'] for m in native.motifs}; self.key = None; self.value = None; self.hits = 0

    def classify(self, poses, location):
        for pose in poses: pose_arrays(pose)
        key = json.dumps(poses, sort_keys=True, separators=(',', ':'), allow_nan=False)
        if key == self.key:
            self.hits += 1; return copy.deepcopy(self.value)
        matches = self.budget.query('main', dict(pair=MEMBERS, **location),
            lambda: self.native.classify_pair(*poses))
        require(type(matches) is list and all(type(m['motif_id']) is int and m['motif_id'] in self.labels for m in matches),
                'Invalid full native classifier result')
        value = [MEMBERS+[m] for m in sorted({v['motif_id'] for v in matches})]
        self.key, self.value = key, value
        return copy.deepcopy(value)


def fixtures(native, source, budget):
    results = []
    for pair, wanted in (([9, 24], [6]), (MEMBERS, [])):
        matches = budget.query('checkpoint', dict(fixture=True, pair=pair),
            lambda: native.classify_pair(*(source[i] for i in pair)))
        require(type(matches) is list and sorted({m['motif_id'] for m in matches}) == wanted,
                'Frozen positive/negative classifier fixture failed')
        results.append(dict(pair=pair, expected_motifs=wanted, passed=True, excluded_from_comparisons=True))
    return results


def validate_plan(plan):
    require(plan['source_sha256'] == source_closure() and plan['runtime'] == inherited.runtime(), 'Source/runtime differs')
    context, inputs = admit_inputs(plan)
    require(inputs == plan['input_sha256'], 'Incomplete or extra input closure')
    return context


def live_authority(path, digest, plan):
    root = Path(plan['root']); execution_path = root/'execution-plan.json'; execution = read(execution_path)
    driver.verify_plan(execution_path, execution, sha(execution_path))
    require(execution['root'] == str(root) and len(execution['jobs']) == 1, 'Exactly one owned child required')
    job = execution['jobs'][0]; claim = read(root/'execution/claim.json'); status = read(root/'execution/status.json')
    require(claim['pid'] == os.getppid() and claim['plan_sha256'] == sha(execution_path)
        and claim['retries'] == claim['replacements'] == 0 and status['failure'] is None
        and status['plan_sha256'] == claim['plan_sha256']
        and status['active'] == dict(ordinal=0, id=job['id'], population=job['population'], phase=job['phase']),
        'Missing live controller authority')
    stat = Path(f'/proc/{claim["pid"]}/stat').read_text().rsplit(')', 1)[1].split()
    require(int(stat[19]) == claim['birth_ticks'] and stat[0] not in ('Z', 'X'), 'Parent identity changed')
    require(job['argv'] == [sys.executable, '-B', str(Path(__file__).resolve()), '--plan', str(path),
            '--plan-sha256', digest, '--out', plan['output']]
        and job['terminal'] == dict(path=str(Path(plan['output'])/'summary.json'), success_contract='complete_and_passed')
        and job['cpu_limit_seconds'] == LIMITS['cpu_seconds'] and job['wall_limit_seconds'] == LIMITS['wall_seconds']
        and job['address_space_limit_bytes'] == LIMITS['address_space_bytes'], 'Worker command/limits differ')
    closure = dict(plan['input_sha256'], **{str(p): plan['source_sha256'][n] for n, p in source_paths().items()})
    closure[str(path)] = digest
    require(all(execution['files'].get(p) == h for p, h in closure.items()), 'Execution omitted bound inputs/source')


def run(plan_path, digest, out):
    path = Path(plan_path).resolve(); require(sha(path) == digest and sys.flags.optimize == 0, 'Plan hash/optimized runtime differs')
    plan = read(path); out = Path(out).resolve(); root = Path(plan['root']).resolve()
    require(str(root) == plan['root'] and str(out) == plan['output'] and out.is_relative_to(root)
        and out != root and not out.exists(), 'Fresh output under frozen root required')
    live_authority(path, digest, plan)
    context = validate_plan(plan)  # All48 raw hashes are fixed before any decode/query.
    require(not any(Path(p).is_relative_to(out) for p in plan['input_sha256']), 'Input/output collision')
    out.mkdir(); (out/'chains').mkdir(); results = []; started = time.process_time(); wall = time.monotonic()
    old_handlers = {}
    def stop(signum, frame): raise RuntimeError('Interrupted signal '+str(signum))
    with (out/'attempts.jsonl').open('x') as journal:
        budget = NativeBudget(ALLOCATION, LIMITS, lambda row: append(journal, row), started=started, wall=wall)
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGXCPU): old_handlers[sig] = signal.signal(sig, stop)
        try:
            native_info = context['native']; refs = native_info['refs']
            def bind(p, h):
                require(plan['input_sha256'].get(str(Path(p).resolve())) == h and sha(p) == h, 'Unbound classifier input')
                return Path(p)
            native, classifier = load_bounded_classifier(refs['definition']['path'], native_info['setup'], budget, bind)
            fixture_results = fixtures(native, native_info['source'], budget)
            observer = PairObserver(native, budget); completed = 0; files = {}
            for chain in context['chains']:
                budget.check(); tick = time.process_time(); trace = []; before = dict(budget.calls); hits = observer.hits
                budget.emit(dict(state='chain_begin', chain_id=chain['id'], job=chain['job']))
                target = out/'chains'/(chain['id']+'.jsonl')
                with Path(chain['trajectory']['path']).open('rb') as raw, target.open('x') as dest:
                    def rows():
                        while True:
                            line = raw.readline(LIMITS['max_record_bytes']+1)
                            if not line: return
                            require(len(line) <= LIMITS['max_record_bytes'] and line.endswith(b'\n'), 'Oversized/truncated admitted row')
                            yield json.loads(line, object_pairs_hook=driver._pairs,
                                parse_constant=lambda s: (_ for _ in ()).throw(ValueError('Nonfinite JSON '+s)))
                    for point in endpoints(rows(), chain):
                        budget.check(); location = dict(chain_id=chain['id'], block=point['block'])
                        budget.emit(dict(state='endpoint_begin', **location))
                        labels = observer.classify(point.pop('selected'), location)
                        point['internal_native_keys'] = labels; append(dest, point); trace.append(point); completed += 1
                        require(completed <= ALLOCATION['retained_endpoints'], 'Endpoint allocation exceeded')
                        budget.emit(dict(state='endpoint_complete', **location, internal_native_keys=labels))
                metrics = reduce_trace(trace, chain['full_sampler_cpu_seconds'])
                files[str(target)] = sha(target)
                results.append(dict(job=chain['job'], chain_id=chain['id'], reused_native_control=False,
                    trajectory=chain['trajectory'], terminal=chain['terminal'], metrics=metrics,
                    endpoints=dict(path=str(target), sha256=files[str(target)]), exact_pose_cache_hits=observer.hits-hits,
                    pair_queries=budget.calls['main_completed']-before.get('main_completed', 0),
                    analysis_cpu_seconds=time.process_time()-tick))
                budget.emit(dict(state='chain_complete', chain_id=chain['id']))
            require(completed == ALLOCATION['retained_endpoints'] and len(results) == 48
                and budget.calls['main_begun'] == budget.calls['main_completed']
                and budget.calls['checkpoint_begun'] == budget.calls['checkpoint_completed'] == 2, 'Incomplete allocation')
            require(source_closure() == plan['source_sha256'] and inherited.runtime() == plan['runtime'] and sha(path) == digest,
                    'Source/runtime/plan changed')
            for p, h in plan['input_sha256'].items(): require(sha(p) == h, 'Input changed '+p)
            budget.check(); files[str(out/'attempts.jsonl')] = sha(out/'attempts.jsonl')
            result = dict(schema=SCHEMA, complete=True, passed=True, plan_sha256=digest, allocation=ALLOCATION,
                chains=results+context['controls'], fixtures=fixture_results, classifier=classifier,
                query_counts=dict(budget.calls), setup_counts=dict(budget.setup_calls), files=files,
                input_sha256=plan['input_sha256'], source_sha256=plan['source_sha256'], runtime=plan['runtime'],
                analysis_cpu_seconds=time.process_time()-started, scope=SCOPE,
                external_native_observer=False, cycle_observer=False, assembly_gate_open=False)
            write(out/'summary.json', result); return result
        except BaseException as error:
            for sig in old_handlers: signal.signal(sig, signal.SIG_IGN)
            budget.emit(dict(state='failure', error=repr(error), counts=dict(budget.calls)))
            write(out/'failure.json', dict(schema=SCHEMA, complete=False, passed=False, error=repr(error),
                plan_sha256=digest, completed_chains=len(results), query_counts=dict(budget.calls),
                setup_counts=dict(budget.setup_calls), partial_outputs_retained=True, retries=0, replacements=0))
            raise
        finally:
            for sig, handler in old_handlers.items(): signal.signal(sig, handler)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True); parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); run(args.plan, args.plan_sha256, args.out)
