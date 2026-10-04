#!/usr/bin/env python3
"""One bounded native audit of all authenticated retained conditional endpoints.

The companion binder admits completed contact replay evidence and the completed
initial native-filter audit. This worker repeats neither proposal arithmetic nor
contact geometry. It keeps every endpoint, including rejection residence, and
uses a predetermined, outcome-independent sparse unpruned check of the filter.
"""
from __future__ import annotations

import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import signal
import sys
import time

import bind_conditional_native_audit as binding
from conditional_native_metrics import iter_authenticated_endpoints, native_metrics
from conditional_native_observer import ConditionalNativeObserver, canonical
from native_class_physical_labels import load_bounded_classifier
from run_native_class_physical_campaign import read, require, sha, write, verify_plan as verify_execution

SCHEMA = 'conditional-native-audit-v1'
ROLES = ('main', 'checkpoint')
GRAPH_FIELDS = ('instantaneous_native_keys', 'fixed_native_keys', 'mobile_related_native_keys',
                'external_native_keys', 'internal_native_keys', 'full_native_graph',
                'mobile_native_components', 'mobile_registry_resolved')
METRIC_FIELDS = tuple(key for key in GRAPH_FIELDS if key != 'full_native_graph')
SCOPE = ('Saved conditional native registry at rd=1.4 A, z=.0275 A^-3 for two mobile '
         'tetramers against 262 fixed spectators. Every retained endpoint is observed. '
         'Native and contact observers have different descriptors; neither establishes '
         'equilibrium, finite-system assembly, physical kinetic rates or unseen coverage. '
         'Full sampler CPU is the sampling denominator; analysis/setup CPU is separate.')


def reference(path): return dict(path=str(Path(path).resolve()), sha256=sha(path))


def append(stream, value):
    stream.write(json.dumps(value, separators=(',', ':'), allow_nan=False)+'\n')
    stream.flush(); os.fsync(stream.fileno())


class Budget:
    def __init__(self, allocation, limits, emit, *, started=None, wall=None):
        self.allocation, self.limits, self.emit = allocation, limits, emit
        self.started = time.process_time() if started is None else started
        self.wall = time.monotonic() if wall is None else wall
        self.calls, self.setup_calls = Counter(), Counter()

    def check(self):
        require(time.process_time()-self.started <= self.limits['cpu_seconds'], 'Native audit CPU budget exhausted')
        require(time.monotonic()-self.wall <= self.limits['wall_seconds'], 'Native audit wall budget exhausted')

    def query(self, role, location, function):
        self.check(); require(role in ROLES, 'Unknown native query role')
        require(self.calls[role+'_begun'] < self.allocation[role+'_pair_query_cap'], 'Native query cap exceeded: '+role)
        self.calls[role+'_begun'] += 1
        identity = dict(role=role, query=self.calls[role+'_begun'], **location)
        self.emit(dict(state='query_begin', **identity))
        started = time.process_time(); value = function(); cpu = time.process_time()-started
        self.calls[role+'_completed'] += 1
        self.emit(dict(state='query_complete', **identity, result=value, classifier_cpu_seconds=cpu))
        self.check()
        return value

    def setup_query(self, kind, limit, function, **extra):
        self.check(); require(self.setup_calls[kind+'_started'] < limit, 'Native setup cap exceeded: '+kind)
        index = self.setup_calls[kind+'_started']; self.setup_calls[kind+'_started'] += 1
        self.emit(dict(state='setup_begin', kind=kind, index=index, **extra))
        value = function(); self.setup_calls[kind+'_completed'] += 1
        self.emit(dict(state='setup_complete', kind=kind, index=index, counts=dict(self.setup_calls)))
        self.check()
        return value


class RecordedCandidates:
    """Remember the actual search, including reuse after whole-endpoint hits."""
    def __init__(self, candidate, members):
        self.candidate, self.members = candidate, tuple(members)
        self.identity = None; self.pairs = None; self.calls = 0; self.cpu_seconds = 0.

    def candidate_pairs(self, poses, *, mobile_labels=None):
        require(tuple(mobile_labels) == self.members, 'Unexpected fixed-pair search after fixed-graph reuse')
        started = time.process_time()
        pairs = self.candidate.candidate_pairs(poses, mobile_labels=mobile_labels)
        self.cpu_seconds += time.process_time()-started; self.calls += 1
        self.identity = canonical([poses[i] for i in self.members])
        self.pairs = tuple(map(tuple, pairs))
        return pairs


def mobile_pairs(count, members):
    return tuple((i, j) for i in range(count) for j in range(i+1, count) if i in members or j in members)


def attach_observer(native, source, members, fixed_keys, budget, context_index):
    """Seed only authenticated fixed labels; keep native/filter predicates intact."""
    observer = ConditionalNativeObserver(native, source, members)
    keys = [tuple(key) for key in fixed_keys]
    require(len(set(keys)) == len(keys) and all(len(k) == 3 and all(type(v) is int for v in k)
            and 0 <= k[0] < k[1] < len(source) and k[2] in observer.labels
            and not set(members).intersection(k[:2]) for k in keys), 'Invalid inherited fixed labels')
    require(observer.fixed_keys is None and observer.counts['fixed_pair_calls'] == 0, 'Unexpected fixed setup query')
    observer.fixed_keys = tuple(sorted(keys))
    tracker = RecordedCandidates(observer.filter, members); observer.filter = tracker
    location = {}
    original = observer._classify_pair

    def classify(state, pair, role):
        require(role == 'mobile' and location, 'Unallocated fixed/endpoint query')
        return budget.query('main', dict(context_index=context_index, **location, pair=list(pair)),
                            lambda: original(state, pair, role))

    observer._classify_pair = classify
    return observer, tracker, location


def checkpoint(native, source, members, selected, tracker, budget, location):
    """Audit only filter-omitted pairs; never invoke the filter a second time."""
    require(tracker.identity == canonical(selected) and tracker.pairs is not None,
            'No current candidate set for checkpoint')
    all_pairs = set(mobile_pairs(len(source), set(members))); candidates = set(tracker.pairs)
    require(candidates <= all_pairs and len(candidates) == len(tracker.pairs), 'Invalid recorded candidate inventory')
    omitted = sorted(all_pairs-candidates); before = tracker.calls
    state = list(source)
    for member, pose in zip(members, selected): state[member] = pose
    budget.emit(dict(state='checkpoint_begin', **location, candidates=sorted(candidates), omitted_pairs=len(omitted)))
    started = time.process_time()
    for i, j in omitted:
        found = budget.query('checkpoint', dict(**location, pair=[i, j]),
                             lambda i=i, j=j: native.classify_pair(state[i], state[j]))
        require(type(found) is list and not found, 'False-negative native pair at production checkpoint')
    require(tracker.calls == before, 'Checkpoint repeated candidate search')
    result = dict(**location, candidates=len(candidates), omitted_pairs=len(omitted),
                  cpu_seconds=time.process_time()-started, passed=True, second_candidate_search=False)
    budget.calls['checkpoint_endpoints'] += 1
    require(budget.calls['checkpoint_endpoints'] <= budget.allocation['checkpoint_endpoints'], 'Too many checkpoint endpoints')
    budget.emit(dict(state='checkpoint_complete', **result))
    return result


def decoded_rows(path, maximum, initial_metadata):
    """Bound line memory; parse the already replayed journal without redoing MC."""
    with Path(path).open('rb') as stream:
        first = True
        while raw := stream.readline(maximum+1):
            require(len(raw) <= maximum and raw.endswith(b'\n'), 'Oversized or truncated journal record')
            row = json.loads(raw, parse_constant=lambda x: (_ for _ in ()).throw(ValueError('Nonfinite JSON '+x)))
            if first:
                initial_metadata.update({k: row[k] for k in ('geometry_load_cpu_seconds', 'cloud_cpu_seconds') if k in row})
                first = False
            yield row


def worker_argv(plan_path, digest, out):
    return [sys.executable, '-B', str(Path(__file__).resolve()), '--plan', str(Path(plan_path).resolve()),
            '--plan-sha256', digest, '--out', str(Path(out).resolve())]


def live_authority(plan_path, digest, plan):
    root = Path(plan['root']); path = root/'execution-plan.json'; execution = read(path)
    require(plan_path == root/'audit-plan.json', 'Wrong native audit plan location')
    verify_execution(path, execution, sha(path))
    require(execution['root'] == str(root) and len(execution['jobs']) == 1, 'Exactly one native-audit child required')
    files = dict(plan['input_sha256']); files[str(plan_path)] = digest
    files.update({str(p): plan['source_sha256'][name] for name, p in binding.source_paths().items()})
    files[plan['runtime']['executable']['path']] = plan['runtime']['executable']['sha256']
    require(all(execution['files'].get(p) == d for p, d in files.items()), 'Controller omitted frozen native inputs')
    claim = read(root/'execution/claim.json')
    require(claim['plan_sha256'] == sha(path) and claim['pid'] == os.getppid()
            and claim['maximum_workers'] == claim['threads'] == 1
            and claim['retries'] == claim['replacements'] == 0, 'Unowned native audit child')
    stat = Path(f"/proc/{claim['pid']}/stat").read_text().rsplit(')', 1)[1].split()
    require(int(stat[19]) == claim['birth_ticks'] and stat[0] not in ('Z', 'X'), 'Controller process identity changed')
    status = read(root/'execution/status.json'); job = execution['jobs'][0]
    require(status['plan_sha256'] == claim['plan_sha256'] and status['failure'] is None
            and status['active'] == dict(ordinal=0, id=job['id'], population=job['population'], phase='geometry')
            and job['phase'] == 'geometry', 'Native audit is not the active allocation')
    require(job['argv'] == worker_argv(plan_path, digest, plan['output'])
            and job['terminal'] == dict(path=str(Path(plan['output'])/'summary.json'), success_contract='complete_and_passed')
            and job['cpu_limit_seconds'] == plan['limits']['cpu_seconds']
            and job['wall_limit_seconds'] == plan['limits']['wall_seconds']
            and job['address_space_limit_bytes'] == plan['limits']['address_space_bytes'], 'Changed native worker command/limits')


def run(plan_path, digest, out):
    started, wall = time.process_time(), time.monotonic()
    require(__debug__ and sys.flags.optimize == 0, 'Native graph assertions must remain enabled')
    plan_path, out = Path(plan_path).resolve(), Path(out).resolve()
    require(sha(plan_path) == digest, 'Changed native audit plan')
    plan = read(plan_path)
    expected = binding.make_plan(plan['contact_root'], plan['initial_audit_root'], root=plan['root'])
    require(plan == expected and out == Path(plan['output']), 'Native audit declaration/inputs differ')
    live_authority(plan_path, digest, plan)
    require(not out.exists(), 'Native audit output already exists; no retry')
    out.mkdir(parents=True); (out/'chains').mkdir(); (out/'graphs').mkdir()
    write(out/'claim.json', dict(schema=SCHEMA, plan_sha256=digest, allocation=plan['allocation'], retries=0, replacements=0))
    results, files = [], {}
    with (out/'attempts.jsonl').open('x') as ledger:
        budget = Budget(plan['allocation'], plan['limits'], lambda v: append(ledger, v), started=started, wall=wall)
        def stopped(signum, _frame): raise RuntimeError('Native audit interrupted by signal '+str(signum))
        signals = (signal.SIGTERM, signal.SIGINT, signal.SIGXCPU)
        previous = {s: signal.signal(s, stopped) for s in signals}
        try:
            def bind(path, expected_sha):
                path = Path(path).resolve()
                require(plan['input_sha256'].get(str(path)) == expected_sha and sha(path) == expected_sha,
                        'Unbound frozen native constructor input')
                return path
            setup_tick = time.process_time()
            native, native_identity = load_bounded_classifier(plan['native_inputs']['definition']['path'],
                plan['setup_inventory'], budget, bind)
            setup_cpu = time.process_time()-setup_tick
            source = read(plan['native_inputs']['source_frame']['path'])['poses']
            context_reports = []
            for ci, context in enumerate(plan['contexts']):
                members = [context['root'], context['child']]
                fixed = plan['fixed_native_keys_by_context'][str(ci)]
                tick = time.process_time()
                observer, tracker, location = attach_observer(native, source, members, fixed, budget, ci)
                attach_cpu = time.process_time()-tick
                seen_graphs = set(); graph_path = out/'graphs'/f'context-{ci}.jsonl'
                with graph_path.open('x') as graph_stream:
                    chains = [chain for chain in plan['chains'] if chain['identity']['context_index'] == ci]
                    require(len(chains) == 32, 'Wrong per-context chain count')
                    for chain in chains:
                        chain_tick = time.process_time(); observer_cpu = checkpoint_cpu = 0.
                        before = dict(budget.calls); trace = []; checks = []; initial_metadata = {}
                        initial_ref = chain['initial']
                        initial = ([source[i] for i in members] if initial_ref['kind'] == 'source'
                                   else read(initial_ref['record']['path'])['selected'])
                        require(chain['members'] == members
                                and chain['checkpoint_blocks'] == binding.checkpoint_blocks(chain['identity']),
                                'Changed labels or predeclared checkpoints')
                        budget.emit(dict(state='chain_begin', chain_id=chain['id'], identity=chain['identity'],
                                         checkpoints=chain['checkpoint_blocks']))
                        path = out/'chains'/(chain['id']+'.jsonl')
                        rows = decoded_rows(chain['trajectory']['path'], plan['limits']['max_record_bytes'], initial_metadata)
                        endpoints = iter_authenticated_endpoints(rows, job=chain['job'], initial=initial,
                            blocks=plan['allocation']['blocks'], warmup=plan['allocation']['warmup'])
                        with path.open('x') as stream:
                            for endpoint in endpoints:
                                budget.check(); budget.calls['endpoints_begun'] += 1
                                require(budget.calls['endpoints_begun'] <= plan['allocation']['retained_endpoints'], 'Endpoint cap exceeded')
                                location.clear(); location.update(chain_id=chain['id'], block=endpoint['block'])
                                budget.emit(dict(state='endpoint_begin', context_index=ci, **location))
                                tick = time.process_time(); observed = observer.classify(endpoint['selected'])
                                observer_cpu += time.process_time()-tick
                                graph_id = hashlib.sha256(canonical(observed['instantaneous_native_keys']).encode()).hexdigest()
                                if graph_id not in seen_graphs:
                                    append(graph_stream, dict(graph_id=graph_id, **{k: observed[k] for k in GRAPH_FIELDS}))
                                    seen_graphs.add(graph_id)
                                if endpoint['block'] in chain['checkpoint_blocks']:
                                    check = checkpoint(native, source, members, endpoint['selected'], tracker, budget,
                                        dict(context_index=ci, **location))
                                    checks.append(check); checkpoint_cpu += check['cpu_seconds']
                                metadata = {k: endpoint[k] for k in ('block', 'production', 'sampler_cpu_seconds')}
                                append(stream, dict(**metadata, graph_id=graph_id, observer_counts=observed['observer_counts']))
                                trace.append(dict(**metadata, **{k: observed[k] for k in METRIC_FIELDS}))
                                budget.calls['endpoints_completed'] += 1
                                if endpoint['production']: budget.calls['production_endpoints'] += 1
                                budget.emit(dict(state='endpoint_complete', context_index=ci, **location, graph_id=graph_id,
                                                 counts=dict(budget.calls)))
                        require([c['block'] for c in checks] == chain['checkpoint_blocks'], 'Missing predeclared checkpoint')
                        require(sha(chain['trajectory']['path']) == chain['trajectory']['sha256'], 'Journal changed while reading')
                        metrics = native_metrics(trace, members=members, full_sampler_cpu_seconds=chain['full_sampler_cpu_seconds'],
                            blocks=plan['allocation']['blocks'], warmup=plan['allocation']['warmup'])
                        result = dict(chain_id=chain['id'], identity=chain['identity'], job=chain['job'],
                            trajectory=chain['trajectory'], terminal=chain['terminal'], metrics=metrics,
                            endpoints=reference(path), checkpoints=checks, observer_cpu_seconds=observer_cpu,
                            checkpoint_cpu_seconds=checkpoint_cpu, analysis_cpu_seconds=time.process_time()-chain_tick,
                            initial_cost_metadata=initial_metadata,
                            geometry_load_cpu_seconds=chain['geometry_load_cpu_seconds'],
                            contact_observer_cpu_seconds=chain['contact_observer_cpu_seconds'],
                            query_counts={k: v-before.get(k, 0) for k, v in budget.calls.items()})
                        results.append(result); files[str(path)] = sha(path)
                        write(out/'chains'/(chain['id']+'-summary.json'), result)
                        files[str(out/'chains'/(chain['id']+'-summary.json'))] = sha(out/'chains'/(chain['id']+'-summary.json'))
                        budget.emit(dict(state='chain_complete', chain_id=chain['id'], counts=dict(budget.calls)))
                files[str(graph_path)] = sha(graph_path)
                require(observer.counts['fixed_pair_calls'] == 0, 'Repeated fixed native geometry')
                context_reports.append(dict(context_index=ci, members=members, fixed_native_keys=fixed,
                    graph_records=len(seen_graphs), graphs=reference(graph_path), observer_counts=observer.counts,
                    candidate_searches=tracker.calls, candidate_cpu_seconds=tracker.cpu_seconds,
                    adapter_setup_cpu_seconds=attach_cpu, new_fixed_pair_queries=0))
            require(len(results) == plan['allocation']['chains'] and
                budget.calls['endpoints_begun'] == budget.calls['endpoints_completed'] == plan['allocation']['retained_endpoints']
                and budget.calls['production_endpoints'] == plan['allocation']['production_endpoints']
                and budget.calls['checkpoint_endpoints'] == plan['allocation']['checkpoint_endpoints']
                and all(budget.calls[r+'_begun'] == budget.calls[r+'_completed'] for r in ROLES), 'Incomplete native audit inventory')
            require(sha(plan_path) == digest and binding.runtime() == plan['runtime'], 'Native plan/runtime changed')
            for path, expected_sha in plan['input_sha256'].items(): require(sha(path) == expected_sha, 'Native audit input changed')
            require(binding.source_closure() == plan['source_sha256'], 'Native audit source changed')
            files[str(out/'attempts.jsonl')] = sha(out/'attempts.jsonl')
            report = dict(schema=SCHEMA, complete=True, passed=True, plan_sha256=digest, allocation=plan['allocation'],
                chains=results, contexts=context_reports, query_counts=dict(budget.calls), setup_counts=dict(budget.setup_calls),
                native_identity=native_identity, native_setup_cpu_seconds=setup_cpu,
                inherited_preparation_costs=plan['inherited_preparation_costs'],
                analysis_cpu_seconds=time.process_time()-started, analysis_wall_seconds=time.monotonic()-wall,
                source_sha256=plan['source_sha256'], input_sha256=plan['input_sha256'], runtime=plan['runtime'],
                files=files, scope=SCOPE, assembly_gate_open=False, equilibrium_established=False)
            write(out/'summary.json', report)
            return report
        except BaseException as error:
            for sig in signals: signal.signal(sig, signal.SIG_IGN)
            budget.emit(dict(state='failure', error=repr(error), counts=dict(budget.calls)))
            write(out/'failure.json', dict(schema=SCHEMA, complete=False, passed=False, error=repr(error),
                plan_sha256=digest, completed_chains=[c['chain_id'] for c in results], query_counts=dict(budget.calls),
                setup_counts=dict(budget.setup_calls), ledger=reference(out/'attempts.jsonl'),
                partial_outputs_retained=True, retries=0, replacements=0))
            raise
        finally:
            for sig, handler in previous.items(): signal.signal(sig, handler)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True); parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True); args = parser.parse_args()
    result = run(args.plan, args.plan_sha256, args.out)
    print(json.dumps(dict(complete=result['complete'], passed=result['passed'], output=str(args.out))))
