#!/usr/bin/env python3
"""Bounded differential candidate audit on frozen initial states only.

make_plan is metadata-only. Execute run only under an externally reviewed,
hash-bound single-child controller. No journal, trajectory, cloud, or bath is
read. One unpruned classifier pass supplies both the reference and the filtered
result: positive pairs must survive the necessary-condition filter. This is not
a production endpoint audit or a new hard/wall-validity certificate.
"""
from __future__ import annotations

import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
from collections import Counter
import copy
from importlib.metadata import version
import itertools
import json
from pathlib import Path
import signal
import sys
import time

from analyze_mobile_native_pocket import local_sources
from native_class_physical_labels import load_bounded_classifier, observer_setup_inventory
from native_graph_consistency import NativeGraphConsistency
from native_pair_candidates import NativePairCandidates
from run_native_class_physical_campaign import read, require, sha, write, verify_plan as verify_execution

PLAN_SCHEMA = 'native-pair-initial-audit-plan-v1'
SCHEMA = 'native-pair-initial-audit-v1'
DEFINITION_SHA = '5cf55ebe0c0f78ec6b8cdc745cac938b0fadfa3732e1e1c730864c62c651d7a9'
SHAPE_SHA = 'c7442034e7ffa4627b67c57a81117f0e9bc2d389ecf956a83dac2a5e915554c9'
WITNESS_SHA = 'd04145f5d458233bd138c384e2b244c5888f44cd9158dc34ccf969507b6ca653'
ALLOCATION = dict(bodies=264, contexts=4, starts_per_context=4, alternative_starts=16,
                  source_pairs=34716, mobile_pairs_per_start=525, alternative_pairs=8400,
                  maximum_reference_pair_queries=43116, second_classifier_pass=False,
                  new_physical_draws=0, production_endpoints=0)
LIMITS = dict(cpu_seconds=1800, wall_seconds=3600, address_space_bytes=16*1024**3, threads=1)
SCOPE = ('Differential native candidate-filter check on one frozen source and sixteen '
         'prepared initial states. Reuses hard-validity preparation evidence; no new '
         'hard/wall certificate, equilibrium estimate, assembly conclusion, production '
         'endpoint validation, or measured sampler speedup.')


def reference(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha(path))


class Bindings:
    def __init__(self): self.files = {}

    def bind(self, path, digest):
        path = Path(path).resolve()
        require(sha(path) == digest, 'Changed frozen input: '+str(path))
        require(str(path) not in self.files or self.files[str(path)] == digest, 'Conflicting input binding')
        self.files[str(path)] = digest
        return path

    def read(self, ref): return read(self.bind(ref['path'], ref['sha256']))


def runtime():
    return dict(python=sys.version, executable=reference(sys.executable),
                optimization=sys.flags.optimize,
                packages={name: version(name) for name in ('numpy', 'scipy')})


def source_closure():
    return {name: sha(path) for name, path in local_sources(__file__).items()}


def make_plan(config_path, definition_path, witness_path, compiled_path, *, root):
    """Validate/freeze metadata only. Does not instantiate any geometry class."""
    b = Bindings()
    refs = dict(config=reference(config_path), definition=reference(definition_path),
                witness=reference(witness_path), compiled_native=reference(compiled_path))
    config, definition, witness, compiled = (b.read(refs[k]) for k in refs)
    require(config['schema'] == 'evolving-dimer-benchmark-v1', 'Wrong benchmark config')
    require(refs['definition']['sha256'] == DEFINITION_SHA and refs['witness']['sha256'] == WITNESS_SHA,
            'Wrong frozen native definition/witness')
    require(config['shape']['sha256'] == SHAPE_SHA == definition['shape_sha256'], 'Wrong repaired shape')
    shape = b.read(config['shape'])
    require(len(shape['atoms']) == 4004, 'Expected complete 4004-atom physical shape')
    require(len(definition['input_sha256']) == 15, 'Incomplete frozen native input closure')
    for name, digest in definition['input_sha256'].items():
        part = Path(name)
        require(not part.is_absolute() and '..' not in part.parts, 'Unsafe native input path')
        b.bind(Path(definition_path).resolve().parent/'inputs'/part, digest)
    require(compiled['source_definition_sha256'] == DEFINITION_SHA
            and compiled['source_input_sha256'] == definition['input_sha256']
            and compiled['criteria'] == definition['criteria'], 'Compiled native identity differs')
    require(witness['native_definition_sha256'] == DEFINITION_SHA
            and witness['shape_sha256'] == SHAPE_SHA
            and witness['compiled_native_sha256'] == refs['compiled_native']['sha256'], 'Witness identity differs')
    report = witness['native_shape_compatibility']
    require(report['compiled_sha256'] == refs['compiled_native']['sha256']
            and report['expected_shape_sha256'] == SHAPE_SHA
            and report['compatible'] is True and report['hard_valid_implication_within_tolerance'] is True
            and report['native_atoms'] == report['physical_atoms'] == report['matched_atoms'] == 4004
            and report['physical_index_by_native_atom'] == list(range(4004))
            and report['unmatched_native_atoms'] == report['unmatched_physical_atoms'] == []
            and report['matched_max_center_error_a'] == report['matched_max_radius_error_a'] == 0.
            and report['pair_overlap_slack_bound_a'] == 0.
            and report['center_tolerance_a'] == 1e-10 and report['radius_tolerance_a'] == 1e-12
            and report['observer_hard_overlap_tolerance_a'] == 1e-8, 'Wrong complete zero-error shape witness')
    # Accept the already saved atom-identity witness; do not expand atom geometry.
    source = b.read(config['source_frame'])
    freeze = b.read(config['source_freeze_manifest'])
    b.read(config['source_config'])
    require(len(source['poses']) == 264 and source['boundary'] == 'spherical'
            and freeze['frame_sha256'] == config['source_frame']['sha256']
            and freeze['shape_sha256'] == SHAPE_SHA, 'Source identity/geometry domain differs')
    require(config['physical']['depletant_radius'] == 1.4 and config['physical']['activity'] == .0275,
            'Wrong conditional diagnostic target')
    authority = config['inherited_campaign']
    original = b.read(authority['config'])
    prepared = b.read(authority['prepared_manifest'])
    run_binding = b.read(authority['run_binding'])
    audit = b.read(authority['preparation_audit'])
    require(prepared['schema'] == 'evolving-dimer-prepared-starts-v1'
            and prepared['complete'] is True and prepared['passed'] is True
            and prepared['all_attempts_retained'] is True
            and prepared['config_sha256'] == authority['config']['sha256'], 'Incomplete initial preparation')
    require(run_binding['complete'] is True and run_binding['prepared_manifest'] == authority['prepared_manifest']
            and run_binding['config_sha256'] == authority['config']['sha256'], 'Prepared-state authority differs')
    require(audit['complete'] is True and audit['passed'] is True and audit['native_classifier_calls'] == 0,
            'Incomplete or unexpected preparation audit')
    require(original['source_frame'] == config['source_frame'] and original['shape'] == config['shape']
            and original['contexts'] == config['contexts'] and len(config['contexts']) == 4, 'Changed source/contexts')
    starts = []
    for item in sorted(prepared['alternative_starts'], key=lambda x: (x['context_index'], x['stream'])):
        index, stream = item['context_index'], item['stream']
        require(type(index) is int and 0 <= index < 4 and type(stream) is int and 0 <= stream < 4
                and item['status'] == 'prepared', 'Invalid prepared start identity')
        record = item['record']; start = b.read(record)
        members = [config['contexts'][index][k] for k in ('root', 'child')]
        require(len(set(members)) == 2 and all(type(i) is int and 0 <= i < 264 for i in members), 'Invalid mobile labels')
        require(start['context_index'] == index and start['stream'] == stream and start['status'] == 'prepared'
                and start['is_equilibrium_sample'] is False and len(start['selected']) == 2
                and start['source'] == [source['poses'][i] for i in members], 'Prepared selected/source poses differ')
        require(audit['input_sha256'].get(record['path']) == record['sha256']
                and run_binding['prepared_files'].get(record['path']) == record['sha256'], 'Start not independently audited/bound')
        starts.append(dict(context_index=index, stream=stream, members=members, record=record))
    require([(x['context_index'], x['stream']) for x in starts] == list(itertools.product(range(4), range(4))),
            'Require exactly sixteen original starts, without replacement')
    require(audit['input_sha256'].get(config['source_frame']['path']) == config['source_frame']['sha256'],
            'Source not bound by completed preparation audit')
    setup = observer_setup_inventory(definition_path)
    root = Path(root).resolve()
    return dict(schema=PLAN_SCHEMA, root=str(root), output=str(root/'audit'),
                **refs, source_frame=config['source_frame'], shape=config['shape'],
                starts=starts, contexts=config['contexts'], allocation=ALLOCATION, limits=LIMITS,
                setup_inventory=setup, input_sha256=b.files, source_sha256=source_closure(), runtime=runtime(),
                source_reference_mode='fresh_full_reference',
                reuse_search=dict(source_freeze=config['source_freeze_manifest'],
                    preparation_audit=authority['preparation_audit'],
                    completed_matching_full_native_receipt=None,
                    scope='Source/preparation metadata contains no stateless full-native labels; preparation audit records zero native calls. '
                          'Hysteretic/viewer labels are not a reusable reference.'),
                scope=SCOPE, launched=False)


def worker_argv(plan_path, plan_sha256, out):
    return [sys.executable, '-B', str(Path(__file__).resolve()), '--plan', str(Path(plan_path).resolve()),
            '--plan-sha256', plan_sha256, '--out', str(Path(out).resolve())]


def live_authority(plan_path, digest, plan):
    """Use the existing generic controller; no standalone scientific execution."""
    root = Path(plan['root'])
    require(plan_path == root/'audit-plan.json', 'Wrong audit plan location')
    path = root/'execution-plan.json'; execution = read(path)
    verify_execution(path, execution, sha(path))
    require(execution['root'] == str(root) and len(execution['jobs']) == 1,
            'Exactly one generic-controller child required')
    expected_files = dict(plan['input_sha256'])
    expected_files[str(plan_path)] = digest
    expected_files.update({str(path): plan['source_sha256'][name] for name, path in local_sources(__file__).items()})
    expected_files[plan['runtime']['executable']['path']] = plan['runtime']['executable']['sha256']
    require(all(execution['files'].get(path) == value for path, value in expected_files.items()),
            'Controller omitted an audit source/runtime/input binding')
    claim = read(root/'execution/claim.json')
    require(claim['plan_sha256'] == sha(path) and claim['pid'] == os.getppid()
            and claim['maximum_workers'] == claim['threads'] == 1
            and claim['retries'] == claim['replacements'] == 0, 'Missing current parent controller authority')
    stat = Path(f"/proc/{claim['pid']}/stat").read_text().rsplit(')', 1)[1].split()
    require(int(stat[19]) == claim['birth_ticks'] and stat[0] not in ('Z', 'X'), 'Parent controller identity changed')
    status = read(root/'execution/status.json'); job = execution['jobs'][0]
    require(status['plan_sha256'] == claim['plan_sha256'] and status['failure'] is None
            and status['active'] == dict(ordinal=0, id=job['id'], population=job['population'], phase='geometry')
            and job['phase'] == 'geometry', 'This audit is not the active job')
    require(job['argv'] == worker_argv(plan_path, digest, plan['output'])
            and job['terminal'] == dict(path=str(Path(plan['output'])/'summary.json'), success_contract='complete_and_passed')
            and job['cpu_limit_seconds'] == LIMITS['cpu_seconds']
            and job['wall_limit_seconds'] == LIMITS['wall_seconds']
            and job['address_space_limit_bytes'] == LIMITS['address_space_bytes'], 'Changed worker command/limits/terminal')


class Budget:
    def __init__(self, maximum, limits, emit):
        self.maximum, self.limits, self.emit = maximum, limits, emit
        self.started, self.wall = time.process_time(), time.monotonic()
        self.begun = self.completed = 0
        self.setup_calls = Counter()

    def check(self):
        require(time.process_time()-self.started <= self.limits['cpu_seconds'], 'Audit CPU budget exhausted')
        require(time.monotonic()-self.wall <= self.limits['wall_seconds'], 'Audit wall budget exhausted')

    def query(self, state_id, pair, function):
        self.check(); require(self.begun < self.maximum, 'Reference query cap exceeded')
        self.begun += 1
        identity = dict(state_id=state_id, pair=list(pair), query=self.begun)
        self.emit(dict(state='pair_begin', **identity))
        start = time.process_time(); result = function(); elapsed = time.process_time()-start
        self.completed += 1
        self.emit(dict(state='pair_complete', **identity, matches=result, classifier_cpu_seconds=elapsed))
        self.check()
        return result

    def setup_query(self, kind, limit, function, **extra):
        self.check(); require(self.setup_calls[kind+'_started'] < limit, 'Setup cap exceeded: '+kind)
        index = self.setup_calls[kind+'_started']; self.setup_calls[kind+'_started'] += 1
        self.emit(dict(state='setup_begin', kind=kind, index=index, **extra))
        value = function(); self.setup_calls[kind+'_completed'] += 1
        self.emit(dict(state='setup_complete', kind=kind, index=index, counts=dict(self.setup_calls)))
        self.check()
        return value


def audit_state(native, candidate, checker, state, state_id, budget, *, members=None, fixed_keys=()):
    """One all-pair (or all-mobile-pair) reference pass, no second classification."""
    start = time.process_time()
    candidates = set(candidate.candidate_pairs(state, mobile_labels=members))
    search_cpu = time.process_time()-start
    selected = None if members is None else set(members)
    pairs = [(a, b) for a, b in itertools.combinations(range(len(state)), 2)
             if selected is None or a in selected or b in selected]
    require(candidates <= set(pairs), 'Candidate pair inventory escapes requested state')
    budget.emit(dict(state='state_begin', state_id=state_id, members=members,
                     candidate_pairs=sorted(candidates), reference_pair_calls=len(pairs),
                     candidate_search_cpu_seconds=search_cpu))
    labels = {m['id'] for m in native.motifs}; keys = []
    classifier_cpu = time.process_time()
    for pair in pairs:
        matches = budget.query(state_id, pair, lambda pair=pair: native.classify_pair(state[pair[0]], state[pair[1]]))
        require(type(matches) is list, 'Classifier must preserve all matching labels')
        for match in matches:
            label = match['motif_id']; require(type(label) is int and label in labels, 'Unknown native label')
            keys.append((*pair, label))
        require(not matches or pair in candidates, 'False-negative candidate pair: '+str((state_id, pair)))
    reference_cpu = time.process_time()-classifier_cpu
    keys = sorted(set(keys)); combined = sorted(set(map(tuple, fixed_keys)) | set(keys))
    result = dict(state_id=state_id, reference_pair_calls=len(pairs), candidate_pair_calls=len(candidates),
                  calls_avoided=len(pairs)-len(candidates), candidate_search_cpu_seconds=search_cpu,
                  reference_loop_cpu_seconds=reference_cpu, reference_native_keys=keys,
                  instantaneous_native_keys=combined, graph=checker.check(combined), passed=True,
                  second_classifier_pass=False)
    budget.emit(dict(state='state_complete', **result))
    return result


def run(plan_path, plan_sha256, out):
    require(__debug__ and sys.flags.optimize == 0, 'Native graph assertions must remain enabled')
    plan_path, out = Path(plan_path).resolve(), Path(out).resolve()
    require(sha(plan_path) == plan_sha256, 'Unbound/changed audit plan')
    plan = read(plan_path)
    expected = make_plan(*(plan[k]['path'] for k in ('config', 'definition', 'witness', 'compiled_native')), root=plan['root'])
    require(plan == expected, 'Frozen audit declaration/input/source/runtime differs')
    require(out == Path(plan['output']), 'Output differs from frozen audit plan')
    live_authority(plan_path, plan_sha256, plan)
    require(not out.exists(), 'Exclusive audit output already exists; no retries')
    out.mkdir(parents=True)
    write(out/'claim.json', dict(schema=SCHEMA, plan_sha256=plan_sha256, allocation=ALLOCATION,
                                 limits=LIMITS, retries=0, replacements=0))
    budget = None
    with (out/'attempts.jsonl').open('x') as ledger:
        def emit(value):
            ledger.write(json.dumps(value, separators=(',', ':'), allow_nan=False)+'\n')
            ledger.flush(); os.fsync(ledger.fileno())
        def interrupted(signum, _frame):
            raise RuntimeError('Audit interrupted by signal '+str(signum))
        signals = (signal.SIGTERM, signal.SIGINT, signal.SIGXCPU)
        previous = {s: signal.signal(s, interrupted) for s in signals}
        try:
            budget = Budget(ALLOCATION['maximum_reference_pair_queries'], LIMITS, emit)
            bindings = Bindings()
            native, native_identity = load_bounded_classifier(plan['definition']['path'],
                plan['setup_inventory'], budget, bindings.bind)
            motifs = [dict(position=m['relative_position'], orientation=m['relative_orientation']) for m in native.motifs]
            candidate = NativePairCandidates(native.member_positions, motifs,
                native.definition['criteria']['body_member_position_entry_A'])
            source = read(plan['source_frame']['path'])['poses']
            checker = NativeGraphConsistency(native.motifs, len(source))
            results = [audit_state(native, candidate, checker, source, 'source', budget)]
            source_keys = results[0]['reference_native_keys']
            fixed = {index: [k for k in source_keys if not {c['root'], c['child']}.intersection(k[:2])]
                     for index, c in enumerate(plan['contexts'])}
            for item in plan['starts']:
                state = copy.deepcopy(source)
                for member, pose in zip(item['members'], read(item['record']['path'])['selected']): state[member] = pose
                ident = f"context-{item['context_index']}-stream-{item['stream']}"
                results.append(audit_state(native, candidate, checker, state, ident, budget,
                    members=item['members'], fixed_keys=fixed[item['context_index']]))
            require(budget.begun == budget.completed == ALLOCATION['maximum_reference_pair_queries'],
                    'Incomplete/unexpected reference inventory')
            for path, digest in plan['input_sha256'].items(): require(sha(path) == digest, 'Input changed during audit')
            require(source_closure() == plan['source_sha256'], 'Source changed during audit')
            require(sha(plan_path) == plan_sha256 and runtime() == plan['runtime'], 'Plan/runtime changed during audit')
            result = dict(schema=SCHEMA, complete=True, passed=True, plan_sha256=plan_sha256,
                allocation=ALLOCATION, reference_queries_begun=budget.begun, reference_queries_completed=budget.completed,
                setup_counts=dict(budget.setup_calls), setup_inventory=plan['setup_inventory'], native_identity=native_identity,
                source_frame=plan['source_frame'], shape=plan['shape'], definition=plan['definition'],
                states=results, fixed_native_keys_by_context=fixed, input_sha256=plan['input_sha256'],
                source_sha256=plan['source_sha256'], runtime=plan['runtime'], scope=SCOPE,
                total_cpu_seconds=time.process_time()-budget.started,
                total_wall_seconds=time.monotonic()-budget.wall,
                ledger=reference(out/'attempts.jsonl'), production_gate_open=False)
            write(out/'summary.json', result)
            return result
        except BaseException as error:
            for s in signals: signal.signal(s, signal.SIG_IGN)
            emit(dict(state='failure', error=repr(error)))
            write(out/'failure.json', dict(schema=SCHEMA, complete=False, passed=False, plan_sha256=plan_sha256,
                error=repr(error), reference_queries_begun=budget.begun if budget else 0,
                reference_queries_completed=budget.completed if budget else 0,
                setup_counts=dict(budget.setup_calls) if budget else {}, ledger=reference(out/'attempts.jsonl'),
                retry=False, replacement=False, scope=SCOPE))
            raise
        finally:
            for s, handler in previous.items(): signal.signal(s, handler)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    result = run(args.plan, args.plan_sha256, args.out)
    print(json.dumps(dict(complete=result['complete'], passed=result['passed'], output=str(args.out))))
