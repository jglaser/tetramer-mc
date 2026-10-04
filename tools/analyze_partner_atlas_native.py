#!/usr/bin/env python3
"""Second-stage internal-native observation of completed partner-atlas chains.

Plan fields: root, output, contact {execution_plan,analysis,manifest,input_binding},
config, protocol, native_plan (BoundFiles), chains[32] {id,job,trajectory,terminal,
initial}, allocation, limits, input_sha256, source_sha256, runtime. The producer's
frozen native plan remains unchanged. Its completed classifier authority supplies
16 cached local/m4 summaries and the original bounded constructor inventory.
No old journal, proposal arithmetic, external native graph or physical draw.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import signal
import sys
import time

import analyze_surrogate_internal_native as prior
import prepare_partner_atlas_benchmark as preparation

driver, inherited = prior.driver, prior.inherited
read, require, sha, write = prior.read, prior.require, prior.sha, prior.write
completed_report = prior.completed_report
PairObserver, NativeBudget, append = prior.PairObserver, prior.NativeBudget, prior.append
reduce_trace, pose_arrays = prior.reduce_trace, prior.pose_arrays
load_bounded_classifier = prior.load_bounded_classifier
ARMS = ('partner_atlas_direct', 'partner_atlas_m1', 'partner_atlas_m8', 'partner_atlas_flat8')
CACHED = ('local', 'm4')
STARTS, MEMBERS = prior.STARTS, prior.MEMBERS
PLAN_SCHEMA = 'partner-atlas-internal-native-plan-v1'
SCHEMA = 'partner-atlas-internal-native-v1'
ALLOCATION = dict(new_chains=32, cached_chains=16, blocks=4608, warmup=512,
    retained_endpoints=147488, production_endpoints=131072,
    main_pair_query_cap=147488, checkpoint_pair_query_cap=0, fixture_queries=0,
    external_pair_queries=0, contact_geometry_queries=0, arithmetic_replays=0, new_draws=0)
LIMITS = dict(prior.LIMITS)
SCOPE = prior.SCOPE
identity, expected = prior.identity, prior.expected


def source_paths():
    return prior.local_sources(__file__)


def source_closure():
    return {name: sha(path) for name, path in source_paths().items()}


def native_context(authority, config, b):
    checked = preparation.validate_native_authority(b, authority['root'], authority['summary']['sha256'], config)
    require(checked == authority, 'Producer native authority changed')
    report = b.load(authority['summary'])
    require(report['schema'] == prior.SCHEMA and report['allocation'] == prior.ALLOCATION
        and report['external_native_observer'] is False and report['cycle_observer'] is False
        and report['assembly_gate_open'] is False and report['classifier'] == authority['classifier'],
        'Wrong completed internal-native authority')
    refs = authority['refs']; initial_plan = b.load(authority['original_native_plan'])
    setup = prior.observer_setup_inventory(refs['definition']['path'])
    require(setup == initial_plan['setup_inventory'], 'Native constructor allocation changed')
    source = b.load(refs['source_frame']); shape = b.load(refs['shape'])
    require(len(source['poses']) == 264 and source['boundary'] == 'spherical'
        and len(shape['atoms']) == 4004, 'Native source/shape inventory changed')
    return dict(refs=refs, source=source['poses'], setup=setup, report=report)


def admit_inputs(plan):
    """Complete contact lifecycle first; metadata/hash admission, no queries."""
    require(plan['schema'] == PLAN_SCHEMA and plan['allocation'] == ALLOCATION and plan['limits'] == LIMITS,
        'Changed native scope/allocation/limits')
    b = inherited.Bindings()
    contact = completed_report(plan['contact'], b)
    require(contact['schema'] == 'partner-atlas-analysis-v1' and contact['new_chains'] == 32
        and contact['reused_control_chains'] == 40 and len(contact['chains']) == 72
        and contact['native_observer'] is False and contact['old_geometry_queries'] == contact['new_physical_draws'] == 0,
        'Completed72-chain partner contact/arithmetic audit required')
    for key in ('config', 'protocol', 'native_plan'):
        ref = plan[key]
        require(contact['input_sha256'].get(ref['path']) == contact['_execution_files'].get(ref['path']) == ref['sha256'],
            'Producer metadata not bound by completed contact audit')
    config, protocol, declared = (b.load(plan[k]) for k in ('config', 'protocol', 'native_plan'))
    require(config['protocol'] == plan['protocol'] and protocol['native_observer_plan'] == plan['native_plan']
        and protocol['schema'] == 'partner-atlas-protocol-v1' and protocol['assembly_gate_open'] is False
        and declared == preparation.native_plan(protocol['native_authority'])
        and declared['new_chains'] == 32 and declared['new_endpoints'] == 147488
        and declared['queries_per_retained_endpoint'] == 1 and declared['dispatched'] is False,
        'Frozen producer/native declaration differs')
    require(config['partner_atlas_policy'] == preparation.policy()
        and [config['contexts'][0][k] for k in ('root', 'child')] == MEMBERS
        and config['physical']['depletant_radius'] == 1.4 and config['physical']['activity'] == .0275
        and config['allocation']['warmup_blocks'] == 512 and config['allocation']['production_blocks'] == 4096,
        'Conditional target/arm schedule differs')
    native = native_context(protocol['native_authority'], config, b)
    require(config['shape'] == native['refs']['shape'] and config['source_frame'] == native['refs']['source_frame'],
        'Classifier shape/source differs from conditional producer')
    index = {identity(c['job']): c for c in contact['chains']}
    require(len(index) == 72 and set(index) == expected(ARMS + prior.RIGID + CACHED), 'Wrong72 contact identity product')
    require(len(config['jobs']) == 32 and len({identity(j) for j in config['jobs']}) == 32
        and {identity(j) for j in config['jobs']} == expected(ARMS), 'Wrong producer identity inventory')
    jobs = {identity(j): j for j in config['jobs']}
    require(len(plan['chains']) == 32 and len({c['id'] for c in plan['chains']}) == 32
        and {identity(c['job']) for c in plan['chains']} == expected(ARMS), 'Wrong32 native-label allocation')
    chains = []
    for chain in plan['chains']:
        require(type(chain['id']) is str and chain['id'] and all(c.isalnum() or c in '-_' for c in chain['id']), 'Unsafe chain ID')
        key = identity(chain['job']); old = index[key]
        require(chain['job'] == old['job'] == jobs[key] and old['reused_control'] is False
            and chain['trajectory'] == old['trajectory'], 'New chain/job/journal differs')
        for ref in (chain['terminal'], chain['trajectory']):
            require(contact['input_sha256'].get(ref['path']) == contact['_execution_files'].get(ref['path']) == ref['sha256'],
                'Journal/terminal lacks completed arithmetic authority')
            b.bind(ref['path'], ref['sha256'])
        terminal = b.load(chain['terminal']); cpu = old['metrics']['full_sampler_cpu_seconds']
        require(terminal['complete'] is True and terminal['conditional_target'] is True
            and terminal['job'] == chain['job'] and terminal['trajectory'] == chain['trajectory']
            and terminal['counts'] == old['counts'] and terminal['blocks'] == 4608
            and terminal['config_sha256'] == plan['config']['sha256']
            and terminal['cpu_seconds'] == cpu and type(cpu) in (int, float) and math.isfinite(cpu) and cpu > 0
            and not (Path(chain['terminal']['path']).parent/'failure.json').exists(), 'Sampler terminal/counts/CPU differs')
        start = chain['initial']; require(start['kind'] == chain['job']['initialization'], 'Initialization differs')
        if start['kind'] == 'source':
            require(set(start) == {'kind'}, 'Unexpected source fields')
            poses = [native['source'][i] for i in MEMBERS]
        else:
            require(set(start) == {'kind', 'record'}, 'Unexpected prepared-start fields')
            ref = start['record']; value = b.load(ref)
            require(contact['input_sha256'].get(ref['path']) == contact['_execution_files'].get(ref['path']) == ref['sha256']
                and value['status'] == 'prepared' and value['context_index'] == 0
                and value['stream'] == chain['job']['stream'], 'Prepared start not bound/matched')
            poses = value['selected']
        require(type(poses) is list and len(poses) == 2, 'Wrong initial pose pair')
        for pose in poses: pose_arrays(pose)
        chains.append(dict(chain, initial_poses=poses, full_sampler_cpu_seconds=cpu))
    native_rows = native['report']['chains']
    require(len(native_rows) == 64 and len({identity(c['job']) for c in native_rows}) == 64
        and {identity(c['job']) for c in native_rows} == expected(prior.FLEXIBLE + prior.RIGID + CACHED),
        'Wrong completed64 native inventory')
    entries = config['control_analysis']['observations']
    require(len(entries) == 40 and len({identity(e['job']) for e in entries}) == 40
        and {identity(e['job']) for e in entries} == expected(prior.RIGID + CACHED), 'Wrong40 contact cache inventory')
    entry_index = {identity(e['job']): e for e in entries}; controls = []
    for c in native_rows:
        key = identity(c['job'])
        if key not in expected(CACHED): continue
        old = index[key]; entry = entry_index[key]; ref = entry['terminal']
        require(contact['input_sha256'].get(ref['path']) == ref['sha256'], 'Cached terminal unbound by contact audit')
        terminal = b.load(ref); m = c['metrics']; cpu = m['full_sampler_cpu_seconds']
        require(c['reused_native_control'] is True and old['reused_control'] is True
            and c['job'] == old['job'] == entry['job'] == terminal['job']
            and c['trajectory'] == old['trajectory'] == terminal['trajectory']
            and terminal['complete'] is True and terminal['conditional_target'] is True
            and terminal['blocks'] == 4608 and terminal['counts'] == old['counts']
            and cpu == old['metrics']['full_sampler_cpu_seconds'] == terminal['cpu_seconds']
            and type(cpu) in (int, float) and math.isfinite(cpu) and cpu > 0
            and m['schema'] == 'internal-native-metrics-v1' and m['production_samples'] == 4096
            and m['retained_endpoints'] == 4609
            and not (Path(ref['path']).parent/'failure.json').exists(), 'Cached native identity/terminal/metric scope differs')
        controls.append(dict(job=c['job'], reused_native_control=True, trajectory=c['trajectory'], terminal=ref,
            metrics=copy.deepcopy(m), native_authority=protocol['native_authority']['summary']))
    require(len(controls) == 16 and {identity(c['job']) for c in controls} == expected(CACHED), 'Incomplete16 native controls')
    return dict(native=native, chains=chains, controls=controls), b.files


def endpoints(rows, chain, *, blocks=4608, warmup=512):
    """Own partner7-row adapter, only after authenticated arithmetic completion."""
    rows = iter(rows); cpu = -1.; direct = chain['job']['arm'] == ARMS[0]
    require(chain['job']['arm'] in ARMS, 'Unknown partner-atlas arm')
    def take(kind, block):
        nonlocal cpu
        row = next(rows, None)
        require(type(row) is dict and row.get('kind') == kind and type(row.get('block')) is int
            and row['block'] == block and not any(k in row for k in ('fatal_error', 'error', 'bath_failure')),
            'Missing/out-of-order/fatal admitted row')
        now = row['sampler_cpu_seconds']
        require(type(now) in (int, float) and math.isfinite(now) and 0 <= now
            and cpu <= now <= chain['full_sampler_cpu_seconds'], 'Nonmonotone/nonfinite CPU')
        cpu = now; return row
    def point(row, block):
        require(type(row['selected']) is list and len(row['selected']) == 2, 'Expected retained pair')
        for pose in row['selected']: pose_arrays(pose)
        return dict(block=block, production=block > warmup, selected=row['selected'], sampler_cpu_seconds=cpu)
    first = take('initial', 0)
    require(first['job'] == chain['job'] and first['conditional_target'] is True
        and first['selected'] == chain['initial_poses'] and first.get('production', False) is False,
        'Initial identity/pose differs')
    selected = copy.deepcopy(first['selected']); yield point(first, 0)
    for block in range(1, blocks+1):
        for attempt, slot in enumerate((0, 1, 0, 1)):
            local = take('local', block)
            require(type(local['attempt']) is int and local['attempt'] == attempt and local['member'] == MEMBERS[slot]
                and local['status'] in ('hard_rejected', 'completed') and type(local['accepted']) is bool
                and local['old'] == selected[slot], 'Local source/sequence differs')
            require(local['status'] != 'hard_rejected' or not local['accepted'], 'Accepted hard rejection')
            if local['accepted']: selected[slot] = local['proposed']
            require(local['retained'] == selected, 'Local retained pair differs')
        begun = take('partner_atlas_attempt_begun', block)
        outcome = take('flexible_partner_atlas_direct' if direct else 'flexible_partner_atlas_chain', block)
        require(begun['status'] == 'begun' and begun['members'] == outcome['members'] == MEMBERS
            and begun['inner_filter'] is outcome['inner_filter'] is (not direct)
            and begun['old'] == outcome['old'] == selected
            and outcome['status'] in ('completed', 'identity_self_loop') and type(outcome['accepted']) is bool,
            'Wrong/incomplete collective source')
        require(outcome['status'] != 'identity_self_loop' or not outcome['accepted'], 'Accepted identity')
        if outcome['accepted']: selected = copy.deepcopy(outcome['proposed'])
        require(outcome['retained'] == selected, 'Collective retained pair differs')
        row = take('retained_block', block)
        require(row['production'] is (block > warmup) and row['selected'] == selected, 'Retained endpoint differs')
        yield point(row, block)
    require(next(rows, None) is None, 'Unexpected journal tail')


def validate_plan(plan):
    require(plan['source_sha256'] == source_closure() and plan['runtime'] == inherited.runtime(), 'Source/runtime differs')
    context, inputs = admit_inputs(plan)
    require(inputs == plan['input_sha256'], 'Incomplete/extra input closure')
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
    require(all(execution['files'].get(p) == h for p, h in closure.items()), 'Unbound worker input/source')


def run(plan_path, digest, out):
    path = Path(plan_path).resolve(); require(sha(path) == digest and sys.flags.optimize == 0, 'Plan hash/runtime differs')
    plan = read(path); out = Path(out).resolve(); root = Path(plan['root']).resolve()
    require(str(root) == plan['root'] and str(out) == plan['output'] and out.is_relative_to(root)
        and out != root and not out.exists(), 'Fresh output under frozen root required')
    live_authority(path, digest, plan); context = validate_plan(plan)
    require(not any(Path(p).is_relative_to(out) for p in plan['input_sha256']), 'Input/output collision')
    out.mkdir(); (out/'chains').mkdir(); results = []; completed = 0
    started = time.process_time(); wall = time.monotonic(); old_handlers = {}
    def stop(signum, frame): raise RuntimeError('Interrupted signal '+str(signum))
    with (out/'attempts.jsonl').open('x') as journal:
        budget = NativeBudget(ALLOCATION, LIMITS, lambda row: append(journal, row), started=started, wall=wall)
        budget.calls.update(dict(main_begun=0, main_completed=0, checkpoint_begun=0, checkpoint_completed=0))
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGXCPU): old_handlers[sig] = signal.signal(sig, stop)
        try:
            info = context['native']; refs = info['refs']
            def bind(p, h):
                require(plan['input_sha256'].get(str(Path(p).resolve())) == h and sha(p) == h, 'Unbound classifier input')
                return Path(p)
            native, classifier = load_bounded_classifier(refs['definition']['path'], info['setup'], budget, bind)
            require(classifier == info['report']['classifier'], 'Loaded classifier differs from completed authority')
            observer = PairObserver(native, budget); files = {}
            for chain in context['chains']:
                budget.check(); tick = time.process_time(); trace = []; before = dict(budget.calls); hits = observer.hits
                budget.emit(dict(state='chain_begin', chain_id=chain['id'], job=chain['job']))
                target = out/'chains'/(chain['id']+'.jsonl')
                with Path(chain['trajectory']['path']).open('rb') as raw, target.open('x') as dest:
                    def rows():
                        while line := raw.readline(LIMITS['max_record_bytes']+1):
                            require(len(line) <= LIMITS['max_record_bytes'] and line.endswith(b'\n'), 'Oversized/truncated row')
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
            require(completed == ALLOCATION['retained_endpoints'] and len(results) == 32
                and budget.calls['main_begun'] == budget.calls['main_completed']
                and budget.calls['checkpoint_begun'] == budget.calls['checkpoint_completed'] == 0, 'Incomplete allocation')
            require(source_closure() == plan['source_sha256'] and inherited.runtime() == plan['runtime'] and sha(path) == digest,
                'Source/runtime/plan changed')
            for p, h in plan['input_sha256'].items(): require(sha(p) == h, 'Input changed '+p)
            budget.check(); files[str(out/'attempts.jsonl')] = sha(out/'attempts.jsonl')
            result = dict(schema=SCHEMA, complete=True, passed=True, plan_sha256=digest, allocation=ALLOCATION,
                chains=results+context['controls'], classifier=classifier, query_counts=dict(budget.calls),
                setup_counts=dict(budget.setup_calls), setup_inventory=info['setup'], files=files,
                inherited_classifier_validation=plan['native_plan'], new_fixture_queries=0,
                input_sha256=plan['input_sha256'], source_sha256=plan['source_sha256'], runtime=plan['runtime'],
                analysis_cpu_seconds=time.process_time()-started, scope=SCOPE,
                primary_control_arms=['local'], contextual_control_arms=['m4'],
                external_native_observer=False, cycle_observer=False, assembly_gate_open=False)
            write(out/'summary.json', result); return result
        except BaseException as error:
            for sig in old_handlers: signal.signal(sig, signal.SIG_IGN)
            budget.emit(dict(state='failure', error=repr(error), counts=dict(budget.calls)))
            write(out/'failure.json', dict(schema=SCHEMA, complete=False, passed=False, error=repr(error),
                plan_sha256=digest, completed_chains=len(results), completed_endpoints=completed,
                query_counts=dict(budget.calls), setup_counts=dict(budget.setup_calls),
                partial_outputs_retained=True, retries=0, replacements=0)); raise
        finally:
            for sig, handler in old_handlers.items(): signal.signal(sig, handler)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True); parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); run(args.plan, args.plan_sha256, args.out)
