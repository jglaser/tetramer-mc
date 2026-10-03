#!/usr/bin/env python3
"""Separately bounded, selected full-interval audit of fresh physical v7 rows.

No work runs on import. ``select(plan)`` is arithmetic/metadata only. Its result
must be written to a fresh file and hash-bound as ``selection`` in an external
execution plan before ``run``. The same plan binds a pre-draw ``preselection``
declaration and an external ``review`` attesting that declaration preceded the
first draw and this population has never undergone this full-geometry audit.
Those chronology/uniqueness attestations are external authority, not inferred
from file timestamps. No preparation here allocates or launches protein work.

Plans otherwise use the endpoint labeler's population/definition/reference_region/
strata/observer_setup/source_sha256/runtime/limits fields, plus target, algebra,
labels and output:{receipt,journal,failure}. Additional strict limits are
max_full_geometry_queries<=20 and max_axis_queries<=60. Selected IDs are exactly
16 unconditional IDs plus at most one largest Qz contributor per decision region,
ties by lowest ID, deduplicated, no replacement for empty regions. Original full
row byte offsets/hashes are bound; no truncated or renumbered pseudo-population.
"""
from __future__ import annotations

import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import hashlib
import heapq
import json
import math
from pathlib import Path
import signal
import sys
import threading
import time

import numpy as np

import analyze_native_class_physical_populations as statistics
import native_class_physical_labels as endpoints
import native_class_line_physical_algebra_audit as streaming
import native_class_line_physical_reference as full
from analyze_mobile_native_pocket import local_sources
from native_class_line_weight_row import validate_row

require, read, sha, close = streaming.require, streaming.read, streaming.sha, streaming.close
SCHEMA = 'native-class-selected-full-geometry-v1'
PLAN_SCHEMA = 'native-class-selected-full-geometry-plan-v1'
PRESELECTION_SCHEMA = 'native-class-unconditional-audit-ids-v1'
SELECTION_SCHEMA = 'native-class-selected-full-row-bindings-v1'


def preselection(population, samples, seed):
    """Metadata-only deterministic hash ranking, to freeze before physical draws."""
    require(type(population) is str and population, 'Missing population ID')
    require(type(samples) is int and samples >= 16, 'At least 16 unconditional attempts required')
    require(type(seed) is int and 0 <= seed < 2**64, 'Invalid independent audit seed')
    def rank(draw):
        return statistics.fingerprint(dict(domain=PRESELECTION_SCHEMA, population=population,
            samples=samples, audit_seed=seed, draw=draw)), draw
    ids = sorted(heapq.nsmallest(16, range(samples), key=rank))
    return dict(schema=PRESELECTION_SCHEMA, population=population, samples=samples,
                audit_seed=seed, draw_ids=ids, rule='16 smallest SHA256 ranks, ties by draw ID')


def selected_inventory(unconditional, maxima):
    """Each region may contribute one ID; identical IDs retain all reasons."""
    require(len(unconditional) == len(set(unconditional)) == 16, 'Expected 16 distinct unconditional IDs')
    require(set(maxima) == set(statistics.DECISIONS), 'Decision regions differ')
    reasons = {draw: ['unconditional'] for draw in unconditional}
    for region in statistics.DECISIONS:
        value = maxima[region]
        if value is not None:
            reasons.setdefault(value['draw'], []).append('largest_Qz:'+region)
    require(16 <= len(reasons) <= 20, 'Selected full-row allocation exceeded')
    return reasons


def _metadata(plan, bindings):
    require(plan['schema'] == PLAN_SCHEMA, 'Wrong selected-geometry plan')
    root = Path(plan['population']['root']).resolve()
    for name in ('manifest', 'summary', 'samples', 'attempts'):
        bindings.bind(root/(name+('.jsonl' if name in ('samples', 'attempts') else '.json')),
                      plan['population'][name+'_sha256'])
    manifest, summary, region, config, law, witness = streaming._provenance(root, bindings.bind)
    require(manifest['samples'] == plan['population']['samples']
            and manifest['seed'] == plan['population']['seed'], 'Population allocation differs')
    target_region, descriptor, target_id = statistics.target_identity(plan['target'], bindings)
    require(region == target_region, 'Selected population target differs')
    require(plan['target']['native_definition'] == plan['definition']
            and plan['target']['old_r5_region'] == plan['reference_region'], 'Target reference aliases differ')
    require(read(bindings.reference(plan['strata'])) == statistics.STRATA, 'Original strata differ')
    algebra, labels = [read(bindings.reference(plan[key])) for key in ('algebra', 'labels')]
    for kind, receipt in [('algebra', algebra), ('labels', labels)]:
        statistics.check_receipt(receipt, root, manifest, summary, target_id, kind, bindings, descriptor)
    label_path = bindings.reference(labels['labels'])
    declaration = read(bindings.reference(plan['preselection']))
    require(declaration == preselection(plan['population']['id'], manifest['samples'], declaration['audit_seed']),
            'Predeclared unconditional IDs differ')
    return dict(root=root, manifest=manifest, summary=summary, region=region, config=config,
                law=law, witness=witness, labels=labels, label_path=label_path,
                declaration=declaration, target_and_regions_sha256=target_id)


def _select(context, plan, budget=None):
    """Scan original all-attempt streams; no observer or geometry constructor."""
    count = context['manifest']['samples']; maximum = plan['limits']['max_record_bytes']
    require(type(maximum) is int and maximum > 0, 'Invalid record byte cap')
    require(count <= plan['limits']['max_rows'], 'Attempt budget exceeded')
    paths = [context['root']/'samples.jsonl', context['root']/'attempts.jsonl', context['label_path']]
    streams = [endpoints.BoundedLines(path, maximum) for path in paths]
    unconditional = context['declaration']['draw_ids']; retained = {}
    maxima = {region: None for region in statistics.DECISIONS}
    try:
        for index in range(count):
            if budget is not None: budget.check()
            row, attempt, label = (stream.next() for stream in streams)
            require(row is not None and label is not None and type(attempt) is dict
                    and type(attempt.get('draw')) is int and attempt == dict(draw=index, state='begin'),
                    'Missing/reordered all-attempt input')
            value = validate_row(row, expected_draw=index, manifest=context['manifest'], region=context['region'])
            regions = statistics.label_regions(label, row, value, streams[0].last['sha256'])
            entry = dict(draw=index, sample_record=dict(streams[0].last), label_record=dict(streams[2].last))
            if index in unconditional: retained[index] = entry
            for region in statistics.DECISIONS:
                previous = maxima[region]
                # Strict > preserves the first (lowest ID) exact weight tie.
                if region in regions and (previous is None or value['z'] > previous['log_weight']):
                    maxima[region] = dict(draw=index, log_weight=value['z'], row=entry)
        expected = [context['summary']['samples_sha256'], context['summary']['attempts_sha256'],
                    context['labels']['labels']['sha256']]
        for stream, digest in zip(streams, expected):
            require(stream.next() is None and stream.lines == count and stream.hash.hexdigest() == digest,
                    'Unconditional input stream changed')
    finally:
        for stream in streams: stream.close()
    for value in maxima.values():
        if value is not None: retained[value['draw']] = value['row']
    reasons = selected_inventory(unconditional, maxima)
    return dict(schema=SELECTION_SCHEMA, population=plan['population'],
        target_and_regions_sha256=context['target_and_regions_sha256'], unconditional_denominator=count,
        preselection=plan['preselection'], algebra=plan['algebra'], labels=plan['labels'],
        decision_maxima={key: None if value is None else {k: value[k] for k in ('draw', 'log_weight')}
                         for key, value in maxima.items()},
        rows=[dict(retained[draw], reasons=reasons[draw]) for draw in sorted(reasons)],
        full_geometry_rows=len(reasons), maximum_full_geometry_rows=20, replacements=0,
        trust_boundary='Largest-contributor IDs are outcome-selected diagnostics, not unbiased samples; '
                       'zero-observation regions receive no replacement and no upper bound.')


def select(plan):
    """Return a reviewable full-row selection without performing geometry."""
    bindings = statistics.Bindings(); context = _metadata(plan, bindings)
    result = _select(context, plan); bindings.recheck()
    return result


def read_bound_record(record, maximum):
    require(type(record['offset']) is int and record['offset'] >= 0
            and type(record['bytes']) is int and 0 < record['bytes'] <= maximum, 'Invalid selected byte span')
    with Path(record['path']).open('rb') as stream:
        stream.seek(record['offset']); raw = stream.read(record['bytes'])
    require(len(raw) == record['bytes'] and raw.endswith(b'\n') and raw.count(b'\n') == 1
            and hashlib.sha256(raw).hexdigest() == record['sha256'], 'Selected full row bytes changed')
    row = streaming.loads(raw)
    require(type(row) is dict, 'Selected row is not an object')
    return row


def frozen_line_observer(observer, compiled, classifier_binding, bind):
    """Nominal interface bridge; preserve the initialized frozen implementation.

    The full reference requires its local NativeContactRegions nominal type,
    while endpoint setup correctly loads the archived class in a distinct
    module. Neither constructor is called here. Frozen methods remain first in
    the MRO and every initialized field is transferred by identity. The line
    reference reads this already checked original data, not a new compiled-only
    classifier. Existing sources and the caller's observer are never modified.
    """
    frozen_class = type(observer); module = sys.modules.get(frozen_class.__module__)
    require(module is not None and frozen_class is getattr(module, 'NativeContactRegions', None),
            'Exact initialized frozen native class required')
    runtime = bind(module.__file__, classifier_binding['runtime_sha256'])
    definition_path = bind(classifier_binding['definition'], classifier_binding['definition_sha256'])
    definition = read(definition_path)
    require(observer.definition == definition and observer.definition_sha256 == classifier_binding['definition_sha256']
            and definition['input_sha256']['source/native_contact_regions.py'] == sha(runtime),
            'Frozen observer source/definition identity differs')
    require(module.CRITERIA == full.line.CRITERIA == definition['criteria'], 'Frozen/reference criteria differ')
    full.line.compare_compiled_definition(compiled, observer)
    local_class = full.line.NativeContactRegions
    local_source = bind(sys.modules[local_class.__module__].__file__)
    fields = ('member_positions', 'member_rotations', 'atoms', 'radii', 'residues', 'residue_count',
              'references', 'motifs', 'motif_positions', 'motif_rotations', 'fixed_poses',
              'definition', 'definition_sha256')
    require(all(name in observer.__dict__ for name in fields), 'Incomplete initialized native line data')
    if isinstance(observer, local_class):
        bridge = observer
    else:
        bridge_type = type('FrozenNativeLineObserverBridge', (frozen_class, local_class), {'__module__': __name__})
        bridge = object.__new__(bridge_type)
        bridge.__dict__.update(observer.__dict__)
        require(bridge_type.__mro__[1] is frozen_class, 'Frozen method precedence changed')
    require(isinstance(bridge, local_class) and all(getattr(bridge, name) is getattr(observer, name) for name in fields),
            'Frozen initialized line data changed in interface bridge')
    for method in ('_contacts', 'classify_pair', 'classify'):
        require(getattr(bridge, method).__func__ is getattr(frozen_class, method), 'Frozen classifier method changed')
    return bridge, dict(mode='initialized frozen class first in nominal local-interface MRO; no constructors',
        frozen_runtime=dict(path=str(runtime), sha256=sha(runtime)),
        local_interface=dict(path=str(local_source), sha256=sha(local_source)),
        definition_sha256=classifier_binding['definition_sha256'], initialized_fields=list(fields),
        initialized_fields_preserved_by_identity=True, additional_observer_setup_queries=0)


def audit_selected_row(row, label, context, recon, budget):
    """Reuse the unpruned full reference, never the saved-interval evaluator."""
    index = row['draw']; u = np.asarray(row['latent'])
    accounting = validate_row(row, expected_draw=index, manifest=context['manifest'], region=context['region'])
    result = budget.query('full_geometry', index,
        lambda: full.compact_density(recon, u, row['native_class_line_density']))
    require(not result.get('conditioning_disabled') and len(result['axes']) == 3,
            'Selected full geometry needs all three active line axes')
    close(row['log_proposal_density'], result['log_density'], 'Full physical proposal q differs', atol=2e-7, rtol=1e-11)
    generated = full.line.audit_draw(row['native_class_line_draw'], recon, u, result)
    raw, position, rotation, jac = recon.decode(u)
    close(row['pose']['position'], position, 'Full physical position differs')
    close(full.line.pose_arrays(row['pose'])[1], rotation, 'Full physical orientation differs')
    close(row['log_physical_jacobian'], jac, 'Full physical Jacobian differs')
    memberships = 0
    for axis in result['axes']:
        coordinate = recon.raw(u)[axis['axis']]
        if 'segment' in axis and axis['segment'][0] <= coordinate <= axis['segment'][1]:
            memberships += 1
            require(full.line.contains(axis['hard_free_intervals'], coordinate) == label['hard_valid'],
                    'Full H membership differs from independent endpoint')
            if label['applicable']:
                require(full.line.contains(axis['native_intervals'], coordinate) == label['native'],
                        'Full native membership differs from complete endpoint classifier')
                require(full.line.contains(axis['exclusion_contact_intervals'], coordinate) == label['exclusion_contact'],
                        'Full exclusion membership differs from independent endpoint')
    if math.isfinite(accounting['z']): require(memberships == 3, 'Contributing endpoint lacks full line membership')
    return dict(draw=index, axes=3, membership_axes=memberships,
        interval_endpoints=max(result['interval_error'], generated['endpoint_error']),
        inverse_cdf=generated['inverse_error'], log_proposal_density=result['log_density'])


def run(plan_path, *, plan_sha256):
    """Exclusive, no-resume worker; external controller owns capacity/launch."""
    require(threading.current_thread() is threading.main_thread(), 'One main worker thread required')
    plan_path = Path(plan_path).resolve(); streaming._digest(plan_sha256, 'execution plan')
    require(sha(plan_path) == plan_sha256, 'Frozen selected-geometry plan changed')
    plan = read(plan_path); bindings = statistics.Bindings(); bindings.bind(plan_path, plan_sha256)
    paths = {key: Path(plan['output'][key]).resolve() for key in ('receipt', 'journal', 'failure')}
    require(len(set(paths.values())) == 3 and all(not p.exists() and p.parent.is_dir() for p in paths.values()),
            'Fresh distinct outputs with existing parents required')
    budget = None; completed = []; active = None; source_bindings = {}; phase = 'metadata'
    def interrupted(signum, _frame): raise SystemExit(128+signum)
    require(all(signal.getitimer(timer) == (0., 0.) for timer in (signal.ITIMER_REAL, signal.ITIMER_PROF)),
            'Existing process timer is unsupported')
    previous = {sig: signal.signal(sig, interrupted) for sig in
                (signal.SIGTERM, signal.SIGXCPU, signal.SIGALRM, signal.SIGPROF)}
    with paths['journal'].open('x') as ledger:
        def emit(event):
            ledger.write(json.dumps(event, allow_nan=False)+'\n'); ledger.flush(); os.fsync(ledger.fileno())
        try:
            budget = endpoints.Budget(plan['limits'], emit)
            signal.setitimer(signal.ITIMER_REAL, plan['limits']['wall_seconds'])
            signal.setitimer(signal.ITIMER_PROF, plan['limits']['cpu_seconds'])
            for role, cap in [('full_geometry', 20), ('axis', 60)]:
                value = plan['limits']['max_'+role+'_queries']
                require(type(value) is int and 0 < value <= cap, 'Invalid selected geometry query cap')
            require(plan['limits']['max_full_geometry_queries'] >= 16, 'Cannot cover unconditional IDs')
            sources = local_sources(__file__)
            require(set(plan['source_sha256']) == set(sources), 'Frozen selected source closure differs')
            for name, source in sources.items():
                module = sys.modules.get(source.stem)
                if module is not None: require(Path(module.__file__).resolve() == source, 'Loaded source path differs')
                bindings.bind(source, plan['source_sha256'][name]); source_bindings[str(source)] = sha(source)
            runtime = streaming.runtime_identity()
            require(runtime == plan['runtime'], 'Frozen selected runtime differs')
            for path, digest in runtime['file_sha256'].items(): bindings.bind(path, digest)
            context = _metadata(plan, bindings)
            selection = read(bindings.reference(plan['selection']))
            require(selection == _select(context, plan, budget), 'Fresh full-row selection differs')
            review = read(bindings.reference(plan['review']))
            require(review['complete'] is True and review['passed'] is True
                    and review['preselection'] == plan['preselection'] and review['selection'] == plan['selection']
                    and review['population'] == plan['population'] and review['limits'] == plan['limits']
                    and review['predeclared_before_first_draw'] is True
                    and review['no_previous_full_geometry_audit_of_population'] is True,
                    'External predeclaration/fresh-query review is required')
            count = selection['full_geometry_rows']
            require(count <= plan['limits']['max_full_geometry_queries']
                    and 3*count <= plan['limits']['max_axis_queries'], 'Selection exceeds frozen query limits')
            require(context['law'].axes == [0, 1, 2] and context['law'].beta > 0 and context['law'].alpha < 1,
                    'Selected audit requires three active raw translation axes')
            require(not any(str(p) in bindings.files for p in paths.values()), 'Output aliases bound input')
            bindings.recheck(); budget.check()
            emit(dict(state='inputs_bound_before_geometry', selection=selection, input_sha256=bindings.files,
                      source_sha256=source_bindings, new_pose_draws=0, new_Poisson_clouds=0))
            phase = 'setup'
            endpoint_context = endpoints._load_context(context['root'], plan, bindings.bind, emit, budget)
            require(endpoint_context['target_and_regions_sha256'] == context['target_and_regions_sha256'],
                    'Endpoint setup target differs')
            setup_before = dict(budget.setup_calls)
            observer, observer_bridge = frozen_line_observer(endpoint_context['observer'],
                read(context['root']/'provenance/compiled-native.json'), endpoint_context['classifier_binding'], bindings.bind)
            require(dict(budget.setup_calls) == setup_before, 'Interface bridge repeated observer setup')
            emit(dict(state='frozen_observer_interface_bound', bridge=observer_bridge))
            recon = full.line.Reconstructor(context['region'], read(context['root']/'provenance/importance-guide.json'),
                context['config'], read(context['root']/'provenance/shape.json'), observer, use_tree=False)
            original_axis = recon.reconstruct_axis
            def axis(u, axis, use_tree=None):
                require(use_tree in (None, False), 'Selected reference must remain unpruned')
                return budget.query('axis', active, lambda: original_axis(u, axis, use_tree=False))
            recon.reconstruct_axis = axis
            phase = 'rows'
            for entry in selection['rows']:
                active = entry['draw']; budget.check(); emit(dict(state='begin', **entry))
                row = read_bound_record(entry['sample_record'], plan['limits']['max_record_bytes'])
                label = read_bound_record(entry['label_record'], plan['limits']['max_record_bytes'])
                require(row['draw'] == label['draw'] == active, 'Selected original ID changed')
                tick = time.process_time(); result = audit_selected_row(row, label, context, recon, budget)
                result['cpu_seconds'] = time.process_time()-tick
                completed.append(result); emit(dict(state='complete', result=result)); active = None
            phase = 'final_validation'; bindings.recheck(); budget.check()
            require(not (context['root']/'failure.json').exists(), 'Producer failure appeared during selected audit')
            require(budget.calls['full_geometry'] == count and budget.calls['axis'] == 3*count,
                    'Incomplete selected geometry allocation')
            receipt = dict(schema=SCHEMA, complete=True, passed=True, full_geometry_selected_rows=count,
                all_row_full_geometry_certified=False, original_attempt_denominator=context['manifest']['samples'],
                target_and_regions_sha256=context['target_and_regions_sha256'], population=plan['population'],
                selection=plan['selection'], preselection=plan['preselection'], review=plan['review'],
                algebra=plan['algebra'], labels=plan['labels'], rows=completed, input_sha256=bindings.files,
                source_sha256=source_bindings, runtime=runtime, query_counts=dict(budget.calls),
                observer_setup_counts=dict(budget.setup_calls), observer_interface_bridge=observer_bridge,
                execution_plan_sha256=plan_sha256,
                cpu_seconds=time.process_time()-budget.started, wall_seconds=time.monotonic()-budget.wall,
                new_pose_draws=0, new_Poisson_clouds=0, retries=0, replacements=0,
                physical_campaign_gate_open=False, full_vessel_gate_open=False, assembly_gate_open=False,
                scope='Unpruned full interval geometry on only the selected original rows; all-row labels and algebra '
                      'are separate bound evidence. Spatial Poisson/envelope correctness, rare geometry bugs, '
                      'unseen mass, convergence and assembly remain unresolved.')
            emit(dict(state='finished', full_geometry_selected_rows=count))
        except BaseException as error:
            failure = dict(schema=SCHEMA, complete=False, passed=False, phase=phase, draw=active,
                completed_rows=len(completed), error_type=type(error).__name__, error=str(error),
                query_counts={} if budget is None else dict(budget.calls),
                observer_setup_counts={} if budget is None else dict(budget.setup_calls),
                input_sha256=bindings.files, source_sha256=source_bindings, retries=0, replacements=0)
            emit(dict(state='failed', **failure)); streaming._write(paths['failure'], failure)
            raise
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0.)
            signal.setitimer(signal.ITIMER_PROF, 0.)
            for sig, handler in previous.items(): signal.signal(sig, handler)
    receipt['journal'] = dict(path=str(paths['journal']), sha256=sha(paths['journal']))
    streaming._write(paths['receipt'], receipt)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    args = parser.parse_args()
    result = run(args.plan, plan_sha256=args.plan_sha256)
    print(json.dumps(dict(complete=True, passed=True, full_geometry_selected_rows=result['full_geometry_selected_rows'])))
