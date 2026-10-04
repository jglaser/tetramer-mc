#!/usr/bin/env python3
"""Frozen proposal-only pilot checks, with no draw, retry or physical weights.

The genuine native-class-line-guide-audit-v1 records retain their own schema.
Algebra uses every component/axis/channel; its compact view is transient only.
Endpoint hard validity is atomic-only on EVERY attempt, including exterior
poses. Full independent geometry is restricted to the 16 predeclared IDs.
"""
from __future__ import annotations

import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import copy
import json
import math
from pathlib import Path
import signal
import time

import numpy as np

import native_class_line_algebra_reference as algebra
import native_class_line_physical_algebra_audit as streaming
import native_class_line_physical_reference as physical
import native_class_line_reference as line
import native_class_physical_labels as endpoints
import native_class_selected_geometry as selected
import native_class_physical_stage as stage
from analyze_mobile_native_pocket import local_sources

require, close, read, sha, write = stage.require, streaming.close, stage.read, stage.sha, stage.write
SCHEMA = 'native-class-support-pilot-v1'
PRODUCER_SCHEMA = 'native-class-line-guide-audit-v1'
PHASES = ('algebra', 'labels', 'geometry')


def load_protocol(path, digest, bindings):
    """Shared metadata authority for this worker and the receipt-only reducer."""
    path = bindings.bind(path, digest); protocol = read(path)
    require(protocol['schema'] == SCHEMA and protocol['producer_schema'] == PRODUCER_SCHEMA,
            'Wrong proposal-only pilot schema')
    root, code = Path(protocol['root']), Path(protocol['code_directory'])
    require(root.is_absolute() and root.resolve() == root and path == root/'protocol.json'
            and Path(__file__).resolve().parent == code.resolve(), 'Wrong frozen worker location')
    require(protocol['maximum_workers'] == protocol['threads'] == 1
            and protocol['total_attempts'] == 512 and protocol['selected_geometry_rows'] == 64
            and protocol['selected_geometry_axes'] == 192 and protocol['classifier_setups'] == 8
            and protocol['new_Poisson_clouds'] == protocol['physical_weight_estimates'] == 0,
            'Pilot allocation changed')
    require([p['id'] for p in protocol['populations']] == ['r00', 'r01', 'r02', 'r03'], 'Population inventory changed')
    for pop in protocol['populations']:
        require(pop['samples'] == 128 and Path(pop['directory']) == root/'queries'/pop['id'], 'Population changed')
        expected = selected.preselection(pop['id'], 128, pop['audit_seed'])
        require(pop['preselection'] == expected and pop['selected_ids'] == expected['draw_ids'],
                'Unconditional geometry selection changed')
    require(protocol['runtime'] == streaming.runtime_identity(), 'Frozen runtime differs')
    for name, value in protocol['files'].items(): bindings.bind(name, value)
    closure = local_sources(__file__)
    require(set(closure) <= set(protocol['source_sha256']), 'Missing worker source closure')
    for name, source in closure.items():
        require(source.parent == code.resolve(), 'Mixed source directories')
        bindings.bind(source, protocol['source_sha256'][name])
    execution_path = bindings.bind(root/'execution-plan.json'); execution = read(execution_path)
    require(execution['schema'] == 'native-class-physical-execution-v1' and execution['root'] == str(root)
            and execution['maximum_workers'] == execution['threads'] == 1,
            'Wrong controller allocation')
    require(execution['files'].get(str(path)) == digest, 'Protocol is not frozen by controller')
    for name, value in protocol['files'].items():
        require(execution['files'].get(name) == value, 'Controller omitted frozen input')
    claim = read(bindings.bind(root/'execution/claim.json'))
    require(claim['plan_sha256'] == sha(execution_path), 'Missing frozen controller claim')
    return protocol, execution


def population_context(protocol, execution, pop, bindings):
    root = Path(pop['directory'])
    _, summary = stage.predecessor(protocol['root'], execution, pop['id']+'-producer', root/'summary.json', bindings)
    manifest = read(bindings.bind(root/'manifest.json'))
    require(summary['manifest'] == manifest and summary['samples'] == manifest['samples'] == 128
            and manifest['seed'] == pop['seed'] and manifest['schema'] == PRODUCER_SCHEMA
            and manifest['physical_jobs'] == 0 and summary['probes'] == 0
            and manifest['probes_sha256'] is None and not (root/'failure.json').exists(),
            'Incomplete, changed or non-proposal population')
    data = {}
    for key, name, field in [('config', 'config', 'config_sha256'), ('region', 'region', 'region_sha256'),
            ('guide', 'importance-guide', 'guide_sha256'), ('shape', 'shape', 'shape_sha256'),
            ('compiled_native', 'compiled-native', 'compiled_native_sha256'),
            ('producer_bundle', 'source-bundle', 'source_bundle_sha256')]:
        require(manifest[field] == protocol[key]['sha256'], 'Producer changed '+key)
        bindings.reference(protocol[key])
        data[key] = read(bindings.bind(root/'provenance'/(name+'.json'), manifest[field]))
    require(manifest['executable_sha256'] == protocol['producer']['sha256']
            and manifest['native_definition_sha256'] == protocol['definition']['sha256'], 'Producer identity changed')
    bindings.reference(protocol['producer']); bindings.reference(protocol['definition'])
    for name in ('samples', 'attempts', 'probes'):
        bindings.bind(root/(name+'.jsonl'), summary[name+'_sha256'])
    require((root/'probes.jsonl').stat().st_size == 0, 'Unexpected saved probes')
    region, guide, config, compiled = (data[k] for k in ('region', 'guide', 'config', 'compiled_native'))
    require(guide['region_sha256'] == manifest['region_sha256'] and region['shape_sha256'] == manifest['shape_sha256']
            and guide['shape_sha256'] == manifest['shape_sha256']
            and guide['compiled_native'] == protocol['compiled_native']
            and compiled['source_definition_sha256'] == protocol['definition']['sha256'], 'Native/shape/chart binding changed')
    require(compiled['fixed_poses'] == config['fixed_poses'] == region['physical_fixed_neighbors'], 'Scaffold changed')
    for key in ('fixed_poses', 'capture_center', 'capture_radius', 'depletant_radius'):
        require(guide[key] == config[key], 'Guide target changed: '+key)
    for rk, ck in [('capture_center', 'capture_center'), ('capture_radius', 'capture_radius'),
                   ('activity', 'reservoir_density'), ('depletant_radius', 'depletant_radius'), ('physical_metric', 'metadata')]:
        require(region[rk] == config[ck], 'Physical target changed: '+rk)
    law = algebra.AlgebraLaw(region, guide, config)
    require(len(law.weights) == 116 and law.axes == [0, 1, 2] and len(law.channels) == 5
            and law.alpha == .5 and law.beta == 1. and law.floor == 1e-12, 'Fixed guide allocation changed')
    inventory = protocol['component_inventory']
    require([c['index'] for c in inventory] == list(range(116))
            and [c['bank'] for c in inventory] == ['original']*92+['added']*24, 'Component banks changed')
    close(manifest['latent_radius'], law.radius, 'Latent radius changed')
    close(manifest['log_latent_ball_volume'], law.logvolume, 'Latent volume changed')
    return dict(data, law=law, manifest=manifest, summary=summary, root=root, inventory=inventory)


def validate_row(row, index, law):
    require(row['kind'] == 'fresh' and type(row['id']) is int and row['id'] == index, 'Missing/reordered attempt')
    u = algebra.finite_array(row['latent'], (6,), 'latent')
    require(all(type(row[k]) is bool for k in ('hard_valid', 'capture_valid', 'shell_valid')), 'Invalid domain flags')
    raw, position, rotation, jac = law.decode(u)
    quaternion = algebra.finite_array(row['pose']['orientation'], (4,), 'orientation')
    require(abs(float(quaternion @ quaternion)-1.) < 1e-10, 'Unnormalized pose quaternion')
    for actual, expected, name in [(row['raw_coordinates'], raw, 'raw coordinates'),
            (row['pose']['position'], position, 'position'), (line.pose_arrays(row['pose'])[1], rotation, 'rotation'),
            (row['log_physical_jacobian'], jac, 'Jacobian'), (row['latent_radius'], np.linalg.norm(u), 'radius')]:
        close(actual, expected, 'Saved '+name+' differs')
    require(row['shell_valid'] == bool(np.linalg.norm(u) <= law.radius), 'Saved shell predicate differs')
    require(type(row['draw']) is dict and type(row['draw']['conditional']) is bool, 'Missing draw path')
    component = row['draw']['component']
    require(component is None or (type(component) is int and 0 <= component < len(law.weights)), 'Invalid sampled component')
    require(row['draw']['conditional'] == (component is not None) if law.beta == 1.
            else not row['draw']['conditional'] or component is not None, 'Conditional/component branch differs')
    require(component is not None or row['shell_valid'], 'Uniform draw outside its latent ball')
    # Class mode retains one zero-width sentinel and zero contact indices.
    # The producer's outer width map therefore emits one empty inner list.
    require(math.isfinite(row['log_proposal_density']) and math.isfinite(row['baseline_log_density'])
            and row['width_contacts'] == [[]], 'Wrong proposal density/contact format')
    error = float(np.max(abs(law.inverse_pose(position, rotation)-u)))
    require(math.isfinite(row['backmap_error']) and row['backmap_error'] >= 0
            and max(error, row['backmap_error']) < 2e-7*(1+max(float(np.linalg.norm(u)), law.radius)), 'Chart inverse failed')
    for key in ('draw_cpu_seconds', 'density_cpu_seconds', 'observer_cpu_seconds'):
        require(type(row[key]) in (int, float) and math.isfinite(row[key]) and row[key] >= 0, 'Invalid producer CPU')
    return u


def compact_view(details):
    """Drop redundant scalar branches in memory only; validate them separately."""
    require('trace_format' not in details, 'Expected genuine FULL proposal trace')
    result = copy.deepcopy(details)
    for axis in result['axes']: axis.pop('components')
    result['trace_format'] = 'class-line-compact-v1'
    return result


def check_full_branches(details, evaluation, law):
    """Check every emitted scalar term against the complete vectorized sum."""
    for actual, expected in zip(details['axes'], evaluation['axes']):
        require(len(actual['components']) == len(law.weights), 'Missing full component records')
        for k, component in enumerate(actual['components']):
            require(component['component'] == k and len(component['channels']) == len(law.channels), 'Component/channel inventory changed')
            for name, values in [('conditional_mean', expected['conditional_means']), ('conditional_sigma', expected['conditional_sigmas'])]:
                close(component[name], values[k], 'Full '+name+' differs')
            for j, (saved, branch) in enumerate(zip(component['channels'], expected['channels'])):
                require(saved['channel'] == j and saved['fallback_target'] == algebra.FALLBACKS[branch['fallback_indices'][k]]
                        and saved['query_coordinate_allowed'] == bool(branch['query_coordinate_allowed'][k]), 'Full fallback/membership changed')
                for name, value in [('class_mass', branch['class_masses'][k]), ('hard_free_mass', expected['hard_free_masses'][k]),
                        ('effective_mass', branch['effective_masses'][k]), ('multiplier', branch['multipliers'][k])]:
                    close(saved[name], value, 'Full '+name+' differs', atol=1e-12 if name == 'multiplier' else 2e-15, rtol=3e-7)
            close(component['channel_mixture_multiplier'], expected['channel_mixture_multipliers'][k],
                  'Full channel sum differs', atol=1e-12, rtol=3e-7)


def algebra_row(row, index, context):
    law = context['law']; u = validate_row(row, index, law)
    evaluation = law.evaluate_saved(u, compact_view(row['density_details']))
    check_full_branches(row['density_details'], evaluation, law)
    line.log_close(row['log_proposal_density'], evaluation['log_density'], 'Complete proposal q differs')
    line.log_close(row['baseline_log_density'], evaluation['baseline_log_density'], 'Baseline q differs')
    checked = law.verify_draw(row['draw'], u, evaluation)
    chosen = None
    if row['draw']['conditional']:
        draw = row['draw']; k, j = draw['component'], draw['channel']
        axis = next(a for a in evaluation['axes'] if a['axis'] == draw['axis']); branch = axis['channels'][j]
        info = context['inventory'][k]
        chosen = dict(component=k, bank=info['bank'], training_id=info['training_id'], latent_sigma=info['latent_sigma'],
            center=context['guide']['gaussian_components'][k]['mean'], channel=j, axis=draw['axis'],
            class_mass=float(branch['class_masses'][k]), hard_mass=float(axis['hard_free_masses'][k]),
            effective_mass=float(branch['effective_masses'][k]), fallback=algebra.FALLBACKS[branch['fallback_indices'][k]],
            empty_class=not bool(branch['intervals']),
            below_floor_nonempty_class=bool(branch['intervals']) and float(branch['class_masses'][k]) <= law.floor)
    return dict(id=index, selected=chosen, log_proposal_density=evaluation['log_density'],
        component_branches=evaluation['component_branches'], fallback_component_branches=evaluation['fallback_component_branches'],
        producer_cpu_seconds={key: row[key+'_cpu_seconds'] for key in ('draw', 'density', 'observer')},
        maximum_errors=dict(interval_endpoints=max(evaluation['interval_error'], checked['endpoint_error']),
                            inverse_cdf=checked['inverse_error'], log_density=abs(row['log_proposal_density']-evaluation['log_density'])))


def observer_context(context, protocol, bindings, budget, phase):
    definition = bindings.reference(protocol['definition'])
    inventory = endpoints.observer_setup_inventory(definition)
    require(inventory == protocol['observer_setup'], 'Frozen observer setup changed')
    witness = physical.validate_shape_witness(context['compiled_native'], context['shape'],
        context['manifest']['native_shape_compatibility'], protocol['compiled_native']['sha256'], protocol['shape']['sha256'])
    wall = endpoints.wall_certificate(context['shape'], context['config'])
    observer, binding = endpoints.load_bounded_classifier(definition, inventory, budget, bindings.bind)
    for name, digest in binding['input_sha256'].items(): bindings.bind(observer.root/name, digest)
    endpoints.validate_classifier_target(context['config'], protocol['shape']['sha256'], observer.definition, definition)
    line.compare_compiled_definition(context['compiled_native'], observer)
    context.update(observer=observer, classifier_binding=binding, shape_witness=witness, wall_certificate=wall)
    if phase == 'labels':
        context['core'] = endpoints.EndpointCore(context['shape'], context['config']['fixed_poses'])
        context['contact'] = endpoints.ExclusionContact(context['shape'], context['config']['fixed_poses'], context['config']['depletant_radius'])
    else:
        bridge, bridge_binding = selected.frozen_line_observer(observer, context['compiled_native'], binding, bindings.bind)
        context['recon'] = line.Reconstructor(context['region'], context['guide'], context['config'], context['shape'], bridge, use_tree=False)
        context['bridge_binding'] = bridge_binding
    budget.check()


def compare_native(saved, classification, anchor_count):
    """Compare complete Rust per-anchor motif/ANY-bond records, not just a bool."""
    require(saved['native_any'] == classification['native_any']
            and saved['matched_anchor_indices'] == classification['matched_anchor_indices']
            and len(saved['per_anchor']) == anchor_count, 'Complete native anchor labels differ')
    for i, record in enumerate(saved['per_anchor']):
        matches = [m for m in classification['matches'] if m['anchor_index'] == i]
        require(record['anchor_index'] == i and record['matched_motif_ids'] == [m['motif_id'] for m in matches]
                and len(record['matches']) == len(matches), 'Native motif union differs')
        for actual, expected in zip(record['matches'], matches):
            require(actual['motif_id'] == expected['motif_id'], 'Native motif order differs')
            for key in ('maximum_member_position_error_A', 'proper_orientation_error_deg'):
                close(actual[key], expected[key], 'Native '+key+' differs', atol=3e-6)
            require(len(actual['supporting_member_bonds']) == len(expected['supporting_member_bonds']), 'Native ANY-bond support differs')
            for a, b in zip(actual['supporting_member_bonds'], expected['supporting_member_bonds']):
                for key in ('members', 'class_label', 'class_family', 'shared_reference_residue_pairs_entry'):
                    require(a[key] == b[key], 'Native supporting bond differs')
                for key in ('position_error_A', 'orientation_error_deg', 'minimum_gap_A'):
                    if b[key] is None: require(a[key] is None, 'Native null gap differs')
                    else: close(a[key], b[key], 'Native bond '+key+' differs', atol=3e-6)


def label_row(row, index, context, budget):
    u = validate_row(row, index, context['law']); position, rotation = line.pose_arrays(row['pose'])
    config = context['config']
    capture = budget.query('capture', index, lambda: bool(np.linalg.norm(position-config['capture_center']) <= config['capture_radius']))
    hard = budget.query('atomic', index, lambda: context['core'].hard_valid(position, rotation))
    shell = bool(np.linalg.norm(u) <= context['law'].radius)
    require((hard, capture, shell) == (row['hard_valid'], row['capture_valid'], row['shell_valid']), 'Independent endpoint validity differs')
    valid = hard and capture and shell
    record = dict(id=index, hard_valid=hard, capture_valid=capture, shell_valid=shell, valid=valid,
        native=None, contact=None, classification=None, exclusion_contact=None,
        orthant=sum(1 << i for i, v in enumerate(u) if v >= 0), selected_channel_match=None)
    if valid:
        classification = budget.query('native', index, lambda: context['observer'].classify(row['pose']))
        contact = budget.query('contact', index, lambda: context['contact'].classify(row['pose']))
        compare_native(row['native_decision'], classification, len(config['fixed_poses']))
        require(row['exclusion_contact_by_anchor'] == [a['exclusion_contact'] for a in contact['anchors']], 'Independent exclusion contacts differ')
        record.update(native=classification['native_any'], contact=contact['exclusion_contact'],
                      classification=classification, exclusion_contact=contact)
        if row['draw']['conditional']:
            channel = context['law'].channels[row['draw']['channel']]
            match = dict(hard_free=True, native=record['native'], contact_without_native=record['contact'] and not record['native'])[channel['class']]
            record['selected_channel_match'] = match and (channel.get('orthant') is None or channel['orthant'] == record['orthant'])
    else:
        require(row['native_decision'] is None and row['exclusion_contact_by_anchor'] is None, 'Invalid/exterior attempt carries native labels')
    return record


def geometry_row(row, label, context, budget):
    index = row['id']; recon = context['recon']; u = validate_row(row, index, context['law'])
    original = recon.reconstruct_axis
    recon.reconstruct_axis = lambda latent, axis: budget.query('axis', index, lambda: original(latent, axis, use_tree=False))
    try: result = budget.query('full_geometry', index, lambda: recon.density(u))
    finally: recon.reconstruct_axis = original
    require([a['axis'] for a in result['axes']] == [0, 1, 2], 'Missing full reference axes')
    line.log_close(row['log_proposal_density'], result['log_density'], 'Independent full q differs')
    line.log_close(row['baseline_log_density'], result['baseline_log_density'], 'Independent baseline q differs')
    error = line.compare_density_trace(row['density_details'], result, recon, u)
    generated = line.audit_draw(row['draw'], recon, u, result)
    for axis in result['axes']:
        coordinate = recon.raw(u)[axis['axis']]
        require(line.contains(axis['hard_free_intervals'], coordinate) == label['valid'], 'Independent H/endpoint membership differs')
        if label['valid']:
            require(line.contains(axis['native_intervals'], coordinate) == label['native'], 'Full native/endpoint membership differs')
            require(line.contains(axis['exclusion_contact_intervals'], coordinate) == label['contact'], 'Full contact/endpoint membership differs')
    return dict(id=index, axes=3, unpruned=True, full_density_certified=True, draw_path_certified=True,
        endpoint_membership_certified=True, log_proposal_density=result['log_density'],
        maximum_errors=dict(interval_endpoints=max(error, generated['endpoint_error']), inverse_cdf=generated['inverse_error']))


def run(protocol_path, digest, population, phase):
    require(phase in PHASES, 'Unknown audit phase')
    protocol_path = Path(protocol_path).resolve(); root = protocol_path.parent
    require(population in ('r00', 'r01', 'r02', 'r03'), 'Unknown population')
    folder = root/'analysis'/population
    paths = {key: folder/(phase+suffix) for key, suffix in [('receipt', '.json'), ('journal', '.journal.jsonl'), ('failure', '.failure.json')]}
    require(folder.is_dir() and all(not p.exists() for p in paths.values()), 'Fresh stage outputs required; no resume')
    bindings = stage.Inputs(); budget = None; readers = []; completed = []; active = None; context = None
    started, wall = time.process_time(), time.monotonic()
    ledger = paths['journal'].open('x')
    def emit(value):
        ledger.write(json.dumps(value, allow_nan=False)+'\n'); ledger.flush(); os.fsync(ledger.fileno())
    handlers = {}
    def interrupted(signum, _frame): raise InterruptedError('Pilot audit signal '+str(signum))
    try:
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGXCPU): handlers[sig] = signal.signal(sig, interrupted)
        emit(dict(state='started', phase=phase, population=population, protocol_sha256=digest))
        protocol, execution = load_protocol(protocol_path, digest, bindings)
        stage.verify_live_authority(protocol, execution, population, phase)
        pop = next(p for p in protocol['populations'] if p['id'] == population)
        limits = protocol['phase_limits'][phase]
        budget = endpoints.Budget(dict(cpu_seconds=limits['cpu_limit_seconds'], wall_seconds=limits['wall_limit_seconds'],
            max_rows=128, max_record_bytes=limits['max_record_bytes'], **{f'max_{role}_queries': 128 for role in endpoints.QUERY_ROLES},
            max_full_geometry_queries=16, max_axis_queries=48), emit)
        context = population_context(protocol, execution, pop, bindings)
        prior = {}
        if phase in ('labels', 'geometry'):
            _, prior['algebra'] = stage.predecessor(root, execution, population+'-algebra', folder/'algebra.json', bindings)
            require(prior['algebra']['protocol_sha256'] == digest and prior['algebra']['population'] == population,
                    'Wrong algebra predecessor')
        if phase == 'geometry':
            _, prior['labels'] = stage.predecessor(root, execution, population+'-labels', folder/'labels.json', bindings)
            require(prior['labels']['protocol_sha256'] == digest and prior['labels']['population'] == population,
                    'Wrong endpoint predecessor')
        for receipt in prior.values():
            require([r['id'] for r in receipt['rows']] == list(range(128)), 'Incomplete all-attempt predecessor')
        if phase != 'algebra': observer_context(context, protocol, bindings, budget, phase)
        samples = endpoints.BoundedLines(context['root']/'samples.jsonl', limits['max_record_bytes'])
        attempts = endpoints.BoundedLines(context['root']/'attempts.jsonl', limits['max_record_bytes']); readers.extend([samples, attempts])
        selected_ids = set(pop['selected_ids'])
        total_cpu = {key: 0. for key in ('draw', 'density', 'observer')}; conditioned = fallback = valid_count = 0
        for index in range(128):
            budget.check(); active = index
            emit(dict(state='row_begin', id=index, selected=phase != 'geometry' or index in selected_ids))
            row, attempt = samples.next(), attempts.next()
            require(row is not None and attempt == dict(ordinal=index, kind='fresh', id=index, state='begin'), 'Attempt journal changed')
            require(row['kind'] == 'fresh' and type(row['id']) is int and row['id'] == index, 'Sample identity changed')
            sample_hash = samples.last['sha256']
            for receipt in prior.values(): require(receipt['rows'][index]['sample_record_sha256'] == sample_hash, 'Predecessor sample line changed')
            if phase == 'algebra':
                result = algebra_row(row, index, context)
                for key in total_cpu: total_cpu[key] += result['producer_cpu_seconds'][key]
                conditioned += int(row['draw']['conditional']); fallback += int(row['draw'].get('fallback', False))
                valid_count += int(row['hard_valid'] and row['capture_valid'] and row['shell_valid'])
            elif phase == 'labels': result = label_row(row, index, context, budget)
            elif index in selected_ids: result = geometry_row(row, prior['labels']['rows'][index], context, budget)
            else: result = None
            if result is not None:
                result['sample_record_sha256'] = sample_hash; completed.append(result)
            emit(dict(state='row_complete', id=index, audited=result is not None, sample_record= samples.last))
            active = None
        require(samples.next() is None and attempts.next() is None, 'Extra original attempt/sample')
        for reader, key in [(samples, 'samples'), (attempts, 'attempts')]:
            require(reader.hash.hexdigest() == context['summary'][key+'_sha256'], 'Consumed stream hash differs')
        if phase == 'algebra':
            require((conditioned, fallback, valid_count) == tuple(context['summary'][k] for k in
                    ('conditioned_draws', 'fallback_draws', 'hard_capture_shell_valid')), 'Producer summary counts differ')
            for key in ('draw', 'density'): close(total_cpu[key], context['summary'][key+'_cpu_seconds'], 'Producer CPU sum differs', atol=1e-9)
        require([r['id'] for r in completed] == (pop['selected_ids'] if phase == 'geometry' else list(range(128))), 'Audited allocation changed')
        if phase == 'geometry': require(budget.calls['full_geometry'] == 16 and budget.calls['axis'] == 48, 'Full reference query count differs')
        if phase == 'labels': require(budget.calls['capture'] == budget.calls['atomic'] == 128, 'Missing hard-zero endpoint checks')
        budget.check(); bindings.recheck()
        require(protocol['runtime'] == streaming.runtime_identity(), 'Runtime changed during audit')
        emit(dict(state='finished', completed=len(completed), counts=dict(budget.calls), setup_counts=dict(budget.setup_calls)))
        ledger.close()
        receipt = dict(schema='native-class-support-pilot-'+phase+'-v1', complete=True, passed=True, phase=phase,
            population=population, protocol_sha256=digest, rows=completed, attempted_records=128,
            counts=dict(budget.calls), setup_counts=dict(budget.setup_calls), input_sha256=bindings.files,
            source_sha256=protocol['source_sha256'], runtime=protocol['runtime'],
            journal=dict(path=str(paths['journal']), sha256=sha(paths['journal'])),
            analysis_cpu_seconds=time.process_time()-started, wall_seconds=time.monotonic()-wall,
            new_pose_draws=0, new_Poisson_clouds=0, physical_weight_estimates=0,
            geometry_certified_rows=16 if phase == 'geometry' else 0,
            all_row_geometry_certified=False)
        if phase != 'algebra':
            receipt.update(classifier_binding=context['classifier_binding'], shape_witness=context['shape_witness'],
                           wall_certificate=context['wall_certificate'])
        if phase == 'geometry': receipt['frozen_line_observer_bridge'] = context['bridge_binding']
        write(paths['receipt'], receipt)
        return receipt
    except BaseException as error:
        failure = dict(schema='native-class-support-pilot-failure-v1', complete=False, passed=False, phase=phase,
            population=population, protocol_sha256=digest, active_id=active, completed_ids=[r['id'] for r in completed],
            error_type=type(error).__name__, error=str(error), input_sha256=bindings.files,
            counts={} if budget is None else dict(budget.calls), setup_counts={} if budget is None else dict(budget.setup_calls),
            consumed_records=[r.last for r in readers], retries=0, replacements=0)
        if not ledger.closed: emit(dict(state='failed', **failure)); ledger.close()
        write(paths['failure'], failure)
        raise
    finally:
        if not ledger.closed: ledger.close()
        for reader in readers: reader.stream.close()
        for sig, handler in handlers.items(): signal.signal(sig, handler)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--protocol-sha256', required=True)
    parser.add_argument('--population', required=True)
    parser.add_argument('--phase', choices=PHASES, required=True)
    args = parser.parse_args()
    result = run(args.protocol, args.protocol_sha256, args.population, args.phase)
    print(json.dumps(dict(complete=result['complete'], passed=result['passed'], phase=args.phase, population=args.population)))
