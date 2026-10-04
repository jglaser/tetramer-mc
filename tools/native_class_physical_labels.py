#!/usr/bin/env python3
"""Bounded streaming physical endpoint labels for completed v6/v7 populations.

This is an execution tool, not a preparation or launch controller. It requires
an externally frozen, hash-bound plan before opening the physical geometry.
Every captured attempt gets a strict atomic hard check, including recorded
invalid/exterior zeros. Only contributing endpoints get the complete frozen
native observer and contact classifier. No line intervals, samples or clouds
are generated. This receipt does not certify proposal line-set geometry.
"""
from __future__ import annotations

import os
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '1'
import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import signal
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
from scipy.spatial import cKDTree

import native_class_line_physical_algebra_audit as infrastructure
from native_class_line_reference import compare_compiled_definition, pose_arrays
from hard_free_line_reference import Reconstructor as EndpointReference
from analyze_mobile_native_pocket import local_sources
from analyze_mobile_threshold_reference import ExclusionContact, validate_classifier_target
from diagnose_conditional_ray_extremes import chart_coordinates
from analyze_contact_bank_reference_partition import validate_regions, r5_contains
from analyze_contact_bank_pilot import stratify_with_exterior
from analyze_native_class_physical_populations import target_identity, STRATA, fingerprint
from prepare_native_class_projected_probe import setup_inventory, check_setup_counts

require, close = infrastructure.require, infrastructure.close
sha, read = infrastructure.sha, infrastructure.read
SCHEMA = 'native-class-independent-physical-labels-v1'
PLAN_SCHEMA = 'native-class-physical-label-plan-v1'
SHAPE_SHA = 'c7442034e7ffa4627b67c57a81117f0e9bc2d389ecf956a83dac2a5e915554c9'
REGION_SHA = '924648f4d9db473045397c300b3b4af7ccde8cfda1b1703a6899239ec12e2f02'
QUERY_ROLES = ('capture', 'atomic', 'native', 'contact')


def validate_row(row, **kwargs):
    """Use the genuine producer schema; v6 rows are never renamed to v7."""
    return infrastructure.validate_row(row, **kwargs)


def native_identity(root, plan, manifest, config, shape, bind):
    """Bind a native observer separately from a native-blind v6 producer."""
    schema = manifest['schema']
    if schema == infrastructure.full.SCHEMA:
        provenance = manifest['compiled_native']
        compiled_path = bind(root/'provenance/compiled-native.json', provenance['compiled_sha256'])
        report = provenance['shape_compatibility']
        report_binding = None
        origin = 'producer_compiled_native'
        require(plan['definition']['sha256'] == provenance['source_definition_sha256'],
                'Frozen native definition differs from producer')
    elif schema == 'importance-latent-region-normalizer-v6':
        require('compiled_native' not in manifest,
                'Native-blind v6 producer must not carry compiled native identity')
        require(type(plan.get('compiled_native')) is dict and type(plan.get('shape_compatibility')) is dict,
                'v6 labels require externally bound compiled native and shape compatibility')
        compiled_path = bind(plan['compiled_native']['path'], plan['compiled_native']['sha256'])
        report_path = bind(plan['shape_compatibility']['path'], plan['shape_compatibility']['sha256'])
        report = read(report_path)
        report_binding = dict(path=str(report_path), sha256=sha(report_path))
        origin = 'external_analysis_plan'
    else:
        raise ValueError('Unsupported physical-label source schema')
    compiled = read(compiled_path)
    compiled_binding = dict(path=str(compiled_path), sha256=sha(compiled_path))
    require(compiled['source_definition_sha256'] == plan['definition']['sha256'],
            'Compiled observer source definition differs from frozen analysis plan')
    require(compiled['fixed_poses'] == config['fixed_poses'] == manifest['physical_fixed_neighbors'],
            'Compiled observer scaffold differs from physical source')
    witness = infrastructure.full.validate_shape_witness(compiled, shape, report,
        compiled_binding['sha256'], manifest['shape_sha256'])
    return dict(source_schema=schema, native_identity_origin=origin, compiled_native=compiled,
                compiled_native_binding=compiled_binding, shape_compatibility_binding=report_binding,
                shape_witness=witness)


class EndpointCore:
    """Only fixed endpoint trees; no guide, native-line or interval initializer."""
    def __init__(self, shape, fixed):
        self.atoms = np.asarray([a['center'] for a in shape['atoms']], float)
        self.radii = np.asarray([a['radius'] for a in shape['atoms']], float)
        require(len(self.radii) > 0 and self.atoms.shape == (len(self.radii), 3)
                and np.isfinite(self.atoms).all() and np.isfinite(self.radii).all()
                and (self.radii > 0).all(), 'Invalid physical atom union')
        self.fixed_world = []
        for pose in fixed:
            position, rotation = pose_arrays(pose)
            self.fixed_world.append(self.atoms @ rotation.T + position)
        require(bool(self.fixed_world), 'Missing physical scaffold')
        self.fixed_trees = [cKDTree(points) for points in self.fixed_world]
        self.maximum_radius = float(self.radii.max())

    # Direct Euclidean distance < radius sum; the 1e-9 cushion only widens the
    # candidate search. The observer's -1e-8 gap tolerance never sets hard truth.
    def hard_valid(self, position, rotation):
        mobile = self.atoms @ rotation.T + position
        for fixed, tree in zip(self.fixed_world, self.fixed_trees):
            # Bounded batches avoid one Python/KD-tree call per atom without
            # constructing an all-atom pair list for deeply invalid endpoints.
            for start in range(0, len(mobile), 64):
                stop = min(start+64, len(mobile))
                candidates = tree.query_ball_point(mobile[start:stop],
                    self.radii[start:stop]+self.maximum_radius+1e-9)
                for i, indices in enumerate(candidates, start):
                    if indices and np.any(np.linalg.norm(fixed[indices]-mobile[i], axis=1)
                                          < self.radii[i]+self.radii[indices]):
                        return False
        return True


def wall_certificate(shape, config):
    bound = max(math.hypot(*atom['center']) + atom['radius'] for atom in shape['atoms'])
    radius = config['metadata']['physical_sphere_radius_A']
    upper = math.hypot(*config['capture_center']) + config['capture_radius'] + bound
    require(math.isfinite(bound) and math.isfinite(radius) and upper < radius,
            'Center capture does not certify the original atomic wall')
    require(all(math.hypot(*p['position']) + bound < radius for p in config['fixed_poses']),
            'Fixed scaffold wall certificate fails')
    return dict(shape_bound_A=bound, physical_sphere_radius_A=radius,
                all_capture_poses_atom_radius_upper_bound_A=upper,
                guaranteed_atomic_wall_clearance_A=radius-upper,
                predicate='center capture only; original atomic wall is redundant by the bound',
                bath_scope='unbounded and wall permeable')


class Budget:
    def __init__(self, limits, emit):
        self.limits, self.emit = limits, emit
        self.started, self.wall = time.process_time(), time.monotonic()
        self.calls = Counter()
        self.setup_calls = Counter()
        for key in ('max_rows', 'max_record_bytes', *(f'max_{r}_queries' for r in QUERY_ROLES)):
            require(type(limits[key]) is int and limits[key] > 0, 'Invalid frozen budget '+key)
        for key in ('cpu_seconds', 'wall_seconds'):
            require(type(limits[key]) in (int, float) and math.isfinite(limits[key])
                    and limits[key] > 0, 'Invalid frozen budget '+key)

    def check(self):
        require(time.process_time()-self.started <= self.limits['cpu_seconds'], 'CPU budget exhausted')
        require(time.monotonic()-self.wall <= self.limits['wall_seconds'], 'Wall budget exhausted')

    def query(self, role, draw, function):
        self.check()
        require(self.calls[role] < self.limits[f'max_{role}_queries'], 'Query budget exhausted: '+role)
        self.calls[role] += 1
        self.emit(dict(state='query_begin', role=role, draw=draw, query=self.calls[role]))
        value = function()
        self.check()
        self.emit(dict(state='query_complete', role=role, draw=draw, query=self.calls[role]))
        return value

    def setup_query(self, kind, limit, function, **extra):
        self.check()
        require(self.setup_calls[kind+'_started'] < limit, 'Observer setup query cap exceeded: '+kind)
        index = self.setup_calls[kind+'_started']; self.setup_calls[kind+'_started'] += 1
        self.emit(dict(state='setup_query_begin', kind=kind, index=index, **extra))
        value = function()
        self.check(); self.setup_calls[kind+'_completed'] += 1
        self.emit(dict(state='setup_query_complete', kind=kind, index=index,
                       counts=dict(self.setup_calls), **extra))
        return value


def observer_setup_inventory(definition):
    """Reuse the metadata-only reference/triple cap; this pass builds no guides."""
    value = setup_inventory(definition)
    value.update(guide_instances=0, setup_scope='One frozen observer initialization: all directed reference '
                 'patch queries and one fixed-scaffold classifier, whose cached member triples bound contacts.')
    return value


@contextmanager
def bounded_observer_setup(cls, criteria, inventory, budget):
    """Instrument the dynamically loaded frozen class before its constructor."""
    old_contacts, old_classifier = cls._contacts, cls.classify_pair
    in_scaffold = False
    def contacts(model, position, rotation, cutoff):
        if in_scaffold:
            require(cutoff == criteria['contact_entry_A'], 'Changed scaffold contact cutoff')
            return budget.setup_query('scaffold_contacts', inventory['maximum_scaffold_contact_calls'],
                lambda: old_contacts(model, position, rotation, cutoff))
        index = budget.setup_calls['reference_contacts_started']
        require(index < inventory['reference_contact_calls'], 'Extra reference setup query')
        expected = inventory['reference_contact_queries'][index]
        require(np.array_equal(position, expected['position']) and np.array_equal(rotation, expected['rotation'])
                and cutoff == criteria['native_reference_patch_gap_A'], 'Changed frozen reference setup query')
        return budget.setup_query('reference_contacts', inventory['reference_contact_calls'],
            lambda: old_contacts(model, position, rotation, cutoff), label=expected['label'])
    def classify(model, anchor, moving):
        nonlocal in_scaffold
        require(not in_scaffold and [anchor, moving] == inventory['fixed_scaffold_poses']
                and budget.setup_calls['reference_contacts_completed'] == inventory['reference_contact_calls'],
                'Changed fixed-scaffold setup query')
        in_scaffold = True
        try:
            return budget.setup_query('scaffold_classifier', inventory['fixed_scaffold_classifier_calls'],
                lambda: old_classifier(model, anchor, moving))
        finally:
            in_scaffold = False
    cls._contacts, cls.classify_pair = contacts, classify
    try:
        yield
        check_setup_counts(inventory, budget.setup_calls)
    finally:
        cls._contacts, cls.classify_pair = old_contacts, old_classifier


def load_bounded_classifier(definition, inventory, budget, bind):
    definition = Path(definition).resolve(); data = read(definition)
    runtime = bind(definition.parent/'inputs/source/native_contact_regions.py',
                   data['input_sha256']['source/native_contact_regions.py'])
    spec = importlib.util.spec_from_file_location('bounded_frozen_native_regions', runtime)
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    with bounded_observer_setup(module.NativeContactRegions, module.CRITERIA, inventory, budget):
        observer = module.NativeContactRegions(definition)
    return observer, dict(definition=str(definition), definition_sha256=sha(definition),
        runtime_sha256=sha(runtime), input_sha256=data['input_sha256'], criteria=data['criteria'], scope=data['scope'])


class BoundedLines(infrastructure.JsonLines):
    def __init__(self, path, maximum):
        super().__init__(path)
        self.maximum = maximum

    def next(self):
        offset = self.bytes
        raw = self.stream.readline(self.maximum+1)
        if not raw:
            self.last = dict(path=str(self.path), offset=offset, eof=True)
            return None
        self.hash.update(raw); self.bytes += len(raw); self.lines += 1
        self.last = dict(path=str(self.path), offset=offset, bytes=len(raw), line=self.lines,
                         sha256=hashlib.sha256(raw).hexdigest())
        require(len(raw) <= self.maximum, 'Record byte budget exceeded')
        require(raw.endswith(b'\n') and raw.strip(), 'Blank or unterminated JSONL record')
        value = infrastructure.loads(raw)
        require(type(value) is dict, 'JSONL record is not an object')
        return value


def _load_context(root, plan, bind, emit, budget):
    # _provenance performs schema-specific geometry-free chart checks. v7 also
    # has a producer-native atom bijection; v6 has no producer native identity.
    # Neither constructs the line Reconstructor nor queries any endpoints.
    manifest, summary, region, config, law, witness = infrastructure._provenance(root, bind)
    require(manifest['shape_sha256'] == SHAPE_SHA and manifest['region_sha256'] == REGION_SHA,
            'Only the unchanged repaired-shape protein R4 target is authorized')
    shape = read(root/'provenance/shape.json')
    reference_path = bind(plan['reference_region']['path'], plan['reference_region']['sha256'])
    strata_path = bind(plan['strata']['path'], plan['strata']['sha256'])
    reference, strata = read(reference_path), read(strata_path)
    validate_regions(region, reference)
    require(strata == STRATA, 'Original fixed strata changed')
    certificate = wall_certificate(shape, config)
    definition_path = bind(plan['definition']['path'], plan['definition']['sha256'])
    identity = native_identity(root, plan, manifest, config, shape, bind)
    _, descriptor, target_id = target_identity(dict(
        region=dict(path=str(root/'provenance/region.json'), sha256=manifest['region_sha256']),
        old_r5_region=plan['reference_region'], native_definition=plan['definition']),
        SimpleNamespace(reference=lambda ref: bind(ref['path'], ref['sha256'])))
    inventory = observer_setup_inventory(definition_path)
    require(plan['observer_setup'] == inventory, 'Frozen observer setup inventory differs')
    budget.check(); emit(dict(state='native_setup_begin', inventory=inventory))
    observer, binding = load_bounded_classifier(definition_path, inventory, budget, bind)
    budget.check(); emit(dict(state='native_setup_complete', counts=dict(budget.setup_calls)))
    validate_classifier_target(config, manifest['shape_sha256'], observer.definition, definition_path)
    for name, digest in binding['input_sha256'].items(): bind(observer.root/name, digest)
    compare_compiled_definition(identity['compiled_native'], observer)
    require(binding['definition_sha256'] == plan['definition']['sha256'], 'Observer binding changed')
    budget.check(); emit(dict(state='endpoint_setup_begin'))
    core = EndpointCore(shape, config['fixed_poses'])
    contact = ExclusionContact(shape, config['fixed_poses'], config['depletant_radius'])
    budget.check(); emit(dict(state='endpoint_setup_complete'))
    return dict(manifest=manifest, summary=summary, region=region, config=config, law=law,
                reference=reference, strata=strata, observer=observer, contact=contact, core=core,
                classifier_binding=binding, physical_input_witness=witness, wall_certificate=certificate,
                target_and_regions_sha256=target_id, target_descriptor=descriptor,
                observer_setup_inventory=inventory, **identity)


def classify_row(row, index, context, budget, sample_digest):
    """One complete original index. Saved guide interval fields are never read."""
    manifest, region, config = (context[k] for k in ('manifest', 'region', 'config'))
    accounting = validate_row(row, expected_draw=index, manifest=manifest, region=region)
    position, rotation = pose_arrays(row['pose'])
    _, decoded_position, decoded_rotation, logj = context['law'].decode(row['latent'])
    close(position, decoded_position, 'Endpoint chart position differs')
    close(rotation, decoded_rotation, 'Endpoint chart orientation differs')
    close(row['log_physical_jacobian'], logj, 'Endpoint chart Jacobian differs')
    original_q = infrastructure.physical.native.native_q(region['physical_metric'], row['pose'])
    close(row['q'], original_q, 'Original physical q differs')
    q_region = dict(region)
    if q_region.get('maximum_original_q') is None: q_region.pop('maximum_original_q', None)
    require(row['region_valid'] == infrastructure.physical.q_contains(original_q, q_region),
            'Independent original-q predicate differs')
    capture = budget.query('capture', index, lambda: bool(
        np.linalg.norm(position-config['capture_center']) <= config['capture_radius']))
    require(capture == row['capture_valid'], 'Independent capture predicate differs')
    atomic = budget.query('atomic', index, lambda: context['core'].hard_valid(position, rotation)) if capture else None
    hard = bool(capture and atomic)
    require(hard == row['hard_valid'], 'Independent atomic hard predicate differs')
    applicable = hard and row['region_valid'] and accounting['support']
    require(applicable == math.isfinite(accounting['z']), 'Contributing endpoint differs')
    record = dict(draw=index, sample_record_sha256=sample_digest, hard_valid=hard, applicable=applicable,
                  native=None, exclusion_contact=None, old_r5_radius=None, old_capture_valid=None,
                  classification=None, contact=None)
    if applicable:
        classification = budget.query('native', index, lambda: context['observer'].classify(row['pose']))
        contact = budget.query('contact', index, lambda: context['contact'].classify(row['pose']))
        native, touching = classification['native_any'], contact['exclusion_contact']
        require(type(native) is bool and type(touching) is bool, 'Invalid physical classifier Boolean')
        require(classification['native_anchor_count'] in (0, 1, 2)
                and native == (classification['native_anchor_count'] > 0), 'Native anchor labels differ')
        require(not classification['registry_consistent_triangle'] or classification['native_anchor_count'] == 2,
                'Registry triangle lacks both anchors')
        radius = float(np.linalg.norm(chart_coordinates([row['pose']], context['reference'])[0]))
        old = context['reference']
        old_capture = bool(np.linalg.norm(position-old['capture_center']) <= old['capture_radius'])
        inside = bool(r5_contains([radius], [original_q], np.asarray([old_capture], bool))[0])
        record.update(native=native, exclusion_contact=touching, old_r5_radius=radius,
                      old_capture_valid=old_capture, classification=classification, contact=contact,
                      native_partition=dict(old_R5_intersection_native=native and inside,
                                            remaining_R4_native=native and not inside))
    bins = stratify_with_exterior(np.asarray([row['latent']]), region, context['strata'],
                                 np.asarray([accounting['support']], bool))
    record['strata'] = {key: int(value[0]) for key, value in bins.items()}
    budget.check()
    return record, accounting


def run(plan_path, *, plan_sha256):
    """Exclusive, no-resume pass under an externally frozen allocation."""
    require(threading.current_thread() is threading.main_thread(), 'One main thread required')
    plan_path = Path(plan_path).resolve()
    infrastructure._digest(plan_sha256, 'execution plan')
    require(sha(plan_path) == plan_sha256, 'Frozen label plan changed')
    plan = read(plan_path)
    require(plan['schema'] == PLAN_SCHEMA, 'Wrong physical-label execution plan')
    root = Path(plan['population']['root']).resolve()
    paths = {key: Path(plan['output'][key]).resolve() for key in ('receipt', 'labels', 'journal', 'failure')}
    require(len(set(paths.values())) == 4, 'Output paths must differ')
    for path in paths.values():
        require(not path.exists() and path.parent.is_dir(), 'Output must be new with an existing parent')
    bindings, source_bindings, readers = {}, {}, []
    begun = completed = 0; active = None; phase = 'metadata'; context = None; runtime = None; manifest = None
    ledger = paths['journal'].open('x'); labels = None
    def emit(value):
        ledger.write(json.dumps(value, allow_nan=False)+'\n'); ledger.flush(); os.fsync(ledger.fileno())
    def bind(path, expected=None):
        path = Path(path).resolve(); actual = sha(path)
        if expected is not None:
            infrastructure._digest(expected, str(path)); require(actual == expected, 'Changed bound input '+str(path))
        require(path not in paths.values(), 'Output aliases an input')
        require(str(path) not in bindings or bindings[str(path)] == actual, 'Input changed during pass '+str(path))
        bindings[str(path)] = actual
        return path
    budget = None
    def interrupted(signum, _frame): raise SystemExit(128+signum)
    prior_term = signal.signal(signal.SIGTERM, interrupted)
    try:
        emit(dict(state='started', schema=SCHEMA, plan_sha256=plan_sha256, new_pose_draws=0, new_Poisson_clouds=0))
        budget = Budget(plan['limits'], emit)
        bind(plan_path, plan_sha256)
        sources = local_sources(__file__)
        require(set(plan['source_sha256']) == set(sources), 'Frozen source closure differs')
        for name, source in sources.items():
            module = sys.modules.get(source.stem)
            if module is not None: require(Path(module.__file__).resolve() == source, 'Imported source path differs')
            bind(source, plan['source_sha256'][name]); source_bindings[str(source)] = bindings[str(source)]
        runtime = infrastructure.runtime_identity()
        for path, digest in runtime['file_sha256'].items(): bind(path, digest)
        for name in ('manifest', 'summary', 'samples', 'attempts'):
            suffix = '.jsonl' if name in ('samples', 'attempts') else '.json'
            bind(root/(name+suffix), plan['population'][name+'_sha256'])
        manifest = read(root/'manifest.json')
        count = manifest['samples']
        require(type(count) is int and 0 < count <= budget.limits['max_rows'], 'Attempt allocation exceeds plan')
        require(count == plan['population']['samples'] and manifest['seed'] == plan['population']['seed'],
                'Frozen population identity differs')
        for role in QUERY_ROLES:
            require(count <= budget.limits[f'max_{role}_queries'], 'Plan cannot cover all possible '+role+' queries')
        context = _load_context(root, plan, bind, emit, budget)
        emit(dict(state='inputs_bound', samples=count, input_sha256=bindings, source_sha256=source_bindings,
                  runtime=runtime, classifier_binding=context['classifier_binding'],
                  shape_witness=context['shape_witness'], wall_certificate=context['wall_certificate'],
                  source_schema=context['source_schema'], native_identity_origin=context['native_identity_origin'],
                  compiled_native_binding=context['compiled_native_binding'],
                  shape_compatibility_binding=context['shape_compatibility_binding']))
        maximum = budget.limits['max_record_bytes']
        samples, attempts = BoundedLines(root/'samples.jsonl', maximum), BoundedLines(root/'attempts.jsonl', maximum)
        readers.extend((samples, attempts)); labels = paths['labels'].open('xb'); label_hash = hashlib.sha256()
        counts = Counter(); phase = 'rows'
        for index in range(count):
            active = index; begun += 1; budget.check()
            emit(dict(state='begin', draw=index, sample_offset=samples.bytes, attempt_offset=attempts.bytes))
            attempt, row = attempts.next(), samples.next()
            require(type(attempt) is dict and attempt == dict(draw=index, state='begin') and type(attempt.get('draw')) is int,
                    'Missing/reordered original attempt')
            require(row is not None, 'Missing original sample')
            record, accounting = classify_row(row, index, context, budget, samples.last['sha256'])
            raw = (json.dumps(record, separators=(',', ':'), allow_nan=False)+'\n').encode()
            require(len(raw) <= maximum, 'Label record byte budget exceeded')
            labels.write(raw); labels.flush(); os.fsync(labels.fileno()); label_hash.update(raw)
            counts.update(accounting['counters']); completed += 1
            counts['native_entry_unbound_anomalies'] += int(record['native'] is True and record['exclusion_contact'] is False)
            emit(dict(state='complete', draw=index, sample_record=samples.last,
                      label_record_sha256=hashlib.sha256(raw).hexdigest(), applicable=record['applicable']))
            active = None
        phase = 'final_validation'
        require(samples.next() is None and attempts.next() is None, 'Extra records beyond original allocation')
        for reader, key in ((samples, 'samples_sha256'), (attempts, 'attempts_sha256')):
            require(reader.lines == count and reader.hash.hexdigest() == plan['population'][key], 'Consumed input differs')
        for key in ('capture_rejected', 'hard_rejected', 'region_rejected', 'shell_rejected'):
            require(context['summary'][key] == counts[key], 'Saved rejection count differs: '+key)
        require(budget.calls['capture'] == count
                and budget.calls['atomic'] == count-counts['capture_rejected']
                and budget.calls['native'] == budget.calls['contact'] == counts['contributing'],
                'Incomplete physical query coverage')
        labels.close()
        for path, digest in bindings.items(): require(sha(path) == digest, 'Input/source/runtime changed during pass '+path)
        require(not (root/'failure.json').exists(), 'Producer failure appeared during pass')
        budget.check()
        result = dict(schema=SCHEMA, complete=True, passed=True, samples=count, seed=manifest['seed'],
            source_schema=context['source_schema'], native_identity_origin=context['native_identity_origin'],
            compiled_native_binding=context['compiled_native_binding'],
            shape_compatibility_binding=context['shape_compatibility_binding'],
            physical_input_witness=context['physical_input_witness'],
            root=str(root), manifest_sha256=plan['population']['manifest_sha256'],
            samples_sha256=plan['population']['samples_sha256'], attempts_sha256=plan['population']['attempts_sha256'],
            labels=dict(path=str(paths['labels']), sha256=label_hash.hexdigest()),
            region_sha256=manifest['region_sha256'], shape_sha256=manifest['shape_sha256'],
            reference_region_sha256=plan['reference_region']['sha256'], strata_sha256=fingerprint(context['strata']),
            strata_file_sha256=plan['strata']['sha256'], target_and_regions_sha256=context['target_and_regions_sha256'],
            definition_sha256=plan['definition']['sha256'], classifier_binding=context['classifier_binding'],
            physical_hard_validity_scope='every_attempt_capture_and_atomic',
            physical_native_contact_scope='every_contributing_pose',
            counts=dict(counts), query_counts=dict(budget.calls), input_sha256=bindings,
            observer_setup_inventory=context['observer_setup_inventory'], observer_setup_counts=dict(budget.setup_calls),
            source_sha256=source_bindings, runtime=runtime, wall_certificate=context['wall_certificate'],
            shape_witness=context['shape_witness'], limits=plan['limits'], execution_plan_sha256=plan_sha256,
            analysis_cpu_seconds=time.process_time()-budget.started, wall_seconds=time.monotonic()-budget.wall,
            full_line_geometry_certified=False, new_pose_draws=0, new_Poisson_clouds=0,
            scope='Every captured attempted endpoint is independently checked, including saved zero rows; complete native/contact records only on contributing targets. Proposal interval completeness, physical bath point geometry, coverage and assembly remain separate gates.')
        emit(dict(state='finished', complete=True, query_counts=dict(budget.calls), counts=dict(counts)))
        ledger.close(); result['journal'] = dict(path=str(paths['journal']), sha256=sha(paths['journal']))
        infrastructure._write(paths['receipt'], result)
        return result
    except BaseException as exc:
        if labels is not None and not labels.closed: labels.flush(); os.fsync(labels.fileno()); labels.close()
        event = dict(state='failed', phase=phase, draw=active, begun_rows=begun, completed_rows=completed,
                     error_type=type(exc).__name__, error=str(exc), input_stream_context=[r.last for r in readers])
        if not ledger.closed: emit(event); ledger.close()
        failure = dict(schema=SCHEMA, complete=False, passed=False, **event, input_sha256=bindings,
                       source_schema=None if manifest is None else manifest.get('schema'),
                       native_identity_origin=None if context is None else context['native_identity_origin'],
                       compiled_native_binding=None if context is None else context['compiled_native_binding'],
                       shape_compatibility_binding=None if context is None else context['shape_compatibility_binding'],
                       source_sha256=source_bindings, runtime=runtime,
                       query_counts={} if budget is None else dict(budget.calls),
                       observer_setup_counts={} if budget is None else dict(budget.setup_calls),
                       journal=dict(path=str(paths['journal']), sha256=sha(paths['journal'])),
                       labels=None if not paths['labels'].exists() else dict(path=str(paths['labels']), sha256=sha(paths['labels'])),
                       new_pose_draws=0, new_Poisson_clouds=0)
        if not paths['failure'].exists(): infrastructure._write(paths['failure'], failure)
        raise
    finally:
        signal.signal(signal.SIGTERM, prior_term)
        for reader in readers: reader.close()
        if labels is not None and not labels.closed: labels.close()
        if not ledger.closed: ledger.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    args = parser.parse_args()
    receipt = run(args.plan, plan_sha256=args.plan_sha256)
    print(json.dumps(dict(complete=True, passed=True, samples=receipt['samples'], counts=receipt['counts'])))
