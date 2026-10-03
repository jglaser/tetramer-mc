#!/usr/bin/env python3
"""Evaluate one reviewed fixed saved-row allocation with optional KD candidates.

No sampling, Poisson clouds, independent redraws, or unpruned comparison passes.
The previous full audit includes extra point predicates, so its CPU cannot be
divided by this worker's CPU to claim a speedup.
"""
import argparse
from collections import Counter
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
import traceback

for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS'):
    os.environ[name] = '1'

import numpy as np
import native_class_line_reference as reference
from prepare_native_class_projected_probe import PROVENANCE, check_setup_counts, read, require, sha, verify, write


def verify_loaded_namespace(root):
    """Every application import must resolve inside the frozen module directory."""
    names = ('prepare_native_class_projected_probe', 'run_evolving_dimer_analysis',
             'native_class_line_reference', 'hard_free_line_reference',
             'analyze_contact_line_audit', 'native_contact_regions')
    frozen = read(root/'freeze.json')['files']
    paths = [Path(__file__).resolve()]
    paths.extend(Path(sys.modules[name].__file__).resolve() for name in names)
    for path in paths:
        name = 'code/'+path.name
        require(path == root/name and sha(path) == frozen[name],
                'Worker imported code outside its frozen namespace: '+str(path))


@contextmanager
def count_observer_setup(inventory, emit, progress):
    """Bound and journal unchanged reference/scaffold initialization queries."""
    cls = reference.NativeContactRegions
    old_contacts, old_classifier = cls._contacts, cls.classify_pair
    counts = Counter(); in_scaffold = False
    progress.update(phase='observer_setup', observer_setup_counts=counts)
    def evaluate(kind, limit, call, extra=None):
        require(counts[kind+'_started'] < limit, 'Observer setup query cap exceeded: '+kind)
        index = counts[kind+'_started']; counts[kind+'_started'] += 1
        started = time.process_time()
        emit(dict(state='setup_query_begin', kind=kind, index=index, **(extra or {})))
        result = call()
        counts[kind+'_completed'] += 1
        emit(dict(state='setup_query_complete', kind=kind, index=index,
                  cpu_seconds=time.process_time()-started, counts=dict(counts), **(extra or {})))
        return result
    def contacts(model, position, rotation, cutoff):
        if in_scaffold:
            require(cutoff == reference.CRITERIA['contact_entry_A'], 'Changed scaffold contact cutoff')
            return evaluate('scaffold_contacts', inventory['maximum_scaffold_contact_calls'],
                            lambda: old_contacts(model, position, rotation, cutoff))
        index = counts['reference_contacts_started']
        require(index < inventory['reference_contact_calls'], 'Extra reference setup query')
        query = inventory['reference_contact_queries'][index]
        require(np.array_equal(position, np.asarray(query['position']))
                and np.array_equal(rotation, np.asarray(query['rotation']))
                and cutoff == reference.CRITERIA['native_reference_patch_gap_A'],
                'Reference setup query differs from frozen catalogue')
        return evaluate('reference_contacts', inventory['reference_contact_calls'],
                        lambda: old_contacts(model, position, rotation, cutoff), dict(label=query['label']))
    def classify(model, anchor, moving):
        nonlocal in_scaffold
        require(counts['reference_contacts_completed'] == inventory['reference_contact_calls']
                and [anchor, moving] == inventory['fixed_scaffold_poses'] and not in_scaffold,
                'Changed fixed-scaffold setup query')
        in_scaffold = True
        try:
            return evaluate('scaffold_classifier', inventory['fixed_scaffold_classifier_calls'],
                            lambda: old_classifier(model, anchor, moving))
        finally:
            in_scaffold = False
    cls._contacts, cls.classify_pair = contacts, classify
    try:
        yield counts
        require(counts['reference_contacts_completed'] == inventory['reference_contact_calls']
                and counts['scaffold_classifier_completed'] == inventory['fixed_scaffold_classifier_calls'],
                'Incomplete observer setup')
        require(counts['reference_contacts_started']+counts['scaffold_contacts_started'] <=
                inventory['maximum_total_contact_calls'], 'Observer setup total cap exceeded')
    finally:
        cls._contacts, cls.classify_pair = old_contacts, old_classifier


def reconstructors(root, plan, emit, progress):
    """Construct the unchanged original observer once and two optional guides."""
    with count_observer_setup(plan['observer_setup'], emit, progress):
        observer = reference.NativeContactRegions(root/plan['definition'])
    progress['phase'] = 'guide_initialization'
    answer = {}
    for label in ('hard_free-r00', 'saved'):
        directory = root/'inputs/queries'/label
        manifest = read(directory/'manifest.json')
        data = {name: read(directory/'provenance'/(name+'.json')) for name in PROVENANCE}
        config, region, guide, shape, compiled = [data[k] for k in
            ('config', 'region', 'importance-guide', 'shape', 'compiled-native')]
        require(guide['schema'] == reference.SCHEMA and guide['region_sha256'] == manifest['region_sha256']
                and guide['shape_sha256'] == region['shape_sha256'] == manifest['shape_sha256']
                and guide['compiled_native']['sha256'] == manifest['compiled_native_sha256'],
                'Changed physical guide binding')
        require(compiled['source_definition_sha256'] == manifest['native_definition_sha256']
                == sha(root/plan['definition']) and compiled['source_input_sha256'] == observer.definition['input_sha256'],
                'Changed complete native definition')
        reference.compare_compiled_definition(compiled, observer)
        for key in ('fixed_poses', 'capture_center', 'capture_radius', 'depletant_radius'):
            require(guide[key] == config[key], 'Guide/physical field differs: '+key)
        require(compiled['fixed_poses'] == config['fixed_poses']
                == region.get('physical_fixed_neighbors', [region['fixed_neighbor']]), 'Changed physical scaffold')
        for a, b in [('capture_center', 'capture_center'), ('capture_radius', 'capture_radius'),
                     ('activity', 'reservoir_density'), ('depletant_radius', 'depletant_radius'),
                     ('physical_metric', 'metadata')]:
            require(region[a] == config[b], 'Changed physical region: '+a)
        recon = reference.Reconstructor(region, guide, config, shape, observer, use_tree=True)
        reference.hard.close(manifest['log_latent_ball_volume'], recon.logvolume, 'Changed latent volume')
        reference.hard.close(manifest['latent_radius'], recon.radius, 'Changed latent radius')
        answer[label] = recon
    return answer


@contextmanager
def count_leaf_work():
    """Observe existing leaf calls/results; do not add predicates or queries."""
    previous = reference.leaf_contact_intervals
    counts = Counter()
    def wrapped(fixed, moving, direction, fr, mr, gap, segment, **kwargs):
        require(kwargs.get('use_tree') is True, 'Optional leaf path was not enabled')
        label = 'native' if kwargs['inclusive'] else 'exclusion'
        pairs = kwargs.get('pairs')
        possible = len(fixed)*len(moving) if pairs is None else len(pairs)
        counts[label+'_calls'] += 1
        counts[label+'_possible_pairs'] += possible
        result = previous(fixed, moving, direction, fr, mr, gap, segment, **kwargs)
        counts[label+'_evaluated_pairs'] += result['leaf_atom_pairs']
        require(result['leaf_atom_pairs'] <= possible, 'Candidate count exceeded complete pair inventory')
        return result
    reference.leaf_contact_intervals = wrapped
    try:
        yield counts
    finally:
        reference.leaf_contact_intervals = previous


def compare_row(row, recon, progress=None):
    """Exactly one density evaluation at an existing latent, then comparisons."""
    require(recon.use_tree is True, 'Unpruned reconstruction is forbidden in this probe')
    u = np.asarray(row['latent'])
    require(u.shape == (6,) and np.isfinite(u).all(), 'Invalid saved latent')
    progress = {} if progress is None else progress
    progress['phase'] = 'decode'
    x, position, rotation, jacobian = recon.decode(u)
    with count_leaf_work() as counts:
        progress.update(phase='density', partial_counts=counts)
        counts['density_queries'] = 1
        density = recon.density(u)
    progress['phase'] = 'comparisons'
    for key, expected in [('latent_radius', np.linalg.norm(u)), ('raw_coordinates', x),
                          ('log_physical_jacobian', jacobian)]:
        reference.hard.close(row[key], expected, 'Changed '+key)
    reference.hard.close(row['pose']['position'], position, 'Pose translation differs')
    reference.hard.close(reference.pose_arrays(row['pose'])[1], rotation, 'Pose rotation differs')
    reference.log_close(row['log_proposal_density'], density['log_density'], 'Full q differs')
    reference.log_close(row['baseline_log_density'], density['baseline_log_density'], 'Baseline q differs')
    maxima = dict(interval_endpoint_error=reference.compare_density_trace(
        row['density_details'], density, recon, u), jacobian_error=abs(row['log_physical_jacobian']-jacobian),
        log_density_error=abs(row['log_proposal_density']-density['log_density'])
            if math.isfinite(density['log_density']) else 0., inverse_CDF_error=0., draw_interval_endpoint_error=0.)
    require(len(density['axes']) <= 3, 'Exceeded translation-axis allocation')
    counts['axis_geometries'] = len(density['axes'])
    counts['hard_core_evaluated_pairs'] = sum(g.get('projected_pair_candidates', 0) for g in density['axes'])
    if row['kind'] == 'fresh':
        require(row['draw'] is not None, 'Missing saved draw trace')
        result = reference.audit_draw(row['draw'], recon, u, density)
        maxima['inverse_CDF_error'], maxima['draw_interval_endpoint_error'] = result['inverse_error'], result['endpoint_error']
    else:
        require(row['kind'] == 'probe' and row['draw'] is None, 'Saved query must not draw')
    progress['phase'] = 'complete'
    return dict(log_density=None if not math.isfinite(density['log_density']) else density['log_density'],
        log_physical_jacobian=jacobian, maxima=maxima, counts=dict(counts),
        pointwise_native_domain_observers_rerun=False)


def worker(root):
    root = Path(root).resolve()
    plan = verify(root)
    claim = read(root/'execution-claim.json')
    require(claim['freeze_sha256'] == sha(root/'freeze.json')
            and claim['allocation_sha256'] == sha(root/'allocation.json'), 'Unclaimed or changed execution')
    review = read(claim['review']['path'])
    require(sha(claim['review']['path']) == claim['review']['sha256'] and review['complete'] is True
            and review['passed'] is True and review['freeze_sha256'] == claim['freeze_sha256']
            and review['allocation_sha256'] == claim['allocation_sha256'], 'Changed execution review')
    started = time.process_time(); wall = time.monotonic(); completed = []
    totals, maxima = Counter(), Counter()
    current = None
    row_started, row_wall = started, wall
    from run_evolving_dimer_analysis import terminate_requested
    verify_loaded_namespace(root)
    previous = {sig: signal.signal(sig, terminate_requested) for sig in (signal.SIGTERM, signal.SIGXCPU)}
    progress = {}
    try:
        with (root/'journal.jsonl').open('x') as stream:
            def emit(event):
                stream.write(json.dumps(event, allow_nan=False)+'\n'); stream.flush(); os.fsync(stream.fileno())
            emit(dict(state='started', row_count=56, maximum_workers=1, optional_candidates=True,
                      freeze_sha256=claim['freeze_sha256'], allocation_sha256=claim['allocation_sha256']))
            try:
                emit(dict(state='setup_begin', inventory=plan['observer_setup']))
                models = reconstructors(root, plan, emit, progress)
                setup_cpu = time.process_time()-started
                setup_counts = dict(progress.get('observer_setup_counts', {}))
                check_setup_counts(plan['observer_setup'], setup_counts)
                emit(dict(state='setup_complete', counts=setup_counts, cpu_seconds=setup_cpu,
                          guide_instances=len(models)))
                receipts = read(root/'prior-unpruned-receipts.json')
                raw = (root/'selected-rows.jsonl').read_bytes().splitlines()
                for entry, value in zip(plan['rows'], raw):
                    current = entry
                    progress = dict(phase='begin', partial_counts={})
                    row_started, row_wall = time.process_time(), time.monotonic()
                    emit(dict(state='begin', **entry))
                    result = compare_row(json.loads(value), models[entry['query']], progress)
                    result.update(cpu_seconds=time.process_time()-row_started,
                                  wall_seconds=time.monotonic()-row_wall)
                    totals.update(result['counts'])
                    require(totals['density_queries'] <= 56 and totals['axis_geometries'] <= 168,
                            'Exceeded frozen query budget')
                    for key, value in result['maxima'].items(): maxima[key] = max(maxima[key], value)
                    result['prior_full_audit_cpu_seconds'] = receipts[entry['ordinal']]['cpu_seconds'] if entry['ordinal'] < 16 else None
                    emit(dict(state='complete', **entry, result=result, cumulative_counts=dict(totals), maxima=dict(maxima)))
                    completed.append(dict(entry, **result))
                    current = None
                progress = dict(phase='final_verification', partial_counts={})
                verify(root)
                require(sha(root/'freeze.json') == claim['freeze_sha256'], 'Freeze changed during geometry')
                summary = dict(complete=True, passed=True, completed_rows=len(completed),
                    rows=completed, counts=dict(totals), maxima=dict(maxima), setup_cpu_seconds=setup_cpu,
                    observer_setup_counts=setup_counts, observer_setup_inventory=plan['observer_setup'],
                    cpu_seconds=time.process_time()-started, wall_seconds=time.monotonic()-wall,
                    optional_matches_Rust=True, fresh_prior_correspondence_rows=16,
                    saved_unpruned_correspondence_pending_rows=40, full_independent_audit_complete=False,
                    physical_campaign_gate_open=False, new_poses=0, new_clouds=0, retries=0,
                    cpu_comparison=plan['timing'], correspondence_scope=plan['equivalence'])
                write(root/'result.json', summary)
                emit(dict(state='finished', completed_rows=56, result_sha256=sha(root/'result.json')))
            except BaseException as error:
                emit(dict(state='failed', current=current, completed_rows=len(completed),
                    error=repr(error), traceback=traceback.format_exc(),
                    current_cpu_seconds=time.process_time()-row_started,
                    current_wall_seconds=time.monotonic()-row_wall, cumulative_counts=dict(totals),
                    current_progress=progress, failed_leaf_internal_progress_available=False,
                    retries=0, partial_outputs_retained=True))
                raise
    finally:
        for sig, handler in previous.items(): signal.signal(sig, handler)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    worker(parser.parse_args().root)
