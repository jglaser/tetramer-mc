#!/usr/bin/env python3
"""Audit the fixed AB shoulder pilot and report descriptive mixing diagnostics.

All attempts are replayed. Occupancy and autocorrelation retain cycle endpoints
including repeats after the fixed burn. A and B contact incidence remain separate.
This observer cannot establish global equilibrium, assembly, or physical rates.
"""
from __future__ import annotations

import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import copy
import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial import cKDTree

from prepare_smc_normalizer_atlas import Density, arrays, read, relative_poses, sha, write
from prepare_deep_far_normalizer_atlas import registration
from analyze_involution_docking_campaign import ChartAudit
from analyze_posterior_docking_pilot import effective_count
from prepare_shoulder_docking_benchmark import LABELS, MODEL_SHA, WINDOW, validate_plan


def contains(q, window=WINDOW):
    q = np.asarray(q)
    lower = q >= window['minimum'] if window['lower_inclusive'] else q > window['minimum']
    upper = q <= window['maximum'] if window['upper_inclusive'] else q < window['maximum']
    return np.isfinite(q) & lower & upper


def validate_gate(gate, log_acceptance):
    for name in ('gained', 'lost', 'raw_points'):
        assert type(gate[name]) is int and gate[name] >= 0, (name, gate[name])
    assert gate['gained']+gate['lost'] <= gate['raw_points']
    assert type(log_acceptance) in (int, float) and np.isfinite(log_acceptance) and log_acceptance <= 0.
    assert type(gate['log_weight']) in (int, float) and np.isfinite(gate['log_weight'])


def validate_source_bundle(campaign, manifest, bundle_path):
    """Tie embedded executable source text to the independently archived source."""
    bundle = read(bundle_path)
    assert bundle['schema'] == 1
    expected = manifest['source_bundle_files']
    mandatory = {'Cargo.toml': 'Cargo.toml', 'Cargo.lock': 'Cargo.lock', 'build.rs': 'build.rs',
                 'vendor/README.md': 'source/vendor/README.md'}
    assert all(expected.get(name) == path for name, path in mandatory.items())
    rust_sources = {name.removeprefix('source/'): name for name in manifest['input_sha256']
                    if name.startswith('source/src/') and name.endswith('.rs')}
    assert rust_sources and expected == dict(mandatory, **rust_sources)
    assert set(bundle['files']) == set(expected), 'Executable source list differs from archived source list'
    for runtime_name, archived_name in expected.items():
        record = bundle['files'][runtime_name]
        raw = (Path(campaign)/'provenance'/archived_name).read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        assert manifest['input_sha256'][archived_name] == digest
        assert record['sha256'] == digest, runtime_name
        assert record['text'].encode('utf-8') == raw, runtime_name


def validate_terminal_status(campaign, manifest):
    status = read(Path(campaign)/'status.json')
    assert status['running'] is False and status['complete'] is True
    jobs = status['jobs']
    ids = [entry['id'] for entry in jobs]
    expected = [job['id'] for job in manifest['jobs']]
    assert len(ids) == len(expected) and len(ids) == len(set(ids)) and set(ids) == set(expected)
    for entry in jobs:
        assert type(entry['exit_code']) is int and entry['exit_code'] == 0 and entry['status'] == 'complete'
    return sha(Path(campaign)/'status.json')


def assign_labels(q, radii):
    """Disjoint physical masks; Gaussian proposal components are not labels."""
    q, radii = np.asarray(q), np.asarray(radii)
    assert radii.shape == (len(q), 3) and np.all(contains(q))
    labels = np.full(len(q), 4, dtype=int)
    inner = q < 1.1
    labels[inner] = 3
    for index in range(3):
        labels[inner & (labels == 3) & (radii[:, index] <= .5)] = index
    return labels


def occupancy(labels):
    return {name: float(np.mean(np.asarray(labels) == index)) for index, name in enumerate(LABELS)}


def descriptor_ess(values, cpu):
    result = effective_count(values)
    result['apparent_ess_per_sampler_cpu_second'] = (
        result['apparent_ess'] / cpu if result['apparent_ess'] is not None and cpu > 0 else None)
    return result


def transitions(labels, selection=None):
    labels = np.asarray(labels, dtype=int)
    matrix = np.zeros((len(LABELS), len(LABELS)), dtype=int)
    mask = np.ones(max(0, len(labels)-1), dtype=bool) if selection is None else np.asarray(selection, dtype=bool)
    assert len(mask) == len(labels)-1
    for old, new in zip(labels[:-1][mask], labels[1:][mask]):
        matrix[old, new] += 1
    n = max(1, int(mask.sum()))
    counts = matrix.sum(axis=1)
    conditional = np.divide(matrix, counts[:, None], out=np.zeros_like(matrix, dtype=float), where=counts[:, None] > 0)
    return dict(labels=LABELS, counts=matrix.tolist(), empirical_flux_per_attempt=(matrix/n).tolist(),
        conditional_transition_probabilities=conditional.tolist(), source_counts=counts.tolist(),
        absent_source_rows=[LABELS[i] for i in np.flatnonzero(counts == 0)],
        off_diagonal_events=int(matrix.sum()-np.trace(matrix)), self_loops=int(np.trace(matrix)),
        interpretation='Includes every selected attempt and rejection. Unvisited-source rows are zero placeholders. Occupancy-weighted flux is distinct from directional transition probabilities; equality of finite counts does not certify stationarity. Ordered local/global cycles need not be reversible.')


def pair_roundtrips(labels):
    """For each pair, count nonoverlapping A-B-A visits, ignoring other labels."""
    result = {}
    for a, b in itertools.combinations(range(len(LABELS)), 2):
        visits = []
        for index, value in enumerate(labels):
            if value in (a, b) and (not visits or value != visits[-1][1]):
                visits.append((index, int(value)))
        trips = [dict(first_attempt=visits[i][0], last_attempt=visits[i+2][0],
                      start_label=LABELS[visits[i][1]], elapsed_attempts=visits[i+2][0]-visits[i][0])
                 for i in range(0, len(visits)-2, 2)]
        result[f'{LABELS[a]}|{LABELS[b]}'] = dict(completed=len(trips), trips=trips,
            convention='Pair-specific first-hit visits, other labels ignored; successive roundtrips do not overlap except their shared endpoint.')
    return result


def allocated_attempt_cpu(moves, labels, burn_stamp, final_stamp):
    """Partition recorded time; intervening logging cost belongs to the next attempt."""
    stamps = np.asarray([burn_stamp]+[move['sampler_cpu_seconds'] for move in moves], dtype=float)
    intervals = np.diff(stamps)
    tail = float(final_stamp-stamps[-1])
    assert np.all(intervals >= -1e-9) and tail >= -1e-9
    assert abs(float(intervals.sum())+tail-(final_stamp-burn_stamp)) < 1e-8
    def group(mask):
        indices = np.flatnonzero(mask)
        seconds = float(intervals[indices].sum())
        accepted = sum(bool(moves[i]['accepted']) for i in indices)
        exchanged = sum(labels[i] != labels[i+1] for i in indices)
        return dict(attempts=len(indices), accepted=int(accepted), label_changes=int(exchanged),
            allocated_sampler_cpu_seconds=seconds,
            accepted_per_allocated_cpu_second=accepted/seconds if seconds > 0 else None,
            label_changes_per_allocated_cpu_second=exchanged/seconds if seconds > 0 else None)
    return dict(by_slot={str(slot): group([m['attempt'] == slot for m in moves]) for slot in range(3)},
        by_kernel={kind: group([m['kind'] == kind for m in moves]) for kind in sorted({m['kind'] for m in moves})},
        final_frame_cpu_tail_seconds=tail, total_allocated_attempt_cpu_seconds=float(intervals.sum()),
        full_postburn_cpu_seconds=float(final_stamp-burn_stamp),
        interpretation='Differences of cumulative move timestamps; first interval starts at the burn frame timestamp. Includes intervening I/O and diagnostics, so these are allocated costs, not isolated kernel profiling. Full post-burn frame difference remains the ESS denominator.')


def contact_incidence(cfg, shape, frames, cpu):
    """Exact expanded-sphere contact incidence at every retained cycle endpoint."""
    centers = np.asarray([a['center'] for a in shape['atoms']])
    radii = np.asarray([a['radius'] for a in shape['atoms']])
    ft, _, fr = arrays(cfg['fixed_poses'])
    assert len(ft) == 2
    fixed = [centers @ rotation.T + position for rotation, position in zip(fr, ft)]
    trees = [cKDTree(points) for points in fixed]
    n, rd = len(centers), cfg['depletant_radius']
    bits = np.empty((len(frames), 2, 2*n), dtype=bool)
    rows, previous_pose, previous_row = [], None, None
    for serial, frame in enumerate(frames):
        pose = frame['pose']
        if pose == previous_pose:
            bits[serial] = bits[serial-1]
            row = copy.deepcopy(previous_row)
        else:
            t, _, rotations = arrays([pose])
            moved = centers @ rotations[0].T + t[0]
            row = {}
            for j, name in enumerate(('A', 'B')):
                candidates = trees[j].query_ball_point(moved, radii+radii.max()+2*rd, workers=1)
                incidence = np.zeros(2*n, dtype=bool)
                pair_count, min_gap = 0, float('inf')
                for i, js in enumerate(candidates):
                    if not js:
                        continue
                    js = np.asarray(js, dtype=int)
                    gaps = np.linalg.norm(fixed[j][js]-moved[i], axis=1)-radii[i]-radii[js]
                    min_gap = min(min_gap, float(gaps.min()))
                    selected = js[gaps < 2*rd]
                    if len(selected):
                        incidence[i], incidence[n+selected] = True, True
                        pair_count += len(selected)
                assert min_gap >= -1e-8, (frame['cycle'], name, min_gap)
                bits[serial, j] = incidence
                row[name] = dict(mobile_contact_atoms=int(incidence[:n].sum()), fixed_contact_atoms=int(incidence[n:].sum()),
                    overlapping_exclusion_sphere_pairs=pair_count, minimum_tested_core_gap_A=min_gap if np.isfinite(min_gap) else None)
            previous_pose, previous_row = pose, copy.deepcopy(row)
        assert frame['depletion_contact'] == any(row[name]['overlapping_exclusion_sphere_pairs'] > 0 for name in ('A', 'B'))
        rows.append(dict(cycle=frame['cycle'], **row))
    result = dict(samples=len(frames), stride_cycles=1, rows=rows, neighbors={})
    half = len(bits)//2
    for j, name in enumerate(('A', 'B')):
        x = bits[:, j]
        centered = x.astype(float)-x.mean(axis=0)
        base = float(np.mean(np.sum(centered*centered, axis=1)))
        lags = []
        for lag in (1, 2, 5, 10, 20, 50, 100, 250, 500):
            if lag >= len(x):
                continue
            union = np.sum(x[:-lag] | x[lag:], axis=1)
            intersection = np.sum(x[:-lag] & x[lag:], axis=1)
            jaccard = np.divide(intersection, union, out=np.ones(len(union)), where=union > 0)
            lags.append(dict(lag_cycles=lag, mean_jaccard_distance=float(np.mean(1-jaccard)),
                centered_incidence_correlation=float(np.mean(np.sum(centered[:-lag]*centered[lag:], axis=1))/base) if base > 0 else None))
        result['neighbors'][name] = dict(apparent_effective_count=descriptor_ess(x, cpu), lags=lags,
            mean_incidence=x.mean(axis=0).tolist(), first_half_mean_incidence=x[:half].mean(axis=0).tolist(),
            last_half_mean_incidence=x[half:].mean(axis=0).tolist(),
            half_mean_incidence_rms_difference=float(np.sqrt(np.mean((x[:half].mean(axis=0)-x[half:].mean(axis=0))**2))))
    result['joint_apparent_effective_count'] = descriptor_ess(bits.reshape(len(bits), -1), cpu)
    result['descriptor'] = 'Separate A and B binary mobile/fixed atom incidence from exact expanded-sphere overlaps. Redundant spheres retained; this is not crystallographic patch registration or an overlap-volume estimator. Every repeat retained.'
    return result


def one(task):
    campaign, out, job = task
    campaign, out = Path(campaign), Path(out)
    begun = time.monotonic()
    manifest = read(campaign/'manifest.json')
    cfg, model = read(job['config']), read(campaign/'provenance/model.json')
    assert sha(job['config']) == job['config_sha256']
    assert cfg['seed'] == job['seed']
    assert cfg['target_region'] == manifest['target_region']
    assert cfg['proposal_anchor_index'] == 0 and len(cfg['fixed_poses']) == 2
    directory = Path(job['directory'])
    summary, checkpoint = read(directory/'summary.json'), read(directory/'checkpoint.json')
    run_manifest = read(directory/'manifest.json')
    assert manifest.get('binary_supplied') and manifest.get('binary_sha256'), 'A reviewed binary must be frozen before production assessment'
    assert run_manifest['executable_sha256'] == manifest['binary_sha256']
    assert sha(campaign/'provenance/docking-mc') == manifest['binary_sha256']
    assert run_manifest['source_bundle_sha256'] == sha(directory/'provenance/source-bundle.json')
    validate_source_bundle(campaign, manifest, directory/'provenance/source-bundle.json')
    assert run_manifest['target_region'] == cfg['target_region']
    assert run_manifest['proposal_anchor_index'] == cfg['proposal_anchor_index']
    assert run_manifest['cycles'] == manifest['cycles'] and run_manifest['sample_every'] == 1
    for key, expected, source_name in (
            ('config_sha256', job['config_sha256'], 'input-config.json'),
            ('model_sha256', manifest['model_sha256'], 'model.json'),
            ('shape_sha256', manifest['shape_sha256'], 'shape.json')):
        assert run_manifest[key] == expected and checkpoint[key] == expected
        assert sha(directory/'provenance'/source_name) == expected
    assert summary['complete'] and summary['completed_cycles'] == manifest['cycles']
    assert summary['config_sha256'] == job['config_sha256']
    assert summary['model_sha256'] == manifest['model_sha256']
    assert summary['shape_sha256'] == manifest['shape_sha256']
    assert summary['method'] == ('local' if job['mode'] == 'local' else 'posterior_involution')
    assert summary['correlation'] == job['correlation']
    assert run_manifest['method'] == checkpoint['method'] == summary['method']
    assert run_manifest['correlation'] == checkpoint['correlation'] == job['correlation']
    with (directory/'trajectory.jsonl').open() as stream:
        frames = [json.loads(line) for line in stream]
    with (directory/'moves.jsonl').open() as stream:
        moves = [json.loads(line) for line in stream]
    cycles, burn = manifest['cycles'], manifest['burn_cycles']
    assert [f['cycle'] for f in frames] == list(range(cycles+1))
    assert len(moves) == cycles*3 and cfg['local_attempts_per_cycle'] == 2
    metric = cfg['target_region']['metric']
    frameq = registration([f['pose'] for f in frames], metric)
    retainedq = registration([m['retained_pose'] for m in moves], metric)
    assert np.all(contains(frameq)) and np.all(contains(retainedq))
    candidate_ids = [i for i, m in enumerate(moves) if m['proposed_pose'] is not None]
    candidateq = dict(zip(candidate_ids, registration([moves[i]['proposed_pose'] for i in candidate_ids], metric))) if candidate_ids else {}
    anchor = cfg['fixed_poses'][0]
    # Frozen geometric masks, with original q and priority exactly unchanged.
    combined_poses = [frames[0]['pose']] + [m['retained_pose'] for m in moves]
    combined_relative = relative_poses(combined_poses, anchor)
    radii = np.column_stack([Density(read(campaign/f'provenance/geometric-{name}.json')).evaluate(combined_relative)[1][:, 0]
                            for name in LABELS[:3]])
    all_labels = assign_labels(np.r_[frameq[0], retainedq], radii)
    frame_labels = all_labels[::3]
    assert len(frame_labels) == len(frames)

    audit, density = ChartAudit(model), Density(model)
    learned = [i for i, m in enumerate(moves) if (m.get('proposal') or {}).get('step') is not None]
    if learned:
        old_relative = relative_poses([moves[i]['old_pose'] for i in learned], anchor)
        new_relative = relative_poses([moves[i]['proposed_pose'] for i in learned], anchor)
        oldg, _, oldlogs = density.evaluate(old_relative)
        newg, _, newlogs = density.evaluate(new_relative)
        logw = np.log(density.weights)
        for index, serial in enumerate(learned):
            p = moves[serial]['proposal']
            assert p['source_law'] == 'posterior' and p['anchor_index'] == 0
            a, b = p['trace']['source'], p['trace']['target']
            audit.close('full_old_gaussian', oldg[index], p['full_old_gaussian_log_density'])
            audit.close('full_new_gaussian', newg[index], p['full_new_gaussian_log_density'])
            audit.close('source_posterior', oldlogs[index, a]-oldg[index], p['source_log_probability'])
            audit.close('reverse_source_posterior', newlogs[index, b]-newg[index], p['inverse_source_log_probability'])
            label = newlogs[index, b]-newg[index]+logw[a]-(oldlogs[index, a]-oldg[index])-logw[b]
            audit.close('posterior_label_ratio', label, p['label_log_reverse_forward'])
            audit.close('expanded_posterior_ratio', p['step']['log_correction']+label, p['expanded_log_reverse_forward'])
            audit.close('collapsed_posterior_ratio', oldg[index]-newg[index], p['log_reverse_forward'])
            audit.close('expanded_equals_collapsed', p['expanded_log_reverse_forward'], p['log_reverse_forward'])
        # Exact full-mixture correction checked above for every learned move;
        # chart maps receive deterministic, evenly spaced checks plus accepts.
        selected = set(np.linspace(0, len(learned)-1, min(128, len(learned)), dtype=int))
        accepted = [i for i, serial in enumerate(learned) if moves[serial]['accepted']]
        if accepted:
            selected.update(accepted[i] for i in np.linspace(0, len(accepted)-1, min(128, len(accepted)), dtype=int))
        for index in sorted(selected):
            p = copy.deepcopy(moves[learned[index]]['proposal'])
            p.pop('full_old_gaussian_log_density', None)
            p.pop('full_new_gaussian_log_density', None)
            actual = audit.involution(old_relative[index], p, job['correlation'])
            audit.compare_pose(audit.absolute(actual, anchor), moves[learned[index]]['proposed_pose'], 'candidate_lab')
    else:
        selected = set()

    state, contact = frames[0]['pose'], frames[0]['depletion_contact']
    audit.compare_pose(state, cfg['initial_pose'], 'initial')
    counters, expected_counts = defaultdict(Counter), [Counter(), Counter()]
    for serial, move in enumerate(moves):
        cycle, attempt = divmod(serial, 3)
        cycle += 1
        assert move['cycle'] == cycle and move['attempt'] == attempt
        expected_kind = 'local' if job['mode'] == 'local' or attempt < 2 else 'global'
        assert move['kind'] == expected_kind
        audit.compare_pose(state, move['old_pose'], 'old_pose')
        assert move['depletion_contact_before'] == contact
        previousq = frameq[0] if serial == 0 else retainedq[serial-1]
        audit.close('old_q', previousq, move['old_q'])
        audit.close('retained_q', retainedq[serial], move['retained_q'])
        branch, p = move['branch'], move.get('proposal') or {}
        count = counters[branch]
        count['attempted'] += 1
        count['accepted'] += int(move['accepted'])
        expected = expected_counts[int(expected_kind == 'global')]
        expected['attempted'] += 1
        expected['accepted'] += int(move['accepted'])
        candidate = move['proposed_pose']
        if candidate is None:
            expected['numerical_nulls'] += 1
            assert not move['accepted'] and move['gate'] is None and move['proposed_q'] is None
        else:
            capture = bool(np.linalg.norm(np.asarray(candidate['position'])-cfg['capture_center']) <= cfg['capture_radius'])
            valid_region = bool(contains(candidateq[serial]))
            assert move['capture_valid'] == capture and move['region_valid'] == valid_region
            audit.close('candidate_q', candidateq[serial], move['proposed_q'])
            if not capture:
                expected['capture_rejected'] += 1
            elif not valid_region:
                expected['region_rejected'] += 1
            elif not move['hard_valid']:
                expected['hard_rejected'] += 1
            else:
                expected['hard_valid'] += 1
            count['region_valid_candidates'] += int(valid_region)
            if expected_kind == 'global':
                assert p['anchor_index'] == 0
        if expected_kind == 'local' or branch == 'uniform':
            audit.close('symmetric_proposal_correction', p['log_reverse_forward'], 0.)
        if move.get('gate') is not None:
            assert move['capture_valid'] and move['region_valid'] and move['hard_valid']
            gate = move['gate']
            validate_gate(gate, move['log_acceptance'])
            coefficient = np.log1p(1. / cfg['poisson_lambda_ratio']) if cfg['reservoir_density'] > 0 else 0.
            audit.close('conditional_poisson_log_weight', coefficient*(gate['gained']-gate['lost']), gate['log_weight'])
            expected['gate_raw_points'] += gate['raw_points']
            alpha = min(0., p['log_reverse_forward']+gate['log_weight'])
            audit.close('acceptance', alpha, move['log_acceptance'])
            count['gate_evaluations'] += 1
        else:
            assert not move['accepted'] and move['log_acceptance'] is None
        if move['log_acceptance'] == 0.:
            assert move['accepted'], 'A valid unit-acceptance move cannot be rejected'
        if move['accepted']:
            audit.compare_pose(candidate, move['retained_pose'], 'accepted')
            pt, pr = audit.arrays(state)
            nt, nr = audit.arrays(candidate)
            changed = np.linalg.norm(nt-pt) > 1e-10 or np.max(np.abs(nr-pr)) > 1e-12
            expected['accepted_pose_changes'] += int(changed)
            count['accepted_pose_changes'] += int(changed)
        else:
            audit.compare_pose(state, move['retained_pose'], 'rejected')
        state, contact = move['retained_pose'], move['depletion_contact']
        if attempt == 2:
            audit.compare_pose(state, frames[cycle]['pose'], 'frame_replay')
            assert contact == frames[cycle]['depletion_contact']
    for index, expected in enumerate(expected_counts):
        for key in set(expected) | set(summary['counts'][index]):
            assert expected[key] == summary['counts'][index].get(key, 0), (index, key, expected[key], summary['counts'][index])
    audit.compare_pose(state, checkpoint['pose'], 'checkpoint')
    assert checkpoint['counts'] == summary['counts']
    assert checkpoint['completed_cycles'] == cycles
    kept = frame_labels[burn+1:]
    cpu = float(frames[-1]['sampler_cpu_seconds']-frames[burn]['sampler_cpu_seconds'])
    half = len(kept)//2
    assert len(kept) == cycles-burn and cpu > 0
    attempt_labels = all_labels[3*burn:]
    kept_moves = moves[3*burn:]
    attempt_cost = allocated_attempt_cpu(kept_moves, attempt_labels,
        frames[burn]['sampler_cpu_seconds'], frames[-1]['sampler_cpu_seconds'])
    flux_by_slot = {str(slot): transitions(attempt_labels, [m['attempt'] == slot for m in kept_moves]) for slot in range(3)}
    flux_by_kernel = {kind: transitions(attempt_labels, [m['kind'] == kind for m in kept_moves])
                      for kind in sorted({m['kind'] for m in kept_moves})}
    contact_stats = contact_incidence(cfg, read(campaign/'provenance/shape.json'), frames[burn+1:], cpu)
    result = dict(id=job['id'], mode=job['mode'], start=job['start'], replicate=job['replicate'], seed=job['seed'],
        passed=True, initial_q=float(frameq[0]), final_q=float(frameq[-1]), burn_cycles=burn,
        retained_cycles=len(kept), sampler_cpu_seconds_after_burn=cpu, total_sampler_cpu_seconds=summary['sampler_cpu_seconds'],
        counts={key: dict(value) for key, value in counters.items()}, occupancy=occupancy(kept),
        first_half_occupancy=occupancy(kept[:half]), last_half_occupancy=occupancy(kept[half:]),
        half_occupancy_total_variation=float(.5*np.sum(np.abs(np.bincount(kept[:half], minlength=5)/half-np.bincount(kept[half:], minlength=5)/(len(kept)-half)))),
        label_ess={name: descriptor_ess(kept == index, cpu) for index, name in enumerate(LABELS)},
        joint_label_ess=descriptor_ess(np.eye(5)[kept], cpu), q_ess=descriptor_ess(frameq[burn+1:], cpu),
        all_attempt_flux=transitions(attempt_labels), attempt_flux_by_slot=flux_by_slot,
        attempt_flux_by_kernel=flux_by_kernel, attempt_cpu_allocation=attempt_cost,
        pair_roundtrips=pair_roundtrips(attempt_labels),
        contact_incidence=contact_stats,
        q_rows=[dict(cycle=f['cycle'], q=float(q), label=LABELS[label], sampler_cpu_seconds=f['sampler_cpu_seconds'])
                for f, q, label in zip(frames, frameq, frame_labels)],
        audit=dict(all_moves_replayed=len(moves), all_learned_density_checks=len(learned), representative_map_checks=len(selected),
            checks=dict(audit.checks), maximum_errors=dict(audit.max_errors), independent_geometry_frames=contact_stats['samples']),
        source_sha256={name: sha(directory/name) for name in ('moves.jsonl', 'trajectory.jsonl', 'summary.json', 'checkpoint.json')},
        inference_scope=manifest['inference_scope'], analysis_wall_seconds=time.monotonic()-begun)
    if manifest.get('physical_reference'):
        reference = manifest['physical_reference']['regions']
        differences = {name: result['occupancy'][name]-reference[name]['probability'] for name in LABELS}
        result['reference_comparison'] = dict(occupancy_minus_observed_reference=differences,
            total_variation=.5*sum(abs(value) for value in differences.values()),
            interpretation='Descriptive discrepancy against a finite independent reference; no convergence threshold or certified equilibrium conclusion.')
    destination = out/'runs'/job['id']
    destination.mkdir(parents=True)
    write(destination/'analysis.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('campaign', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 32:
        parser.error('Require one to 32 workers')
    campaign, out = args.campaign.resolve(), args.out.resolve()
    manifest = read(campaign/'manifest.json')
    assert manifest['schema'] == 'conditional-shoulder-docking-v1'
    if not manifest.get('binary_supplied') or not manifest.get('binary_sha256'):
        parser.error('This is an inert no-binary preparation. Assess only a fresh production preparation with an explicitly frozen reviewed binary.')
    validate_plan(manifest, campaign)
    assert manifest['model_sha256'] == MODEL_SHA
    terminal_status_sha256 = validate_terminal_status(campaign, manifest)
    for name, digest in manifest['input_sha256'].items():
        assert sha(campaign/'provenance'/name) == digest, name
    if out.exists() and any(out.iterdir()):
        parser.error('Analysis output directory must be new or empty')
    out.mkdir(parents=True, exist_ok=True)
    tasks = [(str(campaign), str(out), job) for job in manifest['jobs']]
    if args.workers == 1:
        results = list(map(one, tasks))
    else:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks))) as pool:
            results = list(pool.map(one, tasks))
    comparison = {}
    for mode in ('local', 'c0', 'c09'):
        groups = {start: [r for r in results if r['mode'] == mode and r['start'] == start]
                  for start in ('direct', 'geometry')}
        means = {start: {name: float(np.mean([r['occupancy'][name] for r in group])) for name in LABELS}
                 for start, group in groups.items()}
        comparison[mode] = dict(mean_occupancy_by_initialization=means,
            initialization_mean_total_variation=.5*sum(abs(means['direct'][name]-means['geometry'][name]) for name in LABELS),
            largest_run_half_total_variation=max(r['half_occupancy_total_variation'] for group in groups.values() for r in group))
    result = dict(passed=all(r['passed'] for r in results), runs=[{key: r[key] for key in
        ('id', 'mode', 'start', 'replicate', 'occupancy', 'first_half_occupancy', 'last_half_occupancy', 'label_ess',
         'joint_label_ess', 'q_ess', 'sampler_cpu_seconds_after_burn', 'half_occupancy_total_variation')} for r in results],
        initialization_comparison=comparison, physical_reference=manifest['physical_reference'],
        campaign_manifest_sha256=sha(campaign/'manifest.json'),
        terminal_status_sha256=terminal_status_sha256,
        analyzer_sha256=sha(__file__), inference_scope=manifest['inference_scope'],
        efficiency_conclusion='Not automatically promoted: inspect initialization differences, occupancy coverage, half agreement, and transitions before interpreting apparent ESS as sampler efficiency.')
    write(out/'analysis.json', result)
    print(json.dumps(dict(passed=result['passed'], runs=len(results), out=str(out))))


if __name__ == '__main__':
    main()
