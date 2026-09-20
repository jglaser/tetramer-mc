#!/usr/bin/env python3
"""Independent posterior-label audit and bounded docking mixing diagnostics.

Replay and acceptance arithmetic cover every attempt. Full Gaussian densities
are independently vectorized over every learned proposal; the chart involution
itself is independently reconstructed at a prespecified representative subset.
"""
from __future__ import annotations

import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import copy
import json
from pathlib import Path
import shutil
import time
import numpy as np
from scipy.spatial import cKDTree

from prepare_smc_normalizer_atlas import ROOT, Density, arrays, read, relative_poses, sha, write
from prepare_deep_far_normalizer_atlas import registration
from analyze_involution_docking_campaign import ChartAudit, CorePassages


def qbin(q):
    return 'native_core' if q <= .8 else 'native_shell' if q <= 1 else 'shoulder' if q < 2 else 'intermediate' if q < 5 else 'far'


def occupancy(q):
    count = Counter(qbin(v) for v in q)
    return {key: count[key]/len(q) for key in ('native_core', 'native_shell', 'shoulder', 'intermediate', 'far')}


def scalar_correlation(values):
    """Finite-record correlation, not an equilibrium ESS or convergence claim."""
    x = np.array(values, dtype=float, copy=True)
    x -= x.mean()
    variance = x@x
    if variance < 1e-24:
        return dict(variance=float(variance/len(x)), correlation=None)
    lags = [v for v in (1, 2, 5, 10, 20, 50, 100, 250, 500, 1000) if v < len(x)]
    return dict(variance=float(variance/len(x)), correlation={str(lag): float(x[:-lag]@x[lag:]/variance*len(x)/(len(x)-lag)) for lag in lags},
        apparent_effective_count=effective_count(values),
        interpretation='Sample-mean-centered finite-trajectory diagnostic; initialization and rare-mode bias remain.')


def positive_monotone_iact(rho, maximum_lag):
    """Geyer lag pairs (0,1),(2,3),...; preserve slow positive components."""
    pairs, last = [], float('inf')
    for lag in range(0, min(len(rho)-1, maximum_lag), 2):
        pair = float(rho[lag]+rho[lag+1])
        if pair <= 0:
            break
        pair = min(pair, last)
        pairs.append(pair)
        last = pair
    raw = -1+2*sum(pairs)
    return max(1., raw), raw, 2*len(pairs)-1


def effective_count(values):
    """Initial-positive, monotone-pair correlation sum for scalar/vector records.

    For vectors, sum component autocovariances before normalization. This is a
    trace statistic for this descriptor, never a bound on all physical modes.
    """
    x = np.asarray(values, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    n = len(x)
    x = x-x.mean(axis=0)
    size = 1 << (2*n-1).bit_length()
    power = np.abs(np.fft.rfft(x, n=size, axis=0))**2
    covariance = np.fft.irfft(power.sum(axis=1), n=size)[:n]/n
    if covariance[0] < 1e-24:
        return dict(samples=n, iact_samples=None, apparent_ess=None, reason='Constant observed descriptor; unresolved mixing, not infinite ESS.')
    rho = covariance/covariance[0]
    tau, raw, last_lag = positive_monotone_iact(rho, max(1, n//2))
    return dict(samples=n, iact_samples=tau, unfloored_iact_samples=raw, apparent_ess=n/tau, last_included_lag=last_lag,
        estimator='Biased sample-centered FFT autocovariance; Geyer positive monotone pairs (rho0+rho1),(rho2+rho3),..., tau=-1+2sum; capped at half record length; IACT floor1.',
        scope='Descriptive conditional statistic; valid equilibrium ESS requires stationarity and coverage, neither established by this pilot.')


def contacts(cfg, shape, frames, stride=10, include_atom_indices=False):
    """Exact sphere-pair tests, retaining atom incidence rather than one edge bit."""
    centers = np.asarray([a['center'] for a in shape['atoms']])
    radii = np.asarray([a['radius'] for a in shape['atoms']])
    ft, _, fr = arrays(cfg['fixed_poses'])
    assert len(ft) == 1
    fixed = centers@fr[0].T + ft[0]
    tree = cKDTree(fixed)
    n, rd = len(centers), cfg['depletant_radius']
    cycle_ids = sorted(set(range(0, len(frames), stride)) | {len(frames)-1})
    bits, rows = [], []
    begun = time.monotonic()
    for cycle in cycle_ids:
        t, _, r = arrays([frames[cycle]['pose']])
        moved = centers@r[0].T+t[0]
        candidates = tree.query_ball_point(moved, radii+radii.max()+2*rd, workers=1)
        incidence = np.zeros(2*n, dtype=bool)
        pair_count, min_gap = 0, float('inf')
        for i, js in enumerate(candidates):
            if not js:
                continue
            js = np.asarray(js)
            gaps = np.linalg.norm(fixed[js]-moved[i], axis=1)-radii[i]-radii[js]
            min_gap = min(min_gap, float(gaps.min()))
            js = js[gaps < 2*rd]
            if len(js):
                incidence[i] = True
                incidence[n+js] = True
                pair_count += len(js)
        assert min_gap >= -1e-8, (cycle, min_gap)
        assert bool(pair_count) == frames[cycle]['depletion_contact'], (cycle, pair_count)
        bits.append(incidence)
        rows.append(dict(cycle=cycle, mobile_contact_atoms=int(incidence[:n].sum()),
            fixed_contact_atoms=int(incidence[n:].sum()), overlapping_exclusion_sphere_pairs=pair_count,
            minimum_tested_core_gap_A=min_gap if np.isfinite(min_gap) else None, hard_valid=True))
        if include_atom_indices:
            rows[-1]['mobile_contact_atom_indices'] = np.flatnonzero(incidence[:n]).tolist()
            rows[-1]['fixed_contact_atom_indices'] = np.flatnonzero(incidence[n:]).tolist()
    bits = np.asarray(bits)
    centered = bits.astype(float)-bits.mean(axis=0)
    base = float(np.sum(centered*centered)/len(bits))
    lagrows = []
    for lag in (1, 2, 5, 10, 20, 50):
        if lag >= len(bits):
            continue
        intersection = np.sum(bits[:-lag]&bits[lag:], axis=1)
        union = np.sum(bits[:-lag]|bits[lag:], axis=1)
        jac = np.divide(intersection, union, out=np.ones_like(intersection, dtype=float), where=union>0)
        lagrows.append(dict(lag_cycles=lag*stride, mean_jaccard_distance=float(np.mean(1-jac)),
            centered_incidence_correlation=float(np.mean(np.sum(centered[:-lag]*centered[lag:], axis=1))/base) if base > 0 else None))
    half = len(bits)//2
    return dict(stride_cycles=stride, samples=len(rows), rows=rows, lags=lagrows,
        apparent_effective_count=effective_count(bits),
        last_half_apparent_effective_count=effective_count(bits[half:]),
        mean_incidence=bits.mean(axis=0).tolist(),
        first_half_mean_incidence=bits[:half].mean(axis=0).tolist(),
        last_half_mean_incidence=bits[half:].mean(axis=0).tolist(),
        analysis_wall_seconds=time.monotonic()-begun,
        descriptor='Binary mobile/fixed atom incidence in overlaps of depletant-expanded atom spheres. Exact union contact membership; redundant internal spheres are retained. This is not a crystallographic patch label or an overlap-volume estimator.',
        interpretation='Sample-centered correlation and lagged Jaccard change, with no equilibrium stationarity assumption certified.')


def one(task):
    begun, begun_cpu = time.monotonic(), time.process_time()
    campaign, out, job, region_path = task
    campaign, out = Path(campaign), Path(out)
    manifest, cfg = read(campaign/'manifest.json'), read(job['config'])
    directory = Path(job['directory'])
    summary = read(directory/'summary.json')
    assert summary['complete'] and summary['method'] == 'posterior_involution'
    assert sha(job['config']) == job['config_sha256']
    model = read(campaign/'provenance/model.json')
    audit, density = ChartAudit(model), Density(model)
    frames = [json.loads(line) for line in (directory/'trajectory.jsonl').open()]
    moves = [json.loads(line) for line in (directory/'moves.jsonl').open()]
    cycles = manifest['cycles']
    assert [f['cycle'] for f in frames] == list(range(cycles+1))
    assert len(moves) == cycles*(cfg['local_attempts_per_cycle']+1)
    frameq = registration([f['pose'] for f in frames], cfg['metadata'])
    retainedq = registration([m['retained_pose'] for m in moves], cfg['metadata'])
    candidate_ids = [i for i, m in enumerate(moves) if m['proposed_pose'] is not None]
    candidateq = dict(zip(candidate_ids, registration([moves[i]['proposed_pose'] for i in candidate_ids], cfg['metadata'])))
    learned = [i for i, m in enumerate(moves) if (m.get('proposal') or {}).get('step') is not None]
    anchor = cfg['fixed_poses'][0]
    old_relative = relative_poses([moves[i]['old_pose'] for i in learned], anchor)
    new_relative = relative_poses([moves[i]['proposed_pose'] for i in learned], anchor)
    oldg, _, oldlogs = density.evaluate(old_relative)
    newg, _, newlogs = density.evaluate(new_relative)
    region = read(region_path)
    assert region['shape_sha256'] == model['shape_sha256'] and region['fixed_neighbor'] == anchor
    assert region['activity'] == cfg['reservoir_density'] and region['depletant_radius'] == cfg['depletant_radius']
    region_density = Density(region['gaussian_chart'])
    frame_radius = region_density.evaluate(relative_poses([f['pose'] for f in frames], anchor))[1][:, 0]
    retained_radius = region_density.evaluate(relative_poses([m['retained_pose'] for m in moves], anchor))[1][:, 0]
    logw = np.log(density.weights)
    for index, serial in enumerate(learned):
        p = moves[serial]['proposal']
        assert p['source_law'] == 'posterior'
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
    # Prespecified evenly spaced map checks plus evenly spaced accepted maps.
    selected = set(np.linspace(0, len(learned)-1, min(128, len(learned)), dtype=int))
    accepted = [j for j, serial in enumerate(learned) if moves[serial]['accepted']]
    if accepted:
        selected.update(accepted[j] for j in np.linspace(0, len(accepted)-1, min(128, len(accepted)), dtype=int))
    for index in sorted(selected):
        proposal = copy.deepcopy(moves[learned[index]]['proposal'])
        proposal.pop('full_old_gaussian_log_density', None)
        proposal.pop('full_new_gaussian_log_density', None)
        actual = audit.involution(old_relative[index], proposal, job['correlation'])
        audit.compare_pose(audit.absolute(actual, anchor), moves[learned[index]]['proposed_pose'], 'candidate_lab')

    state, contact = frames[0]['pose'], frames[0]['depletion_contact']
    audit.compare_pose(state, cfg['initial_pose'], 'initial')
    core, strict, qonly = CorePassages(other=2.), CorePassages(other=5.), CorePassages(other=5.)
    deep_trackers = {radius: CorePassages(other=5.) for radius in (3., 8.)}
    for tracker in (core, strict, qonly):
        tracker.observe(float(frameq[0]), 0, -1, 0., 'initial', contact if tracker is not qonly else True)
    for radius, tracker in deep_trackers.items():
        tracker.observe(float(frameq[0]), 0, -1, 0., 'initial', contact and (frameq[0] <= .8 or frame_radius[0] <= radius))
    counters, proposals, crossings = defaultdict(Counter), [], []
    source_counts, target_counts = Counter(), Counter()
    previousq = float(frameq[0])
    native_episodes = []
    native_entry = dict(cycle=0, attempt=-1, q=previousq) if previousq <= 1 else None
    for serial, move in enumerate(moves):
        cycle, attempt = divmod(serial, cfg['local_attempts_per_cycle']+1)
        cycle += 1
        assert move['cycle'] == cycle and move['attempt'] == attempt
        audit.compare_pose(state, move['old_pose'], 'old_pose')
        assert move['depletion_contact_before'] == contact
        branch = move['branch']
        counter = counters[branch]
        counter['attempted'] += 1
        counter['accepted'] += int(move['accepted'])
        counter['hard_valid'] += int(move['hard_valid'])
        counter['capture_valid'] += int(move['capture_valid'])
        candidate, p = move['proposed_pose'], move.get('proposal') or {}
        if p.get('trace') is not None:
            source_counts[p['trace']['source']] += 1
            target_counts[p['trace']['target']] += 1
        if candidate is not None:
            capture = bool(np.linalg.norm(np.asarray(candidate['position'])-cfg['capture_center']) <= cfg['capture_radius'])
            assert move['capture_valid'] == capture
            cq = float(candidateq[serial])
            counter['native_candidates'] += int(cq <= 1.)
            counter['hard_valid_native_candidates'] += int(cq <= 1. and move['hard_valid'] and capture)
            counter['accepted_native_candidates'] += int(cq <= 1. and move['accepted'])
            if move['kind'] == 'global' and ((previousq <= .8 and cq >= 5.) or (previousq >= 5. and cq <= .8)):
                proposals.append(dict(cycle=cycle, old_q=previousq, candidate_q=cq,
                    source='native' if previousq <= .8 else 'far', hard_valid=move['hard_valid'],
                    capture_valid=capture, accepted=move['accepted'],
                    correction=p.get('log_reverse_forward', 0.), gate=(move.get('gate') or {}).get('log_weight'),
                    log_acceptance=move.get('log_acceptance')))
        if move.get('gate') is not None:
            expected = min(0., p.get('log_reverse_forward', 0.)+move['gate']['log_weight'])
            audit.close('acceptance', expected, move['log_acceptance'])
            counter['gate_evaluations'] += 1
        if move['accepted']:
            assert move['capture_valid'] and move['hard_valid'] and candidate is not None
            pt, pr = audit.arrays(state)
            nt, nr = audit.arrays(candidate)
            changed = np.linalg.norm(nt-pt)>1e-10 or np.max(np.abs(nr-pr))>1e-12
            counter['accepted_pose_changes'] += int(changed)
            if p.get('trace') is not None:
                counter['accepted_label_changes'] += int(changed and p['trace']['source'] != p['trace']['target'])
            audit.compare_pose(candidate, move['retained_pose'], 'accepted')
        else:
            audit.compare_pose(state, move['retained_pose'], 'rejected')
        state, contact = move['retained_pose'], move['depletion_contact']
        q = float(retainedq[serial])
        if (q <= 1.) != (previousq <= 1.):
            crossings.append(dict(cycle=cycle, attempt=attempt, branch=branch, old_q=previousq, new_q=q))
            if q <= 1.:
                native_entry = dict(cycle=cycle, attempt=attempt, q=q)
            else:
                native_episodes.append(dict(entry=native_entry, exit=dict(cycle=cycle, attempt=attempt, q=q),
                    duration_cycles=cycle-native_entry['cycle'], right_censored=False))
                native_entry = None
        previousq = q
        for tracker in (core, strict, qonly):
            tracker.observe(q, cycle, attempt, move['sampler_cpu_seconds'], branch, contact if tracker is not qonly else True)
        for radius, tracker in deep_trackers.items():
            tracker.observe(q, cycle, attempt, move['sampler_cpu_seconds'], branch,
                contact and (q <= .8 or retained_radius[serial] <= radius))
        if attempt == cfg['local_attempts_per_cycle']:
            audit.compare_pose(state, frames[cycle]['pose'], 'frame_replay')
            assert contact == frames[cycle]['depletion_contact']
    audit.compare_pose(state, read(directory/'checkpoint.json')['pose'], 'checkpoint')
    if native_entry is not None:
        native_episodes.append(dict(entry=native_entry, exit=None, duration_cycles=cycles-native_entry['cycle'], right_censored=True))
    for tracker in (core, strict, qonly, *deep_trackers.values()):
        for visit in tracker.visits:
            serial = (visit['cycle']-1)*(cfg['local_attempts_per_cycle']+1)+visit['attempt']
            radius = frame_radius[0] if visit['cycle'] == 0 else retained_radius[serial]
            visit['deep_latent_radius'] = float(radius)
            visit['in_deep_R3'] = bool(visit['q'] >= 5. and radius <= 3.)
            visit['in_deep_R8'] = bool(visit['q'] >= 5. and radius <= 8.)
    for index, branches in ((0, ['local']), (1, [b for b in counters if b != 'local'])):
        for key in ('attempted', 'accepted', 'accepted_pose_changes'):
            assert sum(counters[b][key] for b in branches) == summary['counts'][index][key]
    contact_stats = contacts(cfg, read(campaign/'provenance/shape.json'), frames)
    cpu = summary['sampler_cpu_seconds']
    half = cycles//2
    result = dict(id=job['id'], start=job['start'], replicate=job['replicate'], mode=job['mode'],
        correlation=job['correlation'], passed=True, cpu_seconds=cpu, wall_seconds=summary['wall_seconds'],
        cost=summary.get('cost'), counts={key: dict(value) for key, value in counters.items()},
        initial_q=float(frameq[0]), final_q=float(frameq[-1]),
        first_half_occupancy=occupancy(frameq[1:half+1]), last_half_occupancy=occupancy(frameq[half+1:]),
        occupancy=occupancy(frameq[1:]), unbound_fraction=float(np.mean([not f['depletion_contact'] for f in frames[1:]])),
        core=core.result(cpu), strict_core=strict.result(cpu), q_only_strict_core=qonly.result(cpu),
        native_to_deep={str(int(radius)): tracker.result(cpu) for radius, tracker in deep_trackers.items()},
        native_dwell_episodes=native_episodes,
        frozen_deep_region_sha256=sha(region_path),
        frozen_deep_occupancy={str(int(radius)): float(np.mean((frameq[1:] >= 5.) & (frame_radius[1:] <= radius))) for radius in (3., 8.)},
        native_boundary_crossings=crossings, strict_core_candidates=proposals,
        source_label_counts=dict(source_counts), target_label_counts=dict(target_counts),
        q_correlation=scalar_correlation(frameq[1:]), last_half_q_correlation=scalar_correlation(frameq[half+1:]),
        mean_q=float(np.mean(frameq[1:])), last_half_mean_q=float(np.mean(frameq[half+1:])),
        native_indicator_apparent_ess=effective_count(frameq[1:] <= 1.), contact_incidence=contact_stats,
        q_rows=[dict(cycle=i, q=float(q), deep_latent_radius=float(frame_radius[i]), contact=f['depletion_contact'], cpu_seconds=f['sampler_cpu_seconds']) for i, (q, f) in enumerate(zip(frameq, frames))],
        audit=dict(all_moves_replayed=len(moves), all_learned_density_checks=len(learned),
            representative_map_checks=len(selected), representative_map_serials=[learned[i] for i in sorted(selected)],
            checks=dict(audit.checks), maximum_errors=dict(audit.max_errors),
            independent_geometry_frames=contact_stats['samples'],
            scope='Every move replayed; every learned full-mixture density and posterior-label correction checked independently. Chart map checks use a deterministic representative subset. Exact sphere geometry and contact incidence checked every10 cycles.'),
        source_sha256={name: sha(directory/name) for name in ('moves.jsonl', 'trajectory.jsonl', 'summary.json', 'checkpoint.json')},
        config_sha256=sha(job['config']), analyzer_sha256=sha(__file__))
    result['q_only_strict_core']['both_cores_require_depletion_contact'] = False
    assert np.all(frameq >= 0.)
    assert np.max(np.abs(frameq-registration([f['pose'] for f in frames], cfg['metadata']))) < 1e-10
    if (job['start'] == 'deep' and job['replicate'] == 1 and job['mode'] == 'c09'
            and np.any((frameq >= 5.) & (frame_radius > 8.))):
        outside = (frameq >= 5.) & (frame_radius > 8.)
        ids = np.flatnonzero(outside)
        chosen = ids[np.linspace(0, len(ids)-1, min(32, len(ids)), dtype=int)].tolist()
        selected_frames = [frames[i] for i in chosen]
        selected_density = density.evaluate(relative_poses([f['pose'] for f in selected_frames], anchor))[0]
        selected_contacts = contacts(cfg, read(campaign/'provenance/shape.json'), selected_frames, stride=1, include_atom_indices=True)
        ranges = []
        first = None
        for cycle, value in enumerate(np.r_[outside, False]):
            if value and first is None:
                first = cycle
            elif not value and first is not None:
                ranges.append(dict(first_cycle=first, last_cycle=cycle-1, retained_frames=cycle-first))
                first = None
        cohort = dict(source_job=job['id'], selection='At most32 equally spaced indices among all chronological retained frames with original q>=5 and frozen deep latent radius>8. No physical weight selection; duplicate poses retained.',
            source_sha256=result['source_sha256'], model_sha256=manifest['model_sha256'], frozen_region_sha256=sha(region_path),
            outside_frame_count=len(ids), outside_q_range=[float(frameq[ids].min()), float(frameq[ids].max())],
            outside_radius_range=[float(frame_radius[ids].min()), float(frame_radius[ids].max())], dwell_ranges=ranges,
            unique_selected_poses=len({json.dumps(f['pose'], sort_keys=True) for f in selected_frames}),
            final_frame=frames[-1], final_q=float(frameq[-1]), final_latent_radius=float(frame_radius[-1]),
            poses=[dict(cycle=i, q=float(frameq[i]), deep_latent_radius=float(frame_radius[i]),
                gaussian_log_density=float(g), pose=frames[i]['pose'], depletion_contact=frames[i]['depletion_contact'],
                contact_descriptor={key: value for key, value in c.items() if key != 'cycle'})
                for i, g, c in zip(chosen, selected_density, selected_contacts['rows'])])
        write(out/'outside-R8-cohort.json', cohort)
        result['outside_R8_cohort_file'] = str(out/'outside-R8-cohort.json')
    result['analysis_wall_seconds'] = time.monotonic()-begun
    result['analysis_cpu_seconds'] = time.process_time()-begun_cpu
    path = out/'runs'/job['id']
    path.mkdir(parents=True, exist_ok=True)
    write(path/'analysis.json', result)
    print(json.dumps(dict(id=job['id'], passed=True, strict_roundtrips=result['strict_core']['completed_roundtrips'],
        native_fraction=result['occupancy']['native_core']+result['occupancy']['native_shell'], cpu_seconds=cpu)), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('campaign', type=Path)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--region', type=Path, default=ROOT/'runs/smc-normalizer-deep-far/site0/fixed-discovered-region.json')
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    manifest = read(campaign/'manifest.json')
    assert read(campaign/'summary.json')['complete']
    assert all(sha(campaign/'provenance'/name) == digest for name, digest in manifest['input_sha256'].items())
    out = campaign/'assessment'
    out.mkdir(exist_ok=True)
    shutil.copy2(__file__, out/'analyzer.py')
    shutil.copy2(args.region, out/'frozen-deep-region.json')
    tasks = [(str(campaign), str(out), job, str(out/'frozen-deep-region.json')) for job in manifest['jobs']]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(one, tasks))
    totals = []
    for mode in ('c0', 'c09'):
        group = [r for r in results if r['mode'] == mode]
        count = Counter()
        for r in group:
            for key, value in r['counts'].items():
                if key != 'local':
                    count.update(value)
        cpu = sum(r['cpu_seconds'] for r in group)
        contact_ess = sum(r['contact_incidence']['apparent_effective_count']['apparent_ess'] or 0 for r in group)
        totals.append(dict(mode=mode, cpu_seconds=cpu, global_counts=dict(count),
            accepted_global_pose_changes_per_cpu_second=count['accepted_pose_changes']/cpu,
            contact_apparent_ess=contact_ess, contact_apparent_ess_per_cpu_second=contact_ess/cpu,
            strict_roundtrips=sum(r['strict_core']['completed_roundtrips'] for r in group),
            native_to_far=sum(r['strict_core']['native_to_other'] for r in group),
            far_to_native=sum(r['strict_core']['other_to_native'] for r in group)))
    def mean_contact_distance(a, b):
        a, b = np.asarray(a), np.asarray(b)
        return dict(incidence_L1=float(np.sum(np.abs(a-b))),
            weighted_jaccard_distance=float(1-np.minimum(a, b).sum()/np.maximum(a, b).sum()))
    agreement = []
    for mode in ('c0', 'c09'):
        group = [r for r in results if r['mode'] == mode]
        for i, a in enumerate(group):
            for b in group[i+1:]:
                agreement.append(dict(mode=mode, first=a['id'], second=b['id'],
                    same_start_kind=a['start'] == b['start'],
                    last_half_mean_q_difference=a['last_half_mean_q']-b['last_half_mean_q'],
                    **mean_contact_distance(a['contact_incidence']['last_half_mean_incidence'], b['contact_incidence']['last_half_mean_incidence'])))
    within = [dict(id=r['id'], **mean_contact_distance(r['contact_incidence']['first_half_mean_incidence'], r['contact_incidence']['last_half_mean_incidence'])) for r in results]
    write(out/'analysis.json', dict(runs=results, totals=totals, start_agreement=agreement, within_run_half_contact_differences=within, model_sha256=manifest['model_sha256'],
        limitation='Finite, selected-start trajectories. No equilibrium distribution or mixing-time estimate is established by absence of rare returns. Accepted moves per CPU and contact changes are separate algorithmic diagnostics.'))
    lines = ['# Posterior-source docking pilot', '',
        'Eight frozen-model runs compare c=0 and c=0.9, each with 5000 cycles, identical two local attempts per cycle, and a separate 10% uniform branch. The c=0 kernel is a matched independent Gaussian-branch redraw. Starts are selected snapshots, not equilibrium draws. The retained 116-component atlas contains native information.', '',
        '| Start | Rep | c | Native occupancy | Far occupancy | Native→far / far→native | Roundtrips | Global changes | CPU s |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in results:
        occ, core = r['occupancy'], r['strict_core']
        n = sum(v.get('accepted_pose_changes', 0) for k, v in r['counts'].items() if k != 'local')
        lines.append(f"| {r['start']} | {r['replicate']} | {r['correlation']:g} | {occ['native_core']+occ['native_shell']:.4g} | {occ['far']:.4g} | {core['native_to_other']} / {core['other_to_native']} | {core['completed_roundtrips']} | {n} | {r['cpu_seconds']:.2f} |")
    lines += ['', 'All attempts pass retained-state replay and acceptance arithmetic. Full Gaussian mixture densities and posterior source/label corrections are checked independently for every learned proposal; deterministic representative chart-map checks and exact sphere-pair geometry at 501 frames per run also pass.', '',
        'The contact diagnostic tracks atom incidence in overlaps of depletant-expanded spheres. It resolves changing surface contacts beyond a single neighbor-edge bit, but does not identify crystallographic patches. Its sample-centered correlation is descriptive; nonstationarity and unvisited modes remain possible.', '',
        'A faster accepted-pose-change rate is not a measured equilibrium mixing speedup. Rare native occupancy is expected to be small under the provisional normalizer results, so missing returns alone cannot diagnose algorithmic failure or equilibrium.']
    lines += ['', '| c | Apparent contact ESS | Contact ESS / sampler CPU s | Global changes / sampler CPU s |', '|---:|---:|---:|---:|']
    for total in totals:
        lines.append(f"| {0 if total['mode']=='c0' else .9} | {total['contact_apparent_ess']:.2f} | {total['contact_apparent_ess_per_cpu_second']:.4f} | {total['accepted_global_pose_changes_per_cpu_second']:.4f} |")
    lines += ['', 'Contact incidence is recorded every10 cycles; apparent ESS uses a sample-centered positive monotone autocorrelation sum. Values describe sampled contacts only and do not establish equilibrium coverage. Analysis observer cost is excluded from the reported sampler CPU and separately recorded for each run.', '',
        '![Registration trajectories](registration-trajectories.png)', '', '![Frozen deep-region coverage](deep-region-trajectories.png)']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 2, figsize=(11, 10), sharex=True, sharey=True)
    for r in results:
        row = (0 if r['start'] == 'native' else 2)+r['replicate']
        col = 0 if r['mode'] == 'c0' else 1
        ax = axes[row, col]
        ax.plot([v['cycle'] for v in r['q_rows']], [v['q'] for v in r['q_rows']], lw=.7)
        for visit in r['strict_core']['visits']:
            if visit['core'] == 'native' and visit['cycle'] > 0:
                ax.scatter(visit['cycle'], visit['q'], marker='v', color='#c03836', s=25, zorder=4)
                ax.annotate(str(visit['cycle']), (visit['cycle'], visit['q']), xytext=(0, 13), textcoords='offset points', ha='center', fontsize=8, color='#a02828')
        ax.axhspan(0, .8, color='#50a070', alpha=.15)
        ax.axhline(5, color='#888888', ls=':', lw=.7)
        ax.set_title(f"{r['start']} start {r['replicate']}, c={r['correlation']:g}")
        ax.set_ylim(0, 32)
        if col == 0:
            ax.set_ylabel('Registration q')
        if row == 3:
            ax.set_xlabel('MC cycle (2 local + 1 global)')
    fig.suptitle('Matched posterior-source proposals: finite trajectories')
    fig.tight_layout()
    fig.savefig(out/'registration-trajectories.png', dpi=160)
    fig.savefig(out/'registration-trajectories.svg')
    plt.close(fig)
    fig, axes = plt.subplots(4, 2, figsize=(11, 10), sharex=True, sharey=True)
    for r in results:
        row = (0 if r['start'] == 'native' else 2)+r['replicate']
        col = 0 if r['mode'] == 'c0' else 1
        ax = axes[row, col]
        values = [v['deep_latent_radius'] if v['q'] >= 5 else np.nan for v in r['q_rows']]
        ax.plot([v['cycle'] for v in r['q_rows']], values, lw=.7)
        clipped = [v['cycle'] for v in r['q_rows'] if v['q'] >= 5 and v['deep_latent_radius'] > 13]
        if clipped:
            ax.scatter(clipped, [12.7]*len(clipped), color='#a03838', marker='^', s=10)
            ax.text(.98, .93, f'{len(clipped)} frames >13', transform=ax.transAxes, ha='right', fontsize=8, color='#a03838')
        ax.set_ylim(0, 13)
        ax.axhspan(0, 3, color='#50a070', alpha=.15)
        ax.axhline(8, color='#bf7130', ls='--', lw=.8)
        ax.set_title(f"{r['start']} start {r['replicate']}, c={r['correlation']:g}; R8 occupancy={r['frozen_deep_occupancy']['8']:.1%}")
        if col == 0:
            ax.set_ylabel('Frozen deep-chart latent radius')
        if row == 3:
            ax.set_xlabel('MC cycle (2 local + 1 global)')
    fig.suptitle('Deep-region coverage: green R3, dashed R8; q<5 omitted, radius>13 marked')
    fig.tight_layout()
    fig.savefig(out/'deep-region-trajectories.png', dpi=160)
    fig.savefig(out/'deep-region-trajectories.svg')
    plt.close(fig)
    write(out/'plot-provenance.json', dict(plot_source_sha256=sha(__file__),
        analysis_sha256=sha(out/'analysis.json'), replotted_without_recomputing_analysis=False))
    print(json.dumps(dict(complete=True, totals=totals)), flush=True)


if __name__ == '__main__':
    main()
