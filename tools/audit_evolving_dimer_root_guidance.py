#!/usr/bin/env python3
"""Bounded independent count checks and all-attempt two-threshold algebra.

The campaign must freeze ``audit_protocol()`` and this import closure BEFORE
production. This module has no sampling or fitting entry point. ``audit_chain``
audits exactly block 1's collective attempt, including source-outside and cap
self-loops; it never substitutes a successful event. Other events are scalar
replay only. Original cloud thinning/prepared-start audits are reused by binding
their receipts and bytes, not repeated here. A metadata-only outer binder must
authenticate those receipts, the campaign closure, and all 32 terminal journals.

Scalar checks do not independently validate every geometric count, mixture
density, RNG variate or physical Poisson predicate. Independent count checks
use the existing NumPy/SciPy sphere-union reference, never the Rust BVH.
"""
from __future__ import annotations
import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import numpy as np

from analyze_auxiliary_overlap_probe import CountOracle
from analyze_capped_dimer_probe import Checks, DimerGeometry, feasibility, is_feasible
from analyze_evolving_dimer_benchmark import validate_journal
from analyze_factorized_dimer_probe import (
    audit_frame, internal_geometry, internal_ok, root_geometry, root_ok,
    relative_pose, compose_pose, strict_equal,
)


def require(value, message):
    if not value:
        raise ValueError(message)


def audit_protocol():
    return dict(schema='evolving-dimer-root-guidance-audit-protocol-v1', chains=32,
        arm='root_m4', geometric_event=dict(kind='factorized_dimer', block=1,
            after_local_attempts=4, selection='First scheduled collective event, never first candidate.',
            replace_skipped=False),
        geometry_scope='Both source/assembled-endpoint count frames; every raw root/internal hard/contact predicate and every count actually evaluated in these 32 events.',
        maximum_count_queries=2560, maximum_point_membership_tests=41943040,
        maximum_raw_root_trials=1024, maximum_raw_internal_trials=1024,
        maximum_source_frames=32, maximum_endpoint_frames=32,
        maximum_full_fingerprint_evaluations=128,
        maximum_root_hard_predicate_evaluations=1088,
        maximum_internal_predicate_evaluations=1088,
        setup='At most four fixed-spectator DimerGeometry objects and one CountOracle for the unchanged inflated shape; no setup point-membership calls.',
        scalar_dimer_attempts=147456, scalar_local_attempts=589824,
        workers=1, cpu_seconds=1800, wall_seconds=3600, memory_bytes=16*1024**3,
        new_poses=0, new_clouds=0, repeated_cloud_thinning=0,
        cloud_reuse='Reconstruct only the previously audited frozen point arrays from bound raw uniforms/indices; one fixed body cloud used in both frames.',
        thresholds='Independent root_m4/root_threshold and m4/threshold roles; each maximum of four uniform integers refreshed once per eligible event.',
        roles=dict(proposal='m4/proposal', internal_threshold='m4/threshold',
                   root_threshold='root_m4/root_threshold', bath='m4/bath', accept='m4/accept'),
        trust_boundary='Every saved row is checked for replay, threshold support, stopping, query accounting, and one summed MH correction. Only the 32 predeclared events independently reconstruct geometry/counts. No independent replay of RNG, physical Poisson predicates or full mixture densities here.',
        failure_policy='First disagreement is fatal; retain the partial journal/receipt. No replacements, fresh clouds, retries or silent tolerance of integer-count disagreement.')


def _nat(value, label, upper=None):
    require(type(value) is int and value >= 0 and (upper is None or value <= upper), 'Invalid '+label)
    return value


def _log(value):
    if value == '-inf':
        return -math.inf
    require(type(value) in (int, float) and math.isfinite(value), 'Invalid log factor')
    return float(value)


def _close(actual, expected, label):
    actual = _log(actual)
    expected = -math.inf if expected == -math.inf else _log(expected)
    require(actual == expected or (math.isfinite(actual) and math.isfinite(expected)
        and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-8)), label)


def _frame_count(frame, name, expected, point_count):
    counts = frame[name]
    require(set(counts) == {'relative_count', 'recovered_count', 'world_count', 'reconstructed_world_count'}, 'Count frame fields differ')
    require(all(_nat(x, name+' frame count', point_count) == expected for x in counts.values()), 'Saved count frame disagrees')


def _hard_source(value):
    return not value['internal_core_overlap'] and not any(value['spectator_core_collisions']) and all(value['wall_valid'])


def _stage(records, stage, threshold, point_count):
    require(isinstance(records, list) and len(records) <= 32, 'Stage exceeds fixed cap')
    passed = False
    queries = 0
    for index, record in enumerate(records, 1):
        require(not passed and record['index'] == index, 'Hidden retry or wrong stage index')
        f = record['feasibility']
        good = root_ok(f) if stage == 'root' else internal_ok(f)
        require(type(good) is bool, 'Non-Boolean stage predicate')
        if good:
            queries += 1
            k = _nat(record['guidance_count'], 'raw count', point_count)
            passed = k >= threshold
        else:
            require('guidance_count' not in record, 'Skipped count is not a zero observation')
    require(passed or len(records) == 32, 'Premature cap exhaustion')
    return passed, queries


def validate_scalar_record(row, config, point_count):
    """Every rejected state is retained; geometry/count values are trusted here."""
    require(row['kind'] == 'factorized_dimer' and type(row['accepted']) is bool, 'Wrong event kind')
    p = row['proposal']
    require(p['caps'] == dict(root=32, internal=32, joint=1) and p['order'] == 'root_first', 'Changed cap law')
    require(p['old'] == row['old'] and p['members'] == row['members'] and p['anchor_label'] == row['anchor'], 'Outcome identity differs')
    require(_hard_source(p['source_feasibility']), 'Saved source is hard invalid')
    guides = {'root': p['root_guidance'], 'internal': p['guidance']}
    eligible = p['source_feasibility']['internal_exclusion_contact']
    require(type(eligible) is bool, 'Non-Boolean source contact')
    for stage, d in guides.items():
        require(d['m'] == 4 and d['point_count'] == point_count, 'Changed threshold law/cloud')
        old = _nat(d['old_count'], 'old count', point_count)
        _frame_count(p['source_frame'], 'root_guidance' if stage == 'root' else 'guidance', old, point_count)
        if eligible:
            draws = d['integer_draws']
            require(len(draws) == 4 and all(_nat(x, 'integer threshold draw', old) >= 0 for x in draws), 'Threshold trace differs')
            require(d['threshold'] == max(draws), 'Threshold maximum differs')
        else:
            require(d['integer_draws'] == [] and d['threshold'] is None, 'Ineligible source consumed a threshold')
    queries = dict(root=4, internal=4)
    assembled = False
    final = None
    if not eligible:
        require(p['status'] == 'source_outside_domain' and p['attempts'] == [] and p['candidate'] is None, 'Ineligible source retried')
    else:
        require(len(p['attempts']) == 1 and p['attempts'][0]['index'] == 1, 'Changed joint cap')
        a = p['attempts'][0]
        rp, rq = _stage(a['root_draws'], 'root', guides['root']['threshold'], point_count)
        queries['root'] += rq
        ip = False
        if rp:
            ip, iq = _stage(a['internal_draws'], 'internal', guides['internal']['threshold'], point_count)
            queries['internal'] += iq
        else:
            require(a['internal_draws'] == [], 'Internal stage ran after root exhaustion')
        assembled = rp and ip
        if assembled:
            require(a['proposed'] is not None and a['frame'] is not None and a['final_feasibility'] is not None, 'Missing assembled endpoint')
            for stage, d in guides.items():
                count = a[stage+'_draws'][-1]['guidance_count']
                _frame_count(a['frame'], 'root_guidance' if stage == 'root' else 'guidance', count, point_count)
                queries[stage] += 4
                require(count >= d['threshold'], 'Endpoint violates fixed threshold')
            final = 'candidate' if is_feasible(a['final_feasibility']) else 'final_rejected'
        else:
            require(all(a[k] is None for k in ('proposed', 'frame', 'final_feasibility')), 'Endpoint exists after exhausted stage')
            final = 'internal_cap_exhausted' if rp else 'root_cap_exhausted'
        require(a['status'] == final and p['status'] == ('candidate' if final == 'candidate' else 'cap_exhausted'), 'Wrong terminal stage/status')
    candidate = p['candidate']
    require((candidate is not None) == (final == 'candidate'), 'Candidate status differs')
    auxiliary = {}
    for stage, d in guides.items():
        require(_nat(d['count_queries'], 'query count') == queries[stage]
                and _nat(d['point_tests'], 'point tests') == queries[stage]*point_count, 'Count-query accounting differs')
        if candidate is None:
            require(d['new_count'] is None and d['aux_log_correction'] is None, 'Null event has destination correction')
        else:
            k = p['attempts'][0][stage+'_draws'][-1]['guidance_count']
            require(_nat(d['new_count'], 'new count', point_count) == k, 'Final count differs')
            value = 4*(math.log1p(d['old_count'])-math.log1p(k))
            _close(d['aux_log_correction'], value, 'Threshold correction differs')
            auxiliary[stage] = value
    if candidate is None:
        require(row['status'] == 'proposal_self_loop' and not row['accepted'], 'Null event is accepted')
        require(not any(k in row for k in ('bath', 'log_u', 'complete_log_correction', 'log_acceptance_ratio', 'proposed')), 'Null event consumed physical acceptance')
        return dict(status=p['status'], stage_status=final, count_queries=queries, assembled=assembled)
    require(row['status'] == 'completed', 'Missing completed physical decision')
    require(row['proposed'] == [candidate['root'], candidate['child']] == p['attempts'][0]['proposed'], 'Candidate endpoint differs')
    f = _log(candidate['diagnostics']['log_reverse_forward'])
    correction = (f+auxiliary['internal'])+auxiliary['root']
    _close(row['complete_log_correction'], correction, 'Complete correction must include each auxiliary once')
    bath = row['bath']['aggregate']
    ratio = config['physical']['lambda_ratio']
    require(config['physical']['activity'] == .0275 and ratio == 64., 'Changed physical bath')
    b = math.log1p(1/ratio)*(_nat(bath['gained'], 'gained')-_nat(bath['lost'], 'lost'))
    _close(bath['log_weight'], b, 'Physical bath count correction differs')
    legs = row['bath']['legs']
    require(len(legs) == 2, 'Physical two-leg path differs')
    for leg in legs:
        gained, lost = _nat(leg['gained'], 'leg gained'), _nat(leg['lost'], 'leg lost')
        require(_nat(leg['retained_points'], 'leg retained') == gained+lost
                and _nat(leg['raw_points'], 'leg raw') >= gained+lost, 'Leg count accounting differs')
        _close(leg['log_weight'], math.log1p(1/ratio)*(gained-lost), 'Leg log correction differs')
    for key in ('gained', 'lost', 'raw_points', 'retained_points', 'created_cells', 'retained_cells'):
        require(_nat(bath[key], 'aggregate '+key) == sum(_nat(leg[key], 'leg '+key) for leg in legs), 'Two-leg aggregation differs')
    _close(bath['log_weight'], sum(leg['log_weight'] for leg in legs), 'Two-leg log sum differs')
    _close(row['log_acceptance_ratio'], correction+b, 'Complete MH sum differs')
    u = row['log_u']
    require(type(u) in (int, float) and math.isfinite(u) and u < 0
            and row['accepted'] == (u < min(0., correction+b)), 'Physical decision differs')
    return dict(status=p['status'], stage_status=final, count_queries=queries, assembled=assembled,
        full_F_log=f, root_aux_log=auxiliary['root'], internal_aux_log=auxiliary['internal'], bath_log=b,
        acceptance_probability=math.exp(min(0., correction+b)))


def retained_points(bank):
    """Only reconstruct already audited arrays; never repeat sphere thinning."""
    def bound(item):
        data = Path(item['path']).read_bytes()
        require(hashlib.sha256(data).hexdigest() == item['sha256'], 'Changed bound cloud bytes')
        return data
    raw = bound(bank['raw']); meta = json.loads(bound(bank['metadata']))
    require(meta['raw_count'] == 16384 and len(raw) == 16384*24, 'Cloud allocation differs')
    uniform = np.frombuffer(raw, dtype='<f8').reshape((-1, 3))
    require(np.isfinite(uniform).all() and np.all((uniform >= 0) & (uniform < 1)), 'Invalid cloud variates')
    indices = meta['kept_indices']
    require(all(type(i) is int and 0 <= i < 16384 for i in indices)
            and indices == sorted(set(indices)), 'Invalid cloud thinning indices')
    low, high = np.asarray(meta['low']), np.asarray(meta['high'])
    require(low.shape == high.shape == (3,) and np.isfinite(low).all() and np.isfinite(high).all()
            and np.all(low < high), 'Invalid cloud bounding box')
    return low+(high-low)*uniform[indices]


def audit_first_record(row, geometry, counter, points, case, checks):
    """Fixed first-event geometry allocation. No fitting or extra point draws."""
    require(row['kind'] == 'factorized_dimer' and row['block'] == 1, 'Unallocated geometry event')
    p = row['proposal']; old = row['old']; anchor = geometry.state[case['anchor']]
    require(row['members'] == [case['root'], case['child']] and row['anchor'] == case['anchor'], 'Geometry context differs')
    counts = Counter()
    def frame(members, raw_edges, value, tag):
        full = feasibility(geometry, case, members, geometry.fingerprint(row['members'], members))
        expected = p['source_feasibility'] if tag == 'source' else p['attempts'][0]['final_feasibility']
        strict_equal(expected, full, checks, 'independent full predicates', tag)
        audit_frame(geometry, case, anchor, members, full, value, checks, tag)
        counts[tag+'_frames'] += 1
        for index, name in ((0, 'root_guidance'), (1, 'guidance')):
            world = [anchor, members[0]] if index == 0 else members
            rebuilt = [anchor, value['reconstructed_members'][0]] if index == 0 else value['reconstructed_members']
            expected_counts = dict(relative_count=counter.relative(points, raw_edges[index]),
                recovered_count=counter.relative(points, value['recovered_edges'][index]),
                world_count=counter.world(points, world),
                reconstructed_world_count=counter.world(points, rebuilt))
            counts['count_queries'] += 4
            strict_equal(value[name], expected_counts, checks, 'independent '+name, tag)
            require(len(set(expected_counts.values())) == 1, 'Independent frame count disagreement')
    edges = [relative_pose(anchor, old[0]), relative_pose(old[0], old[1])]
    frame(old, edges, p['source_frame'], 'source')
    for a in p['attempts']:
        for stage in ('root', 'internal'):
            for trial in a[stage+'_draws']:
                edge = trial['draw']['proposed_relative_pose']
                if stage == 'root':
                    checks.pose(trial['world_pose'], compose_pose(anchor, edge), 'root recomposition')
                    expected = root_geometry(geometry, case, trial['world_pose'])
                    good = root_ok(expected)
                else:
                    expected = internal_geometry(geometry, edge)
                    good = internal_ok(expected)
                counts[stage+'_raw_trials'] += 1
                strict_equal(trial['feasibility'], expected, checks, 'independent '+stage+' predicates', trial['index'])
                if good:
                    k = counter.relative(points, edge); counts['count_queries'] += 1
                    strict_equal(trial.get('guidance_count'), k, checks, 'independent raw '+stage+' count', trial['index'])
                else:
                    require('guidance_count' not in trial, 'Count present on skipped hard/contact branch')
        if a['proposed'] is not None:
            raw = [a['root_draws'][-1]['draw']['proposed_relative_pose'],
                   a['internal_draws'][-1]['draw']['proposed_relative_pose']]
            frame(a['proposed'], raw, a['frame'], 'endpoint')
    require(counts['count_queries'] <= 80 and counts['root_raw_trials'] <= 32 and counts['internal_raw_trials'] <= 32, 'First-event query budget exceeded')
    require(not checks.failures, 'Independent geometry/pose checks failed')
    counts['point_membership_tests'] = counts['count_queries']*len(points)
    return dict(counts)


def validate_initial_contract(row, context, bank):
    require(row['cloud'] == bank, 'Initial cloud binding differs')
    expected = dict(schema='evolving-dimer-root-guidance-v1', cloud=bank,
        cloud_reuse='identical frozen body-frame points; no additional cloud draws',
        root=dict(m=4, point_frame='fixed_anchor_body', anchor_label=context['anchor'],
                  mobile_label=context['root'], threshold_rng_role='root_m4/root_threshold'),
        internal=dict(m=4, point_frame='mobile_root_body', root_label=context['root'],
                      child_label=context['child'], threshold_rng_role='m4/threshold'),
        matched_control_arm='m4', proposal_rng_role='m4/proposal', bath_rng_role='m4/bath',
        accept_rng_role='m4/accept',
        local_rng_roles='unchanged shared local/{attempt}/{proposal,bath,accept}',
        physical_decisions_per_candidate=1)
    require(row['guidance_contract'] == expected, 'Frozen two-threshold/frame/RNG contract differs')


def audit_chain(config, job, initial, terminal, rows, *, bank, counter, geometry, points, checks):
    """Stream every attempt through the existing replay, auditing one fixed event."""
    require(job['arm'] == 'root_m4', 'Wrong audit arm')
    require(config['allocation']['warmup_blocks'] == 512 and config['allocation']['production_blocks'] == 4096, 'Changed block allocation')
    counts = Counter(); first_geometry = None
    def checked_rows():
        nonlocal first_geometry
        for row in rows:
            if row['kind'] == 'initial':
                validate_initial_contract(row, config['contexts'][job['context_index']], bank)
            elif row['kind'] == 'factorized_dimer':
                result = validate_scalar_record(row, config, len(points))
                counts['scalar_dimer_attempts'] += 1
                counts[result['stage_status'] or result['status']] += 1
                if row['block'] == 1:
                    require(first_geometry is None, 'Duplicate first collective event')
                    first_geometry = audit_first_record(row, geometry, counter, points,
                        config['contexts'][job['context_index']], checks)
            elif row['kind'] == 'local':
                counts['scalar_local_attempts'] += 1
            yield row
    # No ContactObserver is created: validate_journal is a scalar state replay.
    validate_journal(checked_rows(), config, job, initial, terminal)
    require(first_geometry is not None and counts['scalar_dimer_attempts'] == 4608
            and counts['scalar_local_attempts'] == 18432, 'Missing allocated attempts')
    return dict(job=job, scalar_counts=dict(counts), first_event_geometry=first_geometry,
        independently_audited_geometry_events=1, other_events_scalar_only=4607,
        independently_reconstructed_physical_bath_predicates=0)
