#!/usr/bin/env python3
"""Audit guided reset decisions and reuse the authenticated old physical baseline.

All guided outers, including nulls, remain in the denominator. Full F, the
auxiliary target ratio and the bath count factor enter one MH decision exactly
once. This saved-count audit draws no proposals, clouds or random variates.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import time
import analyze_factorized_dimer_physical as prior

BASELINE_RECEIPT = '89fef400daa70578e13201334fe9862f5cfee9400d3ff18b6814640f1abb1ae9'
BASELINE_ANALYSIS = '9b7fc8916c8f3e32a15dc65c2b9f025488e39b7b4de44179b9bf069a82999c8b'
REFERENCE_SHA = '0bde358181834b1848b2b6dbebe84d1b7747bb9faeb6d7a0ac5ce79b3b018237'
LIMITS = dict(raw_per_leg=20_000_000, raw_per_outer=40_000_000, raw_campaign=2_000_000_000,
              retained_per_leg=20_000_000, retained_per_outer=40_000_000,
              retained_campaign=2_000_000_000, cpu_seconds=1200.)
CACHED_FIELDS = ('atlas', 'atlas_index', 'case', 'case_index', 'attempt', 'method', 'm',
                 'old', 'anchor_pose', 'proposal_cpu_seconds', 'contact_diagnostic_cpu_seconds',
                 'raw_edge_draws', 'cloud_construction_cpu_seconds', 'guidance_setup_cpu_seconds',
                 'standalone_proposal_cpu_seconds', 'complete_log_correction')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def bound(record):
    require(sha(record['path']) == record['sha256'], 'Changed bound input: ' + record['path'])
    return Path(record['path'])


def close(a, b, name):
    require(math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, rel_tol=2e-12, abs_tol=2e-10), name)


def log_value(value):
    if value == '-inf':
        return -math.inf
    require(type(value) in (int, float) and math.isfinite(value), 'Invalid log density')
    return value


def nonnegative(value, name):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0, name)
    return value


def integer(value, name):
    require(type(value) is int and value >= 0, name)
    return value


def fingerprint(value, name):
    require(isinstance(value, list) and len(value) == 4 and
            all(type(v) is int and 0 <= v < 2**64 for v in value), name)


def seed(plan, cached, role):
    text = (f"auxiliary-overlap-physical-v1/{plan['master_seed']}/{plan['candidate_ledger']['sha256']}/"
            f"{cached['atlas_index']}/{cached['case_index']}/{cached['attempt']}/{cached['method']}/{role}")
    return int(hashlib.sha256(text.encode()).hexdigest()[:16], 16)


COUNT_FIELDS = ('gained', 'lost', 'raw_points', 'retained_points', 'retained_cells', 'created_cells')


def audit_gate(gate, plan):
    # Reuse the independently tested, unchanged count-factor checker. Its file
    # is part of the frozen executing audit closure checked below.
    prior.audit_gate(gate, plan)


def auxiliary_terms(cached):
    """Check the source conditional, support and the two separate corrections."""
    m = cached['m']
    if type(m) is int and m == 0 and cached['method']=='unguided':
        require(cached['guidance'] is None, 'Unguided arm has auxiliary trace')
        c=cached['candidate']
        if c is None:
            require(cached['complete_log_correction'] is None, 'Unguided null has correction')
            return None
        d=c['diagnostics'];full=log_value(d['log_reverse_forward'])
        old_f,new_f=log_value(d['full_old_log_density']),log_value(d['full_new_log_density'])
        require(math.isfinite(new_f) and full==old_f-new_f, 'Wrong unguided full density')
        require(d['selection_log_reverse_forward']==d['log_tree_coordinate_jacobian']==0, 'Unguided selection/Jacobian')
        require(log_value(cached['complete_log_correction'])==full, 'Wrong unguided complete correction')
        return dict(full=full,auxiliary=0.,complete=full)
    require(type(m) is int and m in (1, 4) and cached['method'] == f'm{m}', 'Changed guided arm')
    d = cached['guidance']
    require(d['m'] == m, 'Guidance multiplicity differs')
    size = integer(d['point_count'], 'Invalid cloud size')
    old = integer(d['old_count'], 'Invalid source count')
    require(old <= size, 'Source count exceeds cloud')
    threshold = integer(d['threshold'], 'Invalid threshold')
    draws = d['integer_draws']
    require(threshold <= old and len(draws) == m and
            all(type(v) is int and 0 <= v <= old for v in draws) and max(draws) == threshold,
            'Invalid saved threshold support')
    candidate = cached['candidate']
    if candidate is None:
        require(d['new_count'] is None and d['aux_log_correction'] is None and
                cached['complete_log_correction'] is None, 'Proposal null acquired a correction')
        return None
    new = integer(d['new_count'], 'Invalid endpoint count')
    require(new <= size and threshold <= new,
            'Invalid saved threshold support')
    auxiliary = m*(math.log1p(old)-math.log1p(new))
    close(d['aux_log_correction'], auxiliary, 'Wrong auxiliary target ratio')
    diagnostics = candidate['diagnostics']
    full = log_value(diagnostics['log_reverse_forward'])
    old_f = log_value(diagnostics['full_old_log_density'])
    new_f = log_value(diagnostics['full_new_log_density'])
    require(math.isfinite(new_f) and full == old_f-new_f, 'Wrong full-density correction')
    require(diagnostics['selection_log_reverse_forward'] == diagnostics['log_tree_coordinate_jacobian'] == 0,
            'Unaccounted selection/Jacobian')
    complete = log_value(cached['complete_log_correction'])
    if math.isfinite(full):
        close(complete, full+auxiliary, 'Saved complete correction omits or duplicates auxiliary term')
    else:
        require(complete == full == -math.inf, 'Invalid infinite complete correction')
    return dict(full=full, auxiliary=auxiliary, complete=complete)


def audit_decision(row, cached, state, source_sha, plan):
    """Small pure checker used by adversarial tests and the full ledger audit."""
    require(row['cached'] == cached and row['index'] == cached['index'], 'Cache/index mismatch')
    require(row['source_state_sha256'] == source_sha, 'Wrong reset-state hash')
    members = [cached['case']['root'], cached['case']['child']]
    anchor = cached['case']['anchor']
    require(len(set(members + [anchor])) == 3 and all(0 <= i < len(state) for i in members + [anchor]), 'Invalid labels')
    old = [state[i] for i in members]
    require(cached['old'] == row['source_selected'] == old and cached['anchor_pose'] == state[anchor], 'Source not reset')
    for role in ('gate', 'mh'):
        require(row[role + '_seed'] == seed(plan, cached, role), 'Wrong independent seed role')
    log_u = row['log_uniform']
    require(type(log_u) in (int, float) and math.isfinite(log_u) and log_u < 0, 'Invalid MH uniform')
    fingerprint(row['mh_rng_after_fingerprint'], 'Invalid MH RNG fingerprint')
    require(type(row['accepted']) is bool and row['gate_failure'] is None, 'Failed gate cannot be a decision')
    nonnegative(row['physical_replay_cpu_seconds'], 'Replay CPU')
    for key in ('proposal_cpu_seconds', 'cloud_construction_cpu_seconds', 'guidance_setup_cpu_seconds',
                'standalone_proposal_cpu_seconds'):
        nonnegative(cached[key], 'Saved '+key)
    close(cached['standalone_proposal_cpu_seconds'], cached['proposal_cpu_seconds'] if cached['m']==0 else sum(cached[key] for key in
          ('proposal_cpu_seconds', 'cloud_construction_cpu_seconds', 'guidance_setup_cpu_seconds')),
          'Incomplete standalone proposal CPU')
    nonnegative(cached['contact_diagnostic_cpu_seconds'], 'Saved contact CPU')
    terms = auxiliary_terms(cached)
    candidate = cached['candidate']
    expected_state = list(state)
    counts = dict(raw_points=0, retained_points=0, gained=0, lost=0)
    if candidate is None:
        require(cached['proposal_status'] in ('cap_exhausted', 'source_outside_domain', 'source_outside_contact'), 'Unknown proposal null')
        require(row['status'] == 'proposal_null' and row['gate'] is None and not row['accepted'] and row['log_ratio'] is None,
                'Proposal null acquired a physical decision')
        require(row['proposed_selected'] is None and all(row[key] is None for key in
                ('q_correction', 'full_f_correction', 'auxiliary_correction')) and
                all(key not in row for key in ('gate_cpu_seconds', 'gate_rng_after_fingerprint')),
                'Proposal null drew a gate')
        acceptance = 0.
    else:
        require(cached['proposal_status'] == 'candidate' and row['status'] == 'physical_decision', 'Incomplete/fatal candidate')
        proposed = [candidate['root'], candidate['child']]
        require(row['proposed_selected'] == proposed, 'Changed proposed pose')
        for key in ('source_feasibility', 'endpoint_feasibility'):
            f = row[key]
            require(f == dict(internal_core_overlap=False, spectator_core_collisions=[[], []],
                              wall_valid=[True, True], internal_exclusion_contact=True), 'Invalid accepted domain')
        correction = terms['complete']
        require(log_value(row['full_f_correction']) == terms['full'], 'Changed full-F correction')
        close(row['auxiliary_correction'], terms['auxiliary'], 'Changed auxiliary correction')
        require(log_value(row['q_correction']) == correction, 'Omitted or duplicated auxiliary correction')
        gate = row['gate']
        require(gate['order'] in ('first_then_second', 'second_then_first'), 'Unknown path order')
        indices = [0, 1] if gate['order'] == 'first_then_second' else [1, 0]
        require(gate['ordered_members'] == [members[i] for i in indices], 'Wrong member order')
        intermediate = list(old)
        intermediate[indices[0]] = proposed[indices[0]]
        require(gate['intermediate_selected'] == intermediate, 'Wrong copied intermediate')
        require(len(gate['legs']) == 2, 'Path must contain both legs')
        for leg in gate['legs']:
            audit_gate(leg, plan)
        aggregate = gate['aggregate']
        for key in COUNT_FIELDS:
            require(aggregate[key] == sum(leg[key] for leg in gate['legs']), 'Wrong aggregate ' + key)
        for key in ('log_weight', 'envelope_volume'):
            close(aggregate[key], sum(leg[key] for leg in gate['legs']), 'Wrong aggregate ' + key)
        for key in ('raw', 'retained'):
            require(aggregate[key + '_points'] <= plan['limits'][key + '_per_outer'], 'Outer cap exceeded')
        expected_log_ratio = correction + aggregate['log_weight']
        observed = log_value(row['log_ratio'])
        if math.isfinite(expected_log_ratio):
            close(observed, expected_log_ratio, 'Wrong MH log ratio')
        else:
            require(observed == expected_log_ratio == -math.inf, 'Invalid infinite MH factor')
        require(row['accepted'] == (log_u < min(0., expected_log_ratio)), 'Wrong MH decision')
        acceptance = math.exp(min(0., expected_log_ratio))
        for key in counts:
            counts[key] = aggregate[key]
        fingerprint(row['gate_rng_after_fingerprint'], 'Invalid gate RNG fingerprint')
        nonnegative(row['gate_cpu_seconds'], 'Gate CPU')
        require(row['physical_replay_cpu_seconds'] + 1e-9 >= row['gate_cpu_seconds'], 'Gate excluded from replay CPU')
        if row['accepted']:
            for i, pose in zip(members, proposed):
                expected_state[i] = pose
    require(row['retained_state'] == expected_state, 'Changed spectator or rejected state')
    require(row['retained_selected'] == [expected_state[i] for i in members], 'Wrong retained selected state')
    return dict(**counts, acceptance_probability=acceptance)


def distribution(values):
    finite = sorted(v for v in values if math.isfinite(v))
    def quantile(p):
        if not finite:
            return None
        pos = p*(len(finite)-1)
        lo = int(pos)
        return finite[lo] + (pos-lo)*(finite[min(lo+1,len(finite)-1)]-finite[lo])
    return dict(count=len(values), negative_infinity=sum(v == -math.inf for v in values),
                quantiles={name:quantile(p) for name,p in [('min',0),('p05',.05),('median',.5),('p95',.95),('max',1)]})


def summarize(rows):
    result = dict(outer_attempts=len(rows), candidates=sum(r['candidate'] for r in rows),
                  accepted=sum(r['accepted'] for r in rows),
                  accepted_with_external_contacts=sum(r['accepted'] and r['external_contacts'] > 0 for r in rows),
                  accepted_changed_contact_fingerprints=sum(r['accepted'] and r['contact_changed'] for r in rows),
                  accepted_gained_contact_edges=sum(r['gained_contact_edges'] for r in rows if r['accepted']),
                  accepted_lost_contact_edges=sum(r['lost_contact_edges'] for r in rows if r['accepted']))
    for key in ('acceptance_probability', 'raw_points', 'retained_points', 'gained', 'lost',
                'saved_proposal_cpu_seconds', 'gate_cpu_seconds', 'physical_replay_cpu_seconds'):
        result[key if key != 'acceptance_probability' else 'sum_conditional_acceptance_probability'] = sum(r[key] for r in rows)
    result['replay_plus_saved_proposal_cpu_seconds'] = result['saved_proposal_cpu_seconds'] + result['physical_replay_cpu_seconds']
    cpu = result['replay_plus_saved_proposal_cpu_seconds']
    result['accepted_per_replay_plus_proposal_cpu_second'] = result['accepted'] / cpu if cpu else None
    for key in ('log_proposal_correction','log_depletion_factor','log_ratio'):
        result[key] = distribution([log_value(r[key]) for r in rows if r['candidate']])
    for key in ('log_full_f_correction', 'log_auxiliary_correction'):
        result[key] = distribution([log_value(r[key]) for r in rows if r['candidate']])
    return result


def validate_contract(plan, protocol):
    width=plan['schema']=='fft-width-physical-reset-v1'
    require((width and protocol['schema']=='fft-width-physical-reset-protocol-v1') or
            (plan['schema']=='auxiliary-overlap-physical-reset-v1' and protocol['schema']=='auxiliary-overlap-physical-reset-protocol-v1'), 'Unknown guided physical protocol')
    require(plan['master_seed'] == protocol['master_seed'] == (6100300501 if width else 6100300301), 'Changed bath seed domain')
    require(plan['total_outer'] == 1536 and type(plan['total_candidates']) is int and
            0 <= plan['total_candidates'] <= 1536, 'Changed allocation')
    require(plan['activity'] == .0275 and plan['depletant_radius'] == 1.4 and
            plan['lambda'] == 64*plan['activity'] == 1.76, 'Changed diagnostic physical model')
    require(plan['limits'] == protocol['limits'] == LIMITS, 'Changed physical budget')
    require(plan['envelope'] == protocol['envelope'] == dict(max_cells=255, max_depth=8, min_width=0.),
            'Changed envelope')
    require(plan['reference_config']['sha256'] == REFERENCE_SHA, 'Changed frozen source context')
    for key in ('candidate_ledger', 'passive', 'baseline', 'baseline_cache'):
        require(plan[key] == protocol[key], 'Plan/protocol differs: '+key)
    allocation = protocol['allocation']
    require(allocation['total_outer'] == 1536 and allocation['candidates'] == plan['total_candidates'] and
            allocation['failed_proposals'] == 1536-plan['total_candidates'] and
            allocation['atlases'] == 3 and allocation['contexts'] == 8 and
            allocation['attempts_per_context_and_method'] == 32 and
            allocation['methods'] == (['unguided','m4'] if width else ['m1', 'm4']) and allocation['extension'] is False,
            'Changed fixed physical allocation')
    require(protocol['reused_baseline_attempts'] == (0 if width else 768) and protocol['new_baseline_baths'] == 0 and
            protocol['no_new_proposal_draws'] is True and protocol['native_classifier'] is False and
            protocol['outcome_filtering'] is False, 'Changed reuse/reset scope')
    if width:require(plan['baseline']=={} and plan['baseline_cache'] is None, 'Width replay included baseline redraw')


def baseline_subset(raw_bytes):
    """Return the exact old factorized physical lines; no new decision is made."""
    selected = []
    keys = []
    for line in raw_bytes.splitlines(keepends=True):
        row = json.loads(line)
        cached = row['cached']
        if cached['method'] != 'factorized':
            continue
        require(row['status'] in ('proposal_null', 'physical_decision') and row['gate_failure'] is None,
                'Failed historical baseline decision')
        keys.append((cached['atlas_index'], cached['case_index'], cached['attempt']))
        selected.append(line)
    require(keys == [(a, c, t) for a in range(3) for c in range(8) for t in range(32)],
            'Missing/reordered/repeated historical baseline slot')
    return b''.join(selected)


def audit_baseline(plan, checked, state, passive_baseline_record):
    records = plan['baseline']
    require(set(records) == {'completed-review.json', 'analysis.json', 'execution/attempts.jsonl',
                            'execution/terminal.json', 'execution/source-state.json'}, 'Incomplete baseline binding')
    require(records['completed-review.json']['sha256'] == BASELINE_RECEIPT and
            records['analysis.json']['sha256'] == BASELINE_ANALYSIS, 'Unreviewed physical baseline')
    files = {name: checked(record) for name, record in records.items()}
    receipt = read(files['completed-review.json'])
    analysis = read(files['analysis.json'])
    require(receipt['schema'] == 'factorized-dimer-physical-completed-review-v1' and
            receipt['complete'] is True and receipt['passed'] is True and
            receipt['physical_exit_code'] == receipt['audit_exit_code'] == 0, 'Historical baseline failed')
    require(analysis['schema'] == 'factorized-dimer-physical-independent-audit-v1' and
            analysis['complete'] is True and analysis['passed'] is True and not analysis['failures'],
            'Historical physical audit failed')
    for name, record in records.items():
        if name != 'completed-review.json':
            require(receipt['output_hashes'][name] == record['sha256'], 'Receipt does not bind '+name)
        if name.startswith('execution/'):
            require(analysis['input_hashes'][name.removeprefix('execution/')] == record['sha256'],
                    'Physical audit does not bind '+name)
    require(read(files['execution/source-state.json']) == state, 'Baseline has different reset state')
    terminal = read(files['execution/terminal.json'])
    require(terminal['summary']['complete'] is True and
            terminal['attempts_sha256'] == records['execution/attempts.jsonl']['sha256'], 'Historical terminal mismatch')
    selected = baseline_subset(files['execution/attempts.jsonl'].read_bytes())
    require(checked(plan['baseline_cache']).read_bytes() == selected, 'Baseline cache is not exact historical subset')
    passive_lines = checked(passive_baseline_record).read_text().splitlines()
    require(len(passive_lines) == 768, 'Incomplete paired passive baseline')
    audited = {row['index']: row for row in analysis['rows'] if row['method'] == 'factorized'}
    require(len(audited) == 768, 'Incomplete historical physical baseline analysis')
    results = []
    for line, passive_line in zip(selected.splitlines(), passive_lines):
        raw = json.loads(line); cached = raw['cached']; passive_row = json.loads(passive_line)
        require(cached['passive_row_sha256'] == hashlib.sha256(passive_line.encode()).hexdigest() and
                cached['candidate'] == passive_row['outcome']['candidate'] and cached['old'] == passive_row['old'],
                'Historical physical baseline uses different passive proposals')
        result = dict(audited[raw['index']])
        require(result['accepted'] == raw['accepted'] and result['candidate'] == (cached['candidate'] is not None) and
                all(result[key] == cached[key] for key in ('atlas', 'method', 'attempt')) and
                result['case'] == cached['case']['name'], 'Historical audited decision mismatch')
        result.update(method='baseline', historical_method='factorized', reused=True,
                      log_full_f_correction=result['log_proposal_correction'],
                      log_auxiliary_correction=0. if result['candidate'] else None)
        results.append(result)
    return results


def audit(run):
    run = Path(run)
    started = time.process_time()
    checked_inputs = {}
    def checked(record):
        path = bound(record)
        checked_inputs[str(path)] = record['sha256']
        return path
    names = ('config.json', 'binding.json', 'protocol.json', 'source-bundle.json', 'example.rs',
             'source-state.json', 'attempts.jsonl', 'terminal.json')
    hashes = {name: sha(run / name) for name in names}
    plan, binding, terminal = (read(run / name) for name in ('config.json', 'binding.json', 'terminal.json'))
    width=plan['schema']=='fft-width-physical-reset-v1'
    require(binding['schema'] == 'auxiliary-overlap-physical-binding-v1', 'Unknown physical binding')
    for filename, key in [('config.json', 'config_sha256'), ('protocol.json', 'protocol_sha256'),
                          ('source-bundle.json', 'compiled_source_bundle_sha256'), ('example.rs', 'example_source_sha256')]:
        require(hashes[filename] == binding[key], 'Execution binding mismatch: ' + filename)
    require(sha(checked(plan['protocol'])) == hashes['protocol.json'], 'Changed protocol copy')
    protocol = read(run / 'protocol.json')
    validate_contract(plan, protocol)
    for record in protocol['audit_files'].values():
        checked(record)
    for name, path in [('analyze_auxiliary_overlap_physical.py', __file__),
                       ('analyze_factorized_dimer_physical.py', prior.__file__)]:
        require(sha(path) == protocol['audit_files'][name]['sha256'], 'Executing audit module differs: '+name)
    frozen_path = run.parent/'prelaunch.json'
    frozen_sha = sha(frozen_path)
    frozen = read(frozen_path)
    require(frozen['schema'] == 'auxiliary-overlap-physical-prelaunch-v1' and frozen['complete'] is True,
            'Missing complete prelaunch closure')
    for path, digest in frozen['files'].items():
        checked(dict(path=path, sha256=digest))
    require(binding['executable_sha256'] in frozen['files'].values(), 'Executable absent from frozen closure')
    for filename in ('config.json', 'binding.json', 'protocol.json', 'source-bundle.json', 'example.rs'):
        require(hashes[filename] in frozen['files'].values(), 'Execution file absent from prelaunch: '+filename)
    bundle = read(run/'source-bundle.json')['files']
    for name, item in bundle.items():
        require(hashlib.sha256(item['text'].encode()).hexdigest() == item['sha256'], 'Corrupt compiled source: '+name)
    compiled = {name: item['sha256'] for name, item in bundle.items()}
    require(compiled == plan['compiled_source_sha256'], 'Compiled source differs from plan')
    passive = {name: checked(record) for name, record in plan['passive'].items()}
    passive_compiled = read(passive['source-bundle.json'])['files']
    for name in ('src/docking.rs','src/proposal.rs','src/basin_involution.rs','src/math.rs',
                 'src/defensive_dimer_proposal.rs','src/dimer_tree_proposal.rs','src/capped_dimer.rs',
                 'src/factorized_dimer.rs','src/auxiliary_overlap_threshold.rs','src/geometry.rs','src/spherical.rs'):
        require(compiled[name] == passive_compiled[name]['sha256'], 'Changed proposal/geometry implementation: ' + name)
        require(compiled[name] == protocol['proposal_source_sha256'][name], 'Proposal source absent from protocol: '+name)
    prior_audit = read(passive['analysis.json'])
    require(prior_audit['schema'] == ('fft-width-independent-analysis-v1' if width else 'auxiliary-overlap-independent-analysis-v1') and
            prior_audit['complete'] is True and prior_audit['passed'] is True and not prior_audit['failures'],
            'Passive audit failed')
    require(prior_audit['summary']['outer_attempts'] == 1536 and
            prior_audit['summary']['candidate_count'] == plan['total_candidates'], 'Changed candidate/null allocation')
    for name, digest in prior_audit['input_hashes'].items():
        require(sha(passive[name]) == digest, 'Changed passive audit input: ' + name)
    review = read(passive['root-review.json'])
    require(review['complete'] and review['passed'] and review['output_hashes']['analysis.json'] == sha(passive['analysis.json']), 'Missing passive completion review')
    reference = read(checked(plan['reference_config']))
    require(read(passive['config.json'])['reference_config']['sha256'] == plan['reference_config']['sha256'], 'Different passive source')
    state = read(run / 'source-state.json')
    require(state == read(checked(reference['source_config']))['initial_poses'] == read(checked(reference['source_frame']))['poses'], 'Different source state')
    for name in ('shape','panel','source_freeze_manifest'):
        checked(reference[name])
    for atlas in reference['atlases']:
        checked(atlas['model'])
    if width:
        pc=read(passive['config.json'])
        require(pc['schema']=='fft-width-screen-v1' and pc['master_seed']==6100300401 and
                [a['tau'] for a in pc['scaled_atlases']]==[.125,.25,.5], 'Wrong width proposal family')
        for atlas in pc['scaled_atlases']:checked(atlas['model'])
    require(reference['depletant_radius'] == plan['depletant_radius'] == 1.4 and reference['activity'] == plan['activity'] == .0275,
            'Changed diagnostic physical conditions')
    require(plan['lambda'] == 1.76 and len(state) == 264, 'Wrong model/intensity')
    baseline_results = [] if width else audit_baseline(plan, checked, state, read(passive['protocol.json'])['baseline_cache'])
    ledger = checked(plan['candidate_ledger'])
    cache = [json.loads(line) for line in ledger.read_text().splitlines()]
    original_lines = passive['attempts.jsonl'].read_text().splitlines()
    rows = [json.loads(line) for line in (run / 'attempts.jsonl').read_text().splitlines()]
    require(terminal['attempts_sha256'] == hashes['attempts.jsonl'], 'Terminal ledger differs')
    require(terminal['summary']['complete'], 'Incomplete campaign remains unresolved; no complete estimator')
    require(len(cache) == len(original_lines) == len(rows) == len(prior_audit['rows']) == plan['total_outer'] == 1536, 'Lost unconditional outer rows')
    keys, streams, results = set(), set(), []
    cumulative = dict(raw_points=0, retained_points=0)
    for i, (row, cached, line, passive_row) in enumerate(zip(rows, cache, original_lines, prior_audit['rows'])):
        original = json.loads(line)
        expected = {k: original[k] for k in CACHED_FIELDS}
        expected.update(index=i, passive_row_sha256=hashlib.sha256(line.encode()).hexdigest(),
                        proposal_status=original['outcome']['status'], candidate=original['outcome']['candidate'],
                        guidance=original['outcome'].get('guidance'))
        require(original['status'] == 'completed' and cached == expected, 'Cached row differs from passive ledger')
        require(all(passive_row[k] == cached[k] for k in ('atlas', 'method', 'attempt')) and passive_row['case'] == cached['case']['name'], 'Wrong contact audit row')
        key = (cached['atlas_index'], cached['case_index'], cached['attempt'], cached['method'])
        require(key not in keys, 'Repeated outer key')
        keys.add(key)
        for role in ('gate', 'mh'):
            value = seed(plan, cached, role)
            require(value not in streams, 'Seed collision')
            streams.add(value)
        metrics = audit_decision(row, cached, state, hashes['source-state.json'], plan)
        for name in cumulative:
            cumulative[name] += metrics[name]
            require(row['campaign_completed_' + name] == cumulative[name], 'Wrong campaign counter')
            require(cumulative[name] <= plan['limits'][name.replace('_points', '_campaign')], 'Campaign budget exceeded')
        contact = passive_row.get('contact_change', {})
        candidate = passive_row['candidate']
        results.append(dict(index=i, atlas=cached['atlas'], method=cached['method'], case=cached['case']['name'],
            attempt=cached['attempt'], m=cached['m'], candidate=candidate is not None, accepted=row['accepted'], **metrics,
            external_contacts=candidate['external_contacts'] if candidate else 0,
            contact_changed=bool(contact.get('gained_contacts', 0) or contact.get('lost_contacts', 0)),
            gained_contact_edges=contact.get('gained_contacts', 0), lost_contact_edges=contact.get('lost_contacts', 0),
            log_proposal_correction=row.get('q_correction'), log_depletion_factor=row['gate']['aggregate']['log_weight'] if row['gate'] else None,
            log_full_f_correction=row['full_f_correction'], log_auxiliary_correction=row['auxiliary_correction'],
            log_ratio=row['log_ratio'],
            saved_proposal_cpu_seconds=cached['standalone_proposal_cpu_seconds'], gate_cpu_seconds=row.get('gate_cpu_seconds', 0.),
            physical_replay_cpu_seconds=row['physical_replay_cpu_seconds']))
    require(keys == {(a,c,t,m) for a in range(3) for c in range(8) for t in range(32) for m in (['unguided','m4'] if width else ['m1','m4'])}, 'Changed frozen allocation')
    summary = summarize(results)
    actual = terminal['summary']['result']
    for key in ('outer_attempts','candidates','accepted','raw_points','retained_points'):
        require(actual[key] == summary[key], 'Wrong terminal ' + key)
    require(actual['candidates'] == plan['total_candidates'] and actual['proposal_nulls'] == 1536-actual['candidates'], 'Lost candidate/null count')
    require(actual['new_proposal_draws'] == actual['sequential_state_updates'] == 0 and actual['reset_state_decisions'] == 1536, 'Wrong reset semantics')
    for key in ('gate_cpu_seconds','saved_proposal_cpu_seconds','physical_replay_cpu_seconds','replay_plus_saved_proposal_cpu_seconds'):
        close(actual[key], summary[key], 'Wrong terminal CPU ' + key)
    close(actual['gate_plus_saved_proposal_cpu_seconds'], summary['gate_cpu_seconds']+summary['saved_proposal_cpu_seconds'], 'Wrong gate plus proposal CPU')
    whole = nonnegative(terminal['cpu_seconds'], 'Whole-process CPU')
    require(whole+1e-8 >= summary['physical_replay_cpu_seconds'], 'Incomplete whole-process clock')
    close(terminal['summary']['whole_process_plus_saved_proposal_cpu_seconds'], whole+summary['saved_proposal_cpu_seconds'], 'Wrong complete CPU')
    summary['whole_process_plus_saved_proposal_cpu_seconds'] = whole+summary['saved_proposal_cpu_seconds']
    summary['passive_whole_process_cpu_seconds']=nonnegative(read(passive['terminal.json'])['cpu_seconds'],'Passive whole CPU')
    summary['diagnostic_execution_cpu_seconds']=whole+summary['passive_whole_process_cpu_seconds']
    groups = defaultdict(list)
    contexts = defaultdict(list)
    for row in baseline_results + results:
        groups[(row['atlas'], row['method'])].append(row)
        contexts[(row['atlas'], row['method'], row['case'])].append(row)
    require(hashes == {name: sha(run / name) for name in names}, 'Inputs changed during audit')
    require(all(sha(path) == digest for path,digest in checked_inputs.items()), 'Bound input changed during audit')
    require(sha(frozen_path) == frozen_sha, 'Prelaunch changed during audit')
    return dict(schema='auxiliary-overlap-physical-independent-audit-v1', complete=True, passed=True,
        failures=[], input_hashes=hashes, checked_input_hashes=checked_inputs, candidate_ledger_sha256=sha(ledger), summary=summary,
        prelaunch_sha256=frozen_sha, baseline_review_sha256=None if width else BASELINE_RECEIPT,
        reused_baseline_summary=summarize(baseline_results), reused_baseline_rows=baseline_results,
        allocation=dict(new_reset_decisions=1536, reused_baseline_decisions=0 if width else 768, new_baseline_baths=0,
                        new_guided_decisions=768 if width else 1536, independent_reset_contexts=False,
                        fixed_purposive_contexts=True, independent_bath_streams=True, extension=False),
        comparisons=[dict(atlas=a, method=m, summary=summarize(r)) for (a,m),r in sorted(groups.items())],
        contexts=[dict(atlas=a,method=m,case=c,summary=summarize(r)) for (a,m,c),r in sorted(contexts.items())],
        rows=results, auditor_cpu_seconds=time.process_time()-started,
        limitations=['Saved-count/MH/state audit; does not regenerate Poisson point locations or RNG.',
            'Geometry and full proposal densities rely on the separately bound complete independent passive audit.',
            'Fixed reset contexts are not an equilibrium trajectory; accepted counts are not contact ESS or native assembly.',
            'Guided arms share passive proposal prefixes; their fresh bath decisions use distinct method seed domains.',
            'Historical comparisons use saved decisions and timings; no new historical baseline cloud or timing replicate is drawn.',
            'Per-arm CPU is a production-equivalent proxy; total diagnostic execution includes actual passive and physical whole-process CPU, with audits separate.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        result = audit(args.run)
    except Exception as error:
        result = dict(schema='auxiliary-overlap-physical-independent-audit-v1',complete=False,passed=False,
                      failures=[str(error)],run=str(args.run))
    with args.output.open('x') as out:
        json.dump(result,out,indent=2,allow_nan=False)
        out.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','reused_baseline_rows','contexts','comparisons')},indent=2))
    raise SystemExit(0 if result['passed'] else 1)
