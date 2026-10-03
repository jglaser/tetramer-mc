#!/usr/bin/env python3
"""Audit cached-pose physical decisions; never infer stationarity from reset trials.

This independently checks bindings, every unconditional outer, Poisson count
arithmetic, the full-density MH decision, and all retained particle states.
It does not regenerate random clouds or prove floating-point geometry predicates.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import time


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


def seed(plan, cached, role):
    text = (f"factorized-dimer-physical-v1/{plan['master_seed']}/{plan['candidate_ledger']['sha256']}/"
            f"{cached['atlas_index']}/{cached['case_index']}/{cached['attempt']}/{cached['method']}/{role}")
    return int(hashlib.sha256(text.encode()).hexdigest()[:16], 16)


COUNT_FIELDS = ('gained', 'lost', 'raw_points', 'retained_points', 'retained_cells', 'created_cells')


def audit_gate(gate, plan):
    for key in COUNT_FIELDS:
        integer(gate[key], key)
    require(gate['retained_points'] == gate['gained'] + gate['lost'] <= gate['raw_points'], 'Inconsistent point counts')
    require(gate['retained_cells'] <= gate['created_cells'], 'Inconsistent cell counts')
    nonnegative(gate['envelope_volume'], 'Envelope volume')
    require(gate['raw_points'] <= plan['limits']['raw_per_leg'], 'Leg raw cap exceeded')
    require(gate['retained_points'] <= plan['limits']['retained_per_leg'], 'Leg retained cap exceeded')
    coefficient = math.log1p(plan['activity'] / plan['lambda'])
    close(gate['log_weight'], coefficient * (gate['gained'] - gate['lost']), 'Wrong gained/lost factor')
    if gate['envelope_volume'] == 0 or plan['activity'] == 0:
        require(gate['raw_points'] == 0 and gate['log_weight'] == 0, 'Zero-limit cloud not empty')


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
    require(type(row['accepted']) is bool and row['gate_failure'] is None, 'Failed gate cannot be a decision')
    nonnegative(row['physical_replay_cpu_seconds'], 'Replay CPU')
    nonnegative(cached['proposal_cpu_seconds'], 'Saved proposal CPU')
    nonnegative(cached['contact_diagnostic_cpu_seconds'], 'Saved contact CPU')
    candidate = cached['candidate']
    expected_state = list(state)
    counts = dict(raw_points=0, retained_points=0, gained=0, lost=0)
    if candidate is None:
        require(cached['proposal_status'] in ('cap_exhausted', 'source_outside_domain', 'source_outside_contact'), 'Unknown proposal null')
        require(row['status'] == 'proposal_null' and row['gate'] is None and not row['accepted'] and row['log_ratio'] is None,
                'Proposal null acquired a physical decision')
        require(row['proposed_selected'] is None and 'q_correction' not in row and 'gate_cpu_seconds' not in row,
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
        diagnostics = candidate['diagnostics']
        correction = log_value(diagnostics['log_reverse_forward'])
        old_f = log_value(diagnostics['full_old_log_density'])
        new_f = log_value(diagnostics['full_new_log_density'])
        require(math.isfinite(new_f) and correction == old_f - new_f, 'Wrong full-density correction')
        require(diagnostics['selection_log_reverse_forward'] == diagnostics['log_tree_coordinate_jacobian'] == 0,
                'Unaccounted selection/Jacobian')
        require(log_value(row['q_correction']) == correction, 'Changed proposal correction')
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
        fingerprint = row['gate_rng_after_fingerprint']
        require(len(fingerprint) == 4 and all(type(v) is int and 0 <= v < 2**64 for v in fingerprint), 'Invalid RNG fingerprint')
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
    return result


def audit(run):
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
    require(plan['schema'] == 'factorized-dimer-physical-reset-v1', 'Unknown plan')
    for filename, key in [('config.json', 'config_sha256'), ('protocol.json', 'protocol_sha256'),
                          ('source-bundle.json', 'compiled_source_bundle_sha256'), ('example.rs', 'example_source_sha256')]:
        require(hashes[filename] == binding[key], 'Execution binding mismatch: ' + filename)
    require(sha(checked(plan['protocol'])) == hashes['protocol.json'], 'Changed protocol copy')
    protocol = read(run / 'protocol.json')
    for record in protocol['audit_files'].values():
        checked(record)
    compiled = {name: item['sha256'] for name, item in read(run / 'source-bundle.json')['files'].items()}
    require(compiled == plan['compiled_source_sha256'], 'Compiled source differs from plan')
    passive = {name: checked(record) for name, record in plan['passive'].items()}
    passive_compiled = read(passive['source-bundle.json'])['files']
    for name in ('src/docking.rs','src/proposal.rs','src/basin_involution.rs','src/math.rs',
                 'src/defensive_dimer_proposal.rs','src/dimer_tree_proposal.rs','src/capped_dimer.rs',
                 'src/factorized_dimer.rs','src/geometry.rs','src/spherical.rs'):
        require(compiled[name] == passive_compiled[name]['sha256'], 'Changed proposal/geometry implementation: ' + name)
    prior_audit = read(passive['analysis.json'])
    require(prior_audit['complete'] and prior_audit['passed'] and not prior_audit['failures'], 'Passive audit failed')
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
    require(reference['depletant_radius'] == plan['depletant_radius'] == 1.4 and reference['activity'] == plan['activity'] == .0275,
            'Changed diagnostic physical conditions')
    require(plan['lambda'] == 1.76 and len(state) == 264, 'Wrong model/intensity')
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
        expected = {k: original[k] for k in ('atlas', 'atlas_index', 'case', 'case_index', 'attempt', 'method',
                    'old', 'anchor_pose', 'proposal_cpu_seconds', 'contact_diagnostic_cpu_seconds', 'raw_edge_draws')}
        expected.update(index=i, passive_row_sha256=hashlib.sha256(line.encode()).hexdigest(),
                        proposal_status=original['outcome']['status'], candidate=original['outcome']['candidate'])
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
            attempt=cached['attempt'], candidate=candidate is not None, accepted=row['accepted'], **metrics,
            external_contacts=candidate['external_contacts'] if candidate else 0,
            contact_changed=bool(contact.get('gained_contacts', 0) or contact.get('lost_contacts', 0)),
            gained_contact_edges=contact.get('gained_contacts', 0), lost_contact_edges=contact.get('lost_contacts', 0),
            log_proposal_correction=row.get('q_correction'), log_depletion_factor=row['gate']['aggregate']['log_weight'] if row['gate'] else None,
            log_ratio=row['log_ratio'],
            saved_proposal_cpu_seconds=cached['proposal_cpu_seconds'], gate_cpu_seconds=row.get('gate_cpu_seconds', 0.),
            physical_replay_cpu_seconds=row['physical_replay_cpu_seconds']))
    require(keys == {(a,c,t,m) for a in range(3) for c in range(8) for t in range(32) for m in ('whole_joint','factorized')}, 'Changed frozen allocation')
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
    groups = defaultdict(list)
    contexts = defaultdict(list)
    for row in results:
        groups[(row['atlas'], row['method'])].append(row)
        contexts[(row['atlas'], row['method'], row['case'])].append(row)
    require(hashes == {name: sha(run / name) for name in names}, 'Inputs changed during audit')
    require(all(sha(path) == digest for path,digest in checked_inputs.items()), 'Bound input changed during audit')
    return dict(schema='factorized-dimer-physical-independent-audit-v1', complete=True, passed=True,
        failures=[], input_hashes=hashes, checked_input_hashes=checked_inputs, candidate_ledger_sha256=sha(ledger), summary=summary,
        comparisons=[dict(atlas=a, method=m, summary=summarize(r)) for (a,m),r in sorted(groups.items())],
        contexts=[dict(atlas=a,method=m,case=c,summary=summarize(r)) for (a,m,c),r in sorted(contexts.items())],
        rows=results, auditor_cpu_seconds=time.process_time()-started,
        limitations=['Saved-count/MH/state audit; does not regenerate Poisson point locations or RNG.',
            'Geometry and full proposal densities rely on the separately bound complete independent passive audit.',
            'Fixed reset contexts are not an equilibrium trajectory; accepted counts are not contact ESS or native assembly.',
            'Per-arm CPU omits shared setup/output overhead; complete observed campaign cost reported separately.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        result = audit(args.run)
    except Exception as error:
        result = dict(schema='factorized-dimer-physical-independent-audit-v1',complete=False,passed=False,
                      failures=[str(error)],run=str(args.run))
    with args.output.open('x') as out:
        json.dump(result,out,indent=2,allow_nan=False)
        out.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','contexts','comparisons')},indent=2))
    raise SystemExit(0 if result['passed'] else 1)
