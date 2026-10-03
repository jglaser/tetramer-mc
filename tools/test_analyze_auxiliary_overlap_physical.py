"""Deterministic adversarial records only; no proposal, bath or random draws."""
import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import analyze_auxiliary_overlap_physical as audit


def pose(x):
    return dict(position=[x, 0., 0.], orientation=[1., 0., 0., 0.])


def fixture(m=4):
    plan = dict(master_seed=6100300301, candidate_ledger={'sha256': '0'*64}, activity=.0275,
                **{'lambda': 1.76}, limits=copy.deepcopy(audit.LIMITS))
    state = [pose(0.), pose(4.), pose(8.)]
    proposed = [pose(1.), pose(5.)]
    full = -5.
    auxiliary = m*(math.log1p(8)-math.log1p(2))
    complete = full+auxiliary
    candidate = dict(root=proposed[0], child=proposed[1], diagnostics=dict(
        full_old_log_density=-14., full_new_log_density=-9., log_reverse_forward=full,
        selection_log_reverse_forward=0., log_tree_coordinate_jacobian=0.))
    integers = [1] if m == 1 else [1, 0, 2, 1]
    cached = dict(index=0, atlas_index=0, case_index=0, attempt=0, method=f'm{m}', m=m,
        case=dict(root=0, child=1, anchor=2), old=state[:2], anchor_pose=state[2],
        proposal_status='candidate', candidate=candidate, proposal_cpu_seconds=.2,
        cloud_construction_cpu_seconds=.1, guidance_setup_cpu_seconds=.01,
        standalone_proposal_cpu_seconds=.31, contact_diagnostic_cpu_seconds=.001,
        complete_log_correction=complete,
        guidance=dict(m=m, point_count=10, old_count=8, new_count=2, threshold=max(integers),
                      integer_draws=integers, aux_log_correction=auxiliary))
    coefficient = math.log1p(plan['activity']/plan['lambda'])
    first = dict(gained=3, lost=1, raw_points=10, retained_points=4, retained_cells=3,
                 created_cells=5, envelope_volume=2., log_weight=2*coefficient)
    second = dict(gained=0, lost=1, raw_points=7, retained_points=1, retained_cells=2,
                  created_cells=3, envelope_volume=1., log_weight=-coefficient)
    aggregate = {key: first[key]+second[key] for key in first}
    gate = dict(order='first_then_second', ordered_members=[0, 1],
                intermediate_selected=[proposed[0], state[1]], legs=[first, second], aggregate=aggregate)
    feasible = dict(internal_core_overlap=False, spectator_core_collisions=[[], []],
                    wall_valid=[True, True], internal_exclusion_contact=True)
    ratio = complete+aggregate['log_weight']
    row = dict(index=0, cached=cached, source_state_sha256='source', source_selected=state[:2],
        proposed_selected=proposed, gate_seed=audit.seed(plan, cached, 'gate'),
        mh_seed=audit.seed(plan, cached, 'mh'), log_uniform=-10., mh_rng_after_fingerprint=[5, 6, 7, 8],
        accepted=True, gate_failure=None, physical_replay_cpu_seconds=.7, status='physical_decision',
        source_feasibility=feasible, endpoint_feasibility=copy.deepcopy(feasible),
        full_f_correction=full, auxiliary_correction=auxiliary, q_correction=complete, gate=gate,
        log_ratio=ratio, gate_rng_after_fingerprint=[1, 2, 3, 4], gate_cpu_seconds=.5,
        retained_selected=proposed, retained_state=proposed+[state[2]])
    return row, cached, state, 'source', plan


def rejected(row, cached, state):
    row.update(accepted=False, retained_state=state, retained_selected=cached['old'])


def null_fixture():
    row, cached, state, source, plan = fixture()
    cached.update(candidate=None, proposal_status='cap_exhausted', complete_log_correction=None)
    cached['guidance'].update(new_count=None, aux_log_correction=None)
    row.update(status='proposal_null', gate=None, log_ratio=None, proposed_selected=None,
               full_f_correction=None, auxiliary_correction=None, q_correction=None)
    rejected(row, cached, state)
    del row['gate_cpu_seconds'], row['gate_rng_after_fingerprint']
    return row, cached, state, source, plan


class GuidedDecisionAudit(unittest.TestCase):
    def test_width_allocation_seed_and_no_baseline_contract(self):
        p=dict(schema='fft-width-physical-reset-v1',master_seed=6100300501,total_outer=1536,total_candidates=10,
               activity=.0275,depletant_radius=1.4,**{'lambda':1.76},limits=audit.LIMITS,
               envelope=dict(max_cells=255,max_depth=8,min_width=0.),reference_config=dict(sha256=audit.REFERENCE_SHA),
               candidate_ledger={},passive={},baseline={},baseline_cache=None)
        q={k:copy.deepcopy(p[k]) for k in ['master_seed','limits','envelope','candidate_ledger','passive','baseline','baseline_cache']}
        q.update(schema='fft-width-physical-reset-protocol-v1',allocation=dict(total_outer=1536,candidates=10,failed_proposals=1526,
                 atlases=3,contexts=8,attempts_per_context_and_method=32,methods=['unguided','m4'],extension=False),
                 reused_baseline_attempts=0,new_baseline_baths=0,no_new_proposal_draws=True,native_classifier=False,outcome_filtering=False)
        audit.validate_contract(p,q)
        for mutate in ['seed','methods','baseline','extension','density']:
            a,b=copy.deepcopy(p),copy.deepcopy(q)
            if mutate=='seed':a['master_seed']=b['master_seed']=6100300301
            elif mutate=='methods':b['allocation']['methods']=['m1','m4']
            elif mutate=='baseline':a['baseline_cache']=b['baseline_cache']={}
            elif mutate=='extension':b['allocation']['extension']=True
            else:a['activity']=.035
            with self.subTest(mutate=mutate),self.assertRaises(ValueError):audit.validate_contract(a,b)
        seeds={audit.seed(dict(master_seed=6100300501,candidate_ledger=dict(sha256='0'*64)),dict(atlas_index=a,case_index=c,attempt=j,method=m),role)
               for a in range(3) for c in range(8) for j in range(32) for m in ['unguided','m4'] for role in ['gate','mh']}
        self.assertEqual(len(seeds),3072)

    def test_unguided_width_physical_decision_and_null(self):
        args=fixture();r,c,state,_,plan=args
        plan['master_seed']=6100300501;c.update(method='unguided',m=0,guidance=None,complete_log_correction=-5.,standalone_proposal_cpu_seconds=.2)
        r.update(gate_seed=audit.seed(plan,c,'gate'),mh_seed=audit.seed(plan,c,'mh'),auxiliary_correction=0.,q_correction=-5.)
        r['log_ratio']=-5.+r['gate']['aggregate']['log_weight']
        audit.audit_decision(*args)
        bad=copy.deepcopy(args);bad[1]['guidance']={}
        with self.assertRaisesRegex(ValueError,'auxiliary'):audit.audit_decision(*bad)
        bad=copy.deepcopy(args);bad[1]['complete_log_correction']=-4.
        with self.assertRaisesRegex(ValueError,'complete'):audit.audit_decision(*bad)
        c.update(candidate=None,proposal_status='cap_exhausted',complete_log_correction=None)
        r.update(status='proposal_null',gate=None,log_ratio=None,proposed_selected=None,full_f_correction=None,auxiliary_correction=None,q_correction=None)
        rejected(r,c,state);del r['gate_cpu_seconds'],r['gate_rng_after_fingerprint']
        self.assertEqual(audit.audit_decision(*args)['raw_points'],0)

    def test_both_m_values_accept_reject_and_path_orders(self):
        for m in (1, 4):
            args = fixture(m)
            row, cached, state, _, _ = args
            result = audit.audit_decision(*args)
            self.assertEqual(result['raw_points'], 17)
            self.assertGreater(result['acceptance_probability'], 0.)
            row['log_uniform'] = -.001
            rejected(row, cached, state)
            row['gate'].update(order='second_then_first', ordered_members=[1, 0],
                               intermediate_selected=[state[0], row['proposed_selected'][1]])
            audit.audit_decision(*args)

    def test_missing_and_double_auxiliary_even_with_consistent_wrong_mh(self):
        for multiplier in (0, 2):
            row, cached, state, source, plan = fixture()
            wrong = row['full_f_correction']+multiplier*row['auxiliary_correction']
            row['q_correction'] = wrong
            row['log_ratio'] = wrong+row['gate']['aggregate']['log_weight']
            row['accepted'] = row['log_uniform'] < min(0., row['log_ratio'])
            with self.assertRaisesRegex(ValueError, 'auxiliary'):
                audit.audit_decision(row, cached, state, source, plan)
            cached['complete_log_correction'] = wrong
            with self.assertRaisesRegex(ValueError, 'auxiliary'):
                audit.audit_decision(row, cached, state, source, plan)

    def test_zero_count_and_negative_infinity(self):
        args = fixture()
        row, cached, state, _, _ = args
        cached['guidance'].update(old_count=0, new_count=0, threshold=0, integer_draws=[0]*4,
                                  aux_log_correction=0.)
        cached['complete_log_correction'] = row['q_correction'] = row['full_f_correction']
        row['auxiliary_correction'] = 0.
        row['log_ratio'] = row['q_correction']+row['gate']['aggregate']['log_weight']
        audit.audit_decision(*args)
        cached['candidate']['diagnostics'].update(full_old_log_density='-inf', log_reverse_forward='-inf')
        cached['complete_log_correction'] = '-inf'
        row.update(full_f_correction='-inf', q_correction='-inf', log_ratio='-inf')
        rejected(row, cached, state)
        self.assertEqual(audit.audit_decision(*args)['acceptance_probability'], 0.)

    def test_every_null_keeps_mh_coin_but_draws_no_bath(self):
        args = null_fixture()
        self.assertEqual(audit.audit_decision(*args)['raw_points'], 0)
        for key, value in [('q_correction', 0.), ('full_f_correction', 0.),
                           ('auxiliary_correction', 0.), ('gate_cpu_seconds', 0.),
                           ('gate_rng_after_fingerprint', [1, 2, 3, 4]), ('accepted', True)]:
            broken = copy.deepcopy(args)
            broken[0][key] = value
            with self.assertRaises(ValueError):
                audit.audit_decision(*broken)
        del args[0]['mh_rng_after_fingerprint']
        with self.assertRaises(KeyError):
            audit.audit_decision(*args)

    def test_corrupt_threshold_density_counts_coins_intermediate_and_state(self):
        mutations = [
            lambda r, c: c['guidance'].update(threshold=3),
            lambda r, c: c['guidance'].update(integer_draws=[1, 0, 2]),
            lambda r, c: c['guidance'].update(old_count=11),
            lambda r, c: c['guidance'].update(aux_log_correction=0.),
            lambda r, c: c['candidate']['diagnostics'].update(full_new_log_density=-8.),
            lambda r, c: c['candidate']['diagnostics'].update(selection_log_reverse_forward=1.),
            lambda r, c: r.update(full_f_correction=0.),
            lambda r, c: r.update(auxiliary_correction=0.),
            lambda r, c: r['gate']['legs'][0].update(gained=8),
            lambda r, c: r['gate']['aggregate'].update(raw_points=18),
            lambda r, c: r['gate']['aggregate'].update(log_weight=1.),
            lambda r, c: r['gate'].update(intermediate_selected=r['proposed_selected']),
            lambda r, c: r['gate'].update(ordered_members=[1, 0]),
            lambda r, c: r.update(log_uniform=0.),
            lambda r, c: r.update(mh_seed=r['gate_seed']),
            lambda r, c: r.update(mh_rng_after_fingerprint=[0]*3),
            lambda r, c: r.update(gate_rng_after_fingerprint=[True]*4),
            lambda r, c: r.update(accepted=False),
            lambda r, c: r['retained_state'].__setitem__(2, pose(99.)),
            lambda r, c: r.update(source_state_sha256='different'),
            lambda r, c: r.update(status='fatal'),
            lambda r, c: r.update(gate_failure={'reason': 'budget'}),
            lambda r, c: r['endpoint_feasibility'].update(wall_valid=[False, True]),
            lambda r, c: c.update(standalone_proposal_cpu_seconds=.2),
            lambda r, c: r.update(gate_cpu_seconds=1.),
        ]
        for mutate in mutations:
            args = fixture()
            mutate(args[0], args[1])
            with self.assertRaises(ValueError):
                audit.audit_decision(*args)

    def test_reused_count_factor_zero_volume_and_cap_failures(self):
        plan = fixture()[-1]
        zero = dict(gained=0, lost=0, raw_points=0, retained_points=0, retained_cells=0,
                    created_cells=0, envelope_volume=0., log_weight=0.)
        audit.audit_gate(zero, plan)
        for key, value in [('raw_points', 1), ('gained', -1), ('log_weight', math.nan)]:
            with self.assertRaises(ValueError):
                audit.audit_gate(dict(zero, **{key: value}), plan)
        with self.assertRaises(ValueError):
            audit.audit_gate(dict(zero, envelope_volume=1., raw_points=20_000_001), plan)

    def test_seed_domains_include_method_role_ledger_and_master(self):
        values = set()
        for m in (1, 4):
            args = fixture(m)
            cached, plan = args[1], args[-1]
            for role in ('gate', 'mh'):
                for master in (6100300301, 6100203201):
                    for digest in ('0'*64, '1'*64):
                        plan.update(master_seed=master, candidate_ledger={'sha256': digest})
                        values.add(audit.seed(plan, cached, role))
        self.assertEqual(len(values), 16)

    def test_contract_rejects_changed_model_budget_allocation_or_reuse(self):
        plan = dict(schema='auxiliary-overlap-physical-reset-v1', master_seed=6100300301,
            total_outer=1536, total_candidates=324, activity=.0275, depletant_radius=1.4,
            **{'lambda': 1.76}, limits=copy.deepcopy(audit.LIMITS),
            envelope=dict(max_cells=255, max_depth=8, min_width=0.),
            reference_config=dict(sha256=audit.REFERENCE_SHA), candidate_ledger={}, passive={},
            baseline={}, baseline_cache={})
        protocol = dict(schema='auxiliary-overlap-physical-reset-protocol-v1',
            **{key: copy.deepcopy(plan[key]) for key in ('master_seed', 'limits', 'envelope',
                    'candidate_ledger', 'passive', 'baseline', 'baseline_cache')},
            allocation=dict(total_outer=1536, candidates=324, failed_proposals=1212,
                atlases=3, contexts=8, attempts_per_context_and_method=32,
                methods=['m1', 'm4'], extension=False), reused_baseline_attempts=768,
            new_baseline_baths=0, no_new_proposal_draws=True, native_classifier=False,
            outcome_filtering=False)
        audit.validate_contract(plan, protocol)
        mutations = [lambda p, q: p.update(master_seed=6100203201),
                     lambda p, q: p.update(activity=.035),
                     lambda p, q: p.update(total_outer=1535),
                     lambda p, q: p['limits'].update(cpu_seconds=2400.),
                     lambda p, q: q['allocation'].update(extension=True),
                     lambda p, q: q.update(new_baseline_baths=768),
                     lambda p, q: q.update(outcome_filtering=True)]
        for mutate in mutations:
            args = copy.deepcopy((plan, protocol))
            mutate(*args)
            with self.assertRaises(ValueError):
                audit.validate_contract(*args)


def baseline_fixture(root):
    state = [pose(0.), pose(4.), pose(8.)]
    raw_lines, passive_lines, audited = [], [], []
    for a in range(3):
        for c in range(8):
            for t in range(32):
                index = len(audited)
                passive = dict(outcome=dict(candidate=None), old=state[:2])
                line = json.dumps(passive)
                passive_lines.append(line+'\n')
                cached = dict(method='factorized', atlas=f'a{a}', atlas_index=a, case_index=c,
                              case=dict(name=f'c{c}'), attempt=t, candidate=None, old=state[:2],
                              passive_row_sha256=hashlib.sha256(line.encode()).hexdigest())
                raw = dict(index=index, cached=cached, accepted=False, gate_failure=None, status='proposal_null')
                raw_lines.append(json.dumps(raw)+'\n')
                audited.append(dict(index=index, method='factorized', atlas=f'a{a}', case=f'c{c}',
                                    attempt=t, candidate=False, accepted=False, log_proposal_correction=None))
    def write(name, value, raw=False):
        path = root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value if raw else json.dumps(value))
        return dict(path=str(path), sha256=audit.sha(path))
    files = {}
    files['execution/attempts.jsonl'] = write('execution/attempts.jsonl', ''.join(raw_lines), raw=True)
    files['execution/source-state.json'] = write('execution/source-state.json', state)
    files['execution/terminal.json'] = write('execution/terminal.json', dict(
        summary=dict(complete=True), attempts_sha256=files['execution/attempts.jsonl']['sha256']))
    files['analysis.json'] = write('analysis.json', dict(schema='factorized-dimer-physical-independent-audit-v1',
        complete=True, passed=True, failures=[], rows=audited,
        input_hashes={name.removeprefix('execution/'): value['sha256'] for name, value in files.items()}))
    receipt = dict(schema='factorized-dimer-physical-completed-review-v1', complete=True, passed=True,
                   physical_exit_code=0, audit_exit_code=0, output_hashes={name: value['sha256'] for name, value in files.items()})
    files['completed-review.json'] = write('completed-review.json', receipt)
    cache = write('baseline-cache.jsonl', ''.join(raw_lines), raw=True)
    passive = write('passive-cache.jsonl', ''.join(passive_lines), raw=True)
    return dict(baseline=files, baseline_cache=cache), state, passive


class BaselineAudit(unittest.TestCase):
    def test_exact_all_768_historical_decisions_and_corrupt_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            plan, state, passive = baseline_fixture(Path(directory))
            with patch.object(audit, 'BASELINE_RECEIPT', plan['baseline']['completed-review.json']['sha256']), \
                 patch.object(audit, 'BASELINE_ANALYSIS', plan['baseline']['analysis.json']['sha256']):
                result = audit.audit_baseline(plan, audit.bound, state, passive)
                self.assertEqual(len(result), 768)
                self.assertTrue(all(row['reused'] and row['method'] == 'baseline' for row in result))
                cache = Path(plan['baseline_cache']['path'])
                cache.write_text(cache.read_text()+'\n')
                plan['baseline_cache']['sha256'] = audit.sha(cache)
                with self.assertRaisesRegex(ValueError, 'exact historical subset'):
                    audit.audit_baseline(plan, audit.bound, state, passive)

    def test_lost_duplicate_reordered_or_fatal_baseline_cannot_be_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            plan, _, _ = baseline_fixture(Path(directory))
            lines = Path(plan['baseline_cache']['path']).read_bytes().splitlines(keepends=True)
            for broken in (lines[:-1], lines+[lines[0]], [lines[1], lines[0]]+lines[2:]):
                with self.assertRaises(ValueError):
                    audit.baseline_subset(b''.join(broken))
            row = json.loads(lines[0]); row['status'] = 'fatal'
            lines[0] = json.dumps(row).encode()+b'\n'
            with self.assertRaises(ValueError):
                audit.baseline_subset(b''.join(lines))


if __name__ == '__main__':
    unittest.main()
