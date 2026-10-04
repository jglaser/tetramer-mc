"""Synthetic scalar/metadata fixtures; never classify or sample protein geometry."""
import copy
import itertools
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import analyze_rigid_surrogate_benchmark as a


def pose(x):
    return dict(position=[float(x), 0., 0.], orientation=[1., 0., 0., 0.])


def gate(lost=0):
    return dict(gained=0, lost=lost, raw_points=lost, retained_points=lost,
                created_cells=1 if lost else 0, retained_cells=1 if lost else 0,
                envelope_volume=1. if lost else 0., log_weight=-lost*math.log1p(1/64))


def score(selected, contract, config):
    x = selected[0]['position'][0]
    covered = 1 if x < 1.5 else (2 if x < 3.5 else 0)
    units = 4*covered; volume = .5*contract['point_volume']*units
    return dict(points_per_body=2, spectator_covered=[covered]*2,
        internal_unshielded=[0, 0], nearby_spectators=[1, 1], spectator_membership_queries=4,
        internal_membership_queries=4-2*covered, twice_overlap_units=units,
        overlap_volume_estimate=volume,
        log_surrogate=volume*config['physical']['activity']*contract['effective_config']['guidance_strength'])


def fixture(arm=a.ARMS[1], initialization='source'):
    config = dict(contexts=[dict(root=9, child=24, anchor=71)],
        surrogate_policy=dict(schema='rigid-surrogate-policy-v1', proposal_scales=dict(source='local')),
        local=dict(translation_std_A=.4, rotation_std_degrees=3.),
        allocation=dict(warmup_blocks=1, production_blocks=3), physical=dict(activity=.0275, lambda_ratio=64.),
        cloud=dict(raw_count=2), source_frame={'synthetic': 'source'})
    job = dict(id=0, context_index=0, arm=arm, initialization=initialization, stream=0)
    bank = dict(synthetic='frozen cloud'); meta = dict(raw_count=2, low=[0., 0., 0.], high=[2., 2., 2.],
        kept_indices=[0, 1], cpu_seconds=.01)
    prepared = dict(synthetic='prepared manifest')
    start = None if initialization == 'source' else dict(synthetic='prepared entry')
    contract = a.surrogate_contract(config, job, bank, meta); settings = contract['effective_config']
    initial = [pose(1), pose(4)]; selected = copy.deepcopy(initial)
    rows = [dict(kind='initial', block=0, job=job, selected=copy.deepcopy(initial), conditional_target=True,
        sampler_cpu_seconds=0., fixed_source=config['source_frame'], cloud=bank, cloud_cpu_seconds=.01,
        prepared_manifest=prepared, prepared_start=start, surrogate_contract=contract)]
    counts = dict(local_attempted=0, local_accepted=0, dimer_attempted=0, dimer_accepted=0, dimer_self_loop=0)
    raw = retained = 0
    for block in range(1, 5):
        for attempt, slot in enumerate((0, 1, 0, 1)):
            rows.append(dict(kind='local', block=block, attempt=attempt, member=(9, 24)[slot],
                old=copy.deepcopy(selected[slot]), proposed=pose(999), accepted=False, status='hard_rejected',
                retained=copy.deepcopy(selected), sampler_cpu_seconds=float(len(rows))))
            counts['local_attempted'] += 1
        old = copy.deepcopy(selected); current = copy.deepcopy(old)
        old_score = score(old, contract, config); current_score = old_score
        rows.append(dict(kind='rigid_surrogate_attempt_begun', status='begun', block=block,
            members=[9, 24], handle=9, config=settings, old=copy.deepcopy(old), raw=raw,
            bath_retained=retained, sampler_cpu_seconds=float(len(rows))))
        steps = []
        for index in range(settings['inner_steps']):
            x = float(block+1) if index == 0 else current[0]['position'][0]+1.
            target = [pose(x), pose(x+3.)]
            step = dict(index=index, old_handle=current[0], old_score=current_score['log_surrogate'],
                        proposed_handle=target[0], proposed=target, accepted=False, status='hard_rejected')
            if index == 0 and block != 3:
                candidate_score = score(target, contract, config)
                delta = candidate_score['log_surrogate']-current_score['log_surrogate']
                log_u = -.0001 if block == 4 else -10.
                accepted = log_u < min(0., delta)
                step.update(status='completed', proposed_score=candidate_score, log_acceptance_ratio=delta,
                            log_u=log_u, accepted=accepted)
                if accepted: current = copy.deepcopy(target); current_score = candidate_score
            step['retained'] = copy.deepcopy(current); steps.append(step)
        correction = old_score['log_surrogate']-current_score['log_surrogate']
        outcome = dict(kind='rigid_surrogate_chain', block=block, members=[9, 24], handle=9,
            config=settings, old=old, old_score=old_score, proposed=current, proposed_score=current_score,
            steps=steps, complete_log_correction=correction, accepted=False,
            physical_decisions=0, status='identity_self_loop', sampler_cpu_seconds=float(len(rows)))
        if current != old:
            bath = gate(64 if block == 2 else 0); ratio = bath['log_weight']+correction
            log_u = -.0001 if block == 2 else -10.; accepted = log_u < min(0., ratio)
            outcome.update(status='completed', physical_decisions=1, bath=bath, log_u=log_u,
                           log_acceptance_ratio=ratio, accepted=accepted)
            raw += bath['raw_points']; retained += bath['retained_points']
            if accepted: selected = copy.deepcopy(current)
        outcome['retained'] = copy.deepcopy(selected); rows.append(outcome)
        counts['dimer_attempted'] += 1; counts['dimer_accepted'] += outcome['accepted']
        counts['dimer_self_loop'] += outcome['status'] == 'identity_self_loop'
        rows.append(dict(kind='retained_block', block=block, production=block > 1, selected=copy.deepcopy(selected),
            counts=copy.deepcopy(counts), raw=raw, retained=retained, sampler_cpu_seconds=float(len(rows))))
    terminal = dict(complete=True, conditional_target=True, job=job, blocks=4, counts=counts,
        raw=raw, retained=retained, cpu_seconds=float(len(rows)))
    return config, job, initial, rows, terminal, bank, meta, prepared, start


def check(data):
    c, j, initial, rows, terminal, bank, meta, prepared, start = data
    with mock.patch.object(a.previous, 'ConditionalObserver', side_effect=AssertionError('No geometry')):
        return a.validate_journal(iter(rows), c, j, initial, terminal, bank, meta, prepared, start)


class SurrogateReplayTests(unittest.TestCase):
    def test_all_arms_starts_rejections_and_projection_preserve_original(self):
        for arm, initialization in itertools.product(a.ARMS, a.STARTS):
            with self.subTest(arm=arm, initialization=initialization):
                data = fixture(arm, initialization); frozen = copy.deepcopy(data)
                points = check(data)
                self.assertEqual(len(points), 5); self.assertEqual(data, frozen)
                self.assertEqual(points[1]['selected'], points[2]['selected'])
                self.assertEqual(points[2]['selected'], points[3]['selected'])
                self.assertEqual(points[-1]['counts']['dimer_attempted'], 4)

    def test_contract_raw_denominator_and_bad_scores_rejected(self):
        mutations = [
            lambda f: f[3][0]['surrogate_contract'].__setitem__('point_volume', 8.),
            lambda f: f[3][6]['old_score'].__setitem__('twice_overlap_units', 99),
            lambda f: f[3][6]['old_score'].__setitem__('internal_membership_queries', 99),
            lambda f: f[3][6]['old_score'].__setitem__('spectator_membership_queries', 99),
            lambda f: f[3][6]['old_score'].__setitem__('log_surrogate', math.nan),
            lambda f: f[3][6]['old_score'].__setitem__('points_per_body', True),
            lambda f: f[3][0].__setitem__('prepared_start', {'wrong': 'start'}),
            lambda f: f[6].__setitem__('kept_indices', [1, 0]),
        ]
        for mutation in mutations:
            data = fixture(); mutation(data)
            with self.assertRaises(ValueError): check(data)

    def test_fixed_horizon_rigidity_and_inner_decisions(self):
        mutations = [
            lambda row: row['steps'].pop(),
            lambda row: row['steps'][0]['proposed'][1]['position'].__setitem__(0, 99.),
            lambda row: row['steps'][0].__setitem__('old_handle', pose(888)),
            lambda row: row['steps'][0].__setitem__('accepted', False),
            lambda row: row['steps'][1].__setitem__('log_u', -1.),
            lambda row: row['steps'][0].__setitem__('log_acceptance_ratio', math.inf),
            lambda row: row['steps'][0].__setitem__('retained', [pose(9), pose(12)]),
        ]
        for mutation in mutations:
            data = fixture(); mutation(data[3][6])
            with self.assertRaises(ValueError): check(data)

    def test_outer_correction_one_gate_and_identity_controls(self):
        mutations = [
            lambda rows: rows[6].__setitem__('complete_log_correction', -rows[6]['complete_log_correction']),
            lambda rows: rows[6].__setitem__('physical_decisions', 2),
            lambda rows: rows[6]['bath'].__setitem__('log_weight', 1.),
            lambda rows: rows[6].__setitem__('accepted', False),
            lambda rows: rows[20].__setitem__('bath', gate()),
            lambda rows: rows[20].__setitem__('status', 'completed'),
            lambda rows: rows[13].__setitem__('retained', rows[13]['proposed']),
        ]
        for mutation in mutations:
            data = fixture(); mutation(data[3])
            with self.assertRaises(ValueError): check(data)

    def test_begun_cadence_cpu_counters_fatal_and_tail(self):
        mutations = [
            lambda rows: rows.pop(5),
            lambda rows: rows[5].__setitem__('raw', 1),
            lambda rows: rows[5].__setitem__('sampler_cpu_seconds', 99.),
            lambda rows: rows[5].__setitem__('old', [pose(90), pose(93)]),
            lambda rows: rows[6].__setitem__('fatal_error', 'partial bath'),
            lambda rows: rows.append(copy.deepcopy(rows[5])),
        ]
        for mutation in mutations:
            data = fixture(); mutation(data[3])
            with self.assertRaises(ValueError): check(data)
        data = fixture(); data[3][:] = data[3][:6]
        with self.assertRaises(ValueError): check(data)

    def test_rotated_rigid_reconstruction_and_diagnostics(self):
        old = [pose(1), pose(4)]
        handle = dict(position=[5., 3., 1.], orientation=[math.sqrt(.5), 0., 0., math.sqrt(.5)])
        child = dict(position=[5., 6., 1.], orientation=handle['orientation'])
        a.validate_rigid(old, handle, [handle, child])
        child['position'][1] += .01
        with self.assertRaises(ValueError): a.validate_rigid(old, handle, [handle, child])
        c, j, initial, rows, t, bank, meta, prep, start = fixture()
        d = dict(warmup_blocks=1)
        a.validate_journal(rows, c, j, initial, t, bank, meta, prep, start, d)
        summary = a.finish_diagnostics(d)
        self.assertEqual(summary['warmup']['outer_attempts'], 1)
        self.assertEqual(summary['production']['outer_attempts'], 3)
        self.assertEqual(summary['production']['physical_decisions'], 1)
        self.assertEqual(summary['production']['candidate_factor_summary']['bath']['count'], 1)
        self.assertLess(summary['production']['candidate_factor_summary']['bath']['mean'], 0)

    def test_exact_inventory_and_incomplete_gate_before_journal_hash(self):
        config = fixture()[0]; config['physical'] = copy.deepcopy(a.scalar.PHYSICAL)
        config['allocation'] = dict(warmup_blocks=512, production_blocks=4096)
        config['jobs'] = [dict(id=n, context_index=0, arm=arm, initialization=start, stream=s)
            for n, (arm, start, s) in enumerate(itertools.product(a.ARMS, a.STARTS, range(4)))]
        a.validate_inventory(config); config['jobs'].pop()
        with self.assertRaises(ValueError): a.validate_inventory(config)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root/'execution').mkdir()
            (root/'execution-plan.json').write_text(json.dumps(dict(jobs=[{}]*24)))
            status = dict(complete=False, passed=False, failure=None, active={'id': 'running'},
                          unstarted=[], completed=[], plan_sha256='unused')
            for name in ('status', 'summary'): (root/'execution'/f'{name}.json').write_text(json.dumps(status))
            with mock.patch.object(a, 'sha', side_effect=AssertionError('No hashing before complete gate')):
                with self.assertRaises(ValueError): a.bind_complete_inputs(root, config)


if __name__ == '__main__':
    unittest.main()
