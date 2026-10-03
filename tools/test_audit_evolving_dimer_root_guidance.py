"""Synthetic sphere-only controls; never read campaign or protein inputs."""
import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import numpy as np

import audit_evolving_dimer_root_guidance as a


def pose(x):
    return dict(position=[float(x), 0., 0.], orientation=[1., 0., 0., 0.])


def fixture():
    shape = dict(atoms=[dict(center=[0., 0., 0.], radius=1.)])
    state = [pose(0), pose(2.2), pose(4.4)]
    case = dict(root=1, child=2, anchor=0)
    geometry = a.DimerGeometry(shape, state, .5, 20.)
    counter = a.CountOracle([[0., 0., 0.]], [1.5])
    points = np.array([[.5, 0., 0.], [1., 0., 0.], [1.4, 0., 0.]])
    old = state[1:]
    new = [pose(2.8), pose(5.6)]
    def frame(members):
        edges = [a.relative_pose(state[0], members[0]), a.relative_pose(*members)]
        rebuilt_root = a.compose_pose(state[0], edges[0])
        rebuilt = [rebuilt_root, a.compose_pose(rebuilt_root, edges[1])]
        def full(poses):
            return a.feasibility(geometry, case, poses, geometry.fingerprint([1, 2], poses))
        value = dict(recovered_edges=edges, reconstructed_members=rebuilt,
            reconstructed_feasibility=full(rebuilt),
            reconstructed_root=a.root_geometry(geometry, case, rebuilt[0]),
            internal_relative=a.internal_geometry(geometry, edges[1]))
        for i, name in [(0, 'root_guidance'), (1, 'guidance')]:
            world = [state[0], members[0]] if i == 0 else members
            k = counter.relative(points, edges[i])
            value[name] = dict(relative_count=k, recovered_count=k,
                world_count=counter.world(points, world), reconstructed_world_count=k)
        return full(members), value
    source, source_frame = frame(old)
    final, final_frame = frame(new)
    roots = []
    for i, x in enumerate([.5, 3.5, 2.8], 1):
        edge = pose(x)
        r = dict(index=i, draw=dict(proposed_relative_pose=edge), world_pose=edge,
                 feasibility=a.root_geometry(geometry, case, edge))
        if a.root_ok(r['feasibility']): r['guidance_count'] = counter.relative(points, edge)
        roots.append(r)
    internals = []
    for i, x in enumerate([1., 3.5, 2.8], 1):
        edge = pose(x)
        r = dict(index=i, draw=dict(proposed_relative_pose=edge),
                 feasibility=a.internal_geometry(geometry, edge))
        if a.internal_ok(r['feasibility']): r['guidance_count'] = counter.relative(points, edge)
        internals.append(r)
    def guide(queries):
        return dict(m=4, point_count=3, old_count=2, new_count=1, threshold=1,
            integer_draws=[0, 1, 1, 0], aux_log_correction=4*(math.log1p(2)-math.log1p(1)),
            count_queries=queries, point_tests=3*queries)
    root_guide, internal_guide = guide(10), guide(9)
    f = -2.
    correction = f+root_guide['aux_log_correction']+internal_guide['aux_log_correction']
    leg = dict(gained=2, lost=3, raw_points=10, retained_points=5,
        retained_cells=1, created_cells=1, log_weight=-math.log1p(1/64))
    aggregate = {k: 2*v for k, v in leg.items()}
    logalpha = correction+aggregate['log_weight']
    proposed = dict(root=new[0], child=new[1], diagnostics=dict(log_reverse_forward=f))
    row = dict(kind='factorized_dimer', block=1, members=[1, 2], anchor=0, old=old,
        proposal=dict(caps=dict(root=32, internal=32, joint=1), order='root_first', old=old,
            members=[1, 2], anchor_label=0, source_feasibility=source, source_frame=source_frame,
            root_guidance=root_guide, guidance=internal_guide, status='candidate', candidate=proposed,
            attempts=[dict(index=1, status='candidate', root_draws=roots, internal_draws=internals,
                proposed=new, final_feasibility=final, frame=final_frame)]),
        proposed=new, status='completed', complete_log_correction=correction,
        bath=dict(aggregate=aggregate, legs=[leg, copy.deepcopy(leg)]), log_acceptance_ratio=logalpha,
        log_u=-1., accepted=-1. < min(0., logalpha))
    config = dict(physical=dict(activity=.0275, lambda_ratio=64.))
    return row, config, geometry, counter, points, case


class TestRootGuidanceAudit(unittest.TestCase):
    def test_protocol_is_fixed_before_data(self):
        p = a.audit_protocol()
        self.assertEqual(p['chains'], 32)
        self.assertEqual(p['maximum_count_queries'], 32*(32+32+8+8))
        self.assertEqual(p['maximum_point_membership_tests'], 2560*16384)
        self.assertFalse(p['geometric_event']['replace_skipped'])

    def test_two_nonzero_auxiliaries_and_exact_count_frames(self):
        row, cfg, geom, counter, points, case = fixture()
        r = a.validate_scalar_record(row, cfg, len(points))
        self.assertGreater(r['root_aux_log'], 0)
        self.assertGreater(r['internal_aux_log'], 0)
        q = a.audit_first_record(row, geom, counter, points, case, a.Checks())
        self.assertEqual(q['count_queries'], 19)
        self.assertEqual(q['point_membership_tests'], 57)
        self.assertEqual(q['root_raw_trials'], 3)
        self.assertEqual(q['internal_raw_trials'], 3)

    def test_auxiliary_omitted_or_doubled_is_fatal(self):
        row, cfg, *_ = fixture()
        for scale in (0, 2):
            broken = copy.deepcopy(row)
            delta = (scale-1)*row['proposal']['root_guidance']['aux_log_correction']
            broken['complete_log_correction'] += delta
            broken['log_acceptance_ratio'] += delta
            broken['accepted'] = broken['log_u'] < min(0., broken['log_acceptance_ratio'])
            with self.assertRaisesRegex(ValueError, 'each auxiliary once'):
                a.validate_scalar_record(broken, cfg, 3)

    def test_skipped_hard_or_contact_count_is_not_zero(self):
        row, cfg, *_ = fixture()
        for stage, index in [('root', 0), ('internal', 0), ('internal', 1)]:
            broken = copy.deepcopy(row)
            broken['proposal']['attempts'][0][stage+'_draws'][index]['guidance_count'] = 0
            with self.assertRaisesRegex(ValueError, 'Skipped count'):
                a.validate_scalar_record(broken, cfg, 3)

    def test_scalar_consistent_false_count_fails_independent_geometry(self):
        row, cfg, geom, counter, points, case = fixture()
        # Failed-threshold root keeps the same decision/counters despite a false count.
        row['proposal']['attempts'][0]['root_draws'][1]['guidance_count'] = 1
        row['proposal']['root_guidance']['threshold'] = 2
        row['proposal']['root_guidance']['integer_draws'] = [2]*4
        with self.assertRaises(ValueError):
            a.audit_first_record(row, geom, counter, points, case, a.Checks())

    def test_source_outside_has_no_threshold_or_replacement(self):
        row, cfg, *_ = fixture(); p = row['proposal']
        p['source_feasibility']['internal_exclusion_contact'] = False
        p['status'] = 'source_outside_domain'; p['attempts'] = []; p['candidate'] = None
        row['status'] = 'proposal_self_loop'; row['accepted'] = False
        for name in ('root_guidance', 'guidance'):
            p[name].update(integer_draws=[], threshold=None, new_count=None, aux_log_correction=None,
                           count_queries=4, point_tests=12)
        for name in ('proposed', 'complete_log_correction', 'bath', 'log_acceptance_ratio', 'log_u'):
            del row[name]
        result = a.validate_scalar_record(row, cfg, 3)
        self.assertEqual(result['status'], 'source_outside_domain')
        p['root_guidance']['threshold'] = 0
        with self.assertRaisesRegex(ValueError, 'consumed a threshold'):
            a.validate_scalar_record(row, cfg, 3)

    def test_root_cap_preserves_all_hard_rejections(self):
        row, cfg, geom, counter, points, case = fixture(); p = row['proposal']
        attempt = p['attempts'][0]
        trial = copy.deepcopy(attempt['root_draws'][0])
        attempt.update(status='root_cap_exhausted',
            root_draws=[dict(copy.deepcopy(trial), index=i) for i in range(1, 33)],
            internal_draws=[], proposed=None, frame=None, final_feasibility=None)
        p.update(status='cap_exhausted', candidate=None)
        for name in ('root_guidance', 'guidance'):
            p[name].update(new_count=None, aux_log_correction=None, count_queries=4, point_tests=12)
        row.update(status='proposal_self_loop', accepted=False)
        for name in ('proposed', 'complete_log_correction', 'bath', 'log_acceptance_ratio', 'log_u'):
            del row[name]
        self.assertEqual(a.validate_scalar_record(row, cfg, 3)['stage_status'], 'root_cap_exhausted')
        audited = a.audit_first_record(row, geom, counter, points, case, a.Checks())
        self.assertEqual(audited['root_raw_trials'], 32)
        self.assertEqual(audited['count_queries'], 8)

    def test_full_chain_audits_first_event_once_and_every_scalar_attempt(self):
        row, cfg, _, _, points, case = fixture(); p = row['proposal']
        p['source_feasibility']['internal_exclusion_contact'] = False
        p.update(status='source_outside_domain', attempts=[], candidate=None)
        row.update(status='proposal_self_loop', accepted=False)
        for name in ('root_guidance', 'guidance'):
            p[name].update(integer_draws=[], threshold=None, new_count=None, aux_log_correction=None,
                           count_queries=4, point_tests=12)
        for name in ('proposed', 'complete_log_correction', 'bath', 'log_acceptance_ratio', 'log_u'):
            del row[name]
        cfg.update(contexts=[case], allocation=dict(warmup_blocks=512, production_blocks=4096))
        job = dict(arm='root_m4', context_index=0, initialization='source', stream=0, id=0)
        initial = row['old']; bank = dict(synthetic='no files or new clouds')
        contract = dict(schema='evolving-dimer-root-guidance-v1', cloud=bank,
            cloud_reuse='identical frozen body-frame points; no additional cloud draws',
            root=dict(m=4, point_frame='fixed_anchor_body', anchor_label=0, mobile_label=1,
                threshold_rng_role='root_m4/root_threshold'),
            internal=dict(m=4, point_frame='mobile_root_body', root_label=1, child_label=2,
                threshold_rng_role='m4/threshold'), matched_control_arm='m4',
            proposal_rng_role='m4/proposal', bath_rng_role='m4/bath', accept_rng_role='m4/accept',
            local_rng_roles='unchanged shared local/{attempt}/{proposal,bath,accept}',
            physical_decisions_per_candidate=1)
        first = dict(kind='initial', block=0, job=job, selected=initial, conditional_target=True,
                     sampler_cpu_seconds=0., cloud=bank, guidance_contract=contract)
        a.validate_initial_contract(first, case, bank)
        bad = copy.deepcopy(first); bad['guidance_contract']['root']['threshold_rng_role'] = 'm4/threshold'
        with self.assertRaisesRegex(ValueError, 'contract differs'):
            a.validate_initial_contract(bad, case, bank)
        def rows():
            yield first
            for block in range(1, 4609):
                for i, slot in enumerate([0, 1, 0, 1]):
                    yield dict(kind='local', block=block, attempt=i, member=slot+1,
                        old=initial[slot], retained=initial, status='hard_rejected', accepted=False,
                        sampler_cpu_seconds=float(block))
                yield dict(row, block=block, retained=initial, sampler_cpu_seconds=float(block))
                yield dict(kind='retained_block', block=block, production=block>512, selected=initial,
                    raw=0, retained=0, sampler_cpu_seconds=float(block),
                    counts=dict(local_attempted=4*block, local_accepted=0, dimer_attempted=block,
                                dimer_accepted=0, dimer_self_loop=block))
        # This is a scalar wrapper control. Independent geometry is tested above.
        with mock.patch.object(a, 'audit_first_record', return_value=dict(count_queries=8)) as first_check:
            report = a.audit_chain(cfg, job, initial, None, rows(), bank=bank,
                                  geometry=None, counter=None, points=points, checks=a.Checks())
        self.assertEqual(first_check.call_count, 1)
        self.assertEqual(report['scalar_counts']['scalar_dimer_attempts'], 4608)
        self.assertEqual(report['scalar_counts']['scalar_local_attempts'], 18432)

    def test_hidden_retry_and_wrong_query_count(self):
        row, cfg, *_ = fixture()
        r = copy.deepcopy(row)
        r['proposal']['attempts'][0]['root_draws'].append(copy.deepcopy(r['proposal']['attempts'][0]['root_draws'][-1]))
        with self.assertRaisesRegex(ValueError, 'Hidden retry'):
            a.validate_scalar_record(r, cfg, 3)
        row['proposal']['guidance']['count_queries'] += 1
        with self.assertRaisesRegex(ValueError, 'accounting'):
            a.validate_scalar_record(row, cfg, 3)

    def test_zero_count_has_a_nonzero_destination_penalty(self):
        row, cfg, *_ = fixture(); d = row['proposal']['root_guidance']
        d.update(old_count=0, threshold=0, integer_draws=[0]*4,
                 aux_log_correction=-4*math.log1p(1))
        row['proposal']['source_frame']['root_guidance'] = dict.fromkeys(dct_keys(), 0)
        # With threshold zero, the previous first geometric pass must end this stage.
        trial = row['proposal']['attempts'][0]['root_draws'][1]
        trial['guidance_count'] = 1
        del row['proposal']['attempts'][0]['root_draws'][2:]
        d.update(count_queries=10, point_tests=30)
        f = row['proposal']['candidate']['diagnostics']['log_reverse_forward']
        c = f+row['proposal']['guidance']['aux_log_correction']+d['aux_log_correction']
        row['complete_log_correction'] = c
        row['log_acceptance_ratio'] = c+row['bath']['aggregate']['log_weight']
        row['accepted'] = row['log_u'] < min(0., row['log_acceptance_ratio'])
        # One fewer count call because threshold0 succeeds at the first hard pass.
        d.update(count_queries=9, point_tests=27)
        result = a.validate_scalar_record(row, cfg, 3)
        self.assertLess(result['root_aux_log'], 0)

    def test_zero_F_acceptance_and_bath_leg_corruption(self):
        row, cfg, *_ = fixture()
        row['proposal']['candidate']['diagnostics']['log_reverse_forward'] = '-inf'
        row['complete_log_correction'] = row['log_acceptance_ratio'] = '-inf'
        row['accepted'] = False
        self.assertEqual(a.validate_scalar_record(row, cfg, 3)['acceptance_probability'], 0)
        row['bath']['legs'][0]['lost'] += 1
        with self.assertRaises(ValueError):
            a.validate_scalar_record(row, cfg, 3)

    def test_bound_cloud_affine_reconstruction_no_thinning(self):
        with tempfile.TemporaryDirectory() as temp:
            raw = np.full((16384, 3), .5, dtype='<f8').tobytes()
            meta = dict(raw_count=16384, low=[-1., -2., -3.], high=[3., 4., 5.], kept_indices=[0, 3])
            bank = {}
            for name, data in [('raw', raw), ('metadata', json.dumps(meta).encode())]:
                path = Path(temp)/name; path.write_bytes(data)
                bank[name] = dict(path=str(path), sha256=hashlib.sha256(data).hexdigest())
            np.testing.assert_array_equal(a.retained_points(bank), [[1., 1., 1.], [1., 1., 1.]])
            Path(bank['raw']['path']).write_bytes(raw[:-1])
            with self.assertRaisesRegex(ValueError, 'Changed bound'):
                a.retained_points(bank)


def dct_keys():
    return ['relative_count', 'recovered_count', 'world_count', 'reconstructed_world_count']


if __name__ == '__main__':
    unittest.main()
