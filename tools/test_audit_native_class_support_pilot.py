"""Synthetic proposal records and mocked stage authority; no protein queries."""
import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

import audit_native_class_support_pilot as worker
import native_class_line_algebra_reference as algebra
import native_class_line_reference as line
from native_contact_regions import make_pose
import test_native_class_line_algebra_reference as fixtures


def full_details(trace, result, law):
    value = copy.deepcopy(trace); value.pop('trace_format')
    for axis, reference in zip(value['axes'], result['axes']):
        axis['components'] = []
        means, sigmas = law.conditional(np.asarray(value['raw_coordinates']), axis['axis'])
        for k in range(len(law.weights)):
            branches = []
            for j, channel in enumerate(reference['channels']):
                branch = line.conditional_branch(reference['hard_free_intervals'], channel['intervals'],
                    means[k], sigmas[k], law.floor, value['raw_coordinates'][axis['axis']])
                branch['fallback_target'] = branch.pop('fallback'); branch['channel'] = j
                branches.append(branch)
            axis['components'].append(dict(component=k, conditional_mean=float(means[k]), conditional_sigma=float(sigmas[k]),
                channels=branches, channel_mixture_multiplier=sum(c['probability']*b['multiplier'] for c, b in zip(law.channels, branches))))
    return value


def case(u=None, components=None):
    args, recon, law = fixtures.case(beta=1., components=components)
    u = np.asarray([0.]*6 if u is None else u, float)
    trace, result = fixtures.compact(recon, u)
    raw, position, rotation, jac = law.decode(u)
    hard = recon.hard_valid(position, rotation)
    capture = bool(np.linalg.norm(position-args[2]['capture_center']) <= args[2]['capture_radius'])
    row = dict(kind='fresh', id=0, latent=u.tolist(), latent_radius=float(np.linalg.norm(u)), raw_coordinates=raw.tolist(),
        pose=make_pose(position, rotation), shell_valid=bool(np.linalg.norm(u) <= law.radius), capture_valid=capture,
        hard_valid=hard, log_proposal_density=result['log_density'], baseline_log_density=result['baseline_log_density'],
        log_physical_jacobian=jac, backmap_error=0., width_contacts=[[]],
        draw=dict(conditional=False, original_latent=u.tolist(), component=None),
        density_details=full_details(trace, result, law), draw_cpu_seconds=.1, density_cpu_seconds=.2, observer_cpu_seconds=.3,
        native_decision=None, exclusion_contact_by_anchor=None)
    inventory = [dict(index=k, bank='original' if k < 92 else 'added', training_id=None if k < 92 else 'toy',
                      latent_sigma=None if k < 92 else .15) for k in range(len(law.weights))]
    return row, dict(law=law, guide=args[1], config=args[2], inventory=inventory, recon=recon), trace, result


def budget(events=None):
    return worker.endpoints.Budget(dict(cpu_seconds=60, wall_seconds=120, max_rows=128, max_record_bytes=2**20,
        max_capture_queries=128, max_atomic_queries=128, max_native_queries=128, max_contact_queries=128,
        max_full_geometry_queries=16, max_axis_queries=48), (events.append if events is not None else lambda _: None))


def empty_classification():
    return dict(native_any=False, matched_anchor_indices=[], matches=[], native_anchor_count=0,
                registry_consistent_triangle=False, registry_consistent_triangles=[])


def attach_empty_native(row, context):
    count = len(context['config']['fixed_poses'])
    row['native_decision'] = dict(native_any=False, matched_anchor_indices=[], per_anchor=[
        dict(anchor_index=i, matched_motif_ids=[], matches=[]) for i in range(count)])
    row['exclusion_contact_by_anchor'] = [False]*count
    context.update(observer=mock.Mock(classify=mock.Mock(return_value=empty_classification())),
        contact=mock.Mock(classify=mock.Mock(return_value=dict(exclusion_contact=False,
            anchors=[dict(exclusion_contact=False) for _ in range(count)]))))


class PilotWorkerTests(unittest.TestCase):
    def test_all_116_components_three_axes_five_channels_and_full_record_tamper(self):
        row, context, _, full = case(components=116)
        with mock.patch.object(worker.endpoints, 'EndpointCore', side_effect=AssertionError('No algebra geometry')):
            result = worker.algebra_row(row, 0, context)
        self.assertEqual(result['component_branches'], 116*3*5)
        self.assertAlmostEqual(result['log_proposal_density'], full['log_density'], places=12)
        self.assertIsNone(result['selected'])
        for change in ('last_component', 'q', 'physical_schema', 'impossible_gaussian_branch', 'missing_width_sentinel'):
            bad = copy.deepcopy(row)
            if change == 'last_component': bad['density_details']['axes'][2]['components'][115]['channels'][4]['effective_mass'] *= .5
            elif change == 'q': bad['log_proposal_density'] += 1.
            elif change == 'physical_schema': bad['density_details']['trace_format'] = 'class-line-compact-v1'
            elif change == 'impossible_gaussian_branch': bad['draw']['component'] = 0
            else: bad['width_contacts'] = []
            with self.subTest(change=change), self.assertRaises(ValueError): worker.algebra_row(bad, 0, context)
        exterior, context, _, _ = case([8., 0., 0., 0., 0., 0.])
        with self.assertRaisesRegex(ValueError, 'Uniform draw outside'): worker.validate_row(exterior, 0, context['law'])

    def test_component_specific_class_H_and_unconditional_fallback_full_sum(self):
        args, recon, _ = fixtures.case(axes=(0,), beta=1.)
        args[0]['gaussian_chart']['covariances'] = [np.eye(6).tolist()]
        args[0]['gaussian_chart']['means'] = [[0.]*6]
        args[1]['minimum_conditional_mass'] = 1e-6
        args[1]['gaussian_components'] = [dict(weight=w, mean=[m, 0., 0., 0., 0., 0.], covariance=np.eye(6).tolist())
            for w, m in ((.6, 0.), (.2, 5.), (.2, 20.))]
        recon = line.Reconstructor(*args); law = algebra.AlgebraLaw(*args[:3]); u = np.zeros(6)
        trace, _ = fixtures.compact(recon, u)
        trace['axes'][0].update(hard_free_intervals=[line.interval(-1., 1.)],
            native_intervals=[line.interval(-.1, .1)], exclusion_contact_intervals=[line.interval(-2., 2.)])
        trace = fixtures.rebuild_saved_channels(law, u, trace); evaluated = law.evaluate_saved(u, trace)
        self.assertEqual([algebra.FALLBACKS[v] for v in evaluated['axes'][0]['channels'][1]['fallback_indices']],
                         ['class', 'hard_free', 'unconditional'])
        actual = full_details(trace, evaluated, law)
        worker.check_full_branches(actual, evaluated, law)
        scalar = .5*math.exp(-law.logvolume)
        for k, logg in enumerate(law.gaussian_logs(u)):
            scalar += .5*math.exp(logg)*sum(c['probability']*actual['axes'][0]['components'][k]['channels'][j]['multiplier']
                                              for j, c in enumerate(law.channels))
        self.assertAlmostEqual(math.exp(evaluated['log_density']), scalar, places=14)
        # All branches are evaluated before component weights enter the mixture.
        changed = copy.deepcopy(args[1]); changed['gaussian_components'][0]['weight'] *= .75
        changed_law = algebra.AlgebraLaw(args[0], changed, args[2]); changed_trace = copy.deepcopy(trace)
        changed_trace['baseline_log_density'] = math.log(.5*math.exp(-changed_law.logvolume)
            +.5*sum(math.exp(v) for v in changed_law.gaussian_logs(u)))
        other = changed_law.evaluate_saved(u, changed_trace)
        for a, b in zip(evaluated['axes'][0]['channels'], other['axes'][0]['channels']):
            np.testing.assert_array_equal(a['fallback_indices'], b['fallback_indices'])
            np.testing.assert_array_equal(a['effective_masses'], b['effective_masses'])

    def test_every_exterior_and_saved_invalid_attempt_gets_atomic_check(self):
        row, context, _, _ = case()
        # Keep a genuine chart pose but exclude it with the capture ball.
        context['config'] = dict(context['config'], capture_center=[1000., 0., 0.], capture_radius=.01)
        row.update(capture_valid=False, hard_valid=True)
        context['core'] = mock.Mock(hard_valid=mock.Mock(return_value=True))
        context['observer'] = mock.Mock(classify=mock.Mock(side_effect=AssertionError('Exterior native query')))
        context['contact'] = mock.Mock(classify=mock.Mock(side_effect=AssertionError('Exterior contact query')))
        calls = budget(); label = worker.label_row(row, 0, context, calls)
        self.assertTrue(label['hard_valid']); self.assertFalse(label['valid']); self.assertIsNone(label['native'])
        context['core'].hard_valid.assert_called_once(); self.assertEqual(calls.calls['atomic'], 1)
        row['hard_valid'] = False; context['core'].hard_valid.return_value = False
        self.assertFalse(worker.label_row(row, 0, context, calls)['hard_valid'])
        self.assertEqual(calls.calls['atomic'], 2)
        row['hard_valid'] = True
        with self.assertRaisesRegex(ValueError, 'validity'): worker.label_row(row, 0, context, calls)
        row['hard_valid'] = False; row['native_decision'] = {}
        with self.assertRaisesRegex(ValueError, 'Invalid/exterior'): worker.label_row(row, 0, context, calls)

    def test_complete_endpoint_labels_orthant_and_selected_channel(self):
        row, context, trace, _ = case()
        context['core'] = mock.Mock(hard_valid=mock.Mock(return_value=True)); row['hard_valid'] = True
        attach_empty_native(row, context)
        evaluation = context['law'].evaluate_saved(row['latent'], trace)
        row['draw'] = fixtures.selected_draw(context['law'], row['latent'], trace, evaluation, channel=0)
        result = worker.label_row(row, 0, context, budget())
        self.assertTrue(result['valid']); self.assertFalse(result['native']); self.assertFalse(result['contact'])
        self.assertEqual(result['orthant'], 63); self.assertTrue(result['selected_channel_match'])
        row['native_decision']['per_anchor'][0]['matched_motif_ids'] = [99]
        with self.assertRaisesRegex(ValueError, 'motif union'): worker.label_row(row, 0, context, budget())

    def test_selected_unpruned_geometry_checks_full_q_draw_and_endpoint(self):
        row, context, trace, full = case()
        evaluation = context['law'].evaluate_saved(row['latent'], trace)
        row['draw'] = fixtures.selected_draw(context['law'], row['latent'], trace, evaluation, channel=0)
        coordinate = context['law'].raw(row['latent'])[0]
        axis = full['axes'][0]
        label = dict(valid=line.contains(axis['hard_free_intervals'], coordinate),
                     native=line.contains(axis['native_intervals'], coordinate),
                     contact=line.contains(axis['exclusion_contact_intervals'], coordinate))
        events = []; calls = budget(events)
        result = worker.geometry_row(row, label, context, calls)
        self.assertTrue(result['full_density_certified']); self.assertTrue(result['draw_path_certified'])
        self.assertEqual(dict(calls.calls), dict(full_geometry=1, axis=3))
        self.assertEqual(len([e for e in events if e['state'] == 'query_complete']), 4)
        bad = copy.deepcopy(row); bad['log_proposal_density'] += .1
        with self.assertRaisesRegex(ValueError, 'full q'): worker.geometry_row(bad, label, context, budget())
        bad = dict(label, valid=not label['valid'])
        with self.assertRaisesRegex(ValueError, 'H/endpoint'): worker.geometry_row(row, bad, context, budget())
        bad = copy.deepcopy(row); bad['draw']['fallback_target'] = 'invented'
        with self.assertRaisesRegex(ValueError, 'fallback'): worker.geometry_row(bad, label, context, budget())

    def test_query_failure_retains_begun_axis_and_never_retries(self):
        row, context, _, _ = case(); events = []; calls = budget(events)
        original = context['recon'].reconstruct_axis
        with mock.patch.object(context['recon'], 'reconstruct_axis', side_effect=ValueError('synthetic geometry failure')):
            with self.assertRaisesRegex(ValueError, 'synthetic geometry'): worker.geometry_row(row, {}, context, calls)
        self.assertEqual(dict(calls.calls), dict(full_geometry=1, axis=1))
        self.assertEqual([e['state'] for e in events], ['query_begin', 'query_begin'])
        self.assertEqual(context['recon'].reconstruct_axis, original)

    def test_mocked_stage_streams_all128_and_preserves_failure_prefix(self):
        row, context, _, _ = case(); context['core'] = mock.Mock(hard_valid=mock.Mock(return_value=False))
        context['observer'] = mock.Mock(classify=mock.Mock(side_effect=AssertionError('Invalid native query')))
        row.update(hard_valid=False, native_decision=None, exclusion_contact_by_anchor=None)
        for corrupt in (False, True):
            with self.subTest(corrupt=corrupt), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary); folder = root/'analysis/r00'; folder.mkdir(parents=True)
                data = root/'queries/r00'; data.mkdir(parents=True)
                samples = [dict(row, id=i) for i in range(128)]
                if corrupt: samples[2]['hard_valid'] = True
                raw = [('\n'.join(json.dumps(r) for r in samples)+'\n').encode(),
                       ('\n'.join(json.dumps(dict(ordinal=i, id=i, kind='fresh', state='begin')) for i in range(128))+'\n').encode()]
                for name, value in zip(('samples', 'attempts'), raw): (data/(name+'.jsonl')).write_bytes(value)
                ctx = dict(context, root=data, summary={name+'_sha256': hashlib.sha256(value).hexdigest() for name, value in zip(('samples', 'attempts'), raw)},
                           classifier_binding={}, shape_witness={}, wall_certificate={})
                prior = dict(protocol_sha256='d'*64, population='r00', rows=[dict(id=i, sample_record_sha256=hashlib.sha256(linebytes).hexdigest())
                    for i, linebytes in enumerate(raw[0].splitlines(keepends=True))])
                pop = dict(id='r00', selected_ids=list(range(16)))
                protocol = dict(populations=[pop], source_sha256={}, runtime={}, phase_limits=dict(labels=dict(
                    cpu_limit_seconds=60, wall_limit_seconds=120, max_record_bytes=2**20)))
                with mock.patch.object(worker, 'load_protocol', return_value=(protocol, {})), \
                     mock.patch.object(worker.stage, 'verify_live_authority'), \
                     mock.patch.object(worker.stage, 'predecessor', return_value=({}, prior)), \
                     mock.patch.object(worker, 'population_context', return_value=ctx), \
                     mock.patch.object(worker, 'observer_context'), \
                     mock.patch.object(worker.streaming, 'runtime_identity', return_value={}):
                    if corrupt:
                        with self.assertRaisesRegex(ValueError, 'validity'): worker.run(root/'protocol.json', 'd'*64, 'r00', 'labels')
                        failure = worker.read(folder/'labels.failure.json')
                        self.assertEqual(failure['active_id'], 2); self.assertEqual(failure['completed_ids'], [0, 1])
                        self.assertEqual(failure['counts']['atomic'], 3); self.assertEqual(failure['retries'], 0)
                        self.assertFalse((folder/'labels.json').exists())
                    else:
                        result = worker.run(root/'protocol.json', 'd'*64, 'r00', 'labels')
                        self.assertEqual(len(result['rows']), 128); self.assertEqual(result['counts']['atomic'], 128)
                        self.assertEqual(result['counts']['capture'], 128); self.assertFalse(any(r['valid'] for r in result['rows']))
                        self.assertEqual(result['rows'][0]['sample_record_sha256'], prior['rows'][0]['sample_record_sha256'])
                        self.assertFalse((folder/'labels.failure.json').exists())
                    with self.assertRaisesRegex(ValueError, 'no resume'): worker.run(root/'protocol.json', 'd'*64, 'r00', 'labels')


if __name__ == '__main__': unittest.main()
