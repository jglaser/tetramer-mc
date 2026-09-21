#!/usr/bin/env python3
"""Synthetic audit regressions; no protein trajectory is launched or inspected."""
import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from analyze_mobile_posterior_pilot import (
    ChartAudit, audit_densities, audit_single_body_gate, density_record,
    environment_episodes, full_capture_log_density, pair_changes,
    proposal_kernel, update_nonspecific_graph, update_registry,
    RECOVERY_COORDINATES, RECOVERY_MARKER, RECOVERY_SCHEMA, validate_reference,
)


def pose(x=0., y=0., z=0.):
    return dict(position=[x, y, z], orientation=[1., 0., 0., 0.])


def model():
    return dict(angular_length=1.5, weights=[.3, .7],
        anchors=[dict(position=[0., 0., 0.], rotation=np.eye(3).tolist()),
                 dict(position=[1., .5, -.2], rotation=np.eye(3).tolist())],
        means=[[0.]*6, [.1, -.2, .3, .2, -.1, .15]],
        covariances=[np.diag([.2, .3, .4, .15, .1, .2]).tolist(),
                     np.diag([.7, .4, .5, .3, .4, .2]).tolist()])


def posterior_record(correlation):
    audit = ChartAudit(model())
    a, b = 1, 0
    z, noise = np.asarray([.2, -.3, .1, .3, -.1, .4]), np.asarray([-.4, .1, .2, -.2, .3, .1])
    sine = math.sqrt(1-correlation**2)
    target, inverse_noise = correlation*z+sine*noise, sine*z-correlation*noise
    old, old_j = audit.decode(a, z)
    new, new_j = audit.decode(b, target)
    old_g, new_g = audit.full_gaussian(old), audit.full_gaussian(new)
    source, destination = audit.gaussian(a, old), audit.gaussian(b, new)
    forward = math.log(audit.weights[a])+source-old_g
    reverse = math.log(audit.weights[b])+destination-new_g
    labels = reverse+math.log(audit.weights[a])-forward-math.log(audit.weights[b])
    jacobian = new_j-old_j
    auxiliary = .5*(noise@noise-inverse_noise@inverse_noise)
    info = dict(branch='involution', kernel='frozen-posterior', source_law='posterior',
        moving_index=0, anchor_index=1, correlation=correlation,
        trace=dict(source=a, target=b, noise=noise.tolist()),
        step=dict(pose=new, source_latent=z.tolist(), target_latent=target.tolist(),
            inverse_trace=dict(source=b, target=a, noise=inverse_noise.tolist()),
            log_extended_jacobian=jacobian, log_auxiliary_ratio=auxiliary,
            log_correction=jacobian+auxiliary),
        selected_source_log_density=source, selected_target_log_density=destination,
        full_old_gaussian_log_density=old_g, full_new_gaussian_log_density=new_g,
        source_log_probability=forward, inverse_source_log_probability=reverse,
        label_log_reverse_forward=labels, expanded_log_reverse_forward=jacobian+auxiliary+labels,
        log_reverse_forward=old_g-new_g)
    anchor = pose(3., -2., 1.)
    return dict(kernel='frozen-posterior', accepted=True, proposal=info,
        old=audit.absolute(old, anchor), new=audit.absolute(new, anchor),
        anchor=anchor, old_relative=old, new_relative=new)


def valid_move(correction=0.):
    gate = dict(gained=0, lost=0, raw_points=0, retained_points=0,
        log_weight=0., envelope_volume=0., retained_cells=0, created_cells=0)
    return dict(kind='global', moving_index=0, old_pose=pose(), proposed_pose=pose(1.),
        retained_pose=pose(1.), hard_valid=True, accepted=True, gate=gate,
        log_acceptance=min(0., correction), proposal=dict(branch='uniform', anchor_index=1,
            kernel='full-mixture-capture', log_reverse_forward=correction))


def reference_fixture(root, wrong_atom_order=False):
    """A tiny frozen helper/data fixture exercises recovery without real audits."""
    campaign, reference = root/'campaign', root/'recovered-reference'
    frozen = campaign/'provenance/reference'
    frozen.mkdir(parents=True)
    reference.mkdir()
    coordinates = dict(positions=[[2., 0., 0.], [0., 0., 0.]] if wrong_atom_order else [[0., 0., 0.], [2., 0., 0.]])
    coordinate_bytes = json.dumps(coordinates).encode()
    digest = lambda raw: hashlib.sha256(raw).hexdigest()
    helper = '''def read_json(path):
    return json.loads(Path(path).read_text())

def repaired_residue_labels(shape_path):
    shape = read_json(shape_path)
    source = PROJECT/'results/native-geometry-repair/rebuilt-hydrogens/heavy-coordinates.json'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == shape['source_coordinates_sha256']
    positions = np.asarray(read_json(source)['positions'])
    body = np.asarray([atom['center'] for atom in shape['atoms']])
    discrepancy = float(np.max(np.abs(positions-positions.mean(axis=0)-body)))
    if discrepancy > 1e-10:
        raise ValueError('Authoritative coordinate rows do not match shape atom ordering.')
    labels = np.arange(len(body))
    return labels, dict(available=True, source=str(source), source_sha256=shape['source_coordinates_sha256'],
        shape_sha256=hashlib.sha256(shape_path.read_bytes()).hexdigest(),
        maximum_coordinate_discrepancy_angstrom=discrepancy, residues=[{} for _ in body],
        atom_order_audit='Synthetic unchanged helper checked exact recentered rows.')
'''
    originals = {'scripts/analyze_depletion_mirror_benchmark.py': helper.encode(),
                 'results/c1c3-scaffold/motifs.json': b'{"frozen": true}'}
    for name, raw in originals.items():
        for folder in (frozen, reference):
            target = folder/name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
    monomer = campaign/'provenance/monomer-shape.json'
    monomer.write_text(json.dumps(dict(source_coordinates_sha256=digest(coordinate_bytes),
        atoms=[dict(center=[-1., 0., 0.]), dict(center=[1., 0., 0.])])))
    original_hashes = {name: digest(raw) for name, raw in originals.items()}
    manifest = dict(reference=str(frozen), observer_sha256='f'*64,
        input_sha256={**{'reference/'+name: value for name, value in original_hashes.items()},
                      'monomer-shape.json': digest(monomer.read_bytes())})
    (campaign/'manifest.json').write_text(json.dumps(manifest))
    added = reference/RECOVERY_COORDINATES
    added.parent.mkdir(parents=True)
    added.write_bytes(coordinate_bytes)
    marker = dict(schema=RECOVERY_SCHEMA, source_campaign=str(campaign),
        source_campaign_manifest_sha256=digest((campaign/'manifest.json').read_bytes()),
        source_frozen_reference=str(frozen), original_reference_sha256=original_hashes,
        monomer_shape=str(monomer), monomer_shape_sha256=digest(monomer.read_bytes()),
        added_coordinates=dict(path=RECOVERY_COORDINATES, sha256=digest(coordinate_bytes)))
    (reference/RECOVERY_MARKER).write_text(json.dumps(marker))
    return campaign, manifest, reference


class MobileAuditTests(unittest.TestCase):
    def test_reference_recovery_is_explicit_hash_bound_and_preserves_originals(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign, manifest, reference = reference_fixture(Path(tmp))
            frozen = Path(manifest['reference'])
            before = {str(path): path.read_bytes() for path in campaign.rglob('*') if path.is_file()}
            original = validate_reference(campaign, manifest, frozen)
            self.assertEqual(original['mode'], 'original-frozen-reference')
            recovered = validate_reference(campaign, manifest, reference)
            self.assertEqual(recovered['mode'], 'separate-reference-data-recovery')
            self.assertEqual(recovered['label_validation']['atoms'], 2)
            self.assertEqual(recovered['label_validation']['maximum_coordinate_discrepancy_angstrom'], 0.)
            self.assertEqual(recovered['source_frozen_reference'], str(frozen))
            self.assertEqual(before, {str(path): path.read_bytes() for path in campaign.rglob('*') if path.is_file()})
            self.assertFalse(any(path.name == '__pycache__' for path in reference.rglob('*')))

    def test_reference_recovery_rejects_changed_scripts_extra_data_or_coordinates(self):
        for change in ('script', 'extra', 'missing', 'coordinate', 'binding', 'symlink'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                campaign, manifest, reference = reference_fixture(Path(tmp))
                script = reference/'scripts/analyze_depletion_mirror_benchmark.py'
                if change == 'script':
                    script.write_text(script.read_text()+'\n# altered\n')
                elif change == 'extra':
                    (reference/'extra.json').write_text('{}')
                elif change == 'missing':
                    (reference/'results/c1c3-scaffold/motifs.json').unlink()
                elif change == 'coordinate':
                    (reference/RECOVERY_COORDINATES).write_text('{"positions": []}')
                elif change == 'binding':
                    path = reference/RECOVERY_MARKER
                    marker = json.loads(path.read_text())
                    marker['source_campaign_manifest_sha256'] = '0'*64
                    path.write_text(json.dumps(marker))
                else:
                    script.unlink()
                    script.symlink_to(Path(manifest['reference'])/'scripts/analyze_depletion_mirror_benchmark.py')
                with self.assertRaises(AssertionError):
                    validate_reference(campaign, manifest, reference)

    def test_reference_recovery_still_runs_unchanged_atom_order_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign, manifest, reference = reference_fixture(Path(tmp), wrong_atom_order=True)
            with self.assertRaisesRegex(ValueError, 'atom ordering'):
                validate_reference(campaign, manifest, reference)

    def test_preassociated_registry_changes_do_not_create_or_detach_body_bonds(self):
        initial = {(0, 1, 7), (1, 2, 8)}
        changed = update_registry(initial, {(0, 1, 9)}, {(0, 1, 9)}, {(0, 1)})
        self.assertEqual(changed, {(0, 1, 9), (1, 2, 8)})
        self.assertEqual(pair_changes(initial, changed), dict(formed_body_pairs=[],
            broken_body_pairs=[], changed_registry_body_pairs=[(0, 1)]))
        detached = update_registry(changed, set(), set(), {(0, 1)})
        self.assertEqual(pair_changes(changed, detached)['broken_body_pairs'], [(0, 1)])
        self.assertEqual(pair_changes(changed, detached)['formed_body_pairs'], [])
        # A preexisting label within retention but outside entry remains active.
        self.assertEqual(update_registry(initial, set(), {(0, 1, 7)}, {(0, 1)}), initial)

    def test_environment_episodes_start_with_actual_bound_neighbors(self):
        rows = [dict(sweep=i, native=dict(edges=edges)) for i, edges in enumerate(
            [[[0, 1]], [[0, 1]], [], [[0, 1]], [[0, 1]]])]
        result = environment_episodes(rows, 'native', 3)
        body0 = [e for e in result['episodes'] if e['body'] == 0]
        self.assertEqual([e['neighbors'] for e in body0], [[1], [], [1]])
        self.assertEqual(body0[0]['first_observed_sweep'], 0)
        self.assertEqual(body0[0]['last_observed_sweep'], 1)
        self.assertTrue(body0[-1]['right_censored'])
        self.assertEqual(result['observed_returns'], 2)

    def test_full_mixture_uses_actual_cube_and_half_open_support(self):
        eta, log_g = .1, -80.
        floor = math.log(eta)-3*math.log(8.)
        self.assertAlmostEqual(full_capture_log_density(log_g, pose(3.), [8.]*3, eta), floor)
        self.assertAlmostEqual(full_capture_log_density(log_g, pose(-4.), [8.]*3, eta), floor)
        for x in (4., -4.00001, 20.):
            self.assertAlmostEqual(full_capture_log_density(log_g, pose(x), [8.]*3, eta), math.log1p(-eta)+log_g)
        self.assertEqual(full_capture_log_density(log_g, pose(), [8.]*3, 0.), log_g)
        self.assertEqual(full_capture_log_density(log_g, pose(5.), [8.]*3, 1.), -math.inf)
        cfg = dict(box_lengths=[4.]*3, uniform_proposal_cube_lengths=[8.]*3, learned_uniform_weight=eta)
        audit = ChartAudit(model())
        old, new = pose(), pose(3.)
        old_q = full_capture_log_density(audit.full_gaussian(old), old, [8.]*3, eta)
        new_q = full_capture_log_density(audit.full_gaussian(new), new, [8.]*3, eta)
        record = dict(kernel='full-mixture-capture', old=old, new=new, old_relative=old, new_relative=new,
            accepted=False, proposal=dict(old_log_density=old_q, new_log_density=new_q, log_reverse_forward=old_q-new_q))
        result = audit_densities(model(), [record], cfg, audit)
        self.assertEqual(result['full_mixture_checks'], 1)
        wrong = copy.deepcopy(cfg)
        wrong['uniform_proposal_cube_lengths'] = wrong['box_lengths']
        with self.assertRaises(AssertionError):
            audit_densities(model(), [record], wrong, ChartAudit(model()))

    def test_density_records_snapshot_before_common_shift_mutates_state(self):
        audit = ChartAudit(model())
        state = [pose(1.), pose(2.), pose(3.)]
        move = valid_move()
        saved = density_record(0, 'full-mixture-capture', move, state, audit)
        before = copy.deepcopy(saved)
        for p in state:
            p['position'][0] += 20.
        move['proposed_pose']['position'][0] += 40.
        move['proposal']['anchor_index'] = 2
        self.assertEqual(saved, before)

    def test_posterior_label_ratios_maps_and_separate_uniform_counts(self):
        for c in (0., .9):
            record = posterior_record(c)
            uniform = copy.deepcopy(record)
            uniform['proposal'] = dict(branch='uniform', log_reverse_forward=0.)
            cfg = dict(frozen_posterior=dict(correlation=c))
            audit = ChartAudit(model())
            result = audit_densities(model(), [record, uniform], cfg, audit)
            self.assertEqual(result, dict(global_density_records=2, full_mixture_checks=0,
                posterior_checks=1, posterior_uniform_checks=1, map_checks=1))
            self.assertEqual(audit.checks['posterior_inverse_map_position'], 1)
            wrong = copy.deepcopy(record)
            wrong['proposal']['label_log_reverse_forward'] += .01
            with self.assertRaises(AssertionError):
                audit_densities(model(), [wrong], cfg, ChartAudit(model()))

    def test_zero_gate_acceptance_and_schema_are_checked(self):
        cfg = dict(reservoir_density=.2, poisson_lambda_ratio=4.)
        move = valid_move()
        self.assertEqual(audit_single_body_gate(move, cfg, ChartAudit(model())), 1)
        wrong = copy.deepcopy(move)
        wrong['accepted'] = False
        with self.assertRaises(AssertionError):
            audit_single_body_gate(wrong, cfg, ChartAudit(model()))
        for mutate in (
            lambda m: m['gate'].update(raw_points=-1),
            lambda m: m['gate'].update(retained_points=1),
            lambda m: m.update(hard_valid=False),
            lambda m: m.update(gate=None),
            lambda m: m['proposal'].update(log_reverse_forward=None),
        ):
            wrong = copy.deepcopy(move)
            mutate(wrong)
            with self.subTest(wrong=wrong), self.assertRaises((AssertionError, TypeError)):
                audit_single_body_gate(wrong, cfg, ChartAudit(model()))
        zero_activity = dict(cfg, reservoir_density=0.)
        self.assertEqual(audit_single_body_gate(move, zero_activity, ChartAudit(model())), 1)
        sampled = valid_move(-1.)
        sampled['gate'].update(raw_points=8, retained_points=4, gained=3, lost=1,
            log_weight=2*math.log1p(.25), envelope_volume=10., retained_cells=1, created_cells=1)
        sampled['log_acceptance'] = -1.+sampled['gate']['log_weight']
        sampled['accepted'] = False  # A negative alpha alone cannot decide an unrecorded uniform.
        self.assertEqual(audit_single_body_gate(sampled, cfg, ChartAudit(model())), 1)
        with self.assertRaises(AssertionError):
            audit_single_body_gate(sampled, zero_activity, ChartAudit(model()))
        rejected = copy.deepcopy(move)
        rejected.update(hard_valid=False, accepted=False, gate=None, log_acceptance=None)
        self.assertEqual(audit_single_body_gate(rejected, cfg, ChartAudit(model())), 0)
        null = copy.deepcopy(rejected)
        null['proposed_pose'] = None
        null['proposal'].pop('log_reverse_forward')
        self.assertEqual(audit_single_body_gate(null, cfg, ChartAudit(model())), 0)
        self.assertEqual(proposal_kernel(move, cfg), 'full-mixture-capture')
        bad_label = copy.deepcopy(move)
        bad_label['proposal']['kernel'] = 'typo-capture'
        with self.assertRaises(AssertionError):
            proposal_kernel(bad_label, cfg)

    def test_cached_nonspecific_graph_uses_non_anchor_pairs_and_rejects_overlap(self):
        class Sphere:
            bound = .5
            @staticmethod
            def exact_gap(displacement, rotation):
                return float(np.linalg.norm(displacement)-1.)
        state = [pose(), pose(1.5), pose(10.)]
        self.assertEqual(update_nonspecific_graph(state, [(0, 1), (0, 2)], {(1, 2)}, Sphere(), .5), {(0, 1), (1, 2)})
        state[1] = pose(.8)
        with self.assertRaises(AssertionError):
            update_nonspecific_graph(state, [(0, 1)], set(), Sphere(), .5)


if __name__ == '__main__':
    unittest.main()
