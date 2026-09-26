"""Synthetic fit/freeze checks only; no discovery MC, native data or protein runs."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from fit_depletion_contact_atlas import (
    DISCOVERY_SCHEMA, FitOptions, chart_coordinates, fit_model, prepare,
    proposal_preflight, regularized_moments, rolling_covariance, serialized,
    sha_bytes, shape_scale,
)
from analyze_involution_docking_campaign import ChartAudit
from prepare_smc_normalizer_atlas import Density


def pose(position, rotation=None):
    rotation = np.eye(3) if rotation is None else rotation
    quaternion = Rotation.from_matrix(rotation).as_quat()[[3, 0, 1, 2]]
    return dict(position=np.asarray(position, float).tolist(), orientation=quaternion.tolist())


def contact(direction, rotation=None):
    direction = np.asarray(direction, float)
    direction /= np.linalg.norm(direction)
    value = pose(2.1*direction, rotation)
    witness = dict(radial_contact=dict(direction=direction.tolist(), distance=2.,
        orientation=value['orientation'], moving_atom=0, fixed_atom=0),
        surface_point=direction.tolist(), mobile_surface_lever=(-direction).tolist())
    return value, witness


def chart_pose(reference, coordinates, ell=2.):
    x = np.asarray(coordinates, float)
    anchor_r = Rotation.from_quat(np.asarray(reference['orientation'])[[1, 2, 3, 0]]).as_matrix()
    rotation = Rotation.from_quat(np.r_[x[3:]/ell, 1.]).as_matrix()@anchor_r
    return pose(np.asarray(reference['position'])+x[:3], rotation)


def fixture(count=2, seed=44):
    shape = dict(atoms=[dict(center=[0., 0., 0.], radius=1.)])
    shape_hash = sha_bytes(serialized(shape))
    slots = []
    for index in range(count):
        initial, initial_contact = contact([1., .2*index, .1*index])
        optimized, optimized_contact = contact([.1*index, 1., .2*index],
            Rotation.from_rotvec([.1*index, -.2*index, .15*index]).as_matrix())
        coordinates = np.array([[0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0.],
            [1.+index, -.3, .4, .15, -.1, .2]])
        samples = [dict(step=10+i, pose=chart_pose(optimized, x), accepted=i in (0, 3))
                   for i, x in enumerate(coordinates)]
        slots.append(dict(slot=index, seeds=dict(start=seed+10*index, refinement=seed+10*index+1),
            initial_pose=initial, initial_contact=initial_contact, optimized_pose=optimized,
            optimized_contact=optimized_contact, refinement_samples=samples,
            initial_validation=[dict(volume=10.-index), dict(volume=9.-index)],
            optimized_validation=[dict(volume=20.-index), dict(volume=19.-index)]))
    discovery = dict(schema=DISCOVERY_SCHEMA, shape_sha256=shape_hash,
                     config=dict(starts=count, seed=seed), slots=slots)
    return shape, discovery, shape_hash


def archive_fixture(root, shape, discovery):
    root.mkdir()
    shape_bytes, discovery_bytes = serialized(shape), serialized(discovery)
    files = {'shape.json': shape_bytes, 'discovery.json': discovery_bytes,
             'source-bundle.json': serialized({'files': {}}),
             'provenance.json': serialized({'native_geometry_inputs': 0})}
    for name, raw in files.items():
        (root/name).write_bytes(raw)
    manifest = dict(schema=DISCOVERY_SCHEMA, complete=True, shape_sha256=sha_bytes(shape_bytes),
        outputs_sha256={name: sha_bytes(raw) for name, raw in files.items()})
    (root/'manifest.json').write_bytes(serialized(manifest))
    return root/'shape.json'


class DepletionContactFitTests(unittest.TestCase):
    def test_shape_rms_scale_is_independent_of_translation_and_metadata(self):
        shape = dict(atoms=[dict(center=[1., 2., 3.], radius=.5),
                            dict(center=[5., 2., 3.], radius=.7)], native_motif='unused')
        ell, metrics = shape_scale(shape)
        self.assertEqual(ell, 4.)
        self.assertEqual(metrics['center_rms'], 2.)
        for atom in shape['atoms']:
            atom['center'] = (np.asarray(atom['center'])+123.).tolist()
        self.assertEqual(shape_scale(shape)[0], ell)

    def test_chart_coordinates_and_full_covariance_moments(self):
        shape, discovery, _ = fixture(1)
        reference = discovery['slots'][0]['optimized_pose']
        lower = np.diag([.5, .7, .3, .2, .4, .6])
        lower[3, 0] = .14
        lower[5, 2] = -.09
        desired_mean = np.array([.2, -.1, .4, -.3, .15, .06])
        # Twelve symmetric coordinates have exact zero mean and identity
        # population covariance before the full linear transformation.
        standardized = np.r_[np.sqrt(6)*np.eye(6), -np.sqrt(6)*np.eye(6)]
        expected_coordinates = standardized@lower.T+desired_mean
        poses = [chart_pose(reference, value) for value in expected_coordinates]
        actual, _ = chart_coordinates(reference, poses, shape_scale(shape)[0])
        np.testing.assert_allclose(actual, expected_coordinates, atol=1e-15)
        options = FitOptions(shrinkage=0, covariance_floor=1e-12)
        mean, covariance, metrics = regularized_moments(actual, np.eye(6), options)
        np.testing.assert_allclose(mean, desired_mean, atol=1e-15)
        np.testing.assert_allclose(covariance, lower@lower.T, atol=1e-15)
        self.assertEqual(metrics['scatter_denominator'], 12)
        self.assertGreater(abs(covariance[0, 3]), .01)

    def test_rejected_repeats_contribute_to_mean_and_scatter(self):
        shape, discovery, shape_hash = fixture(1)
        model, _, metrics = fit_model(shape, discovery, shape_hash)
        samples = discovery['slots'][0]['refinement_samples']
        coordinates, _ = chart_coordinates(discovery['slots'][0]['optimized_pose'],
                                           [sample['pose'] for sample in samples], 2.)
        np.testing.assert_allclose(model['base_model']['means'][1], coordinates.mean(axis=0), atol=1e-15)
        self.assertAlmostEqual(model['base_model']['means'][1][0], .25)
        self.assertEqual(metrics['retained_count'], 4)
        self.assertEqual(metrics['retained_rejected_count'], 2)
        self.assertEqual(metrics['slots'][0]['unique_retained_poses'], 2)
        self.assertEqual(metrics['slots'][0]['retained_steps'], [10, 11, 12, 13])
        np.testing.assert_allclose(metrics['slots'][0]['empirical_covariance'], np.cov(coordinates.T, bias=True), atol=1e-15)

    def test_scores_acceptance_and_native_metadata_do_not_select_or_weight_slots(self):
        shape, discovery, shape_hash = fixture(3)
        expected, initial, _ = fit_model(shape, discovery, shape_hash)
        changed = copy.deepcopy(discovery)
        changed['native_model'] = '/must/not/be/read/model-native.json'
        changed['native_motifs'] = '/must/not/be/read/motifs.json'
        changed['production_poses'] = '/must/not/be/read/trajectory.json'
        for i, slot in enumerate(changed['slots']):
            slot['native_label'] = ['winner', 'loser', 'unknown'][i]
            slot['initial_validation'] = [dict(volume=-1e100)]*2
            slot['optimized_validation'] = [dict(volume=1e100*(i+1))]*2
            slot['search_attempts'] = [{'score': 'ignored', 'native_rank': -i}]
            for sample in slot['refinement_samples']:
                sample['accepted'] = not sample['accepted']
        actual, actual_initial, metrics = fit_model(shape, changed, shape_hash)
        self.assertEqual(serialized(actual), serialized(expected))
        self.assertEqual(serialized(actual_initial), serialized(initial))
        self.assertEqual([row['local_slot'] for row in metrics['slots']], [0, 1, 2])
        np.testing.assert_array_equal(actual['base_model']['weights'], [1/6.]*6)
        self.assertEqual([row['raw_slot_weight'] for row in metrics['slots']], [1.]*3)

    def test_multiple_populations_keep_all_slots_even_with_unequal_trace_lengths(self):
        shape, first, shape_hash = fixture(2, seed=10)
        _, second, _ = fixture(1, seed=30)
        samples = second['slots'][0]['refinement_samples']
        samples += [dict(step=14+i, pose=copy.deepcopy(samples[-1]['pose']), accepted=False) for i in range(12)]
        model, initial, metrics = fit_model(shape, [first, second], shape_hash)
        self.assertEqual([(row['source_index'], row['local_slot']) for row in metrics['slots']], [(0, 0), (0, 1), (1, 0)])
        np.testing.assert_array_equal(model['base_model']['weights'], [1/6.]*6)
        np.testing.assert_array_equal(initial['base_model']['weights'], [1/3.]*3)
        self.assertEqual(metrics['source_populations'], 2)
        self.assertEqual(metrics['retained_count'], 24)
        self.assertEqual(metrics['slots'][2]['source_seed'], 30)
        self.assertEqual(metrics['slots'][2]['seeds'], second['slots'][0]['seeds'])

    def test_protective_initial_component_and_baseline_regularization(self):
        shape, discovery, shape_hash = fixture(1)
        samples = discovery['slots'][0]['refinement_samples']
        for sample in samples:
            sample['pose'] = copy.deepcopy(samples[0]['pose'])
        options = FitOptions(shrinkage=.25, covariance_floor=.4)
        model, initial, metrics = fit_model(shape, discovery, shape_hash, options)
        base = model['base_model']
        self.assertEqual(base['weights'], [.5, .5])
        self.assertEqual(base['anchors'][0], initial['base_model']['anchors'][0])
        self.assertEqual(base['covariances'][0], initial['base_model']['covariances'][0])
        self.assertEqual(base['means'][0], [0.]*6)
        baseline = rolling_covariance(np.array([0., -1., 0.]), 2., options)
        np.testing.assert_allclose(base['covariances'][1], .4*baseline, atol=1e-15)
        self.assertTrue((np.linalg.eigvalsh(base['covariances']) > 0).all())
        self.assertEqual(metrics['slots'][0]['floored_eigenvalues'], 6)
        self.assertIn('No IID', metrics['limitation'])

    def test_density_normalization_by_construction_and_roundtrip(self):
        shape, discovery, shape_hash = fixture(2)
        model, _, _ = fit_model(shape, discovery, shape_hash)
        density, audit = Density(model), ChartAudit(model)
        self.assertAlmostEqual(sum(density.weights), 1., places=15)
        self.assertEqual(model['reciprocal_components'], [True]*4)
        self.assertTrue(proposal_preflight(model, 18)['passed'])
        rng = np.random.default_rng(101)
        for label in range(8):
            for latent in rng.normal(size=(4, 6)):
                value, jacobian = audit.decode(label, latent)
                np.testing.assert_allclose(audit.encode(label, value), latent, atol=2e-13)
                component = density.evaluate([value])[2][0, label]-np.log(density.weights[label])
                self.assertAlmostEqual(component+jacobian, -3*np.log(2*np.pi)-.5*latent@latent, places=11)
                inverse = audit.reciprocal(value)
                self.assertAlmostEqual(density.evaluate([value])[0][0], density.evaluate([inverse])[0][0], places=11)

    def test_invalid_slots_and_witnesses_fail_without_dropping_data(self):
        shape, original, shape_hash = fixture(2)
        variants = []
        altered = copy.deepcopy(original); altered['slots'][0]['refinement_samples'] = []; variants.append(altered)
        altered = copy.deepcopy(original); altered['slots'].pop(); variants.append(altered)
        altered = copy.deepcopy(original); altered['slots'][1]['slot'] = 0; variants.append(altered)
        altered = copy.deepcopy(original); altered['slots'][0]['initial_contact']['mobile_surface_lever'][0] += .1; variants.append(altered)
        altered = copy.deepcopy(original); altered['slots'][0]['refinement_samples'][1]['step'] = 10; variants.append(altered)
        altered = copy.deepcopy(original); altered['slots'][0]['refinement_samples'][1]['accepted'] = 1; variants.append(altered)
        for altered in variants:
            with self.subTest(altered=altered):
                with self.assertRaises(ValueError):
                    fit_model(shape, altered, shape_hash)

    def test_cayley_seam_fails_entire_fit_instead_of_trimming_sample(self):
        shape, discovery, shape_hash = fixture(1)
        discovery['slots'][0]['refinement_samples'][0]['pose']['orientation'] = [0., 1., 0., 0.]
        with self.assertRaisesRegex(ValueError, 'seam'):
            fit_model(shape, discovery, shape_hash)

    def test_invalid_regularization_and_branch_weights_rejected(self):
        for options in (FitOptions(initial_weight=0), FitOptions(initial_weight=1),
                        FitOptions(shrinkage=-.1), FitOptions(shrinkage=1.1),
                        FitOptions(covariance_floor=0), FitOptions(translation_width=float('nan')),
                        FitOptions(angle_width_degrees=-1)):
            with self.subTest(options=options):
                with self.assertRaises(ValueError):
                    options.validate()

    def test_one_slot_freeze_is_deterministic_and_reads_no_native_data(self):
        shape, discovery, _ = fixture(1)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root/'discovery'
            shape_path = archive_fixture(source, shape, discovery)
            forbidden = root/'native-model.json'
            forbidden.write_text('{"native": "a sentinel unrelated to the input data"}')
            original_read_bytes, original_read_text = Path.read_bytes, Path.read_text
            observed = []
            def audit_read_bytes(path):
                path = Path(path)
                observed.append(path)
                self.assertNotEqual(path, forbidden)
                self.assertTrue(path.is_relative_to(root) or path.suffix == '.py', str(path))
                return original_read_bytes(path)
            def audit_read_text(path, *args, **kwargs):
                path = Path(path)
                observed.append(path)
                self.assertNotEqual(path, forbidden)
                self.assertTrue(path.is_relative_to(root) or path.suffix == '.py', str(path))
                return original_read_text(path, *args, **kwargs)
            with patch.object(Path, 'read_bytes', audit_read_bytes), patch.object(Path, 'read_text', audit_read_text):
                first = prepare(shape_path, [source], root/'fit-a')
                forbidden.write_text('changed native information must not affect a native-blind fit')
                second = prepare(shape_path, [source], root/'fit-b')
            self.assertEqual(first, second)
            self.assertGreater(len(observed), 0)
            names = sorted(p.relative_to(root/'fit-a') for p in (root/'fit-a').rglob('*') if p.is_file())
            self.assertEqual(names, sorted(p.relative_to(root/'fit-b') for p in (root/'fit-b').rglob('*') if p.is_file()))
            for name in names:
                self.assertEqual((root/'fit-a'/name).read_bytes(), (root/'fit-b'/name).read_bytes(), str(name))
            freeze = json.loads((root/'fit-a'/'freeze.json').read_text())
            for name, digest in freeze['files'].items():
                self.assertEqual(sha_bytes((root/'fit-a'/name).read_bytes()), digest)
            self.assertEqual(first['independent_slots'], 1)
            self.assertFalse(first['production_launched'])
            with self.assertRaisesRegex(ValueError, 'fresh'):
                prepare(shape_path, [source], root/'fit-a')

    def test_manifest_integrity_and_duplicate_populations_fail_closed(self):
        shape, discovery, _ = fixture(1)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root/'discovery'
            shape_path = archive_fixture(source, shape, discovery)
            with self.assertRaisesRegex(ValueError, 'Repeated discovery'):
                prepare(shape_path, [source, source], root/'duplicate')
            self.assertFalse((root/'duplicate').exists())
            raw = (source/'discovery.json').read_bytes()
            (source/'discovery.json').write_bytes(raw+b' ')
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                prepare(shape_path, [source], root/'modified')
            self.assertFalse((root/'modified').exists())
            (source/'discovery.json').write_bytes(raw)
            manifest = json.loads((source/'manifest.json').read_text())
            manifest['complete'] = False
            (source/'manifest.json').write_bytes(serialized(manifest))
            with self.assertRaisesRegex(ValueError, 'completed'):
                prepare(shape_path, [source], root/'incomplete')

    def test_changed_metadata_does_not_make_repeated_random_streams_independent(self):
        shape, first, shape_hash = fixture(2)
        second = copy.deepcopy(first)
        second['config']['validation_points'] = 123456
        second['slots'][0]['optimized_validation'] = [{'volume': 9.}]*2
        with self.assertRaisesRegex(ValueError, 'Repeated source seed'):
            fit_model(shape, [first, second], shape_hash)


if __name__ == '__main__':
    unittest.main()
