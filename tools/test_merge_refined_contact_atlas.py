"""Frozen proposal bookkeeping tests; no protein runs or native references."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from fit_depletion_contact_atlas import FitOptions, fit_model, serialized, sha_bytes
from merge_refined_contact_atlas import merge
from prepare_mobile_reciprocal_benchmark import reciprocal_envelope
from test_fit_depletion_contact_atlas import chart_pose, fixture


def archive(directory, model, metrics):
    directory.mkdir()
    (directory/'model.json').write_bytes(serialized(model))
    (directory/'fit-metrics.json').write_bytes(serialized(metrics))
    (directory/'manifest.json').write_bytes(serialized(dict(complete=True,
        model_sha256=sha_bytes((directory/'model.json').read_bytes()))))
    refresh(directory)


def refresh(directory):
    """Synthetic fixtures deliberately update their freeze after semantic edits."""
    manifest = json.loads((directory/'manifest.json').read_bytes())
    manifest['model_sha256'] = sha_bytes((directory/'model.json').read_bytes())
    (directory/'manifest.json').write_bytes(serialized(manifest))
    (directory/'freeze.json').write_bytes(serialized(dict(files={
        p.name: sha_bytes(p.read_bytes()) for p in directory.glob('*.json') if p.name != 'freeze.json'})))


def edit(directory, name, operation):
    path = directory/name
    value = json.loads(path.read_bytes())
    operation(value)
    path.write_bytes(serialized(value))
    refresh(directory)


def prepare_fixture(root):
    shape, discovery, shape_hash = fixture(3, seed=501)
    original, _, original_metrics = fit_model(shape, discovery, shape_hash)
    # Unequal original masses ensure replacing fewer charts cannot flatten or
    # renormalize the original slot allocation accidentally.
    original['base_model']['weights'] = [.1, .1, .15, .15, .25, .25]
    mapping = [1, 1, 2]
    replacement = copy.deepcopy(discovery)
    replacement['config']['seed'] = 701
    replacement['slots'] = []
    for index, old_slot in enumerate(mapping):
        slot = copy.deepcopy(discovery['slots'][old_slot])
        slot['slot'] = index
        slot['seeds']['refinement'] = 70100+index
        rng = np.random.default_rng(80100+index)
        reference = slot['optimized_pose']
        points = rng.normal(size=(64, 6))*.002
        points[:, 0] += index*.002
        slot['refinement_samples'] = [dict(step=i+1, pose=chart_pose(reference, point), accepted=True)
                                      for i, point in enumerate(points)]
        replacement['slots'].append(slot)
    newer, _, newer_metrics = fit_model(shape, replacement, shape_hash,
        FitOptions(initial_weight=0, shrinkage=.02, covariance_floor=1e-6,
                   minimum_unique_poses=32, minimum_empirical_rank=6))
    archive(root/'old', original, original_metrics)
    archive(root/'new', newer, newer_metrics)
    (root/'mapping.json').write_bytes(serialized(mapping))
    return SimpleNamespace(base_fit=root/'old', refined_fit=root/'new',
                           mapping=root/'mapping.json', out=root/'merged')


class MergeRefinedAtlasTests(unittest.TestCase):
    def run_merge(self, args):
        with contextlib.redirect_stdout(io.StringIO()):
            return merge(args)

    def test_preserves_exact_slot_masses_and_untouched_components(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = prepare_fixture(root)
            old = json.loads((root/'old'/'model.json').read_bytes())['base_model']
            new = json.loads((root/'new'/'model.json').read_bytes())['base_model']
            self.run_merge(args)
            merged = json.loads((args.out/'model.json').read_bytes())['base_model']
            self.assertEqual(merged['weights'], [.1, .1, .15, .15, .5])
            for key in ['anchors', 'means', 'covariances', 'weights']:
                self.assertEqual(merged[key][:2], old[key][:2])
            for key in ['anchors', 'means', 'covariances']:
                self.assertEqual(merged[key][2:], new[key])
            audit = json.loads((args.out/'slot-map.json').read_bytes())
            self.assertEqual([row['mass'] for row in audit], [.2, .3, .5])
            self.assertEqual(audit[1]['refined_rows'], [0, 1])
            self.assertEqual(audit[1]['refinement_streams'], [70100, 70101])
            self.assertEqual(audit[2]['refined_rows'], [2])
            for name in ['old-updated-slots-only', 'new-updated-slots-only']:
                subset = json.loads((args.out/(name+'.json')).read_bytes())['base_model']
                self.assertAlmostEqual(sum(subset['weights']), 1.)
            self.assertTrue(json.loads((args.out/'preflight.json').read_bytes())['passed'])

    def test_new_fit_allocation_does_not_replace_original_slot_masses(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = prepare_fixture(root)
            self.run_merge(args)
            expected = (args.out/'model.json').read_bytes()
            edit(root/'new', 'model.json', lambda value: value['base_model'].update(weights=[.98, .01, .01]))
            args.out = root/'unequal-source-weights'
            self.run_merge(args)
            self.assertEqual((args.out/'model.json').read_bytes(), expected)

    def test_no_failed_or_regularizer_only_slot_can_be_dropped(self):
        operations = [
            lambda metrics: metrics.update(exploration_gate_passed=False),
            lambda metrics: metrics['slots'][1].update(exploration_gate_passed=False),
            lambda metrics: metrics['slots'][1].update(unique_physical_chart_poses=1, empirical_rank=0),
            lambda metrics: metrics['slots'][1]['requested_gate'].update(minimum_unique_poses=1000),
            lambda metrics: metrics['slots'][1].update(initial_component=0),
        ]
        for operation in operations:
            with self.subTest(operation=operation), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                args = prepare_fixture(root)
                edit(root/'new', 'fit-metrics.json', operation)
                with self.assertRaises(ValueError):
                    self.run_merge(args)
                self.assertFalse(args.out.exists())

    def test_duplicate_or_missing_refinement_stream_is_not_independent(self):
        for seed in [70100, None]:
            with self.subTest(seed=seed), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                args = prepare_fixture(root)
                edit(root/'new', 'fit-metrics.json',
                     lambda metrics: metrics['slots'][1]['seeds'].update(refinement=seed))
                with self.assertRaisesRegex(ValueError, 'RNG|independent'):
                    self.run_merge(args)
                self.assertFalse(args.out.exists())

    def test_mapping_and_reference_poses_are_checked(self):
        for mapping in [[], [1, 2], [True, 1, 2], [1, 1, 30], [0, 1, 2]]:
            with self.subTest(mapping=mapping), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                args = prepare_fixture(root)
                args.mapping.write_bytes(serialized(mapping))
                with self.assertRaises(ValueError):
                    self.run_merge(args)
                self.assertFalse(args.out.exists())

    def test_source_component_partition_is_complete_and_disjoint(self):
        operations = [
            lambda metrics: metrics['slots'][1].update(refined_component=1),
            lambda metrics: metrics['slots'][0].update(initial_component=None),
            lambda metrics: metrics['slots'][0].update(global_slot=4),
        ]
        for operation in operations:
            with self.subTest(operation=operation), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                args = prepare_fixture(root)
                edit(root/'old', 'fit-metrics.json', operation)
                with self.assertRaises(ValueError):
                    self.run_merge(args)
                self.assertFalse(args.out.exists())

    def test_changed_uncovered_or_unsafe_frozen_input_fails_before_writing(self):
        for mode in ['changed', 'uncovered', 'unsafe']:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                args = prepare_fixture(root)
                if mode == 'changed':
                    with (root/'new'/'model.json').open('ab') as stream:
                        stream.write(b' ')
                else:
                    path = root/'new'/'freeze.json'
                    value = json.loads(path.read_bytes())
                    if mode == 'uncovered':
                        del value['files']['fit-metrics.json']
                    else:
                        (root/'outside.json').write_text('{}')
                        value['files']['../outside.json'] = sha_bytes(b'{}')
                    path.write_bytes(serialized(value))
                with self.assertRaises(ValueError):
                    self.run_merge(args)
                self.assertFalse(args.out.exists())

    def test_output_archives_verified_inputs_and_is_deterministic(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = prepare_fixture(root)
            first = self.run_merge(args)
            freeze = json.loads((args.out/'freeze.json').read_bytes())
            for name, digest in freeze['files'].items():
                self.assertEqual(sha_bytes((args.out/name).read_bytes()), digest)
            for label in ['base', 'refined']:
                source = root/('old' if label == 'base' else 'new')
                for name in ['model.json', 'fit-metrics.json', 'manifest.json', 'freeze.json']:
                    self.assertEqual((args.out/'provenance'/label/name).read_bytes(), (source/name).read_bytes())
            self.assertEqual((args.out/'provenance'/'mapping.json').read_bytes(), args.mapping.read_bytes())
            before = args.out
            args.out = root/'second'
            second = self.run_merge(args)
            self.assertEqual(first, second)
            for relative in freeze['files']:
                self.assertEqual((before/relative).read_bytes(), (args.out/relative).read_bytes())
            with self.assertRaisesRegex(ValueError, 'Fresh'):
                self.run_merge(args)


if __name__ == '__main__':
    unittest.main()
