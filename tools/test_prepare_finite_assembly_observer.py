"""Geometry-only patch construction and frozen measurement-asset controls."""
import copy
import itertools
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import prepare_finite_assembly_observer as prep
from finite_assembly_observer import graph_summary
from native_contact_regions import pose_arrays


def fixture(points=None):
    points = (list(itertools.product((-1., 2.), repeat=3)) if points is None else points)
    monomer = dict(atoms=[dict(center=list(p), radius=.2+i*.01) for i, p in enumerate(points)])
    c = float(np.sqrt(.5))
    members = [dict(position=[7.*i, 2., -3.], orientation=q) for i, q in enumerate(
        ([1., 0., 0., 0.], [c, 0., 0., c], [0., 1., 0., 0.], [c, 0., c, 0.]))]
    atoms = []
    for member in members:
        t, r = pose_arrays(member)
        atoms.extend(dict(center=(r@np.asarray(a['center'])+t).tolist(), radius=a['radius'])
                     for a in monomer['atoms'])
    return dict(atoms=atoms, rigid_members=members), monomer


class PatchConstructionTests(unittest.TestCase):
    def build(self, shape, monomer):
        return prep.build_patch_map(shape, monomer, 'a'*64, 'b'*64)

    def test_local_octants_preserved_across_member_rotation_and_translation(self):
        shape, monomer = fixture()
        result, check = self.build(shape, monomer)
        self.assertEqual(result['atom_patch_ids'], prep.PATCH_IDS)
        self.assertEqual(result['patch_dictionary'], prep.PATCH_IDS)
        self.assertEqual(set(result['patch_atom_counts'].values()), {1})
        self.assertEqual(check['dictionary_size'], 32)
        self.assertLessEqual(check['maximum_coordinate_error_A'], 1e-14)
        self.assertEqual(check['maximum_radius_error_A'], 0.)
        self.assertEqual(check['physical_draws'], 0)
        self.assertEqual(check['native_classifier_calls'], 0)

    def test_empty_patch_ids_and_zero_sign_ties_are_retained(self):
        shape, monomer = fixture([[0., -0., 0.], [-1., 0., -1.]])
        result, _ = self.build(shape, monomer)
        self.assertEqual(len(result['patch_dictionary']), 32)
        self.assertEqual(sum(result['patch_atom_counts'].values()), 8)
        for member in range(4):
            self.assertEqual(result['atom_patch_ids'][2*member:2*member+2],
                             [f'member{member}:octant7', f'member{member}:octant2'])
        self.assertEqual(sum(v == 0 for v in result['patch_atom_counts'].values()), 24)

    def test_origin_is_not_inferred_from_centroid(self):
        shape, monomer = fixture([[9., 10., 11.], [20., 30., 40.]])
        result, _ = self.build(shape, monomer)
        self.assertTrue(all(label.endswith('octant7') for label in result['atom_patch_ids']))

    def test_roundoff_is_reported_but_shape_change_rejected(self):
        shape, monomer = fixture()
        shape['atoms'][0]['center'][0] += prep.RECONSTRUCTION_TOLERANCE_A*.25
        _, check = self.build(shape, monomer)
        self.assertGreater(check['maximum_coordinate_error_A'], 0.)
        shape['atoms'][0]['center'][0] += prep.RECONSTRUCTION_TOLERANCE_A*2
        with self.assertRaisesRegex(ValueError, 'reconstruct'):
            self.build(shape, monomer)

    def test_radii_must_match_exactly_even_below_coordinate_tolerance(self):
        shape, monomer = fixture()
        shape['atoms'][0]['radius'] += 1e-14
        with self.assertRaisesRegex(ValueError, 'radii differ'):
            self.build(shape, monomer)

    def test_atom_order_missing_members_and_invalid_numbers_fail(self):
        shape, monomer = fixture()
        cases = []
        changed = copy.deepcopy(shape); changed['atoms'][0], changed['atoms'][1] = changed['atoms'][1], changed['atoms'][0]; cases.append(changed)
        changed = copy.deepcopy(shape); changed['atoms'].pop(); cases.append(changed)
        changed = copy.deepcopy(shape); changed['rigid_members'].pop(); cases.append(changed)
        changed = copy.deepcopy(shape); changed['atoms'][0]['center'][0] = float('nan'); cases.append(changed)
        changed = copy.deepcopy(shape); changed['rigid_members'][0]['orientation'] = [0., 0., 0., 0.]; cases.append(changed)
        for changed in cases:
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                self.build(changed, monomer)


class FrozenRulesTests(unittest.TestCase):
    def graph(self, size=0, exclusion=(), keys=None, positions=None, box=None):
        motifs = [dict(id=i, relative_position=[float(i+1), 0., 0.],
                       relative_orientation=[1., 0., 0., 0.]) for i in range(2)]
        n = 12
        keys = [(i, i+1, 0) for i in range(size-1)] if keys is None else keys
        positions = [[float(i), 0., 0.] for i in range(n)] if positions is None else positions
        graph = graph_summary(n, exclusion, keys, motifs, positions, box)
        self.assertEqual(prep.rule_label(graph), graph['environment'])
        return graph

    def test_all_six_regions_are_exhaustive_on_component_examples(self):
        self.assertEqual(self.graph()['environment'], 'dispersed')
        self.assertEqual(self.graph(exclusion=[(0, 1)])['environment'], 'contact_no_entry')
        for size, label in [(2, 'registered_small'), (7, 'registered_small'),
                            (8, 'registered_eight'), (9, 'registered_growth'), (12, 'registered_growth')]:
            self.assertEqual(self.graph(size)['environment'], label)
        bad = self.graph(keys=[(0, 1, 0), (1, 2, 0), (0, 2, 0)])
        self.assertEqual(bad['environment'], 'remaining')
        self.assertTrue(bad['native_cycle_frustrated'])

    def test_winding_retained_as_remaining_without_catalogue_failure(self):
        positions = [[0., 0., 0.], [4., 0., 0.], [8., 0., 0.]]+[[1., 1., 1.]]*9
        graph = self.graph(keys=[(0, 1, 0), (1, 2, 0), (0, 2, 1)], positions=positions, box=[12.]*3)
        self.assertEqual(graph['environment'], 'remaining')
        self.assertTrue(graph['native_periodic_winding'])
        self.assertFalse(graph['native_cycle_frustrated'])


class ArtifactRetentionTests(unittest.TestCase):
    def test_failure_retains_partial_outputs_and_freezes_them(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'failed'
            def fail(starts, path):
                path.mkdir(); (path/'partial.jsonl').write_text('{"attempt":0}\n')
                (path/'nested').mkdir(); (path/'nested/freeze.json').write_text('{}\n')
                raise ValueError('synthetic preparation failure')
            with patch.object(prep, '_prepare', side_effect=fail), self.assertRaisesRegex(ValueError, 'synthetic'):
                prep.prepare(Path(tmp)/'input', out)
            self.assertFalse(json.loads((out/'failure.json').read_text())['complete'])
            frozen = json.loads((out/'freeze.json').read_text())['files']
            self.assertEqual(set(frozen), {'failure.json', 'partial.jsonl', 'nested/freeze.json'})
            for name, expected in frozen.items():
                self.assertEqual(prep.sha(out/name), expected)

    def test_existing_output_is_never_modified(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp); (out/'keep').write_text('unchanged')
            with self.assertRaisesRegex(ValueError, 'Fresh'):
                prep.prepare('unused', out)
            self.assertEqual([p.name for p in out.iterdir()], ['keep'])


if __name__ == '__main__':
    unittest.main()
