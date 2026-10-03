"""Fixed synthetic checks for optional KD culling; no protein inputs or draws."""
import copy
import unittest

import numpy as np

import native_class_line_reference as line
import test_native_class_line_reference as fixtures

IDENTITY, observer = fixtures.IDENTITY, fixtures.observer


class ProjectedCandidateTests(unittest.TestCase):
    def compare(self, fixed, moving, direction, fr, mr, gap, segment, *, pairs=None):
        counts = []
        for inclusive in (False, True):
            kwargs = dict(inclusive=inclusive, pairs=pairs)
            complete = line.leaf_contact_intervals(fixed, moving, direction, fr, mr, gap, segment, **kwargs)
            culled = line.leaf_contact_intervals(fixed, moving, direction, fr, mr, gap, segment,
                                                use_tree=True, **kwargs)
            # Exact represented endpoints and topology, not sampled membership.
            self.assertEqual(complete['intervals'], culled['intervals'])
            self.assertLessEqual(culled['leaf_atom_pairs'], complete['leaf_atom_pairs'])
            counts.append((complete['leaf_atom_pairs'], culled['leaf_atom_pairs']))
        return counts

    def test_sparse_cloud_actually_culls_and_default_remains_complete(self):
        fixed = np.array([[0., 20.*i, -30.*i] for i in range(65)])
        moving = np.array([[1., 0., 0.], [1., 20., -30.]])
        counts = self.compare(fixed, moving, [1., 0., 0.], [.5]*65, [.3, .6], .2, [-4., 4.])
        self.assertEqual(counts, [(130, 2), (130, 2)])

    def test_fixed_random_clouds_variable_radii_and_residue_whitelists(self):
        rng = np.random.default_rng(620041)
        for trial in range(24):
            fixed, moving = rng.normal(size=(11, 3))*4, rng.normal(size=(9, 3))*4
            fr, mr, direction = rng.uniform(0, 1, 11), rng.uniform(0, 1, 9), rng.normal(size=3)
            pairs = None if trial % 2 == 0 else [(int(i), int(j)) for i, j in
                zip(rng.integers(0, 11, 25), rng.integers(0, 9, 25))]
            with self.subTest(trial=trial):
                self.compare(fixed, moving, direction, fr, mr, trial/20, [-3.4, 2.7], pairs=pairs)

    def test_tangencies_neighboring_ulps_and_closed_singleton_segment(self):
        for y in (np.nextafter(2., 0.), 2., np.nextafter(2., np.inf)):
            for direction in ([1., 0., 0.], [-1., 0., 0.], [2., 0., 0.]):
                for segment in ([-2., 3.], [0., 0.], [-3., 0.], [0., 2.]):
                    with self.subTest(y=y, direction=direction, segment=segment):
                        self.compare([[0., 0., 0.]], [[0., y, 0.]], direction, [1.], [1.], 0., segment)
        self.assertEqual(line.leaf_contact_intervals([[0., 0., 0.]], [[0., 2., 0.]],
            [1., 0., 0.], [1.], [1.], 0., [0., 0.], inclusive=True, use_tree=True)['intervals'],
            [line.interval(0., 0.)])

    def test_large_parallel_offsets_and_oblique_rounded_directions(self):
        directions = ([1., 1./3, -1./7], [-.23, .79, .51], [1., np.nextafter(1., 0.), 1.])
        for direction in directions:
            direction = np.array(direction)
            for distance in (1.e8, 1.e14):
                for offset in (np.zeros(3), np.array([1.e15, -1.e15, 1.e15])):
                    fixed = np.array([offset, offset+[0., 40., 0.]])
                    moving = np.array([offset+distance*direction+[0., .1, 0.], offset+[0., 100., 100.]])
                    with self.subTest(direction=direction, distance=distance, offset=offset):
                        self.compare(fixed, moving, direction, [.5, .8], [.7, .2], .1,
                                     [-distance-5, -distance+5])

    def test_nonunit_extreme_and_zero_direction_full_fallback(self):
        for scale in (1.e-200, 1.e200, -3.):
            self.compare([[0., 0., 0.], [0., 100., 0.]], [[0., .5, 0.]],
                [scale, 0., 0.], [.5, .8], [.5], .2, [-2/abs(scale), 3/abs(scale)])
        fixed, moving = np.array([[0., 0., 0.], [1., 3., 2.]]), np.array([[.1, 0., 0.]])
        self.assertIsNone(line.projected_candidates(fixed, moving, [0., 0., 0.], [.5, .5], [.5], 0., [-2., 3.]))
        self.assertEqual(self.compare(fixed, moving, [0., 0., 0.], [.5, .5], [.5], 0., [-2., 3.]),
                         [(2, 2), (2, 2)])

    def test_empty_whitelist_duplicates_and_empty_atoms(self):
        fixed, moving = [[0., 0., 0.], [0., 50., 0.]], [[0., 0., 0.], [0., 50., 0.]]
        for pairs in ([], [(0, 0), (0, 0), (1, 1), (0, 1)], [(0, 1)]):
            self.compare(fixed, moving, [1., 0., 0.], [.2, .8], [.3, .5], .1, [-2., 2.], pairs=pairs)
        self.compare(np.empty((0, 3)), moving, [1., 0., 0.], [], [.3, .5], .1, [-2., 2.])
        self.compare(fixed, np.empty((0, 3)), [1., 0., 0.], [.2, .8], [], .1, [-2., 2.])

    def test_unsafe_bounds_fallback_and_leaf_errors_are_not_hidden(self):
        fixed, moving = np.array([[0., 0., 0.]]), np.array([[0., 1.e308, 0.]])
        self.assertIsNone(line.projected_candidates(fixed, moving, [1., 0., 0.], [1.], [1.], 0., [-1., 1.]))
        self.assertEqual(self.compare(fixed, moving, [1., 0., 0.], [1.], [1.], 0., [-1., 1.]), [(1, 1), (1, 1)])
        for use_tree in (False, True):
            with self.subTest(use_tree=use_tree), np.errstate(over='ignore'), self.assertRaisesRegex(ValueError, 'Unrepresentable leaf roots'):
                line.leaf_contact_intervals(fixed, fixed, [1., 0., 0.], [1.e308], [1.e308], 0.,
                    [-1., 1.], inclusive=True, use_tree=use_tree)

    def test_native_all_motifs_anchors_and_residue_pairs_preserved(self):
        atoms = np.array([[0., 0., 0.], [0., 1., 0.], [0., 80., 0.]])
        model = observer(atoms=atoms, residues=[0, 1, 2])
        model.references['good']['native_residue_pairs'] = {0, 1, 3}
        duplicate = copy.deepcopy(model.motifs[0]); duplicate['id'] = 17
        model.motifs.append(duplicate)
        model.motif_positions = np.vstack([model.motif_positions, model.motif_positions])
        model.motif_rotations = np.concatenate([model.motif_rotations, model.motif_rotations])
        anchors = [IDENTITY, dict(position=[10., 0., 0.], orientation=[1., 0., 0., 0.])]
        full, fast = line.NativeLineReference(model), line.NativeLineReference(model, use_tree=True)
        for origin in ([0., 0., 0.], [0., 2.5, 0.], [0., 40., 0.]):
            a = full.all_anchors(anchors, origin, np.eye(3), [1., 0., 0.], [-2., 16.])
            b = fast.all_anchors(anchors, origin, np.eye(3), [1., 0., 0.], [-2., 16.])
            self.assertEqual(a['intervals'], b['intervals'])
            self.assertEqual([v['motifs'] for v in a['anchors']], [v['motifs'] for v in b['anchors']])
            self.assertLessEqual(b['leaf_atom_pairs'], a['leaf_atom_pairs'])

    def test_full_class_density_geometry_and_per_call_override(self):
        args = fixtures.ConditionalLawTests().setup()
        full, fast = line.Reconstructor(*args), line.Reconstructor(*args, use_tree=True)
        rng = np.random.default_rng(620042)
        for u in [np.zeros(6), *rng.normal(0., .7, size=(12, 6))]:
            a, b = full.density(u), fast.density(u)
            self.assertEqual(a['log_density'], b['log_density'])
            self.assertEqual(a['component_multipliers'], b['component_multipliers'])
            for x, y in zip(a['axes'], b['axes']):
                for key in ('intervals', 'hard_free_intervals', 'native_intervals',
                            'exclusion_contact_intervals', 'channels', 'components'):
                    self.assertEqual(x[key], y[key], key)
                self.assertEqual(fast.reconstruct_axis(u, x['axis'], use_tree=False),
                                 {k: v for k, v in x.items() if k != 'components'})
                self.assertEqual(full.reconstruct_axis(u, x['axis'], use_tree=True),
                                 {k: v for k, v in y.items() if k != 'components'})


if __name__ == '__main__':
    unittest.main()
