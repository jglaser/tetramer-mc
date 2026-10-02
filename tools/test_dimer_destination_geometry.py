import unittest
import numpy as np
from dimer_destination_geometry import DimerGeometry


def pose(x, y=0):
    return dict(position=[x, y, 0.], orientation=[1., 0., 0., 0.])


class GeometryTests(unittest.TestCase):
    def test_removed_old_bodies_and_new_mutual_pair(self):
        shape = dict(atoms=[dict(center=[0., 0., 0.], radius=1.)])
        g = DimerGeometry(shape, [pose(0), pose(3), pose(8)], .5, 20.)
        result = g.fingerprint([0, 1], [pose(6.5), pose(7)])
        self.assertEqual(result['hard_overlap_edges'], [[0, 1], [0, 2], [1, 2]])
        self.assertEqual(g.fingerprint([0, 1], [pose(0), pose(3)])['contacts'], [])  # Strict tangency.
        moved = g.fingerprint([0, 1], [pose(3), pose(-8)])
        self.assertTrue(moved['hard_valid'])  # Old selected body at x=3 was removed.

    def test_variable_radii_against_direct_all_pairs_even_wall_invalid(self):
        shape = dict(atoms=[dict(center=[0., 0., 0.], radius=.2), dict(center=[1., 0., 0.], radius=.8)])
        state = [pose(-4), pose(0), pose(4), pose(8)]
        g = DimerGeometry(shape, state, .3, 8.)
        selected = [pose(7.5), pose(3.4, .2)]
        r = g.fingerprint([0, 1], selected)
        hard, contacts = [], []
        for i, j in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3)]:
            a = g.placed(selected[i]); b = g.placed(selected[j] if j < 2 else state[j])
            d = np.linalg.norm(a[:, None]-b[None, :], axis=2)
            radius = g.radii[:, None]+g.radii[None, :]
            if (d < radius).any(): hard.append([i, j])
            if (d < radius+.6).any(): contacts.append([i, j])
        self.assertEqual(r['hard_overlap_edges'], hard)
        self.assertEqual(r['contacts'], contacts)
        self.assertFalse(r['wall_valid'])
        self.assertTrue(r['contacts'])  # Invalid-wall endpoints still classified.


if __name__ == '__main__':
    unittest.main()
