"""Deterministic finite-set coverage in rigid-member geometry."""
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from prepare_far_contact_centers import farthest_first
from prepare_far_contact_candidates import member_distances


class FarContactCenterTests(unittest.TestCase):
    def test_fixed_first_and_exact_tie_order(self):
        x = np.array([0., 2., -2., 2.])
        cover = farthest_first(abs(x[:, None]-x[None, :]), 0, .5)
        self.assertEqual(cover['center_candidate_indices'], [0, 1, 2])
        self.assertEqual(cover['nearest_center_indices'], [0, 1, 2, 1])
        self.assertEqual(cover['nearest_member_RMS_A'], [0., 0., 0., 0.])

    def test_closed_boundary_and_no_unnecessary_center(self):
        x = np.array([0., .5, np.nextafter(.5, 1.)])
        cover = farthest_first(abs(x[:, None]-x[None, :]), 0, .5)
        self.assertEqual(cover['center_candidate_indices'], [0, 2])
        self.assertLessEqual(max(cover['nearest_member_RMS_A']), .5)
        self.assertEqual(farthest_first(np.array([[0., .5], [.5, 0.]]), 0, .5)['center_candidate_indices'], [0])

    def test_assignment_tie_uses_first_selected_center(self):
        x = np.array([0., 1., 2.])
        cover = farthest_first(abs(x[:, None]-x[None, :]), 2, 1.)
        self.assertEqual(cover['center_candidate_indices'], [2, 0])
        self.assertEqual(cover['nearest_center_indices'][1], 0)

    def test_coverage_and_center_separation_in_rigid_member_frame(self):
        members = np.array([[1., 2., 3.], [-1., -2., 3.], [1., -2., -3.], [-1., 2., -3.]])
        poses = [dict(position=[x, 0., 0.], orientation=[1., 0., 0., 0.]) for x in (0., .3, 1.2, -2.)]
        distances = member_distances(poses, members); cover = farthest_first(distances, 0, .5)
        self.assertEqual(cover['center_candidate_indices'], [0, 3, 2])
        selected = cover['center_candidate_indices']
        self.assertTrue(np.all(np.min(distances[:, selected], axis=1) <= .5))
        self.assertTrue(all(distances[a, b] > .5 for i, a in enumerate(selected) for b in selected[:i]))
        rotation = Rotation.from_rotvec([.3, -.4, .7]); quaternion = rotation.as_quat()[[3, 0, 1, 2]].tolist()
        moved = [dict(position=(rotation.apply(p['position'])+[8., -2., 4.]).tolist(), orientation=quaternion) for p in poses]
        moved_distances = member_distances(moved, members)
        np.testing.assert_allclose(moved_distances, distances, atol=1e-14)
        self.assertEqual(farthest_first(moved_distances, 0, .5)['center_candidate_indices'], selected)

    def test_invalid_geometry_and_zero_radius_duplicates(self):
        cover = farthest_first(np.zeros((3, 3)), 1, 0.)
        self.assertEqual(cover['center_candidate_indices'], [1])
        for matrix in [np.array([[1.]]), np.array([[0., -1.], [-1., 0.]]), np.array([[0., 1.], [2., 0.]]), np.array([[0., np.inf], [np.inf, 0.]])]:
            with self.assertRaises(ValueError): farthest_first(matrix, 0, .5)


if __name__ == '__main__': unittest.main()
