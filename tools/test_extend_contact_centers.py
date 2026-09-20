"""Preservation, explicit-center, coverage and deterministic tie controls."""
import unittest

import numpy as np

from extend_contact_centers import extend_indices


class ExtendContactCentersTest(unittest.TestCase):
    def test_preserve_order_and_append_already_covered_mandatory_center(self):
        positions = np.array([0., 1., .1, 2.])
        matrix = np.abs(positions[:, None]-positions[None, :])
        selected, _ = extend_indices(matrix, [1, 0], 2, .5)
        self.assertEqual(selected, [1, 0, 2, 3])
        self.assertLessEqual(np.min(matrix[:, selected], axis=1).max(), .5)

    def test_farthest_tie_uses_first_candidate_and_boundary_is_closed(self):
        positions = np.array([0., .1, 2., -2., .6])
        matrix = np.abs(positions[:, None]-positions[None, :])
        selected, _ = extend_indices(matrix, [0], 1, .5)
        # Mandatory .1 makes -2 farther; after that, +2 must be covered as well.
        self.assertEqual(selected, [0, 1, 3, 2])
        self.assertNotIn(4, selected)
        tied = np.abs(np.array([0., 2., -2.])[:, None]-np.array([0., 2., -2.])[None, :])
        self.assertEqual(extend_indices(tied, [0], 0, .5)[0], [0, 1, 2])

    def test_asymmetric_matrix_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'symmetric'):
            extend_indices(np.array([[0., 1.], [2., 0.]]), [0], 1)


if __name__ == '__main__':
    unittest.main()
