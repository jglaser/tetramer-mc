"""Small independent controls; no protein queries or stochastic draws."""
import copy
import unittest

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from audit_dimer_mean_clearance import minimum_gap, validate_queries


class MeanClearanceTests(unittest.TestCase):
    def test_radius_classes_match_all_pairs_and_not_nearest_center_only(self):
        centers = np.array([[0., 0., 0.], [2., 0., 0.], [-1., 4., 0.]])
        radii = np.array([.1, 3., .4])
        points = centers + np.array([.5, 0., 0.])
        groups = [(float(r), np.flatnonzero(radii == r), cKDTree(centers[radii == r]))
                  for r in np.unique(radii)]
        got = minimum_gap(points, centers, radii, groups)
        distances = np.linalg.norm(points[:, None] - centers[None, :], axis=2)
        reference = distances - radii[:, None] - radii[None, :]
        self.assertAlmostEqual(got['clearance_A'], float(reference.min()), places=14)
        # Atom0's closest center is atom0, but its smallest surface gap is to atom1.
        self.assertEqual(int(np.argmin(distances[0])), 0)
        self.assertEqual(int(np.argmin(reference[0])), 1)

    def test_reciprocal_has_same_exact_minimum(self):
        centers = np.array([[0., 0., 0.], [2., 0., 0.], [-1., 4., 0.]])
        radii = np.array([.1, 3., .4])
        groups = [(float(r), np.flatnonzero(radii == r), cKDTree(centers[radii == r]))
                  for r in np.unique(radii)]
        rotation = Rotation.from_rotvec([.2, -.4, .1]).as_matrix()
        position = np.array([7., 2., 1.])
        forward = minimum_gap(centers @ rotation.T + position, centers, radii, groups)
        reverse = minimum_gap((centers - position) @ rotation, centers, radii, groups)
        self.assertAlmostEqual(forward['clearance_A'], reverse['clearance_A'], places=13)

    def test_query_allocation_rejects_filtering_reordering_and_noncenters(self):
        config = dict(scientific_threads=1, queries=2, atlases=[dict(name='a', virtual_branches=2)])
        queries = [dict(query=i, atlas='a', branch=i, branch_weight=.5, latent=[0.] * 6,
                        relative_pose=dict(position=[0., 0., 0.], orientation=[1., 0., 0., 0.]))
                   for i in range(2)]
        validate_queries(queries, config)
        bads = [queries[:1], list(reversed(queries))]
        noncenter = copy.deepcopy(queries)
        noncenter[0]['latent'][0] = .1
        bads.append(noncenter)
        duplicate = copy.deepcopy(queries)
        duplicate[1]['branch'] = 0
        bads.append(duplicate)
        for bad in bads:
            with self.assertRaises(ValueError):
                validate_queries(bad, config)


if __name__ == '__main__':
    unittest.main()
