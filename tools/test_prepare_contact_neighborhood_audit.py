import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from prepare_contact_neighborhood_audit import relative_scaffolds


class NeighborhoodTests(unittest.TestCase):
    def test_each_anchor_reconstructs_every_supplied_body(self):
        rng = np.random.default_rng(671)
        rotations = Rotation.random(4, random_state=rng)
        translations = rng.normal(size=(4, 3))*70
        poses = [dict(position=p.tolist(), orientation=q[[3, 0, 1, 2]].tolist())
                 for p, q in zip(translations, rotations.as_quat())]
        rows = relative_scaffolds(poses, [3, 0, 2])
        for row in rows:
            a = row['anchor_index']
            self.assertEqual(row['poses'][0], dict(position=[0., 0., 0.], orientation=[1., 0., 0., 0.]))
            self.assertEqual(set(row['fixed_body_indices']), {0, 2, 3})
            for i, pose in zip(row['fixed_body_indices'], row['poses']):
                q = np.asarray(pose['orientation'])
                relative = Rotation.from_quat(q[[1, 2, 3, 0]])
                np.testing.assert_allclose(rotations[a].apply(pose['position'])+translations[a], translations[i], atol=1e-12)
                np.testing.assert_allclose((rotations[a]*relative).as_matrix(), rotations[i].as_matrix(), atol=1e-12)

    def test_invalid_or_repeated_scaffold_rejected(self):
        poses = [dict(position=[0., 0., 0.], orientation=[1., 0., 0., 0.])]
        for indices in [[], [0, 0], [1], [-1]]:
            with self.assertRaises(ValueError):
                relative_scaffolds(poses, indices)
        poses[0]['orientation'] = [2., 0., 0., 0.]
        with self.assertRaises(ValueError):
            relative_scaffolds(poses, [0])


if __name__ == '__main__':
    unittest.main()
