"""Proposal allocation, isolated width control, and original-q cover checks."""
import copy
import math
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from prepare_shoulder_contact_atlas import assemble, cover_model, CENTERS
from analyze_native_region_reference import native_q, cover_coordinates


def fixture():
    one = dict(schema='weighted-pose-mixture-v1', angular_length=10.,
        coordinate_convention='anchor-body-relative', shape_sha256='shape',
        anchors=[dict(position=[0., 0., 0.], rotation=np.eye(3).tolist())],
        means=[[0.]*6], covariances=[np.eye(6).tolist()], weights=[1.])
    legacy = copy.deepcopy(one)
    for key in ('anchors', 'means', 'covariances'):
        legacy[key] += copy.deepcopy(legacy[key])
    legacy['weights'] = [.3, .7]
    fits = {name: copy.deepcopy(one) for name in CENTERS}
    for i, name in enumerate(CENTERS):
        fits[name]['means'][0][0] = i+1.
    return legacy, fits


class ShoulderAtlasPreparationTests(unittest.TestCase):
    def test_mass_allocation_normalizes_and_retains_legacy_family(self):
        legacy, fits = fixture(); before = copy.deepcopy((legacy, fits))
        masses = dict(zip(CENTERS, map(math.log, (3., 1., 6.))))
        model, allocation = assemble(legacy, fits, masses, 1.)
        np.testing.assert_allclose(model['weights'], [.15, .35, .15, .05, .3])
        self.assertAlmostEqual(sum(model['weights']), 1.)
        np.testing.assert_allclose(list(allocation.values()), [.3, .1, .6])
        self.assertEqual((legacy, fits), before)
        shifted, _ = assemble(legacy, fits, {k: v+100 for k, v in masses.items()}, 1.)
        np.testing.assert_allclose(shifted['weights'], model['weights'], rtol=2e-14)

    def test_broad_control_changes_only_new_covariances(self):
        legacy, fits = fixture(); masses = {name: 0. for name in CENTERS}
        narrow, _ = assemble(legacy, fits, masses, 1.)
        broad, _ = assemble(legacy, fits, masses, 4.)
        for key in narrow:
            if key != 'covariances': self.assertEqual(narrow[key], broad[key])
        self.assertEqual(narrow['covariances'][:2], broad['covariances'][:2])
        np.testing.assert_array_equal(np.asarray(broad['covariances'][2:]), 4*np.asarray(narrow['covariances'][2:]))

    def test_invalid_or_incompatible_sources_are_rejected(self):
        legacy, fits = fixture(); masses = {name: 0. for name in CENTERS}
        with self.assertRaises(ValueError): assemble(legacy, fits, dict(masses, geometry=math.nan), 1.)
        with self.assertRaises(ValueError): assemble(legacy, fits, masses, 2.)
        fits['mixture']['angular_length'] = 11.
        with self.assertRaises(ValueError): assemble(legacy, fits, masses, 1.)

    def test_product_cover_contains_original_q_targets_without_changing_metric(self):
        identity = dict(position=[0., 0., 0.], orientation=[1., 0., 0., 0.])
        positions = [[-10., -10., -10.], [-10., 10., 10.], [10., -10., 10.], [10., 10., -10.]]
        metric = dict(native_poses=[identity], rigid_members=[dict(position=p, orientation=identity['orientation']) for p in positions],
                      member_error_scale=2., angle_error_scale_deg=15.)
        cfg = dict(metadata=metric); original = copy.deepcopy(cfg)
        cover, proof = cover_model(cfg)
        self.assertEqual(cfg, original)
        self.assertEqual(cover['covers'][0]['ball_radius'], 4.)
        self.assertEqual(proof['enclosing_metric']['member_error_scale'], 4.)
        self.assertGreater(cover['covers'][0]['angle_cap'], 0.)
        self.assertLess(cover['covers'][0]['angle_cap'], math.radians(30.))
        rng = np.random.default_rng(991)
        found = 0
        for _ in range(500):
            q = Rotation.from_rotvec(rng.normal(size=3)*.08).as_quat()[[3, 0, 1, 2]].tolist()
            pose = dict(position=(rng.normal(size=3)*1.2).tolist(), orientation=q)
            if native_q(metric, pose) <= 2.:
                radius, angle = cover_coordinates(cover['covers'][0], pose)
                self.assertLessEqual(radius, 4.)
                self.assertLessEqual(angle, cover['covers'][0]['angle_cap'])
                found += 1
        self.assertGreater(found, 30)


if __name__ == '__main__':
    unittest.main()
