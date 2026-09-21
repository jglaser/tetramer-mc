"""Matched target and all-mobile preparation checks; no physical simulation."""
import copy
from pathlib import Path
import unittest
from run_mobile_four_body_growth import (
    DESIGN, EXAMPLE, RADIUS, SEED_BASE, STARTS, command, configured, read,
)


class FourBodyCampaign(unittest.TestCase):
    def setUp(self):
        self.example = read(EXAMPLE)
        self.starts = read(DESIGN/'recommended-poses.json')

    def make(self, arm='original', start='triangle_free', replicate=0, index=0):
        return configured(self.example, self.starts, Path('/tmp')/arm, arm, start, replicate, index)

    def test_config_changes_no_move_or_bath_parameter(self):
        original = copy.deepcopy((self.example, self.starts)); cfg = self.make()
        for k in ('depletant_radius', 'reservoir_density', 'poisson_lambda_ratio', 'endpoint_gate',
                  'global_probability', 'learned_uniform_weight', 'local_translation_std_A',
                  'local_small_angle_std_degrees', 'gca_probability', 'center_shift_probability',
                  'frozen_posterior', 'method'):
            self.assertEqual(cfg[k], self.example[k], k)
        self.assertEqual((self.example, self.starts), original)
        self.assertEqual(cfg['boundary'], dict(kind='spherical', radius=RADIUS))
        self.assertEqual(cfg['box_lengths'], [2*RADIUS]*3)
        self.assertEqual(cfg['fixed_body_indices'], [])
        self.assertEqual(cfg['seed_labels'], [])

    def test_each_start_is_identical_between_proposal_arms(self):
        for start in STARTS:
            a = self.make(start=start); b = self.make('coverage', start, 1, 7)
            omit = {'seed', 'metadata', 'shape', 'monomer_shape'}
            self.assertEqual({k:v for k,v in a.items() if k not in omit},
                             {k:v for k,v in b.items() if k not in omit})
        self.assertEqual(self.make()['seed'], SEED_BASE)
        self.assertEqual(self.make(index=7)['seed'], SEED_BASE+7*1009)

    def test_only_fourth_pose_changes_between_starts(self):
        free, bound = [self.make(start=start) for start in STARTS]
        self.assertEqual(len(free['initial_poses']), 4)
        self.assertEqual(free['initial_poses'][:3], bound['initial_poses'][:3])
        self.assertNotEqual(free['initial_poses'][3], bound['initial_poses'][3])
        self.assertEqual(free['initial_poses'][3]['position'], [0., 0., 0.])
        self.assertEqual(bound['metadata']['initial_fragments'], [[0, 1, 2, 3]])

    def test_all_four_body_starts_fit_atomic_wall(self):
        import numpy as np
        from scipy.spatial.transform import Rotation
        shape = read(EXAMPLE.parent/'tetramer-shape.json')
        centers = np.asarray([a['center'] for a in shape['atoms']]); radii = np.asarray([a['radius'] for a in shape['atoms']])
        for start in STARTS:
            for pose in self.make(start=start)['initial_poses']:
                q = np.asarray(pose['orientation']); rotation = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
                world = centers@rotation.T+pose['position']
                self.assertGreater(float(np.min(RADIUS-np.linalg.norm(world, axis=1)-radii)), 0.)

    def test_command_uses_assembly_and_preserves_every_sweep(self):
        argv = command(Path('/tmp/arm'), dict(config='/tmp/cfg', directory='/tmp/out'))
        self.assertEqual(argv[1], 'run')
        self.assertEqual(argv[argv.index('--sample-every')+1], '1')
        self.assertEqual(argv[argv.index('--sweeps')+1], '2000')
        self.assertNotIn('--proposal-anchor-index', argv)
        self.assertNotIn('--region', argv)


if __name__ == '__main__': unittest.main()
