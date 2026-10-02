#!/usr/bin/env python3
"""Pure analytic/fixture checks; no compiled sampler or physical jobs invoked."""
import math
from pathlib import Path
import tempfile
import unittest
from scipy.integrate import quad
from physical_hard_free_line_vessel import PhysicalHardFreeLineGuide
import prepare_hard_free_smc_reference as prep


class ReferencePreparationTests(unittest.TestCase):
    def test_haar_angle_cap_and_lens_integrals(self):
        for activity in (0., 4.):
            for upper in (1.4, 2.2):
                def oracle(radius):
                    theta = 2*math.atan((1.3/1.1)*math.sqrt(max(0., 16-(radius/.6)**2)))
                    haar = (theta-math.sin(theta))/math.pi
                    overlap = math.pi*(2.8+radius)*max(0., 1.4-radius)**2/12
                    return 4*math.pi*radius**2*haar*math.exp(activity*overlap)
                expected = quad(oracle, .6, upper, points=[1.4] if upper > 1.4 else None,
                                epsabs=1e-11, epsrel=1e-12)[0]
                self.assertAlmostEqual(prep.reference_integral(activity, upper), expected, delta=1e-10)
        self.assertGreater(prep.reference_integral(4.), prep.reference_integral(0.))
        with self.assertRaises(ValueError): prep.reference_integral(0., subdivisions=3)

    def test_fixed_fixture_and_normalized_guide_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)/'z4'; prep.fixture(root, 4.)
            shape, region, config, guide = [prep.read(root/name) for name in
                ['shape.json', 'region.json', 'config.json', 'guide.json']]
            model = PhysicalHardFreeLineGuide(region, guide, config, shape,
                region_sha256=prep.sha(root/'region.json'), expected_shape_sha256=prep.sha(root/'shape.json'))
            self.assertEqual(model.recon.alpha, .5)
            self.assertEqual(model.recon.beta, 1.)
            self.assertEqual(model.recon.axes, [0, 1, 2])
            self.assertEqual(len(guide['gaussian_components']), 2)
            self.assertAlmostEqual(sum(c['weight'] for c in guide['gaussian_components']), 1.)
            self.assertNotEqual(guide['gaussian_components'][0]['mean'], guide['gaussian_components'][1]['mean'])
            self.assertEqual(region['gaussian_chart']['covariances'][0], prep.diagonal([.6]*3+[1.3]*3))
            uniform = prep.read(root/'guide-uniform.json')
            self.assertEqual(uniform, dict(guide, defensive_uniform_shell_probability=1.))
            zero = Path(temporary)/'zero'; prep.fixture(zero, 0., 2.)
            self.assertGreater(2*prep.read(zero/'shape.json')['atoms'][0]['radius'], prep.CAPTURE)

    def test_exact_allocation_and_intentional_parity_pair(self):
        rows = prep.jobs(Path('/tmp/frozen-reference'))
        self.assertEqual(len(rows), 11)
        self.assertEqual(len({r['seed'] for r in rows}), 10)
        self.assertEqual(sum(r['initial_draws'] for r in rows), 33536)
        for z in (0, 4):
            main = [r for r in rows if r['purpose'] == 'analytic' and r['activity'] == z]
            self.assertEqual(len(main), 4)
            self.assertTrue(all((r['initial_draws'], r['population'], r['stages'], r['sweeps']) ==
                                (4096, 192, 8, 6) for r in main))
        pair = [r for r in rows if r['purpose'] == 'paired-uniform-parity']
        self.assertEqual(pair[0]['seed'], pair[1]['seed'])
        self.assertIn('--initial-guide', pair[0]['argv'])
        self.assertNotIn('--initial-guide', pair[1]['argv'])
        for r in rows:
            self.assertIn('proposal-density', r['argv'])
            self.assertNotIn('--exclude-native-entry', r['argv'])


if __name__ == '__main__':
    unittest.main()
