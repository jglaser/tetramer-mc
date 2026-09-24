"""Allocation, analytic geometry, independent quadrature and audit fixture tests."""
import copy
import math
from pathlib import Path
import tempfile
import unittest

import validate_native_excluded_smc_reference as reference


class ReferenceTests(unittest.TestCase):
    def test_quadrature_and_unbound_identity(self):
        for z, exact in reference.EXACT.items():
            total = reference.reference_integral(z, 2.2)
            contact = reference.reference_integral(z, 1.4)
            self.assertAlmostEqual(total, exact['total'], places=12)
            self.assertAlmostEqual(contact, exact['contact'], places=12)
            self.assertAlmostEqual(total-contact, .002892781726561475, places=12)

    def test_compiled_fixture_and_density_geometry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'fixture'; reference.fixture(root, 4.)
            config = reference.read(root / 'config.json'); region = reference.read(root / 'region.json')
            adapter = reference.AnalyticSphereNative(root / 'shape.json')
            self.assertEqual(adapter.expected_compiled, reference.read(root / 'native-compiled.json'))
            self.assertEqual(config['reservoir_density'], 4.)
            cov = region['gaussian_chart']['covariances'][0]
            self.assertEqual([cov[i][i] for i in range(6)], [.36]*3 + [.0009]*3)
            anchor = dict(position=[0.,0.,0.], orientation=[1.,0.,0.,0.])
            pose = dict(position=[1.5,0.,0.], orientation=[1.,0.,0.,0.])
            matches = adapter.classify_pair(anchor, pose)
            self.assertEqual(matches[0]['motif_id'], 0)
            self.assertAlmostEqual(matches[0]['supporting_member_bonds'][0]['minimum_gap_A'], .9)
            pose['position'] = [-.6,0.,0.]
            self.assertEqual(adapter.classify_pair(anchor, pose), [])
            pose['position'] = [1.5,0.,0.]
            pose['orientation'] = [math.cos(math.radians(8)),0.,0.,math.sin(math.radians(8))]
            self.assertEqual(adapter.classify_pair(anchor, pose), [])

    def test_predicate_boundary_and_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'fixture'; reference.fixture(root, 0.)
            adapter = reference.AnalyticSphereNative(root / 'shape.json')
            anchor = dict(position=[4.,5.,6.], orientation=[math.sqrt(.5),0.,0.,math.sqrt(.5)])
            pose = dict(position=[4.,6.5,6.], orientation=anchor['orientation'])
            self.assertEqual(len(adapter.classify_pair(anchor, pose)), 1)
            pose['position'] = [4.,4.4,6.]
            self.assertEqual(adapter.classify_pair(anchor, pose), [])

    def test_seed_inventory_nested_and_fresh(self):
        seeds = reference.seed_values(dict(jobs=[dict(seed=1), dict(nested=dict(seeds=[2,3]))], seed=True))
        self.assertEqual(seeds, {1,2,3})
        self.assertEqual(len(set(reference.SEEDS)), 17)
        self.assertEqual(reference.SEEDS[0], 148301010)
        self.assertEqual(reference.SEEDS[-1], 148317154)

    def test_exact_command_allocation(self):
        job = dict(id='z0-r00', fixture='z0', seed=reference.SEEDS[0], initial_draws=4096)
        argv = reference.command('/example', job)
        pairs = dict(zip(argv[1::2], argv[2::2]))
        self.assertEqual(pairs['--initial-draws'], '4096')
        self.assertEqual(pairs['--population'], '192')
        self.assertEqual(pairs['--stages'], '8')
        self.assertEqual(pairs['--sweeps-per-stage'], '6')
        self.assertEqual(pairs['--cloud-replicates'], '2')
        self.assertEqual(pairs['--lambda-ratio'], '8')
        self.assertEqual(pairs['--bridge'], 'proposal-density')

    def test_fixture_refuses_overwrite_and_zero_geometry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'fixture'; reference.fixture(root, 0., 3.)
            with self.assertRaises(FileExistsError): reference.fixture(root, 0.)
            config = reference.read(root / 'config.json')
            shape = reference.read(root / 'shape.json')
            self.assertGreater(2*shape['atoms'][0]['radius'], config['capture_radius'])


if __name__ == '__main__':
    unittest.main()
