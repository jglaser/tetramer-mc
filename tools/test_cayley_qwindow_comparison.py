#!/usr/bin/env python3
"""Keep unconditional denominators when selecting a physical q subregion."""
import json
import math
from pathlib import Path
import tempfile
import unittest

from compare_cayley_qwindow import load_latent
from prepare_cayley_rms_cover import sha, write


class MatchingBandTests(unittest.TestCase):
    def fixture(self, root):
        provenance = root/'provenance'; population = root/'runs/r00'
        provenance.mkdir(); population.mkdir(parents=True)
        p = lambda x: {'position': [x, 0., 0.], 'orientation': [1., 0., 0., 0.]}
        metadata = {'native_poses': [p(0)], 'rigid_members': [p(-1), p(1)],
                    'member_error_scale': 1., 'angle_error_scale_deg': 15.}
        cfg = {'fixed_poses': [p(100)], 'capture_center': [0., 0., 0.], 'capture_radius': 18.,
               'depletant_radius': 1.5, 'reservoir_density': .035, 'metadata': metadata}
        write(provenance/'config.json', cfg)
        write(provenance/'shape.json', {'atoms': [{'center': [0., 0., 0.], 'radius': .2}]})
        shape_hash = sha(provenance/'shape.json')
        region = {'minimum_original_q': 1., 'maximum_original_q': 2., 'physical_metric': metadata,
                  'physical_fixed_neighbors': cfg['fixed_poses'], 'shape_sha256': shape_hash}
        write(provenance/'region.json', region)
        rows = []
        for i, q in enumerate([1., 1.05, 1.5, 2.]):
            valid = 1 < q < 2
            row = {'draw': i, 'q': q, 'pose': p(q), 'region_valid': valid, 'capture_valid': True, 'hard_valid': True}
            if valid:
                # Unit hard weight, physical weights 2 and 20 in separate bands.
                logw = math.log(2 if q < 1.1 else 20)
                row.update(log_physical_jacobian=0., log_hard_weight=0., log_importance_weight=logw,
                           clouds=[{'log_weight': logw}, {'log_weight': logw}])
            rows.append(row)
        (population/'samples.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
        digest = sha(population/'samples.jsonl')
        write(population/'summary.json', {'complete': True, 'samples': 4, 'samples_sha256': digest})
        rsha = sha(provenance/'region.json')
        write(root/'manifest.json', {'region_sha256': rsha, 'archive_sha256': {'shape.json': shape_hash},
                                    'lambda_ratio': 64., 'cloud_replicates': 2,
                                    'jobs': [{'id': 'r00', 'seed': 10, 'samples': 4, 'directory': str(population)}]})
        (root/'assessment').mkdir()
        write(root/'assessment/analysis.json', {'region_sha256': rsha, 'estimate': {'draws': 4},
              'independently_reconstructed_poses': 4, 'sampler_cpu_seconds': 2., 'log_latent_volume': 0.,
              'original_q_window': {'minimum': 1., 'maximum': 2.}, 'populations': [{'id': 'r00', 'samples_sha256': digest}]})

    def test_band_keeps_all_unconditional_zeros_and_excludes_endpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); self.fixture(root)
            result = load_latent(root, 1., 1.1)
            self.assertEqual(result['physical']['draws'], 4)
            self.assertEqual(result['physical']['nonzero'], 1)
            self.assertAlmostEqual(math.exp(result['physical']['logQ']), .5)
            self.assertAlmostEqual(math.exp(result['hard']['logQ']), .25)
            self.assertEqual(result['CPU_seconds'], 2.)
            self.assertEqual(result['populations'][0]['physical']['draws'], 4)

    def test_outside_band_and_changed_shape_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); self.fixture(root)
            with self.assertRaises(ValueError): load_latent(root, .9, 1.1)
            write(root/'provenance/shape.json', {'changed': True})
            with self.assertRaises(ValueError): load_latent(root, 1., 1.1)

    def test_unweighted_population_scatter_requires_equal_budgets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); self.fixture(root)
            manifest = json.loads((root/'manifest.json').read_text())
            extra = dict(manifest['jobs'][0], id='r01', seed=11, samples=8)
            manifest['jobs'].append(extra); write(root/'manifest.json', manifest)
            with self.assertRaisesRegex(ValueError, 'equal unconditional population budgets'):
                load_latent(root, 1., 1.1)


if __name__ == '__main__': unittest.main()
