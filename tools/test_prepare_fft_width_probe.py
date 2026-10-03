"""Exact covariance-only transformation and immutable historical allocation checks."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import prepare_fft_width_probe as prep


def model():
    lower = np.eye(6)
    lower[3, 0] = .3
    lower[4, 1] = -.2
    lower[5, 2] = .1
    covariance = lower @ lower.T
    return dict(schema='reciprocal-pose-mixture-v1', reciprocal_components=[True, False],
        base_model=dict(schema='weighted-pose-mixture-v1', covariances=[covariance.tolist(), (2*covariance).tolist()],
            means=[[1, 2, 3, 4, 5, 6], [-1, -2, -3, -4, -5, -6]],
            weights=[.3, .7], anchors=[dict(position=[1, 2, 3]), dict(position=[-1, -2, -3])],
            angular_length=35., coordinate_convention='unchanged', shape_sha256='fixed'))


def history_fixture(root, method='factorized'):
    (root/'execution').mkdir()
    rows = []
    for c in range(8):
        for j in range(32):
            for atlas in [0, 1, 2]:
                rows.append(json.dumps(dict(atlas_index=atlas, case_index=c, attempt=j,
                    method=method, status='completed', candidate=None if j % 2 else 'candidate'))+'\n')
    (root/'execution/attempts.jsonl').write_text(''.join(rows))
    (root/'analysis.json').write_text(json.dumps(dict(complete=True, passed=True, failures=[])))
    receipt = dict(complete=True, passed=True, audit_exit_code=0,
        output_hashes={name: prep.sha(root/name) for name in ['analysis.json', 'execution/attempts.jsonl']})
    (root/'completed-review.json').write_text(json.dumps(receipt))
    return prep.sha(root/'completed-review.json')


class WidthPreparationTests(unittest.TestCase):
    def test_complete_cross_covariance_transform_preserves_every_other_field(self):
        original = model()
        snapshot = copy.deepcopy(original)
        cov = np.asarray(original['base_model']['covariances'])
        for tau in prep.SCALES:
            scaled = prep.scale_atlas(original, tau)
            np.testing.assert_array_equal(scaled['base_model']['covariances'], cov*(tau*tau))
            self.assertNotEqual(scaled['base_model']['covariances'][0][0][3], 0.)
            np.testing.assert_allclose(np.linalg.cholesky(scaled['base_model']['covariances']),
                np.linalg.cholesky(cov)*tau, rtol=2e-15, atol=1e-16)
            scaled['base_model']['covariances'] = original['base_model']['covariances']
            self.assertEqual(scaled, original)
        self.assertEqual(original, snapshot)

    def test_invalid_or_extended_transforms_rejected(self):
        for tau in [0., 1., -.25, float('nan'), float('inf'), True]:
            with self.subTest(tau=tau), self.assertRaises(ValueError):
                prep.scale_atlas(model(), tau)
        for defect in ['schema', 'length', 'dimension', 'nan', 'asymmetry', 'indefinite', 'ambiguous']:
            value = model()
            if defect == 'schema': value['schema'] = 'unknown'
            elif defect == 'length': value['base_model']['weights'].pop()
            elif defect == 'dimension': value['base_model']['covariances'][0].pop()
            elif defect == 'nan': value['base_model']['covariances'][0][0][0] = float('nan')
            elif defect == 'asymmetry': value['base_model']['covariances'][0][0][1] = .5
            elif defect == 'indefinite': value['base_model']['covariances'][0][0][0] = -1.
            else: value['base_model']['scales'] = [1.]
            with self.subTest(defect=defect), self.assertRaises((ValueError, np.linalg.LinAlgError)):
                prep.scale_atlas(value, .25)

    def test_historical_bytes_preserve_nulls_without_redrawing(self):
        for method in ['factorized', 'm4']:
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                digest = history_fixture(root, method)
                rows = prep.history_rows(root, digest, method)
                original = (root/'execution/attempts.jsonl').read_bytes().splitlines(keepends=True)
                self.assertEqual(rows, original[1::3])
                self.assertEqual(len(rows), 256)
                self.assertEqual(sum(json.loads(x)['candidate'] is None for x in rows), 128)
                (root/'execution/attempts.jsonl').write_bytes(b''.join(original[:-2]))
                with self.assertRaisesRegex(ValueError, 'Changed historical output'):
                    prep.history_rows(root, digest, method)

    def test_historical_relabel_or_failed_review_rejected_even_when_rehashed(self):
        for defect in ['failed', 'slot', 'review', 'analysis']:
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                history_fixture(root)
                receipt = prep.read(root/'completed-review.json')
                if defect == 'review': receipt['passed'] = False
                elif defect == 'analysis':
                    path = root/'analysis.json'
                    value = prep.read(path); value['passed'] = False
                    path.write_text(json.dumps(value))
                    receipt['output_hashes']['analysis.json'] = prep.sha(path)
                else:
                    path = root/'execution/attempts.jsonl'
                    rows = path.read_text().splitlines(keepends=True)
                    value = json.loads(rows[1])
                    value['status' if defect == 'failed' else 'attempt'] = 'fatal' if defect == 'failed' else 99
                    rows[1] = json.dumps(value)+'\n'
                    path.write_text(''.join(rows))
                    receipt['output_hashes']['execution/attempts.jsonl'] = prep.sha(path)
                (root/'completed-review.json').write_text(json.dumps(receipt))
                with self.subTest(defect=defect), self.assertRaises(ValueError):
                    prep.history_rows(root, prep.sha(root/'completed-review.json'), 'factorized')

    def test_fixed_allocation_and_no_hidden_uniform_change(self):
        a = prep.allocation()
        self.assertEqual(a['contexts']*a['slots_per_context'], a['clouds'])
        self.assertEqual(a['clouds']*len(a['covariance_scales'])*len(a['methods']), 1536)
        self.assertEqual(a['clouds']*a['raw_points_per_cloud']*24, 100663296)
        self.assertEqual(a['new_outer_attempts']*(a['caps']['root']+a['caps']['internal']), 98304)
        self.assertEqual((a['uniform_probability'], a['uniform_half_width_A']), (.5, 160.))
        self.assertEqual(a['historical_new_draws'], 0)
        self.assertEqual(a['physical_bath_draws'], 0)
        self.assertFalse(a['extension'])


if __name__ == '__main__':
    unittest.main()
