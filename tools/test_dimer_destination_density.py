"""Deterministic coefficient, reciprocal-map and recorded-trace controls."""
import copy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import dimer_destination_density as audit


def fixture(correlated=False):
    lower = np.diag([1.1, 1.2, .7, .8, 1.3, .9])
    if correlated:
        lower[np.tril_indices(6, -1)] = [.3, -.2, .4, .7, .1, -.3, .2, -.8, .6, .1,
                                        .2, .3, -.4, .1, .9]
    base = dict(schema='weighted-pose-mixture-v1', coordinate_convention='anchor-body-relative',
                angular_length=2., shape_sha256='a'*64,
                anchors=[dict(position=[.2, -.1, .3], rotation=np.eye(3).tolist())],
                means=[[.1, .2, -.1, .05, -.03, .02]],
                covariances=[audit.exported_covariance(lower).tolist()], weights=[1.])
    return dict(schema='reciprocal-pose-mixture-v1', base_model=base, reciprocal_components=[True])


class DimerDensityTests(unittest.TestCase):
    def test_constructor_keeps_distinct_arithmetic_and_reports_exact_covariance_difference(self):
        value = audit.DimerDestinationDensity(fixture(correlated=True))
        self.assertEqual(value.factor_report['different_factor_components'], 1)
        self.assertGreater(value.factor_report['factor_absolute']['value'], 0.)
        self.assertGreater(value.factor_report['covariance_exact_real_absolute']['value'], 0.)
        self.assertLess(value.factor_report['covariance_exact_per_component_maxnorm_relative'], 1e-14)
        self.assertEqual(value.score.weights.tolist(), [.5, .5])

    def test_reciprocal_decode_encode_and_correlated_inverse(self):
        value = audit.DimerDestinationDensity(fixture(correlated=True))
        latent = np.array([.7, -.3, .2, .1, -.2, .4])
        for branch in (0, 1):
            pose = value.decode(branch, latent)
            np.testing.assert_allclose(value.encode(branch, pose), latent, atol=2e-14, rtol=0.)
        old = value.decode(1, latent)
        forward = value.reconstruct_tree_edge(old, 1, 0, [.2, -.4, .1, .7, .3, -.6])
        inverse = forward['inverse_trace']
        backward = value.reconstruct_tree_edge(forward['pose'], inverse['source'], inverse['target'], inverse['noise'])
        self.assertLess(max(audit.pose_errors(old, backward['pose']).values()), 3e-14)
        self.assertAlmostEqual(forward['log_extended_jacobian']+backward['log_extended_jacobian'], 0., places=12)
        self.assertAlmostEqual(forward['log_auxiliary_ratio']+backward['log_auxiliary_ratio'], 0., places=12)

    def test_complete_defensive_density_and_closed_cube_boundary(self):
        value = audit.DimerDestinationDensity(fixture())
        pose = dict(position=[3., 0., 0.], orientation=[1., 0., 0., 0.])
        a = value.edge_density(pose, .5, 3.)
        self.assertEqual(a['log_uniform'], -3*(math.log(2)+math.log(3)))
        self.assertAlmostEqual(a['log_full'], float(np.logaddexp(a['log_uniform'], a['log_learned']))-math.log(2))
        exterior = copy.deepcopy(pose)
        exterior['position'][0] = math.nextafter(3., math.inf)
        self.assertEqual(value.edge_density(exterior, 1., 3.)['log_full'], -math.inf)
        self.assertEqual(value.edge_density(pose, 0., 3.)['log_full'], a['log_learned'])
        for alpha in [-.1, 1.1, float('nan')]:
            with self.assertRaises(ValueError): value.edge_density(pose, alpha, 3.)

    def test_cached_tree_audit_never_rescores_and_detects_corrupt_trace(self):
        value = audit.DimerDestinationDensity(fixture())
        old = value.decode(0, [.1, .2, .3, -.1, .2, -.3])
        noise = [.3, -.2, .1, .6, -.5, .4]
        rebuilt = value.reconstruct_tree_edge(old, 0, 1, noise)
        edge = dict(old_relative_pose=old, trace=dict(source=0, target=1, noise=noise),
                    step=dict(handle=rebuilt['pose'], step=rebuilt,
                              **{key:rebuilt[key] for key in ['full_old_member_log_density',
                                  'full_new_member_log_density', 'log_reverse_forward',
                                  'label_log_reverse_forward', 'expanded_log_reverse_forward']}))
        full, _, components = value.score.evaluate([old, rebuilt['pose']])
        cache = (full[0], full[1], components[0, 0], components[1, 1])
        with patch.object(value.score, 'evaluate', side_effect=AssertionError('unexpected rescoring')):
            result = value.audit_tree_edge(edge, density_values=cache)
        self.assertLess(max(result['errors'].values()), 1e-12)
        bad = copy.deepcopy(edge)
        bad['step']['step']['inverse_trace']['noise'][0] += .1
        self.assertGreater(value.audit_tree_edge(bad, density_values=cache)['errors']['inverse_noise_max_abs'], .09)
        bad['step']['step']['inverse_trace']['source'] = 0
        with self.assertRaisesRegex(ValueError, 'Inverse labels'): value.audit_tree_edge(bad, density_values=cache)

    def test_source_binding_rejects_changed_constructor_and_same_hash_lie(self):
        root = Path(__file__).resolve().parents[1]
        files = {}
        for name, *_ in audit.SOURCE_EXCERPTS:
            text = (root/name).read_text()
            files[name] = dict(text=text, sha256=audit.hashlib.sha256(text.encode()).hexdigest())
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'bundle.json'
            path.write_text(json.dumps(dict(files=files)))
            self.assertEqual(len(audit.bind_source_bundle(path)['excerpts']), len(audit.SOURCE_EXCERPTS))
            files['src/docking.rs']['text'] = files['src/docking.rs']['text'].replace(
                'let base_parameters = model.component_parameters();', 'let base_parameters = changed();')
            path.write_text(json.dumps(dict(files=files)))
            with self.assertRaisesRegex(ValueError, 'Corrupt'): audit.bind_source_bundle(path)
            files['src/docking.rs']['sha256'] = audit.hashlib.sha256(files['src/docking.rs']['text'].encode()).hexdigest()
            path.write_text(json.dumps(dict(files=files)))
            with self.assertRaisesRegex(ValueError, 'Unknown constructor'): audit.bind_source_bundle(path)


if __name__ == '__main__':
    unittest.main()
