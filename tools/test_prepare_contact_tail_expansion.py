import copy
import math
import unittest

import numpy as np
from scipy.special import logsumexp

import prepare_contact_tail_expansion as target
from prepare_contact_bank_guides import log_proposal


class TailExpansionTests(unittest.TestCase):
    def setUp(self):
        self.guide = dict(schema='defensive-latent-shell-guide-v1',
            region_sha256=target.PINS['region'], defensive_uniform_shell_probability=.5,
            gaussian_components=[dict(weight=.3, mean=[0.]*6, covariance=np.eye(6).tolist()),
                dict(weight=.7, mean=[.5]*6, covariance=(np.eye(6)*2).tolist())])
        self.addition = dict(weight=1., mean=[1.]*6,
            covariance=(np.eye(6)*.3+np.ones((6, 6))*.2).tolist())

    def test_normalized_full_density_and_old_guide_lower_bound(self):
        candidate = target.expand_guide(self.guide, [self.addition])
        result = target.validate_guide(candidate, self.guide, [[1.]*6])
        self.assertLess(result['independent_density_max_abs_log_error'], 1e-9)
        u = np.random.default_rng(12).normal(size=(100, 6))*3
        addition = copy.deepcopy(self.guide)
        addition['gaussian_components'] = [self.addition]
        expected = np.logaddexp(math.log(.8)+log_proposal(u, self.guide),
                               math.log(.2)+log_proposal(u, addition))
        np.testing.assert_allclose(log_proposal(u, candidate), expected, atol=1e-12)

    def test_full_covariance_and_untruncated_tails_are_preserved(self):
        before = copy.deepcopy(self.guide)
        candidate = target.expand_guide(self.guide, [self.addition])
        self.assertEqual(self.guide, before)
        self.assertEqual(candidate['gaussian_components'][-1]['covariance'], self.addition['covariance'])
        self.assertTrue(np.isfinite(log_proposal(np.full((1, 6), 9.), candidate)).all())

    def test_allocation_control_does_not_change_geometry(self):
        candidate, assignments = target.allocation_control(self.guide, [[1.]*6, [-1.]*6])
        for original, changed in zip(self.guide['gaussian_components'], candidate['gaussian_components']):
            self.assertEqual(original['mean'], changed['mean'])
            self.assertEqual(original['covariance'], changed['covariance'])
        self.assertEqual(len(assignments), 2)
        self.assertAlmostEqual(sum(c['weight'] for c in candidate['gaussian_components']), 1.)

    def test_bad_allocation_rejected(self):
        for fraction in (-1, 1, 2):
            with self.assertRaises(ValueError): target.expand_guide(self.guide, [self.addition], fraction)
        with self.assertRaises(ValueError): target.expand_guide(self.guide, [])

    def test_anchored_fit_uses_distinct_same_class_rows_and_source_N(self):
        rng = np.random.default_rng(500)
        pieces = []
        populations = [dict(arm='a' if p < 2 else 'b', id=f'r{p%2:02}', seed=p, samples=10000) for p in range(4)]
        for p in range(4):
            for cid, orthant in ((1, 55), (2, 63)):
                u = rng.normal(size=(160, 6))*.1+cid
                u[-1] = u[0]  # Exact duplicates must not count twice in the covariance fit.
                z = rng.normal(size=160)
                pieces.append(dict(u=u, z=z, pairs=np.column_stack([z, z]), log_q=np.zeros(160),
                    class_id=np.full(160, cid), bin_orthant=np.full(160, orthant),
                    draw=np.arange(160)+(cid-1)*160, population=np.full(160, p)))
        data = {key: np.concatenate([p[key] for p in pieces]) for key in pieces[0]}
        components, details = target.fit_components(data, populations, self.guide, np.eye(6)*.01)
        self.assertEqual(len(components), 8)
        for component, detail in zip(components, details):
            self.assertEqual(len(detail['neighbors']), 128)
            self.assertLessEqual(max(r['weight'] for r in detail['neighbors']), 1/16+1e-12)
            self.assertTrue(np.all(np.linalg.eigvalsh(component['covariance']) >= .01-1e-12))
            self.assertEqual(component['mean'], detail['anchor'])
        # The two independent cloud logs are an arithmetic mean, not mean logs.
        self.assertAlmostEqual(float(logsumexp([1., 3.])-math.log(2)), 2.433780830483027)


if __name__ == '__main__': unittest.main()
