import copy
import math
import unittest

from analyze_context_overlap_panel import score_counts, audit_rows


class ScoreCountTests(unittest.TestCase):
    def test_score_and_positive_weight_are_distinct(self):
        result = score_counts(3., 100., [7, 19], 2., .7)
        self.assertAlmostEqual(result['z_overlap'], .7*(3+26/4))
        self.assertAlmostEqual(result['variance_estimate'], .7**2*26/16)
        self.assertAlmostEqual(result['variance_upper'], .7**2*100/4)
        self.assertNotAlmostEqual(result['z_overlap'], result['log_mean_positive_weight'])

    def test_poisson_generating_function(self):
        z, lam, lower, residual = .3, 2., 1.7, 2.3
        probability = math.exp(-lam*residual)
        expected_weight = expected_count = probability_sum = 0.
        for k in range(100):
            expected_weight += probability*math.exp(z*lower+k*math.log1p(z/lam))
            expected_count += probability*k
            probability_sum += probability
            probability *= lam*residual/(k+1)
        self.assertAlmostEqual(probability_sum, 1.)
        self.assertAlmostEqual(expected_weight, math.exp(z*(lower+residual)))
        self.assertAlmostEqual(z*(lower+expected_count/lam), z*(lower+residual))
        # Exponentiating the unbiased log score is biased even with two clouds.
        exponentiated_score_mean = math.exp(z*lower+2*lam*residual*math.expm1(z/(2*lam)))
        self.assertGreater(exponentiated_score_mean, math.exp(z*(lower+residual)))

    def test_zero_hits_have_nonzero_upper_uncertainty(self):
        r = score_counts(0., 10., [0, 0], 2., .035)
        self.assertEqual(r['variance_estimate'], 0.)
        self.assertEqual(r['score_interval'][0], 0.)
        self.assertGreater(r['score_interval'][1], 0.)
        self.assertAlmostEqual(r['count_mean_interval'][1], -math.log(.025))

    def test_empty_residual_and_hard_only_limits(self):
        r = score_counts(2., 0., [0, 0], 3., .2)
        self.assertEqual(r['score_interval'], [.4, .4])
        self.assertEqual(r['variance_upper'], 0.)
        self.assertAlmostEqual(r['log_mean_positive_weight'], .4)
        r = score_counts(2., 4., [2, 5], 3., 0.)
        self.assertEqual(r['z_overlap'], 0.)
        self.assertEqual(r['log_mean_positive_weight'], 0.)

    def test_invalid_counts_and_parameters(self):
        for counts in ([0], [1, -1], [True, 0], [1., 0]):
            with self.assertRaises(ValueError): score_counts(0., 1., counts, 2., .035)
        with self.assertRaises(ValueError): score_counts(0., 0., [1, 0], 2., .035)
        with self.assertRaises(ValueError): score_counts(0., 1., [1, 0], 0., .035)

    def test_simultaneous_intervals_widen_and_cloud_order_cancels(self):
        a = score_counts(1., 100., [13, 18], 2., .035)
        b = score_counts(1., 100., [18, 13], 2., .035, .05/24)
        self.assertEqual(a['z_overlap'], b['z_overlap'])
        self.assertEqual(a['log_mean_positive_weight'], b['log_mean_positive_weight'])
        self.assertLess(b['score_interval'][0], a['score_interval'][0])
        self.assertGreater(b['score_interval'][1], a['score_interval'][1])

    def test_empty_envelope_cloud_accounting(self):
        cfg,panel,rows=self.accounting_fixture()
        records,totals=audit_rows(cfg,panel,rows)
        self.assertEqual(totals,dict(raw_points=0,processed_points=0,overlap_points=0,clouds=2))
        self.assertEqual(records[0]['score_interval'],[.4,.4])

    def test_corrupted_pose_density_or_incomplete_cloud_is_rejected(self):
        cfg,panel,rows=self.accounting_fixture()
        mutations=[lambda r:r[0].update(pose={'changed':True}),
                   lambda r:r[0].update(log_q_balanced=4.),
                   lambda r:r[0]['clouds'][0]['progress'].update(complete=False),
                   lambda r:r[0]['envelope'].update(uncertain_volume=1.)]
        for mutate in mutations:
            corrupted=copy.deepcopy(rows);mutate(corrupted)
            with self.assertRaises(ValueError):audit_rows(cfg,panel,corrupted)

    @staticmethod
    def accounting_fixture():
        cfg=dict(lambda_=3.,activity=.2);cfg['lambda']=cfg.pop('lambda_')
        pose=dict(position=[0,0,0],orientation=[1,0,0,0])
        panel=dict(entries=[dict(id='fixed',pose=pose,metadata=dict(log_q_balanced=2.))])
        envelope=dict(lower_volume=2.,upper_volume=2.,uncertain_volume=0.)
        progress=dict(begun=True,complete=True,planned_points=None,processed_points=0,overlap_points=0,log_weight=.4)
        weight=dict(**envelope,raw_points=0,overlap_points=0,log_weight=.4)
        row=dict(id='fixed',pose_index=0,complete=True,pose=pose,envelope=envelope,
                 clouds=[dict(progress=progress,weight=weight) for _ in range(2)],
                 score=score_counts(2.,0.,[0,0],3.,.2),log_q_balanced=2.,log_mean_positive_weight=.4)
        return cfg,panel,[row]


if __name__ == '__main__': unittest.main()
