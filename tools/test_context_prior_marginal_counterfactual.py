"""Deterministic log-density algebra; no trajectory reads or geometry."""
import copy
import math
import unittest
from analyze_context_prior_marginal_counterfactual import learned_counterfactual,logadd,reduction


def row(gx,gy,w=0.,u=.5):
    raw=gx-gy+w;accepted=(math.log(u) if u else -math.inf)<min(0.,raw)
    return dict(old_pose=dict(position=[0.,0.,0.]),proposed_pose=dict(position=[.1,0.,0.]),
        proposal=dict(full_old_gaussian_log_density=gx,full_new_gaussian_log_density=gy,log_reverse_forward=gx-gy),
        log_proposal_reverse_forward=gx-gy,wall_valid=True,core_valid=True,accepted=accepted,
        gate=dict(log_weight=w),raw_log_acceptance=raw,log_acceptance=min(0.,raw),
        acceptance_uniform=u,log_uniform=math.log(u) if u else None)


class Counterfactual(unittest.TestCase):
    def test_marginal_rule_can_gain_or_lose_on_one_component(self):
        gain=learned_counterfactual(row(-1000.,-2.),.5)
        loss=learned_counterfactual(row(0.,-1000.,-3.),.5)
        self.assertFalse(gain['original_accepted']);self.assertTrue(gain['counterfactual_accepted'])
        self.assertTrue(loss['original_accepted']);self.assertFalse(loss['counterfactual_accepted'])
        self.assertAlmostEqual(gain['marginal_logq'],-math.log1p(math.exp(-2.)))
        self.assertAlmostEqual(loss['marginal_logq'],math.log(2.))

    def test_equal_density_and_zero_uniform_conventions(self):
        for g in (-1000.,0.,1000.):
            value=learned_counterfactual(row(g,g),.5)
            self.assertEqual(value['marginal_logq'],0.)
        self.assertTrue(learned_counterfactual(row(-1000.,-2.,u=0.),.5)['counterfactual_accepted'])

    def test_outside_cube_is_not_assigned_uniform_density(self):
        hard=row(-2.,-3.);hard.update(proposed_pose=dict(position=[1.,0.,0.]),accepted=False,gate=None,
            wall_valid=False,core_valid=None,raw_log_acceptance=None,log_acceptance=None,acceptance_uniform=None,log_uniform=None)
        value=learned_counterfactual(hard,.5)
        self.assertFalse(value['candidate_inside_uniform_cube']);self.assertIsNone(value['new_G_minus_logU'])
        self.assertAlmostEqual(value['marginal_logq'],logadd(0.,-2.)+3.)
        self.assertFalse(value['counterfactual_accepted'])
        invalid=copy.deepcopy(hard);invalid['old_pose']['position']=[1.,0.,0.]
        with self.assertRaises(ValueError):learned_counterfactual(invalid,.5)

    def test_unknown_uniform_and_acceptance_losses_are_retained(self):
        values=[dict(branch='involution',**learned_counterfactual(row(-1000.,-2.),.5)),
                dict(branch='involution',**learned_counterfactual(row(0.,-1000.,-3.),.5)),
                dict(branch='uniform',status='bath_rejected',original_accepted=False,counterfactual_accepted=None)]
        result=reduction(values)
        self.assertEqual(result['global_attempts'],3);self.assertEqual(result['missing_uniform_counterfactual'],1)
        self.assertEqual(result['acceptance_gains'],1);self.assertEqual(result['acceptance_losses'],1)
        self.assertEqual(result['original_gated_accepts'],result['counterfactual_gated_accepts'])


if __name__=='__main__':unittest.main()
