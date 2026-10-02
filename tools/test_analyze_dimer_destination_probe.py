import math
import unittest
import numpy as np

from analyze_dimer_destination_probe import Checks, fingerprints, full_density, log_value, log_difference, summarize, evaluate_tasks


class PassiveAnalyzerTests(unittest.TestCase):
    def test_full_mixture_and_closed_cube_support(self):
        pose=dict(position=[2.,0.,0.],orientation=[1.,0.,0.,0.])
        density=full_density(math.log(.1),pose,.5,2.)
        self.assertAlmostEqual(math.exp(density['log_full']),.5/64+.05)
        pose['position'][0]=np.nextafter(2.,math.inf)
        self.assertEqual(full_density(math.log(.1),pose,.5,2.)['log_uniform'],-math.inf)
        self.assertEqual(log_value('-inf'),-math.inf)
        self.assertEqual(log_difference(-math.inf,-math.inf),'undefined_both_zero_density')
        for x in (math.inf,math.nan):
            with self.assertRaises(ValueError):log_value(x)

    def test_failures_accumulate_with_fixed_tolerance(self):
        checks=Checks();checks.close(2.,2.,'same');checks.close(2.001,2.,'changed',1)
        checks.exact(4,5,'lost attempt');checks.close(-math.inf,-math.inf,'structural zero')
        self.assertEqual(checks.count,4);self.assertEqual(len(checks.failures),2)
        self.assertEqual(checks.failures[0]['row'],1)

    def test_all_attempt_denominator_and_empty_fingerprint(self):
        rows=[dict(status='proposal_error'),dict(status='proposal_null'),dict(status='finite_endpoint',hard_valid=True,log_reverse_forward=-100.)]
        result=summarize(rows)
        self.assertEqual(result['attempts'],3)
        self.assertEqual(result['hard_valid_all_attempt_fraction'],1/3)
        self.assertEqual(fingerprints([],[])['fingerprint_jaccard'],1.)
        self.assertEqual(fingerprints([[0,1]],[[1,2]])['lost_contacts'],1)

    def test_batched_scores_preserve_branch_and_generation_difference(self):
        class Density:
            def __init__(self,offset):self.offset=offset
            def evaluate(self,poses):
                a=np.asarray([p['x'] for p in poses])+self.offset
                return a,np.zeros((len(a),2)),np.column_stack([a,a+1])
        class Helper:
            score=Density(0.);map_density=Density(.25)
        values=evaluate_tasks(Helper(),[dict(pose=dict(x=i),branch=i%2) for i in range(5)],batch=2)
        self.assertEqual(len(values),5)
        self.assertEqual(values[3],dict(log_g=3.,map_log_g=3.25,branch_log=4.))


if __name__=='__main__':unittest.main()
