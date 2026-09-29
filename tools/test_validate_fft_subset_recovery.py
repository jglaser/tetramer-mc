import unittest
import numpy as np
from validate_fft_subset_recovery import matching_pose, summarize, uniform_slots


class RecoveryTest(unittest.TestCase):
    def test_fixed_uniform_selection(self):
        self.assertEqual(uniform_slots(768, 32), list(range(0, 768, 24)))
        self.assertEqual(uniform_slots(5, 3), [0, 1, 3])
        with self.assertRaises(ValueError):
            uniform_slots(2, 3)

    def test_quaternion_sign_match(self):
        a = dict(position=[1, 2, 3], orientation=[1, 0, 0, 0])
        b = dict(position=[1, 2, 3], orientation=[-1, 0, 0, 0])
        self.assertTrue(matching_pose(a, b))
        b['position'][0] += .01
        self.assertFalse(matching_pose(a, b))

    def test_failures_preserved_in_denominator_and_cost(self):
        validation = [dict(volume=100., standard_error=4.), dict(volume=100., standard_error=3.)]
        plan = dict(all_slot_denominator=2, slots=[dict(slot=i, archived_validation=validation) for i in (0, 24)])
        rows = [dict(slot=0, execution_error=None, repair_succeeded=True,
                     score=dict(hard_valid=True, score=dict(volume=80.)), worker_cpu_seconds=2., scorer_child_cpu_seconds=1.),
                dict(slot=24, execution_error=None, repair_succeeded=False, score=None,
                     worker_cpu_seconds=3., scorer_child_cpu_seconds=0.)]
        r = summarize(rows, plan)
        self.assertEqual(r['denominator'], 2)
        self.assertEqual(r['repaired_hard_valid'], 1)
        self.assertEqual(r['mean_overlap_ratio_unconditional'], .4)
        self.assertEqual(r['total_cpu_seconds'], 6.)
        self.assertEqual(r['slots'][0]['archived_overlap_se_A3'], 2.5)
        self.assertTrue(r['complete'])
        rows[-1]['execution_error'] = 'test error'
        self.assertFalse(summarize(rows, plan)['complete'])


if __name__ == '__main__':
    unittest.main()
