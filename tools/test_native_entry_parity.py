"""Make the cross-language audit fail on physically material label changes."""
import unittest

from check_native_entry_parity import compare


class ParityAuditTests(unittest.TestCase):
    def setUp(self):
        self.errors = dict(maximum_angle_error_deg=0., maximum_position_or_gap_error_A=0.)

    def test_missing_motif_or_anchor_is_not_silently_ignored(self):
        for expected, actual in ((dict(native_any=True), dict(native_any=False)),
                                 (dict(matched_anchor_indices=[0, 1]), dict(matched_anchor_indices=[0])),
                                 (dict(motif_id=7), dict()),
                                 (dict(motif_id=7), dict(motif_id=7, extra=True))):
            with self.subTest(expected=expected, actual=actual), self.assertRaises(ValueError):
                compare(expected, actual, self.errors)

    def test_shared_residue_pairs_require_exact_integer_agreement(self):
        with self.assertRaisesRegex(ValueError, 'Exact label'):
            compare(dict(shared_reference_residue_pairs_entry=[4, 18]),
                    dict(shared_reference_residue_pairs_entry=[4, 19]), self.errors)

    def test_small_angle_roundoff_is_reported_but_cutoff_labels_stay_exact(self):
        compare(dict(proper_orientation_error_deg=0.),
                dict(proper_orientation_error_deg=1e-6), self.errors)
        self.assertEqual(self.errors['maximum_angle_error_deg'], 1e-6)
        with self.assertRaises(ValueError):
            compare(dict(native_any=True), dict(native_any=False), self.errors)

    def test_position_and_gap_have_tighter_tolerance(self):
        compare(dict(minimum_gap_A=.02), dict(minimum_gap_A=.02+1e-11), self.errors)
        with self.assertRaises(ValueError):
            compare(dict(minimum_gap_A=.02), dict(minimum_gap_A=.02+1e-7), self.errors)

    def test_nonfinite_and_null_mismatch_are_rejected(self):
        for value in (float('nan'), float('inf'), None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                compare(dict(position_error_A=0.), dict(position_error_A=value), self.errors)


if __name__ == '__main__':
    unittest.main()
