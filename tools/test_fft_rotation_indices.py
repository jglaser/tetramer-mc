import hashlib
from pathlib import Path
import tempfile
import unittest
import numpy as np
from fft_depletion_docking import super_fibonacci
from fft_rotation_indices import (freeze_rotation_indices, load_rotation_indices,
                                  shard_rotation_indices, validate_rotation_indices)


class RotationIndexTest(unittest.TestCase):
    def test_validation_and_ascending_order(self):
        np.testing.assert_array_equal(validate_rotation_indices(np.array([9, 0, 7], np.uint32), 10), [0, 7, 9])
        for bad in ([1, 1], [-1, 2], [1, 10], [], [1., 2.], [[1, 2]], [True, False]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                validate_rotation_indices(np.asarray(bad), 10)
        with self.assertRaises(ValueError):
            validate_rotation_indices(np.array([2**64-1], np.uint64), 10)

    def test_complete_disjoint_sorted_sharding(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'ids.npy'
            np.save(p, np.array([99, 3, 71, 0, 12, 63, 2]))
            for parts in (1, 2, 4, 11):
                shards = [shard_rotation_indices(100, k, parts, p) for k in range(parts)]
                combined = np.concatenate(shards)
                np.testing.assert_array_equal(np.sort(combined), [0, 2, 3, 12, 63, 71, 99])
                self.assertEqual(len(combined), len(set(combined)))
                for shard in shards:
                    self.assertTrue(np.all(shard[1:] > shard[:-1]))

    def test_no_option_exact_original_arithmetic(self):
        for n in (1, 5, 101):
            for parts in (1, 2, 7):
                for rank in range(parts):
                    np.testing.assert_array_equal(shard_rotation_indices(n, rank, parts), np.arange(rank, n, parts))

    def test_fixed_grid_quaternions_match_full_grid_exactly(self):
        n = 300000
        selected = np.array([0, 7, 19024, 75000, 178001, n-1])
        np.testing.assert_array_equal(super_fibonacci(n, selected), super_fibonacci(n)[selected])
        self.assertFalse(np.array_equal(super_fibonacci(len(selected)), super_fibonacci(n, selected)))

    def test_exact_file_freeze_and_hash(self):
        with tempfile.TemporaryDirectory() as d:
            src, dst = Path(d)/'input.npy', Path(d)/'frozen.npy'
            np.save(src, np.array([5, 0, 3], dtype=np.int32))
            raw = src.read_bytes()
            metadata = freeze_rotation_indices(src, dst, 9)
            self.assertEqual(dst.read_bytes(), raw)
            self.assertEqual(metadata['rotation_indices_sha256'], hashlib.sha256(raw).hexdigest())
            self.assertEqual(metadata['selected_rotations'], 3)
            self.assertEqual(metadata['rotation_indices'], str(dst.resolve()))
            np.testing.assert_array_equal(load_rotation_indices(dst, 9), [0, 3, 5])
            with self.assertRaises(FileExistsError):
                freeze_rotation_indices(src, dst, 9)

    def test_invalid_shard_rejected(self):
        for n, rank, parts in ((0, 0, 1), (10, -1, 2), (10, 2, 2), (10, 0, 0)):
            with self.assertRaises(ValueError):
                shard_rotation_indices(n, rank, parts)


if __name__ == '__main__':
    unittest.main()
