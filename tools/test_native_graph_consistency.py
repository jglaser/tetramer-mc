import unittest
from native_graph_consistency import NativeGraphConsistency


def motif(label, translation, quaternion=None):
    return dict(id=label, relative_position=translation, relative_orientation=quaternion or [1., 0., 0., 0.])


class Consistency(unittest.TestCase):
    def test_tree_and_consistent_triangle_tail_do_not_require_clique(self):
        checker = NativeGraphConsistency([motif(0, [1, 0, 0]), motif(1, [2, 0, 0]), motif(2, [0, 1, 0])], 4)
        tree = checker.check([(0, 1, 0), (1, 2, 0), (2, 3, 2)])
        triangle = checker.check([(0, 1, 0), (1, 2, 0), (0, 2, 1), (2, 3, 2)])
        self.assertTrue(tree['consistent_connected']); self.assertEqual(tree['independent_cycles'], 0)
        self.assertTrue(triangle['consistent_connected']); self.assertEqual(triangle['independent_cycles'], 1)

    def test_connected_pair_native_graph_can_be_frustrated(self):
        checker = NativeGraphConsistency([motif(0, [1, 0, 0])], 3)
        value = checker.check([(0, 1, 0), (1, 2, 0), (0, 2, 0)])
        self.assertTrue(value['connected']); self.assertFalse(value['consistent_connected'])

    def test_rotation_composition_and_reverse_traversal(self):
        # A 180-degree rotation makes the next +x translation point along -x.
        checker = NativeGraphConsistency([motif(0, [1, 0, 0], [0, 0, 0, 1]),
            motif(1, [1, 0, 0]), motif(2, [0, 0, 0], [0, 0, 0, 1])], 3)
        self.assertTrue(checker.check([(0, 1, 0), (1, 2, 1), (0, 2, 2)])['consistent_connected'])
        reverse = checker.check([(0, 2, 2), (1, 2, 1)])
        self.assertTrue(reverse['consistent_connected'])

    def test_alternative_labels_backtrack_instead_of_false_rejection(self):
        checker = NativeGraphConsistency([motif(0, [1, 0, 0]), motif(1, [2, 0, 0]), motif(2, [3, 0, 0])], 3)
        keys = [(0, 1, 0), (0, 1, 1), (1, 2, 0), (0, 2, 2)]
        value = checker.check(keys)
        self.assertTrue(value['consistent_connected']); self.assertEqual(value['chosen_edge_labels']['0-1']['motif_id'], 1)
        self.assertIs(value, checker.check(list(reversed(keys))+keys))

    def test_disconnected_components_and_isolates_are_not_growth(self):
        checker = NativeGraphConsistency([motif(0, [1, 0, 0])], 4)
        for keys in ([], [(0, 1, 0)], [(0, 1, 0), (2, 3, 0)]):
            value = checker.check(keys)
            self.assertTrue(value['consistent']); self.assertFalse(value['connected'])


if __name__ == '__main__': unittest.main()
