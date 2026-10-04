"""Integer-affine toys only; no catalogue, classifier, trajectory, or geometry.

The small-graph oracle enumerates complete edge-label products first, then
propagates and checks every occupied site. It does not share the solver's
incremental search or collision-pruning implementation.
"""
from collections import deque
from itertools import product
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
import exact_native_embedding as e


IR = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
RZ = ((0, -1, 0), (1, 0, 0), (0, 0, 1))
ONE = (IR, (0, 0, 0))


def op(t=(0, 0, 0), r=IR):
    return e.Affine(r, tuple(t))


def shift(x):
    return op((24*x, 0, 0))


def raw(g):
    return tuple(tuple(row) for row in g.rotation), tuple(g.translation_numerator)


def oracle_product(a, b):
    ar, at = a; br, bt = b
    rotation = tuple(tuple(sum(ar[i][k]*br[k][j] for k in range(3))
                           for j in range(3)) for i in range(3))
    translation = tuple(at[i]+sum(ar[i][k]*bt[k] for k in range(3)) for i in range(3))
    return rotation, translation


def oracle_inverse(a):
    r, t = a
    rt = tuple(tuple(r[j][i] for j in range(3)) for i in range(3))
    return rt, tuple(-sum(rt[i][j]*t[j] for j in range(3)) for i in range(3))


def exhaustive_embedding(bodies, keys, motifs, members):
    """Independent finite oracle for one connected graph, including a singleton."""
    labels = {}
    for i, j, label in sorted(set(keys)):
        labels.setdefault((i, j), []).append(label)
    edges = sorted(labels)
    for picked in product(*(labels[edge] for edge in edges)):
        adjacent = {i: [] for i in bodies}
        for (i, j), label in zip(edges, picked):
            g = raw(motifs[label])
            adjacent[i].append((j, g)); adjacent[j].append((i, oracle_inverse(g)))
        assigned = {min(bodies): ONE}; queue = deque(assigned); conflict = False
        while queue:
            i = queue.popleft()
            for j, g in adjacent[i]:
                wanted = oracle_product(assigned[i], g)
                if j in assigned:
                    if assigned[j] != wanted: conflict = True
                else:
                    assigned[j] = wanted; queue.append(j)
        if conflict or set(assigned) != set(bodies): continue
        sites = [oracle_product(assigned[i], raw(member)) for i in bodies for member in members]
        if len(set(sites)) == len(sites): return True
    return False


def checker(motifs=None, members=None):
    return e.ExactNativeEmbedding({0: shift(2)} if motifs is None else motifs,
                                  [shift(0), shift(1)] if members is None else members,
                                  denominator=24)


class ExactNativeEmbeddingTests(unittest.TestCase):
    def single(self, result):
        self.assertEqual(len(result['components']), 1)
        return result['components'][0]

    def assert_witness(self, component, keys, motifs, members):
        """Verify a successful certificate without calling solver arithmetic."""
        self.assertIs(component['embeddable'], True)
        self.assertEqual(component['status'], 'embedding_found')
        witness = component['witness']
        assigned = {r['body']: (tuple(map(tuple, r['operator']['rotation'])),
                               tuple(r['operator']['translation_numerator']))
                    for r in witness['body_operators']}
        self.assertEqual(set(assigned), set(component['bodies']))
        self.assertEqual(assigned[min(assigned)], ONE)
        available = set(map(tuple, keys)); chosen = list(map(tuple, witness['chosen_labels']))
        self.assertEqual(len(chosen), len({(i, j) for i, j, _ in available}))
        self.assertEqual({(i, j) for i, j, _ in chosen}, {(i, j) for i, j, _ in available})
        for i, j, label in chosen:
            self.assertIn((i, j, label), available)
            self.assertEqual(oracle_product(assigned[i], raw(motifs[label])), assigned[j])
        expected = {(body, member): oracle_product(g, raw(h))
                    for body, g in assigned.items() for member, h in enumerate(members)}
        actual = {(r['body'], r['member_index']):
                  (tuple(map(tuple, r['operator']['rotation'])), tuple(r['operator']['translation_numerator']))
                  for r in witness['occupied_sites']}
        self.assertEqual(len(witness['occupied_sites']), len(expected))
        self.assertEqual(actual, expected)
        self.assertEqual(len(set(actual.values())), len(expected))
        for row in witness['body_operators']+witness['occupied_sites']:
            self.assertEqual(row['operator']['denominator'], 24)

    def test_affine_inverse_screw_conjugation_and_serialized_roundtrip(self):
        screw = op((36, 12, -6), RZ)
        self.assertEqual(raw(e.compose(screw, e.inverse(screw))), ONE)
        self.assertEqual(raw(e.compose(e.inverse(screw), screw)), ONE)
        square = e.compose(screw, screw)
        self.assertEqual(raw(e.compose(square, square)), (IR, (0, 0, -24)))
        conjugated = e.compose(e.compose(screw, shift(1)), e.inverse(screw))
        self.assertEqual(raw(conjugated), (IR, (0, 24, 0)))
        record = screw.to_record(denominator=24)
        self.assertEqual(e.Affine.from_record(record, denominator=24), screw)
        self.assertEqual(raw(e.IDENTITY), ONE)

    def test_empty_graph_and_singletons_require_no_search_or_shared_gauge(self):
        result = checker().check(3, [], max_search_nodes=0)
        self.assertIs(result['componentwise_embeddable'], True)
        self.assertEqual(result['search_nodes'], 0)
        self.assertFalse(result['budget_exhausted'])
        self.assertEqual([c['bodies'] for c in result['components']], [[0], [1], [2]])
        for c in result['components']:
            self.assert_witness(c, [], {0: shift(2)}, [shift(0), shift(1)])

    def test_valid_reverse_traversal_tree_and_noncommuting_cycle(self):
        motifs = {0: shift(2), 1: shift(4)}; members = [shift(0), shift(1)]
        keys = [(0, 2, 0), (1, 2, 1)]
        component = self.single(checker(motifs, members).check(3, keys, max_search_nodes=100))
        self.assert_witness(component, keys, motifs, members)
        assigned = {r['body']: r['operator']['translation_numerator'] for r in component['witness']['body_operators']}
        self.assertEqual(list(assigned[1]), [-48, 0, 0])
        screw = op((36, 12, -6), RZ)
        motifs = {0: screw, 1: op((0, 0, -24))}; members = [op()]
        keys = [(i, i+1, 0) for i in range(4)]+[(0, 4, 1)]
        component = self.single(checker(motifs, members).check(5, keys, max_search_nodes=100))
        self.assert_witness(component, keys, motifs, members)

    def test_inconsistent_cycle_exhausts_without_dropping_an_edge(self):
        keys = [(0, 1, 0), (1, 2, 0), (0, 2, 0)]
        result = checker().check(3, keys, max_search_nodes=100)
        component = self.single(result)
        self.assertIs(component['embeddable'], False)
        self.assertEqual(component['status'], 'no_admissible_embedding')
        self.assertGreater(component['diagnostics']['cycle_constraint_rejections'], 0)
        self.assertIsNone(component.get('witness'))
        self.assertFalse(result['budget_exhausted'])

    def test_tree_can_have_nonadjacent_partial_monomer_collision(self):
        # Each neighboring pair is injective. The chain ends at +1, sharing a
        # monomer with its root despite having no graph cycle at all.
        members = [shift(0), shift(1), shift(10), shift(11)]
        motifs = {0: shift(4), 1: shift(-3)}
        for label in motifs:
            self.assertTrue(exhaustive_embedding([0, 1], [(0, 1, label)], motifs, members))
        keys = [(0, 1, 0), (1, 2, 1)]
        self.assertFalse(exhaustive_embedding([0, 1, 2], keys, motifs, members))
        component = self.single(checker(motifs, members).check(3, keys, max_search_nodes=100))
        self.assertIs(component['embeddable'], False)
        self.assertEqual(component['status'], 'no_admissible_embedding')
        self.assertGreater(component['diagnostics']['site_collision_rejections'], 0)
        self.assertEqual(component['diagnostics']['cycle_constraint_rejections'], 0)

    def test_first_colliding_label_backtracks_and_rolls_back_partial_sites(self):
        # -1 first inserts a free site, then collides at0. Retrying -2 needs
        # the formerly free -1 site: leaking a partial insertion rejects it.
        motifs = {0: shift(-1), 1: shift(-2)}; members = [shift(0), shift(1)]
        keys = [(0, 1, 0), (0, 1, 1)]
        component = self.single(checker(motifs, members).check(2, keys, max_search_nodes=100))
        self.assert_witness(component, keys, motifs, members)
        self.assertEqual(component['witness']['chosen_labels'], [[0, 1, 1]])
        self.assertGreater(component['diagnostics']['site_collision_rejections'], 0)

    def test_descendant_collision_backtracks_an_earlier_accepted_placement(self):
        motifs = {0: shift(2), 1: shift(4), 2: shift(-2)}
        members = [shift(0), shift(1)]
        keys = [(0, 1, 0), (0, 1, 1), (1, 2, 2)]
        component = self.single(checker(motifs, members).check(3, keys, max_search_nodes=100))
        self.assert_witness(component, keys, motifs, members)
        self.assertIn([0, 1, 1], component['witness']['chosen_labels'])
        self.assertGreater(component['diagnostics']['site_collision_rejections'], 0)

    def test_all_alternative_labels_can_be_invalid(self):
        motifs = {0: shift(0), 1: shift(1), 2: shift(-1)}
        keys = [(0, 1, label) for label in motifs]
        result = checker(motifs).check(2, keys, max_search_nodes=100)
        component = self.single(result)
        self.assertIs(component['embeddable'], False)
        self.assertEqual(component['status'], 'no_admissible_embedding')
        self.assertIsNone(component.get('witness'))
        self.assertFalse(result['budget_exhausted'])

    def test_search_cap_is_unresolved_with_deterministic_prefix(self):
        motifs = {0: shift(1), 1: shift(2)}; keys = [(0, 1, 0), (0, 1, 1)]
        solver = checker(motifs)
        zero = solver.check(2, keys, max_search_nodes=0)
        self.assertIsNone(self.single(zero)['embeddable'])
        self.assertEqual(zero['search_nodes'], 0)
        limited = solver.check(2, keys, max_search_nodes=1)
        repeated = solver.check(2, list(reversed(keys))+keys, max_search_nodes=1)
        self.assertEqual(limited['components'], repeated['components'])
        self.assertEqual(limited['unique_native_keys'], repeated['unique_native_keys'])
        self.assertEqual(limited['search_nodes'], repeated['search_nodes'])
        component = self.single(limited)
        self.assertIsNone(component['embeddable'])
        self.assertEqual(component['status'], 'search_budget_exhausted')
        self.assertEqual(limited['search_nodes'], 1)
        self.assertTrue(limited['budget_exhausted']); self.assertIsNone(component.get('witness'))
        complete = solver.check(2, keys, max_search_nodes=2)
        self.assert_witness(self.single(complete), keys, motifs, [shift(0), shift(1)])
        self.assertEqual(complete['search_nodes'], 2)
        self.assertEqual(component['diagnostics'], self.single(complete)['diagnostics'])
        self.assertEqual(component['search_node_start'], 0)
        self.assertEqual(component['search_node_end'], 1)

    def test_disconnected_components_have_independent_site_gauges(self):
        keys = [(0, 1, 0), (2, 3, 0)]
        result = checker().check(5, keys, max_search_nodes=100)
        self.assertIs(result['componentwise_embeddable'], True)
        self.assertEqual([c['bodies'] for c in result['components']], [[0, 1], [2, 3], [4]])
        for component in result['components']:
            local = [k for k in keys if k[0] in component['bodies']]
            self.assert_witness(component, local, {0: shift(2)}, [shift(0), shift(1)])
        self.assertEqual(result['search_nodes'], sum(c['search_nodes'] for c in result['components']))
        limited = checker().check(5, keys, max_search_nodes=1)
        self.assertEqual([c['embeddable'] for c in limited['components']], [True, None, True])
        self.assertEqual(limited['search_nodes'], 1)
        self.assertEqual(limited['components'][1]['search_node_start'], 1)
        self.assertEqual(limited['components'][1]['search_node_end'], 1)

    def test_nonlifted_periodic_component_stays_unresolved_without_search(self):
        keys = [(0, 1, 0), (2, 3, 0)]
        result = checker().check(4, keys, max_search_nodes=100, non_lifted_components=[(0, 1)])
        first, second = result['components']
        self.assertIsNone(first['embeddable'])
        self.assertEqual(first['status'], 'non_lifted_periodic_component')
        self.assertEqual(first['search_nodes'], 0); self.assertIsNone(first.get('witness'))
        self.assertIs(second['embeddable'], True)
        self.assertFalse(result['budget_exhausted'])

    def test_strict_operator_graph_and_budget_inputs(self):
        invalid_ops = [lambda: op((0., 0, 0)), lambda: op((False, 0, 0)),
            lambda: op((0, 0)), lambda: op(r=((1, 0, 0), (0, 1, 0), (0, 0, -1))),
            lambda: op(r=((1, 1, 0), (0, 1, 0), (0, 0, 1))),
            lambda: op(r=((True, 0, 0), (0, 1, 0), (0, 0, 1)))]
        for make in invalid_ops:
            with self.subTest(operator=make), self.assertRaises(ValueError): make()
        for den in (0, -1, True, 24.):
            with self.subTest(denominator=den), self.assertRaises(ValueError):
                e.ExactNativeEmbedding({0: shift(2)}, [op()], denominator=den)
        for motifs, members in [({True: shift(2)}, [op()]), ({0: shift(2)}, []),
                                ({0: shift(2)}, [op(), op()])]:
            with self.subTest(constructor=(motifs, members)), self.assertRaises(ValueError):
                e.ExactNativeEmbedding(motifs, members, denominator=24)
        record = op().to_record(denominator=24)
        for changed in (dict(record, denominator=12), dict(record, denominator=True),
                        dict(record, unexpected=0), {k: v for k, v in record.items() if k != 'rotation'}):
            with self.subTest(record=changed), self.assertRaises(ValueError):
                e.Affine.from_record(changed, denominator=24)
        solver = checker()
        for n in (0, -1, True, 3.):
            with self.subTest(body_count=n), self.assertRaises(ValueError): solver.check(n, [], max_search_nodes=10)
        for keys in [[(0, 0, 0)], [(1, 0, 0)], [(0, 3, 0)], [(-1, 1, 0)],
                     [(False, 1, 0)], [(0, 1, True)], [(0, 1, 999)], [(0, 1)], [(0, 1, 0, 2)]]:
            with self.subTest(keys=keys), self.assertRaises(ValueError): solver.check(3, keys, max_search_nodes=10)
        for cap in (-1, True, 1.5):
            with self.subTest(cap=cap), self.assertRaises(ValueError): solver.check(3, [], max_search_nodes=cap)
        for winding in ([(0, 2)], [(0, 1), (0, 1)], [(0, True)]):
            with self.subTest(non_lifted=winding), self.assertRaises(ValueError):
                solver.check(3, [(0, 1, 0)], max_search_nodes=10, non_lifted_components=winding)

    def test_duplicate_key_order_body_and_motif_label_permutations(self):
        motifs = {0: shift(2), 1: shift(-2), 2: shift(4), 3: shift(-4)}
        members = [shift(0), shift(1)]; keys = [(0, 1, 0), (1, 2, 0), (0, 2, 2)]
        solver = checker(motifs, members)
        original = solver.check(3, keys, max_search_nodes=100)
        repeated = solver.check(3, list(reversed(keys))*2, max_search_nodes=100)
        self.assertEqual(original['components'], repeated['components'])
        self.assertEqual(original['unique_native_keys'], repeated['unique_native_keys'])
        self.assertEqual(original['search_nodes'], repeated['search_nodes'])
        self.assertEqual((original['duplicate_key_count'], repeated['duplicate_key_count']), (0, 3))
        self.assert_witness(self.single(original), keys, motifs, members)
        permutation = [2, 0, 1]; relabel = {0: 19, 1: 7, 2: 11, 3: 3}
        renamed = {relabel[label]: operator for label, operator in motifs.items()}
        changed = []
        for i, j, label in keys:
            a, b = permutation[i], permutation[j]
            changed.append((min(a, b), max(a, b), relabel[label if a < b else label ^ 1]))
        result = checker(renamed, members).check(3, changed, max_search_nodes=100)
        self.assert_witness(self.single(result), changed, renamed, members)
        self.assertEqual(original['componentwise_embeddable'], result['componentwise_embeddable'])

    def test_independent_complete_edge_product_oracle_for_small_graphs(self):
        motifs = {0: shift(1), 1: shift(2), 2: shift(-2), 3: shift(4),
                  4: op((24, 0, 0), RZ), 5: op((0, 24, 0), ((0, 1, 0), (-1, 0, 0), (0, 0, 1)))}
        members = [shift(0), shift(1)]
        alternatives = [(0,), (1,), (2,), (1, 3), (4, 5), (0, 1, 3)]
        solver = checker(motifs, members); successes = failures = 0
        # Complete products of label *sets* on a tree and a triangle, each
        # independently expanded into all assignments by the oracle above.
        for edges in ([(0, 1), (1, 2)], [(0, 1), (1, 2), (0, 2)]):
            for selected in product(alternatives, repeat=len(edges)):
                keys = [(i, j, label) for (i, j), labels in zip(edges, selected) for label in labels]
                expected = exhaustive_embedding([0, 1, 2], keys, motifs, members)
                result = solver.check(3, keys, max_search_nodes=10000); component = self.single(result)
                with self.subTest(keys=keys):
                    self.assertIs(component['embeddable'], expected)
                    self.assertFalse(result['budget_exhausted'])
                    if expected: self.assert_witness(component, keys, motifs, members)
                successes += expected; failures += not expected
        self.assertGreater(successes, 0); self.assertGreater(failures, 0)


if __name__ == '__main__':
    unittest.main()
