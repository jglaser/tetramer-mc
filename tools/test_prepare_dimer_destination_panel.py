"""Panel selection must depend on immutable reset geometry, never outcomes."""
import copy
import unittest

from prepare_dimer_destination_panel import collect_pairs, make_cases, select_pairs, ordered_sum, relative_translation


def fixture():
    state = [dict(position=[float(i), 0., 0.], orientation=[1., 0., 0., 0.]) for i in range(12)]
    specifications = [([6, 7], [6, 7]), ([4, 5], [4, 5]), ([0, 1], [0, 1, 2]),
                      ([0, 2], [0, 1, 2]), ([8, 9], [8, 9, 10]), ([2, 3], [2, 3])]
    rows = [dict(event=1, phase=i, members=members, old_poses=[state[j] for j in members],
                 subset_context=dict(whole_component=members == parent, parent_component_members=parent))
            for i, (members, parent) in enumerate(specifications)]
    return state, rows


class PanelSelectionTests(unittest.TestCase):
    def test_original_scalar_sum_and_saved_translation_across_python_versions(self):
        self.assertEqual(ordered_sum([1e16, 1., -1e16]), 0.)
        # Archived root13 -> child215 under the original Python3.9 preparation.
        root = dict(orientation=[.2762252148774449, -.3604561619419589, -.8724885433129739, -.1803738555655635],
                    position=[-153.04313342120602, 386.9495660364342, 247.10120184637864])
        child = dict(orientation=[.8605944711459798, .1745069229416419, .2815857064523771, -.3868255162216252],
                     position=[-193.93321494408036, 372.6576148936235, 256.88343056148966])
        self.assertEqual(relative_translation(root, child), [22.44645510131526, -38.311109735257034, -.6051093192743293])

    def test_lexicographic_strata_distinct_parents_and_no_outcome_filter(self):
        state, rows = fixture()
        table, first = collect_pairs(state, [('x', rows)])
        self.assertEqual([e['members'] for e in select_pairs(table)], [[2, 3], [4, 5], [0, 1], [8, 9]])
        mutated = copy.deepcopy(rows)
        for row in mutated:
            row.update(hard_valid=False, accepted=False, physical_accepted=False,
                       proposed_gained_contacts=['unread'], native_labels=['unread'])
        self.assertEqual(collect_pairs(state, [('x', mutated)]), (table, first))
        self.assertEqual(len(table), 6)  # Unselected eligible rows remain available.

    def test_stale_state_duplicate_phase_and_inconsistent_parent_fail(self):
        state, rows = fixture()
        bad = copy.deepcopy(rows);bad[0]['old_poses'][0]['position'][0] += .1
        with self.assertRaisesRegex(ValueError, 'Non-reset'):
            collect_pairs(state, [('x', bad)])
        with self.assertRaisesRegex(ValueError, 'Duplicate first'):
            collect_pairs(state, [('x', rows + [rows[0]])])
        bad = copy.deepcopy(rows[0]);bad['phase'] = 20
        bad['subset_context'] = dict(whole_component=False, parent_component_members=[6, 7, 8])
        with self.assertRaisesRegex(ValueError, 'Inconsistent reset'):
            collect_pairs(state, [('x', rows+[bad])])

    def test_anchor_ties_external_distinct_and_fixed_support(self):
        state, rows = fixture()
        selected = select_pairs(collect_pairs(state, [('x', rows)])[0])
        cases, details = make_cases(state, selected)
        # Root 2 has equal-distance candidates 1 and 3; child 3 is ineligible.
        self.assertEqual(cases[0]['anchor'], 1)
        self.assertEqual(cases[1]['anchor'], 0)
        self.assertEqual(len(cases), 8)
        self.assertTrue(all(d['maximum_absolute_old_edge_coordinate'] <= 160 for d in details))
        with self.assertRaisesRegex(ValueError, 'outside fixed cube'):
            make_cases(state, selected, half_width=.1)


if __name__ == '__main__':
    unittest.main()
