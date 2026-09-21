"""Frozen guide-peak identities and disjoint priority-ball assignment."""
import copy
import itertools
import math
import unittest

from prepare_shoulder_guide_peak_references import selected_peaks, union_assignment


SOURCES = ('mixture-confirmation', 'geometry-confirmation')


def selection_fixture():
    """Independent records preserve every field carried by the original rows."""
    identities = [('r03', 30939, 1.038236897099948),
                  ('r01', 33280, 1.0232210223101685)]
    rows = []
    for index, (population, draw, q) in enumerate(identities):
        rows.append(dict(population=population, draw=draw, q=q,
            pose=dict(position=[.1+index, .2, -.3], orientation=[1., 0., 0., 0.]),
            log_importance_weight=35.+index, log_hard_weight=-10.-index,
            log_proposal_density=10.+index, log_boltzmann_mean=45.+2*index,
            cloud_log_weights=[45.+2*index, 45.+2*index],
            cloud_overlap_counts=[2900+index, 2900+index],
            cloud_raw_points=[30000+index, 30000+index],
            lower_volume=0., upper_volume=14000., proposal_family='guide',
            proposal_component=None, guide_branch='learned', guide_component=index))
    diagnostic = dict(complete=True, comparison_peaks=[dict(source_campaign=source, original_row=copy.deepcopy(row))
                                       for source, row in zip(SOURCES, rows)])
    guided = dict(campaigns={source: dict(top_16=[copy.deepcopy(row), dict(copy.deepcopy(row), draw=draw+1)])
                            for source, row, (_, draw, _) in zip(SOURCES, rows, identities)})
    return diagnostic, guided


class ShoulderGuidePeakReferenceTests(unittest.TestCase):
    def test_selection_keeps_exact_source_rows_in_fixed_source_order(self):
        diagnostic, guided = selection_fixture()
        diagnostic['comparison_peaks'].reverse()
        before = copy.deepcopy((diagnostic, guided))
        selected = selected_peaks(diagnostic, guided)
        self.assertEqual([record['source_campaign'] for record in selected], list(SOURCES))
        for record in selected:
            expected = guided['campaigns'][record['source_campaign']]['top_16'][0]
            self.assertEqual(record['original_row'], expected)
            self.assertEqual(record['original_row']['pose'], expected['pose'])
            self.assertEqual(record['original_row']['log_importance_weight'], expected['log_importance_weight'])
            self.assertEqual(record['original_row']['cloud_log_weights'], expected['cloud_log_weights'])
        self.assertEqual((diagnostic, guided), before)
        self.assertEqual(selected_peaks(diagnostic, guided), selected)

    def test_selection_rejects_replacing_a_source_maximum(self):
        for source in SOURCES:
            with self.subTest(source=source):
                diagnostic, guided = selection_fixture()
                ranking = guided['campaigns'][source]['top_16']
                ranking[0], ranking[1] = ranking[1], ranking[0]
                before = copy.deepcopy((diagnostic, guided))
                with self.assertRaises(ValueError):
                    selected_peaks(diagnostic, guided)
                self.assertEqual((diagnostic, guided), before)

    def test_selection_rejects_modified_weight_or_other_original_fields(self):
        for source in SOURCES:
            for field, replacement in [('log_importance_weight', 99.),
                                       ('log_hard_weight', -99.),
                                       ('cloud_log_weights', [99., 99.]),
                                       ('pose', dict(position=[9., 9., 9.], orientation=[1., 0., 0., 0.]))]:
                with self.subTest(source=source, field=field):
                    diagnostic, guided = selection_fixture()
                    record = next(record for record in diagnostic['comparison_peaks']
                                  if record['source_campaign'] == source)
                    record['original_row'][field] = replacement
                    before = copy.deepcopy((diagnostic, guided))
                    with self.assertRaises(ValueError):
                        selected_peaks(diagnostic, guided)
                    self.assertEqual((diagnostic, guided), before)

    def test_frozen_identities_are_checked_even_when_both_copies_agree(self):
        for source in SOURCES:
            for field, replacement in [('population', 'r99'), ('draw', 12345), ('q', 1.09)]:
                with self.subTest(source=source, field=field):
                    diagnostic, guided = selection_fixture()
                    row = next(record['original_row'] for record in diagnostic['comparison_peaks']
                               if record['source_campaign'] == source)
                    row[field] = replacement
                    guided['campaigns'][source]['top_16'][0][field] = replacement
                    with self.assertRaises(ValueError):
                        selected_peaks(diagnostic, guided)

    def test_union_closed_edges_overlap_priority_and_outside(self):
        just_outside = math.nextafter(.5, math.inf)
        cases = [((.5, .5, .5), 0),
                 ((.1, .01, .001), 0),
                 ((just_outside, .5, .1), 1),
                 ((math.inf, .4, .2), 1),
                 ((just_outside, just_outside, .5), 2),
                 ((math.inf, math.inf, 0.), 2),
                 ((just_outside, just_outside, just_outside), None),
                 ((math.inf, math.inf, math.inf), None)]
        for radii, expected in cases:
            with self.subTest(radii=radii):
                self.assertEqual(union_assignment(radii), expected)
                self.assertEqual(union_assignment(radii), expected)

    def test_priority_pieces_are_disjoint_and_cover_exact_ball_union(self):
        grid = (0., .49, .5, math.nextafter(.5, math.inf), .75, math.inf)
        for direct, mixture, geometry in itertools.product(grid, repeat=3):
            radii = (direct, mixture, geometry)
            # Explicit disjoint masks provide an independent partition oracle.
            pieces = (direct <= .5,
                      direct > .5 and mixture <= .5,
                      direct > .5 and mixture > .5 and geometry <= .5)
            self.assertEqual(sum(pieces), int(any(radius <= .5 for radius in radii)))
            assignment = union_assignment(radii)
            if not any(pieces):
                self.assertIsNone(assignment)
            else:
                self.assertEqual(sum(index == assignment for index in range(3)), 1)
                self.assertTrue(pieces[assignment])

    def test_union_rejects_bad_radius_even_after_an_earlier_ball_hit(self):
        for invalid in (-.1, -math.inf, math.nan):
            for index in range(3):
                with self.subTest(invalid=invalid, index=index):
                    radii = [0., 0., 0.]
                    radii[index] = invalid
                    with self.assertRaises(ValueError):
                        union_assignment(tuple(radii))


if __name__ == '__main__':
    unittest.main()
