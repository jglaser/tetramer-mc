"""Synthetic all-attempt receipt accounting; no raw rows or scientific calls."""
import copy
import unittest

import summarize_native_class_support_pilot as summary


def fixture(identity='r00'):
    pop = dict(id=identity, samples=128, selected_ids=list(range(16)))
    inventory = [dict(index=i, bank='original' if i < 92 else 'added',
                      training_id=None if i < 92 else 'training', latent_sigma=None if i < 92 else .05)
                 for i in range(116)]
    algebra = []; labels = []
    for i in range(128):
        digest = f'{i:064x}'
        algebra.append(dict(id=i, sample_record_sha256=digest, selected=None,
                            producer_cpu_seconds=dict(draw=.1, density=.2, observer=.3)))
        labels.append(dict(id=i, sample_record_sha256=digest, valid=False, native=None, contact=None,
                           orthant=i % 64, selected_channel_match=None))
    # Keep a legitimate native/contact anomaly and two competing endpoints.
    labels[0].update(valid=True, native=True, contact=False, orthant=55, selected_channel_match=True)
    algebra[0]['selected'] = dict(component=92, channel=4, class_mass=.4, hard_mass=.7,
                                 effective_mass=.4, fallback='class', empty_class=False)
    for i, orthant in [(1,22), (2,62)]:
        labels[i].update(valid=True, native=False, contact=True, orthant=orthant, selected_channel_match=True)
        algebra[i]['selected'] = dict(component=92+i, channel=i+1, class_mass=.4, hard_mass=.7,
            effective_mass=.4, fallback='class', empty_class=False)
    receipts = []
    for phase, rows in [('algebra', algebra), ('labels', labels),
                        ('geometry', [dict(id=i, sample_record_sha256=f'{i:064x}') for i in range(16)])]:
        receipts.append(dict(schema='native-class-support-pilot-'+phase+'-v1', phase=phase, population=identity,
                             complete=True, passed=True, rows=rows, analysis_cpu_seconds=1., counts={}, setup_counts={}))
    return pop, *receipts, inventory


class ReductionTests(unittest.TestCase):
    def test_invalid_denominators_anomaly_and_branch_rates_remain_separate(self):
        report = summary.population_report(*fixture())
        self.assertEqual(report['attempted'], 128)
        self.assertEqual(report['counts']['invalid_or_exterior'], 125)
        self.assertEqual(report['counts']['native'], 1)
        self.assertEqual(report['counts']['native_without_contact_anomaly'], 1)
        self.assertEqual(report['critical_endpoints'], dict(competing22=1, competing62=1, native55=1))
        self.assertEqual(report['target_line_hit_rates']['channel:2'], dict(selected=1, hits=1, hit_fraction=1.))
        reports = [summary.population_report(*fixture(f'r{i:02}')) for i in range(4)]
        combined = summary.aggregate(reports)
        self.assertEqual(combined['attempted_denominator'], 512)
        self.assertEqual(combined['fractions']['competing']['count'], 8)
        self.assertEqual(combined['fractions']['competing']['fraction'], 8/512)
        self.assertEqual(combined['access_screen'], 'access_observed_requires_review')
        self.assertFalse(combined['physical_campaign_gate_open'])

    def test_missing_or_substituted_audit_rows_fail(self):
        value = fixture(); value[2]['rows'].pop()
        with self.assertRaisesRegex(ValueError, 'IDs'): summary.population_report(*value)
        value = fixture(); value[3]['rows'][-1]['id'] = 16
        with self.assertRaisesRegex(ValueError, 'IDs'): summary.population_report(*value)
        value = fixture(); value[3]['rows'][0]['sample_record_sha256'] = 'different'
        with self.assertRaisesRegex(ValueError, 'same exact row bytes'): summary.population_report(*value)

    def test_endpoint_hits_cannot_replace_selected_five_coordinate_line_hits(self):
        reports = []
        for i in range(4):
            value = fixture(f'r{i:02}')
            value[1]['rows'][1]['selected'].update(class_mass=0., hard_mass=.7, effective_mass=.7,
                                                 fallback='hard_free', empty_class=True)
            reports.append(summary.population_report(*value))
        result = summary.aggregate(reports)
        self.assertEqual(result['critical_endpoints']['competing22'], 4)
        self.assertEqual(result['selected_target_line_access']['2']['hits'], 0)
        self.assertEqual(result['access_screen'], 'unsuccessful_or_inconclusive')
        self.assertFalse(result['physical_campaign_gate_open'])

    def test_native55_is_an_explicit_access_requirement(self):
        reports = [summary.population_report(*fixture(f'r{i:02}')) for i in range(4)]
        for report in reports: report['critical_endpoints']['native55'] = 0
        self.assertEqual(summary.aggregate(reports)['access_screen'], 'unsuccessful_or_inconclusive')
        reports = [summary.population_report(*fixture(f'r{i:02}')) for i in range(4)]
        for report in reports: report['target_line_hit_rates']['channel:4']['hits'] = 0
        self.assertEqual(summary.aggregate(reports)['access_screen'], 'unsuccessful_or_inconclusive')


if __name__ == '__main__': unittest.main()
