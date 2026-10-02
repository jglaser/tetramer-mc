"""Deterministic baseline-audit controls; no random physical jobs."""
import copy
import json
import math
from pathlib import Path
import tempfile
import unittest

from scipy.special import logsumexp

import audit_vessel_baseline_streaming as baseline
from test_audit_hard_free_vessel_streaming import complete_fixture


def fixture(root):
    binary = complete_fixture(root); config = baseline.read(root/'config.json')
    manifest = baseline.read(root/'manifest.json'); manifest['schema'] = 4
    manifest.pop('outer_mixture_schema'); manifest.pop('outer_vessel_probability')
    for key in list(manifest):
        if key.startswith('latent_'): manifest.pop(key)
    manifest.update(attempt_journal='attempts.jsonl; begin before each attempt; no retries',resume_supported=False)
    vessel = baseline.baseline.VesselDensity(config,manifest,baseline.read(root/'provenance/model.json'))
    old_log = float(vessel.evaluate([config['initial_pose']])[1]['anchor_log_densities'][0,0])
    rows = [json.loads(s) for s in (root/'samples.jsonl').read_text().splitlines()]
    for row in rows:
        logq = row['log_vessel_proposal_density']
        for key in ['outer_branch','latent_proposal','latent_density','log_vessel_proposal_density','log_latent_physical_density']:
            row.pop(key)
        row['log_proposal_density'] = logq
        if row['hard_valid']: row['log_hard_weight'] = row['log_importance_weight'] = -logq
        if row['proposal'] is None:
            row['proposal'] = dict(moving_index=0,anchor_index=1,null_reason=None,candidate=copy.deepcopy(row['pose']),
                branch='uniform',component_index=None,new_log_density=logq,old_log_density=old_log,
                log_reverse_forward=old_log-logq)
    (root/'samples.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    baseline.write(root/'manifest.json',manifest)
    summary = baseline.read(root/'summary.json'); summary['manifest'] = manifest
    summary['samples_sha256'] = baseline.sha(root/'samples.jsonl')
    total = float(logsumexp([r['log_hard_weight'] for r in rows if r['hard_valid']])-math.log(len(rows)))
    summary['estimates'] = {k:dict(log_normalizer=total) for k in ['total','hard_total']}
    baseline.write(root/'summary.json',summary)
    return binary,root/'provenance/latent-region.json'


class BaselineAuditTests(unittest.TestCase):
    def test_reporting_chart_changes_labels_but_not_target_or_weights(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); population = root/'population'; binary,region = fixture(population)
            first = baseline.audit(population,root/'first',binary,region,2)
            alternate = baseline.read(region)
            alternate['gaussian_chart']['covariances'] = [[[100. if i == j else 0. for j in range(6)] for i in range(6)]]
            alternate['capture_radius'] = .01
            baseline.write(root/'alternate.json',alternate)
            second = baseline.audit(population,root/'second',binary,root/'alternate.json',64)
            self.assertEqual(first['estimates']['total'],second['estimates']['total'])
            self.assertNotEqual(first['estimates']['inside_R4']['Q0']['nonzero'],second['estimates']['inside_R4']['Q0']['nonzero'])
            self.assertEqual(first['estimates']['total']['Q0']['nonzero'],4)
            self.assertEqual(first['primitive_count_audit'],second['primitive_count_audit'])
            self.assertEqual(first['density_audit']['checked_attempts'],7)
            self.assertEqual(first['batching']['peak_rows'],2)
            self.assertEqual(second['batching']['peak_rows'],7)
            self.assertFalse(first['reporting_region_binding']['affects_proposal'])
            self.assertFalse(first['reporting_region_binding']['restricts_target'])

    def test_missing_journal_and_wrong_reporting_shape_fail_before_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); population = root/'population'; binary,region = fixture(population)
            bad = baseline.read(region); bad['shape_sha256'] = '0'*64
            baseline.write(root/'bad.json',bad)
            with self.assertRaisesRegex(ValueError,'Reporting shape/scaffold'):
                baseline.audit(population,root/'wrong-shape',binary,root/'bad.json')
            self.assertFalse((root/'wrong-shape').exists())
            manifest = baseline.read(population/'manifest.json'); manifest.pop('attempt_journal')
            baseline.write(population/'manifest.json',manifest)
            summary = baseline.read(population/'summary.json'); summary['manifest'] = manifest
            baseline.write(population/'summary.json',summary)
            with self.assertRaisesRegex(ValueError,'journal/domain absent'):
                baseline.audit(population,root/'no-journal',binary,region)
            self.assertFalse((root/'no-journal').exists())

    def test_an_extra_density_factor_fails_and_preserves_earlier_attempts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); population = root/'population'; binary,region = fixture(population)
            rows = [json.loads(s) for s in (population/'samples.jsonl').read_text().splitlines()]
            rows[4]['log_hard_weight'] += .2
            (population/'samples.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            summary = baseline.read(population/'summary.json'); summary['samples_sha256'] = baseline.sha(population/'samples.jsonl')
            baseline.write(population/'summary.json',summary)
            with self.assertRaisesRegex(ValueError,'no mixture or extra J'):
                baseline.audit(population,root/'failed',binary,region,2)
            self.assertEqual(len((root/'failed/geometry.jsonl').read_text().splitlines()),4)
            self.assertEqual(baseline.read(root/'failed/status.json')['phase'],'failed')
            self.assertFalse((root/'failed/analysis.json').exists())


if __name__ == '__main__': unittest.main()
