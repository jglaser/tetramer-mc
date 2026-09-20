"""Selection provenance must resolve an arm and the original full-row maximum."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from analyze_peak_neighborhood import audit_peak_selection, selected_source_record
from prepare_cayley_rms_cover import sha


class PeakSelectionAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.cfg = dict(fixed_poses=[{'position': [1., 2., 3.]}], capture_radius=18.)
        self.pose = {'position': [0., 0., 0.], 'orientation': [1., 0., 0., 0.]}
        self.selected = dict(population='r01', seed=102, draw=1, pose=self.pose,
                             q=2.4, original_log_importance_weight=5.)
        self.paths = []; hashes = {}; populations = []
        for index, weights in enumerate(([None, 2.], [1., 5.])):
            path = self.root/'campaign'/'runs'/f'r{index:02d}'/'samples.jsonl'
            path.parent.mkdir(parents=True); self.paths.append(path)
            rows = [dict(draw=i, pose=self.pose, q=2.4, log_importance_weight=w) for i, w in enumerate(weights)]
            if index == 0: rows[0].pop('log_importance_weight')
            path.write_text(''.join(json.dumps(row)+'\n' for row in rows)); hashes[str(path)] = sha(path)
            populations.append(dict(id=f'r{index:02d}', seed=101+index, samples=len(rows)))
        window = dict(minimum=2., maximum=5., lower_inclusive=True, upper_inclusive=False)
        self.campaign = dict(arm='broad', root=str(self.root/'campaign'), top_poses=[self.selected],
            populations=populations, original_full_window_audit=dict(physical_signature=self.cfg,
                shape_sha256='shape-hash', original_integration_window=window, sample_sha256=hashes))
        self.historical = dict(complete=True, original_q_window=window, campaigns=[self.campaign])
        self.source = self.root/'source-audit.json'; self.archive = self.root/'extremes.json'
        self.freeze()

    def freeze(self):
        data = json.dumps(self.historical)
        self.source.write_text(data); self.archive.write_text(data)
        self.protocol = dict(selection_source=dict(kind='atlas_maximum', arm='broad',
            path=str(self.source), sha256=sha(self.source)))

    def audit(self, **kwargs):
        return audit_peak_selection(self.protocol, self.archive, self.selected,
                                    kwargs.get('cfg', self.cfg), kwargs.get('shape', 'shape-hash'))

    def test_full_arm_maximum_keeps_zero_rows(self):
        result = self.audit()
        self.assertTrue(result['whole_arm_maximum_checked'])
        self.assertEqual(result['original_unconditional_draws'], 4)
        self.assertEqual(result['maximum_original_log_importance_weight'], 5.)

    def test_larger_omitted_pose_rejected_even_with_updated_hashes(self):
        rows = [json.loads(line) for line in self.paths[0].read_text().splitlines()]
        rows[0]['log_importance_weight'] = 6.
        self.paths[0].write_text(''.join(json.dumps(row)+'\n' for row in rows))
        self.campaign['original_full_window_audit']['sample_sha256'][str(self.paths[0])] = sha(self.paths[0])
        self.freeze()
        with self.assertRaisesRegex(ValueError, 'whole-arm importance maximum'): self.audit()

    def test_wrong_arm_and_nonmaximum_shortlist_rejected(self):
        self.protocol['selection_source']['arm'] = 'narrow'
        with self.assertRaisesRegex(ValueError, 'selected contact-atlas arm'): self.audit()
        self.protocol['selection_source']['arm'] = 'broad'
        altered = copy.deepcopy(self.historical)
        altered['campaigns'][0]['top_poses'].append(dict(self.selected, original_log_importance_weight=6.))
        with self.assertRaisesRegex(ValueError, 'listed importance maximum'):
            selected_source_record(altered, self.protocol['selection_source'])

    def test_physical_target_and_original_rows_immutable(self):
        with self.assertRaisesRegex(ValueError, 'physical target differs'):
            self.audit(cfg=dict(self.cfg, capture_radius=19.))
        with self.assertRaisesRegex(ValueError, 'Selection shape changed'): self.audit(shape='another-shape')
        self.paths[0].write_text(self.paths[0].read_text().replace('2.0', '2.5'))
        with self.assertRaisesRegex(ValueError, 'original rows changed'): self.audit()

    def test_source_and_archive_must_match_frozen_audit(self):
        self.archive.write_text(self.archive.read_text()+' ')
        with self.assertRaisesRegex(ValueError, 'source audit changed'): self.audit()

    def test_legacy_absent_selection_source_preserves_remainder_rule(self):
        historical = dict(pieces=[dict(name='r4', top_poses=[{'pose': 'other'}]),
                                  dict(name='remainder', top_poses=[self.selected])])
        self.archive.write_text(json.dumps(historical))
        result = audit_peak_selection({}, self.archive, self.selected, self.cfg, 'shape-hash')
        self.assertEqual(result['kind'], 'legacy_remainder_maximum')
        with self.assertRaisesRegex(ValueError, 'historical maximum'):
            audit_peak_selection({}, self.archive, dict(self.selected, draw=0), self.cfg, 'shape-hash')

    def test_explicit_remainder_source_checks_archive_and_source_hash(self):
        historical = dict(pieces=[dict(name='remainder', top_poses=[self.selected])])
        data = json.dumps(historical); self.archive.write_text(data); self.source.write_text(data)
        protocol = dict(selection_source=dict(kind='remainder_maximum', path=str(self.source),
                                             sha256=sha(self.source)))
        result = audit_peak_selection(protocol, self.archive, self.selected, self.cfg, 'shape-hash')
        self.assertEqual(result['kind'], 'remainder_maximum')
        self.assertTrue(result['source_hash_checked'])
        self.source.write_text(data+' ')
        with self.assertRaisesRegex(ValueError, 'source audit changed'):
            audit_peak_selection(protocol, self.archive, self.selected, self.cfg, 'shape-hash')


if __name__ == '__main__': unittest.main()
