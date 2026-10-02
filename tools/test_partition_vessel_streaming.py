"""Deterministic reporting/accounting tests, not protein physical evidence."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import partition_vessel_streaming as part
import audit_vessel_baseline_streaming as baseline
import audit_hard_free_vessel_streaming as guided
from test_audit_vessel_baseline_streaming import fixture
from test_audit_hard_free_vessel_streaming import complete_fixture


def prepared(root, guided_arm=False):
    population = root/'population'
    if guided_arm:
        binary = complete_fixture(population); region_path = population/'provenance/latent-region.json'
    else:
        binary, region_path = fixture(population)
    config = part.read(population/'config.json'); manifest = part.read(population/'manifest.json')
    region = part.read(region_path)
    region.update(activity=config['reservoir_density'], depletant_radius=config['depletant_radius'],
                  physical_metric=config['metadata'], physical_fixed_neighbors=config['fixed_poses'],
                  minimum_original_q=0., minimum_original_q_inclusive=True,
                  capture_radius=region['capture_radius'] if guided_arm else 3.5)
    part.write(region_path, region)
    paths = {}
    for name, radius, q in [('current_R4', 4., 0.), ('old_native_R4', 4., 0.),
                            ('old_alternative_R5', 5., 1.), ('old_alternative_R32', 32., 1.)]:
        value = copy.deepcopy(region); value.update(mahalanobis_radius=radius,
            minimum_original_q=q, minimum_original_q_inclusive=q == 0.)
        if name == 'old_native_R4': value['maximum_original_q'] = 1.
        path = root/(name+'.json'); part.write(path, value); paths[name] = path
    if guided_arm:
        guide_path = population/'provenance/latent-guide.json'; guide = part.read(guide_path)
        guide['region_sha256'] = part.sha(region_path); part.write(guide_path, guide)
        manifest.update(latent_region_sha256=part.sha(region_path), latent_guide_sha256=part.sha(guide_path))
        manifest['latent_source_capture']['radius'] = region['capture_radius']
        part.write(population/'manifest.json', manifest)
        summary = part.read(population/'summary.json'); summary['manifest'] = manifest
        part.write(population/'summary.json', summary)
        guided.audit(population, root/'audit', binary, 2)
    else:
        baseline.audit(population, root/'audit', binary, paths['current_R4'], 2)
    native_root = root/'native'; inputs = native_root/'inputs'; (inputs/'source').mkdir(parents=True)
    # This deliberately simple frozen observer tests orchestration only.
    runtime = inputs/'source/native_contact_regions.py'
    runtime.write_text('import json\nfrom pathlib import Path\n'
        'class NativeContactRegions:\n'
        '    def __init__(self, path): self.definition = json.loads(Path(path).read_text())\n'
        '    def classify(self, pose): return dict(native_any=pose["position"][0] < 3.1)\n')
    source_config = copy.deepcopy(config); source_config['capture_radius'] = region['capture_radius']
    part.write(inputs/'physical-config.json', source_config)
    definition = dict(shape_sha256=manifest['shape_sha256'], fixed_poses=config['fixed_poses'],
        physical_config_sha256=part.sha(inputs/'physical-config.json'),
        input_sha256={str(p.relative_to(inputs)): part.sha(p) for p in inputs.rglob('*') if p.is_file()},
        criteria=dict(test_only=True), scope='Synthetic native predicate for deterministic software tests only')
    part.write(native_root/'definition.json', definition)
    return root/'audit/analysis.json', paths, native_root/'definition.json'


class ReportingTests(unittest.TestCase):
    def test_guided_arm_uses_same_partition_with_its_own_full_mixture_weights(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); audit, paths, definition = prepared(root, guided_arm=True)
            result = part.analyze(audit, paths, definition, root/'partition')
            self.assertEqual(result['manifest']['schema'], 6)
            self.assertEqual(result['new_native_classifier_calls'], 4)
            self.assertEqual(result['samples'], 7)
            self.assertAlmostEqual(result['estimates']['total']['Qz']['logQ'],
                                   part.read(audit)['estimates']['total']['Qz']['logQ'])
            self.assertEqual(result['new_geometry_queries'], 0)

    def test_complete_disjoint_primary_and_native_subsets_for_all_memberships(self):
        for mask in range(16):
            supports = {k: bool(mask & (1 << i)) for i, k in enumerate(part.SUPPORTS)}
            for contact in (False, True):
                for native in (False, True):
                    classes, regional = part.classify(True, contact, native, supports)
                    self.assertEqual(sum(classes[k] for k in part.PRIMARY), 1)
                    self.assertEqual(classes['inside_measured_pockets'], mask != 0)
                    self.assertEqual(classes['outside_measured_pockets'], mask == 0)
                    self.assertEqual(sum(regional[k] for k in part.REGIONAL[2:4]),
                                     native and supports['current_R4'])
                    self.assertEqual(classes['registered_native_entry'], native)
        with self.assertRaises(ValueError):
            part.classify(False, False, True, dict.fromkeys(part.SUPPORTS, False))

    def test_entrypoint_reuses_geometry_preserves_all_denominators_and_native_anomalies(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); audit, paths, definition = prepared(root)
            # A call into the geometric oracle here would be an accidental replay.
            with patch.object(baseline.baseline.PrunedExclusionContact, 'classify', side_effect=AssertionError('repeated geometry')):
                result = part.analyze(audit, paths, definition, root/'partition')
            self.assertEqual(result['samples'], 7); self.assertEqual(result['invalid_draws'], 3)
            self.assertEqual(result['new_native_classifier_calls'], 4)
            self.assertEqual(result['new_geometry_queries'], 0)
            self.assertGreater(result['native_unbound_anomalies'], 0)
            labels = [json.loads(s) for s in (root/'partition/labels.jsonl').read_text().splitlines()]
            self.assertEqual([x['draw'] for x in labels], list(range(7)))
            for label in labels:
                if not label['classes']['total']:
                    self.assertIsNone(label['native'])
                    self.assertFalse(any(label['classes'].values()))
            total = result['estimates']['total']['Qz']
            self.assertEqual(total['draws'], 7); self.assertEqual(total['nonzero'], 4)
            old = part.read(audit)['estimates']['total']['Qz']['logQ']
            self.assertAlmostEqual(total['logQ'], old)
            self.assertGreater(result['estimates']['outside_measured_pockets']['Qz']['nonzero'], 0)
            for name in part.REGIONAL:
                for family, count in part.BIN_COUNTS.items():
                    bins = result['strata'][family][name]
                    self.assertEqual(len(bins), count)
                    self.assertTrue(all(b['Qz']['draws'] == 7 for b in bins))
                    self.assertEqual(sum(b['Qz']['nonzero'] for b in bins),
                                     result['regional_estimates'][name]['Qz']['nonzero'])
            self.assertEqual(result['regional_estimates'][part.PRIMARY[0]],
                             result['estimates'][part.PRIMARY[0]+':inside_current_R4'])
            part.Ledger().frozen(root/'partition')
            with self.assertRaisesRegex(ValueError, 'Fresh native'):
                part.analyze(audit, paths, definition, root/'partition')

    def test_bad_observer_target_and_changed_audit_fail_before_classification(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); audit, paths, definition = prepared(root)
            value = part.read(definition); value['fixed_poses'] = []
            part.write(definition, value)
            with self.assertRaisesRegex(ValueError, 'Native geometry identity'):
                part.analyze(audit, paths, definition, root/'bad-native')
            self.assertFalse((root/'bad-native').exists())
            with (root/'audit/geometry.jsonl').open('a') as stream: stream.write('{}\n')
            with self.assertRaisesRegex(ValueError, 'Hash mismatch'):
                part.analyze(audit, paths, definition, root/'bad-audit')
            self.assertFalse((root/'bad-audit').exists())

    def test_failed_classifier_retains_previous_draws_and_does_not_retry(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); audit, paths, definition = prepared(root)
            classifier, binding = part.load_classifier(definition)
            original = classifier.classify; calls = []
            def failing(pose):
                calls.append(pose)
                if len(calls) == 2: raise ValueError('injected observer failure')
                return original(pose)
            classifier.classify = failing
            with patch.object(part, 'load_classifier', return_value=(classifier, binding)):
                with self.assertRaisesRegex(ValueError, 'injected observer'):
                    part.analyze(audit, paths, definition, root/'failed')
            status = part.read(root/'failed/status.json')
            self.assertEqual(status['phase'], 'failed'); self.assertEqual(len(calls), 2)
            self.assertEqual(status['new_native_classifier_calls'], 2)
            saved = (root/'failed/labels.jsonl').read_text().splitlines()
            self.assertEqual(len(saved), status['processed_attempts'])
            self.assertGreater(len(saved), 0); self.assertFalse((root/'failed/analysis.json').exists())

    def test_old_r5_keeps_original_capture_and_strict_registration(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); audit, paths, definition = prepared(root)
            config = part.read(root/'population/config.json'); manifest = part.read(root/'population/manifest.json')
            regions = {k: part.read(v) for k, v in paths.items()}
            part.validate_supports(regions, config, manifest)
            for key, value in [('mahalanobis_radius', 4.), ('minimum_original_q_inclusive', True),
                               ('capture_radius', config['capture_radius'])]:
                changed = copy.deepcopy(regions); changed['old_alternative_R5'][key] = value
                with self.subTest(key=key), self.assertRaises(ValueError):
                    part.validate_supports(changed, config, manifest)


if __name__ == '__main__': unittest.main()
