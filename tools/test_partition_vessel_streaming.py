"""Deterministic reporting/accounting tests, not protein physical evidence."""
import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import partition_vessel_streaming as part
import audit_vessel_baseline_streaming as baseline
import audit_hard_free_vessel_streaming as guided
from test_audit_vessel_baseline_streaming import fixture, envelope_fixture
from test_audit_hard_free_vessel_streaming import complete_fixture


def prepared(root, guided_arm=False, binary_path=None, native_directory='native', envelope=False):
    population = root/'population'
    if guided_arm:
        binary = complete_fixture(population); region_path = population/'provenance/latent-region.json'
    else:
        binary, region_path = (envelope_fixture(population) if envelope else fixture(population))
    if binary_path is not None:
        shutil.copy2(binary,binary_path); binary = binary_path
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
    summary = part.read(population/'summary.json'); summary['sampler_cpu_seconds'] = 1.
    part.write(population/'summary.json',summary)
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
    native_root = root/native_directory; inputs = native_root/'inputs'; (inputs/'source').mkdir(parents=True)
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


def prepared_class(root, schema=7, binary_path=None, native_directory='native'):
    """Orchestration fixture: reuse seven deterministic rows, not class sampling.

    The geometry/weight records are existing hard-free toy controls. A small
    compiled observer tests the new provenance adapter and physical partition;
    complete native-class densities have their own saved-fixture audit suite.
    """
    audit, paths, definition = prepared(root, True, binary_path, native_directory)
    population = root/'population'; value = part.read(audit); manifest = value['manifest']
    shape = part.read(population/'provenance/shape.json')
    shutil.copy2(population/'provenance/shape.json', definition.parent/'inputs/tetramer-shape.json')
    native = part.read(definition)
    native['input_sha256']['tetramer-shape.json'] = manifest['shape_sha256']; part.write(definition, native)
    identity = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
    compiled = dict(schema='native-entry-compiled-v1', source_definition_sha256=part.sha(definition),
        source_input_sha256=native['input_sha256'], criteria=part.native_class.reference.line.CRITERIA,
        fixed_poses=native['fixed_poses'], members=[dict(position=[0., 0., 0.], rotation=identity)],
        monomer_atoms=[dict(a, residue=0) for a in shape['atoms']], residue_count=1, references=[], motifs=[])
    compiled_path = population/'provenance/compiled-native.json'; part.write(compiled_path, compiled)
    n_atoms = len(shape['atoms'])
    witness = dict(compiled_sha256=part.sha(compiled_path), expected_shape_sha256=manifest['shape_sha256'],
        center_tolerance_a=1e-10, radius_tolerance_a=1e-12, observer_hard_overlap_tolerance_a=1e-8,
        native_atoms=n_atoms, physical_atoms=n_atoms, matched_atoms=n_atoms,
        physical_index_by_native_atom=list(range(n_atoms)), compatible=True,
        unmatched_native_atoms=[], unmatched_physical_atoms=[], matched_max_center_error_a=0.,
        matched_max_radius_error_a=0., pair_overlap_slack_bound_a=0., hard_valid_implication_within_tolerance=True)
    manifest.update(schema=schema, outer_mixture_schema=part.native_class.reference.SCHEMA,
        latent_guide_schema=part.native_class.reference.line.SCHEMA,
        attempt_journal='attempts.jsonl; begin before each attempt; no retries', resume_supported=False,
        compiled_native=dict(compiled_sha256=part.sha(compiled_path), source_definition_sha256=part.sha(definition),
            source_input_sha256=native['input_sha256'], shape_compatibility=witness))
    manifest['latent_source_capture']['conditions_guide'] = True
    if schema == 8:
        from audit_full_vessel_latent import AtomWallEnvelope
        import math
        atom_index = max(range(n_atoms), key=lambda i: shape['atoms'][i]['radius']); atom = shape['atoms'][atom_index]
        wall = manifest['atomic_wall']; radius = wall['radius']-atom['radius']
        envelope = dict(atom_index=atom_index, atom_center=atom['center'], atom_radius=atom['radius'],
            wall_center=wall['center'], wall_radius=wall['radius'], envelope_radius=radius,
            log_volume=math.log(4*math.pi/3)+3*math.log(radius))
        manifest.update(pre_envelope_schema=7, vessel_uniform_schema='one-atom-wall-envelope-v1',
                        vessel_uniform_envelope=envelope)
        value['vessel_uniform_envelope'] = AtomWallEnvelope(shape, wall, envelope).witness
    part.write(population/'manifest.json', manifest)
    summary = part.read(population/'summary.json'); summary['manifest'] = manifest; part.write(population/'summary.json', summary)
    value.update(schema=part.native_class.SCHEMA, manifest=manifest,
        shape_witness=part.native_class.reference.regional.validate_shape_witness(compiled, shape, witness,
            part.sha(compiled_path), manifest['shape_sha256']),
        geometry_reconstruction=dict(physical_pose_checks=7, density_pose_checks=7),
        new_pose_draws=0, new_Poisson_clouds=0, new_native_classifier_calls=0)
    sources = part.local_sources(Path(part.native_class.__file__))
    for name, path in sources.items():
        shutil.copy2(path, audit.parent/'provenance'/name); value['source_sha256'][str(path.resolve())] = part.sha(path)
    for path in [definition, compiled_path, *[definition.parent/'inputs'/n for n in native['input_sha256']]]:
        value['source_sha256'][str(path.resolve())] = part.sha(path)
    value['source_sha256'] = {p: part.sha(p) for p in value['source_sha256']}
    part.write(audit, value)
    status = part.read(audit.parent/'status.json'); status['analysis_sha256'] = part.sha(audit); part.write(audit.parent/'status.json', status)
    refreeze(audit.parent)
    observer = part.native_class.reference.line.observer_from_compiled_for_synthetic(compiled)
    observer.definition = native; observer.definition_sha256 = part.sha(definition)
    observer.classify = lambda pose: dict(native_any=pose['position'][0] < 3.1)
    _, binding = part.load_classifier(definition)
    return audit, paths, definition, observer, binding


def refreeze(root):
    part.write(root/'freeze.json', dict(files={str(p.relative_to(root)): part.sha(p)
        for p in root.rglob('*') if p.is_file() and p.name != 'freeze.json'}))


class NativeClassAdapterTests(unittest.TestCase):
    def test_class_schema7_and8_preserve_partition_denominator_and_original_classifier(self):
        for schema in (7, 8):
            with self.subTest(schema=schema), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); audit, paths, definition, observer, binding = prepared_class(root, schema)
                with patch.object(part, 'load_classifier', return_value=(observer, binding)):
                    result = part.analyze(audit, paths, definition, root/'partition')
                self.assertEqual(result['samples'], 7)
                self.assertEqual(result['invalid_draws'], 3)
                self.assertEqual(result['new_native_classifier_calls'], 4)
                self.assertEqual(result['new_geometry_queries'], 0)
                self.assertEqual(result['proposal_native_binding']['source_definition_sha256'], part.sha(definition))
                self.assertAlmostEqual(result['estimates']['total']['Qz']['logQ'],
                                       part.read(audit)['estimates']['total']['Qz']['logQ'], places=12)

    def test_native_source_mapping_compilation_shape_and_attempt_coverage_rejected(self):
        for defect in ('definition_binding', 'compiled_binding', 'source_archive', 'shape_mapping', 'coverage'):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); audit, _, definition, observer, _ = prepared_class(root)
                value = part.read(audit)
                if defect == 'definition_binding': value['source_sha256'].pop(str(definition.resolve()))
                elif defect == 'compiled_binding': value['manifest']['compiled_native']['compiled_sha256'] = '0'*64
                elif defect == 'source_archive':
                    path = audit.parent/'provenance/audit_native_class_vessel_streaming.py'
                    path.write_text(path.read_text()+'\n# tampered archive\n')
                elif defect == 'shape_mapping': value['manifest']['compiled_native']['shape_compatibility']['matched_atoms'] += 1
                else: value['geometry_reconstruction']['physical_pose_checks'] = 6
                # Keep the completion snapshot consistent so rejection reaches
                # the identity check rather than a generic stale-summary check.
                summary = part.read(root/'population/summary.json'); summary['manifest'] = value['manifest']
                part.write(root/'population/summary.json', summary)
                value['source_sha256'][str((root/'population/summary.json').resolve())] = part.sha(root/'population/summary.json')
                refreeze(audit.parent)
                with self.assertRaises(ValueError):
                    part.bind_class_audit(part.Ledger(), audit, value, part.read(audit.parent/'freeze.json')['files'], definition, observer)

    def test_missing_external_original_and_unfrozen_extra_file_are_not_synthetic_escape_hatches(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); audit, _, definition, observer, _ = prepared_class(root)
            value = part.read(audit); frozen = part.read(audit.parent/'freeze.json')['files']
            (audit.parent/'extra.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'unfrozen'):
                part.bind_class_audit(part.Ledger(), audit, value, frozen, definition, observer)


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
