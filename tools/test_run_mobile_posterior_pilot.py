#!/usr/bin/env python3
"""Synthetic freezer/controller tests: fake embedded binary, mocked children only."""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import run_mobile_posterior_pilot as controller


class FakeChild:
    def __init__(self, index, code=0):
        self.pid, self.code, self.waits = 10000+index, code, 0

    def poll(self):
        return self.code

    def wait(self):
        self.waits += 1
        return self.code


class MobileControllerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='mobile-controller-synthetic-test-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root/'reviewed'
        self.files = ('Cargo.toml', 'Cargo.lock', 'build.rs', 'vendor/README.md', 'src/main.rs')
        records = {}
        for name in self.files:
            path = self.source/name
            path.parent.mkdir(parents=True, exist_ok=True)
            raw = ('Synthetic source fixture '+name+'\n').encode()
            path.write_bytes(raw)
            records[name] = dict(text=raw.decode(), sha256=hashlib.sha256(raw).hexdigest())
        self.bundle = self.root/'source-bundle.json'
        controller.write(self.bundle, dict(schema=1, files=records))
        self.binary = self.root/'inert-binary-fixture'
        self.binary.write_bytes(b'Synthetic bytes: never execute\0'+self.bundle.read_bytes()+b'\0end')
        self.binary.chmod(0o700)
        self.observer = self.root/'analyze_mobile_posterior_pilot.py'
        self.observer.write_text('from synthetic_observer_dependency import marker\n# Never executed in these tests.\n')
        (self.root/'synthetic_observer_dependency.py').write_text('marker = "archived dependency"\n')
        self.campaign, self.journal, self.assessment = [self.root/name for name in ('campaign', 'journal', 'assessment')]
        self.addCleanup(patch.stopall)
        patch.object(controller, 'print', create=True).start()

    def freeze(self):
        # Read the real inert preparation, but all writes go to a temporary
        # synthetic campaign with fake source bytes and a never-executed binary.
        return controller.freeze(controller.PREPARATION, self.campaign, self.binary, self.bundle,
                                 reviewed_root=self.source, analyzer=self.observer)

    def refresh_manifest_freeze_hash(self, manifest):
        controller.write(self.campaign/'manifest.json', manifest)
        frozen = controller.read(self.campaign/'freeze.json')
        frozen['files']['manifest.json'] = controller.sha(self.campaign/'manifest.json')
        for job in manifest['jobs']:
            frozen['files'][str(Path(job['config']).relative_to(self.campaign))] = controller.sha(job['config'])
        controller.write(self.campaign/'freeze.json', frozen)

    def test_freeze_preflight_are_inert_and_archive_all_dependencies(self):
        with patch.object(controller.subprocess, 'Popen') as launch, patch.object(controller.subprocess, 'run') as audit:
            result = self.freeze()
            verified = controller.check(self.campaign, self.journal, self.assessment)
        launch.assert_not_called()
        audit.assert_not_called()
        self.assertEqual(result, verified)
        self.assertEqual(result['total_attempts'], 120000)
        self.assertEqual(len(result['jobs']), 12)
        self.assertEqual(set(result['rust_sources']), set(self.files))
        self.assertTrue((self.campaign/'provenance/synthetic_observer_dependency.py').is_file())
        for name in ('scripts/tetramer_order.py', 'scripts/analyze_fluid_sampling.py',
                     'results/c1c3-scaffold/motifs.json', 'results/native-neighbor-classes/classification.json',
                     'results/native-geometry-repair/rebuilt-hydrogens/heavy-coordinates.json'):
            path = self.campaign/'provenance/reference'/name
            self.assertTrue(path.is_file())
            self.assertIn('reference/'+name, result['input_sha256'])
        labels = controller.validate_residue_coordinates(self.campaign/'provenance/monomer-shape.json',
                                                         self.campaign/'provenance'/controller.COORDINATE_ARCHIVE)
        self.assertEqual(labels, result['residue_coordinate_validation'])
        self.assertEqual((labels['atoms'], labels['residues']), (1001, 129))
        self.assertEqual(labels['maximum_coordinate_discrepancy_angstrom'], 0.)
        for job in result['jobs']:
            config = controller.read(job['config'])
            self.assertTrue(config['shape'].startswith(str(self.campaign/'provenance')))
            self.assertTrue(config['monomer_shape'].startswith(str(self.campaign/'provenance')))
            self.assertTrue(config['metadata']['native_pair_motifs'].startswith(str(self.campaign/'provenance')))
            self.assertNotIn('--no-moves', job['command'])
        self.assertFalse(self.journal.exists())
        self.assertFalse(self.assessment.exists())
        self.assertFalse((self.campaign/'status.json').exists())
        with self.assertRaisesRegex(ValueError, 'Fresh campaign'):
            self.freeze()

    def test_bundle_requires_exact_embedded_bytes_and_reviewed_source(self):
        controller.verify_bundle(self.binary, self.bundle, self.source)
        original = self.bundle.read_bytes()
        self.bundle.write_bytes(original+b'\n')
        with self.assertRaisesRegex(ValueError, 'Exact supplied source-bundle bytes'):
            controller.verify_bundle(self.binary, self.bundle, self.source)
        self.bundle.write_bytes(original)
        (self.source/'src/main.rs').write_text('Changed reviewed source\n')
        with self.assertRaisesRegex(ValueError, 'Reviewed source differs'):
            controller.verify_bundle(self.binary, self.bundle, self.source)
        records = controller.read(self.bundle)
        records['files']['src/main.rs']['sha256'] = '0'*64
        controller.write(self.bundle, records)
        self.binary.write_bytes(b'fake\0'+self.bundle.read_bytes())
        with self.assertRaisesRegex(ValueError, 'text/hash mismatch'):
            controller.verify_bundle(self.binary, self.bundle, self.source)

    def coordinate_fixture(self):
        directory = self.root/'synthetic-residue-input'
        directory.mkdir()
        coordinate_path = directory/'heavy-coordinates.json'
        shape_path = directory/'monomer-shape.json'
        records = []
        for index in range(3):
            line = [' ']*80
            line[12:16] = list(' CA ')
            line[17:20] = list('ALA')
            line[21] = 'A'
            line[22:27] = list(f'{index//2+1:4d} ')
            records.append(''.join(line))
        controller.write(coordinate_path, dict(positions=[[4., 5., 6.], [6., 5., 6.], [8., 5., 6.]], atom_records=records))
        controller.write(shape_path, dict(source_coordinates_sha256=controller.sha(coordinate_path),
            atoms=[dict(center=[x, 0., 0.], radius=.5) for x in (-2., 0., 2.)]))
        return shape_path, coordinate_path

    def test_residue_data_contract_and_correct_sibling_archive_lookup(self):
        shape_path, coordinates = self.coordinate_fixture()
        source, labels = controller.locate_residue_coordinates(shape_path, fallback=self.root/'missing')
        self.assertEqual(source, coordinates)
        self.assertEqual(labels['atom_to_residue'], [0, 0, 1])
        self.assertEqual((labels['atoms'], labels['residues']), (3, 2))
        self.assertEqual(labels['maximum_coordinate_discrepancy_angstrom'], 0.)
        archived_shape = self.root/'archived/monomer-shape.json'
        archived_shape.parent.mkdir()
        archived_shape.write_bytes(shape_path.read_bytes())
        source, labels_again = controller.locate_residue_coordinates(archived_shape, [shape_path], self.root/'missing')
        self.assertEqual(source, coordinates)
        self.assertEqual(labels, labels_again)
        fallback = self.root/'addon'/controller.COORDINATE_ARCHIVE
        fallback.parent.mkdir(parents=True)
        fallback.write_bytes(coordinates.read_bytes())
        source, fallback_labels = controller.locate_residue_coordinates(archived_shape, fallback=fallback)
        self.assertEqual(source, fallback)
        self.assertEqual(fallback_labels, labels)

    def test_residue_data_rejects_missing_wrong_hash_count_and_atom_order(self):
        shape_path, coordinates = self.coordinate_fixture()
        original = controller.read(coordinates)
        shape = controller.read(shape_path)
        coordinates.unlink()
        with self.assertRaisesRegex(ValueError, 'heavy-coordinate records unavailable'):
            controller.locate_residue_coordinates(shape_path, fallback=self.root/'missing')
        controller.write(coordinates, dict(original, unexpected_change=True))
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            controller.validate_residue_coordinates(shape_path, coordinates)
        for mutation in ('count', 'order'):
            value = copy.deepcopy(original)
            if mutation == 'count': value['atom_records'].pop()
            else: value['positions'].reverse()
            controller.write(coordinates, value)
            shape['source_coordinates_sha256'] = controller.sha(coordinates)
            controller.write(shape_path, shape)
            with self.subTest(mutation=mutation), self.assertRaisesRegex(ValueError, 'count/dimensions|atom ordering'):
                controller.validate_residue_coordinates(shape_path, coordinates)

    def test_freeze_rejects_missing_native_observer_data_before_creating_campaign(self):
        with patch.object(controller, 'AUTHORITATIVE_COORDINATES', self.root/'missing-authoritative-records.json'):
            with self.assertRaisesRegex(ValueError, 'cannot freeze native observer data closure'):
                self.freeze()
        self.assertFalse(self.campaign.exists())

    def test_changed_physics_command_and_archived_bytes_are_rejected(self):
        original = self.freeze()
        changed = copy.deepcopy(original)
        changed['jobs'][0]['command'].append('--no-moves')
        self.refresh_manifest_freeze_hash(changed)
        with self.assertRaisesRegex(ValueError, 'Exact frozen command changed'):
            controller.validate_immutable(self.campaign)
        self.refresh_manifest_freeze_hash(original)
        path = Path(original['jobs'][0]['config'])
        config = controller.read(path)
        config['reservoir_density'] = .04
        controller.write(path, config)
        changed = copy.deepcopy(original)
        changed['jobs'][0]['config_sha256'] = controller.sha(path)
        self.refresh_manifest_freeze_hash(changed)
        with self.assertRaisesRegex(ValueError, 'Physical configuration changed'):
            controller.validate_immutable(self.campaign)
        (self.campaign/'provenance/model.json').write_text('changed model\n')
        with self.assertRaisesRegex(ValueError, 'Frozen campaign file changed'):
            controller.validate_immutable(self.campaign)

    def test_preflight_refuses_existing_outputs_journals_status_or_assessment(self):
        manifest = self.freeze()
        for path in (self.journal, self.assessment, self.campaign/'status.json',
                     Path(manifest['jobs'][0]['directory']), Path(manifest['jobs'][0]['log'])):
            path.write_text('existing, must preserve\n')
            with self.subTest(path=path), self.assertRaises(ValueError):
                controller.check(self.campaign, self.journal, self.assessment)
            self.assertEqual(path.read_text(), 'existing, must preserve\n')
            path.unlink()

    def test_success_waits_for_all_children_then_invokes_only_archived_observer(self):
        manifest = self.freeze()
        children = [FakeChild(i) for i in range(12)]

        def observer(command, **_):
            self.assertTrue(all(child.waits == 1 for child in children))
            self.assertEqual(command[1], str(self.campaign/'provenance/analyze_mobile_posterior_pilot.py'))
            self.assertEqual(command[command.index('--reference')+1], manifest['reference'])
            self.assertEqual(command[command.index('--workers')+1], '12')
            self.assertTrue(controller.read(self.campaign/'status.json')['complete'])
            self.assessment.mkdir()
            rows = [dict({key: job[key] for key in ('id', 'mode', 'start', 'replicate', 'seed')}, passed=True)
                    for job in manifest['jobs']]
            controller.write(self.assessment/'analysis.json', dict(complete=True, runs=rows,
                manifest_sha256=controller.sha(self.campaign/'manifest.json'),
                terminal_status_sha256=controller.sha(self.campaign/'status.json'),
                analyzer_sha256=manifest['observer_sha256']))
            return subprocess.CompletedProcess(command, 0)

        with patch.object(controller.subprocess, 'Popen', side_effect=children) as launches, \
             patch.object(controller.subprocess, 'run', side_effect=observer) as audit:
            state = controller.run(self.campaign, self.journal, self.assessment)
        self.assertEqual(launches.call_count, 12)
        audit.assert_called_once()
        self.assertTrue(state['complete'])
        self.assertEqual(state['phase'], 'complete')
        self.assertEqual(len(controller.read(self.campaign/'status.json')['jobs']), 12)
        with self.assertRaisesRegex(ValueError, 'No restart'):
            controller.run(self.campaign, self.journal, self.assessment)

    def test_failed_child_is_drained_with_every_launched_result_and_no_observer(self):
        self.freeze()
        children = [FakeChild(i, 7 if i == 3 else 0) for i in range(12)]
        with patch.object(controller.subprocess, 'Popen', side_effect=children) as launches, \
             patch.object(controller.subprocess, 'run') as audit:
            with self.assertRaisesRegex(RuntimeError, 'all children drained'):
                controller.run(self.campaign, self.journal, self.assessment)
        self.assertEqual(launches.call_count, 12)
        self.assertTrue(all(child.waits == 1 for child in children))
        audit.assert_not_called()
        status = controller.read(self.campaign/'status.json')
        self.assertFalse(status['complete'])
        self.assertFalse(status['running'])
        self.assertEqual(len(status['jobs']), 12)
        self.assertEqual(status['jobs'][3]['exit_code'], 7)
        self.assertEqual(controller.read(self.journal/'status.json')['phase'], 'physical_failed')

    def test_launch_exception_drains_started_children_without_retries(self):
        self.freeze()
        children = [FakeChild(i) for i in range(3)]
        with patch.object(controller.subprocess, 'Popen', side_effect=children+[OSError('synthetic launch failure')]) as launches, \
             patch.object(controller.subprocess, 'run') as audit:
            with self.assertRaisesRegex(OSError, 'synthetic launch failure'):
                controller.run(self.campaign, self.journal, self.assessment)
        self.assertEqual(launches.call_count, 4)
        self.assertTrue(all(child.waits == 1 for child in children))
        audit.assert_not_called()
        status = controller.read(self.campaign/'status.json')
        self.assertEqual(status['jobs'][3]['status'], 'launch_failed')
        self.assertTrue(all(job['status'] == 'not_started' for job in status['jobs'][4:]))
        self.assertFalse(status['running'])

    def test_observer_failure_preserves_successful_physical_terminal_status(self):
        self.freeze()
        children = [FakeChild(i) for i in range(12)]
        with patch.object(controller.subprocess, 'Popen', side_effect=children), \
             patch.object(controller.subprocess, 'run', return_value=subprocess.CompletedProcess(['observer'], 9)):
            with self.assertRaises(subprocess.CalledProcessError):
                controller.run(self.campaign, self.journal, self.assessment)
        self.assertTrue(controller.read(self.campaign/'status.json')['complete'])
        state = controller.read(self.journal/'status.json')
        self.assertEqual(state['phase'], 'audit_failed')
        self.assertEqual(state['analysis_exit_code'], 9)
        self.assertFalse(state['complete'])

    def test_observer_duplicate_run_identity_cannot_mark_campaign_complete(self):
        manifest = self.freeze()
        children = [FakeChild(i) for i in range(12)]

        def observer(command, **_):
            self.assessment.mkdir()
            first = manifest['jobs'][0]
            row = dict({key: first[key] for key in ('id', 'mode', 'start', 'replicate', 'seed')}, passed=True)
            controller.write(self.assessment/'analysis.json', dict(complete=True, runs=[row]*12,
                manifest_sha256=controller.sha(self.campaign/'manifest.json'),
                terminal_status_sha256=controller.sha(self.campaign/'status.json'), analyzer_sha256=manifest['observer_sha256']))
            return subprocess.CompletedProcess(command, 0)

        with patch.object(controller.subprocess, 'Popen', side_effect=children), \
             patch.object(controller.subprocess, 'run', side_effect=observer):
            with self.assertRaisesRegex(ValueError, 'Observer result identities'):
                controller.run(self.campaign, self.journal, self.assessment)
        self.assertTrue(controller.read(self.campaign/'status.json')['complete'])
        self.assertEqual(controller.read(self.journal/'status.json')['phase'], 'audit_failed')

    def test_exclusive_status_reservation_blocks_a_racing_second_controller(self):
        manifest = self.freeze()
        status_path = self.campaign/'status.json'
        status_path.write_text('existing first controller reservation\n')
        with patch.object(controller, 'check', return_value=manifest), \
             patch.object(controller.subprocess, 'Popen') as launch:
            with self.assertRaises(FileExistsError):
                controller.run(self.campaign, self.journal, self.assessment)
        launch.assert_not_called()
        self.assertEqual(status_path.read_text(), 'existing first controller reservation\n')


if __name__ == '__main__':
    unittest.main()
