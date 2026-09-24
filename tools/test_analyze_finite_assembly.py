"""Authenticated synthetic trajectory adapter tests; no protein MC or audits.

Only the synthetic native-definition reader and its input inventory are mocked.
Run hashes, runtime source closure, retained frames, configurations, and output
freezes use the real implementation.
"""
import copy
from contextlib import contextmanager
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import analyze_finite_assembly as adapter
import contact_benchmark_contract as benchmark
from test_analyze_contact_efficiency import file_fixture
from test_finite_assembly_observer import LINE_MOTIFS, pose


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False)+'\n')


def read(path):
    return json.loads(Path(path).read_text())


def rebind_declared(run):
    digest = adapter.sha(run/'provenance/input-config.json')
    for name in ('manifest.json', 'summary.json', 'checkpoint.json'):
        content = read(run/name)
        content['config_sha256'] = digest
        save(run/name, content)


def rebind_definition(definition, plan):
    content = read(plan)
    content['definition_sha256'] = adapter.sha(definition)
    save(plan, content)


def bind_local_benchmark(data):
    """Declare a three-arm comparison and bind this single synthetic local run.

    The other streams are deliberately absent: one valid report must not imply
    that a complete matched comparison or convergence has been established.
    """
    for name in ('config.json', 'provenance/input-config.json'):
        config = read(data.run/name)
        config['global_probability'] = 0.
        save(data.run/name, config)
    rebind_declared(data.run)
    config = read(data.run/'config.json')
    definition = read(data.definition)
    plan = read(data.plan)
    weight = config['learned_uniform_weight']
    contract = dict(schema=benchmark.SCHEMA,
        physical_identity_sha256=adapter.digest(definition['physical_identity']),
        observer_definition_sha256=adapter.digest(definition),
        region_definition_sha256=adapter.digest(definition['regions']),
        common_schedule={key: config[key] for key in benchmark.COMMON},
        window=dict(burn_sweep=0, end_sweep=8, cadence_sweeps=1),
        preparations=[plan['preparation_id']], streams_per_preparation=4,
        single_body_attempt_budget=benchmark.ATTEMPTS,
        arms={plan['proposal_arm']: dict(role='local', global_probability=0., method='local-uniform',
                model_sha256=None, frozen_posterior=None, learned_uniform_weight=weight),
              'synthetic-redraw': dict(role='independent-redraw', global_probability=.5, method='learned',
                model_sha256='b'*64, frozen_posterior=None, learned_uniform_weight=weight),
              'synthetic-transport': dict(role='correlated-transport', global_probability=.5, method='learned',
                model_sha256='b'*64, frozen_posterior=dict(probability=.9, correlation=.9), learned_uniform_weight=weight)})
    path = data.root/'benchmark.json'
    save(path, contract)
    plan['benchmark_contract'] = dict(path=path.name, sha256=adapter.sha(path))
    save(data.plan, plan)
    return path, contract


@contextmanager
def synthetic_run(periodic=False, native_pairs=False, during_classification=None):
    from prepare_finite_assembly_observer import REGION_RULES

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        run, _ = file_fixture(root)
        config = read(run/'config.json')
        if periodic:
            initial = [pose([0., 0., 0.]), pose([4., 0., 0.]), pose([8., 0., 0.])]
            update = dict(boundary=dict(kind='periodic'), box_lengths=[12., 12., 12.],
                          depletant_radius=2., initial_poses=initial)
            config.update(update, uniform_proposal_cube_lengths=[12., 12., 12.])
            save(run/'config.json', config)
            declared = read(run/'provenance/input-config.json'); declared.update(update)
            save(run/'provenance/input-config.json', declared)
            manifest = read(run/'manifest.json'); manifest['boundary'] = update['boundary']
            save(run/'manifest.json', manifest)
            frames = [json.loads(line) for line in (run/'trajectory.jsonl').read_text().splitlines()]
            for frame in frames:
                frame.update(poses=initial, boundary='periodic')
                frame.pop('spherical_wall_radius', None)
            (run/'trajectory.jsonl').write_text(''.join(json.dumps(f)+'\n' for f in frames))
            checkpoint = read(run/'checkpoint.json'); checkpoint['poses'] = initial
            save(run/'checkpoint.json', checkpoint)
            rebind_declared(run)

        definition_dir = root/'definition'
        definition_dir.mkdir()
        sources = {}
        for name in adapter.RUNTIME_SOURCES+('prepare_finite_assembly_observer.py',):
            source = Path(adapter.__file__).with_name(name)
            archived = definition_dir/'source'/name
            archived.parent.mkdir(exist_ok=True)
            shutil.copy2(source, archived)
            sources['source/'+name] = adapter.sha(archived)
        shape_sha = adapter.sha(run/'provenance/shape.json')
        save(definition_dir/'patch.json', dict(schema='body-frame-atom-patch-map-v1',
            shape_sha256=shape_sha, atom_patch_ids=['surface']))
        save(definition_dir/'native.json', dict(synthetic=True, shape_sha256=shape_sha))
        identity = adapter.physical_identity(config, shape_sha)
        definition = definition_dir/'definition.json'
        save(definition, dict(schema='finite-assembly-observer-definition-v1', reference_size=8,
            regions=copy.deepcopy(REGION_RULES), physical_identity=identity,
            physical_identity_sha256=adapter.digest(identity), source_sha256=sources,
            patch_map=dict(path='patch.json', sha256=adapter.sha(definition_dir/'patch.json')),
            native_definition=dict(path='native.json', sha256=adapter.sha(definition_dir/'native.json'))))
        plan = root/'window.json'
        save(plan, dict(schema=adapter.PLAN_SCHEMA, definition_sha256=adapter.sha(definition),
            burn_sweep=0, end_sweep=8, preparation_id='synthetic-dispersed', proposal_arm='synthetic-local'))
        calls = []

        def classify_pair(anchor, moving):
            calls.append((anchor, moving))
            if during_classification is not None and len(calls) == 1:
                during_classification(run, definition_dir)
            return [dict(motif_id=0)] if native_pairs else []

        native = SimpleNamespace(shape_sha256=shape_sha, motifs=LINE_MOTIFS,
                                 classify_pair=classify_pair)
        with patch.object(adapter, 'NativeContactRegions', return_value=native), \
             patch.object(adapter, 'native_input_bindings', side_effect=lambda path: {str(path): adapter.sha(path)}):
            yield SimpleNamespace(root=root, run=run, definition=definition, plan=plan,
                                  out=root/'analysis', calls=calls)


class FiniteAssemblyAdapterTests(unittest.TestCase):
    def test_actual_report_matches_bound_benchmark_without_claiming_complete_allocation(self):
        with synthetic_run() as data:
            path, contract = bind_local_benchmark(data)
            result = adapter.observe(data.run, data.definition, data.plan, data.out)
            self.assertIsNone(benchmark.validate_report(result, contract))
            self.assertEqual(result['window_attempts'], dict(local=24, **{'global': 0}))
            self.assertEqual(result['benchmark_contract']['sha256'], adapter.sha(path))
            self.assertEqual(result['benchmark_contract']['content_sha256'], adapter.digest(contract))
            self.assertEqual(result['dependency_sha256'][str(path)], adapter.sha(path))
            self.assertEqual(result['schedule']['global_probability'], 0.)
            with self.assertRaisesRegex(ValueError, 'Incomplete or enlarged'):
                benchmark.validate_comparison([result])

    def test_changed_benchmark_schedule_law_hash_or_missing_input_fails_before_output(self):
        for mode in ('schedule', 'arm-law', 'stale-hash', 'missing-file'):
            with self.subTest(mode=mode), synthetic_run() as data:
                path, contract = bind_local_benchmark(data)
                if mode == 'missing-file':
                    path.unlink()
                else:
                    if mode == 'arm-law':
                        plan = read(data.plan); plan['proposal_arm'] = 'synthetic-redraw'; save(data.plan, plan)
                    else:
                        contract['common_schedule']['local_translation_std_A'] *= 2.
                        save(path, contract)
                        if mode != 'stale-hash':
                            plan = read(data.plan); plan['benchmark_contract']['sha256'] = adapter.sha(path); save(data.plan, plan)
                error = FileNotFoundError if mode == 'missing-file' else ValueError
                expected = {'schedule': 'common local/collective', 'arm-law': 'Actual global kernel',
                            'stale-hash': 'Frozen analysis input changed', 'missing-file': 'benchmark.json'}[mode]
                with self.assertRaisesRegex(error, expected):
                    adapter.observe(data.run, data.definition, data.plan, data.out)
                self.assertFalse(data.out.exists())

    def test_authenticated_retained_run_outputs_every_frame_and_freezes_results(self):
        with synthetic_run() as data:
            result = adapter.observe(data.run, data.definition, data.plan, data.out)
            self.assertTrue(result['complete'])
            self.assertEqual(result['fingerprint']['samples'], 8)
            self.assertEqual(result['fingerprint']['sampling_CPU_seconds'], 4.)
            rows = [json.loads(line) for line in (data.out/'observations.jsonl').read_text().splitlines()]
            self.assertEqual([row['sweep'] for row in rows], list(range(9)))
            self.assertEqual(len(data.calls), 27)
            self.assertEqual(result['environment_occupancies']['contact_no_entry']['fraction'], 1.)
            self.assertEqual(result['labels_by_frame'], [row['environment'] for row in rows])
            self.assertEqual(result['initialization']['preparation_id'], 'synthetic-dispersed')
            self.assertEqual(result['initialization']['proposal_arm'], 'synthetic-local')
            self.assertEqual(result['plan_sha256'], adapter.sha(data.plan))
            self.assertEqual(set(result['implementation_sha256']),
                             set(adapter.RUNTIME_SOURCES+('prepare_finite_assembly_observer.py',)))
            self.assertIsNone(result['physical_reference'])
            self.assertFalse(result['frozen_bias_supported'])
            freeze = read(data.out/'freeze.json')
            self.assertEqual(set(freeze['files']), {'analysis.json', 'observations.jsonl'})
            for name, expected in freeze['files'].items():
                self.assertEqual(adapter.sha(data.out/name), expected)
            with self.assertRaisesRegex(ValueError, 'Fresh analysis'):
                adapter.observe(data.run, data.definition, data.plan, data.out)

    def test_missing_reordered_and_duplicate_saved_frames_fail_audit(self):
        for mode in ('missing', 'reordered', 'duplicate'):
            with self.subTest(mode=mode), synthetic_run() as data:
                path = data.run/'trajectory.jsonl'
                lines = path.read_text().splitlines()
                if mode == 'missing': lines.pop(3)
                elif mode == 'reordered': lines[2], lines[3] = lines[3], lines[2]
                else: lines.insert(3, lines[3])
                path.write_text('\n'.join(lines)+'\n')
                with self.assertRaisesRegex(ValueError, 'saved endpoints'):
                    adapter.observe(data.run, data.definition, data.plan, data.out)
                self.assertFalse(data.out.exists())
                self.assertEqual(data.calls, [])

    def test_stale_definition_hash_and_changed_region_law_fail(self):
        for stale_hash in (True, False):
            with self.subTest(stale_hash=stale_hash), synthetic_run() as data:
                definition = read(data.definition)
                definition['regions'] = []
                save(data.definition, definition)
                if not stale_hash:
                    rebind_definition(data.definition, data.plan)
                with self.assertRaisesRegex(ValueError, 'bind the observer' if stale_hash else 'Region definitions'):
                    adapter.observe(data.run, data.definition, data.plan, data.out)
                self.assertFalse(data.out.exists())

    def test_archived_source_and_runtime_source_closure_are_checked_separately(self):
        for mode in ('archive-change', 'runtime-mismatch', 'missing-source'):
            with self.subTest(mode=mode), synthetic_run() as data:
                definition = read(data.definition)
                relative = 'source/finite_assembly_observer.py'
                path = data.definition.parent/relative
                if mode == 'missing-source':
                    del definition['source_sha256'][relative]
                else:
                    path.write_text(path.read_text()+'\n# Synthetic tamper control.\n')
                    if mode == 'runtime-mismatch':
                        definition['source_sha256'][relative] = adapter.sha(path)
                save(data.definition, definition)
                rebind_definition(data.definition, data.plan)
                expected = {'archive-change': 'Frozen observer source changed',
                            'runtime-mismatch': 'Runtime observer differs',
                            'missing-source': 'Incomplete observer implementation'}[mode]
                with self.assertRaisesRegex(ValueError, expected):
                    adapter.observe(data.run, data.definition, data.plan, data.out)
                self.assertFalse(data.out.exists())

    def test_changes_during_observation_are_detected_before_outputs_are_written(self):
        for mode in ('run', 'native'):
            def change(run, definition_dir):
                target = run/'trajectory.jsonl' if mode == 'run' else definition_dir/'native.json'
                with target.open('a') as stream:
                    stream.write('\n')
            with self.subTest(mode=mode), synthetic_run(during_classification=change) as data:
                with self.assertRaisesRegex(ValueError, 'changed during observation'):
                    adapter.observe(data.run, data.definition, data.plan, data.out)
                self.assertGreater(len(data.calls), 0)
                self.assertFalse(data.out.exists())

    def test_physical_identity_mismatch_is_rejected_even_after_definition_rebinding(self):
        with synthetic_run() as data:
            definition = read(data.definition)
            definition['physical_identity']['reservoir_density'] = .04
            definition['physical_identity_sha256'] = adapter.digest(definition['physical_identity'])
            save(data.definition, definition); rebind_definition(data.definition, data.plan)
            with self.assertRaisesRegex(ValueError, 'Different physical target'):
                adapter.observe(data.run, data.definition, data.plan, data.out)
            self.assertEqual(data.calls, [])

    def test_winding_frames_are_counted_in_remaining_and_written_without_filtering(self):
        with synthetic_run(periodic=True, native_pairs=True) as data:
            result = adapter.observe(data.run, data.definition, data.plan, data.out)
            self.assertEqual(result['fingerprint']['samples'], 8)
            self.assertEqual(result['environment_occupancies']['remaining']['fraction'], 1.)
            self.assertIsNone(result['fingerprint']['apparent_ess'])
            self.assertEqual(result['size_pair_occupancies'], [dict(largest_exclusion=3,
                largest_native_raw=3, largest_native_certified=1, registry_resolved=False,
                observations=8, fraction=1.)])
            rows = [json.loads(line) for line in (data.out/'observations.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows), 9)
            self.assertTrue(all(row['native_periodic_winding'] for row in rows))
            self.assertTrue(all(len(row['instantaneous_native_keys']) == 3 for row in rows))

    def test_frozen_bias_requires_separate_physical_occupancy_path(self):
        with synthetic_run() as data:
            bias = dict(schema='synthetic-frozen-bias', values=[0., 1.])
            for name in ('config.json', 'manifest.json', 'summary.json', 'provenance/input-config.json'):
                value = read(data.run/name); value['assembly_bias'] = bias
                save(data.run/name, value)
            rebind_declared(data.run)
            with self.assertRaisesRegex(ValueError, 'unbiased proposal'):
                adapter.observe(data.run, data.definition, data.plan, data.out)
            self.assertEqual(data.calls, [])
            self.assertFalse(data.out.exists())


if __name__ == '__main__':
    unittest.main()
