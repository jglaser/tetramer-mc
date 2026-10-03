"""Synthetic selected-row allocation, binding and gate checks; no protein queries."""
import hashlib
import importlib.util
import json
from pathlib import Path
import signal
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
from contextlib import ExitStack

import numpy as np

import native_class_selected_geometry as selected
from test_native_class_line_physical_reference import fixture


def save(path, value):
    path.write_text(json.dumps(value, allow_nan=False)+'\n')
    return dict(path=str(path), sha256=selected.sha(path))


def streams(base, count=24):
    rows = [dict(draw=i, z=float(i), regions=['native', 'native_remainder']) for i in range(count)]
    rows[22]['z'] = rows[23]['z'] = 100.
    paths = [base/name for name in ('samples.jsonl', 'attempts.jsonl', 'labels.jsonl')]
    for path, values in zip(paths, [rows, [dict(draw=i, state='begin') for i in range(count)],
                                   [dict(draw=i) for i in range(count)]]):
        path.write_text(''.join(json.dumps(v)+'\n' for v in values))
    context = dict(root=base, manifest=dict(samples=count), region={}, label_path=paths[2],
        summary=dict(samples_sha256=selected.sha(paths[0]), attempts_sha256=selected.sha(paths[1])),
        labels=dict(labels=dict(sha256=selected.sha(paths[2]))),
        declaration=dict(draw_ids=list(range(16))), target_and_regions_sha256='a'*64)
    plan = dict(population=dict(samples=count), limits=dict(max_record_bytes=10000, max_rows=count),
                preselection={}, algebra={}, labels={})
    return rows, context, plan


def select_mocked(context, plan):
    with mock.patch.object(selected, 'validate_row', side_effect=lambda row, **_: dict(z=row['z'])), \
         mock.patch.object(selected.statistics, 'label_regions', side_effect=lambda label, row, *args: row['regions']):
        return selected._select(context, plan)


class SelectedFullGeometry(unittest.TestCase):
    def test_unconditional_ids_are_deterministic_distinct_and_do_not_use_scientific_rng(self):
        with mock.patch.object(selected.full.line, 'Reconstructor', side_effect=AssertionError('No geometry')):
            a = selected.preselection('primary-r00', 128, 123)
            self.assertEqual(a, selected.preselection('primary-r00', 128, 123))
            self.assertNotEqual(a['draw_ids'], selected.preselection('primary-r00', 128, 124)['draw_ids'])
        self.assertEqual(len(set(a['draw_ids'])), 16)
        self.assertEqual(a['draw_ids'], sorted(a['draw_ids']))
        self.assertTrue(all(0 <= i < 128 for i in a['draw_ids']))
        for n in [0, 15, True]:
            with self.assertRaises(ValueError): selected.preselection('p', n, 0)

    def test_stream_selection_keeps_original_ids_ties_and_empty_regions_without_replacement(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows, context, plan = streams(Path(tmp))
            with mock.patch.object(selected.full.line, 'Reconstructor', side_effect=AssertionError('No geometry')):
                value = select_mocked(context, plan)
            self.assertEqual(value['unconditional_denominator'], 24)
            self.assertEqual(value['full_geometry_rows'], 17)
            self.assertEqual([r['draw'] for r in value['rows']], list(range(16))+[22])
            self.assertEqual(value['rows'][-1]['reasons'], ['largest_Qz:native', 'largest_Qz:native_remainder'])
            self.assertIsNone(value['decision_maxima']['competing'])
            self.assertIsNone(value['decision_maxima']['native_old_r5'])
            self.assertEqual(selected.read_bound_record(value['rows'][-1]['sample_record'], 10000), rows[22])

    def test_four_distinct_region_maxima_cap_at_twenty_and_deduplicate_unconditional(self):
        maxima = {key: dict(draw=16+i) for i, key in enumerate(selected.statistics.DECISIONS)}
        self.assertEqual(len(selected.selected_inventory(list(range(16)), maxima)), 20)
        maxima['native']['draw'] = 0
        self.assertEqual(len(selected.selected_inventory(list(range(16)), maxima)), 19)
        with self.assertRaises(ValueError): selected.selected_inventory([0]*16, maxima)

    def test_stream_rejects_missing_reordered_extra_and_changed_hash_records(self):
        for mode in ('missing', 'reordered', 'extra', 'hash'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                rows, context, plan = streams(Path(tmp))
                path = Path(tmp)/'attempts.jsonl'; raw = path.read_bytes().splitlines(keepends=True)
                if mode == 'missing': raw.pop()
                if mode == 'reordered': raw[0], raw[1] = raw[1], raw[0]
                if mode == 'extra': raw.append(raw[-1])
                if mode == 'hash': raw[0] = b'{"draw": 0, "state": "begin"}  \n'
                path.write_bytes(b''.join(raw))
                with self.assertRaises(ValueError): select_mocked(context, plan)

    def test_selected_full_row_hash_includes_lf_and_rejects_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, context, plan = streams(Path(tmp)); value = select_mocked(context, plan)
            record = value['rows'][0]['sample_record']; path = Path(record['path'])
            raw = path.read_bytes(); self.assertEqual(record['sha256'], hashlib.sha256(raw[:record['bytes']]).hexdigest())
            path.write_bytes(raw.replace(b'0.0', b'9.0', 1))
            with self.assertRaisesRegex(ValueError, 'bytes changed'): selected.read_bound_record(record, 10000)

    def test_selected_reference_reuses_full_density_and_draw_once_for_toy_geometry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, recon, rows, manifest, summary = fixture(Path(tmp), beta=.5)
            context = dict(manifest=manifest, region=selected.read(root/'provenance/region.json'))
            calls = []
            budget = SimpleNamespace(query=lambda role, draw, fn: (calls.append((role, draw)), fn())[1])
            # Saved toy row 0 contributes. Use its fully reconstructed native
            # and exclusion memberships as synthetic independent label inputs.
            row = rows[0]; density = recon.density(row['latent']); coordinate = recon.raw(row['latent'])[0]
            label = dict(hard_valid=row['hard_valid'], applicable=True,
                native=selected.full.line.contains(density['axes'][0]['native_intervals'], coordinate),
                exclusion_contact=selected.full.line.contains(density['axes'][0]['exclusion_contact_intervals'], coordinate))
            with mock.patch.object(selected.full, 'compact_density', wraps=selected.full.compact_density) as full_score, \
                 mock.patch.object(selected.full.line, 'audit_draw', wraps=selected.full.line.audit_draw) as draw:
                result = selected.audit_selected_row(row, label, context, recon, budget)
            self.assertEqual(full_score.call_count, 1); self.assertEqual(draw.call_count, 1)
            self.assertEqual(calls, [('full_geometry', row['draw'])]); self.assertEqual(result['axes'], 3)
            bad = dict(label, native=not label['native'])
            with self.assertRaisesRegex(ValueError, 'native membership'):
                selected.audit_selected_row(row, bad, context, recon, budget)

    def test_dynamically_loaded_frozen_observer_bridges_real_line_reference_without_setup(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp); root, original, rows, _, _ = fixture(base, beta=.5)
            local_class = selected.full.line.NativeContactRegions
            source = Path(sys.modules[local_class.__module__].__file__)
            frozen_path = base/'frozen_native_regions.py'; frozen_path.write_bytes(source.read_bytes())
            name = 'selected_test_frozen_native_regions'
            spec = importlib.util.spec_from_file_location(name, frozen_path)
            module = importlib.util.module_from_spec(spec); sys.modules[name] = module; spec.loader.exec_module(module)
            try:
                # The data are a tiny synthetic four-member fixture. The class
                # is loaded from actual frozen module bytes, not a local alias.
                frozen = object.__new__(module.NativeContactRegions)
                frozen.__dict__.update(original.native.observer.__dict__)
                compiled = selected.read(root/'provenance/compiled-native.json')
                compiled['source_input_sha256']['source/native_contact_regions.py'] = selected.sha(frozen_path)
                definition = dict(criteria=module.CRITERIA, input_sha256=compiled['source_input_sha256'])
                definition_path = base/'bridge-definition.json'; save(definition_path, definition)
                frozen.definition = definition; frozen.definition_sha256 = selected.sha(definition_path)
                compiled['source_definition_sha256'] = frozen.definition_sha256
                binding = dict(definition=str(definition_path), definition_sha256=frozen.definition_sha256,
                               runtime_sha256=selected.sha(frozen_path))
                with self.assertRaisesRegex(ValueError, 'Original native observer required'):
                    selected.full.line.NativeLineReference(frozen)
                bindings = selected.statistics.Bindings()
                with mock.patch.object(module.NativeContactRegions, '__init__', side_effect=AssertionError('No repeated setup')), \
                     mock.patch.object(local_class, '__init__', side_effect=AssertionError('No local setup')):
                    adapter, receipt = selected.frozen_line_observer(frozen, compiled, binding, bindings.bind)
                    rebuilt = selected.full.line.Reconstructor(original.region, original.guide, original.config,
                        original.shape, adapter, use_tree=False)
                    result = selected.full.compact_density(rebuilt, np.asarray(rows[0]['latent']), rows[0]['native_class_line_density'])
                self.assertEqual(result['log_density'], rows[0]['log_proposal_density'])
                self.assertIs(type(adapter).__mro__[1], module.NativeContactRegions)
                self.assertIs(adapter.atoms, frozen.atoms)
                self.assertIs(adapter.classify.__func__, module.NativeContactRegions.classify)
                self.assertEqual(receipt['additional_observer_setup_queries'], 0)
                self.assertEqual(receipt['frozen_runtime']['sha256'], selected.sha(frozen_path))
                with self.assertRaisesRegex(ValueError, 'Changed bound file'):
                    selected.frozen_line_observer(frozen, compiled, dict(binding, runtime_sha256='0'*64), bindings.bind)
                definition['criteria'] = dict(definition['criteria'], contact_entry_A=99.)
                save(definition_path, definition)
                with self.assertRaises(ValueError): selected.frozen_line_observer(frozen, compiled, binding, bindings.bind)
            finally:
                del sys.modules[name]

    def test_bad_plan_fails_before_any_geometry_and_keeps_failure_journal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = dict(output={name: str(root/(name+'.json')) for name in ('receipt', 'journal', 'failure')},
                limits=dict(cpu_seconds=30., wall_seconds=30., max_rows=16, max_record_bytes=10000,
                    max_full_geometry_queries=21, max_axis_queries=60,
                    **{f'max_{role}_queries': 16 for role in selected.endpoints.QUERY_ROLES}))
            path = root/'plan.json'; save(path, plan)
            before = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGALRM, signal.SIGPROF)}
            with mock.patch.object(selected.endpoints, '_load_context', side_effect=AssertionError('No setup')) as setup:
                with self.assertRaisesRegex(ValueError, 'query cap'):
                    selected.run(path, plan_sha256=selected.sha(path))
            self.assertEqual(setup.call_count, 0)
            failure = selected.read(root/'failure.json')
            self.assertFalse(failure['passed']); self.assertEqual(failure['completed_rows'], 0)
            self.assertFalse((root/'receipt.json').exists())
            self.assertEqual({sig: signal.getsignal(sig) for sig in before}, before)
            self.assertEqual(signal.getitimer(signal.ITIMER_REAL), (0., 0.))
            self.assertEqual(signal.getitimer(signal.ITIMER_PROF), (0., 0.))

    def test_worker_retains_begun_failed_row_and_never_replaces_or_resumes(self):
        for fail in (False, True):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); _, context, plan = streams(root)
                context.update(config={}, law=SimpleNamespace(axes=[0, 1, 2], beta=.5, alpha=.5))
                (root/'provenance').mkdir()
                for name in ('importance-guide.json', 'shape.json', 'compiled-native.json'): save(root/'provenance'/name, {})
                limits = dict(plan['limits'], cpu_seconds=30., wall_seconds=30.,
                    max_full_geometry_queries=20, max_axis_queries=60,
                    **{f'max_{role}_queries': 24 for role in selected.endpoints.QUERY_ROLES})
                plan.update(schema=selected.PLAN_SCHEMA, limits=limits,
                    runtime=selected.streaming.runtime_identity(),
                    source_sha256={Path(selected.__file__).name: selected.sha(selected.__file__)},
                    output={name: str(root/(name+'.json')) for name in ('receipt', 'journal', 'failure')})
                inventory = select_mocked(context, plan)
                plan['selection'] = save(root/'selection.json', inventory)
                plan['review'] = save(root/'review.json', dict(complete=True, passed=True,
                    preselection=plan['preselection'], selection=plan['selection'], population=plan['population'],
                    limits=limits, predeclared_before_first_draw=True,
                    no_previous_full_geometry_audit_of_population=True))
                plan_path = root/'plan.json'; save(plan_path, plan)
                def metadata(frozen, bindings):
                    for name in ('samples.jsonl', 'attempts.jsonl', 'labels.jsonl'):
                        bindings.bind(root/name)
                    return context
                def audit(row, label, ctx, recon, budget):
                    def full_query():
                        for axis in range(3):
                            recon.reconstruct_axis([], axis)
                            if fail and row['draw'] == 1: raise ValueError('synthetic full reference failure')
                        return dict(draw=row['draw'], axes=3)
                    return budget.query('full_geometry', row['draw'], full_query)
                with ExitStack() as stack:
                    stack.enter_context(mock.patch.object(selected, '_metadata', side_effect=metadata))
                    stack.enter_context(mock.patch.object(selected, 'validate_row', side_effect=lambda row, **_: dict(z=row['z'])))
                    stack.enter_context(mock.patch.object(selected.statistics, 'label_regions', side_effect=lambda label, row, *args: row['regions']))
                    stack.enter_context(mock.patch.object(selected, 'local_sources', return_value={Path(selected.__file__).name: Path(selected.__file__).resolve()}))
                    setup = stack.enter_context(mock.patch.object(selected.endpoints, '_load_context', return_value=dict(
                        target_and_regions_sha256=context['target_and_regions_sha256'], observer=None, classifier_binding={})))
                    stack.enter_context(mock.patch.object(selected, 'frozen_line_observer', return_value=(None, {})))
                    stack.enter_context(mock.patch.object(selected.full.line, 'Reconstructor', return_value=SimpleNamespace(
                        reconstruct_axis=lambda u, axis, use_tree=None: dict(axis=axis))))
                    stack.enter_context(mock.patch.object(selected, 'audit_selected_row', side_effect=audit))
                    if fail:
                        with self.assertRaisesRegex(ValueError, 'synthetic full reference failure'):
                            selected.run(plan_path, plan_sha256=selected.sha(plan_path))
                        failure = selected.read(root/'failure.json')
                        self.assertEqual((failure['draw'], failure['completed_rows']), (1, 1))
                        self.assertEqual(failure['query_counts'], dict(full_geometry=2, axis=4))
                        self.assertFalse((root/'receipt.json').exists())
                    else:
                        result = selected.run(plan_path, plan_sha256=selected.sha(plan_path))
                        self.assertEqual(result['full_geometry_selected_rows'], 17)
                        self.assertEqual(result['query_counts'], dict(full_geometry=17, axis=51))
                        self.assertFalse(result['all_row_full_geometry_certified'])
                        self.assertEqual(result['original_attempt_denominator'], 24)
                    self.assertEqual(setup.call_count, 1)
                    with self.assertRaisesRegex(ValueError, 'Fresh distinct outputs'):
                        selected.run(plan_path, plan_sha256=selected.sha(plan_path))
                    self.assertEqual(setup.call_count, 1)
                journal = [json.loads(row) for row in (root/'journal.json').read_text().splitlines()]
                if fail:
                    self.assertEqual([row['draw'] for row in journal if row['state'] == 'begin'], [0, 1])
                    self.assertEqual(journal[-1]['state'], 'failed')


if __name__ == '__main__': unittest.main()
