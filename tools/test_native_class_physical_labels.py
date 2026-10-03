"""Synthetic endpoint and streaming contracts only; no protein geometry/draws."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

import native_class_physical_labels as labels
import analyze_native_class_physical_populations as consumer
from native_class_line_reference import observer_from_compiled_for_synthetic
from test_native_class_line_physical_reference import fixture, save


def rows_write(path, rows):
    path.write_text(''.join(json.dumps(row, allow_nan=False)+'\n' for row in rows))


def synthetic(base):
    root, law, rows, manifest, summary = fixture(base)
    shape = labels.read(root/'provenance/shape.json')
    config = labels.read(root/'provenance/input-config.json')
    region = labels.read(root/'provenance/region.json')
    reference = copy.deepcopy(region)
    reference.update(mahalanobis_radius=5., minimum_original_q=1., minimum_original_q_inclusive=False)
    config['metadata']['physical_sphere_radius_A'] = 20.
    region['physical_metric'] = config['metadata']
    old = base/'old-r5.json'; save(old, reference)
    strata = base/'strata.json'; save(strata, labels.STRATA)
    definition = root/'provenance/compiled-native.json'
    context = dict(manifest=manifest, summary=summary, region=region, config=config, law=law,
        reference=reference, strata=labels.STRATA,
        observer=observer_from_compiled_for_synthetic(labels.read(definition)),
        core=labels.EndpointCore(shape, config['fixed_poses']),
        contact=labels.ExclusionContact(shape, config['fixed_poses'], config['depletant_radius']),
        classifier_binding=dict(synthetic=True), shape_witness=dict(synthetic=True),
        wall_certificate=labels.wall_certificate(shape, config), target_and_regions_sha256='f'*64,
        observer_setup_inventory=dict(synthetic=True))
    context['observer'].fixed_body_ids = [2]
    context['observer'].fixed_scaffold_matches = []
    output = base/'labels-output'; output.mkdir()
    plan = dict(schema=labels.PLAN_SCHEMA,
        population=dict(root=str(root), seed=manifest['seed'], samples=len(rows), **{
            name+'_sha256': labels.sha(root/(name+('.jsonl' if name in ('samples', 'attempts') else '.json')))
            for name in ('manifest', 'summary', 'samples', 'attempts')}),
        reference_region=dict(path=str(old), sha256=labels.sha(old)),
        definition=dict(path=str(definition), sha256=labels.sha(definition)),
        strata=dict(path=str(strata), sha256=labels.sha(strata)),
        output={key: str(output/name) for key, name in [('receipt', 'receipt.json'), ('labels', 'labels.jsonl'),
                                                      ('journal', 'journal.jsonl'), ('failure', 'failure.json')]},
        limits=dict(cpu_seconds=60., wall_seconds=90., max_rows=len(rows), max_record_bytes=200000,
                    **{f'max_{role}_queries': len(rows) for role in labels.QUERY_ROLES}),
        source_sha256={Path(labels.__file__).name: labels.sha(labels.__file__)})
    path = base/'label-plan.json'; save(path, plan)
    return root, rows, context, plan, path


def execute(context, plan, path):
    def setup(root, frozen, bind, emit, budget):
        # Tiny fixture authorities replace production-only native/protein pins.
        for ref in (frozen['reference_region'], frozen['definition'], frozen['strata']):
            bind(ref['path'], ref['sha256'])
        bind(root/'provenance/region.json')
        return context
    with mock.patch.object(labels, '_load_context', side_effect=setup), \
         mock.patch.object(labels, 'local_sources', return_value={Path(labels.__file__).name: Path(labels.__file__).resolve()}):
        return labels.run(path, plan_sha256=labels.sha(path))


class PhysicalEndpointLabels(unittest.TestCase):
    def test_strict_core_variable_radii_tangency_and_capture_wall_certificate(self):
        pose = lambda x: dict(position=[x, 0., 0.], orientation=[1., 0., 0., 0.])
        shape = dict(atoms=[dict(center=[0., 0., 0.], radius=1.), dict(center=[0., 4., 0.], radius=.3)])
        with mock.patch.object(labels.EndpointReference, '__init__', side_effect=AssertionError('No line initializer')):
            core = labels.EndpointCore(shape, [pose(0.)])
        self.assertTrue(core.hard_valid(np.array([2., 0., 0.]), np.eye(3)))
        self.assertFalse(core.hard_valid(np.array([np.nextafter(2., 0.), 0., 0.]), np.eye(3)))
        self.assertTrue(core.hard_valid(np.array([np.nextafter(2., math.inf), 0., 0.]), np.eye(3)))
        config = dict(capture_center=[0., 0., 0.], capture_radius=10., fixed_poses=[pose(0.)],
                      metadata=dict(physical_sphere_radius_A=20.))
        self.assertGreater(labels.wall_certificate(shape, config)['guaranteed_atomic_wall_clearance_A'], 0.)
        config['metadata']['physical_sphere_radius_A'] = 14.3
        with self.assertRaisesRegex(ValueError, 'does not certify'): labels.wall_certificate(shape, config)

    def test_full_stream_checks_invalid_and_exterior_zeros_and_consumer_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root, rows, context, plan, path = synthetic(Path(directory))
            with mock.patch.object(context['core'], 'hard_valid', wraps=context['core'].hard_valid) as hard, \
                 mock.patch.object(context['observer'], 'classify', wraps=context['observer'].classify) as native:
                result = execute(context, plan, path)
            self.assertEqual(hard.call_count, 3)
            self.assertEqual(native.call_count, 1)
            self.assertEqual(result['query_counts'], dict(capture=3, atomic=3, native=1, contact=1))
            records = [json.loads(line) for line in Path(plan['output']['labels']).read_text().splitlines()]
            self.assertEqual([r['draw'] for r in records], [0, 1, 2])
            self.assertEqual([r['applicable'] for r in records], [True, False, False])
            self.assertIn('matches', records[0]['classification'])
            self.assertIn('anchors', records[0]['contact'])
            self.assertEqual(records[1]['strata']['orthant'], 63)
            self.assertEqual(records[2]['strata'], dict(radial=-1, angular=-1, orthant=-1))
            raw = (root/'samples.jsonl').read_bytes().splitlines(keepends=True)
            for index, record in enumerate(records):
                self.assertEqual(record['sample_record_sha256'], hashlib.sha256(raw[index]).hexdigest())
                accounting = labels.validate_row(rows[index], expected_draw=index,
                    manifest=context['manifest'], region=context['region'])
                selected = consumer.label_regions(record, rows[index], accounting, record['sample_record_sha256'])
                self.assertEqual(bool(selected), index == 0)
            self.assertEqual(result['strata_sha256'], consumer.fingerprint(consumer.STRATA))
            journal = [json.loads(line) for line in Path(plan['output']['journal']).read_text().splitlines()]
            beginnings = [(r['role'], r['draw']) for r in journal if r['state'] == 'query_begin']
            self.assertIn(('atomic', 1), beginnings)
            self.assertIn(('atomic', 2), beginnings)
            self.assertFalse(Path(plan['output']['failure']).exists())

    def test_batched_endpoint_matches_existing_reference_and_bruteforce(self):
        pose = lambda p: dict(position=p, orientation=[1., 0., 0., 0.])
        # Unequal-radius dumbbell, two differently located fixed neighbors.
        shape = dict(atoms=[dict(center=[-2., 0., 0.], radius=.2),
                            dict(center=[2., 0., 0.], radius=.6)])
        core = labels.EndpointCore(shape, [pose([0., 0., 0.]), pose([1., 3., 0.])])
        rotations = [np.eye(3), np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])]
        for rotation in rotations:
            for x in [-4., -2., 0., .4, 1.2, 2., 4., 6.]:
                for y in [0., .4, 1.2, 3., 5.]:
                    position = np.array([x, y, 0.])
                    moving = core.atoms @ rotation.T + position
                    brute = not any(np.any(np.linalg.norm(fixed[:, None, :]-moving[None, :, :], axis=2)
                        < core.radii[:, None]+core.radii[None, :]) for fixed in core.fixed_world)
                    self.assertEqual(core.hard_valid(position, rotation), brute)
                    self.assertEqual(core.hard_valid(position, rotation),
                                     labels.EndpointReference.hard_valid(core, position, rotation))

    def test_overlap_only_in_second_or_third_atom_batch(self):
        fixed = [dict(position=[0., 0., 0.], orientation=[1., 0., 0., 0.])]
        for count in (65, 129):
            shape = dict(atoms=[dict(center=[100.*i, 0., 0.], radius=.6 if i == count-1 else .1)
                                for i in range(count)])
            core = labels.EndpointCore(shape, fixed)
            position = np.array([0., 0., 1.])
            self.assertFalse(core.hard_valid(position, np.eye(3)))
            self.assertFalse(labels.EndpointReference.hard_valid(core, position, np.eye(3)))
            prefix = labels.EndpointCore(dict(atoms=shape['atoms'][:-1]), fixed)
            self.assertTrue(prefix.hard_valid(position, np.eye(3)))

    def test_frozen_dynamic_constructor_setup_queries_are_bounded_and_restored(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory); source = base/'inputs/source/native_contact_regions.py'
            source.parent.mkdir(parents=True)
            source.write_text('''import json
from pathlib import Path
CRITERIA = {'contact_entry_A': 2., 'native_reference_patch_gap_A': 1.}
IDENTITY = [[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]
class NativeContactRegions:
    def __init__(self, path):
        self.data = json.loads(Path(path).read_text())
        for _ in range(self.data['reference_repeats']):
            self._contacts([0.,0.,0.], IDENTITY, 1.)
        self.classify_pair(*self.data['fixed_poses'])
    def _contacts(self, position, rotation, cutoff):
        if self.data['fail_contact']: raise ValueError('synthetic setup contact failure')
        return {'synthetic': True}
    def classify_pair(self, anchor, moving):
        for _ in range(self.data['scaffold_repeats']):
            self._contacts([0.,0.,0.], IDENTITY, 2.)
        return []
''')
            fixed = [dict(position=[0., 0., 0.], orientation=[1., 0., 0., 0.]),
                     dict(position=[3., 0., 0.], orientation=[1., 0., 0., 0.])]
            definition = base/'definition.json'
            data = dict(input_sha256={'source/native_contact_regions.py': labels.sha(source)},
                        criteria=dict(contact_entry_A=2., native_reference_patch_gap_A=1.), scope='synthetic',
                        fixed_poses=fixed, reference_repeats=1, scaffold_repeats=1, fail_contact=False)
            inventory = dict(reference_contact_calls=1, fixed_scaffold_classifier_calls=1,
                reference_contact_queries=[dict(label='synthetic', position=[0., 0., 0.], rotation=np.eye(3).tolist())],
                fixed_scaffold_poses=fixed, maximum_scaffold_contact_calls=1, maximum_total_contact_calls=2)
            limits = dict(cpu_seconds=60., wall_seconds=90., max_rows=1, max_record_bytes=10000,
                          **{f'max_{role}_queries': 1 for role in labels.QUERY_ROLES})
            def bind(path, expected):
                self.assertEqual(labels.sha(path), expected)
                return path
            for mode in ('success', 'extra_reference', 'extra_scaffold', 'query_failure'):
                data.update(reference_repeats=2 if mode == 'extra_reference' else 1,
                            scaffold_repeats=2 if mode == 'extra_scaffold' else 1,
                            fail_contact=mode == 'query_failure')
                save(definition, data); events = []; budget = labels.Budget(limits, events.append)
                if mode == 'success':
                    observer, _ = labels.load_bounded_classifier(definition, inventory, budget, bind)
                    labels.check_setup_counts(inventory, budget.setup_calls)
                    before = dict(budget.setup_calls)
                    observer._contacts([0., 0., 0.], np.eye(3), 2.)
                    self.assertEqual(before, dict(budget.setup_calls))
                    self.assertEqual(sum(v for k, v in before.items() if k.endswith('_started')), 3)
                else:
                    with self.assertRaises(ValueError):
                        labels.load_bounded_classifier(definition, inventory, budget, bind)
                    if mode == 'query_failure':
                        self.assertEqual(events[-1]['state'], 'setup_query_begin')
                        self.assertEqual(budget.setup_calls['reference_contacts_started'], 1)
                        self.assertEqual(budget.setup_calls['reference_contacts_completed'], 0)
                # The wrappers are removed from the actual dynamically loaded
                # class even after a constructor failure, not a different copy.
                cls = labels.sys.modules['bounded_frozen_native_regions'].NativeContactRegions
                self.assertEqual(cls._contacts.__name__, '_contacts')
                self.assertEqual(cls.classify_pair.__name__, 'classify_pair')

    def test_false_saved_hard_zero_fails_without_native_call_or_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            _, rows, context, _, _ = synthetic(Path(directory))
            row = copy.deepcopy(rows[0]); row.update(hard_valid=False, clouds=[], log_hard_weight=None, log_importance_weight=None)
            events = []; budget = labels.Budget(dict(cpu_seconds=60., wall_seconds=90., max_rows=3,
                max_record_bytes=200000, **{f'max_{r}_queries': 3 for r in labels.QUERY_ROLES}), events.append)
            with mock.patch.object(context['observer'], 'classify', side_effect=AssertionError('No native on mismatch')):
                with self.assertRaisesRegex(ValueError, 'atomic hard predicate differs'):
                    labels.classify_row(row, 0, context, budget, '0'*64)
            self.assertEqual(dict(budget.calls), dict(capture=1, atomic=1))

    def test_capture_rejection_still_checked_but_needs_no_atomic_query(self):
        with tempfile.TemporaryDirectory() as directory:
            _, rows, context, plan, _ = synthetic(Path(directory))
            row = copy.deepcopy(rows[2]); context['config']['capture_radius'] = 7.
            row.update(capture_valid=False, hard_valid=False)
            budget = labels.Budget(plan['limits'], lambda event: None)
            with mock.patch.object(context['core'], 'hard_valid', side_effect=AssertionError('Capture short circuit')):
                record, _ = labels.classify_row(row, 2, context, budget, '0'*64)
            self.assertFalse(record['hard_valid']); self.assertFalse(record['applicable'])
            self.assertEqual(dict(budget.calls), dict(capture=1))

    def test_exact_old_r5_partition_preserves_strict_q_and_closed_radius(self):
        with tempfile.TemporaryDirectory() as directory:
            _, rows, context, plan, _ = synthetic(Path(directory))
            for q, radius, inside in [(1., 5., False), (np.nextafter(1., math.inf), 5., True),
                                      (2., np.nextafter(5., math.inf), False)]:
                q = float(q)
                row = copy.deepcopy(rows[0]); row['q'] = q
                with mock.patch.object(labels.infrastructure.physical.native, 'native_q', return_value=q), \
                     mock.patch.object(labels, 'chart_coordinates', return_value=np.array([[radius, 0., 0., 0., 0., 0.]])):
                    record, _ = labels.classify_row(row, 0, context, labels.Budget(plan['limits'], lambda e: None), '0'*64)
                self.assertEqual(record['native_partition']['old_R5_intersection_native'], inside)
                self.assertEqual(record['native_partition']['remaining_R4_native'], not inside)

    def test_failure_preserves_begun_query_and_partial_labels_and_cannot_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, context, plan, path = synthetic(Path(directory))
            hard = context['core'].hard_valid
            def fail_second(position, rotation):
                if position[0] < 1.: raise ValueError('synthetic atomic failure')
                return hard(position, rotation)
            with mock.patch.object(context['core'], 'hard_valid', side_effect=fail_second):
                with self.assertRaisesRegex(ValueError, 'synthetic atomic failure'): execute(context, plan, path)
            self.assertFalse(Path(plan['output']['receipt']).exists())
            failure = labels.read(plan['output']['failure'])
            self.assertEqual(failure['completed_rows'], 1); self.assertEqual(failure['draw'], 1)
            partial = Path(plan['output']['labels']).read_text().splitlines()
            self.assertEqual(len(partial), 1)
            events = [json.loads(line) for line in Path(plan['output']['journal']).read_text().splitlines()]
            self.assertEqual(events[-2]['state'], 'query_begin')
            self.assertEqual(events[-2]['role'], 'atomic')
            self.assertEqual(events[-1]['state'], 'failed')
            with self.assertRaisesRegex(ValueError, 'Output must be new'): execute(context, plan, path)

    def test_plan_hash_budget_and_source_changes_fail_before_queries(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, context, plan, path = synthetic(Path(directory))
            with self.assertRaisesRegex(ValueError, 'plan changed'): labels.run(path, plan_sha256='0'*64)
            plan['source_sha256'][Path(labels.__file__).name] = '0'*64; save(path, plan)
            with mock.patch.object(labels, '_load_context', side_effect=AssertionError('No geometry')):
                with self.assertRaisesRegex(ValueError, 'Changed bound input'): execute(context, plan, path)
            self.assertFalse(Path(plan['output']['labels']).exists())
        with tempfile.TemporaryDirectory() as directory:
            _, _, context, plan, path = synthetic(Path(directory))
            plan['limits']['max_rows'] = 2; save(path, plan)
            with self.assertRaisesRegex(ValueError, 'allocation exceeds'): execute(context, plan, path)

    def test_unterminated_and_overlong_records_retain_failure_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)/'rows.jsonl'; p.write_bytes(b'{"draw":0}')
            reader = labels.BoundedLines(p, 100)
            try:
                with self.assertRaisesRegex(ValueError, 'unterminated'): reader.next()
                self.assertEqual(reader.last['offset'], 0)
            finally: reader.close()
            p.write_bytes(b'{"long":"'+b'x'*100+b'"}\n')
            reader = labels.BoundedLines(p, 20)
            try:
                with self.assertRaisesRegex(ValueError, 'byte budget'): reader.next()
                self.assertEqual(reader.last['bytes'], 21)
            finally: reader.close()


if __name__ == '__main__': unittest.main()
