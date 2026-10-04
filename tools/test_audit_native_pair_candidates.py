"""Synthetic-only tests for the bounded initial-state differential audit."""
from collections import Counter
import copy
import itertools
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import audit_native_pair_candidates as audit
from native_graph_consistency import NativeGraphConsistency
from native_pair_candidates import NativePairCandidates


def pose(x): return dict(position=[float(x), 0., 0.], orientation=[1., 0., 0., 0.])


class ToyNative:
    def __init__(self):
        self.motifs = [dict(id=k, relative_position=[1., 0., 0.],
                            relative_orientation=[1., 0., 0., 0.]) for k in (0, 1)]
        self.calls = []

    def classify_pair(self, a, b):
        self.calls.append((a, b))
        if abs(b['position'][0]-a['position'][0]-1.) <= .1:
            return [dict(motif_id=0), dict(motif_id=1)]
        return []


def metadata_fixture(root):
    """Fake metadata only, with no native observer or physical shape query."""
    def save(name, value):
        path = root/name; path.parent.mkdir(parents=True, exist_ok=True)
        audit.write(path, value)
        return audit.reference(path)
    shape = save('shape.json', dict(atoms=[{} for _ in range(4004)]))
    inputs = {}
    for i in range(15):
        ref = save(f'native/inputs/input-{i}.json', {})
        inputs[f'input-{i}.json'] = ref['sha256']
    criteria = dict(body_member_position_entry_A=2.)
    definition = save('native/definition.json', dict(input_sha256=inputs, shape_sha256=shape['sha256'], criteria=criteria))
    compiled = save('compiled.json', dict(source_definition_sha256=definition['sha256'],
                                          source_input_sha256=inputs, criteria=criteria))
    report = dict(compiled_sha256=compiled['sha256'], expected_shape_sha256=shape['sha256'], compatible=True,
                  hard_valid_implication_within_tolerance=True, native_atoms=4004, physical_atoms=4004,
                  matched_atoms=4004, physical_index_by_native_atom=list(range(4004)),
                  unmatched_native_atoms=[], unmatched_physical_atoms=[], matched_max_center_error_a=0.,
                  matched_max_radius_error_a=0., pair_overlap_slack_bound_a=0., center_tolerance_a=1e-10,
                  radius_tolerance_a=1e-12, observer_hard_overlap_tolerance_a=1e-8)
    witness = save('witness.json', dict(native_definition_sha256=definition['sha256'], shape_sha256=shape['sha256'],
                                       compiled_native_sha256=compiled['sha256'], native_shape_compatibility=report))
    poses = [pose(i) for i in range(264)]
    source = save('source.json', dict(poses=poses, boundary='spherical'))
    freeze = save('freeze.json', dict(frame_sha256=source['sha256'], shape_sha256=shape['sha256']))
    source_config = save('source-config.json', {})
    contexts = [dict(root=2*i, child=2*i+1) for i in range(4)]
    original = save('original-config.json', dict(source_frame=source, shape=shape, contexts=contexts))
    starts = []
    for c, s in itertools.product(range(4), range(4)):
        selected = [poses[2*c], poses[2*c+1]]
        record = save(f'start-{c}-{s}.json', dict(context_index=c, stream=s, status='prepared',
                      is_equilibrium_sample=False, selected=selected, source=selected))
        starts.append(dict(context_index=c, stream=s, status='prepared', record=record))
    prepared = save('prepared.json', dict(schema='evolving-dimer-prepared-starts-v1', complete=True, passed=True,
                 all_attempts_retained=True, config_sha256=original['sha256'], alternative_starts=starts))
    files = {s['record']['path']: s['record']['sha256'] for s in starts}
    binding = save('run-binding.json', dict(complete=True, prepared_manifest=prepared,
                   config_sha256=original['sha256'], prepared_files=files))
    preparation_audit = save('preparation-audit.json', dict(complete=True, passed=True, native_classifier_calls=0,
                            input_sha256={**files, source['path']: source['sha256']}))
    config = save('config.json', dict(schema='evolving-dimer-benchmark-v1', shape=shape, source_frame=source,
                 source_freeze_manifest=freeze, source_config=source_config,
                 physical=dict(depletant_radius=1.4, activity=.0275), contexts=contexts,
                 inherited_campaign=dict(config=original, prepared_manifest=prepared,
                                         run_binding=binding, preparation_audit=preparation_audit)))
    return dict(config=config, definition=definition, witness=witness, compiled=compiled,
                shape=shape, starts=starts)


class InitialAuditTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.budget = audit.Budget(100, dict(cpu_seconds=60, wall_seconds=60), self.events.append)
        self.native = ToyNative()
        self.candidate = NativePairCandidates([[0., 0., 0.]], [pose(1.)], .1)
        self.state = [pose(0), pose(1), pose(2), pose(100)]
        self.checker = NativeGraphConsistency(self.native.motifs, 4)

    def test_one_reference_pass_preserves_all_labels_and_graph(self):
        result = audit.audit_state(self.native, self.candidate, self.checker, self.state, 'source', self.budget)
        self.assertEqual(len(self.native.calls), 6)
        self.assertEqual(self.budget.begun, self.budget.completed)
        self.assertEqual(result['reference_native_keys'], [(0, 1, 0), (0, 1, 1), (1, 2, 0), (1, 2, 1)])
        self.assertEqual(result['candidate_pair_calls'], 2)
        self.assertTrue(result['graph']['consistent'])
        self.assertFalse(result['second_classifier_pass'])

    def test_mobile_pair_inventory_and_fixed_keys_are_reused(self):
        source = audit.audit_state(self.native, self.candidate, self.checker, self.state, 'source', self.budget)
        fixed = [k for k in source['reference_native_keys'] if not {2, 3}.intersection(k[:2])]
        changed = copy.deepcopy(self.state); changed[2] = pose(3.)
        result = audit.audit_state(self.native, self.candidate, self.checker, changed, 'start', self.budget,
                                   members=[2, 3], fixed_keys=fixed)
        self.assertEqual(result['reference_pair_calls'], 5)
        self.assertEqual(len(self.native.calls), 11)
        self.assertEqual(result['instantaneous_native_keys'], fixed)
        self.assertEqual(result['graph'], self.checker.check(fixed))

    def test_false_negative_is_fatal_with_preserved_completed_query(self):
        with mock.patch.object(self.candidate, 'candidate_pairs', return_value=[]):
            with self.assertRaisesRegex(ValueError, 'False-negative'):
                audit.audit_state(self.native, self.candidate, self.checker, self.state, 'source', self.budget)
        self.assertEqual((self.budget.begun, self.budget.completed), (1, 1))
        self.assertEqual(self.events[-1]['state'], 'pair_complete')
        self.assertEqual(self.events[-1]['matches'], [dict(motif_id=0), dict(motif_id=1)])

    def test_classifier_failure_preserves_uncompleted_attempt(self):
        with mock.patch.object(self.native, 'classify_pair', side_effect=RuntimeError('fatal classifier')):
            with self.assertRaisesRegex(RuntimeError, 'fatal classifier'):
                audit.audit_state(self.native, self.candidate, self.checker, self.state, 'source', self.budget)
        self.assertEqual((self.budget.begun, self.budget.completed), (1, 0))
        self.assertEqual(self.events[-1]['state'], 'pair_begin')

    def test_caps_refuse_extra_queries_and_count_setup_separately(self):
        self.budget.maximum = 1
        self.budget.query('state', (0, 1), lambda: [])
        with self.assertRaisesRegex(ValueError, 'cap'):
            self.budget.query('state', (0, 2), lambda: self.fail('No second query'))
        self.budget.setup_query('reference_contacts', 1, lambda: None)
        with self.assertRaisesRegex(ValueError, 'Setup cap'):
            self.budget.setup_query('reference_contacts', 1, lambda: self.fail('No second setup'))
        self.assertEqual(self.budget.setup_calls, Counter(reference_contacts_started=1, reference_contacts_completed=1))
        self.assertEqual((self.budget.begun, self.budget.completed), (1, 1))

    def test_plan_is_metadata_only_and_binds_all_starts(self):
        with tempfile.TemporaryDirectory() as directory:
            data = metadata_fixture(Path(directory))
            with mock.patch.multiple(audit, DEFINITION_SHA=data['definition']['sha256'],
                    SHAPE_SHA=data['shape']['sha256'], WITNESS_SHA=data['witness']['sha256']), \
                 mock.patch.object(audit, 'observer_setup_inventory', return_value=dict(observer_instances=1)), \
                 mock.patch.object(audit, 'load_bounded_classifier', side_effect=AssertionError('No classifier')), \
                 mock.patch.object(audit, 'source_closure', return_value={'test': 'fake'}), \
                 mock.patch.object(audit, 'runtime', return_value={'synthetic': True}):
                plan = audit.make_plan(*(data[k]['path'] for k in ('config', 'definition', 'witness', 'compiled')),
                                       root=Path(directory)/'audit-root')
                self.assertEqual(plan['allocation']['maximum_reference_pair_queries'], 43116)
                self.assertEqual(len(plan['starts']), 16)
                self.assertTrue(all(s['record']['path'] in plan['input_sha256'] for s in data['starts']))
                with Path(data['starts'][0]['record']['path']).open('a') as stream: stream.write(' ')
                with self.assertRaisesRegex(ValueError, 'Changed frozen input'):
                    audit.make_plan(*(data[k]['path'] for k in ('config', 'definition', 'witness', 'compiled')),
                                    root=Path(directory)/'audit-root')

    def test_incomplete_or_nonidentity_witness_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            data = metadata_fixture(Path(directory))
            witness = audit.read(data['witness']['path'])
            witness['native_shape_compatibility']['matched_max_center_error_a'] = 1e-12
            bad = Path(directory)/'bad-witness.json'; audit.write(bad, witness)
            with mock.patch.multiple(audit, DEFINITION_SHA=data['definition']['sha256'],
                    SHAPE_SHA=data['shape']['sha256'], WITNESS_SHA=audit.sha(bad)):
                with self.assertRaisesRegex(ValueError, 'zero-error'):
                    audit.make_plan(data['config']['path'], data['definition']['path'], bad, data['compiled']['path'],
                                    root=Path(directory)/'audit-root')


if __name__ == '__main__': unittest.main()
