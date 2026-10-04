"""Synthetic worker primitives only: no protein inputs, jobs or native queries."""
from collections import Counter
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

import run_conditional_native_audit as audit


def pose(x): return dict(position=[float(x), 0., 0.], orientation=[1., 0., 0., 0.])


class ToyNative:
    def __init__(self):
        self.member_positions = np.asarray([[0., 0., 0.]])
        self.definition = dict(criteria=dict(body_member_position_entry_A=.1))
        self.motifs = [dict(id=i, relative_position=[4., 0., 0.],
                           relative_orientation=[1., 0., 0., 0.]) for i in (7, 8)]
        self.calls = []

    def classify_pair(self, a, b):
        self.calls.append(copy.deepcopy((a, b)))
        return ([dict(motif_id=7), dict(motif_id=8)]
                if abs(b['position'][0]-a['position'][0]-4.) <= .1 else [])


def authority_fixture(root):
    """A fake owned-child declaration; no controller is launched."""
    (root/'execution').mkdir()
    source = root/'worker.py'; source.write_text('# synthetic source\n')
    executable = root/'python'; executable.write_text('synthetic executable\n')
    plan_path = root/'audit-plan.json'; plan_path.write_text('{}\n')
    digest = audit.sha(plan_path)
    plan = dict(root=str(root), output=str(root/'analysis'), input_sha256={},
                source_sha256={'worker.py': audit.sha(source)},
                runtime=dict(executable=audit.reference(executable)),
                limits=dict(cpu_seconds=60, wall_seconds=120, address_space_bytes=1024))
    job = dict(id='native', population='all', phase='geometry',
               argv=audit.worker_argv(plan_path, digest, plan['output']),
               terminal=dict(path=str(root/'analysis/summary.json'), success_contract='complete_and_passed'),
               cpu_limit_seconds=60, wall_limit_seconds=120, address_space_limit_bytes=1024)
    execution = dict(root=str(root), jobs=[job], files={str(source): audit.sha(source),
                     str(executable): audit.sha(executable), str(plan_path): digest})
    audit.write(root/'execution-plan.json', execution)
    pid = os.getppid()
    birth = int(Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19])
    claim = dict(plan_sha256=audit.sha(root/'execution-plan.json'), pid=pid, birth_ticks=birth,
                 maximum_workers=1, threads=1, retries=0, replacements=0)
    audit.write(root/'execution/claim.json', claim)
    audit.write(root/'execution/status.json', dict(plan_sha256=claim['plan_sha256'], failure=None,
                active=dict(ordinal=0, id='native', population='all', phase='geometry')))
    return plan_path, digest, plan, source


class WorkerPrimitiveTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.budget = audit.Budget(dict(main_pair_query_cap=20, checkpoint_pair_query_cap=20,
                                       checkpoint_endpoints=2),
                                   dict(cpu_seconds=60, wall_seconds=60), self.events.append)

    def test_role_and_setup_caps_are_separate_and_attempts_are_preserved(self):
        self.budget.allocation.update(main_pair_query_cap=1, checkpoint_pair_query_cap=1)
        self.assertEqual(self.budget.query('main', {'pair': [0, 1]}, lambda: []), [])
        self.assertEqual(self.budget.query('checkpoint', {'pair': [0, 2]}, lambda: []), [])
        for role in ('main', 'checkpoint'):
            with self.assertRaisesRegex(ValueError, 'cap exceeded'):
                self.budget.query(role, {}, lambda: self.fail('Cap must precede query'))
        self.budget.setup_query('reference', 1, lambda: None)
        with self.assertRaisesRegex(ValueError, 'setup cap'):
            self.budget.setup_query('reference', 1, lambda: self.fail('No extra setup'))
        self.assertEqual(self.budget.calls, Counter(main_begun=1, main_completed=1,
                                                  checkpoint_begun=1, checkpoint_completed=1))
        self.assertEqual(self.budget.setup_calls, Counter(reference_started=1, reference_completed=1))

    def test_query_exception_leaves_begun_record_without_completion(self):
        def fail(): raise RuntimeError('synthetic classifier failure')
        with self.assertRaisesRegex(RuntimeError, 'synthetic classifier failure'):
            self.budget.query('main', dict(block=3), fail)
        self.assertEqual(self.events, [dict(state='query_begin', role='main', query=1, block=3)])
        self.assertEqual(self.budget.calls, Counter(main_begun=1))

    def test_fixed_graph_reuse_and_whole_endpoint_cache_preserve_candidates(self):
        native = ToyNative(); source = [pose(x) for x in (0, 4, 8, 30)]
        observer, tracker, location = audit.attach_observer(native, source, [0, 3],
            [(1, 2, 7), (1, 2, 8)], self.budget, 0)
        location.update(chain_id='toy', block=1)
        selected = [pose(0), pose(12)]
        result = observer.classify(selected)
        self.assertEqual(result['fixed_native_keys'], [(1, 2, 7), (1, 2, 8)])
        self.assertEqual(result['external_native_keys'], [(0, 1, 7), (0, 1, 8), (2, 3, 7), (2, 3, 8)])
        self.assertEqual(observer.counts['fixed_pair_calls'], 0)
        self.assertEqual(len(native.calls), 2)
        observer.classify(copy.deepcopy(selected))
        self.assertEqual(len(native.calls), 2)
        self.assertEqual(tracker.calls, 1)
        check = audit.checkpoint(native, source, [0, 3], selected, tracker, self.budget, dict(block=1))
        self.assertEqual((check['candidates'], check['omitted_pairs']), (2, 3))
        self.assertEqual(len(native.calls), 5)
        self.assertEqual(tracker.calls, 1)
        self.assertEqual(self.budget.calls['main_completed'], 2)
        self.assertEqual(self.budget.calls['checkpoint_completed'], 3)
        self.assertEqual(self.budget.calls['checkpoint_endpoints'], 1)

    def test_checkpoint_false_negative_is_fatal_after_completed_query(self):
        native = ToyNative(); source = [pose(x) for x in (0, 4, 8, 30)]
        observer, tracker, location = audit.attach_observer(native, source, [0, 3], [], self.budget, 0)
        location.update(block=1); selected = [source[0], source[3]]
        observer.classify(selected)
        with mock.patch.object(native, 'classify_pair', return_value=[dict(motif_id=7)]):
            with self.assertRaisesRegex(ValueError, 'False-negative'):
                audit.checkpoint(native, source, [0, 3], selected, tracker, self.budget, dict(block=1))
        self.assertEqual(self.budget.calls['checkpoint_begun'], 1)
        self.assertEqual(self.budget.calls['checkpoint_completed'], 1)
        self.assertEqual(self.budget.calls['checkpoint_endpoints'], 0)
        self.assertEqual(self.events[-1]['result'], [dict(motif_id=7)])

    def test_stale_candidates_and_invalid_fixed_keys_fail_before_queries(self):
        native = ToyNative(); source = [pose(x) for x in (0, 4, 8, 30)]
        for keys in ([(0, 1, 7)], [(1, 2, 99)], [(1, 2, 7), (1, 2, 7)]):
            with self.assertRaisesRegex(ValueError, 'inherited fixed'):
                audit.attach_observer(native, source, [0, 3], keys, self.budget, 0)
        observer, tracker, location = audit.attach_observer(native, source, [0, 3], [], self.budget, 0)
        with self.assertRaisesRegex(ValueError, 'current candidate'):
            audit.checkpoint(native, source, [0, 3], [source[0], source[3]], tracker, self.budget, {})
        self.assertEqual(native.calls, [])
        self.assertEqual(self.budget.calls, Counter())

    def test_mobile_pairs_cover_both_directions_and_one_internal_pair(self):
        self.assertEqual(audit.mobile_pairs(4, [3, 1]), ((0, 1), (0, 3), (1, 2), (1, 3), (2, 3)))
        self.assertEqual(len(audit.mobile_pairs(264, [132, 27])), 525)

    def test_bounded_journal_decode_preserves_rows_and_initial_costs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'toy.jsonl'; metadata = {}
            rows = [dict(kind='initial', geometry_load_cpu_seconds=2., cloud_cpu_seconds=3.),
                    dict(kind='retained', block=1), dict(kind='retained', block=2)]
            path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            self.assertEqual(list(audit.decoded_rows(path, 1000, metadata)), rows)
            self.assertEqual(metadata, dict(geometry_load_cpu_seconds=2., cloud_cpu_seconds=3.))
            for data, limit in (('{"x":NaN}\n', 100), ('{}', 100), ('{"x":"long"}\n', 5)):
                path.write_text(data)
                with self.assertRaises(ValueError): list(audit.decoded_rows(path, limit, {}))

    def test_append_flushes_and_syncs_a_durable_failure_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'ledger.jsonl'
            with path.open('x') as stream, mock.patch.object(audit.os, 'fsync', wraps=os.fsync) as sync:
                audit.append(stream, dict(state='query_begin', query=1))
                self.assertEqual(sync.call_count, 1)
                self.assertEqual(json.loads(path.read_text()), dict(state='query_begin', query=1))

    def test_live_authority_binds_source_paths_rather_than_digest_names(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, digest, plan, source = authority_fixture(root)
            with mock.patch.object(audit, 'verify_execution'), \
                 mock.patch.object(audit.binding, 'source_paths', return_value={'worker.py': source}), \
                 mock.patch.object(audit.binding, 'source_closure', side_effect=AssertionError('Path API required')):
                audit.live_authority(path, digest, plan)
                plan['source_sha256']['worker.py'] = '0'*64
                with self.assertRaisesRegex(ValueError, 'omitted frozen'):
                    audit.live_authority(path, digest, plan)


if __name__ == '__main__': unittest.main()
