"""Synthetic metadata/lifecycle tests; no protein rows or geometry queries."""
import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import types
import unittest
from unittest import mock

import audit_evolving_dimer_root_guidance as reference
import run_evolving_dimer_root_guidance_audit as driver


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, bytes): path.write_bytes(value)
    else: path.write_text(json.dumps(value)+'\n')
    return dict(path=str(path.resolve()), sha256=driver.sha(path))


def fixture(root):
    """Canonical schemas and complete inventory with inert bytes, not fake bypasses."""
    base = root/'campaign'; old = root/'original'; base.mkdir(); old.mkdir()
    contexts = [dict(name=f'context{c}', root=2*c+1, child=2*c+2, anchor=0) for c in range(4)]
    physical = dict(depletant_radius=1.4, activity=.0275, lambda_ratio=64., wall_radius=593.742500239952)
    old_config = put(old/'config.json', dict(physical=physical, contexts=contexts))
    old_binding = put(old/'binding.json', dict(config_sha256=old_config['sha256']))
    prepared_files = []; banks = []; starts = []
    for c in range(4):
        for init in ('source', 'proposal_prepared'):
            for stream in range(4):
                raw = put(old/f'prepared/cloud-{c}-{init}-{stream}.bin', b'inert raw bytes; never parsed')
                meta = put(old/f'prepared/cloud-{c}-{init}-{stream}.json', dict(inert=True))
                prepared_files.extend([raw, meta]); banks.append(dict(context_index=c, initialization=init, stream=stream, raw=raw, metadata=meta))
        for stream in range(4):
            record = put(old/f'prepared/start-{c}-{stream}.json', dict(inert=True))
            ledger = put(old/f'prepared/start-{c}-{stream}.jsonl', b'inert ledger; never parsed\n')
            prepared_files.extend([record, ledger]); starts.append(dict(context_index=c, stream=stream, record=record, ledger=ledger))
    prepared = put(old/'prepared/manifest.json', dict(complete=True, passed=True, all_attempts_retained=True,
        config_sha256=old_config['sha256'], binding_sha256=old_binding['sha256'], files=prepared_files,
        cloud_banks=banks, alternative_starts=starts))
    old_protocol = put(old/'audit-protocol.json', dict(inert=True))
    audit_inputs = {f['path']: f['sha256'] for f in prepared_files+[old_config, old_binding, prepared]}
    old_audit = put(old/'audit.json', dict(schema='evolving-dimer-preparation-independent-audit-v1',
        complete=True, passed=True, physical=physical, source_contexts=contexts,
        input_sha256=audit_inputs, protocol=old_protocol))
    authority = dict(config=old_config, binding=old_binding, prepared_manifest=prepared, preparation_audit=old_audit)
    audit_plan = put(base/'root-guidance-audit-plan.json', reference.audit_protocol())
    source = put(base/'common/source/tools'/driver.AUDITOR, b'# synthetic archived source\n')
    example = put(base/'common/source/examples/evolving_dimer_benchmark.rs', b'// synthetic\n')
    protocol = put(base/'protocol.json', dict(schema='evolving-dimer-root-extension-protocol-v1',
        root_guidance_audit_plan=audit_plan,
        source_files={'tools/'+driver.AUDITOR: source['sha256'],
                      'examples/evolving_dimer_benchmark.rs': example['sha256']}))
    jobs = [dict(id=3*k+2, context_index=c, initialization=i, stream=s, arm='root_m4')
        for k, (c, i, s) in enumerate((c, i, s) for c in range(4)
                for i in ('source', 'proposal_prepared') for s in range(4))]
    config = put(base/'config.json', dict(schema='evolving-dimer-benchmark-v1', jobs=jobs, contexts=contexts,
        allocation=dict(warmup_blocks=512, production_blocks=4096, local_attempts=589824, dimer_attempts=147456),
        physical=physical, inherited_campaign=authority, output=str(base/'execution'), protocol=protocol))
    freeze = put(base/'freeze.json', dict(schema='evolving-dimer-root-extension-freeze-v1', complete=True,
        scientific_execution_started=False,
        files={str(Path(f['path']).relative_to(base)): f['sha256'] for f in [config, protocol, audit_plan, source, example]},
        input_sha256={f['path']: f['sha256'] for f in [old_config, old_binding, prepared, old_audit]}))
    exe = put(base/'common/evolving_dimer_benchmark', b'inert executable never run')
    bundle = put(base/'common/source-bundle.json', dict(inert=True))
    binding = put(base/'binding.json', dict(config_sha256=config['sha256'], protocol_sha256=protocol['sha256'],
        freeze_sha256=freeze['sha256'], executable_sha256=exe['sha256'],
        compiled_source_bundle_sha256=bundle['sha256'], example_source_sha256=example['sha256']))
    run = put(base/'run-binding.json', dict(complete=True, config_sha256=config['sha256'], protocol_sha256=protocol['sha256'],
        prelaunch_binding=binding, prepared_manifest=prepared, prepared_files={f['path']: f['sha256'] for f in prepared_files}))
    review = put(base/'review.json', dict(complete=True, passed=True, input_sha256={config['path']: config['sha256']}))
    dispatch = put(base/'dispatch/plan.json', dict(jobs=jobs, retries=False, replacements=False, review=review,
        input_sha256={f['path']: f['sha256'] for f in [config, run, binding, exe]}))
    completed = []
    for job in jobs:
        directory = base/'execution'/f"job-{job['id']:03}"
        trajectory = put(directory/'trajectory.jsonl', b'this is deliberately not JSON: metadata must never parse\n')
        terminal = put(directory/'terminal.json', dict(complete=True, conditional_target=True, job=job, blocks=4608,
            config_sha256=config['sha256'], binding_sha256=run['sha256'], trajectory=trajectory,
            counts=dict(local_attempted=18432, dimer_attempted=4608)))
        completed.append(dict(job=job, success=True, returncode=0, error=None, child_drained=True, terminal_sha256=terminal['sha256']))
    put(base/'dispatch/status.json', dict(complete=True, passed=True, failure=None, failure_draining=False,
        active=[], unstarted=[], completed=completed, plan_sha256=dispatch['sha256']))
    return base


class MetadataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        self.base = fixture(self.root)

    def tearDown(self): self.temp.cleanup()

    def test_complete_exact32_metadata_without_row_parse(self):
        with mock.patch.object(reference, 'CountOracle', side_effect=AssertionError('No geometry')):
            bound = driver.campaign_bindings(self.base)
        self.assertEqual(len(bound['jobs']), 32)
        self.assertEqual(bound['jobs'][0]['job']['id'], 2)
        self.assertEqual(bound['jobs'][-1]['job']['id'], 95)
        self.assertIn(str(self.base/'execution/job-002/trajectory.jsonl'), bound['files'])

    def test_partial_duplicate_failed_inventory_rejected(self):
        path = self.base/'dispatch/status.json'; original = driver.read(path)
        for mutate in [lambda s: s.update(complete=False),
                       lambda s: s['completed'].pop(),
                       lambda s: s['completed'].__setitem__(1, s['completed'][0]),
                       lambda s: s['completed'][0].update(child_drained=False)]:
            changed = copy.deepcopy(original); mutate(changed); put(path, changed)
            with self.assertRaises(ValueError): driver.campaign_bindings(self.base)
        put(path, original)

    def test_terminal_identity_even_if_hash_rebound(self):
        path = self.base/'execution/job-002/terminal.json'; value = driver.read(path)
        value['job']['arm'] = 'm4'; ref = put(path, value)
        status = driver.read(self.base/'dispatch/status.json'); status['completed'][0]['terminal_sha256'] = ref['sha256']
        put(self.base/'dispatch/status.json', status)
        with self.assertRaisesRegex(ValueError, 'Terminal provenance'):
            driver.campaign_bindings(self.base)

    def test_changed_trajectory_and_failure_receipt(self):
        path = self.base/'execution/job-002/trajectory.jsonl'; raw = path.read_bytes(); path.write_bytes(raw+b'x')
        with self.assertRaisesRegex(ValueError, 'Changed bound'):
            driver.campaign_bindings(self.base)
        path.write_bytes(raw); put(path.parent/'failure.json', {'complete': False})
        with self.assertRaisesRegex(ValueError, 'Terminal provenance'):
            driver.campaign_bindings(self.base)

    def test_original_preparation_receipt_is_mandatory(self):
        original = self.root/'original/audit.json'; value = driver.read(original); value['passed'] = False
        put(original, value)
        with self.assertRaisesRegex(ValueError, 'Changed bound'):
            driver.campaign_bindings(self.base)

    def test_protocol_query_and_event_limits(self):
        p = reference.audit_protocol()
        for key, value in [('maximum_count_queries', 2561), ('new_clouds', 1), ('cpu_seconds', 1801)]:
            broken = copy.deepcopy(p); broken[key] = value
            with self.assertRaisesRegex(ValueError, 'allocation'):
                driver.check_protocol(broken)
        p['geometric_event']['block'] = 2
        with self.assertRaises(ValueError): driver.check_protocol(p)


class BudgetTests(unittest.TestCase):
    def test_query_begin_is_durable_before_oracle_and_failure_has_no_completion(self):
        events = []; oracle = mock.Mock()
        def compute(points, pose):
            self.assertEqual(events[-1]['state'], 'count_begin')
            self.assertEqual(events[-1]['points'], len(points))
            return 2
        oracle.relative.side_effect = compute
        budget = driver.CountBudget(oracle, on_event=events.append)
        self.assertEqual(budget.relative([1, 2, 3], None), 2)
        self.assertEqual([r['state'] for r in events], ['count_begin', 'count_complete'])
        oracle.relative.side_effect = ValueError('query failure')
        with self.assertRaisesRegex(ValueError, 'query failure'):
            budget.relative([1], None)
        self.assertEqual(events[-1]['state'], 'count_begin')
        self.assertEqual(events[-1]['ordinal'], 2)
        self.assertEqual(budget.completed, 1)
        blocked = driver.CountBudget(oracle, on_event=lambda _: (_ for _ in ()).throw(OSError('journal write')))
        with self.assertRaisesRegex(OSError, 'journal write'):
            blocked.world([1], None)
        oracle.world.assert_not_called()

    def test_budget_reserved_before_call_and_partial_failures(self):
        oracle = mock.Mock(); oracle.relative.return_value = 2
        budget = driver.CountBudget(oracle)
        self.assertEqual(budget.relative([1, 2, 3], None), 2)
        self.assertEqual((budget.started, budget.completed, budget.point_tests), (1, 1, 3))
        oracle.world.side_effect = ValueError('partial count error')
        with self.assertRaisesRegex(ValueError, 'partial'):
            budget.world([1, 2], None)
        self.assertEqual((budget.started, budget.completed, budget.point_tests), (2, 1, 5))
        budget.started = 2560
        with self.assertRaisesRegex(ValueError, 'budget exhausted'):
            budget.relative([1], None)
        self.assertEqual(oracle.relative.call_count, 1)
        budget.started = 1; budget.point_tests = 41943040
        with self.assertRaisesRegex(ValueError, 'budget exhausted'):
            budget.relative([1], None)


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        (self.root/'child').mkdir()
        self.jobs = [dict(job=dict(id=i)) for i in range(32)]
        self.plan = dict(base='synthetic-base', jobs=self.jobs, argv=['never-executed'])
        put(self.root/'execution-plan.json', self.plan)
        put(self.root/'review.json', dict(complete=True, passed=True,
            plan_sha256=driver.sha(self.root/'execution-plan.json'), chains=32, maximum_count_queries=2560))

    def tearDown(self): self.temp.cleanup()

    def fake_success(self, *args):
        put(self.root/'attempts.jsonl', b'{"state":"synthetic"}\n')
        put(self.root/'audit.json', dict(complete=True, passed=True,
            plan_sha256=driver.sha(self.root/'execution-plan.json'), chains=[dict(job=j['job']) for j in self.jobs],
            scalar_dimer_attempts=147456, scalar_local_attempts=589824, independent_geometry_events=32,
            count_queries_completed=0, count_queries_started=0, point_membership_tests_started=0,
            journal_sha256=driver.sha(self.root/'attempts.jsonl')))

    def launch(self, action):
        with mock.patch.object(driver, 'verify'), mock.patch.object(driver, 'checked_import',
            return_value=types.SimpleNamespace(owned_child=action)):
            driver.run(self.root)

    def test_success_exclusive_claim_and_exact_resources(self):
        calls = []
        def invoke(*args):
            calls.append(args); self.fake_success()
        self.launch(invoke)
        self.assertEqual(calls[0][-3:], (1800, 3600, 16*1024**3))
        summary = driver.read(self.root/'summary.json')
        self.assertTrue(summary['passed']); self.assertEqual(summary['geometric_events'], 32)
        self.assertEqual(summary['result']['sha256'], driver.sha(self.root/'audit.json'))
        with self.assertRaises(FileExistsError): self.launch(invoke)
        self.assertEqual(len(calls), 1)

    def test_timeout_failure_keeps_claim_and_begin_no_retry(self):
        def timeout(*args): raise subprocess.TimeoutExpired(args[0], 3600)
        with self.assertRaises(subprocess.TimeoutExpired): self.launch(timeout)
        self.assertTrue((self.root/'claim.json').is_file())
        self.assertTrue((self.root/'child/begin.json').is_file())
        self.assertFalse(driver.read(self.root/'failure.json')['passed'])
        self.assertFalse((self.root/'summary.json').exists())

    def test_sigterm_failure_restores_handler(self):
        previous = signal.getsignal(signal.SIGTERM)
        def stop(*args): os.kill(os.getpid(), signal.SIGTERM)
        with self.assertRaises(SystemExit): self.launch(stop)
        self.assertEqual(signal.getsignal(signal.SIGTERM), previous)
        self.assertTrue(driver.read(self.root/'failure.json')['partial_outputs_retained'])

    def test_overbudget_result_rejected(self):
        def invoke(*args):
            self.fake_success(); p = self.root/'audit.json'; r = driver.read(p)
            r['count_queries_started'] = r['count_queries_completed'] = 2561; put(p, r)
        with self.assertRaisesRegex(ValueError, 'allocation exceeded'):
            self.launch(invoke)
        self.assertFalse((self.root/'summary.json').exists())

    def test_unreviewed_plan_never_claimed(self):
        review = driver.read(self.root/'review.json'); review['plan_sha256'] = 'wrong'
        put(self.root/'review.json', review)
        with self.assertRaisesRegex(ValueError, 'launch review'):
            self.launch(lambda *a: self.fail('Must not launch'))
        self.assertFalse((self.root/'claim.json').exists())


if __name__ == '__main__': unittest.main()
