"""Synthetic metadata and tiny journals only; no protein inputs or geometry."""
from contextlib import ExitStack
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
import analyze_partner_atlas_native as a
import test_analyze_surrogate_internal_native as fixtures

pose, ToyNative = fixtures.pose, fixtures.ToyNative


def tiny_chain(arm='partner_atlas_direct', start='source'):
    job = dict(id=0, context_index=0, arm=arm, initialization=start, stream=0)
    initial = [pose(0), pose(1)]; current = copy.deepcopy(initial); cpu = 0.
    rows = [dict(kind='initial', block=0, job=copy.deepcopy(job), selected=initial,
                 conditional_target=True, sampler_cpu_seconds=cpu)]
    direct = arm == 'partner_atlas_direct'
    # Boundary1=A, production2=empty,3=B,4=A, with collective rejection/null residence.
    for block, x in enumerate((0, 3, 2, 0), 1):
        for attempt, member in enumerate((27, 132, 27, 132)):
            slot = attempt % 2; old = copy.deepcopy(current[slot])
            if attempt == 2:
                current[0] = pose(x)
            cpu += 1
            rows.append(dict(kind='local', block=block, attempt=attempt, member=member,
                status='completed' if attempt == 2 else 'hard_rejected', accepted=attempt == 2,
                old=old, proposed=copy.deepcopy(current[slot]), retained=copy.deepcopy(current), sampler_cpu_seconds=cpu))
        cpu += 1
        rows.append(dict(kind='partner_atlas_attempt_begun', block=block, status='begun',
            members=a.MEMBERS, inner_filter=not direct, old=copy.deepcopy(current), sampler_cpu_seconds=cpu))
        cpu += 1
        rows.append(dict(kind='flexible_partner_atlas_direct' if direct else 'flexible_partner_atlas_chain',
            block=block, status='identity_self_loop' if block % 2 else 'completed', accepted=False,
            members=a.MEMBERS, inner_filter=not direct, old=copy.deepcopy(current),
            retained=copy.deepcopy(current), steps=[dict(status='null_proposal' if block % 2 else 'completed')],
            sampler_cpu_seconds=cpu))
        cpu += 1
        rows.append(dict(kind='retained_block', block=block, production=block > 1,
            selected=copy.deepcopy(current), sampler_cpu_seconds=cpu))
    return dict(id='toy', job=job, initial_poses=initial, full_sampler_cpu_seconds=cpu+10.), rows


def budget(events, cap=20):
    return a.NativeBudget(dict(main_pair_query_cap=cap, checkpoint_pair_query_cap=0),
                          a.LIMITS, events.append)


def admission_fixture(root):
    """Hash synthetic new bytes; historical journals deliberately do not exist."""
    def save(name, value):
        path = root/name; a.write(path, value)
        return dict(path=str(path), sha256=a.sha(path))
    frame = dict(path='/frozen/frame.json', sha256='f'*64)
    shape = dict(path='/frozen/shape.json', sha256='e'*64)
    source = [pose(i) for i in range(264)]
    contact = dict(schema='partner-atlas-analysis-v1', new_chains=32, reused_control_chains=40,
        complete=True, native_observer=False, old_geometry_queries=0, new_physical_draws=0,
        chains=[], input_sha256={}, _execution_files={})
    def bind(ref):
        for key in ('input_sha256', '_execution_files'): contact[key][ref['path']] = ref['sha256']
    authority = dict(root='/frozen/native', summary=dict(path='/frozen/native/summary.json', sha256='d'*64))
    native_plan = save('native-plan.json', a.preparation.native_plan(authority))
    protocol = save('protocol.json', dict(schema='partner-atlas-protocol-v1', assembly_gate_open=False,
        native_observer_plan=native_plan, native_authority=authority))
    declared, entries, records = [], [], {}
    for n, key in enumerate(sorted(a.expected(a.ARMS + a.prior.RIGID + a.CACHED))):
        ci, arm, start, stream = key
        job = dict(id=n, context_index=ci, arm=arm, initialization=start, stream=stream)
        new = arm in a.ARMS
        path = root/f'{n}-trajectory.jsonl' if new else Path(f'/historical/{n}-trajectory.jsonl')
        if new: path.write_text('Admission hashes synthetic bytes without decoding.\n')
        trajectory = dict(path=str(path), sha256=a.sha(path) if new else 'a'*64)
        record = dict(job=job, trajectory=trajectory, counts={'attempts':1}, reused_control=not new,
            metrics=dict(full_sampler_cpu_seconds=38., production_samples=4096, retained_endpoints=4609))
        contact['chains'].append(record); records[key] = record
        if new:
            initial = dict(kind=start)
            if start == 'proposal_prepared':
                initial['record'] = save(f'{n}-start.json', dict(status='prepared', context_index=0,
                    stream=stream, selected=[source[27], source[132]]))
                bind(initial['record'])
            declared.append(dict(id=f'chain-{n}', job=job, trajectory=trajectory, initial=initial))
        else:
            terminal = save(f'{n}-terminal.json', dict(complete=True, conditional_target=True, job=job,
                trajectory=trajectory, blocks=4608, counts=record['counts'], cpu_seconds=38.))
            bind(terminal); entries.append(dict(job=job, terminal=terminal))
    config = save('config.json', dict(source_frame=frame, shape=shape, protocol=protocol,
        contexts=[dict(root=27, child=132)], physical=dict(depletant_radius=1.4, activity=.0275),
        allocation=dict(warmup_blocks=512, production_blocks=4096), partner_atlas_policy=a.preparation.policy(),
        jobs=[c['job'] for c in declared], control_analysis=dict(observations=entries)))
    for chain in declared:
        old = records[a.identity(chain['job'])]
        chain['terminal'] = save(chain['id']+'-terminal.json', dict(complete=True, conditional_target=True,
            job=chain['job'], trajectory=chain['trajectory'], blocks=4608, counts=old['counts'],
            cpu_seconds=38., config_sha256=config['sha256']))
        bind(chain['terminal']); bind(chain['trajectory'])
    for ref in (config, protocol, native_plan): bind(ref)
    native_rows = []
    for n, key in enumerate(sorted(a.expected(a.prior.FLEXIBLE + a.prior.RIGID + a.CACHED))):
        ci, arm, start, stream = key
        old = records.get(key)
        job = old['job'] if old else dict(id=n+100, context_index=ci, arm=arm, initialization=start, stream=stream)
        trajectory = old['trajectory'] if old else dict(path=f'/historical/flexible-{n}.jsonl', sha256='b'*64)
        native_rows.append(dict(job=job, trajectory=trajectory, reused_native_control=arm in a.CACHED,
            metrics=dict(schema='internal-native-metrics-v1', full_sampler_cpu_seconds=38.,
                production_samples=4096, retained_endpoints=4609, production_window_cpu_seconds=30.,
                internal_native_attachment_fraction=.125,
                internal_motifs={'already_projected':True, 'counts':{'nonempty_returns':3}})))
    native = dict(refs=dict(source_frame=frame, shape=shape), source=source, setup={},
                  report=dict(chains=native_rows, classifier={}))
    plan = dict(schema=a.PLAN_SCHEMA, allocation=a.ALLOCATION, limits=a.LIMITS,
        contact={'tag':'contact'}, config=config, protocol=protocol, native_plan=native_plan, chains=declared)
    return plan, contact, native


class PartnerAtlasNativeTests(unittest.TestCase):
    def test_complete32_plus16_metadata_join_and_unchanged_projected_controls(self):
        with tempfile.TemporaryDirectory() as temp:
            plan, contact, native = admission_fixture(Path(temp))
            with mock.patch.object(a, 'completed_report', return_value=contact), \
                    mock.patch.object(a, 'native_context', return_value=native):
                context, files = a.admit_inputs(plan)
                self.assertEqual((len(context['chains']), len(context['controls'])), (32, 16))
                self.assertEqual({a.identity(c['job']) for c in context['chains']}, a.expected(a.ARMS))
                self.assertTrue(all(c['trajectory']['path'] in files for c in plan['chains']))
                self.assertFalse(any(path.startswith('/historical/') for path in files))
                saved = {a.identity(c['job']):c['metrics'] for c in native['report']['chains']}
                for control in context['controls']:
                    self.assertEqual(control['metrics'], saved[a.identity(control['job'])])
                    self.assertIsNot(control['metrics'], saved[a.identity(control['job'])])
                changed = copy.deepcopy(plan); changed['chains'][0]['trajectory'] = changed['chains'][1]['trajectory']
                with self.assertRaises(ValueError): a.admit_inputs(changed)
                changed = copy.deepcopy(plan); changed['chains'].pop()
                with self.assertRaises(ValueError): a.admit_inputs(changed)
                saved_row = native['report']['chains'][-1]
                native['report']['chains'][-1] = native['report']['chains'][0]
                with self.assertRaises(ValueError): a.admit_inputs(plan)
                native['report']['chains'][-1] = saved_row
                cached = next(c for c in native['report']['chains'] if c['job']['arm'] in a.CACHED)
                cached['metrics']['full_sampler_cpu_seconds'] = 39.
                with self.assertRaises(ValueError): a.admit_inputs(plan)
                cached['metrics']['full_sampler_cpu_seconds'] = 38.
                for ref in (plan['native_plan'], plan['protocol'], plan['chains'][0]['trajectory']):
                    digest = contact['_execution_files'].pop(ref['path'])
                    with self.assertRaises(ValueError): a.admit_inputs(plan)
                    contact['_execution_files'][ref['path']] = digest

    def test_scope_and_query_allocation_fail_before_classifier(self):
        self.assertEqual((a.ALLOCATION['new_chains'], a.ALLOCATION['cached_chains']), (32, 16))
        self.assertEqual((a.ALLOCATION['retained_endpoints'], a.ALLOCATION['production_endpoints']), (147488, 131072))
        self.assertEqual(a.ALLOCATION['main_pair_query_cap'], 147488)
        for key in ('checkpoint_pair_query_cap', 'fixture_queries', 'external_pair_queries',
                    'contact_geometry_queries', 'arithmetic_replays', 'new_draws'):
            self.assertEqual(a.ALLOCATION[key], 0)
        plan = dict(schema=a.PLAN_SCHEMA, allocation=dict(a.ALLOCATION, new_draws=1), limits=a.LIMITS)
        with mock.patch.object(a, 'load_bounded_classifier') as loader:
            with self.assertRaises(ValueError): a.admit_inputs(plan)
            loader.assert_not_called()

    def test_all_four_arms_both_starts_and_rejected_null_residence(self):
        for arm in a.ARMS:
            for start in a.STARTS:
                with self.subTest(arm=arm, start=start):
                    chain, rows = tiny_chain(arm, start)
                    points = list(a.endpoints(iter(rows), chain, blocks=4, warmup=1))
                    self.assertEqual([p['block'] for p in points], list(range(5)))
                    self.assertEqual([p['production'] for p in points], [False, False, True, True, True])
                    self.assertEqual([p['selected'][0]['position'][0] for p in points], [0., 0., 3., 2., 0.])
                    self.assertEqual(points[0]['selected'], points[1]['selected'])

    def test_cadence_fatal_wrong_family_filter_and_copied_retained_rejected(self):
        for arm in a.ARMS:
            chain, original = tiny_chain(arm)
            mutations = [
                lambda r: r.pop(3),
                lambda r: r[2].update(member=27),
                lambda r: r[5].update(kind='flexible_surrogate_attempt_begun'),
                lambda r: r[5].update(inner_filter=arm == 'partner_atlas_direct'),
                lambda r: r[6].update(kind='flexible_surrogate_chain'),
                lambda r: r[6].update(fatal_error='preserved'),
                lambda r: r[6].update(status='fatal'),
                lambda r: r[6].update(retained=[pose(19), pose(1)]),
                lambda r: r[7].update(production=True),
                lambda r: r[7]['selected'][0].update(orientation=[2., 0., 0., 0.]),
                lambda r: r[0].update(sampler_cpu_seconds=-.1),
                lambda r: r[0]['job'].update(stream=1),
                lambda r: r.append(copy.deepcopy(r[-1])),
            ]
            for number, change in enumerate(mutations):
                with self.subTest(arm=arm, change=number):
                    rows = copy.deepcopy(original); change(rows)
                    with self.assertRaises(ValueError):
                        list(a.endpoints(rows, chain, blocks=4, warmup=1))

    def test_multilabel_exact_pose_cache_and_main_cap_without_checkpoint_queries(self):
        events = []; b = budget(events, cap=2); observer = a.PairObserver(ToyNative(), b)
        poses = [pose(2), pose(1)]
        self.assertEqual(observer.classify(poses, {'block':0}), [[27,132,6], [27,132,7]])
        self.assertEqual(observer.classify(copy.deepcopy(poses), {'block':1}), [[27,132,6], [27,132,7]])
        self.assertEqual((observer.hits, b.calls['main_completed']), (1, 1))
        self.assertEqual(observer.classify([pose(3), pose(1)], {'block':2}), [])
        with self.assertRaises(ValueError): observer.classify(poses, {'block':3})
        self.assertEqual(b.calls['checkpoint_begun'], 0)
        broken = mock.Mock(motifs=ToyNative.motifs)
        broken.classify_pair.side_effect = RuntimeError('classification failed')
        events = []; b = budget(events); observer = a.PairObserver(broken, b)
        with self.assertRaises(RuntimeError): observer.classify(poses, {'block':0})
        self.assertEqual((b.calls['main_begun'], b.calls['main_completed']), (1, 0))
        self.assertEqual(events[-1]['state'], 'query_begin')

    def test_baseline_empty_returns_full_cpu_and_constant_null_ess(self):
        chain, rows = tiny_chain(); observer = a.PairObserver(ToyNative(), budget([])); trace = []
        for point in a.endpoints(rows, chain, blocks=4, warmup=1):
            point['internal_native_keys'] = observer.classify(point.pop('selected'), {'block':point['block']})
            trace.append(point)
        metrics = a.reduce_trace(trace, 38., blocks=4, warmup=1); descriptor = metrics['internal_motifs']
        self.assertEqual(metrics['internal_native_attachment_fraction'], 2/3)
        self.assertEqual(metrics['production_window_cpu_seconds'], 21.)
        self.assertEqual(descriptor['counts']['enter_nonempty'], 1)
        self.assertEqual(descriptor['counts']['direct_nonempty_changes'], 1)
        self.assertEqual(descriptor['counts']['nonempty_returns'], 1)
        self.assertTrue(descriptor['environments']['completed_returns'][0]['passed_through_empty'])
        self.assertEqual(descriptor['rates_per_full_sampler_cpu_second']['nonempty_returns'], 1/38.)
        empty = [dict(row, internal_native_keys=[]) for row in trace]
        zero = a.reduce_trace(empty, 38., blocks=4, warmup=1)
        self.assertIsNone(zero['internal_motifs']['presence_ess']['apparent_ess'])
        self.assertEqual(zero['internal_motifs']['counts']['nonempty_returns'], 0)

    def test_completed_lifecycle_source_change_and_failure_receipt(self):
        for change in (None, 'source', 'failure'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)/'case'; refs = fixtures.lifecycle(root)
                if change == 'source': (root/'fake.py').write_text('# changed\n')
                if change == 'failure': a.write(root/'execution/failure.json', dict(error='preserved'))
                if change is None:
                    self.assertTrue(a.completed_report(refs, a.inherited.Bindings())['complete'])
                else:
                    with self.assertRaises(ValueError): a.completed_report(refs, a.inherited.Bindings())

    def test_mock_worker_zero_checkpoints_and_failure_keeps_attempted_prefix(self):
        for fail in (False, True):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as temp:
                root = Path(temp); path = root/'plan.json'; out = root/'analysis'; journal = root/'toy.jsonl'
                chain, rows = tiny_chain()
                journal.write_text(''.join(json.dumps(row)+'\n' for row in rows))
                chain.update(trajectory=dict(path=str(journal), sha256=a.sha(journal)),
                             terminal={'path':'/unused', 'sha256':'unused'})
                plan = dict(root=str(root), output=str(out), input_sha256={str(journal):a.sha(journal)},
                            source_sha256={}, runtime={}, native_plan={'path':'/unused', 'sha256':'unused'})
                a.write(path, plan); native = ToyNative()
                if fail:
                    original = native.classify_pair
                    def classify(left, right):
                        if left['position'][0] == 3.: raise RuntimeError('deliberate query failure')
                        return original(left, right)
                    native.classify_pair = classify
                context = dict(chains=[dict(chain, id=f'toy-{i}') for i in range(32)], controls=[],
                    native=dict(source=[pose(i) for i in range(264)], setup={}, refs={'definition':{'path':'/unused'}},
                                report={'classifier':{}}))
                actual_endpoints, actual_reduce = a.endpoints, a.reduce_trace
                patches = [mock.patch.object(a, 'live_authority'), mock.patch.object(a, 'validate_plan', return_value=context),
                    mock.patch.object(a, 'load_bounded_classifier', return_value=(native, {})),
                    mock.patch.object(a, 'source_closure', return_value={}),
                    mock.patch.object(a.inherited, 'runtime', return_value={}),
                    mock.patch.object(a, 'ALLOCATION', dict(a.ALLOCATION, retained_endpoints=160)),
                    mock.patch.object(a, 'endpoints', side_effect=lambda values, c: actual_endpoints(values, c, blocks=4, warmup=1)),
                    mock.patch.object(a, 'reduce_trace', side_effect=lambda values, cpu: actual_reduce(values, cpu, blocks=4, warmup=1))]
                with ExitStack() as stack:
                    for item in patches: stack.enter_context(item)
                    if fail:
                        with self.assertRaises(RuntimeError): a.run(path, a.sha(path), out)
                        failure = a.read(out/'failure.json')
                        self.assertFalse(failure['complete'])
                        self.assertGreater(failure['query_counts']['main_begun'], failure['query_counts']['main_completed'])
                        self.assertFalse((out/'summary.json').exists())
                    else:
                        result = a.run(path, a.sha(path), out)
                        self.assertTrue(result['complete']); self.assertTrue(result['passed'])
                        self.assertEqual(len(result['chains']), 32)
                        self.assertEqual(result['query_counts']['checkpoint_completed'], 0)
                ledger = [json.loads(line) for line in (out/'attempts.jsonl').read_text().splitlines()]
                self.assertEqual(ledger[-1]['state'], 'failure' if fail else 'chain_complete')
                self.assertTrue((out/'chains/toy-0.jsonl').exists())


if __name__ == '__main__': unittest.main()
