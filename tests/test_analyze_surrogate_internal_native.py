"""Synthetic dictionaries and tiny journals only; no protein geometry or data."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
import analyze_surrogate_internal_native as a


def pose(x): return dict(position=[float(x), 0., 0.], orientation=[1., 0., 0., 0.])


def tiny_chain(arm=a.FLEXIBLE[0], start='source'):
    job = dict(id=0, context_index=0, arm=arm, initialization=start, stream=0)
    initial = [pose(0), pose(1)]; cpu = 0.
    rows = [dict(kind='initial', block=0, job=job, selected=initial, conditional_target=True, sampler_cpu_seconds=cpu)]
    family = 'flexible' if arm in a.FLEXIBLE else 'rigid'
    # Boundary1=A, production2=empty,3=B,4=A; preserved return through empty.
    for block, x in enumerate((0, 3, 2, 0), 1):
        for attempt, member in enumerate((27, 132, 27, 132)):
            cpu += 1
            rows.append(dict(kind='local', block=block, attempt=attempt, member=member,
                             status='hard_rejected', sampler_cpu_seconds=cpu))
        cpu += 1; rows.append(dict(kind=family+'_surrogate_attempt_begun', block=block,
            status='begun', members=a.MEMBERS, sampler_cpu_seconds=cpu))
        cpu += 1; rows.append(dict(kind=family+'_surrogate_chain', block=block,
            status='identity_self_loop', members=a.MEMBERS, sampler_cpu_seconds=cpu))
        cpu += 1; rows.append(dict(kind='retained_block', block=block, production=block > 1,
            selected=[pose(x), pose(1)], sampler_cpu_seconds=cpu))
    chain = dict(id='toy', job=job, initial_poses=initial, full_sampler_cpu_seconds=cpu+10.)
    return chain, rows


class ToyNative:
    motifs = [dict(id=6), dict(id=7)]
    def classify_pair(self, left, right):
        x = left['position'][0]
        if x == 9.: return [dict(motif_id=6)]
        if x == 27. or x == 3.: return []
        if x == 2.: return [dict(motif_id=7), dict(motif_id=6)]
        return [dict(motif_id=6)]


def budget(events, cap=20):
    return a.NativeBudget(dict(main_pair_query_cap=cap, checkpoint_pair_query_cap=2),
        a.LIMITS, events.append)


def lifecycle(root):
    """Actual generic receipt shape, synthetic bytes, no subprocess."""
    root.mkdir(); (root/'execution').mkdir(); (root/'analysis').mkdir()
    source = root/'fake.py'; source.write_text('# toy\n')
    source_ref = dict(path=str(source), sha256=a.sha(source))
    result = dict(complete=True, source_sha256={'fake.py':source_ref['sha256']}, input_sha256={})
    terminal = root/'analysis/analysis.json'; a.write(terminal, result)
    tr = dict(path=str(terminal), sha256=a.sha(terminal))
    inp = root/'analysis/input-binding.json'; a.write(inp, dict(input_sha256={}))
    manifest = root/'analysis/manifest.json'
    a.write(manifest, dict(complete=True, files={'analysis.json':tr['sha256'], 'input-binding.json':a.sha(inp)}))
    job = dict(id='analysis', population='all', phase='statistics', argv=['/fake/python'],
        terminal=dict(path=str(terminal), success_contract='complete'))
    plan = dict(schema=a.driver.SCHEMA, root=str(root), maximum_workers=1, threads=1,
        jobs=[job], files={str(source):source_ref['sha256']})
    pp = root/'execution-plan.json'; a.write(pp, plan); ph = a.sha(pp)
    directory = a.driver.job_directory(root, 0, job); directory.mkdir(parents=True)
    done = dict(id=job['id'], population='all', phase='statistics', argv=job['argv'],
        terminal=tr, success_contract='complete', success=True, child_started=True,
        child_drained=True, returncode=0, error=None, timeout=False, retries=0, replacements=0,
        pid=101, birth_ticks=20)
    a.write(directory/'success.json', done); a.write(directory/'exit.json', done)
    a.write(directory/'attempt.json', dict(job=job))
    a.write(directory/'process.json', {k:done[k] for k in ('id','pid','birth_ticks','argv')})
    status = dict(complete=True, passed=True, failure=None, active=None, unstarted=[], completed=[done], plan_sha256=ph)
    a.write(root/'execution/status.json', status); a.write(root/'execution/summary.json', status)
    a.write(root/'execution/claim.json', dict(schema=a.driver.SCHEMA, plan_sha256=ph,
        maximum_workers=1, threads=1, retries=0, replacements=0))
    return dict(execution_plan=dict(path=str(pp), sha256=ph), analysis=tr,
        manifest=dict(path=str(manifest), sha256=a.sha(manifest)),
        input_binding=dict(path=str(inp), sha256=a.sha(inp)))


class InternalNativeTests(unittest.TestCase):
    def test_metadata48_plus16_join_rejects_swapped_cache_missing_and_unbound_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); frame={'path':'/frozen/frame.json','sha256':'frame'}
            shape={'path':'/frozen/shape.json','sha256':'shape'}
            def save(name,value):
                path=root/name;a.write(path,value);return dict(path=str(path),sha256=a.sha(path))
            config=save('config.json',dict(source_frame=frame,shape=shape,
                contexts=[dict(root=27,child=132)],physical=dict(depletant_radius=1.4,activity=.0275)))
            source=[pose(i) for i in range(264)]
            contact=dict(schema='flexible-surrogate-analysis-v1',new_chains=24,reused_control_chains=40,
                complete=True,native_observer=False,old_geometry_queries=0,new_physical_draws=0,
                chains=[],input_sha256={config['path']:config['sha256']},_execution_files={})
            rigid=dict(schema='rigid-surrogate-analysis-v1',chains=[],input_sha256={},_execution_files={})
            rigid_ref={'path':'/frozen/rigid-analysis.json','sha256':'rigid'}
            contact['input_sha256'][rigid_ref['path']]=rigid_ref['sha256']
            audit=dict(chains=[]); declared=[]
            for n, ident in enumerate(sorted(a.expected(a.FLEXIBLE+a.RIGID+('local','m4')))):
                ci,arm,start,stream=ident;job=dict(id=n,context_index=ci,arm=arm,initialization=start,stream=stream)
                path=root/f'{n}-trajectory.jsonl';path.write_text('admission hashes bytes without decoding\n')
                trajectory=dict(path=str(path),sha256=a.sha(path));counts={'attempts':1}
                metrics=dict(full_sampler_cpu_seconds=38.,production_samples=4096,retained_endpoints=4609)
                record=dict(job=job,trajectory=trajectory,counts=counts,metrics=metrics)
                contact['chains'].append(record)
                if arm not in a.FLEXIBLE:rigid['chains'].append(record)
                if arm in a.FLEXIBLE+a.RIGID:
                    terminal=save(f'{n}-terminal.json',dict(complete=True,conditional_target=True,job=job,
                        trajectory=trajectory,blocks=4608,counts=counts,cpu_seconds=38.))
                    initial={'kind':start};origin=contact if arm in a.FLEXIBLE else rigid
                    if start=='proposal_prepared':
                        initial['record']=save(f'{n}-start.json',dict(status='prepared',context_index=0,
                            stream=stream,selected=[source[27],source[132]]))
                        origin['input_sha256'][initial['record']['path']]=initial['record']['sha256']
                    for ref in (terminal,trajectory):
                        origin['input_sha256'][ref['path']]=ref['sha256'];origin['_execution_files'][ref['path']]=ref['sha256']
                    declared.append(dict(id=f'chain-{n}',job=job,trajectory=trajectory,terminal=terminal,initial=initial))
                else:
                    desc=a._environment_metrics([[],[],[]],[True]*3,[512,513,514],38.)
                    old=dict(metrics,production_window_cpu_seconds=30.,internal_native_attachment_fraction=0.,
                        internal_motifs=desc,mobile_registry_resolved_fraction=1,mobile_cycle_frustration_fraction=0)
                    audit['chains'].append(dict(job=job,trajectory=trajectory,metrics=old))
            plan=dict(schema=a.PLAN_SCHEMA,allocation=a.ALLOCATION,limits=a.LIMITS,contact={'tag':'flexible'},
                rigid_contact={'tag':'rigid','analysis':rigid_ref},native_audit={'tag':'native','summary':{'sha256':a.NATIVE_SHA}},
                config=config,chains=declared)
            def completed(ref,b,**kwargs):
                return {'flexible':contact,'rigid':rigid,'native':audit}[ref['tag']]
            with mock.patch.object(a,'completed_report',side_effect=completed),mock.patch.object(a,'native_identity',
                    return_value=dict(refs={'source_frame':frame,'shape':shape},source=source)):
                ctx,files=a.admit_inputs(plan)
                self.assertEqual((len(ctx['chains']),len(ctx['controls'])),(48,16))
                self.assertTrue(all(c['trajectory']['path'] in files for c in declared))
                modified=copy.deepcopy(plan);modified['chains'][0]['trajectory']=modified['chains'][1]['trajectory']
                with self.assertRaises(ValueError):a.admit_inputs(modified)
                modified=copy.deepcopy(plan);modified['chains'].pop()
                with self.assertRaises(ValueError):a.admit_inputs(modified)
                old=audit['chains'][-1];audit['chains'][-1]=audit['chains'][0]
                with self.assertRaises(ValueError):a.admit_inputs(plan)
                audit['chains'][-1]=old
                unbound=declared[0];origin=contact if unbound['job']['arm'] in a.FLEXIBLE else rigid
                del origin['_execution_files'][unbound['trajectory']['path']]
                with self.assertRaises(ValueError):a.admit_inputs(plan)

    def test_cadence_both_families_starts_and_all_residence(self):
        for arm in (a.FLEXIBLE[0], a.RIGID[0]):
            for start in a.STARTS:
                chain, rows = tiny_chain(arm, start)
                points = list(a.endpoints(iter(rows), chain, blocks=4, warmup=1))
                self.assertEqual([p['block'] for p in points], list(range(5)))
                self.assertEqual([p['production'] for p in points], [False, False, True, True, True])
                self.assertEqual(points[0]['selected'], points[1]['selected'])

    def test_reject_missing_repeated_fatal_swapped_and_invalid_rows(self):
        chain, original = tiny_chain()
        variants = []
        rows = copy.deepcopy(original); del rows[3]; variants.append(rows)
        rows = copy.deepcopy(original); rows[2]['member'] = 27; variants.append(rows)
        rows = copy.deepcopy(original); rows[6]['fatal_error'] = 'failed'; variants.append(rows)
        rows = copy.deepcopy(original); rows[6]['status'] = 'fatal'; variants.append(rows)
        rows = copy.deepcopy(original); rows[7]['production'] = True; variants.append(rows)
        rows = copy.deepcopy(original); rows[7]['selected'][0]['orientation'] = [2, 0, 0, 0]; variants.append(rows)
        rows = copy.deepcopy(original); rows[0]['sampler_cpu_seconds'] = -.1; variants.append(rows)
        rows = copy.deepcopy(original); rows.append(rows[-1]); variants.append(rows)
        rows = copy.deepcopy(original); rows[0]['job']['stream'] = 1; variants.append(rows)
        for rows in variants:
            with self.assertRaises(ValueError): list(a.endpoints(rows, chain, blocks=4, warmup=1))

    def test_multilabel_exact_cache_cap_and_error_prefix(self):
        events=[]; b=budget(events, cap=2); observer=a.PairObserver(ToyNative(), b)
        poses=[pose(2), pose(1)]
        self.assertEqual(observer.classify(poses, {'block':0}), [[27,132,6],[27,132,7]])
        self.assertEqual(observer.classify(copy.deepcopy(poses), {'block':1}), [[27,132,6],[27,132,7]])
        self.assertEqual(observer.hits, 1); self.assertEqual(b.calls['main_completed'], 1)
        observer.classify([pose(3),pose(1)], {'block':2})
        with self.assertRaises(ValueError): observer.classify(poses, {'block':3})
        broken=mock.Mock(motifs=ToyNative.motifs)
        broken.classify_pair.side_effect=RuntimeError('classification failed')
        events=[]; b=budget(events); observer=a.PairObserver(broken,b)
        with self.assertRaises(RuntimeError): observer.classify(poses, {'block':0})
        self.assertEqual(b.calls['main_begun'], 1); self.assertEqual(b.calls['main_completed'], 0)
        self.assertEqual(events[-1]['state'], 'query_begin')

    def test_positive_negative_fixtures_are_separately_bounded(self):
        b=budget([]); source=[pose(i) for i in range(264)]
        result=a.fixtures(ToyNative(),source,b)
        self.assertEqual(b.calls['checkpoint_completed'],2); self.assertEqual(b.calls['main_completed'],0)
        self.assertTrue(all(x['excluded_from_comparisons'] for x in result))
        wrong=mock.Mock(); wrong.classify_pair.return_value=[]
        with self.assertRaises(ValueError): a.fixtures(wrong,source,budget([]))

    def test_metrics_baseline_empty_returns_full_cpu_and_null_ess(self):
        chain, rows=tiny_chain(); observer=a.PairObserver(ToyNative(),budget([])); trace=[]
        for p in a.endpoints(rows,chain,blocks=4,warmup=1):
            p['internal_native_keys']=observer.classify(p.pop('selected'),{'block':p['block']});trace.append(p)
        m=a.reduce_trace(trace,38.,blocks=4,warmup=1); d=m['internal_motifs']
        self.assertEqual(m['internal_native_attachment_fraction'],2/3)
        self.assertEqual(m['production_window_cpu_seconds'],21.)
        self.assertEqual(d['counts']['enter_nonempty'],1)
        self.assertEqual(d['counts']['direct_nonempty_changes'],1)
        self.assertEqual(d['counts']['nonempty_returns'],1)
        self.assertTrue(d['environments']['completed_returns'][0]['passed_through_empty'])
        self.assertNotIn('return_endpoints_resolved', d['environments']['completed_returns'][0])
        self.assertEqual(d['rates_per_full_sampler_cpu_second']['nonempty_returns'],1/38.)
        empty=[dict(r,internal_native_keys=[]) for r in trace]
        zero=a.reduce_trace(empty,38.,blocks=4,warmup=1)
        self.assertIsNone(zero['internal_motifs']['presence_ess']['apparent_ess'])
        self.assertEqual(zero['internal_motifs']['counts']['nonempty_returns'],0)
        # Same old reducer descriptor, projected without changing values/denominator.
        original=a._environment_metrics([r['internal_native_keys'] for r in trace[1:]], [True]*4, list(range(1,5)),38.)
        old=dict(m,internal_motifs=original)
        self.assertEqual(a.control_metrics(old),m)

    def test_completed_receipt_and_changed_source_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'case'; refs=lifecycle(root)
            result=a.completed_report(refs,a.inherited.Bindings())
            self.assertTrue(result['complete'])
            (root/'fake.py').write_text('# changed\n')
            with self.assertRaises(ValueError): a.completed_report(refs,a.inherited.Bindings())
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'case'; refs=lifecycle(root)
            a.write(root/'execution/failure.json',dict(error='preserved'))
            with self.assertRaises(ValueError): a.completed_report(refs,a.inherited.Bindings())

    def test_plan_scope_and_native_reference_fail_before_classifier(self):
        plan=dict(schema=a.PLAN_SCHEMA,allocation=dict(a.ALLOCATION,new_chains=47),limits=a.LIMITS)
        with mock.patch.object(a,'load_bounded_classifier') as loader:
            with self.assertRaises(ValueError):a.admit_inputs(plan)
            loader.assert_not_called()

    def test_mock_worker_success_and_failure_durable_prefix(self):
        for fail in (False, True):
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp); path=root/'plan.json'; out=root/'analysis'; journal=root/'toy.jsonl'
                chain,rows=tiny_chain(); journal.write_text(''.join(json.dumps(r)+'\n' for r in rows))
                chain.update(trajectory=dict(path=str(journal),sha256=a.sha(journal)),terminal={'path':'/unused','sha256':'unused'})
                plan=dict(root=str(root),output=str(out),input_sha256={str(journal):a.sha(journal)},
                    source_sha256={},runtime={})
                a.write(path,plan); native=ToyNative()
                if fail:
                    original=native.classify_pair
                    def classify(left,right):
                        if left['position'][0] == 3:raise RuntimeError('deliberate query failure')
                        return original(left,right)
                    native.classify_pair=classify
                chains=[dict(chain,id='toy-'+str(i)) for i in range(48)]
                context=dict(chains=chains,controls=[],native=dict(source=[pose(i) for i in range(264)],
                    setup={},refs={'definition':{'path':'/unused'}}))
                allocation=dict(a.ALLOCATION,retained_endpoints=240)
                actual_endpoints=a.endpoints; actual_reduce=a.reduce_trace
                patches=[mock.patch.object(a,'live_authority'), mock.patch.object(a,'validate_plan',return_value=context),
                    mock.patch.object(a,'load_bounded_classifier',return_value=(native,{})),
                    mock.patch.object(a,'source_closure',return_value={}),mock.patch.object(a.inherited,'runtime',return_value={}),
                    mock.patch.object(a,'ALLOCATION',allocation),
                    mock.patch.object(a,'endpoints',side_effect=lambda rows,c:actual_endpoints(rows,c,blocks=4,warmup=1)),
                    mock.patch.object(a,'reduce_trace',side_effect=lambda rows,cpu:actual_reduce(rows,cpu,blocks=4,warmup=1))]
                from contextlib import ExitStack
                with ExitStack() as stack:
                    for patch in patches:stack.enter_context(patch)
                    if fail:
                        with self.assertRaises(RuntimeError):a.run(path,a.sha(path),out)
                        failure=a.read(out/'failure.json')
                        self.assertFalse(failure['complete'])
                        self.assertGreater(failure['query_counts']['main_begun'],failure['query_counts']['main_completed'])
                    else:
                        result=a.run(path,a.sha(path),out)
                        self.assertTrue(result['complete']);self.assertTrue(result['passed'])
                        self.assertEqual(len(result['chains']),48)
                        self.assertEqual(result['query_counts']['checkpoint_completed'],2)
                ledger=[json.loads(line) for line in (out/'attempts.jsonl').read_text().splitlines()]
                self.assertEqual(ledger[-1]['state'],'failure' if fail else 'chain_complete')
                self.assertTrue((out/'chains/toy-0.jsonl').exists())


if __name__ == '__main__': unittest.main()
