"""Synthetic admission/ownership controls; no physical observer or atom queries."""
import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import run_two_neighbor_singleton_analysis as runner
from test_analyze_two_neighbor_singleton_benchmark import metadata_fixture, put


def fixture(parent):
    base, config = metadata_fixture(parent)
    root = parent/'observer'
    sources = runner.source_closure()
    frozen_sources = {}
    for name, source in sources.items():
        if name in (runner.NAME, 'test_'+runner.NAME): continue
        relative = 'tools/'+name
        target = base/'common/source'/relative
        put(target, source.read_bytes()); frozen_sources[relative] = runner.sha(target)
    example = base/'common/source/examples/evolving_dimer_benchmark.rs'
    put(example, b'// inert synthetic example, never compiled\n')
    frozen_sources['examples/evolving_dimer_benchmark.rs'] = runner.sha(example)
    put(base/'protocol.json', dict(source_files=frozen_sources))
    put(base/'freeze.json', dict(schema='two-neighbor-singleton-freeze-v1', complete=True,
        files={name:runner.sha(base/name) for name in ('config.json', 'protocol.json', 'analysis-plan.json')},
        input_sha256={}))
    put(base/'common/evolving_dimer_benchmark', b'inert synthetic executable, never run')
    put(base/'common/source-bundle.json', dict(inert=True))
    put(base/'binding.json', {key:runner.sha(base/name) for key,name in [
        ('config_sha256','config.json'), ('protocol_sha256','protocol.json'), ('freeze_sha256','freeze.json'),
        ('executable_sha256','common/evolving_dimer_benchmark'), ('compiled_source_bundle_sha256','common/source-bundle.json'),
        ('example_source_sha256','common/source/examples/evolving_dimer_benchmark.rs')]})
    review = put(base/'review.json', dict(complete=True, passed=True,
        input_sha256={str(base/'config.json'):runner.sha(base/'config.json')}))
    dispatch = runner.read(base/'dispatch/plan.json')
    dispatch.update(review=review, input_sha256={str(base/'binding.json'):runner.sha(base/'binding.json')})
    ref = put(base/'dispatch/plan.json', dispatch)
    status = runner.read(base/'dispatch/status.json'); status['plan_sha256'] = ref['sha256']
    put(base/'dispatch/status.json', status)
    return base, root, config


def successful_observer(base, root, plan, *, exit_receipt=True):
    out = root/'analysis'; out.mkdir()
    chains = [dict(job=j, reused_control=False, pair_classifications=1,
                   metrics=dict(production_samples=4096)) for j in plan['new_jobs']]
    chains += [dict(job=j, reused_control=True, new_geometry_queries=0,
                    metrics=dict(production_samples=4096)) for j in plan['control_jobs']]
    for job in plan['new_jobs']:
        put(out/f"job-{job['id']:03}-observations.jsonl", b'inert synthetic observation bytes\n')
    scientific_plan = runner.read(base/'analysis-plan.json')
    put(out/'input-binding.json', dict(input_sha256=plan['required_analysis_inputs'], plan=scientific_plan))
    put(out/'analysis.json', dict(schema='two-neighbor-singleton-analysis-v1', complete=True,
        new_chains=64, reused_control_chains=64, chains=chains, new_geometry_endpoints=294976,
        new_physical_draws=0, old_geometry_queries=0, native_observer=False,
        analysis_plan=scientific_plan, input_sha256=plan['required_analysis_inputs']))
    put(out/'manifest.json', dict(complete=True, files={p.name:runner.sha(p) for p in out.iterdir()}))
    if exit_receipt:
        put(root/'exit.json', dict(child_started=True, child_drained=True, returncode=0, error=None))


class ObserverWrapperTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base, self.root, self.config = fixture(Path(self.temp.name))
        # The production controller's separate authority validator is already
        # tested. Its returned config here is synthetic; complete input/cache
        # admission and all wrapper source/hash checks remain real and unmocked.
        self.authority = mock.patch.object(runner.campaign, 'review_inputs', return_value=self.config)
        self.authority.start()

    def tearDown(self): self.authority.stop(); self.temp.cleanup()

    def prepare(self): return runner.prepare(self.base, self.root)

    def test_complete_metadata_prepare_and_exact_inventory(self):
        with mock.patch.object(runner.analyzer.previous, 'ConditionalObserver', side_effect=AssertionError('No geometry')):
            plan = self.prepare(); runner.verify_plan(self.root, plan)
        self.assertEqual(len(plan['new_jobs']), 64); self.assertEqual(len(plan['control_jobs']), 64)
        self.assertEqual(sum(p.endswith('/trajectory.jsonl') for p in plan['files']), 64)
        self.assertEqual(sum(p.endswith('-observations.jsonl') for p in plan['files']), 64)
        self.assertEqual({j['arm'] for j in plan['control_jobs']}, {'local','m4'})
        self.assertEqual((plan['maximum_workers'],plan['cpu_limit_seconds'],plan['wall_limit_seconds']), (1,3600,7200))
        self.assertFalse((self.root/'claim.json').exists())

    def test_unfinished_failed_undrained_inventory_rejected_before_output_directory(self):
        path = self.base/'dispatch/status.json'; original = runner.read(path)
        changes = [lambda s:s.update(complete=False), lambda s:s.update(passed=False),
            lambda s:s.update(failure='failed'), lambda s:s.update(failure_draining=True),
            lambda s:s.update(active=[{}]), lambda s:s.update(unstarted=[{}]),
            lambda s:s['completed'].pop(), lambda s:s['completed'][0].update(child_drained=False),
            lambda s:s['completed'].__setitem__(1,s['completed'][0])]
        for i, mutate in enumerate(changes):
            changed = copy.deepcopy(original); mutate(changed); put(path,changed)
            with self.subTest(change=i), self.assertRaises(ValueError): self.prepare()
            self.assertFalse(self.root.exists())
        put(path, original)

    def test_changed_journal_or_cached_control_is_not_reused(self):
        path = self.base/'execution/job-000/trajectory.jsonl'; original = path.read_bytes()
        path.write_bytes(original+b'changed\n')
        with self.assertRaises(ValueError):self.prepare()
        self.assertFalse(self.root.exists());path.write_bytes(original)
        cache = Path(next(iter(self.config['control_analysis']['observations'].values()))['path'])
        cache.write_bytes(b'changed cached bytes\n')
        with self.assertRaises(ValueError):self.prepare()
        self.assertFalse(self.root.exists())

    def test_scope_command_and_missing_binding_cannot_launch(self):
        plan = self.prepare()
        for key, value in [('cpu_limit_seconds',3601),('wall_limit_seconds',7201),('maximum_workers',2),
            ('new_chains',63),('new_physical_draws',1),('old_geometry_queries',1),('retries',1),
            ('replacements',1),('native_observer',True)]:
            with self.subTest(key=key),self.assertRaises(ValueError):runner.verify_plan(self.root,dict(plan,**{key:value}))
        changed = copy.deepcopy(plan);changed['argv'][-1]=str(self.root/'wrong')
        with self.assertRaisesRegex(ValueError,'command'):runner.verify_plan(self.root,changed)
        changed = copy.deepcopy(plan);del changed['files'][str(self.base/'config.json')]
        with self.assertRaisesRegex(ValueError,'admission input'):runner.verify_plan(self.root,changed)
        changed = copy.deepcopy(plan);changed['control_jobs'][0]['id']+=1000
        with self.assertRaisesRegex(ValueError,'admission input'):runner.verify_plan(self.root,changed)

    def test_bound_source_mutation_rejected_before_claim(self):
        self.prepare();source=self.root/'code'/runner.NAME
        source.write_bytes(source.read_bytes()+b'\n# modified\n')
        with mock.patch.object(runner.launcher,'owned_child') as child:
            with self.assertRaisesRegex(ValueError,'wrapper source'):runner.run(self.root)
            child.assert_not_called()
        self.assertFalse((self.root/'claim.json').exists())

    def test_success_once_and_exclusive_claim(self):
        plan=self.prepare()
        with mock.patch.object(runner.launcher,'owned_child',side_effect=lambda *args:successful_observer(self.base,self.root,plan)) as child:
            runner.run(self.root)
            self.assertEqual(child.call_args.args[3:],(3600,7200,16*1024**3))
            self.assertEqual(child.call_count,1)
            with self.assertRaises(FileExistsError):runner.run(self.root)
            self.assertEqual(child.call_count,1)
        summary=runner.read(self.root/'summary.json')
        self.assertTrue(summary['passed']);self.assertFalse(summary['assembly_gate_open'])
        self.assertEqual(summary['analysis']['sha256'],runner.sha(self.root/'analysis/analysis.json'))

    def test_output_checks_control_ids_zero_queries_and_bounded_samples(self):
        plan=self.prepare();successful_observer(self.base,self.root,plan)
        path=self.root/'analysis/analysis.json';original=runner.read(path)
        changes=[lambda r:r.update(new_geometry_endpoints=294975),
            lambda r:r['chains'][64]['job'].update(id=999),
            lambda r:r['chains'].__setitem__(65,copy.deepcopy(r['chains'][64])),
            lambda r:r['chains'][64].update(new_geometry_queries=1),
            lambda r:r['chains'][0]['metrics'].update(production_samples=4095),
            lambda r:r['chains'][0].update(pair_classifications=4609*525+1),
            lambda r:r.update(input_sha256={}),lambda r:r.update(native_observer=True)]
        for i, mutate in enumerate(changes):
            changed=copy.deepcopy(original);mutate(changed);put(path,changed)
            with self.subTest(change=i),self.assertRaises(ValueError):runner.verify_output(self.base,self.root,plan)

    def test_extra_prepared_start_inputs_must_remain_in_frozen_closure(self):
        plan=self.prepare()
        start=put(self.base/'synthetic-prepared/start.json',dict(selected='inert synthetic start metadata'))
        # The analyzer loads starts after bind_complete_inputs, so these refs
        # belong to the complete frozen closure, beyond required_analysis_inputs.
        plan['files'][start['path']]=start['sha256']
        successful_observer(self.base,self.root,plan)
        out=self.root/'analysis';result=runner.read(out/'analysis.json')
        result['input_sha256'][start['path']]=start['sha256'];put(out/'analysis.json',result)
        binding=runner.read(out/'input-binding.json');binding['input_sha256']=result['input_sha256'];put(out/'input-binding.json',binding)
        put(out/'manifest.json',dict(complete=True,files={p.name:runner.sha(p) for p in out.iterdir() if p.name!='manifest.json'}))
        runner.verify_output(self.base,self.root,plan)
        result['input_sha256'][start['path']]='0'*64;put(out/'analysis.json',result)
        with self.assertRaisesRegex(ValueError,'unbound input'):runner.verify_output(self.base,self.root,plan)

    def test_failed_verification_retains_partial_outputs_without_success(self):
        plan=self.prepare()
        def invalid(*args):
            successful_observer(self.base,self.root,plan)
            path=self.root/'analysis/analysis.json';r=runner.read(path);r['new_geometry_endpoints']=1;put(path,r)
        with mock.patch.object(runner.launcher,'owned_child',side_effect=invalid) as child:
            with self.assertRaisesRegex(ValueError,'observer result'):runner.run(self.root)
            with self.assertRaises(FileExistsError):runner.run(self.root)
            self.assertEqual(child.call_count,1)
        self.assertTrue((self.root/'analysis/analysis.json').exists())
        self.assertTrue(runner.read(self.root/'failure.json')['partial_outputs_retained'])
        self.assertFalse((self.root/'summary.json').exists())

    def test_real_owned_child_timeout_drains_and_is_not_retried(self):
        self.prepare();owned=runner.launcher.owned_child
        def timeout(argv,cwd,directory,*limits):
            return owned([sys.executable,'-c','import time; time.sleep(30)'],cwd,directory,2,.05,16*1024**3)
        with mock.patch.object(runner.launcher,'owned_child',side_effect=timeout) as child:
            with self.assertRaises(subprocess.TimeoutExpired):runner.run(self.root)
            with self.assertRaises(FileExistsError):runner.run(self.root)
            self.assertEqual(child.call_count,1)
        result=runner.read(self.root/'exit.json')
        self.assertTrue(result['child_drained']);self.assertEqual(result['returncode'],-signal.SIGKILL)
        self.assertTrue((self.root/'failure.json').exists());self.assertFalse((self.root/'summary.json').exists())

    def test_real_child_receives_exact_limits_and_single_thread_environment(self):
        plan=self.prepare();owned=runner.launcher.owned_child
        script="import os,resource; assert resource.getrlimit(resource.RLIMIT_CPU)==(3600,3601); assert resource.getrlimit(resource.RLIMIT_AS)==(16*1024**3,16*1024**3); assert all(os.environ[k]=='1' for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','RAYON_NUM_THREADS'))"
        def invoke(argv,cwd,directory,*limits):
            owned([sys.executable,'-c',script],cwd,directory,*limits)
            successful_observer(self.base,self.root,plan,exit_receipt=False)
        with mock.patch.object(runner.launcher,'owned_child',side_effect=invoke):runner.run(self.root)
        self.assertTrue(runner.read(self.root/'summary.json')['passed'])
        self.assertTrue(runner.read(self.root/'exit.json')['child_drained'])

    def test_signal_failure_restores_handlers_and_keeps_claim(self):
        self.prepare();previous=signal.getsignal(signal.SIGTERM)
        def stop(*args):os.kill(os.getpid(),signal.SIGTERM)
        with mock.patch.object(runner.launcher,'owned_child',side_effect=stop):
            with self.assertRaises(SystemExit):runner.run(self.root)
        self.assertEqual(signal.getsignal(signal.SIGTERM),previous)
        self.assertTrue((self.root/'claim.json').exists());self.assertTrue((self.root/'failure.json').exists())
        self.assertFalse((self.root/'summary.json').exists())


if __name__=='__main__':unittest.main()
