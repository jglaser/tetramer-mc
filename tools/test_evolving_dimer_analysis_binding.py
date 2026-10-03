"""Synthetic all-96 metadata binding checks; no observers or physical draws."""
import copy
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import unittest
from unittest import mock

import run_evolving_dimer_analysis as runner


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def bound(path):
    return dict(path=str(Path(path).resolve()), sha256=runner.sha(path))


def fixture(root):
    """Populate every metadata edge expected by campaign_bindings, with tiny files."""
    base = root/'campaign'
    (base/'dispatch').mkdir(parents=True)
    (base/'common/source/tools').mkdir(parents=True)
    analyzer = base/'common/source/tools/analyze_evolving_dimer_benchmark.py'
    analyzer.write_text('# Synthetic placeholder. Never executed.\n')
    dependency = base/'common/source/tools/synthetic_dependency.py'
    dependency.write_text('# Bound transitive fixture dependency.\n')
    (base/'common/evolving_dimer_benchmark').write_text('Synthetic executable placeholder.\n')
    (base/'dispatch/run.py').write_text('# Synthetic dispatch source.\n')
    jobs = [dict(id=k, context_index=c, arm=a, initialization=i, stream=s)
        for k, (c,a,i,s) in enumerate((c,a,i,s) for c in range(4)
            for a in ('local','unguided','m4') for i in ('source','proposal_prepared') for s in range(4))]
    raw = base/'common/input.txt'; raw.write_text('Fixture input, no geometry.\n')
    config = dict(jobs=jobs, output=str(base/'run'),
        allocation=dict(blocks_per_chain=4608, warmup_blocks=512, production_blocks=4096),
        synthetic_input=bound(raw))
    save(base/'config.json', config)
    protocol = dict(source_files={str(p.relative_to(base/'common/source')):runner.sha(p)
        for p in [analyzer,dependency]})
    save(base/'protocol.json', protocol)
    prepared_file = base/'common/prepared.txt'; prepared_file.write_text('No sampled poses.\n')
    prepared = dict(complete=True, passed=True, all_attempts_retained=True, synthetic_record=bound(prepared_file))
    save(base/'prepared.json', prepared)
    binding = dict(config_sha256=runner.sha(base/'config.json'), protocol_sha256=runner.sha(base/'protocol.json'),
        prepared_files={str(prepared_file):runner.sha(prepared_file)}, prepared_manifest=bound(base/'prepared.json'))
    save(base/'run-binding.json', binding)
    review = dict(complete=True, passed=True, input_sha256={str(raw):runner.sha(raw)})
    save(base/'review.json', review)
    freeze = dict(complete=True, scientific_execution_started=False,
        files={name:runner.sha(base/name) for name in ['config.json','protocol.json','common/input.txt']},
        original_input_bindings=dict(source=bound(raw)))
    save(base/'freeze.json', freeze)
    dispatch = dict(jobs=jobs, config_sha256=runner.sha(base/'config.json'),
        binding_sha256=runner.sha(base/'run-binding.json'),
        executable_sha256=runner.sha(base/'common/evolving_dimer_benchmark'),
        source_sha256=runner.sha(base/'dispatch/run.py'), review=bound(base/'review.json'))
    save(base/'dispatch/plan.json', dispatch)
    terminals = []
    for job in jobs:
        directory = base/'run'/f"job-{job['id']:03}"
        directory.mkdir(parents=True)
        trajectory = directory/'trajectory.jsonl'
        trajectory.write_text(json.dumps(dict(synthetic=True, job=job['id']))+'\n')
        terminal = dict(complete=True, job=job, conditional_target=True, blocks=4608,
            config_sha256=runner.sha(base/'config.json'),binding_sha256=runner.sha(base/'run-binding.json'),
            trajectory=bound(trajectory))
        save(directory/'terminal.json', terminal)
        terminals.append(dict(job=job, success=True, returncode=0, terminal_error=None,
            terminal_sha256=runner.sha(directory/'terminal.json')))
    status = dict(complete=True, passed=True, failure_draining=False, active=[], unstarted=[],
        completed=terminals,plan_sha256=runner.sha(base/'dispatch/plan.json'))
    save(base/'dispatch/status.json', status)
    return base, config


def fake_success(base, output, jobs):
    """Write observer-shaped result artifacts; no observation computation."""
    output.mkdir()
    result = dict(complete=True, chains=[dict(job=job,metrics=dict(production_samples=4096)) for job in jobs])
    save(output/'analysis.json', result)
    save(output/'input-binding.json', dict(synthetic=True))
    for job in jobs:
        (output/f"job-{job['id']:03}-observations.jsonl").write_text('{"synthetic":true}\n')
    save(output/'manifest.json',dict(complete=True,files={str(p.relative_to(output)):runner.sha(p)
        for p in output.iterdir() if p.is_file()}))


class AnalysisBindingTests(unittest.TestCase):
    def test_launch_window_sigterm_is_deferred_until_child_can_be_drained(self):
        """Deliver SIGTERM after fork succeeds but before owned_child's assignment."""
        prior_mask = signal.pthread_sigmask(signal.SIG_BLOCK, set())
        if signal.SIGTERM in prior_mask:
            self.skipTest('Test requires SIGTERM initially unblocked')
        real_popen = runner.subprocess.Popen
        children = []
        previous_handler = signal.signal(signal.SIGTERM, runner.terminate_requested)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp)
                def signal_before_assignment(*args, **kwargs):
                    child = real_popen(*args, **kwargs)
                    children.append(child)
                    self.assertIn(signal.SIGTERM,
                        signal.pthread_sigmask(signal.SIG_BLOCK, set()))
                    os.kill(os.getpid(), signal.SIGTERM)
                    # The signal is pending here; delivery before this return
                    # would leave the caller without the child handle.
                    self.assertIn(signal.SIGTERM, signal.sigpending())
                    return child
                with mock.patch.object(runner.subprocess, 'Popen', side_effect=signal_before_assignment):
                    with self.assertRaisesRegex(SystemExit, 'received signal 15'):
                        runner.owned_child([sys.executable, '-c', 'import time; time.sleep(30)'],
                            tmp, path, 5, 5)
                self.assertEqual(len(children), 1)
                self.assertEqual(children[0].returncode, -signal.SIGKILL)
                with self.assertRaises(ChildProcessError):
                    os.waitpid(children[0].pid, os.WNOHANG)
                result = runner.read(path/'exit.json')
                self.assertTrue(result['child_started'] and result['child_drained'])
                self.assertEqual(result['returncode'], -signal.SIGKILL)
                self.assertIn('received signal 15', result['error'])
                self.assertEqual(signal.pthread_sigmask(signal.SIG_BLOCK, set()), prior_mask)
                self.assertNotIn(signal.SIGTERM, signal.sigpending())
        finally:
            # Preserve test-runner state, and drain even if an assertion exposed
            # a regression in the controller's ownership window.
            for child in children:
                if child.poll() is None:
                    runner.drain(child)
            signal.signal(signal.SIGTERM, previous_handler)
            signal.pthread_sigmask(signal.SIG_SETMASK, prior_mask)

    def test_full_synthetic_metadata_prepare_verify_and_successful_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            base,config = fixture(Path(tmp)); root=Path(tmp)/'analysis-execution'
            plan=runner.prepare(base,root,cpu_seconds=5,wall_seconds=10)
            self.assertEqual(sum(p.endswith('/trajectory.jsonl') for p in plan['files']),96)
            self.assertIn(str(base/'common/source/tools/synthetic_dependency.py'),plan['files'])
            runner.verify_plan(root,plan)
            with mock.patch.object(runner,'owned_child',side_effect=lambda *a,**k:fake_success(base,root/'analysis',config['jobs'])) as child:
                runner.run(root)
                self.assertEqual(child.call_count,1)
                self.assertEqual(child.call_args.args[0],plan['argv'])
            summary=runner.read(root/'summary.json')
            self.assertTrue(summary['complete'] and summary['passed'])
            self.assertFalse(summary['assembly_gate_open'])
            self.assertEqual(summary['plan_sha256'],runner.read(root/'claim.json')['plan_sha256'])
            with mock.patch.object(runner,'owned_child') as child:
                with self.assertRaises(FileExistsError):runner.run(root)
                child.assert_not_called()

    def test_omitted_or_changed_bound_file_and_scope_fields_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            base,_ = fixture(Path(tmp)); root=Path(tmp)/'analysis-execution';plan=runner.prepare(base,root)
            source=str(base/'common/source/tools/synthetic_dependency.py')
            for change in ('omit','digest'):
                altered=copy.deepcopy(plan)
                if change=='omit':del altered['files'][source]
                else:altered['files'][source]='0'*64
                with self.subTest(change=change),self.assertRaises(ValueError):runner.verify_plan(root,altered)
            changes=dict(partial_analysis_allowed=True,restart=True,new_sampling_attempts=1,
                expected_chains=95,expected_retained_endpoints=442463,expected_production_endpoints=393215,
                maximum_workers=2,address_space_limit_bytes=17*1024**3,cpu_limit_seconds=86401,
                wall_limit_seconds=172801)
            for name,value in changes.items():
                with self.subTest(field=name),self.assertRaises(ValueError):runner.verify_plan(root,dict(plan,**{name:value}))
            bad=copy.deepcopy(plan);bad['argv'][0]='/not-the-bound-python';bad['files'][bad['argv'][0]]='0'*64
            with self.assertRaisesRegex(ValueError,'interpreter'):runner.verify_plan(root,bad)
            bad=copy.deepcopy(plan);bad['argv'][-1]=str(root/'other-output')
            with self.assertRaisesRegex(ValueError,'command'):runner.verify_plan(root,bad)

    def test_changed_terminal_failure_and_trajectory_prevent_preparation(self):
        for mode in ('terminal','failure','trajectory'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as tmp:
                base,_=fixture(Path(tmp));directory=base/'run/job-000';output=Path(tmp)/'analysis-execution'
                if mode=='terminal':
                    value=runner.read(directory/'terminal.json');value['blocks']=1;save(directory/'terminal.json',value)
                elif mode=='failure':save(directory/'failure.json',dict(complete=False))
                else:(directory/'trajectory.jsonl').write_text('Changed synthetic data\n')
                with self.assertRaises(ValueError):runner.prepare(base,output)
                self.assertFalse(output.exists())

    def test_mutated_plan_during_child_is_retained_as_failure_and_never_retried(self):
        with tempfile.TemporaryDirectory() as tmp:
            base,config=fixture(Path(tmp));root=Path(tmp)/'analysis-execution';runner.prepare(base,root)
            def change_plan(*args,**kwargs):
                fake_success(base,root/'analysis',config['jobs'])
                value=runner.read(root/'execution-plan.json');value['scope']='Changed after launch'
                save(root/'execution-plan.json',value)
            with mock.patch.object(runner,'owned_child',side_effect=change_plan) as child:
                with self.assertRaisesRegex(ValueError,'Plan changed'):runner.run(root)
                self.assertEqual(child.call_count,1)
            self.assertFalse((root/'summary.json').exists())
            self.assertTrue((root/'analysis/analysis.json').exists())
            failure=runner.read(root/'failure.json');self.assertFalse(failure['complete'])
            self.assertTrue(failure['partial_outputs_retained']);self.assertEqual(failure['retries'],0)
            with mock.patch.object(runner,'owned_child') as child:
                with self.assertRaises(FileExistsError):runner.run(root)
                child.assert_not_called()

    def test_duplicate_analysis_jobs_and_bad_result_manifest_are_not_success(self):
        for mode in ('duplicate','manifest_digest','missing_chain'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as tmp:
                base,config=fixture(Path(tmp));root=Path(tmp)/'analysis-execution';runner.prepare(base,root)
                def bad_result(*args,**kwargs):
                    fake_success(base,root/'analysis',config['jobs'])
                    p=root/'analysis/analysis.json';value=runner.read(p)
                    if mode=='duplicate':value['chains'][-1]['job']=value['chains'][0]['job'];save(p,value)
                    elif mode=='missing_chain':value['chains'].pop();save(p,value)
                    else:
                        m=root/'analysis/manifest.json';value=runner.read(m);value['files']['analysis.json']='0'*64;save(m,value)
                with mock.patch.object(runner,'owned_child',side_effect=bad_result):
                    with self.assertRaises(ValueError):runner.run(root)
                self.assertFalse((root/'summary.json').exists());self.assertTrue((root/'failure.json').exists())


if __name__=='__main__':unittest.main()
