"""No protein sampling: synthetic child scheduler, output and frozen-gate tests."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from analyze_r4_smc_control import Ledger, sha, write
import run_full_vessel_comparison as runner


class Child:
    def __init__(self, pid, polls, final=0):
        self.pid, self.polls, self.final, self.waits = pid, list(polls), final, 0
    def poll(self): return self.polls.pop(0) if self.polls else self.final
    def wait(self): self.waits+=1; return self.final


def items(root, n=3, kind='physical'):
    return [dict(id=str(i), kind=kind, directory=str(root/('output'+str(i))),
        log=str(root/('log'+str(i))), command=['synthetic',str(i)], status='pending',returncode=None) for i in range(n)]


class VesselRunnerTests(unittest.TestCase):
    def execute(self, steps, children, capacity=None, workers=2, snapshot=None):
        calls=[]; snapshots=[]; paused=[]
        def launch(command, **kwargs):
            calls.append((command, kwargs))
            value=children[len(calls)-1]
            if isinstance(value, BaseException): raise value
            return value
        def record():
            snapshots.append(copy.deepcopy(steps))
            if snapshot: snapshot()
        runner.execute_group(steps,record,workers,'/synthetic/repo',{'OMP_NUM_THREADS':'1'},
            popen=launch,pause=lambda delay:paused.append(delay),
            capacity=capacity or (lambda _:dict(physical_pids=[],workers=0)),birth=lambda pid:str(100+pid))
        return calls,snapshots,paused

    def test_all_jobs_complete_with_birth_tokens_and_closed_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            steps=items(Path(tmp)); children=[Child(i,[None,0]) for i in range(3)]
            calls,snapshots,_=self.execute(steps,children)
            self.assertEqual(len(calls),3)
            self.assertTrue(all(s['status']=='complete' and s['returncode']==0 for s in steps))
            self.assertTrue(all(s['process_birth']==str(100+s['pid']) for s in steps))
            self.assertTrue(all(call[1]['stdout'].closed for call in calls))
            self.assertLessEqual(max(sum(s['status']=='running' for s in snap) for snap in snapshots),2)

    def test_poll_failure_before_refill_and_drain_other_started_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            steps=items(Path(tmp)); first=Child(11,[None,1],1); second=Child(12,[None,None],0)
            with self.assertRaisesRegex(RuntimeError,'Child failed'):
                self.execute(steps,[first,second])
            self.assertEqual([s['status'] for s in steps],['failed','complete','not_started'])
            self.assertEqual(second.waits,1)
            self.assertFalse(Path(steps[2]['log']).exists())

    def test_spawn_exception_drains_started_child_preserves_every_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); steps=items(root); child=Child(1,[None,None],0)
            with self.assertRaisesRegex(OSError,'cannot spawn'):
                self.execute(steps,[child,OSError('cannot spawn')])
            self.assertEqual(child.waits,1)
            self.assertEqual([s['status'] for s in steps],['complete','not_started','not_started'])
            self.assertTrue(Path(steps[0]['log']).exists());self.assertTrue(Path(steps[1]['log']).exists())

    def test_snapshot_failure_after_spawn_still_drains(self):
        with tempfile.TemporaryDirectory() as tmp:
            steps=items(Path(tmp)); child=Child(1,[None],0); count=0
            def snapshot():
                nonlocal count
                count+=1
                if count==1: raise RuntimeError('snapshot failed')
            with self.assertRaisesRegex(RuntimeError,'snapshot failed'):
                self.execute(steps,[child],snapshot=snapshot)
            self.assertEqual(child.waits,1)
            self.assertEqual(steps[0]['status'],'complete')
            self.assertEqual(steps[1]['status'],'not_started')

    def test_capacity_is_checked_at_each_launch_for_both_budgets(self):
        with tempfile.TemporaryDirectory() as tmp:
            steps=items(Path(tmp),1); values=[dict(physical_pids=list(range(8)),workers=8),
                dict(physical_pids=[],workers=32),dict(physical_pids=[],workers=31),dict(physical_pids=[],workers=31)]
            seen=[]
            def capacity(_): seen.append(1);return values.pop(0)
            calls,_,paused=self.execute(steps,[Child(1,[0])],capacity=capacity)
            self.assertEqual(len(calls),1);self.assertEqual(len(seen),4);self.assertEqual(len(paused),2)

    def test_existing_output_and_log_refuse_without_overwrite(self):
        for mode in ('directory','log'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as tmp:
                steps=items(Path(tmp),1);path=Path(steps[0][mode])
                if mode=='directory':path.mkdir();(path/'partial').write_text('keep')
                else:path.write_text('keep')
                with self.assertRaises((ValueError,FileExistsError)): self.execute(steps,[])
                self.assertEqual((path/'partial' if mode=='directory' else path).read_text(),'keep')

    def test_failed_gate_creates_one_shot_failure_without_any_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            with (mock.patch.object(runner,'validate',side_effect=ValueError('regional checks failed')),
                  mock.patch.object(runner.subprocess,'Popen') as launch):
                with self.assertRaisesRegex(ValueError,'regional checks failed'):runner.run(out,'a'*64)
                launch.assert_not_called()
                state=json.loads((out/'status.json').read_text())
                self.assertEqual(state['phase'],'failed');self.assertEqual(state['physics_launched'],0)
                with self.assertRaises(FileExistsError):runner.run(out,'a'*64)

    def test_stage_order_does_not_depend_on_scientific_pass_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); prep=root/'prep';out=root/'workflow';prep.mkdir();out.mkdir()
            steps=items(root,1); steps[0]['kind']='aggregate'
            plan=dict(preparation=dict(path=str(prep)),repository=str(root),stages=[
                dict(name=stage,groups=[dict(kind='aggregate',workers=1,steps=[dict(steps[0],id=stage,directory=str(root/stage),log=str(root/(stage+'.log')))])])
                for stage in ('standard','large')])
            calls=[]
            def group(jobs,snapshot,*args,**kwargs):
                for job in jobs:Path(job['log']).write_text('synthetic');job.update(status='complete',returncode=0)
                calls.append(jobs[0]['id']);snapshot()
            with (mock.patch.object(runner,'validate',return_value=(plan,{},Ledger(),{'regional_checks_passed':True})),
                  mock.patch.object(runner,'execute_group',side_effect=group),
                  mock.patch.object(runner,'verify_step',return_value={'diagnostic_passed':False})):
                result=runner.run(out,'a'*64)
            self.assertEqual(calls,['standard','large']);self.assertTrue(result['complete'])
            self.assertEqual(result['physics_launched'],0)

    def test_failed_standard_execution_never_starts_large(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);prep=root/'prep';out=root/'workflow';prep.mkdir();out.mkdir()
            plan=dict(preparation=dict(path=str(prep)),repository=str(root),stages=[dict(name=stage,
                groups=[dict(kind='physical',workers=1,steps=[dict(items(root,1)[0],id=stage,directory=str(root/stage),log=str(root/(stage+'.log')))])])
                for stage in ('standard','large')])
            with (mock.patch.object(runner,'validate',return_value=(plan,{},Ledger(),{})),
                  mock.patch.object(runner,'execute_group',side_effect=RuntimeError('physical failed')) as execute):
                with self.assertRaises(RuntimeError):runner.run(out,'a'*64)
            self.assertEqual(execute.call_count,1)
            self.assertEqual(json.loads((out/'status.json').read_text())['phase'],'failed')


    def test_matching_target_requires_every_physical_and_classifier_identity(self):
        prep=dict(input_sha256={'shape.json':'shape','current_R4.json':'r4','old_alternative_R5.json':'r5'},native_definition_sha256='native')
        confirmation=dict(shape_sha256='shape',region_sha256='r4',reference_region_sha256='r5',native_definition=dict(definition_sha256='native'))
        config=dict(depletant_radius=1.5,reservoir_density=.035,fixed_poses=['a','b'],metadata={'q':'frozen'})
        runner.match_target(confirmation,prep,config,config)
        for key in ('shape_sha256','region_sha256','reference_region_sha256','native_definition'):
            bad=copy.deepcopy(confirmation)
            if key=='native_definition':bad[key]['definition_sha256']='other'
            else:bad[key]='other'
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'different shape'):
                runner.match_target(bad,prep,config,config)
        for key in config:
            bad=copy.deepcopy(config);bad[key]=None
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'bath, scaffold'):
                runner.match_target(confirmation,prep,bad,config)

    def test_prelaunch_input_failure_drains_without_second_spawn(self):
        with tempfile.TemporaryDirectory() as tmp:
            steps=items(Path(tmp));child=Child(1,[None],0);launches=[];checks=[]
            def launch(*args,**kwargs):launches.append(1);return child
            def check():
                checks.append(1)
                if len(checks)==2:raise ValueError('input changed')
            with self.assertRaisesRegex(ValueError,'input changed'):
                runner.execute_group(steps,lambda:None,2,'/synthetic',{},popen=launch,pause=lambda _:None,
                    capacity=lambda _:dict(physical_pids=[],workers=0),birth=lambda _:None,before_launch=check)
            self.assertEqual(len(launches),1);self.assertEqual(child.waits,1)
            self.assertEqual(steps[1]['status'],'not_started')

    def test_standard_output_validation_failure_never_starts_large(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);prep=root/'prep';out=root/'workflow';prep.mkdir();out.mkdir()
            plan=dict(preparation=dict(path=str(prep)),repository=str(root),stages=[dict(name=stage,
                groups=[dict(kind='physical',workers=1,steps=[dict(items(root,1)[0],id=stage,directory=str(root/stage),log=str(root/(stage+'.log')))])])
                for stage in ('standard','large')])
            with (mock.patch.object(runner,'validate',return_value=(plan,{},Ledger(),{})),
                  mock.patch.object(runner,'execute_group') as execute,
                  mock.patch.object(runner,'verify_step',side_effect=ValueError('wrong source'))):
                with self.assertRaisesRegex(ValueError,'wrong source'):runner.run(out,'a'*64)
            self.assertEqual(execute.call_count,1)
            self.assertEqual(json.loads((out/'status.json').read_text())['phase'],'failed')


if __name__=='__main__':unittest.main()
