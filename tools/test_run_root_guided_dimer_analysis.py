"""Synthetic admission/controller checks; no scientific observer or atom queries."""
import copy
import json
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import run_root_guided_dimer_analysis as runner


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value)+'\n')


def fixture(parent):
    base=parent/'campaign';audit=parent/'guidance';root=parent/'observer'
    base.mkdir();audit.mkdir()
    jobs=[dict(id=3*n+5,arm='root_m4',context_index=c,initialization=i,stream=s)
        for n,(c,i,s) in enumerate((c,i,s) for c in range(4) for i in ('source','proposal_prepared') for s in range(4))]
    config={'jobs':jobs}
    save(base/'config.json',config)
    observer_plan=dict(schema='root-guided-dimer-analysis-plan-v1',new_chains=32,reused_control_chains=32,
        new_retained_initial_observations=147488,new_production_observations=131072,
        maximum_new_pair_classifications=77431200,new_geometry_only=True,new_physical_draws=0,
        native_observer=False,complete_inventory_required=True)
    save(base/'analysis-plan.json',observer_plan)
    analyzer=base/'common/source/tools/analyze_root_guided_dimer_benchmark.py'
    analyzer.parent.mkdir(parents=True);analyzer.write_text('# Synthetic placeholder; never used as a scientific observer.\n')
    files={str(p):runner.sha(p) for p in (base/'config.json',base/'analysis-plan.json',analyzer)}
    items=[]
    for j in jobs:
        directory=base/'execution'/f"job-{j['id']:03}"
        save(directory/'terminal.json',{'complete':True,'job':j});save(directory/'trajectory.jsonl',{'synthetic':True})
        save(base/'controls'/f"job-{j['id']:03}.jsonl",{'synthetic_cached_control':True})
        for p in (directory/'terminal.json',directory/'trajectory.jsonl',base/'controls'/f"job-{j['id']:03}.jsonl"):
            files[str(p)]=runner.sha(p)
        items.append(dict(job=j,terminal=runner.record(directory/'terminal.json'),trajectory=runner.record(directory/'trajectory.jsonl')))
    current=dict(files=files,jobs=items,audit_protocol={'synthetic':True})
    plan=dict(base=str(base),files=files,jobs=items)
    save(audit/'execution-plan.json',plan);digest=runner.sha(audit/'execution-plan.json')
    save(audit/'attempts.jsonl',{'synthetic_completed_geometry_audit':True})
    result=dict(complete=True,passed=True,plan_sha256=digest,chains=[{'job':j} for j in jobs],
        scalar_dimer_attempts=147456,scalar_local_attempts=589824,independent_geometry_events=32,
        count_queries_started=12,count_queries_completed=12,point_membership_tests_started=100,
        new_poses=0,new_clouds=0,journal_sha256=runner.sha(audit/'attempts.jsonl'))
    save(audit/'audit.json',result)
    summary=dict(schema='evolving-dimer-root-guidance-audit-summary-v1',base=str(base),complete=True,passed=True,
        plan_sha256=digest,chains=32,scalar_dimer_attempts=147456,scalar_local_attempts=589824,
        independent_geometry_events=32,new_poses=0,new_clouds=0,count_queries=12,
        audit_sha256=runner.sha(audit/'audit.json'),journal_sha256=result['journal_sha256'])
    save(audit/'summary.json',summary)
    save(audit/'review.json',{'complete':True,'passed':True,'plan_sha256':digest})
    save(audit/'claim.json',{'plan_sha256':digest,'review_sha256':runner.sha(audit/'review.json')})
    save(audit/'child/exit.json',{'child_started':True,'child_drained':True,'returncode':0,'error':None})
    return base,audit,root,current


def successful_observer(base,out):
    out.mkdir();jobs=runner.read(base/'config.json')['jobs'];chains=[]
    for j in jobs:
        chains.append(dict(job=j,reused_control=False,pair_classifications=1,metrics={'production_samples':4096}))
        chains.append(dict(job=dict(j,arm='m4'),reused_control=True,new_geometry_queries=0,metrics={'production_samples':4096}))
        save(out/f"job-{j['id']:03}-observations.jsonl",{'synthetic':True})
    save(out/'input-binding.json',{'synthetic':True})
    save(out/'analysis.json',dict(schema='root-guided-dimer-analysis-v1',complete=True,new_chains=32,reused_control_chains=32,
        chains=chains,new_geometry_endpoints=147488,new_physical_draws=0,native_observer=False,
        analysis_plan=runner.read(base/'analysis-plan.json')))
    save(out/'manifest.json',{'complete':True,'files':{p.name:runner.sha(p) for p in out.iterdir()}})


class ObserverWrapperTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.base,self.audit,self.root,self.current=fixture(Path(self.temp.name))
        self.patch=mock.patch.object(runner.guidance,'campaign_bindings',return_value=self.current);self.patch.start()
        self.verify=mock.patch.object(runner.guidance,'verify_plan',create=True);self.verify.start()
    def tearDown(self):self.verify.stop();self.patch.stop();self.temp.cleanup()
    def prepare(self):return runner.prepare(self.base,self.root,self.audit)
    def test_metadata_prepare_full_admission_then_one_success_and_no_retry(self):
        plan=self.prepare();runner.verify_plan(self.root,plan)
        self.assertEqual(sum(p.endswith('/trajectory.jsonl') for p in plan['files']),32)
        self.assertEqual(sum('/controls/' in p for p in plan['files']),32)
        self.assertEqual(plan['maximum_workers'],1);self.assertEqual(plan['cpu_limit_seconds'],1800)
        with mock.patch.object(runner.launcher,'owned_child',side_effect=lambda *a,**k:successful_observer(self.base,self.root/'analysis')) as child:
            runner.run(self.root);self.assertEqual(child.call_count,1)
            self.assertEqual(child.call_args.args[3:],(1800,3600,16*1024**3))
            with self.assertRaises(FileExistsError):runner.run(self.root)
            self.assertEqual(child.call_count,1)
        s=runner.read(self.root/'summary.json');self.assertTrue(s['passed']);self.assertEqual(s['old_geometry_queries'],0)
    def test_incomplete_or_failed_guidance_admission_before_any_directory(self):
        p=self.audit/'summary.json';original=runner.read(p)
        for field,value in [('passed',False),('complete',False),('chains',31),('scalar_dimer_attempts',147455),
                ('scalar_local_attempts',589823),('independent_geometry_events',31),('base','wrong'),('plan_sha256','bad')]:
            with self.subTest(field=field):
                save(p,dict(original,**{field:value}))
                with self.assertRaises(ValueError):self.prepare()
                self.assertFalse(self.root.exists())
        save(p,original);save(self.audit/'worker-failure.json',{'failed':True})
        with self.assertRaisesRegex(ValueError,'Failed or interrupted'):self.prepare()
    def test_audit_inventory_and_count_budget_are_not_trusted_from_summary(self):
        p=self.audit/'audit.json';original=runner.read(p);summary=runner.read(self.audit/'summary.json')
        for field,value in [('chains',original['chains'][:-1]),('count_queries_completed',11),
                ('point_membership_tests_started',41943041),('new_poses',1)]:
            with self.subTest(field=field):
                save(p,dict(original,**{field:value}));save(self.audit/'summary.json',dict(summary,audit_sha256=runner.sha(p)))
                with self.assertRaises(ValueError):self.prepare()
                self.assertFalse(self.root.exists())
    def test_changed_omitted_input_and_scope_cannot_launch(self):
        plan=self.prepare();key=str(self.audit/'audit.json')
        for name,value in [('cpu_limit_seconds',1801),('wall_limit_seconds',3601),('maximum_workers',2),
                ('new_physical_draws',1),('old_geometry_queries',1),('retries',1),('new_chains',31)]:
            with self.subTest(name=name),self.assertRaises(ValueError):runner.verify_plan(self.root,dict(plan,**{name:value}))
        altered=copy.deepcopy(plan);del altered['files'][key]
        with self.assertRaisesRegex(ValueError,'admission input'):runner.verify_plan(self.root,altered)
        altered=copy.deepcopy(plan);altered['argv'][-1]=str(self.root/'wrong')
        with self.assertRaisesRegex(ValueError,'command'):runner.verify_plan(self.root,altered)
        (self.audit/'attempts.jsonl').write_text('changed\n')
        with mock.patch.object(runner.launcher,'owned_child') as child:
            with self.assertRaises(ValueError):runner.run(self.root)
            child.assert_not_called();self.assertFalse((self.root/'claim.json').exists())
    def test_wrong_observer_inventory_fails_and_preserves_outputs(self):
        self.prepare()
        def wrong(*a,**kw):
            successful_observer(self.base,self.root/'analysis')
            p=self.root/'analysis/analysis.json';r=runner.read(p);r['new_geometry_endpoints']=1;save(p,r)
        with mock.patch.object(runner.launcher,'owned_child',side_effect=wrong):
            with self.assertRaisesRegex(ValueError,'observer result'):runner.run(self.root)
        self.assertTrue((self.root/'analysis/analysis.json').exists());self.assertTrue((self.root/'failure.json').exists())
        self.assertFalse((self.root/'summary.json').exists())
    def test_real_owned_child_timeout_is_drained_failure_retained_no_retry(self):
        self.prepare();owned=runner.launcher.owned_child
        def timeout(argv,cwd,directory,*limits):
            return owned([sys.executable,'-c','import time; time.sleep(30)'],cwd,directory,2,.05,16*1024**3)
        with mock.patch.object(runner.launcher,'owned_child',side_effect=timeout) as child:
            with self.assertRaises(subprocess.TimeoutExpired):runner.run(self.root)
            with self.assertRaises(FileExistsError):runner.run(self.root)
            self.assertEqual(child.call_count,1)
        exit_status=runner.read(self.root/'exit.json');self.assertTrue(exit_status['child_drained'])
        self.assertEqual(exit_status['returncode'],-signal.SIGKILL);self.assertTrue((self.root/'failure.json').exists())
    def test_terminal_failure_during_admission_never_calls_observer(self):
        with mock.patch.object(runner.guidance,'campaign_bindings',side_effect=ValueError('All32 completed required')):
            with self.assertRaisesRegex(ValueError,'All32'):self.prepare()
        self.assertFalse(self.root.exists())

if __name__=='__main__':unittest.main()
