"""Synthetic controller lifecycle tests: no geometry, scientific binary or draws."""
import copy
import json
import os
from pathlib import Path
import signal
import shutil
import tempfile
import unittest
from unittest.mock import patch

import prepare_two_neighbor_singleton_benchmark as prep
import run_two_neighbor_singleton_benchmark as runner


class FakeChild:
    counter=10000000
    def __init__(self,argv,*,code=0,**kwargs):
        FakeChild.counter+=1;self.pid=FakeChild.counter;self.returncode=code
        config=json.loads(Path(argv[argv.index('--config')+1]).read_text())
        identity=int(argv[argv.index('--job')+1]);out=Path(config['output'])/f'job-{identity:03}'
        if code==0:
            out.mkdir(parents=True);prep.write(out/'terminal.json',dict(complete=True,conditional_target=True,blocks=4608,
                job=next(j for j in config['jobs'] if j['id']==identity),config_sha256=prep.sha(argv[argv.index('--config')+1]),
                binding_sha256=prep.sha(argv[argv.index('--binding')+1]),
                counts=dict(local_attempted=18432,singleton_attempted=4608,local_accepted=0,singleton_accepted=0,singleton_self_loop=4608)))
    def poll(self):return self.returncode
    def wait(self):return self.returncode


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.base=Path(self.tmp.name);(self.base/'dispatch').mkdir()
        self.jobs=prep.jobs()
        self.config={'jobs':self.jobs,'output':str(self.base/'execution')}
        prep.write(self.base/'config.json',self.config);prep.write(self.base/'review.json',{});prep.write(self.base/'run-binding.json',{})
        self.plan=dict(schema='two-neighbor-singleton-dispatch-v1',jobs=self.jobs,limits=prep.LIMITS,
            runtime=prep.runtime(),retries=False,replacements=False,workers=3,
            source=prep.record(runner.__file__),review=prep.record(self.base/'review.json'),
            input_sha256={str(self.base/p):prep.sha(self.base/p) for p in ('config.json','run-binding.json')})
        prep.write(self.base/'dispatch/plan.json',self.plan)
    def tearDown(self):self.tmp.cleanup()
    def run_controller(self,popen):
        with patch.object(runner,'review_inputs',return_value=self.config),patch.object(runner.subprocess,'Popen',side_effect=popen),patch.object(runner.time,'sleep'):
            runner.run(self.base)
    def test_complete_inventory_exclusive_no_retry(self):
        self.run_controller(FakeChild)
        status=prep.read(self.base/'dispatch/status.json')
        self.assertTrue(status['passed']);self.assertEqual([x['job'] for x in status['completed']],self.jobs)
        self.assertEqual(len(list((self.base/'dispatch').glob('*-begin.json'))),64)
        self.assertTrue(all(x['child_drained'] for x in status['completed']))
        with self.assertRaises(FileExistsError):self.run_controller(FakeChild)
    def test_failure_drains_all_owned_and_keeps_unstarted(self):
        children=[]
        def launch(argv,**kw):
            p=FakeChild(argv,code=1 if not children else None,**kw);children.append(p);return p
        def drain(p):p.returncode=-9
        with patch.object(runner,'drain',side_effect=drain) as d:
            with self.assertRaisesRegex(RuntimeError,'Chain failed'):self.run_controller(launch)
        s=prep.read(self.base/'dispatch/status.json');self.assertFalse(s['passed']);self.assertEqual(len(s['completed']),3)
        self.assertEqual(len(s['unstarted']),61);self.assertEqual(d.call_count,2);self.assertFalse(s['active'])
        self.assertTrue(all(p.returncode is not None for p in children))
    def test_launch_failure_drains_preexisting_and_preserves_failed_job(self):
        children=[]
        def launch(argv,**kw):
            if children:raise OSError('synthetic launch error')
            p=FakeChild(argv,code=None,**kw);children.append(p);return p
        def drain(p):p.returncode=-9
        with patch.object(runner,'drain',side_effect=drain):
            with self.assertRaisesRegex(OSError,'synthetic'):self.run_controller(launch)
        s=prep.read(self.base/'dispatch/status.json');self.assertEqual(len(s['completed']),2)
        self.assertEqual(len(s['unstarted']),62);self.assertTrue(all(x['child_drained'] for x in s['completed']))
    def test_launch_window_sigterm_is_owned_then_drained(self):
        before=signal.getsignal(signal.SIGTERM);children=[]
        def launch(argv,**kw):
            p=FakeChild(argv,code=None,**kw);children.append(p)
            os.kill(os.getpid(),signal.SIGTERM);return p
        def drain(p):p.returncode=-9
        with patch.object(runner,'drain',side_effect=drain):
            with self.assertRaisesRegex(SystemExit,'signal'):self.run_controller(launch)
        self.assertEqual(signal.getsignal(signal.SIGTERM),before)
        s=prep.read(self.base/'dispatch/status.json');self.assertEqual(len(s['completed']),1)
        self.assertEqual(len(s['unstarted']),63);self.assertTrue(s['completed'][0]['child_drained'])
    def test_wall_timeout_and_scope_tampering(self):
        def launch(argv,**kw):return FakeChild(argv,code=None,**kw)
        def drain(p):p.returncode=-9
        with patch.object(runner.time,'monotonic',side_effect=[0.,0.,0.,7201.]),patch.object(runner,'drain',side_effect=drain):
            with self.assertRaisesRegex(ValueError,'wall limit'):self.run_controller(launch)
        s=prep.read(self.base/'dispatch/status.json');self.assertEqual(len(s['completed']),3)
        self.assertFalse(s['active']);self.assertFalse(s['passed'])
    def test_changed_input_fails_before_claim_or_launch(self):
        (self.base/'config.json').write_text('{}')
        with patch.object(runner.subprocess,'Popen') as p,patch.object(runner,'review_inputs',return_value=self.config):
            with self.assertRaisesRegex(ValueError,'input changed'):runner.run(self.base)
        p.assert_not_called();self.assertFalse((self.base/'dispatch/claimed.json').exists())
    def test_wrong_terminal_is_failed_and_no_replacements(self):
        def launch(argv,**kw):
            p=FakeChild(argv,**kw);identity=int(argv[argv.index('--job')+1])
            terminal=Path(self.config['output'])/f'job-{identity:03}'/'terminal.json'
            v=prep.read(terminal);v['config_sha256']='wrong';terminal.write_text(json.dumps(v));return p
        with patch.object(runner,'drain',side_effect=lambda p:p.wait()):
            with self.assertRaisesRegex(RuntimeError,'Chain failed'):self.run_controller(launch)
        s=prep.read(self.base/'dispatch/status.json');self.assertFalse(s['passed']);self.assertEqual(len(s['unstarted']),61)
    def test_begin_write_failure_keeps_job_unstarted(self):
        actual=runner.write
        def broken(p,v):
            if str(p).endswith('-begin.json'):raise OSError('begin disk failure')
            return actual(p,v)
        with patch.object(runner,'write',side_effect=broken),patch.object(runner.subprocess,'Popen') as popen:
            with self.assertRaisesRegex(OSError,'begin disk failure'):
                with patch.object(runner,'review_inputs',return_value=self.config):runner.run(self.base)
        popen.assert_not_called();s=prep.read(self.base/'dispatch/status.json');self.assertEqual(s['unstarted'],self.jobs)
    def test_exit_write_failure_cannot_prevent_other_children_draining(self):
        children=[];actual=runner.write
        def launch(argv,**kw):
            p=FakeChild(argv,code=1 if not children else None,**kw);children.append(p);return p
        def broken(p,v):
            if str(p).endswith('-exit.json'):raise OSError('exit disk failure')
            return actual(p,v)
        def drain(p):p.returncode=-9
        with patch.object(runner,'write',side_effect=broken),patch.object(runner,'drain',side_effect=drain):
            with self.assertRaisesRegex(OSError,'exit disk failure'):self.run_controller(launch)
        self.assertEqual(len(children),3);self.assertTrue(all(p.returncode is not None for p in children))

    def test_transient_drain_error_still_drains_every_owned_child(self):
        children=[];attempts=[]
        def launch(argv,**kw):
            p=FakeChild(argv,code=1 if not children else None,**kw);children.append(p);return p
        def drain(p):
            attempts.append(p.pid)
            if len(attempts)==1:raise OSError('synthetic transient wait error')
            p.returncode=-9
        with patch.object(runner,'drain',side_effect=drain):
            with self.assertRaisesRegex(RuntimeError,'Chain failed'):self.run_controller(launch)
        self.assertEqual(attempts,[children[1].pid,children[2].pid,children[1].pid])
        self.assertTrue(all(p.returncode is not None for p in children))
        s=prep.read(self.base/'dispatch/status.json')
        self.assertFalse(s['passed']);self.assertFalse(s['active']);self.assertEqual(len(s['unstarted']),61)

    def test_matched_config_scope_and_frozen_input_changes(self):
        root=self.base/'frozen';root.mkdir()
        old=dict(schema='evolving-dimer-benchmark-v1',jobs=prep.old.jobs(),limits={'cpu_seconds':1800},
            physical={'activity':.0275},preparation_output='original-prepared',shape={'path':'original-shape','sha256':'old'})
        prep.write(root/'original.json',old)
        c=copy.deepcopy(old);c.update(jobs=self.jobs,allocation=prep.allocation(),output=str(root/'execution'),
            scientific_allocation={},protocol={},compiled_source_sha256={},control_analysis={},singleton_policy=prep.singleton_policy(),
            inherited_campaign={'config':prep.record(root/'original.json')})
        c['limits']['cpu_seconds']=3600
        prep.write(root/'config.json',c);prep.write(root/'protocol.json',{'roles':prep.ROLE_MAP,'singleton_policy':prep.singleton_policy()})
        freeze=dict(schema='two-neighbor-singleton-freeze-v1',complete=True,scientific_execution_started=False,
            files={},input_sha256={str(root/'original.json'):prep.sha(root/'original.json')})
        prep.write(root/'freeze.json',freeze)
        self.assertEqual(prep.verify(root),c)
        c['physical']['activity']=.04
        (root/'config.json').write_text(json.dumps(c))
        with self.assertRaisesRegex(ValueError,'matched scope'):prep.verify(root)
        c['physical']['activity']=.0275;c['jobs'][0]['seed_family']['stream']=99
        (root/'config.json').write_text(json.dumps(c))
        with self.assertRaisesRegex(ValueError,'matched scope'):prep.verify(root)
        c['jobs']=prep.jobs()
        for key,value in [('trial_cap',33),('uniform_half_width',159.),('uniform_probability',.4),
                          ('member_schedule','alternating_1_first')]:
            c['singleton_policy']=prep.singleton_policy();c['singleton_policy'][key]=value
            (root/'config.json').write_text(json.dumps(c))
            with self.assertRaisesRegex(ValueError,'matched scope'):prep.verify(root)
        (root/'original.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'Inherited input changed'):prep.verify(root)

    def test_fixed_allocations_and_roles(self):
        a=prep.allocation();self.assertEqual(a['chains'],64);self.assertEqual(a['singleton_attempts'],64*4608)
        self.assertEqual(a['local_attempts'],64*4608*4)
        self.assertEqual(a['retained_initial_observations'],64*4609)
        self.assertEqual(a['new_cloud_banks'],0);self.assertEqual(a['new_preparation_attempts'],0)
        self.assertEqual(prep.ROLE_MAP['proposal'],'singleton_two_neighbor/proposal')
        self.assertEqual(prep.ROLE_MAP['bath'],'singleton_two_neighbor/bath')
        self.assertEqual(len({(j['context_index'],j['initialization'],j['stream']) for j in self.jobs}),32)
        self.assertEqual(len({j['id'] for j in self.jobs}),64)
        self.assertEqual({j['arm'] for j in self.jobs},set(prep.ARMS))
        self.assertEqual(self.jobs,prep.jobs())
        self.assertEqual(prep.singleton_policy()['oligomer']['multi_contact_mass'],.8)


class LauncherTests(unittest.TestCase):
    """Detached-controller publication uses fake Popen only; no child starts."""
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.base=Path(self.tmp.name)
        (self.base/'common/source/tools').mkdir(parents=True)
        shutil.copyfile(runner.__file__,self.base/'common/source/tools/run_two_neighbor_singleton_benchmark.py')
        self.config={'jobs':prep.jobs(),'output':str(self.base/'execution')}
        for name in ('config.json','binding.json','run-binding.json','freeze.json','review.json'):
            prep.write(self.base/name,self.config if name=='config.json' else {})
        (self.base/'common/evolving_dimer_benchmark').write_bytes(b'synthetic nonexecutable placeholder')
        self.before={s:signal.getsignal(s) for s in (signal.SIGINT,signal.SIGTERM)}
        class Child:
            pid=10000001
            returncode=None
            terminated=False
            reaped=False
            def terminate(child):child.terminated=True;child.returncode=-15
            def wait(child):child.reaped=True;return child.returncode
        self.child=Child()
    def tearDown(self):
        for signum,handler in self.before.items():self.assertEqual(signal.getsignal(signum),handler)
        self.tmp.cleanup()
    def launch(self,popen):
        with patch.object(runner,'review_inputs',return_value=self.config),\
             patch.object(runner.subprocess,'Popen',side_effect=popen):
            return runner.launch(self.base,self.base/'review.json')
    def test_success_publishes_owned_controller_without_stopping_it(self):
        result=self.launch(lambda *args,**kwargs:self.child)
        self.assertEqual(result['pid'],self.child.pid)
        self.assertEqual(prep.read(self.base/'dispatch/launch.json')['pid'],self.child.pid)
        self.assertFalse(self.child.terminated);self.assertFalse(self.child.reaped)
    def test_failed_launch_receipt_stops_and_reaps_controller_without_retry(self):
        actual=runner.write
        def broken(path,value):
            if path.name=='launch.json':raise OSError('synthetic publication failure')
            actual(path,value)
        with patch.object(runner,'write',side_effect=broken):
            with self.assertRaisesRegex(OSError,'publication failure'):
                self.launch(lambda *args,**kwargs:self.child)
        self.assertTrue(self.child.terminated);self.assertTrue(self.child.reaped)
        self.assertTrue((self.base/'dispatch/plan.json').exists())
        with self.assertRaises(FileExistsError):self.launch(lambda *args,**kwargs:self.child)
    def test_launch_window_stop_is_owned_then_gracefully_drained(self):
        def launch(*args,**kwargs):
            self.assertIn('preexec_fn',kwargs)
            mask=signal.pthread_sigmask(signal.SIG_BLOCK,set())
            self.assertTrue({signal.SIGINT,signal.SIGTERM}<=mask)
            os.kill(os.getpid(),signal.SIGTERM)
            return self.child
        with self.assertRaisesRegex(SystemExit,'signal'):
            self.launch(launch)
        self.assertTrue(self.child.terminated);self.assertTrue(self.child.reaped)
        self.assertFalse((self.base/'dispatch/launch.json').exists())
    def test_second_stop_cannot_interrupt_controller_drain(self):
        actual=runner.write
        def broken(path,value):
            if path.name=='launch.json':raise OSError('publication failure')
            actual(path,value)
        def terminate():
            self.child.terminated=True;self.child.returncode=-15
            os.kill(os.getpid(),signal.SIGTERM)
            os.kill(os.getpid(),signal.SIGINT)
        self.child.terminate=terminate
        with patch.object(runner,'write',side_effect=broken):
            with self.assertRaisesRegex(OSError,'publication failure'):
                self.launch(lambda *args,**kwargs:self.child)
        self.assertTrue(self.child.terminated);self.assertTrue(self.child.reaped)
    def test_fork_failure_leaves_no_controller_to_reap(self):
        def launch(*args,**kwargs):raise OSError('synthetic fork failure')
        with self.assertRaisesRegex(OSError,'fork failure'):self.launch(launch)
        self.assertFalse(self.child.terminated);self.assertFalse(self.child.reaped)

if __name__=='__main__':unittest.main()
