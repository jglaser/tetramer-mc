"""Nonphysical scheduler tests; never create protein populations."""
import copy
import os
from pathlib import Path
import tempfile
import unittest

from run_r4_smc_controls import control_jobs, physical_preflight, prerequisite_state, process_token
from run_conditional_ray_campaign import execute_jobs


class SchedulerTests(unittest.TestCase):
    def test_wait_and_dead_prerequisite(self):
        state = dict(complete=False, steps=[dict(returncode=None)])
        self.assertEqual(prerequisite_state(state, True), 'waiting')
        with self.assertRaises(ValueError): prerequisite_state(state, False)

    def test_success_is_execution_not_scientific_gate(self):
        state = dict(complete=True, confirmation_passed=False,
                     steps=[dict(returncode=0),dict(returncode=0)])
        self.assertEqual(prerequisite_state(state, False), 'ready')
        for changed in [dict(state,error='failure'), dict(state,steps=[dict(returncode=0)]),
                        dict(state,steps=[dict(returncode=0),dict(returncode=1)])]:
            with self.assertRaises(ValueError): prerequisite_state(changed,True)

    def test_birth_identity(self):
        self.assertIsNotNone(process_token(os.getpid()))
        self.assertEqual(process_token(os.getpid()), process_token(os.getpid()))
        self.assertIsNone(process_token(2147483647))

    def test_interleaving_preserves_exact_commands_and_seeds(self):
        contexts = [dict(protocol=dict(jobs=[dict(id=f'r{i}',seed=100*arm+i,
                    argv=['test',str(arm),str(i)],output=f'/unused/{arm}/{i}') for i in range(4)])) for arm in range(2)]
        before = copy.deepcopy(contexts); jobs=control_jobs(contexts,Path('/unused'))
        self.assertEqual(contexts,before)
        self.assertEqual([j['arm'] for j in jobs],['broad','narrow']*4)
        self.assertEqual([j['seed'] for j in jobs],[0,100,1,101,2,102,3,103])
        for i,j in enumerate(jobs): self.assertEqual(j['command'],contexts[i%2]['protocol']['jobs'][i//2]['argv'])
        contexts[1]['protocol']['jobs'][0]['seed']=0
        with self.assertRaises(ValueError): control_jobs(contexts,Path('/unused'))

    def test_preflight_refuses_existing_output(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(ValueError,'Existing SMC output'):
                physical_preflight(dict(jobs=[dict(directory=root)]), [])

    def test_failure_drains_started_children_and_never_retries(self):
        class Child:
            def __init__(self,index): self.pid=10+index;self.index=index;self.waited=False
            def poll(self): return 1 if self.index==0 else None
            def wait(self): self.waited=True;return 1 if self.index==0 else 0
        with tempfile.TemporaryDirectory() as root:
            jobs=[dict(directory=str(Path(root)/f'output{i}'),log=str(Path(root)/f'log{i}'),
                       command=['inert'],status='pending') for i in range(6)]
            children=[]
            def launch(*args,**kwargs):
                child=Child(len(children));children.append(child);return child
            with self.assertRaisesRegex(RuntimeError,'Physical failure'):
                execute_jobs(jobs,lambda:None,workers=4,popen=launch,pause=lambda _:None)
            self.assertEqual(len(children),4)
            self.assertTrue(all(c.waited for c in children))
            self.assertEqual([j['status'] for j in jobs],['failed','complete','complete','complete','not_started','not_started'])


if __name__ == '__main__': unittest.main()
