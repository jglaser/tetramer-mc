import copy
import math
from pathlib import Path
import tempfile
import unittest
from run_contact_arc_score import make_jobs, validate_design
from run_full_vessel_comparison import execute_group
from report_contact_arc_score import moment


class ArcScoreController(unittest.TestCase):
    def dispatch(self,root,returncode=0):
        jobs=make_jobs(root,root/'common');launched=[]
        class FakeChild:
            pid=999999
            def poll(self):return returncode
            def wait(self):return returncode
        def launch(command,**kwargs):launched.append(command);return FakeChild()
        execute_group(jobs,lambda:None,1,root,{},popen=launch,pause=lambda _:None,
            capacity=lambda _:dict(workers=0,physical_pids=[]),birth=lambda _:0)
        return jobs,launched

    def test_exact_two_job_contract_no_sampling_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            jobs,commands=self.dispatch(Path(directory))
            self.assertEqual([j['status'] for j in jobs],['complete','complete'])
            self.assertEqual(len(commands),2)
            for command,flag in zip(commands,['--importance-guide','--guide']):
                guides=[command[i+1] for i,x in enumerate(command) if x==flag]
                self.assertEqual([Path(g).stem for g in guides],['uniform_phi92','localized_phi92'])
                self.assertNotIn('--seed',command);self.assertNotIn('--samples',command)
                self.assertEqual(command[command.index('--minimum-arc-mass')+1],'1e-12')

    def test_failure_prevents_independent_job_and_no_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with self.assertRaisesRegex(RuntimeError,'no further launches'):self.dispatch(root,1)
            self.assertTrue((root/'rust.log').exists());self.assertFalse((root/'python.log').exists())

    def test_frozen_allocation_rejects_new_pose_or_missing_invalid_class(self):
        plan=dict(arms=['uniform_phi92','localized_phi92'],unique_saved_queries=206,candidate_density_evaluations=412,
            critical_queries=78,critical_original_sources={'baseline':4,'expanded':74},breadth_queries=128,
            breadth_classes={'native_R5':32,'native_complement':32,'competing':32,'invalid':32},
            minimum_arc_mass=1e-12,defensive_uniform_probability=.5,conditional_probability=.5,
            widths_A=[.02,.1,.5],maximum_CPU_workers=1,new_pose_draws=0,new_Poisson_clouds=0,
            guides_unchanged=True,labels_unchanged=True,all_components_and_widths_in_density=True,
            no_optional_stopping=True,no_retries=True,no_autoextension=True)
        validate_design(plan)
        for key,value in [('new_pose_draws',1),('minimum_arc_mass',1e-10),('candidate_density_evaluations',410),
                          ('breadth_classes',{'native_R5':32,'native_complement':32,'competing':32})]:
            candidate=copy.deepcopy(plan);candidate[key]=value
            with self.assertRaises(ValueError):validate_design(candidate)

    def test_moment_uses_all_contributions_and_is_log_shift_invariant(self):
        a=moment([math.log(1),math.log(3)]);b=moment([1000.,1000.+math.log(3)])
        self.assertAlmostEqual(a['contribution_ESS'],1.6)
        self.assertAlmostEqual(a['largest_contribution'],.75)
        self.assertAlmostEqual(b['contribution_ESS'],a['contribution_ESS'],places=10)
        self.assertAlmostEqual(b['log_second_moment']-a['log_second_moment'],1000.)


if __name__=='__main__':unittest.main()
