from pathlib import Path
import tempfile
import unittest
from run_hard_free_line_score import job
from run_full_vessel_comparison import execute_group


class ScoreOnlyDispatch(unittest.TestCase):
    def execute(self,root,returncode):
        jobs=[job(root)];launched=[]
        class Child:
            pid=999999
            def poll(self):return returncode
            def wait(self):return returncode
        def launch(cmd,**kwargs):launched.append(cmd);return Child()
        execute_group(jobs,lambda:None,1,root,{},popen=launch,pause=lambda _:None,
            capacity=lambda _:dict(workers=0,physical_pids=[]),birth=lambda _:0)
        return jobs,launched

    def test_only_xyz_and_exact_zero_draw_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            jobs,launched=self.execute(Path(directory),0)
            self.assertEqual(len(launched),1);self.assertEqual(jobs[0]['status'],'complete')
            cmd=launched[0]
            self.assertEqual(cmd[cmd.index('--samples')+1],'0');self.assertEqual(cmd[cmd.index('--seed')+1],'0')
            self.assertEqual(Path(cmd[cmd.index('--importance-guide')+1]).name,'xyz.json')

    def test_failure_drains_and_preserves_log(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError,'no further launches'):self.execute(Path(directory),1)
            self.assertTrue((Path(directory)/'score.log').exists())


if __name__=='__main__':unittest.main()
