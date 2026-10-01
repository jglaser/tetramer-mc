import tempfile
from pathlib import Path
import unittest
from run_contact_circle_feasibility import make_jobs
from run_full_vessel_comparison import execute_group


class CircleController(unittest.TestCase):
    def execute(self, root, code=0):
        jobs = make_jobs(root, root/'common', root/'preparation')
        launched = []
        class FakeChild:
            pid = 999999
            def poll(self): return code
            def wait(self): return code
        def launch(command, **kwargs):
            launched.append(command)
            return FakeChild()
        execute_group(jobs, lambda:None, 1, root, {}, popen=launch, pause=lambda _:None,
            capacity=lambda _:dict(workers=0,physical_pids=[]), birth=lambda _:0)
        return jobs, launched

    def test_real_dispatcher_accepts_both_job_contracts_without_launching_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            jobs, launched = self.execute(Path(directory))
            self.assertEqual([j['status'] for j in jobs], ['complete','complete'])
            self.assertEqual(len(launched), 2)
            self.assertTrue(all(j['kind']=='geometry' for j in jobs))
            self.assertTrue(all('--cases' in c for c in launched))

    def test_child_failure_stops_before_second_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, 'no further launches'):
                self.execute(Path(directory), code=1)
            self.assertTrue((Path(directory)/'rust.log').exists())
            self.assertFalse((Path(directory)/'python.log').exists())


if __name__ == '__main__': unittest.main()
