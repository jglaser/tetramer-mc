"""Driver-only synthetic metadata/control tests; no sample rows are audited."""
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import run_native_class_sphere_algebra_replay as driver


def save(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value)+'\n')


def fixture(base):
    root = base/'replay'; original = base/'original'
    (root/'code').mkdir(parents=True); (root/'audits').mkdir(); (root/'children').mkdir()
    sources = Path(driver.__file__).resolve().parent
    files = {}
    for name in driver.SOURCE_NAMES:
        target = root/'code'/name; shutil.copyfile(sources/name, target); files[str(target)] = driver.sha(target)
    jobs = []
    prior = dict(estimate={'draws': 2048, 'logQ': 1.25}, hard_region={'draws': 2048, 'logQ': .5},
                 paired_noise={'value': .25}, counts={'attempted': 2048}, samples_sha256='1'*64,
                 attempts_sha256='2'*64, analysis_cpu_seconds=1.)
    for name in driver.IDS:
        population = original/'populations'/name
        for filename in ('manifest.json', 'summary.json', 'samples.jsonl', 'attempts.jsonl',
                         'provenance/input-config.json', 'provenance/region.json', 'provenance/shape.json',
                         'provenance/importance-guide.json', 'provenance/source-bundle.json', 'provenance/compiled-native.json'):
            path = population/filename; save(path, {'synthetic_metadata_only': True}); files[str(path)] = driver.sha(path)
        previous = original/'audits'/(name+'.json'); save(previous, prior); files[str(previous)] = driver.sha(previous)
        output = root/'audits'/(name+'.json')
        jobs.append(dict(id=name, samples=2048, population=str(population), prior_audit=str(previous), output=str(output),
            argv=[sys.executable, str(root/'code/native_class_line_physical_algebra_audit.py'), '--root', str(population), '--out', str(output)]))
    plan = dict(schema=driver.SCHEMA, execution_ready=False, review_required=True, root=str(root), base=str(original),
        expected_populations=16, expected_saved_attempts=32768, maximum_workers=1,
        cpu_limit_per_population_seconds=120, wall_limit_per_population_seconds=240, address_space_limit_bytes=4*1024**3,
        jobs=jobs, files=files, python=sys.executable, runtime=driver.audit.runtime_identity(),
        new_pose_draws=0, new_Poisson_clouds=0, new_atom_geometry_queries=0, retries=0, scope='synthetic metadata/control only')
    save(root/'execution-plan.json', plan)
    save(root/'review.json', dict(complete=True, passed=True, plan_sha256=driver.sha(root/'execution-plan.json'),
                                  maximum_workers=1, expected_saved_attempts=32768))
    return root, plan, prior


class ReplayDriverTests(unittest.TestCase):
    def patch_archive(self, root):
        return mock.patch.multiple(driver, __file__=str(root/'code/run_native_class_sphere_algebra_replay.py'))

    def test_complete_inventory_and_exclusive_claim(self):
        with tempfile.TemporaryDirectory() as temp:
            root, plan, prior = fixture(Path(temp)); called = []
            def child(argv, cwd, directory, cpu, wall, memory):
                called.append((argv, cpu, wall, memory))
                job = plan['jobs'][len(called)-1]
                result = dict(prior, complete=True, passed=True, all_rows_algebra=2048,
                    independently_reconstructed_geometry_rows=0, geometry_certified=False,
                    new_pose_draws=0, new_Poisson_clouds=0)
                save(job['output'], result)
            with (self.patch_archive(root), mock.patch.object(driver.launcher, '__file__', str(root/'code/run_evolving_dimer_analysis.py')),
                    mock.patch.object(driver.launcher, 'owned_child', side_effect=child)):
                driver.run(root)
                self.assertEqual(len(called), 16)
                self.assertTrue(all(c[1:] == (120, 240, 4*1024**3) for c in called))
                result = driver.read(root/'completion.json'); self.assertEqual(result['all_rows_algebra'], 32768)
                with self.assertRaises(FileExistsError): driver.run(root)
                self.assertEqual(len(called), 16)

    def test_failure_does_not_start_later_population_or_retry(self):
        with tempfile.TemporaryDirectory() as temp:
            root, plan, prior = fixture(Path(temp))
            with (self.patch_archive(root), mock.patch.object(driver.launcher, '__file__', str(root/'code/run_evolving_dimer_analysis.py')),
                    mock.patch.object(driver.launcher, 'owned_child', side_effect=subprocess.TimeoutExpired('mock-child', 240)) as child):
                with self.assertRaises(subprocess.TimeoutExpired): driver.run(root)
                self.assertEqual(child.call_count, 1)
                failure = driver.read(root/'failure.json')
                self.assertEqual(failure['current'], driver.IDS[0]); self.assertEqual(failure['remaining_jobs_not_started'], 15)
                self.assertEqual(failure['retries'], 0)
                self.assertTrue((root/'children'/driver.IDS[0]/'begin.json').exists())
                with self.assertRaises(FileExistsError): driver.run(root)
                self.assertEqual(child.call_count, 1)

    def test_changed_inventory_missing_binding_and_command_rejected_before_claim(self):
        for mode in ('inventory', 'missing_binding', 'command', 'scope'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temp:
                root, plan, prior = fixture(Path(temp))
                if mode == 'inventory': plan['jobs'][-1] = copy.deepcopy(plan['jobs'][0])
                if mode == 'missing_binding': plan['files'].pop(str(Path(plan['jobs'][0]['population'])/'samples.jsonl'))
                if mode == 'command': plan['jobs'][0]['argv'] = ['unbound-command']
                if mode == 'scope': plan['new_atom_geometry_queries'] = 1
                save(root/'execution-plan.json', plan)
                with (self.patch_archive(root), mock.patch.object(driver.launcher, '__file__', str(root/'code/run_evolving_dimer_analysis.py')),
                        mock.patch.object(driver.launcher, 'owned_child') as child):
                    with self.assertRaises(ValueError): driver.run(root)
                    child.assert_not_called(); self.assertFalse((root/'claim.json').exists())


if __name__ == '__main__': unittest.main()
