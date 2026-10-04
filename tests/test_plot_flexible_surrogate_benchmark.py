"""Synthetic completed metadata and metrics; no trajectory or geometry inputs."""
import copy
import importlib.util
import itertools
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

SOURCE = Path(__file__).resolve().parents[1]/'tools/plot_flexible_surrogate_benchmark.py'
SPEC = importlib.util.spec_from_file_location('flexible_plot', SOURCE)
p = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(p)


def ess(value, cpu):
    return dict(apparent_ess=value, apparent_ess_per_sampling_CPU_second=None if value is None else value/cpu)


def activity(fraction, changes, returns, cpu):
    return dict(production_samples=4096, full_sampler_cpu_seconds=cpu, nonempty_fraction=fraction,
        direct_nonempty_changes=changes, entering_nonempty=2, leaving_nonempty=1,
        completed_nonempty_returns=[{'synthetic_return':i} for i in range(returns)],
        direct_nonempty_changes_per_full_CPU_second=changes/cpu,
        completed_nonempty_returns_per_full_CPU_second=returns/cpu)


def synthetic_result():
    chains = []; comparisons = []
    for arm, start, stream in itertools.product(p.ARMS, p.STARTS, range(4)):
        cpu = 100.+stream
        # One source has positive presence ESS from just one occupied frame.
        external_fraction = 1/4096 if stream == 0 else (stream+1)/16
        internal_fraction = .5
        internal_activity = activity(internal_fraction, stream+1, stream, cpu)
        external_activity = activity(external_fraction, stream, stream, cpu)
        internal = dict(production_samples=4096, full_sampler_cpu_seconds=cpu, transition_baseline_block=512,
            internal_contact_fraction=internal_fraction, attachments=2, detachments=1,
            patch_set_activity=internal_activity, patch_set_ess=ess(10., cpu),
            patch_set_occupancy=[dict(fingerprint='empty', fraction=.5), dict(fingerprint='contact', fraction=.5)])
        chains.append(dict(job=dict(context_index=0, arm=arm, initialization=start, stream=stream, id=stream),
            reused_control=arm in p.CONTROLS, trajectory={'path':'/must-not-read/trajectory.jsonl', 'sha256':'unused'},
            metrics=dict(production_samples=4096, full_sampler_cpu_seconds=cpu,
                internal_contact_fraction=internal_fraction, internal_organization=internal,
                external_activity=external_activity,
                external_only=dict(any_contact_fraction=external_fraction, ess=ess(31. if stream == 0 else None, cpu)),
                fingerprint_ess=ess(None if arm == 'local' else 16.+stream, cpu), singleton_fingerprint_fraction=.25)))
    for arm, stream in itertools.product(p.ARMS, range(4)):
        comparisons.append(dict(
            left=dict(context_index=0, arm=arm, initialization='source', stream=stream),
            right=dict(context_index=0, arm=arm, initialization='proposal_prepared', stream=stream),
            internal_organization=dict(patch_set_occupancy_total_variation=stream/4., contact_fraction_difference=-.125),
            external_only=dict(environment_total_variation=stream/8.)))
    return dict(schema='flexible-surrogate-analysis-v1', complete=True, new_chains=24, reused_control_chains=40,
        chains=chains, comparisons=dict(descriptive_paired_comparisons=comparisons), new_geometry_endpoints=110616,
        old_geometry_queries=0, new_physical_draws=0, native_observer=False, analysis_plan={'synthetic_plan':True})


def completed_fixture(root):
    def save(relative, value):
        path = root/relative; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, allow_nan=False)+'\n'); return path
    result = synthetic_result()
    analysis = save('analysis/analysis.json', result)
    save('analysis/manifest.json', dict(complete=True, files={'analysis.json':p.sha(analysis)}))
    declaration = save('frozen-analysis-plan.json', result['analysis_plan'])
    protocol = save('protocol.json', dict(schema='flexible-surrogate-analysis-dispatch-v1',
        new_chains=24, cached_control_chains=40, analysis_plan=p.record(declaration)))
    job = dict(id='whole-contact-analysis', phase='statistics', population='context0-all-24',
        argv=['synthetic-observer'], terminal=dict(path=str(analysis), success_contract='complete'))
    plan = save('execution-plan.json', dict(schema='native-class-physical-execution-v1', root=str(root),
        maximum_workers=1, threads=1, jobs=[job], files={str(protocol):p.sha(protocol), str(declaration):p.sha(declaration)}))
    prepared = save('preparation.json', dict(schema='flexible-surrogate-analysis-preparation-v1',
        complete=True, launched=False, execution_plan=p.record(plan), protocol=p.record(protocol)))
    save('execution/claim.json', dict(schema='native-class-physical-execution-v1', plan_sha256=p.sha(plan),
        maximum_workers=1, threads=1, retries=0, replacements=0, preparation_receipt=p.record(prepared)))
    done = dict(id=job['id'], phase=job['phase'], population=job['population'], argv=job['argv'],
        success=True, child_started=True, child_drained=True, returncode=0, error=None, timeout=False,
        retries=0, replacements=0, success_contract='complete', pid=123, birth_ticks=456, terminal=p.record(analysis))
    status = dict(complete=True, passed=True, failure=None, active=None, unstarted=[], completed=[done], plan_sha256=p.sha(plan))
    save('execution/status.json', status); save('execution/summary.json', status)
    directory = 'execution/jobs/000-whole-contact-analysis/'
    save(directory+'attempt.json', dict(job=job))
    save(directory+'process.json', {k:done[k] for k in ('id', 'pid', 'birth_ticks', 'argv')})
    save(directory+'success.json', done); save(directory+'exit.json', done)
    return result


class SavedMetricsTests(unittest.TestCase):
    def test_all64_distinct_rare_contact_occupancy_and_null_ess(self):
        result = synthetic_result(); untouched = copy.deepcopy(result)
        rows = p.saved_values(result)
        self.assertEqual(len(rows), 64)
        self.assertEqual(sum(r['reused_control'] for r in rows), 40)
        self.assertEqual(sum(not r['reused_control'] for r in rows), 24)
        self.assertEqual(len({tuple(r['job'][k] for k in ('context_index','arm','initialization','stream')) for r in rows}), 64)
        rare = [r for r in rows if r['job']['stream'] == 0]
        self.assertTrue(all(r['external_contact_frames'] == 1 and r['external_presence_ess'] == 31. for r in rare))
        self.assertTrue(any(r['contact_fingerprint_ess_per_cpu'] is None for r in rows))
        self.assertEqual(result, untouched)

    def test_inventory_origin_residence_cpu_and_metric_mutations_fail(self):
        mutations = (
            lambda r:r['chains'].pop(),
            lambda r:r['chains'].__setitem__(1, copy.deepcopy(r['chains'][0])),
            lambda r:r['chains'][0].update(reused_control=False),
            lambda r:r['chains'][0]['metrics'].update(production_samples=4095),
            lambda r:r['chains'][0]['metrics']['internal_organization'].update(transition_baseline_block=513),
            lambda r:r['chains'][0]['metrics']['external_activity'].update(full_sampler_cpu_seconds=1.),
            lambda r:r['chains'][0]['metrics']['internal_organization']['patch_set_activity'].update(direct_nonempty_changes_per_full_CPU_second=99.),
            lambda r:r['chains'][0]['metrics']['external_only']['ess'].update(apparent_ess=None),
            lambda r:r['chains'][0]['metrics']['external_only'].update(any_contact_fraction=-1.),
            lambda r:r['chains'][0]['metrics'].update(singleton_fingerprint_fraction=2.),
        )
        for mutate in mutations:
            result = synthetic_result(); mutate(result)
            with self.assertRaises(ValueError): p.saved_values(result)

    def test_all32_initialization_pairs_preserve_signed_source_and_magnitude(self):
        result = synthetic_result(); values = p.saved_agreement(result)
        self.assertEqual(len(values), 32)
        self.assertTrue(all(row['internal_contact_fraction_absolute_difference'] == .125 for row in values))
        self.assertTrue(all(row['source_start_comparison']['saved_right_minus_left_contact_difference'] == -.125 for row in values))
        for defect in ('missing', 'wrong stream', 'duplicate', 'invalid TV'):
            result = synthetic_result(); pairs = result['comparisons']['descriptive_paired_comparisons']
            if defect == 'missing': pairs.pop()
            if defect == 'wrong stream': pairs[0]['right']['stream'] = 2
            if defect == 'duplicate': pairs[-1] = copy.deepcopy(pairs[0])
            if defect == 'invalid TV': pairs[0]['internal_organization']['patch_set_occupancy_total_variation'] = 1.1
            with self.subTest(defect=defect), self.assertRaises(ValueError): p.saved_agreement(result)


class CompletedOnlyTests(unittest.TestCase):
    def test_completed_metadata_authenticates_and_result_tamper_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); expected = completed_fixture(root)
            result, refs = p.authenticate(root)
            self.assertEqual(result, expected); self.assertIn(str(root/'analysis/analysis.json'), refs)
            with (root/'analysis/analysis.json').open('a') as stream: stream.write(' ')
            with self.assertRaisesRegex(ValueError, 'Completed result hash differs'): p.authenticate(root)

    def test_incomplete_controller_stops_before_reading_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); completed_fixture(root)
            for name in ('status', 'summary'):
                path = root/f'execution/{name}.json'; value = json.loads(path.read_text())
                value.update(complete=False, passed=False, active='running')
                path.write_text(json.dumps(value))
            original = Path.read_text
            def guarded(path, *args, **kwargs):
                if path == root/'analysis/analysis.json': raise AssertionError('Read incomplete scientific result')
                return original(path, *args, **kwargs)
            with mock.patch.object(Path, 'read_text', guarded), self.assertRaisesRegex(ValueError, 'incomplete, failed or undrained'):
                p.authenticate(root)

    def test_complete_synthetic_render_has_three_figures_and_no_journal_reads(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'controller'; root.mkdir(); completed_fixture(root)
            output = Path(tmp)/'figures'; original_open = Path.open
            def guarded(path, *args, **kwargs):
                if path.suffix == '.jsonl': raise AssertionError('Plotter read a scientific journal')
                return original_open(path, *args, **kwargs)
            with mock.patch.object(Path, 'open', guarded): receipt = p.plot(root, output)
            self.assertTrue(receipt['complete']); self.assertEqual(receipt['scientific_journals_read'], 0)
            self.assertEqual(receipt['chains'], 64); self.assertFalse(receipt['streams_or_starts_pooled'])
            for name in ('contact-organization', 'contact-efficiency', 'initialization-differences'):
                self.assertGreater((output/f'{name}.png').stat().st_size, 1000)
                self.assertIn('Cached 40 controls', (output/f'{name}.svg').read_text())
            values = json.loads((output/'plotted-values.json').read_text())
            self.assertEqual(len(values['rows']), 64); self.assertEqual(len(values['initialization_agreement']), 32)
            self.assertTrue(any(row['external_presence_ess_per_cpu'] is None for row in values['rows']))
            for name, ref in receipt['outputs'].items(): self.assertEqual(p.sha(output/name), ref['sha256'])
            with self.assertRaisesRegex(ValueError, 'Fresh plot output required'): p.plot(root, output)


if __name__ == '__main__':
    unittest.main()
