#!/usr/bin/env python3
"""Synthetic completed metadata/native-label fixtures; no geometry or simulation."""
import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

import plot_growth_z004_native as plotter


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False)+'\n')


def manifest_hashes(root):
    return {arm: plotter.sha(root/arm/'strict-native/manifest.json') for arm in plotter.ARMS}


def observations():
    rows = []; previous_members = set(); previous_edges = set()
    for index in range(1001):
        members = list(range(9 if 250 <= index < 750 else 8))
        keys = [[i, i+1, 0] for i in range(7)]
        if len(members) == 9: keys.append([0, 8, 1])
        components = [dict(bodies=members, certified=True)]
        if index >= 500:
            keys.append([100, 101, 2]); components.append(dict(bodies=[100, 101], certified=index >= 800))
        edges = {(i, j) for i, j, _ in keys}; member_set = set(members)
        rows.append(dict(sweep=index*100, sampler_cpu_seconds=index*.5, native_keys=keys,
            certified_seed_bodies=members, largest_certified_seed_component=len(members),
            registered_edges=len(edges), registered_edges_outside_original_seed=len(edges)-7,
            registered_components=components, nonseed_registered_component_sizes=[2] if index >= 500 else [],
            seed_entered_bodies=sorted(member_set-previous_members) if index else [],
            seed_left_bodies=sorted(previous_members-member_set) if index else [],
            gained_edges=[list(e) for e in sorted(edges-previous_edges)] if index else [],
            lost_edges=[list(e) for e in sorted(previous_edges-edges)] if index else []))
        previous_members, previous_edges = member_set, edges
    full = dict(start_sweep=0, end_sweep=100000, intervals=1000, seed_member_entries=1, seed_member_exits=1)
    assessment = dict(complete=True, frames=1001, first_sweep=0, last_sweep=100000,
        cadence_sweeps=100, seed_labels=list(range(8)), initial=rows[0], final=rows[-1],
        physical=dict(radius_A=1.4, activity_A_minus3=.04, bodies=264, concentration_uM=500.),
        final_full_pair_check=dict(passed=True, pairs=34716, keys=len(rows[-1]['native_keys'])),
        reuse=None, newly_classified_frames=1001, windows=dict(full=full, earlier={'saved': 'earlier'}, last_quarter={'saved': 'late'}),
        quarters=[{'saved_quarter': i} for i in range(4)], families={'2/9': {'final': 1}},
        missing_outside_seed_families=['0/1'], seed_newcomers=[{'body': 8, 'saved': 'censored episodes'}],
        motif_change_windows=[], observed_edge_reentry_episodes=0, cycle_frustrated_frames=300)
    return assessment, rows


def campaign(root):
    """Real stable-driver lifecycle contract around synthetic saved labels."""
    source = root/'code/run_native_class_physical_campaign.py'; source.parent.mkdir(parents=True)
    shutil.copyfile(plotter.driver.__file__, source)
    executable = root/'code/fake-executable'; executable.write_text('synthetic: never executed\n')
    protocol = root/'protocol.json'
    declaration = dict(schema='growth-z004-native-protocol-v1', root=str(root), arms={},
        physical_parameters=plotter.PHYSICAL, shared_provenance=plotter.SHARED,
        geometry_jobs=2, new_saved_frames=2002, cached_classifications=0,
        new_physical_draws=0, move_log_scans=0, exclusion_graph_jobs=0)
    put(protocol, declaration)
    files = {str(p): plotter.sha(p) for p in (source, executable, protocol)}
    jobs = []; completed = []
    for ordinal, arm in enumerate(plotter.ARMS):
        snapshot = root/arm/'snapshot'; snapshot.mkdir(parents=True)
        input_receipt = root/arm/'snapshot-manifest.json'
        put(input_receipt, dict(actual_model_sha256=plotter.MODELS[arm], sha256={'config.json': '0'*64},
            frames=1001, first_sweep=0, last_sweep=100000, cadence_sweeps=100))
        files[str(input_receipt)] = plotter.sha(input_receipt)
        declaration['arms'][arm] = dict(snapshot_manifest=dict(path=str(input_receipt), sha256=files[str(input_receipt)]),
            model_sha256=plotter.MODELS[arm], native_informed_proposal=arm == 'coverage',
            effective_config_sha256='0'*64, frames=1001, final_exhaustive_pairs=34716)
        assessment, rows = observations(); assessment.update(snapshot=str(snapshot), source_sha256={str(input_receipt): files[str(input_receipt)]})
        directory = root/arm/'strict-native'; put(directory/'assessment.json', assessment)
        (directory/'frames.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
        (directory/'report.md').write_text('Synthetic observer report\n')
        put(directory/'manifest.json', dict(complete=True, physical_jobs_launched=0,
            source_sha256=assessment['source_sha256'],
            files={name: plotter.sha(directory/name) for name in ('assessment.json', 'frames.jsonl', 'report.md')}))
        job = dict(id=arm+'-native', population=arm, phase='geometry', argv=[str(executable), arm],
            cpu_limit_seconds=3600, wall_limit_seconds=7200, address_space_limit_bytes=8*1024**3,
            terminal=dict(path=str(directory/'assessment.json'), success_contract='complete'))
        jobs.append(job)
        done = dict(id=job['id'], population=arm, phase='geometry', argv=job['argv'],
            success=True, error=None, timeout=False, child_started=True, child_drained=True,
            returncode=0, retries=0, replacements=0, pid=123+ordinal, birth_ticks=100+ordinal,
            terminal=dict(path=str(directory/'assessment.json'), sha256=plotter.sha(directory/'assessment.json')),
            success_contract='complete')
        completed.append(done); lifecycle = root/'execution/jobs'/f'{ordinal:03d}-{job["id"]}'
        put(lifecycle/'attempt.json', {'job': job})
        put(lifecycle/'process.json', {k: done[k] for k in ('id', 'pid', 'birth_ticks', 'argv')})
        put(lifecycle/'exit.json', done); put(lifecycle/'success.json', done)
    put(protocol, declaration); files[str(protocol)] = plotter.sha(protocol)
    plan = dict(schema=plotter.driver.SCHEMA, root=str(root), maximum_workers=1, threads=1,
        files=files, executable_resolutions={str(executable): str(executable)}, jobs=jobs,
        preparation_receipt=str(root/'preparation.json'))
    put(root/'execution-plan.json', plan); digest = plotter.sha(root/'execution-plan.json')
    prepared = dict(schema='growth-z004-native-preparation-v1', complete=True, launched=False,
        execution_plan=dict(path=str(root/'execution-plan.json'), sha256=digest),
        protocol=dict(path=str(protocol), sha256=plotter.sha(protocol)))
    put(root/'preparation.json', prepared)
    put(root/'execution/claim.json', dict(schema=plotter.driver.SCHEMA, plan_sha256=digest,
        maximum_workers=1, threads=1, retries=0, replacements=0,
        preparation_receipt=dict(path=str(root/'preparation.json'), sha256=plotter.sha(root/'preparation.json'))))
    summary = dict(schema=plotter.driver.SCHEMA, complete=True, passed=True, failure=None,
        active=None, unstarted=[], completed=completed, plan_sha256=digest,
        maximum_workers=1, threads=1, retries=0, replacements=0, files=files)
    put(root/'execution/summary.json', summary); put(root/'execution/status.json', summary)


class GrowthPresentationTests(unittest.TestCase):
    def test_all_frames_baseline_residence_and_component_distinction(self):
        assessment, rows = observations(); series = plotter.saved_series(assessment, rows)
        self.assertEqual(len(series), 1001)
        self.assertEqual(series[0]['certified_seed_members'], 8)
        self.assertEqual(series[249]['certified_seed_members'], 8)
        self.assertEqual(series[250]['certified_seed_members'], 9)
        self.assertEqual(series[750]['certified_seed_members'], 8)
        self.assertEqual(series[600]['largest_nonseed_registered_component'], 2)
        self.assertEqual(series[600]['largest_nonseed_certified_component'], 0)
        self.assertEqual(series[800]['largest_nonseed_certified_component'], 2)
        reduced = plotter.numeric_summary(assessment, series)
        self.assertEqual((reduced['seed_entries'], reduced['seed_exits']), (1, 1))
        self.assertEqual(reduced['saved_windows'], assessment['windows'])
        self.assertEqual(reduced['saved_quarters'], assessment['quarters'])
        self.assertEqual(reduced['motif_families'], assessment['families'])

    def test_missing_frame_false_certificate_and_bad_changes_rejected(self):
        assessment, rows = observations()
        for mutation in ('missing', 'certificate', 'baseline', 'cpu', 'edge'):
            with self.subTest(mutation=mutation):
                bad = copy.deepcopy(rows)
                if mutation == 'missing': bad.pop(400)
                elif mutation == 'certificate': bad[600]['registered_components'][1]['certified'] = 1
                elif mutation == 'baseline': bad[400]['seed_entered_bodies'] = [8]
                elif mutation == 'cpu': bad[400]['sampler_cpu_seconds'] = -1
                else: bad[400]['registered_edges_outside_original_seed'] += 1
                with self.assertRaises(ValueError): plotter.saved_series(assessment, bad)

    def test_authenticates_both_real_driver_lifecycles(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve(); campaign(root)
            pins = manifest_hashes(root); assessments, rows, refs = plotter.authenticate(root, pins)
            self.assertEqual(set(assessments), set(plotter.ARMS))
            self.assertEqual([len(rows[a]) for a in plotter.ARMS], [1001, 1001])
            self.assertIn(str(root/'discovery/strict-native/frames.jsonl'), refs)
            # The driver hashes this frozen input; no content decoding is needed.
            (root/'code/fake-executable').write_text('changed\n')
            with self.assertRaisesRegex(ValueError, 'Frozen input changed'): plotter.authenticate(root, pins)

    def test_undrained_or_duplicate_inventory_refused_before_native_rows(self):
        for mutation in ('undrained', 'duplicate'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve(); campaign(root)
                summary = plotter.driver.read(root/'execution/summary.json')
                if mutation == 'undrained': summary['completed'][1]['child_drained'] = False
                else: summary['completed'][1] = summary['completed'][0]
                put(root/'execution/summary.json', summary); put(root/'execution/status.json', summary)
                with mock.patch.object(plotter, 'saved_series', side_effect=AssertionError('must not reduce')):
                    with self.assertRaises(ValueError): plotter.authenticate(root, manifest_hashes(root))

    def test_native_frame_bytes_and_terminal_are_bound(self):
        for mutation in ('frames', 'terminal', 'replaced_manifest'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve(); campaign(root)
                pins = manifest_hashes(root)
                path = root/'discovery/strict-native'/('assessment.json' if mutation == 'terminal' else 'frames.jsonl')
                with path.open('a') as stream: stream.write(' ')
                if mutation == 'replaced_manifest':
                    manifest_path = path.parent/'manifest.json'; manifest = plotter.driver.read(manifest_path)
                    manifest['files']['frames.jsonl'] = plotter.sha(path); put(manifest_path, manifest)
                with self.assertRaises(ValueError): plotter.authenticate(root, pins)

    def test_fresh_output_and_report_preserve_descriptive_scope(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()/'campaign'; root.mkdir(); campaign(root)
            output = root.parent/'presentation'
            def render(summaries, destination):
                self.assertEqual(set(summaries), set(plotter.ARMS))
                for suffix in ('png', 'svg'): (destination/('native-growth.'+suffix)).write_text('synthetic figure')
                return 'synthetic'
            with mock.patch.object(plotter, 'render', side_effect=render):
                receipt = plotter.plot(root, output, manifest_hashes(root))
            result = plotter.driver.read(receipt)
            self.assertEqual(result['saved_native_rows'], 2002)
            self.assertEqual(result['new_geometry_queries'], 0)
            self.assertTrue(result['frozen_input_byte_hashing'])
            self.assertIn('not independent replicates', (output/'report.md').read_text())
            self.assertIn('not necessarily cycle-certified', (output/'report.md').read_text())
            with self.assertRaisesRegex(ValueError, 'Fresh output'): plotter.plot(root, output, manifest_hashes(root))


if __name__ == '__main__': unittest.main()
