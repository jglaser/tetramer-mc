#!/usr/bin/env python3
"""Synthetic comparison checks; no binaries, physical audits or live campaigns."""
from __future__ import annotations

import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np

from analyze_shoulder_docking_benchmark import allocated_attempt_cpu, descriptor_ess, pair_roundtrips, transitions
from compare_shoulder_docking_repeat import (
    LABELS, MODES, STARTS, checked_ess, compare, mode_summaries,
    read, sha, validate_comparison_plan, write,
)


def fixture_campaign(root, name, reference, cycles, burn, seed_base, constant=False):
    campaign, assessment = root/name, root/(name+'-assessment')
    provenance = campaign/'provenance'
    provenance.mkdir(parents=True)
    assessment.mkdir()
    source_files = {name: name for name in ('Cargo.toml', 'Cargo.lock', 'build.rs')}
    source_files.update({name: 'source/'+name for name in
                        ('vendor/README.md', 'src/docking.rs', 'src/normalizer.rs', 'src/latent_region.rs')})
    input_names = set(source_files.values()) | {
        'model.json', 'shape.json', 'input-config.json', 'input-protocol.json', 'geometric-direct.json',
        'geometric-mixture.json', 'geometric-geometry.json', 'analyze_shoulder_docking_benchmark.py', 'docking-mc'}
    for filename in input_names:
        path = provenance/filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('Synthetic inert fixture: '+filename+'\n')
    (provenance/'physical-reference-assessment.json').write_bytes(Path(reference['source_path']).read_bytes())
    hashes = {str(p.relative_to(provenance)): sha(p) for p in provenance.rglob('*') if p.is_file()}
    target = dict(metric=dict(member_error_scale=2., angle_error_scale_deg=15.),
                  window=dict(minimum=1., maximum=2., lower_inclusive=False, upper_inclusive=False))
    jobs, summaries = [], []
    retained = cycles-burn
    for start in STARTS:
        for replicate in (0, 1):
            for mode in MODES:
                index = len(jobs)
                identity = f'{start}-{replicate}-{mode}'
                run = campaign/'runs'/identity
                run.mkdir(parents=True)
                config_path = campaign/'configs'/(identity+'.json')
                config_path.parent.mkdir(exist_ok=True)
                seed = seed_base+1009*index
                write(config_path, dict(seed=seed, shape=str(provenance/'shape.json'), target_region=target,
                                        capture_radius=18., proposal_anchor_index=0, uniform_probability=.05,
                                        initial_pose=start, fixed_poses=['A', 'B'], local_attempts_per_cycle=2))
                method, correlation = ('local', 0.) if mode == 'local' else ('posterior-involution', .9 if mode == 'c09' else 0.)
                job = dict(id=identity, index=index, start=start, replicate=replicate, mode=mode,
                           method=method, correlation=correlation, seed=seed, config=str(config_path),
                           config_sha256=sha(config_path), directory=str(run))
                job['command'] = [str(provenance/'docking-mc'), '--config', str(config_path),
                                  '--model', str(provenance/'model.json'), '--out', str(run), '--cycles', str(cycles),
                                  '--sample-every', '1', '--method', method, '--correlation', str(correlation)]
                jobs.append(job)
                for filename in ('moves.jsonl', 'trajectory.jsonl', 'summary.json', 'checkpoint.json'):
                    (run/filename).write_text('Synthetic source bytes, not a physical trajectory: '+filename+'\n')
                offset = 0 if start == 'direct' else 2
                labels = np.array([offset if constant else ((i//2+replicate+(start == 'geometry')) % 2)*2
                                   for i in range(retained)])
                half = retained//2
                occupancy = lambda values: {name: float(np.mean(values == i)) for i, name in enumerate(LABELS)}
                occ, first, last = occupancy(labels), occupancy(labels[:half]), occupancy(labels[half:])
                cpu = float(retained*4)
                cpu_start = float(5+burn*4)
                cpu_end = float(5+cycles*4)
                q_rows = [dict(cycle=i, label=LABELS[offset if i <= burn else labels[i-burn-1]],
                               q=1.05, sampler_cpu_seconds=float(5+4*i)) for i in range(cycles+1)]
                # Every cycle has all three attempts; repeats are retained explicitly.
                attempt_labels = [offset]+[int(label) for label in labels for _ in range(3)]
                moves = [dict(attempt=i % 3, kind='local' if mode == 'local' or i % 3 < 2 else 'global',
                              accepted=attempt_labels[i] != attempt_labels[i+1],
                              sampler_cpu_seconds=cpu_start+(i+1)*(cpu-3)/(3*retained))
                         for i in range(3*retained)]
                neighbors = {}
                bits = np.column_stack([labels == 0, labels == 2]).astype(float)
                for neighbor, values in (('A', bits), ('B', bits[:, ::-1])):
                    neighbors[neighbor] = dict(apparent_effective_count=descriptor_ess(values, cpu), lags=[],
                        mean_incidence=values.mean(axis=0).tolist(),
                        first_half_mean_incidence=values[:half].mean(axis=0).tolist(),
                        last_half_mean_incidence=values[half:].mean(axis=0).tolist(),
                        half_mean_incidence_rms_difference=float(np.sqrt(np.mean(
                            (values[:half].mean(axis=0)-values[half:].mean(axis=0))**2))))
                differences = {label: occ[label]-reference['regions'][label]['probability'] for label in LABELS}
                detail = dict(id=identity, mode=mode, start=start, replicate=replicate, seed=seed, passed=True,
                    burn_cycles=burn, retained_cycles=retained, sampler_cpu_seconds_after_burn=cpu,
                    total_sampler_cpu_seconds=cpu_end, occupancy=occ, first_half_occupancy=first, last_half_occupancy=last,
                    half_occupancy_total_variation=.5*sum(abs(first[k]-last[k]) for k in LABELS),
                    reference_comparison=dict(occupancy_minus_observed_reference=differences,
                                              total_variation=.5*sum(abs(x) for x in differences.values())),
                    label_ess={label: descriptor_ess(labels == i, cpu) for i, label in enumerate(LABELS)},
                    joint_label_ess=descriptor_ess(np.eye(5)[labels], cpu),
                    q_rows=q_rows, audit=dict(all_moves_replayed=3*cycles, independent_geometry_frames=retained),
                    source_sha256={filename: sha(run/filename) for filename in
                                   ('moves.jsonl', 'trajectory.jsonl', 'summary.json', 'checkpoint.json')},
                    contact_incidence=dict(samples=retained, stride_cycles=1,
                        rows=[dict(cycle=i) for i in range(burn+1, cycles+1)], neighbors=neighbors,
                        joint_apparent_effective_count=descriptor_ess(np.column_stack([bits, bits[:, ::-1]]), cpu),
                        descriptor='Synthetic separate A+B binary atomic incidence; every repeat retained.'),
                    all_attempt_flux=transitions(attempt_labels),
                    attempt_flux_by_slot={str(i): transitions(attempt_labels, [m['attempt'] == i for m in moves]) for i in range(3)},
                    attempt_flux_by_kernel={k: transitions(attempt_labels, [m['kind'] == k for m in moves]) for k in sorted({m['kind'] for m in moves})},
                    attempt_cpu_allocation=allocated_attempt_cpu(moves, attempt_labels, cpu_start, cpu_end),
                    pair_roundtrips=pair_roundtrips(attempt_labels), analysis_wall_seconds=999.)
                detail_path = assessment/'runs'/identity/'analysis.json'
                detail_path.parent.mkdir(parents=True)
                write(detail_path, detail)
                summaries.append({k: detail[k] for k in ('id', 'mode', 'start', 'replicate', 'occupancy',
                    'first_half_occupancy', 'last_half_occupancy', 'label_ess', 'joint_label_ess',
                    'sampler_cpu_seconds_after_burn', 'half_occupancy_total_variation')})
    manifest = dict(schema='conditional-shoulder-docking-v1', jobs=jobs, cycles=cycles, burn_cycles=burn,
        workers=4, sample_every=1, attempts_per_cycle=3, total_attempts=12*3*cycles,
        physical_labels=LABELS, target_region=target, physical_reference=reference, input_sha256=hashes,
        source_bundle_files=source_files, model_sha256=hashes['model.json'], shape_sha256=hashes['shape.json'],
        binary_sha256=hashes['docking-mc'], seeds=[j['seed'] for j in jobs], seed_base=seed_base)
    write(campaign/'manifest.json', manifest)
    write(campaign/'status.json', dict(complete=True, running=False,
        jobs=[dict(id=j['id'], exit_code=0, status='complete') for j in jobs]))
    initializations = {}
    for mode in MODES:
        means = {s: {label: float(np.mean([r['occupancy'][label] for r in summaries if r['mode'] == mode and r['start'] == s]))
                     for label in LABELS} for s in STARTS}
        initializations[mode] = dict(mean_occupancy_by_initialization=means,
            initialization_mean_total_variation=.5*sum(abs(means['direct'][k]-means['geometry'][k]) for k in LABELS),
            largest_run_half_total_variation=max(r['half_occupancy_total_variation'] for r in summaries if r['mode'] == mode))
    analysis = assessment/'analysis.json'
    write(analysis, dict(passed=True, runs=summaries, physical_reference=reference,
        campaign_manifest_sha256=sha(campaign/'manifest.json'), terminal_status_sha256=sha(campaign/'status.json'),
        analyzer_sha256=hashes['analyze_shoulder_docking_benchmark.py'],
        initialization_comparison=initializations, efficiency_conclusion='Synthetic apparent ESS; no equilibrium inference.'))
    return campaign, analysis


class RepeatComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='shoulder-comparison-test-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        reference_path = self.root/'reference.json'
        write(reference_path, dict(synthetic=True))
        reference = dict(source_path=str(reference_path), source_sha256=sha(reference_path),
            regions={name: dict(probability=.2, observed_delta_method_SE=.01) for name in LABELS})
        self.pilot, self.old_analysis = fixture_campaign(self.root, 'pilot', reference, 8, 2, 100)
        self.repeat, self.new_analysis = fixture_campaign(self.root, 'repeat', reference, 16, 4, 100000)
        self.plan_path = self.root/'plan.json'
        self.freeze_plan()

    def freeze_plan(self):
        plan = dict(schema='shoulder-docking-independent-length-control-v1', frozen_before_repeat_trajectories=True,
                    unchanged_physical_and_proposal_law=True, seed_streams_disjoint=True,
                    pilot_cycles=8, repeat_cycles=16, repeat_burn_cycles=4, rules=['Keep every run and repeat.'],
                    inputs={str(p): sha(p) for p in (self.pilot/'manifest.json', self.repeat/'manifest.json', self.old_analysis)})
        write(self.plan_path, plan)
        self.plan_hash = sha(self.plan_path)

    def run_compare(self):
        return compare(self.plan_path, self.pilot, self.repeat, self.old_analysis, self.new_analysis,
                       self.root/'out', self.plan_hash)

    def validate_plan(self):
        return validate_comparison_plan(self.plan_path, self.plan_hash, self.pilot, self.repeat, self.old_analysis)

    def test_end_to_end_keeps_all_runs_repeats_fluxes_and_full_cpu(self):
        data = self.run_compare()
        self.assertFalse(data['stationary_efficiency_established'])
        for budget, retained in (('pilot', 6), ('repeat', 12)):
            result = data['campaigns'][budget]
            self.assertEqual(len(result['records']), 12)
            for row in result['records']:
                self.assertEqual(row['full_postburn_CPU_seconds'], 4*retained)
                self.assertEqual(row['joint_label_apparent_ESS']['samples'], retained)
                self.assertAlmostEqual(row['joint_label_apparent_ESS']['apparent_ess_per_sampler_cpu_second'],
                                       row['joint_label_apparent_ESS']['apparent_ess']/(4*retained))
                self.assertEqual(sum(map(sum, row['all_attempt_flux']['counts'])), 3*retained)
                self.assertGreater(row['all_attempt_flux']['self_loops'], 0)
                self.assertEqual(set(row['attempt_flux_by_slot']), {'0', '1', '2'})
                self.assertIsNone(row['label_apparent_ESS']['mixture']['apparent_ess'])
            for summary in result['mode_summaries'].values():
                self.assertEqual(summary['full_postburn_sampling_CPU_seconds'], 16*retained)
                self.assertAlmostEqual(summary['roundtrips_per_full_postburn_CPU_second'],
                                       summary['direct_geometry_roundtrips']/(16*retained))
        self.assertTrue((self.root/'out/provenance.json').is_file())
        self.assertTrue((self.root/'out/comparison.md').is_file())
        with self.assertRaisesRegex(ValueError, 'Fresh comparison'):
            self.run_compare()

    def test_cpu_denominator_cannot_use_kernel_time_or_omit_tail(self):
        stat = descriptor_ess([0, 0, 1, 1, 0, 0], 24.)
        checked_ess(stat, 6, 24.)
        for denominator in (21., 8., 999., 32.):
            bad = dict(stat, apparent_ess_per_sampler_cpu_second=stat['apparent_ess']/denominator)
            with self.subTest(denominator=denominator), self.assertRaisesRegex(ValueError, 'Saved statistic differs'):
                checked_ess(bad, 6, 24.)

    def test_constant_descriptors_remain_unresolved(self):
        stat = descriptor_ess(np.ones((6, 3)), 24.)
        result = checked_ess(stat, 6, 24., constant=True)
        self.assertIsNone(result['apparent_ess'])
        self.assertIsNone(result['apparent_ess_per_sampler_cpu_second'])
        bad = dict(stat, apparent_ess=6., iact_samples=1., apparent_ess_per_sampler_cpu_second=.25)
        with self.assertRaisesRegex(ValueError, 'Constant descriptor'):
            checked_ess(bad, 6, 24., constant=True)

    def test_constant_joint_contact_and_labels_survive_full_comparison(self):
        reference = read(self.pilot/'manifest.json')['physical_reference']
        self.pilot, self.old_analysis = fixture_campaign(self.root, 'constant-pilot', reference, 8, 2, 100, constant=True)
        self.repeat, self.new_analysis = fixture_campaign(self.root, 'constant-repeat', reference, 16, 4, 100000, constant=True)
        self.freeze_plan()
        data = self.run_compare()
        for campaign in data['campaigns'].values():
            for row in campaign['records']:
                for descriptor in ('joint_atomic_contact_apparent_ESS', 'joint_label_apparent_ESS'):
                    self.assertIsNone(row[descriptor]['apparent_ess'])
                    self.assertIsNone(row[descriptor]['apparent_ess_per_sampler_cpu_second'])
                    self.assertIn('Constant', row[descriptor]['reason'])
        for mode in data['independent_length_comparison']['by_mode'].values():
            self.assertIsNone(mode['repeat_over_pilot_roundtrip_rate_ratio'])

    def test_mode_event_rate_uses_sum_of_cpu_not_mean_individual_rates(self):
        data = self.run_compare()
        records = data['campaigns']['pilot']['records']
        cpus, trips = [1., 2., 4., 8.], [1, 2, 0, 5]
        for mode in MODES:
            for row, cpu, count in zip([r for r in records if r['mode'] == mode], cpus, trips):
                row['full_postburn_CPU_seconds'] = cpu
                row['direct_geometry_roundtrips']['completed'] = count
        for summary in mode_summaries(records).values():
            self.assertAlmostEqual(summary['roundtrips_per_full_postburn_CPU_second'], 8/15)
            self.assertNotAlmostEqual(summary['roundtrips_per_full_postburn_CPU_second'],
                                      sum(n/cpu for n, cpu in zip(trips, cpus))/4)

    def test_missing_or_duplicate_runs_rejected_before_output(self):
        original = read(self.new_analysis)
        for runs in (original['runs'][:-1], original['runs'][:-1]+[original['runs'][0]]):
            write(self.new_analysis, dict(original, runs=runs))
            with self.assertRaises(ValueError):
                self.run_compare()
            self.assertFalse((self.root/'out').exists())

    def test_changed_target_and_proposal_config_rejected_even_if_refrozen(self):
        original = read(self.repeat/'manifest.json')
        job = original['jobs'][0]
        config_original = read(job['config'])
        for field, value in (('capture_radius', 19.), ('uniform_probability', .1)):
            write(job['config'], dict(config_original, **{field: value}))
            changed = copy.deepcopy(original)
            changed['jobs'][0]['config_sha256'] = sha(job['config'])
            write(self.repeat/'manifest.json', changed)
            self.freeze_plan()
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'Physical/proposal configuration changed'):
                self.validate_plan()
        write(job['config'], config_original)
        changed = copy.deepcopy(original)
        changed['target_region']['window']['maximum'] = 2.1
        write(self.repeat/'manifest.json', changed)
        self.freeze_plan()
        with self.assertRaisesRegex(ValueError, 'Physical target'):
            self.validate_plan()

    def test_overlap_seed_and_changed_docking_source_rejected(self):
        original = read(self.repeat/'manifest.json')
        old = read(self.pilot/'manifest.json')
        changed = copy.deepcopy(original)
        changed['seeds'], changed['seed_base'] = old['seeds'], old['seed_base']
        for job, seed in zip(changed['jobs'], changed['seeds']):
            job['seed'] = seed
        write(self.repeat/'manifest.json', changed)
        self.freeze_plan()
        with self.assertRaisesRegex(ValueError, 'seed streams overlap'):
            self.validate_plan()
        changed = copy.deepcopy(original)
        changed['input_sha256']['source/src/docking.rs'] = '0'*64
        write(self.repeat/'manifest.json', changed)
        self.freeze_plan()
        with self.assertRaisesRegex(ValueError, 'physical/proposal source changed'):
            self.validate_plan()

    def test_plan_and_audited_source_hash_changes_rejected(self):
        with self.assertRaisesRegex(ValueError, 'plan SHA-256'):
            validate_comparison_plan(self.plan_path, '0'*64, self.pilot, self.repeat, self.old_analysis)
        source = self.repeat/'runs/direct-0-local/moves.jsonl'
        source.write_text('Changed source after saved audit\n')
        with self.assertRaisesRegex(ValueError, 'audited source bytes changed'):
            self.run_compare()
        self.assertFalse((self.root/'out').exists())

    def test_missing_selfloops_and_wrong_slot_totals_are_rejected(self):
        path = self.new_analysis.parent/'runs/direct-0-local/analysis.json'
        original = read(path)
        for key in ('all_attempt_flux', 'attempt_flux_by_slot'):
            changed = copy.deepcopy(original)
            flux = changed[key] if key == 'all_attempt_flux' else changed[key]['0']
            flux['counts'][0][0] -= 1
            write(path, changed)
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'Missing attempts or self-loops'):
                self.run_compare()
            self.assertFalse((self.root/'out').exists())


if __name__ == '__main__':
    unittest.main()
