#!/usr/bin/env python3
"""Compare two completed, independently seeded shoulder budgets under a frozen plan.

Reads saved audits and checks byte identities; never launches or replays MC.
All 12 runs and retained repeats remain separate. Apparent ESS is descriptive,
and additive event/CPU summaries are not pooled-chain autocorrelations.
"""
from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path
import shutil

from plot_shoulder_docking_benchmark import (
    LABELS, MODES, STARTS, ROOT, load_completed, near, read, require, sha, write,
)
import numpy as np

DEFAULT_PLAN = ROOT/'runs/ab-shoulder-docking-repeat-comparison-plan-20260921/plan.json'
PLAN_SHA256 = '097798cebf3bc5caa8d6c0c0c69f5f29f15c7228e00523f032b92410aa769b20'
DEFAULT_PILOT = ROOT/'runs/ab-shoulder-docking-pilot-12x5000-20260921'
DEFAULT_REPEAT = ROOT/'runs/ab-shoulder-docking-repeat-12x40000-20260921'
DEFAULT_PILOT_ANALYSIS = ROOT/'runs/ab-shoulder-docking-assessment-20260921/analysis.json'
DEFAULT_REPEAT_ANALYSIS = ROOT/'runs/ab-shoulder-docking-repeat-assessment-20260921/analysis.json'
UNUSED_NORMALIZER_CHANGES = {'source/src/normalizer.rs', 'source/src/latent_region.rs'}
SCOPE = (
    'Independent fixed budgets, all 12 runs per budget and every retained repeat. '
    'Apparent label and atomic-contact ESS are individual finite-record statistics; '
    'occupancy/reference, half and initialization discrepancies remain separate. '
    'No stationary efficiency or speedup is established by apparent ESS alone. '
    'No concatenation, refits, thinning, optional stopping, full-target native '
    'escape, assembly, or physical kinetic-rate inference.'
)


def validate_allocation(manifest, campaign):
    """Check the entire factorial allocation and exact commands without execution."""
    jobs = manifest['jobs']
    require(len(jobs) == len({j['id'] for j in jobs}) == 12, 'Need all 12 distinct runs')
    seeds = manifest['seeds']
    require(len(seeds) == 12 and all(type(s) is int for s in seeds), 'Invalid seed allocation')
    require(seeds == [manifest.get('seed_base', seeds[0])+1009*i for i in range(12)],
            'Seed allocation differs from the frozen independent streams')
    require(manifest['total_attempts'] == 12*3*manifest['cycles'], 'Attempt budget differs')
    allocation = [(s, r, m, 'local' if m == 'local' else 'posterior-involution', .9 if m == 'c09' else 0.)
                  for s in STARTS for r in (0, 1) for m in MODES]
    for index, (job, design) in enumerate(zip(jobs, allocation)):
        require(tuple(job[k] for k in ('start', 'replicate', 'mode', 'method', 'correlation')) == design,
                'Method/initialization/replicate allocation changed')
        require(job['index'] == index and job['seed'] == seeds[index], 'Job stream differs')
        expected = [str(campaign/'provenance/docking-mc'), '--config', job['config'],
                    '--model', str(campaign/'provenance/model.json'), '--out', job['directory'],
                    '--cycles', str(manifest['cycles']), '--sample-every', '1',
                    '--method', job['method'], '--correlation', str(job['correlation'])]
        require(job['command'] == expected, 'Frozen command allocation changed')


def validate_comparison_plan(plan_path, plan_sha256, pilot, repeat, pilot_analysis):
    """Bind input identity before looking at completed-run outcomes."""
    require(sha(plan_path) == plan_sha256, 'Frozen comparison plan SHA-256 differs')
    plan = read(plan_path)
    require(plan['schema'] == 'shoulder-docking-independent-length-control-v1', 'Unknown comparison plan')
    for key in ('frozen_before_repeat_trajectories', 'unchanged_physical_and_proposal_law', 'seed_streams_disjoint'):
        require(plan[key] is True, 'Plan does not declare the frozen independent control: '+key)
    required = {str(pilot/'manifest.json'), str(repeat/'manifest.json'), str(pilot_analysis)}
    require(set(plan['inputs']) == required, 'Comparison plan does not bind exactly these campaigns and pilot audit')
    for name, digest in plan['inputs'].items():
        require(sha(name) == digest, 'Frozen comparison input changed: '+name)
    old, new = read(pilot/'manifest.json'), read(repeat/'manifest.json')
    require(old['cycles'] == plan['pilot_cycles'] and new['cycles'] == plan['repeat_cycles']
            and new['burn_cycles'] == plan['repeat_burn_cycles'], 'Fixed comparison budget changed')
    require(new['cycles'] > old['cycles'], 'Repeat must be the longer independent budget')
    for manifest, campaign in ((old, pilot), (new, repeat)):
        validate_allocation(manifest, campaign)
    require(set(old['seeds']).isdisjoint(new['seeds']), 'Pilot and repeat seed streams overlap')
    # All metadata not explicitly related to budget, provenance or execution is fixed.
    varying = {'jobs', 'seeds', 'seed_base', 'cycles', 'burn_cycles', 'total_attempts',
               'workers', 'binary_sha256', 'input_sha256', 'source_paths'}
    require({k: v for k, v in old.items() if k not in varying} ==
            {k: v for k, v in new.items() if k not in varying}, 'Physical target, labels, reference, or proposal law changed')
    fixed_inputs = {'model.json', 'shape.json', 'input-config.json', 'input-protocol.json',
                    'physical-reference-assessment.json', 'geometric-direct.json',
                    'geometric-mixture.json', 'geometric-geometry.json',
                    'Cargo.toml', 'Cargo.lock', 'build.rs', 'source/vendor/README.md'}
    require(old['source_bundle_files'] == new['source_bundle_files'], 'Executable source bundle membership changed')
    fixed_inputs.update(old['source_bundle_files'].values())
    changed_sources = {}
    for name in sorted(fixed_inputs):
        before, after = old['input_sha256'][name], new['input_sha256'][name]
        if name in UNUSED_NORMALIZER_CHANGES and before != after:
            changed_sources[name] = dict(pilot_sha256=before, repeat_sha256=after)
        else:
            require(before == after, 'Frozen physical/proposal source changed: '+name)
    for previous, current in zip(old['jobs'], new['jobs']):
        require(previous['id'] == current['id'], 'Matched run identities changed')
        configs = []
        for job, manifest in ((previous, old), (current, new)):
            require(sha(job['config']) == job['config_sha256'], 'Per-run configuration changed')
            cfg = read(job['config'])
            require(cfg['seed'] == job['seed'], 'Configuration seed differs')
            require(sha(cfg['shape']) == manifest['shape_sha256'], 'Configuration shape identity changed')
            require(cfg['target_region'] == manifest['target_region'], 'Configuration target differs')
            configs.append({k: v for k, v in cfg.items() if k not in ('shape', 'seed')})
        require(configs[0] == configs[1], 'Physical/proposal configuration changed: '+previous['id'])
    return plan, dict(unchanged_physical_and_proposal_law=True, disjoint_seed_streams=True,
                     allowed_unused_normalizer_source_changes=changed_sources,
                     pilot_manifest_sha256=sha(pilot/'manifest.json'),
                     repeat_manifest_sha256=sha(repeat/'manifest.json'))


def checked_ess(statistic, samples, cpu, constant=None):
    """Use the full post-burn CPU including every slot, rejection and frame tail."""
    result = copy.deepcopy(statistic)
    require(result['samples'] == samples, 'ESS record length differs')
    apparent, rate = result['apparent_ess'], result['apparent_ess_per_sampler_cpu_second']
    if constant is True:
        require(apparent is None, 'Constant descriptor must have unresolved ESS')
    if apparent is None:
        require(rate is None and result.get('reason') and result['iact_samples'] is None,
                'Unresolved ESS must retain its reason and null rate')
    else:
        require(math.isfinite(apparent) and 0 < apparent <= samples, 'Invalid apparent ESS')
        near(apparent, samples/result['iact_samples'])
        near(rate, apparent/cpu)
    return result


def checked_flux(flux, attempts, cpu):
    """Validate saved all-attempt counts; retain self-loops and absent sources."""
    require(tuple(flux['labels']) == LABELS, 'Flux labels changed')
    raw = flux['counts']
    require(len(raw) == 5 and all(len(row) == 5 for row in raw), 'Invalid flux matrix')
    require(all(type(n) is int and n >= 0 for row in raw for n in row), 'Invalid flux counts')
    matrix = np.asarray(raw, dtype=np.int64)
    require(int(matrix.sum()) == attempts, 'Missing attempts or self-loops in flux')
    require(flux['source_counts'] == matrix.sum(axis=1).tolist(), 'Flux source counts differ')
    require(flux['self_loops'] == int(np.trace(matrix)) and
            flux['off_diagonal_events'] == attempts-int(np.trace(matrix)), 'Flux event counts differ')
    require(flux['absent_source_rows'] == [LABELS[i] for i, n in enumerate(matrix.sum(axis=1)) if n == 0],
            'Absent flux-source rows changed')
    for i in range(5):
        for j in range(5):
            near(flux['empirical_flux_per_attempt'][i][j], matrix[i, j]/attempts)
            near(flux['conditional_transition_probabilities'][i][j],
                 matrix[i, j]/matrix[i].sum() if matrix[i].sum() else 0.)
    result = copy.deepcopy(flux)
    result['counts_per_full_postburn_CPU_second'] = (matrix/cpu).tolist()
    return result


def diagnostics(detail, record, retained):
    """Check saved descriptive statistics; no geometry or gate calculation."""
    cpu = record['full_postburn_CPU_seconds']
    require(math.isfinite(cpu) and cpu > 0, 'Invalid full post-burn CPU denominator')
    require(detail['total_sampler_cpu_seconds'] >= cpu, 'Full-run CPU is smaller than retained CPU')
    labels = {name: checked_ess(detail['label_ess'][name], retained, cpu,
                              constant=record['occupancy'][i] in (0., 1.))
              for i, name in enumerate(LABELS)}
    joint_labels = checked_ess(detail['joint_label_ess'], retained, cpu,
                               constant=max(record['occupancy']) == 1.)
    contact = detail['contact_incidence']
    require(set(contact['neighbors']) == {'A', 'B'}, 'Both contact neighbors must remain separate')
    neighbors, constants = {}, []
    for name, descriptor in contact['neighbors'].items():
        means = descriptor['mean_incidence']
        require(means and all(math.isfinite(x) and 0 <= x <= 1 for x in means), 'Invalid contact incidence mean')
        constant = all(x in (0., 1.) for x in means)
        constants.append(constant)
        neighbors[name] = {k: copy.deepcopy(v) for k, v in descriptor.items()
                           if k not in ('mean_incidence', 'first_half_mean_incidence', 'last_half_mean_incidence')}
        neighbors[name]['apparent_effective_count'] = checked_ess(descriptor['apparent_effective_count'], retained, cpu, constant)
        first, last = descriptor['first_half_mean_incidence'], descriptor['last_half_mean_incidence']
        require(len(first) == len(last) == len(means), 'Contact half descriptor dimensions differ')
        require(all(math.isfinite(x) and 0 <= x <= 1 for x in first+last), 'Invalid half contact mean')
        for average, a, b in zip(means, first, last):
            near(average, (a*(retained//2)+b*(retained-retained//2))/retained)
        near(descriptor['half_mean_incidence_rms_difference'],
             np.sqrt(np.mean((np.asarray(first)-np.asarray(last))**2)))
    joint_contact = checked_ess(contact['joint_apparent_effective_count'], retained, cpu, all(constants))
    total = checked_flux(detail['all_attempt_flux'], 3*retained, cpu)
    require(set(detail['attempt_flux_by_slot']) == {'0', '1', '2'}, 'Missing attempt slot')
    slots = {name: checked_flux(value, retained, cpu) for name, value in detail['attempt_flux_by_slot'].items()}
    kernel_counts = {'local': 3*retained} if record['mode'] == 'local' else {'local': 2*retained, 'global': retained}
    require(set(detail['attempt_flux_by_kernel']) == set(kernel_counts), 'Kernel allocation differs')
    kernels = {name: checked_flux(value, kernel_counts[name], cpu) for name, value in detail['attempt_flux_by_kernel'].items()}
    allocation = detail['attempt_cpu_allocation']
    near(allocation['full_postburn_cpu_seconds'], cpu)
    near(allocation['total_allocated_attempt_cpu_seconds']+allocation['final_frame_cpu_tail_seconds'], cpu)
    require(allocation['final_frame_cpu_tail_seconds'] >= -1e-9, 'Negative frame CPU tail')
    for group, data, counts in (('by_slot', slots, dict.fromkeys(slots, retained)), ('by_kernel', kernels, kernel_counts)):
        require(set(allocation[group]) == set(data), 'Incomplete attempt CPU allocation')
        require(np.array_equal(sum(np.asarray(x['counts']) for x in data.values()), np.asarray(total['counts'])),
                'Slot/kernel counts do not sum to all-attempt flux')
        near(sum(x['allocated_sampler_cpu_seconds'] for x in allocation[group].values()),
             allocation['total_allocated_attempt_cpu_seconds'])
        for name, value in allocation[group].items():
            require(value['attempts'] == counts[name] and 0 <= value['accepted'] <= counts[name], 'Attempt CPU count differs')
            require(value['label_changes'] == data[name]['off_diagonal_events'], 'Allocated CPU exchange count differs')
            require(value['allocated_sampler_cpu_seconds'] >= 0, 'Negative allocated CPU')
    trips = copy.deepcopy(detail['pair_roundtrips']['direct|geometry'])
    require(type(trips['completed']) is int and trips['completed'] == len(trips['trips']), 'Roundtrip count differs')
    previous = -1
    for trip in trips['trips']:
        first, last = trip['first_attempt'], trip['last_attempt']
        require(type(first) is int and type(last) is int and 0 <= first < last <= 3*retained
                and first >= previous and trip['start_label'] in ('direct', 'geometry')
                and trip['elapsed_attempts'] == last-first, 'Invalid or overlapping pair roundtrip')
        previous = last
    trips['per_full_postburn_CPU_second'] = trips['completed']/cpu
    return dict(total_sampler_CPU_seconds=detail['total_sampler_cpu_seconds'],
                label_apparent_ESS=labels, joint_label_apparent_ESS=joint_labels,
                joint_atomic_contact_apparent_ESS=joint_contact,
                atomic_contact_neighbors=neighbors, atomic_contact_descriptor=contact['descriptor'],
                direct_geometry_roundtrips=trips, all_attempt_flux=total,
                attempt_flux_by_slot=slots, attempt_flux_by_kernel=kernels,
                attempt_CPU_allocation=copy.deepcopy(allocation),
                observer_wall_seconds=detail['analysis_wall_seconds'])


def campaign_data(analysis, campaign):
    data, inputs, archives = load_completed(analysis, campaign)
    records = []
    for record in data['records']:
        detail = read(analysis.parent/'runs'/record['id']/'analysis.json')
        row = {k: v for k, v in record.items() if k != 'labels'}
        row.update(diagnostics(detail, record, data['retained_cycles']))
        row['occupancy_minus_observed_reference'] = detail['reference_comparison']['occupancy_minus_observed_reference']
        for i, label in enumerate(LABELS):
            near(row['occupancy_minus_observed_reference'][label],
                 row['occupancy'][i]-data['reference_probabilities'][i])
        row['unvisited_physical_labels'] = [name for name, value in zip(LABELS, record['occupancy']) if value == 0.]
        records.append(row)
    summary = read(analysis)
    for mode in MODES:
        initialization = summary['initialization_comparison'][mode]
        for start in STARTS:
            selected = [r for r in records if r['mode'] == mode and r['start'] == start]
            for i, label in enumerate(LABELS):
                near(initialization['mean_occupancy_by_initialization'][start][label],
                     sum(r['occupancy'][i] for r in selected)/len(selected))
        near(initialization['largest_run_half_total_variation'],
             max(r['half_total_variation'] for r in records if r['mode'] == mode))
    return dict(total_cycles=data['total_cycles'], burn_cycles=data['burn_cycles'],
                retained_cycles_per_run=data['retained_cycles'], records=records,
                initialization_comparison=summary['initialization_comparison'],
                physical_reference=data['reference_source'],
                mode_summaries=mode_summaries(records), efficiency_conclusion=summary['efficiency_conclusion']), inputs, archives


def aggregate_flux(fluxes, cpu):
    matrix = sum(np.asarray(f['counts'], dtype=np.int64) for f in fluxes)
    attempts = int(matrix.sum())
    return dict(labels=LABELS, counts=matrix.tolist(), attempts=attempts,
                empirical_flux_per_attempt=(matrix/attempts).tolist(),
                counts_per_full_postburn_CPU_second=(matrix/cpu).tolist(),
                self_loops=int(np.trace(matrix)), off_diagonal_events=attempts-int(np.trace(matrix)))


def mode_summaries(records):
    result = {}
    for mode in MODES:
        selected = [r for r in records if r['mode'] == mode]
        require(len(selected) == 4, 'Need all four independent runs per mode')
        cpu = sum(r['full_postburn_CPU_seconds'] for r in selected)
        trips = sum(r['direct_geometry_roundtrips']['completed'] for r in selected)
        result[mode] = dict(run_ids=[r['id'] for r in selected],
            full_run_sampling_CPU_seconds=sum(r['total_sampler_CPU_seconds'] for r in selected),
            full_postburn_sampling_CPU_seconds=cpu, direct_geometry_roundtrips=trips,
            individual_roundtrips=[r['direct_geometry_roundtrips']['completed'] for r in selected],
            roundtrips_per_full_postburn_CPU_second=trips/cpu,
            all_attempt_flux=aggregate_flux([r['all_attempt_flux'] for r in selected], cpu),
            attempt_flux_by_slot={key: aggregate_flux([r['attempt_flux_by_slot'][key] for r in selected], cpu)
                                  for key in ('0', '1', '2')},
            attempt_flux_by_kernel={key: aggregate_flux([r['attempt_flux_by_kernel'][key] for r in selected], cpu)
                                    for key in selected[0]['attempt_flux_by_kernel']})
    return result


def length_comparisons(campaigns):
    """Side-by-side finite-record contrasts; never compute a pooled ESS."""
    modes, strata = {}, []
    for mode in MODES:
        before = campaigns['pilot']['mode_summaries'][mode]
        after = campaigns['repeat']['mode_summaries'][mode]
        old_rate, new_rate = [x['roundtrips_per_full_postburn_CPU_second'] for x in (before, after)]
        modes[mode] = dict(pilot_roundtrips_per_full_postburn_CPU_second=old_rate,
            repeat_roundtrips_per_full_postburn_CPU_second=new_rate,
            repeat_minus_pilot_roundtrips_per_full_postburn_CPU_second=new_rate-old_rate,
            repeat_over_pilot_roundtrip_rate_ratio=new_rate/old_rate if old_rate > 0 else None,
            ratio_unresolved_reason=None if old_rate > 0 else 'Pilot observed zero roundtrips; no finite rate ratio.',
            interpretation='Observed event-count/CPU contrast between independent fixed budgets, not stationary efficiency or physical kinetics.')
    old_records = {r['id']: r for r in campaigns['pilot']['records']}
    new_records = {r['id']: r for r in campaigns['repeat']['records']}
    require(len(old_records) == len(new_records) == 12 and old_records.keys() == new_records.keys(),
            'Independent budgets do not contain the same 12 design strata')
    for identity, before in old_records.items():
        after = new_records[identity]
        strata.append(dict(id=identity, pilot_seed=before['seed'], repeat_seed=after['seed'],
            repeat_minus_pilot_occupancy={label: after['occupancy'][i]-before['occupancy'][i] for i, label in enumerate(LABELS)},
            repeat_minus_pilot_reference_total_variation=after['reference_total_variation']-before['reference_total_variation'],
            repeat_minus_pilot_half_total_variation=after['half_total_variation']-before['half_total_variation']))
    return dict(by_mode=modes, individual_design_strata=strata,
                interpretation='Matched design strata have independent seeds; differences are not paired-trajectory uncertainty estimates.')


def markdown(data):
    lines = ['# Independent shoulder length control', '', SCOPE, '',
             'CPU denominators include all sampling slots, rejected moves, gate work and recorded overhead. '
             'Observer wall time is separate. Roundtrips use all post-burn attempts, including self-loops; '
             'occupancy and ESS use every retained cycle endpoint. Matched rows denote design strata, not paired seed streams.', '',
             '| Budget | Mode | Roundtrips (individual runs) | Full post-burn CPU s | Roundtrips / CPU s | Initialization TV | Largest half TV |',
             '|---|---|---|---:|---:|---:|---:|']
    for budget in ('pilot', 'repeat'):
        campaign = data['campaigns'][budget]
        for mode, summary in campaign['mode_summaries'].items():
            init = campaign['initialization_comparison'][mode]
            lines.append(f"| {campaign['total_cycles']:,} cycles | {mode} | {summary['direct_geometry_roundtrips']} ({', '.join(map(str, summary['individual_roundtrips']))}) | "
                         f"{summary['full_postburn_sampling_CPU_seconds']:.4f} | {summary['roundtrips_per_full_postburn_CPU_second']:.6g} | "
                         f"{init['initialization_mean_total_variation']:.6g} | {init['largest_run_half_total_variation']:.6g} |")
    lines += ['', 'Individual rows below preserve all runs. Null apparent ESS means unresolved, including constant descriptors; '
              'no ESS values are pooled or averaged into a speedup. Full occupancies, retained halves, all five label ESS values, '
              'neighbor contact diagnostics and slot/kernel flux matrices are in `comparison.json`.', '',
              '| Budget | Run | Reference TV | Half TV | Joint contact apparent ESS / CPU s | Joint label apparent ESS / CPU s |',
              '|---|---|---:|---:|---:|---:|']
    for budget in ('pilot', 'repeat'):
        for row in data['campaigns'][budget]['records']:
            values = [row[k]['apparent_ess_per_sampler_cpu_second'] for k in ('joint_atomic_contact_apparent_ESS', 'joint_label_apparent_ESS')]
            rates = ['unresolved' if x is None else f'{x:.6g}' for x in values]
            lines.append(f"| {budget} | {row['id']} | {row['reference_total_variation']:.6g} | {row['half_total_variation']:.6g} | {' | '.join(rates)} |")
    return '\n'.join(lines)+'\n'


def compare(plan_path, pilot, repeat, pilot_analysis, repeat_analysis, out, plan_sha256=PLAN_SHA256):
    plan_path, pilot, repeat, pilot_analysis, repeat_analysis, out = [Path(p).resolve() for p in
        (plan_path, pilot, repeat, pilot_analysis, repeat_analysis, out)]
    require(not out.exists(), 'Fresh comparison output directory required')
    plan, checks = validate_comparison_plan(plan_path, plan_sha256, pilot, repeat, pilot_analysis)
    inputs = {str(plan_path): plan_sha256}
    archive = {'plan.json': plan_path}
    campaigns = {}
    for name, analysis, campaign in (('pilot', pilot_analysis, pilot), ('repeat', repeat_analysis, repeat)):
        campaigns[name], hashes, files = campaign_data(analysis, campaign)
        inputs.update(hashes)
        archive.update({name+'/'+key: path for key, path in files.items()})
    require(campaigns['pilot']['physical_reference'] == campaigns['repeat']['physical_reference'], 'Physical reference changed')
    source = Path(__file__).resolve()
    inputs[str(source)] = sha(source)
    archive[source.name] = source
    data = dict(schema='shoulder-docking-independent-length-comparison-v1', complete=True,
                plan_sha256=plan_sha256, validation=checks, labels=LABELS, campaigns=campaigns,
                independent_length_comparison=length_comparisons(campaigns),
                scope=SCOPE, frozen_comparison_rules=plan['rules'],
                stationary_efficiency_established=False,
                flux_interpretation='Additive all-attempt counts are summed only within a budget/mode. '
                'Slots and kernels remain separate; ordered cycles need not be reversible. '
                'Finite flux imbalance and conditional transition rates are not stationarity tests or kinetic rates.')
    out.mkdir(parents=True)
    provenance = out/'provenance'
    for name, path in archive.items():
        target = provenance/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        require(sha(target) == inputs[str(path)], 'Input changed while archiving: '+str(path))
    write(out/'comparison.json', data)
    (out/'comparison.md').write_text(markdown(data))
    write(out/'provenance.json', dict(complete=True, input_sha256=inputs,
        archived_sha256={str(p.relative_to(provenance)): sha(p) for p in provenance.rglob('*') if p.is_file()},
        output_sha256={p.name: sha(p) for p in out.iterdir() if p.is_file()},
        verification='Frozen plan, complete audits, terminal statuses, source hashes and saved arithmetic. No physical replay.',
        scope=SCOPE))
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=DEFAULT_PLAN)
    parser.add_argument('--plan-sha256', default=PLAN_SHA256, help='Expected frozen plan digest; default is the predeclared real control')
    parser.add_argument('--pilot', type=Path, default=DEFAULT_PILOT)
    parser.add_argument('--repeat', type=Path, default=DEFAULT_REPEAT)
    parser.add_argument('--pilot-analysis', type=Path, default=DEFAULT_PILOT_ANALYSIS)
    parser.add_argument('--repeat-analysis', type=Path, default=DEFAULT_REPEAT_ANALYSIS)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    compare(args.plan, args.pilot, args.repeat, args.pilot_analysis, args.repeat_analysis, args.out, args.plan_sha256)
    print(args.out.resolve()/'comparison.md')


if __name__ == '__main__':
    main()
