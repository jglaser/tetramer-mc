#!/usr/bin/env python3
"""Compare complete independent-stream finite-assembly blocks, without pooling.

Frozen observer outputs are authenticated and their lightweight retained-record
accounting is checked. Geometry, native classification and FFT estimates are not
repeated. This tool neither launches simulations nor certifies equilibrium.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform
import shutil

import numpy as np
import scipy
from scipy.stats import t as student_t

from analyze_contact_efficiency import (MATCHED_SCHEDULE, digest, physical_identity,
    read, require, sampled_exchanges, validate_effective_config, validate_frames)
from analyze_finite_assembly import SCHEMA as REPORT_SCHEMA, RUNTIME_SOURCES
from contact_benchmark_contract import validate_comparison
from finite_assembly_observer import REGIONS, _components
from prepare_finite_assembly_observer import rule_label


SCHEMA = 'finite-assembly-independent-stream-comparison-v1'
PREPARATIONS = ('dispersed', 'competing-aggregate', 'native-seeded')
FINGERPRINTS = ('fingerprint', 'native_fingerprint')
COMPONENTS = ('exclusion', 'native_raw', 'native_certified')
SIZES = ('largest_exclusion', 'largest_native_raw', 'largest_native_certified')
IMPLEMENTATION = ('compare_finite_assembly.py', 'analyze_contact_efficiency.py',
    'analyze_finite_assembly.py', 'contact_benchmark_contract.py',
    'finite_assembly_observer.py', 'prepare_finite_assembly_observer.py',
    'native_graph_consistency.py', 'native_contact_regions.py', 'mobile_posterior_metrics.py')
SCOPE = ('Four independent stream means per arm/preparation; trajectories are never concatenated. '
         'Student-t intervals and binwise contrasts are descriptive, not simultaneous confidence '
         'bounds or convergence guarantees. Zero observations and zero observed variance do not '
         'establish zero equilibrium probability. Apparent ESS/CPU is not physical kinetics or '
         'proof of important-region coverage. No equal directional transition rates are required.')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def close(a, b, message):
    require(math.isfinite(float(a)) and math.isfinite(float(b))
            and math.isclose(a, b, rel_tol=2e-12, abs_tol=2e-12), message)


def mean_statistics(values):
    x = np.asarray(values, float)
    require(x.ndim == 1 and len(x) >= 2 and np.isfinite(x).all(),
            'At least two finite independent population values required')
    se = float(x.std(ddof=1)/math.sqrt(len(x)))
    mean = float(x.mean())
    interval = None if se == 0 else [mean-float(student_t.ppf(.975, len(x)-1))*se,
                                    mean+float(student_t.ppf(.975, len(x)-1))*se]
    return dict(values=x.tolist(), populations=len(x), mean=mean,
        population_standard_error=se, descriptive_t95=interval,
        zero_observed_variance=se == 0,
        equilibrium_bound=False)


def _contrast(a, b):
    se_a, se_b = a['population_standard_error'], b['population_standard_error']
    se = math.hypot(se_a, se_b)
    delta = a['mean']-b['mean']
    if se:
        df = se**4/(se_a**4/(a['populations']-1)+se_b**4/(b['populations']-1))
        width = float(student_t.ppf(.975, df))*se
        interval = [delta-width, delta+width]
    else:
        df, interval = None, None
    return dict(difference=delta, combined_population_standard_error=se,
        descriptive_welch_t95=interval, descriptive_degrees_of_freedom=df,
        diagnostic_agreement_within_three_observed_se=abs(delta) <= 3*se if se else None,
        zero_observed_variance=se == 0, equilibrium_agreement_established=False)


def _fingerprint_rate(value, samples, cpu):
    require(value['samples'] == samples, 'Fingerprint sample denominator differs')
    close(value['sampling_CPU_seconds'], cpu, 'Fingerprint CPU denominator differs')
    ess, rate = value['apparent_ess'], value['apparent_ess_per_sampling_CPU_second']
    if ess is None:
        require(rate is None, 'Null ESS must retain a null CPU rate')
    else:
        require(type(ess) in (int, float) and math.isfinite(ess) and 0 < ess <= samples,
                'Invalid apparent ESS')
        close(rate, ess/cpu, 'ESS/CPU differs from retained-work denominator')
    return rate


def _validate_statistics(report):
    n = report['physical_identity']['bodies']
    require(type(n) is int and n > 0, 'Invalid body count')
    window = report['window']
    samples = (window['end_sweep']-window['burn_sweep'])//window['cadence_sweeps']
    require(samples > 0 and samples*window['cadence_sweeps'] == window['end_sweep']-window['burn_sweep'],
            'Window/cadence mismatch')
    cpu = window['cpu_seconds']
    require(type(cpu) in (int, float) and math.isfinite(cpu) and cpu > 0,
            'Positive full sampler CPU required')
    require(set(report['environment_occupancies']) == set(REGIONS), 'Incomplete exhaustive region inventory')
    total = 0
    for name in REGIONS:
        row = report['environment_occupancies'][name]
        count = row['observations']; require(type(count) is int and 0 <= count <= samples, 'Invalid region count')
        total += count
        close(row['fraction'], count/samples, 'Region denominator differs')
        _fingerprint_rate(row['apparent_ess'], samples, cpu)
    require(total == samples, 'Every retained endpoint must enter one region')
    for name in FINGERPRINTS:
        _fingerprint_rate(report[name], samples, cpu)
    for name in SIZES:
        row = report['size_statistics'][name]
        require(math.isfinite(row['mean']) and 1 <= row['mean'] <= n, 'Invalid mean component size')
        _fingerprint_rate(row['apparent_ess'], samples, cpu)
    for name in COMPONENTS:
        row = report['component_number_statistics'][name]
        require(row['sizes'] == list(range(1, n+1)), 'Missing component-size bins')
        values = row['mean_component_counts']
        require(len(values) == n and all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in values),
                'Invalid mean component counts')
        mass = math.fsum(k*v for k, v in enumerate(values, 1))
        if name == 'native_certified':
            require(mass <= n+2e-12, 'Certified component mass exceeds inventory')
        else:
            close(mass, n, 'Exclusion/raw component histogram must conserve all bodies')
        _fingerprint_rate(row['apparent_ess'], samples, cpu)
    joint, seen, count = report['size_pair_occupancies'], set(), 0
    for row in joint:
        key = tuple(row[k] for k in SIZES)+ (row['registry_resolved'],)
        require(key not in seen and all(type(k) is int and 1 <= k <= n for k in key[:3])
                and key[2] <= key[1] and type(key[3]) is bool, 'Invalid/repeated joint size bin')
        seen.add(key)
        require(type(row['observations']) is int and 0 < row['observations'] <= samples, 'Invalid joint size count')
        count += row['observations']
        close(row['fraction'], row['observations']/samples, 'Joint size denominator differs')
    require(count == samples, 'Unresolved joint sizes must remain in the denominator')
    pairs = report['environment_exchanges']['pairs']
    expected = set(itertools.combinations(sorted(set(REGIONS)-{'remaining'}), 2))
    require(len(pairs) == len(expected) and {tuple(p['regions']) for p in pairs} == expected,
            'Incomplete exchange-pair inventory')
    for pair in pairs:
        for key in ('forward', 'reverse', 'completed_nonoverlapping_roundtrips'):
            require(type(pair[key]) is int and pair[key] >= 0, 'Invalid saved passage count')
        close(pair['completed_passages_per_sampler_cpu_second'], (pair['forward']+pair['reverse'])/cpu,
              'Passage CPU denominator differs')
        require(pair['equilibrium_interpretation']['status'] == 'unassessed_equilibrium_weights',
                'Version-one comparison has no independently authenticated finite-system reference')


def compare_reports(reports):
    """Pure comparison; use compare_report_files for authenticated file inputs."""
    require(len(reports) == 36, 'Exactly 36 independent reports required per complete block')
    require(all(r['schema'] == REPORT_SCHEMA and r['complete'] is True for r in reports),
            'Incomplete or incompatible assembly report')
    contract_check = validate_comparison(reports)
    require(contract_check is not None, 'A frozen matched-kernel contract is required')
    contract = reports[0]['benchmark_contract']['content']
    require(contract['preparations'] == list(PREPARATIONS), 'Exact three preparation labels/order required')
    first = reports[0]
    keys = ('physical_identity', 'definition_sha256', 'region_definition_sha256',
            'observer_definition_sha256', 'implementation_sha256')
    for report in reports:
        require(all(report[k] == first[k] for k in keys), 'Target/observer/implementation identity differs')
        require(report.get('physical_reference') is None and report.get('frozen_bias_supported') is False,
                'Biased or referenced reports require a different comparison law')
        seed = report['initialization']['master_seed']
        require(type(seed) is int and 0 <= seed < 2**64, 'Independent uint64 seed required')
        _validate_statistics(report)
    require(len({str(Path(r['run']).resolve()) for r in reports}) == len(reports), 'Repeated run directory')
    groups = {}
    for r in reports:
        key = (r['initialization']['proposal_arm'], r['initialization']['preparation_id'])
        groups.setdefault(key, []).append(r)
    arms = list(contract['arms'])
    for preparation in PREPARATIONS:
        initial = [Counter(r['initialization']['initial_poses_sha256'] for r in groups[arm, preparation]) for arm in arms]
        require(all(v == initial[0] for v in initial), 'Arms do not share matched initial preparations')
    output = []
    for (arm, preparation), members in sorted(groups.items()):
        members = sorted(members, key=lambda r: r['initialization']['master_seed'])
        row = dict(proposal_arm=arm, preparation_id=preparation, populations=len(members),
            seeds=[r['initialization']['master_seed'] for r in members],
            initialization_pose_hashes=[r['initialization']['initial_poses_sha256'] for r in members],
            sampler_cpu_seconds=mean_statistics([r['window']['cpu_seconds'] for r in members]),
            region_occupancies={}, fingerprint_rates={}, component_histograms={}, size_statistics={}, exchanges=[])
        for name in REGIONS:
            values = [r['environment_occupancies'][name] for r in members]
            stats = mean_statistics([v['fraction'] for v in values])
            stats.update(observed_counts=[v['observations'] for v in values],
                all_zero_observations=all(v['observations'] == 0 for v in values),
                first_half_values=[v['first_half_fraction'] for v in values],
                second_half_values=[v['second_half_fraction'] for v in values],
                half_window_changes=[None if v['first_half_fraction'] is None else
                    v['second_half_fraction']-v['first_half_fraction'] for v in values])
            row['region_occupancies'][name] = stats
        for name in FINGERPRINTS:
            values = [r[name]['apparent_ess_per_sampling_CPU_second'] for r in members]
            row['fingerprint_rates'][name] = dict(values=values, null_populations=sum(v is None for v in values),
                population_statistics=None if any(v is None for v in values) else mean_statistics(values))
        for name in SIZES:
            row['size_statistics'][name] = mean_statistics([r['size_statistics'][name]['mean'] for r in members])
        n = first['physical_identity']['bodies']
        for name in COMPONENTS:
            values = [r['component_number_statistics'][name]['mean_component_counts'] for r in members]
            row['component_histograms'][name] = dict(sizes=list(range(1, n+1)),
                mean_count_by_size=[mean_statistics([v[k] for v in values]) for k in range(n)],
                represented_body_fraction=mean_statistics([math.fsum((k+1)*x for k, x in enumerate(v))/n for v in values]))
        joint_keys = sorted({tuple(j[k] for k in SIZES)+(j['registry_resolved'],)
                             for r in members for j in r['size_pair_occupancies']})
        joint_maps = [{tuple(j[k] for k in SIZES)+(j['registry_resolved'],): j['fraction']
                       for j in r['size_pair_occupancies']} for r in members]
        row['joint_size_occupancies'] = [dict(zip(SIZES+('registry_resolved',), key),
            population_statistics=mean_statistics([m.get(key, 0.) for m in joint_maps])) for key in joint_keys]
        for pair in itertools.combinations(sorted(set(REGIONS)-{'remaining'}), 2):
            records = [next(p for p in r['environment_exchanges']['pairs'] if tuple(p['regions']) == pair) for r in members]
            row['exchanges'].append(dict(regions=list(pair),
                per_stream=[dict(seed=r['initialization']['master_seed'], **p) for r, p in zip(members, records)],
                streams_with_both_directions=sum(p['forward'] > 0 and p['reverse'] > 0 for p in records),
                completed_passages_per_cpu=mean_statistics([p['completed_passages_per_sampler_cpu_second'] for p in records]),
                completed_roundtrips=sum(p['completed_nonoverlapping_roundtrips'] for p in records),
                missing_return_interpretation='Unassessed equilibrium weights; no thermodynamic-zero or sampling-failure conclusion.'))
        row['per_stream_episode_histories'] = [dict(seed=r['initialization']['master_seed'],
            sampled_episodes=r['environment_exchanges']['sampled_episodes']) for r in members]
        output.append(row)
    contrasts, speedups = [], []
    for left, right in itertools.combinations(output, 2):
        kind = ('initialization' if left['proposal_arm'] == right['proposal_arm'] else
                'proposal' if left['preparation_id'] == right['preparation_id'] else None)
        if kind is None:
            continue
        ids = dict(kind=kind, left=[left['proposal_arm'], left['preparation_id']],
                   right=[right['proposal_arm'], right['preparation_id']])
        for region in REGIONS:
            contrasts.append(dict(ids, observable='region:'+region,
                **_contrast(left['region_occupancies'][region], right['region_occupancies'][region])))
        for name in SIZES:
            contrasts.append(dict(ids, observable=name,
                **_contrast(left['size_statistics'][name], right['size_statistics'][name])))
        for name in FINGERPRINTS:
            a = left['fingerprint_rates'][name]['population_statistics']
            b = right['fingerprint_rates'][name]['population_statistics']
            resolved = a is not None and b is not None and b['mean'] > 0
            speedups.append(dict(ids, fingerprint=name,
                ratio_of_population_mean_rates=a['mean']/b['mean'] if resolved else None,
                rate_difference=_contrast(a, b) if resolved else None,
                status='apparent_efficiency_only' if resolved else 'unresolved_null_population_efficiency',
                ratio_uncertainty=None, equilibrium_speedup_established=False))
    return dict(schema=SCHEMA, complete=True, physical_identity=first['physical_identity'],
        definition_sha256=first['definition_sha256'], observer_definition_sha256=first['observer_definition_sha256'],
        region_definition_sha256=first['region_definition_sha256'], groups=output, contrasts=contrasts,
        speedup_diagnostics=speedups, benchmark_contract=contract_check,
        raw_populations=[dict(run=r['run'], seed=r['initialization']['master_seed'],
            trajectory_sha256=r['source_sha256']['trajectory.jsonl']) for r in reports],
        unresolved_size_mass_scope='Certified histograms retain missing body mass; no renormalization over certified components.',
        convergence_established=False, scope=SCOPE)


def _cached_accounting(report, observations):
    """Check already-classified records, without native geometry or FFT replay."""
    window = report['window']; chosen, selected = window['indices'], window['sample_indices']
    require(chosen == list(range(chosen[0], chosen[-1]+1)) and selected == chosen[1:]
            and 0 <= chosen[0] < chosen[-1] < len(observations), 'Invalid retained-record window')
    labels = [o['environment'] for o in observations]
    require(labels == report['labels_by_frame'], 'Saved labels/report inventory differs')
    n = report['physical_identity']['bodies']
    for obs in observations:
        require(obs['environment'] == rule_label(obs), 'Cached environment/graph classification differs')
        exclusion_edges = sorted({(i, j) for i, j, _, _ in obs['tokens']})
        native_edges = sorted({(i, j) for i, j, _ in obs['instantaneous_native_keys']})
        for edges in (exclusion_edges, native_edges):
            require(all(type(i) is type(j) is int and 0 <= i < j < n for i, j in edges),
                    'Cached edge has invalid body IDs')
        require(obs['exclusion_edges'] == [list(e) for e in exclusion_edges]
                and obs['native_edges'] == [list(e) for e in native_edges], 'Cached edges and contact tokens differ')
        require(obs['exclusion_components'] == _components(n, exclusion_edges)
                and [c['bodies'] for c in obs['native_components']] == _components(n, native_edges),
                'Cached component inventory differs from saved edges')
        certified = []
        for c in obs['native_components']:
            require(c['certified'] is (c['catalogue_consistent'] is True and c['ordinary_space_lift'] is True),
                    'Cached native certification differs')
            if c['certified']:
                certified.append(len(c['bodies']))
        require(obs['native_registry_resolved'] is all(c['certified'] for c in obs['native_components'])
                and obs['largest_exclusion'] == max(map(len, obs['exclusion_components']))
                and obs['largest_native_raw'] == max(len(c['bodies']) for c in obs['native_components'])
                and obs['largest_native_certified'] == max([1]+certified), 'Cached largest sizes or unresolved status differ')
    sweeps = [observations[i]['sweep'] for i in chosen]
    require(sweeps == list(range(window['burn_sweep'], window['end_sweep']+1, window['cadence_sweeps'])),
            'Cached record cadence/window differs')
    samples = len(selected); split = samples//2
    for name in REGIONS:
        values = [labels[i] == name for i in selected]
        stats = report['environment_occupancies'][name]
        require(stats['observations'] == sum(values), 'Resealed region counts differ from retained records')
        if split:
            close(stats['first_half_fraction'], sum(values[:split])/split, 'First-half count mismatch')
        else:
            require(stats['first_half_fraction'] is None, 'Empty first half must remain null')
        close(stats['second_half_fraction'], sum(values[split:])/(samples-split), 'Second-half count mismatch')
    for name, field in [('fingerprint', 'tokens'), ('native_fingerprint', 'instantaneous_native_keys')]:
        dictionary = sorted({tuple(t) for i in chosen for t in observations[i][field]})
        require(report[name]['observed_token_dictionary'] == [list(t) for t in dictionary], 'Fingerprint token coverage differs')
        counts = Counter(t for i in selected for t in set(map(tuple, observations[i][field])))
        constants = sum(counts[t] in (0, samples) for t in dictionary)
        require(constants == report[name]['constant_observed_token_count'], 'Constant-token accounting differs')
        if constants == len(dictionary):
            require(report[name]['apparent_ess'] is None, 'Constant fingerprint cannot claim independent samples')
    for name in SIZES:
        close(report['size_statistics'][name]['mean'], math.fsum(observations[i][name] for i in selected)/samples,
              'Cached mean size differs')
    for name in COMPONENTS:
        counts = np.zeros(n)
        for i in selected:
            obs = observations[i]
            groups = obs['exclusion_components'] if name == 'exclusion' else [
                c['bodies'] for c in obs['native_components'] if name == 'native_raw' or c['certified']]
            for group in groups:
                require(1 <= len(group) <= n, 'Invalid cached component')
                counts[len(group)-1] += 1
        require(np.allclose(counts/samples, report['component_number_statistics'][name]['mean_component_counts'],
                            rtol=2e-12, atol=2e-12), 'Cached component histogram differs')
    counts = Counter(tuple(observations[i][k] for k in SIZES)+(observations[i]['native_registry_resolved'],) for i in selected)
    recorded = {tuple(j[k] for k in SIZES)+(j['registry_resolved'],): j['observations'] for j in report['size_pair_occupancies']}
    require(dict(counts) == recorded, 'Cached joint-size denominator differs')
    expected = sampled_exchanges([labels[i] for i in chosen], sweeps, set(REGIONS)-{'remaining'})
    actual = report['environment_exchanges']
    require(expected['events'] == actual['events'] and expected['sampled_episodes'] == actual['sampled_episodes'],
            'Cached passages or censored episodes differ')
    actual_pairs = {tuple(p['regions']): p for p in actual['pairs']}
    for pair in expected['pairs']:
        other = actual_pairs[tuple(pair['regions'])]
        require(all(other[k] == v for k, v in pair.items()), 'Cached per-stream roundtrips differ')


def compare_report_files(paths):
    require(len(paths) > 0, 'Report files required')
    reports, bindings = [], {}
    def authenticate(path, expected=None):
        path = Path(path).resolve(); key = str(path)
        if key not in bindings:
            bindings[key] = sha(path)
        require(expected is None or bindings[key] == expected, 'Frozen input changed: '+key)
        return bindings[key]
    implementation = {name: authenticate(Path(__file__).with_name(name)) for name in IMPLEMENTATION}
    for value in paths:
        path = Path(value).resolve()
        require(path.name == 'analysis.json', 'Frozen analysis.json inputs required')
        freeze_path = path.with_name('freeze.json'); authenticate(freeze_path)
        freeze = read(freeze_path)
        require(freeze['schema'] == 'finite-assembly-observation-freeze-v1'
                and set(freeze['files']) == {'analysis.json', 'observations.jsonl'}, 'Incomplete observer freeze')
        for name, expected in freeze['files'].items():
            authenticate(path.with_name(name), expected)
        report = read(path)
        require(freeze['implementation_sha256'] == report['implementation_sha256'], 'Observer implementation freeze differs')
        require(set(RUNTIME_SOURCES+('prepare_finite_assembly_observer.py',)) <= set(report['implementation_sha256']),
                'Incomplete observer source closure')
        for name, expected in report['implementation_sha256'].items():
            require(any(Path(p).name == name and value == expected for p, value in report['dependency_sha256'].items()),
                    'Observer source is not dependency-bound')
        for p, expected in report['dependency_sha256'].items():
            authenticate(p, expected)
        run = Path(report['run']).resolve()
        required = {'config.json', 'manifest.json', 'summary.json', 'checkpoint.json',
            'trajectory.jsonl', 'provenance/input-config.json', 'provenance/shape.json',
            'provenance/source-bundle.json'}
        require(required <= set(report['source_sha256']), 'Incomplete original run source inventory')
        for p, expected in report['source_sha256'].items():
            source = (run/p).resolve()
            require(source.is_relative_to(run), 'Run source path escapes archive')
            authenticate(source, expected)
        config, summary, manifest, checkpoint = [read(run/name) for name in
            ('config.json', 'summary.json', 'manifest.json', 'checkpoint.json')]
        for field, name in [('config_sha256', 'provenance/input-config.json'),
                            ('shape_sha256', 'provenance/shape.json')]:
            require(manifest[field] == summary[field] == checkpoint[field] == report['source_sha256'][name],
                    'Original run provenance differs')
        require(manifest['source_bundle_sha256'] == report['source_sha256']['provenance/source-bundle.json'],
                'Original executable source binding differs')
        require(manifest.get('model_sha256') == summary.get('model_sha256') == checkpoint.get('model_sha256'),
                'Original frozen model identity differs')
        if manifest.get('model_sha256') is not None:
            require(report['source_sha256'].get('provenance/frozen-relative-model.json') == manifest['model_sha256'],
                    'Missing original frozen proposal binding')
        shape = read(run/'provenance/shape.json')
        validate_effective_config(config, read(run/'provenance/input-config.json'), manifest, summary, shape)
        with (run/'trajectory.jsonl').open() as stream:
            frames = [json.loads(line) for line in stream]
        window = validate_frames(frames, config, summary, manifest, checkpoint,
                                 report['window']['burn_sweep'], report['window']['end_sweep'])
        require(window == report['window'], 'Reported window/CPU differs from original retained trajectory')
        require(physical_identity(config, manifest['shape_sha256']) == report['physical_identity'],
                'Reported physical identity differs from original run')
        require(config['seed'] == report['initialization']['master_seed']
                and summary['initial_sweep'] == report['initialization']['invocation_initial_sweep']
                and manifest.get('resume') == report['initialization']['resume']
                and digest(frames[0]['poses']) == report['initialization']['initial_poses_sha256'],
                'Report independent-run/initial-pose identity differs')
        require(report['schedule'] == {k: config[k] for k in MATCHED_SCHEDULE}
                and report['proposal'] == dict(method=config['method'], model_sha256=manifest.get('model_sha256'),
                    frozen_posterior=config.get('frozen_posterior'), learned_uniform_weight=config.get('learned_uniform_weight')),
                'Reported proposal/schedule differs from original run')
        require(report['window_attempts'] == {kind:
            frames[window['indices'][-1]]['counts'][kind]['attempted']
            -frames[window['indices'][0]]['counts'][kind]['attempted'] for kind in ('local', 'global')},
            'Reported attempt budget differs from original retained counters')
        definitions = [Path(p) for p, value in report['dependency_sha256'].items() if value == report['definition_sha256']]
        definitions = [read(p) for p in definitions]
        require(definitions and any(digest(d) == report['observer_definition_sha256']
                and digest(d['regions']) == report['region_definition_sha256']
                and d['physical_identity'] == report['physical_identity'] for d in definitions),
                'Measured observer definition/regions are not bound')
        plans = [read(p) for p, value in report['dependency_sha256'].items() if value == report['plan_sha256']]
        require(plans and any(p['definition_sha256'] == report['definition_sha256']
                and p['preparation_id'] == report['initialization']['preparation_id']
                and p['proposal_arm'] == report['initialization']['proposal_arm']
                and p['burn_sweep'] == window['burn_sweep'] and p['end_sweep'] == window['end_sweep'] for p in plans),
                'Measured preparation/arm/window differs from bound analysis plan')
        observations = [json.loads(line) for line in path.with_name('observations.jsonl').read_text().splitlines()]
        require([o['sweep'] for o in observations] == [f['sweep'] for f in frames],
                'Cached observations omit or reorder original retained frames')
        _cached_accounting(report, observations)
        reports.append(report)
    result = compare_reports(reports)
    for p, expected in bindings.items():
        require(sha(p) == expected, 'Comparison input changed during analysis: '+p)
    result['input_sha256'] = bindings
    result['implementation_sha256'] = implementation
    result['runtime'] = dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__)
    result['geometry_replays'] = result['native_classifier_calls'] = result['fft_replays'] = 0
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reports', nargs='+', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    require(not args.out.exists(), 'Fresh comparison directory required')
    result = compare_report_files(args.reports)
    args.out.mkdir(parents=True)
    (args.out/'source').mkdir()
    for name, expected in result['implementation_sha256'].items():
        shutil.copy2(Path(__file__).with_name(name), args.out/'source'/name)
        require(sha(args.out/'source'/name) == expected, 'Comparator source changed before archival')
    (args.out/'comparison.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    (args.out/'freeze.json').write_text(json.dumps(dict(
        files={p.relative_to(args.out).as_posix(): sha(p) for p in sorted(args.out.rglob('*')) if p.is_file()},
        implementation_sha256=result['implementation_sha256']
        ), indent=2)+'\n')
    print(json.dumps(dict(complete=True, groups=len(result['groups']), convergence_established=False,
                         physical_simulations=0), indent=2))


if __name__ == '__main__':
    main()
