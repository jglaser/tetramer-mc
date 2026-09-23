#!/usr/bin/env python3
"""Passive physical occupancy ratios from frozen-bias retained trajectories.

No simulation or training is performed. Self-normalized finite-record ratios
are not unbiased estimators or evidence of equilibrium/coverage.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np

from analyze_contact_efficiency import (
    ContactObserver, MATCHED_SCHEDULE, digest, environment_label, integer,
    load_bound_file, native_input_bindings, physical_identity, population_mean,
    read, recheck_bindings, require, save, sha, summarize_observations,
    validate_effective_config, validate_frames,
)
from mobile_posterior_metrics import apparent_effective_count

SCHEMA = 'frozen-biased-contact-occupancy-v1'
PLAN_SCHEMA = 'frozen-biased-contact-occupancy-plan-v1'
PROTOCOL = 'elementary-reversible-kernel-factorized-bias-v1'
TARGET = 'hard(X) wall(X) exp[-z * exclusion_union_volume(X) - B(largest_exclusion_contact_component(X))]'
REWEIGHTING = ('Normalized physical expectations use exp(log_reweight), where log_reweight = B; '
              'every saved frame including rejected updates is retained.')
SCOPE = ('Physical occupancies are self-normalized finite-record importance ratios, not finite-sample '
         'unbiased estimates. Frozen bias and complete saved frames do not establish equilibrium, '
         'overlap, coverage, independent preparations, native stability or physical kinetics. '
         'Weight concentration ESS ignores serial correlation; apparent autocorrelation ESS is a '
         'separate finite-record diagnostic. Unobserved regions are unresolved, not thermodynamic zeros. '
         'Conditional scaffold probabilities do not certify these all-mobile physical populations.')


def largest_component_size(tokens, bodies):
    """Instantaneous graph only; patch multiplicities and previous labels do not matter."""
    neighbors = [set() for _ in range(bodies)]
    for i, j, *_ in tokens:
        require(type(i) is type(j) is int and 0 <= i < j < bodies, 'Invalid instantaneous contact edge')
        neighbors[i].add(j); neighbors[j].add(i)
    unseen, largest = set(range(bodies)), 0
    while unseen:
        pending = [unseen.pop()]; size = 0
        while pending:
            node = pending.pop(); size += 1
            found = neighbors[node] & unseen
            unseen.difference_update(found); pending.extend(found)
        largest = max(largest, size)
    return largest


def validate_bias_counts(counts, constant=False):
    bias = counts.get('assembly_bias')
    require(isinstance(bias, dict) and set(bias) == {'local', 'global', 'gca', 'center_shift'},
            'Missing or unknown assembly bias count family')
    for kind, gate in bias.items():
        require(isinstance(gate, dict) and set(gate) == {'physical_accepted', 'accepted', 'rejected'},
                'Invalid assembly bias count schema')
        for name, value in gate.items():
            integer(value, 'assembly bias '+kind+' '+name)
        require(gate['physical_accepted'] == gate['accepted']+gate['rejected'],
                'Assembly bias gate dispositions do not sum')
        ordinary = counts[kind]
        if kind in ('local', 'global'):
            require(gate['physical_accepted'] <= ordinary['hard_valid']
                    and gate['accepted'] == ordinary['accepted'], 'Single-body bias counts disagree')
        else:
            # Rust completed counts finished physical collective kernels, including
            # subsequent bias rejection; only transformed_bodies counts retention.
            # The other collective correction modes are excluded by validate_frames.
            require(gate['physical_accepted'] == ordinary['completed'] == ordinary['attempted'],
                    'Collective physical completion and bias counts disagree')
            bodies = len(counts['selected_body_updates_by_body'])
            transformed = integer(ordinary['transformed_bodies'], 'collective transformed bodies')
            require((kind == 'gca' and transformed <= bodies*gate['accepted'])
                    or (kind == 'center_shift' and transformed == bodies*gate['accepted']),
                    'Collective retained transformed-body counts disagree')
        require(not constant or gate['rejected'] == 0, 'Constant assembly bias cannot reject')


def validate_bias(frames, observations, config, manifest, summary, checkpoint):
    """Audit table, all cached instantaneous states, and both acceptance counters."""
    bodies = len(config['initial_poses']); table = config.get('assembly_bias')
    require(isinstance(table, dict) and set(table) == {'values'}, 'Invalid frozen assembly bias schema')
    values = table['values']
    require(isinstance(values, list) and len(values) == bodies
            and all(type(x) in (int, float) and math.isfinite(x) for x in values),
            'Assembly bias needs exactly N finite values')
    require(math.isfinite(max(values)-min(values)), 'Assembly bias differences must be representable')
    for obj in (manifest, summary, checkpoint):
        other = obj.get('assembly_bias')
        require(isinstance(other, dict) and set(other) == {'values'}
                and isinstance(other['values'], list)
                and all(type(x) in (int, float) and math.isfinite(x) for x in other['values'])
                and other == table, 'Frozen assembly bias table differs across outputs')
    require(manifest.get('assembly_bias_protocol') == PROTOCOL
            and manifest.get('physical_target') == TARGET
            and manifest.get('assembly_bias_reweighting') == REWEIGHTING,
            'Unknown assembly bias target, elementary correction protocol or weight sign')
    require(len(frames) == len(observations), 'Missing instantaneous observations')
    constant = min(values) == max(values)
    for frame, obs in zip(frames, observations):
        state = frame.get('assembly_bias')
        require(isinstance(state, dict) and set(state) == {'largest_component_size', 'bias', 'log_reweight'},
                'Invalid assembly bias frame state')
        size = integer(state['largest_component_size'], 'assembly largest component', 1)
        require(size == largest_component_size(obs['tokens'], bodies),
                'Recorded assembly component differs from instantaneous geometry')
        require(all(type(state[k]) in (int, float) and math.isfinite(state[k]) for k in ('bias', 'log_reweight'))
                and state['bias'] == state['log_reweight'] == values[size-1],
                'Recorded assembly bias/log weight differs from frozen +B table')
        validate_bias_counts(frame['counts'], constant)
    for obj in (checkpoint, summary):
        state = obj.get('assembly_bias_state')
        require(isinstance(state, dict) and set(state) == {'largest_component_size', 'bias', 'log_reweight'}
                and type(state['largest_component_size']) is int
                and all(type(state[k]) in (int, float) and math.isfinite(state[k]) for k in ('bias', 'log_reweight'))
                and state == frames[-1]['assembly_bias'], 'Terminal assembly bias checkpoint/summary state differs')
    for counts in (checkpoint['counts'], summary['counts'], summary['initial_counts'], summary['segment_counts']):
        validate_bias_counts(counts, constant)
    return values


def weight_statistics(log_weights, labels, ids):
    """Log-shifted ratio and sufficient moments; never exponentiate an absolute B."""
    logs = np.asarray(log_weights, float)
    require(logs.ndim == 1 and len(logs) and np.isfinite(logs).all()
            and len(labels) == len(logs), 'Invalid retained log weights')
    require(math.isfinite(float(logs.max())-float(logs.min())), 'Log weight differences must be representable')
    require(set(labels) <= set(ids) and len(ids) == len(set(ids)), 'Invalid exhaustive occupancy labels')
    shift = float(logs.max()); weights = np.exp(logs-shift)
    denominator = math.fsum(map(float, weights))
    fractions, numerator_means = {}, {}
    for name in ids:
        numerator = math.fsum(float(w) for w, label in zip(weights, labels) if label == name)
        fractions[name] = numerator/denominator
        numerator_means[name] = numerator/len(weights)
    square_sum = math.fsum(float(w)*float(w) for w in weights)
    return dict(fractions=fractions,
        weight_concentration=dict(samples=len(weights), effective_count=denominator**2/square_sum,
            effective_fraction=denominator**2/square_sum/len(weights), largest_normalized_weight=1/denominator,
            log_weight_span=float(logs.max()-logs.min()), numerically_underflowed_weights=int(np.sum(weights == 0)),
            scope='Kish concentration (sum w)^2/sum w^2 only; ignores time ordering and is not autocorrelation ESS or an equilibrium certificate.'),
        ratio_moments=dict(log_scale=shift, mean_scaled_weight=denominator/len(weights),
                           mean_scaled_weighted_indicators=numerator_means))


def summarize_occupancies(observations, frames, window, definitions):
    ids = [d['id'] for d in definitions]+['remaining']
    labels = [environment_label(obs, definitions) for obs in observations]
    selected = window['sample_indices']; sample_labels = [labels[k] for k in selected]
    logs = [frames[k]['assembly_bias']['log_reweight'] for k in selected]
    stats = weight_statistics(logs, sample_labels, ids)
    log_array = np.asarray(logs); weights = np.exp(log_array-log_array.max())
    split = len(selected)//2
    halves = [weight_statistics(logs[:split], sample_labels[:split], ids) if split else None,
              weight_statistics(logs[split:], sample_labels[split:], ids)]
    occupancies = {}
    for name in ids:
        p = stats['fractions'][name]
        indicator = np.asarray([label == name for label in sample_labels], float)
        residual = weights*(indicator-p)
        scale = float(np.max(np.abs(residual)))
        # Normalize amplitude so gauge/weight scale cannot create a constant-trace diagnostic.
        trace = residual/scale if scale else residual
        occupancies[name] = dict(physical_fraction=p, biased_observations=int(indicator.sum()),
            biased_fraction=float(indicator.mean()), observed_support=bool(indicator.any()),
            first_half_physical_fraction=None if halves[0] is None else halves[0]['fractions'][name],
            second_half_physical_fraction=halves[1]['fractions'][name],
            ratio_residual_autocorrelation=apparent_effective_count(trace, window['cpu_seconds']))
    return dict(window=window, labels_by_frame=labels, physical_occupancies=occupancies,
        weight_concentration=stats['weight_concentration'], ratio_moments=stats['ratio_moments'],
        biased_chain_diagnostics=summarize_observations(observations, frames, window, definitions),
        ratio_residual_scope='Trace w*(I-p_hat) describes ratio fluctuations; its apparent ESS is neither physical occupancy ESS nor a confidence interval.',
        scope=SCOPE)


def validate_definitions(plan):
    definitions = plan['environments']; ids = [d['id'] for d in definitions]
    require(ids and all(isinstance(s, str) and s for s in ids) and len(set(ids)) == len(ids)
            and 'remaining' not in ids, 'Unique environments required; remaining is reserved')
    for definition in definitions:
        require(set(definition) <= {'id', 'required_tokens', 'forbidden_tokens', 'native_consistent_connected'},
                'Unknown environment criterion')
        if 'native_consistent_connected' in definition:
            require(type(definition['native_consistent_connected']) is bool, 'Native criterion must be boolean')
        for key in ('required_tokens', 'forbidden_tokens'):
            for token in definition.get(key, []):
                require(len(token) == 4 and type(token[0]) is type(token[1]) is int
                        and 0 <= token[0] < token[1] and all(isinstance(x, str) and x for x in token[2:]),
                        'Environment tokens require ordered body IDs and two patch IDs')
    for key in ('preparation_id', 'proposal_arm'):
        require(isinstance(plan[key], str) and plan[key], 'Missing declared '+key)
    require(plan.get('physical_reference') is None,
            'This occupancy estimator does not certify external physical/conditional references')
    return definitions


def analyze(run, plan_path, out):
    run, plan_path, out = (Path(p).resolve() for p in (run, plan_path, out))
    require(__debug__, 'Run with PYTHONOPTIMIZE=0 for checked native predicates')
    require(not out.exists(), 'Use a fresh output directory')
    started = time.process_time(); plan = read(plan_path)
    require(plan['schema'] == PLAN_SCHEMA, 'Unsupported frozen bias occupancy plan')
    definitions = validate_definitions(plan)
    names = ('config.json', 'manifest.json', 'summary.json', 'checkpoint.json', 'trajectory.jsonl',
             'provenance/input-config.json', 'provenance/shape.json', 'provenance/source-bundle.json')
    bindings = {name: sha(run/name) for name in names}
    config, manifest, summary, checkpoint = [read(run/name) for name in names[:4]]
    for field, filename in [('config_sha256', 'provenance/input-config.json'), ('shape_sha256', 'provenance/shape.json')]:
        require(manifest[field] == summary[field] == checkpoint[field] == bindings[filename], 'Run provenance mismatch: '+field)
    require(manifest['source_bundle_sha256'] == bindings['provenance/source-bundle.json'], 'Source bundle changed')
    require(manifest.get('model_sha256') == summary.get('model_sha256') == checkpoint.get('model_sha256'), 'Model identity differs')
    if manifest.get('model_sha256') is not None:
        model = run/'provenance/frozen-relative-model.json'
        require(sha(model) == manifest['model_sha256'], 'Frozen proposal changed')
        bindings[str(model.relative_to(run))] = sha(model)
    shape = read(run/'provenance/shape.json')
    validate_effective_config(config, read(run/'provenance/input-config.json'), manifest, summary, shape)
    frames = [json.loads(line) for line in (run/'trajectory.jsonl').read_text().splitlines()]
    window = validate_frames(frames, config, summary, manifest, checkpoint, plan['burn_sweep'], plan['end_sweep'],
                             allow_frozen_assembly_bias=True)
    identity = physical_identity(config, manifest['shape_sha256'])
    require(digest(identity) == plan['physical_identity_sha256'], 'Analysis plan targets different physical measure')
    patch_path = load_bound_file(plan['patch_map'], plan_path.parent); patch = read(patch_path)
    require(patch['schema'] == 'body-frame-atom-patch-map-v1' and patch['shape_sha256'] == manifest['shape_sha256'],
            'Patch/shape identity differs')
    implementation = {name: sha(Path(__file__).with_name(name)) for name in
                      ('analyze_biased_contact_occupancy.py', 'analyze_contact_efficiency.py', 'mobile_posterior_metrics.py')}
    dependencies = {str(plan_path): sha(plan_path), str(patch_path): sha(patch_path)}
    dependencies.update({str(Path(__file__).with_name(name).resolve()): value for name, value in implementation.items()})
    native = None
    if plan.get('native_definition') is not None:
        from native_contact_regions import NativeContactRegions
        native_path = load_bound_file(plan['native_definition'], plan_path.parent)
        dependencies.update(native_input_bindings(native_path)); native = NativeContactRegions(native_path)
        require(native.shape_sha256 == manifest['shape_sha256'], 'Native observer uses another shape')
        for source in ('native_contact_regions.py', 'native_graph_consistency.py'):
            path = Path(__file__).with_name(source).resolve()
            dependencies[str(path)] = implementation[source] = sha(path)
    observer = ContactObserver(shape, patch['atom_patch_ids'], config, native)
    for definition in definitions:
        for name in ('required_tokens', 'forbidden_tokens'):
            for i, j, a, b in definition.get(name, []):
                require(j < window['body_count'] and a in observer.patches and b in observer.patches,
                        'Environment token is outside the fixed body/patch dictionary')
    observations = [observer.classify(frame['poses']) for frame in frames]
    values = validate_bias(frames, observations, config, manifest, summary, checkpoint)
    result = summarize_occupancies(observations, frames, window, definitions)
    observer_sha = digest(dict(environments=definitions, patch_map_sha256=sha(patch_path),
                               native_definition_sha256=None if native is None else native.definition_sha256))
    result.update(schema=SCHEMA, complete=True, run=str(run), plan_sha256=sha(plan_path), physical_identity=identity,
        assembly_bias=dict(values=values), assembly_bias_protocol=PROTOCOL,
        region_definition_sha256=digest(definitions), observer_definition_sha256=observer_sha,
        initialization=dict(master_seed=config['seed'], preparation=config.get('metadata'),
            preparation_id=plan['preparation_id'], proposal_arm=plan['proposal_arm'],
            initial_poses_sha256=digest(frames[0]['poses']), invocation_initial_sweep=summary['initial_sweep'],
            resume=manifest.get('resume'), initial_environment=result['labels_by_frame'][0]),
        measurement_identity=dict(patch_map_sha256=sha(patch_path),
            native_definition_sha256=None if native is None else native.definition_sha256,
            cadence_sweeps=window['cadence_sweeps']), schedule={key: config[key] for key in MATCHED_SCHEDULE},
        proposal=dict(method=config['method'], model_sha256=manifest.get('model_sha256'),
            frozen_posterior=config.get('frozen_posterior'), learned_uniform_weight=config.get('learned_uniform_weight')),
        source_sha256=bindings, dependency_sha256=dependencies, implementation_sha256=implementation,
        observer_cpu_seconds=time.process_time()-started,
        validation_scope='Saved-state geometry, counters, target declaration and provenance checked. Elementary physical gate/reversibility correctness remains the runner contract, not a trajectory-based proof.')
    recheck_bindings({str(run/name): expected for name, expected in bindings.items()}, 'Input changed during observation')
    recheck_bindings(dependencies, 'Observer dependency changed during observation')
    out.mkdir(parents=True); save(out/'analysis.json', result)
    with (out/'observations.jsonl').open('w') as stream:
        for frame, obs, label in zip(frames, observations, result['labels_by_frame']):
            stream.write(json.dumps(dict(sweep=frame['sweep'], sampler_cpu_seconds=frame['sampler_cpu_seconds'],
                environment=label, assembly_bias=frame['assembly_bias'], **obs), allow_nan=False)+'\n')
    save(out/'freeze.json', dict(files={name: sha(out/name) for name in ('analysis.json', 'observations.jsonl')},
                               implementation_sha256=implementation))
    return result


def independent_ratio_diagnostic(moments, region):
    """Delta-method covariance across independent, equal-length stream means."""
    require(len(moments) > 0, 'No independent stream moments')
    shift = max(m['log_scale'] for m in moments)
    scale = [math.exp(m['log_scale']-shift) for m in moments]
    denominator = np.asarray([s*m['mean_scaled_weight'] for s, m in zip(scale, moments)])
    numerator = np.asarray([s*m['mean_scaled_weighted_indicators'][region] for s, m in zip(scale, moments)])
    require(np.isfinite(numerator).all() and np.isfinite(denominator).all() and np.all(denominator > 0)
            and np.all(numerator >= 0) and np.all(numerator <= denominator), 'Invalid independent ratio moments or numerically lost stream')
    ratio = float(numerator.mean()/denominator.mean()); n = len(moments)
    covariance = None; se = None
    if n > 1:
        cov = np.cov(np.stack([numerator, denominator]), ddof=1)/n
        # Residual form is stable and exactly includes numerator/denominator covariance.
        se = float(np.std(numerator-ratio*denominator, ddof=1)/math.sqrt(n)/denominator.mean())
        covariance = dict(numerator_mean_variance=float(cov[0, 0]), denominator_mean_variance=float(cov[1, 1]),
                          numerator_denominator_mean_covariance=float(cov[0, 1]), common_log_scale=shift)
    return dict(ratio_of_stream_moments=ratio, delta_method_standard_error=se,
        zero_observed_variance=se == 0 if se is not None else None, scaled_covariance=covariance,
        scope='Independent stream means with a shared frozen bias; asymptotic delta-method diagnostic only. Few streams, common trapping and unseen support invalidate confidence interpretations; zero observed variance is not certainty.')


def compare_reports(reports):
    require(len(reports) >= 2, 'At least two independent reports required')
    first = reports[0]
    for report in reports:
        require(report['schema'] == SCHEMA and report['complete'] is True, 'Incomplete biased occupancy report')
        for key in ('physical_identity', 'assembly_bias', 'assembly_bias_protocol', 'region_definition_sha256',
                    'observer_definition_sha256', 'measurement_identity', 'schedule', 'implementation_sha256'):
            require(report[key] == first[key], 'Independent comparison must match '+key)
        require(tuple(report['window'][k] for k in ('burn_sweep', 'end_sweep', 'cadence_sweeps')) ==
                tuple(first['window'][k] for k in ('burn_sweep', 'end_sweep', 'cadence_sweeps')),
                'Equal regular stream windows required')
        require(set(report['physical_occupancies']) == set(first['physical_occupancies']), 'Physical partitions differ')
    seeds = [r['initialization']['master_seed'] for r in reports]
    require(len(seeds) == len(set(seeds)), 'Independent-stream comparison rejects reused/paired seeds or continued segments')
    groups, arms = {}, {}
    for report in reports:
        key = (report['initialization']['proposal_arm'], report['initialization']['preparation_id'])
        arms.setdefault(key[0], report['proposal'])
        require(arms[key[0]] == report['proposal'], 'A proposal arm requires one frozen proposal')
        groups.setdefault(key, []).append(report)
    output = []
    for (arm, preparation), members in sorted(groups.items()):
        output.append(dict(proposal_arm=arm, preparation_id=preparation, populations=len(members),
            physical_occupancies={name: dict(
                equal_stream_ratio_mean=population_mean([r['physical_occupancies'][name]['physical_fraction'] for r in members]),
                ratio_covariance_diagnostic=independent_ratio_diagnostic([r['ratio_moments'] for r in members], name))
                for name in first['physical_occupancies']}))
    return dict(schema='frozen-biased-contact-independent-stream-comparison-v1', groups=output,
                physical_identity=first['physical_identity'], assembly_bias=first['assembly_bias'],
                scope=SCOPE+' Streams stay separate; distinct seeds are a necessary bookkeeping check, not proof of independent preparation. No chains are concatenated.')


def compare_report_files(paths):
    reports, bindings = [], {}
    for value in paths:
        path = Path(value).resolve(); require(path.name == 'analysis.json', 'Frozen analysis.json required')
        freeze_path = path.with_name('freeze.json'); bindings[str(freeze_path)] = sha(freeze_path)
        freeze = read(freeze_path)
        require(set(freeze['files']) == {'analysis.json', 'observations.jsonl'}, 'Incomplete report freeze')
        for name, expected in freeze['files'].items():
            item = path.with_name(name); require(sha(item) == expected, 'Frozen occupancy report changed: '+str(item))
            bindings[str(item)] = expected
        report = read(path); implementation = report['implementation_sha256']
        require(freeze['implementation_sha256'] == implementation
                and {'analyze_biased_contact_occupancy.py', 'analyze_contact_efficiency.py', 'mobile_posterior_metrics.py'} <= set(implementation),
                'Report implementation freeze differs or lacks both observer modules')
        for name, expected in implementation.items():
            require(any(Path(source).name == name and value == expected for source, value in report['dependency_sha256'].items()),
                    'Report implementation absent from dependency bindings: '+name)
        reports.append(report)
    result = compare_reports(reports); recheck_bindings(bindings, 'Comparison input changed')
    result['source_sha256'] = bindings
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path); parser.add_argument('--plan', type=Path)
    parser.add_argument('--compare', nargs='+', type=Path); parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    if args.compare:
        require(args.run is None and args.plan is None, 'Compare does not accept a run/plan')
        require(not args.out.exists(), 'Use a fresh comparison output')
        result = compare_report_files(args.compare); args.out.mkdir(parents=True)
        save(args.out/'comparison.json', result)
    else:
        require(args.run is not None and args.plan is not None, 'Run and frozen plan required')
        analyze(args.run, args.plan, args.out)
    print(json.dumps(dict(complete=True, out=str(args.out.resolve()), physical_simulations=0)))


if __name__ == '__main__':
    main()
