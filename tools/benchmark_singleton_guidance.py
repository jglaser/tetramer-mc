#!/usr/bin/env python3
"""Freeze, run and audit matched singleton contact-guidance reset diagnostics.

Every reset phase starts from the same archived physical state. These are
conditional transition diagnostics, not assembly trajectories or ESS estimates.
No running production directory is written and no failed allocation is retried.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SNAPSHOT = ROOT / 'runs/growth-merged-analysis-20261001/snapshot'
SCHEMA = 'singleton-guidance-reset-benchmark-v1'
SCOPE = ('Conditional transition probabilities from a fixed physical configuration. '
         'Reset phases and streams are independent; events within a phase are not. '
         'This does not measure equilibrium, assembly, contact ESS or mixing speedup.')


def require(test, message):
    if not test:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x' if exclusive else 'w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def copy_checked(source, destination):
    source, destination = Path(source), Path(destination)
    digest = sha(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    require(sha(destination) == digest == sha(source), 'Input changed while copying: ' + str(source))


def configured(source, checkpoint, provenance, arm, seed, singleton_rate, duration, epsilon,
               pool_selection='nearest', independent_max_trials=None, local_only=False):
    config = copy.deepcopy(source)
    config['initial_poses'] = copy.deepcopy(checkpoint['poses'])
    config['seed'] = seed
    config['shape'] = str(provenance / 'shape.json')
    if config.get('monomer_shape'):
        config['monomer_shape'] = str(provenance / 'monomer-shape.json')
    config['cluster_phase'] = dict(
        duration=duration, singleton_rate=singleton_rate, dimer_rate=0., trimer_rate=0.,
        transport_probability=0. if local_only else 1., correlation=.9,
        local_translation_std_A=source['local_translation_std_A'],
        local_small_angle_std_degrees=source['local_small_angle_std_degrees'],
        transport_charts='members', anchor_count=2,
        anchor_contact_uniform_probability=1. if arm == 'uniform-primary' else epsilon,
    )
    if arm == 'fused':
        config['cluster_phase']['oligomer'] = {}
    if pool_selection != 'nearest':
        config['cluster_phase']['anchor_pool_selection'] = pool_selection
    if independent_max_trials is not None:
        require(isinstance(independent_max_trials, int) and independent_max_trials > 0,
                'Independent trial cap must be a positive integer')
        config['cluster_phase']['singleton_independent_max_trials'] = independent_max_trials
    # Metadata and supplied native seed labels are retained for provenance and
    # passive observations. The benchmark's proposal never inspects them.
    config.setdefault('metadata', {})['singleton_guidance_diagnostic'] = dict(
        arm=arm, source_completed_sweeps=checkpoint['completed_sweeps'],
        independent_reset_phases=True, proposal_uses_native_labels=False)
    return config


def freeze(out, snapshot, binary, bundle, populations=4, phases=128, workers=2,
           singleton_rate=1., duration=.01, epsilon=.1, seed=2026100100,
           uniform_primary=False, native_definition=None, pool_selection='nearest',
           independent_max_trials=None, local_only=False):
    out, snapshot, binary, bundle = map(lambda p: Path(p).resolve(), (out, snapshot, binary, bundle))
    require(not out.exists(), 'Fresh output directory required')
    require(populations > 0 and phases > 0 and 1 <= workers <= 2, 'Invalid allocation; maximum two physical children')
    require(singleton_rate > 0 and duration > 0 and 0 < epsilon <= 1, 'Invalid phase parameters')
    require(pool_selection in ('nearest', 'contact_without_replacement'), 'Invalid anchor pool law')
    require(not local_only or (independent_max_trials is None and not uniform_primary),
            'Local control cannot enable independent draws or a primary-selection arm')
    source, checkpoint = read(snapshot/'config.json'), read(snapshot/'checkpoint.json')
    require(not source.get('fixed_body_indices') and not source.get('assembly_bias'), 'Require all-mobile unbiased target')
    require(source['boundary']['kind'] == 'spherical', 'Require spherical frozen state')
    require(all(checkpoint.get(k) is None for k in ('auxiliary_eta', 'rj_state', 'contact_memory_state',
            'conditional_state', 'atlas_state', 'atlas_mask_state')), 'Physical reset cannot discard auxiliary state')
    raw_bundle = bundle.read_bytes()
    compiled = json.loads(raw_bundle)
    require(os.access(binary, os.X_OK) and raw_bundle in binary.read_bytes(), 'Source bundle must be embedded in executable')
    require('singleton_rate' in compiled['files']['src/cluster_phase.rs']['text'], 'Binary lacks singleton channels')
    if pool_selection != 'nearest':
        require('ContactWithoutReplacement' in compiled['files']['src/cluster_phase.rs']['text'],
                'Binary lacks retained contact-pool selection')
    if independent_max_trials is not None:
        require(isinstance(independent_max_trials, int) and independent_max_trials > 0,
                'Independent trial cap must be a positive integer')
        require('singleton_independent_max_trials' in compiled['files']['src/cluster_phase.rs']['text'],
                'Binary lacks independent hard-conditioned redraw')
    for name, entry in compiled['files'].items():
        require(hashlib.sha256(entry['text'].encode()).hexdigest() == entry['sha256'], 'Invalid bundled source: '+name)
    provenance = out/'provenance'
    sources = {'binary': binary, 'source-bundle.json': bundle, 'source-config.json': snapshot/'config.json',
        'checkpoint.json': snapshot/'checkpoint.json', 'model.json': snapshot/'provenance/frozen-relative-model.json',
        'shape.json': snapshot/'provenance/shape.json', 'benchmark_singleton_guidance.py': Path(__file__)}
    if source.get('monomer_shape'):
        path = Path(source['monomer_shape'])
        sources['monomer-shape.json'] = path if path.is_absolute() else snapshot/path
    require(sha(sources['model.json']) == checkpoint['model_sha256'], 'Checkpoint/model mismatch')
    require(sha(sources['shape.json']) == checkpoint['shape_sha256'], 'Checkpoint/shape mismatch')
    for name, path in sources.items():
        copy_checked(path, provenance/name)
    if native_definition:
        native_definition = Path(native_definition).resolve()
        definition = read(native_definition)
        require(definition['shape_sha256'] == checkpoint['shape_sha256'], 'Native observer shape mismatch')
        copy_checked(native_definition, provenance/'native/definition.json')
        for name, digest in definition['input_sha256'].items():
            rel = Path(name)
            require(not rel.is_absolute() and '..' not in rel.parts, 'Unsafe observer path')
            path = native_definition.parent/'inputs'/rel
            require(sha(path) == digest, 'Native observer input changed')
            copy_checked(path, provenance/'native/inputs'/rel)
        copy_checked(ROOT/'tools/native_contact_regions.py', provenance/'native_contact_regions.py')
    arms = ['local'] if local_only else ['unfused', 'fused'] + (['uniform-primary'] if uniform_primary else [])
    jobs = []
    for population in range(populations):
        for arm_index, arm in enumerate(arms):
            # Unique streams: matching refers to allocations and physical setup,
            # not to common random numbers consumed differently by the arms.
            job_seed = seed + population * len(arms) + arm_index
            name = f'{arm}-p{population}'
            config = configured(source, checkpoint, provenance, arm, job_seed, singleton_rate, duration,
                                epsilon, pool_selection, independent_max_trials, local_only)
            config_path = out/'configs'/f'{name}.json'
            write(config_path, config)
            output = out/'runs'/name
            command = [str(provenance/'binary'), '--config', str(config_path), '--model', str(provenance/'model.json'),
                '--phases', str(phases), '--seed', str(job_seed), '--out', str(output)]
            jobs.append(dict(id=name, arm=arm, population=population, seed=job_seed, config=str(config_path),
                directory=str(output), log=str(out/'logs'/f'{name}.log'), command=command))
    for folder in ('runs', 'logs'):
        (out/folder).mkdir()
    protocol = dict(schema=SCHEMA, scope=SCOPE, snapshot=str(snapshot), checkpoint_sweep=checkpoint['completed_sweeps'],
        body_count=len(checkpoint['poses']), arms=arms, populations=populations, phases=phases, workers=workers,
        singleton_rate=singleton_rate, duration=duration, correlation=.9, primary_uniform_probability=epsilon,
        anchor_pool_selection=pool_selection,
        independent_max_trials=independent_max_trials,
        learned_kernel='local' if local_only else ('independent_conditioned' if independent_max_trials is not None else 'involution'),
        local_translation_std_A=source['local_translation_std_A'],
        local_small_angle_std_degrees=source['local_small_angle_std_degrees'],
        expected_events_per_population=len(checkpoint['poses'])*singleton_rate*duration*phases,
        depletant_radius=source['depletant_radius'], depletant_activity=source['reservoir_density'],
        model_sha256=checkpoint['model_sha256'], shape_sha256=checkpoint['shape_sha256'],
        native_observer=bool(native_definition), jobs=jobs,
        allocation='Frozen before sampling; failed jobs are not replaced; all started children drain before exit.',
        comparison=('Local singleton baseline with unchanged translation/rotation scales.' if local_only else
                    'Fused versus unfused two-anchor singleton maps. Optional uniform-primary arm only tests primary selection.'),
        physical_jobs_launched=0)
    write(out/'protocol.json', protocol)
    write(out/'freeze.json', {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob('*')) if p.is_file()})
    validate(out)
    return protocol


def validate(out):
    out = Path(out).resolve()
    for name, digest in read(out/'freeze.json').items():
        require(sha(out/name) == digest, 'Frozen input changed: '+name)
    protocol = read(out/'protocol.json')
    require(protocol['schema'] == SCHEMA and 1 <= protocol['workers'] <= 2, 'Invalid frozen protocol')
    require(sha(__file__) == sha(out/'provenance/benchmark_singleton_guidance.py'), 'Use frozen controller version')
    return protocol


def run(out):
    out = Path(out).resolve()
    protocol = validate(out)
    require(not any((out/'runs').iterdir()) and not any((out/'logs').iterdir()), 'Existing output; no retry or overwrite')
    state = dict(complete=False, running=True, jobs=[dict(id=j['id'], status='pending') for j in protocol['jobs']])
    write(out/'status.json', state, exclusive=True)
    active, next_index, failure = {}, 0, None

    def finish(index, child):
        code = child.wait()
        state['jobs'][index].update(status='complete' if code == 0 else 'failed', exit_code=code, finished=time.time())
        return code

    try:
        while active or (next_index < len(protocol['jobs']) and failure is None):
            while failure is None and next_index < len(protocol['jobs']) and len(active) < protocol['workers']:
                validate(out)
                index = next_index
                next_index += 1
                job = protocol['jobs'][index]
                state['jobs'][index].update(status='launching', started=time.time())
                write(out/'status.json', state)
                try:
                    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', RAYON_NUM_THREADS='1')
                    with Path(job['log']).open('xb') as log:
                        active[index] = subprocess.Popen(job['command'], stdout=log, stderr=subprocess.STDOUT, env=env)
                except BaseException as error:
                    state['jobs'][index].update(status='launch_failed', exception=repr(error))
                    raise
                state['jobs'][index].update(status='running', pid=active[index].pid)
            for index, child in list(active.items()):
                if child.poll() is not None:
                    code = finish(index, child)
                    del active[index]
                    if code != 0:
                        failure = RuntimeError('Physical benchmark failed; preserving all attempts and draining started children')
            write(out/'status.json', state)
            if active:
                time.sleep(.1)
    except BaseException as error:
        failure = error
    finally:
        for index, child in active.items():
            finish(index, child)
        for job in state['jobs']:
            if job['status'] == 'pending':
                job['status'] = 'not_started'
        state.update(running=False, complete=failure is None, finished=time.time())
        if failure is not None:
            state['exception'] = repr(failure)
        write(out/'status.json', state)
    if failure is not None:
        raise failure
    return state


def quantiles(values):
    values = sorted(values)
    if not values:
        return dict(count=0)
    def at(f):
        t = f*(len(values)-1)
        i = int(t)
        return values[i] + (t-i)*(values[min(i+1, len(values)-1)]-values[i])
    return dict(count=len(values), minimum=values[0], q10=at(.1), median=at(.5), q90=at(.9),
        maximum=values[-1], mean=math.fsum(values)/len(values))


def event_statistics(rows):
    counts, metrics = Counter(), defaultdict(list)
    for row in rows:
        if row['kind'] != 'cluster_event':
            continue
        require(len(row['members']) == 1, 'Non-singleton event in singleton experiment')
        p = row['proposal']
        accepted, valid = row['accepted'], row['hard_valid']
        if p['branch'] == 'independent_conditioned':
            raw = p.get('raw_trials')
            require(isinstance(raw, list) and len(raw) == p['raw_trial_count'],
                    'Independent raw attempts missing')
            require(all(not r['hard_valid'] for r in raw[:-1]),
                    'Independent kernel retried after a hard-valid candidate')
            if p['cap_exhausted']:
                require(len(raw) == p['max_trials'] and all(not r['hard_valid'] for r in raw),
                        'Invalid exhausted-cap record')
                require(row['proposed_poses'] is None and not accepted,
                        'Exhausted cap produced a physical candidate')
            if row['proposed_poses'] is not None:
                require(bool(raw) and raw[-1]['hard_valid'] and valid and
                        row['proposed_poses'] == [raw[-1]['candidate']],
                        'Independent endpoint differs from first hard-valid raw pose')
            counts['independent_raw_hard_valid'] += sum(r['hard_valid'] for r in raw)
            counts['independent_valid_candidates'] += int(valid)
        gained, lost = len(row['proposed_gained_contacts']), len(row['proposed_lost_contacts'])
        counts.update(attempted=1, hard_valid=int(valid), accepted=int(accepted),
            accepted_exchanges=int(accepted and gained > 0 and lost > 0),
            accepted_contact_changing=int(accepted and gained+lost > 0),
            accepted_attachments=int(accepted and gained > 0 and lost == 0),
            accepted_detachments=int(accepted and lost > 0 and gained == 0),
            proposal_nulls=int(row['proposed_poses'] is None),
            learned=int(p['branch'] in ('involution', 'independent_conditioned')),
            uniform=int(p['branch'] == 'uniform'),
            independent_raw_trials=p.get('raw_trial_count', 0),
            independent_cap_exhaustions=int(p.get('cap_exhausted', False)))
        d = row['diagnostic']
        counts['embedded'] += d['topology'] == 'embedded'
        counts['pool_recorded'] += d['pool_contains_direct_partner'] is not None
        counts['pool_contains_direct_partner'] += d['pool_contains_direct_partner'] is True
        if p.get('oligomer'):
            o = p['oligomer']
            counts['fused_catalogue_available'] += o['fused_components'] > 0
            counts['source_fused'] += o.get('source_label', {}).get('kind') == 'fused'
            counts['target_fused'] += o.get('target_label', p.get('target_label', {})).get('kind') == 'fused'
            metrics['catalogue_build_seconds'].append(sum(o['build_seconds']))
            metrics['fused_components'].append(o['fused_components'])
        numbers = {}
        for key in ('full_old_log_density', 'full_new_log_density', 'full_old_member_log_density',
                    'full_new_member_log_density', 'map_log_reverse_forward', 'label_log_reverse_forward',
                    'anchor_log_reverse_forward', 'log_reverse_forward',
                    'pool_forward_log_probability', 'pool_reverse_log_probability'):
            if isinstance(p.get(key), (int, float)):
                numbers[key] = p[key]
        for output, alternatives in (
            ('source_gaussian_log_density', ('full_old_log_density', 'full_old_member_log_density')),
            ('destination_gaussian_log_density', ('full_new_log_density', 'full_new_member_log_density')),
        ):
            for key in alternatives:
                if isinstance(p.get(key), (int, float)):
                    numbers[output] = p[key]
                    break
        step = p.get('step', {})
        if step.get('source_latent') is not None:
            numbers['source_latent_norm'] = math.sqrt(sum(x*x for x in step['source_latent']))
        if step.get('log_auxiliary_ratio') is not None:
            numbers['log_auxiliary_ratio'] = step['log_auxiliary_ratio']
        if row.get('log_acceptance') is not None:
            numbers['conditional_acceptance_probability'] = math.exp(row['log_acceptance'])
        for key, value in numbers.items():
            require(math.isfinite(value), 'Nonfinite diagnostic statistic')
            for suffix, keep in [('all', True), ('valid', valid), ('accepted', accepted)]:
                if keep:
                    metrics[f'{key}_{suffix}'].append(value)
    n = counts['attempted']
    return dict(counts=dict(counts), fractions={k:counts[k]/n if n else None for k in
        ('hard_valid', 'accepted', 'accepted_contact_changing', 'accepted_exchanges')},
        metrics={key:quantiles(value) for key, value in metrics.items()})


def native_changes(rows, initial, classifier):
    """Observe accepted single-body moves only; labels never guide the kernel."""
    state, phase, events = None, None, []
    for row in rows:
        if row['kind'] != 'cluster_event':
            continue
        if row['phase'] != phase:
            state, phase = copy.deepcopy(initial), row['phase']
        i = row['members'][0]
        require(state[i] == row['old_poses'][0], 'Native observer replay mismatch')
        if row['accepted']:
            def keys(pose):
                result = set()
                for j, other in enumerate(state):
                    if j == i:
                        continue
                    left, right = (pose, other) if i < j else (other, pose)
                    result.update((min(i,j), max(i,j), m['motif_id']) for m in classifier.classify_pair(left, right))
                return result
            before, after = keys(state[i]), keys(row['retained_poses'][0])
            if before != after:
                events.append(dict(phase=phase, event_time=row['event_time'], member=i,
                    gained=sorted(after-before), lost=sorted(before-after)))
        state[i] = row['retained_poses'][0]
    return dict(scope='Native registry is observed only after acceptance; no native proposal filtering.',
        events=events, gained=sum(len(e['gained']) for e in events), lost=sum(len(e['lost']) for e in events))


def analyze(out):
    out = Path(out).resolve()
    protocol = validate(out)
    status = read(out/'status.json')
    require(status['complete'] and not status['running'], 'All physical jobs must finish successfully before combined analysis')
    classifier = None
    if protocol['native_observer']:
        from native_contact_regions import NativeContactRegions
        classifier = NativeContactRegions(out/'provenance/native/definition.json')
    populations, by_arm = [], defaultdict(list)
    input_hashes = {}
    for job in protocol['jobs']:
        directory = Path(job['directory'])
        summary, manifest = read(directory/'summary.json'), read(directory/'manifest.json')
        require(summary['complete'] and summary['all_attempted_events_retained'] and summary['all_phase_endpoints_replayed'], 'Incomplete job audit')
        require(summary['phases'] == protocol['phases'] and summary['population_seed'] == job['seed'], 'Job allocation changed')
        require(manifest['executable_sha256'] == sha(out/'provenance/binary'), 'Executed binary mismatch')
        require(manifest['config_sha256'] == sha(job['config']) and manifest['model_sha256'] == protocol['model_sha256'], 'Executed inputs mismatch')
        for name in ('events.jsonl', 'phases.jsonl'):
            digest = sha(directory/name)
            require(digest == summary[name.split('.')[0]+'_sha256'], 'Event/phase output changed')
            input_hashes[str(directory/name)] = digest
        rows = [json.loads(line) for line in (directory/'events.jsonl').read_text().splitlines()]
        stats = event_statistics(rows)
        require(stats['counts']['attempted'] == summary['counts']['events'], 'Attempt denominator mismatch')
        for name in ('independent_raw_trials', 'independent_cap_exhaustions',
                     'independent_valid_candidates'):
            require(stats['counts'].get(name, 0) == summary['counts'].get(name, 0),
                    'Independent attempt counter mismatch: '+name)
        result = dict(id=job['id'], arm=job['arm'], population=job['population'], seed=job['seed'], **stats,
            kernel_cpu_seconds=summary['kernel_cpu_seconds'], total_cpu_seconds=summary['total_cpu_seconds'])
        grouped = defaultdict(list)
        for row in rows:
            if row['kind'] == 'cluster_event':
                branch, topology = row['proposal']['branch'], row['diagnostic']['topology']
                grouped['branch='+branch].append(row)
                grouped['topology='+topology].append(row)
                grouped[f'branch={branch}/topology={topology}'].append(row)
        result['strata'] = {key:event_statistics(value) for key, value in grouped.items()}
        result['contact_changes_per_kernel_cpu_second'] = stats['counts']['accepted_contact_changing']/summary['kernel_cpu_seconds']
        if classifier:
            result['native'] = native_changes(rows, read(job['config'])['initial_poses'], classifier)
        populations.append(result)
        by_arm[job['arm']].append(result)
    arms = {}
    for arm, records in by_arm.items():
        counts = Counter()
        for record in records:
            counts.update(record['counts'])
        cpu = sum(r['kernel_cpu_seconds'] for r in records)
        arms[arm] = dict(counts=dict(counts), kernel_cpu_seconds=cpu,
            fractions={key:counts[key]/counts['attempted'] if counts['attempted'] else None for key in
                ('hard_valid', 'accepted', 'accepted_contact_changing', 'accepted_exchanges')},
            contact_changes_per_kernel_cpu_second=counts['accepted_contact_changing']/cpu,
            population_contact_change_fractions=[r['fractions']['accepted_contact_changing'] for r in records])
    analysis = dict(complete=True, scope=SCOPE, protocol_sha256=sha(out/'protocol.json'), populations=populations,
        arms=arms, physical_output_sha256=input_hashes,
        uncertainty='Inspect independent population results; no binomial independence assumption for within-phase events.')
    write(out/'analysis.json', analysis)
    return analysis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    p = commands.add_parser('freeze')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--snapshot', type=Path, default=DEFAULT_SNAPSHOT)
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--populations', type=int, default=4)
    p.add_argument('--phases', type=int, default=128)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--singleton-rate', type=float, default=1.)
    p.add_argument('--duration', type=float, default=.01)
    p.add_argument('--epsilon', type=float, default=.1)
    p.add_argument('--seed', type=int, default=2026100100)
    p.add_argument('--uniform-primary', action='store_true')
    p.add_argument('--pool-selection', choices=('nearest', 'contact_without_replacement'), default='nearest')
    p.add_argument('--independent-max-trials', type=int,
                   help='Use direct independent learned draws, stopping at the first hard-valid pose or this cap')
    p.add_argument('--local-only', action='store_true',
                   help='One local singleton arm, with the source configuration translation/rotation scales')
    p.add_argument('--native-definition', type=Path)
    for name in ('validate', 'run', 'analyze'):
        p = commands.add_parser(name)
        p.add_argument('--out', type=Path, required=True)
    args = vars(parser.parse_args())
    action = args.pop('action')
    result = globals()[action](**args)
    print(json.dumps(dict(action=action, complete=result.get('complete'), jobs=len(result.get('jobs', []))), sort_keys=True))


if __name__ == '__main__':
    main()
