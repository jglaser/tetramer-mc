#!/usr/bin/env python3
"""Independent audit of a separately frozen matched bath comparison.

One independent equilibrium source and defensive candidate supplies three MH
endpoints with a common acceptance uniform. Five contact tests are prespecified:
three source-to-endpoint changes and two bath-minus-analytic contrasts. Every
variable is in {-1,0,1}; exact conditional sign tests use Bonferroni alpha .05.
All rejected pairs and secondary observables remain. No prior flag is replaced.

This imports the previously independent quaternion/density/source audit helpers;
all three analyzer files belong to the frozen closure. Generation is also
reconstructed from saved branch/latent variates. This does not replay the PRNG,
rejected source variates or each Poisson point, nor certify floating arithmetic.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from scipy.stats import chi2, norm
import analyze_dimer_one_step_stationarity as base
from analyze_dimer_tree_equilibrium import quaternion_product

PROTOCOL = 'matched-defensive-bath-bonferroni5-v1'
ARMS = ['analytic', 'poisson', 'singleton_path']
GROUPS = ARMS + ['poisson_minus_analytic', 'singleton_path_minus_analytic']
OBSERVABLES = base.OBSERVABLES
ENDPOINT_KEYS = {'arm', 'state', 'd', 'gate', 'path_gate', 'log_ratio', 'accepted'}


def validate_config(c):
    expected = {'schema': 1, 'analysis_protocol': PROTOCOL, 'draws_per_population': 65536,
                'core_radius': .2, 'exclusion_radius': 1., 'activity': 1.5,
                'auxiliary_intensity': 24., 'root_radius': 2., 'min_separation': .4,
                'max_separation': 2.5, 'rho': .7, 'uniform_probability': .5,
                'uniform_half_width': 2.5,
                'gate': {'max_cells': 255, 'max_depth': 5, 'min_width': 0.}}
    if any(c.get(k) != v for k, v in expected.items()):
        raise ValueError('frozen matched physical/allocation contract changed')
    populations = [{'id': f'matched-r{i:02d}', 'proposal_kind': 'defensive_independent',
                    'arm': 'poisson', 'seed': 610027000+i} for i in range(8)]
    if c.get('populations') != populations or set(c) != set(expected)|{'populations'}:
        raise ValueError('frozen matched population/seed contract changed')


def rotate(q, v):
    return quaternion_product(quaternion_product(q, [0.]+v), [q[0], -q[1], -q[2], -q[3]])[1:]


def compose(a, b):
    delta = rotate(a['orientation'], b['position'])
    return {'position': [x+y for x, y in zip(a['position'], delta)],
            'orientation': quaternion_product(a['orientation'], b['orientation'])}


def pose_error(actual, expected):
    base.validate_state([actual, expected])
    t = max(abs(x-y) for x, y in zip(actual['position'], expected['position']))
    qa, qb = actual['orientation'], expected['orientation']
    q = min(max(abs(x-y) for x, y in zip(qa, qb)), max(abs(x+y) for x, y in zip(qa, qb)))
    if max(t, q) > 3e-12:
        raise ValueError(f'independent generated pose differs: {t}, {q}')
    return t, q


def audit_generation(row, config):
    record = row['defensive_proposal']
    if len(record['edges']) != 2:
        raise ValueError('both independent edge draws must be retained')
    reconstructed, maximum = [], [0., 0.]
    for edge in record['edges']:
        tr = edge['trace']; u = tr['branch_uniform']
        if not 0 <= u < 1:
            raise ValueError('invalid branch uniform')
        branch = 'uniform' if u < config['uniform_probability'] else 'learned'
        if edge['branch'] != branch:
            raise ValueError('branch differs from frozen selection coin')
        if branch == 'uniform':
            raw = tr['translation_uniforms']
            if len(raw) != 3 or not all(0 <= v < 1 for v in raw) or len(tr['quaternion_normals']) != 4:
                raise ValueError('invalid uniform trace')
            pose = {'position': [(2*v-1)*config['uniform_half_width'] for v in raw],
                    'orientation': base.unit(tr['quaternion_normals'])}
        else:
            label = tr['target_label']; z = tr['target_latent']; j = label['branch']
            if (label != {'kind': 'single', 'member': 0, 'anchor': 0, 'branch': j}
                    or type(j) is not int or j not in range(4) or len(z) != 6
                    or not all(math.isfinite(v) for v in z)):
                raise ValueError('invalid identity-atlas label/latent')
            pose = {'position': z[:3], 'orientation': quaternion_product(
                base.unit([1.]+z[3:]), [float(i == j) for i in range(4)])}
        if edge['proposed_relative_pose'] is None or edge['null_reason'] is not None:
            raise ValueError('finite independent reconstruction disagrees with numerical-null trace')
        errors = pose_error(pose, edge['proposed_relative_pose'])
        maximum = [max(a, b) for a, b in zip(maximum, errors)]
        reconstructed.append(pose)
    if row['proposed_state'] is None:
        raise ValueError('two finite generated edges unexpectedly became null')
    for pose, expected in zip([reconstructed[0], compose(*reconstructed)], row['proposed_state']):
        errors = pose_error(pose, expected)
        maximum = [max(a, b) for a, b in zip(maximum, errors)]
    return maximum


def audit_pair(pair, config, spec):
    if set(pair) != {'common', 'outcomes'}:
        raise ValueError('unexpected matched record structure')
    row, endpoints = pair['common'], pair['outcomes']
    if (row['state'] != row['old_state'] or row['d'] != row['old_d'] or row['accepted']
            or any(row[k] is not None for k in ['gate', 'path_gate', 'log_ratio'])):
        raise ValueError('common record must remain the original source')
    if len(endpoints) != 3 or [e['arm'] for e in endpoints] != ARMS:
        raise ValueError('missing, duplicated, reordered or excess matched arm')
    if any(set(endpoint) != ENDPOINT_KEYS for endpoint in endpoints):
        raise ValueError('endpoint cannot override shared proposal/source/uniform')
    max_q = 0.
    for endpoint in endpoints:
        combined = {**row, **endpoint}
        max_q = max(max_q, base.audit_row(combined, config, {**spec, 'arm': endpoint['arm']}))
    generation = audit_generation(row, config)
    return {'max_logq_error': max_q, 'max_generation_translation_error': generation[0],
            'max_generation_quaternion_error': generation[1]}


def source_checks(populations, refs):
    family = len(OBSERVABLES)+2
    critical = float(norm.ppf(1-.05/(2*family)))
    moments = {}
    for key in OBSERVABLES:
        stats = base.merge([p['source'][key] for p in populations]); target = refs['observables'][key]
        moments[key] = {**stats, 'reference': target, 'error_over_se': (stats['mean']-target)/stats['se'],
                        'estimated_family_interval_contains_reference': bool(abs(stats['mean']-target) <= critical*stats['se'])}
    hist = np.sum([p['source_radial_counts'] for p in populations], axis=0)
    expected = np.asarray(refs['radial_bin_probabilities'])*sum(hist)
    statistic = float(np.sum((hist-expected)**2/expected)); p = float(chi2.sf(statistic, len(hist)-1))
    trials = base.merge([p['source_radial_trials'] for p in populations])
    se = math.sqrt(refs['source_radial_trials_variance']/trials['n'])
    error = (trials['mean']-refs['source_radial_trials_mean'])/se
    return {'family_size': family, 'moments': moments,
            'radial_histogram': {'bins': base.RADIAL_BINS, 'counts': hist.tolist(),
                'expected_counts': expected.tolist(), 'pearson_statistic': statistic,
                'asymptotic_p': p, 'bonferroni_adjusted_p': min(1., p*family)},
            'radial_rejection_trials': {**trials, 'reference': refs['source_radial_trials_mean'],
                'known_geometric_se': se, 'error_over_known_se': error,
                'estimated_family_interval_contains_reference': bool(abs(error) <= critical)}}


def analyze(config, directory):
    validate_config(config); refs = base.reference(config)
    populations, hashes = [], {}
    n = config['draws_per_population']; keys = OBSERVABLES
    for spec in config['populations']:
        path = directory/(spec['id']+'.jsonl'); digest = hashlib.sha256()
        xs = np.empty((n, len(keys))); ys = np.empty((3, n, len(keys)))
        trials = np.empty(n); hist = np.zeros(len(base.RADIAL_BINS)-1, dtype=np.int64)
        maxima = {k: 0. for k in ['max_logq_error', 'max_generation_translation_error', 'max_generation_quaternion_error']}
        cpu = 0.; accepted = [0]*3; raw = [0]*3
        with path.open('rb') as stream:
            for step, line in enumerate(stream):
                if step >= n: raise ValueError('excess matched attempts')
                digest.update(line); pair = json.loads(line); row = pair['common']
                if row['step'] != step: raise ValueError('missing/reordered matched attempt')
                audit = audit_pair(pair, config, spec)
                for key, value in audit.items(): maxima[key] = max(maxima[key], value)
                if row['cpu_seconds'] < cpu: raise ValueError('CPU accounting decreased')
                cpu = row['cpu_seconds']
                values = base.observables(row['old_state']); xs[step] = [values[k] for k in keys]
                trials[step] = row['source']['radial_trials']
                index = min(len(hist)-1, int(np.searchsorted(base.RADIAL_BINS, values['d'], side='right')-1))
                hist[index] += 1
                for i, endpoint in enumerate(pair['outcomes']):
                    v = base.observables(endpoint['state']); ys[i, step] = [v[k] for k in keys]
                    accepted[i] += int(endpoint['accepted']); raw[i] += (endpoint['gate'] or {}).get('raw_points', 0)
        if step+1 != n: raise ValueError('incomplete matched population')
        hashes[str(path)] = digest.hexdigest()
        changes = [ys[i]-xs for i in range(3)] + [ys[1]-ys[0], ys[2]-ys[0]]
        groups = {}
        for name, data in zip(GROUPS, changes):
            contact = data[:, 0]
            if not np.isin(contact, [-1., 0., 1.]).all(): raise ValueError('primary contrast outside sign-test support')
            groups[name] = {'moments': {k: base.moments(data[:, i]) for i, k in enumerate(keys)},
                            'positive': int(np.count_nonzero(contact == 1)), 'negative': int(np.count_nonzero(contact == -1))}
        populations.append({**spec, 'n': n, 'cpu_seconds': cpu, **maxima,
            'accepted_by_arm': dict(zip(ARMS, accepted)), 'raw_points_by_arm': dict(zip(ARMS, raw)),
            'source': {k: base.moments(xs[:, i]) for i, k in enumerate(keys)},
            'endpoint': {arm: {k: base.moments(ys[j, :, i]) for i, k in enumerate(keys)} for j, arm in enumerate(ARMS)},
            'changes': groups, 'source_radial_trials': base.moments(trials), 'source_radial_counts': hist.tolist()})
    groups = {}
    for name in GROUPS:
        moments = {k: base.merge([p['changes'][name]['moments'][k] for p in populations]) for k in keys}
        primary = base.primary_contact(moments['contact'], sum(p['changes'][name]['positive'] for p in populations), sum(p['changes'][name]['negative'] for p in populations))
        secondary = {}
        for key, stats in moments.items():
            if key != 'contact':
                secondary[key] = {**stats, 'normal_95_interval': [stats['mean']-1.96*stats['se'], stats['mean']+1.96*stats['se']],
                                  'mean_over_se': stats['mean']/stats['se'] if stats['se'] else None}
        groups[name] = {'primary_contact': primary, 'secondary_paired_changes': secondary,
                       'population_contact_means': [p['changes'][name]['moments']['contact']['mean'] for p in populations]}
    return {'schema': PROTOCOL, 'references': refs, 'observables': keys,
            'primary_family': {'alpha': .05, 'groups': GROUPS, 'exact_test': 'conditional binomial sign on nonzero paired contact changes or matched contrasts'},
            'groups': groups, 'populations': populations, 'source_checks': source_checks(populations, refs),
            'source_sha256': hashes, 'interpretation': 'Separately allocated matched diagnostic; retains previous primary rejection. Same source/proposal/uniform induces correlation between arms; tests use independent pairs and Bonferroni validity does not require independent arms. Does not prove equilibrium or protein assembly.'}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--directory', type=Path, required=True); parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists(): raise FileExistsError(args.out)
    result = analyze(json.loads(args.config.read_text()), args.directory)
    result['config_sha256'] = hashlib.sha256(args.config.read_bytes()).hexdigest()
    with args.out.open('x') as f: json.dump(result, f, indent=2, allow_nan=False); f.write('\n')


if __name__ == '__main__': main()
