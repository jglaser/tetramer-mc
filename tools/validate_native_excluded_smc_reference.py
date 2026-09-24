#!/usr/bin/env python3
"""Fresh, fixed-budget sphere fixtures for the independent restricted-SMC auditor.

These fixtures exercise the real compiled native predicate. They are not a rerun
of old reference populations and are never protein or assembly evidence.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

THREAD_ENV = {name: '1' for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
    'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'RAYON_NUM_THREADS')}
os.environ.update(THREAD_ENV)

from prepare_shoulder_docking_benchmark import local_dependencies
from run_full_vessel_comparison import execute_group, process_token

SCHEMA = 'native-excluded-smc-reference-v1'
SEEDS = tuple(148301010 + 1009 * i for i in range(17))
EXACT = {0.: dict(total=.004462706426614964, contact=.001569924700053489),
         4.: dict(total=.005747109017931148, contact=.002854327291369672)}
for item in EXACT.values():
    item['unbound'] = item['total'] - item['contact']


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def seed_values(value):
    """Find explicit seeds in old plans, including nested independent allocations."""
    found = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key == 'seed' and type(item) is int:
                found.add(item)
            elif key == 'seeds' and isinstance(item, list):
                found.update(s for s in item if type(s) is int)
            found.update(seed_values(item))
    elif isinstance(value, list):
        for item in value:
            found.update(seed_values(item))
    return found


def fixture(root, activity, core=.3):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    fixed = dict(position=[0., 0., 0.], orientation=[1., 0., 0., 0.])
    native = dict(position=[1., 0., 0.], orientation=[1., 0., 0., 0.])
    identity = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
    shape = dict(name='Independent native-excluded sphere reference',
        volume=4 * math.pi * core ** 3 / 3,
        atoms=[dict(center=[0., 0., 0.], radius=core)])
    write(root / 'shape.json', shape)
    shape_hash = sha(root / 'shape.json')
    metadata = dict(native_poses=[native], rigid_members=[fixed],
        member_error_scale=1., angle_error_scale_deg=15.)
    config = dict(shape=str(root / 'shape.json'), fixed_poses=[fixed], initial_pose=native,
        capture_center=[0., 0., 0.], capture_radius=2.2, depletant_radius=.4,
        reservoir_density=activity, poisson_lambda_ratio=8., translation_steps=[.15, .4],
        rotation_steps_deg=[8., 25.], rotation_probability=.5, local_attempts_per_cycle=1,
        uniform_probability=.1, seed=1, metadata=metadata,
        endpoint_gate=dict(max_cells=31, max_depth=8, min_width=.3))
    write(root / 'config.json', config)
    covariance = [[0.] * 6 for _ in range(6)]
    for i in range(6):
        covariance[i][i] = (.6 if i < 3 else .03) ** 2
    region = dict(fixed_neighbor=fixed, physical_fixed_neighbors=[fixed],
        capture_center=[0., 0., 0.], capture_radius=2.2, shape_sha256=shape_hash,
        activity=activity, depletant_radius=.4, physical_metric=metadata,
        minimum_original_q=0., minimum_mahalanobis_radius=0., mahalanobis_radius=4.,
        gaussian_chart=dict(shape_sha256=shape_hash, angular_length=1.1,
            coordinate_convention='anchor-body-relative',
            anchors=[dict(position=[0., 0., 0.], rotation=identity)],
            means=[[0.] * 6], covariances=[covariance], weights=[1.]))
    write(root / 'region.json', region)
    write(root / 'native-compiled.json', compiled_definition(shape_hash, core))


def compiled_definition(shape_hash, core=.3):
    fixed = dict(position=[0., 0., 0.], orientation=[1., 0., 0., 0.])
    identity = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
    compiled = dict(schema='native-entry-compiled-v1', source_definition_sha256='a' * 64,
        source_input_sha256={'tetramer-shape.json': shape_hash},
        criteria=dict(body_member_position_entry_A=2., body_orientation_entry_deg=15.,
            monomer_position_entry_A=3., monomer_orientation_entry_deg=20., contact_entry_A=2.,
            native_reference_patch_gap_A=1., minimum_shared_native_residue_pairs=1,
            hard_overlap_tolerance_A=1e-8, catalogue_cycle_position_tolerance_A=1e-6,
            catalogue_cycle_angle_tolerance_deg=1e-6),
        fixed_poses=[fixed], members=[dict(position=[0., 0., 0.], rotation=identity)],
        monomer_atoms=[dict(center=[0., 0., 0.], radius=core, residue=0)], residue_count=1,
        references=[dict(label='A', family='A', position=[1.5, 0., 0.], rotation=identity,
            native_residue_pairs=[0])],
        motifs=[dict(id=0, position=[1.5, 0., 0.], rotation=identity,
            member_contacts=[dict(member_i=0, member_j=0, directed_class='A')])])
    return compiled



class AnalyticSphereNative:
    """Independent one-atom classifier, with explicit complete match diagnostics."""
    def __init__(self, shape_path, core=.3):
        self.core = core
        self.expected_compiled = compiled_definition(sha(shape_path), core)
        self.source_sha256 = {str(Path(__file__).resolve()): sha(__file__),
                              str(Path(shape_path).resolve()): sha(shape_path)}

    @staticmethod
    def rotation(q):
        w, x, y, z = q
        return [[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]]

    def classify_pair(self, anchor, moving):
        ra = self.rotation(anchor['orientation']); rm = self.rotation(moving['orientation'])
        relative = [moving['position'][i] - anchor['position'][i] for i in range(3)]
        t = [sum(ra[j][i]*relative[j] for j in range(3)) for i in range(3)]
        trace = sum(ra[i][j]*rm[i][j] for i in range(3) for j in range(3))
        angle = math.degrees(math.acos(min(1., max(-1., (trace - 1.)/2.))))
        error = math.sqrt(sum((t[i] - (1.5 if i == 0 else 0.))**2 for i in range(3)))
        gap = math.sqrt(sum(x*x for x in t)) - 2*self.core
        if error > 2. or angle > 15. or gap > 2.:
            return []
        return [dict(motif_id=0, maximum_member_position_error_A=error,
            proper_orientation_error_deg=angle, supporting_member_bonds=[dict(
                members=[0, 0], class_label='A', class_family='A', position_error_A=error,
                orientation_error_deg=angle, minimum_gap_A=gap,
                shared_reference_residue_pairs_entry=[0])])]


def reference_integral(activity, upper, subdivisions=32768):
    """Independent physical radial quadrature; no sampler densities used."""
    def f(r):
        a = .03 / 1.1 * math.sqrt(max(0., 16. - (r / .6) ** 2))
        remainder = min(1., max(0., (1. + (r*r + 2.25 - 4.) / (3.*r)) / 2.))
        lens = math.pi * (2.8 + r) * (1.4 - r) ** 2 / 12. if r < 1.4 else 0.
        return 8.*r*r*(math.atan(a) - a/(1. + a*a))*math.exp(activity*lens)*remainder
    hi = min(2.2, upper)
    h = (hi - .6) / subdivisions
    return (f(.6) + f(hi) + math.fsum((4. if i % 2 else 2.) * f(.6 + h*i)
        for i in range(1, subdivisions))) * h / 3.


def command(out, job):
    out = Path(out)
    inputs = out / 'inputs' / job['fixture']
    return [str(out / 'common/latent-region-smc'), '--config', str(inputs / 'config.json'),
        '--region', str(inputs / 'region.json'), '--exclude-native-entry',
        str(inputs / 'native-compiled.json'), '--out', str(out / 'populations' / job['id']),
        '--seed', str(job['seed']), '--bridge', 'proposal-density',
        '--initial-draws', str(job['initial_draws']), '--population', '192', '--stages', '8',
        '--sweeps-per-stage', '6', '--cloud-replicates', '2', '--lambda-ratio', '8']


def freeze(out, repository, target):
    out = Path(out).resolve(); repository = Path(repository).resolve(); target = Path(target).resolve()
    require(not out.exists(), 'Fresh reference output required; no overwrite')
    require(sys.flags.optimize == 0, 'Disable Python optimization')
    binary = target / 'release/latent-region-smc'
    bundles = list((target / 'release/build').glob('tetramer-mc-*/out/source-bundle.json'))
    require(binary.is_file() and len(bundles) == 1, 'One completed isolated release build required')
    bundle = read(bundles[0])
    require(all(sha(repository / name) == value['sha256'] for name, value in bundle['files'].items()),
        'Compiled source bundle differs from current source')
    inventory = []
    previous_seeds = set()
    for path in sorted({p for name in ('protocol.json', 'plan.json', 'manifest.json')
                        for p in (repository / 'runs').rglob(name)}):
        value = read(path); seeds = seed_values(value)
        previous_seeds.update(seeds)
        inventory.append(dict(path=str(path), sha256=sha(path), seeds=sorted(seeds)))
    require(not previous_seeds.intersection(SEEDS), 'Reference seed collision')
    out.mkdir(parents=True); common = out / 'common'; common.mkdir()
    shutil.copy2(binary, common / 'latent-region-smc')
    shutil.copy2(target / 'release/native-entry-probe', common / 'native-entry-probe')
    shutil.copy2(bundles[0], common / 'source-bundle.json')
    for name, entry in bundle['files'].items():
        destination = common / 'source' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(entry['text'])
        require(sha(destination) == entry['sha256'], 'Source extraction changed bytes')
    for name, path in local_dependencies([Path(__file__), Path(__file__).with_name('test_native_excluded_smc_reference.py')]).items():
        shutil.copy2(path, common / name)
    write(out / 'seed-inventory.json', dict(files=inventory, all_seeds=sorted(previous_seeds)))
    for label, activity, core in [('z0', 0., .3), ('z4', 4., .3), ('zero', 0., 3.)]:
        fixture(out / 'inputs' / label, activity, core)
    jobs = [dict(id=f'z{int(z)}-r{i:02d}', fixture=f'z{int(z)}', activity=z,
        seed=SEEDS[8*j+i], initial_draws=4096, reference=True)
        for j, z in enumerate((0., 4.)) for i in range(8)]
    jobs.append(dict(id='zero', fixture='zero', activity=0., seed=SEEDS[16],
        initial_draws=256, reference=False))
    for job in jobs:
        job['command'] = command(out, job)
    exact = {str(z): {key: reference_integral(z, 2.2 if key == 'total' else 1.4)
        for key in ('total', 'contact')} for z in EXACT}
    for z, values in exact.items():
        values['unbound'] = values['total'] - values['contact']
        require(all(abs(values[key] - EXACT[float(z)][key]) < 1e-12 for key in values),
            'Independent quadrature disagrees with frozen reference')
    source = repository / 'runs/native-excluded-smc-validation-20260924/validation.json'
    plan = dict(schema=SCHEMA, repository=str(repository), created=time.time(),
        python=sys.executable, python_sha256=sha(sys.executable),
        git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repository, text=True).strip(),
        build_command=['cargo', 'build', '--release', '--locked', '--offline', '--jobs', '4',
            '--bin', 'latent-region-smc', '--bin', 'native-entry-probe', '--target-dir', str(target)],
        binary_sha256=sha(common / 'latent-region-smc'),
        probe_sha256=sha(common / 'native-entry-probe'),
        source_bundle_sha256=sha(common / 'source-bundle.json'),
        maximum_physical_workers=4, maximum_total_workers=32, thread_environment=THREAD_ENV,
        total_initial_attempts=16*4096+256, jobs=jobs, exact=exact,
        numerical_reference_tolerance=dict(population_standard_errors=6., absolute=1e-6),
        prior_validation=dict(path=str(source), sha256=sha(source), reused_without_rerun=True),
        audit_required='Independent analyze_native_excluded_smc.py must audit each completed fixture before protein production.',
        scope='Fresh independent implementation/auditor references, not a protein weight or assembly claim.',
        failure='No replacement populations, retries, extension, or partial-population omission; drain started children.')
    write(out / 'plan.json', plan)
    write(out / 'freeze.json', dict(files={str(p.relative_to(out)): sha(p)
        for p in out.rglob('*') if p.is_file()}))
    return plan


def validate(out, expected):
    out = Path(out).resolve()
    require(sha(out / 'plan.json') == expected, 'Frozen reference plan changed')
    require(all(sha(out / name) == digest for name, digest in read(out / 'freeze.json')['files'].items()),
        'Frozen reference inputs changed')
    plan = read(out / 'plan.json')
    require(plan['schema'] == SCHEMA and len(plan['jobs']) == 17
        and plan['maximum_physical_workers'] == 4 and plan['maximum_total_workers'] == 32
        and plan['total_initial_attempts'] == 65792 and plan['thread_environment'] == THREAD_ENV,
        'Reference allocation differs')
    require([j['seed'] for j in plan['jobs']] == list(SEEDS)
        and all(j['command'] == command(out, j) for j in plan['jobs']), 'Reference commands differ')
    require(plan['python'] == sys.executable and plan['python_sha256'] == sha(sys.executable),
        'Frozen Python runtime differs')
    return plan


def run(out, expected):
    out = Path(out).resolve(); plan = validate(out, expected)
    require(Path(__file__).resolve() == out / 'common' / Path(__file__).name,
        'Execute the archived reference driver')
    with (out / 'claim.json').open('x') as stream:
        json.dump(dict(pid=os.getpid(), process_birth=process_token(os.getpid()), plan_sha256=expected), stream)
    state = dict(schema=SCHEMA, complete=False, phase='physical', pid=os.getpid(),
        process_birth=process_token(os.getpid()), plan_sha256=expected, jobs=[])
    (out / 'logs').mkdir(); (out / 'populations').mkdir()
    for job in plan['jobs']:
        state['jobs'].append(dict(id=job['id'], kind='physical', command=job['command'],
            directory=str(out / 'populations' / job['id']), log=str(out / 'logs' / (job['id'] + '.txt')),
            status='pending'))
    def snapshot():
        write(out / 'status.json', state)
    def interrupted(signum, frame):
        raise InterruptedError(f'Reference interrupted by signal {signum}; draining children')
    old = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        execute_group(state['jobs'], snapshot, 4, plan['repository'],
            dict(os.environ, **THREAD_ENV), before_launch=lambda: validate(out, expected))
        report = summarize(out, plan)
        write(out / 'reference.json', report)
        state.update(complete=True, phase='physical_complete_audit_required',
            reference_sha256=sha(out / 'reference.json'))
    except BaseException as exc:
        state.update(phase='failed', error=repr(exc)); raise
    finally:
        snapshot()
        for sig, handler in old.items():
            signal.signal(sig, handler)


def summarize(out, plan):
    result = dict(schema=SCHEMA, complete=True, auditor_validated=False, populations=[], references={})
    for job in plan['jobs']:
        path = Path(out) / 'populations' / job['id']
        summary = read(path / 'summary.json'); manifest = read(path / 'manifest.json')
        require(summary['complete'] is True and manifest['options']['seed'] == job['seed'], 'Unfinished/mismatched population')
        with (path / 'initialization.jsonl').open() as stream:
            attempts = sum(1 for _ in stream)
        require(attempts == job['initial_draws'], 'Missing attempted draws')
        if not job['reference']:
            require(summary['zero_estimate'] is True and summary['initial_hits'] == 0
                and summary['log_Z'] is None, 'All-zero fixture changed')
            result['zero_fixture'] = dict(attempts=attempts, log_Z=None)
            continue
        particles = summary['terminal_particles']
        require(not summary['zero_estimate'] and len(particles) == 192, 'Missing fixed terminal population')
        for particle in particles:
            t = particle['pose']['position']
            require(math.dist(t, [1.5, 0., 0.]) > 2. - 1e-12, 'Retained native terminal pose')
        q = math.exp(summary['log_Z'])
        fraction = sum(math.hypot(*p['pose']['position']) < 1.4 for p in particles) / len(particles)
        result['populations'].append(dict(id=job['id'], activity=job['activity'],
            total=q, contact=q*fraction, unbound=q*(1. - fraction),
            files={name: sha(path / name) for name in ('manifest.json', 'summary.json',
                'initialization.jsonl', 'stages.jsonl')}))
    for activity in (0., 4.):
        group = [p for p in result['populations'] if p['activity'] == activity]
        require(len(group) == 8, 'All independent reference populations required')
        stats = {}
        for key in ('total', 'contact', 'unbound'):
            values = [p[key] for p in group]; mean = math.fsum(values) / 8
            se = math.sqrt(math.fsum((x-mean)**2 for x in values) / (8*7))
            exact = plan['exact'][str(activity)][key]
            stats[key] = dict(mean=mean, population_se=se, exact=exact,
                passed=abs(mean-exact) <= 6*se + 1e-6)
        result['references'][str(activity)] = stats
    result['reference_tolerance_passed'] = all(s['passed'] for g in result['references'].values() for s in g.values())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    freeze_parser = sub.add_parser('freeze')
    freeze_parser.add_argument('--out', type=Path, required=True)
    freeze_parser.add_argument('--repository', type=Path, required=True)
    freeze_parser.add_argument('--target', type=Path, required=True)
    run_parser = sub.add_parser('run')
    run_parser.add_argument('--out', type=Path, required=True)
    run_parser.add_argument('--expected-plan-sha256', required=True)
    args = parser.parse_args()
    if args.action == 'freeze':
        freeze(args.out, args.repository, args.target)
        print(sha(args.out / 'plan.json'))
    else:
        run(args.out, args.expected_plan_sha256)


if __name__ == '__main__':
    main()
