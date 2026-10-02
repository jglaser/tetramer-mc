#!/usr/bin/env python3
"""Freeze two bounded sphere references; only --run starts their one-shot execution.

The alpha=.5/lambda64 reference is reused. The new alpha=.2/lambda128
reference uses exactly two 8192-attempt sphere jobs followed by two full
geometry audits. No protein draws, native classification or retries. Failure
stops new launches while already started, allocated children finish.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
import os
import sys
from importlib.metadata import version
from pathlib import Path
import shutil
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
PILOT = Path('/vast/xvg/tetramer-mc-runs/hard-free-line-physical-pilot-20261001')
REFERENCES = ROOT/'results/hard-free-line-production-validation-20261001'
PYTHON = '/home/xvg/protein-nucleation/.venv/bin/python'
BINARY_SHA = '752c5d7aa36249cb923ca0959b0872b4f96204101090ac1cfca4606732554c7b'
BUNDLE_SHA = '30f7234102da852263c2853d3d32a885f9913553f3c632086913da6cb23f3ed3'
SCHEMA = 'hard-free-line-sensitivity-reference-allocation-v1'
SEEDS = [610016211, 610016212]
SAMPLES = 8192
ARMS = [dict(id='alpha02', beta=1., alpha=.2, component_count=92, samples=16384, lambda_ratio=128.),
        dict(id='lambda64', beta=1., alpha=.5, component_count=92, samples=16384, lambda_ratio=64.)]
AUDIT_SOURCES = ['hard_free_line_physical_reference.py', 'hard_free_line_reference.py',
                 'analyze_contact_line_audit.py', 'analyze_native_region_reference.py', 'analyze_basin_normalizers.py']
INVENTORY_ROOTS = ['/vast/xvg/tetramer-mc-runs', '/vast/xvg/tetramer-mc/runs', '/vast/xvg/tetramer-mc/results',
                   '/vast/xvg/protein-nucleation/results', '/vast/xvg/protein-nucleation-20260920/results']
DECLARATIONS = ['protocol.json', 'plan.json', 'allocation.json', 'manifest.json', 'config.json',
                'declaration.json', 'prospective-plan.json', 'prospective-pilot-plan.json', 'campaign.json']
PHYSICAL_NAMES = {'basin-normalizer', 'latent-region-normalizer', 'latent-region-smc',
                  'native-region-normalizer', 'tetramer-mc', 'docking-mc', 'gate-benchmark', 'gate_benchmark', 'contact_atlas'}


def require(value, message):
    if not value: raise ValueError(message)


def runtime():
    require(sys.flags.optimize == 0, 'Unoptimized Python is required for reference assertions')
    require(sys.executable == PYTHON, 'Use the pinned virtual-environment Python without resolving its symlink')
    return dict(executable=sys.executable, python_version=sys.version,
                numpy_version=version('numpy'), scipy_version=version('scipy'), optimize=0)


def read(path): return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''): digest.update(chunk)
    return digest.hexdigest()


def write(path, value):
    with Path(path).open('x') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False)+'\n')


def check_files(files):
    for path, digest in files.items(): require(sha(path) == digest, 'Changed bound file: '+str(path))


def seeds_in(value):
    found = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if 'seed' in key.lower():
                for number in item if isinstance(item, list) else [item]:
                    if type(number) is int or isinstance(number, str) and number.isdecimal(): found.add(int(number))
            found.update(seeds_in(item))
    elif isinstance(value, list):
        for i, item in enumerate(value):
            if item == '--seed' and i+1 < len(value):
                number = value[i+1]
                if type(number) is int or isinstance(number, str) and number.isdecimal(): found.add(int(number))
            found.update(seeds_in(item))
    return found


def inventory(roots):
    require(all(Path(p).is_dir() for p in roots), 'Missing seed inventory root')
    command = ['rg', '--files', '--hidden', '--no-ignore', *map(str, roots)]
    for name in DECLARATIONS: command += ['-g', name]
    result = subprocess.run(command, text=True, capture_output=True)
    require(result.returncode in (0, 1), 'Seed inventory failed: '+result.stderr)
    files, seen = {}, set()
    for path in sorted({str(Path(p).resolve()) for p in result.stdout.splitlines()}):
        seeds = seeds_in(read(path)); seen.update(seeds)
        files[path] = dict(sha256=sha(path), declared_seeds=sorted(seeds))
    require(files and not seen.intersection(SEEDS), 'Empty seed inventory or fresh seed collision')
    return dict(roots=list(map(str, roots)), command=command, files=files, files_scanned=len(files),
                previous_unique_seeds=len(seen), fresh_seeds=SEEDS, collisions=[],
                scope='Declared seeds in metadata and command arguments; no trajectory replay.')


def reused_references(root):
    """Authenticate existing reference receipts and bytes, without re-auditing rows."""
    root = Path(root).resolve(); files = {}; jobs = []
    def bind(path):
        files[str(path)] = sha(path)
        return read(path)
    validation = bind(root/'validation.json')
    require(validation['complete'] is True and validation['production_binary']['sha256'] == BINARY_SHA
            and validation['production_binary']['source_bundle_sha256'] == BUNDLE_SHA, 'Unvalidated source references')
    for activity in (0, 2):
        base = root/'toys/analytic'/f'activity{activity}'
        ref = bind(base/'reference.json'); audit = bind(base/'independent-physical-audit.json')
        manifest = bind(base/'run/manifest.json'); summary = bind(base/'run/summary.json')
        require(ref['activity'] == activity and ref['samples'] == SAMPLES and ref['lambda_ratio'] == 64
                and all(ref[k]['passed'] is True and ref[k]['attempted_denominator'] == SAMPLES
                        for k in ('hard', 'depletion', 'latent_volume')), 'Existing analytic reference failed')
        require(audit['complete'] is True and audit['geometry_mode'] == 'full' and audit['samples'] == SAMPLES
                and audit['independently_reconstructed_geometry_rows'] == SAMPLES, 'Missing full reused geometry audit')
        require(summary['complete'] is True and summary['manifest'] == manifest == ref['manifest']
                and manifest['executable_sha256'] == BINARY_SHA and manifest['source_bundle_sha256'] == BUNDLE_SHA
                and manifest['importance_uniform_probability'] == .5 and manifest['lambda_ratio'] == 64
                and manifest['cloud_replicates'] == 2 and manifest['activity'] == activity, 'Reused reference identity differs')
        require(audit['executable_sha256'] == BINARY_SHA and audit['source_bundle_sha256'] == BUNDLE_SHA,
                'Reused audit executable differs')
        for key in ('samples_sha256', 'attempts_sha256'):
            require(ref[key] == summary[key] == audit[key], 'Reused attempts or rows differ')
        check_files(audit['files']); files.update(audit['files'])
        jobs.append(dict(activity=activity, reference=str(base/'reference.json'),
                         audit=str(base/'independent-physical-audit.json'), samples=SAMPLES))
    return dict(complete=True, alpha=.5, beta=1., lambda_ratio=64., jobs=jobs, files=files,
                no_rows_replayed=True, no_new_draws=True)


def prepare(out, pilot=PILOT, references=REFERENCES):
    environment = runtime()
    out, pilot = Path(out).resolve(), Path(pilot).resolve()
    require(not out.exists(), 'Fresh reference preparation directory required')
    prior = read(pilot/'protocol.json'); status = read(pilot/'status.json'); common = pilot/'common'
    require(status['complete'] is True and status['phase'] == 'complete'
            and status['protocol_sha256'] == sha(pilot/'protocol.json'), 'Completed original pilot required')
    require(prior['binary_sha256'] == sha(common/'latent-region-normalizer') == BINARY_SHA
            and prior['source_bundle_sha256'] == sha(common/'source-bundle.json') == BUNDLE_SHA,
            'Exact archived production binary/source required')
    source = read(common/'source-bundle.json')
    require((common/'source-bundle.json').read_bytes() in (common/'latent-region-normalizer').read_bytes(),
            'Source bundle is not embedded in pinned executable')
    for name, entry in source['files'].items():
        require(hashlib.sha256(entry['text'].encode()).hexdigest() == entry['sha256']
                == sha(common/'rust-source'/name), 'Embedded Rust source differs: '+name)
    for name in AUDIT_SOURCES:
        require(sha(common/name) == prior['python_sources'][name], 'Archived Python audit source differs: '+name)
    reused = reused_references(references)
    seeds = inventory([ROOT/'runs', ROOT/'results', *INVENTORY_ROOTS])
    out.mkdir(parents=True); target = out/'common'; target.mkdir()
    for name in ('latent-region-normalizer', 'source-bundle.json', *AUDIT_SOURCES): shutil.copy2(common/name, target/name)
    shutil.copy2(__file__, target/Path(__file__).name)
    fixture = Path(references).resolve()/'archive/sources/tests/latent_region_hard_free_line.rs'
    shutil.copy2(fixture, target/'latent_region_hard_free_line.rs')
    write(out/'reused-references.json', reused); write(out/'seed-inventory.json', seeds)
    jobs = []
    for activity, seed in zip((0, 2), SEEDS):
        base = out/f'activity{activity}'; base.mkdir()
        original = Path(references).resolve()/'toys/analytic'/f'activity{activity}'/'run/provenance'
        for name in ('shape.json', 'region.json'): shutil.copy2(original/name, base/name)
        config = read(original/'input-config.json'); config['shape'] = str(base/'shape.json')
        config['poisson_lambda_ratio'] = 128.; write(base/'config.json', config)
        guide = read(original/'importance-guide.json'); guide['defensive_uniform_shell_probability'] = .2
        require(guide['conditional_probability'] == 1 and len(guide['gaussian_components']) == 2
                and guide['raw_translation_axes'] == [0, 1, 2] and guide['minimum_conditional_mass'] == 1e-12,
                'Analytic sphere fixture changed')
        write(base/'guide.json', guide)
        jobs.append(dict(id=f'activity{activity}', activity=activity, seed=seed, samples=SAMPLES,
                         directory=str(base/'run'), audit=str(base/'audit.json')))
    plan = dict(schema=SCHEMA, jobs=jobs, target_arms=ARMS, alpha=.2, beta=1., lambda_ratio=128.,
                reference_component_count=2, cloud_replicates=2, total_unconditional_draws=2*SAMPLES,
                maximum_physical_workers=2, maximum_audit_workers=2, maximum_global_physical_workers=8,
                maximum_total_workers=32, python=PYTHON, runtime=environment,
                binary_sha256=BINARY_SHA, source_bundle_sha256=BUNDLE_SHA,
                no_protein_jobs=True, no_native_classifier=True, no_retries=True, failure_draining=True,
                no_autoextension=True, no_optional_stopping=True, all_attempts_retained=True,
                analytic_standard_error_multiplier=6.5, independent_geometry='full',
                reference_fixture_sha256=sha(fixture), preparer_sha256=sha(__file__),
                reused_reference_sha256=sha(out/'reused-references.json'),
                seed_inventory_sha256=sha(out/'seed-inventory.json'),
                scope='Sphere normalization and depletion oracle at activities 0 and 2 plus full independent geometry. '
                      'Production arm metadata identifies the downstream 92-component control; toy mixture has two components. '
                      'Existing lambda64 references are authenticated and reused. No classifier or protein validation claim.')
    write(out/'allocation.json', plan)
    write(out/'freeze.json', dict(files={str(p.relative_to(out)): sha(p) for p in sorted(out.rglob('*')) if p.is_file()}))
    return dict(out=str(out), allocation_sha256=sha(out/'allocation.json'), preparation_only=True,
                fresh_seeds=SEEDS, total_new_attempts=2*SAMPLES, launched=False)


def analytic_reference(activity):
    """Same radial/Haar sphere oracle as the archived Rust integration test."""
    def integrand(radius):
        angle = math.sqrt(max(0., 2.4**2-radius**2))
        angular = .5*(math.atan(angle)-angle/(1+angle**2))
        overlap = math.pi*(2.8+radius)*(1.4-radius)**2/12 if radius < 1.4 else 0.
        return 16*radius**2*angular*math.exp(activity*overlap)
    n = 32768; step = (2.4-.6)/n
    return (integrand(.6)+integrand(2.4)+math.fsum(integrand(.6+i*step)*(2 if i%2 == 0 else 4)
            for i in range(1, n)))*step/3


def check_mean(values, expected):
    n = len(values); mean = math.fsum(values)/n
    se = math.sqrt(math.fsum((v-mean)**2 for v in values)/(n*(n-1)))
    passed = abs(mean-expected) < 6.5*se+1e-7
    require(passed, f'Analytic sphere check failed: mean={mean}, reference={expected}, SE={se}')
    return dict(mean=mean, reference=expected, standard_error=se, passed=passed, attempted_denominator=n)


def validate_result(out, job):
    base = out/job['id']; run = base/'run'; summary = read(run/'summary.json'); audit = read(base/'audit.json')
    manifest = summary['manifest']
    expected = dict(seed=job['seed'], samples=SAMPLES, cloud_replicates=2, lambda_ratio=128.,
                    activity=job['activity'], executable_sha256=BINARY_SHA, source_bundle_sha256=BUNDLE_SHA,
                    importance_uniform_probability=.2, importance_component_count=2,
                    config_sha256=sha(base/'config.json'), region_sha256=sha(base/'region.json'),
                    shape_sha256=sha(base/'shape.json'), importance_guide_sha256=sha(base/'guide.json'))
    require(summary['complete'] is True and all(manifest[k] == v for k, v in expected.items()), 'New reference identity differs')
    require(audit['complete'] is True and audit['geometry_mode'] == 'full' and audit['samples'] == SAMPLES
            and audit['independently_reconstructed_geometry_rows'] == SAMPLES
            and audit['samples_sha256'] == summary['samples_sha256']
            and audit['attempts_sha256'] == summary['attempts_sha256'], 'New full geometry audit incomplete')
    check_files(audit['files'])
    mass, hard, volume = [], [], []; exterior = hard_zero = conditioned = different = 0
    for text in (run/'samples.jsonl').read_text().splitlines():
        row = json.loads(text); inside = row['shell_valid']; valid = inside and row['hard_valid'] and row['region_valid'] and row['capture_valid']
        volume.append(math.exp(-row['log_proposal_density']) if inside else 0.)
        hard.append(math.exp(row['log_hard_weight']) if valid else 0.)
        mass.append(math.exp(row['log_importance_weight']) if valid else 0.)
        exterior += not inside; hard_zero += inside and not row['hard_valid']; conditioned += row['hard_free_line_draw']['conditional']
        different += valid and row['clouds'][0] != row['clouds'][1]
    require(len(mass) == SAMPLES and exterior > 10 and hard_zero > 10 and conditioned > 1000,
            'Missing attempted rows, exterior/hard zeros or conditioned draws')
    require(job['activity'] == 0 or different > 10, 'No observed distinct cloud records at positive activity')
    result = dict(activity=job['activity'], seed=job['seed'], samples=SAMPLES, cloud_replicates=2,
                  alpha=.2, beta=1., lambda_ratio=128., hard=check_mean(hard, analytic_reference(0.)),
                  depletion=check_mean(mass, analytic_reference(job['activity'])),
                  latent_volume=check_mean(volume, math.pi**3*2.4**6/6), outside_zeros=exterior,
                  hard_zeros=hard_zero, conditioned_draws=conditioned, different_cloud_records=different,
                  audit_sha256=sha(base/'audit.json'), samples_sha256=summary['samples_sha256'],
                  attempts_sha256=summary['attempts_sha256'])
    write(base/'reference.json', result)
    return result


def capacity():
    physical, workers = [], 0
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit(): continue
        try:
            if proc.stat().st_uid != os.getuid(): continue
            stat = (proc/'stat').read_text().rsplit(')', 1)[1].split()
            if stat[0] == 'Z': continue
            executable = Path(os.readlink(proc/'exe').removesuffix(' (deleted)')).name
            args = os.fsdecode((proc/'cmdline').read_bytes())
            is_physical = executable in PHYSICAL_NAMES
            if is_physical: physical.append(int(proc.name))
            if is_physical or ('python' in executable and ('tetramer-mc' in args or 'protein-nucleation' in args)):
                workers += int(stat[17])
        except (FileNotFoundError, ProcessLookupError, PermissionError): continue
    private_namespace = 'codex-linux-san' in (Path('/proc/1/comm').read_text())
    return dict(physical_pids=physical, workers=workers, private_pid_namespace=private_namespace)


def execute_group(out, jobs, kind, expected_runtime=None):
    environment = runtime()
    require(expected_runtime is None or environment == expected_runtime, 'Reference Python runtime changed')
    active = []; records = []; env = os.environ.copy(); interrupted = False
    env.update({k: '1' for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS')})
    env['PYTHONOPTIMIZE'] = '0'
    try:
        for job in jobs:
            require(runtime() == environment, 'Reference Python runtime changed before launch')
            counts = capacity()
            require(not counts['private_pid_namespace'], 'Global capacity cannot be observed inside private sandbox PID namespace')
            require(counts['workers'] < 32 and (kind != 'physical' or len(counts['physical_pids']) < 8),
                    'Global capacity unavailable; no waiting, retry or extension')
            require(all(child.poll() in (None, 0) for _, child in active), 'Earlier child failed; stop further launches')
            base = out/job['id']
            if kind == 'physical':
                require(not (base/'run').exists(), 'Existing reference output; no retry')
                command = [str(out/'common/latent-region-normalizer'), '--config', str(base/'config.json'),
                           '--region', str(base/'region.json'), '--importance-guide', str(base/'guide.json'),
                           '--out', str(base/'run'), '--samples', str(SAMPLES), '--seed', str(job['seed']),
                           '--cloud-replicates', '2', '--lambda-ratio', '128']
            else:
                command = [PYTHON, '-B', str(out/'common/hard_free_line_physical_reference.py'), '--directory',
                           str(base/'run'), '--out', str(base/'audit.json'), '--geometry', 'full']
            record = dict(id=job['id'], kind=kind, command=command, started=time.time())
            with (base/(kind+'.log')).open('xb') as log:
                child = subprocess.Popen(command, cwd=out, env=env, stdout=log, stderr=subprocess.STDOUT)
            record['pid'] = child.pid; active.append((record, child)); records.append(record)
            write(base/(kind+'-launch.json'), record)
        while active:
            for record, child in list(active):
                code = child.poll()
                if code is None: continue
                record.update(returncode=code, finished=time.time()); active.remove((record, child))
                require(code == 0, 'Reference child failed; stop launches and drain already allocated peers')
            if active: time.sleep(.1)
    except (KeyboardInterrupt, InterruptedError):
        interrupted = True
        raise
    finally:
        if interrupted:
            for _, child in active:
                if child.poll() is None: child.terminate()
        for record, child in active:
            try: code = child.wait(timeout=5 if interrupted else None)
            except (KeyboardInterrupt, InterruptedError):
                interrupted = True
                for _, peer in active:
                    if peer.poll() is None: peer.terminate()
                try: code = child.wait(timeout=5)
                except subprocess.TimeoutExpired: child.kill(); code = child.wait()
            except subprocess.TimeoutExpired: child.kill(); code = child.wait()
            record.update(returncode=code, finished=time.time(), aborted=interrupted)
        write(out/(kind+'-execution.json'), dict(jobs=records, no_retries=True, failure_draining=True))


def run(out):
    environment = runtime()
    out = Path(out).resolve(); plan = read(out/'allocation.json')
    require(plan['runtime'] == environment, 'Reference Python runtime differs from frozen preparation')
    frozen = {str(out/name): digest for name, digest in read(out/'freeze.json')['files'].items()}
    check_files(frozen)
    require(sha(__file__) == plan['preparer_sha256'], 'Run the frozen preparer bytes')
    require(plan['schema'] == SCHEMA and plan['target_arms'] == ARMS and plan['total_unconditional_draws'] == 2*SAMPLES
            and [j['seed'] for j in plan['jobs']] == SEEDS and len(plan['jobs']) == 2, 'Changed fixed reference allocation')
    reused = read(out/'reused-references.json'); check_files(reused['files'])
    counts = capacity()
    require(not counts['private_pid_namespace'], 'Run only in the host PID namespace; sandbox cannot observe global capacity')
    require(len(counts['physical_pids']) <= 6 and counts['workers'] <= 30,
            'Two global worker slots required before starting reference allocation')
    write(out/'launch-claim.json', dict(started=time.time(), pid=os.getpid(), allocation_sha256=sha(out/'allocation.json'), capacity=counts))
    def interrupt(signum, frame): raise InterruptedError('Reference execution interrupted; terminate children and retain journals')
    previous = signal.signal(signal.SIGTERM, interrupt)
    try:
        execute_group(out, plan['jobs'], 'physical', environment)
        execute_group(out, plan['jobs'], 'audit', environment)
        results = [validate_result(out, job) for job in plan['jobs']]
        check_files(frozen); check_files(reused['files'])
        files = dict(reused['files']); files.update(frozen); files[str(out/'freeze.json')] = sha(out/'freeze.json')
        for job in plan['jobs']:
            audit = read(out/job['id']/'audit.json'); files.update(audit['files'])
            for name in ('audit.json', 'reference.json'):
                path = out/job['id']/name; files[str(path)] = sha(path)
        receipt = dict(schema='hard-free-line-sensitivity-prerequisites-v1', complete=True, all_checks_passed=True,
                       binary_sha256=BINARY_SHA, source_bundle_sha256=BUNDLE_SHA, arms=ARMS, files=files,
                       independent_geometry='full', new_references=results, reused_references=reused,
                       reference_component_count=2, reference_attempts=2*SAMPLES, no_protein_jobs=True,
                       no_native_classifier=True, checks=['sphere latent-volume normalization', 'full J/q hard sphere mass',
                       'two-cloud depletion sphere mass', 'whole-line atom geometry, density and conditional inverse CDF',
                       'all attempted rows, journals, invalid/exterior zeros', 'exact archived executable and source bundle'],
                       scope=plan['scope'])
        write(out/'prerequisites.json', receipt)
        return dict(complete=True, receipt=str(out/'prerequisites.json'), receipt_sha256=sha(out/'prerequisites.json'))
    except BaseException as exc:
        write(out/'failure.json', dict(complete=False, error=str(exc), finished=time.time(), no_retry=True))
        raise
    finally: signal.signal(signal.SIGTERM, previous)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True); parser.add_argument('--run', action='store_true')
    parser.add_argument('--pilot', type=Path, default=PILOT); parser.add_argument('--references', type=Path, default=REFERENCES)
    args = parser.parse_args()
    print(json.dumps(run(args.out) if args.run else prepare(args.out, args.pilot, args.references), indent=2))
