#!/usr/bin/env python3
"""Freeze, preflight and run the fixed independent entry-shell R4 reference.

Freeze/preflight never execute the sampler. All 65,536 attempted draws belong
to the unchanged physical R4 target; native/contact labels are downstream only.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from entry_shell_proposal import EntryShellGuide
from prepare_shoulder_docking_benchmark import local_dependencies
from run_latent_reference_campaign import validate_region
from run_mobile_posterior_pilot import verify_bundle

ROOT = Path(__file__).resolve().parents[1]
RUNNER_NAME = 'run_entry_shell_reference_campaign.py'
SCHEMA = 'entry-shell-reference-controller-v1'
CAMPAIGN_SCHEMA = 'importance-latent-region-campaign-v2'
POPULATION_SCHEMA = 'importance-latent-region-normalizer-v2'
GUIDE_SCHEMA = 'defensive-entry-shell-guide-v1'
PROPOSAL_KIND = 'orientation-marginal-member-shell-mixture'
DENSITY_MEASURE = 'Lebesgue measure in the original six-dimensional whitened region chart'
SEEDS = [130101010 + 1009*i for i in range(4)]
SAMPLES = 16384
ALLOCATION = dict(populations=4, samples_per_population=SAMPLES,
                  total_unconditional_draws=65536, seeds=SEEDS,
                  cloud_replicates=2, lambda_ratio=64., maximum_physical_workers=4)
PINS = {
    'plan.json': '8e4012e36a008412509f9d2b7aed7ea75f7de3c507bb90fac066ea2f706cce60',
    'freeze.json': 'a1abf1b1a9e7fbdb6d1504075e7a825ffae997f35c1336640e4095f2f02cc670',
    'guide.json': '80df09a9a802dd4a7f2bd2a04de7f7745055067f09afc7b08cdf2eee2a6826de',
    'region.json': '924648f4d9db473045397c300b3b4af7ccde8cfda1b1703a6899239ec12e2f02',
    'config.json': 'a983737d3d4c974871ee5eac31a228bcae1820d2b98956cda9252b729a9c9b45',
}


def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path, value): Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
def require(condition, message):
    if not condition:
        raise ValueError(message)


def inside(root, name):
    root, name = Path(root).resolve(), Path(name)
    require(not name.is_absolute() and '..' not in name.parts, 'Unsafe archive path')
    path = (root/name).resolve()
    require(path.is_relative_to(root), 'Archive path escapes root')
    return path


def file_hashes(root):
    return {p.relative_to(root).as_posix(): sha(p)
            for p in sorted(Path(root).rglob('*')) if p.is_file()}


def validate_package(package):
    """Bind the selected proposal and the old physical target independently."""
    package = Path(package).resolve()
    for name, digest in PINS.items():
        require(sha(package/name) == digest, 'Selected preparation changed: '+name)
    frozen = read(package/'freeze.json')['files']
    require(frozen and 'plan.json' in frozen, 'Empty preparation freeze')
    for name, digest in frozen.items():
        require(sha(inside(package, name)) == digest, 'Preparation dependency changed: '+name)
    plan = read(package/'plan.json')
    require(plan['schema'] == 'mobile-threshold-entry-shell-preparation-v1'
            and plan['complete'] is True and plan['launched'] is False, 'Require inert completed preparation')
    require(plan['allocation'] == ALLOCATION, 'Independent fixed-N allocation changed')
    for field in ('guide', 'config', 'region'):
        require(plan[field] == field+'.json' and plan[field+'_sha256'] == sha(package/plan[field]),
                'Preparation input binding differs: '+field)
    require(sha(inside(package, plan['native_definition'])) == plan['native_definition_sha256'],
            'Frozen native definition changed')
    cfg, region, guide = (read(package/(name+'.json')) for name in ('config', 'region', 'guide'))
    source = read(package/'inputs/source-config.json')
    restored = copy.deepcopy(cfg); restored['shape'] = source['shape']
    require(restored == source and sha(package/'inputs/source-config.json') == plan['source_config_sha256'],
            'Preparation physical source changed beyond shape relocation')
    shape = package/'inputs/tetramer-shape.json'
    require(sha(shape) == plan['shape_sha256'], 'Physical shape changed')
    require(cfg['depletant_radius'] == 1.5 and cfg['reservoir_density'] == .035
            and cfg['poisson_lambda_ratio'] == 64. and cfg['capture_radius'] == 170.
            and len(cfg['fixed_poses']) == 2, 'Wrong physical scaffold/bath/capture')
    require(region.get('minimum_mahalanobis_radius', 0.) == 0. and region['mahalanobis_radius'] == 4.
            and region['minimum_original_q'] == 0.
            and region.get('minimum_original_q_inclusive', True) is True
            and 'maximum_original_q' not in region, 'R4 target gained a q/native filter')
    parsed = EntryShellGuide(guide, region, plan['region_sha256'])
    require(parsed.schema == GUIDE_SCHEMA and parsed.alpha == .2 and parsed.count == 24,
            'Wrong frozen entry-shell proposal')
    geometry = validate_region(region, cfg, read(shape), plan['shape_sha256'])
    require(geometry['chart_anchor_index'] == 1 and geometry['entire_chart_inside_capture'],
            'Require the same B-anchored R4 inside capture')
    return plan, geometry


def command(archive, job):
    return [str(archive/'latent-region-normalizer'), '--config', str(archive/'config.json'),
            '--region', str(archive/'region.json'), '--importance-guide', str(archive/'importance-guide.json'),
            '--out', job['directory'], '--samples', str(job['samples']), '--seed', str(job['seed']),
            '--cloud-replicates', '2', '--lambda-ratio', '64.0']


def freeze(out, binary, source_bundle, package):
    out, binary, source_bundle, package = map(lambda p: Path(p).resolve(), (out, binary, source_bundle, package))
    require(not out.exists(), 'Use a fresh campaign; no overwrite')
    plan, geometry = validate_package(package)
    bundle, rust_sources = verify_bundle(binary, source_bundle, ROOT)
    # Both the observer and controller import closures are resolved before copying.
    python = local_dependencies([Path(__file__), ROOT/'tools/analyze_latent_region.py',
                                 ROOT/'tools/entry_shell_proposal.py'])
    require('entry_shell_proposal.py' in python and 'analyze_latent_region.py' in python,
            'Incomplete current observer closure')
    archive = out/'provenance'; archive.mkdir(parents=True)
    for name, path in {'source-config.json': package/'config.json', 'region.json': package/'region.json',
                       'importance-guide.json': package/'guide.json', 'shape.json': package/'inputs/tetramer-shape.json',
                       'latent-region-normalizer': binary, 'source-bundle.json': source_bundle,
                       'controller.py': Path(__file__)}.items():
        shutil.copy2(path, archive/name)
    for name, entry in bundle['files'].items():
        path = inside(archive/'source', name); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(entry['text'].encode('utf-8'))
    for name, path in python.items():
        shutil.copy2(path, archive/name)
    # Copy exactly the immutable package ledger; ignore incidental Python caches.
    package_archive = archive/'reference-package'; package_archive.mkdir()
    for name in [*read(package/'freeze.json')['files'], 'freeze.json']:
        target = inside(package_archive, name); target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(inside(package, name), target)
    cfg = read(package/'config.json'); cfg['shape'] = str(archive/'shape.json')
    write(archive/'config.json', cfg); write(archive/'geometry-preflight.json', geometry)
    for name in ('runs', 'logs'): (out/name).mkdir()
    jobs = []
    for i, seed in enumerate(SEEDS):
        ident = f'r{i:02d}'
        job = dict(id=ident, seed=seed, samples=SAMPLES, directory=str(out/'runs'/ident),
                   log=str(out/'logs'/f'{ident}.log'))
        job['command'] = command(archive, job); jobs.append(job)
    manifest = dict(schema=CAMPAIGN_SCHEMA, jobs=jobs, workers=4, physical_activity=.035,
                    lambda_ratio=64., cloud_replicates=2, config_sha256=sha(archive/'config.json'),
                    region_sha256=sha(archive/'region.json'), shape_sha256=sha(archive/'shape.json'),
                    importance_guide_sha256=sha(archive/'importance-guide.json'),
                    archive_sha256=file_hashes(archive),
                    source_inputs=dict(package=str(package), binary=str(binary), source_bundle=str(source_bundle)),
                    scope='Same fixed physical R4, full normalized defensive entry-shell proposal. Every attempted draw remains in N; native/contact classes are downstream only. No discovery-row pooling.')
    write(out/'manifest.json', manifest)
    protocol = dict(schema=SCHEMA, **ALLOCATION, manifest_sha256=sha(out/'manifest.json'),
                    controller_sha256=sha(__file__), physical_executable_sha256=sha(binary),
                    source_bundle_sha256=sha(source_bundle), rust_sources=rust_sources,
                    python_sources={name: sha(path) for name, path in python.items()},
                    source_config_sha256=plan['config_sha256'], region_sha256=plan['region_sha256'],
                    importance_guide_sha256=plan['guide_sha256'], shape_sha256=plan['shape_sha256'],
                    chart_anchor_index=1, reference_package=dict(path='provenance/reference-package',
                        plan_sha256=sha(package/'plan.json'), freeze_sha256=sha(package/'freeze.json')),
                    native_definition='provenance/reference-package/'+plan['native_definition'],
                    native_definition_sha256=plan['native_definition_sha256'],
                    stopping='Exactly four independent fixed-N streams. No retries/resume/extensions; drain started children. One archived audit only after all four outputs validate.',
                    interpretation='Fresh guide estimates remain separate from the earlier equal-N uniform population used for discovery. No whole-basin, equilibrium, tail-coverage or efficiency claim.')
    write(out/'protocol.json', protocol)
    write(out/'freeze.json', dict(files=file_hashes(out)))
    validate(out)
    return protocol


def validate(out):
    out = Path(out).resolve(); archive = out/'provenance'
    protocol, manifest = read(out/'protocol.json'), read(out/'manifest.json')
    require(protocol['schema'] == SCHEMA and manifest['schema'] == CAMPAIGN_SCHEMA, 'Unsupported campaign law')
    for name, digest in read(out/'freeze.json')['files'].items():
        require(sha(inside(out, name)) == digest, 'Frozen file changed: '+name)
    require(sha(out/'manifest.json') == protocol['manifest_sha256'], 'Manifest binding differs')
    require(sha(archive/'controller.py') == protocol['controller_sha256'], 'Controller binding differs')
    for name, digest in manifest['archive_sha256'].items():
        require(sha(inside(archive, name)) == digest, 'Archived dependency changed: '+name)
    require(all(protocol.get(k) == v for k, v in ALLOCATION.items()), 'Fixed allocation changed')
    package = inside(out, protocol['reference_package']['path']); plan, _ = validate_package(package)
    require(sha(package/'plan.json') == protocol['reference_package']['plan_sha256']
            and sha(package/'freeze.json') == protocol['reference_package']['freeze_sha256'], 'Package binding changed')
    require(sha(inside(out, protocol['native_definition'])) == protocol['native_definition_sha256']
            == plan['native_definition_sha256'], 'Native reporting definition changed')
    cfg, original = read(archive/'config.json'), read(archive/'source-config.json')
    restored = copy.deepcopy(cfg); restored['shape'] = original['shape']
    require(restored == original and cfg['shape'] == str(archive/'shape.json'), 'Changed physical config beyond shape relocation')
    require(sha(archive/'source-config.json') == protocol['source_config_sha256'] == plan['config_sha256'], 'Source config differs')
    require(sha(archive/'config.json') == manifest['config_sha256'], 'Resolved config differs')
    for field, name in [('region_sha256', 'region.json'), ('shape_sha256', 'shape.json'),
                        ('importance_guide_sha256', 'importance-guide.json')]:
        require(sha(archive/name) == manifest[field] == protocol[field], 'Physical/proposal binding differs: '+name)
        require(protocol[field] == plan['guide_sha256' if field == 'importance_guide_sha256' else field],
                'Campaign differs from selected package: '+name)
    require(sha(archive/'latent-region-normalizer') == protocol['physical_executable_sha256']
            and sha(archive/'source-bundle.json') == protocol['source_bundle_sha256'], 'Executable/source identity differs')
    _, rust = verify_bundle(archive/'latent-region-normalizer', archive/'source-bundle.json', archive/'source')
    require(rust == protocol['rust_sources'], 'Archived Rust source membership changed')
    for name, digest in protocol['python_sources'].items():
        require(sha(inside(archive, name)) == digest, 'Archived Python closure changed: '+name)
    closure = local_dependencies([archive/RUNNER_NAME, archive/'analyze_latent_region.py', archive/'entry_shell_proposal.py'])
    require(set(closure) == set(protocol['python_sources']), 'Python import closure membership changed')
    region = read(archive/'region.json')
    geometry = validate_region(region, cfg, read(archive/'shape.json'), manifest['shape_sha256'])
    require(geometry == read(archive/'geometry-preflight.json') and geometry['chart_anchor_index'] == 1,
            'Geometry enclosure changed')
    require(manifest['workers'] == 4 and manifest['cloud_replicates'] == 2 and manifest['lambda_ratio'] == 64.
            and manifest['physical_activity'] == cfg['reservoir_density'] == .035, 'Bath/allocation differs')
    require(len(manifest['jobs']) == 4, 'Exactly four independent populations required')
    for i, (job, seed) in enumerate(zip(manifest['jobs'], SEEDS)):
        require(job['id'] == f'r{i:02d}' and job['seed'] == seed and job['samples'] == SAMPLES, 'Job allocation differs')
        require(job['directory'] == str(out/'runs'/job['id']) and job['log'] == str(out/'logs'/f"{job['id']}.log"), 'Job output escaped campaign')
        require(job['command'] == command(archive, job), 'Physical command changed')
    return protocol


def check(out):
    out = Path(out).resolve(); protocol = validate(out)
    require(sha(__file__) == protocol['controller_sha256'], 'Run the exact archived controller')
    require(not (out/'status.json').exists() and not (out/'assessment').exists(), 'No retry/resume/audit overwrite')
    require(not any((out/'runs').iterdir()) and not any((out/'logs').iterdir()), 'Existing physical outputs or logs')
    require(not list(out.glob('r*-status.json')), 'Existing population status')
    return protocol


def execute_jobs(jobs, snapshot, popen=subprocess.Popen):
    """Launch once, at most four children, then reap all before reporting failure."""
    require(len(jobs) <= 4, 'Maximum four physical workers')
    active, error = [], None
    try:
        for job in jobs:
            require(not Path(job['directory']).exists(), 'Existing population output')
            with Path(job['log']).open('xb') as log:
                child = popen(job['command'], stdout=log, stderr=subprocess.STDOUT)
            active.append((job, child)); job.update(status='running', pid=child.pid, started=time.time()); snapshot()
    except BaseException as exc:
        error = exc
    finally:
        for job, child in active:
            while True:
                try:
                    code = child.wait(); break
                except KeyboardInterrupt as exc:
                    # Preserve interruption, but never orphan a started sampler.
                    if error is None: error = exc
            job.update(returncode=code, status='complete' if code == 0 else 'failed', finished=time.time())
            if code != 0 and error is None: error = RuntimeError('Physical failure; no retry or audit')
            try: snapshot()
            except BaseException as exc:
                if error is None: error = exc
        for job in jobs:
            if job['status'] == 'pending': job['status'] = 'not_started'
        snapshot()
    if error is not None: raise error


def verify_output(out, manifest, job):
    out = Path(out); directory = Path(job['directory']); archive = out/'provenance'
    summary, pm = read(directory/'summary.json'), read(directory/'manifest.json')
    require(summary['complete'] is True and summary['samples'] == job['samples'] and summary['manifest'] == pm,
            'Incomplete population')
    expected = dict(schema=POPULATION_SCHEMA, samples=job['samples'], seed=job['seed'], cloud_replicates=2,
        activity=manifest['physical_activity'], lambda_ratio=64., guide_schema=GUIDE_SCHEMA, proposal_kind=PROPOSAL_KIND,
        importance_uniform_probability=.2, importance_component_count=24, proposal_density_measure=DENSITY_MEASURE,
        physical_fixed_neighbors=read(archive/'config.json')['fixed_poses'], chart_anchor=read(archive/'region.json')['fixed_neighbor'],
        minimum_latent_radius=0., latent_radius=4., minimum_original_q=0., maximum_original_q=None,
        minimum_original_q_inclusive=True, maximum_original_q_inclusive=True)
    expected['lambda'] = manifest['physical_activity']*64.
    for field in ('region_sha256', 'shape_sha256', 'config_sha256', 'importance_guide_sha256'): expected[field] = manifest[field]
    expected.update(source_bundle_sha256=manifest['archive_sha256']['source-bundle.json'],
                    executable_sha256=manifest['archive_sha256']['latent-region-normalizer'])
    require(all(pm.get(k) == v for k, v in expected.items()), 'Population law/input identity differs')
    for name, field in [('input-config.json', 'config_sha256'), ('region.json', 'region_sha256'),
                         ('shape.json', 'shape_sha256'), ('source-bundle.json', 'source_bundle_sha256'),
                         ('importance-guide.json', 'importance_guide_sha256')]:
        require(sha(directory/'provenance'/name) == pm[field], 'Population input copy differs: '+name)
    sample_sha = sha(directory/'samples.jsonl')
    require(summary['samples_sha256'] == sample_sha, 'Saved row bytes differ')
    require(summary['estimates']['region']['draws'] == summary['estimates']['hard_region']['draws'] == job['samples'],
            'Unconditional denominator changed')
    return dict(samples_sha256=sample_sha, summary_sha256=sha(directory/'summary.json'),
                manifest_sha256=sha(directory/'manifest.json'), sampler_cpu_seconds=summary['sampler_cpu_seconds'])


def verify_assessment(out, manifest, analysis, jobs):
    total = sum(j['samples'] for j in jobs)
    require(analysis['region_sha256'] == manifest['region_sha256'], 'Audit region differs')
    require(analysis['estimate']['draws'] == analysis['hard_region']['draws'] == total
            == analysis['independently_reconstructed_poses'], 'Audit dropped unconditional draws')
    require(analysis['physical_fixed_neighbors'] == read(Path(out)/'provenance/config.json')['fixed_poses'], 'Audit scaffold differs')
    guide = analysis['importance_sampling']
    require(guide['guide_sha256'] == manifest['importance_guide_sha256'] and guide['uniform_shell_probability'] == .2
            and guide['entry_shell_component_count'] == 24 and guide['guide_schema'] == GUIDE_SCHEMA
            and guide['proposal_kind'] == PROPOSAL_KIND and guide['density_measure'] == DENSITY_MEASURE,
            'Audit guide law differs')
    require(guide['draws'] == total and set(guide['branch_counts']) == {'uniform-shell', 'entry-shell'}
            and sum(guide['branch_counts'].values()) == total, 'Audit branch denominator differs')
    populations = analysis['populations']
    require(len(populations) == len(jobs) and {p['id'] for p in populations} == {j['id'] for j in jobs}, 'Audit omitted populations')
    for job in jobs:
        record = next(p for p in populations if p['id'] == job['id'])
        require(record['seed'] == job['seed'] and record['estimate']['draws'] == record['hard_region']['draws'] == job['samples'],
                'Audit population allocation differs')
        require(record['samples_sha256'] == job['output']['samples_sha256'] == sha(Path(job['directory'])/'samples.jsonl'), 'Audit row binding differs')
        audit = record['importance_sampling_audit']
        require(audit['draws'] == job['samples'] and set(audit['branch_counts']) == {'uniform-shell', 'entry-shell'}
                and sum(audit['branch_counts'].values()) == job['samples']
                and len(audit['entry_shell_component_counts']) == 24
                and sum(audit['entry_shell_component_counts']) == audit['branch_counts']['entry-shell'], 'Audit branch/components differ')
        for name in ('samples', 'manifest', 'summary'):
            suffix = 'jsonl' if name == 'samples' else 'json'
            require(sha(Path(job['directory'])/f'{name}.{suffix}') == job['output'][name+'_sha256'], 'Output changed during audit')


def run(out):
    out = Path(out).resolve(); check(out); manifest = read(out/'manifest.json')
    state = dict(schema='entry-shell-reference-status-v1', complete=False, phase='physical', started=time.time(),
                 protocol_sha256=sha(out/'protocol.json'), jobs=[dict(j, status='pending') for j in manifest['jobs']], audit=None)
    with (out/'status.json').open('x') as stream: json.dump(state, stream, indent=2, allow_nan=False)
    def snapshot(): write(out/'status.json', state)
    try:
        execute_jobs(state['jobs'], snapshot)
        state['phase'] = 'physical_validation'; snapshot(); validate(out)
        for job in state['jobs']:
            job['output'] = verify_output(out, manifest, job)
            write(out/f"{job['id']}-status.json", dict(id=job['id'], returncode=job['returncode'], seed=job['seed'], samples=job['samples'], output=job['output']))
        state['phase'] = 'audit'; snapshot()
        argv = [sys.executable, '-B', str(out/'provenance/analyze_latent_region.py'), '--root', str(out)]
        state['audit'] = dict(command=argv, started=time.time(), returncode=None); snapshot()
        with (out/'logs/audit.log').open('xb') as stream:
            result = subprocess.run(argv, stdout=stream, stderr=subprocess.STDOUT)
        state['audit'].update(returncode=result.returncode, finished=time.time()); snapshot(); result.check_returncode()
        verify_assessment(out, manifest, read(out/'assessment/analysis.json'), state['jobs'])
        validate(out)
        state['audit']['analysis_sha256'] = sha(out/'assessment/analysis.json')
        state.update(complete=True, phase='complete', finished=time.time()); snapshot()
    except BaseException as error:
        state.update(complete=False, phase=state['phase']+'_failed', exception=repr(error), finished=time.time()); snapshot(); raise
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('freeze', 'validate', 'preflight', 'run'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--binary', type=Path); parser.add_argument('--source-bundle', type=Path)
    parser.add_argument('--package', type=Path)
    args = parser.parse_args()
    if args.action == 'freeze':
        if any(v is None for v in (args.binary, args.source_bundle, args.package)):
            parser.error('freeze requires --binary, --source-bundle and --package')
        result = freeze(args.out, args.binary, args.source_bundle, args.package)
    else:
        result = {'validate': validate, 'preflight': check, 'run': run}[args.action](args.out)
    print(json.dumps(dict(action=args.action, out=str(args.out.resolve()), complete=result.get('complete'),
                          physical_launches=0 if args.action != 'run' else len(result['jobs'])), indent=2))


if __name__ == '__main__': main()
