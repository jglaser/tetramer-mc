#!/usr/bin/env python3
"""Freeze and execute the predeclared 786432-draw conditional-ray comparison.

Every stage is a separate estimator. The uniform arm uses alpha=1, whose
physical stream is tested bitwise against the original uniform sampler.
"""
from __future__ import annotations
import argparse
import copy
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from conditional_ray_proposal import ConditionalRayGuide, SCHEMA as GUIDE_SCHEMA, POPULATION_SCHEMA, PROPOSAL_KIND
from run_entry_shell_reference_campaign import read, write, sha, require, inside, file_hashes, validate_package
from run_mobile_posterior_pilot import verify_bundle
from prepare_shoulder_docking_benchmark import local_dependencies

ROOT = Path(__file__).resolve().parents[1]
RUNNER_NAME = 'run_conditional_ray_campaign.py'
SCHEMA = 'conditional-ray-reference-controller-v1'
ARMS = [
    dict(id='pilot_uniform', samples=16384, alpha=1., lambda_ratio=64.),
    dict(id='pilot_ray', samples=16384, alpha=.5, lambda_ratio=64.),
    dict(id='large_uniform', samples=65536, alpha=1., lambda_ratio=64.),
    dict(id='large_ray', samples=65536, alpha=.5, lambda_ratio=64.),
    dict(id='alpha02', samples=16384, alpha=.2, lambda_ratio=64.),
    dict(id='lambda128', samples=16384, alpha=.5, lambda_ratio=128.),
]
DENSITY_MEASURE = 'Lebesgue measure in the original six-dimensional whitened region chart'
TOTAL = 786432
SEEDS = [131101010+1009*i for i in range(24)]


def make_guide(package, alpha):
    plan, _ = validate_package(package)
    source, construction = read(package/'guide.json'), read(package/'construction.json')
    interfaces = []
    for anchor, motif in [(0, 7), (1, 4)]:
        entries = []
        for member in range(4):
            matches = [e for e in construction['entries'] if e['anchor_index'] == anchor and
                       e['motif_id'] == motif and e['moving_member_index'] == member and e['width_A'] == .02]
            require(len(matches) == 1, 'Member correspondence changed')
            entries.append(source['entries'][matches[0]['entry']])
        interfaces.append(dict(moving_members=[e['moving_member'] for e in entries],
                               target_world_members=[e['target_world_member'] for e in entries]))
    guide = dict(schema=GUIDE_SCHEMA, region_sha256=plan['region_sha256'],
                 defensive_uniform_shell_probability=alpha, inner_radius=2., widths=[.02, .1, .5], interfaces=interfaces)
    ConditionalRayGuide(guide, read(package/'region.json'), plan['region_sha256'])
    return guide


def command(root, arm, job):
    archive = root/arm['id']/'provenance'
    return [str(root/'common/latent-region-normalizer'), '--config', str(archive/'config.json'),
            '--region', str(archive/'region.json'), '--importance-guide', str(archive/'importance-guide.json'),
            '--out', job['directory'], '--samples', str(job['samples']), '--seed', str(job['seed']),
            '--cloud-replicates', '2', '--lambda-ratio', str(arm['lambda_ratio'])]


def freeze(out, binary, bundle_path, package):
    out, binary, bundle_path, package = map(lambda p: Path(p).resolve(), (out, binary, bundle_path, package))
    require(not out.exists(), 'Fresh campaign directory required')
    package_plan, geometry = validate_package(package)
    bundle, rust = verify_bundle(binary, bundle_path, ROOT)
    python = local_dependencies([Path(__file__), ROOT/'tools/analyze_latent_region.py'])
    common = out/'common'; common.mkdir(parents=True)
    shutil.copy2(binary, common/'latent-region-normalizer'); shutil.copy2(bundle_path, common/'source-bundle.json')
    shutil.copy2(__file__, common/'controller.py')
    for name, path in python.items(): shutil.copy2(path, common/name)
    for name, entry in bundle['files'].items():
        path = inside(common/'source', name); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(entry['text'].encode())
    frozen = read(package/'freeze.json')['files']
    for name in [*frozen, 'freeze.json']:
        dest = inside(common/'reference-package', name); dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(inside(package, name), dest)
    write(common/'geometry-preflight.json', geometry)
    jobs = []; arms = []
    for arm_index, spec in enumerate(ARMS):
        arm = dict(spec); base = out/arm['id']; archive = base/'provenance'; archive.mkdir(parents=True)
        for folder in ('runs', 'logs'): (base/folder).mkdir()
        for name, path in python.items(): shutil.copy2(path, archive/name)
        for name, path in [('source-bundle.json', bundle_path), ('latent-region-normalizer', binary),
                           ('source-config.json', package/'config.json'), ('shape.json', package/'inputs/tetramer-shape.json'),
                           ('region.json', package/'region.json')]: shutil.copy2(path, archive/name)
        config = read(package/'config.json'); config['shape'] = str(archive/'shape.json')
        write(archive/'config.json', config); write(archive/'importance-guide.json', make_guide(package, arm['alpha']))
        arm_jobs = []
        for index in range(4):
            job = dict(id=f'r{index:02d}', arm=arm['id'], seed=SEEDS[4*arm_index+index], samples=arm['samples'],
                       directory=str(base/'runs'/f'r{index:02d}'), log=str(base/'logs'/f'r{index:02d}.log'))
            job['command'] = command(out, arm, job); arm_jobs.append(job); jobs.append(job)
        manifest = dict(schema='importance-latent-region-campaign-v3', jobs=arm_jobs, workers=4,
            physical_activity=.035, lambda_ratio=arm['lambda_ratio'], cloud_replicates=2,
            config_sha256=sha(archive/'config.json'), shape_sha256=sha(archive/'shape.json'),
            region_sha256=sha(archive/'region.json'), importance_guide_sha256=sha(archive/'importance-guide.json'),
            archive_sha256=file_hashes(archive), allocation=arm)
        write(base/'manifest.json', manifest); arm['manifest_sha256'] = sha(base/'manifest.json'); arms.append(arm)
    require(sum(j['samples'] for j in jobs) == TOTAL, 'Wrong total allocation')
    protocol = dict(schema=SCHEMA, arms=arms, jobs=jobs, total_unconditional_draws=TOTAL, maximum_physical_workers=8,
        controller_sha256=sha(__file__), binary_sha256=sha(binary), source_bundle_sha256=sha(bundle_path),
        rust_sources=rust, python_sources={k: sha(p) for k, p in python.items()},
        native_definition='common/reference-package/'+package_plan['native_definition'],
        native_definition_sha256=package_plan['native_definition_sha256'],
        region_sha256=package_plan['region_sha256'], shape_sha256=package_plan['shape_sha256'],
        primary_classes=['registered_native_entry', 'contact_no_native_entry', 'unbound_no_native_entry'],
        strata=dict(radial_edges=[0., 2., 3., 4.], angular_projection_squared_edges=[0., 4., 9., 16.],
                    latent_orthants='six signs, zero assigned positive; all 64 bins retained'),
        convergence=dict(population_relative_SE_max=.1, importance_ESS_min=200., largest_draw_max=.02,
            log_agreement_absolute_max=.2, log_agreement_combined_SE_max=3., deltaF_95_halfwidth_max=.5,
            significant_stratum_mass_fraction=.01, stratum_log_agreement_absolute_max=.2),
        scope='Frozen finite R4 target. Six separate fresh estimates; no pooling, retries, adaptive allocation or verdict from sampler failure. Full-wall and assembly production require passed downstream convergence gates.')
    write(out/'protocol.json', protocol); write(out/'freeze.json', dict(files=file_hashes(out)))
    validate(out)
    return protocol


def validate(out):
    out = Path(out).resolve(); protocol = read(out/'protocol.json'); common = out/'common'
    require(protocol['schema'] == SCHEMA and protocol['total_unconditional_draws'] == TOTAL, 'Wrong campaign')
    for name, digest in read(out/'freeze.json')['files'].items():
        require(sha(inside(out, name)) == digest, 'Frozen file changed: '+name)
    require(sha(common/'controller.py') == protocol['controller_sha256'], 'Controller changed')
    require(sha(common/'latent-region-normalizer') == protocol['binary_sha256'] and
            sha(common/'source-bundle.json') == protocol['source_bundle_sha256'], 'Executable/source changed')
    _, rust = verify_bundle(common/'latent-region-normalizer', common/'source-bundle.json', common/'source')
    require(rust == protocol['rust_sources'], 'Rust closure changed')
    closure = local_dependencies([common/RUNNER_NAME, common/'analyze_latent_region.py'])
    require(set(closure) == set(protocol['python_sources']), 'Python closure changed')
    for name, digest in protocol['python_sources'].items(): require(sha(common/name) == digest, 'Python source changed')
    package = common/'reference-package'; package_plan, _ = validate_package(package)
    require(sha(inside(out, protocol['native_definition'])) == protocol['native_definition_sha256']
            == package_plan['native_definition_sha256'], 'Native definition changed')
    require(protocol['region_sha256'] == package_plan['region_sha256'] and
            protocol['shape_sha256'] == package_plan['shape_sha256'], 'Physical target changed')
    all_jobs = []
    for index, (arm, expected) in enumerate(zip(protocol['arms'], ARMS)):
        require({k: arm[k] for k in expected} == expected, 'Arm allocation changed')
        base = out/arm['id']; archive = base/'provenance'; manifest = read(base/'manifest.json')
        require(sha(base/'manifest.json') == arm['manifest_sha256'], 'Arm manifest changed')
        require(manifest['schema'] == 'importance-latent-region-campaign-v3' and len(manifest['jobs']) == 4, 'Arm schema/count changed')
        require(read(archive/'importance-guide.json') == make_guide(package, arm['alpha']), 'Guide construction changed')
        config = read(archive/'config.json'); original = read(package/'config.json')
        restored = copy.deepcopy(config); restored['shape'] = original['shape']
        require(restored == original and config['shape'] == str(archive/'shape.json'), 'Physical config changed')
        for i, job in enumerate(manifest['jobs']):
            require(job['seed'] == SEEDS[4*index+i] and job['samples'] == arm['samples'] and job['arm'] == arm['id']
                    and job['id'] == f'r{i:02d}' and job['command'] == command(out, arm, job), 'Job law changed')
            require(job['directory'] == str(base/'runs'/job['id']) and job['log'] == str(base/'logs'/f"{job['id']}.log"), 'Job path escaped')
            all_jobs.append(job)
    require(len(protocol['arms']) == 6 and all_jobs == protocol['jobs'] and len(all_jobs) == 24, 'Incomplete allocation')
    return protocol


def preflight(out):
    out = Path(out).resolve(); protocol = validate(out)
    require(sha(__file__) == protocol['controller_sha256'], 'Use exact archived controller')
    require(not (out/'status.json').exists(), 'Campaign already launched; no retries')
    for arm in protocol['arms']:
        base = out/arm['id']
        require(not any((base/'runs').iterdir()) and not any((base/'logs').iterdir()) and
                not (base/'assessment').exists(), 'Existing outputs; no overwrite')
    return protocol


def execute_jobs(jobs, snapshot, workers=8, popen=subprocess.Popen, pause=time.sleep):
    require(type(workers) is int and 1 <= workers <= 8, 'At most eight physical workers')
    active, cursor, error = [], 0, None
    try:
        while cursor < len(jobs) or active:
            while error is None and cursor < len(jobs) and len(active) < workers:
                job = jobs[cursor]; cursor += 1
                require(not Path(job['directory']).exists(), 'Existing population output')
                with Path(job['log']).open('xb') as stream:
                    child = popen(job['command'], stdout=stream, stderr=subprocess.STDOUT)
                job.update(status='running', pid=child.pid, started=time.time()); active.append((job, child)); snapshot()
            progressed = False
            for job, child in list(active):
                code = child.poll()
                if code is None: continue
                child.wait(); active.remove((job, child)); progressed = True
                job.update(status='complete' if code == 0 else 'failed', returncode=code, finished=time.time()); snapshot()
                if code != 0: error = RuntimeError('Physical failure; drain children, do not retry')
            if error is not None: break
            if not progressed and active: pause(.1)
    except BaseException as exc:
        error = exc
    finally:
        for job, child in active:
            while True:
                try: code = child.wait(); break
                except KeyboardInterrupt as exc:
                    if error is None: error = exc
            job.update(status='complete' if code == 0 else 'failed', returncode=code, finished=time.time())
        for job in jobs:
            if job['status'] == 'pending': job['status'] = 'not_started'
        snapshot()
    if error is not None: raise error


def verify_output(out, protocol, job):
    arm = next(a for a in protocol['arms'] if a['id'] == job['arm'])
    root = Path(out)/arm['id']; manifest = read(root/'manifest.json')
    directory = Path(job['directory']); summary = read(directory/'summary.json'); pm = read(directory/'manifest.json')
    require(summary['complete'] is True and summary['manifest'] == pm and summary['samples'] == job['samples'], 'Incomplete physical population')
    expected = dict(schema=POPULATION_SCHEMA, samples=job['samples'], seed=job['seed'], cloud_replicates=2,
        activity=.035, lambda_ratio=arm['lambda_ratio'], guide_schema=GUIDE_SCHEMA, proposal_kind=PROPOSAL_KIND,
        importance_uniform_probability=arm['alpha'], importance_component_count=3, proposal_density_measure=DENSITY_MEASURE,
        executable_sha256=protocol['binary_sha256'], source_bundle_sha256=protocol['source_bundle_sha256'],
        minimum_latent_radius=0., latent_radius=4., minimum_original_q=0., maximum_original_q=None,
        minimum_original_q_inclusive=True, maximum_original_q_inclusive=True,
        physical_fixed_neighbors=read(root/'provenance/config.json')['fixed_poses'],
        chart_anchor=read(root/'provenance/region.json')['fixed_neighbor'])
    expected['lambda'] = .035*arm['lambda_ratio']
    for key in ('region_sha256', 'shape_sha256', 'config_sha256', 'importance_guide_sha256'): expected[key] = manifest[key]
    require(all(pm.get(k) == v for k, v in expected.items()), 'Population law changed')
    require(summary['estimates']['region']['draws'] == summary['estimates']['hard_region']['draws'] == job['samples'], 'Unconditional denominator changed')
    require(summary['shell_rejected'] == 0, 'Ray proposal left R4')
    require(summary['samples_sha256'] == sha(directory/'samples.jsonl'), 'Rows changed')
    for name, field in [('input-config.json', 'config_sha256'), ('region.json', 'region_sha256'), ('shape.json', 'shape_sha256'),
                        ('importance-guide.json', 'importance_guide_sha256'), ('source-bundle.json', 'source_bundle_sha256')]:
        require(sha(directory/'provenance'/name) == pm[field], 'Population provenance changed')
    return dict(samples_sha256=sha(directory/'samples.jsonl'), summary_sha256=sha(directory/'summary.json'),
                manifest_sha256=sha(directory/'manifest.json'), sampler_cpu_seconds=summary['sampler_cpu_seconds'])


def verify_assessment(out, protocol, arm, analysis, jobs):
    base = Path(out)/arm['id']; manifest = read(base/'manifest.json')
    selected = [j for j in jobs if j['arm'] == arm['id']]
    require(len(selected) == 4 and len({j['id'] for j in selected}) == 4, 'Wrong audit allocation')
    total = 4*arm['samples']
    require(analysis['region_sha256'] == protocol['region_sha256'] == manifest['region_sha256']
            and analysis['physical_fixed_neighbors'] == read(base/'provenance/config.json')['fixed_poses'], 'Audit physical target changed')
    require(analysis['estimate']['draws'] == analysis['hard_region']['draws'] == total
            == analysis['independently_reconstructed_poses'], 'Audit dropped unconditional draws')
    guide = analysis['importance_sampling']
    require(guide['guide_sha256'] == manifest['importance_guide_sha256'] and
            guide['conditional_ray_component_count'] == 3 and guide['uniform_shell_probability'] == arm['alpha']
            and guide['guide_schema'] == GUIDE_SCHEMA and guide['proposal_kind'] == PROPOSAL_KIND
            and guide['density_measure'] == DENSITY_MEASURE and guide['shell_rejected'] == 0
            and guide['draws'] == total, 'Audit guide identity differs')
    require(set(guide['branch_counts']) == {'uniform-shell', 'conditional-ray'} and
            all(type(v) is int and v >= 0 for v in guide['branch_counts'].values()) and
            sum(guide['branch_counts'].values()) == total, 'Audit branch denominator differs')
    if arm['alpha'] == 1.: require(guide['branch_counts']['conditional-ray'] == 0, 'Uniform reference gained guided draws')
    populations = analysis['populations']
    require(len(populations) == 4 and {p['id'] for p in populations} == {j['id'] for j in selected}, 'Audit population missing/repeated')
    aggregate_fallbacks = [0, 0, 0]
    aggregate_branches = {'uniform-shell': 0, 'conditional-ray': 0}
    for job in selected:
        population = next(p for p in populations if p['id'] == job['id'])
        require(population['seed'] == job['seed'] and population['estimate']['draws'] == population['hard_region']['draws'] == job['samples']
                and population['samples_sha256'] == job['output']['samples_sha256'], 'Audit population binding changed')
        audit = population['importance_sampling_audit']
        require(audit['draws'] == job['samples'] and audit['shell_rejected'] == 0 and
                set(audit['branch_counts']) == {'uniform-shell', 'conditional-ray'} and
                all(type(v) is int and v >= 0 for v in audit['branch_counts'].values()) and
                sum(audit['branch_counts'].values()) == job['samples'] and len(audit['ray_component_counts']) == 3 and
                all(type(v) is int and v >= 0 for v in audit['ray_component_counts']) and
                sum(audit['ray_component_counts']) == audit['branch_counts']['conditional-ray'], 'Audit component denominator changed')
        require(len(audit['ray_fallback_counts']) == 3 and all(type(f) is int and 0 <= f <= c for f, c in
                zip(audit['ray_fallback_counts'], audit['ray_component_counts'])), 'Audit fallback counts invalid')
        aggregate_fallbacks = [a+b for a,b in zip(aggregate_fallbacks, audit['ray_fallback_counts'])]
        aggregate_branches = {k: v + audit['branch_counts'][k] for k, v in aggregate_branches.items()}
        for name in ('samples', 'manifest', 'summary'):
            extension = 'jsonl' if name == 'samples' else 'json'
            require(sha(Path(job['directory'])/f'{name}.{extension}') == job['output'][name+'_sha256'], 'Output changed during audit')
    require(guide['branch_counts'] == aggregate_branches, 'Aggregate branch counts differ')
    require(all(type(v) is int and v >= 0 for v in guide['ray_fallback_counts']) and
            guide['ray_fallback_counts'] == aggregate_fallbacks, 'Aggregate fallback counts differ')


def run(out):
    out = Path(out).resolve(); protocol = preflight(out)
    state = dict(schema='conditional-ray-reference-status-v1', complete=False, phase='physical', started=time.time(),
                 protocol_sha256=sha(out/'protocol.json'), jobs=[dict(j, status='pending') for j in protocol['jobs']], audits={})
    with (out/'status.json').open('x') as stream: stream.write('{}\n')
    def snapshot(): write(out/'status.json', state)
    snapshot()
    try:
        execute_jobs(state['jobs'], snapshot)
        state['phase'] = 'physical_validation'; snapshot(); validate(out)
        for job in state['jobs']: job['output'] = verify_output(out, protocol, job)
        state['phase'] = 'audit'; snapshot()
        for arm in protocol['arms']:
            base = out/arm['id']; argv = [sys.executable, '-B', str(base/'provenance/analyze_latent_region.py'), '--root', str(base)]
            audit = state['audits'][arm['id']] = dict(command=argv, started=time.time(), returncode=None); snapshot()
            with (base/'logs/audit.log').open('xb') as stream: result = subprocess.run(argv, stdout=stream, stderr=subprocess.STDOUT)
            audit.update(returncode=result.returncode, finished=time.time()); snapshot(); result.check_returncode()
            verify_assessment(out, protocol, arm, read(base/'assessment/analysis.json'), state['jobs'])
            audit['analysis_sha256'] = sha(base/'assessment/analysis.json'); snapshot()
        validate(out); state.update(complete=True, phase='complete', finished=time.time()); snapshot()
    except BaseException as error:
        state.update(phase=state['phase']+'_failed', exception=repr(error), finished=time.time()); snapshot(); raise
    return state


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('action', choices=['freeze', 'validate', 'preflight', 'run'])
    p.add_argument('--out', type=Path, required=True); p.add_argument('--binary', type=Path)
    p.add_argument('--source-bundle', type=Path); p.add_argument('--package', type=Path)
    args = p.parse_args()
    if args.action == 'freeze':
        if any(x is None for x in (args.binary, args.source_bundle, args.package)): p.error('freeze requires binary, source-bundle and package')
        result = freeze(args.out, args.binary, args.source_bundle, args.package)
    else: result = dict(validate=validate, preflight=preflight, run=run)[args.action](args.out)
    print(dict(action=args.action, out=str(args.out.resolve()), complete=result.get('complete')))
