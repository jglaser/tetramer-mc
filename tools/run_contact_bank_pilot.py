#!/usr/bin/env python3
"""Freeze and run an independent eight-population full-6D contact-bank pilot.

This controller never fits a guide. It retains 48 source-population/class anchors
and eight R5-reference native anchors at their explicitly frozen group weights. Both arms integrate the same frozen R4;
untruncated Gaussian draws outside it remain zeros in the attempted denominator.
The pilot is not an assembly or whole-vessel thermodynamic calculation.
"""
from __future__ import annotations

import argparse
import copy
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import scipy

from analyze_latent_region import LatentImportanceGuide
from analyze_mobile_native_pocket import load_classifier
from analyze_mobile_threshold_reference import validate_classifier_target
from prepare_shoulder_docking_benchmark import local_dependencies
from run_conditional_ray_campaign import execute_jobs
from run_entry_shell_reference_campaign import read, write, sha, require, inside, file_hashes
from run_latent_reference_campaign import validate_region
from run_mobile_posterior_pilot import verify_bundle

ROOT = Path(__file__).resolve().parents[1]
RUNNER_NAME = 'run_contact_bank_pilot.py'
SCHEMA = 'contact-bank-pilot-controller-v2'
PACKAGE_SCHEMA = 'reference-contact-bank-preparation-v1'
GUIDE_SCHEMA = 'defensive-latent-shell-guide-v1'
CAMPAIGN_SCHEMA = 'importance-latent-region-campaign-v1'
POPULATION_SCHEMA = 'importance-latent-region-normalizer-v1'
DENSITY_MEASURE = 'Lebesgue measure in the original six-dimensional whitened region chart'
ARMS = [dict(id='bank', covariance_factor=1., samples=16384, alpha=.5, lambda_ratio=128., component_count=56),
        dict(id='wide', covariance_factor=4., samples=16384, alpha=.5, lambda_ratio=128., component_count=56)]
SEEDS = [132101010+1009*i for i in range(8)]
TOTAL = 131072
COMPONENTS = 56
PINS = {
    'region.json': '924648f4d9db473045397c300b3b4af7ccde8cfda1b1703a6899239ec12e2f02',
    'shape.json': 'c7442034e7ffa4627b67c57a81117f0e9bc2d389ecf956a83dac2a5e915554c9',
    'config.json': '340050d107c4e862f62cde293630daf5ae70aecfb5264bd5d8ac0676a589078e',
    'native-definition.json': '5cf55ebe0c0f78ec6b8cdc745cac938b0fadfa3732e1e1c730864c62c651d7a9',
}
COMPONENT_GROUPS = [
    dict(source_group='conditional-ray', class_name='native', count=24, component_weight=1/96, full_probability=.125),
    dict(source_group='conditional-ray', class_name='no-entry', count=24, component_weight=1/48, full_probability=.25),
    dict(source_group='R5-reference', class_name='native', count=8, component_weight=1/32, full_probability=.125),
]
THREAD_ENV = {name: '1' for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                                   'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'RAYON_NUM_THREADS')}


def runtime():
    return dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__)


def worker_environment():
    # The independent auditor deliberately uses assertions for row-level checks.
    return dict(os.environ, PYTHONOPTIMIZE='0', **THREAD_ENV)


def nonnegative_count(value):
    return type(value) is int and value >= 0


def validate_package(package):
    require(sys.flags.optimize == 0, 'Correctness audits require enabled Python assertions')
    package = Path(package).resolve()
    frozen = read(package/'freeze.json')['files']
    required = {'plan.json', 'config.json', 'region.json', 'shape.json', 'native-definition.json',
                'guide-bank.json', 'guide-wide.json', 'native-region/definition.json'}
    require(required <= set(frozen) and 'freeze.json' not in frozen, 'Incomplete preparation freeze')
    for name, digest in frozen.items():
        require(sha(inside(package, name)) == digest, 'Preparation dependency changed: '+name)
    for name, digest in PINS.items():
        require(sha(package/name) == digest, 'Frozen physical target changed: '+name)
    plan = read(package/'plan.json')
    require(plan['schema'] == PACKAGE_SCHEMA and plan['complete'] is True and plan['production_launched'] is False,
            'Require an inert completed guide preparation')
    for name in ('config', 'region', 'shape'):
        require(plan[name+'_sha256'] == sha(package/(name+'.json')), 'Preparation input binding differs: '+name)
    require(plan['native_definition'] == 'native-region/definition.json' and
            plan['native_definition_sha256'] == sha(package/'native-definition.json') == sha(package/'native-region/definition.json'),
            'Native definition changed')
    cfg, region = read(package/'config.json'), read(package/'region.json')
    require(cfg['depletant_radius'] == 1.5 and cfg['reservoir_density'] == .035 and
            cfg['poisson_lambda_ratio'] == 64. and cfg['capture_radius'] == 170. and len(cfg['fixed_poses']) == 2,
            'Wrong physical scaffold/bath/capture')
    require(region.get('minimum_mahalanobis_radius', 0.) == 0. and region['mahalanobis_radius'] == 4.
            and region['minimum_original_q'] == 0. and region.get('minimum_original_q_inclusive', True) is True
            and 'maximum_original_q' not in region, 'R4 target gained a q/native filter')
    classifier, classifier_binding = load_classifier(package/'native-region/definition.json')
    validate_classifier_target(cfg, plan['shape_sha256'], classifier.definition, package/'native-region/definition.json')
    geometry = validate_region(region, cfg, read(package/'shape.json'), plan['shape_sha256'])
    geometry['native_classifier_binding'] = classifier_binding
    require(geometry['chart_anchor_index'] == 1 and geometry['entire_chart_inside_capture'], 'Wrong R4 chart/capture')
    guides = {}
    for arm in ARMS:
        name = arm['id']; path = package/f'guide-{name}.json'; guide = read(path)
        require(plan['guide_sha256'][name] == sha(path), 'Prepared guide binding changed')
        parsed = LatentImportanceGuide(guide, plan['region_sha256'])
        require(parsed.count == COMPONENTS and parsed.alpha == .5, 'Wrong defensive guide allocation')
        guides[name] = guide
    for base, wide in zip(guides['bank']['gaussian_components'], guides['wide']['gaussian_components']):
        require(base['mean'] == wide['mean'] and base['weight'] == wide['weight'] and
                np.array_equal(np.asarray(wide['covariance']), 4*np.asarray(base['covariance'])),
                'Wide arm must only multiply the full covariance by four')
    require(plan['covariance_multipliers'] == {'bank': 1, 'wide': 4}, 'Covariance construction changed')
    anchors = plan['anchor_metadata']
    require(len(anchors) == COMPONENTS and [a['component_index'] for a in anchors] == list(range(COMPONENTS)),
            'Anchor slots changed')
    require(all(a['source_group'] == ('conditional-ray' if i < 48 else 'R5-reference') for i, a in enumerate(anchors)),
            'Mandatory source-group slots changed')
    for group in COMPONENT_GROUPS:
        indices = [i for i, a in enumerate(anchors) if (a['source_group'], a['class']) == (group['source_group'], group['class_name'])]
        require(len(indices) == group['count'], 'Anchor class allocation changed')
        require(all(guides['bank']['gaussian_components'][i]['weight'] == group['component_weight'] for i in indices),
                'Source-group component weights changed')
    original_populations = {(a['arm'], a['population_id']) for a in anchors[:48]}
    require(len(original_populations) == 24 and all(
        {a['class'] for a in anchors[:48] if (a['arm'], a['population_id']) == population} == {'native', 'no-entry'}
        for population in original_populations), 'Original source populations lost a class slot')
    require(all(a['seed'] not in SEEDS for a in anchors), 'Training and pilot seeds overlap')
    require(len({(a['source_group'], a['arm'], a['population_id'], a['class']) for a in anchors}) == COMPONENTS,
            'Repeated source-population class slot')
    return plan, geometry


def command(root, arm, job):
    archive = root/arm['id']/'provenance'
    return [str(root/'common/latent-region-normalizer'), '--config', str(archive/'config.json'),
            '--region', str(archive/'region.json'), '--importance-guide', str(archive/'importance-guide.json'),
            '--out', job['directory'], '--samples', str(job['samples']), '--seed', str(job['seed']),
            '--cloud-replicates', '2', '--lambda-ratio', '128.0']


def freeze(out, binary, bundle_path, package):
    out, binary, bundle_path, package = map(lambda p: Path(p).resolve(), (out, binary, bundle_path, package))
    require(not out.exists(), 'Fresh campaign directory required; no overwrite')
    plan, geometry = validate_package(package)
    bundle, rust = verify_bundle(binary, bundle_path, ROOT)
    python = local_dependencies([Path(__file__), ROOT/'tools/analyze_latent_region.py', ROOT/'tools/analyze_contact_bank_pilot.py'])
    out.mkdir(); common = out/'common'; common.mkdir()
    shutil.copy2(binary, common/'latent-region-normalizer'); shutil.copy2(bundle_path, common/'source-bundle.json')
    shutil.copy2(__file__, common/'controller.py')
    for name, path in python.items(): shutil.copy2(path, common/name)
    for name, entry in bundle['files'].items():
        path = inside(common/'source', name); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(entry['text'].encode())
    for name in [*read(package/'freeze.json')['files'], 'freeze.json']:
        dest = inside(common/'reference-package', name); dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(inside(package, name), dest)
    write(common/'geometry-preflight.json', geometry)
    jobs, arms = [], []
    for arm_index, spec in enumerate(ARMS):
        arm = dict(spec); base = out/arm['id']; archive = base/'provenance'; archive.mkdir(parents=True)
        for folder in ('runs', 'logs'): (base/folder).mkdir()
        for name, path in python.items(): shutil.copy2(path, archive/name)
        for name, path in [('source-bundle.json', bundle_path), ('latent-region-normalizer', binary),
                           ('source-config.json', package/'config.json'), ('shape.json', package/'shape.json'),
                           ('region.json', package/'region.json'), ('importance-guide.json', package/f"guide-{arm['id']}.json")]:
            shutil.copy2(path, archive/name)
        config = read(package/'config.json'); config['shape'] = str(archive/'shape.json'); write(archive/'config.json', config)
        arm_jobs = []
        for i in range(4):
            job = dict(id=f'r{i:02d}', arm=arm['id'], seed=SEEDS[4*arm_index+i], samples=arm['samples'],
                       directory=str(base/'runs'/f'r{i:02d}'), log=str(base/'logs'/f'r{i:02d}.log'))
            job['command'] = command(out, arm, job); arm_jobs.append(job); jobs.append(job)
        manifest = dict(schema=CAMPAIGN_SCHEMA, jobs=arm_jobs, workers=4, physical_activity=.035,
            lambda_ratio=128., cloud_replicates=2, config_sha256=sha(archive/'config.json'),
            shape_sha256=sha(archive/'shape.json'), region_sha256=sha(archive/'region.json'),
            importance_guide_sha256=sha(archive/'importance-guide.json'), archive_sha256=file_hashes(archive), allocation=arm)
        write(base/'manifest.json', manifest); arm['manifest_sha256'] = sha(base/'manifest.json'); arms.append(arm)
    protocol = dict(schema=SCHEMA, arms=arms, jobs=jobs, total_unconditional_draws=TOTAL, maximum_physical_workers=8,
        controller_sha256=sha(__file__), binary_sha256=sha(binary), source_bundle_sha256=sha(bundle_path),
        rust_sources=rust, python_sources={k: sha(p) for k, p in python.items()}, runtime=runtime(), thread_environment=THREAD_ENV,
        preparation_plan_sha256=sha(package/'plan.json'), preparation_freeze_sha256=sha(package/'freeze.json'),
        native_definition='common/reference-package/native-region/definition.json', native_definition_sha256=plan['native_definition_sha256'],
        region_sha256=plan['region_sha256'], shape_sha256=plan['shape_sha256'], component_count=COMPONENTS,
        component_groups=COMPONENT_GROUPS,
        primary_classes=['registered_native_entry', 'contact_no_native_entry', 'unbound_no_native_entry'],
        estimator='Unconditional mean of H_R4 H_capture H_hard J W/q; two independent Poisson clouds per valid pose. Full untruncated defensive mixture q, exterior and hard-invalid zeros retained.',
        strata=dict(radial_edges=[0., 2., 3., 4.], angular_projection_squared_edges=[0., 4., 9., 16.],
                    latent_orthants='six signs, zero assigned positive; all 64 bins retained'),
        convergence=dict(population_relative_SE_max=.1, importance_ESS_min=200., largest_draw_max=.02,
            log_agreement_absolute_max=.2, log_agreement_combined_SE_max=3., deltaF_95_halfwidth_max=.5,
            significant_stratum_mass_fraction=.01, stratum_log_agreement_absolute_max=.2),
        scope='Independent pilot only. Each arm is analyzed separately; no retries, adaptive stopping, pooling with training draws, whole-vessel or assembly inference. All eight physical populations complete and validate before two archived audits.')
    write(out/'protocol.json', protocol); write(out/'freeze.json', dict(files=file_hashes(out)))
    validate(out)
    return protocol


def validate(out):
    out = Path(out).resolve(); protocol = read(out/'protocol.json'); common = out/'common'
    require(protocol['schema'] == SCHEMA and protocol['total_unconditional_draws'] == TOTAL and
            protocol['maximum_physical_workers'] == 8 and protocol['component_count'] == COMPONENTS
            and protocol['component_groups'] == COMPONENT_GROUPS,
            'Wrong pilot allocation')
    frozen = read(out/'freeze.json')['files']
    require('protocol.json' in frozen and 'common/controller.py' in frozen, 'Incomplete campaign freeze')
    for name, digest in frozen.items(): require(sha(inside(out, name)) == digest, 'Frozen file changed: '+name)
    require(sha(common/'controller.py') == protocol['controller_sha256'], 'Controller changed')
    require(sha(common/'latent-region-normalizer') == protocol['binary_sha256'] and
            sha(common/'source-bundle.json') == protocol['source_bundle_sha256'], 'Executable/source changed')
    _, rust = verify_bundle(common/'latent-region-normalizer', common/'source-bundle.json', common/'source')
    require(rust == protocol['rust_sources'], 'Rust closure changed')
    closure = local_dependencies([common/RUNNER_NAME, common/'analyze_latent_region.py', common/'analyze_contact_bank_pilot.py'])
    require(set(closure) == set(protocol['python_sources']), 'Python closure changed')
    for name, digest in protocol['python_sources'].items(): require(sha(common/name) == digest, 'Python source changed')
    require(protocol['thread_environment'] == THREAD_ENV, 'Worker limits changed')
    package = common/'reference-package'; plan, _ = validate_package(package)
    require(sha(package/'plan.json') == protocol['preparation_plan_sha256'] and
            sha(package/'freeze.json') == protocol['preparation_freeze_sha256'], 'Preparation binding changed')
    require(protocol['native_definition'] == 'common/reference-package/native-region/definition.json' and
            sha(inside(out, protocol['native_definition'])) == protocol['native_definition_sha256'] == plan['native_definition_sha256'],
            'Native definition changed')
    require(protocol['region_sha256'] == plan['region_sha256'] and protocol['shape_sha256'] == plan['shape_sha256'], 'Physical target changed')
    all_jobs = []
    require(len(protocol['arms']) == 2, 'Incomplete arm allocation')
    for index, (arm, expected) in enumerate(zip(protocol['arms'], ARMS)):
        require({k: arm[k] for k in expected} == expected, 'Arm allocation changed')
        base = out/arm['id']; archive = base/'provenance'; manifest = read(base/'manifest.json')
        require(sha(base/'manifest.json') == arm['manifest_sha256'], 'Arm manifest changed')
        require(manifest['schema'] == CAMPAIGN_SCHEMA and len(manifest['jobs']) == 4 and manifest['workers'] == 4
                and manifest['physical_activity'] == .035 and manifest['lambda_ratio'] == 128. and manifest['cloud_replicates'] == 2,
                'Arm law/schema changed')
        require(manifest['archive_sha256'] == file_hashes(archive), 'Arm source/input closure changed')
        for name in ('config', 'region', 'shape'):
            require(manifest[name+'_sha256'] == sha(archive/(name+'.json')), 'Arm input hash changed')
        require(sha(archive/'importance-guide.json') == manifest['importance_guide_sha256'] == plan['guide_sha256'][arm['id']], 'Guide changed')
        for name in ('region.json', 'shape.json'):
            require(sha(archive/name) == sha(package/name), 'Arm physical input changed')
        require(sha(archive/'source-config.json') == sha(package/'config.json'), 'Physical source config changed')
        config = read(archive/'config.json'); original = read(package/'config.json')
        restored = copy.deepcopy(config); restored['shape'] = original['shape']
        require(restored == original and config['shape'] == str(archive/'shape.json'), 'Physical config changed')
        for i, job in enumerate(manifest['jobs']):
            require(job['seed'] == SEEDS[4*index+i] and job['samples'] == arm['samples'] and job['arm'] == arm['id']
                    and job['id'] == f'r{i:02d}' and job['command'] == command(out, arm, job), 'Job law changed')
            require(job['directory'] == str(base/'runs'/job['id']) and job['log'] == str(base/'logs'/f"{job['id']}.log"), 'Job path escaped')
            all_jobs.append(job)
    require(all_jobs == protocol['jobs'] and len(all_jobs) == 8 and sum(j['samples'] for j in all_jobs) == TOTAL, 'Incomplete allocation')
    return protocol


def preflight(out):
    out = Path(out).resolve(); protocol = validate(out)
    require(sha(__file__) == protocol['controller_sha256'], 'Use exact archived controller')
    require(runtime() == protocol['runtime'], 'Python audit runtime changed')
    require(not (out/'status.json').exists(), 'Pilot already launched; no retries')
    for arm in protocol['arms']:
        base = out/arm['id']
        require(not any((base/'runs').iterdir()) and not any((base/'logs').iterdir()) and
                not (base/'assessment').exists(), 'Existing outputs; no overwrite')
    return protocol


def verify_output(out, protocol, job):
    root = Path(out)/job['arm']; manifest = read(root/'manifest.json')
    directory = Path(job['directory']); summary = read(directory/'summary.json'); pm = read(directory/'manifest.json')
    require(summary['complete'] is True and summary['manifest'] == pm and summary['samples'] == job['samples'], 'Incomplete physical population')
    expected = dict(schema=POPULATION_SCHEMA, samples=job['samples'], seed=job['seed'], cloud_replicates=2,
        activity=.035, lambda_ratio=128., importance_uniform_probability=.5, importance_component_count=COMPONENTS,
        proposal_density_measure=DENSITY_MEASURE, executable_sha256=protocol['binary_sha256'],
        source_bundle_sha256=protocol['source_bundle_sha256'], minimum_latent_radius=0., latent_radius=4.,
        minimum_original_q=0., maximum_original_q=None, minimum_original_q_inclusive=True, maximum_original_q_inclusive=True,
        physical_fixed_neighbors=read(root/'provenance/config.json')['fixed_poses'], chart_anchor=read(root/'provenance/region.json')['fixed_neighbor'])
    expected['lambda'] = .035*128.
    for key in ('region_sha256', 'shape_sha256', 'config_sha256', 'importance_guide_sha256'): expected[key] = manifest[key]
    require(all(pm.get(k) == v for k, v in expected.items()), 'Population law changed')
    require(summary['estimates']['region']['draws'] == summary['estimates']['hard_region']['draws'] == job['samples'], 'Unconditional denominator changed')
    require(nonnegative_count(summary['shell_rejected']) and summary['shell_rejected'] <= job['samples'], 'Invalid exterior draw count')
    require(summary['samples_sha256'] == sha(directory/'samples.jsonl'), 'Rows changed')
    require(type(summary['sampler_cpu_seconds']) in (int, float) and math.isfinite(summary['sampler_cpu_seconds'])
            and summary['sampler_cpu_seconds'] >= 0., 'Invalid sampler CPU time')
    for name, field in [('input-config.json', 'config_sha256'), ('region.json', 'region_sha256'), ('shape.json', 'shape_sha256'),
                        ('importance-guide.json', 'importance_guide_sha256'), ('source-bundle.json', 'source_bundle_sha256')]:
        require(sha(directory/'provenance'/name) == pm[field], 'Population provenance changed')
    return dict(samples_sha256=sha(directory/'samples.jsonl'), summary_sha256=sha(directory/'summary.json'),
                manifest_sha256=sha(directory/'manifest.json'), sampler_cpu_seconds=summary['sampler_cpu_seconds'],
                shell_rejected=summary['shell_rejected'])


def verify_assessment(out, protocol, arm, analysis, jobs):
    base = Path(out)/arm['id']; manifest = read(base/'manifest.json'); selected = [j for j in jobs if j['arm'] == arm['id']]
    require(len(selected) == 4 and len({j['id'] for j in selected}) == 4, 'Wrong audit allocation')
    total = 4*arm['samples']; guide = analysis['importance_sampling']
    require(analysis['region_sha256'] == protocol['region_sha256'] == manifest['region_sha256'] and
            analysis['physical_fixed_neighbors'] == read(base/'provenance/config.json')['fixed_poses'], 'Audit physical target changed')
    require(analysis['estimate']['draws'] == analysis['hard_region']['draws'] == total == analysis['independently_reconstructed_poses'],
            'Audit dropped unconditional draws')
    require(guide['guide_sha256'] == manifest['importance_guide_sha256'] and guide['gaussian_component_count'] == COMPONENTS
            and guide['uniform_shell_probability'] == .5 and guide['density_measure'] == DENSITY_MEASURE
            and guide['draws'] == total, 'Audit guide identity differs')
    populations = analysis['populations']
    require(len(populations) == 4 and {p['id'] for p in populations} == {j['id'] for j in selected}, 'Audit population missing/repeated')
    aggregate = {'uniform-shell': 0, 'gaussian': 0}; exterior = 0
    for job in selected:
        population = next(p for p in populations if p['id'] == job['id']); audit = population['importance_sampling_audit']
        require(population['seed'] == job['seed'] and population['estimate']['draws'] == population['hard_region']['draws'] == job['samples']
                and population['samples_sha256'] == job['output']['samples_sha256'], 'Audit population binding changed')
        counts = audit['branch_counts']; components = audit['gaussian_component_counts']
        require(audit['draws'] == job['samples'] and set(counts) == set(aggregate) and all(nonnegative_count(v) for v in counts.values())
                and sum(counts.values()) == job['samples'] and len(components) == COMPONENTS
                and all(nonnegative_count(v) for v in components) and sum(components) == counts['gaussian'], 'Audit component denominator changed')
        require(nonnegative_count(audit['shell_rejected']) and audit['shell_rejected'] <= counts['gaussian']
                and audit['shell_rejected'] == job['output']['shell_rejected'], 'Audit exterior zeros changed')
        for key in ('maximum_log_proposal_density_error', 'maximum_latent_vector_reconstruction_error', 'maximum_log_jacobian_error'):
            require(type(audit[key]) in (int, float) and math.isfinite(audit[key]) and 0. <= audit[key] < 2e-8, 'Audit reconstruction failed')
        aggregate = {k: v+counts[k] for k, v in aggregate.items()}; exterior += audit['shell_rejected']
        for name in ('samples', 'manifest', 'summary'):
            extension = 'jsonl' if name == 'samples' else 'json'
            require(sha(Path(job['directory'])/f'{name}.{extension}') == job['output'][name+'_sha256'], 'Output changed during audit')
    require(set(guide['branch_counts']) == set(aggregate) and all(nonnegative_count(v) for v in guide['branch_counts'].values())
            and guide['branch_counts'] == aggregate, 'Aggregate branch counts differ')
    require(nonnegative_count(guide['shell_rejected']) and guide['shell_rejected'] == exterior, 'Aggregate exterior zeros differ')


def run(out):
    out = Path(out).resolve(); protocol = preflight(out)
    state = dict(schema='contact-bank-pilot-status-v2', complete=False, phase='physical', started=time.time(),
                 protocol_sha256=sha(out/'protocol.json'), jobs=[dict(j, status='pending') for j in protocol['jobs']], audits={})
    with (out/'status.json').open('x') as stream: stream.write('{}\n')
    def snapshot(): write(out/'status.json', state)
    snapshot()
    try:
        execute_jobs(state['jobs'], snapshot, workers=8,
                     popen=lambda *args, **kwargs: subprocess.Popen(*args, **kwargs, env=worker_environment()))
        state['phase'] = 'physical_validation'; snapshot(); validate(out)
        for job in state['jobs']: job['output'] = verify_output(out, protocol, job)
        state['phase'] = 'audit'; snapshot()
        for arm in protocol['arms']:
            base = out/arm['id']; argv = [sys.executable, '-B', str(base/'provenance/analyze_latent_region.py'), '--root', str(base)]
            audit = state['audits'][arm['id']] = dict(command=argv, started=time.time(), returncode=None); snapshot()
            with (base/'logs/audit.log').open('xb') as stream:
                result = subprocess.run(argv, stdout=stream, stderr=subprocess.STDOUT, env=worker_environment())
            audit.update(returncode=result.returncode, finished=time.time()); snapshot(); result.check_returncode()
            verify_assessment(out, protocol, arm, read(base/'assessment/analysis.json'), state['jobs'])
            audit['analysis_sha256'] = sha(base/'assessment/analysis.json'); snapshot()
        validate(out); state.update(complete=True, phase='complete', finished=time.time()); snapshot()
    except BaseException as error:
        state.update(phase=state['phase']+'_failed', exception=repr(error), finished=time.time()); snapshot(); raise
    return state


def validate_completed(out):
    """Bind a completed pilot to all immutable inputs, raw outputs and audits."""
    out = Path(out).resolve(); protocol = validate(out); status = read(out/'status.json')
    require(status['schema'] == 'contact-bank-pilot-status-v2' and status['complete'] is True
            and status['phase'] == 'complete' and status['protocol_sha256'] == sha(out/'protocol.json'),
            'Pilot is not complete')
    require(len(status['jobs']) == len(protocol['jobs']), 'Terminal job allocation changed')
    for frozen, job in zip(protocol['jobs'], status['jobs']):
        require({k: job[k] for k in frozen} == frozen and job['status'] == 'complete' and job['returncode'] == 0,
                'Terminal physical job differs')
        require(job['output'] == verify_output(out, protocol, job), 'Terminal output binding changed')
    require(set(status['audits']) == {a['id'] for a in protocol['arms']}, 'Terminal audit allocation changed')
    for arm in protocol['arms']:
        base = out/arm['id']; audit = status['audits'][arm['id']]
        expected = [sys.executable, '-B', str(base/'provenance/analyze_latent_region.py'), '--root', str(base)]
        require(audit['command'] == expected and audit['returncode'] == 0 and
                audit['analysis_sha256'] == sha(base/'assessment/analysis.json'), 'Terminal audit binding changed')
        verify_assessment(out, protocol, arm, read(base/'assessment/analysis.json'), status['jobs'])
    return protocol, status


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['freeze', 'validate', 'preflight', 'run'])
    parser.add_argument('--out', type=Path, required=True); parser.add_argument('--binary', type=Path)
    parser.add_argument('--source-bundle', type=Path); parser.add_argument('--package', type=Path)
    args = parser.parse_args()
    if args.action == 'freeze':
        if any(x is None for x in (args.binary, args.source_bundle, args.package)):
            parser.error('freeze requires binary, source-bundle and package')
        result = freeze(args.out, args.binary, args.source_bundle, args.package)
    else:
        result = dict(validate=validate, preflight=preflight, run=run)[args.action](args.out)
    print(dict(action=args.action, out=str(args.out.resolve()), complete=result.get('complete')))
