#!/usr/bin/env python3
"""Freeze/run the fixed eight-population contact-tail pilot with audited outputs.

Reuses the validated failure-draining executor, row auditor and complete native
classifier. This new pilot cannot change old convergence gates or start assembly.
"""
from __future__ import annotations
import argparse
import copy
import os
from pathlib import Path
import shutil
import signal
import sys
import time
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
            'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'RAYON_NUM_THREADS'):
    os.environ[key] = '1'

from prepare_contact_tail_expansion import PILOT_SEEDS, PINS, read, require, sha, write_new
from prepare_shoulder_docking_benchmark import local_dependencies
from run_contact_confirmation import (runtime, worker_environment, THREAD_ENV, CAMPAIGN_SCHEMA)
from run_entry_shell_reference_campaign import file_hashes, inside
from run_full_vessel_comparison import execute_group
from run_mobile_posterior_pilot import verify_bundle
from run_smc_guide_pilot import (verify_output, verify_assessment, audit_step,
    validate_base_package, STRATA, CONVERGENCE)
from run_smc_importance_bridge import process_token
from analyze_contact_confirmation import (classify_population, summarize_arm,
    load_classifier, validate_classifier_target, validate_regions, PRIMARY, PARTS,
    DECISION_CLASSES, quality_gate, free_energy_interval, compare_mass,
    compare_free_energy_intervals, unbound_volume_bound)

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'contact-tail-independent-pilot-v1'
ARMS = [dict(id=arm, samples=16384, alpha=.5, lambda_ratio=128., component_count=count)
        for arm, count in [('baseline', 84), ('expanded', 92)]]


def verify_frozen(directory, filename='freeze.json'):
    directory = Path(directory)
    frozen = read(directory/filename)['files']
    for name, digest in frozen.items():
        require(sha(inside(directory, name)) == digest, 'Frozen artifact changed: '+name)
    return frozen


def copy_frozen(source, destination, filename='freeze.json'):
    source, destination = Path(source), Path(destination)
    for name in [*verify_frozen(source, filename), filename]:
        target = inside(destination, name); target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(inside(source, name), target)


def command(root, arm, job):
    archive = Path(root)/arm['id']/'provenance'
    return [str(Path(root)/'common/latent-region-normalizer'), '--config', str(archive/'config.json'),
        '--region', str(archive/'region.json'), '--importance-guide', str(archive/'importance-guide.json'),
        '--out', job['directory'], '--samples', '16384', '--seed', str(job['seed']),
        '--cloud-replicates', '2', '--lambda-ratio', '128.0']


def validate_design(protocol):
    require(protocol['schema'] == SCHEMA and protocol['arms'] == ARMS, 'Pilot allocation changed')
    require(protocol['total_unconditional_draws'] == 131072 and
            protocol['maximum_physical_workers'] == 2 and protocol['maximum_total_workers'] == 32,
            'Pilot budget changed')
    require(protocol['strata'] == STRATA and protocol['convergence'] == CONVERGENCE,
            'Original strata/diagnostics changed')
    jobs = protocol['jobs']
    require(len(jobs) == 8 and [j['seed'] for j in jobs] == PILOT_SEEDS,
            'Eight distinct predeclared seeds required')
    require(len({(j['arm'], j['id']) for j in jobs}) == 8 and all(j['samples'] == 16384 for j in jobs),
            'Population identity/attempt count changed')
    require(protocol['full_vessel_gate_open'] is False and protocol['assembly_gate_open'] is False,
            'Small pilot cannot open thermodynamic production gates')


def freeze(out, preparation, sphere_reference, repository=ROOT):
    out, preparation, sphere_reference, repository = map(lambda p: Path(p).resolve(),
        (out, preparation, sphere_reference, repository))
    require(not out.exists(), 'Fresh pilot directory required; no overwrite')
    verify_frozen(preparation, 'completion.json'); verify_frozen(sphere_reference)
    proposal = read(preparation/'prospective-pilot-plan.json')
    reference = read(sphere_reference/'validation.json')
    require(reference['complete'] and reference['all_checks_passed'] and
            reference['candidate_sha256'] == sha(preparation/'expanded-guide.json') and
            [r['activity'] for r in reference['results']] == [0., 2.] and
            reference['total_attempted_draws'] == 32768, 'Incomplete candidate sphere reference')
    require(proposal['seeds'] == PILOT_SEEDS and proposal['total_attempted_draws'] == 131072,
            'Previously prepared allocation differs')
    old = repository/'runs/protected-guide-validation-20260923/common'
    package = old/'reference-package'; plan, geometry = validate_base_package(package)
    binary, bundle_path = old/'latent-region-normalizer', old/'source-bundle.json'
    bundle, rust = verify_bundle(binary, bundle_path, old/'source')
    require(reference['binary_sha256'] == sha(binary) == proposal['source_binary_sha256'] and
            reference['source_bundle_sha256'] == sha(bundle_path) == proposal['source_bundle_sha256'],
            'Reference and pilot executables differ')
    require(plan['region_sha256'] == PINS['region'] and plan['shape_sha256'] == PINS['shape'],
            'Original physical target differs')
    sources = local_dependencies([Path(__file__)])
    out.mkdir(parents=True); common = out/'common'; common.mkdir()
    for name, path in sources.items(): shutil.copy2(path, common/name)
    shutil.copy2(binary, common/'latent-region-normalizer'); shutil.copy2(bundle_path, common/'source-bundle.json')
    for name, entry in bundle['files'].items():
        target = inside(common/'source', name); target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(entry['text'])
    copy_frozen(package, common/'reference-package')
    # Small authenticated receipts bind the immutable preparation and sphere archives;
    # their frozen files remain external and are checked by validate().
    write_new(common/'geometry-preflight.json', geometry)
    jobs = []
    for arm in ARMS:
        base = out/arm['id']; archive = base/'provenance'; archive.mkdir(parents=True)
        (base/'runs').mkdir(); (base/'logs').mkdir()
        for name, path in sources.items(): shutil.copy2(path, archive/name)
        for name, path in [('source-bundle.json', bundle_path), ('latent-region-normalizer', binary),
            ('source-config.json', package/'config.json'), ('shape.json', package/'shape.json'),
            ('region.json', package/'region.json'), ('importance-guide.json', preparation/
             ('baseline.json' if arm['id'] == 'baseline' else 'expanded-guide.json'))]:
            shutil.copy2(path, archive/name)
        config = read(package/'config.json'); config['shape'] = str(archive/'shape.json')
        write_new(archive/'config.json', config)
        arm_jobs = []
        for index in range(4):
            job = dict(id=f'r{index:02}', arm=arm['id'], kind='physical', samples=16384,
                seed=PILOT_SEEDS[len(jobs)], directory=str(base/'runs'/f'r{index:02}'),
                log=str(base/'logs'/f'r{index:02}.log'))
            job['command'] = command(out, arm, job); jobs.append(job); arm_jobs.append(job)
        manifest = dict(schema=CAMPAIGN_SCHEMA, jobs=arm_jobs, workers=2, physical_activity=.035,
            lambda_ratio=128., cloud_replicates=2, config_sha256=sha(archive/'config.json'),
            shape_sha256=sha(archive/'shape.json'), region_sha256=sha(archive/'region.json'),
            importance_guide_sha256=sha(archive/'importance-guide.json'),
            archive_sha256=file_hashes(archive), allocation=arm)
        write_new(base/'manifest.json', manifest)
    protocol = dict(schema=SCHEMA, arms=ARMS, jobs=jobs, total_unconditional_draws=131072,
        maximum_physical_workers=2, maximum_total_workers=32, maximum_classification_workers=1,
        repository=str(repository), runtime=runtime(), thread_environment=THREAD_ENV,
        controller_sha256=sha(__file__), python_sources={name: sha(common/name) for name in sources},
        binary_sha256=sha(binary), source_bundle_sha256=sha(bundle_path), rust_sources=rust,
        preparation=str(preparation), preparation_completion_sha256=sha(preparation/'completion.json'),
        sphere_reference=str(sphere_reference), sphere_reference_sha256=sha(sphere_reference/'validation.json'),
        native_definition='common/reference-package/native-region/definition.json',
        native_definition_sha256=plan['native_definition_sha256'],
        reference_region='common/reference-package/old-r5-region.json', reference_region_sha256=plan['reference_region_sha256'],
        supplemental_definition='common/reference-package/native-partition-definition.json',
        supplemental_definition_sha256=plan['supplemental_definition_sha256'],
        region_sha256=plan['region_sha256'], shape_sha256=plan['shape_sha256'],
        strata=STRATA, convergence=CONVERGENCE, full_vessel_gate_open=False, assembly_gate_open=False,
        scope='Fresh fixed-allocation feasibility pilot. Every attempt and invalid zero retained. No previous rows pooled, no retries/extension, no historical convergence-gate changes, no automatic production launch.')
    validate_design(protocol)
    write_new(out/'protocol.json', protocol); write_new(out/'freeze.json', dict(files=file_hashes(out)))
    validate(out, sha(out/'protocol.json'))
    return protocol


def validate(out, expected):
    out = Path(out).resolve()
    require(sha(out/'protocol.json') == expected, 'Pinned pilot protocol changed')
    protocol = read(out/'protocol.json'); validate_design(protocol); verify_frozen(out)
    require(sys.flags.optimize == 0 and runtime() == protocol['runtime'], 'Audit runtime changed')
    common = out/'common'
    require(sha(common/'latent-region-normalizer') == protocol['binary_sha256'] and
            sha(common/'source-bundle.json') == protocol['source_bundle_sha256'], 'Executable changed')
    _, rust = verify_bundle(common/'latent-region-normalizer', common/'source-bundle.json', common/'source')
    require(rust == protocol['rust_sources'], 'Executable source closure changed')
    for name, digest in protocol['python_sources'].items():
        require(sha(common/name) == digest, 'Python source changed: '+name)
    preparation = Path(protocol['preparation']); sphere = Path(protocol['sphere_reference'])
    require(sha(preparation/'completion.json') == protocol['preparation_completion_sha256'] and
            sha(sphere/'validation.json') == protocol['sphere_reference_sha256'], 'Prepared evidence changed')
    verify_frozen(preparation, 'completion.json'); verify_frozen(sphere)
    package = common/'reference-package'; verify_frozen(package)
    for name, key in [('region.json', 'region_sha256'), ('shape.json', 'shape_sha256')]:
        require(sha(package/name) == protocol[key], 'Physical target changed')
    for key in ('native_definition', 'reference_region', 'supplemental_definition'):
        require(sha(inside(out, protocol[key])) == protocol[key+'_sha256'], 'Classifier partition changed')
    expected_jobs = []
    for arm in protocol['arms']:
        base = out/arm['id']; archive = base/'provenance'; manifest = read(base/'manifest.json')
        require(manifest['schema'] == CAMPAIGN_SCHEMA and manifest['allocation'] == arm and
                manifest['archive_sha256'] == file_hashes(archive), 'Arm archive changed')
        candidate = preparation/('baseline.json' if arm['id']=='baseline' else 'expanded-guide.json')
        require(sha(archive/'importance-guide.json') == sha(candidate) == manifest['importance_guide_sha256'],
                'Candidate guide changed')
        config = read(archive/'config.json'); original = read(package/'config.json')
        restored = copy.deepcopy(config); restored['shape'] = original['shape']
        require(restored == original and config['shape'] == str(archive/'shape.json'), 'Physical configuration changed')
        require(sha(archive/'region.json') == protocol['region_sha256'] and
                sha(archive/'shape.json') == protocol['shape_sha256'], 'Physical geometry changed')
        for job in manifest['jobs']:
            require(job['command'] == command(out, arm, job) and
                    job['directory'] == str(base/'runs'/job['id']) and
                    job['log'] == str(base/'logs'/(job['id']+'.log')), 'Job command/path changed')
            expected_jobs.append(job)
    require(expected_jobs == protocol['jobs'], 'Manifest allocation differs')
    return protocol


def compare(arms, protocol):
    checks = {}; left, right = arms['baseline'], arms['expanded']; gates = protocol['convergence']
    for region in DECISION_CLASSES:
        checks[region] = compare_mass(left['estimates'][region], right['estimates'][region], 'Qz', gates)
    strata = []
    for family in ('radial', 'angular', 'orthant'):
        for region in DECISION_CLASSES:
            for a, b in zip(left['strata'][family][region], right['strata'][family][region]):
                significant = max(a['observed_class_fraction']['Qz'] or 0., b['observed_class_fraction']['Qz'] or 0.) >= .01
                strata.append(dict(family=family, region=region, bin=a['bin'], significant=significant,
                    comparison=compare_mass(a, b, 'Qz', gates)))
    return dict(quality={arm: {r: quality_gate(data['estimates'][r], gates) for r in DECISION_CLASSES}
            for arm, data in arms.items()}, regional_comparisons=checks, all_stratum_comparisons=strata,
        free_energy_intervals={arm: free_energy_interval(data, gates) for arm, data in arms.items()},
        full_vessel_gate_open=False, assembly_gate_open=False,
        interpretation='Feasibility diagnostics only; original failed gates unchanged; no size/intensity control or unseen-mass bound supplied by this pilot')


def classify_and_compare(out, protocol, state, snapshot):
    destination = out/'comparison'; require(not destination.exists(), 'No repeated classification')
    destination.mkdir(); definition = inside(out, protocol['native_definition'])
    reference = inside(out, protocol['reference_region'])
    classifier, binding = load_classifier(definition)
    assessments = {a['id']: read(out/a['id']/'assessment/analysis.json') for a in ARMS}
    records = []
    for job in state['jobs']:
        arm = next(a for a in ARMS if a['id'] == job['arm'])
        config = read(out/arm['id']/'provenance/config.json')
        validate_classifier_target(config, protocol['shape_sha256'], classifier.definition, definition)
        validate_regions(read(out/arm['id']/'provenance/region.json'), read(reference))
        audit = next(r for r in assessments[arm['id']]['populations'] if r['id'] == job['id'])
        task = dict(root=str(out), out=str(destination), job=job, arm=arm, output=job['output'],
            audit=audit, definition=str(definition), classifier_binding=binding, strata=STRATA,
            reference_region=str(reference), reference_region_sha256=protocol['reference_region_sha256'],
            supplemental_definition_sha256=protocol['supplemental_definition_sha256'])
        records.append(classify_population(task))
        state['classified'].append(arm['id']+'/'+job['id']); snapshot()
    arms = {arm['id']: summarize_arm(destination, arm,
        [r for r in records if r['arm'] == arm['id']], STRATA, assessments[arm['id']]) for arm in ARMS}
    result = dict(schema=SCHEMA, complete=True, total_unconditional_draws=131072,
        protocol_sha256=sha(out/'protocol.json'), native_definition=binding,
        arms=arms, diagnostics=compare(arms, protocol), old_audits_replayed=0,
        old_samples_pooled=False, unbound_R4_bound=unbound_volume_bound(read(out/'baseline/provenance/region.json')))
    write_new(destination/'analysis.json', result)
    state['comparison_sha256'] = sha(destination/'analysis.json')


def run(out, expected):
    out = Path(out).resolve(); protocol = validate(out, expected)
    require(Path(__file__).resolve() == out/'common/run_contact_tail_pilot.py', 'Use the frozen controller')
    require(not (out/'status.json').exists(), 'No retry or overwrite of a started pilot')
    for arm in ARMS:
        base = out/arm['id']
        require(not any((base/'runs').iterdir()) and not any((base/'logs').iterdir()) and
                not (base/'assessment').exists(), 'Existing pilot outputs')
    write_new(Path(protocol['preparation'])/'pilot-execution-claim.json', dict(
        campaign=str(out), protocol_sha256=expected, pid=os.getpid(), process_birth=process_token(os.getpid())))
    state = dict(schema=SCHEMA, complete=False, phase='physical', started=time.time(),
        pid=os.getpid(), process_birth=process_token(os.getpid()), protocol_sha256=expected,
        jobs=[dict(j, status='pending') for j in protocol['jobs']], audits={}, classified=[])
    write_new(out/'status.json', state)
    def snapshot():
        temporary = out/'status.tmp'; temporary.write_text(__import__('json').dumps(state, indent=2)+'\n')
        temporary.replace(out/'status.json')
    def interrupted(signum, frame):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        raise InterruptedError('Draining started children after termination request')
    previous = signal.signal(signal.SIGTERM, interrupted)
    try:
        execute_group(state['jobs'], snapshot, 2, protocol['repository'], worker_environment(),
                      before_launch=lambda: validate(out, expected))
        state['phase'] = 'audit'; snapshot()
        for job in state['jobs']: job['output'] = verify_output(out, protocol, job)
        state['audits'] = {arm['id']: audit_step(out, arm) for arm in ARMS}; snapshot()
        execute_group(list(state['audits'].values()), snapshot, 1, protocol['repository'], worker_environment())
        for arm in ARMS:
            path = out/arm['id']/'assessment/analysis.json'
            verify_assessment(out, protocol, arm, read(path), state['jobs'])
            state['audits'][arm['id']]['analysis_sha256'] = sha(path)
        state['phase'] = 'classification'; snapshot()
        classify_and_compare(out, protocol, state, snapshot)
        validate(out, expected); state.update(complete=True, phase='complete', finished=time.time()); snapshot()
    except BaseException as error:
        state.update(complete=False, phase=state['phase']+'_failed', error=repr(error), finished=time.time()); snapshot()
        raise
    finally: signal.signal(signal.SIGTERM, previous)
    return state


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('freeze'); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--preparation', type=Path, required=True); p.add_argument('--sphere-reference', type=Path, required=True)
    for action in ('preflight', 'run'):
        p = sub.add_parser(action); p.add_argument('--out', type=Path, required=True)
        p.add_argument('--expected-protocol-sha256', required=True)
    args = parser.parse_args()
    if args.action == 'freeze':
        freeze(args.out, args.preparation, args.sphere_reference)
        print(dict(protocol_sha256=sha(args.out/'protocol.json'), physical_jobs_launched=0))
    elif args.action == 'preflight':
        validate(args.out, args.expected_protocol_sha256); print('Frozen pilot validation passed; no jobs launched')
    else: print(run(args.out, args.expected_protocol_sha256)['phase'])
