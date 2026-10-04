#!/usr/bin/env python3
"""Materialize the fixed 512-proposal support screen; never launch a child.

The original proposal-audit executable/schema is retained.  Only its Gaussian
mixture changes.  Completed metadata, not original sample streams, supplies
the eight development centers.  A separate launch review remains necessary.
"""
from __future__ import annotations

import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import shutil
import signal
import sys

from analyze_mobile_native_pocket import local_sources
from native_class_line_physical_algebra_audit import runtime_identity
from native_class_physical_labels import observer_setup_inventory, validate_classifier_target
from native_class_selected_geometry import preselection
from prepare_native_class_physical_campaign import seed_inventory
from prepare_hard_free_line_fresh import ROOTS
from run_native_class_physical_campaign import read, require, sha, write

SCHEMA = 'native-class-support-pilot-v1'
INPUT_SCHEMA = 'native-class-support-pilot-inputs-v1'
PRODUCER_SCHEMA = 'native-class-line-guide-audit-v1'
GUIDE_SCHEMA = 'defensive-native-class-line-guide-v1'
NAMESPACE = 'native-class-training-center-support-pilot-20261004-v1'
TRAINING_IDS = [
    'critical:competing22:r00:518', 'critical:competing22:r00:1003',
    'critical:competing62:r00:4006', 'critical:competing62:r00:5233',
    'critical:old-native55:r00:584', 'critical:old-native55:r00:599',
    'critical:remaining-native55:r00:16', 'critical:remaining-native55:r00:17']
WIDTHS = [.05, .15, .45]
CHANNELS = [dict(**{'class': kind}, probability=.2, **({} if orthant is None else dict(orthant=orthant)))
            for kind, orthant in [('hard_free', None), ('contact_without_native', None),
                                  ('contact_without_native', 22), ('contact_without_native', 62), ('native', 55)]]
PINS = {
    'recommendation': '4e5477659bbd82204542b160c8dad22d71a327d4b89f0880edfc3eb35a5e76d2',
    'diagnosis': 'c986b86f0927243597ab1b5c180bd3f1ad7ba3b24f5aaae8ad9784bfbce5110d',
    'diagnosis_completion': '09c3967bd9c71d93846a6118ff651f64dd2bf7f491eea4d98dc2214cb4f1b419',
    'original_plan': '7f1bfeb01f8c0d371a5e1f14fbc66371b68fb485452be5fe25f5a2ca72fc246e',
    'original_completion': '6b104d1436f947d4d16b5e316752ebcc130c563b617b1bd8c879a1130998e436',
    'original_summary': '990644192945028974fb21d0a373158d46d43b368cb6a78207e9ce15bccd3201',
    'original_guide': 'f14837081dc2fb08873d4617b2d9bac3b56cf55640242a77fe8b5cd968cb329e',
    'config': '1656dc5650e2fbb18a3086950622a73dfb7ca0ec46b8031abd45ebbec042a42e',
    'region': '924648f4d9db473045397c300b3b4af7ccde8cfda1b1703a6899239ec12e2f02',
    'shape': 'c7442034e7ffa4627b67c57a81117f0e9bc2d389ecf956a83dac2a5e915554c9',
    'compiled_native': 'dbb3c3e32259f506b9c979ebb9d1fd773cb78507c1f808fd8dd0ed319e2eade4',
    'definition': '5cf55ebe0c0f78ec6b8cdc745cac938b0fadfa3732e1e1c730864c62c651d7a9',
    'producer': 'f9afc90000004062b67e5cc872124cd8203e2acd6bb462df2eb4835b330474bc',
    'producer_bundle': '3af8b65ca4800deb5d343521f73cfc8c1ff80db910da413e5c64c96e16c9038b',
}
PHASE_LIMITS = {phase: dict(cpu_limit_seconds=cpu, wall_limit_seconds=2*cpu,
                           address_space_limit_bytes=gib*2**30)
                for phase, cpu, gib in [('producer', 300, 4), ('algebra', 300, 4),
                                       ('labels', 600, 8), ('geometry', 1800, 8), ('statistics', 300, 4)]}
for _phase in ('algebra', 'labels', 'geometry'):
    PHASE_LIMITS[_phase]['max_record_bytes'] = 32*2**20
ENTRIES = ('prepare_native_class_support_pilot.py', 'audit_native_class_support_pilot.py',
           'summarize_native_class_support_pilot.py', 'run_native_class_physical_campaign.py')


def bound(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha(path))


class Bindings:
    def __init__(self): self.files = {}

    def bind(self, path, expected=None):
        path = Path(path).resolve(); digest = sha(path)
        require(expected is None or digest == expected, 'Changed input '+str(path))
        require(str(path) not in self.files or self.files[str(path)] == digest, 'Input changed during admission')
        self.files[str(path)] = digest
        return path

    def ref(self, reference):
        require(set(reference) == {'path', 'sha256'}, 'Expected exact BoundFile')
        return self.bind(reference['path'], reference['sha256'])

    def recheck(self):
        for path, digest in self.files.items(): require(sha(path) == digest, 'Input changed '+path)


def default_inputs(repository):
    repository = Path(repository).resolve()
    old = repository/'results/native-class-line-proposal-preparation-20261003'
    diag = repository/'results/native-class-saved-support-diagnosis-20261004'
    paths = dict(recommendation=diag/'pilot-recommendation-selected-v2.json', diagnosis=diag/'analysis.json',
        diagnosis_completion=diag/'completion.json', original_plan=old/'execution-plan.json',
        original_completion=old/'execution/summary.json',
        original_summary=repository/'results/native-class-line-proposal-summary-20261004.json',
        original_guide=old/'guides/class.json', config=old/'common/config.json', region=old/'common/region.json',
        shape=old/'common/shape.json', compiled_native=old/'common/compiled-native.json',
        definition=old/'execution-code/native-definition/definition.json',
        producer=old/'execution-code/contact-line-guide-audit',
        producer_bundle=old/'queries/class-r00/provenance/source-bundle.json')
    return dict(schema=INPUT_SCHEMA, repository=str(repository), seed_namespace=NAMESPACE,
                inputs={key: dict(path=str(path), sha256=PINS[key]) for key, path in paths.items()},
                seed_inventory_roots=list(dict.fromkeys([*ROOTS, str(repository/'results'), str(repository/'runs')])) )


def build_guide(original, diagnosis, compiled_ref):
    require(original['schema'] == GUIDE_SCHEMA and len(original['gaussian_components']) == 92,
            'Expected original 92-component class guide')
    require(original['defensive_uniform_shell_probability'] == .5 and original['conditional_probability'] == 1.
            and original['raw_translation_axes'] == [0, 1, 2] and original['minimum_conditional_mass'] == 1e-12
            and original['class_channels'] == CHANNELS, 'Original class law changed')
    require(diagnosis['schema'] == 'native-class-saved-support-diagnosis-v1'
            and diagnosis['complete'] is True and diagnosis['passed'] is True, 'Incomplete saved diagnosis')
    training = [row for row in diagnosis['rows'] if row['training']]
    require([row['id'] for row in training] == TRAINING_IDS, 'Critical training inventory changed')
    require(len({tuple(row['latent']) for row in training}) == 8, 'Repeated training center')
    added = diagnosis['new_components']
    require(len(added) == 24, 'Expected 24 additions')
    components = copy.deepcopy(original['gaussian_components'])
    total = math.fsum(component['weight'] for component in components)
    require(math.isfinite(total) and total > 0 and all(math.isfinite(c['weight']) and c['weight'] > 0 for c in components),
            'Invalid original Gaussian weights')
    inventory = []
    for i, component in enumerate(components):
        component['weight'] = .75*(component['weight']/total)
        inventory.append(dict(index=i, bank='original', training_id=None, latent_sigma=None))
    for i, (row, width) in enumerate((row, width) for row in training for width in WIDTHS):
        entry = added[i]
        require(entry['training_id'] == row['id'] and entry['latent_sigma'] == width
                and entry['mean'] == row['latent'] and len(entry['mean']) == 6
                and all(type(v) in (int, float) and math.isfinite(v) for v in entry['mean'])
                and entry['weight'] == .25/24
                and entry['covariance'] == [[width**2 if a == b else 0. for b in range(6)] for a in range(6)],
                'Added center, width, covariance or mass differs from fixed design')
        components.append({key: copy.deepcopy(entry[key]) for key in ('weight', 'mean', 'covariance')})
        inventory.append(dict(index=92+i, bank='added', training_id=row['id'], latent_sigma=width))
    require(abs(math.fsum(v['weight'] for v in components)-1.) <= 2e-15, 'Candidate weights do not normalize')
    result = copy.deepcopy(original)
    result['gaussian_components'] = components; result['compiled_native'] = compiled_ref
    return result, inventory


def binary64_retention(original, candidate):
    """Reproduce the producer's ordered f64 weight sum/division, without it.

    This is a coefficient diagnostic, not an IEEE guarantee for a subsequently
    evaluated density.  The checked 3/4 theorem concerns exact arithmetic.
    """
    def normalized(guide):
        values = [float(c['weight']) for c in guide['gaussian_components']]
        total = 0.
        for value in values: total += value
        return total, [value/total for value in values]
    old_sum, old = normalized(original); new_sum, new = normalized(candidate)
    ratios = [new[i]/value for i, value in enumerate(old)]
    return dict(parser_rule='src/latent_region.rs ImportanceGuide: ordered f64 sum then weight/total',
        original_binary64_weight_sum=old_sum, candidate_binary64_weight_sum=new_sum,
        original_component_retention_min=min(ratios), original_component_retention_max=max(ratios),
        ideal_exact_arithmetic_coefficient=.75,
        implemented_defensive_coefficient=1.,
        scope='Normalized coefficient ratios only; no IEEE pointwise-density inequality is certified.')


def source_paths():
    base = Path(__file__).resolve().parent; result = {}
    for entry in ENTRIES:
        require((base/entry).is_file(), 'Missing pilot implementation '+entry)
        for name, path in local_sources(base/entry).items():
            require(name not in result or result[name] == path, 'Ambiguous module '+name)
            result[name] = path
    return result


def population_design(root, namespace, previous_seeds):
    require(namespace == NAMESPACE, 'Fixed prospective namespace required')
    populations = []; fresh = []
    for i in range(4):
        identity = f'r{i:02}'
        seeds = {role: int.from_bytes(hashlib.sha256(f'{namespace}/{identity}/{role}'.encode()).digest()[:8], 'little')
                 for role in ('proposal', 'audit')}
        fresh.extend(seeds.values())
        selection = preselection(identity, 128, seeds['audit'])
        populations.append(dict(id=identity, samples=128, seed=seeds['proposal'], audit_seed=seeds['audit'],
            directory=str(Path(root)/'queries'/identity), selected_ids=selection['draw_ids'], preselection=selection))
    require(len(set(fresh)) == 8 and not set(fresh).intersection(previous_seeds), 'Fresh or audit seed collision')
    return populations


def validate(inputs_path, validation_path):
    bindings = Bindings(); inputs_path = bindings.bind(inputs_path); inputs = read(inputs_path)
    require(inputs['schema'] == INPUT_SCHEMA and set(inputs['inputs']) == set(PINS), 'Wrong pilot input schema/inventory')
    data = {}
    for key, expected in PINS.items():
        ref = inputs['inputs'][key]
        require(ref['sha256'] == expected, 'Unreviewed input identity '+key)
        path = bindings.ref(ref)
        if key != 'producer': data[key] = read(path)
    old = Path(inputs['inputs']['original_plan']['path']).resolve().parent
    plan, done = data['original_plan'], data['original_completion']
    require(plan['schema'] == 'native-class-line-reviewed-execution-v1' and plan['physical_clouds'] == 0
            and plan['executable_sha256'] == PINS['producer'] and len(plan['jobs']) == 18
            and done['complete'] is True and done['passed'] is True and done['plan_sha256'] == PINS['original_plan']
            and len(done['completed']) == 18 and done['fresh_draws'] == 1024 and done['saved_development_probes'] == 40
            and done['new_physical_clouds'] == 0 and not (old/'execution/failure.json').exists(), 'Original 18-job gate failed')
    for ordinal, (job, terminal) in enumerate(zip(plan['jobs'], done['completed'])):
        require(terminal['id'] == job['id'] and terminal['terminal'] == job['terminal'], 'Original job identity changed')
        result = read(bindings.bind(terminal['terminal'], terminal['sha256']))
        require(result['complete'] is True and (job['phase'] != 'audit' or result['passed'] is True), 'Original terminal failed')
        folder = old/'execution'/f'{ordinal:02}-{job["id"]}'
        exit_record = read(bindings.bind(folder/'exit.json'))
        require(exit_record['returncode'] == 0 and exit_record['error'] is None and exit_record['child_drained'] is True,
                'Original child exit failed')
        require(read(bindings.bind(folder/'success.json')) == terminal, 'Original success binding changed')
    summary, diagnosis, completion = data['original_summary'], data['diagnosis'], data['diagnosis_completion']
    require(summary['complete'] is True and summary['execution_plan_sha256'] == PINS['original_plan']
            and summary['execution_summary_sha256'] == PINS['original_completion'], 'Original once-only reduction differs')
    require(completion['complete'] is True and completion['passed'] is True and completion['source_unchanged'] is True
            and completion['analysis_sha256'] == PINS['diagnosis'] and completion['query_counts'] ==
            dict(saved_rows=40, saved_axes=120, new_geometry=0, new_pose_draws=0, new_clouds=0), 'Diagnosis receipt differs')
    recommendation = data['recommendation']
    require(recommendation['independent_geometry']['total_rows'] == 64
            and recommendation['total_new_unconditional_proposals'] == 512 and recommendation['total_new_clouds'] == 0
            and recommendation['supersedes']['sha256'] == '4a1decb5cc0ab273ffa290bc9b6916b167c8f35e41b70ceed9b2abc95968b340',
            'Selected recommendation differs')
    bindings.ref(recommendation['supersedes'])
    config, guide, definition, compiled = [data[k] for k in ('config', 'original_guide', 'definition', 'compiled_native')]
    require(config['depletant_radius'] == 1.5 and config['reservoir_density'] == .035 and config['capture_radius'] == 170
            and guide['region_sha256'] == PINS['region'] and guide['shape_sha256'] == PINS['shape']
            and guide['fixed_poses'] == config['fixed_poses'] == compiled['fixed_poses'], 'Physical/scaffold scope changed')
    definition_path = Path(inputs['inputs']['definition']['path']).resolve()
    require(compiled['source_definition_sha256'] == PINS['definition']
            and compiled['source_input_sha256'] == definition['input_sha256'], 'Frozen native identity differs')
    for name, digest in definition['input_sha256'].items():
        relative = Path(name)
        require(not relative.is_absolute() and '..' not in relative.parts, 'Unsafe native input member')
        bindings.bind(definition_path.parent/'inputs'/relative, digest)
    validate_classifier_target(config, PINS['shape'], definition, definition_path)
    build_guide(guide, diagnosis, inputs['inputs']['compiled_native'])
    bundle = data['producer_bundle']; executable = Path(inputs['inputs']['producer']['path'])
    require(bundle['schema'] == 1 and bundle['files'] and os.access(executable, os.X_OK)
            and Path(inputs['inputs']['producer_bundle']['path']).read_bytes() in executable.read_bytes(), 'Producer bundle is not embedded')
    for name, entry in bundle['files'].items():
        relative = Path(name)
        require(not relative.is_absolute() and '..' not in relative.parts
                and hashlib.sha256(entry['text'].encode()).hexdigest() == entry['sha256'], 'Invalid embedded source member')
    source = source_paths(); source_hashes = {str(path): sha(path) for path in source.values()}
    validation_path = bindings.bind(validation_path); validation = read(validation_path)
    require(validation['complete'] is True and validation['passed'] is True and validation['returncode'] == 0
            and validation['source_before'] == validation['source_after']
            and all(validation['source_after'].get(path) == digest for path, digest in source_hashes.items()),
            'Passing synthetic validation of current pilot closure required')
    for path, digest in source_hashes.items(): bindings.bind(path, digest)
    return dict(inputs=inputs, inputs_path=inputs_path, validation_path=validation_path, data=data,
                bindings=bindings, source=source, runtime=runtime_identity())


def execution_plan(protocol, reference):
    root, code = Path(protocol['root']), Path(protocol['code_directory']); jobs = []
    def add(identity, population, phase, argv, terminal, contract='complete_and_passed'):
        limits = protocol['phase_limits'][phase]
        jobs.append(dict(id=identity, population=population, phase=phase, argv=argv,
            terminal=dict(path=str(terminal), success_contract=contract),
            **{k: limits[k] for k in ('cpu_limit_seconds', 'wall_limit_seconds', 'address_space_limit_bytes')}))
    for population in protocol['populations']:
        identity = population['id']
        add(identity+'-producer', identity, 'producer', [protocol['producer']['path'],
            '--config', protocol['config']['path'], '--region', protocol['region']['path'],
            '--importance-guide', protocol['guide']['path'], '--out', population['directory'],
            '--samples', '128', '--seed', str(population['seed'])], Path(population['directory'])/'summary.json', 'complete')
        for phase in ('algebra', 'labels', 'geometry'):
            add(identity+'-'+phase, identity, phase, [protocol['python'], '-B', str(code/'audit_native_class_support_pilot.py'),
                '--protocol', reference['path'], '--protocol-sha256', reference['sha256'],
                '--population', identity, '--phase', phase], root/'analysis'/identity/(phase+'.json'))
    add('statistics', 'all', 'statistics', [protocol['python'], '-B', str(code/'summarize_native_class_support_pilot.py'),
        '--protocol', reference['path'], '--protocol-sha256', reference['sha256']], root/'analysis/statistics.json')
    files = dict(protocol['files']); files[reference['path']] = reference['sha256']
    resolutions = {job['argv'][0]: str(Path(job['argv'][0]).resolve()) for job in jobs}
    for path in resolutions.values(): files[path] = sha(path)
    return dict(schema='native-class-physical-execution-v1', root=str(root), maximum_workers=1, threads=1,
        files=files, executable_resolutions=resolutions, jobs=jobs, preparation_receipt=str(root/'preparation.json'),
        scope='Proposal-only 512 attempts, all-row algebra/labels, exactly64 selected full references. No clouds, retries or extension.')


def materialize(out, context, inventory):
    inputs, data, bindings = context['inputs'], context['data'], context['bindings']
    common, code = out/'common', out/'code'; common.mkdir(); code.mkdir(); (out/'analysis').mkdir()
    copied = {}
    def copy_file(source, destination):
        source = Path(source).resolve(); expected = bindings.files[str(source)]
        require(sha(source) == expected, 'Copy source changed')
        destination.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, destination)
        require(sha(destination) == expected, 'Copied bytes differ from admitted source')
        copied[str(destination)] = expected
        return bound(destination)
    copy_file(context['inputs_path'], common/'inputs.json')
    validation = copy_file(context['validation_path'], common/'validation.json')
    for name, source in context['source'].items(): copy_file(source, code/name)
    refs = {}
    for name in ('shape', 'region', 'compiled_native', 'producer_bundle'):
        refs[name] = copy_file(inputs['inputs'][name]['path'], common/(name+'.json'))
    refs['producer'] = copy_file(inputs['inputs']['producer']['path'], common/'contact-line-guide-audit')
    definition_source = Path(inputs['inputs']['definition']['path']).resolve()
    refs['definition'] = copy_file(definition_source, out/'native-definition/definition.json')
    for name in data['definition']['input_sha256']:
        copy_file(definition_source.parent/'inputs'/name, out/'native-definition/inputs'/name)
    for name, entry in data['producer_bundle']['files'].items():
        destination = common/'rust-source'/name; destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(entry['text'].encode())
        require(sha(destination) == entry['sha256'], 'Extracted embedded source differs')
        copied[str(destination)] = entry['sha256']
    config = copy.deepcopy(data['config']); config['shape'] = refs['shape']['path']; write(common/'config.json', config)
    refs['config'] = bound(common/'config.json')
    guide, components = build_guide(data['original_guide'], data['diagnosis'], refs['compiled_native'])
    write(common/'guide.json', guide); refs['guide'] = bound(common/'guide.json')
    write(common/'seed-inventory.json', inventory)
    populations = population_design(out, inputs['seed_namespace'], inventory['seeds'])
    for population in populations:
        (out/'analysis'/population['id']).mkdir()
    (out/'queries').mkdir()
    setup = observer_setup_inventory(Path(refs['definition']['path']))
    for path, digest in copied.items(): require(sha(path) == digest, 'Copied closure changed before publication')
    bindings.recheck()
    files = {str(path): sha(path) for path in sorted(out.rglob('*')) if path.is_file()}
    files.update(context['runtime']['file_sha256'])
    protocol = dict(schema=SCHEMA, root=str(out), code_directory=str(code), files=files,
        source_sha256={name: sha(path) for name, path in context['source'].items()}, runtime=context['runtime'],
        python=str(Path(sys.executable).absolute()), **refs, producer_schema=PRODUCER_SCHEMA,
        populations=populations, component_inventory=components, observer_setup=setup,
        phase_limits=copy.deepcopy(PHASE_LIMITS), maximum_workers=1, threads=1, total_attempts=512,
        selected_geometry_rows=64, selected_geometry_axes=192, classifier_setups=8,
        new_Poisson_clouds=0, saved_probe_replays=0, physical_weight_estimates=0,
        storage_reserve_bytes=2*2**30, aggregate_cpu_ceiling_seconds=12300, aggregate_wall_ceiling_seconds=24600,
        prior_inputs=bindings.files, validation=validation, launched=False, launch_review_complete=False,
        binary64_weight_retention=binary64_retention(data['original_guide'], guide),
        physical_campaign_gate_open=False, full_vessel_gate_open=False, assembly_gate_open=False,
        training_critical_points=8, nontraining_critical_holdouts=0,
        scope='Training-center access screen only; all attempts retained. Selected references do not certify all-row geometry.')
    write(out/'protocol.json', protocol)
    execution = execution_plan(protocol, bound(out/'protocol.json'))
    for path, digest in files.items(): require(sha(path) == digest, 'Frozen closure changed before publication')
    write(out/'execution-plan.json', execution)
    receipt = dict(schema='native-class-support-pilot-preparation-v1', complete=True, launched=False,
        root=str(out), protocol=bound(out/'protocol.json'), execution_plan=bound(out/'execution-plan.json'),
        total_attempts=512, selected_reference_rows=64, new_Poisson_clouds=0,
        launch_review_complete=False, scope='Metadata preparation only; external review required before launching.')
    write(out/'preparation.json', receipt)
    return receipt


def prepare(out, inputs_path, validation_path):
    out = Path(out).resolve(); require(not out.exists(), 'Preparation requires a fresh directory')
    context = validate(inputs_path, validation_path)
    inventory = seed_inventory(context['inputs']['seed_inventory_roots'])
    for path, item in inventory['files'].items(): context['bindings'].bind(path, item['sha256'])
    population_design(out, context['inputs']['seed_namespace'], inventory['seeds'])
    require(shutil.disk_usage(out.parent).free >= 2*2**30, 'Need fixed2GiB storage reserve')
    context['bindings'].recheck(); out.mkdir()
    write(out/'preparation-attempt.json', dict(schema=SCHEMA, launched=False, input_sha256=context['bindings'].files))
    handlers = {}
    def interrupted(signum, frame): raise InterruptedError('Preparation signal '+str(signum))
    try:
        for sig in (signal.SIGTERM, signal.SIGINT): handlers[sig] = signal.signal(sig, interrupted)
        return materialize(out, context, inventory)
    except BaseException as error:
        write(out/'preparation-failure.json', dict(schema=SCHEMA, complete=False, launched=False,
            error_type=type(error).__name__, error=str(error), retries=0,
            scope='Partial directory preserved. Never launch or reuse it.'))
        raise
    finally:
        for sig, previous in handlers.items(): signal.signal(sig, previous)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--validation', type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(prepare(arguments.out, arguments.inputs, arguments.validation), indent=2))
