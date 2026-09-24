#!/usr/bin/env python3
"""Prepare immutable finite-assembly inputs; no simulation or timing selection.

This deliberately is not a finite-assembly-campaign-v1 manifest. A later binder
must supply an executable, observation window and actual output paths, and must
independently enforce the unchanged physical launch gates.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil

import finite_assembly_contract as contract
from contact_benchmark_contract import digest, validate_contract as validate_benchmark
from prepare_shoulder_docking_benchmark import local_dependencies
from summarize_finite_assembly_starts import authenticated_inputs

SCHEMA = 'finite-assembly-config-bank-v1'
OBSERVER_MANIFEST_SHA = 'd2c2bd6dc0571df919361e41090351523ec7566c2ff0629824443d323300d871'
OBSERVER_FREEZE_SHA = '4407fa5d4004f423ead506b1db441431d09e59b2f5aac0e8dd81c3350c1dc08a'
DEFAULT_SCHEDULE = dict(
    common=dict(local_translation_std_A=.2, local_small_angle_std_degrees=1.,
        gca_probability=1., center_shift_probability=1., poisson_lambda_ratio=64.,
        endpoint_gate=dict(max_cells=2047, max_depth=14, min_width=.5)),
    global_probability=.5, learned_uniform_weight=.1,
    frozen_posterior=dict(probability=.5, correlation=.9))
ROLES = ('local', 'redraw', 'transport')
UNRESOLVED = dict(executable='Validated release executable and embedded source bundle',
    window='Frozen burn/end/cadence; none selected by this preparation',
    output_root='Fresh independent output paths',
    physical_gates='Regional and full-vessel convergence and remaining-mass checks',
    dispatch='Resource-limited dispatcher with launch-time seed/provenance checks')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(1024*1024), b''):
            h.update(data)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def inside(root, relative):
    root = Path(root).resolve()
    require(isinstance(relative, str) and relative and not Path(relative).is_absolute(),
            'Require relative archive path')
    path = (root/relative).resolve()
    require(path.is_relative_to(root) and path != root, 'Archive path escapes root')
    return path


def archive_files(root, expected_manifest=None, expected_freeze=None):
    root = Path(root).resolve()
    if expected_manifest:
        require(sha(root/'manifest.json') == expected_manifest, 'Pinned manifest changed')
    if expected_freeze:
        require(sha(root/'freeze.json') == expected_freeze, 'Pinned freeze changed')
    freeze = read(root/'freeze.json')
    require(set(freeze) == {'files'} and isinstance(freeze['files'], dict)
            and freeze['files'], 'Invalid archive freeze')
    result = {}
    for relative, expected in freeze['files'].items():
        require(relative != 'freeze.json', 'Freeze cannot bind itself')
        path = inside(root, relative)
        require(path.is_file() and sha(path) == expected, 'Archived bytes changed: '+relative)
        result[relative] = path
    result['freeze.json'] = root/'freeze.json'
    return result


def copy_archive(root, destination, files):
    for relative, source in files.items():
        path = inside(destination, relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, path)


def seeds_in(value):
    """Conservative declared seeds, including scalar and plural prefixed names."""
    result = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if (key == 'seed' or key.endswith('_seed')) and type(item) is int:
                result.add(item)
            if (key == 'seeds' or key.endswith('_seeds')) and isinstance(item, list):
                result.update(x for x in item if type(x) is int)
            result.update(seeds_in(item))
    elif isinstance(value, list):
        for item in value:
            result.update(seeds_in(item))
    return result


def seed_inventory(repository, excluded=None):
    repository = Path(repository).resolve()
    names = {'protocol.json', 'plan.json', 'manifest.json', 'declaration.json',
             'config.json', 'input-config.json', 'seed-inventory.json'}
    paths = set()
    for directory in (repository/'runs', repository/'examples'):
        for path in directory.rglob('*.json'):
            if excluded is not None and path.resolve().is_relative_to(Path(excluded).resolve()):
                continue
            if (directory.name == 'examples' or path.name in names
                    or path.name.endswith(('-config.json', '_config.json'))
                    or path.name.startswith('config-')
                    or path.parent.name in ('configs', 'sourceconfigs', 'configurations')):
                paths.add(path.resolve())
    rows, seeds = [], set()
    for path in sorted(paths):
        values = seeds_in(read(path))
        rows.append(dict(path=str(path), sha256=sha(path), seeds=sorted(values)))
        seeds.update(values)
    return dict(schema='finite-assembly-seed-inventory-v1', files=rows, seeds=sorted(seeds),
        scope='Recursive run declarations, named config files, configs/sourceconfigs/configurations '
              'directories and all example JSON; named seed fields only. '
              'No trajectory/sample replay. Re-inventory at launch for concurrent reservations.')


def arms(schedule, model_sha):
    base = dict(global_probability=schedule['global_probability'], method='learned',
                model_sha256=model_sha, learned_uniform_weight=schedule['learned_uniform_weight'])
    return dict(local=dict(base, role='local', global_probability=0., method='local-uniform',
                           model_sha256=None, frozen_posterior=None),
        redraw=dict(base, role='independent-redraw', frozen_posterior=None),
        transport=dict(base, role='correlated-transport',
                       frozen_posterior=copy.deepcopy(schedule['frozen_posterior'])))


def validate_schedule(schedule):
    require(isinstance(schedule, dict) and set(schedule) == set(DEFAULT_SCHEDULE),
            'Incomplete or unknown schedule')
    # Reuse pure schedule validation with an internal synthetic window. This
    # window is not emitted, reserved, sampled or used in a scientific contract.
    validate_benchmark(dict(schema='matched-contact-kernel-benchmark-v1',
        physical_identity_sha256='0'*64, observer_definition_sha256='0'*64,
        region_definition_sha256='0'*64, common_schedule=schedule['common'],
        window=dict(burn_sweep=0, end_sweep=1, cadence_sweeps=1),
        preparations=list(contract.PREPARATIONS), streams_per_preparation=4,
        arms=arms(schedule, contract.MODEL_SHA['geometry-only']),
        single_body_attempt_budget='one single-body update per mobile body per sweep'))
    require(all(0 < schedule['common'][key] <= 1 for key in
                ('gca_probability', 'center_shift_probability')),
            'Primary study must retain GCA and center shift')
    return schedule


def job_layout():
    for study in contract.STUDIES:
        for n in (12, 24):
            for boundary in (('spherical',) if study == 'primary-spherical'
                             else ('spherical', 'periodic')):
                for family in contract.MODEL_SHA:
                    block = f'{study}-N{n}-{boundary}-{family}'
                    for arm in ROLES:
                        for preparation in contract.PREPARATIONS:
                            for stream in range(4):
                                yield dict(id=f'{block}-{arm}-{preparation}-r{stream:02d}', block=block,
                                    study=study, bodies=n, boundary=boundary, model_family=family,
                                    arm=arm, preparation=preparation, stream=stream)


def runtime_seed(master_seed, job_id):
    require(type(master_seed) is int and 0 <= master_seed < 2**64, 'Invalid master seed')
    message = f'{SCHEMA}:runtime:{master_seed}:{job_id}'.encode()
    return int.from_bytes(hashlib.sha256(message).digest()[:8], 'big')


def configured(state, job, schedule):
    arm = arms(schedule, contract.MODEL_SHA[job['model_family']])[job['arm']]
    common = copy.deepcopy(schedule['common'])
    if job['study'] == 'boundary-comparison':
        common.update(gca_probability=0., center_shift_probability=0.)
    return dict(shape='../inputs/shape.json', initial_poses=copy.deepcopy(state['initial_poses']),
        box_lengths=state['box_lengths'], boundary=state['boundary'], depletant_radius=1.5,
        reservoir_density=.035, fixed_body_indices=[], seed_labels=state['seed_labels'], seed=job['seed'],
        global_probability=arm['global_probability'], learned_uniform_weight=arm['learned_uniform_weight'],
        frozen_posterior=arm['frozen_posterior'], **common,
        metadata=dict(schema=SCHEMA, job_id=job['id'], preparation=job['preparation'], stream=job['stream'],
            model_family=job['model_family'], model_sha256=arm['model_sha256'],
            initial_poses_sha256=state['initial_poses_sha256'], preparation_equilibrated=False,
            proposal_training_feedback=False, timing_window_selected=False))


def state_identity(state):
    return dict(shape_sha256=contract.SHAPE_SHA, bodies=len(state['initial_poses']),
        boundary=state['boundary'], box_lengths=state['box_lengths'], depletant_radius=1.5,
        reservoir_density=.035, fixed_body_indices=[], measure=contract.MEASURE)


def argv_template(job):
    result = [dict(parameter='executable', type='path'), 'run', '--config', job['config'],
              '--method', 'local-uniform' if job['arm'] == 'local' else 'learned']
    if job['model'] is not None:
        result += ['--model', job['model']]
    return result + ['--out', dict(parameter='output_root', type='path', child=job['id']),
        '--sweeps', dict(parameter='end_sweep', type='positive_integer'),
        '--sample-every', dict(parameter='cadence_sweeps', type='positive_integer')]


def binding(root, relative):
    return dict(path=relative, sha256=sha(inside(root, relative)))


def authenticate_assets(repository, starts, observers):
    root, manifest, _, dependencies, states, _, _, _ = authenticated_inputs(starts)
    start_files = archive_files(root)
    observer_files = archive_files(observers, OBSERVER_MANIFEST_SHA, OBSERVER_FREEZE_SHA)
    om = read(Path(observers)/'manifest.json')
    require(om['complete'] and om['timing_windows'] is None and not om['production_ready']
            and om['starting_states'] == 48 and om['physical_draws'] == 0,
            'Require completed observer-only preparation')
    require(read(Path(observers)/'starting-states-manifest.json') == manifest,
            'Observer uses different prepared states')
    definitions = {}
    for entry in om['definitions']:
        d = read(inside(observers, entry['path']))
        require(sha(inside(observers, entry['path'])) == entry['sha256']
                and d['physical_identity_sha256'] == digest(d['physical_identity']),
                'Observer identity mismatch')
        key = entry['bodies'], entry['boundary']
        require(key not in definitions, 'Repeated observer target')
        contract.target_volume(d['physical_identity'], *key)
        require(all(state_identity(state) == d['physical_identity']
                    for k, state in states.items() if (k[0], k[3]) == key),
                'Observer physical identity differs from prepared states')
        for relative, expected in d['source_sha256'].items():
            p = (inside(observers, entry['path']).parent/relative).resolve()
            require(p.is_relative_to(Path(observers).resolve()) and sha(p) == expected,
                    'Observer source closure differs')
        definitions[key] = entry['path']
    require(set(definitions) == {(n, b) for n in (12, 24) for b in ('spherical', 'periodic')},
            'Incomplete observer targets')
    bank = Path(repository)/'runs/native-blind-memory-proposal-preparation-20260921'
    model_files = archive_files(bank, contract.BLIND_MANIFEST_SHA, contract.BLIND_FREEZE_SHA)
    bm = read(bank/'manifest.json')
    require(bm['source_slot_order'] == [dict(replicate=i, slot=j) for i in range(4) for j in range(16)]
            and bm['base_components'] == 64 and bm['virtual_components'] == 128
            and bm['shape_sha256'] == contract.SHAPE_SHA, 'Incomplete native-blind model source')
    native = Path(repository)/'examples/frozen-coverage-reciprocal-mixture.json'
    require(sha(bank/'model.json') == contract.MODEL_SHA['geometry-only']
            and sha(native) == contract.MODEL_SHA['native-informed'], 'Pinned model changed')
    return dict(starts=root, start_files=start_files, states=states, definitions=definitions,
        observers=Path(observers).resolve(), observer_files=observer_files,
        blind=bank, blind_files=model_files, native=native, dependencies=dependencies)


def source_bindings(repository):
    repository = Path(repository).resolve()
    receipt = repository/'runs/periodic-reciprocal-validation-20260924/receipt.json'
    value = read(receipt)
    require(value['complete'] is True and value['rust_tests']['passed'] == 41
            and value['rust_tests']['no_release_binary_built'] is True,
            'Missing completed periodic validation receipt')
    result = {str(receipt): sha(receipt)}
    for relative, expected in value['sources'].items():
        path = inside(repository, relative)
        require(sha(path) == expected, 'Validated source changed: '+relative)
        result[str(path)] = expected
    for relative in ('Cargo.toml', 'Cargo.lock', 'build.rs', 'vendor/README.md'):
        path = repository/relative
        result[str(path)] = sha(path)
    for path in (repository/'src').rglob('*.rs'):
        result[str(path.resolve())] = sha(path)
    return result


def prepare(repository, starts, observers, out, *, master_seed=152101010, schedule_path=None):
    repository, out = Path(repository).resolve(), Path(out).resolve()
    require(not out.exists(), 'Fresh config-bank directory required; no overwrite or retry')
    schedule = validate_schedule(read(schedule_path) if schedule_path else copy.deepcopy(DEFAULT_SCHEDULE))
    assets = authenticate_assets(repository, starts, observers)
    source = source_bindings(repository)
    inventory = seed_inventory(repository, excluded=out)
    jobs = [dict(job, seed=runtime_seed(master_seed, job['id'])) for job in job_layout()]
    seeds = [j['seed'] for j in jobs]
    require(len(set(seeds)) == 432 and not set(seeds).intersection(inventory['seeds']),
            'Runtime seeds collide with each other or a historical declaration')
    for state in assets['states'].values():
        require(state['preparation_seed'] not in seeds, 'Runtime/construction seed collision')
    dependencies = dict(source)
    for files in (assets['start_files'], assets['observer_files'], assets['blind_files']):
        dependencies.update({str(p.resolve()): sha(p) for p in files.values()})
    dependencies[str(assets['native'].resolve())] = sha(assets['native'])
    dependencies.update({row['path']: row['sha256'] for row in inventory['files']})
    if schedule_path:
        dependencies[str(Path(schedule_path).resolve())] = sha(schedule_path)
    try:
        out.mkdir(parents=True)
        copy_archive(assets['starts'], out/'inputs/starts', assets['start_files'])
        copy_archive(assets['observers'], out/'inputs/observer', assets['observer_files'])
        copy_archive(assets['blind'], out/'inputs/blind', assets['blind_files'])
        shutil.copy2(assets['starts']/'inputs/shape.json', out/'inputs/shape.json')
        shutil.copy2(assets['native'], out/'inputs/native-model.json')
        write(out/'schedule.json', schedule)
        for row in inventory['files']:
            path = out/'seed-declarations'/(row['sha256']+'.json')
            if not path.exists():
                path.parent.mkdir(exist_ok=True)
                shutil.copy2(row['path'], path)
        write(out/'seed-inventory.json', inventory)
        for path, expected in source.items():
            destination = out/'validated-source'/Path(path).relative_to(repository)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            require(sha(destination) == expected, 'Validated source changed while copying')
        sources = local_dependencies([Path(__file__), Path(__file__).with_name('test_prepare_finite_assembly_config_bank.py')])
        for name, path in sources.items():
            target = out/'source'/name
            target.parent.mkdir(exist_ok=True)
            shutil.copy2(path, target)
            dependencies[str(path)] = sha(path)
            require(sha(target) == dependencies[str(path)], 'Preparation source changed while copying')
        for job in jobs:
            n, boundary = job['bodies'], job['boundary']
            state = assets['states'][n, job['preparation'], job['stream'], boundary]
            state_id = f'N{n}-{job["preparation"]}-r{job["stream"]:02d}-{boundary}'
            job['state'] = binding(out, 'inputs/starts/starts/'+state_id+'.json')
            job['saved_geometry_check'] = binding(out, 'inputs/starts/starts/'+state_id+'-validation.json')
            job['initial_poses_sha256'] = state['initial_poses_sha256']
            job['observer'] = binding(out, 'inputs/observer/'+assets['definitions'][n, boundary])
            definition = read(inside(out, job['observer']['path']))
            job['observer_definition_sha256'] = digest(definition)
            job['region_definition_sha256'] = digest(definition['regions'])
            job['model'] = (None if job['arm'] == 'local' else binding(out,
                'inputs/blind/model.json' if job['model_family'] == 'geometry-only' else 'inputs/native-model.json'))
            config_path = 'configs/'+job['id']+'.json'
            write(out/config_path, configured(state, job, schedule))
            job['config'] = binding(out, config_path)
            job['argv_template'] = argv_template(job)
        result = dict(schema=SCHEMA, repository=str(repository), complete=True,
            production_ready=False, production_authorized=False,
            window=None, executable=None, dispatcher=None, unresolved=UNRESOLVED,
            master_seed=master_seed, seeds=seeds,
            seed_derivation='First eight SHA256 bytes, big-endian, of schema:runtime:master_seed:job_id',
            schedule=binding(out, 'schedule.json'), schedule_source='explicit-file' if schedule_path else 'documented-historical-N12',
            blocks=12, jobs=jobs, sources=dependencies,
            starting_states_manifest_sha256=sha(assets['starts']/'manifest.json'),
            observer_manifest_sha256=OBSERVER_MANIFEST_SHA,
            validated_source=source,
            preparation_sources={name: sha(out/'source'/name) for name in sources},
            reused_geometry_checks=48, physical_draws=0, native_classifier_calls=0,
            interbody_geometry_replays=0, maximum_physical_jobs=8, maximum_total_workers=32,
            scope='Input preparation only. No observation window, executable, launch, equilibrium or stability claim.')
        for path, expected in dependencies.items():
            require(sha(path) == expected, 'Dependency changed during preparation: '+path)
        write(out/'manifest.json', result)
        return result
    except BaseException as error:
        if out.exists():
            write(out/'failure.json', dict(complete=False, error_type=type(error).__name__, message=str(error)))
        raise
    finally:
        if out.exists():
            write(out/'freeze.json', dict(files={p.relative_to(out).as_posix(): sha(p)
                for p in sorted(out.rglob('*')) if p.is_file() and p != out/'freeze.json'}))


def validate_bank(root, *, check_external=True):
    root = Path(root).resolve()
    archive_files(root)
    require(not (root/'failure.json').exists(), 'Failed bank cannot be used')
    manifest = read(root/'manifest.json')
    require(manifest['schema'] == SCHEMA and manifest['complete'] is True
            and manifest['production_ready'] is False and manifest['production_authorized'] is False
            and manifest['window'] is None
            and manifest['executable'] is None and manifest['dispatcher'] is None
            and manifest['unresolved'] == UNRESOLVED, 'Bank cannot select timing or authorize production')
    repository = Path(manifest['repository'])
    for source, expected in manifest['validated_source'].items():
        require(Path(source).is_relative_to(repository), 'Source lies outside recorded repository')
        path = inside(root/'validated-source', Path(source).relative_to(repository).as_posix())
        require(sha(path) == expected, 'Copied validated source differs')
    for name, expected in manifest['preparation_sources'].items():
        require(sha(inside(root/'source', name)) == expected, 'Copied preparation source differs')
    schedule = validate_schedule(read(root/'schedule.json'))
    require(manifest['schedule'] == binding(root, 'schedule.json'), 'Schedule binding differs')
    require(sha(root/'inputs/shape.json') == contract.SHAPE_SHA, 'Shape changed')
    archive_files(root/'inputs/blind', contract.BLIND_MANIFEST_SHA, contract.BLIND_FREEZE_SHA)
    archive_files(root/'inputs/observer', OBSERVER_MANIFEST_SHA, OBSERVER_FREEZE_SHA)
    require(sha(root/'inputs/blind/model.json') == contract.MODEL_SHA['geometry-only']
            and sha(root/'inputs/native-model.json') == contract.MODEL_SHA['native-informed'], 'Model changed')
    _, _, _, _, states, _, _, _ = authenticated_inputs(root/'inputs/starts')
    inventory = read(root/'seed-inventory.json')
    previous = set()
    for row in inventory['files']:
        path = root/'seed-declarations'/(row['sha256']+'.json')
        require(sha(path) == row['sha256'] and sorted(seeds_in(read(path))) == row['seeds'],
                'Historical seed declaration changed')
        previous.update(row['seeds'])
    require(sorted(previous) == inventory['seeds'], 'Historical seed union differs')
    layout = list(job_layout())
    require(manifest['blocks'] == 12 and len(manifest['jobs']) == 432, 'Incomplete allocation')
    actual_seeds = []
    for job, expected in zip(manifest['jobs'], layout):
        require({k: job[k] for k in expected} == expected, 'Allocation/order changed')
        seed = runtime_seed(manifest['master_seed'], job['id'])
        require(job['seed'] == seed and seed not in previous, 'Seed changed or reused')
        actual_seeds.append(seed)
        state = states[job['bodies'], job['preparation'], job['stream'], job['boundary']]
        state_id = f'N{job["bodies"]}-{job["preparation"]}-r{job["stream"]:02d}-{job["boundary"]}'
        require(job['state']['path'] == 'inputs/starts/starts/'+state_id+'.json'
                and job['saved_geometry_check']['path'] == 'inputs/starts/starts/'+state_id+'-validation.json'
                and read(inside(root, job['state']['path'])) == state,
                'Wrong saved starting state or geometry record')
        require(seed != state['preparation_seed'], 'Runtime/construction seed collision')
        require(job['initial_poses_sha256'] == digest(state['initial_poses']), 'Pose binding differs')
        for key in ('state', 'saved_geometry_check', 'observer', 'config'):
            require(job[key] == binding(root, job[key]['path']), 'Job binding changed: '+key)
        require(read(inside(root, job['config']['path'])) == configured(state, job, schedule),
                'Config differs from frozen starting state/schedule')
        definition = read(inside(root, job['observer']['path']))
        require(job['observer_definition_sha256'] == digest(definition)
                and job['region_definition_sha256'] == digest(definition['regions'])
                and definition['physical_identity'] == state_identity(state), 'Observer target/digest differs')
        expected_model = None if job['arm'] == 'local' else binding(root,
            'inputs/blind/model.json' if job['model_family'] == 'geometry-only' else 'inputs/native-model.json')
        require(job['model'] == expected_model and job['argv_template'] == argv_template(job),
                'Wrong model or executable command template')
    require(manifest['seeds'] == actual_seeds and len(set(actual_seeds)) == 432, 'Incomplete seed allocation')
    if check_external:
        for path, expected in manifest['sources'].items():
            require(sha(path) == expected, 'External source changed: '+path)
    return dict(schema=SCHEMA, complete=True, configs=432, independent_seeds=432,
        geometry_checks_reused=48, physical_draws=0, window=None, executable=None,
        production_ready=False, external_sources_rechecked=check_external,
        manifest_sha256=sha(root/'manifest.json'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--starts', type=Path, required=True)
    parser.add_argument('--observers', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--master-seed', type=int, default=152101010)
    parser.add_argument('--schedule', type=Path)
    args = parser.parse_args()
    prepare(args.repository, args.starts, args.observers, args.out,
            master_seed=args.master_seed, schedule_path=args.schedule)
    print(json.dumps(validate_bank(args.out), indent=2))


if __name__ == '__main__':
    main()
