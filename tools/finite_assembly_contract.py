"""Inert finite-system assembly design and input-binding checks; never launch MC.

Compose separate single-target contracts. A complete design is not a populated
preparation, geometry certificate, convergence result, or production permission.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from contact_benchmark_contract import COMMON, digest, finite, require
from contact_benchmark_contract import validate_contract as validate_benchmark

SCHEMA = 'finite-assembly-campaign-v1'
SHAPE_SHA = 'c7442034e7ffa4627b67c57a81117f0e9bc2d389ecf956a83dac2a5e915554c9'
MODEL_SHA = {
    'geometry-only': 'b6d06b0a076d7f3cd4f591d116a77799b2dc4caa3ea6c21081ad5da8a45b3e9b',
    'native-informed': 'feb4011c622c3104bbe909a29685bd7f077e28f0c847f630c69d8fa87939c20e',
}
BLIND_MANIFEST_SHA = 'b34846304687c20c7d671826cef603c5400d03cd036816d3d480dbbc7818fd5a'
BLIND_FREEZE_SHA = '2ea917e076ba0a397ea9a8eb91fa284e1557adb1ab8cb244ed2510a83efd9db2'
PREPARATIONS = ('dispersed', 'competing-aggregate', 'native-seeded')
STUDIES = ('primary-spherical', 'boundary-comparison')
MEASURE = 'd^3t times normalized Haar for each labelled rigid body; ideal bath wall-permeable'
CONCENTRATION_UM = 106.8
CONCENTRATION_TOLERANCE_UM = .05
VOLUME_RELATIVE_TOLERANCE = 1e-12
AVOGADRO = 6.02214076e23
UNSUPPORTED = ('auxiliary_transport', 'reversible_jump', 'contact_memory',
               'conditional_closure', 'atlas_transport', 'atlas_mask', 'assembly_bias')


def keys(value, expected, label):
    require(isinstance(value, dict) and set(value) == set(expected), 'Incomplete/unknown '+label)


def hash_value(value):
    require(isinstance(value, str) and len(value) == 64
            and all(c in '0123456789abcdef' for c in value), 'Invalid SHA256')


def binding(value, expected=None):
    keys(value, ('path', 'sha256'), 'file binding')
    require(isinstance(value['path'], str) and value['path'], 'Empty binding path')
    hash_value(value['sha256'])
    if expected is not None:
        require(value['sha256'] == expected, 'Pinned asset hash differs')


def target_volume(identity, bodies, boundary):
    keys(identity, ('shape_sha256', 'bodies', 'boundary', 'box_lengths',
                    'depletant_radius', 'reservoir_density', 'fixed_body_indices', 'measure'),
         'physical identity')
    require(type(identity['bodies']) is int and identity['bodies'] == bodies,
            'Physical body count differs')
    require(identity['shape_sha256'] == SHAPE_SHA and identity['depletant_radius'] == 1.5
            and identity['reservoir_density'] == .035 and identity['fixed_body_indices'] == []
            and identity['measure'] == MEASURE, 'Fixed physical target or all-mobile law differs')
    lengths = identity['box_lengths']
    require(isinstance(lengths, list) and len(lengths) == 3
            and all(finite(x) and x > 0 for x in lengths), 'Invalid box/display lengths')
    wall = identity['boundary']
    keys(wall, ('kind', 'radius') if boundary == 'spherical' else ('kind',), 'boundary')
    require(wall['kind'] == boundary, 'Boundary identity differs')
    if boundary == 'spherical':
        require(finite(wall['radius']) and wall['radius'] > 0, 'Invalid sphere radius')
        volume = (4 * math.pi / 3) * wall['radius']**3
    else:
        volume = math.prod(lengths)
    require(math.isfinite(volume) and volume > 0, 'Unrepresentable physical volume')
    concentration = bodies / volume * 1e33 / AVOGADRO
    require(abs(concentration-CONCENTRATION_UM) <= CONCENTRATION_TOLERANCE_UM,
            'Tetramer concentration differs from fixed 106.8 uM tolerance')
    return volume


def expected_blocks():
    return {(study, n, boundary, family)
            for study in STUDIES for n in (12, 24)
            for boundary in (('spherical',) if study == 'primary-spherical'
                             else ('spherical', 'periodic'))
            for family in MODEL_SHA}


def validate_contract(plan):
    """Check the entire design, including unfilled config bindings, without I/O."""
    keys(plan, ('schema', 'physical', 'models', 'primary_common_schedule', 'window', 'blocks'),
         'finite assembly contract')
    require(plan['schema'] == SCHEMA, 'Wrong finite assembly schema')
    physical = plan['physical']
    keys(physical, ('shape', 'concentration_uM', 'concentration_tolerance_uM',
                    'volume_relative_tolerance', 'depletant_radius', 'reservoir_density'),
         'physical declaration')
    binding(physical['shape'], SHAPE_SHA)
    for name, expected in dict(concentration_uM=CONCENTRATION_UM,
            concentration_tolerance_uM=CONCENTRATION_TOLERANCE_UM,
            volume_relative_tolerance=VOLUME_RELATIVE_TOLERANCE,
            depletant_radius=1.5, reservoir_density=.035).items():
        require(finite(physical[name]) and physical[name] == expected,
                'Fixed physical constant/tolerance changed: '+name)
    keys(plan['models'], MODEL_SHA, 'model families')
    for family, item in plan['models'].items():
        keys(item, ('model', 'provenance'), 'model declaration')
        binding(item['model'], MODEL_SHA[family])
        if family == 'geometry-only':
            keys(item['provenance'], ('manifest', 'freeze'), 'native-blind provenance')
            binding(item['provenance']['manifest'], BLIND_MANIFEST_SHA)
            binding(item['provenance']['freeze'], BLIND_FREEZE_SHA)
        else:
            require(item['provenance'] is None, 'Native control uses the pinned coverage model')
    keys(plan['primary_common_schedule'], COMMON, 'primary common schedule')
    require(all(finite(plan['primary_common_schedule'][k])
                and 0 < plan['primary_common_schedule'][k] <= 1
                for k in ('gca_probability', 'center_shift_probability')),
            'Primary spherical study retains GCA and center shifts')
    boundary_schedule = dict(plan['primary_common_schedule'], gca_probability=0.,
                             center_shift_probability=0.)
    require(isinstance(plan['blocks'], list), 'Blocks must be a list')
    seen, block_ids, job_ids, seeds = set(), set(), set(), set()
    targets, observers, volumes, initial_hashes, proposals = {}, {}, {}, {}, {}
    bound_jobs = 0
    for block in plan['blocks']:
        keys(block, ('id', 'study', 'bodies', 'boundary', 'model_family', 'physical_identity',
                     'benchmark_contract', 'jobs'), 'study block')
        require(type(block['bodies']) is int, 'Integer body count required')
        key = (block['study'], block['bodies'], block['boundary'], block['model_family'])
        require(key in expected_blocks() and key not in seen, 'Missing/duplicate/unplanned study block')
        seen.add(key)
        require(isinstance(block['id'], str) and block['id'] and block['id'] not in block_ids,
                'Unique block IDs required')
        block_ids.add(block['id'])
        study, n, boundary, family = key
        identity = block['physical_identity']
        volume = target_volume(identity, n, boundary)
        target_key = (n, boundary)
        require(target_key not in targets or targets[target_key] == identity,
                'Identical N/boundary targets differ across studies/model families')
        targets[target_key], volumes[target_key] = identity, volume
        contract = validate_benchmark(block['benchmark_contract'])
        require(contract['physical_identity_sha256'] == digest(identity),
                'Nested contract physical identity differs')
        require(contract['preparations'] == list(PREPARATIONS), 'Exact three preparations required')
        require(contract['window'] == plan['window'], 'Observation windows differ')
        window = contract['window']
        require(window['burn_sweep'] % window['cadence_sweeps'] == 0
                and window['end_sweep'] % window['cadence_sweeps'] == 0,
                'Observation window endpoints must align with saved cadence')
        schedule = plan['primary_common_schedule'] if study == 'primary-spherical' else boundary_schedule
        require(contract['common_schedule'] == schedule, 'Matched study move schedule differs')
        observer = (contract['observer_definition_sha256'], contract['region_definition_sha256'])
        require(target_key not in observers or observers[target_key] == observer,
                'Observer/region definitions differ for the same physical target')
        observers[target_key] = observer
        for arm in contract['arms'].values():
            if arm['role'] != 'local':
                require(arm['model_sha256'] == MODEL_SHA[family], 'Wrong frozen model family')
            # Only model content differs between families, not kernel scheduling.
            proposal = {k: v for k, v in arm.items() if k != 'model_sha256'}
            role = arm['role']
            require(role not in proposals or proposals[role] == proposal,
                    'Proposal settings differ across model/size/boundary blocks')
            proposals[role] = proposal
        require(isinstance(block['jobs'], list), 'Jobs must be a list')
        jobs = set()
        for job in block['jobs']:
            keys(job, ('id', 'arm', 'preparation', 'stream', 'seed', 'config',
                       'initial_poses_sha256', 'invocation'), 'job declaration')
            require(isinstance(job['id'], str) and job['id'] and job['id'] not in job_ids,
                    'Unique job IDs required')
            job_ids.add(job['id'])
            require(type(job['stream']) is int and 0 <= job['stream'] < 4,
                    'Four numbered streams required')
            jk = (job['arm'], job['preparation'], job['stream'])
            require(job['arm'] in contract['arms'] and job['preparation'] in PREPARATIONS
                    and jk not in jobs, 'Unplanned or duplicate arm/preparation/stream')
            jobs.add(jk)
            invocation = job['invocation']
            keys(invocation, ('method', 'sweeps', 'sample_every', 'model_sha256'), 'invocation')
            arm = contract['arms'][job['arm']]
            require(invocation == dict(method=arm['method'], sweeps=window['end_sweep'],
                        sample_every=window['cadence_sweeps'], model_sha256=arm['model_sha256'])
                    and type(invocation['sweeps']) is int
                    and type(invocation['sample_every']) is int,
                    'Declared invocation differs from kernel or observation window')
            require(type(job['seed']) is int and 0 <= job['seed'] < 2**64
                    and job['seed'] not in seeds, 'Every job needs a unique uint64 seed')
            seeds.add(job['seed'])
            if job['config'] is None:
                require(job['initial_poses_sha256'] is None, 'Unfilled config cannot certify poses')
            else:
                binding(job['config'])
                hash_value(job['initial_poses_sha256'])
                pose_key = (n, boundary, job['preparation'], job['stream'])
                require(pose_key not in initial_hashes
                        or initial_hashes[pose_key] == job['initial_poses_sha256'],
                        'Matched controls must use the same declared initial poses')
                initial_hashes[pose_key] = job['initial_poses_sha256']
                bound_jobs += 1
        require(len(jobs) == 36, 'Every block requires all 36 independent jobs')
    require(seen == expected_blocks(), 'Incomplete size/boundary/model study matrix')
    for n in (12, 24):
        require(math.isclose(volumes[n, 'spherical'], volumes[n, 'periodic'],
                            rel_tol=VOLUME_RELATIVE_TOLERANCE, abs_tol=0.),
                'Sphere and periodic vessel volumes differ')
    require(math.isclose(volumes[24, 'spherical'], 2*volumes[12, 'spherical'],
                        rel_tol=VOLUME_RELATIVE_TOLERANCE, abs_tol=0.),
            'N24 must have twice the N12 volume at identical concentration')
    return dict(schema=SCHEMA, complete_design=True, blocks=12, planned_jobs=432,
                bound_configurations=bound_jobs, input_bindings_complete=(bound_jobs == 432),
                bound_files_verified=False, input_files_verified=False, geometry_validated=False,
                production_authorized=False,
                contract_sha256=digest(plan), local_control_policy='duplicated independent streams',
                concentration_uM={str(n): n/volumes[n, 'spherical']*1e33/AVOGADRO for n in (12, 24)},
                unresolved_obligations=['Initial hard/wall validity and preparation-label meaning',
                    'Seed exclusion against historical campaign inventory',
                    'Observer definition/source closure, executable and actual argv/model binding',
                    'Regional/full-vessel gates, mobile equilibrium coverage and production authorization'])


def file_sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def validate_configuration(config, block, job):
    """Pure effective-law/pose checks; file hashes and geometric validity are separate."""
    contract = block['benchmark_contract']
    require(all(config.get(k) is None for k in UNSUPPORTED),
            'This contract supports frozen unbiased kernels only')
    require(config.get('fixed_body_indices', []) == [] and config.get('resume') is None,
            'Fresh all-mobile configuration required')
    poses = config['initial_poses']
    require(isinstance(poses, list) and len(poses) == block['bodies']
            and digest(poses) == job['initial_poses_sha256'], 'Initial pose binding differs')
    for pose in poses:
        keys(pose, ('position', 'orientation'), 'initial pose')
        t, q = pose['position'], pose['orientation']
        require(isinstance(t, list) and len(t) == 3 and all(finite(x) for x in t)
                and isinstance(q, list) and len(q) == 4 and all(finite(x) for x in q)
                and abs(sum(x*x for x in q)-1) <= 1e-10, 'Invalid rigid-body pose')
    require(type(config['seed']) is int and config['seed'] == job['seed'], 'Config seed differs')
    identity = dict(shape_sha256=SHAPE_SHA, bodies=len(poses),
        boundary=config['boundary'], box_lengths=config['box_lengths'],
        depletant_radius=config['depletant_radius'], reservoir_density=config['reservoir_density'],
        fixed_body_indices=config.get('fixed_body_indices', []), measure=MEASURE)
    require(identity == block['physical_identity'], 'Config physical identity differs')
    require({k: config[k] for k in COMMON} == contract['common_schedule'],
            'Config common schedule differs')
    arm = contract['arms'][job['arm']]
    require(config['global_probability'] == arm['global_probability']
            and config['learned_uniform_weight'] == arm['learned_uniform_weight']
            and config.get('frozen_posterior') == arm['frozen_posterior'],
            'Config arm proposal differs')
    return True


def validate_files(plan, base_dir):
    """Verify declared bytes/configuration laws; no atom geometry or simulation."""
    result = validate_contract(plan)
    base_dir = Path(base_dir).resolve()
    checked = {}

    def read_bound(item, relative_to=base_dir):
        binding(item)
        path = (relative_to/item['path']).resolve()
        require(path.is_file(), 'Missing bound input: '+str(path))
        require(file_sha(path) == item['sha256'], 'Input hash mismatch: '+str(path))
        checked[str(path)] = item['sha256']
        return path, json.loads(path.read_text())

    read_bound(plan['physical']['shape'])
    for item in plan['models'].values():
        read_bound(item['model'])
    provenance = plan['models']['geometry-only']['provenance']
    manifest_path, manifest = read_bound(provenance['manifest'])
    freeze_path, freeze = read_bound(provenance['freeze'])
    require(manifest_path.parent == freeze_path.parent, 'Native-blind receipt directory differs')
    archive = manifest_path.parent
    require(isinstance(freeze, dict) and set(freeze) == {'files'}, 'Invalid export freeze')
    for relative, sha256 in freeze['files'].items():
        hash_value(sha256)
        path = (archive/relative).resolve()
        require(path.is_relative_to(archive) and path.is_file(), 'Invalid archived provenance path')
        require(file_sha(path) == sha256, 'Native-blind archive changed: '+relative)
        checked[str(path)] = sha256
    require(freeze['files'].get('manifest.json') == BLIND_MANIFEST_SHA
            and freeze['files'].get('model.json') == MODEL_SHA['geometry-only']
            and manifest['model_sha256'] == MODEL_SHA['geometry-only']
            and manifest['shape_sha256'] == SHAPE_SHA,
            'Pinned native-blind source closure differs')
    require(manifest['source_slot_order'] == [dict(replicate=i, slot=j)
                for i in range(4) for j in range(16)]
            and manifest['base_components'] == 64 and manifest['virtual_components'] == 128,
            'Complete ordered native-blind slot inventory differs')
    # The pinned export is the reviewed all-slot construction, not a boolean claim.
    for block in plan['blocks']:
        contract = block['benchmark_contract']
        for job in block['jobs']:
            if job['config'] is None:
                continue
            path, config = read_bound(job['config'])
            validate_configuration(config, block, job)
            actual_shape = (path.parent/config['shape']).resolve()
            require(actual_shape.is_file() and file_sha(actual_shape) == SHAPE_SHA,
                    'Config shape differs from repaired tetramer')
            checked[str(actual_shape)] = SHAPE_SHA
    for path, expected in checked.items():
        require(file_sha(path) == expected, 'Input changed during validation: '+path)
    result.update(bound_files_verified=True, input_files_verified=result['input_bindings_complete'],
                  verified_file_sha256=checked,
                  native_blind_provenance_verified=True,
                  scope='Read-only byte/configuration checks. No geometry replay, classification, fitting, or physical draws.')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--verify-files', action='store_true')
    args = parser.parse_args()
    plan = json.loads(args.manifest.read_text())
    result = validate_files(plan, args.manifest.resolve().parent) if args.verify_files else validate_contract(plan)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
