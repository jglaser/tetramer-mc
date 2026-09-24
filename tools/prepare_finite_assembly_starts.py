#!/usr/bin/env python3
"""Prepare physical starting poses, never Monte Carlo trajectories or proposals."""
from __future__ import annotations
import os
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
              'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_name] = '1'
import argparse
import copy
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import shutil
import sys
import time
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from contact_benchmark_contract import digest

ROOT = Path(__file__).resolve().parents[1]
SHAPE_SHA = 'c7442034e7ffa4627b67c57a81117f0e9bc2d389ecf956a83dac2a5e915554c9'
SEED_SHA = '7cbea7c17634605e7fc8863dff3a4755c8b3bddd4b753a7f177141ebc23437f1'
NATIVE_SHA = '5cf55ebe0c0f78ec6b8cdc745cac938b0fadfa3732e1e1c730864c62c651d7a9'
PREPARATIONS = ('dispersed', 'competing-aggregate', 'native-seeded')
RD, ACTIVITY, CONCENTRATION = 1.5, .035, 106.8
RAY_GAP = .1
MAX_TRIALS = 10000


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def sphere_data(xyz, radii):
    xyz, radii = np.asarray(xyz, float), np.asarray(radii, float)
    require(xyz.ndim == 2 and xyz.shape[1] == 3 and len(xyz) > 0
            and radii.shape == (len(xyz),) and np.isfinite(xyz).all()
            and np.isfinite(radii).all() and (radii > 0).all(), 'Invalid sphere union')
    return xyz, radii


def last_core_exit(fixed_xyz, fixed_radii, moving_xyz, moving_radii, direction):
    """Last forward closed-core intersection for moving_xyz + s*unit(direction).

    Projected KD candidates are a conservative acceleration only. For every
    retained pair solve the exact one-dimensional sphere quadratic. Multiple
    components, cavities and tangencies require no convexity assumption.
    """
    fixed, fr = sphere_data(fixed_xyz, fixed_radii)
    moving, mr = sphere_data(moving_xyz, moving_radii)
    n = np.asarray(direction, float)
    require(n.shape == (3,) and np.isfinite(n).all(), 'Invalid ray direction')
    norm = float(np.linalg.norm(n))
    require(math.isfinite(norm) and norm > 0, 'Invalid ray direction')
    n = n/norm
    axis = np.eye(3)[int(np.argmin(np.abs(n)))]
    e1 = np.cross(n, axis); e1 /= np.linalg.norm(e1)
    basis = np.column_stack((e1, np.cross(n, e1)))
    f2, m2 = fixed@basis, moving@basis
    tree = cKDTree(f2)
    guard = 1024*np.finfo(float).eps*(1+np.max(np.abs(fixed))
                                    + np.max(np.abs(moving)) + fr.max()+mr.max())
    last = 0.
    for start in range(0, len(moving), 128):
        neighbors = tree.query_ball_point(m2[start:start+128],
            mr[start:start+128]+fr.max()+guard, eps=0., workers=1)
        for offset, ids in enumerate(neighbors):
            if not ids:
                continue
            j = start+offset
            delta = fixed[ids]-moving[j]
            along = delta@n
            # Transverse coordinates avoid subtracting nearly equal large
            # longitudinal squared distances at grazing intersections.
            transverse = delta@basis
            perpendicular2 = np.sum(transverse*transverse, axis=1)
            discriminant = (fr[ids]+mr[j])**2-perpendicular2
            valid = discriminant >= 0.
            if np.any(valid):
                last = max(last, float(np.max(along[valid]+np.sqrt(discriminant[valid]))))
    return last


def quaternion(rng):
    q = rng.normal(size=4)
    return q/np.linalg.norm(q)


def matrix(q):
    return Rotation.from_quat(np.asarray(q)[[1, 2, 3, 0]]).as_matrix()


def pose(position, q):
    return dict(position=np.asarray(position).tolist(), orientation=np.asarray(q).tolist())


def world_atoms(shape_xyz, poses):
    return np.vstack([shape_xyz@matrix(p['orientation']).T+p['position'] for p in poses])


def dimensions(n):
    volume = n*1e27/(CONCENTRATION*1e-6*6.02214076e23)
    return dict(sphere_radius=(3*volume/(4*math.pi))**(1/3),
                box_lengths=[volume**(1/3)]*3, volume_A3=volume)


def prepare_family(shape, seed_poses, kind, n, rng, geometry, record=None):
    """One common-coordinate preparation fitting sphere and periodic vessel.

    Returns poses and every construction event. Native labels never guide
    construction. Periodic coordinates are wrapped only when exporting states.
    """
    require(kind in PREPARATIONS and type(n) is int and n >= 1, 'Invalid preparation')
    xyz, radii = sphere_data([a['center'] for a in shape['atoms']],
                            [a['radius'] for a in shape['atoms']])
    bound = float(np.max(np.linalg.norm(xyz, axis=1)+radii))
    lengths = np.asarray(geometry['box_lengths'], float)
    sphere_radius = float(geometry['sphere_radius'])
    require(lengths.shape == (3,) and np.isfinite(lengths).all()
            and (lengths > 4*(bound+RD)).all()
            and math.isfinite(sphere_radius) and sphere_radius > bound+RD+1,
            'Insufficient common preparation domain')
    common_radius = min(sphere_radius, .5*float(lengths.min()))
    events = []

    def emit(event):
        event = dict(index=len(events), **event)
        events.append(event)
        if record is not None:
            record(event)

    def place_fragment(fragment):
        positions = np.asarray([p['position'] for p in fragment], float)
        centroid = positions.mean(axis=0)
        centered = positions-centroid
        extent = float(np.max(np.linalg.norm(centered, axis=1)))+bound+RD+1.
        require(extent < common_radius, 'Fragment cannot fit the common preparation ball')
        global_q = quaternion(rng); rotation = matrix(global_q)
        direction = rng.normal(size=3); direction /= np.linalg.norm(direction)
        center = direction*(common_radius-extent)*rng.random()**(1/3)
        result = []
        for p, position in zip(fragment, centered):
            q = Rotation.from_matrix(rotation@matrix(p['orientation'])).as_quat()[[3, 0, 1, 2]]
            result.append(pose(rotation@position+center, q))
        emit(dict(kind='proper-fragment-isometry', source_centroid=centroid.tolist(),
            orientation=global_q.tolist(), center=center.tolist(), exclusion_bound_A=extent,
            bodies=len(fragment), accepted=True))
        return result

    if kind == 'native-seeded':
        require(seed_poses is not None and len(seed_poses) == 8 and n >= 8,
                'Exactly eight prescribed seed bodies required')
        poses = place_fragment(copy.deepcopy(seed_poses))
    elif kind == 'competing-aggregate':
        poses = [pose(np.zeros(3), quaternion(rng))]
        emit(dict(kind='aggregate-root', candidate=copy.deepcopy(poses[0]), accepted=True))
        for body in range(1, n):
            fixed = world_atoms(xyz, poses)
            fixed_radii = np.tile(radii, len(poses))
            for attempt in range(MAX_TRIALS):
                parent = int(rng.integers(len(poses)))
                origin = np.asarray(poses[parent]['position'])
                direction = rng.normal(size=3); direction /= np.linalg.norm(direction)
                q = quaternion(rng); moving = xyz@matrix(q).T
                exit_distance = last_core_exit(fixed-origin, fixed_radii, moving, radii, direction)
                candidate = pose(origin+direction*(exit_distance+RAY_GAP), q)
                positions = np.asarray([p['position'] for p in poses]+[candidate['position']])
                extent = np.max(np.linalg.norm(positions-positions.mean(axis=0), axis=1))+bound+RD+1.
                accepted = bool(extent < common_radius)
                emit(dict(kind='aggregate-ray', body=body, attempt=attempt, parent=parent,
                    direction=direction.tolist(), core_exit_A=exit_distance, gap_A=RAY_GAP,
                    candidate=candidate, proposed_exclusion_extent_A=float(extent), accepted=accepted,
                    reason='inside-common-ball' if accepted else 'outside-common-ball'))
                if accepted:
                    poses.append(candidate)
                    break
            else:
                raise ValueError('Aggregate preparation exhausted its fixed geometric trial budget')
        poses = place_fragment(poses)
    else:
        poses = []
    if kind != 'competing-aggregate':
        half = lengths/2-bound-RD-1.
        for body in range(len(poses), n):
            for attempt in range(MAX_TRIALS):
                position = rng.uniform(-half, half)
                wall = bool(np.linalg.norm(position)+bound+RD+1. < sphere_radius)
                minimum = None
                separated = True
                if poses:
                    delta = np.asarray([p['position'] for p in poses])-position
                    delta -= lengths*np.floor(delta/lengths+.5)
                    minimum = float(np.min(np.linalg.norm(delta, axis=1)))
                    separated = minimum > 2*(bound+RD)+1e-7
                accepted = wall and separated
                event = dict(kind='free-placement', body=body, attempt=attempt,
                    position=position.tolist(), minimum_periodic_center_distance_A=minimum,
                    accepted=accepted, reason=('accepted' if accepted else
                        'common-wall-bound' if not wall else 'exclusion-bound-separation'))
                if accepted:
                    candidate = pose(position, quaternion(rng))
                    event['candidate'] = candidate
                    poses.append(candidate)
                emit(event)
                if accepted:
                    break
            else:
                raise ValueError('Separated preparation exhausted its fixed geometric trial budget')
    return poses, events


def preparation_seed(master, n, kind, stream):
    return int.from_bytes(hashlib.sha256(
        f'finite-assembly-start-v1:{master}:{n}:{kind}:{stream}'.encode()).digest()[:8], 'little')


def prepare(out, master_seed=151001007):
    from native_contact_regions import NativeContactRegions
    from validate_finite_assembly_starts import validate_start
    from prepare_shoulder_docking_benchmark import local_dependencies
    out = Path(out).resolve()
    require(not out.exists(), 'Fresh preparation directory required')
    require(type(master_seed) is int and 0 <= master_seed < 2**64, 'Invalid preparation seed')
    paths = dict(shape=ROOT/'examples/tetramer-shape.json', seed=ROOT/'examples/seeded.json',
        native=ROOT/'runs/native-excluded-smc-control-20260924/inputs/native/definition.json',
        seed_check=ROOT/'runs/finite-assembly-seed-preflight-20260924/validation.json')
    for name, expected in dict(shape=SHAPE_SHA, seed=SEED_SHA, native=NATIVE_SHA,
            seed_check='472b7c829cb8bb6a7e4f3c843e2f19b5fb2697776650dbd61d5c05174285c88f').items():
        require(sha(paths[name]) == expected, 'Frozen construction input differs: '+name)
    seeds = [preparation_seed(master_seed, n, kind, stream)
             for n in (12, 24) for kind in PREPARATIONS for stream in range(4)]
    require(len(set(seeds)) == 24, 'Preparation seed collision')
    out.mkdir(); (out/'inputs').mkdir(); (out/'starts').mkdir(); (out/'events').mkdir()
    shutil.copy2(paths['shape'], out/'inputs/shape.json')
    shutil.copy2(paths['seed'], out/'inputs/native-seed-source.json')
    shutil.copytree(paths['native'].parent, out/'inputs/native')
    shutil.copytree(paths['seed_check'].parent, out/'inputs/native-seed-preflight')
    source = local_dependencies([Path(__file__), Path(__file__).with_name('validate_finite_assembly_starts.py'),
                                  Path(__file__).with_name('test_finite_assembly_starts.py')])
    (out/'source').mkdir()
    for name, path in source.items():
        shutil.copy2(path, out/'source'/name)
    plan = dict(schema='finite-assembly-geometric-start-plan-v1', sizes=[12,24],
        preparations=list(PREPARATIONS), streams=4, boundaries=['spherical','periodic'],
        canonical_preparations=24, boundary_states=48, master_seed=master_seed,
        preparation_seeds=seeds, concentration_uM=CONCENTRATION, depletant_radius=RD,
        reservoir_density=ACTIVITY, native_seed_indices=list(range(8)),
        aggregate_outward_ray_gap_A=RAY_GAP, geometric_trials_per_body=MAX_TRIALS,
        initialization_law='Nonequilibrium geometric construction; no equilibrium or proposal-density claim.',
        common_boundary_preparation='Identical centered coordinates; periodic coordinates wrapped only at export.',
        scope='Starting-pose assets only. No proposal model, runtime MC seeds, burn length, production budget or physical gate supplied.',
        input_sha256={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()},
        python=sys.version, numpy=version('numpy'), scipy=version('scipy'),
        preparation_rng='numpy.random.Generator(numpy.random.PCG64(seed))')
    write_new(out/'plan.json', plan)
    states = []
    started = time.monotonic()
    try:
        shape = json.loads((out/'inputs/shape.json').read_text())
        seed = json.loads((out/'inputs/native-seed-source.json').read_text())
        require(seed['seed_labels'] == list(range(8)), 'Prescribed seed indices changed')
        native = NativeContactRegions(out/'inputs/native/definition.json')
        for n in (12, 24):
            geometry = dimensions(n); lengths = np.asarray(geometry['box_lengths'])
            for kind in PREPARATIONS:
                for stream in range(4):
                    name = f'N{n}-{kind}-r{stream:02d}'
                    rng_seed = preparation_seed(master_seed, n, kind, stream)
                    with (out/'events'/(name+'.jsonl')).open('x') as log:
                        def record(event):
                            log.write(json.dumps(event, allow_nan=False)+'\n'); log.flush()
                        poses, events = prepare_family(shape, seed['initial_poses'][:8], kind, n,
                            np.random.Generator(np.random.PCG64(rng_seed)), geometry, record)
                    canonical_hash = digest(poses)
                    for boundary in ('spherical','periodic'):
                        exported = copy.deepcopy(poses)
                        if boundary == 'periodic':
                            for p in exported:
                                p['position'] = np.mod(p['position'], lengths).tolist()
                        wall = (dict(kind='spherical', radius=geometry['sphere_radius'])
                                if boundary=='spherical' else dict(kind='periodic'))
                        box = [2*geometry['sphere_radius']]*3 if boundary=='spherical' else lengths.tolist()
                        state = dict(schema='finite-assembly-initial-state-v1', preparation=kind,
                            bodies=n, stream=stream, preparation_seed=rng_seed,
                            shape='../inputs/shape.json', shape_sha256=SHAPE_SHA, initial_poses=exported,
                            initial_poses_sha256=digest(exported), canonical_poses_sha256=canonical_hash,
                            boundary=wall, box_lengths=box, depletant_radius=RD, reservoir_density=ACTIVITY,
                            fixed_body_indices=[], seed_labels=list(range(8)) if kind=='native-seeded' else [],
                            preparation_equilibrated=False, proposal_training_feedback=False)
                        filename=name+'-'+boundary+'.json'
                        write_new(out/'starts'/filename, state)
                        report = validate_start(shape, exported, wall, box, native, kind)
                        write_new(out/'starts'/(filename[:-5]+'-validation.json'), report)
                        require(report['passed'] is True,
                            'Independent start validation failed: '+filename+': '+str(report['failure_reasons']))
                        states.append(dict(id=filename[:-5], state='starts/'+filename,
                            state_sha256=sha(out/'starts'/filename),
                            validation='starts/'+filename[:-5]+'-validation.json',
                            construction_events=len(events), rejected_construction_events=sum(not e['accepted'] for e in events)))
                        print(json.dumps(dict(validated=states[-1]['id'], bodies=n)), flush=True)
        for name, expected in plan['input_sha256'].items():
            require(sha(out/name)==expected, 'Construction input changed: '+name)
        write_new(out/'manifest.json', dict(schema='finite-assembly-geometric-starts-v1',
            complete=True, states=states, canonical_preparations=24, boundary_states=len(states),
            plan_sha256=sha(out/'plan.json'), physical_draws=0, production_ready=False,
            wall_seconds=time.monotonic()-started,
            outstanding=['Frozen observation definitions and production windows', 'Matched kernel configurations and fresh MC seeds',
                         'Physical regional/full-vessel checks and production launch prerequisites']))
    except BaseException as error:
        write_new(out/'failure.json', dict(complete=False, error=repr(error), validated_states=states,
            physical_draws=0, wall_seconds=time.monotonic()-started))
        raise
    finally:
        write_new(out/'freeze.json', dict(files={str(p.relative_to(out)):sha(p)
            for p in out.rglob('*') if p.is_file()}))
    return states


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--master-seed', type=int, default=151001007)
    args = parser.parse_args()
    prepare(args.out, args.master_seed)
