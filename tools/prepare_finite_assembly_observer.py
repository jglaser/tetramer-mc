#!/usr/bin/env python3
"""Freeze geometry-only patches and instantaneous assembly observer definitions.

This is asset preparation from completed checks: no trajectories, classifier
calls, geometry replay, model reads, physical draws, or invented timing windows.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import platform
from pathlib import Path
import shutil

import numpy as np
import scipy

from analyze_contact_efficiency import physical_identity
from contact_benchmark_contract import digest
from native_contact_regions import pose_arrays
from summarize_finite_assembly_starts import authenticated_inputs

SCHEMA = 'finite-assembly-observer-definition-v1'
REFERENCE_SIZE = 8
RECONSTRUCTION_TOLERANCE_A = 1e-10
REGION_RULES = [
    dict(id='dispersed', native_registry_resolved=True, native_edge_count=dict(equals=0),
         largest_exclusion=dict(equals=1)),
    dict(id='contact_no_entry', native_registry_resolved=True, native_edge_count=dict(equals=0),
         largest_exclusion=dict(min=2)),
    dict(id='registered_small', native_registry_resolved=True, largest_native_raw=dict(min=2, max=7)),
    dict(id='registered_eight', native_registry_resolved=True, largest_native_raw=dict(equals=8)),
    dict(id='registered_growth', native_registry_resolved=True, largest_native_raw=dict(min=9)),
    dict(id='remaining', native_registry_resolved=False),
]
PATCH_IDS = [f'member{member}:octant{octant}' for member in range(4) for octant in range(8)]
MEASUREMENTS = dict(
    occupancy='Permutation-invariant instantaneous six-region indicators; all bodies included.',
    aggregation='Complete exclusion-component sizes, edge counts and largest component.',
    registry='Complete raw native components and whole-component catalogue/image-lift certification; '
             'retain frustrated/winding components and their raw sizes.',
    certified_size='Largest whole native component certified by the current catalogue/lift test; '
                   'not a maximum compatible subgraph or a global crystal certificate.',
    fingerprint='Tagged (body_i,body_j,patch_i,patch_j) contact-vector apparent ESS per full sampler CPU; '
                'separate from permutation-invariant region/size indicators. Coarse octants can miss finer surface changes.',
    remaining='Any winding or catalogue-frustrated native component puts the complete frame in remaining; '
              'retain known certified components and all original denominators.',
    periodic='Minimum-image pairs under existing box restriction; winding is unsupported ordinary-space '
             'registry, not proof of a nonnative phase. No quotient-space lattice claim.',
    reference_size='Eight is the supplied seed size, not an inferred critical nucleus.',
    limits='No equilibrium, stability, mixing, or physical kinetic conclusion from a prepared state.')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def build_patch_map(shape, monomer, shape_sha256, monomer_sha256):
    """Verify four full atom blocks, then label in the monomer's body frame."""
    atoms = np.asarray([a['center'] for a in monomer['atoms']], float)
    radii = np.asarray([a['radius'] for a in monomer['atoms']], float)
    body = np.asarray([a['center'] for a in shape['atoms']], float)
    body_radii = np.asarray([a['radius'] for a in shape['atoms']], float)
    members = shape['rigid_members']
    require(atoms.ndim == 2 and atoms.shape == (len(radii), 3) and len(radii) > 0
            and len(members) == 4 and body.shape == (4*len(radii), 3)
            and body_radii.shape == (4*len(radii),) and np.isfinite(atoms).all()
            and np.isfinite(radii).all() and np.isfinite(body).all()
            and np.isfinite(body_radii).all() and np.all(radii > 0) and np.all(body_radii > 0),
            'Require four complete finite monomer atom blocks')
    # The monomer body origin is frozen; do not infer or refit a centroid here.
    octants = ((atoms[:, 0] >= 0).astype(int)*4+(atoms[:, 1] >= 0).astype(int)*2
               +(atoms[:, 2] >= 0).astype(int))
    patches, reconstruction = [], []
    for member, pose in enumerate(members):
        position, rotation = pose_arrays(pose)
        predicted = atoms@rotation.T+position
        lo, hi = member*len(atoms), (member+1)*len(atoms)
        error = float(np.max(np.abs(predicted-body[lo:hi])))
        radius_error = float(np.max(np.abs(radii-body_radii[lo:hi])))
        require(error <= RECONSTRUCTION_TOLERANCE_A,
                f'Member {member} atom centers do not reconstruct the frozen body')
        require(np.array_equal(radii, body_radii[lo:hi]),
                f'Member {member} atomic radii differ; no shape repair allowed')
        patches.extend(f'member{member}:octant{int(o)}' for o in octants)
        reconstruction.append(dict(member=member, first_atom=lo, atoms=len(atoms),
                                   maximum_coordinate_error_A=error, maximum_radius_error_A=radius_error))
    counts = Counter(patches)
    patch = dict(schema='body-frame-atom-patch-map-v1', shape_sha256=shape_sha256,
        monomer_shape_sha256=monomer_sha256, atom_patch_ids=patches, patch_dictionary=PATCH_IDS,
        patch_atom_counts={name: counts[name] for name in PATCH_IDS},
        construction=dict(schema='four-member-local-octants-v1', members=4,
            atom_order='Four unchanged consecutive monomer atom blocks in rigid_members order',
            origin='Archived monomer body origin [0,0,0]; no recentering or fitted frame',
            octant='4*(x>=0)+2*(y>=0)+(z>=0), in the archived monomer body coordinates',
            boundary_tie='Zero belongs to the nonnegative half-space',
            full_dictionary=True, filtering=False, native_labels_used=False,
            reconstruction_tolerance_A=RECONSTRUCTION_TOLERANCE_A))
    summary = dict(complete=True, atoms_per_member=len(atoms), atom_count=len(body),
        dictionary_size=32, occupied_patches=sum(counts[name] > 0 for name in PATCH_IDS),
        members=reconstruction, maximum_coordinate_error_A=max(r['maximum_coordinate_error_A'] for r in reconstruction),
        maximum_radius_error_A=max(r['maximum_radius_error_A'] for r in reconstruction),
        atom_center_reconstruction_only=True, interbody_geometry_replays=0,
        native_classifier_calls=0, physical_draws=0)
    return patch, summary


def rule_label(graph):
    """Evaluate the frozen structural selectors independently of graph_summary."""
    observed = dict(graph, native_edge_count=len(graph['native_edges']))
    matches = []
    for rule in REGION_RULES:
        keep = True
        for name, condition in rule.items():
            if name == 'id':
                continue
            value = observed[name]
            if isinstance(condition, bool):
                keep &= value is condition
            else:
                keep &= all(value == bound if op == 'equals' else value >= bound if op == 'min'
                            else value <= bound for op, bound in condition.items())
        if keep:
            matches.append(rule['id'])
    require(len(matches) == 1, 'Frozen region rules are not exhaustive/disjoint on this graph')
    return matches[0]


def source_closure():
    from prepare_shoulder_docking_benchmark import local_dependencies
    folder = Path(__file__).resolve().parent
    required = ['prepare_finite_assembly_observer.py', 'test_prepare_finite_assembly_observer.py',
                'finite_assembly_observer.py', 'analyze_finite_assembly.py',
                'analyze_contact_efficiency.py', 'mobile_posterior_metrics.py',
                'contact_benchmark_contract.py', 'native_contact_regions.py', 'native_graph_consistency.py']
    require(all((folder/name).is_file() for name in required), 'Final observer source closure not available')
    paths = [folder/name for name in required]
    for name in ('test_finite_assembly_observer.py', 'test_analyze_finite_assembly.py'):
        if (folder/name).is_file():
            paths.append(folder/name)
    return local_dependencies(paths)


def _prepare(starts, out):
    from finite_assembly_observer import REGIONS, graph_summary
    require(tuple(rule['id'] for rule in REGION_RULES) == REGIONS, 'Observer region order differs')
    root, manifest, plan, dependencies, states, records, _, _ = authenticated_inputs(starts)
    require(len(read(root/'freeze.json')['files']) == 158, 'Expected exact completed 158-file preparation')
    out = Path(out).resolve()
    require(not out.exists(), 'Fresh observer preparation directory required')
    sources = source_closure()
    source_hashes = {name: sha(path) for name, path in sources.items()}
    shape = read(root/'inputs/shape.json')
    monomer_path = root/'inputs/native/inputs/monomer-shape.json'
    monomer = read(monomer_path)
    require(len(monomer['atoms']) == 1001 and len(shape['atoms']) == 4004,
            'Actual protein requires four complete 1001-atom members')
    patch, reconstruction = build_patch_map(shape, monomer, sha(root/'inputs/shape.json'), sha(monomer_path))
    runtime = dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__)
    out.mkdir(); (out/'source').mkdir(); (out/'definitions').mkdir()
    for name, path in sources.items():
        shutil.copy2(path, out/'source'/name)
    shutil.copytree(root/'inputs/native', out/'native')
    shutil.copy2(root/'inputs/shape.json', out/'shape.json')
    for name in ('manifest.json', 'plan.json', 'freeze.json'):
        shutil.copy2(root/name, out/('starting-states-'+name))
    write(out/'preparation-plan.json', dict(schema='finite-assembly-observer-preparation-plan-v1',
        starting_states_manifest_sha256=sha(root/'manifest.json'),
        starting_states_plan_sha256=sha(root/'plan.json'),
        starting_states_freeze_sha256=sha(root/'freeze.json'),
        regions=REGION_RULES, reference_size=REFERENCE_SIZE, source_sha256=source_hashes, runtime=runtime,
        allocation='Four target definitions; all 48 cached starting-state graphs; no record selection',
        timing_windows=None, physical_draws=0, native_classifier_calls=0, interbody_geometry_replays=0))
    write(out/'patch-map.json', patch)
    write(out/'patch-reconstruction.json', reconstruction)
    write(out/'regions.json', dict(reference_size=REFERENCE_SIZE, regions=REGION_RULES, measurements=MEASUREMENTS))
    definitions = []
    for n in (12, 24):
        for boundary in ('spherical', 'periodic'):
            state = states[n, 'dispersed', 0, boundary]
            identity = physical_identity(state, patch['shape_sha256'])
            require(all(physical_identity(s, patch['shape_sha256']) == identity
                        for key, s in states.items() if key[0] == n and key[3] == boundary),
                    'Preparation changes a target identity')
            definition = dict(schema=SCHEMA, physical_identity=identity,
                physical_identity_sha256=digest(identity),
                patch_map=dict(path='../patch-map.json', sha256=sha(out/'patch-map.json')),
                native_definition=dict(path='../native/definition.json', sha256=sha(out/'native/definition.json')),
                regions=REGION_RULES, reference_size=REFERENCE_SIZE,
                source_sha256={'../source/'+name: value for name, value in source_hashes.items()},
                measurements=MEASUREMENTS, timing_windows=None,
                scope='Frozen measurement law only. No trajectory, production window or equilibrium probability supplied.')
            path = out/'definitions'/f'N{n}-{boundary}.json'
            write(path, definition)
            definitions.append(dict(bodies=n, boundary=boundary, path=path.relative_to(out).as_posix(),
                                    sha256=sha(path), physical_identity_sha256=digest(identity)))
    motifs = read(out/'native/inputs/native-pair-motifs.json')['motifs']
    projections, counts = [], Counter()
    with (out/'starting-state-graphs.jsonl').open('x') as stream:
        for key in sorted(states):
            state, saved = states[key], records[key]
            positions = [p['position'] for p in state['initial_poses']]
            graph = graph_summary(state['bodies'], saved['exclusion']['edges'], saved['native']['keys'],
                motifs, positions, state['box_lengths'] if key[3] == 'periodic' else None)
            label = rule_label(graph)
            require(label == graph['environment'], 'Frozen rules differ from implemented observer')
            expected = {'dispersed': 'dispersed', 'competing-aggregate': 'contact_no_entry',
                        'native-seeded': 'registered_eight'}[state['preparation']]
            require(label == expected, 'Saved initial-state graph falls outside its prescribed region')
            identifier = f'N{key[0]}-{key[1]}-r{key[2]:02d}-{key[3]}'
            entry = dict(id=identifier, bodies=key[0], preparation=key[1], stream=key[2], boundary=key[3],
                state_sha256=sha(root/'starts'/(identifier+'.json')),
                saved_validation_sha256=sha(root/'starts'/(identifier+'-validation.json')),
                graph=graph, origin='Saved exact edge/motif keys; graph projection only, no atom or native classification')
            stream.write(json.dumps(entry, allow_nan=False)+'\n')
            projections.append(identifier); counts[label] += 1
    require(len(projections) == 48 and len(set(projections)) == 48, 'Cached state inventory differs')
    for path, expected in dependencies.items():
        require(sha(path) == expected, 'Starting-state archive changed during preparation')
    for original in sorted((root/'inputs/native').rglob('*')):
        if original.is_file():
            require(sha(original) == sha(out/'native'/original.relative_to(root/'inputs/native')),
                    'Copied native-definition input differs from authenticated source')
    require(sha(root/'inputs/shape.json') == sha(out/'shape.json'), 'Copied shape differs')
    for name, expected in source_hashes.items():
        require(sha(sources[name]) == sha(out/'source'/name) == expected, 'Observer source changed during preparation')
    result = dict(schema='finite-assembly-observer-assets-v1', complete=True, definitions=definitions,
        patch_map_sha256=sha(out/'patch-map.json'), reconstruction=reconstruction,
        starting_states=48, graph_projection_counts={label: counts[label] for label in REGIONS},
        cached_graph_projection_sha256=sha(out/'starting-state-graphs.jsonl'),
        starting_state_dependency_sha256=dependencies, observer_source_sha256=source_hashes, runtime=runtime,
        timing_windows=None, production_ready=False, native_classifier_calls=0,
        interbody_geometry_replays=0, physical_draws=0,
        scope='Frozen measurement assets and initial-state label checks. These imposed starting states do not estimate occupancies.')
    write(out/'manifest.json', result)
    return result


def prepare(starts, out):
    """Fresh-only preparation, retaining every output and freezing failed attempts."""
    out = Path(out).resolve()
    require(not out.exists(), 'Fresh observer preparation directory required')
    try:
        return _prepare(starts, out)
    except BaseException as error:
        if out.exists():
            write(out/'failure.json', dict(complete=False, error_type=type(error).__name__,
                  message=str(error), physical_draws=0, native_classifier_calls=0,
                  interbody_geometry_replays=0))
        raise
    finally:
        if out.exists():
            write(out/'freeze.json', dict(files={p.relative_to(out).as_posix(): sha(p)
                for p in sorted(out.rglob('*')) if p.is_file() and p != out/'freeze.json'}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--starts', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.starts, args.out)
    print(json.dumps(dict(complete=result['complete'], definitions=len(result['definitions']),
                         starting_states=result['starting_states'], physical_draws=0), indent=2))
