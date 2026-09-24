"""Independent, static atomic and instantaneous-native preparation checks.

No sampler, proposal model, fit, pose repair, or input-file mutation is involved.
The caller authenticates/freeze-binds shape and native-definition source bytes.
Geometric or preparation failures are retained as JSON results, not filtered.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
from scipy.spatial import cKDTree

from native_contact_regions import pose_arrays
from native_graph_consistency import NativeGraphConsistency

RD = 1.5
PREPARATIONS = ('dispersed', 'competing-aggregate', 'native-seeded')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def components(count, edges):
    unseen = set(range(count))
    result = []
    while unseen:
        reached = {min(unseen)}
        while True:
            following = reached | {x for a, b in edges if a in reached or b in reached
                                   for x in (a, b)}
            if following == reached:
                break
            reached = following
        result.append(sorted(reached))
        unseen -= reached
    return result


def periodic_lift(positions, lengths, edges):
    """Check that every native component has an ordinary-space image lift."""
    images = {}
    for root in range(len(positions)):
        if root in images:
            continue
        images[root] = np.zeros(3, dtype=np.int64)
        stack = [root]
        while stack:
            i = stack.pop()
            for a, b in edges:
                if i not in (a, b):
                    continue
                j = b if i == a else a
                winding = -np.floor((positions[j]-positions[i])/lengths+.5).astype(np.int64)
                predicted = images[i]+winding
                if j in images:
                    if not np.array_equal(images[j], predicted):
                        return False
                else:
                    images[j] = predicted
                    stack.append(j)
    return True


def validate_start(shape, poses, boundary, box_lengths, native, preparation):
    """Return a complete static check; classify native entry only if hard-valid.

    Exact contact convention: atomic spheres inflated separately by RD intersect
    strictly. Hard geometry uses squared distances with no overlap tolerance.
    Geometry-failure outputs deliberately have native.evaluated=False.
    """
    require(preparation in PREPARATIONS, 'Unknown preparation')
    require(isinstance(poses, list) and len(poses) >= 2, 'At least two poses required')
    require(native is not None, 'Complete instantaneous native classifier required')
    if preparation == 'native-seeded':
        require(len(poses) >= 8, 'Native seed requires fixed bodies 0 through 7')
    atoms = np.asarray([a['center'] for a in shape['atoms']], dtype=float)
    radii = np.asarray([a['radius'] for a in shape['atoms']], dtype=float)
    require(atoms.ndim == 2 and atoms.shape == (len(radii), 3) and len(radii) > 0
            and np.isfinite(atoms).all() and np.isfinite(radii).all()
            and np.all(radii > 0), 'Invalid atomic shape')
    arrays = [pose_arrays(p) for p in poses]
    positions = np.asarray([p for p, _ in arrays])
    rotations = np.asarray([r for _, r in arrays])
    lengths = np.asarray(box_lengths, dtype=float)
    require(lengths.shape == (3,) and np.isfinite(lengths).all() and np.all(lengths > 0),
            'Invalid box/display lengths')
    require(isinstance(boundary, dict) and boundary.get('kind') in ('spherical', 'periodic'),
            'Unsupported boundary')
    spherical = boundary['kind'] == 'spherical'
    if spherical:
        require(set(boundary) == {'kind', 'radius'} and type(boundary['radius']) in (float, int)
                and math.isfinite(boundary['radius']) and boundary['radius'] > 0,
                'Invalid spherical radius')
    else:
        require(set(boundary) == {'kind'}, 'Unexpected periodic boundary fields')
    bound = float(np.max(np.linalg.norm(atoms, axis=1)+radii))
    require(math.isfinite(bound), 'Unrepresentable body bound')
    if not spherical:
        require(np.all(lengths > 4*(bound+RD)), 'Unique minimum-image contact bound required')
    # Match the explicit arithmetic grouping, rather than implicit BLAS choices.
    rotated = ((rotations[:, None, :, 0]*atoms[None, :, 0, None]
                + rotations[:, None, :, 1]*atoms[None, :, 1, None])
               + rotations[:, None, :, 2]*atoms[None, :, 2, None])
    world = rotated+positions[:, None, :]
    require(np.isfinite(world).all(), 'Unrepresentable transformed atom coordinates')
    if spherical:
        wall_clearance = boundary['radius']-np.linalg.norm(world, axis=2)-radii
        wall_min = float(wall_clearance.min())
        wall_valid = bool(np.all(wall_clearance >= 0))
    else:
        wall_min, wall_valid = None, True
    edges, core_overlaps, pair_checks = [], [], []
    pair_images = {}
    max_radius = float(radii.max())
    for i, j in itertools.combinations(range(len(poses)), 2):
        partner = positions[j].copy()
        if not spherical:
            partner -= lengths*np.floor((positions[j]-positions[i])/lengths+.5)
        pair_images[i, j] = partner
        separation = float(np.linalg.norm(partner-positions[i]))
        guard = 1024*np.finfo(float).eps*(1+np.linalg.norm(positions[i])
                                         + np.linalg.norm(partner)+bound+RD)
        row = dict(bodies=[i, j], center_separation_A=separation,
                   bounding_pruned=False, queried_atomic_pairs=0, core_overlap_count=0,
                   exclusion_overlap_count=0, queried_minimum_core_gap_A=None)
        if separation > 2*(bound+RD)+guard:
            row.update(bounding_pruned=True, core_gap_lower_bound_A=separation-2*bound-guard)
            pair_checks.append(row)
            continue
        left, right = world[i], rotated[j]+partner
        near = cKDTree(left).sparse_distance_matrix(cKDTree(right),
                    2*(max_radius+RD)+guard, output_type='ndarray')
        row['queried_atomic_pairs'] = int(len(near))
        if len(near):
            a, b = near['i'], near['j']
            d = left[a]-right[b]
            squared = (d[:, 0]*d[:, 0]+d[:, 1]*d[:, 1])+d[:, 2]*d[:, 2]
            core_sum = radii[a]+radii[b]
            exclusion_sum = (radii[a]+RD)+(radii[b]+RD)
            hard = squared < core_sum*core_sum
            contact = squared < exclusion_sum*exclusion_sum
            row.update(core_overlap_count=int(hard.sum()), exclusion_overlap_count=int(contact.sum()),
                       queried_minimum_core_gap_A=float((np.sqrt(squared)-core_sum).min()))
            if np.any(hard):
                k = int(np.flatnonzero(hard)[0])
                core_overlaps.append(dict(bodies=[i, j], atoms=[int(a[k]), int(b[k])],
                    squared_distance_A2=float(squared[k]), squared_core_threshold_A2=float(core_sum[k]**2),
                    core_overlap_count=int(hard.sum())))
            if np.any(contact):
                edges.append([i, j])
        pair_checks.append(row)
    hard_valid = not core_overlaps
    exclusion_components = components(len(poses), edges)
    result = dict(schema='finite-assembly-start-validation-v1', preparation=preparation,
        bodies=len(poses), atoms_per_body=len(atoms), depletant_radius_A=RD,
        geometry=dict(hard_valid=hard_valid, wall_valid=wall_valid, body_bound_A=bound,
                      minimum_atomic_wall_clearance_A=wall_min, core_overlaps=core_overlaps,
                      pair_checks=pair_checks),
        exclusion=dict(edges=edges, components=exclusion_components,
                       largest_component_size=max(map(len, exclusion_components))),
        native=dict(evaluated=False, definition_sha256=getattr(native, 'definition_sha256', None)),
        passed=False, failure_reasons=[], physical_draws=0,
        scope='Static preparation geometry and instantaneous registry. No equilibrium, sampling, or assembly claim.')
    if not hard_valid:
        result['failure_reasons'].append('strict_atomic_core_overlap')
    if not wall_valid:
        result['failure_reasons'].append('strict_atomic_wall_violation')
    if not hard_valid or not wall_valid:
        return result
    native_keys, matches = [], []
    for i, j in itertools.combinations(range(len(poses)), 2):
        partner = dict(poses[j], position=pair_images[i, j].tolist())
        entry = native.classify_pair(poses[i], partner)
        if entry:
            matches.append(dict(bodies=[i, j], matches=entry))
            native_keys.extend((i, j, int(m['motif_id'])) for m in entry)
    checker = NativeGraphConsistency(native.motifs, len(poses))
    cycle = checker.check(native_keys)
    native_edges = sorted({(i, j) for i, j, _ in native_keys})
    image_lift = spherical or periodic_lift(positions, lengths, native_edges)
    native_components = cycle['components']
    result['native'].update(evaluated=True, keys=[list(k) for k in sorted(set(native_keys))],
        edges=[list(e) for e in native_edges], matches=matches, cycle=cycle,
        largest_component_size=max(map(len, native_components)), ordinary_space_image_lift=image_lift)
    if not image_lift:
        result['failure_reasons'].append('native_graph_has_periodic_winding')
    if not cycle['consistent']:
        result['failure_reasons'].append('native_catalogue_cycle_inconsistent')
    if preparation == 'dispersed':
        if edges:
            result['failure_reasons'].append('dispersed_has_exclusion_contacts')
        if native_edges:
            result['failure_reasons'].append('dispersed_has_native_entry')
    elif preparation == 'competing-aggregate':
        if len(exclusion_components) != 1:
            result['failure_reasons'].append('competing_exclusion_graph_not_connected')
        if native_edges:
            result['failure_reasons'].append('competing_has_native_entry')
    else:
        expected = [list(range(8))]+[[i] for i in range(8, len(poses))]
        if native_components != expected:
            result['failure_reasons'].append('native_seed_is_not_exact_eight_body_component')
        if any(i >= 8 or j >= 8 for i, j in edges):
            result['failure_reasons'].append('native_seed_remainder_has_exclusion_contacts')
    result['passed'] = not result['failure_reasons']
    return result
