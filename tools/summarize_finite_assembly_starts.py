#!/usr/bin/env python3
"""Authenticate and visualize saved initial-state checks; no geometry replay."""
from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
from pathlib import Path
import shutil

import numpy as np

from contact_benchmark_contract import digest

PREPARATIONS = ('dispersed', 'competing-aggregate', 'native-seeded')
SHAPE_SHA = 'c7442034e7ffa4627b67c57a81117f0e9bc2d389ecf956a83dac2a5e915554c9'
NATIVE_SHA = '5cf55ebe0c0f78ec6b8cdc745cac938b0fadfa3732e1e1c730864c62c651d7a9'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def authenticated_inputs(root):
    root = Path(root).resolve()
    manifest, plan, freeze = (read(root/name) for name in ('manifest.json', 'plan.json', 'freeze.json'))
    require(manifest['schema'] == 'finite-assembly-geometric-starts-v1'
            and manifest['complete'] is True and manifest['physical_draws'] == 0
            and manifest['production_ready'] is False and not (root/'failure.json').exists(),
            'Require the complete successful non-production preparation')
    require(manifest['plan_sha256'] == sha(root/'plan.json'), 'Preparation plan binding differs')
    require(plan['sizes'] == [12, 24] and plan['preparations'] == list(PREPARATIONS)
            and plan['streams'] == 4 and plan['boundaries'] == ['spherical', 'periodic']
            and plan['canonical_preparations'] == 24 and plan['boundary_states'] == 48
            and plan['concentration_uM'] == 106.8 and plan['depletant_radius'] == 1.5
            and plan['reservoir_density'] == .035 and plan['native_seed_indices'] == list(range(8)),
            'Declared initial-state allocation or physical target differs')
    dependencies = {}
    for relative, expected in freeze['files'].items():
        path = (root/relative).resolve()
        require(path.is_relative_to(root) and path.is_file(), 'Invalid frozen input path')
        require(sha(path) == expected, 'Frozen preparation changed: '+relative)
        dependencies[str(path)] = expected
    for name in ('manifest.json', 'plan.json', 'inputs/shape.json', 'inputs/native/definition.json'):
        require(name in freeze['files'], 'Missing required frozen preparation file: '+name)
    require(freeze['files']['inputs/shape.json'] == SHAPE_SHA
            and freeze['files']['inputs/native/definition.json'] == NATIVE_SHA,
            'Physical shape or complete native definition differs')
    for name, expected in plan['input_sha256'].items():
        require(freeze['files'].get(name) == expected, 'Predeclared source/input changed: '+name)
    dependencies[str(root/'freeze.json')] = sha(root/'freeze.json')
    expected = set(itertools.product((12, 24), PREPARATIONS, range(4), ('spherical', 'periodic')))
    states, records, rows = {}, {}, []
    require(len(manifest['states']) == 48 and manifest['boundary_states'] == 48
            and manifest['canonical_preparations'] == 24, 'Incomplete or enlarged state inventory')
    for item in manifest['states']:
        require(item['state'] in freeze['files'] and item['validation'] in freeze['files'],
                'State or validation missing from freeze')
        require(item['state_sha256'] == freeze['files'][item['state']], 'State hash mismatch')
        state, report = read(root/item['state']), read(root/item['validation'])
        n, kind, stream, boundary = state['bodies'], state['preparation'], state['stream'], state['boundary']['kind']
        key = (n, kind, stream, boundary)
        require(type(n) is int and type(stream) is int and key in expected and key not in states,
                'Repeated/unplanned state')
        require(item['id'] == f'N{n}-{kind}-r{stream:02d}-{boundary}', 'State identifier mismatch')
        require(state['shape_sha256'] == SHAPE_SHA and state['depletant_radius'] == 1.5
                and state['reservoir_density'] == .035 and state['fixed_body_indices'] == []
                and state['preparation_equilibrated'] is False and state['proposal_training_feedback'] is False,
                'Initial-state law differs')
        require(len(state['initial_poses']) == n
                and digest(state['initial_poses']) == state['initial_poses_sha256'], 'Pose hash/count differs')
        require(report['schema'] == 'finite-assembly-start-validation-v1' and report['passed'] is True
                and report['failure_reasons'] == [] and report['physical_draws'] == 0
                and report['bodies'] == n and report['preparation'] == kind,
                'Unpassed or mismatched saved validation')
        geometry, exclusion, native = report['geometry'], report['exclusion'], report['native']
        require(geometry['hard_valid'] is True and geometry['wall_valid'] is True
                and not geometry['core_overlaps'] and native['evaluated'] is True
                and native['definition_sha256'] == NATIVE_SHA and native['cycle']['consistent'] is True
                and native['ordinary_space_image_lift'] is True, 'Saved validity/registry check failed')
        if kind == 'dispersed':
            require(not exclusion['edges'] and not native['edges'], 'Dispersed saved contact checks differ')
        elif kind == 'competing-aggregate':
            require(exclusion['largest_component_size'] == n and not native['edges'],
                    'Competing saved contact checks differ')
        else:
            require(native['cycle']['components'] == [list(range(8))]+[[i] for i in range(8, n)]
                    and not any(max(edge) >= 8 for edge in exclusion['edges']),
                    'Native seed/remainder saved checks differ')
        gaps = [p['queried_minimum_core_gap_A'] for p in geometry['pair_checks']
                if p['queried_minimum_core_gap_A'] is not None]
        row = dict(id=item['id'], bodies=n, preparation=kind, stream=stream, boundary=boundary,
            minimum_queried_core_gap_A=min(gaps) if gaps else None,
            minimum_atomic_wall_clearance_A=geometry['minimum_atomic_wall_clearance_A'],
            exclusion_components=exclusion['components'], exclusion_edges=len(exclusion['edges']),
            largest_exclusion_component=exclusion['largest_component_size'],
            native_components=native['cycle']['components'], native_edges=len(native['edges']),
            largest_native_component=native['largest_component_size'],
            independent_native_cycles=native['cycle']['independent_cycles'],
            construction_events=item['construction_events'],
            rejected_construction_events=item['rejected_construction_events'])
        states[key], records[key] = state, report
        rows.append(row)
    require(set(states) == expected, 'Missing boundary/state allocation')
    events, seed_set = [], set()
    for n, kind, stream in itertools.product((12, 24), PREPARATIONS, range(4)):
        left, right = states[n, kind, stream, 'spherical'], states[n, kind, stream, 'periodic']
        require(left['canonical_poses_sha256'] == digest(left['initial_poses'])
                and right['canonical_poses_sha256'] == left['canonical_poses_sha256'],
                'Paired boundary canonical identity differs')
        wrapped = copy.deepcopy(left['initial_poses'])
        for pose in wrapped:
            pose['position'] = np.mod(pose['position'], right['box_lengths']).tolist()
        require(wrapped == right['initial_poses'], 'Periodic boundary is not the unchanged wrapped canonical state')
        require(left['preparation_seed'] == right['preparation_seed']
                and left['preparation_seed'] not in seed_set, 'Preparation seeds not independent by canonical state')
        seed_set.add(left['preparation_seed'])
        name = f'events/N{n}-{kind}-r{stream:02d}.jsonl'
        require(name in freeze['files'], 'Construction events not frozen')
        log = [json.loads(line) for line in (root/name).read_text().splitlines()]
        require([e['index'] for e in log] == list(range(len(log))), 'Construction event accounting differs')
        rejected = sum(e['accepted'] is False for e in log)
        require(all(type(e['accepted']) is bool for e in log), 'Invalid event disposition')
        group = [r for r in rows if (r['bodies'], r['preparation'], r['stream']) == (n, kind, stream)]
        require(all(r['construction_events'] == len(log) and r['rejected_construction_events'] == rejected
                    for r in group), 'Construction attempts/zeros differ across boundary exports')
        events.append(dict(bodies=n, preparation=kind, stream=stream, events=len(log), rejections=rejected))
    require(seed_set == set(plan['preparation_seeds']) and len(plan['preparation_seeds']) == 24,
            'Declared preparation seed inventory differs')
    return root, manifest, plan, dependencies, states, records, rows, events


def overview(path, shape, states, records):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle, Rectangle
    from scipy.spatial.transform import Rotation

    atoms = np.asarray([a['center'] for a in shape['atoms']], float)[::4]
    colors = {'dispersed': '#526d82', 'competing-aggregate': '#168c98', 'native-seeded': '#31975c'}
    names = {'dispersed': 'Dispersed', 'competing-aggregate': 'Competing aggregate', 'native-seeded': 'Native seed + free tetramers'}
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 9.2))
    for row, n in enumerate((12, 24)):
        for col, kind in enumerate(PREPARATIONS):
            ax = axes[row, col]
            state, report = states[n, kind, 0, 'spherical'], records[n, kind, 0, 'spherical']
            poses = state['initial_poses']; positions = np.asarray([p['position'] for p in poses])
            radius = state['boundary']['radius']; length = states[n, kind, 0, 'periodic']['box_lengths'][0]
            ax.add_patch(Circle((0, 0), radius, fill=False, color='#66717c', linewidth=.8, linestyle='--'))
            ax.add_patch(Rectangle((-length/2, -length/2), length, length, fill=False,
                                   color='#abb2ba', linewidth=.8, linestyle=':'))
            for i, pose in enumerate(poses):
                q = np.asarray(pose['orientation']); rotation = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
                xyz = atoms@rotation.T+positions[i]
                color = '#8997a3' if kind == 'native-seeded' and i >= 8 else colors[kind]
                ax.scatter(xyz[:, 0], xyz[:, 1], s=1.25, color=color, alpha=.42, linewidths=0,
                           rasterized=True)
            edges = report['native']['edges'] if kind == 'native-seeded' else report['exclusion']['edges']
            for i, j in edges:
                ax.plot(positions[[i, j], 0], positions[[i, j], 1], color=colors[kind], linewidth=.9, alpha=.7)
            native_size = report['native']['largest_component_size'] if report['native']['edges'] else 0
            text = (f'Exclusion cluster: {report["exclusion"]["largest_component_size"]}/{n}'
                    f'   Native cluster: {native_size}/{n}')
            ax.text(.02, .025, text, transform=ax.transAxes, fontsize=8, color='#28333d',
                    bbox=dict(facecolor='white', edgecolor='none', alpha=.85, pad=2))
            ax.set_title(f'N = {n} · {names[kind]}', fontsize=11, loc='left')
            ax.set(xlim=(-1.08*radius, 1.08*radius), ylim=(-1.08*radius, 1.08*radius),
                   xlabel='x (Å)', ylabel='y (Å)')
            ax.set_aspect('equal'); ax.tick_params(labelsize=8)
            for spine in ax.spines.values():
                spine.set_color('#d6dbe0')
    fig.suptitle('Prepared starting states at 106.8 μM tetramers', fontsize=16, x=.065, ha='left')
    fig.legend(handles=[Line2D([], [], color='#66717c', linestyle='--', label='Spherical wall projection'),
                        Line2D([], [], color='#abb2ba', linestyle=':', label='Equal-volume periodic cell'),
                        Line2D([], [], color='#168c98', label='Exclusion contacts'),
                        Line2D([], [], color='#31975c', label='Instantaneous native contacts')],
               loc='lower center', ncol=4, frameon=False, fontsize=9, bbox_to_anchor=(.5, .055))
    fig.text(.065, .025, 'Stream 0; XY projection of every fourth atom center. Depth is omitted. '
             'These are imposed initial conditions, not equilibrium assembly.', fontsize=9, color='#47525e')
    fig.tight_layout(rect=(0, .095, 1, .95))
    fig.savefig(path, dpi=190)
    plt.close(fig)


def summarize(root, out):
    root, manifest, plan, dependencies, states, records, rows, events = authenticated_inputs(root)
    out = Path(out).resolve()
    require(not out.exists(), 'Fresh report directory required')
    out.mkdir()
    shutil.copy2(Path(__file__), out/'summarize_finite_assembly_starts.py')
    write(out/'report-plan.json', dict(schema='finite-assembly-starts-report-plan-v1',
        preparation_manifest_sha256=sha(root/'manifest.json'), source_sha256=sha(__file__),
        allocation='All 48 saved states; four streams per N/preparation/boundary; no selection',
        visualization='Stream 0, spherical canonical coordinates, XY projection, every fourth atom center',
        geometry_replays=0, native_classifier_calls=0, physical_draws=0))
    groups = []
    for n, kind in itertools.product((12, 24), PREPARATIONS):
        group = [r for r in rows if r['bodies'] == n and r['preparation'] == kind]
        construction = [e for e in events if e['bodies'] == n and e['preparation'] == kind]
        def span(field):
            values = [r[field] for r in group if r[field] is not None]
            return [min(values), max(values)] if values else None
        groups.append(dict(bodies=n, preparation=kind, canonical_streams=4, boundary_states=8,
            minimum_queried_core_gap_range_A=span('minimum_queried_core_gap_A'),
            spherical_wall_clearance_range_A=span('minimum_atomic_wall_clearance_A'),
            exclusion_edge_count_range=span('exclusion_edges'),
            largest_exclusion_component_range=span('largest_exclusion_component'),
            native_edge_count_range=span('native_edges'), largest_native_component_range=span('largest_native_component'),
            independent_native_cycles_range=span('independent_native_cycles'),
            construction_events=sum(e['events'] for e in construction),
            rejected_construction_events=sum(e['rejections'] for e in construction)))
    overview(out/'initial-preparations.png', read(root/'inputs/shape.json'), states, records)
    result = dict(schema='finite-assembly-starts-report-v1', complete=True,
        canonical_preparations=24, boundary_states=48, all_saved_validations_passed=True,
        paired_boundary_canonical_hashes_verified=True, groups=groups, states=rows,
        construction=events, preparation_manifest_sha256=sha(root/'manifest.json'),
        preparation_plan_sha256=manifest['plan_sha256'], source_and_input_sha256=dependencies,
        preparation_wall_seconds=manifest['wall_seconds'], figure_sha256=sha(out/'initial-preparations.png'),
        geometry_replays=0, native_classifier_calls=0, physical_draws=0,
        production_ready=False, outstanding=manifest['outstanding'],
        scope='Static prepared geometries. Saved validation reused without replay. No equilibrium, '
              'thermodynamic, mixing, native-discovery or assembly-stability conclusion.')
    for path, expected in dependencies.items():
        require(sha(path) == expected, 'Preparation changed during report: '+path)
    write(out/'analysis.json', result)
    write(out/'freeze.json', dict(files={p.relative_to(out).as_posix(): sha(p)
                                       for p in sorted(out.rglob('*')) if p.is_file()}))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.root, args.out)
    print(json.dumps(dict(complete=result['complete'], boundary_states=result['boundary_states'],
                         physical_draws=0, output=str(args.out.resolve())), indent=2))
