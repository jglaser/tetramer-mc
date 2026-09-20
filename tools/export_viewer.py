#!/usr/bin/env python3
"""Bundle a saved run into a self-contained, offline sphere trajectory viewer.

This lightweight Canvas/WebGL viewer is not hoomd-bevy. The simulation files
remain authoritative; embedded display coordinates use GPU float32 buffers.
No external scripts, libraries, fonts or network resources are referenced.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
import math
import sys
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def poses(frame):
    if 'poses' in frame:
        result = frame['poses']
    elif 'positions' in frame and 'orientations' in frame:
        result = [dict(position=p, orientation=q) for p, q in zip(frame['positions'], frame['orientations'])]
    else:
        raise ValueError('Expected poses or positions/orientations in each saved frame')
    normalized = []
    for pose in result:
        p, q = pose['position'], pose['orientation']
        if len(p) != 3 or len(q) != 4 or not all(math.isfinite(v) for v in [*p, *q]):
            raise ValueError('Nonfinite or malformed saved pose')
        if abs(sum(v*v for v in q)-1.) > 1e-7:
            raise ValueError('Saved orientation must be a scalar-first unit quaternion')
        normalized.append(dict(position=p, orientation=q))
    return normalized


def vector3(value, name):
    if not isinstance(value, (list, tuple)) or len(value) != 3 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in value):
        raise ValueError(name + ' must contain three finite coordinates')
    return list(map(float, value))


def shift_count(record):
    counts = record.get('counts', {})
    if 'center_shift' in counts:
        return counts['center_shift'].get('completed')
    if 'shift' in counts:
        return counts['shift'].get('accepted')
    return None


def coordinate_history(run, frames, rows, initial_center):
    """Use persisted coordinate origins or certify their reconstruction from logs.

    Stored poses stay in the sphere frame. In the accumulated-shift convention,
    p_lab=p_sphere+C and an accepted common shift d changes C to C-d.
    A missing origin is never inferred from a center of mass or later frame.
    """
    convention = 'Accumulated center-shift convention: p_coordinate = p_sphere + C; C_new = C_old - accepted_common_displacement. This changes the display frame, not the physical target.'
    try:
        if all('coordinate_wall_center' in row for row in rows):
            origins = [row.get('coordinate_origin_sweep', 0) for row in rows]
            if any(not isinstance(s, int) or s < 0 or s > rows[0].get('sweep', rows[0].get('step', 0)) for s in origins) or len(set(origins)) != 1:
                raise ValueError('Persisted coordinate origin sweeps are invalid or inconsistent')
            for frame, row in zip(frames, rows):
                frame['coordinate_wall_center'] = vector3(row['coordinate_wall_center'], 'coordinate_wall_center')
                frame['coordinate_origin_sweep'] = origins[0]
            return dict(available=True, source='persisted trajectory field', origin_sweep=origins[0], convention=convention)
        first, last = frames[0]['sweep'], frames[-1]['sweep']
        if any(not isinstance(f['sweep'], int) for f in frames) or any(a['sweep'] >= b['sweep'] for a, b in zip(frames, frames[1:])):
            raise ValueError('Saved sweeps must increase strictly to reconstruct shift history')
        manifest_path = run/'manifest.json'
        manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
        if manifest.get('initial_sweep', first) != first:
            raise ValueError('First saved sweep differs from the segment start')
        origin_sources = {}
        origin_sweep = rows[0].get('coordinate_origin_sweep', 0)
        if 'coordinate_wall_center' in rows[0]:
            center = vector3(rows[0]['coordinate_wall_center'], 'initial coordinate_wall_center')
        elif first == 0 and not manifest.get('resume'):
            center = vector3(initial_center, 'initial coordinate wall center')
        else:
            resume = manifest.get('resume')
            if not resume:
                raise ValueError('Resumed/trimmed trajectory has no known initial coordinate wall center')
            path = Path(resume)
            candidates = [path] if path.is_absolute() else [run/path, Path.cwd()/path]
            path = next((p for p in candidates if p.is_file()), None)
            if path is None:
                raise ValueError('Resume checkpoint unavailable; initial coordinate wall center cannot be reconstructed')
            checkpoint = json.loads(path.read_text())
            if checkpoint.get('completed_sweeps', checkpoint.get('sweep')) != first or 'coordinate_wall_center' not in checkpoint:
                raise ValueError('Legacy resume checkpoint lacks the coordinate origin; provide its full earlier shift history')
            center = vector3(checkpoint['coordinate_wall_center'], 'checkpoint coordinate_wall_center')
            origin_sweep = checkpoint.get('coordinate_origin_sweep', 0)
            origin_sources[str(path.resolve())] = sha(path)
        baseline = shift_count(rows[0])
        if baseline is None and first == 0:
            baseline = 0
        if baseline is None or not isinstance(baseline, int) or baseline < 0 or (first == 0 and baseline != 0):
            raise ValueError('Initial shift counter is missing or inconsistent with the segment start')
        for row in rows[1:]:
            count = shift_count(row)
            if not isinstance(count, int) or count < baseline:
                raise ValueError('Saved shift counters are missing or inconsistent')
        path = next((run/name for name in ('moves.jsonl', 'moves.jsonl.gz') if (run/name).is_file()), None)
        if path is None:
            raise ValueError('Move log unavailable; cumulative shifts cannot be reconstructed')
        events = []
        previous_sweep = first
        with (gzip.open(path, 'rt') if path.suffix == '.gz' else path.open()) as stream:
            for line in stream:
                if not line.strip():
                    continue
                move = json.loads(line)
                sweep = move.get('sweep')
                if not isinstance(sweep, int) or sweep < previous_sweep or sweep <= first:
                    raise ValueError('Move log does not start after the initial frame or is out of sweep order')
                previous_sweep = sweep
                if move.get('kind') not in ('shift', 'center_shift') or move.get('accepted') is False:
                    continue
                if move['kind'] == 'center_shift':
                    displacement = move.get('result', {}).get('displacement')
                else:
                    direction = vector3(move.get('direction'), 'shift direction')
                    distance = move.get('distance')
                    if not isinstance(distance, (int, float)) or not math.isfinite(distance):
                        raise ValueError('Invalid logged shift distance')
                    displacement = [distance * v for v in direction]
                events.append((sweep, vector3(displacement, 'shift displacement')))
        event_index = 0
        centers = []
        for frame, row in zip(frames, rows):
            while event_index < len(events) and events[event_index][0] <= frame['sweep']:
                center = [c-d for c, d in zip(center, events[event_index][1])]
                event_index += 1
            expected = shift_count(row)
            if expected is None and frame['sweep'] == 0:
                expected = 0
            if expected != baseline + event_index:
                raise ValueError(f"Shift log/count mismatch at sweep {frame['sweep']}: {event_index} segment shifts, counter {expected}, baseline {baseline}")
            if 'coordinate_wall_center' in row and any(abs(a-b) > 1e-9*(1+abs(a)+abs(b)) for a,b in zip(center,vector3(row['coordinate_wall_center'],'coordinate_wall_center'))):
                raise ValueError('Persisted coordinate origin disagrees with logged displacements')
            centers.append(center[:])
        for frame, center in zip(frames, centers):
            frame['coordinate_wall_center'] = center
            frame['coordinate_origin_sweep'] = origin_sweep
        origin_sources[str(path.resolve())] = sha(path)
        if manifest_path.is_file():
            origin_sources[str(manifest_path.resolve())] = sha(manifest_path)
        return dict(available=True, source='validated cumulative accepted shift log',
                    initial_sweep=first, last_sweep=last, origin_sweep=origin_sweep, segment_shifts=event_index,
                    convention=convention, source_sha256=origin_sources)
    except (ValueError, TypeError, KeyError, OSError) as error:
        for frame in frames:
            frame.pop('coordinate_wall_center', None)
            frame.pop('coordinate_origin_sweep', None)
        return dict(available=False, reason=str(error), convention=convention)


def native_bonds(config, shape, frames, run, boundary, wall_radius, reference, mode):
    """Optional posthoc classifier; never substitute a center-distance graph."""
    unavailable = lambda reason: dict(available=False, reason=reason)
    if mode == 'off':
        return unavailable('Native-bond extraction disabled')
    members = config.get('rigid_members') or shape.get('rigid_members')
    monomer = config.get('monomer_shape')
    if not members or not monomer:
        reason = 'Rigid-member poses and monomer_shape are required for native bonds'
        if mode == 'on':
            raise ValueError(reason)
        return unavailable(reason)
    def resolve(path):
        path = Path(path)
        return path if path.is_absolute() else run/path
    monomer = resolve(monomer)
    inputs = [monomer, reference/'results/c1c3-scaffold/motifs.json',
              reference/'results/native-neighbor-classes/classification.json']
    code = [reference/'scripts'/name for name in
            ('tetramer_order.py', 'analyze_precursor_exchange.py', 'audit_tetramer_assembly.py')]
    missing = [str(path) for path in inputs+code if not path.is_file()]
    if missing:
        reason = 'Native analysis inputs unavailable: ' + ', '.join(missing)
        if mode == 'on':
            raise FileNotFoundError(reason)
        return unavailable(reason)
    sys.path.insert(0, str(reference/'scripts'))
    try:
        from tetramer_order import TetramerOrder
        from analyze_precursor_exchange import prepare_templates
    except ImportError as error:
        if mode == 'on':
            raise RuntimeError('Native bonds require the optional research NumPy/SciPy environment') from error
        return unavailable('Optional native-classifier dependencies unavailable: ' + str(error))
    analysis = dict(config, rigid_members=members, monomer_shape=str(monomer))
    if boundary == 'spherical':
        # The unchanged analyzer uses periodic wrapping. This fictitious box
        # makes that mapping the identity for every actual monomer displacement.
        # A recorded span bound also handles a translated wall center.
        bound = max(math.hypot(*a['center'])+a['radius'] for a in shape['atoms'])
        span = max(max(p['position'][k] for f in frames for p in f['poses']) -
                   min(p['position'][k] for f in frames for p in f['poses']) for k in range(3))
        member_bound = max(math.hypot(*m['position']) for m in members)
        analysis['box_lengths'] = [max(8*(wall_radius+bound), 4*(span+2*member_bound+bound))]*3
    motif = config.get('native_pair_motifs', config.get('metadata', {}).get('native_pair_motifs'))
    if motif:
        motif_path = resolve(motif)
        if not motif_path.is_file():
            reason = 'Native motif input unavailable: ' + str(motif_path)
            if mode == 'on':
                raise FileNotFoundError(reason)
            return unavailable(reason)
        analysis['native_pair_motifs'] = str(motif_path)
        inputs.append(motif_path)
    templates = prepare_templates(monomer, inputs[1], inputs[2])
    classifier = TetramerOrder(analysis, run, templates=templates)
    for frame in frames:
        result = classifier.classify(frame)
        unique = {}
        for edge in result['native_entry_monomer_edges']:
            key = (*edge['bodies'], *edge['members'])
            bond = unique.setdefault(key, dict(bodies=edge['bodies'], members=edge['members'], classes=[]))
            bond['classes'].append(edge['class_label'])
        frame['native_bonds'] = [dict(unique[key], classes=sorted(set(unique[key]['classes']))) for key in sorted(unique)]
    return dict(available=True, classifier='TetramerOrder.native_entry_monomer_edges',
        criterion='Strict per-frame native entry: proper relative monomer pose, atomic gap, and shared native residue patch; intrinsic contacts within a tetramer excluded.',
        history='Each saved frame classified independently; no retention hysteresis or inferred events between frames.',
        member_positions=[m['position'] for m in members], protocol=classifier.protocol,
        analysis_box_lengths=list(map(float, classifier.box)),
        source_sha256={str(path.resolve()): sha(path) for path in inputs+code})


def bundle(run, output, native_mode='auto', reference=None):
    run = run.resolve()
    config_path, trajectory_path = run/'config.json', run/'trajectory.jsonl'
    config = json.loads(config_path.read_text())
    shape_path = run/'provenance/shape.json'
    if not shape_path.is_file():
        candidate = Path(config['shape'])
        shape_path = candidate if candidate.is_absolute() else run/candidate
    shape = json.loads(shape_path.read_text())
    atoms = []
    for atom in shape['atoms']:
        row = [*atom['center'], atom['radius']]
        if len(row) != 4 or not all(math.isfinite(v) for v in row) or row[3] <= 0:
            raise ValueError('Invalid atom sphere')
        atoms.append(row)
    if not atoms:
        raise ValueError('No sphere geometry to display')
    frames, rows = [], []
    for line in trajectory_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows.append(row)
        frame = dict(sweep=row.get('sweep', row.get('step', len(frames))), poses=poses(row),
                     seed_labels=row.get('seed_labels', config.get('seed_labels', [])))
        if 'contact_edges' in row:
            frame['contact_edges'] = row['contact_edges']
        frames.append(frame)
    if not frames or not frames[0]['poses']:
        raise ValueError('No saved poses to display')
    count = len(frames[0]['poses'])
    if any(len(f['poses']) != count for f in frames):
        raise ValueError('This viewer requires a fixed number of rigid bodies')
    box = config.get('box_lengths', config.get('box_lengths_A'))
    if box is None or len(box) != 3 or not all(math.isfinite(v) and v > 0 for v in box):
        raise ValueError('Require three positive display/periodic box lengths')
    boundary_spec = config.get('boundary', 'periodic')
    boundary = boundary_spec.get('kind') if isinstance(boundary_spec, dict) else boundary_spec
    if boundary not in ('periodic', 'spherical'):
        raise ValueError('Only periodic or spherical boundaries are supported')
    wall_radius = None
    wall_center = config.get('spherical_wall_center', config.get('metadata', {}).get('spherical_wall_center', [0., 0., 0.]))
    if boundary == 'spherical':
        if isinstance(boundary_spec, dict):
            wall_center = boundary_spec.get('center', wall_center)
        wall_radius = boundary_spec.get('radius') if isinstance(boundary_spec, dict) else config.get('spherical_wall_radius', config.get('spherical_radius'))
        if not isinstance(wall_radius, (int, float)) or not math.isfinite(wall_radius) or wall_radius <= 0:
            raise ValueError('Spherical boundary requires a finite positive wall radius')
        box = [2 * wall_radius] * 3
    if len(wall_center) != 3 or not all(math.isfinite(v) for v in wall_center):
        raise ValueError('spherical_wall_center must contain three finite coordinates')
    for frame in frames:
        if any(not isinstance(i, int) or i < 0 or i >= count for i in frame['seed_labels']):
            raise ValueError('Seed label outside body index range')
    coordinate_frame = coordinate_history(run, frames, rows, wall_center) if boundary == 'spherical' else dict(available=False, reason='Periodic view uses its existing image convention')
    if coordinate_frame['available']:
        coordinate_frame['fixed_view_extent'] = wall_radius + max(math.hypot(*f['coordinate_wall_center']) for f in frames)
    reference = (reference or Path(__file__).resolve().parents[2]/'protein-nucleation').resolve()
    native = native_bonds(config, shape, frames, run, boundary, wall_radius, reference, native_mode)
    data = dict(schema='tetramer-offline-viewer-v1', run_name=run.name, atoms=atoms, frames=frames,
        body_count=count, box_lengths=box, boundary=boundary, spherical_wall_radius=wall_radius,
        spherical_wall_center=wall_center, coordinate_frame=coordinate_frame, stored_pose_frame='sphere-centered' if boundary == 'spherical' else 'periodic', native_bonds=native,
        body_bound=max(math.hypot(*a[:3])+a[3] for a in atoms),
        depletant_radius=config.get('depletant_radius', 0.),
        source_sha256=dict(config=sha(config_path), trajectory=sha(trajectory_path), shape=sha(shape_path)),
        geometry_scope='Actual saved rigid-body sphere union; body colors and proximity groups do not indicate native registration. Optional native bonds use the recorded classifier and criteria.')
    web = Path(__file__).resolve().parents[1]/'web'
    html = (web/'index.template.html').read_text()
    encoded = json.dumps(data, separators=(',', ':'), allow_nan=False)
    # Prevent embedded JSON strings from terminating the data script element.
    encoded = encoded.replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    html = html.replace('/*__VIEWER_CSS__*/', (web/'viewer.css').read_text())
    html = html.replace('/*__TRAJECTORY_DATA__*/', encoded)
    html = html.replace('/*__VIEWER_JS__*/', (web/'viewer.js').read_text())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html)
    return dict(output=str(output.resolve()), bodies=count, atoms_per_body=len(atoms),
        displayed_spheres=count*len(atoms), frames=len(frames), bytes=output.stat().st_size,
        output_sha256=sha(output), sources=data['source_sha256'],
        native_bonds_available=native['available'],
        native_bonds_per_frame=[len(f.get('native_bonds', [])) for f in frames],
        native_bonds_unavailable_reason=native.get('reason'), coordinate_frame=coordinate_frame)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--native-bonds', choices=('auto', 'on', 'off'), default='auto',
                        help='Optional unchanged research classifier; on requires its inputs and dependencies')
    parser.add_argument('--reference', type=Path, default=Path(__file__).resolve().parents[2]/'protein-nucleation',
                        help='Optional research repository containing TetramerOrder and native templates')
    args = parser.parse_args()
    print(json.dumps(bundle(args.run, args.out, args.native_bonds, args.reference), indent=2))


if __name__ == '__main__':
    main()
