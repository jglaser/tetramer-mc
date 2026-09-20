#!/usr/bin/env python3
"""Bundle a saved run into a self-contained, offline sphere trajectory viewer.

This lightweight Canvas/WebGL viewer is not hoomd-bevy. The simulation files
remain authoritative; embedded display coordinates use GPU float32 buffers.
No external scripts, libraries, fonts or network resources are referenced.
"""
from __future__ import annotations
import argparse
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
    frames = []
    for line in trajectory_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
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
    reference = (reference or Path(__file__).resolve().parents[2]/'protein-nucleation').resolve()
    native = native_bonds(config, shape, frames, run, boundary, wall_radius, reference, native_mode)
    data = dict(schema='tetramer-offline-viewer-v1', run_name=run.name, atoms=atoms, frames=frames,
        body_count=count, box_lengths=box, boundary=boundary, spherical_wall_radius=wall_radius,
        spherical_wall_center=wall_center, native_bonds=native,
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
        native_bonds_unavailable_reason=native.get('reason'))


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
