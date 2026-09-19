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


def bundle(run, output):
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
        raise ValueError('Require three positive periodic box lengths')
    for frame in frames:
        if any(not isinstance(i, int) or i < 0 or i >= count for i in frame['seed_labels']):
            raise ValueError('Seed label outside body index range')
    data = dict(schema='tetramer-offline-viewer-v1', run_name=run.name, atoms=atoms, frames=frames,
        body_count=count, box_lengths=box, body_bound=max(math.hypot(*a[:3])+a[3] for a in atoms),
        depletant_radius=config.get('depletant_radius', 0.),
        source_sha256=dict(config=sha(config_path), trajectory=sha(trajectory_path), shape=sha(shape_path)),
        geometry_scope='Actual saved rigid-body sphere union; colors indicate body identity or original seed membership, never native registration.')
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
        output_sha256=sha(output), sources=data['source_sha256'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(bundle(args.run, args.out), indent=2))


if __name__ == '__main__':
    main()
