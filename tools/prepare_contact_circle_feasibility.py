#!/usr/bin/env python3
"""Freeze every conditioned circle from the completed passive comparison.

This selects no new random poses, uses no native labels, and generates no
Poisson clouds. It measures geometry only; it is not an importance campaign.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(passive, out):
    passive, out = Path(passive).resolve(), Path(out).resolve()
    if out.exists():
        raise ValueError('Fresh output required')
    status, analysis = read(passive/'status.json'), read(passive/'analysis.json')
    if not status['complete'] or not analysis['complete']:
        raise ValueError('Completed audited passive comparison required')
    if sha(passive/'analysis.json') != status['analysis_sha256']:
        raise ValueError('Analysis changed')
    if analysis['fresh_attempts'] != 1536 or analysis['new_Poisson_clouds'] != 0:
        raise ValueError('Unexpected source allocation')
    cases, sources = [], {}
    for job in status['jobs']:
        directory = Path(job['directory'])
        for name, expected in job['output_sha256'].items():
            if sha(directory/name) != expected:
                raise ValueError('Audited source output changed')
        path = directory/'samples.jsonl'
        sources[str(path)] = sha(path)
        guide_path = passive/'common'/(job['arm']+'.json')
        guide = read(guide_path)
        sources[str(guide_path)] = sha(guide_path)
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if len(rows) != 128 or [r['id'] for r in rows] != list(range(128)):
            raise ValueError('Attempt identities changed')
        for row in rows:
            draw = row['draw']
            if not draw['conditional']:
                continue
            if draw['fallback']:
                raise ValueError('Unexpected fallback; cannot silently exclude it')
            cases.append(dict(id=f"{job['arm']}/{job['id']}/{row['id']}",
                arm=job['arm'], population=job['id'], source_attempt=row['id'],
                source_file=str(path), source_seed=job['seed'],
                component=draw['component'], width_index=draw['width_index'],
                pose=row['pose'], radii=draw['radii'], phi=draw['phi'],
                circle_center=draw['circle_center'], circle_radius=draw['circle_radius'],
                contact_pairs=guide['component_contact_pairs'][draw['component']],
                azimuth_law=draw['azimuth_law'], hard_valid=row['hard_valid'],
                shell_valid=row['shell_valid'], capture_valid=row['capture_valid']))
    if len(cases) != 240 or len({r['id'] for r in cases}) != 240:
        raise ValueError('Expected every one of the 240 conditioned circles')
    out.mkdir()
    for name in ('shape.json', 'config.json'):
        shutil.copy2(passive/'common'/name, out/name)
    config = read(out/'config.json')
    config['shape'] = str(out/'shape.json')
    (out/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    shutil.copy2(__file__, out/Path(__file__).name)
    protocol = dict(schema='contact-circle-feasibility-cases-v1',
        passive=str(passive), passive_analysis_sha256=sha(passive/'analysis.json'),
        passive_protocol_sha256=sha(passive/'protocol.json'), sources=sources,
        config=str(out/'config.json'), config_sha256=sha(out/'config.json'),
        shape_sha256=sha(out/'shape.json'), cases=cases,
        maximum_CPU_workers=1, new_pose_draws=0, new_Poisson_clouds=0,
        scope='Post-pilot geometry of every conditioned circle; no label filtering, physical weights, or resampling',
        planned_checks=['whole-union hard-free arc length and original azimuth mass',
            'original hard-valid predicate and reconstructed circle/pose',
            'independent Python arcs, masses, direct-pose witnesses',
            'empty circles, runtime and tree traversal counts, separated by arm/width'])
    (out/'cases.json').write_text(json.dumps(protocol, indent=2)+'\n')
    manifest = {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()}
    (out/'freeze.json').write_text(json.dumps(dict(files=manifest), indent=2)+'\n')
    return dict(cases=len(cases), cases_sha256=sha(out/'cases.json'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--passive', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.passive, args.out)))
