#!/usr/bin/env python3
"""Freeze four-member synthetic line-guide CLI inputs; no poses or clouds drawn."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
from native_contact_regions import CRITERIA


def save(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(out):
    out = Path(out).resolve()
    if out.exists():
        raise ValueError('Fresh toy fixture directory required')
    out.mkdir(parents=True)
    identity = np.eye(3).tolist()
    pose = lambda x: dict(position=x, orientation=[1., 0., 0., 0.])
    fixed = [pose([0., 0., 0.]), pose([8., 0., 0.])]
    members = [dict(position=[0., 5.*i, 0.], rotation=identity) for i in range(4)]
    monomer = [dict(center=[0., 0., z], radius=.2, residue=i) for i, z in enumerate([0., .5])]
    atoms = [dict(center=(np.array(m['position'])+a['center']).tolist(), radius=a['radius'])
             for m in members for a in monomer]
    save(out/'shape.json', dict(name='four-member-two-atom-toy', volume=1., atoms=atoms))
    shape_sha = sha(out/'shape.json')
    native = dict(schema='native-entry-compiled-v1', source_definition_sha256='0'*64,
        source_input_sha256={'tetramer-shape.json': shape_sha}, criteria=CRITERIA, fixed_poses=fixed,
        members=members, monomer_atoms=monomer, residue_count=2, references=[], motifs=[])
    for i, x in enumerate([1.6, -1.6, 1.6]):
        label = f'toy-{i}'
        native['references'].append(dict(label=label, family='toy', position=[x, 0., 0.], rotation=identity,
                                         native_residue_pairs=[0]))
        native['motifs'].append(dict(id=i, position=[x, 0., 0.], rotation=identity,
            member_contacts=[dict(member_i=0, member_j=0, directed_class=label),
                             dict(member_i=3, member_j=3, directed_class=label)]))
    save(out/'native.json', native)
    metadata = dict(native_poses=[pose([1.6, 0., 0.])], rigid_members=[pose(m['position']) for m in members],
                    member_error_scale=2., angle_error_scale_deg=15.)
    config = dict(shape=str(out/'shape.json'), fixed_poses=fixed, initial_pose=pose([3.8, 0., 0.]),
        capture_center=[3.8, 0., 0.], capture_radius=6., depletant_radius=.5, reservoir_density=0.,
        poisson_lambda_ratio=64., translation_steps=[.1], rotation_steps_deg=[1.], rotation_probability=.5,
        local_attempts_per_cycle=1, uniform_probability=.5, seed=6100402001, metadata=metadata)
    save(out/'config.json', config)
    lower = np.diag([1.5, .5, .7, .2, .3, .4]); lower[1, 0] = .15; lower[3, 0] = .12; lower[4, 1] = -.08
    chart = dict(shape_sha256=shape_sha, coordinate_convention='anchor-body-relative', angular_length=8.,
        weights=[1.], anchors=[dict(position=[3.8, 0., 0.], rotation=identity)], means=[[0.]*6],
        covariances=[(lower@lower.T).tolist()])
    region = dict(shape_sha256=shape_sha, fixed_neighbor=fixed[0], physical_fixed_neighbors=fixed,
        capture_center=config['capture_center'], capture_radius=6., activity=0., depletant_radius=.5,
        physical_metric=metadata, minimum_original_q=0., mahalanobis_radius=4., gaussian_chart=chart)
    save(out/'region.json', region)
    components = [dict(weight=.6, mean=[0.]*6, covariance=np.eye(6).tolist()),
                  dict(weight=.4, mean=[.6, -.2, .1, .1, .2, -.1], covariance=(np.eye(6)*.7).tolist())]
    channels = [{'class': 'hard_free', 'probability': .2}, {'class': 'native', 'probability': .2},
                {'class': 'contact_without_native', 'probability': .2},
                {'class': 'contact_without_native', 'probability': .2, 'orthant': 22},
                {'class': 'native', 'probability': .2, 'orthant': 55}]
    guide = dict(schema='defensive-native-class-line-guide-v1', region_sha256=sha(out/'region.json'),
        defensive_uniform_shell_probability=.5, gaussian_components=components, raw_translation_axes=[0, 1, 2],
        conditional_probability=1., minimum_conditional_mass=1e-12, class_channels=channels,
        compiled_native=dict(path=str(out/'native.json'), sha256=sha(out/'native.json')),
        shape_sha256=shape_sha, fixed_poses=fixed, capture_center=config['capture_center'], capture_radius=6., depletant_radius=.5)
    probes = [dict(id=f'fixed-{i}', latent=u) for i, u in enumerate([
        [0.]*6, [-1., 0., 0., 0., 0., 0.], [1., 0., 0., 0., 0., 0.],
        [-.4, .2, -.3, .1, -.2, .4], [8., 0., 0., 0., 0., 0.],
        [0., 0., 8., 0., 0., 0.], [0., 0., 0., 8., 0., 0.], [.1, .2, .3, .4, .5, .6]])]
    (out/'probes.jsonl').write_text(''.join(json.dumps(p)+'\n' for p in probes))
    jobs = []
    for index, (name, count) in enumerate([('classes', 64), ('hard-free-only', 64), ('uniform', 32), ('disabled', 32), ('floor-fallback', 64)]):
        current = copy.deepcopy(guide)
        if name == 'hard-free-only': current['class_channels'] = [{'class': 'hard_free', 'probability': 1.}]
        if name == 'uniform': current['defensive_uniform_shell_probability'] = 1.
        if name == 'disabled': current['conditional_probability'] = 0.
        if name == 'floor-fallback': current['minimum_conditional_mass'] = .99
        save(out/f'{name}-guide.json', current)
        jobs.append(dict(id=name, samples=count, seed=6100402001+index, config=str(out/'config.json'),
            region=str(out/'region.json'), guide=str(out/f'{name}-guide.json'), probes=str(out/'probes.jsonl'), out=str(out/name)))
    paths = list(out.glob('*.json'))+[out/'probes.jsonl', Path(__file__).resolve()]
    allocation = dict(schema='native-class-line-synthetic-validation-v1', jobs=jobs, total_fresh_draws=256,
        saved_queries_per_job=8, physical_jobs=0, protein_queries=0, Poisson_clouds=0, maximum_workers=1,
        source_and_input_sha256={str(p): sha(p) for p in paths},
        scope='All 256 synthetic fresh draws and 40 fixed toy queries; complete native/class/full q/J and draw trace independent audit.')
    save(out/'allocation.json', allocation)
    return allocation


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    value = prepare(args.out)
    print(json.dumps(dict(jobs=len(value['jobs']), total_fresh_draws=value['total_fresh_draws'])))
