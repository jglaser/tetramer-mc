#!/usr/bin/env python3
"""Derive a monomer-body assembly config from a tetramer-body config.

Each labelled seed tetramer is expanded into its rigid member monomers at their
exact native poses (pose_tetramer ∘ member_pose), so the monomer seed is the
same crystal fragment. Unlabelled template bodies are dropped; free monomers
are generated at run time with --free-bodies/--concentration-um. The bath,
schedule and cluster-phase settings are copied. The tetramer-level observer
fields (monomer_shape, native_pair_motifs) are removed; monomer-level native
classification is a separate post-hoc analysis.
"""
import argparse, copy, hashlib, json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation


def arrays(p):
    w, x, y, z = p['orientation']
    return np.asarray(p['position'], float), Rotation.from_quat([x, y, z, w]).as_matrix()


def pose(t, R):
    x, y, z, w = Rotation.from_matrix(R).as_quat()
    q = np.array([w, x, y, z]); q = q if q[0] >= 0 else -q
    return dict(position=[float(v) for v in t], orientation=[float(v) for v in q])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--template', default='examples/spherical-cluster-oligomer.json')
    p.add_argument('--tetramer-shape', default='examples/tetramer-shape.json')
    p.add_argument('--monomer-shape', default='monomer-shape.json', help='path written into the config (relative to it)')
    p.add_argument('--depletant-radius', type=float, default=1.4)
    p.add_argument('--depletant-activity', type=float, default=0.0275)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    cfg = json.loads(Path(a.template).read_text())
    members = [arrays(m) for m in json.loads(Path(a.tetramer_shape).read_text())['rigid_members']]
    poses = []
    for label in sorted(cfg['seed_labels']):
        T, R = arrays(cfg['initial_poses'][label])
        poses += [pose(T+R@t, R@r) for t, r in members]
    out = copy.deepcopy(cfg)
    out.update(shape=a.monomer_shape, initial_poses=poses, seed_labels=list(range(len(poses))),
               depletant_radius=a.depletant_radius, reservoir_density=a.depletant_activity)
    out.pop('monomer_shape', None)
    meta = {k: v for k, v in cfg['metadata'].items() if k in ('cluster_phase_protocol', 'cluster_phase_status')}
    meta.update(arm='seeded-monomer', body='lysozyme monomer (1LYZ, repaired, P43212 cell)',
                seed_origin=f'{len(cfg["seed_labels"])} labelled seed tetramers of {Path(a.template).name} expanded into '
                            f'{len(poses)} member monomers at their exact native poses',
                template_config_sha256=hashlib.sha256(Path(a.template).read_bytes()).hexdigest(),
                tetramer_shape_sha256=hashlib.sha256(Path(a.tetramer_shape).read_bytes()).hexdigest(),
                native_observer='none in config; classify monomer contacts post hoc',
                usage='tetramer-mc run --config <this> --model <monomer map> --free-bodies N --concentration-um C ...')
    out['metadata'] = meta
    Path(a.out).write_text(json.dumps(out, indent=2)+'\n')
    print(json.dumps(dict(out=a.out, seed_monomers=len(poses))))


if __name__ == '__main__':
    main()
