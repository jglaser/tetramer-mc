#!/usr/bin/env python3
"""Freeze supplied scaffold poses in every anchor frame for proposal evaluation.

This prepares evaluation inputs only. It neither discovers contacts nor alters
the proposal. Seed geometry may be supplied; no native labels enter a fit.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from scipy.spatial.transform import Rotation


def relative_scaffolds(poses, indices):
    if not indices or len(set(indices)) != len(indices):
        raise ValueError('A nonempty set of distinct fixed indices is required')
    if any(type(i) is not int or not 0 <= i < len(poses) for i in indices):
        raise ValueError('Invalid scaffold index')
    translations, rotations = [], []
    for i in indices:
        p = np.asarray(poses[i]['position'], dtype=float)
        q = np.asarray(poses[i]['orientation'], dtype=float)
        if (p.shape != (3,) or q.shape != (4,) or not np.isfinite(p).all()
                or not np.isfinite(q).all() or abs(q@q-1) > 1e-10):
            raise ValueError('Invalid rigid pose')
        translations.append(p)
        rotations.append(Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix())
    result = []
    for local_anchor, anchor in enumerate(indices):
        inverse = rotations[local_anchor].T
        order = [local_anchor] + [i for i in range(len(indices)) if i != local_anchor]
        relative = [dict(position=[0., 0., 0.], orientation=[1., 0., 0., 0.])]
        for i in order[1:]:
            p = inverse@(translations[i]-translations[local_anchor])
            q = Rotation.from_matrix(inverse@rotations[i]).as_quat(canonical=True)
            relative.append(dict(position=p.tolist(), orientation=q[[3, 0, 1, 2]].tolist()))
        result.append(dict(anchor_index=anchor, fixed_body_indices=[indices[i] for i in order], poses=relative))
    return result


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def prepare(config, indices, out):
    config, out = Path(config).resolve(), Path(out).resolve()
    if out.exists():
        raise ValueError('Fresh output required')
    source = json.loads(config.read_text())
    if source.get('boundary', {}).get('kind') == 'periodic':
        raise ValueError('Periodic scaffolds require an explicitly unwrapped input')
    rows = relative_scaffolds(source['initial_poses'], indices)
    shape = Path(source['shape'])
    if not shape.is_absolute():
        shape = config.parent/shape
    out.mkdir(parents=True)
    shutil.copy2(config, out/'source-config.json')
    shutil.copy2(shape, out/'shape.json')
    shutil.copy2(__file__, out/'executed-prepare.py')
    manifest_rows = []
    for row in rows:
        name = f"anchor-{row['anchor_index']:03d}.json"
        write(out/name, row['poses'])
        manifest_rows.append({**{k: v for k, v in row.items() if k != 'poses'}, 'file': name})
    manifest = dict(schema='fixed-contact-neighborhood-audit-v1', complete=True,
        source_config=str(config), source_config_sha256=sha(config),
        shape_sha256=sha(shape), scaffold_indices=indices, neighborhoods=manifest_rows,
        physical_jobs_launched=0, native_labels_read=False,
        evaluation_only=True, model_selection=False,
        transformation='t_rel=R_anchor^T(t_fixed-t_anchor); R_rel=R_anchor^T R_fixed. Anchor is exactly identity.',
        scope='Supplied scaffold geometry for unconditional proposal-draw diagnostics. '
              'Includes all selected anchor environments. No wall, periodic images or physical MC. '
              'The Rust audit separately checks scaffold hard validity before drawing. '
              'No scaffold information enters contact discovery or covariance training.',
        files={p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()})
    write(out/'manifest.json', manifest)
    return manifest


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True, type=Path)
    p.add_argument('--indices', nargs='+', type=int, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(prepare(args.config, args.indices, args.out), indent=2))
