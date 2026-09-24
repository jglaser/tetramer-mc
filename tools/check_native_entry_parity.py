#!/usr/bin/env python3
"""Cross-check the compiled Rust native-entry predicate on geometric fixtures.

No MC, bath weights, old population reclassification, or equilibrium inference.
Every positive hard-valid assertion in a probe request is checked independently
using complete tetramer atom distances before either entry predicate is called.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
              'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'RAYON_NUM_THREADS'):
    os.environ[_name] = '1'
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from compile_native_entry import compile_definition
from native_contact_regions import compose, make_pose, pose_arrays


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def hard_checker(model):
    shape = json.loads((model.root / 'tetramer-shape.json').read_text())
    xyz = np.asarray([a['center'] for a in shape['atoms']])
    radii = np.asarray([a['radius'] for a in shape['atoms']])
    tree = cKDTree(xyz)
    def check(anchor, moving):
        ta, ra = pose_arrays(anchor); tm, rm = pose_arrays(moving)
        d, r = ra.T @ (tm-ta), ra.T @ rm
        transformed = xyz @ r.T + d
        near = tree.sparse_distance_matrix(cKDTree(transformed), 2*radii.max(),
                                          output_type='ndarray')
        if not len(near):
            return True, None
        minimum = float(np.min(near['v']-radii[near['i']]-radii[near['j']]))
        return minimum >= -1e-8, minimum
    return check


def fixture_requests(model):
    origin = make_pose([0., 0., 0.], np.eye(3))
    cases = []
    def add(name, moving, anchor=None):
        row = dict(id=name, moving=copy.deepcopy(moving), hard_valid=True)
        if anchor is not None:
            row['anchor'] = copy.deepcopy(anchor)
        cases.append(row)
    for motif in model.motifs:
        mid = motif['id']; pose = model.motif_poses[mid]
        add(f'exact-{mid}', pose, origin)
        add(f'reversed-{mid}', origin, pose)
        negated = copy.deepcopy(pose)
        negated['orientation'] = (-np.asarray(pose['orientation'])).tolist()
        add(f'quaternion-sign-{mid}', negated, origin)
        # Offset on both sides of the 2 A member gate. Full atomic hard checks
        # explicitly discard clashes; no assumed entry label is imposed here.
        direction = np.asarray(pose['position'], dtype=float)
        direction /= np.linalg.norm(direction)
        for delta in (0.01, 0.1, 1.999999, 2.000001, 2.5):
            shifted = copy.deepcopy(pose)
            shifted['position'] = (np.asarray(pose['position']) + delta*direction).tolist()
            add(f'outward-{mid}-{delta}', shifted, origin)
    transform = make_pose([103., -47., 29.], Rotation.from_rotvec([.8, -.5, .2]).as_matrix())
    for index in (0, 3, 6, 13):
        mid = model.motifs[index]['id']
        add(f'global-isometry-{mid}', compose(transform, model.motif_poses[mid]), transform)
    physical = json.loads((model.root / 'physical-config.json').read_text())
    add('fixed-scaffold-native', physical['metadata']['native_poses'][0])
    add('fixed-scaffold-unbound', make_pose([500., 300., -200.],
                                          Rotation.from_rotvec([.1, .7, -.2]).as_matrix()))
    add('native-outside-capture', dict(position=[-110.48995177833392, 136.0113205273436, -2.7406713685702364],
        orientation=[.6882360353297444, -.1701385672833947, .5760388146466761, -.4068947180989217]))
    return cases


def expected_result(model, request):
    anchors = [request['anchor']] if 'anchor' in request else model.fixed_poses
    groups = []
    for i, anchor in enumerate(anchors):
        matches = model.classify_pair(anchor, request['moving'])
        groups.append(dict(anchor_index=i, matched_motif_ids=[m['motif_id'] for m in matches],
                           matches=matches))
    indices = [g['anchor_index'] for g in groups if g['matches']]
    return dict(id=request['id'], native_any=bool(indices),
                matched_anchor_indices=indices, per_anchor=groups)


def compare(expected, actual, errors, path=''):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError(f'Output keys differ at {path}: {actual!r}')
        for key in expected:
            compare(expected[key], actual[key], errors, path+'/'+key)
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f'Output length differs at {path}')
        for i, (x, y) in enumerate(zip(expected, actual)):
            compare(x, y, errors, path+f'/{i}')
    elif isinstance(expected, float):
        if not isinstance(actual, (float, int)) or not np.isfinite(actual):
            raise ValueError(f'Nonfinite or nonnumeric output at {path}')
        difference = abs(expected-actual)
        angular = 'orientation' in path
        tolerance = 3e-6 if angular else 2e-9
        key = 'maximum_angle_error_deg' if angular else 'maximum_position_or_gap_error_A'
        errors[key] = max(errors[key], difference)
        if difference > tolerance:
            raise ValueError(f'Numerical parity failed at {path}: {expected} versus {actual}')
    elif actual != expected:
        raise ValueError(f'Exact label parity failed at {path}: {expected!r} versus {actual!r}')


def run(definition, compiled, binary, out):
    definition, compiled, binary, out = [Path(p).resolve() for p in (definition, compiled, binary, out)]
    if out.exists():
        raise ValueError('Fresh parity output required')
    exported, model = compile_definition(definition)
    if json.loads(compiled.read_text()) != exported:
        raise ValueError('Compiled definition differs from verified Python materialization')
    hard = hard_checker(model)
    requests, expected, attempted = [], [], []
    for request in fixture_requests(model):
        anchors = [request['anchor']] if 'anchor' in request else model.fixed_poses
        checks = [hard(a, request['moving']) for a in anchors]
        valid = all(ok for ok, _ in checks)
        attempted.append(dict(request=request, full_atomic_hard_valid=valid,
                              anchor_core_gaps=[gap for _, gap in checks]))
        if valid:
            requests.append(request); expected.append(expected_result(model, request))
    if not any(e['native_any'] for e in expected) or not any(not e['native_any'] for e in expected):
        raise ValueError('Fixtures must exercise both classes')
    if not all(any(e['id'] == f'exact-{m["id"]}' for e in expected) for m in model.motifs):
        raise ValueError('Every exact native motif must be hard-valid and tested')
    # Probe must reject a missing/false assertion instead of implying that the
    # entry predicate establishes full atomic hard validity by itself.
    for value in (None, False):
        row = dict(id=f'hard-precondition-{value}', moving=requests[0]['moving'])
        if value is not None:
            row['hard_valid'] = value
        requests.append(row)
    out.mkdir(parents=True)
    (out/'attempted-fixtures.json').write_text(json.dumps(attempted, indent=2, allow_nan=False)+'\n')
    lines = ''.join(json.dumps(r, allow_nan=False)+'\n' for r in requests)
    (out/'requests.jsonl').write_text(lines)
    proc = subprocess.run([str(binary), '--definition', str(compiled)], input=lines,
                          text=True, capture_output=True, check=False)
    (out/'stdout.jsonl').write_text(proc.stdout); (out/'stderr.log').write_text(proc.stderr)
    if proc.returncode:
        raise ValueError(f'Rust probe failed with code {proc.returncode}; output preserved')
    actual = [json.loads(s) for s in proc.stdout.splitlines()]
    if len(actual) != len(requests):
        raise ValueError('Probe dropped or added requests')
    errors = dict(maximum_angle_error_deg=0., maximum_position_or_gap_error_A=0.)
    for e, a in zip(expected, actual):
        compare(e, a, errors)
    if not all('error' in a and a['id'] == r['id'] for r, a in zip(requests[-2:], actual[-2:])):
        raise ValueError('Probe accepted a missing hard-valid precondition')
    (out/'expected.json').write_text(json.dumps(expected, indent=2, allow_nan=False)+'\n')
    receipt = dict(schema='native-entry-crosslanguage-v1', complete=True,
        attempted_geometric_fixtures=len(attempted), hard_valid_fixtures=len(expected),
        discarded_geometric_clashes=len(attempted)-len(expected),
        native_fixtures=sum(e['native_any'] for e in expected),
        nonnative_fixtures=sum(not e['native_any'] for e in expected),
        full_motif_and_patch_label_parity=True, hard_precondition_checks=2, **errors,
        input_sha256={str(p):digest(p) for p in (definition, compiled, binary, Path(__file__).resolve(),
                                               Path(__file__).with_name('compile_native_entry.py'))},
        output_sha256={p.name:digest(p) for p in out.iterdir() if p.is_file()},
        scope='Geometric implementation parity only. No physical sampling, restricted-SMC '
              'normalizer, unseen-mode guarantee, global cycle certification or assembly conclusion.',
        physical_draws=0, old_population_classifier_passes=0)
    (out/'validation.json').write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n')
    return receipt


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('definition', 'compiled', 'binary', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    args=parser.parse_args()
    print(json.dumps(run(args.definition, args.compiled, args.binary, args.out), indent=2))
