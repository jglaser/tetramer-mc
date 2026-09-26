#!/usr/bin/env python3
"""Turn a shape-only FFT depletion scan into hard-valid discovery start poses.

1. Pool every scan peak and keep the best `--pool` by grid overlap.
2. Repair each to an exactly hard-valid pose by greedy random perturbation that
   lowers the summed squared atom-pair penetration (KD-tree, exact radii).
3. Score repaired poses with the exact fixed-allocation union-overlap estimator
   (target/release/pair-overlap-score).
4. Greedy non-maximum suppression in pose space, treating a pose and its
   physical inverse I(t,R)=(-R^T t, R^T) as the same pair contact.

Only the particle shape and the scan enter. No native motif, label or
production pose is read; native evaluation is a separate later diagnostic.
"""
from __future__ import annotations
import os
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(name, '1')
import argparse, hashlib, json, subprocess, tempfile, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
SCORER = ROOT/'target/release/pair-overlap-score'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def to_pose(t, R):
    x, y, z, w = Rotation.from_matrix(R).as_quat()
    q = np.array([w, x, y, z]); q = q if q[0] >= 0 else -q
    return dict(position=[float(v) for v in t], orientation=[float(v) for v in q])


def from_pose(p):
    w, x, y, z = p['orientation']
    return np.asarray(p['position'], float), Rotation.from_quat([x, y, z, w]).as_matrix()


_G = {}


def _init(shape_path):
    shape = json.loads(Path(shape_path).read_text())
    X = np.array([a['center'] for a in shape['atoms']]); r = np.array([a['radius'] for a in shape['atoms']])
    _G.update(X=X, r=r, tree=cKDTree(X), rmax=r.max())


def penetration(t, R):
    X, r, tree = _G['X'], _G['r'], _G['tree']
    Y = X@R.T+t
    pairs = tree.query_ball_point(Y, 2*_G['rmax'])
    total, worst = 0., 0.
    for i, js in enumerate(pairs):
        if js:
            d = np.linalg.norm(X[js]-Y[i], axis=1)
            p = r[js]+r[i]-d
            p = p[p > 0]
            if len(p):
                total += float(np.sum(p*p)); worst = max(worst, float(p.max()))
    return total, worst


def repair(job):
    t, q, seed, steps, clearance = job
    rng = np.random.default_rng(seed)
    R = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    t = np.asarray(t, float)
    cost, worst = penetration(t, R)
    for k in range(steps):
        if worst < -clearance or cost == 0.:
            break
        level = rng.integers(3)
        tt = t+rng.normal(scale=[0.05, 0.2, 0.5][level], size=3)
        RR = Rotation.from_rotvec(np.radians([0.1, 0.5, 1.5][level])*rng.normal(size=3)).as_matrix()@R
        c, w = penetration(tt, RR)
        if c < cost:
            t, R, cost, worst = tt, RR, c, w
    if cost > 0.:
        return None
    return to_pose(t, R)


def score(poses, shape, rd, points, seed):
    with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as f:
        json.dump(poses, f); name = f.name
    try:
        out = subprocess.run([str(SCORER), '--shape', shape, '--poses', name, '--rd', str(rd),
                              '--points', str(points), '--seed', str(seed)],
                             check=True, capture_output=True, text=True).stdout
    finally:
        os.unlink(name)
    return json.loads(out)


def score_chunk(job):
    return score(*job)


def pose_distance(a, b):
    """(translation A, rotation degrees) to b or its inverse, whichever is closer."""
    ta, Ra = from_pose(a)
    tb, Rb = from_pose(b)
    best = (np.inf, np.inf)
    for t, R in ((tb, Rb), (-Rb.T@tb, Rb.T)):
        d = (float(np.linalg.norm(ta-t)), float(np.degrees(Rotation.from_matrix(Ra.T@R).magnitude())))
        if max(d) < max(best):
            best = d
    return best


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scan', required=True)
    p.add_argument('--shape', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--pool', type=int, default=20000)
    p.add_argument('--select', type=int, default=64)
    p.add_argument('--rd', type=float, default=1.4)
    p.add_argument('--points', type=int, default=8192)
    p.add_argument('--repair-steps', type=int, default=400)
    p.add_argument('--clearance', type=float, default=0.02)
    p.add_argument('--distinct-translation', type=float, default=4.0)
    p.add_argument('--distinct-angle', type=float, default=20.0)
    p.add_argument('--workers', type=int, default=100)
    p.add_argument('--seed', type=int, default=20260926501)
    a = p.parse_args()
    out = Path(a.out); out.mkdir(parents=True)
    scan = Path(a.scan)
    assert json.loads((scan/'manifest.json').read_text())['complete']
    rows = np.concatenate([np.load(f) for f in sorted(scan.glob('shard-*.npy'))])
    # Deduplicate identical (rotation, translation) peaks found under both tolerances.
    key = np.round(rows[:, [0, 6, 7, 8]]).astype(np.int64)
    _, first = np.unique(key, axis=0, return_index=True)
    rows = rows[first]
    # Equal pool share per clash-tolerance class, each ranked by grid overlap.
    classes = np.unique(rows[:, 1])
    share = a.pool//len(classes)
    rows = np.concatenate([(lambda c: c[np.argsort(-c[:, 9], kind='stable')][:share])(rows[rows[:, 1] == k])
                           for k in classes])
    started = time.time()
    jobs = [(r[6:9], r[2:6], a.seed+i, a.repair_steps, a.clearance) for i, r in enumerate(rows)]
    with Pool(a.workers, initializer=_init, initargs=(a.shape,)) as pool:
        repaired = pool.map(repair, jobs, chunksize=8)
        valid = [(i, q) for i, q in enumerate(repaired) if q is not None]
        chunks = [valid[k:k+64] for k in range(0, len(valid), 64)]
        scored = pool.map(score_chunk, [([q for _, q in c], a.shape, a.rd, a.points, a.seed) for c in chunks])
    records = []
    for c, s in zip(chunks, scored):
        for (i, q), res in zip(c, s):
            if res['hard_valid']:
                records.append(dict(scan_row=rows[i].tolist(), pose=q, exact_overlap=res['score']['volume'],
                                    exact_overlap_se=res['score']['standard_error']))
    records.sort(key=lambda r: -r['exact_overlap'])
    chosen = []
    for r in records:
        if all(not (d[0] < a.distinct_translation and d[1] < a.distinct_angle)
               for d in (pose_distance(r['pose'], c['pose']) for c in chosen)):
            chosen.append(r)
            if len(chosen) == a.select:
                break
    (out/'candidates.json').write_text(json.dumps(records)+'\n')
    (out/'selected.json').write_text(json.dumps(chosen, indent=1)+'\n')
    (out/'initial-poses.json').write_text(json.dumps([c['pose'] for c in chosen], indent=1)+'\n')
    summary = dict(native_information=False, arguments=vars(a), scan_manifest_sha256=sha(scan/'manifest.json'),
                   shape_sha256=sha(a.shape), script_sha256=sha(__file__), scorer_sha256=sha(SCORER),
                   pooled=len(rows), repaired_hard_valid=len(records), selected=len(chosen),
                   selected_overlap=[c['exact_overlap'] for c in chosen], wall_seconds=time.time()-started,
                   rule='Top pooled grid peaks -> exact repair -> exact overlap ranking -> NMS with pose/inverse equivalence. Validation clouds and native data never used.')
    (out/'summary.json').write_text(json.dumps(summary, indent=1)+'\n')
    print(json.dumps({k: summary[k] for k in ('pooled', 'repaired_hard_valid', 'selected', 'wall_seconds')}))
    print('top exact overlaps', [round(v) for v in summary['selected_overlap'][:20]])


if __name__ == '__main__':
    main()
