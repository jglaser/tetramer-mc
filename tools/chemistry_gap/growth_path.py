"""Native growth path for site binding constants: from the 32-molecule native fragment
(examples/spherical-cluster-monomer.json), start with the molecule nearest the centroid and its
m6 partner, then repeatedly add the remaining molecule with the most native contacts to the
cluster (ties: nearest the cluster centroid). Writes site_path_jobs.json: job n has the current
cluster as `environment` and the next molecule as `pose` (job 0: the m6 pair).
Usage: growth_path.py N_STEPS"""
import json
import sys

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

sys.path.insert(0, '/vast/xvg/tetramer-mc-runs/morph-growth-20261012')
import observe_growth as og  # noqa: E402

steps = int(sys.argv[1]) if len(sys.argv) > 1 else 12
cfg = json.load(open('/home/xvg/tetramer-mc/examples/spherical-cluster-monomer.json'))
P = cfg['initial_poses'][:32]
T = [og.arr(p) for p in P]
world = [t + og.X @ R.T for t, R in T]
trees = [cKDTree(w) for w in world]


def native(i, j):
    (ti, Ri), (tj, Rj) = T[i], T[j]
    if np.linalg.norm(tj - ti) > 2 * og.BOUND:
        return None
    d, k = trees[j].query(world[i], distance_upper_bound=2 * og.R_ATOM.max() + 2 * og.RD)
    ok = np.isfinite(d)
    if not np.any(d[ok] < og.R_ATOM[ok] + og.R_ATOM[k[ok]] + 2 * og.RD):
        return None
    e = [og.scaled((Ri.T @ (tj - ti), Ri.T @ Rj), n) for n in og.NAT]
    m = int(np.argmin(e))
    return f'm{m}' if e[m] <= 1 else None


c = np.mean([t for t, _ in T], axis=0)
a = int(np.argmin([np.linalg.norm(t - c) for t, _ in T]))
b = next(j for j in range(32) if j != a and native(a, j) == 'm6')
cluster = [a]
order = [(b, ['m6'])]
cluster_next = b
jobs = [dict(label='step 1: m6 pair', environment=[P[a]], pose=P[b], contacts=['m6'])]
cluster.append(b)
for n in range(2, steps + 1):
    best = None
    cen = np.mean([T[i][0] for i in cluster], axis=0)
    for j in range(32):
        if j in cluster:
            continue
        types = [native(i, j) for i in cluster]
        types = [x for x in types if x]
        key = (len(types), -np.linalg.norm(T[j][0] - cen))
        if best is None or key > best[0]:
            best = (key, j, types)
    _, j, types = best
    jobs.append(dict(label=f'step {n}: +{len(types)} ({" ".join(sorted(types))})',
                     environment=[P[i] for i in cluster], pose=P[j], contacts=types))
    cluster.append(j)
for jb in jobs:
    print(jb['label'], ' cluster size before', len(jb['environment']))
json.dump(jobs, open('/vast/xvg/tetramer-mc-runs/morph-growth-20261012/b22/site_path_jobs.json', 'w'), indent=1)
