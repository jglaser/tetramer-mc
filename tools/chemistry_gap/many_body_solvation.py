"""Many-body part of the morphometric solvation along the greedy native growth path.

For step n (molecule c added to cluster S; order from greedy_path_steps.json):
  dF_site  = F(S + c) - F(S) - F(c)
  dF_pairs = sum_{i in S} [F(i + c) - F(i) - F(c)]   (pairs within 2 bound + 2.8 A; others 0)
  many-body = dF_site - dF_pairs
F = White Bear II morphometry, beta p V + sigma A + kappa M + kbar 4 pi chi, of the exact
union of r + 1.4 balls (union_measures.measures). Also per measure (V, A, M, chi).
Usage: many_body_solvation.py [N_PROC]
"""
import json
import sys
from multiprocessing import Pool

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, '/vast/xvg/tetramer-mc-runs/oligomer-capture-20261010/dft-ref')
from union_measures import measures  # noqa: E402

D = '/vast/xvg/tetramer-mc-runs/morph-growth-20261012/b22'
COEF = dict(V=0.2090, A=-0.1564, M=0.1362, chi=-0.0369 * 4 * np.pi)
shape = json.load(open('/home/xvg/tetramer-mc/examples/monomer-shape.json'))
X = np.array([a['center'] for a in shape['atoms']])
RAD = np.array([a['radius'] for a in shape['atoms']]) + 1.4
P = json.load(open('/home/xvg/tetramer-mc/examples/spherical-cluster-monomer.json'))['initial_poses'][:32]
path = json.load(open(f'{D}/greedy_path_steps.json'))
order = path['cluster']                       # molecule indices in addition order (first = root)
BOUND = np.linalg.norm(X, axis=1).max() + RAD.max()


def world(i):
    w, x, y, z = P[i]['orientation']
    return np.array(P[i]['position']) + X @ Rotation.from_quat([x, y, z, w]).as_matrix().T


def meas(ids):
    ids = list(ids)
    m = measures(np.vstack([world(i) for i in ids]), np.concatenate([RAD] * len(ids)))
    return tuple(ids), {k: float(m[k]) for k in ('V', 'A', 'M', 'chi')}


def F(m):
    return sum(COEF[k] * m[k] for k in COEF)


if __name__ == '__main__':
    nproc = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    jobs = [(order[0],)]
    for n in range(1, len(order)):
        jobs.append(tuple(order[:n + 1]))
        c = order[n]
        for i in order[:n]:
            if np.linalg.norm(np.array(P[i]['position']) - np.array(P[c]['position'])) < 2 * BOUND + 2.8:
                jobs.append((i, c))
    with Pool(nproc) as pool:
        res = dict(pool.map(meas, jobs))
    single = res[(order[0],)]
    rows = []
    print(f"{'step':>4s} {'added':>5s}{'site':>9s}{'pairs':>9s}{'many-body':>11s}   many-body by measure (kT): V, A, M, chi")
    for n in range(1, len(order)):
        c, S = order[n], order[:n]
        site = {k: res[tuple(order[:n + 1])][k] - (res[tuple(S)][k] if n > 1 else single[k]) - single[k] for k in COEF}
        pairs = {k: 0. for k in COEF}
        for i in S:
            if (i, c) in res:
                for k in COEF:
                    pairs[k] += res[(i, c)][k] - 2 * single[k]
        mb = {k: site[k] - pairs[k] for k in COEF}
        rows.append(dict(step=n + 1, added=c, site=F(site), pairs=F(pairs), many_body=F(mb),
                         many_body_terms={k: COEF[k] * mb[k] for k in COEF}, many_body_measures=mb))
        print(f"{n + 1:4d} {c:5d}{F(site):9.2f}{F(pairs):9.2f}{F(mb):11.2f}   "
              + ', '.join(f'{COEF[k] * mb[k]:.2f}' for k in COEF)
              + f"   (dV {mb['V']:.0f}, dA {mb['A']:.0f}, dM {mb['M']:.1f}, dchi {mb['chi']:.0f})")
    json.dump(rows, open(f'{D}/many_body_solvation.json', 'w'), indent=1)
    tot = sum(r['many_body'] for r in rows)
    print(f'sum over the path: many-body {tot:.1f} kT, pairwise {sum(r["pairs"] for r in rows):.1f} kT')
