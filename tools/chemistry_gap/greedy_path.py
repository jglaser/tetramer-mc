"""Free-energy-greedy native growth path. Start: the molecule of the 32-molecule native fragment
nearest its centroid plus its m6 partner. Each step: site constants (pair-binding with the
current cluster as environment; 6 A, like-charge w 4; 30k samples, 12k walk steps) for every
fragment molecule in contact with the cluster; add the one with the largest ln K at the
selection eps (1.14 eps_c). Results of all candidates per step: greedy_path_steps.json.
This is one constructive path (an upper bound on the barrier along native clusters), not the
transition path. Usage: greedy_path.py N_STEPS"""
import json
import subprocess
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, '/vast/xvg/tetramer-mc-runs/morph-growth-20261012')
import observe_growth as og  # noqa: E402

D = '/vast/xvg/tetramer-mc-runs/morph-growth-20261012/b22'
BIN = f'{D}/pair-binding-site.bin'
EPS = [0.0029, 0.00325, 0.0037, 0.0039, 0.00423, 0.00455]
SELECT = 0.0037
LN_RHO = np.log(1000e-6 * 6.02214076e23 / 1e27)
steps = int(sys.argv[1]) if len(sys.argv) > 1 else 13

cfg = json.load(open('/home/xvg/tetramer-mc/examples/spherical-cluster-monomer.json'))
P = cfg['initial_poses'][:32]
T = [og.arr(p) for p in P]
world = [t + og.X @ R.T for t, R in T]
trees = [cKDTree(w) for w in world]


def contact_type(i, j):
    (ti, Ri), (tj, Rj) = T[i], T[j]
    if np.linalg.norm(tj - ti) > 2 * og.BOUND:
        return None
    d, k = trees[j].query(world[i], distance_upper_bound=2 * og.R_ATOM.max() + 2 * og.RD)
    ok = np.isfinite(d)
    if not np.any(d[ok] < og.R_ATOM[ok] + og.R_ATOM[k[ok]] + 2 * og.RD):
        return None
    e = [og.scaled((Ri.T @ (tj - ti), Ri.T @ Rj), n) for n in og.NAT]
    m = int(np.argmin(e))
    return f'm{m}' if e[m] <= 1 else 'other'


def site_constants(cluster, candidates, tag):
    jobs = [dict(label=str(j), environment=[P[i] for i in cluster], pose=P[j]) for j in candidates]
    jf = f'{D}/greedy_jobs_{tag}.json'
    json.dump(jobs, open(jf, 'w'))
    out = subprocess.run([BIN, '--shape', '/home/xvg/tetramer-mc/examples/monomer-shape.json',
                          '--poses', jf, '--width', '6', '--epsilons', ','.join(map(str, EPS)),
                          '--eps-walk', '0.0039', '--charge-groups', f'{D}/../pair-binding/charge_groups_monomer_ph4.5.json',
                          '--like-charge-weight', '4', '--like-charge-cutoff', '5', '--samples', '30000',
                          '--walk-steps', '12000', '--threads', str(min(32, len(jobs)))],
                         capture_output=True, text=True, check=True)
    res = json.loads(out.stdout)
    return {int(r['label']): {c['epsilon']: (c['lnK'] + LN_RHO, c['lnK_se'], c['ess']) for c in r['K']}
            for r in res}


c = np.mean([t for t, _ in T], axis=0)
a = int(np.argmin([np.linalg.norm(t - c) for t, _ in T]))
b = next(j for j in range(32) if j != a and contact_type(a, j) == 'm6')
record = []
k0 = site_constants([a], [b], 'step1')[b]
record.append(dict(step=1, added=b, contacts=['m6'], lnKrho={str(e): k0[e] for e in EPS}, candidates={}))
cluster = [a, b]
print(f"step 1: m6 pair, ln(K rho) at select eps {k0[SELECT][0]:.1f}", flush=True)
for n in range(2, steps + 1):
    cand = [j for j in range(32) if j not in cluster and any(contact_type(i, j) for i in cluster)]
    ks = site_constants(cluster, cand, f'step{n}')
    best = max(cand, key=lambda j: ks[j][SELECT][0])
    types = [t for t in (contact_type(i, best) for i in cluster) if t]
    record.append(dict(step=n, added=best, contacts=types,
                       lnKrho={str(e): ks[best][e] for e in EPS},
                       candidates={str(j): {str(e): ks[j][e] for e in EPS} for j in cand}))
    cluster.append(best)
    print(f"step {n}: +{best} ({' '.join(sorted(types))}), ln(K rho) {ks[best][SELECT][0]:.1f} "
          f"(ess {ks[best][SELECT][2]:.0f}); {len(cand)} candidates", flush=True)
    json.dump(dict(eps=EPS, select=SELECT, cluster=cluster, steps=record),
              open(f'{D}/greedy_path_steps.json', 'w'), indent=1)
