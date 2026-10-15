"""Can screened electrostatics pick out the real chloride sites?

Charges: pH 4.5 per-atom charges as in pair-binding/screened_coulomb.py. Potential
phi(r) = l_B sum_i q_i exp(-|r - x_i| / L_D) / |r - x_i| (kT/e), l_B 7.1 A, L_D 4.3 A (0.5 M).
Candidate anion positions: grid (0.5 A) points 2.9-4.0 A from the nearest protein heavy atom.
Observed chlorides: from 1IEE, 9RZL, 7DER (pdb-hewl/, read as data), superposed on the model
frame by CA; their asymmetric-unit chlorides and, for the m1 bridge, the chloride between
A and its m1 neighbour.
Test 1 (single molecule): percentile of phi at each observed chloride among first-shell points.
Test 2 (contacts): bridging points (first shell of both A and B); best phi per contact for the
native contacts and the non-native reference / blind-run contacts; distance of the best m1
point from the observed m1 chloride. Writes anion_electrostatics.json.
"""
import json
import sys

import gemmi
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, '/vast/xvg/tetramer-mc-runs/morph-growth-20261012')
import observe_growth as og  # noqa: E402

M = '/vast/xvg/tetramer-mc-runs/morph-growth-20261012'
LB, LD = 7.1, 4.3
names = json.load(open('/vast/xvg/tetramer-mc-runs/oligomer-capture-20261010/dft-ref/atom_types_monomer.json'))['atoms']
X = og.X
PKA = {('HIS', 15): 5.5, ('GLU', 7): 2.9, ('GLU', 35): 6.2, ('ASP', 18): 2.7, ('ASP', 48): 1.4,
       ('ASP', 52): 3.7, ('ASP', 66): 1.2, ('ASP', 87): 2.1, ('ASP', 101): 4.1, ('ASP', 119): 3.2}
q = np.zeros(len(X))
first, last = min(n[1] for n in names), max(n[1] for n in names)
for k, (res, num, atom) in enumerate(names):
    if res == 'LYS' and atom == 'NZ':
        q[k] += 1.
    if res == 'ARG' and atom in ('NE', 'NH1', 'NH2'):
        q[k] += 1. / 3.
    if res == 'HIS' and atom in ('ND1', 'NE2'):
        q[k] += 0.5 / (1 + 10 ** (4.5 - PKA[('HIS', num)]))
    if num == first and atom == 'N':
        q[k] += 1.
    if num == last and atom in ('O', 'OXT'):
        q[k] -= 0.5
    if (res, atom) in (('ASP', 'OD1'), ('ASP', 'OD2'), ('GLU', 'OE1'), ('GLU', 'OE2')):
        q[k] -= 0.5 / (1 + 10 ** (PKA.get((res, num), 4.0) - 4.5))
ch = np.abs(q) > 1e-9


def phi(points, coords, charges):
    out = np.zeros(len(points))
    for c, qq in zip(coords, charges):
        r = np.maximum(np.linalg.norm(points - c, axis=1), 1.0)
        out += qq * np.exp(-r / LD) / r
    return LB * out


def shell(coords, lo=2.9, hi=4.0, step=0.5):
    lo_c, hi_c = coords.min(0) - hi, coords.max(0) + hi
    g = np.mgrid[lo_c[0]:hi_c[0]:step, lo_c[1]:hi_c[1]:step, lo_c[2]:hi_c[2]:step].reshape(3, -1).T
    d = cKDTree(coords).query(g)[0]
    return g[(d >= lo) & (d <= hi)]


# observed chlorides in the model frame
CA_model = {n[1]: X[i] for i, n in enumerate(names) if n[2] == 'CA'}


def kabsch(P, Q):
    pc, qc = P.mean(0), Q.mean(0)
    U, _, Vt = np.linalg.svd((P - pc).T @ (Q - qc))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, qc - pc @ R.T


observed = {}
for pid in ('1IEE', '9RZL', '7DER'):
    st = gemmi.read_structure(f'{M}/pdb-hewl/{pid}.pdb')
    st.remove_alternative_conformations()
    ca, cls = {}, []
    for chn in st[0]:
        for r in chn:
            for a in r:
                if a.name == 'CA' and r.het_flag != 'H':
                    ca[r.seqid.num] = np.array(a.pos.tolist())
                if r.name == 'CL':
                    cls.append(np.array(a.pos.tolist()))
    common = sorted(set(ca) & set(CA_model))
    R, t = kabsch(np.array([ca[k] for k in common]), np.array([CA_model[k] for k in common]))
    observed[pid] = [c @ R.T + t for c in cls]

# Test 1: single molecule
S1 = shell(X)
p1 = phi(S1, X[ch], q[ch])
print(f'single molecule: {len(S1)} first-shell points; phi range {p1.min():.2f}..{p1.max():.2f} kT/e, '
      f'median {np.median(p1):.2f}')
res1 = {}
for pid, cls in observed.items():
    rows = []
    for c in cls:
        d = cKDTree(X).query(c)[0]
        v = phi(c[None], X[ch], q[ch])[0]
        pct = 100 * np.mean(p1 < v)
        rows.append((round(float(d), 2), round(float(v), 2), round(float(pct), 1)))
    res1[pid] = rows
    print(f'  {pid}: observed Cl (dist to protein, phi, percentile among shell points): {rows}')

# Test 2: contacts
D = f'{M}/pair-binding'
refs = [(f'native m{k}', n) for k, n in enumerate(og.NAT)]
refs += [(r['label'], og.arr(r['pose'])) for r in json.load(open(f'{D}/contacts_monomer.json'))
         + json.load(open(f'{D}/contacts_blind_top32.json'))[:12] if not r['label'].startswith('native')]
res2 = {}
for lab_, (t, R) in refs:
    B = t + X @ R.T
    both = np.vstack([X, B])
    S = shell(both)
    dA, dB = cKDTree(X).query(S)[0], cKDTree(B).query(S)[0]
    S = S[(dA <= 4.0) & (dB <= 4.0)]
    if len(S) == 0:
        res2[lab_] = dict(bridging_points=0, best_phi=None)
        continue
    v = phi(S, np.vstack([X[ch], B[ch]]), np.concatenate([q[ch], q[ch]]))
    i = int(np.argmax(v))
    res2[lab_] = dict(bridging_points=int(len(S)), best_phi=float(v[i]), best_site=S[i].tolist(),
                      n_above_p99=int(np.sum(v > np.percentile(p1, 99))))
# observed m1 chloride: the one within 4 A of both A and its m1 neighbour (from 1IEE)
t1, R1 = og.NAT[1]
B1 = t1 + X @ R1.T
cands = []
for pid, cls in observed.items():
    for c in cls:
        for (tt, RR) in (og.NAT[1], (-(og.NAT[1][1].T @ og.NAT[1][0]), og.NAT[1][1].T)):
            pass
    # chlorides of A's copy and of the neighbour's copy (B frame -> A frame)
    for c in cls + [t1 + c @ R1.T for c in cls]:
        if cKDTree(X).query(c)[0] < 4.2 and cKDTree(B1).query(c)[0] < 4.2:
            cands.append((pid, c))
best_m1 = np.array(res2['native m1']['best_site']) if res2['native m1'].get('best_site') else None
print('\nobserved m1-bridging chlorides (model frame):', [(p, np.round(c, 1).tolist()) for p, c in cands])
if best_m1 is not None and cands:
    print('distance of the best-phi m1 bridging point from the observed m1 chloride:',
          [round(float(np.linalg.norm(best_m1 - c)), 1) for _, c in cands])
nat = {k: v for k, v in res2.items() if k.startswith('native')}
non = {k: v for k, v in res2.items() if not k.startswith('native') and v['best_phi'] is not None}
print('\nbest bridging phi (kT/e): natives', {k[7:]: (round(v['best_phi'], 2) if v['best_phi'] is not None else None)
                                             for k, v in nat.items()})
nv = np.array([v['best_phi'] for v in non.values()])
print(f'non-natives ({len(nv)}): median {np.median(nv):.2f}, 90th pct {np.percentile(nv, 90):.2f}, max {nv.max():.2f}')
m1v = nat['native m1']['best_phi']
print(f'fraction of non-natives with best phi >= m1 ({m1v:.2f}): {np.mean(nv >= m1v):.2f}')
json.dump(dict(single=res1, contacts=res2, observed_m1=[(p, c.tolist()) for p, c in cands]),
          open(f'{M}/anion_electrostatics.json', 'w'), indent=1)
