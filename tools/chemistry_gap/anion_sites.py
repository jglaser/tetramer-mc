"""Anion-bridge sites at contacts: a generic geometric rule, not lysozyme-specific.

A site is a hydrogen-bond donor on A and one on B, 4.8-7.2 A apart, with an anion placed at
their midpoint 2.6-3.6 A from both and no other protein heavy atom (of A or B) within 2.9 A;
sites closer than 2.5 A to each other are merged. Donors: Tyr OH, Asn ND2, Gln NE2,
Arg NE/NH1/NH2, Lys NZ, Ser OG, Thr OG1, Trp NE1, His ND1/NE2, backbone N (not Pro).
Counts sites at the native contacts m0-m6 and at the non-native reference and blind-run
contacts (contact poses as in the binding-constant runs). Writes anion_sites.json.
"""
import json
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, '/vast/xvg/tetramer-mc-runs/morph-growth-20261012')
import observe_growth as og  # noqa: E402

names = json.load(open('/vast/xvg/tetramer-mc-runs/oligomer-capture-20261010/dft-ref/atom_types_monomer.json'))['atoms']
SIDE = {('TYR', 'OH'), ('ASN', 'ND2'), ('GLN', 'NE2'), ('ARG', 'NE'), ('ARG', 'NH1'), ('ARG', 'NH2'), ('LYS', 'NZ'),
        ('SER', 'OG'), ('THR', 'OG1'), ('TRP', 'NE1'), ('HIS', 'ND1'), ('HIS', 'NE2')}
donor = np.array([(n[0], n[2]) in SIDE or (n[2] == 'N' and n[0] != 'PRO') for n in names])
lab = [f'{n[0][:3]}{n[1]}:{n[2]}' for n in names]
X = og.X
D = '/vast/xvg/tetramer-mc-runs/morph-growth-20261012/pair-binding'


def sites(rel):
    t, R = rel
    B = t + X @ R.T
    allc = np.vstack([X, B])
    tree = cKDTree(allc)
    ia, ib = np.where(donor)[0], np.where(donor)[0]
    tb = cKDTree(B[ib])
    found = []
    for i in ia:
        for jj in tb.query_ball_point(X[i], 7.2):
            j = ib[jj]
            d = np.linalg.norm(X[i] - B[j])
            if d < 4.8:
                continue
            m = (X[i] + B[j]) / 2
            if not (2.6 <= d / 2 <= 3.6):
                continue
            near = tree.query_ball_point(m, 2.9)
            if any(k not in (i, len(X) + j) for k in near):
                continue
            if all(np.linalg.norm(m - f[0]) > 2.5 for f in found):
                found.append((m, lab[i], lab[j]))
    return [(a, b) for _, a, b in found]


out = {}
for k, nat in enumerate(og.NAT):
    s = sites(nat)
    out[f'native m{k}'] = s
    print(f'native m{k}: {len(s)} site(s) {s}')
refs = json.load(open(f'{D}/contacts_monomer.json')) + json.load(open(f'{D}/contacts_blind_top32.json'))[:12]
counts = {}
for r in refs:
    if r['label'].startswith('native'):
        continue
    s = sites(og.arr(r['pose']))
    out[r['label']] = s
    counts[r['label']] = len(s)
print('\nnon-native contacts, sites:', {k: v for k, v in counts.items() if v})
print('non-native contacts with >= 1 site:', sum(v > 0 for v in counts.values()), 'of', len(counts),
      '; mean sites', round(np.mean(list(counts.values())), 2))
json.dump(out, open('/vast/xvg/tetramer-mc-runs/morph-growth-20261012/anion_sites.json', 'w'), indent=1)
