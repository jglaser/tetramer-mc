"""Chemistry of the native crystal contacts m0-m6 (1LYZ).

The 1LYZ protein atoms are superimposed (Kabsch) on the monomer-shape frame; the 101 crystal
waters are carried along, so each molecule has its own copy. For each native contact (B at
the native relative pose):
  - interface residues (heavy atoms within 4 A of the partner);
  - hydrogen-bond candidates: polar N/O pairs across the interface within 3.5 A;
  - salt bridges: Lys NZ / Arg NE,NH1,NH2 / N-term N with Asp OD / Glu OE / C-term O,OXT within 4 A;
  - apolar contacts: carbon-carbon pairs within 4 A;
  - bridging waters: a water (from A's or B's copy) within 3.5 A of polar atoms of both A and B.
With the DFT (fixed code) and morphometric contact free energies for comparison.
"""
import json
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, '/vast/xvg/tetramer-mc-runs/morph-growth-20261012')
import observe_growth as og  # noqa: E402

pdb = [l for l in open('/home/xvg/protein-nucleation/data/1LYZ.pdb') if l.startswith(('ATOM', 'HETATM'))]
prot = np.array([[float(l[30:38]), float(l[38:46]), float(l[46:54])] for l in pdb if l.startswith('ATOM')])
wat = np.array([[float(l[30:38]), float(l[38:46]), float(l[46:54])] for l in pdb
                if l.startswith('HETATM') and l[17:20] == 'HOH' and l[12:16].strip() == 'O'])
names = json.load(open('/vast/xvg/tetramer-mc-runs/oligomer-capture-20261010/dft-ref/atom_types_monomer.json'))['atoms']
X = og.X
assert len(prot) == len(X) == len(names)
# Kabsch: prot -> X
pc, xc = prot.mean(0), X.mean(0)
H = (prot - pc).T @ (X - xc)
U, _, Vt = np.linalg.svd(H)
d = np.sign(np.linalg.det(Vt.T @ U.T))
R = Vt.T @ np.diag([1, 1, d]) @ U.T
fit = (prot - pc) @ R.T + xc
rmsd = np.sqrt(np.mean(np.sum((fit - X) ** 2, axis=1)))
W = (wat - pc) @ R.T + xc
print(f'1LYZ superposition on the shape frame: rmsd {rmsd:.3f} A; {len(W)} waters')

elem = np.array([n[2][0] for n in names])
polar = np.isin(elem, ['N', 'O'])
last = max(n[1] for n in names)
pos = np.array([(n[0] == 'LYS' and n[2] == 'NZ') or (n[0] == 'ARG' and n[2] in ('NE', 'NH1', 'NH2'))
                or (n[1] == 1 and n[2] == 'N') for n in names])
neg = np.array([(n[0] == 'ASP' and n[2] in ('OD1', 'OD2')) or (n[0] == 'GLU' and n[2] in ('OE1', 'OE2'))
                or (n[1] == last and n[2] in ('O', 'OXT')) for n in names])
carbon = elem == 'C'
ONE = dict(ALA='A', ARG='R', ASN='N', ASP='D', CYS='C', GLN='Q', GLU='E', GLY='G', HIS='H', ILE='I', LEU='L',
           LYS='K', MET='M', PHE='F', PRO='P', SER='S', THR='T', TRP='W', TYR='Y', VAL='V')
lab = [f'{ONE[n[0]]}{n[1]}:{n[2]}' for n in names]
res = [f'{ONE[n[0]]}{n[1]}' for n in names]
DFT = dict(m6=-16.5, m1=-11.3, m4=-5.2, m2=-3.5, m3=-0.3, m0=0.1, m5=0.6)
WB = dict(m6=-10.2, m1=-9.3, m4=-4.6, m2=-0.9, m3=0.0, m0=0.3, m5=-0.1)


def pairs(A, B, ma, mb, cut):
    ta, tb = cKDTree(A[ma]), cKDTree(B[mb])
    ia, ib = np.where(ma)[0], np.where(mb)[0]
    return [(ia[i], ib[j]) for i, js in enumerate(ta.query_ball_tree(tb, cut)) for j in js]


out = {}
for k, (t, Rk) in enumerate(og.NAT):
    B = t + X @ Rk.T
    WB_ = t + W @ Rk.T
    ia = sorted({res[i] for i, _ in pairs(X, B, np.ones(len(X), bool), np.ones(len(X), bool), 4.0)}, key=lambda s: int(s[1:]))
    ib = sorted({res[j] for _, j in pairs(X, B, np.ones(len(X), bool), np.ones(len(X), bool), 4.0)}, key=lambda s: int(s[1:]))
    hb = pairs(X, B, polar, polar, 3.5)
    sb = pairs(X, B, pos, neg, 4.0) + [(i, j) for j, i in pairs(B, X, pos, neg, 4.0)]
    hp = pairs(X, B, carbon, carbon, 4.0)
    bridging = 0
    tA, tB = cKDTree(X[polar]), cKDTree(B[polar])
    for w in np.vstack([W, WB_]):
        if tA.query(w)[0] < 3.5 and tB.query(w)[0] < 3.5:
            bridging += 1
    name = f'm{k}'
    out[name] = dict(residues_A=ia, residues_B=ib, hbonds=[(lab[i], lab[j]) for i, j in hb],
                     salt_bridges=[(lab[i], lab[j]) for i, j in sb], apolar_pairs=len(hp), bridging_waters=bridging)
    print(f'\n{name}: DFT {DFT[name]:+.1f}, morphometry {WB[name]:+.1f} kT; residues A {len(ia)} / B {len(ib)}; '
          f'H-bond pairs {len(hb)}, salt bridges {len(sb)}, apolar C-C {len(hp)}, bridging waters {bridging}')
    print('   A:', ' '.join(ia))
    print('   B:', ' '.join(ib))
    if hb:
        print('   H-bond pairs:', ', '.join(f'{lab[i]}-{lab[j]}' for i, j in hb))
    if sb:
        print('   salt bridges:', ', '.join(f'{lab[i]}-{lab[j]}' for i, j in sb))
json.dump(out, open('/vast/xvg/tetramer-mc-runs/morph-growth-20261012/interface_chemistry.json', 'w'), indent=1)
