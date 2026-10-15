"""Ions and water at the native crystal contacts of high-resolution / neutron lysozyme structures.

For each structure (read as data with gemmi): the asymmetric unit (protein heavy atoms, first
altloc; waters O; Cl, Na) and its crystal neighbours from the space-group operations and
lattice translations. Each neighbour is classified as a native contact m0..m6 by its relative
pose in the model frame (structure CA superposed on the monomer-shape CA; scaled error <= 1 to
observe_growth.NAT). Per contact (summed over neighbours of that type around molecule A):
  direct polar pairs (N/O across within 3.5 A), salt bridges (4 A),
  bridging ions (Cl / Na within 3.6 A of atoms of both A and B; all symmetry copies),
  bridging waters (within 3.5 A of polar atoms of both A and B; all copies),
  second-shell water links (water-water pairs within 3.2 A bridging A-bound and B-bound waters).
Usage: hydration_contacts.py PDB...   (run with the gudhienv python, -I)
"""
import itertools
import json
import sys

import gemmi
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, '/vast/xvg/tetramer-mc-runs/morph-growth-20261012')
import observe_growth as og  # noqa: E402

names = json.load(open('/vast/xvg/tetramer-mc-runs/oligomer-capture-20261010/dft-ref/atom_types_monomer.json'))['atoms']
CA_model = {n[1]: og.X[i] for i, n in enumerate(names) if n[2] == 'CA'}


def kabsch(P, Q):
    pc, qc = P.mean(0), Q.mean(0)
    U, _, Vt = np.linalg.svd((P - pc).T @ (Q - qc))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, qc - pc @ R.T


for path in sys.argv[1:]:
    st = gemmi.read_structure(path)
    st.remove_hydrogens()
    st.remove_alternative_conformations()
    cell, sg = st.cell, gemmi.SpaceGroup(st.spacegroup_hm)
    prot, plab, ca, wat, ions = [], [], {}, [], []
    for ch in st[0]:
        for r in ch:
            for a in r:
                p = np.array(a.pos.tolist())
                if r.name in ('HOH', 'DOD', 'WAT'):
                    if a.element.name == 'O':
                        wat.append(p)
                elif r.name in ('CL', 'NA'):
                    ions.append((r.name, p))
                elif r.het_flag != 'H' or r.name in gemmi.find_tabulated_residue(r.name).name:
                    if gemmi.find_tabulated_residue(r.name) and gemmi.find_tabulated_residue(r.name).is_amino_acid():
                        prot.append(p)
                        plab.append((r.name, r.seqid.num, a.name, a.element.name))
                        if a.name == 'CA':
                            ca[r.seqid.num] = p
    prot, wat = np.array(prot), np.array(wat)
    common = sorted(set(ca) & set(CA_model))
    Rm, tm = kabsch(np.array([ca[k] for k in common]), np.array([CA_model[k] for k in common]))
    rms = np.sqrt(np.mean(np.sum((np.array([ca[k] for k in common]) @ Rm.T + tm - np.array([CA_model[k] for k in common])) ** 2, 1)))
    elem = np.array([x[3] for x in plab])
    polar = np.isin(elem, ['N', 'O'])
    last = max(x[1] for x in plab)
    posq = np.array([(x[0] == 'LYS' and x[2] == 'NZ') or (x[0] == 'ARG' and x[2] in ('NE', 'NH1', 'NH2'))
                     or (x[1] == 1 and x[2] == 'N') for x in plab])
    negq = np.array([(x[0] == 'ASP' and x[2] in ('OD1', 'OD2')) or (x[0] == 'GLU' and x[2] in ('OE1', 'OE2'))
                     or (x[1] == last and x[2] in ('O', 'OXT')) for x in plab])
    F = np.array(cell.frac.mat.tolist())     # fractional <- orthogonal (no offsets in PDB files)
    O = np.array(cell.orth.mat.tolist())
    cen = prot.mean(0)

    def apply(op, t, X):
        rot = np.array(op.rot) / gemmi.Op.DEN
        tr = np.array(op.tran) / gemmi.Op.DEN + t
        return ((X @ F.T) @ rot.T + tr) @ O.T

    images = []
    for op in sg.operations():
        for t in itertools.product((-1, 0, 1), repeat=3):
            if op.triplet() == 'x,y,z' and t == (0, 0, 0):
                continue
            B = apply(op, np.array(t, float), prot)
            if np.linalg.norm(B.mean(0) - cen) < 2 * og.BOUND + 4:
                images.append((op, np.array(t, float), B))
    # all symmetry copies of waters and ions near A
    allw, alli = [wat], [(n, p) for n, p in ions]
    for op in sg.operations():
        for t in itertools.product((-2, -1, 0, 1, 2), repeat=3):
            if op.triplet() == 'x,y,z' and t == (0, 0, 0):
                continue
            Wt = apply(op, np.array(t, float), wat)
            allw.append(Wt[np.linalg.norm(Wt - cen, axis=1) < 45])
            for n, p in ions:
                q = apply(op, np.array(t, float), p[None])[0]
                if np.linalg.norm(q - cen) < 45:
                    alli.append((n, q))
    W = np.vstack(allw)
    tA, tApol = cKDTree(prot), cKDTree(prot[polar])
    stats = {}
    for op, t, B in images:
        # relative pose in the model frame
        RB, tB = kabsch(prot, B)               # B = prot @ RB.T + tB
        Rrel_s = RB
        Rmod = Rm @ Rrel_s @ Rm.T
        tmod = Rm @ tB + tm - Rmod @ tm
        rel = (tmod, Rmod)                     # pose of B relative to A (A at identity) in model frame
        errs = [og.scaled(rel, n) for n in og.NAT]
        k = int(np.argmin(errs))
        if errs[k] > 1:
            continue
        name = f'm{k}'
        s = stats.setdefault(name, dict(neighbours=0, polar_pairs=0, salt_bridges=0, ions=[], waters=0, water_links=0, err=[]))
        s['neighbours'] += 1
        s['err'].append(round(errs[k], 2))
        tB_ = cKDTree(B)
        tBpol = cKDTree(B[polar])
        s['polar_pairs'] += sum(len(x) for x in tApol.query_ball_tree(tBpol, 3.5))
        s['salt_bridges'] += sum(len(x) for x in cKDTree(prot[posq]).query_ball_tree(cKDTree(B[negq]), 4.0)) + \
            sum(len(x) for x in cKDTree(prot[negq]).query_ball_tree(cKDTree(B[posq]), 4.0))
        for n, q in alli:
            if tA.query(q)[0] < 3.6 and tB_.query(q)[0] < 3.6:
                ra = sorted({f'{plab[i][0]}{plab[i][1]}:{plab[i][2]}' for i in tA.query_ball_point(q, 3.6)})
                rb = sorted({f'{plab[i][0]}{plab[i][1]}:{plab[i][2]}' for i in tB_.query_ball_point(q, 3.6)})
                s['ions'].append(f"{n} [A {' '.join(ra)} | B {' '.join(rb)}]")
        dA = tApol.query(W)[0]
        dB = tBpol.query(W)[0]
        bridge = (dA < 3.5) & (dB < 3.5)
        s['waters'] += int(bridge.sum())
        onlyA, onlyB = np.where((dA < 3.5) & ~bridge)[0], np.where((dB < 3.5) & ~bridge)[0]
        if len(onlyA) and len(onlyB):
            s['water_links'] += sum(len(x) for x in cKDTree(W[onlyA]).query_ball_tree(cKDTree(W[onlyB]), 3.2))
    print(f"\n{path.split('/')[-1]}: CA superposition rms {rms:.2f} A; {len(wat)} waters, ions "
          f"{[n for n, _ in ions]}; {sum(s['neighbours'] for s in stats.values())} native-classified neighbours")
    for name in sorted(stats, key=lambda x: int(x[1:])):
        s = stats[name]
        print(f"  {name}: x{s['neighbours']} (pose err {s['err']}); direct polar pairs {s['polar_pairs']}, salt bridges "
              f"{s['salt_bridges']}, bridging ions {s['ions'] or '-'}, bridging waters {s['waters']}, "
              f"second-shell water links {s['water_links']}")
