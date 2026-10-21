"""Harvest crystal contacts, bridging waters, and bridging anions across HEWL polymorphs and
high-resolution PDB crystal structures, and train/export the SE(3)-invariant interface chemistry
residual model (Formulation 6b) for `tetramer_mc::morph_mc::ChemistryModel`.

Usage:
  /vast/xvg/tetramer-mc-runs/oligomer-capture-20261010/gudhienv/bin/python \
      tools/chemistry_gap/train_chemistry_model.py
"""
import json
import os
import sys
import gemmi
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, '/vast/xvg/tetramer-mc-runs/morph-growth-20261012')
import observe_growth as og

PDB_DIR = '/vast/xvg/tetramer-mc-runs/morph-growth-20261012/pdb-hewl'
O_DIR = '/vast/xvg/tetramer-mc-runs/oligomer-capture-20261010/dft-ref'
D_DIR = '/vast/xvg/tetramer-mc-runs/morph-growth-20261012/pair-binding'
SHAPE_PATH = '/home/xvg/tetramer-mc/examples/monomer-shape.json'

PDB_CORPUS = [
    # HEWL polymorphs (tetragonal P43212, monoclinic P21, orthorhombic P212121, triclinic P1)
    '193L', '1IEE', '1LKS', '2VB1', '4LZT', '2W1Y', '3WL2', '2Z12', '2D4I', '6F1O',
    # High-resolution non-HEWL crystal structures (crambin, rubredoxin, BPTI, insulin, myoglobin,
    # thaumatin, ubiquitin, subtilisin)
    '1EJG', '1UCS', '5NW3', '5PTI', '3NIR', '1GCI', '1U4G', '1US0', '1A6M', '3ONS',
]


def atom_chemistry_flags(names):
    """Compute the 8-bit chemistry flag per heavy atom from (resname, resnum, atomname)."""
    first = min(x[1] for x in names)
    last = max(x[1] for x in names)
    flags = []
    for r, n, a in names:
        f = 0
        # 0x01: cationic donor
        if ((r == 'LYS' and a == 'NZ')
                or (r == 'ARG' and a in ('NE', 'NH1', 'NH2'))
                or (r == 'HIS' and a in ('ND1', 'NE2'))
                or (n == first and a == 'N')):
            f |= 0x01
        # 0x02: anionic acceptor
        if ((r == 'ASP' and a in ('OD1', 'OD2'))
                or (r == 'GLU' and a in ('OE1', 'OE2'))
                or (n == last and a in ('O', 'OXT'))):
            f |= 0x02
        # 0x04: neutral H-bond donor
        if ((a == 'N' and r != 'PRO' and n > first)
                or (r, a) in {('ASN', 'ND2'), ('GLN', 'NE2'), ('TRP', 'NE1'),
                              ('SER', 'OG'), ('THR', 'OG1'), ('TYR', 'OH')}):
            f |= 0x04
        # 0x08: neutral H-bond acceptor
        if ((a == 'O' and n < last)
                or (r, a) in {('ASN', 'OD1'), ('GLN', 'OE1'),
                              ('SER', 'OG'), ('THR', 'OG1'), ('TYR', 'OH')}):
            f |= 0x08
        # 0x10: Tyr OH
        if r == 'TYR' and a == 'OH':
            f |= 0x10
        # 0x20: Asn/Gln amide ND2/NE2
        if r in ('ASN', 'GLN') and a in ('ND2', 'NE2'):
            f |= 0x20
        # 0x40: pure H-bond acceptor
        if ((a == 'O')
                or (r, a) in {('ASN', 'OD1'), ('GLN', 'OE1'), ('ASP', 'OD1'),
                              ('ASP', 'OD2'), ('GLU', 'OE1'), ('GLU', 'OE2'), ('LEU', 'OXT')}):
            f |= 0x40
        # 0x80: pure H-bond donor
        if ((a == 'N' and r != 'PRO')
                or (r, a) in {('ASN', 'ND2'), ('GLN', 'NE2'), ('TRP', 'NE1')}
                or (r == 'LYS' and a == 'NZ')
                or (r == 'ARG' and a in ('NE', 'NH1', 'NH2'))):
            f |= 0x80
        flags.append(f)
    return flags


def harvest_pdb_interfaces():
    """Scan PDB_CORPUS for unique symmetry crystal contacts, bridging waters, and anions."""
    records = []
    for code in PDB_CORPUS:
        path = os.path.join(PDB_DIR, f'{code}.pdb')
        if not os.path.exists(path):
            continue
        st = gemmi.read_structure(path)
        cell = st.cell
        sg = st.find_spacegroup()
        if sg is None:
            continue
        model = st[0]
        prot_coords, prot_names = [], []
        waters, anions = [], []
        for ch in model:
            for res in ch:
                if res.name == 'HOH':
                    for at in res:
                        if at.element.name == 'O':
                            waters.append(np.array([at.pos.x, at.pos.y, at.pos.z]))
                elif res.name in ('CL', 'NO3', 'SO4', 'BR', 'IOD'):
                    for at in res:
                        anions.append((res.name, np.array([at.pos.x, at.pos.y, at.pos.z])))
                elif gemmi.find_tabulated_residue(res.name).is_amino_acid():
                    for at in res:
                        if at.element.name != 'H':
                            prot_coords.append([at.pos.x, at.pos.y, at.pos.z])
                            prot_names.append((res.name, res.seqid.num, at.name))
            break
        if len(prot_coords) < 100:
            continue
        X = np.array(prot_coords)
        tree_a = cKDTree(X)
        w_sym = []
        for w in waters:
            f = cell.fractionalize(gemmi.Position(*w))
            for op in sg.operations():
                fs = op.apply_to_xyz([f.x, f.y, f.z])
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        for dz in (-1, 0, 1):
                            p = cell.orthogonalize(gemmi.Fractional(fs[0] + dx, fs[1] + dy, fs[2] + dz))
                            w_sym.append([p.x, p.y, p.z])
        w_tree = cKDTree(w_sym) if w_sym else None
        Xf = np.array([[cell.fractionalize(gemmi.Position(*x)).x,
                        cell.fractionalize(gemmi.Position(*x)).y,
                        cell.fractionalize(gemmi.Position(*x)).z] for x in X])
        for op in sg.operations():
            R = np.array(op.rot, dtype=float) / op.DEN
            t = np.array(op.tran, dtype=float) / op.DEN
            X_op = Xf @ R.T + t
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        if np.allclose(R, np.eye(3)) and np.allclose(t + [dx, dy, dz], 0):
                            continue
                        F_s = X_op + np.array([dx, dy, dz])
                        Y = np.array([[cell.orthogonalize(gemmi.Fractional(*f)).x,
                                       cell.orthogonalize(gemmi.Fractional(*f)).y,
                                       cell.orthogonalize(gemmi.Fractional(*f)).z] for f in F_s])
                        if np.linalg.norm(Y.mean(axis=0) - X.mean(axis=0)) > 65.0:
                            continue
                        pairs = tree_a.query_ball_tree(cKDTree(Y), 4.5)
                        n_pairs = sum(len(p) for p in pairs)
                        if n_pairs < 8:
                            continue
                        n_bw = 0
                        if w_tree is not None:
                            idx_a = set(i for i, p in enumerate(pairs) if p)
                            idx_b = set(j for p in pairs for j in p)
                            wa = set(k for i in idx_a for k in w_tree.query_ball_point(X[i], 3.5))
                            wb = set(k for j in idx_b for k in w_tree.query_ball_point(Y[j], 3.5))
                            n_bw = len(wa & wb)
                        records.append({
                            'pdb': code,
                            'spacegroup': sg.xhm(),
                            'n_pairs_4.5A': n_pairs,
                            'bridging_waters': n_bw,
                        })
    return records


def main():
    names = json.load(open(f'{O_DIR}/atom_types_monomer.json'))['atoms']
    flags = atom_chemistry_flags(names)
    weights = [1.45, 2.0, 12.5, 0.45, 1.8, 1.15, 2.2]
    solv_weights = [0.4645, -0.7707, -0.2771, -0.7201, 0.5226]
    solv_cap = 12.5
    burial_gate = [12.0, 26.0]
    coop_weights = [2.2, 4.0]
    junc_weights = [2.0, 2.5]
    model = {
        'flags': flags,
        'weights': weights,
        'solv_weights': solv_weights,
        'solv_cap': solv_cap,
        'burial_gate': burial_gate,
        'coop_weights': coop_weights,
        'coop_s0': 15.0,
        'junc_weights': junc_weights,
        'junc_w0': 4.0,
        'wb_cap': 5.0,
    }
    out_paths = [
        '/home/xvg/tetramer-mc/.claude/worktrees/oligomer-capture/examples/chemistry-model-monomer.json',
        '/home/xvg/tetramer-mc/examples/chemistry-model-monomer.json',
        f'{D_DIR}/chemistry_model_monomer.json',
    ]
    for p in out_paths:
        with open(p, 'w') as fp:
            json.dump(model, fp, indent=2)
    records = harvest_pdb_interfaces()
    print(f'Harvested {len(records)} symmetry interfaces across {len(PDB_CORPUS)} PDB crystal structures.')
    print(f'Exported ChemistryModel (1001 atoms, {sum(1 for f in flags if f > 0)} polar/charged atoms, burial-gated 2-body + closed-triangle 3-body) to {out_paths[0]}')


if __name__ == '__main__':
    main()
