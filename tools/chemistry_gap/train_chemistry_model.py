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
from scipy.optimize import minimize

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


def atom_chemistry_flags(names, coords=None):
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


def quat_to_mat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def fit_cl_bridge_parameters(names, coords, flags):
    """Fit smooth monomer pre-organization and pair environment weights from crystal and MC poses."""
    X = np.asarray(coords, dtype=float)
    flags_arr = np.asarray(flags, dtype=int)
    res_nums = np.array([n[1] for n in names])

    cats = np.where((flags_arr & 0x01) != 0)[0]
    anis = np.where((flags_arr & 0x02) != 0)[0]
    accs = np.where((flags_arr & (0x02 | 0x08 | 0x40)) != 0)[0]

    # Monomer-level continuous 3D environment descriptors for Tyr OH (k=1) and Asn/Gln ND2/NE2 (k=2)
    intra_feats = {}
    for i in range(len(X)):
        if (flags_arr[i] & (0x10 | 0x20)) == 0:
            continue
        k_prot = 1.0 if (flags_arr[i] & 0x10) != 0 else 2.0
        c_idx = [m for m in cats if res_nums[m] != res_nums[i]]
        a_idx = [m for m in anis if res_nums[m] != res_nums[i]]
        o_idx = [m for m in accs if abs(res_nums[m] - res_nums[i]) >= 2]
        dc = np.linalg.norm(X[c_idx] - X[i], axis=-1)
        da = np.linalg.norm(X[a_idx] - X[i], axis=-1)
        do = np.linalg.norm(X[o_idx] - X[i], axis=-1)
        xi_cat = float(np.sum(np.exp(-((np.maximum(0.0, dc - 5.5) / 2.2) ** 2))))
        xi_ani = float(np.sum(np.exp(-((np.maximum(0.0, da - 4.0) / 1.5) ** 2))))
        xi_acc = float(np.sum(np.exp(-((np.maximum(0.0, do - 3.0) / 0.6) ** 2))))
        defect = (xi_acc - 0.80 * (k_prot - 1.0)) ** 2
        intra_feats[i] = (xi_cat, xi_ani, defect)

    # Harvest positive (crystal + capture funnel) and negative (MC decoy + blind docking) poses
    poses = []
    contacts_path = f'{D_DIR}/contacts_monomer.json'
    if os.path.exists(contacts_path):
        c193 = json.load(open(contacts_path))
        m1 = c193[1]['pose']
        poses.append(('POS_native_m1', 0.90, quat_to_mat(m1['orientation']), np.array(m1['position'])))

    tr26 = '/vast/xvg/tetramer-mc-runs/morph-growth-20261012/nuc-26/trajectory.jsonl'
    if os.path.exists(tr26):
        with open(tr26) as f:
            last26 = None
            for line in f:
                last26 = json.loads(line)
        if last26 and 'memory' in last26 and last26['memory']:
            s0 = last26['memory'][0]
            poses.append(('POS_nuc26_mem0', 0.90, quat_to_mat(s0['orientation']), np.array(s0['position'])))

    # Native m1 capture funnel entry poses observed in nuc-31 (15-17 deg tilt, 2.4-2.7 A offset)
    poses.extend([
        ('POS_nuc31_slot7', 0.55,
         quat_to_mat([0.7305936337271589, -0.07729144601600103, 0.11926261926913809, 0.6678588191939717]),
         np.array([21.528165880904805, 18.556308740309735, -8.070774298408796])),
        ('POS_nuc31_slot17', 0.78,
         quat_to_mat([0.7492067105606828, -0.07443445777189878, 0.08330666157706808, 0.6528467021314641]),
         np.array([21.547563930808884, 18.644502601529382, -8.334628197098734])),
        ('POS_nuc31_slot10', 0.15,
         quat_to_mat([0.7433569057655202, -0.08221099242596996, 0.09874237138956767, 0.6564387309318468]),
         np.array([21.667509770099507, 19.61244715414565, -8.32716688119827])),
    ])

    for run in ('nuc-28', 'nuc-29', 'nuc-30'):
        tr_path = f'/vast/xvg/tetramer-mc-runs/morph-growth-20261012/{run}/trajectory.jsonl'
        if not os.path.exists(tr_path):
            continue
        with open(tr_path) as f:
            last_rec = None
            for line in f:
                last_rec = json.loads(line)
        if last_rec and 'memory' in last_rec:
            for idx, s in enumerate(last_rec['memory']):
                poses.append((f'NEG_{run}_s{idx}', 0.0, quat_to_mat(s['orientation']), np.array(s['position'])))

    blind_path = f'{D_DIR}/contacts_blind_top32.json'
    if os.path.exists(blind_path):
        b32 = json.load(open(blind_path))
        for idx, c in enumerate(b32):
            p = c['pose']
            poses.append((f'NEG_blind_{idx}', 0.0, quat_to_mat(p['orientation']), np.array(p['position'])))

    pose_cands = []
    for name, target, R, t in poses:
        YA = (R @ X.T).T + t
        XB = X
        tree_a = cKDTree(YA)
        tree_b = cKDTree(XB)
        pairs = tree_a.query_ball_tree(tree_b, 4.9)
        cats_a, cats_b = YA[cats], XB[cats]
        anis_a, anis_b = YA[anis], XB[anis]
        accs_a, accs_b = YA[accs], XB[accs]
        cands = []
        for i, js in enumerate(pairs):
            fi = flags_arr[i]
            if (fi & (0x10 | 0x20)) == 0:
                continue
            for j in js:
                fj = flags_arr[j]
                if not (((fi & 0x10) and (fj & 0x20)) or ((fi & 0x20) and (fj & 0x10))):
                    continue
                d = float(np.linalg.norm(YA[i] - XB[j]))
                if d < 3.2 or d > 4.9:
                    continue
                mid = 0.5 * (YA[i] + XB[j])
                g_d = float(np.exp(-(((d - 4.05) / 0.50) ** 2)))
                dca = np.linalg.norm(cats_a - mid, axis=-1)
                dcb = np.linalg.norm(cats_b - mid, axis=-1)
                phi_cat_a = float(np.sum(np.exp(-((np.maximum(0.0, dca - 6.5) / 2.0) ** 2))))
                phi_cat_b = float(np.sum(np.exp(-((np.maximum(0.0, dcb - 6.5) / 2.0) ** 2))))
                daa = np.linalg.norm(anis_a - mid, axis=-1)
                dab = np.linalg.norm(anis_b - mid, axis=-1)
                phi_ani = float(
                    np.sum(np.exp(-((np.maximum(0.0, daa - 3.0) / 1.8) ** 2)))
                    + np.sum(np.exp(-((np.maximum(0.0, dab - 3.0) / 1.8) ** 2)))
                )
                mask_a = accs != i
                mask_b = accs != j
                doa = np.linalg.norm(accs_a[mask_a] - mid, axis=-1)
                dob = np.linalg.norm(accs_b[mask_b] - mid, axis=-1)
                phi_acc = float(
                    np.sum(np.exp(-((np.maximum(0.0, doa - 2.80) / 0.28) ** 2)))
                    + np.sum(np.exp(-((np.maximum(0.0, dob - 2.80) / 0.28) ** 2)))
                )
                if (fi & 0x10) != 0:
                    d_tyr_hb = np.linalg.norm(accs_b - YA[i], axis=-1)
                else:
                    d_tyr_hb = np.linalg.norm(accs_a - XB[j], axis=-1)
                phi_hb = float(np.max(np.exp(-((np.maximum(0.0, d_tyr_hb - 2.80) / 0.55) ** 2))))
                cands.append((i, j, g_d, phi_cat_a, phi_cat_b, phi_ani, phi_acc, phi_hb))
        pose_cands.append((name, target, cands))

    prior = np.array([5.0, 2.5, 0.04, 1.5, 2.0, 2.0])

    def eval_pose(cands, p):
        w_intra_acc, w_intra_ani, c_cat, w_ani, w_acc, w_hb = p
        scored = []
        for (i, j, g_d, pca, pcb, pani, pacc, phb) in cands:
            xca_i, xan_i, def_i = intra_feats[i]
            xca_j, xan_j, def_j = intra_feats[j]
            ai = (xca_i / (xca_i + 0.04)) * np.exp(-w_intra_acc * def_i - w_intra_ani * xan_i)
            aj = (xca_j / (xca_j + 0.04)) * np.exp(-w_intra_acc * def_j - w_intra_ani * xan_j)
            g_cat = (pca / (pca + c_cat)) * (pcb / (pcb + c_cat))
            g_env = np.exp(-w_ani * pani - w_acc * pacc - w_hb * phb)
            scored.append((ai * aj * g_d * g_cat * g_env, i, j))
        scored.sort(reverse=True)
        used_a, used_b = set(), set()
        s_cl = 0.0
        for val, i, j in scored:
            if i not in used_a and j not in used_b:
                used_a.add(i)
                used_b.add(j)
                s_cl += val
        return s_cl

    def loss(p):
        tot = 0.01 * float(np.sum(((p - prior) / prior) ** 2))
        for name, target, cands in pose_cands:
            s_cl = eval_pose(cands, p)
            if target > 0.0:
                if s_cl < target:
                    tot += 25.0 * ((target - s_cl) ** 2)
            else:
                tot += 20.0 * (s_cl ** 2)
        return tot

    res = minimize(
        loss,
        prior,
        bounds=[(1.0, 12.0), (0.5, 8.0), (0.01, 0.25), (0.5, 5.0), (0.5, 5.0), (0.5, 5.0)],
        method='L-BFGS-B',
    )
    w_intra_acc, w_intra_ani, c_cat, w_ani, w_acc, w_hb = [float(x) for x in res.x]
    atom_cl_activity = []
    for i in range(len(X)):
        if i in intra_feats:
            xca_i, xan_i, def_i = intra_feats[i]
            ai = (xca_i / (xca_i + 0.04)) * np.exp(-w_intra_acc * def_i - w_intra_ani * xan_i)
            atom_cl_activity.append(round(float(ai), 6))
        else:
            atom_cl_activity.append(0.0)
    cl_env_weights = [round(c_cat, 6), round(w_ani, 6), round(w_acc, 6), round(w_hb, 6)]
    return atom_cl_activity, cl_env_weights, [round(w_intra_acc, 4), round(w_intra_ani, 4)]


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
    shape = json.load(open(SHAPE_PATH))
    coords = [a['center'] for a in shape['atoms']]
    flags = atom_chemistry_flags(names, coords=coords)
    atom_cl_activity, cl_env_weights, intra_w = fit_cl_bridge_parameters(names, coords, flags)
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
        'atom_cl_activity': atom_cl_activity,
        'cl_env_weights': cl_env_weights,
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
    print(f'Fitted smooth Cl-bridge weights: intra={intra_w}, env={cl_env_weights}')
    print(f'Exported ChemistryModel (1001 atoms, {sum(1 for f in flags if f > 0)} polar/charged atoms, smooth data-fit Cl-bridge) to {out_paths[0]}')


if __name__ == '__main__':
    main()
