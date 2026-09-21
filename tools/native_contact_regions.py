"""Stateless registered-native entry regions for independent fixed-scaffold poses.

This observer is not a hard/wall validator or a thermodynamic target. Callers
retain the original q/rho partition and all unconditional zero-weight draws.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
from pathlib import Path
import shutil

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'runs/mobile-reciprocal-atlas-benchmark-20260921/reciprocal/provenance'
CONFIG = ROOT/'runs/mobile-competing-reference-preparation-20260921/config.json'
SNAPSHOT = ROOT/'runs/mobile-competing-geometry-20260921/fixed-snapshot.json'
SCHEMA = 'stateless-native-contact-regions-v1'
CRITERIA = dict(body_member_position_entry_A=2., body_orientation_entry_deg=15.,
    monomer_position_entry_A=3., monomer_orientation_entry_deg=20., contact_entry_A=2.,
    native_reference_patch_gap_A=1., minimum_shared_native_residue_pairs=1,
    hard_overlap_tolerance_A=1e-8, catalogue_cycle_position_tolerance_A=1e-6,
    catalogue_cycle_angle_tolerance_deg=1e-6)
SCOPE = ('Instantaneous registered native entry relative to each fixed neighbor. All directed motif matches '
    'are retained. No exit criterion, hysteresis, persistence, q filter or physical weight enters this observer. '
    'No-entry is not proof of nonnative geometry. Both-anchor entry is a geometric attachment descriptor, '
    'not energetic cooperativity. Exact local catalogue cycle closure is not global lattice certification. '
    'Physical hard/wall validity is a separate caller precondition.')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def require(value, message):
    if not value:
        raise ValueError(message)


def pose_arrays(pose):
    t, q = np.asarray(pose['position'], float), np.asarray(pose['orientation'], float)
    require(t.shape == (3,) and q.shape == (4,) and np.isfinite(t).all() and np.isfinite(q).all(), 'Invalid finite pose')
    require(abs(np.linalg.norm(q)-1) < 1e-8, 'Pose quaternion must be normalized')
    return t, Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()


def make_pose(t, r):
    return dict(position=np.asarray(t).tolist(), orientation=Rotation.from_matrix(r).as_quat()[[3, 0, 1, 2]].tolist())


def compose(anchor, relative):
    t, r = pose_arrays(anchor);d, s = pose_arrays(relative)
    return make_pose(t+r@d, r@s)


def inverse(pose):
    t, r = pose_arrays(pose)
    return make_pose(-r.T@t, r.T)


def angle_degrees(a, b):
    return float(np.degrees(np.arccos(np.clip((np.sum(a*b)-1)/2, -1., 1.))))


def freeze_definition(out, source=SOURCE, config=CONFIG, snapshot=SNAPSHOT):
    """Archive fixed criteria/data; this performs no MC or bath-weight queries."""
    out, source = Path(out).resolve(), Path(source).resolve()
    require(not out.exists(), 'Fresh definition directory required')
    inputs = {'physical-config.json': Path(config), 'fixed-snapshot.json': Path(snapshot),
        'monomer-shape.json': source/'monomer-shape.json', 'native-pair-motifs.json': source/'native-pair-motifs.json',
        'tetramer-shape.json': Path(read(config)['shape'])}
    inputs.update({p.relative_to(source).as_posix():p for p in sorted((source/'reference').rglob('*')) if p.is_file()})
    require(len([n for n in inputs if n.startswith('reference/')]) == 8, 'Expected complete eight-file archived reference')
    for name in ('native_contact_regions.py', 'test_native_contact_regions.py'):
        path = Path(__file__).with_name(name)
        if path.exists():inputs['source/'+name] = path
    for name, path in inputs.items():
        target = out/'inputs'/name;target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        require(sha(path) == sha(target), 'Input changed while archiving')
    cfg = read(config)
    definition = dict(schema=SCHEMA, criteria=CRITERIA, scope=SCOPE,
        moving_body_id=0, fixed_body_ids=[2, 1], fixed_poses=cfg['fixed_poses'],
        shape_sha256=sha(inputs['tetramer-shape.json']),
        physical_config_sha256=sha(config), original_capture_radius_A=cfg['capture_radius'],
        interpretation='Original q<=1 uses one reference. The union of these registered native entry motifs is a separate deterministic reporting label, never a replacement target.',
        reference_definition='TetramerOrder.registered_body_graphs entry, with prescribed monomer bonds from match_classes entry; open geometry with no periodic images.',
        input_sha256={name:sha(path) for name,path in inputs.items()},
        source_paths={name:str(path.resolve()) for name,path in inputs.items()})
    save(out/'definition.json', definition)
    classifier = NativeContactRegions(out/'definition.json')
    save(out/'data-validation.json', dict(shape_sha256=classifier.shape_sha256,
        monomer_atoms=len(classifier.atoms), residue_count=classifier.residue_count,
        residue_coordinate_maximum_error_A=classifier.coordinate_error,
        body_motifs=len(classifier.motifs), directed_monomer_classes=len(classifier.references),
        fixed_scaffold_matches=classifier.fixed_scaffold_matches,
        physical_launches=0, bath_weight_queries=0))
    return definition


class NativeContactRegions:
    """Independent implementation of archived TetramerOrder's entry predicates.

    Runtime imports are only stdlib, numpy and scipy. The frozen observer sources
    are retained as provenance and used independently in tests, not imported here.
    """
    def __init__(self, definition_path):
        self.definition_path = Path(definition_path).resolve()
        self.definition_sha256 = sha(self.definition_path)
        self.definition = read(self.definition_path)
        require(self.definition['schema'] == SCHEMA and self.definition['criteria'] == CRITERIA, 'Unsupported native entry definition')
        self.root = self.definition_path.parent/'inputs'
        required = {'physical-config.json','fixed-snapshot.json','monomer-shape.json','tetramer-shape.json',
            'native-pair-motifs.json','source/native_contact_regions.py',
            'reference/results/native-neighbor-classes/classification.json',
            'reference/results/c1c3-scaffold/motifs.json',
            'reference/results/native-geometry-repair/rebuilt-hydrogens/heavy-coordinates.json',
            'reference/scripts/tetramer_order.py','reference/scripts/analyze_precursor_exchange.py'}
        require(required <= set(self.definition['input_sha256']), 'Incomplete native definition input binding')
        require(sha(__file__) == self.definition['input_sha256']['source/native_contact_regions.py'], 'Use the frozen native classifier source')
        for name, digest in self.definition['input_sha256'].items():
            p = Path(name)
            require(not p.is_absolute() and '..' not in p.parts, 'Unsafe definition input path')
            require(sha(self.root/p) == digest, 'Frozen native definition input changed: '+name)
        cfg = read(self.root/'physical-config.json')
        self.fixed_poses = copy.deepcopy(cfg['fixed_poses'])
        self.fixed_body_ids = self.definition['fixed_body_ids']
        require(self.fixed_poses == self.definition['fixed_poses'] and len(self.fixed_poses) == 2 and self.fixed_body_ids == [2, 1], 'Exact observed two-neighbor scaffold required')
        require(sha(self.root/'physical-config.json') == self.definition['physical_config_sha256'], 'Physical config identity mismatch')
        self.shape_sha256 = sha(self.root/'tetramer-shape.json')
        require(self.shape_sha256 == self.definition['shape_sha256'], 'Physical shape identity mismatch')
        shape = read(self.root/'tetramer-shape.json')
        members = shape['rigid_members']
        self.member_positions = np.asarray([p['position'] for p in members])
        self.member_rotations = np.asarray([pose_arrays(p)[1] for p in members])
        require(members == cfg['metadata']['rigid_members'], 'Body member geometry changed')
        monomer = read(self.root/'monomer-shape.json')
        self.atoms = np.asarray([a['center'] for a in monomer['atoms']])
        self.radii = np.asarray([a['radius'] for a in monomer['atoms']])
        coordinate_path = self.root/'reference/results/native-geometry-repair/rebuilt-hydrogens/heavy-coordinates.json'
        require(sha(coordinate_path) == monomer['source_coordinates_sha256'], 'Authoritative residue coordinate hash mismatch')
        coordinates = read(coordinate_path);xyz = np.asarray(coordinates['positions'])
        records = coordinates['atom_records']
        require(xyz.shape == self.atoms.shape and len(records) == len(self.atoms), 'Residue atom count/order mismatch')
        self.coordinate_error = float(np.max(abs(xyz-xyz.mean(axis=0)-self.atoms)))
        require(self.coordinate_error <= 1e-10, 'Authoritative coordinate rows do not match the monomer')
        keys = [(line[21], line[22:27].strip(), line[17:20].strip()) for line in records]
        names = list(dict.fromkeys(keys));self.residue_count = len(names)
        mapping = {key:i for i,key in enumerate(names)}
        self.residues = np.asarray([mapping[key] for key in keys], dtype=int)
        self.tree = cKDTree(self.atoms)
        self.references = {}
        for ref in read(self.root/'reference/results/native-neighbor-classes/classification.json')['classes']:
            name = ref['label'];require(name not in self.references, 'Duplicate directed monomer class')
            position, rotation = np.asarray(ref['body_delta']), np.asarray(ref['body_relative_rotation'])
            contact = self._contacts(position, rotation, CRITERIA['native_reference_patch_gap_A'])
            require(contact['residue_pairs'], 'Empty native reference residue patch')
            self.references[name] = dict(label=name, family=ref['family'], position=position, rotation=rotation,
                                         native_residue_pairs=set(contact['residue_pairs']))
        reference_motifs = read(self.root/'reference/results/c1c3-scaffold/motifs.json')
        require(reference_motifs['shape_sha256'] == sha(self.root/'monomer-shape.json'), 'Monomer template shape differs')
        catalogue = read(self.root/'native-pair-motifs.json')
        require(catalogue['proper_internal_symmetry_order'] == 1, 'Nontrivial body symmetry requires explicit quotienting')
        self.motifs = catalogue['motifs']
        require(len({m['id'] for m in self.motifs}) == len(self.motifs), 'Duplicate motif IDs')
        self.motif_by_id = {m['id']:m for m in self.motifs}
        self.motif_poses = {m['id']:dict(position=m['relative_position'], orientation=m['relative_orientation']) for m in self.motifs}
        self.motif_positions = np.asarray([m['relative_position'] for m in self.motifs])
        self.motif_rotations = np.asarray([pose_arrays(self.motif_poses[m['id']])[1] for m in self.motifs])
        self.fixed_scaffold_matches = self.classify_pair(*self.fixed_poses)

    def _contacts(self, d, r, cutoff):
        points = self.atoms@r.T+d
        near = self.tree.sparse_distance_matrix(cKDTree(points), cutoff+2*self.radii.max(), output_type='ndarray')
        if not len(near):return dict(minimum_gap_A=None, residue_pairs=[])
        gaps = near['v']-self.radii[near['i']]-self.radii[near['j']]
        require(float(gaps.min()) >= -CRITERIA['hard_overlap_tolerance_A'], 'Native observer requires hard-valid poses')
        keep = gaps <= cutoff;near, gaps = near[keep], gaps[keep]
        pairs = self.residues[near['i']]*self.residue_count+self.residues[near['j']]
        return dict(minimum_gap_A=float(gaps.min()) if len(gaps) else None,
                    residue_pairs=sorted(set(map(int,pairs))))

    def classify_pair(self, anchor, moving):
        """Return all entry motif matches, directed anchor→moving, in open space."""
        ta, ra = pose_arrays(anchor);tm, rm = pose_arrays(moving)
        d, r = ra.T@(tm-ta), ra.T@rm
        angles = np.degrees(np.arccos(np.clip((np.einsum('ij,kij->k',r,self.motif_rotations)-1)/2,-1.,1.)))
        observed = self.member_positions@r.T+d
        matches, cache = [], {}
        for k in np.flatnonzero(angles <= CRITERIA['body_orientation_entry_deg']):
            motif = self.motifs[k]
            expected = self.member_positions@self.motif_rotations[k].T+self.motif_positions[k]
            error = float(np.max(np.linalg.norm(observed-expected,axis=1)))
            if error > CRITERIA['body_member_position_entry_A']:continue
            bonds = []
            for c in motif['member_contacts']:
                i,j,label = c['member_i'],c['member_j'],c['directed_class']
                key = (i,j,label)
                if key not in cache:
                    ref = self.references[label]
                    mr = self.member_rotations[i].T@r@self.member_rotations[j]
                    md = self.member_rotations[i].T@(r@self.member_positions[j]+d-self.member_positions[i])
                    position_error = float(np.linalg.norm(md-ref['position']))
                    angle = angle_degrees(mr,ref['rotation'])
                    bond = None
                    if position_error <= CRITERIA['monomer_position_entry_A'] and angle <= CRITERIA['monomer_orientation_entry_deg']:
                        contact = self._contacts(md,mr,CRITERIA['contact_entry_A'])
                        common = sorted(ref['native_residue_pairs'].intersection(contact['residue_pairs']))
                        if len(common) >= CRITERIA['minimum_shared_native_residue_pairs']:
                            bond = dict(members=[i,j],class_label=label,class_family=ref['family'],
                                position_error_A=position_error,orientation_error_deg=angle,
                                minimum_gap_A=contact['minimum_gap_A'],shared_reference_residue_pairs_entry=common)
                    cache[key] = bond
                if cache[key] is not None:bonds.append(copy.deepcopy(cache[key]))
            if bonds:matches.append(dict(motif_id=motif['id'],maximum_member_position_error_A=error,
                proper_orientation_error_deg=float(angles[k]),supporting_member_bonds=bonds))
        return matches

    def _cycles(self, matches):
        output = []
        a = [m for m in matches if m['anchor_index'] == 0]
        b = [m for m in matches if m['anchor_index'] == 1]
        for ma,mb,ab in itertools.product(a,b,self.fixed_scaffold_matches):
            # A→M followed by M→B must equal a registered ideal A→B motif.
            implied = compose(self.motif_poses[ma['motif_id']],inverse(self.motif_poses[mb['motif_id']]))
            t,r = pose_arrays(implied);target,s = pose_arrays(self.motif_poses[ab['motif_id']])
            error = float(np.max(np.linalg.norm(self.member_positions@r.T+t-(self.member_positions@s.T+target),axis=1)))
            angle = float(np.degrees(Rotation.from_matrix(r.T@s).magnitude()))
            if error <= CRITERIA['catalogue_cycle_position_tolerance_A'] and angle <= CRITERIA['catalogue_cycle_angle_tolerance_deg']:
                output.append(dict(anchor0_to_moving_motif=ma['motif_id'],anchor1_to_moving_motif=mb['motif_id'],
                    anchor0_to_anchor1_motif=ab['motif_id'],maximum_catalogue_member_closure_error_A=error,
                    catalogue_rotation_closure_error_deg=angle))
        return output

    def classify(self, moving_pose):
        matches=[]
        for i,anchor in enumerate(self.fixed_poses):
            matches.extend(dict(record,anchor_index=i,anchor_body_id=self.fixed_body_ids[i],moving_body_id=0)
                           for record in self.classify_pair(anchor,moving_pose))
        anchors=sorted({m['anchor_index'] for m in matches});cycles=self._cycles(matches)
        return dict(native_any=bool(matches),native_anchor_count=len(anchors),matched_anchor_indices=anchors,
            matches=matches,cooperative_entry=len(anchors)==2,registry_consistent_triangle=bool(cycles),
            registry_consistent_triangles=cycles)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['freeze']);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();freeze_definition(args.out)
    print(json.dumps(dict(definition=str(args.out.resolve()/'definition.json'),sha256=sha(args.out/'definition.json'),physical_launches=0)))


if __name__ == '__main__':main()
