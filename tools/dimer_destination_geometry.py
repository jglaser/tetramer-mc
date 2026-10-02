"""Independent all-atom saved-endpoint fingerprints using sparse KD distances.

The KD tree only supplies conservative candidates. Strict atomic predicates are
evaluated on direct Euclidean distances, with the two old selected bodies
removed and their new mutual pair included exactly once. No native labels.
"""
from __future__ import annotations

import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


class DimerGeometry:
    def __init__(self, shape, state, rd, wall_radius):
        self.centers = np.asarray([a['center'] for a in shape['atoms']], float)
        self.radii = np.asarray([a['radius'] for a in shape['atoms']], float)
        if not (self.centers.shape == (len(self.radii), 3) and len(self.radii)
                and np.isfinite(self.centers).all() and np.isfinite(self.radii).all()
                and (self.radii > 0).all() and np.isfinite(rd) and rd >= 0
                and np.isfinite(wall_radius) and wall_radius > max(self.radii)):
            raise ValueError('Invalid atomic shape, radius, or wall')
        self.rd, self.wall_radius = float(rd), float(wall_radius)
        self.state = state
        self.fixed = np.concatenate([self.placed(p) for p in state])
        self.tree = cKDTree(self.fixed)
        self.count = len(self.radii)
        self.cutoff = 2*float(max(self.radii))+2*self.rd+1e-8

    def placed(self, pose):
        q = np.asarray(pose['orientation'], float)
        p = np.asarray(pose['position'], float)
        if q.shape != (4,) or p.shape != (3,) or not np.isfinite(q).all() or not np.isfinite(p).all() or abs(q@q-1) > 1e-8:
            raise ValueError('Invalid pose')
        r = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
        return self.centers@r.T+p

    def fingerprint(self, members, selected):
        if len(members) != 2 or len(selected) != 2 or len(set(members)) != 2 or any(type(i) is not int or not 0 <= i < len(self.state) for i in members):
            raise ValueError('Exactly two distinct valid labels required')
        points = [self.placed(p) for p in selected]
        trees = [cKDTree(p) for p in points]
        hard, contacts = set(), set()
        margins, candidates = [], 0

        def classify(pair, distance, radius):
            nonlocal candidates
            candidates += len(distance)
            if len(distance):
                margins.extend([float(np.min(abs(distance-radius))), float(np.min(abs(distance-radius-2*self.rd)))])
            for label in np.unique(pair[distance < radius]):
                hard.add(tuple(sorted((current, int(label)))))
            for label in np.unique(pair[distance < radius+2*self.rd]):
                contacts.add(tuple(sorted((current, int(label)))))

        for current, p, tree in zip(members, points, trees):
            pairs = tree.sparse_distance_matrix(self.tree, self.cutoff, output_type='ndarray')
            labels = pairs['j']//self.count
            keep = (labels != members[0]) & (labels != members[1])
            ii, jj, labels = pairs['i'][keep], pairs['j'][keep], labels[keep]
            distance = np.linalg.norm(p[ii]-self.fixed[jj], axis=1)
            classify(labels, distance, self.radii[ii]+self.radii[jj % self.count])
        current = members[0]
        pairs = trees[0].sparse_distance_matrix(trees[1], self.cutoff, output_type='ndarray')
        ii, jj = pairs['i'], pairs['j']
        classify(np.full(len(ii), members[1]), np.linalg.norm(points[0][ii]-points[1][jj], axis=1), self.radii[ii]+self.radii[jj])
        wall_margin = min(float(np.min(self.wall_radius-self.radii-np.linalg.norm(p, axis=1))) for p in points)
        wall_valid = all(bool(np.all(np.sum(p*p, axis=1) <= (self.wall_radius-self.radii)**2)) for p in points)
        boundary_margin = min(margins) if margins else None
        return dict(hard_overlap_edges=[list(e) for e in sorted(hard)], contacts=[list(e) for e in sorted(contacts)],
                    wall_valid=wall_valid, hard_valid=not hard and wall_valid,
                    candidate_atom_pairs=candidates, minimum_examined_threshold_margin_A=boundary_margin,
                    minimum_wall_margin_A=wall_margin,
                    threshold_ambiguity=bool((boundary_margin is not None and boundary_margin <= 1e-8) or abs(wall_margin) <= 1e-8))
