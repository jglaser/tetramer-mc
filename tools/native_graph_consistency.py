"""Check whether local native motifs admit one consistent ideal pose assignment.

This is a catalogue-label check, not a new physical constraint or a repair of
observed poses. Trees always admit an assignment. Cycles can be frustrated.
Multiple labels on one pair are alternatives; one compatible label suffices.
"""
from __future__ import annotations
import math
import numpy as np
from scipy.spatial.transform import Rotation


class NativeGraphConsistency:
    def __init__(self, motifs, body_count, position_tolerance=1e-6, angle_tolerance=1e-6):
        assert type(body_count) is int and body_count > 0
        self.n = body_count
        self.position_tolerance, self.angle_tolerance = position_tolerance, angle_tolerance
        assert position_tolerance > 0 and angle_tolerance > 0
        self.transforms = {}
        for motif in motifs:
            t = np.asarray(motif['relative_position'], float); q = np.asarray(motif['relative_orientation'], float)
            assert t.shape == (3,) and q.shape == (4,) and np.isfinite(t).all() and np.isfinite(q).all()
            assert abs(np.linalg.norm(q)-1.) < 1e-8 and motif['id'] not in self.transforms
            self.transforms[motif['id']] = (t, Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix())
        self.cache = {}

    def check(self, registered_keys):
        key = tuple(sorted(set(tuple(v) for v in registered_keys)))
        if key in self.cache: return self.cache[key]
        pairs = {}
        for i, j, label in key:
            assert type(i) is type(j) is type(label) is int and 0 <= i < j < self.n
            assert label in self.transforms
            pairs.setdefault((i, j), []).append(label)
        components = []; unseen = set(range(self.n))
        while unseen:
            reached = {min(unseen)}
            while True:
                added = reached | {v for pair in pairs if set(pair) & reached for v in pair}
                if added == reached: break
                reached = added
            components.append(sorted(reached)); unseen -= reached
        chosen = {}; labels = {}

        def matching(i, j):
            pi, ri = chosen[i]; pj, rj = chosen[j]
            t = ri.T@(pj-pi); rotation = ri.T@rj
            found = []
            for label in pairs[i, j]:
                expected_t, expected_r = self.transforms[label]
                position_error = float(np.linalg.norm(t-expected_t))
                angle_error = 2*math.asin(min(1., float(np.linalg.norm(rotation-expected_r))/(2*math.sqrt(2))))
                if position_error <= self.position_tolerance and angle_error <= self.angle_tolerance:
                    found.append((label, position_error, angle_error))
            return found

        def assign(component):
            for i, j in pairs:
                if i in chosen and j in chosen and not matching(i, j): return False
            if all(i in chosen for i in component): return True
            i, j = next(pair for pair in pairs if (pair[0] in chosen) != (pair[1] in chosen)
                        and pair[0] in component)
            for label in pairs[i, j]:
                t, rotation = self.transforms[label]
                if i in chosen:
                    p, r = chosen[i]; added = j
                    chosen[j] = (p+r@t, r@rotation)
                else:
                    p, r = chosen[j]; added = i
                    ri = r@rotation.T; chosen[i] = (p-ri@t, ri)
                if assign(component): return True
                del chosen[added]
            return False

        consistent = True
        for component in components:
            chosen[component[0]] = (np.zeros(3), np.eye(3))
            if not assign(component): consistent = False; break
        if consistent:
            for i, j in pairs:
                label, position_error, angle_error = matching(i, j)[0]
                labels[f'{i}-{j}'] = dict(motif_id=label, position_error_A=position_error, angle_error_radians=angle_error)
        result = dict(consistent=consistent, connected=len(components) == 1,
            consistent_connected=consistent and len(components) == 1, components=components,
            independent_cycles=len(pairs)-self.n+len(components),
            chosen_edge_labels=labels, position_tolerance_A=self.position_tolerance,
            angle_tolerance_radians=self.angle_tolerance,
            scope='Existence of an ideal catalogue pose assignment; measured poses are neither repaired nor constrained. This does not measure free energy or require a clique.')
        self.cache[key] = result
        return result
