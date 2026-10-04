"""Conservative native-pair broad phase; no atoms, patches, or native decisions.

If every corresponding body member differs from an ideal directed motif by at
most e, its mean difference also has Euclidean norm at most e. With member mean
b, the actual moving centroid p_j+R_j*b must therefore lie within e of one of
the anchor targets p_i+R_i*(d_m+R_m*b). This necessary condition is not sufficient
for native registry. Call the unchanged classifier for every returned pair.

The tree uses a closed, inflated infinity-norm ball, a superset of the required
Euclidean ball. Centering and construction use longdouble arithmetic; the tree
uses FP64. A scale-aware outward margin covers rounding in both coordinate
paths, the frozen classifier's relative-pose calculation, and tree arithmetic.
This is an engineering numerical guard, not formal verification of SciPy. Unsafe
arithmetic disables pruning. There is no orientation pruning or periodic logic.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _pose_arrays(pose):
    """Match frozen native_contact_regions.pose_arrays, including normalization."""
    try:
        position = np.asarray(pose['position'], dtype=float)
        quaternion = np.asarray(pose['orientation'], dtype=float)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError('Invalid finite pose') from error
    _require(position.shape == (3,) and quaternion.shape == (4,)
             and np.isfinite(position).all() and np.isfinite(quaternion).all(),
             'Invalid finite pose')
    _require(abs(np.linalg.norm(quaternion)-1) < 1e-8,
             'Pose quaternion must be normalized')
    return position, Rotation.from_quat(quaternion[[1, 2, 3, 0]]).as_matrix()


class NativePairCandidates:
    """A fixed directed motif catalogue; output indices preserve i<j semantics.

    Empty motif catalogues are valid and yield no candidates. Duplicate motif
    geometries are valid: pairs are deduplicated, while the downstream classifier
    remains responsible for retaining all matching motif labels. Construction
    copies inputs. Every call validates all poses, even for an empty catalogue.
    """

    def __init__(self, member_positions, motif_poses, body_member_tolerance, *, boundary='open'):
        _require(boundary in ('open', 'spherical'),
                 'Only open/spherical native candidates are supported; periodic boundaries are unsupported')
        members = np.asarray(member_positions, dtype=float)
        _require(members.ndim == 2 and members.shape[1] == 3 and len(members) > 0
                 and np.isfinite(members).all(), 'Nonempty finite member positions required')
        tolerance = float(body_member_tolerance)
        _require(math.isfinite(tolerance) and tolerance >= 0, 'Invalid member tolerance')
        motifs = [_pose_arrays(pose) for pose in motif_poses]
        self.tolerance = tolerance
        self._members = members.copy()
        self._motif_positions = np.asarray([p for p, _ in motifs], float).reshape((-1, 3))
        self._motif_rotations = np.asarray([r for _, r in motifs], float).reshape((-1, 3, 3))
        with np.errstate(over='ignore', invalid='ignore', under='ignore'):
            # Divide before summation to avoid an unnecessary large-member sum.
            self._centroid = (members.astype(np.longdouble)/len(members)).sum(axis=0)
            self._motif_centroids = (self._motif_positions.astype(np.longdouble)
                + np.einsum('mij,j->mi', self._motif_rotations.astype(np.longdouble), self._centroid))

    def candidate_pairs(self, poses, *, mobile_labels=None):
        """Sorted unique pairs whose unchanged directed classifier may match.

        ``mobile_labels`` optionally retains only pairs involving at least one
        listed body index; it must not be used to observe fixed-fixed bonds.
        Unsafe intermediate arithmetic or a numerical tree failure returns all
        such pairs. Caller input errors are never hidden by that fallback.
        """
        decoded = [_pose_arrays(pose) for pose in poses]
        count = len(decoded)
        if mobile_labels is None:
            mobile = None
        else:
            labels = list(mobile_labels)
            _require(all(type(i) is int and 0 <= i < count for i in labels)
                     and len(set(labels)) == len(labels), 'Invalid mobile labels')
            mobile = set(labels)

        def selected(i, j):
            return mobile is None or i in mobile or j in mobile

        def unpruned():
            return [(i, j) for i, j in itertools.combinations(range(count), 2) if selected(i, j)]

        if count < 2 or not len(self._motif_positions) or mobile == set():
            return []
        positions = np.asarray([p for p, _ in decoded], dtype=np.longdouble)
        rotations = np.asarray([r for _, r in decoded], dtype=np.longdouble)
        with np.errstate(over='ignore', invalid='ignore', under='ignore'):
            centered = positions-positions[0]
            actual = np.asarray(centered+np.einsum('nij,j->ni', rotations, self._centroid), float)
            # Absolute world scale also covers cancellation in the classifier's
            # original p_j-p_i relative-pose path, even though our tree is centered.
            scale = (1+np.abs(positions).max()+np.abs(self._members).max()
                     +np.abs(self._motif_positions).max()+np.abs(self._motif_centroids).max()
                     +np.longdouble(self.tolerance))
            allowance = 512*np.finfo(float).eps*scale
            radius = np.nextafter(float(np.longdouble(self.tolerance)+allowance), math.inf)
        safe = np.finfo(float).max/16
        if (not np.isfinite(actual).all() or not np.isfinite(self._motif_centroids).all()
                or not math.isfinite(radius) or radius > safe or np.abs(actual).max() > safe):
            return unpruned()
        found = set()
        try:
            tree = cKDTree(actual, copy_data=True)
            for i in range(count-1):
                with np.errstate(over='ignore', invalid='ignore', under='ignore'):
                    targets = np.asarray(centered[i]+self._motif_centroids@rotations[i].T, float)
                if not np.isfinite(targets).all() or np.abs(targets).max() > safe:
                    return unpruned()
                near = tree.query_ball_point(targets, radius, p=np.inf, eps=0,
                                              workers=1, return_sorted=True)
                for neighbors in near:
                    found.update((i, int(j)) for j in neighbors if j > i and selected(i, int(j)))
        except (ValueError, OverflowError, FloatingPointError):
            return unpruned()
        return sorted(found)
