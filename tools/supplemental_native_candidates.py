"""Necessary center-distance bound for an open supplemental native catalogue.

If every corresponding member has error at most e, averaging those errors gives
|d + (R-S)b - t| <= e, b=mean(member positions). Proper rotations imply
|d| <= max_m|t_m| + 2|b| + e. This is an exact real-arithmetic necessary condition,
independent of the actual orientation. No atom, residue-patch or label query is
performed here. The enclosing observer still evaluates its full frozen predicate.

Floating-point implementation uses centered coordinates, a scale-aware outward
allowance and eps=0 tree queries. This is an engineering guard, not a formal proof
of SciPy arithmetic. Unsafe arithmetic or a numerical tree failure returns EVERY
pair with an explicit fallback reason, never an empty/no-label result. Invalid
inputs remain fatal. Periodic images are intentionally outside this helper.
"""
from __future__ import annotations
import itertools
import math

import numpy as np
from scipy.spatial import cKDTree


def require(value,message):
    if not value:raise ValueError(message)


class SupplementalCenterCandidates:
    def __init__(self,member_positions,motif_positions,body_member_tolerance,*,boundary='open'):
        require(boundary in ('open','spherical'),'Supplemental center pruning requires open/spherical geometry')
        members=np.asarray(member_positions,float);motifs=np.asarray(motif_positions,float)
        if motifs.size==0:motifs=motifs.reshape((0,3))
        require(members.ndim==2 and members.shape[1]==3 and len(members)>0
            and np.isfinite(members).all(),'Nonempty finite member centers required')
        require(motifs.ndim==2 and motifs.shape[1]==3 and np.isfinite(motifs).all(),'Finite motif translations required')
        require(type(body_member_tolerance) is not bool and math.isfinite(float(body_member_tolerance))
            and float(body_member_tolerance)>=0,'Invalid body member tolerance')
        self.members=members.copy();self.motifs=motifs.copy();self.tolerance=float(body_member_tolerance)
        self.boundary=boundary
        with np.errstate(over='ignore',invalid='ignore',under='ignore'):
            mean=(members.astype(np.longdouble)/len(members)).sum(axis=0)
            norm=lambda a:np.sqrt(np.sum(a*a,axis=-1))
            longest=np.max(norm(motifs.astype(np.longdouble))) if len(motifs) else np.longdouble(0)
            self.nominal=longest+2*norm(mean)+np.longdouble(self.tolerance)

    def candidate_pairs(self,positions):
        """Return canonical pairs plus the complete inventory/pruning accounting."""
        p=np.asarray(positions,float)
        if p.size==0:p=p.reshape((0,3))
        require(p.ndim==2 and p.shape[1]==3 and np.isfinite(p).all(),'Finite body positions required')
        n=len(p);total=n*(n-1)//2
        result=dict(body_count=n,total_pairs=total,candidate_pairs=[],candidate_count=0,pruned_pairs=total,
            motif_count=len(self.motifs),nominal_bound_A=float(self.nominal) if np.isfinite(self.nominal)
                and self.nominal<=np.finfo(float).max else None,
            query_radius_A=None,roundoff_allowance_A=None,tree_queries=0,fallback=False,fallback_reason=None,
            boundary=self.boundary,atoms_or_classifier_queries=0)
        def fallback(reason):
            result.update(candidate_pairs=[list(pair) for pair in itertools.combinations(range(n),2)],
                candidate_count=total,pruned_pairs=0,fallback=True,fallback_reason=reason)
            return result
        if total==0 or len(self.motifs)==0:return result
        with np.errstate(over='ignore',invalid='ignore',under='ignore'):
            center=p.astype(np.longdouble)-p[0].astype(np.longdouble)
            centered=np.asarray(center,float)
            scale=(1+np.max(np.abs(p.astype(np.longdouble)))+np.max(np.abs(center))
                +np.max(np.abs(self.members.astype(np.longdouble)))
                +np.max(np.abs(self.motifs.astype(np.longdouble)))+self.nominal)
            allowance=1024*np.finfo(float).eps*scale
            radius=np.nextafter(float(self.nominal+allowance),math.inf)
        # Avoid squared-distance overflow in both construction and query. A very
        # loose but finite allowance may prune nothing; that is still correct.
        safe=math.sqrt(np.finfo(float).max)/16
        if (not np.isfinite(centered).all() or not np.isfinite(allowance)
                or not math.isfinite(radius) or radius>safe or np.max(np.abs(centered))>safe):
            return fallback('unsafe_floating_point_bound_or_coordinates')
        result.update(query_radius_A=radius,roundoff_allowance_A=float(allowance),tree_queries=1)
        try:
            raw=cKDTree(centered,copy_data=True).query_pairs(radius,p=2.,eps=0.,output_type='ndarray')
        except (ValueError,OverflowError,FloatingPointError) as error:
            return fallback('tree_numerical_failure:'+type(error).__name__)
        pairs=sorted((int(i),int(j)) for i,j in raw)
        require(len(set(pairs))==len(pairs) and all(0<=i<j<n for i,j in pairs),'Invalid tree pair inventory')
        result.update(candidate_pairs=[list(pair) for pair in pairs],candidate_count=len(pairs),pruned_pairs=total-len(pairs))
        return result
