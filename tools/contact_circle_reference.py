#!/usr/bin/env python3
"""Independent deterministic hard-free azimuth geometry and probability mass.

For t(phi)=center+radius*(e1*cos(phi)+e2*sin(phi)), orientation is fixed.
Atom KD trees provide conservative candidates; long-double leaf geometry
produces analytic strict-overlap arcs. No poses or Poisson clouds are drawn.
"""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

TAU=2*math.pi


def require(ok,message):
    if not ok:raise ValueError(message)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def interval(lower,upper,lower_closed=False,upper_closed=False):
    return dict(lower=float(lower),upper=float(upper),lower_closed=bool(lower_closed),upper_closed=bool(upper_closed))
def nonempty(i):return i['lower']<i['upper'] or (i['lower']==i['upper'] and i['lower_closed'] and i['upper_closed'])
def contains(intervals,phi):
    phi=float(phi)%TAU
    return any((phi>i['lower'] or(phi==i['lower'] and i['lower_closed'])) and
        (phi<i['upper'] or(phi==i['upper'] and i['upper_closed'])) for i in intervals)


def union(intervals):
    result=[]
    for raw in sorted(intervals,key=lambda i:(i['lower'],not i['lower_closed'],i['upper'])):
        i=raw.copy();require(0<=i['lower']<=i['upper']<=TAU,'Arc outside canonical domain')
        require(i['upper']<TAU or not i['upper_closed'],'Upper 2pi must remain open')
        if not nonempty(i):continue
        if result and (i['lower']<result[-1]['upper'] or
                (i['lower']==result[-1]['upper'] and(i['lower_closed'] or result[-1]['upper_closed']))):
            previous=result[-1]
            if i['lower']==previous['lower']:previous['lower_closed']|=i['lower_closed']
            if i['upper']>previous['upper']:previous['upper']=i['upper'];previous['upper_closed']=i['upper_closed']
            elif i['upper']==previous['upper']:previous['upper_closed']|=i['upper_closed']
        else:result.append(i)
    return result


def complement(forbidden):
    forbidden=union(forbidden);result=[];lower=0.;closed=True
    for i in forbidden:
        piece=interval(lower,i['lower'],closed,not i['lower_closed'])
        if nonempty(piece):result.append(piece)
        lower=i['upper'];closed=not i['upper_closed']
    if lower<TAU:result.append(interval(lower,TAU,closed,False))
    return result


def open_arc(mode,half_width):
    """Strict interior of a circular arc; preserve any tangent singleton."""
    require(math.isfinite(mode) and 0<=half_width<=math.pi,'Invalid angular arc')
    if half_width==0:return []
    if half_width==math.pi:
        missing=(mode+math.pi)%TAU
        return [i for i in [interval(0,missing,True,False),interval(missing,TAU,False,False)] if nonempty(i)]
    start=(mode-half_width)%TAU;end=start+2*half_width
    if end<=TAU:return [interval(start,end,False,False)]
    return [interval(0,end-TAU,True,False),interval(start,TAU,False,False)]


def rotation(q):
    q=np.asarray(q,float);require(q.shape==(4,) and abs(np.linalg.norm(q)-1)<1e-8,'Invalid orientation')
    return Rotation.from_quat(q[[1,2,3,0]]).as_matrix()


def pair_forbidden_arcs(baseline,fixed,radius_sum,circle_radius,basis):
    """One moving/fixed atomic pair, with strict core overlap.

    Wider intermediate arithmetic independently checks the ordinary-double
    implementation near tangencies. No threshold clamp or retry is used.
    """
    require(radius_sum>0 and circle_radius>=0 and math.isfinite(radius_sum+circle_radius),'Invalid radius')
    delta=np.asarray(fixed,dtype=np.longdouble)-np.asarray(baseline,dtype=np.longdouble)
    e1,e2=np.asarray(basis,dtype=np.longdouble);axis=np.cross(e1,e2)
    a=delta@e1;b=delta@e2;axial=delta@axis;q=np.hypot(a,b)
    rho=np.longdouble(circle_radius);s=np.longdouble(radius_sum)
    if rho==0:
        return [interval(0,TAU,True,False)] if np.sqrt(delta@delta)<s else []
    if q==0:
        return [interval(0,TAU,True,False)] if np.hypot(axial,rho)<s else []
    minimum=np.hypot(axial,q-rho);maximum=np.hypot(axial,q+rho)
    if s<=minimum:return []
    if s>maximum:return [interval(0,TAU,True,False)]
    mode=float(np.arctan2(b,a))
    if s==maximum:return open_arc(mode,math.pi)
    # A 2D circle/disc triangle after removing the axial separation.
    disc=np.sqrt((s-abs(axial))*(s+abs(axial)))
    aa,bb,cc=sorted([rho,q,disc],reverse=True)
    factors=[aa+(bb+cc),cc-(aa-bb),cc+(aa-bb),aa+(bb-cc)]
    require(min(factors)>=0,'Inconsistent leaf triangle; do not clamp')
    sine_numerator=np.sqrt(factors[0])*np.sqrt(factors[1])*np.sqrt(factors[2])*np.sqrt(factors[3])
    cosine_numerator=rho*rho+q*q-disc*disc
    half=float(np.arctan2(sine_numerator,cosine_numerator))
    require(0<half<=math.pi,'Invalid strict overlap angle')
    return open_arc(mode,half)


def wrapped_cauchy_interval_mass(lower,upper,mode,gamma):
    """Analytic unwrapped phase difference; stable even for short intervals.

    The argument change of (t*cos(delta/2),sin(delta/2)), t=tanh(gamma/2),
    divided by pi equals the integrated wrapped Cauchy. A determinant/dot
    atan2 computes that difference without subtracting nearly equal CDFs.
    """
    require(math.isfinite(mode) and math.isfinite(gamma) and gamma>0,'Invalid Cauchy parameters')
    require(0<=lower<=upper<=TAU,'Invalid canonical mass interval')
    width=upper-lower
    if width==0:return 0.
    if width==TAU:return 1.
    t=np.tanh(np.longdouble(gamma)/2);a=(np.longdouble(lower)-mode)/2;b=(np.longdouble(upper)-mode)/2
    cross=t*np.sin(np.longdouble(width)/2)
    dot=t*t*np.cos(a)*np.cos(b)+np.sin(a)*np.sin(b)
    mass=float(np.arctan2(cross,dot)/np.longdouble(math.pi))
    require(0<=mass<=1,'Invalid analytic angular mass')
    return mass


def azimuth_mass(intervals,law):
    intervals=union(intervals);b=law['localized_probability'];require(0<=b<1,'Invalid azimuth mixture')
    terms=[]
    for i in intervals:
        uniform=(i['upper']-i['lower'])/TAU
        wrapped=wrapped_cauchy_interval_mass(i['lower'],i['upper'],law['mode'],law['gamma']) if b else uniform
        terms.append((1-b)*uniform+b*wrapped)
    result=math.fsum(terms);require(0<=result<=1+2e-14,'Angular union mass exceeds one')
    return result


def hard_free_arcs(shape,fixed_poses,moving_orientation,circle_center,circle_radius,basis,*,use_tree=True):
    started=time.process_time();atoms=np.asarray([a['center'] for a in shape['atoms']]);radii=np.asarray([a['radius'] for a in shape['atoms']])
    require(len(atoms)>0 and np.isfinite(atoms).all() and np.isfinite(radii).all() and np.all(radii>0),'Invalid atom shape')
    center=np.asarray(circle_center,float);basis=np.asarray(basis,float)
    require(center.shape==(3,) and np.isfinite(center).all() and basis.shape==(2,3) and np.isfinite(basis).all(),'Invalid circle frame')
    require(np.allclose(basis@basis.T,np.eye(2),atol=1e-12,rtol=1e-12),'Nonorthonormal transverse basis')
    require(math.isfinite(circle_radius) and circle_radius>=0,'Invalid circle radius')
    baseline=atoms@rotation(moving_orientation).T+center;forbidden=[];counts=Counter();saturated=False
    def complete_circle(arcs):
        return len(arcs)==1 and arcs[0]['lower']==0 and arcs[0]['upper']==TAU and arcs[0]['lower_closed']
    for pose in fixed_poses:
        fixed=atoms@rotation(pose['orientation']).T+pose['position']
        if use_tree:
            # Conservative circumsphere around each atom's circular path. The
            # padding is candidate-only; exact leaf inequalities set all arcs.
            coordinate_scale=max(1.,float(abs(baseline).max()),float(abs(fixed).max()))
            padding=1e-9+64*np.finfo(float).eps*coordinate_scale
            candidates=cKDTree(fixed).query_ball_point(baseline,circle_radius+radii+radii.max()+padding)
        else:candidates=[list(range(len(atoms))) for _ in atoms]
        counts['candidate_pairs']+=sum(len(indices) for indices in candidates)
        pending=[]
        for i,indices in enumerate(candidates):
            for j in indices:
                arcs=pair_forbidden_arcs(baseline[i],fixed[j],float(radii[i]+radii[j]),circle_radius,basis)
                counts['leaf_pairs']+=1
                if arcs:counts['overlapping_pairs']+=1;pending.extend(arcs)
                if complete_circle(arcs) or len(pending)>=32:
                    forbidden=union(forbidden+pending);pending=[]
                    if complete_circle(forbidden):saturated=True;break
            if saturated:break
        forbidden=union(forbidden+pending)
        if saturated:break
    forbidden=union(forbidden);allowed=complement(forbidden)
    length=math.fsum(i['upper']-i['lower'] for i in allowed)
    return dict(forbidden=forbidden,allowed=allowed,allowed_length=length,uniform_allowed_probability=length/TAU,
        has_positive_allowed_length=length>0,has_any_allowed_point=bool(allowed),counts=dict(counts),
        full_coverage_early_exit=saturated,analysis_cpu_seconds=time.process_time()-started)


def direct_hard_valid(shape,fixed_poses,moving_orientation,circle_center,circle_radius,basis,phi):
    """Deliberately brute-force endpoint witness, independent of arc algebra."""
    require(len(shape['atoms'])<=256,'Brute endpoint witness is restricted to tiny unit-test shapes')
    atoms=np.asarray([a['center'] for a in shape['atoms']]);radii=np.asarray([a['radius'] for a in shape['atoms']])
    basis=np.asarray(basis);position=np.asarray(circle_center)+circle_radius*(basis[0]*math.cos(phi)+basis[1]*math.sin(phi))
    moving=atoms@rotation(moving_orientation).T+position
    for pose in fixed_poses:
        fixed=atoms@rotation(pose['orientation']).T+pose['position']
        distances=np.linalg.norm(moving[:,None,:]-fixed[None,:,:],axis=2)
        if np.any(distances<radii[:,None]+radii[None,:]):return False
    return True


def circle_from_case(shape,config,case):
    """Reconstruct the basis from frozen atom labels, never from Rust output.

    Circle center and radius are retained verbatim from the saved draw, after
    an independent two-radius reconstruction checks their consistency.
    """
    atoms=np.asarray([a['center'] for a in shape['atoms']]);R=rotation(case['pose']['orientation']);centers=[]
    for label in case['contact_pairs']:
        fixed=config['fixed_poses'][label['neighbor_index']]
        point=rotation(fixed['orientation'])@atoms[label['fixed_atom']]+fixed['position']
        centers.append(point-R@atoms[label['moving_atom']])
    require(len(centers)==2,'Exactly two frozen atom-pair labels required')
    centers=np.asarray(centers);delta=centers[1]-centers[0];D=np.linalg.norm(delta)
    require(math.isfinite(D) and D>0,'Coincident frozen contact centers')
    axis=delta/D;reference=np.eye(3)[np.argmin(abs(axis))];e1=np.cross(axis,reference);e1/=np.linalg.norm(e1);e2=np.cross(axis,e1)
    r1,r2=np.asarray(case['radii'],dtype=np.longdouble);distance=np.longdouble(D)
    height=(distance+(r1-r2)*(r1+r2)/distance)/2
    aa,bb,cc=sorted([r1,r2,distance],reverse=True);factors=[aa+(bb+cc),cc-(aa-bb),cc+(aa-bb),aa+(bb-cc)]
    require(min(factors)>=0,'Saved radii violate triangle inequalities')
    rho=float(np.sqrt(np.prod(np.asarray(factors)))/(2*distance));center=centers[0]+float(height)*axis
    center_error=float(np.max(abs(center-case['circle_center'])));radius_error=abs(rho-case['circle_radius'])
    require(center_error<2e-8 and radius_error<2e-8,'Saved circle disagrees with independent atom-label reconstruction')
    point=np.asarray(case['circle_center'])+case['circle_radius']*(e1*math.cos(case['phi'])+e2*math.sin(case['phi']))
    point_error=float(np.max(abs(point-case['pose']['position'])));require(point_error<2e-8,'Saved phi/frame does not reconstruct original pose')
    return dict(moving_orientation=case['pose']['orientation'],circle_center=case['circle_center'],circle_radius=case['circle_radius'],basis=[e1.tolist(),e2.tolist()],
        reconstruction_errors=dict(circle_center_absolute=center_error,circle_radius_absolute=radius_error,original_position_absolute=point_error))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--shape',type=Path,required=True);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--cases',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    require(not a.out.exists(),'Output must be new');shape=read(a.shape);config=read(a.config)
    contents=a.cases.read_text()
    try:bundle=json.loads(contents)
    except json.JSONDecodeError:bundle=None
    cases=bundle['cases'] if isinstance(bundle,dict) else [json.loads(line) for line in contents.splitlines()]
    require(len({str(c['id']) for c in cases})==len(cases),'Repeated case identity')
    paths=[a.shape,a.config,a.cases,Path(__file__)];bindings={str(p.resolve()):sha(p) for p in paths};rows=[]
    if isinstance(bundle,dict):
        require(bundle['schema']=='contact-circle-feasibility-cases-v1' and bundle['new_pose_draws']==bundle['new_Poisson_clouds']==0,'Wrong saved-circle scope')
        require(sha(a.shape)==bundle['shape_sha256'] and sha(a.config)==bundle['config_sha256'],'Frozen shape/config changed')
        for path,digest in bundle['sources'].items():require(sha(path)==digest,'Frozen source changed');bindings[str(Path(path).resolve())]=digest
    for case in cases:
        frame=circle_from_case(shape,config,case) if 'contact_pairs' in case else case
        result=hard_free_arcs(shape,config['fixed_poses'],frame['moving_orientation'],frame['circle_center'],frame['circle_radius'],frame['basis'])
        result.update(id=case['id'],azimuth_allowed_probability=azimuth_mass(result['allowed'],case['azimuth_law']))
        if 'contact_pairs' in case:
            require(contains(result['allowed'],case['phi'])==case['hard_valid'],'Arc classification contradicts saved original hard predicate')
            result.update(arm=case['arm'],population=case['population'],original_phi=case['phi'],original_hard_valid=case['hard_valid'],basis=frame['basis'],reconstruction_errors=frame['reconstruction_errors'])
        rows.append(result)
    for path,h in bindings.items():require(sha(path)==h,'Input changed during diagnosis')
    output=dict(complete=True,schema='independent-contact-circle-diagnostic-v1',scope='Saved-circle feasibility only; no new poses, Poisson clouds or physical weights.',
        rows=rows,input_sha256=bindings,cases=len(cases))
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(output,indent=2,allow_nan=False)+'\n');print(a.out)


if __name__=='__main__':main()
