#!/usr/bin/env python3
"""Fixed-context, geometry-only Gaussian children of an unchanged blind atlas.

Preparation tool, not an MC kernel. No current moving pose or native classifier
is an input. Production must freeze this model while its outside context is
unchanged and retain the full proposal correction. The small rigid numerical
core derives from the archived incoming solver, SHA
c62ee86d43602f987f694b0889a2f8e66d8309b66ba7c1fc1615872526411eb1;
native entry constraints are deliberately absent. That authority is unchanged.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import sys
import time

import numpy as np
import scipy
from scipy.optimize import minimize
from scipy.spatial import cKDTree

SCHEMA = 'fixed-context-relaxed-atlas-v1'
CHILD_VARIANCES = (0.0625,)*3 + (0.014409934431278726,)*3
REFERENCE_ELL = 55.02283113084892


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def skew(v):
    x,y,z = v
    return np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])


def rotation(q):
    q = np.asarray(q, float)
    require(q.shape == (4,) and np.isfinite(q).all() and abs(float(q@q)-1.) < 1e-10, 'Invalid unit quaternion')
    w,x,y,z = q / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
        [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
        [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


def pose(value):
    require(set(value) in ({'position','rotation'}, {'position','orientation'}), 'Pose requires position and exactly one orientation representation')
    t = np.asarray(value['position'], float)
    r = np.asarray(value['rotation'], float) if 'rotation' in value else rotation(value['orientation'])
    require(t.shape == (3,) and r.shape == (3,3) and np.isfinite(t).all() and np.isfinite(r).all(), 'Invalid pose arrays')
    require(np.max(abs(r.T@r-np.eye(3))) < 1e-10 and abs(np.linalg.det(r)-1) < 1e-10, 'Improper rotation')
    return r,t


def pose_record(r,t):
    return dict(position=np.asarray(t).tolist(), rotation=np.asarray(r).tolist())


def cayley(u):
    q = np.r_[1., np.asarray(u, float)]
    q /= np.linalg.norm(q)
    return rotation(q)


def exponential(phi):
    """SO(3) exponential and left Jacobian, without any pose sampling."""
    theta2=float(phi@phi); k=skew(phi); k2=k@k
    if theta2 < 1e-8:
        a=1-theta2/6+theta2**2/120-theta2**3/5040
        b=.5-theta2/24+theta2**2/720-theta2**3/40320
        c=1/6-theta2/120+theta2**2/5040-theta2**3/362880
    else:
        theta=math.sqrt(theta2); a=math.sin(theta)/theta
        b=(1-math.cos(theta))/theta2; c=(theta-math.sin(theta))/(theta2*theta)
    return np.eye(3)+a*k+b*k2, np.eye(3)+b*k+c*k2


def validate_model(model):
    require(model.get('schema') == 'reciprocal-pose-mixture-v1', 'Require the explicit reciprocal envelope')
    base=model['base_model']; n=len(base['weights'])
    require(base.get('schema') == 'weighted-pose-mixture-v1' and base.get('coordinate_convention') == 'anchor-body-relative', 'Wrong base convention')
    require(n>0 and all(len(base[k])==n for k in ('anchors','means','covariances')) and len(model['reciprocal_components'])==n, 'Incomplete component inventory')
    require(all(type(v) is bool for v in model['reciprocal_components']), 'Nonboolean reciprocal flags')
    weights=np.asarray(base['weights'],float); ell=float(base['angular_length'])
    require(np.isfinite(weights).all() and np.all(weights>0) and abs(math.fsum(weights)-1)<2e-12, 'Weights must be positive and normalized')
    require(math.isfinite(ell) and ell>0, 'Invalid angular length')
    for a,m,c in zip(base['anchors'],base['means'],base['covariances']):
        pose(a); m=np.asarray(m,float); c=np.asarray(c,float)
        require(m.shape==(6,) and c.shape==(6,6) and np.isfinite(m).all() and np.isfinite(c).all(), 'Invalid Gaussian arrays')
        require(np.max(abs(c-c.T)) <= 1e-12*max(1.,float(np.max(abs(c)))), 'Asymmetric covariance')
        np.linalg.cholesky(c)
    return base


def branches(model):
    """Exact virtual parent labels in production order; no component filtering."""
    base=validate_model(model); result=[]
    for k,(weight,inverse) in enumerate(zip(base['weights'],model['reciprocal_components'])):
        for inverted in ((False,True) if inverse else (False,)):
            r,t=pose(base['anchors'][k]); mean=np.asarray(base['means'][k],float)
            r=cayley(mean[3:]/base['angular_length'])@r; t=t+mean[:3]
            if inverted:
                r,t=r.T,-r.T@t
            result.append(dict(component=k,inverted=inverted,weight=weight/(2 if inverse else 1),center=pose_record(r,t)))
    return result


@dataclass(frozen=True)
class Settings:
    child_mass: float=.5
    translation_A: float=1.
    rotation_degrees: float=2.
    desired_gap_A: float=.002
    max_evaluations: int=64
    max_iterations: int=64
    max_pairs: int=32768

    def validate(self):
        require(math.isfinite(self.child_mass) and 0<=self.child_mass<1, 'Child mass must lie in [0,1)')
        require(math.isfinite(self.translation_A) and self.translation_A>0, 'Invalid translation bound')
        require(math.isfinite(self.rotation_degrees) and 0<self.rotation_degrees<180/math.sqrt(3), 'Invalid rotation bound')
        require(math.isfinite(self.desired_gap_A) and self.desired_gap_A>0, 'Invalid clearance objective margin')
        require(type(self.max_evaluations) is int and self.max_evaluations>=1 and type(self.max_iterations) is int and self.max_iterations>=1, 'Invalid evaluation/iteration caps')
        require(type(self.max_pairs) is int and self.max_pairs>=0, 'Invalid pair cap')


class OutsideContext:
    """Immutable atomic arrays and one shared KD-tree in the fixed anchor frame."""
    def __init__(self, atoms, radii, fixed_atoms, fixed_radii, identity):
        self.atoms=np.array(atoms,float,copy=True); self.radii=np.array(radii,float,copy=True)
        self.fixed=np.array(fixed_atoms,float,copy=True); self.fixed_radii=np.array(fixed_radii,float,copy=True)
        require(self.atoms.ndim==self.fixed.ndim==2 and self.atoms.shape[1]==self.fixed.shape[1]==3, 'Invalid atom arrays')
        require(len(self.atoms)>0 and len(self.fixed)>0 and self.radii.shape==(len(self.atoms),) and self.fixed_radii.shape==(len(self.fixed),), 'Invalid atom inventory')
        for a in (self.atoms,self.radii,self.fixed,self.fixed_radii):
            require(np.isfinite(a).all(), 'Nonfinite atoms'); a.flags.writeable=False
        require(np.all(self.radii>0) and np.all(self.fixed_radii>0), 'Nonpositive core radii')
        self.identity=copy.deepcopy(identity); self.tree=cKDTree(self.fixed,copy_data=True)
        self.maximum_fixed_radius=float(self.fixed_radii.max())
        self.maximum_fixed_center_norm=float(np.max(np.linalg.norm(self.fixed,axis=1)))

    @classmethod
    def from_records(cls, shape, context):
        require(set(context)=={'schema','bodies','anchor_label','excluded_moving_labels'} and context['schema']=='fixed-outside-context-v1', 'Context must contain fixed outside bodies only and explicit excluded moving labels')
        bodies=context['bodies']; require(bodies, 'Empty outside context')
        labels=[b['label'] for b in bodies]
        require(all(set(b)=={'label','pose'} for b in bodies) and all(type(i) is int and i>=0 for i in labels) and len(set(labels))==len(labels), 'Invalid fixed body labels')
        excluded=context['excluded_moving_labels']
        require(isinstance(excluded,list) and excluded and all(type(i) is int and i>=0 for i in excluded) and len(set(excluded))==len(excluded), 'Invalid excluded moving labels')
        require(not set(excluded).intersection(labels), 'Moving labels appear in the outside context')
        require(context['anchor_label'] in labels, 'Missing fixed anchor')
        bodies=sorted(bodies,key=lambda b:b['label'])
        ar,at=pose(next(b['pose'] for b in bodies if b['label']==context['anchor_label']))
        atoms=np.asarray([a['center'] for a in shape['atoms']],float)
        radii=np.asarray([a['radius'] for a in shape['atoms']],float)
        fixed=[]
        for b in bodies:
            r,t=pose(b['pose']); rr=ar.T@r; rt=ar.T@(t-at)
            fixed.append(atoms@rr.T+rt)
        canonical=dict(schema=context['schema'],bodies=bodies,anchor_label=context['anchor_label'],excluded_moving_labels=sorted(excluded))
        return cls(atoms,radii,np.concatenate(fixed),np.tile(radii,len(bodies)),
            dict(context_sha256=digest(canonical),shape_sha256=digest(shape),outside_labels=[b['label'] for b in bodies],excluded_moving_labels=sorted(excluded),anchor_label=context['anchor_label']))


class EvaluationCap(RuntimeError):
    pass


class RigidIncomingProblem:
    """Core-only hinge objective; no native gates or current moving pose."""
    def __init__(self, context, initializer, settings, on_event=None):
        settings.validate(); self.context=context; self.settings=settings
        self.on_event=on_event
        self.r0,self.t0=pose(initializer); self.base=context.atoms@self.r0.T
        self.rs=math.radians(settings.rotation_degrees); self.ts=settings.translation_A
        self.margin=settings.desired_gap_A; self.evaluations=[]; self.cache_x=None; self.cache=None
        # Cover the whole coordinate box, including optimizer trials outside
        # the admitted trust balls. Bounds over only the balls would be wrong.
        rotated_norms=np.linalg.norm(self.base,axis=1)
        movement=math.sqrt(3)*self.ts+2*rotated_norms*math.sin(math.sqrt(3)*self.rs/2)
        self.full_box_motion_bounds=movement
        magnitude=1+np.max(rotated_norms)+np.linalg.norm(self.t0)+context.maximum_fixed_center_norm
        self.guard=float(1024*np.finfo(float).eps*magnitude)
        moving=[]; fixed=[]; self.pair_cap_hit=False; self.atom_queries=0
        for i,p in enumerate(self.base+self.t0):
            cutoff=np.nextafter(context.radii[i]+context.maximum_fixed_radius+self.margin+movement[i]+self.guard,np.inf)
            js=sorted(context.tree.query_ball_point(p,cutoff,workers=1)); self.atom_queries+=1
            for j in js:
                distance=np.linalg.norm(context.fixed[j]-p)
                if distance<=np.nextafter(context.radii[i]+context.fixed_radii[j]+self.margin+movement[i]+self.guard,np.inf):
                    if len(moving)==settings.max_pairs:
                        self.pair_cap_hit=True; break
                    moving.append(i); fixed.append(j)
            if self.pair_cap_hit: break
        self.mi=np.asarray(moving,int); self.fi=np.asarray(fixed,int)
        self.radius_sum=context.radii[self.mi]+context.fixed_radii[self.fi]

    def geometry(self,x):
        x=np.asarray(x,float)
        require(x.shape==(6,) and np.isfinite(x).all() and np.max(abs(x))<=1, 'Optimizer escaped full coordinate box')
        r,jl=exponential(self.rs*x[3:]); rotated=self.base@r.T
        return rotated+self.t0+self.ts*x[:3], rotated, r@self.r0, self.t0+self.ts*x[:3], jl

    @staticmethod
    def constraints(x):
        x=np.asarray(x,float); values=np.array([1-x[:3]@x[:3],1-x[3:]@x[3:]])
        jac=np.zeros((2,6)); jac[0,:3]=-2*x[:3]; jac[1,3:]=-2*x[3:]
        return values,jac

    def objective(self,x):
        require(not self.pair_cap_hit, 'Cannot score an incomplete pair list')
        x=np.asarray(x,float)
        if self.cache_x is not None and np.array_equal(x,self.cache_x): return self.cache
        if len(self.evaluations)>=self.settings.max_evaluations: raise EvaluationCap('Declared evaluation cap')
        if self.on_event is not None:
            self.on_event(dict(kind='objective_begun',evaluation=len(self.evaluations),x=x.tolist()))
        points,rotated,_,_,jl=self.geometry(x)
        delta=points[self.mi]-self.context.fixed[self.fi]; d=np.linalg.norm(delta,axis=1)
        gap=d-self.radius_sum; hinge=np.maximum(self.margin-gap,0); active=hinge>0
        unit=np.divide(delta[active],d[active,None],out=np.zeros_like(delta[active]),where=d[active,None]>0)
        force=-2*hinge[active,None]*unit
        grad=np.r_[force.sum(axis=0)*self.ts,np.cross(rotated[self.mi[active]],force).sum(axis=0)@jl*self.rs]
        value=float(hinge@hinge); require(math.isfinite(value) and np.isfinite(grad).all(), 'Nonfinite objective')
        constraint=self.constraints(x)[0]
        self.evaluations.append(dict(index=len(self.evaluations),x=x.tolist(),objective_A2=value,
            minimum_retained_gap_A=float(gap.min()) if len(gap) else None,inside_trust_balls=bool(np.all(constraint>=0))))
        if self.on_event is not None:
            self.on_event(dict(kind='objective_complete',**self.evaluations[-1]))
        self.cache_x=x.copy(); self.cache=(value,grad)
        return self.cache

    def solve(self):
        original=pose_record(self.r0,self.t0)
        result=dict(center=original,fallback=None,optimizer=None,evaluations=self.evaluations,
            retained_pairs=len(self.mi),atom_queries=self.atom_queries,full_box_guard_A=self.guard)
        if self.pair_cap_hit:
            result['fallback']='pair_cap'; return result
        self.objective(np.zeros(6))
        try:
            fit=minimize(self.objective,np.zeros(6),jac=True,method='SLSQP',bounds=[(-1.,1.)]*6,
                constraints=[dict(type='ineq',fun=lambda x:self.constraints(x)[0],jac=lambda x:self.constraints(x)[1])],
                options=dict(maxiter=self.settings.max_iterations,ftol=1e-14,disp=False))
            result['optimizer']=dict(success=bool(fit.success),status=int(fit.status),iterations=int(fit.nit),message=str(fit.message))
        except EvaluationCap:
            result['fallback']='evaluation_cap'; return result
        if not fit.success:
            result['fallback']='optimizer_status'; return result
        eligible=[e for e in self.evaluations if e['inside_trust_balls']]
        chosen=min(eligible,key=lambda e:(e['objective_A2'],e['index']))
        _,_,r,t,_=self.geometry(chosen['x'])
        result.update(center=pose_record(r,t),selected_evaluation=chosen['index'],selected_objective_A2=chosen['objective_A2'])
        return result


def build_model(source, context, settings=Settings(), on_event=None):
    settings.validate(); base=validate_model(source)
    if settings.child_mass==0:
        return copy.deepcopy(source),dict(zero_child_mass=True,children=[],context_used=False)
    require(context is not None, 'Missing fixed outside context')
    parents=branches(source); result=copy.deepcopy(source); b=result['base_model']; mass=settings.child_mass
    b['weights']=[(1-mass)*w for w in base['weights']]
    # Same physical width as the reviewed baseline, even if source angular units differ.
    covariance=np.diag(CHILD_VARIANCES); covariance[3:,3:] *= (base['angular_length']/REFERENCE_ELL)**2
    records=[]
    for label,parent in enumerate(parents):
        if on_event is not None: on_event(dict(kind='child_begun',parent_virtual_label=label,parent=parent))
        callback=None if on_event is None else lambda event,k=label:on_event(dict(parent_virtual_label=k,**event))
        search=RigidIncomingProblem(context,parent['center'],settings,callback).solve()
        child=len(b['weights']); b['anchors'].append(search['center']); b['means'].append([0.]*6)
        b['covariances'].append(covariance.tolist()); b['weights'].append(mass*parent['weight'])
        result['reciprocal_components'].append(False)
        records.append(dict(parent_virtual_label=label,parent_component=parent['component'],inverted=parent['inverted'],
            parent_weight=parent['weight'],child_component=child,child_weight=mass*parent['weight'],search=search))
        if on_event is not None: on_event(dict(kind='child_complete',**records[-1]))
    validate_model(result)
    return result,dict(zero_child_mass=False,context_used=True,context=copy.deepcopy(context.identity),settings=asdict(settings),
        parent_components=len(base['weights']),parent_virtual_branches=len(parents),children=records,
        child_covariance=covariance.tolist(),uses_native_classifier=False,original_model_provenance_unmodified=True,
        source_model_sha256=digest(source),model_sha256=digest(result),
        scope='Conditional core-geometry proposal asset; blind status depends on source-model provenance. Optimization is not projection of a draw, basin mass, physical energy, or evidence of equilibrium.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('model','shape','context','allocation','out'): parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args(); allocation=json.loads(args.allocation.read_bytes())
    require(allocation['schema']=='fixed-context-relaxed-atlas-allocation-v1', 'Wrong allocation schema')
    for name in ('model','shape','context'):
        require(sha(getattr(args,name))==allocation[name+'_sha256'], 'Changed '+name)
    require(sha(__file__)==allocation['exporter_sha256'], 'Changed exporter source')
    limits=allocation['limits']; require(limits['workers']==limits['threads']==1, 'Single-thread allocation required')
    require(all(type(limits[k]) is int and limits[k]>0 for k in ('cpu_seconds','wall_seconds','memory_bytes')), 'Invalid execution limits')
    require(all(os.environ.get(k)=='1' for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')), 'Set numerical thread environment to one before Python starts')
    resource.setrlimit(resource.RLIMIT_CPU,(limits['cpu_seconds'],limits['cpu_seconds']))
    resource.setrlimit(resource.RLIMIT_AS,(limits['memory_bytes'],limits['memory_bytes']))
    def timeout(*_): raise TimeoutError('Declared preparation wall cap')
    signal.signal(signal.SIGALRM,timeout)
    signal.alarm(limits['wall_seconds'])
    require(not args.out.exists(), 'Output must be fresh')
    source=json.loads(args.model.read_bytes()); shape=json.loads(args.shape.read_bytes()); context=json.loads(args.context.read_bytes())
    require(source['base_model']['shape_sha256']==sha(args.shape), 'Model/shape binding mismatch')
    settings=Settings(**allocation['settings']); settings.validate()
    require(len(branches(source))==allocation['virtual_branches'], 'Changed complete branch allocation')
    args.out.mkdir(parents=True); started=time.monotonic(); cpu=time.process_time()
    try:
        fixed=None if settings.child_mass==0 else OutsideContext.from_records(shape,context)
        with (args.out/'attempts.jsonl').open('x') as journal:
            def event(value):
                journal.write(json.dumps(value,allow_nan=False)+'\n'); journal.flush()
            model,report=build_model(source,fixed,settings,event)
        for name in ('model','shape','context'):
            require(sha(getattr(args,name))==allocation[name+'_sha256'], 'Input changed during calculation: '+name)
        report.update(complete=True,passed=True,wall_seconds=time.monotonic()-started,cpu_seconds=time.process_time()-cpu,
            input_sha256={str(getattr(args,k).resolve()):sha(getattr(args,k)) for k in ('model','shape','context','allocation')},
            exporter_sha256=sha(__file__),runtime=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,executable=sys.executable))
        for name,value in [('model.json',model),('report.json',report)]:
            with (args.out/name).open('x') as f: json.dump(value,f,indent=2,allow_nan=False); f.write('\n')
    except BaseException as error:
        with (args.out/'failure.json').open('x') as f: json.dump(dict(complete=False,passed=False,error=repr(error)),f,indent=2)
        raise


if __name__=='__main__': main()
