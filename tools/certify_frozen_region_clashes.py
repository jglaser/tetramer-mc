#!/usr/bin/env python3
"""Cover a frozen Cayley-chart ball by cells with explicit hard-clash witnesses.

Analytic enclosures are evaluated in ordinary floating point with conservative
slack and independently audited; this is not a formal interval-arithmetic proof.
"""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse
import hashlib
import itertools
import json
import time
from pathlib import Path
import shutil
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
OLD=Path('/home/xvg/protein-nucleation/results/coordination-test/narrow-water/high-activity/inputs')
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def rotation(p):return Rotation.from_quat(np.asarray(p['orientation'])[[1,2,3,0]]).as_matrix()
def cayley(c):return Rotation.from_quat(np.r_[c,1.]/np.sqrt(1.+np.dot(c,c))).as_matrix()

class Chart:
    def __init__(self,region):
        g=region['gaussian_chart'];cov=np.asarray(g['covariances'][0]);self.mean=np.asarray(g['means'][0]);self.ell=g['angular_length']
        assert len(g['means'])==1 and self.ell>0 and np.max(np.abs(cov-cov.T))<1e-12
        self.lower=np.zeros((6,6))
        # Same loop/order as Rust Chart::new, rather than relying on a BLAS factorization.
        for i in range(6):
            for j in range(i+1):
                rem=.5*(cov[i,j]+cov[j,i])-sum(self.lower[i,k]*self.lower[j,k] for k in range(j))
                self.lower[i,j]=np.sqrt(rem) if i==j else rem/self.lower[j,j]
        self.anchor_t=np.asarray(g['anchors'][0]['position']);self.anchor_r=np.asarray(g['anchors'][0]['rotation'])
        self.fixed_t=np.asarray(region['fixed_neighbor']['position']);self.fixed_r=rotation(region['fixed_neighbor'])
        assert np.max(np.abs(self.anchor_r.T@self.anchor_r-np.eye(3)))<1e-12
        self.tmat=self.lower[:3];self.rmat=self.lower[3:]
        self.tnorm=float(np.linalg.norm(self.tmat));self.rnorm=float(np.linalg.norm(self.rmat))
        self.cholesky_residual=float(np.max(np.abs(self.lower@self.lower.T-cov)))
    def pose(self,u):
        x=np.array([self.mean[i]+sum(self.lower[i,j]*u[j] for j in range(i+1)) for i in range(6)])
        return self.fixed_t+self.fixed_r@(self.anchor_t+x[:3]),self.fixed_r@cayley(x[3:]/self.ell)@self.anchor_r
    def displacement(self,center,half,radius,body_norm):
        h=min(float(np.linalg.norm(half)),radius+float(np.linalg.norm(center)))
        # Independent valid bounds for each linear map, then the triangle inequality.
        t=min(h*self.tnorm,float(np.linalg.norm(np.abs(self.tmat)@half)))
        c=min(h*self.rnorm,float(np.linalg.norm(np.abs(self.rmat)@half)))/self.ell
        return (t+2.*body_norm*c)*(1.+1e-12)+1e-10,dict(latent_radius=h,translation_bound_A=t,cayley_parameter_bound=c)

def cayley_audit():
    max_operator_ratio=0.;max_jacobian=0.;cases=0
    directions=np.array(list(itertools.product((-1.,0.,1.),repeat=3)));directions=directions[np.linalg.norm(directions,axis=1)>0];directions/=np.linalg.norm(directions,axis=1)[:,None]
    for magnitude in (0.,1e-6,.1,1.,10.,1e3):
        for direction in directions:
            c=magnitude*direction;s=1.+np.dot(c,c)
            metric=(s*np.eye(3)-np.outer(c,c))/(s*s)
            max_jacobian=max(max_jacobian,float(np.linalg.eigvalsh(metric).max()))
            for step in (1e-5,.01,.5,5.):
                for d in directions[::3]:
                    other=c+step*d;difference=np.linalg.norm(cayley(c)-cayley(other),ord=2)
                    ratio=difference/(2.*np.linalg.norm(c-other));max_operator_ratio=max(max_operator_ratio,float(ratio));cases+=1
                    assert difference<=2.*np.linalg.norm(c-other)+1e-12
    assert max_jacobian<=1.+1e-12
    return dict(cases=cases,max_rotation_operator_difference_ratio=max_operator_ratio,max_quaternion_metric_eigenvalue=max_jacobian,
        scope='Deterministic numerical check supporting the analytic global bound; no equilibrium or Monte Carlo sampling.')

def solve(chart,body,radii,fixed,rd,root_radius,out,max_nodes,deadline):
    del rd  # Core geometry only; depletion does not modify hard validity.
    norms=np.linalg.norm(body,axis=1);groups=[]
    for value in np.unique(radii):
        indices=np.flatnonzero(radii==value);groups.append((float(value),indices,cKDTree(fixed[indices])))
    cache=[];cache_set=set();full_queries=0
    def add_cache(pairs):
        for pair in pairs:
            pair=tuple(map(int,pair))
            if pair not in cache_set:cache.append(pair);cache_set.add(pair)
        while len(cache)>512:cache_set.remove(cache.pop(0))
    def witness(points,bounds,force=False):
        nonlocal full_queries
        best=None
        if cache:
            pairs=np.asarray(cache);i,j=pairs.T;distance=np.linalg.norm(points[i]-fixed[j],axis=1)
            margin=radii[i]+radii[j]-distance-bounds[i];k=int(np.argmax(margin))
            best=(float(margin[k]),int(i[k]),int(j[k]),float(distance[k]))
            if best[0]>args.slack and not force:return best
        eligible=np.arange(len(body)) if force else np.flatnonzero(bounds<radii+radii.max()-args.slack)
        if len(eligible):
            full_queries+=1;records=[]
            for value,indices,tree in groups:
                distance,nearest=tree.query(points[eligible],k=1,workers=1);j=indices[nearest]
                margins=radii[eligible]+value-distance-bounds[eligible]
                chosen=np.argsort(margins)[-min(8,len(margins)):]
                for k in chosen:
                    records.append((float(margins[k]),int(eligible[k]),int(j[k]),float(distance[k])))
            records.sort(reverse=True)
            if records:
                add_cache([(r[1],r[2]) for r in records[:16]])
                if best is None or records[0][0]>best[0]:best=records[0]
        return best
    t,r=chart.pose(np.zeros(6));witness(body@r.T+t,np.zeros(len(body)),force=True)
    stack=[('',np.zeros(6),np.full(6,root_radius))];records=[];visited=0;certified=0;outside=0;split=0;min_margin=float('inf');start=time.monotonic()
    while stack and visited<max_nodes and time.monotonic()<deadline:
        path,center,half=stack.pop();visited+=1
        min_radius=float(np.linalg.norm(np.maximum(np.abs(center)-half,0.)))
        base=dict(path=path,center=center.tolist(),half_width=half.tolist())
        if min_radius>root_radius+1e-10:
            records.append(dict(base,kind='outside_ball',minimum_latent_radius=min_radius));outside+=1;continue
        t,r=chart.pose(center);points=body@r.T+t;bounds,details=chart.displacement(center,half,root_radius,norms)
        best=witness(points,bounds)
        if best is not None and best[0]>args.slack:
            margin,i,j,distance=best;min_margin=min(min_margin,margin)
            records.append(dict(base,kind='collision',moving_atom=i,fixed_atom=j,center_distance_A=distance,
                center_penetration_A=float(radii[i]+radii[j]-distance),atom_displacement_bound_A=float(bounds[i]),
                certified_penetration_lower_A=margin,**details));certified+=1;continue
        # Axis choice changes efficiency only. Every split replaces a closed box
        # by its two closed half boxes; both are retained until certified/pruned.
        atom_norm=norms[best[1]] if best is not None else float(norms.max())
        score=half*(np.linalg.norm(chart.tmat,axis=0)+2.*atom_norm*np.linalg.norm(chart.rmat,axis=0)/chart.ell)
        axis=int(np.argmax(score));child_half=half.copy();child_half[axis]*=.5
        assert child_half[axis]>1e-12,'Subdivision below allowed numerical scale'
        records.append(dict(base,kind='split',axis=axis));split+=1
        for branch,sign in [('1',1.),('0',-1.)]:
            child_center=center.copy();child_center[axis]+=sign*child_half[axis]
            stack.append((path+branch,child_center,child_half.copy()))
    unresolved=[dict(path=p,center=c.tolist(),half_width=h.tolist()) for p,c,h in stack]
    with (out/f'R{root_radius:g}-tree.jsonl').open('w') as f:
        for record in records:f.write(json.dumps(record,separators=(',',':'))+'\n')
    save(out/f'R{root_radius:g}-unresolved.json',unresolved)
    result=dict(radius=root_radius,certified_entire_ball=not unresolved,visited_nodes=visited,collision_leaves=certified,
        outside_ball_leaves=outside,split_nodes=split,unresolved_cells=len(unresolved),max_depth=max((len(v['path']) for v in records),default=0),
        minimum_certified_penetration_lower_A=None if not certified else min_margin,full_nearest_neighbor_searches=full_queries,
        wall_seconds=time.monotonic()-start,slack_A=args.slack)
    audit_tree(chart,body,radii,fixed,root_radius,records,unresolved,result)
    save(out/f'R{root_radius:g}-result.json',result);return result

def audit_tree(chart,body,radii,fixed,radius,records,unresolved,result):
    by_path={v['path']:v for v in records};assert len(by_path)==len(records)
    pending={v['path']:v for v in unresolved};assert not set(by_path)&set(pending)
    all_nodes=dict(by_path,**pending);assert '' in all_nodes
    # Verify partition structure separately from traversal order.
    for path,row in all_nodes.items():
        center=np.asarray(row['center']);half=np.asarray(row['half_width'])
        if path:
            parent=by_path[path[:-1]];assert parent['kind']=='split';axis=parent['axis'];expected_half=np.array(parent['half_width']);expected_half[axis]*=.5
            expected_center=np.array(parent['center']);expected_center[axis]+=(1 if path[-1]=='1' else -1)*expected_half[axis]
            assert np.array_equal(half,expected_half) and np.array_equal(center,expected_center)
        else:assert np.array_equal(center,np.zeros(6)) and np.array_equal(half,np.full(6,radius))
        if row.get('kind')=='split':assert path+'0' in all_nodes and path+'1' in all_nodes
        elif row.get('kind')=='outside_ball':assert np.linalg.norm(np.maximum(np.abs(center)-half,0.))>radius+1e-10
        elif row.get('kind')=='collision':
            i,j=row['moving_atom'],row['fixed_atom'];t,r=chart.pose(center)
            # Independent direct pair distance; KD-tree output is never itself a proof.
            gap=np.linalg.norm(r@body[i]+t-fixed[j])-radii[i]-radii[j]
            bound,_=chart.displacement(center,half,radius,np.asarray([np.linalg.norm(body[i])]))
            assert -gap-bound[0]>args.slack
    # Deterministic checks on representative certified leaves. When a box corner
    # is outside the ball, truncate its segment from the closest box point to
    # the origin at the ball boundary. Both endpoints remain in the convex box.
    leaves=[v for v in records if v['kind']=='collision'];checked=0;max_violation=-float('inf')
    signs=np.array(list(itertools.product((-1.,1.),repeat=6)))
    for index in np.linspace(0,len(leaves)-1,min(128,len(leaves)),dtype=int) if leaves else []:
        row=leaves[int(index)];center=np.array(row['center']);half=np.array(row['half_width']);i,j=row['moving_atom'],row['fixed_atom'];t0,r0=chart.pose(center);x0=r0@body[i]+t0
        closest=np.clip(np.zeros(6),center-half,center+half)
        if np.linalg.norm(closest)>radius:continue
        for corner in center+signs*half:
            point=corner
            if np.linalg.norm(corner)>radius:
                direction=corner-closest;a=np.dot(direction,direction);b=2.*np.dot(closest,direction);c=np.dot(closest,closest)-radius*radius
                fraction=max(0.,min(1.,(-b+np.sqrt(max(0.,b*b-4.*a*c)))/(2.*a)))*(1.-1e-12)
                point=closest+fraction*direction
            assert np.linalg.norm(point)<=radius+1e-10 and np.all(np.abs(point-center)<=half+1e-10)
            t,r=chart.pose(point);x=r@body[i]+t;displacement=np.linalg.norm(x-x0)
            max_violation=max(max_violation,float(displacement-row['atom_displacement_bound_A']));assert displacement<=row['atom_displacement_bound_A']+1e-9
            assert np.linalg.norm(x-fixed[j])<radii[i]+radii[j];checked+=1
    result['audit']=dict(partition_structure_verified=True,all_witness_pairs_recomputed=True,deterministic_cell_ball_boundary_checks=checked,
        largest_tested_displacement_minus_bound_A=None if not checked else max_violation,scope='Ordinary floating-point audit of analytic bounds, not a formal rounding proof.')

def main(args):
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False);archive=out/'provenance';archive.mkdir();start=time.monotonic()
    region=read(args.region);environment=read(args.environment);shape=read(args.shape)
    assert sha(args.shape)==region['shape_sha256']
    assert len(environment['fixed_poses'])==2 and environment['fixed_poses'][0]==region['fixed_neighbor']
    assert environment['capture_center']==region['capture_center'] and environment['capture_radius']==region['capture_radius']
    chart=Chart(region);body=np.array([a['center'] for a in shape['atoms']]);radii=np.array([a['radius'] for a in shape['atoms']]);neighbor=environment['fixed_poses'][1]
    fixed=body@rotation(neighbor).T+neighbor['position']
    for path in [args.region,args.environment,args.shape,Path(__file__)]:shutil.copy2(path,archive/Path(path).name)
    sources=[args.region,args.environment,args.shape,Path(__file__),ROOT/'src/latent_region.rs',ROOT/'src/math.rs']
    save(out/'provenance.json',dict(source_sha256={str(p):sha(p) for p in sources},chart_lower=chart.lower.tolist(),cholesky_residual=chart.cholesky_residual,
        prescribed_neighbor=neighbor,core_only=True,configuration_inputs_modified=False,
        scope='Entire frozen latent ball tested, a superset of its hard-valid capture/q-restricted physical region. No claim about other competitor poses, formation costs, or equilibrium weights.'))
    save(out/'cayley-audit.json',cayley_audit());results=[];remaining=args.max_nodes;deadline=start+args.max_seconds
    for radius in args.radii:
        if results and (not results[-1]['certified_entire_ball'] or time.monotonic()-start>60 or remaining<args.max_nodes//2):break
        result=solve(chart,body,radii,fixed,region['depletant_radius'],radius,out,remaining,deadline);results.append(result);remaining-=result['visited_nodes'];print(json.dumps(result),flush=True)
    save(out/'results.json',dict(results=results,wall_seconds=time.monotonic()-start,total_node_budget=args.max_nodes,total_time_budget_seconds=args.max_seconds,
        scope='Analytic geometric enclosure with conservative numerical slack and numerical audits; not formally rounded interval arithmetic. Hard exclusion only for this prescribed second neighbor and the stated latent balls.'))
    lines=['# Frozen deep-contact region: prescribed-neighbor clash certificate','',
        'The test covers the entire stated latent ball, including poses outside the capture sphere or failing the original neighbor constraint. Certifying that larger set also certifies the frozen physical subregion.','',
        '| Radius | Entire ball certified? | Visited cells | Collision leaves | Outside leaves | Unresolved | Minimum penetration lower bound / Å | Seconds |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in results:lines.append(f"|{r['radius']:g}|{r['certified_entire_ball']}|{r['visited_nodes']}|{r['collision_leaves']}|{r['outside_ball_leaves']}|{r['unresolved_cells']}|{r['minimum_certified_penetration_lower_A']}|{r['wall_seconds']:.2f}|")
    lines+=['','Every collision leaf records one actual atomic pair whose center penetration exceeds an analytic bound on that moving atom’s displacement throughout the cell intersected with the latent ball. All witness pairs and the binary partition are rechecked independently. Unresolved cells remain unresolved; no probabilistic conclusion is drawn from their count.','',
        'The rotation map uses q(c)=(1,c)/sqrt(1+|c|²). Its pullback metric is [(1+|c|²)I−ccᵀ]/(1+|c|²)², bounded above by I. A straight c-path therefore has rotational angular length at most2|Δc|. An atom at body radiusr moves at most2r|Δc|. Translation and rotation contributions are added by the triangle inequality.','',
        'For a cell centeru₀ and half-widthsw, h=min(||w||₂,R+||u₀||₂) encloses u−u₀ over the cell/ball intersection. Each linear-map bound is the minimum of h times its Frobenius norm and the norm of its rowwise absolute coefficient sum againstw. Both are valid upper bounds. A 1e−12 relative inflation,1e−10Å additive bound inflation and the reported positive clearance slack protect ordinary numerical evaluation. This is an analytic enclosure with a numerical audit, not a formal floating-point proof.','',
        'Implementation choices that can be revisited: axis-aligned six-dimensional cells; cached witness pairs; nearest-neighbor trees grouped by atomic radius; Frobenius/rowwise bounds instead of sharper coupled bounds; geometric axis scoring; explicit cell/time limits. These affect efficiency or unresolved coverage, not the witness criterion.','',
        'This result concerns only the frozen chart and the prescribed second native neighbor. It does not exclude other competing arrangements, establish an assembly pathway, or supply a neighbor-formation free energy.','']
    (out/'report.md').write_text('\n'.join(lines))
    save(out/'artifacts.json',{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.name!='artifacts.json'})

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--region',type=Path,default=ROOT/'runs/smc-normalizer-deep-far/site0/fixed-discovered-region.json')
    parser.add_argument('--shape',type=Path,default=OLD/'tetramer-shape.json');parser.add_argument('--environment',type=Path,default=OLD/'geometry/environment-site0-m2.json')
    parser.add_argument('--radii',type=float,nargs='+',default=[3.,5.,8.]);parser.add_argument('--max-nodes',type=int,default=100000);parser.add_argument('--max-seconds',type=float,default=300.);parser.add_argument('--slack',type=float,default=1e-6)
    args=parser.parse_args();assert all(r>0 for r in args.radii) and args.max_nodes>0 and args.max_seconds>0 and args.slack>=1e-8
    main(args)
