#!/usr/bin/env python3
"""Independent raw translation/Cayley-coordinate Gaussian conditioner audit.

The angular geometry is reconstructed from atomic quadratic inequalities,
independently of the Rust BVH traversal. Frozen Normal integration and Gaussian
coordinate code are reused unchanged. No sampling or physics runs on import.
"""
import os
for _name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[_name]='1'
import argparse
import copy
from collections import Counter
import json
import math
from pathlib import Path
import time
import numpy as np
from scipy.linalg import solve, solve_triangular
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from scipy.special import logsumexp
import hard_free_line_reference as old
import analyze_contact_line_audit as normal

require,read,sha,close=normal.require,normal.read,normal.sha,normal.close
contains,interval_masses=normal.contains,normal.interval_masses
interval,union,subtract_closed_segment=old.interval,old.union,old.subtract_closed_segment


def quadratic_negative_intervals(coefficients,segment):
    """Strict negativity of c+b*s+a*s², clipped to a closed finite segment.

    All degree/topology cases are explicit; downward-opening tangencies retain
    their isolated allowed root. Roots use the cancellation-resistant q formula.
    Long-double arithmetic is a numerical reference, not interval certification.
    """
    c,b,a=map(np.longdouble,coefficients);lo,hi=map(float,segment)
    require(np.isfinite([c,b,a]).all() and math.isfinite(lo) and math.isfinite(hi) and lo<=hi,'Invalid quadratic/chord')
    if lo==hi:return [interval(lo,hi)] if (a*np.longdouble(lo)+b)*np.longdouble(lo)+c<0 else []
    pieces=[]
    if a==0:
        if b==0:
            if c<0:pieces=[(-np.inf,np.inf,False,False)]
        else:
            root=-c/b
            pieces=[(-np.inf,root,False,False)] if b>0 else [(root,np.inf,False,False)]
    else:
        disc=b*b-4*a*c
        if disc<0:
            if a<0:pieces=[(-np.inf,np.inf,False,False)]
        elif disc==0:
            if a<0:
                root=-b/(2*a);pieces=[(-np.inf,root,False,False),(root,np.inf,False,False)]
        else:
            rootdisc=np.sqrt(disc);q=-np.longdouble(.5)*(b+np.copysign(rootdisc,b))
            roots=sorted([q/a,c/q])
            if a>0:pieces=[(roots[0],roots[1],False,False)]
            else:pieces=[(-np.inf,roots[0],False,False),(roots[1],np.inf,False,False)]
    out=[]
    for lower,upper,lc,uc in pieces:
        lower,upper=float(lower),float(upper)
        l=max(lo,lower);h=min(hi,upper)
        lower_closed=lo>lower or (lo==lower and lc)
        upper_closed=hi<upper or (hi==upper and uc)
        if l<h or (l==h and lower_closed and upper_closed):out.append(interval(l,h,lower_closed,upper_closed))
    return union(out)


def sphere_coefficients(atom_ref,fixed_relative,c0,angular_axis,ell,radius):
    """Ascending raw-coordinate coefficients, with c=c0+(s/ell)e_axis."""
    a,b,v=(np.asarray(x,np.longdouble) for x in (atom_ref,fixed_relative,c0))
    require(v.shape==a.shape==b.shape==(3,) and v[angular_axis]==0 and ell>0,'Invalid angular fiber')
    e=np.eye(3,dtype=np.longdouble)[angular_axis]
    n0=(1-v@v)*a+2*v*(v@a)+2*np.cross(v,a)
    n1=2*(e*(v@a)+v*(e@a)+np.cross(e,a))
    n2=-a+2*e*(e@a);k=a@a+b@b-np.longdouble(radius)**2
    return np.array([k*(1+v@v)-2*(b@n0),-2*(b@n1)/ell,(k-2*(b@n2))/(np.longdouble(ell)**2)],np.longdouble)


def angular_hard_free_intervals(shape,fixed_poses,world_center,Rfixed,Ranchor,c0,angular_axis,ell,segment,use_tree=True):
    """KD-tree swept-circle candidates; analytic per-atom quadratic leaves."""
    atoms=np.asarray([a['center']for a in shape['atoms']]);radii=np.asarray([a['radius']for a in shape['atoms']])
    center=np.asarray(world_center);Rf=np.asarray(Rfixed);Ra=np.asarray(Ranchor);v=np.asarray(c0)
    require(v[angular_axis]==0 and ell>0,'Selected raw Cayley coordinate was not removed')
    reference=atoms@Ra.T;h=math.sqrt(1+v@v);e=np.eye(3)[angular_axis]
    k=(e+np.cross(v,e))/h;world_axis=Rf@k
    R0=Rotation.from_quat(np.r_[v,1.]/h).as_matrix();offset=reference@R0.T@Rf.T
    projection=offset@world_axis;circle_centers=center+projection[:,None]*world_axis
    circle_radii=np.linalg.norm(offset-projection[:,None]*world_axis,axis=1)
    blocked=[];candidates=0
    for pose in fixed_poses:
        fixed=atoms@normal.rotation(pose['orientation']).T+pose['position']
        tree=cKDTree(fixed) if use_tree else None
        for i,a in enumerate(reference):
            # This enclosing sphere contains the full rotation circle, including
            # all finite Cayley values. It can only add conservative candidates.
            js=tree.query_ball_point(circle_centers[i],circle_radii[i]+radii[i]+radii.max()+1e-9) if use_tree else range(len(fixed))
            candidates+=len(js)
            for j in js:
                b=Rf.T@(fixed[j]-center)
                coefficients=sphere_coefficients(a,b,v,angular_axis,ell,radii[i]+radii[j])
                blocked.extend(quadratic_negative_intervals(coefficients,segment))
    core=union(blocked);free=subtract_closed_segment(segment,core)
    return dict(intervals=free,hard_overlap=core,swept_circle_pair_candidates=candidates)


class Reconstructor(old.Reconstructor):
    def __init__(self,region,guide,config,shape):
        require(guide['schema']=='defensive-hard-free-pose-line-guide-v1','Wrong hard-free pose-line schema')
        require('raw_translation_axes' not in guide and 'contact_widths_A' not in guide and 'contact_neighbor_indices' not in guide,'Unexpected translation/contact constraints')
        axes=guide['raw_pose_axes'];require(axes and len(set(axes))==len(axes) and all(type(a)is int and 0<=a<6 for a in axes),'Bad pose axes')
        # Initialize only the unchanged Gaussian/chart machinery through a
        # private translation-schema adapter, then compute Schur conditionals
        # for the actual six-coordinate axis set independently.
        adapted=copy.deepcopy(guide);adapted.pop('raw_pose_axes');adapted.update(schema='defensive-hard-free-line-guide-v1',raw_translation_axes=[0])
        super().__init__(region,adapted,config,shape)
        self.guide=copy.deepcopy(guide);self.axes=list(axes);self.conditionals={}
        covariance=np.asarray([c['covariance']for c in guide['gaussian_components']]).reshape((-1,6,6))
        rawcov=np.einsum('ij,kjl,ml->kim',self.L0,covariance,self.L0)
        for axis in self.axes:
            others=[j for j in range(6) if j!=axis];reg=[];sigma=[]
            for cov in rawcov:
                slope=solve(cov[np.ix_(others,others)],cov[others,axis],assume_a='pos')
                variance=cov[axis,axis]-cov[axis,others]@slope
                require(variance>0 and math.isfinite(variance),'Invalid angular Schur variance')
                reg.append(slope);sigma.append(math.sqrt(variance))
            self.conditionals[axis]=(others,np.asarray(reg),np.asarray(sigma))

    def reconstruct_axis(self,u,axis,use_tree=True):
        require(axis in self.axes,'Unknown raw pose axis')
        if axis<3:return super().reconstruct_axis(u,axis,use_tree)
        x=self.raw(u);raw=x.copy();raw[axis]=0
        u0=solve_triangular(self.L0,raw-self.m0,lower=True);du=solve_triangular(self.L0,np.eye(6)[axis],lower=True)
        r4=self.chord(u0,du,self.radius)
        if r4 is None:return dict(axis=axis,intervals=[],empty_reason='no_R4_chord')
        _,position,_,_=self.decode(u0)
        if np.linalg.norm(position-np.asarray(self.config['capture_center']))>self.config['capture_radius']:
            return dict(axis=axis,intervals=[],empty_reason='no_capture_at_fixed_center')
        c0=raw[3:]/self.ell
        geometry=angular_hard_free_intervals(self.shape,self.config['fixed_poses'],position,self.Rf,np.asarray(self.anchor['rotation']),c0,axis-3,self.ell,r4,use_tree)
        result=dict(axis=axis,coordinate_kind='raw-scaled-Cayley',segment=r4,world_center=position.tolist(),fixed_cayley=c0.tolist(),length_scale=self.ell,**geometry)
        if not any(i['lower']<i['upper']for i in geometry['intervals']):
            result['empty_reason']='no_positive_hard_free_length'
            # Angular diagnostics retain isolated feasible tangencies. Their
            # Normal mass is zero, so the identical original-draw fallback applies.
        return result


def compare_axis(actual,expected):
    maximum=old.compare_axis(actual,expected)
    if expected['axis']>=3:
        require('origin' not in actual and 'direction' not in actual,'Angular fiber mistaken for translation')
        if 'segment' in expected and 'coordinate_kind' in expected:
            require(actual['coordinate_kind']=='raw-scaled-Cayley','Wrong angular coordinate')
            for key in ('world_center','fixed_cayley','length_scale'):close(actual[key],expected[key],'Different angular '+key)
    return maximum

# Audit ledger/row contract follows the frozen translation reference, with
# this module's independently reconstructed six-coordinate geometry.
def audit(directory):
    root=Path(directory).resolve();started=time.process_time();manifest=read(root/'manifest.json');summary=read(root/'summary.json')
    require(manifest['schema']=='hard-free-pose-line-guide-audit-v1' and manifest['physical_jobs']==0,'Wrong audit scope')
    require(summary['complete'] and summary['manifest']==manifest,'Incomplete audit')
    require(not (root/'failure.json').exists(),'A failed query must not be hidden by a complete receipt')
    bindings={str(p):sha(p) for p in [root/'manifest.json',root/'summary.json',Path(__file__).resolve(),Path(normal.__file__).resolve(),Path(old.__file__).resolve()]}
    for name,key in [('config.json','config_sha256'),('region.json','region_sha256'),('importance-guide.json','guide_sha256'),('shape.json','shape_sha256'),('source-bundle.json','source_bundle_sha256')]:
        p=root/'provenance'/name;require(sha(p)==manifest[key],'Changed provenance '+name);bindings[str(p)]=sha(p)
    region,guide,config,shape=[read(root/'provenance'/name) for name in ['region.json','importance-guide.json','config.json','shape.json']]
    require(guide['region_sha256']==manifest['region_sha256'] and region['shape_sha256']==manifest['shape_sha256'],'Broken guide/region/shape binding')
    require(config['fixed_poses']==region.get('physical_fixed_neighbors',[region['fixed_neighbor']]),'Changed scaffold')
    for rk,ck in [('capture_center','capture_center'),('capture_radius','capture_radius'),('depletant_radius','depletant_radius'),('activity','reservoir_density'),('physical_metric','metadata')]:require(region[rk]==config[ck],'Changed physical field '+rk)
    recon=Reconstructor(region,guide,config,shape);rows={}
    close(manifest['log_latent_ball_volume'],recon.logvolume,'Changed latent ball volume');close(manifest['latent_radius'],recon.radius,'Changed latent radius')
    for name in ['samples','probes']:
        p=root/(name+'.jsonl');require(sha(p)==summary[name+'_sha256'],'Changed saved '+name);bindings[str(p)]=sha(p)
        rows[name]=[json.loads(s) for s in p.read_text().splitlines()]
    probes=[]
    if manifest['probes_sha256'] is not None:
        p=root/'provenance/probes.jsonl';require(sha(p)==manifest['probes_sha256'],'Changed fixed probes');bindings[str(p)]=sha(p);probes=[json.loads(s) for s in p.read_text().splitlines()]
    require(len(rows['samples'])==manifest['samples']==summary['samples'] and [r['id'] for r in rows['samples']]==list(range(manifest['samples'])),'Incomplete attempted-draw allocation')
    require(len(rows['probes'])==len(probes)==summary['probes'],'Incomplete saved queries')
    for r,p in zip(rows['probes'],probes):require(r['id']==p['id'] and r['latent']==p['latent'],'Changed query identity')
    journal=root/'attempts.jsonl';require(sha(journal)==summary['attempts_sha256'],'Attempt journal changed');bindings[str(journal)]=sha(journal)
    attempts=[json.loads(s) for s in journal.read_text().splitlines()]
    expected_attempts=[dict(ordinal=i,kind=row['kind'],id=row['id'],state='begin') for i,row in enumerate(rows['samples']+rows['probes'])]
    require(attempts==expected_attempts,'Missing, duplicated or reordered attempted query')
    maxima=Counter();counts=Counter();audited=[]
    for kind,records in rows.items():
        for row in records:
            require(row['kind']==('fresh' if kind=='samples' else 'probe'),'Wrong row type')
            u=np.asarray(row['latent']);x,position,R,jac=recon.decode(u);result=recon.density(u)
            require(u.shape==(6,) and normal.finite(u),'Invalid latent coordinate')
            close(row['latent_radius'],np.linalg.norm(u),'Changed latent radius record')
            close(row['raw_coordinates'],x,'Raw reconstruction failed');close(row['pose']['position'],position,'Position reconstruction failed')
            close(normal.rotation(row['pose']['orientation']),R,'Orientation reconstruction failed');close(row['log_physical_jacobian'],jac,'Jacobian differs')
            maxima['log_physical_jacobian']=max(maxima['log_physical_jacobian'],abs(row['log_physical_jacobian']-jac))
            q=result['log_density'];base=result['baseline_log_density']
            for value,saved,label in [(q,row['log_proposal_density'],'log_q'),(base,row['baseline_log_density'],'log_q_old')]:
                if value==-math.inf:require(saved is None,'Nonzero density outside support')
                else:close(saved,value,'Different '+label,atol=2e-7,rtol=1e-11);maxima[label]=max(maxima[label],abs(saved-value))
            hard=recon.hard_valid(position,R);capture=np.linalg.norm(position-config['capture_center'])<=config['capture_radius'];inside=np.linalg.norm(u)<=recon.radius
            require(row['hard_valid']==hard and row['capture_valid']==capture and row['shell_valid']==inside,'Different physical indicators')
            if hard and capture and inside:require(q>=base-2e-9,'Feasible conditioning reduced physical-support density')
            details=row['density_details']
            if result.get('conditioning_disabled'):require(details.get('conditioning_disabled'),'Missing disabled guide flag')
            else:
                require(details['component_branches']==result['component_branches'] and details['fallback_component_branches']==result['fallback_component_branches'],'Missing branches or changed fallback')
                require(len(details['axes'])==len(result['axes']),'Missing axis')
                close(details['raw_coordinates'],x,'Changed detailed raw coordinates');close(details['baseline_log_density'],base,'Changed detailed baseline')
                weighted_logs=recon.gaussian_logs(u);gaussian_logs=weighted_logs-np.log(recon.weights)
                uniform=math.log(recon.alpha)-recon.logvolume if inside else -math.inf
                for actual,expected in zip(details['axes'],result['axes']):
                    maxima['interval_endpoints']=max(maxima['interval_endpoints'],compare_axis(actual,expected))
                    require(len(actual['components'])==len(recon.weights),'Missing per-axis Gaussian components')
                    multiplier=[];allowed=contains(expected['intervals'],x[expected['axis']])
                    for k,c in enumerate(actual['components']):
                        require(c['component']==k,'Changed per-axis Gaussian order')
                        for key,value in [('gaussian_log_density',gaussian_logs[k]),('conditional_mean',expected['conditional_means'][k]),('conditional_sigma',expected['conditional_sigmas'][k])]:close(c[key],value,'Different per-axis '+key)
                        mass=expected['conditional_masses'][k];fallback=expected['component_fallbacks'][k]
                        close(c['conditional_mass'],mass,'Different per-axis conditional mass',atol=2e-15,rtol=2e-7)
                        require(c['fallback']==fallback and c['query_coordinate_allowed']==allowed,'Different per-axis branch or predicate')
                        multiplier.append(1. if fallback else (1/mass if allowed else 0.))
                    correction=1-recon.beta+recon.beta*np.array(multiplier);positive=correction>0
                    axis_q=float(np.logaddexp(uniform,math.log1p(-recon.alpha)+logsumexp(weighted_logs[positive]+np.log(correction[positive]))))
                    if axis_q==-math.inf:require(actual['axis_log_proposal_density'] is None,'Nonzero unsupported axis density')
                    else:close(actual['axis_log_proposal_density'],axis_q,'Different complete single-axis density',atol=2e-7,rtol=1e-11)
                    require(math.isfinite(actual['geometry_cpu_seconds']) and actual['geometry_cpu_seconds']>=0,'Invalid geometry CPU')
            draw=row['draw']
            if kind=='samples':counts.update(attempted=1,conditioned_draws=int(draw['conditional']),fallback_draws=int(draw.get('fallback',False)),hard_capture_shell_valid=int(hard and capture and inside))
            else:require(draw is None,'Saved query drew a new pose')
            if draw and draw['conditional']:
                axis=draw['axis'];k=draw['component'];original=recon.raw(draw['original_latent']);others=[a for a in range(6) if a!=axis]
                require(axis in recon.axes and 0<=k<len(recon.weights),'Invalid selected component or axis')
                require('width_index' not in draw and 'width_A' not in draw,'Unexpected contact-width draw label')
                close(original[others],x[others],'Conditioner changed retained coordinates')
                if draw['fallback']:close(draw['original_latent'],u,'Fallback did not retain original draw')
                geometry=next(g for g in result['axes'] if g['axis']==axis);means,sigmas=recon.conditional(x,axis);mass=float(interval_masses(geometry['intervals'],means[k],sigmas[k]))
                maxima['draw_interval_endpoints']=max(maxima['draw_interval_endpoints'],compare_axis(draw['geometry'],geometry))
                close(draw['conditional_mean'],means[k],'Conditional mean differs');close(draw['conditional_sigma'],sigmas[k],'Conditional sigma differs')
                close(draw['conditional_mass'],mass,'Conditional mass differs',atol=2e-15,rtol=2e-7);require(draw['fallback']==(mass<=recon.floor),'Selected floor branch differs')
                if not draw['fallback']:
                    require(hard and capture and inside and contains(geometry['intervals'],x[axis]),'Successful line draw violates feasibility')
                    selected=draw['selected_interval'];selected_index=next((i for i,r in enumerate(geometry['intervals']) if all(abs(selected[key]-r[key])<1e-9 for key in ['lower','upper'])),None)
                    require(selected_index is not None,'Selected interval absent')
                    m=float(interval_masses([selected],means[k],sigmas[k]));cdf=float(normal.normal_masses((selected['lower']-means[k])/sigmas[k],(x[axis]-means[k])/sigmas[k]))/m
                    close(draw['selected_interval_mass'],m,'Selected interval mass differs',atol=2e-15,rtol=2e-7)
                    lower_mass=float(interval_masses(geometry['intervals'][:selected_index],means[k],sigmas[k]));selection=draw['uniform_interval_selection']*mass
                    require(0<draw['uniform_interval_selection']<1 and lower_mass-2e-14<=selection<=lower_mass+m+2e-14,'Categorical Gaussian interval selection differs')
                    close(draw['returned_raw_coordinate'],x[axis],'Changed returned coordinate')
                    error=abs(cdf-draw['uniform_within_interval']);require(error<2e-6,'Inverse conditional CDF mismatch');maxima['inverse_cdf']=max(maxima['inverse_cdf'],error)
            elif draw:close(draw['original_latent'],u,'Unconditioned draw changed the baseline sample')
            audited.append(dict(id=row['id'],kind=row['kind'],log_q=q if math.isfinite(q) else None,log_q_old=base if math.isfinite(base) else None,hard_valid=hard,capture_valid=bool(capture),shell_valid=bool(inside)))
    for key in ['conditioned_draws','fallback_draws','hard_capture_shell_valid']:require(counts[key]==summary[key],'Summary count differs '+key)
    close(summary['maximum_backmap_error'],max((r['backmap_error'] for r in rows['samples']),default=0.),'Backmap summary differs',atol=1e-15,rtol=1e-12)
    for key in ['draw_cpu_seconds','density_cpu_seconds']:close(summary[key],math.fsum(r[key] for r in rows['samples']),'Summary CPU differs '+key,atol=1e-9)
    for p,h in bindings.items():require(sha(p)==h,'Input changed during audit')
    return dict(schema='independent-hard-free-pose-line-audit-v1',complete=True,root=str(root),rows=audited,counts=dict(counts),maximum_errors=dict(maxima),input_sha256=bindings,
        probes=len(probes),analysis_cpu_seconds=time.process_time()-started,new_pose_draws=0,new_Poisson_clouds=0,
        executable_sha256=manifest['executable_sha256'],source_bundle_sha256=manifest['source_bundle_sha256'],
        scope='Independent atomic quadratic/angular and translation geometry, complete density and draw-trace audit. No new poses, target weights or native labels.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);a=parser.parse_args()
    require(not a.out.exists(),'New receipt path required');result=audit(a.root);a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(a.out)
