#!/usr/bin/env python3
"""Independent reconstruction of a completed proposal-only contact-line audit.

No new random points, physical weights, or native labels are generated. Rust
interval endpoints are checked structurally; pose hard/contact predicates are
recomputed with independent atom KD trees. Interval completeness has a separate
brute-force geometry validation obligation.
"""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
from scipy.linalg import solve,solve_triangular
from scipy.special import log_ndtr,logsumexp
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def require(ok,message):
    if not ok:raise ValueError(message)
def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def finite(value):return np.isfinite(np.asarray(value,dtype=float)).all()
def close(actual,expected,label,atol=2e-8,rtol=2e-10):
    require(finite(actual) and finite(expected) and np.allclose(actual,expected,atol=atol,rtol=rtol),label)
def rotation(q):
    q=np.asarray(q,float);require(q.shape==(4,) and abs(np.linalg.norm(q)-1)<1e-8,'Invalid quaternion')
    return Rotation.from_quat(q[[1,2,3,0]]).as_matrix()


_NODES,_WEIGHTS=np.polynomial.legendre.leggauss(16)
def normal_masses(lo,hi):
    """Independent 16-point quadrature / log-CDF difference, vectorized."""
    lo,hi=np.broadcast_arrays(np.asarray(lo,float),np.asarray(hi,float))
    require(finite(lo) and finite(hi) and np.all(lo<=hi),'Invalid normal endpoints')
    width=hi-lo;mid=lo+width/2;answer=np.zeros(lo.shape)
    small=(width>0)&(width*(1+abs(mid))<.25)
    if np.any(small):
        points=mid[small,None]+width[small,None]*_NODES/2
        answer[small]=width[small]/(2*math.sqrt(2*math.pi))*np.sum(_WEIGHTS*np.exp(-points*points/2),axis=-1)
    other=(width>0)&~small
    if np.any(other):
        a,b=lo[other],hi[other]
        positive=a>=0
        loglarge=np.where(positive,log_ndtr(-a),log_ndtr(b))
        logsmall=np.where(positive,log_ndtr(-b),log_ndtr(a))
        with np.errstate(divide='ignore',invalid='ignore',under='ignore'):
            answer[other]=np.exp(loglarge+np.log(-np.expm1(logsmall-loglarge)))
    require(finite(answer) and np.all((answer>=0)&(answer<=1+1e-14)),'Invalid independent normal mass')
    return answer


def validate_intervals(intervals,segment=None):
    previous=None
    for r in intervals:
        require(set(r)=={'lower','upper','lower_closed','upper_closed'},'Unknown interval fields')
        require(finite([r['lower'],r['upper']]) and r['lower']<=r['upper'],'Invalid finite interval')
        require(type(r['lower_closed']) is bool and type(r['upper_closed']) is bool,'Invalid endpoint flags')
        require(r['lower']<r['upper'] or (r['lower_closed'] and r['upper_closed']),'Empty stored interval')
        if previous is not None:
            require(previous['upper']<=r['lower'],'Overlapping intervals double-count mass')
            if previous['upper']==r['lower']:
                require(not(previous['upper_closed'] or r['lower_closed']),'Touching intervals were not merged')
        if segment is not None:
            require(segment[0]-1e-9<=r['lower']<=r['upper']<=segment[1]+1e-9,'Interval outside declared segment')
        previous=r


def contains(intervals,s):
    return any((s>r['lower'] or(s==r['lower'] and r['lower_closed'])) and
        (s<r['upper'] or(s==r['upper'] and r['upper_closed'])) for r in intervals)


def interval_masses(intervals,means,sigmas):
    means=np.asarray(means,float);sigmas=np.asarray(sigmas,float)
    if not intervals:return np.zeros(means.shape)
    a=np.asarray([r['lower'] for r in intervals]);b=np.asarray([r['upper'] for r in intervals])
    values=normal_masses((a-means[...,None])/sigmas[...,None],(b-means[...,None])/sigmas[...,None])
    result=values.sum(axis=-1)
    require(np.all(result<=1+1e-12),'Union Gaussian mass exceeds one')
    return result


class Reconstructor:
    def __init__(self,region,guide,config,shape):
        self.region,self.guide,self.config=region,guide,config
        require(guide['schema']=='defensive-contact-line-guide-v1','Wrong guide schema')
        require(region.get('minimum_mahalanobis_radius',0.)==0.,'Only complete R4 supported')
        chart=region['gaussian_chart'];require(len(chart['means'])==len(chart['anchors'])==1,'Expected single raw chart')
        self.m0=np.asarray(chart['means'][0]);self.L0=np.linalg.cholesky(np.asarray(chart['covariances'][0]))
        self.ell=chart['angular_length'];self.anchor=chart['anchors'][0];self.fixed=region['fixed_neighbor']
        self.Rf=rotation(self.fixed['orientation']);self.radius=region['mahalanobis_radius']
        self.logvolume=3*math.log(math.pi)+6*math.log(self.radius)-math.log(6)
        self.alpha=guide['defensive_uniform_shell_probability'];self.beta=guide['conditional_probability']
        self.floor=guide['minimum_conditional_mass'];self.axes=guide['raw_translation_axes'];self.widths=guide['contact_widths_A']
        self.contact_indices=guide['contact_neighbor_indices']
        require(0<self.alpha<=1 and 0<=self.beta<=1 and 0<self.floor<1,'Invalid mixture controls')
        require(self.axes and len(set(self.axes))==len(self.axes) and all(a in (0,1,2) for a in self.axes),'Bad axes')
        require(self.widths and all(math.isfinite(w) and w>0 for w in self.widths),'Bad contact widths')
        comps=guide['gaussian_components'];weights=np.asarray([c['weight'] for c in comps])
        require(len(comps)>0 or self.alpha==1.,'Missing Gaussian mixture')
        self.weights=weights/weights.sum() if len(weights) else weights
        require(np.all(self.weights>0),'Invalid Gaussian weights')
        self.means=np.asarray([c['mean'] for c in comps]).reshape((-1,6));cov=np.asarray([c['covariance'] for c in comps]).reshape((-1,6,6))
        self.lowers=np.linalg.cholesky(cov)
        self.normalizers=-3*math.log(2*math.pi)-np.log(np.diagonal(self.lowers,axis1=1,axis2=2)).sum(axis=1)
        self.rawmeans=self.m0+self.means@self.L0.T
        rawcov=np.einsum('ij,kjl,ml->kim',self.L0,cov,self.L0)
        self.conditionals={}
        for a in self.axes:
            others=[j for j in range(6) if j!=a];reg=[];sigma=[]
            for s in rawcov:
                b=solve(s[np.ix_(others,others)],s[others,a],assume_a='pos')
                variance=s[a,a]-s[a,others]@b
                require(variance>0 and math.isfinite(variance),'Nonpositive Schur variance')
                reg.append(b);sigma.append(math.sqrt(variance))
            self.conditionals[a]=(others,np.asarray(reg),np.asarray(sigma))
        self.atoms=np.asarray([a['center'] for a in shape['atoms']]);self.radii=np.asarray([a['radius'] for a in shape['atoms']])
        self.fixed_world=[self.atoms@rotation(p['orientation']).T+p['position'] for p in config['fixed_poses']]
        self.fixed_trees=[cKDTree(p) for p in self.fixed_world]

    def raw(self,u):return self.m0+self.L0@np.asarray(u)
    def conditional(self,x,axis):
        others,b,sigma=self.conditionals[axis]
        return self.rawmeans[:,axis]+np.einsum('ki,ki->k',b,np.asarray(x)[others]-self.rawmeans[:,others]),sigma
    def gaussian_logs(self,u):
        if not len(self.weights):return np.empty(0)
        z=np.asarray([solve_triangular(L,np.asarray(u)-m,lower=True) for L,m in zip(self.lowers,self.means)])
        return np.log(self.weights)+self.normalizers-np.sum(z*z,axis=1)/2
    def decode(self,u):
        x=self.raw(u);c=x[3:]/self.ell
        Q=Rotation.from_quat(np.r_[c,1.]/math.sqrt(1+c@c)).as_matrix()
        R=self.Rf@Q@np.asarray(self.anchor['rotation'])
        p=np.asarray(self.fixed['position'])+self.Rf@(np.asarray(self.anchor['position'])+x[:3])
        J=np.log(np.diag(self.L0)).sum()-3*math.log(self.ell)-2*math.log(math.pi)-2*math.log1p(c@c)
        return x,p,R,float(J)
    def chord(self,origin,direction,radius):
        aa=direction@direction;center=-(origin@direction)/aa;nearest=origin+center*direction
        residual=radius*radius-nearest@nearest
        if residual<=0:return None
        half=math.sqrt(residual/aa);return [center-half,center+half]
    def geometry_sets(self,detail,x,axis):
        require(detail['axis']==axis,'Wrong axis detail')
        raw=np.array(x);raw[axis]=0
        u0=solve_triangular(self.L0,raw-self.m0,lower=True)
        d=solve_triangular(self.L0,np.eye(6)[axis],lower=True)
        _,position,orientation,_=self.decode(u0);direction=self.Rf[:,axis]
        r4=self.chord(u0,d,self.radius)
        cap=self.chord(position-np.asarray(self.config['capture_center']),direction,self.config['capture_radius'])
        if 'empty_reason' in detail:
            reason=detail['empty_reason']
            require(reason in ('no_R4_chord','no_capture_chord','disjoint_chords','no_positive_hard_free_length'),'Unknown fallback geometry reason')
            if reason=='no_R4_chord':require(r4 is None,'Incorrect empty R4 chord')
            if reason=='no_capture_chord':require(cap is None,'Incorrect empty capture chord')
            if reason=='disjoint_chords':require(r4 is not None and cap is not None and max(r4[0],cap[0])>=min(r4[1],cap[1]),'Incorrect disjoint chords')
            sets=[[] for _ in self.widths]
        else:
            require([d['width_A'] for d in detail['widths']]==self.widths,'Changed width order')
            sets=[d['intervals'] for d in detail['widths']]
        if 'origin' in detail:
            close(detail['origin']['position'],position,'Wrong line origin')
            close(rotation(detail['origin']['orientation']),orientation,'Wrong fixed line orientation')
            close(detail['direction'],direction,'Wrong world line direction')
            require(r4 is not None and cap is not None,'Segment for empty chord')
            segment=[max(r4[0],cap[0]),min(r4[1],cap[1])]
            close(detail['segment'],segment,'Wrong R4/capture line clipping')
        for intervals in sets:validate_intervals(intervals,detail.get('segment'))
        return sets
    def density(self,u,details):
        x=self.raw(u);inside=np.linalg.norm(u)<=self.radius;g=self.gaussian_logs(u)
        uniform=math.log(self.alpha)-self.logvolume if inside else -math.inf
        baseline=np.logaddexp(uniform,math.log1p(-self.alpha)+logsumexp(g)) if self.alpha<1 else uniform
        if self.beta==0 or self.alpha==1:
            require(details.get('conditioning_disabled') is True,'Expected disabled conditioner')
            return float(baseline),float(baseline),0
        require([r['axis'] for r in details['axes']]==self.axes,'Missing or reordered density axes')
        close(details['raw_coordinates'],x,'Density raw coordinates mismatch')
        factors=np.zeros(len(g));fallbacks=0
        for a,detail in zip(self.axes,details['axes']):
            means,sigmas=self.conditional(x,a)
            for intervals in self.geometry_sets(detail,x,a):
                masses=interval_masses(intervals,means,sigmas);fallback=masses<=self.floor
                factors[fallback]+=1.;fallbacks+=int(fallback.sum())
                if contains(intervals,x[a]):factors[~fallback]+=1/masses[~fallback]
        correction=1-self.beta+self.beta*factors/(len(self.axes)*len(self.widths))
        positive=correction>0
        result=np.logaddexp(uniform,math.log1p(-self.alpha)+logsumexp(g[positive]+np.log(correction[positive])))
        require(details['fallback_component_branches']==fallbacks,'Full-mixture fallback decisions differ')
        require(details['component_branches']==len(g)*len(self.axes)*len(self.widths),'Incomplete mixture denominator')
        close(details['baseline_log_density'],baseline,'Internal baseline mismatch')
        if self.beta<1:require(result+2e-9>=baseline+math.log1p(-self.beta),'Pointwise baseline support bound violated')
        return float(result),float(baseline),fallbacks
    def direct_geometry(self,position,R):
        mobile=self.atoms@R.T+position;moving=cKDTree(mobile);minimum=[]
        reach=2*self.radii.max()+max(self.widths)
        for fixed,tree in zip(self.fixed_world,self.fixed_trees):
            pairs=moving.query_ball_tree(tree,reach+1e-9);best=math.inf
            for i,js in enumerate(pairs):
                if js:
                    js=np.asarray(js);gap=np.linalg.norm(fixed[js]-mobile[i],axis=1)-self.radii[i]-self.radii[js]
                    best=min(best,float(gap.min()))
            minimum.append(best)
        return min(minimum)>=0,[[minimum[i]<w for i in self.contact_indices] for w in self.widths]


def audit(directory):
    root=Path(directory).resolve();started=time.process_time();manifest=read(root/'manifest.json');summary=read(root/'summary.json')
    require(manifest['schema']=='contact-line-guide-audit-v1' and manifest['physical_jobs']==0,'Wrong audit scope')
    require(summary['complete'] is True and summary['manifest']==manifest,'Incomplete/mismatched summary')
    paths=[root/'manifest.json',root/'summary.json'];bindings={}
    for name,key in [('config.json','config_sha256'),('region.json','region_sha256'),('importance-guide.json','guide_sha256'),('shape.json','shape_sha256'),('source-bundle.json','source_bundle_sha256')]:
        p=root/'provenance'/name;require(sha(p)==manifest[key],'Changed provenance: '+name);paths.append(p)
    for name in ('samples','probes'):
        p=root/(name+'.jsonl');require(sha(p)==summary[name+'_sha256'],'Changed output: '+name);paths.append(p)
    archived_probes=[]
    if manifest['probes_sha256'] is not None:
        p=root/'provenance/probes.jsonl';require(sha(p)==manifest['probes_sha256'],'Changed probe input');paths.append(p)
        archived_probes=[json.loads(s) for s in p.read_text().splitlines()]
    bindings={str(p):sha(p) for p in paths}
    region=read(root/'provenance/region.json');guide=read(root/'provenance/importance-guide.json')
    config=read(root/'provenance/config.json');shape=read(root/'provenance/shape.json')
    require(guide['region_sha256']==manifest['region_sha256'] and region['shape_sha256']==manifest['shape_sha256'],'Guide/region/shape binding mismatch')
    require(config['fixed_poses']==region.get('physical_fixed_neighbors',[region['fixed_neighbor']]),'Changed physical neighbors')
    for a,b in [('capture_center','capture_center'),('capture_radius','capture_radius'),('depletant_radius','depletant_radius'),('activity','reservoir_density'),('physical_metric','metadata')]:
        require(region[a]==config[b],'Changed physical field: '+a)
    recon=Reconstructor(region,guide,config,shape)
    close(manifest['log_latent_ball_volume'],recon.logvolume,'Ball volume mismatch');close(manifest['latent_radius'],recon.radius,'Radius mismatch')
    rows={name:[json.loads(s) for s in (root/(name+'.jsonl')).read_text().splitlines()] for name in ('samples','probes')}
    require(len(rows['samples'])==summary['samples']==manifest['samples'],'Fresh allocation missing')
    require([r['id'] for r in rows['samples']]==list(range(manifest['samples'])),'Fresh draw IDs changed')
    require(len(rows['probes'])==summary['probes']==len(archived_probes),'Probe allocation missing')
    require([r['id'] for r in rows['probes']]==[p['id'] for p in archived_probes],'Probe identities changed')
    for r,p in zip(rows['probes'],archived_probes):require(r['latent']==p['latent'],'Probe pose changed')
    maxima=Counter();counts=Counter();audited=[]
    for name,records in rows.items():
        for row in records:
            require(row['kind']==('fresh' if name=='samples' else 'probe'),'Wrong row kind')
            u=np.asarray(row['latent'],float);require(u.shape==(6,) and finite(u),'Invalid latent pose')
            x,position,R,jac=recon.decode(u)
            close(row['raw_coordinates'],x,'Raw coordinate reconstruction failed')
            close(row['pose']['position'],position,'Pose translation reconstruction failed')
            close(rotation(row['pose']['orientation']),R,'Pose orientation reconstruction failed')
            close(row['log_physical_jacobian'],jac,'Physical Jacobian reconstruction failed')
            radial=float(np.linalg.norm(u));close(row['latent_radius'],radial,'Latent radius differs')
            require(row['shell_valid']==(radial<=recon.radius),'R4 target flag differs')
            capture=np.linalg.norm(position-np.asarray(config['capture_center']))<=config['capture_radius']
            require(row['capture_valid']==capture,'Center capture flag differs')
            q,baseline,fallbacks=recon.density(u,row['density_details'])
            close(row['log_proposal_density'],q,'Complete proposal density reconstruction failed',atol=2e-7,rtol=1e-11)
            close(row['baseline_log_density'],baseline,'Baseline Gaussian density reconstruction failed')
            maxima['log_density_absolute']=max(maxima['log_density_absolute'],abs(q-row['log_proposal_density']))
            maxima['baseline_log_density_absolute']=max(maxima['baseline_log_density_absolute'],abs(baseline-row['baseline_log_density']))
            maxima['jacobian_absolute']=max(maxima['jacobian_absolute'],abs(jac-row['log_physical_jacobian']))
            hard,contacts=recon.direct_geometry(position,R)
            require(hard==row['hard_valid'],'Independent atom hard predicate differs')
            require(contacts==row['width_contacts'],'Independent atom contact predicate differs')
            draw=row['draw']
            if row['kind']=='fresh':
                counts.update(attempted=1,hard_capture_shell_valid=int(hard and capture and row['shell_valid']),conditioned_draws=int(draw['conditional']),fallback_draws=int(draw.get('fallback',False)))
            else:require(draw is None,'Probe unexpectedly sampled')
            if draw and draw['conditional']:
                a=draw['axis'];k=draw['component'];wi=draw['width_index']
                require(a in recon.axes and 0<=k<len(recon.weights) and 0<=wi<len(recon.widths),'Invalid selected label')
                require(draw['width_A']==recon.widths[wi],'Selected width mismatch')
                means,sigmas=recon.conditional(x,a);sets=recon.geometry_sets(draw['geometry'],x,a)
                mass=float(interval_masses(sets[wi],means[k],sigmas[k]))
                close(draw['conditional_mean'],means[k],'Selected conditional mean mismatch')
                close(draw['conditional_sigma'],sigmas[k],'Selected conditional sigma mismatch')
                close(draw['conditional_mass'],mass,'Selected conditional mass mismatch',atol=2e-15,rtol=2e-7)
                require(draw['fallback']==(mass<=recon.floor),'Selected fallback decision differs')
                if not draw['fallback']:
                    require(hard and capture and row['shell_valid'] and all(contacts[wi]),'Successful conditioned draw violates its constraints')
                    require(contains(sets[wi],x[a]),'Successful draw outside chosen contact intervals')
                    r=draw['selected_interval'];require(r in sets[wi],'Selected interval missing from geometry')
                    selected=float(interval_masses([r],means[k],sigmas[k]));close(draw['selected_interval_mass'],selected,'Selected interval probability differs',atol=2e-15,rtol=2e-7)
                    returned=draw['returned_raw_coordinate'];close(returned,x[a],'Returned raw translation changed')
                    cumulative=float(normal_masses((r['lower']-means[k])/sigmas[k],(returned-means[k])/sigmas[k]))/selected
                    error=abs(cumulative-draw['uniform_within_interval']);require(error<2e-6,'Independent conditional inverse-CDF failure')
                    maxima['inverse_cdf_probability_error']=max(maxima['inverse_cdf_probability_error'],error)
            audited.append(dict(kind=row['kind'],id=row['id'],log_proposal_density=q,baseline_log_density=baseline,
                log_physical_jacobian=jac,hard_valid=hard,shell_valid=bool(row['shell_valid']),capture_valid=bool(capture),
                width_contacts=contacts,fallback_component_branches=fallbacks))
    for key in ('hard_capture_shell_valid','conditioned_draws','fallback_draws'):require(counts[key]==summary[key],'Summary counter mismatch: '+key)
    maximum_backmap=max((r['backmap_error'] for r in rows['samples']),default=0.)
    close(summary['maximum_backmap_error'],maximum_backmap,'Maximum backmap receipt differs',atol=1e-15,rtol=1e-12)
    for name,key in [('draw_cpu_seconds','draw_cpu_seconds'),('density_cpu_seconds','density_cpu_seconds')]:
        close(summary[key],math.fsum(r[name] for r in rows['samples']),'CPU counter mismatch',atol=1e-9)
    for p,digest in bindings.items():require(sha(p)==digest,'Input changed during independent audit')
    return dict(schema='independent-contact-line-audit-v1',complete=True,root=str(root),scope='Independent deterministic proposal/output audit; no new physical samples or native classification',
        counts=dict(counts),probes=len(rows['probes']),maximum_errors=dict(maxima),rows=audited,
        input_sha256=bindings,executable_sha256=manifest['executable_sha256'],source_bundle_sha256=manifest['source_bundle_sha256'],
        script_sha256=sha(__file__),analysis_cpu_seconds=time.process_time()-started,
        conditional_parameters='Independent Schur complement with SciPy positive-definite solve; all components and declared axis/width labels.',
        interval_validation='Finite disjoint endpoint sets, raw line and R4/center-capture chords; KD-tree atom predicates at every recorded pose. Entire-line interval completeness is separately tested against brute-force geometry.',
        physical_measure='Lebesgue density in original latent u; physical Jacobian includes det(L0) and normalized SO(3) Haar factor; no regional truncation of fallback Gaussians.')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();result=audit(args.root);require(not args.out.exists(),'Refusing to overwrite audit receipt')
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('complete','counts','probes','maximum_errors','analysis_cpu_seconds')},indent=2))


if __name__=='__main__':main()
