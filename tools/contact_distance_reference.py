#!/usr/bin/env python3
"""Independent reference for the two-distance contact guide.

The geometry uses boundary-line intersections, rather than polygon clipping.
Gaussian conditioning uses a Schur complement, rather than reordered Cholesky.
No Poisson points, physical weights or native classifications are generated.
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
from scipy.special import logsumexp
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


_POLYGON_NORMALS=np.asarray([[-1,0],[1,0],[0,-1],[0,1],[-1,-1],[1,-1],[-1,1]],float)
_POLYGON_PAIRS=np.asarray([(i,j) for i in range(7) for j in range(i) if abs(np.linalg.det(_POLYGON_NORMALS[[i,j]]))>.5])
_POLYGON_INVERSES=np.linalg.inv(_POLYGON_NORMALS[_POLYGON_PAIRS])


def polygon(d1,d2,D,width):
    """Radius polygon, independently enumerating its boundary intersections.

    Coordinates are offsets from the lower rectangle corner to avoid area
    cancellation when the atomic radii are large relative to the contact gap.
    """
    require(finite([d1,d2,D,width]) and min(d1,d2,D,width)>=0,'Invalid radius polygon')
    # The implemented rectangle endpoints are binary64 d+w. Preserve those
    # endpoints explicitly, including in tiny areas near the fallback floor.
    w1=(d1+width)-d1;w2=(d2+width)-d2
    bounds=np.asarray([0,w1,0,w2,d1+d2-D,D-d1+d2,D+d1-d2])
    vertices=[]
    # A few ulps include the exact boundary intersections; they do not expand
    # endpoint density support, which is evaluated separately below.
    tol=32*np.finfo(float).eps*max(1.,abs(d1),abs(d2),abs(D),abs(width))
    candidates=np.einsum('kij,kj->ki',_POLYGON_INVERSES,bounds[_POLYGON_PAIRS])
    for point in candidates[np.all(candidates@_POLYGON_NORMALS.T<=bounds+tol,axis=1)]:
        if not any(np.linalg.norm(point-p)<=tol for p in vertices):vertices.append(point)
    if len(vertices)<3:return np.empty((0,2)),0.
    v=np.asarray(vertices);center=v.mean(axis=0)
    v=v[np.argsort(np.arctan2(v[:,1]-center[1],v[:,0]-center[0]))]
    shifted=v-v[0];cross=shifted[:-1,0]*shifted[1:,1]-shifted[:-1,1]*shifted[1:,0]
    area=float(abs(cross.sum())/2)
    if area==0:return np.empty((0,2)),0.
    return v+np.asarray([d1,d2]),area


def radii_supported(r1,r2,d1,d2,D,width):
    return (d1<=r1<=d1+width and d2<=r2<=d2+width and
        r1+r2>=D and abs(r1-r2)<=D)


def transverse_basis(c1,c2):
    delta=np.asarray(c2,float)-c1;D=float(np.linalg.norm(delta))
    require(D>0 and math.isfinite(D),'Coincident effective centers')
    e=delta/D;reference=np.eye(3)[np.argmin(abs(e))]
    e1=np.cross(e,reference);e1/=np.linalg.norm(e1);e2=np.cross(e,e1)
    return D,e,e1,e2


def circle_coordinates(r1,r2,D):
    require(min(r1,r2,D)>0 and abs(r1-r2)<=D<=r1+r2,'Impossible distance triangle')
    h=((r1-r2)*(r1+r2)+D*D)/(2*D)
    # Heron's product for the triangle altitude; no r1^2-h^2 cancellation.
    sides=sorted([r1,r2,D],reverse=True);a,b,c=sides
    factors=[a+(b+c),c-(a-b),c+(a-b),a+(b-c)]
    require(min(factors)>=0,'Negative triangle area')
    rho=math.sqrt(math.prod(factors))/(2*D)
    return h,rho


def cartesian_from_radii(c1,c2,r1,r2,phi):
    D,e,e1,e2=transverse_basis(c1,c2);h,rho=circle_coordinates(r1,r2,D)
    return np.asarray(c1)+h*e+rho*(e1*math.cos(phi)+e2*math.sin(phi))


def radii_from_cartesian(c1,c2,t):
    D,e,e1,e2=transverse_basis(c1,c2);delta=np.asarray(t)-c1
    r1=float(np.linalg.norm(delta));r2=float(np.linalg.norm(np.asarray(t)-c2))
    h=float(delta@e);v=delta-h*e;rho=float(np.linalg.norm(v))
    phi=math.atan2(v@e2,v@e1)%(2*math.pi)
    return dict(r1=r1,r2=r2,h=h,rho=rho,phi=phi,D=D)


def wrapped_cauchy_log_density(phi,mean,gamma):
    require(math.isfinite(gamma) and gamma>0,'Nonpositive wrapped-Cauchy scale')
    delta=np.asarray(phi)-mean;half=math.sinh(gamma/2)
    # sinh(gamma)/(cosh(gamma)-cos(delta)), using half angles.
    return np.log(math.sinh(gamma))-math.log(2*math.pi)-np.log(2*(half*half+np.sin(delta/2)**2))


def wrapped_cauchy_inverse(u,mean,gamma):
    u=np.asarray(u);require(np.all((u>0)&(u<1)),'Inverse needs open-unit uniforms')
    return (mean+2*np.arctan(math.tanh(gamma/2)*np.tan(math.pi*(u-.5))))%(2*math.pi)


def azimuth_parameters(c1,c2,rho,conditional_mean,conditional_covariance,controls):
    _,_,e1,e2=transverse_basis(c1,c2);v=np.asarray(conditional_mean)-c1
    projection=np.asarray([v@e1,v@e2]);length=float(np.linalg.norm(projection))
    if length<=controls['projection_floor']:
        return dict(localization_enabled=False,phi_mean=0.,gamma=controls['gamma_max'],projection_norm=length)
    phi=math.atan2(projection[1],projection[0])%(2*math.pi)
    tangent=-e1*math.sin(phi)+e2*math.cos(phi)
    variance=float(tangent@conditional_covariance@tangent)
    require(variance>0 and math.isfinite(variance),'Invalid conditional tangential variance')
    scale=math.sqrt(variance)/max(rho,controls['radius_floor'])
    gamma=min(controls['gamma_max'],max(controls['gamma_min'],scale))
    return dict(localization_enabled=True,phi_mean=phi,gamma=gamma,projection_norm=length)


def azimuth_log_density(phi,parameters,controls):
    b=controls['localized_probability'] if parameters['localization_enabled'] else 0.
    if b==0:return -math.log(2*math.pi)
    return float(np.logaddexp(math.log1p(-b)-math.log(2*math.pi),
        math.log(b)+wrapped_cauchy_log_density(phi,parameters['phi_mean'],parameters['gamma'])))


def normal_log_density(x,mean,covariance):
    L=np.linalg.cholesky(covariance);z=solve_triangular(L,np.asarray(x)-mean,lower=True)
    return float(-len(z)/2*math.log(2*math.pi)-np.log(np.diag(L)).sum()-z@z/2)


def gaussian_conditionals(mean,covariance,angular):
    mean=np.asarray(mean);s=np.asarray(covariance)
    regression=solve(s[3:,3:],s[3:,:3],assume_a='pos').T
    conditional_mean=mean[:3]+regression@(np.asarray(angular)-mean[3:])
    conditional_covariance=s[:3,:3]-regression@s[3:,:3]
    np.linalg.cholesky(conditional_covariance)
    return conditional_mean,conditional_covariance,normal_log_density(angular,mean[3:],s[3:,3:])


class Reconstructor:
    def __init__(self,region,guide,config,shape):
        self.region,self.guide,self.config=region,guide,config
        require(guide['schema']=='defensive-contact-distance-guide-v1','Wrong guide schema')
        require(region.get('minimum_mahalanobis_radius',0.)==0.,'Only complete R4 supported')
        chart=region['gaussian_chart'];require(len(chart['means'])==len(chart['anchors'])==1,'Expected one raw chart')
        self.m0=np.asarray(chart['means'][0]);self.L0=np.linalg.cholesky(np.asarray(chart['covariances'][0]))
        self.logdet=float(np.log(np.diag(self.L0)).sum());self.ell=chart['angular_length'];self.anchor=chart['anchors'][0]
        self.fixed=region['fixed_neighbor'];self.Rf=rotation(self.fixed['orientation']);self.radius=region['mahalanobis_radius']
        self.offset=np.asarray(self.fixed['position'])+self.Rf@np.asarray(self.anchor['position'])
        self.logvolume=3*math.log(math.pi)+6*math.log(self.radius)-math.log(6)
        self.alpha=guide['defensive_uniform_shell_probability'];self.beta=guide['conditional_probability']
        self.widths=guide['contact_widths_A'];self.Dfloor=guide['minimum_center_distance'];self.Afloor=guide['minimum_polygon_area']
        self.controls=guide['azimuth'];self.labels=guide['component_contact_pairs']
        self.contact_indices=guide['contact_neighbor_indices']
        require(0<self.alpha<=1 and 0<=self.beta<=1 and self.Dfloor>0 and self.Afloor>0,'Invalid mixture controls')
        require(self.widths and all(math.isfinite(w) and w>0 for w in self.widths),'Invalid contact widths')
        c=self.controls
        require(0<=c['localized_probability']<1 and c['radius_floor']>0 and c['projection_floor']>0 and 0<c['gamma_min']<=c['gamma_max']<=math.pi,'Invalid azimuth controls')
        comps=guide['gaussian_components'];weights=np.asarray([c['weight'] for c in comps])
        require(len(comps)>0 or self.alpha==1,'Empty nonuniform mixture')
        self.weights=weights/weights.sum() if len(weights) else weights
        require(np.all(self.weights>0),'Invalid Gaussian weights')
        require(len(self.labels)==len(comps),'Incomplete component label list')
        self.means=np.asarray([c['mean'] for c in comps]).reshape((-1,6));cov=np.asarray([c['covariance'] for c in comps]).reshape((-1,6,6))
        self.lowers=np.linalg.cholesky(cov)
        self.normalizers=-3*math.log(2*math.pi)-np.log(np.diagonal(self.lowers,axis1=1,axis2=2)).sum(axis=1)
        self.rawmeans=self.m0+self.means@self.L0.T
        self.rawcov=np.einsum('ij,kjl,ml->kim',self.L0,cov,self.L0)
        self.angular_lowers=np.linalg.cholesky(self.rawcov[:,3:,3:]);self.regression=[];self.conditional_covariance=[]
        for s in self.rawcov:
            regression=solve(s[3:,3:],s[3:,:3],assume_a='pos').T
            conditional=s[:3,:3]-regression@s[3:,:3];np.linalg.cholesky(conditional)
            self.regression.append(regression);self.conditional_covariance.append(conditional)
        self.regression=np.asarray(self.regression).reshape((-1,3,3));self.conditional_covariance=np.asarray(self.conditional_covariance).reshape((-1,3,3))
        self.world_covariance=np.einsum('ij,kjl,ml->kim',self.Rf,self.conditional_covariance,self.Rf)
        self.atoms=np.asarray([a['center'] for a in shape['atoms']]);self.radii=np.asarray([a['radius'] for a in shape['atoms']])
        self.fixed_world=[self.atoms@rotation(p['orientation']).T+p['position'] for p in config['fixed_poses']]
        self.fixed_trees=[cKDTree(p) for p in self.fixed_world]
        for pair in self.labels:
            require(len(pair)==2,'Expected two atom labels')
            require([p['neighbor_index'] for p in pair]==self.contact_indices and len(set(self.contact_indices))==2,'Two ordered distinct scaffold contacts required')
            for p in pair:
                require(0<=p['neighbor_index']<len(self.fixed_world) and 0<=p['moving_atom']<len(self.atoms) and 0<=p['fixed_atom']<len(self.atoms),'Invalid frozen atom label')

    def raw(self,u):return self.m0+self.L0@np.asarray(u)
    def gaussian_logs(self,u):
        if not len(self.weights):return np.empty(0)
        z=np.asarray([solve_triangular(L,np.asarray(u)-m,lower=True) for L,m in zip(self.lowers,self.means)])
        return np.log(self.weights)+self.normalizers-np.sum(z*z,axis=1)/2
    def decode(self,u):
        x=self.raw(u);c=x[3:]/self.ell
        Q=Rotation.from_quat(np.r_[c,1.]/math.sqrt(1+c@c)).as_matrix()
        R=self.Rf@Q@np.asarray(self.anchor['rotation']);p=self.offset+self.Rf@x[:3]
        J=self.logdet-3*math.log(self.ell)-2*math.log(math.pi)-2*math.log1p(c@c)
        return x,p,R,float(J)
    def conditionals(self,x):
        means=self.rawmeans[:,:3]+np.einsum('kij,kj->ki',self.regression,x[3:]-self.rawmeans[:,3:])
        angular=np.asarray([normal_log_density(x[3:],m[3:],s[3:,3:]) for m,s in zip(self.rawmeans,self.rawcov)])
        return means,self.offset+means@self.Rf.T,angular
    def centers(self,R,k):
        centers=[];distances=[]
        for p in self.labels[k]:
            centers.append(self.fixed_world[p['neighbor_index']][p['fixed_atom']]-R@self.atoms[p['moving_atom']])
            distances.append(self.radii[p['moving_atom']]+self.radii[p['fixed_atom']])
        return np.asarray(centers),np.asarray(distances)
    def density(self,u,details=False):
        x,position,R,_=self.decode(u);g=self.gaussian_logs(u)
        uniform=math.log(self.alpha)-self.logvolume if np.linalg.norm(u)<=self.radius else -math.inf
        baseline=float(np.logaddexp(uniform,math.log1p(-self.alpha)+logsumexp(g))) if self.alpha<1 else uniform
        if self.beta==0 or self.alpha==1:return dict(log_density=baseline,baseline_log_density=baseline,conditioning_disabled=True,branches=[])
        rawmeans,worldmeans,angular=self.conditionals(x);branches=[];h=[];fallbacks=0
        for k in range(len(self.weights)):
            centers,distances=self.centers(R,k);D=float(np.linalg.norm(centers[1]-centers[0]))
            coordinates=radii_from_cartesian(*centers,position) if D>self.Dfloor else None
            for wi,width in enumerate(self.widths):
                vertices,area=polygon(*distances,D,width) if D>self.Dfloor else (np.empty((0,2)),0.)
                fallback=D<=self.Dfloor or area<=self.Afloor
                branch=dict(component=k,width_index=wi,D=D,polygon_area=area,fallback=bool(fallback))
                if fallback:
                    logh=float(g[k]);fallbacks+=1
                    branch['fallback_reason']='center_distance' if D<=self.Dfloor else 'polygon_area'
                else:
                    inside=radii_supported(coordinates['r1'],coordinates['r2'],*distances,D,width)
                    parameters=azimuth_parameters(*centers,coordinates['rho'],worldmeans[k],self.world_covariance[k],self.controls)
                    phi_log=azimuth_log_density(coordinates['phi'],parameters,self.controls)
                    translation_log=math.log(D)-math.log(area)-math.log(coordinates['r1'])-math.log(coordinates['r2'])+phi_log if inside and coordinates['r1']>0 and coordinates['r2']>0 and coordinates['rho']>0 else -math.inf
                    logh=math.log(self.weights[k])+self.logdet+angular[k]+translation_log
                    branch.update(inside=bool(inside),angular_log_density=float(angular[k]),translation_log_density=float(translation_log),**parameters)
                    if details:branch.update(effective_centers=centers.tolist(),contact_distances=distances.tolist(),vertices=vertices.tolist(),coordinates=coordinates,conditional_world_mean=worldmeans[k].tolist(),conditional_raw_mean=rawmeans[k].tolist())
                branch['weighted_latent_log_density']=float(logh);branches.append(branch);h.append(logh)
        H=float(logsumexp(h)-math.log(len(self.widths)))
        G=float(logsumexp(g));weighted=[]
        if self.beta<1:weighted.append(math.log1p(-self.beta)+G)
        if self.beta>0:weighted.append(math.log(self.beta)+H)
        q=float(np.logaddexp(uniform,math.log1p(-self.alpha)+logsumexp(weighted)))
        if self.beta<1:require(q+1e-12>=baseline+math.log1p(-self.beta),'Defensive baseline bound failed')
        return dict(log_density=q,baseline_log_density=baseline,conditioned_log_density=H,component_branches=len(branches),fallback_component_branches=fallbacks,branches=branches)
    def direct_geometry(self,position,R):
        mobile=self.atoms@R.T+position;tree=cKDTree(mobile);minimum=[]
        reach=2*self.radii.max()+max(self.widths)
        for fixed,fixed_tree in zip(self.fixed_world,self.fixed_trees):
            candidates=tree.query_ball_tree(fixed_tree,reach+1e-9);best=math.inf
            for i,js in enumerate(candidates):
                if js:
                    js=np.asarray(js);best=min(best,float((np.linalg.norm(fixed[js]-mobile[i],axis=1)-self.radii[i]-self.radii[js]).min()))
            minimum.append(best)
        return min(minimum)>=0,[[minimum[i]<w for i in self.contact_indices] for w in self.widths]


def check_density_details(recon,u,result,recorded,maxima):
    if result.get('conditioning_disabled'):
        require(recorded.get('conditioning_disabled') is True,'Conditioner not disabled')
        return
    close(recorded['raw_coordinates'],recon.raw(u),'Density detail raw coordinates differ')
    close(recorded['baseline_log_density'],result['baseline_log_density'],'Density detail baseline differs')
    for name in ('component_branches','fallback_component_branches'):
        require(recorded[name]==result[name],'Density branch count differs: '+name)
    components=recorded['components'];require([c['component'] for c in components]==list(range(len(recon.weights))),'Missing/reordered density components')
    raw=recon.raw(u);_,worldmeans,angular=recon.conditionals(raw)
    for k,c in enumerate(components):
        close(c['angular_log_density'],angular[k],'Angular marginal differs',atol=2e-8)
        close(c['conditional_world_mean'],worldmeans[k],'Conditional translation mean differs')
        maxima['angular_log_density_absolute']=max(maxima['angular_log_density_absolute'],abs(c['angular_log_density']-angular[k]))
        maxima['conditional_mean_absolute']=max(maxima['conditional_mean_absolute'],float(np.max(abs(np.asarray(c['conditional_world_mean'])-worldmeans[k]))))
        require(len(c['widths'])==len(recon.widths),'Incomplete width mixture')
        for wi,width in enumerate(c['widths']):
            branch=result['branches'][k*len(recon.widths)+wi]
            require(width['width_index']==wi and width['width_A']==recon.widths[wi],'Reordered/changed widths')
            require(width['fallback']==branch['fallback'],'Component-specific fallback differs')
            if 'distance' in width:close(width['distance'],branch['D'],'Effective-center distance differs',atol=2e-10)
            if 'polygon_area' in width:
                close(width['polygon_area'],branch['polygon_area'],'Independent polygon area differs',atol=5e-14,rtol=2e-9)
                maxima['polygon_area_absolute']=max(maxima['polygon_area_absolute'],abs(width['polygon_area']-branch['polygon_area']))
            if branch['fallback']:require(width['fallback_reason']==branch['fallback_reason'],'Fallback reason differs')
            else:
                expected=branch['translation_log_density'];actual=width['translation_log_density']
                if actual is None:require(expected==-math.inf,'Missing positive translation density')
                else:
                    close(actual,expected,'Conditional translation density differs',atol=2e-7,rtol=1e-10)
                    maxima['translation_log_density_absolute']=max(maxima['translation_log_density_absolute'],abs(actual-expected))


def check_draw(recon,row,result,maxima):
    draw=row['draw']
    if not draw or not draw['conditional']:return dict(radius_uniforms_replayed=False)
    k=draw['component'];wi=draw['width_index'];require(0<=k<len(recon.weights) and 0<=wi<len(recon.widths),'Invalid selected label')
    require(draw['width_A']==recon.widths[wi],'Selected width changed')
    original=np.asarray(draw['original_latent']);raw,position,R,_=recon.decode(row['latent'])
    oldraw,_,oldR,_=recon.decode(original)
    close(draw['retained_raw_angles'],oldraw[3:],'Recorded retained angles differ from original')
    close(raw[3:],oldraw[3:],'Conditioning changed angular draw',atol=2e-11,rtol=1e-12)
    close(R,oldR,'Conditioning changed orientation',atol=2e-11,rtol=1e-12)
    maxima['retained_angles_absolute']=max(maxima['retained_angles_absolute'],float(np.max(abs(raw[3:]-oldraw[3:]))))
    branch=result['branches'][k*len(recon.widths)+wi]
    require(draw['fallback']==branch['fallback'],'Draw fallback differs from full-density branch')
    if draw['fallback']:
        require(draw['fallback_reason']==branch['fallback_reason'],'Draw fallback reason differs')
        require(row['latent']==draw['original_latent'],'Fallback did not retain complete original draw')
        return dict(radius_uniforms_replayed=False)
    centers,distances=recon.centers(R,k);coordinates=radii_from_cartesian(*centers,position)
    close(draw['distance'],coordinates['D'],'Selected center distance differs')
    close(draw['radii'],[coordinates['r1'],coordinates['r2']],'Selected radius inverse differs',atol=2e-9)
    require(radii_supported(*draw['radii'],*distances,coordinates['D'],recon.widths[wi]),'Selected radii outside complete polygon')
    vertices,area=polygon(*distances,coordinates['D'],recon.widths[wi]);close(draw['polygon_area'],area,'Selected polygon area differs',atol=5e-14,rtol=2e-9)
    D,e,_,_=transverse_basis(*centers);h,rho=circle_coordinates(*draw['radii'],D)
    close(draw['circle_center'],centers[0]+h*e,'Selected circle center differs')
    close(draw['circle_radius'],rho,'Selected circle radius differs',atol=2e-9)
    reconstructed=cartesian_from_radii(*centers,*draw['radii'],draw['phi'])
    close(draw['world_translation'],reconstructed,'Generated translation reconstruction differs')
    close(position,reconstructed,'Returned latent changed generated translation')
    _,means,angular=recon.conditionals(raw);parameters=azimuth_parameters(*centers,rho,means[k],recon.world_covariance[k],recon.controls)
    close(draw['angular_log_density'],angular[k],'Selected angular density differs')
    close(draw['conditional_world_mean'],means[k],'Selected conditional world mean differs')
    close(draw['azimuth_log_density'],azimuth_log_density(draw['phi'],parameters,recon.controls),'Selected full azimuth mixture density differs',atol=2e-8)
    uniforms=np.asarray(draw['azimuth_uniforms']);require(uniforms.shape==(2,) and np.all((uniforms>0)&(uniforms<1)),'Invalid azimuth uniforms')
    b=recon.controls['localized_probability'] if parameters['localization_enabled'] else 0.
    phi=float(wrapped_cauchy_inverse(uniforms[1],parameters['phi_mean'],parameters['gamma'])) if uniforms[0]<b else 2*math.pi*uniforms[1]
    error=abs((draw['phi']-phi+math.pi)%(2*math.pi)-math.pi);require(error<2e-8,'Azimuth inverse sampling differs')
    maxima['azimuth_inverse_absolute']=max(maxima['azimuth_inverse_absolute'],error)
    ru=np.asarray(draw['radius_uniforms']);require(ru.shape==(3,) and np.all((ru>0)&(ru<1)),'Invalid radius uniforms')
    replayed=False
    if 'polygon_vertices' in draw:
        stored=np.asarray(draw['polygon_vertices']);require(stored.ndim==2 and stored.shape[1]==2,'Invalid polygon vertices')
        require(all(min(np.linalg.norm(v-p) for p in vertices)<2e-10 for v in stored),'Stored polygon adds vertices')
        require(all(min(np.linalg.norm(v-p) for p in stored)<2e-10 for v in vertices),'Stored polygon omits vertices')
        a=stored[1:-1]-stored[0];b=stored[2:]-stored[0]
        areas=(a[:,0]*b[:,1]-a[:,1]*b[:,0])/2
        require(np.all(areas>=-1e-15),'Non-convex/reversed stored polygon')
        close(areas.sum(),area,'Stored fan area differs',atol=5e-14,rtol=2e-9)
        selected=int(np.searchsorted(np.cumsum(areas),ru[0]*areas.sum(),side='right'))
        require(selected<len(areas) and areas[selected]>0,'Invalid sampled fan triangle')
        s=math.sqrt(ru[1]);weights=np.asarray([1-s,s*(1-ru[2]),s*ru[2]])
        sampled=weights@stored[[0,selected+1,selected+2]]
        close(draw['radii'],sampled,'Polygon uniform sampling differs',atol=2e-10)
        maxima['radius_sample_absolute']=max(maxima['radius_sample_absolute'],float(np.max(abs(sampled-draw['radii']))))
        replayed=True
    return dict(radius_uniforms_replayed=replayed)


def audit(directory):
    root=Path(directory).resolve();started=time.process_time();manifest=read(root/'manifest.json');summary=read(root/'summary.json')
    require(manifest['schema']=='contact-distance-guide-audit-v1' and manifest['physical_jobs']==0,'Wrong audit scope')
    require(summary['complete'] is True and summary['manifest']==manifest,'Incomplete/mismatched summary')
    paths=[root/'manifest.json',root/'summary.json']
    for name,key in [('config.json','config_sha256'),('region.json','region_sha256'),('importance-guide.json','guide_sha256'),('shape.json','shape_sha256'),('source-bundle.json','source_bundle_sha256')]:
        p=root/'provenance'/name;require(sha(p)==manifest[key],'Changed provenance: '+name);paths.append(p)
    for name in ('samples','probes'):
        p=root/(name+'.jsonl');require(sha(p)==summary[name+'_sha256'],'Changed output: '+name);paths.append(p)
    archived_probes=[]
    if manifest['probes_sha256'] is not None:
        p=root/'provenance/probes.jsonl';require(sha(p)==manifest['probes_sha256'],'Changed probe input');paths.append(p)
        archived_probes=[json.loads(s) for s in p.read_text().splitlines()]
    bindings={str(p):sha(p) for p in paths}
    region=read(root/'provenance/region.json');guide=read(root/'provenance/importance-guide.json');config=read(root/'provenance/config.json');shape=read(root/'provenance/shape.json')
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
    for r,p in zip(rows['probes'],archived_probes):require(r['latent']==p['latent'],'Probe latent changed')
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
            require(row['shell_valid']==(radial<=recon.radius),'R4 flag differs')
            capture=np.linalg.norm(position-np.asarray(config['capture_center']))<=config['capture_radius']
            require(row['capture_valid']==capture,'Center capture flag differs')
            result=recon.density(u);q=result['log_density'];baseline=result['baseline_log_density']
            close(row['log_proposal_density'],q,'Full mixture density differs',atol=2e-7,rtol=1e-11)
            close(row['baseline_log_density'],baseline,'Baseline mixture density differs')
            check_density_details(recon,u,result,row['density_details'],maxima)
            maxima['log_density_absolute']=max(maxima['log_density_absolute'],abs(q-row['log_proposal_density']))
            maxima['baseline_log_density_absolute']=max(maxima['baseline_log_density_absolute'],abs(baseline-row['baseline_log_density']))
            maxima['jacobian_absolute']=max(maxima['jacobian_absolute'],abs(jac-row['log_physical_jacobian']))
            hard,contacts=recon.direct_geometry(position,R)
            require(hard==row['hard_valid'],'Independent atom hard predicate differs')
            require(contacts==row['width_contacts'],'Independent atom contact predicate differs')
            draw=row['draw'];replay=dict(radius_uniforms_replayed=False)
            if row['kind']=='fresh':
                counts.update(attempted=1,hard_capture_shell_valid=int(hard and capture and row['shell_valid']),conditioned_draws=int(draw['conditional']),fallback_draws=int(draw.get('fallback',False)))
                replay=check_draw(recon,row,result,maxima)
                counts['radius_uniforms_replayed']+=int(replay['radius_uniforms_replayed'])
            else:require(draw is None,'Probe unexpectedly sampled')
            audited.append(dict(kind=row['kind'],id=row['id'],log_proposal_density=q,baseline_log_density=baseline,
                log_physical_jacobian=jac,hard_valid=hard,shell_valid=bool(row['shell_valid']),capture_valid=bool(capture),
                width_contacts=contacts,fallback_component_branches=result.get('fallback_component_branches',0),**replay))
    for key in ('hard_capture_shell_valid','conditioned_draws','fallback_draws'):require(counts[key]==summary[key],'Summary counter differs: '+key)
    maximum_backmap=max((r['backmap_error'] for r in rows['samples']),default=0.)
    close(summary['maximum_backmap_error'],maximum_backmap,'Backmap receipt differs',atol=1e-15,rtol=1e-12)
    for key in ('draw_cpu_seconds','density_cpu_seconds'):close(summary[key],math.fsum(r[key] for r in rows['samples']),'CPU counter differs',atol=1e-9)
    for p,digest in bindings.items():require(sha(p)==digest,'Input changed during independent audit')
    return dict(schema='independent-contact-distance-audit-v1',complete=True,root=str(root),scope='Independent deterministic proposal/output audit; no physical samples or native classification',
        counts=dict(counts),probes=len(rows['probes']),maximum_errors=dict(maxima),rows=audited,input_sha256=bindings,
        executable_sha256=manifest['executable_sha256'],source_bundle_sha256=manifest['source_bundle_sha256'],
        script_sha256=sha(__file__),analysis_cpu_seconds=time.process_time()-started,
        conditional_parameters='Independent 3+3 Gaussian Schur complement, all components and widths; normalized angular marginal preserved before physical endpoint selection.',
        geometry_validation='Independent radius polygons from boundary intersections; distances/azimuth inverse, Jacobian analytic and finite-difference tests, retained original angular coordinates, atom KD-tree hard/contact predicates at every recorded pose.',
        physical_measure='Latent Lebesgue density, det(L0) in H; physical J includes normalized Haar factor. No R4, capture or whole-union conditioning beyond unconditional indicator zeros.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();result=audit(args.root);require(not args.out.exists(),'Refusing to overwrite audit receipt')
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('complete','counts','probes','maximum_errors','analysis_cpu_seconds')},indent=2))


if __name__=='__main__':main()
