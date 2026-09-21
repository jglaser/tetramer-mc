"""Independent density for angular-marginal/member-shell importance proposals.

The full latent ball remains the physical integration domain. Shell proposals
may land outside that ball or in the hard core; those draws are never retried.
This module uses NumPy/SciPy linear algebra, independently of the Rust sampler.
"""
from __future__ import annotations
import math
import numpy as np
from scipy.linalg import solve_triangular
from scipy.spatial.transform import Rotation
from scipy.special import logsumexp

SCHEMA='defensive-entry-shell-guide-v1'
POPULATION_SCHEMA='importance-latent-region-normalizer-v2'
PROPOSAL_KIND='orientation-marginal-member-shell-mixture'

def require(ok,message):
    if not ok:raise ValueError(message)
def finite_number(x):return type(x)in(int,float)and math.isfinite(x)


class EntryShellGuide:
    branch='entry-shell'
    schema=SCHEMA
    population_schema=POPULATION_SCHEMA
    proposal_kind=PROPOSAL_KIND

    def __init__(self,guide,region,region_sha256):
        require(set(guide)=={'schema','region_sha256','defensive_uniform_shell_probability','entries'},'Unknown guide fields')
        require(guide['schema']==SCHEMA and guide['region_sha256']==region_sha256,'Wrong guide or region')
        require(isinstance(region_sha256,str)and len(region_sha256)==64
            and all(x in '0123456789abcdefABCDEF'for x in region_sha256),'Invalid region hash')
        self.alpha=guide['defensive_uniform_shell_probability']
        require(finite_number(self.alpha)and 0<self.alpha<=1,'A positive defensive floor is required')
        require(region.get('minimum_mahalanobis_radius',0.)==0.,'Angular marginal v1 requires a complete latent ball')
        self.radius=region['mahalanobis_radius'];require(finite_number(self.radius)and self.radius>0,'Invalid ball')
        self.log_volume=3*math.log(math.pi)+6*math.log(self.radius)-math.log(6)
        chart=region['gaussian_chart'];require(len(chart['weights'])==1 and chart['weights']==[1.]
            and chart['coordinate_convention']=='anchor-body-relative'and 'base_model'not in chart,'One ordinary chart required')
        self.ell=chart['angular_length'];require(finite_number(self.ell)and self.ell>0,'Invalid angular scale')
        self.mean=np.asarray(chart['means'][0],float);covariance=np.asarray(chart['covariances'][0],float)
        require(self.mean.shape==(6,)and covariance.shape==(6,6)and np.isfinite(self.mean).all()
            and np.isfinite(covariance).all()and np.allclose(covariance,covariance.T,rtol=1e-12,atol=1e-14),'Invalid chart')
        self.lower=np.linalg.cholesky(covariance);self.log_det=np.log(np.diag(self.lower)).sum()
        # Full marginal covariance, including the cross-block contributions of
        # all six latent coordinates. The lower-right Cholesky block is wrong.
        self.angular_lower=np.linalg.cholesky(covariance[3:,3:])
        self.angular_log_det=np.log(np.diag(self.angular_lower)).sum()
        self.anchor_t=np.asarray(chart['anchors'][0]['position'],float)
        self.anchor_r=np.asarray(chart['anchors'][0]['rotation'],float)
        fixed=region['fixed_neighbor'];q=np.asarray(fixed['orientation'],float)
        self.fixed_t=np.asarray(fixed['position'],float)
        require(q.shape==(4,)and np.isfinite(q).all()and abs(np.linalg.norm(q)-1)<1e-9,'Invalid fixed orientation')
        self.fixed_r=Rotation.from_quat(q[[1,2,3,0]]).as_matrix()
        require(self.anchor_t.shape==self.fixed_t.shape==(3,)and self.anchor_r.shape==(3,3)
            and np.isfinite(self.anchor_t).all()and np.isfinite(self.fixed_t).all()and np.isfinite(self.anchor_r).all()
            and np.allclose(self.anchor_r.T@self.anchor_r,np.eye(3),atol=1e-10)
            and abs(np.linalg.det(self.anchor_r)-1)<1e-10,'Invalid chart anchor')
        entries=guide['entries'];require(isinstance(entries,list)and bool(entries),'Missing shell entries')
        self.count=len(entries);weights=[];moving=[];target=[];inner=[];outer=[];volumes=[]
        for e in entries:
            require(set(e)=={'moving_member','target_world_member','inner_radius','outer_radius','weight'},'Unknown shell-entry fields')
            require(all(finite_number(e[k])for k in ('weight','inner_radius','outer_radius'))
                and e['weight']>0 and 0<=e['inner_radius']<e['outer_radius'],'Invalid shell mass or radii')
            a=np.asarray(e['moving_member'],float);b=np.asarray(e['target_world_member'],float)
            require(a.shape==b.shape==(3,)and np.isfinite(a).all()and np.isfinite(b).all(),'Invalid member coordinates')
            lo,hi=e['inner_radius'],e['outer_radius']
            volume=(4*math.pi/3)*(hi-lo)*(hi*hi+hi*lo+lo*lo)
            require(math.isfinite(volume)and volume>0,'Unrepresentable shell volume')
            weights.append(e['weight']);moving.append(a);target.append(b);inner.append(lo);outer.append(hi);volumes.append(volume)
        total=sum(weights);require(not weights or (math.isfinite(total)and total>0),'Invalid total shell mass')
        self.weights=np.asarray(weights)/total if weights else np.empty(0)
        self.moving=np.asarray(moving).reshape((-1,3));self.target=np.asarray(target).reshape((-1,3))
        self.inner=np.asarray(inner);self.outer=np.asarray(outer);self.volumes=np.asarray(volumes)
        require(np.all(self.weights>0),'Unrepresentable normalized shell weight')

    def coordinates(self,latents):
        u=np.asarray(latents,float);require(u.ndim==2 and u.shape[1]==6 and np.isfinite(u).all(),'Invalid latent points')
        return u@self.lower.T+self.mean

    def world(self,latents):
        x=self.coordinates(latents);c=x[:,3:]/self.ell
        cayley=Rotation.from_quat(np.column_stack((c,np.ones(len(c))))).as_matrix()
        rotation=self.fixed_r@cayley@self.anchor_r
        translation=(x[:,:3]+self.anchor_t)@self.fixed_r.T+self.fixed_t
        return translation,rotation,x

    def angular_log_density_raw(self,angular):
        residual=solve_triangular(self.angular_lower,(np.asarray(angular)-self.mean[3:]).T,lower=True).T
        remainder=self.radius**2-np.sum(residual**2,axis=1)
        result=np.full(len(remainder),-np.inf);inside=remainder>0
        result[inside]=math.log(8)-2*math.log(math.pi)-6*math.log(self.radius)-self.angular_log_det+1.5*np.log(remainder[inside])
        return result

    def entry_support(self,latents):
        t,r,_=self.world(latents)
        centers=self.target[None,:,:]-np.einsum('nij,kj->nki',r,self.moving)
        d2=np.sum((t[:,None,:]-centers)**2,axis=2)
        return (d2>=self.inner**2)&(d2<=self.outer**2)

    def log_density(self,latents,shell_valid,log_volume):
        u=np.asarray(latents,float);inside=np.asarray(shell_valid,bool)
        require(inside.shape==(len(u),)and abs(log_volume-self.log_volume)<1e-10,'Ball density measure differs')
        total=np.where(inside,math.log(self.alpha)-log_volume,-np.inf)
        if self.alpha<1.:
            x=self.coordinates(u);support=self.entry_support(u)
            terms=np.where(support,np.log(self.weights)-np.log(self.volumes),-np.inf)
            shell=self.log_det+self.angular_log_density_raw(x[:,3:])+logsumexp(terms,axis=1)
            total=np.logaddexp(total,math.log1p(-self.alpha)+shell)
        return total

    def draw_for_validation(self,rng,count):
        """Synthetic proposal probes only, independently generated with NumPy."""
        normal=rng.normal(size=(count,6));length=np.linalg.norm(normal,axis=1)
        require(np.all(length>0),'Invalid normal draw; no silent retry')
        u=normal/length[:,None]*(self.radius*rng.random(count)**(1/6))[:,None]
        if self.alpha==1.:return u,np.full(count,-1)
        chosen=np.where(rng.random(count)<self.alpha,-1,rng.choice(self.count,size=count,p=self.weights))
        indices=np.flatnonzero(chosen>=0)
        if not len(indices):return u,chosen
        t,r,x=self.world(u[indices]);k=chosen[indices]
        centers=self.target[k]-np.einsum('nij,nj->ni',r,self.moving[k])
        directions=rng.normal(size=(len(k),3));directions/=np.linalg.norm(directions,axis=1)[:,None]
        distance=np.cbrt(self.inner[k]**3+rng.random(len(k))*(self.outer[k]**3-self.inner[k]**3))
        t=centers+directions*distance[:,None]
        x[:,:3]=(t-self.fixed_t)@self.fixed_r-self.anchor_t
        u[indices]=solve_triangular(self.lower,(x-self.mean).T,lower=True).T
        return u,chosen
