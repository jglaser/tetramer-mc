#!/usr/bin/env python3
"""Independent feasibility-only Gaussian line guide; no protein jobs on import.

Every line is reconstructed from atom geometry. The earlier frozen audit module
supplies conditional Gaussian/Normal integration machinery without modification.
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
from scipy.linalg import solve_triangular
from scipy.special import logsumexp
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import analyze_contact_line_audit as normal

require,read,sha,close=normal.require,normal.read,normal.sha,normal.close
contains,interval_masses=normal.contains,normal.interval_masses


def interval(a,b,lc=True,uc=True):
    return dict(lower=float(a),upper=float(b),lower_closed=bool(lc),upper_closed=bool(uc))
def nonempty(i):return i['lower']<i['upper'] or (i['lower']==i['upper'] and i['lower_closed'] and i['upper_closed'])
def union(intervals):
    answer=[]
    for v in sorted((dict(i) for i in intervals if nonempty(i)),key=lambda i:(i['lower'],not i['lower_closed'])):
        if not answer:answer.append(v);continue
        p=answer[-1]
        if v['lower']<p['upper'] or (v['lower']==p['upper'] and (v['lower_closed'] or p['upper_closed'])):
            if v['upper']>p['upper']:p['upper']=v['upper'];p['upper_closed']=v['upper_closed']
            elif v['upper']==p['upper']:p['upper_closed']=p['upper_closed'] or v['upper_closed']
        else:answer.append(v)
    return answer
def subtract_closed_segment(segment,blocked):
    """Endpoint-aware difference, including isolated feasible tangencies."""
    lo,hi=segment;cursor=lo;closed=True;result=[]
    for b in union(blocked):
        require(lo<=b['lower']<=b['upper']<=hi,'Unclipped blocked interval')
        part=interval(cursor,b['lower'],closed,not b['lower_closed'])
        if nonempty(part):result.append(part)
        cursor=b['upper'];closed=not b['upper_closed']
    part=interval(cursor,hi,closed,True)
    if nonempty(part):result.append(part)
    normal.validate_intervals(result,segment)
    return result


def hard_free_intervals(shape,fixed_poses,origin,R,direction,segment,use_tree=True):
    """Independent projected KD-tree candidates and long-double leaf roots."""
    atoms=np.asarray([a['center'] for a in shape['atoms']]);radii=np.asarray([a['radius'] for a in shape['atoms']])
    moving=atoms@np.asarray(R).T+origin;d=np.asarray(direction);speed=np.linalg.norm(d)
    require(speed>0 and np.isfinite(speed),'Invalid translation direction')
    transverse=np.linalg.svd((d/speed)[None,:])[2][1:].T
    projected=(moving-origin)@transverse;blocked=[];candidates=0
    dl=np.asarray(d,np.longdouble);a=dl@dl
    for pose in fixed_poses:
        fixed=atoms@normal.rotation(pose['orientation']).T+pose['position']
        tree=cKDTree((fixed-origin)@transverse) if use_tree else None
        # Bound temporary pair arrays by 64 moving atoms. Vectorization changes
        # only arithmetic batching, never candidate radii or leaf predicates.
        for start in range(0,len(moving),64):
            end=min(start+64,len(moving))
            lists=tree.query_ball_point(projected[start:end],radii[start:end]+radii.max()+1e-9) if use_tree else [list(range(len(fixed))) for _ in range(start,end)]
            count=sum(map(len,lists));candidates+=count
            if not count:continue
            ii=np.repeat(np.arange(start,end),[len(js) for js in lists]);jj=np.concatenate(lists).astype(int)
            delta=np.asarray(fixed[jj],np.longdouble)-np.asarray(moving[ii],np.longdouble)
            center=(delta@dl)/a;nearest=delta-center[:,None]*dl
            radius=np.asarray(radii[ii],np.longdouble)+np.asarray(radii[jj],np.longdouble)
            residual=radius*radius-np.sum(nearest*nearest,axis=1);positive=residual>0
            if not positive.any():continue  # tangencies do not overlap cores
            half=np.sqrt(residual[positive]/a);lower=np.asarray(center[positive]-half,float);upper=np.asarray(center[positive]+half,float)
            lo=np.maximum(segment[0],lower);hi=np.minimum(segment[1],upper)
            lc=lo>lower;uc=hi<upper;keep=(lo<hi)|((lo==hi)&lc&uc)
            blocked.extend(interval(l,h,lclosed,uclosed) for l,h,lclosed,uclosed in zip(lo[keep],hi[keep],lc[keep],uc[keep]))
    core=union(blocked);free=subtract_closed_segment(segment,core)
    return dict(intervals=free,hard_overlap=core,projected_pair_candidates=candidates)


class Reconstructor(normal.Reconstructor):
    def __init__(self,region,guide,config,shape):
        require(guide['schema']=='defensive-hard-free-line-guide-v1','Wrong hard-free guide schema')
        require('contact_widths_A' not in guide and 'contact_neighbor_indices' not in guide,'Feasibility-only guide must not carry contact constraints')
        # Private compatibility adapter only initializes the frozen Gaussian
        # coordinate/Schur machinery. Neither sentinel contact field enters
        # geometry, density, target predicates or the external guide schema.
        adapted=copy.deepcopy(guide);adapted.update(schema='defensive-contact-line-guide-v1',contact_widths_A=[1.],contact_neighbor_indices=[])
        super().__init__(region,adapted,config,shape)
        self.guide=copy.deepcopy(guide);self.shape=copy.deepcopy(shape)

    def reconstruct_axis(self,u,axis,use_tree=True):
        require(axis in self.axes,'Unknown raw translation axis');x=self.raw(u);raw=x.copy();raw[axis]=0
        u0=solve_triangular(self.L0,raw-self.m0,lower=True);du=solve_triangular(self.L0,np.eye(6)[axis],lower=True)
        _,position,R,_=self.decode(u0);direction=self.Rf[:,axis]
        r4=self.chord(u0,du,self.radius)
        if r4 is None:return dict(axis=axis,intervals=[],empty_reason='no_R4_chord')
        cap=self.chord(position-np.asarray(self.config['capture_center']),direction,self.config['capture_radius'])
        if cap is None:return dict(axis=axis,intervals=[],empty_reason='no_capture_chord')
        segment=[max(r4[0],cap[0]),min(r4[1],cap[1])]
        if segment[0]>=segment[1]:return dict(axis=axis,intervals=[],empty_reason='disjoint_chords')
        geometry=hard_free_intervals(self.shape,self.config['fixed_poses'],position,R,direction,segment,use_tree)
        result=dict(axis=axis,segment=segment,origin=dict(position=position.tolist(),orientation=Rotation.from_matrix(R).as_quat()[[3,0,1,2]].tolist()),direction=direction.tolist(),**geometry)
        if not any(i['lower']<i['upper'] for i in geometry['intervals']):
            result['empty_reason']='no_positive_hard_free_length'
            # The law uses fallback on zero-measure sets; retain isolated points
            # separately for the strict-overlap geometry audit.
            result['zero_measure_feasible_points']=geometry['intervals'];result['intervals']=[]
        return result

    def density(self,u):
        u=np.asarray(u);x=self.raw(u);g=self.gaussian_logs(u)
        U=math.log(self.alpha)-self.logvolume if np.linalg.norm(u)<=self.radius else -math.inf
        baseline=float(np.logaddexp(U,math.log1p(-self.alpha)+logsumexp(g))) if self.alpha<1 else U
        if self.beta==0 or self.alpha==1:return dict(log_density=baseline,baseline_log_density=baseline,conditioning_disabled=True,axes=[],fallback_component_branches=0)
        factors=np.zeros(len(g));axes=[];fallbacks=0
        for axis in self.axes:
            geometry=self.reconstruct_axis(u,axis);means,sigmas=self.conditional(x,axis)
            masses=interval_masses(geometry['intervals'],means,sigmas);fallback=masses<=self.floor
            factors[fallback]+=1.;fallbacks+=int(fallback.sum())
            if contains(geometry['intervals'],x[axis]):factors[~fallback]+=1/masses[~fallback]
            axes.append(dict(**geometry,conditional_means=means.tolist(),conditional_sigmas=sigmas.tolist(),conditional_masses=masses.tolist(),component_fallbacks=fallback.tolist()))
        correction=1-self.beta+self.beta*factors/len(self.axes);positive=correction>0
        q=float(np.logaddexp(U,math.log1p(-self.alpha)+logsumexp(g[positive]+np.log(correction[positive]))))
        return dict(log_density=q,baseline_log_density=baseline,axes=axes,component_branches=len(g)*len(self.axes),fallback_component_branches=fallbacks,
            component_multipliers=correction.tolist())

    def hard_valid(self,position,R):
        mobile=self.atoms@R.T+position
        for fixed,tree in zip(self.fixed_world,self.fixed_trees):
            for i,point in enumerate(mobile):
                js=tree.query_ball_point(point,self.radii[i]+self.radii.max()+1e-9)
                if js and np.any(np.linalg.norm(fixed[js]-point,axis=1)<self.radii[i]+self.radii[js]):return False
        return True


def compare_axis(actual,expected):
    require(actual['axis']==expected['axis'],'Changed axis')
    require(actual.get('empty_reason')==expected.get('empty_reason'),'Different empty-line fallback')
    for key in ['segment','direction']:
        if key in actual:close(actual[key],expected[key],'Different line '+key)
    if 'origin' in actual:
        close(actual['origin']['position'],expected['origin']['position'],'Different line origin')
        close(normal.rotation(actual['origin']['orientation']),normal.rotation(expected['origin']['orientation']),'Different line orientation')
    got=actual.get('hard_free_intervals',[]);want=expected['intervals'];normal.validate_intervals(got,actual.get('segment'))
    require(len(got)==len(want),'Different whole-line hard-free topology')
    maximum=0.
    for a,b in zip(got,want):
        require(a['lower_closed']==b['lower_closed'] and a['upper_closed']==b['upper_closed'],'Different strict hard-overlap endpoint inclusion')
        close([a['lower'],a['upper']],[b['lower'],b['upper']],'Different hard-free interval',atol=1e-9,rtol=0)
        maximum=max(maximum,abs(a['lower']-b['lower']),abs(a['upper']-b['upper']))
    return maximum


def audit(directory):
    root=Path(directory).resolve();started=time.process_time();manifest=read(root/'manifest.json');summary=read(root/'summary.json')
    require(manifest['schema']=='hard-free-line-guide-audit-v1' and manifest['physical_jobs']==0,'Wrong audit scope')
    require(summary['complete'] and summary['manifest']==manifest,'Incomplete audit')
    require(not (root/'failure.json').exists(),'A failed query must not be hidden by a complete receipt')
    bindings={str(p):sha(p) for p in [root/'manifest.json',root/'summary.json',Path(__file__).resolve(),Path(normal.__file__).resolve()]}
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
    return dict(schema='independent-hard-free-line-audit-v1',complete=True,root=str(root),rows=audited,counts=dict(counts),maximum_errors=dict(maxima),input_sha256=bindings,
        probes=len(probes),analysis_cpu_seconds=time.process_time()-started,new_pose_draws=0,new_Poisson_clouds=0,
        executable_sha256=manifest['executable_sha256'],source_bundle_sha256=manifest['source_bundle_sha256'],
        scope='Independent entire-line geometry, complete density and draw-trace audit. No new poses, target weights or native labels.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);a=parser.parse_args()
    require(not a.out.exists(),'New receipt path required');result=audit(a.root);a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(a.out)
