#!/usr/bin/env python3
"""Extract a frozen chart region from full global importance draws.

Every original full-proposal weight and unconditional zero is retained.
Optional comparison is against an independently sampled uniform chart region.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from scipy.spatial.transform import Rotation

from analyze_basin_normalizers import moments, paired_noise, population
from prepare_deep_far_normalizer_atlas import registration
from prepare_smc_normalizer_atlas import Density, arrays, read, relative_poses, sha, write


def region_latent(poses, region):
    """Matrix inverse Cayley, with a stable quaternion fallback at the seam."""
    chart=region['gaussian_chart'];assert len(chart['weights'])==1
    relative=relative_poses(poses,region['fixed_neighbor'])
    t,_,r=arrays(relative)
    delta=r@np.asarray(chart['anchors'][0]['rotation']).T
    condition=np.linalg.cond(delta+np.eye(3))
    stable=condition<1e10
    cayley=np.empty((len(poses),3))
    skew=(delta[stable]-np.eye(3))@np.linalg.inv(delta[stable]+np.eye(3))
    cayley[stable]=skew[:,[2,0,1],[1,2,0]]
    if not stable.all():
        q=Rotation.from_matrix(delta[~stable]).as_quat()
        with np.errstate(divide='ignore',invalid='ignore'):
            cayley[~stable]=q[:,:3]/q[:,3,None]
    x=np.column_stack((t-chart['anchors'][0]['position'],chart['angular_length']*cayley))
    latent=np.full_like(x,np.inf);finite=np.isfinite(x).all(axis=1)
    latent[finite]=np.linalg.solve(np.linalg.cholesky(chart['covariances'][0]),(x[finite]-chart['means'][0]).T).T
    radius=np.linalg.norm(latent,axis=1)
    quaternion_radius=Density(chart).evaluate(relative)[1][:,0]
    finite=np.isfinite(radius)&np.isfinite(quaternion_radius)
    error=np.max(np.abs(radius[finite]-quaternion_radius[finite])/(1+quaternion_radius[finite])) if finite.any() else 0.
    assert error<2e-7
    assert np.array_equal(np.isfinite(radius),np.isfinite(quaternion_radius))
    return latent,radius,dict(matrix_radius_relative_error=float(error),quaternion_seam_fallback_count=int((~stable).sum()))


def summarize(logs,pairs,poplogs):
    estimate=moments(logs)
    estimate['paired_noise']=paired_noise(np.asarray(logs),np.asarray(pairs))
    estimate['independent_populations']=moments([-np.inf if x is None else x for x in poplogs])
    estimate['independent_populations']['logQ_values']=poplogs
    return estimate


def analyze(root,region_path,out,reference=None,covariance_scale=None):
    campaign=read(root/'manifest.json');region=read(region_path);cfg=read(root/'provenance/config.json')
    assert cfg['fixed_poses']==region.get('physical_fixed_neighbors',[region['fixed_neighbor']])
    assert region['fixed_neighbor'] in cfg['fixed_poses']
    for name in ('capture_center','capture_radius','depletant_radius'):assert cfg[name]==region[name]
    assert cfg['reservoir_density']==region['activity'] and cfg['metadata']==region['physical_metric']
    assert sha(root/'provenance/shape.json')==region['shape_sha256']==region['gaussian_chart']['shape_sha256']
    for name,digest in campaign['archive_sha256'].items():assert sha(root/'provenance'/name)==digest
    jobs=campaign['jobs'] if covariance_scale is None else [j for j in campaign['jobs'] if j['covariance_std_scale']==covariance_scale]
    assert jobs and len({j['covariance_std_scale'] for j in jobs})==1, 'Compare proposal scales separately using --covariance-scale'
    assert len({j['samples'] for j in jobs})==1, 'Equal population sizes required for the reported population-mean error'
    assert len({j['seed'] for j in jobs})==len(jobs)
    qmin=region['minimum_original_q'];qmax=region.get('maximum_original_q',math.inf)
    inner=region.get('minimum_mahalanobis_radius',0.);outer=region['mahalanobis_radius']
    assert 0<=inner<outer and 0<=qmin<=qmax
    names=('region','outside_region','total')
    logs={k:[] for k in names};pairs={k:[] for k in names};hard={k:[] for k in names}
    populations=[];sources={str(root/'manifest.json'):sha(root/'manifest.json'),str(region_path):sha(region_path)}
    maxerror=0.;seams=0;actual_count=0;top=[]
    for job in jobs:
        audited=population(job);rows=audited['rows'];runtime=audited['summary']['manifest']
        for key,name in [('model_sha256','model.json'),('config_sha256','config.json'),('shape_sha256','shape.json'),('executable_sha256','basin-normalizer')]:
            assert runtime[key]==campaign['archive_sha256'][name]
        assert runtime.get('proposal_anchor_index')==campaign.get('proposal_anchor_index')
        actual=np.asarray([i for i,r in enumerate(rows) if r.get('pose') is not None],dtype=int)
        radii=np.full(len(rows),np.inf);qs=np.full(len(rows),np.inf)
        if len(actual):
            poses=[rows[i]['pose'] for i in actual]
            _,r,diagnostic=region_latent(poses,region)
            radii[actual]=r;qs[actual]=registration(poses,region['physical_metric'])
            maxerror=max(maxerror,diagnostic['matrix_radius_relative_error']);seams+=diagnostic['quaternion_seam_fallback_count']
            actual_count+=len(actual)
        valid=np.asarray([r['hard_valid'] for r in rows])
        included=valid&(qs>=qmin)&(qs<=qmax)&(radii>=inner)&(radii<=outer)
        masks={'region':included,'outside_region':valid&~included,'total':valid}
        statistics={}
        for name,mask in masks.items():
            lp=np.asarray([r['log_importance_weight'] if hit else -np.inf for r,hit in zip(rows,mask)])
            hp=np.asarray([r['log_hard_weight'] if hit else -np.inf for r,hit in zip(rows,mask)])
            cp=np.asarray([[c['log_weight']-r['log_proposal_density'] for c in r['clouds']] if hit else [-np.inf,-np.inf] for r,hit in zip(rows,mask)])
            statistics[name]=moments(lp);statistics[name]['hard_region']=moments(hp)
            logs[name].extend(lp);pairs[name].extend(cp);hard[name].extend(hp)
        combined=[statistics[name]['logQ'] for name in ('region','outside_region') if statistics[name]['logQ'] is not None]
        if combined:assert abs(logsumexp(combined)-statistics['total']['logQ'])<1e-10
        for i in sorted(np.flatnonzero(included),key=lambda i:rows[i]['log_importance_weight'],reverse=True)[:12]:
            top.append(dict(population=job['id'],draw=int(i),pose=rows[i]['pose'],original_q=float(qs[i]),latent_radius=float(radii[i]),log_importance_weight=rows[i]['log_importance_weight']))
        sample_path=Path(job['directory'])/'samples.jsonl';sources[str(sample_path)]=sha(sample_path)
        sources[str(Path(job['directory'])/'summary.json')]=sha(Path(job['directory'])/'summary.json')
        populations.append(dict(id=job['id'],seed=job['seed'],estimates=statistics,proposal_audit=audited['proposal_audit'],samples=job['samples'],sampler_cpu_seconds=audited['summary']['sampler_cpu_seconds']))
    estimates={}
    for name in names:
        estimates[name]=summarize(logs[name],pairs[name],[p['estimates'][name]['logQ'] for p in populations])
        estimates[name]['hard_region']=moments(hard[name])
    comparison=None
    if reference is not None:
        ref_manifest=read(reference/'manifest.json');ref_region=reference/'provenance/region.json'
        assert sha(ref_region)==ref_manifest['region_sha256']==sha(region_path)
        ref=read(reference/'assessment/analysis.json')
        assert ref['region_sha256']==sha(region_path)
        assert set(p['seed'] for p in ref['populations']).isdisjoint(p['seed'] for p in populations)
        for p in ref['populations']:
            job=next(j for j in ref_manifest['jobs'] if j['id']==p['id'])
            assert sha(Path(job['directory'])/'samples.jsonl')==p['samples_sha256']
        a,b=estimates['region'],ref['estimate']
        comparison=dict(global_region=a,uniform_region=b)
        if a['logQ'] is not None and b['logQ'] is not None:
            offset=max(a['logQ'],b['logQ']);wa,wb=np.exp(a['logQ']-offset),np.exp(b['logQ']-offset)
            error=math.hypot(wa*a['relative_se'],wb*b['relative_se'])
            comparison.update(log_mass_difference=a['logQ']-b['logQ'],observed_standardized_difference=(wa-wb)/error if error else None,
                              qualification='Observed independent-sample uncertainties only; shared frozen chart is allowed, training observations are excluded. Agreement does not establish missing-region coverage.')
        sources[str(reference/'assessment/analysis.json')]=sha(reference/'assessment/analysis.json')
    result=dict(region_sha256=sha(region_path),covariance_scale=jobs[0]['covariance_std_scale'],estimates=estimates,populations=populations,comparison=comparison,
        every_actual_pose_reconstructed=actual_count,maximum_matrix_radius_relative_error=maxerror,quaternion_seam_fallback_count=seams,
        top_region_contributors=sorted(top,key=lambda r:r['log_importance_weight'],reverse=True)[:12],
        sampler_cpu_seconds=sum(p['sampler_cpu_seconds'] for p in populations),input_sha256=sources,
        scope='Same frozen finite chart region extracted from global full-density importance contributions. All N unconditional draws including invalid zeros remain in every denominator. Outside-region mass is reported explicitly; no claim of global convergence or independent accepted moves.')
    out.parent.mkdir(parents=True,exist_ok=True);write(out,result)
    print(json.dumps({k:result[k] for k in ('region_sha256','estimates','comparison','every_actual_pose_reconstructed','maximum_matrix_radius_relative_error')},indent=2))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--region',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--reference',type=Path)
    parser.add_argument('--covariance-scale',type=float,help='Select one pre-existing scale group; never pool distinct proposal arms')
    args=parser.parse_args();analyze(args.root.resolve(),args.region.resolve(),args.out.resolve(),args.reference.resolve() if args.reference else None,args.covariance_scale)
