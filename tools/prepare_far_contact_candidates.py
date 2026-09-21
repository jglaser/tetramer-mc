#!/usr/bin/env python3
"""Freeze historical AB far-contact discovery rows without physical resampling."""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse
import copy
import hashlib
import heapq
import json
import math
from pathlib import Path
import shutil

import numpy as np
from scipy.special import logsumexp
from analyze_global_fixed_region import region_latent
from analyze_native_region_reference import native_q
from prepare_native_confirmation_atlas import AtomUnionAudit
from prepare_peak_neighborhood import geometry_model
from prepare_smc_normalizer_atlas import Density,arrays,relative_poses
from run_shoulder_mis_campaign import local_dependencies

ROOT=Path(__file__).resolve().parents[1]
PREP=ROOT/'runs/ab-competing-frozen-atlas-20260920'
CAMPAIGNS=[ROOT/'runs/ab-global-contact-4x8192-l64-20260920',ROOT/'runs/ab-global-widths-2-4-4x16384-l64-20260920']
PER_POPULATION=8


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,a):Path(p).write_text(json.dumps(a,indent=2,allow_nan=False)+'\n')
def require(x,message):
    if not x:raise ValueError(message)


def select_rows(rows,count):
    """Deterministic top importance rows within one fixed proposal population."""
    require(type(count) is int and count>0,'positive count required')
    heap=[]
    for row in rows:
        if not row['hard_valid']:continue
        require(row['capture_valid'] and row['q'] is not None,'invalid positive row')
        if row['q']<5:continue
        weight=row['log_importance_weight']
        require(weight is not None and math.isfinite(weight),'finite original weight required')
        entry=(weight,-row['draw'],row)
        if len(heap)<count:heapq.heappush(heap,entry)
        elif entry[:2]>heap[0][:2]:heapq.heapreplace(heap,entry)
    return [entry[2] for entry in sorted(heap,key=lambda e:e[:2],reverse=True)]


def member_distances(poses,members):
    t,_,rot=arrays(poses)
    world=np.einsum('nij,mj->nmi',rot,np.asarray(members))+t[:,None,:]
    return np.sqrt(np.mean(np.sum((world[:,None]-world[None,:])**2,axis=-1),axis=-1))


def prepare(out):
    require(not out.exists(),'fresh output required')
    cfg=read(PREP/'config.json');region=read(PREP/'region-competitor-r3.json');model=read(PREP/'model-mixture.json');report=read(PREP/'report.json')
    for name,h in report['outputs'].items():require(sha(PREP/name)==h,f'frozen source changed: {name}')
    require(len(cfg['fixed_poses'])==2 and cfg['depletant_radius']==1.5 and cfg['reservoir_density']==.035,'AB physical model changed')
    require(cfg['capture_radius']==18. and cfg['metadata']['member_error_scale']==2. and cfg['metadata']['angle_error_scale_deg']==15.,'physical domain/metric changed')
    out.mkdir(parents=True);(out/'provenance').mkdir()
    protocol={'selection':'Eight largest original q>=5 importance rows per population within each historical width; lower draw index breaks exact ties. All current source populations included.',
              'per_population':PER_POPULATION,'selected_peak_rule':'Highest original importance weight among width4 candidates (all width4 populations have equal N).',
              'scope':'Geometry discovery only. No candidate weight is an equilibrium mixture weight; no full-far estimate is created or pooled.',
              'source_preparation_report_sha256':sha(PREP/'report.json')}
    write(out/'protocol.json',protocol)
    candidates=[];sources={str(PREP/'report.json'):sha(PREP/'report.json')};campaign_records=[]
    for root in CAMPAIGNS:
        master=read(root/'manifest.json');analysis=read(root/'assessment/analysis.json');require(not analysis['pending'],'historical campaign incomplete')
        sources[str(root/'manifest.json')]=sha(root/'manifest.json');sources[str(root/'assessment/analysis.json')]=sha(root/'assessment/analysis.json')
        for name,h in master['archive_sha256'].items():require(sha(root/'provenance'/name)==h,f'campaign archive changed {name}')
        current_cfg=read(root/'provenance/config.json')
        for key in ['fixed_poses','capture_center','capture_radius','depletant_radius','reservoir_density','metadata']:
            require(cfg[key]==current_cfg[key],f'physical signature differs: {key}')
        require(master['proposal_anchor_index']==0 and master['archive_sha256']['model.json']==sha(PREP/'model-mixture.json'),'proposal identity differs')
        for job in master['jobs']:
            path=Path(job['directory'])/'samples.jsonl';manifest_path=path.parent/'manifest.json';manifest=read(manifest_path)
            require(manifest['seed']==job['seed'] and manifest['samples']==job['samples'],'runtime seed or budget differs')
            require(manifest['activity']==.035 and manifest['lambda']==64*.035 and manifest['cloud_replicates']==2,'runtime cloud law changed')
            require(manifest['model_sha256']==sha(PREP/'model-mixture.json') and manifest['shape_sha256']==region['shape_sha256'],'runtime physical input changed')
            digest=hashlib.sha256();num=0
            def stream():
                nonlocal num
                with path.open('rb') as handle:
                    for line in handle:
                        digest.update(line);row=json.loads(line);require(row['draw']==num,'draw ordering');num+=1;yield row
            selected=select_rows(stream(),PER_POPULATION)
            require(num==job['samples'] and digest.hexdigest()==analysis['provenance'][str(path)],'raw source changed')
            require(len(selected)==PER_POPULATION,'not enough far observations for declared panel')
            sources[str(path)]=digest.hexdigest();sources[str(manifest_path)]=sha(manifest_path)
            for rank,row in enumerate(selected):
                candidates.append({'pose':row['pose'],'q':row['q'],'seed':job['seed'],'draw':row['draw'],
                    'population':job['id'],'width':job['covariance_std_scale'],'within_population_rank':rank+1,
                    'source_samples_path':str(path),'source_samples_sha256':digest.hexdigest(),
                    'source_manifest_path':str(manifest_path),'source_manifest_sha256':sha(manifest_path),
                    'source_config_path':str(root/'provenance/config.json'),'source_config_sha256':sha(root/'provenance/config.json'),
                    'unconditional_population_draws':job['samples'],
                    'original_log_importance_weight':row['log_importance_weight'],
                    'original_row':row})
        campaign_records.append({'root':str(root),'far_statistics_by_width':{w:g['estimates']['distant'] for w,g in analysis['groups'].items()},
                                 'unconditional_draws_by_width':{w:g['unconditional_draws'] for w,g in analysis['groups'].items()}})
    poses=[r['pose'] for r in candidates];rel=relative_poses(poses,cfg['fixed_poses'][0]);_,radii,diagnostic=region_latent(poses,region)
    widths=[1.,2.,4.];log_densities={}
    for width in widths:
        widened=copy.deepcopy(model);widened['covariances']=(np.asarray(model['covariances'])*width**2).tolist()
        logG=Density(widened).evaluate(rel)[0]
        log_densities[width]=np.logaddexp(math.log(.9)+logG,math.log(.1)-3*math.log(36.))
    atom=AtomUnionAudit(read(cfg['shape']),cfg['fixed_poses'])
    for i,row in enumerate(candidates):
        q=native_q(cfg['metadata'],row['pose']);require(abs(q-row['q'])<1e-10 and q>=5,'selected original q mismatch')
        require(abs(log_densities[row['width']][i]-row['original_row']['log_proposal_density'])<2e-9,'selected full proposal density mismatch')
        clouds=row['original_row']['clouds'];require(len(clouds)==2,'wrong selected cloud count')
        for cloud in clouds:
            expected=.035*cloud['lower_volume']+cloud['overlap_points']*math.log1p(1/64)
            require(abs(expected-cloud['log_weight'])<1e-10,'selected Poisson factor mismatch')
        logf=float(logsumexp([c['log_weight'] for c in clouds])-math.log(2))
        require(abs(logf-row['original_row']['log_proposal_density']-row['original_log_importance_weight'])<1e-10,'selected numerator denominator mismatch')
        gaps=atom.gaps(row['pose']);require(min(gaps)>=0,'selected pose clashes physical AB')
        row.update(old_competitor_radius=float(radii[i]),minimum_AB_gaps_A=gaps,
                   depletion_contact_neighbors=[gap<3 for gap in gaps],
                   full_proposal_log_density_by_width={str(w):float(log_densities[w][i]) for w in widths},
                   original_cloud_mean_log_weight=logf)
    selected_peak=max((i for i,r in enumerate(candidates) if r['width']==4.),key=lambda i:candidates[i]['original_log_importance_weight'])
    second_peak=sorted((i for i,r in enumerate(candidates) if r['width']==4.),key=lambda i:candidates[i]['original_log_importance_weight'],reverse=True)[1]
    members=np.asarray([p['position'] for p in cfg['metadata']['rigid_members']]);distance=member_distances(poses,members)
    peak_model,_=geometry_model(cfg['metadata'],cfg['fixed_poses'][0],candidates[selected_peak]['pose'],region['shape_sha256'],model['angular_length'])
    _,near_radius,_=region_latent([candidates[second_peak]['pose']],{'gaussian_chart':peak_model,'fixed_neighbor':cfg['fixed_poses'][0]})
    archived={**local_dependencies([Path(__file__)]),'config.json':PREP/'config.json','shape.json':Path(cfg['shape']),
              'old-competitor-model.json':PREP/'model-competitor.json','old-mixture-model.json':PREP/'model-mixture.json',
              'old-competitor-region.json':PREP/'region-competitor-r3.json','old-preparation-report.json':PREP/'report.json'}
    for name,path in archived.items():shutil.copy2(path,out/'provenance'/name)
    result={'complete':True,'protocol_sha256':sha(out/'protocol.json'),'candidates':candidates,'selected_peak':selected_peak,
       'second_width4_peak':second_peak,'top_two_member_RMS_A':float(distance[selected_peak,second_peak]),
       'second_peak_radius_in_first_peak_geometry_chart_A':float(near_radius[0]),
       'member_RMS_distance_matrix_A':distance.tolist(),'campaigns':campaign_records,
       'old_competitor_region_sha256':sha(PREP/'region-competitor-r3.json'),
       'source_input_sha256':sources,'archived_sha256':{p.name:sha(p) for p in (out/'provenance').iterdir()},
       'coordinate_audit':diagnostic,
       'scope':'Selected historical q>=5 AB-compatible poses only, with independently checked geometry and full original densities. Original weights are retained as discovery provenance, never equilibrium mixture allocations. No physical draws, counts, fits, new regional integral or global-coverage certificate.'}
    write(out/'analysis.json',result)
    print(json.dumps({'out':str(out),'candidates':len(candidates),'selected_peak':selected_peak,'second_width4_peak':second_peak,
                      'top_two_member_RMS_A':result['top_two_member_RMS_A'],'second_geometry_radius':float(near_radius[0])}))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args();prepare(args.out.resolve())
