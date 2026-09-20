#!/usr/bin/env python3
"""Read-only comparison of a frozen intermediate chart ball and original guides.

Every estimator retains its original proposal and unconditional draw count.
The new uniform reference estimates only its ball; it says nothing about the
complement. Historical guides used to select the ball are exploratory controls.
"""
from __future__ import annotations
import os
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[_key]='1'
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from scipy.spatial.transform import Rotation

from analyze_latent_region import original_q_window,original_q_contains,shell_log_volume,validate_manifest_q_window
from analyze_native_region_reference import native_q
from audit_shoulder_mis_independently import chart_values,independent_densities,near
from compare_intermediate_reference import load_guided as audit_guided,population_rse
from compare_cayley_qwindow import PHYSICAL_KEYS
from prepare_cayley_rms_cover import read,require,sha,write
from run_shoulder_mis_campaign import local_dependencies
from shoulder_mis import LogMoments,quota_summary

ROOT=Path(__file__).resolve().parents[1]
WINDOW=dict(minimum=2.,maximum=5.,lower_inclusive=True,upper_inclusive=False)
DEFAULT_CHART=ROOT/'runs/ab-intermediate-guide-preparation-20260920/model-weighted.json'
DEFAULT_GUIDES=tuple(ROOT/f'runs/ab-intermediate-{kind}-4x16384-l64-20260920' for kind in ('mixture','geometry'))
KEYS=('full','ball','complement','shell0','shell1','shell2','shell3')


def partition(radius,limit):
    """Closed ball and four disjoint shells; no tolerance changes membership."""
    require(math.isfinite(limit) and limit>0 and radius>=0,'Invalid chart radius')
    if radius>limit:return ('full','complement')
    shell=next(i for i in range(4) if radius<=limit*(i+1)/4)
    return ('full','ball',f'shell{shell}')


class PartitionMoments:
    def __init__(self,radius):
        self.radius=radius
        self.values={kind:{key:LogMoments() for key in KEYS} for kind in ('physical','hard')}

    def add(self,radius=None,physical=None,hard=None,cloud_pair=None):
        selected=() if physical is None else partition(radius,self.radius)
        if physical is not None:
            require(math.isfinite(physical) and math.isfinite(hard),'Invalid nonzero weight')
            require(len(cloud_pair)==2 and all(math.isfinite(c) for c in cloud_pair),'Require independent cloud pair')
        for key in KEYS:
            use=key in selected
            self.values['physical'][key].add(physical if use else -math.inf,cloud_pair if use else None)
            self.values['hard'][key].add(hard if use else -math.inf)

    def merge(self,other):
        require(self.radius==other.radius,'Cannot merge different partitions')
        for kind in self.values:
            for key in KEYS:self.values[kind][key].merge(other.values[kind][key])

    def report(self,supported=KEYS):
        result={}
        for kind in self.values:
            result[kind]={}
            for key in supported:
                row=quota_summary([self.values[kind][key]])
                row['row_RSE']=row.pop('stratified_RSE')
                row['variance_rule']='IID row variance over full original unconditional N, including every zero'
                result[kind][key]=row
        return result


def finish(aggregate,populations,supported=KEYS):
    require(populations and len({p['samples'] for p in populations})==1,'Population RSE requires equal fixed budgets')
    result=aggregate.report(supported)
    for kind in result:
        for key,row in result[kind].items():
            row['independent_population_RSE']=population_rse([p[kind][key]['logQ'] for p in populations])
    return dict(**result,populations=populations)


def read_batches(path,size=4096):
    with path.open('rb') as handle:
        while True:
            lines=[line for _,line in zip(range(size),handle)]
            if not lines:return
            yield lines,[json.loads(line) for line in lines]


def validate_region(region,model,cfg,shape_hash):
    require(original_q_window(region)==WINDOW,'Require the exact original 2<=q<5 window')
    require(region.get('minimum_mahalanobis_radius',0.)==0.,'Reference must be a ball without an inner hole')
    require(math.isfinite(region['mahalanobis_radius']) and region['mahalanobis_radius']>0,'Invalid frozen ball')
    require(len(model['weights'])==1 and model['weights']==[1.],'Ball requires a single normalized chart')
    require(region['gaussian_chart']==model,'Reference chart differs from the exact frozen model')
    require(region['physical_metric']==cfg['metadata'] and region['physical_fixed_neighbors']==cfg['fixed_poses'],'Physical neighbors/metric changed')
    require(region['fixed_neighbor'] in cfg['fixed_poses'],'Chart anchor is not a physical neighbor')
    require(region['shape_sha256']==model['shape_sha256']==shape_hash,'Shape changed')
    for key,field in [('capture_center','capture_center'),('capture_radius','capture_radius'),
        ('depletant_radius','depletant_radius'),('activity','reservoir_density')]:
        require(region[key]==cfg[field],f'Physical target changed: {key}')


def reference_weights(row,logj,logv,manifest):
    require(row['region_valid']==original_q_contains(row['q'],WINDOW),'Original q predicate changed')
    valid=row['capture_valid'] and row['hard_valid'] and row['region_valid']
    if not valid:
        require(row['log_importance_weight'] is None and row['log_hard_weight'] is None and not row['clouds'],'Invalid reference row carries weight')
        return None,None,None
    require(len(row['clouds'])==2,'Missing reference cloud pair')
    hard=logv+logj;near(hard,row['log_hard_weight'])
    clouds=[]
    for cloud in row['clouds']:
        value=manifest['activity']*cloud['lower_volume']+cloud['overlap_points']*math.log1p(manifest['activity']/manifest['lambda'])
        near(value,cloud['log_weight']);clouds.append(value+hard)
    weight=float(logsumexp(clouds))-math.log(2);near(weight,row['log_importance_weight'])
    # Keep the recorded original weight after the independent reconstruction.
    return row['log_importance_weight'],row['log_hard_weight'],[c['log_weight']+row['log_hard_weight'] for c in row['clouds']]


def load_reference(root,model):
    root=Path(root).resolve();master=read(root/'manifest.json');cfg=read(root/'provenance/config.json')
    region=read(root/'provenance/region.json');shape_hash=sha(root/'provenance/shape.json')
    for name,digest in master['archive_sha256'].items():require(sha(root/'provenance'/name)==digest,f'Changed reference input: {name}')
    require(master['region_sha256']==sha(root/'provenance/region.json'),'Changed region hash')
    validate_region(region,model,cfg,shape_hash)
    radius=region['mahalanobis_radius'];logv=shell_log_volume(region)
    aggregate=PartitionMoments(radius);populations=[];hashes={};cpu=0.;errors=dict(radius=0.,jacobian=0.,original_q=0.)
    supported=('ball','shell0','shell1','shell2','shell3')
    for job in master['jobs']:
        path=Path(job['directory']);run=read(path/'manifest.json');summary=read(path/'summary.json')
        require(summary['complete'] and summary['manifest']==run,'Incomplete or inconsistent reference manifest')
        require(run['schema']=='uniform-latent-region-normalizer-v2','Require all-pose extended reference rows')
        require(run['samples']==summary['samples']==job['samples'] and run['seed']==job['seed'],'Reference budget or seed changed')
        for key,expected in dict(config_sha256=sha(root/'provenance/config.json'),shape_sha256=shape_hash,
            region_sha256=master['region_sha256'],executable_sha256=master['archive_sha256']['latent-region-normalizer'],
            physical_fixed_neighbors=cfg['fixed_poses'],chart_anchor=region['fixed_neighbor'],minimum_latent_radius=0.,
            latent_radius=radius,activity=cfg['reservoir_density'],lambda_ratio=master['lambda_ratio'],cloud_replicates=2).items():
            require(run[key]==expected,f'Reference manifest changed: {key}')
        require(master['cloud_replicates']==2 and run['lambda']==(run['activity']*run['lambda_ratio'] if run['activity']>0 else 1.),'Reference cloud law changed')
        validate_manifest_q_window(run,region,WINDOW);near(run['log_latent_shell_volume'],logv)
        for name,field in [('input-config.json','config_sha256'),('shape.json','shape_sha256'),
            ('region.json','region_sha256'),('source-bundle.json','source_bundle_sha256')]:
            require(sha(path/'provenance'/name)==run[field],'Reference population provenance changed')
        local=PartitionMoments(radius);digest=hashlib.sha256();n=0
        for lines,rows in read_batches(path/'samples.jsonl'):
            for line in lines:digest.update(line)
            positions=np.asarray([r['pose']['position'] for r in rows]);quats=np.asarray([r['pose']['orientation'] for r in rows])
            near(np.sum(quats*quats,axis=1),np.ones(len(rows)),1e-8)
            rotations=Rotation.from_quat(quats[:,[1,2,3,0]]).as_matrix()
            _,radii,jac=chart_values(model,positions,rotations,region['fixed_neighbor'])
            for i,row in enumerate(rows):
                require(row['draw']==n,'Missing reference draw');n+=1
                errors['radius']=max(errors['radius'],near(radii[i,0],row['latent_radius']))
                errors['jacobian']=max(errors['jacobian'],near(jac[i,0],row['log_physical_jacobian']))
                errors['original_q']=max(errors['original_q'],near(native_q(cfg['metadata'],row['pose']),row['q']))
                require(row['capture_valid']==(math.dist(row['pose']['position'],cfg['capture_center'])<=cfg['capture_radius']),'Reference capture predicate changed')
                require(0<=row['latent_radius']<=radius,'Reference draw outside its frozen support')
                near(np.linalg.norm(row['latent']),row['latent_radius'])
                weight,hard,pair=reference_weights(row,jac[i,0],logv,run)
                # Membership is the independently reconstructed chart metric;
                # the latent draw is retained only to audit the inverse map.
                if weight is not None:require(radii[i,0]<=radius,'Reconstructed valid reference pose outside ball')
                local.add(radii[i,0],weight,hard,pair)
        require(n==job['samples'] and digest.hexdigest()==summary['samples_sha256'],'Reference rows or unconditional N changed')
        current=local.report(supported)
        for kind,key in [('physical','region'),('hard','hard_region')]:
            actual=current[kind]['ball']['logQ'];recorded=summary['estimates'][key]['logQ']
            require(actual==recorded if actual is None else recorded is not None and abs(actual-recorded)<2e-8,'Reference summary mass mismatch')
        populations.append(dict(id=job['id'],seed=job['seed'],samples=n,**current,samples_sha256=digest.hexdigest()))
        aggregate.merge(local);hashes[str(path/'samples.jsonl')]=digest.hexdigest();cpu+=summary['sampler_cpu_seconds']
    return dict(root=str(root),**finish(aggregate,populations,supported),sample_sha256=hashes,
        manifest_sha256=sha(root/'manifest.json'),region_sha256=master['region_sha256'],CPU_seconds=cpu,
        physical_signature={k:cfg[k] for k in PHYSICAL_KEYS},shape_sha256=shape_hash,reconstruction_max_errors=errors,
        complement=dict(estimated=False,reason='Outside the reference proposal support; no complement or full-window mass estimate is available.'),
        scope='Fresh fixed-region uniform-ball estimate conditional on the previously selected chart and radius.'),region,cfg


def load_guide(root,region,cfg):
    root=Path(root).resolve();audited=audit_guided(root);master=read(root/'manifest.json')
    require(audited['original_integration_window']==WINDOW,'Guide must use the exact original window')
    require(audited['physical_signature']=={k:cfg[k] for k in PHYSICAL_KEYS} and audited['shape_sha256']==region['shape_sha256'],'Guide physical AB target changed')
    model=read(root/'provenance/guide-model.json');radius=region['mahalanobis_radius']
    aggregate=PartitionMoments(radius);populations=[];errors=dict(original_density=0.,original_q=0.)
    for job in master['jobs']:
        path=Path(job['output']);run=read(path/'manifest.json');local=PartitionMoments(radius);digest=hashlib.sha256();n=0
        for lines,rows in read_batches(path/'samples.jsonl'):
            for line in lines:digest.update(line)
            lg,_,radii,_,_,q,capture=independent_densities(rows,cfg,model,region,run['guide'],run['cover_mixture'])
            errors['original_density']=max(errors['original_density'],near(lg,[r['log_proposal_density'] for r in rows]))
            errors['original_q']=max(errors['original_q'],near(q,[r['q'] for r in rows]))
            for i,row in enumerate(rows):
                require(row['draw']==n,'Missing guide draw');n+=1
                inside=original_q_contains(row['q'],WINDOW);reason=row.get('zero')
                require((reason=='q')==(not inside),'Guide q predicate changed')
                if inside:require((reason=='capture')==(not capture[i]),'Guide capture predicate changed')
                if reason is not None:local.add();continue
                hard=-row['log_proposal_density'];pair=[x+hard for x in row['cloud_log_weights']]
                near(float(logsumexp(pair))-math.log(2),row['log_importance_weight'])
                local.add(radii[i],row['log_importance_weight'],hard,pair)
        original_population=next(p for p in audited['populations'] if p['seed']==job['seed'])
        require(n==run['samples']==original_population['samples'] and digest.hexdigest()==audited['sample_sha256'][str(path/'samples.jsonl')],'Guide original rows or N changed')
        populations.append(dict(id=path.name,seed=job['seed'],samples=n,**local.report(),samples_sha256=digest.hexdigest()));aggregate.merge(local)
    result=dict(root=str(root),**finish(aggregate,populations),original_campaign_audit=audited,reconstruction_max_errors=errors,
        CPU_seconds=audited['CPU_seconds'],scope='Historical original-density estimates in a data-selected region and its complement; exploratory, not held-out validation.')
    for kind in ('physical','hard'):
        actual=result[kind]['full']['logQ'];recorded=audited[kind]['all']['logQ']
        require(actual==recorded if actual is None else recorded is not None and abs(actual-recorded)<2e-8,'Guide full mass changed')
    return result


def compare(reference,chart_model,guided_roots,out):
    out=Path(out).resolve();require(not out.exists(),'Use a fresh derived output directory')
    chart_model=Path(chart_model).resolve();model=read(chart_model)
    ref,region,cfg=load_reference(reference,model);guides={};seeds={p['seed'] for p in ref['populations']}
    require(len(seeds)==len(ref['populations']),'Repeated reference seed')
    for path in guided_roots:
        name=Path(path).name;require(name not in guides,'Duplicate guide campaign')
        guides[name]=load_guide(path,region,cfg)
        for population in guides[name]['populations']:
            require(population['seed'] not in seeds,'Require fresh distinct random streams');seeds.add(population['seed'])
    (out/'provenance').mkdir(parents=True)
    dependencies=local_dependencies([Path(__file__)])
    for name,path in dependencies.items():(out/'provenance'/name).write_bytes(path.read_bytes())
    (out/'provenance/chart-model.json').write_bytes(chart_model.read_bytes())
    radius=region['mahalanobis_radius']
    scope=('The chart and radius were selected using earlier reference/guide data. The fresh uniform-ball draws give a conditional estimate of that frozen selected region. '
        'The historical guide restrictions are exploratory controls, not held-out validation, and nominal cross-campaign significance is not reported. '
        'Each source keeps its original proposal density, paired clouds and full unconditional N; no source training rows or campaign estimates are pooled. '
        'The uniform reference cannot estimate the complement or the full intermediate region. Hard flags are from the archived kernel; this audit reconstructs poses, metrics, densities and weights, not atom overlaps. '
        'All-zero supported regions are unresolved, not physical zeros or upper bounds. Observed errors do not certify unseen mass or equilibrium mixing.')
    result=dict(complete=True,q_window=WINDOW,radius=radius,shells=[dict(minimum=radius*i/4,maximum=radius*(i+1)/4,lower_inclusive=i==0,upper_inclusive=True) for i in range(4)],
        chart_model_sha256=sha(chart_model),reference=ref,guided_campaigns=guides,physical=ref['physical_signature'],shape_sha256=ref['shape_sha256'],
        source_sha256={name:sha(out/'provenance'/name) for name in dependencies},scope=scope)
    write(out/'comparison.json',result)
    lines=['# Frozen intermediate chart-ball comparison','',f'Mahalanobis radius {radius:g}; original 2<=q<5 and unchanged capture/AB constraints.','',
        '| Source | Region | Full N | Nonzero | log Q | Row / population RSE | Largest weight | Paired-cloud variance |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    def number(x):return 'unresolved' if x is None else f'{x:.6g}'
    for name,item in [('Fresh uniform reference',ref),*guides.items()]:
        for key,row in item['physical'].items():
            label=key
            if key.startswith('shell'):
                i=int(key[-1]);label=f"{'[' if i==0 else '('}{radius*i/4:g},{radius*(i+1)/4:g}]"
            lines.append(f"| {name} | {label} | {row['draws']} | {row['nonzero']} | {number(row['logQ'])} | {number(row['row_RSE'])} / {number(row['independent_population_RSE'])} | {number(row['maximum_fraction'])} | {number(row['paired_cloud_variance_fraction'])} |")
    lines+=['',scope,'','Hard masses, population results, source hashes and reconstruction errors are in comparison.json.','']
    (out/'report.md').write_text('\n'.join(lines));return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--chart-model',type=Path,default=DEFAULT_CHART);parser.add_argument('--guided-root',type=Path,action='append')
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    result=compare(args.reference,args.chart_model,args.guided_root or DEFAULT_GUIDES,args.out)
    print(json.dumps(dict(complete=True,reference=result['reference']['physical']['ball'],guided_campaigns={k:v['physical'] for k,v in result['guided_campaigns'].items()}),indent=2))


if __name__=='__main__':main()
