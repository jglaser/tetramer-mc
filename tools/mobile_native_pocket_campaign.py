#!/usr/bin/env python3
"""Freeze and execute independent uniform-volume references for native motif 7/4."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
REVIEW=ROOT/'runs/mobile-competing-importance-source-review-20260921'
REFERENCE=ROOT/'runs/mobile-competing-reference-preparation-20260921'
MODEL=ROOT/'examples/frozen-reciprocal-mixture.json'
DEFINITION=ROOT/'runs/mobile-native-region-definition-20260921/definition.json'
CONFIG_SHA='2ed28d5b618209d51dd8a0381c95a8e58853a7e9780f66d53fb027ec07318cf1'
MODEL_SHA='dc9218c9706e1691af9336ef1c0e87a75a86994933dfd41cc6e1d1cd3dc52aa3'
SHAPE_SHA='c7442034e7ffa4627b67c57a81117f0e9bc2d389ecf956a83dac2a5e915554c9'
DEFINITION_SHA='5cf55ebe0c0f78ec6b8cdc745cac938b0fadfa3732e1e1c730864c62c651d7a9'
BINARY_SHA='55fd708b5c58477be516f710d78abbe371ede05e8b621e7e3ddd4ac86ebe8b55'
BUNDLE_SHA='033d3776f2b49fbb66694b52334c284ae8bdbd95c3d310ceb7ba98bbaa18b302'
SAMPLES,REPLICATES,SEED_BASE=8192,4,124101010
ARMS={'r5':(0.,5.),'shell5to8':(5.,8.)}
DESIGNS={
    'initial':dict(seed=SEED_BASE,arms={k:dict(inner=v[0],outer=v[1],replicates=4,samples=8192)for k,v in ARMS.items()}),
    'coverage':dict(seed=125101010,arms={
        'r5repeat':dict(inner=0.,outer=5.,replicates=8,samples=16384),
        'shell8to12':dict(inner=8.,outer=12.,replicates=4,samples=8192),
        'shell12to16':dict(inner=12.,outer=16.,replicates=4,samples=8192),
        'shell16to24':dict(inner=16.,outer=24.,replicates=4,samples=8192),
        'shell24to32':dict(inner=24.,outer=32.,replicates=4,samples=8192)})}
SCHEMA='mobile-native-pocket-controller-v1'

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(ok,message):
    if not ok:raise ValueError(message)
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def exclusive(p,x):
    with Path(p).open('x')as stream:json.dump(x,stream,indent=2,allow_nan=False)

def make_regions(config,base,motif,design='initial'):
    """An unchanged covariance defines coordinates, never a Gaussian target."""
    from scipy.spatial.transform import Rotation
    import numpy as np
    chart={key:copy.deepcopy(base[key])for key in ('schema','coordinate_convention','angular_length','shape_sha256')}
    rotation=Rotation.from_quat(np.asarray(motif['relative_orientation'])[[1,2,3,0]]).as_matrix()
    require(np.array_equal(rotation,np.array(base['anchors'][19]['rotation'])),'Recenter would change angular chart')
    chart.update(anchors=[dict(position=copy.deepcopy(motif['relative_position']),rotation=rotation.tolist())],
        means=[[0.]*6],covariances=[copy.deepcopy(base['covariances'][19])],weights=[1.])
    result={}
    for arm,spec in DESIGNS[design]['arms'].items():
        lo,hi=spec['inner'],spec['outer']
        result[arm]=dict(fixed_neighbor=copy.deepcopy(config['fixed_poses'][0]),physical_fixed_neighbors=copy.deepcopy(config['fixed_poses']),
            capture_center=copy.deepcopy(config['capture_center']),capture_radius=config['capture_radius'],shape_sha256=SHAPE_SHA,
            activity=config['reservoir_density'],depletant_radius=config['depletant_radius'],physical_metric=copy.deepcopy(config['metadata']),
            gaussian_chart=copy.deepcopy(chart),minimum_mahalanobis_radius=lo,mahalanobis_radius=hi,
            minimum_original_q=1.,minimum_original_q_inclusive=False,maximum_original_q_inclusive=True,
            definition=f'Ideal motif7/body2 centered component19 chart, {lo}<rho<={hi}, center included for inner0; original q>1; exact hard scaffold and capture170. Uniform latent-volume reference; no native-contact mask or Gaussian target.')
    return result

def region_checks(config,regions,base,shape):
    import numpy as np
    from prepare_mobile_competing_reference import capture_wall_certificate,jacobian_check
    from prepare_native_tail_regions import decode_latent,encode_matrix,validate_physics
    chart=next(iter(regions.values()))['gaussian_chart'];lower=np.linalg.cholesky(chart['covariances'][0])
    shift=np.r_[np.array(base['anchors'][19]['position'])-chart['anchors'][0]['position'],[0.]*3]+base['means'][19]
    offset=float(np.linalg.norm(np.linalg.solve(lower,shift)))
    require(offset+4<5,'Ideal R5 does not contain original component19 R4')
    bounds={};bound=max(float(np.linalg.norm(a['center']))+a['radius']for a in shape['atoms'])
    for arm,region in regions.items():
        validate_physics(region,config,SHAPE_SHA)
        hi=region['mahalanobis_radius'];latent=np.r_[np.zeros((1,6)),hi*np.eye(6),-hi*np.eye(6)]
        poses,_=decode_latent(latent,chart,config['fixed_poses'][0])
        error=float(np.max(abs(encode_matrix(poses,chart,config['fixed_poses'][0])-latent)))
        require(error<1e-8,'Chart roundtrip differs')
        bounds[arm]=dict(capture_and_wall=capture_wall_certificate(chart,config['fixed_poses'][0],hi,
            config['capture_radius'],config['metadata']['physical_sphere_radius_A'],bound),maximum_backmap_error=error,
            maximum_differential_relative_error=max(jacobian_check(chart,config['fixed_poses'][0],u)for u in latent[[0,1,-1]]))
    return dict(passed=True,physical_draws=0,old_center_whitened_offset=offset,
        covariance_condition=float(np.linalg.cond(chart['covariances'][0])),regions=bounds,
        center_scope='Ideal center may be hard-invalid on the perturbed scaffold. No repair; all invalid samples retain zero weight.')

def freeze(out,design='initial'):
    out=Path(out).resolve();require(not out.exists(),'Fresh campaign required')
    require(sha(REFERENCE/'config.json')==CONFIG_SHA and sha(MODEL)==MODEL_SHA,'Source physical config or atlas changed')
    require(sha(DEFINITION)==DEFINITION_SHA,'Native diagnostic definition changed')
    original=read(REFERENCE/'config.json');base=read(MODEL)['base_model'];definition=read(DEFINITION)
    shape=Path(original['shape']);require(sha(shape)==SHAPE_SHA,'Physical shape changed')
    for name,digest in definition['input_sha256'].items():require(sha(DEFINITION.parent/'inputs'/name)==digest,'Native input changed')
    motifs=read(DEFINITION.parent/'inputs/native-pair-motifs.json')['motifs']
    require(design in DESIGNS,'Unknown prespecified design');specifications=DESIGNS[design]
    regions=make_regions(original,base,next(m for m in motifs if m['id']==7),design)
    preflight=region_checks(original,regions,base,read(shape))
    review=read(REVIEW/'validation.json');binary=Path(review['binary']);bundle=Path(review['source_bundle'])
    require(sha(binary)==BINARY_SHA and sha(bundle)==BUNDLE_SHA,'Reviewed executable changed')
    require(bundle.read_bytes()in binary.read_bytes(),'Embedded source identity differs')
    for name,entry in read(bundle)['files'].items():
        require(sha(REVIEW/'source'/name)==entry['sha256'] and (REVIEW/'source'/name).read_text()==entry['text'],'Reviewed source changed')
    python=read(REVIEW/'python-closure-complete.json')['source_sha256']
    for name,digest in python.items():require(sha(REVIEW/'python'/name)==digest,'Reviewed observer changed')
    top=out/'provenance';top.mkdir(parents=True)
    shutil.copytree(DEFINITION.parent,top/'native-region')
    for name,path in {'source-config.json':REFERENCE/'config.json','source-model.json':MODEL,
        'source-review.json':REVIEW/'validation.json','python-review.json':REVIEW/'python-closure-complete.json',
        'design.md':ROOT/'docs/mobile-native-pocket-reference-design.md','controller.py':Path(__file__)}.items():shutil.copy2(path,top/name)
    write(top/'geometry-preflight.json',preflight)
    if design=='coverage':
        previous=ROOT/'runs/mobile-native-pocket-campaign-20260921';previous_state=read(previous/'status.json')
        require(previous_state['complete']and previous_state['phase']=='complete','First control must be completed')
        for name in ('protocol.json','freeze.json','status.json'):shutil.copy2(previous/name,top/('initial-'+name))
        for name in ('recommendation.json','radial-diagnostic.json'):
            shutil.copy2(ROOT/'runs/mobile-native-pocket-coverage-design-20260921'/name,top/name)
    campaigns=[];job_index=0
    for index,(arm,region)in enumerate(regions.items()):
        spec=specifications['arms'][arm]
        folder=out/arm;archive=folder/'provenance';archive.mkdir(parents=True)
        for name in python:shutil.copy2(REVIEW/'python'/name,archive/name)
        for name,path in {'latent-region-normalizer':binary,'source-bundle.json':bundle,'shape.json':shape}.items():shutil.copy2(path,archive/name)
        config=copy.deepcopy(original);config['shape']=str(archive/'shape.json');write(archive/'config.json',config);write(archive/'region.json',region)
        for name in ('runs','logs'):(folder/name).mkdir()
        jobs=[]
        for replicate in range(spec['replicates']):
            ident=f'r{replicate:02d}';seed=specifications['seed']+1009*job_index;job_index+=1
            job=dict(id=ident,seed=seed,samples=spec['samples'],directory=str(folder/'runs'/ident),log=str(folder/'logs'/f'{ident}.log'))
            job['command']=[str(archive/'latent-region-normalizer'),'--config',str(archive/'config.json'),'--region',str(archive/'region.json'),
                '--out',job['directory'],'--samples',str(spec['samples']),'--seed',str(seed),'--cloud-replicates','2','--lambda-ratio','64']
            jobs.append(job)
        manifest=dict(schema='uniform-latent-region-campaign-v1',jobs=jobs,workers=min(8,spec['replicates']),physical_activity=.035,lambda_ratio=64.,cloud_replicates=2,
            region_sha256=sha(archive/'region.json'),config_sha256=sha(archive/'config.json'),shape_sha256=SHAPE_SHA,
            archive_sha256={p.name:sha(p)for p in archive.iterdir()},source_inputs={'launcher.py':str(REVIEW/'python/launcher.py')},
            scope='Independent finite uniform-latent-volume integral. Invalid zeros retained. Not an entire native basin or assembly ensemble.')
        write(folder/'manifest.json',manifest)
        campaigns.append(dict(arm=arm,path=str(folder),manifest_sha256=sha(folder/'manifest.json'),region_sha256=manifest['region_sha256']))
    protocol=dict(schema=SCHEMA,design=design,allocation=specifications['arms'],campaigns=campaigns,native_definition='provenance/native-region/definition.json',
        samples_per_population=SAMPLES if design=='initial'else None,populations_per_region=4 if design=='initial'else None,
        total_jobs=job_index,total_unconditional_draws=sum(v['samples']*v['replicates']for v in specifications['arms'].values()),maximum_physical_workers=8,
        cloud_replicates=2,lambda_ratio=64.,physical_executable_sha256=BINARY_SHA,source_bundle_sha256=BUNDLE_SHA,
        controller_sha256=sha(__file__),original_config_sha256=CONFIG_SHA,model_sha256=MODEL_SHA,seed_base=specifications['seed'],
        primary_region=next(iter(regions)),sensitivity_regions=list(regions)[1:],stopping='Exactly the declared draw count in each prespecified stream; drain started children on failure; no retries or extensions; audit once after all successful physical jobs.',
        estimands='Qz,Q0,paired Qz/Q0; separate finite-region ratios. Original scaffold and bath unchanged. No global mixture density in weights.',
        discovery_scope='Region selected after prior global discovery, with preexisting covariance and ideal motif center. Only fresh independent rows estimate weight.')
    write(out/'protocol.json',protocol)
    write(out/'freeze.json',dict(files={p.relative_to(out).as_posix():sha(p)for p in out.rglob('*')if p.is_file()}))
    validate(out);return protocol

def validate(out):
    out=Path(out).resolve();p=read(out/'protocol.json')
    require(p['schema']==SCHEMA and p['controller_sha256']==sha(out/'provenance/controller.py'),'Controller identity differs')
    design=p.get('design','initial');require(design in DESIGNS,'Unknown prespecified design');specifications=DESIGNS[design]
    for name,digest in read(out/'freeze.json')['files'].items():require(sha(out/name)==digest,'Frozen input changed: '+name)
    require([c['arm']for c in p['campaigns']]==list(specifications['arms']),'Wrong prespecified regions')
    seeds=[];regions=[];original=read(out/'provenance/source-config.json')
    require(sha(out/'provenance/source-config.json')==CONFIG_SHA,'Wrong physical source')
    for c in p['campaigns']:
        folder=Path(c['path']);require(folder==out/c['arm'],'Campaign moved or directory differs')
        spec=specifications['arms'][c['arm']]
        require(sha(folder/'manifest.json')==c['manifest_sha256'],'Manifest changed')
        m=read(folder/'manifest.json');r=read(folder/'provenance/region.json');config=read(folder/'provenance/config.json')
        restored=copy.deepcopy(config);restored['shape']=original['shape'];require(restored==original,'Changed physical config')
        require(sha(folder/'provenance/latent-region-normalizer')==BINARY_SHA,'Wrong executable')
        require((r['minimum_mahalanobis_radius'],r['mahalanobis_radius'])==(spec['inner'],spec['outer']),'Wrong latent domain')
        require(r['physical_fixed_neighbors']==original['fixed_poses'] and r['physical_metric']==original['metadata'],'Scaffold or metric changed')
        require(r['minimum_original_q']==1. and not r['minimum_original_q_inclusive'] and 'maximum_original_q'not in r,'q window changed')
        require(m['cloud_replicates']==2 and m['lambda_ratio']==64. and len(m['jobs'])==spec['replicates'],'Allocation differs')
        regions.append(r)
        for j in m['jobs']:
            require(j['samples']==spec['samples'] and Path(j['directory'])==folder/'runs'/j['id'],'Fixed population changed')
            seeds.append(j['seed'])
    count=sum(s['replicates']for s in specifications['arms'].values())
    require(seeds==[specifications['seed']+1009*i for i in range(count)],'Independent seeds changed')
    require(p['total_jobs']==count and p['maximum_physical_workers']==8,'Worker/job allocation differs')
    require(all(regions[0]['gaussian_chart']==r['gaussian_chart']for r in regions[1:]),'Shell and ball charts differ')
    return p

def verify_output(folder,m,j):
    d=Path(j['directory']);s=read(d/'summary.json');pm=read(d/'manifest.json')
    require(s['complete'] and s['samples']==j['samples'] and s['manifest']==pm,'Incomplete physical output')
    wanted=dict(schema='uniform-latent-region-normalizer-v2',samples=j['samples'],seed=j['seed'],cloud_replicates=2,activity=.035,lambda_ratio=64.,
        region_sha256=m['region_sha256'],config_sha256=m['config_sha256'],shape_sha256=SHAPE_SHA,source_bundle_sha256=BUNDLE_SHA,executable_sha256=BINARY_SHA)
    require(all(pm.get(k)==v for k,v in wanted.items()),'Output physical identity differs')
    require(s['samples_sha256']==sha(d/'samples.jsonl'),'Saved row bytes differ')
    require(s['estimates']['region']['draws']==s['estimates']['hard_region']['draws']==j['samples'],'Invalid draws excluded')
    return dict(samples_sha256=sha(d/'samples.jsonl'),manifest_sha256=sha(d/'manifest.json'),summary_sha256=sha(d/'summary.json'),sampler_cpu_seconds=s['sampler_cpu_seconds'])

def execute_all(jobs,snapshot,popen=subprocess.Popen):
    """One batch of at most eight; always reap every launched child before raising."""
    require(len(jobs)<=8,'Worker cap exceeded');active=[];error=None
    try:
        for record in jobs:
            require(not Path(record['directory']).exists(),'Existing population output')
            with Path(record['log']).open('xb')as log:child=popen(record['command'],stdout=log,stderr=subprocess.STDOUT)
            active.append((record,child));record.update(status='running',pid=child.pid,started=time.time());snapshot()
    except BaseException as exc:error=exc
    finally:
        for record,child in active:
            code=child.wait();record.update(returncode=code,status='complete'if code==0 else'failed',finished=time.time())
            if code!=0 and error is None:error=RuntimeError('Physical failure; no retry or audit')
            snapshot()
        for record in jobs:
            if record['status']=='pending':record['status']='not_started'
        snapshot()
    if error is not None:raise error

def execute_batches(jobs,snapshot,popen=subprocess.Popen):
    try:
        for offset in range(0,len(jobs),8):execute_all(jobs[offset:offset+8],snapshot,popen)
    finally:
        for record in jobs:
            if record['status']=='pending':record['status']='not_started'
        snapshot()

def run(out):
    out=Path(out).resolve();p=validate(out)
    require(sha(__file__)==p['controller_sha256'],'Run the exact archived controller')
    require(not(out/'status.json').exists(),'No retry or overwrite')
    jobs=[];manifests={}
    for c in p['campaigns']:
        folder=Path(c['path']);require(not(folder/'assessment').exists()and not any((folder/'runs').iterdir()),'Existing outputs')
        m=read(folder/'manifest.json');manifests[c['arm']]=m
        jobs.extend(dict(j,arm=c['arm'],status='pending')for j in m['jobs'])
    state=dict(schema='mobile-native-pocket-status-v1',complete=False,phase='physical',protocol_sha256=sha(out/'protocol.json'),jobs=jobs,audits={},started=time.time())
    exclusive(out/'status.json',state)
    def snapshot():write(out/'status.json',state)
    try:
        execute_batches(jobs,snapshot);validate(out);state['phase']='physical_validation';snapshot()
        for j in jobs:j['output']=verify_output(out/j['arm'],manifests[j['arm']],j)
        state['phase']='audit';snapshot()
        for c in p['campaigns']:
            folder=Path(c['path']);m=manifests[c['arm']]
            argv=[sys.executable,'-B',str(folder/'provenance/analyze_latent_region.py'),'--root',str(folder)]
            a=dict(argv=argv,started=time.time());state['audits'][c['arm']]=a;snapshot()
            with(out/(c['arm']+'-audit.log')).open('xb')as log:result=subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT)
            a.update(returncode=result.returncode,finished=time.time());snapshot();result.check_returncode()
            result=read(folder/'assessment/analysis.json')
            draws=sum(j['samples']for j in m['jobs'])
            require(result['estimate']['draws']==result['hard_region']['draws']==draws and len(result['populations'])==len(m['jobs']),'Incomplete audit')
            require(result['independently_reconstructed_poses']==draws,'Audit did not check every pose')
            a['analysis_sha256']=sha(folder/'assessment/analysis.json');snapshot()
        state.update(complete=True,phase='complete',finished=time.time());snapshot();return state
    except BaseException as error:
        state.update(phase=state['phase']+'_failed',exception=repr(error),finished=time.time());snapshot();raise

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('freeze','validate','run'))
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--design',choices=tuple(DESIGNS),default='initial');args=parser.parse_args()
    if args.action!='freeze'and args.design!='initial':parser.error('--design is only used when freezing')
    result=freeze(args.out,args.design)if args.action=='freeze'else globals()[args.action](args.out)
    print(json.dumps(dict(action=args.action,complete=result.get('complete'),out=str(args.out))))
