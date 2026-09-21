#!/usr/bin/env python3
"""Freeze, validate, then run one immutable uniform-latent-region reference.

This is a finite-region integral, not a Gaussian target or a learned proposal.
The supplied region bytes are unchanged; invalid draws remain zero in fixed N.
Native definitions in optional reference packages are downstream-only data.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time
from mobile_native_pocket_campaign import execute_batches

ROOT=Path(__file__).resolve().parents[1]
REVIEW=ROOT/'runs/mobile-competing-importance-source-review-20260921'
REVIEW_SHA='eb82773c7eb5c4f8a1ea01744a85537f88c8754d9e9295c2bc88901ee77b8e35'
PYTHON_REVIEW_SHA='c1543dd0fde8dd7a625e03b1d8dfff3895a804bfca4777b9b6f31bcee05c42c0'
BINARY_SHA='55fd708b5c58477be516f710d78abbe371ede05e8b621e7e3ddd4ac86ebe8b55'
BUNDLE_SHA='033d3776f2b49fbb66694b52334c284ae8bdbd95c3d310ceb7ba98bbaa18b302'
SCHEMA='latent-reference-controller-v1'
CLOUDS=2


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,value):Path(p).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def require(ok,message):
    if not ok:raise ValueError(message)
def inside(root,name):
    root=Path(root).resolve();name=Path(name)
    require(not name.is_absolute() and '..' not in name.parts,'Unsafe archived relative path')
    path=(root/name).resolve();require(path.is_relative_to(root),'Archive path escapes root')
    return path


def allocation(samples,replicates,seed):
    require(type(samples) is int and 1<=samples<2**64,'Positive u64 sample count required')
    require(type(replicates) is int and 1<=replicates<=1024,'Require one to 1024 independent populations')
    require(type(seed) is int and 0<=seed<=2**64-1-1009*(replicates-1),'Independent seeds overflow u64')
    return [seed+1009*i for i in range(replicates)]


def validate_region(region,config,shape,shape_sha):
    """Match the whole scaffold, allowing any explicit fixed chart anchor."""
    import numpy as np
    from scipy.spatial.transform import Rotation
    fixed=config['fixed_poses'];anchor=region['fixed_neighbor']
    require(fixed and region.get('physical_fixed_neighbors',[anchor])==fixed,'Physical scaffold differs')
    indices=[i for i,p in enumerate(fixed)if p==anchor]
    require(len(indices)==1,'Chart anchor must identify one supplied fixed neighbor')
    for field in ('capture_center','capture_radius','depletant_radius'):
        require(region[field]==config[field],'Region/config target mismatch: '+field)
    require(region['physical_metric']==config['metadata'] and region['activity']==config['reservoir_density'],
        'Physical metric or activity changed')
    require(config.get('target_region') is None,'Unimplemented extra config target filter')
    chart=region['gaussian_chart']
    require(region['shape_sha256']==chart['shape_sha256']==shape_sha,'Physical shape mismatch')
    require(chart.get('coordinate_convention')=='anchor-body-relative' and chart['weights']==[1.]
        and all(len(chart[k])==1 for k in ('anchors','means','covariances')) and 'base_model'not in chart,
        'Exactly one ordinary body-relative Gaussian coordinate chart required')
    ell=chart['angular_length'];require(math.isfinite(ell)and ell>0,'Invalid angular coordinate length')
    c=np.asarray(chart['covariances'][0],float);m=np.asarray(chart['means'][0],float)
    a=chart['anchors'][0];r=np.asarray(a['rotation'],float);t=np.asarray(a['position'],float)
    require(c.shape==(6,6)and m.shape==(6,)and np.isfinite(c).all()and np.isfinite(m).all(),'Invalid finite chart')
    require(np.allclose(c,c.T,rtol=1e-12,atol=1e-14),'Chart covariance is not symmetric')
    require(r.shape==(3,3)and t.shape==(3,)and np.isfinite(r).all()and np.isfinite(t).all()
        and np.allclose(r.T@r,np.eye(3),atol=1e-10)and abs(np.linalg.det(r)-1)<1e-10,'Invalid proper chart pose')
    lower=np.linalg.cholesky(c) # No jitter or covariance modification.
    lo,hi=region.get('minimum_mahalanobis_radius',0.),region['mahalanobis_radius']
    require(all(type(x)in(int,float)and math.isfinite(x)for x in(lo,hi))and 0<=lo<hi,'Invalid finite latent shell')
    qmin=region['minimum_original_q'];qmax=region.get('maximum_original_q')
    require(type(qmin)in(int,float)and math.isfinite(qmin)and qmin>=0,'Invalid q minimum')
    require(qmax is None or(type(qmax)in(int,float)and math.isfinite(qmax)and qmax>qmin),'Invalid q maximum')
    for field in ('minimum_original_q_inclusive','maximum_original_q_inclusive'):
        require(type(region.get(field,True))is bool,'q boundary inclusion must be Boolean')
    for fixed_pose in fixed:
        quat=np.asarray(fixed_pose['orientation']);position=np.asarray(fixed_pose['position'])
        require(quat.shape==(4,)and position.shape==(3,)and np.isfinite(quat).all()and np.isfinite(position).all()
            and abs(np.linalg.norm(quat)-1)<1e-9,'Invalid fixed proper pose')
    ar=Rotation.from_quat(np.asarray(anchor['orientation'])[[1,2,3,0]]).as_matrix()
    center=np.asarray(anchor['position'])+ar@(t+m[:3]);displacement=float(hi*np.linalg.norm(lower[:3],ord=2))
    upper=float(np.linalg.norm(center-config['capture_center'])+displacement)
    shape_bound=max(float(np.linalg.norm(atom['center']))+atom['radius']for atom in shape['atoms'])
    world_upper=float(np.linalg.norm(center)+displacement)
    sphere=config['metadata'].get('physical_sphere_radius_A')
    # This executable has no atomic-wall predicate. If the supplied metadata
    # declares the original sphere, certify that the finite chart lies inside.
    # A loose bound fails closed instead of silently dropping that wall.
    if sphere is not None:
        require(math.isfinite(sphere)and world_upper+shape_bound+1e-8<sphere,
            'Declared origin-centered physical sphere is not certified redundant on this finite region')
    return dict(passed=True,chart_anchor_index=indices[0],fixed_neighbor_count=len(fixed),
        physical_draws=0,minimum_latent_radius=lo,maximum_latent_radius=hi,
        q_window=dict(minimum=qmin,maximum=qmax,lower_inclusive=region.get('minimum_original_q_inclusive',True),
                      upper_inclusive=region.get('maximum_original_q_inclusive',True)),
        covariance_condition=float(np.linalg.cond(c)),chart_center=center.tolist(),maximum_center_displacement_A=displacement,
        capture_center_norm_upper_A=upper,entire_chart_inside_capture=upper+1e-8<config['capture_radius'],
        shape_bound_A=shape_bound,origin_center_norm_upper_A=world_upper,declared_origin_sphere_radius_A=sphere,
        guaranteed_wall_clearance_A=None if sphere is None else sphere-world_upper-shape_bound,
        scope='Coordinate and target identity only. No hard-validity conditioning, native/contact mask, fitted covariance or bath query. Capture/q/core failures stay zero in unconditional N.')


def reviewed_sources(review):
    review=Path(review).resolve()
    require(sha(review/'validation.json')==REVIEW_SHA and sha(review/'python-closure-complete.json')==PYTHON_REVIEW_SHA,
        'Unreviewed executable or Python closure declaration')
    data=read(review/'validation.json');binary=Path(data['binary']);bundle=Path(data['source_bundle'])
    require(sha(binary)==BINARY_SHA and sha(bundle)==BUNDLE_SHA,'Reviewed executable/source identity changed')
    require(bundle.read_bytes()in binary.read_bytes(),'Source bundle not embedded verbatim in executable')
    source=read(bundle)['files']
    require(set(source)==set(data['source_sha256']),'Reviewed Rust closure differs')
    for name,item in source.items():
        path=inside(review/'source',name)
        require(sha(path)==item['sha256']==data['source_sha256'][name]
            and path.read_text()==item['text'],'Reviewed Rust source changed: '+name)
    python=read(review/'python-closure-complete.json')['source_sha256']
    require(len(python)==8 and 'launcher.py'in python and 'analyze_latent_region.py'in python,'Incomplete reviewed Python closure')
    for name,digest in python.items():require(sha(inside(review/'python',name))==digest,'Reviewed observer changed: '+name)
    return binary,bundle,source,python


def validate_package(package,config_path,region_path):
    package=Path(package).resolve();frozen=read(package/'freeze.json')['files']
    require(frozen,'Empty reference-package freeze')
    for name,digest in frozen.items():require(sha(inside(package,name))==digest,'Reference package changed: '+name)
    plan=read(package/'plan.json')
    require(sha(config_path)==plan['config_sha256']==sha(inside(package,plan['config'])),'Package config differs')
    require(sha(region_path)==plan['region_sha256']==sha(inside(package,plan['region'])),'Package region differs')
    return dict(path=str(package),plan_sha256=sha(package/'plan.json'),freeze_sha256=sha(package/'freeze.json'),
        native_definition=plan.get('native_definition'),native_definition_sha256=plan.get('native_definition_sha256'))


def command(archive,job,ratio):
    return [str(archive/'latent-region-normalizer'),'--config',str(archive/'config.json'),'--region',str(archive/'region.json'),
        '--out',job['directory'],'--samples',str(job['samples']),'--seed',str(job['seed']),
        '--cloud-replicates','2','--lambda-ratio',str(ratio)]


def freeze(out,config_path,region_path,*,samples=16384,replicates=4,seed=128101010,reference_package=None,review=REVIEW):
    out=Path(out).resolve();require(not out.exists(),'Use a fresh campaign; no overwrite')
    config_path,region_path=Path(config_path).resolve(),Path(region_path).resolve()
    seeds=allocation(samples,replicates,seed);original=read(config_path);region=read(region_path)
    shape=Path(original['shape']);shape=shape if shape.is_absolute()else config_path.parent/shape
    ratio=original['poisson_lambda_ratio'];require(type(ratio)in(int,float)and math.isfinite(ratio)and ratio>0,'Invalid fixed lambda ratio')
    require(math.isfinite(original['reservoir_density'])and original['reservoir_density']>=0,'Invalid physical activity')
    geometry=validate_region(region,original,read(shape),sha(shape))
    binary,bundle,rust,python=reviewed_sources(review)
    package=validate_package(reference_package,config_path,region_path)if reference_package is not None else None
    archive=out/'provenance';archive.mkdir(parents=True)
    for name,path in {'source-config.json':config_path,'region.json':region_path,'shape.json':shape,
        'latent-region-normalizer':binary,'source-bundle.json':bundle,
        'source-review.json':Path(review)/'validation.json','python-review.json':Path(review)/'python-closure-complete.json',
        'controller.py':Path(__file__),'mobile_native_pocket_campaign.py':Path(__file__).with_name('mobile_native_pocket_campaign.py')}.items():
        shutil.copy2(path,archive/name)
    shutil.copytree(Path(review)/'source',archive/'source')
    for name in python:shutil.copy2(Path(review)/'python'/name,archive/name)
    if package:shutil.copytree(reference_package,archive/'reference-package',symlinks=False)
    cfg=copy.deepcopy(original);cfg['shape']=str(archive/'shape.json');write(archive/'config.json',cfg)
    write(archive/'geometry-preflight.json',geometry)
    for directory in ('runs','logs'):(out/directory).mkdir()
    jobs=[]
    for i,value in enumerate(seeds):
        ident=f'r{i:02d}';job=dict(id=ident,seed=value,samples=samples,directory=str(out/'runs'/ident),log=str(out/'logs'/f'{ident}.log'))
        job['command']=command(archive,job,ratio);jobs.append(job)
    manifest=dict(schema='uniform-latent-region-campaign-v1',jobs=jobs,workers=min(8,replicates),
        physical_activity=cfg['reservoir_density'],lambda_ratio=ratio,cloud_replicates=2,
        config_sha256=sha(archive/'config.json'),region_sha256=sha(archive/'region.json'),shape_sha256=sha(archive/'shape.json'),
        archive_sha256={p.relative_to(archive).as_posix():sha(p)for p in sorted(archive.rglob('*'))if p.is_file()},
        source_inputs=dict(config=str(config_path),region=str(region_path),review=str(Path(review).resolve())),
        scope='Independent fixed-N uniform latent ellipsoid/shell reference on the supplied immutable physical region. Every invalid draw remains a zero. No global mixture denominator or native/contact conditioning; native labels downstream only.')
    write(out/'manifest.json',manifest)
    protocol=dict(schema=SCHEMA,manifest_sha256=sha(out/'manifest.json'),controller_sha256=sha(__file__),
        batch_executor_sha256=sha(archive/'mobile_native_pocket_campaign.py'),samples_per_population=samples,
        populations=replicates,seed_base=seed,seeds=seeds,maximum_physical_workers=min(8,replicates),
        total_unconditional_draws=samples*replicates,cloud_replicates=2,lambda_ratio=ratio,
        physical_executable_sha256=BINARY_SHA,source_bundle_sha256=BUNDLE_SHA,
        source_config_sha256=sha(config_path),source_region_sha256=sha(region_path),shape_sha256=sha(shape),
        chart_anchor_index=geometry['chart_anchor_index'],reference_package=package,
        stopping='Exactly declared N per stream. No retries, extensions or old-row reuse. Drain every started process before failure; audit once only after all physical jobs succeed and validate.',
        estimands='Qz, Q0 and paired Qz/Q0 on this finite region. No claim about the whole pocket, equilibrium contact frequencies or assembly.',
        native_classifier_scope='Optional archived definition is provenance for later stateless reporting, never an integration filter.')
    write(out/'protocol.json',protocol)
    write(out/'freeze.json',dict(files={p.relative_to(out).as_posix():sha(p)for p in sorted(out.rglob('*'))if p.is_file()}))
    validate(out);return protocol


def validate(out):
    out=Path(out).resolve();protocol=read(out/'protocol.json');archive=out/'provenance'
    require(protocol['schema']==SCHEMA,'Unsupported controller schema')
    for name,digest in read(out/'freeze.json')['files'].items():require(sha(inside(out,name))==digest,'Frozen file changed: '+name)
    require(sha(archive/'controller.py')==protocol['controller_sha256'],'Archived controller differs')
    require(sha(archive/'mobile_native_pocket_campaign.py')==protocol['batch_executor_sha256'],'Batch executor differs')
    require(sha(out/'manifest.json')==protocol['manifest_sha256'],'Campaign manifest differs')
    m=read(out/'manifest.json');require(m['schema']=='uniform-latent-region-campaign-v1','Unexpected proposal law')
    cfg=read(archive/'config.json');original=read(archive/'source-config.json');region=read(archive/'region.json')
    restored=copy.deepcopy(cfg);restored['shape']=original['shape'];require(restored==original,'Physical config changed beyond shape relocation')
    require(cfg['shape']==str(archive/'shape.json'),'Wrong archived shape path')
    for name,digest in m['archive_sha256'].items():require(sha(inside(archive,name))==digest,'Archived dependency differs: '+name)
    require(sha(archive/'region.json')==protocol['source_region_sha256']==m['region_sha256'],'Region bytes changed')
    require(sha(archive/'source-config.json')==protocol['source_config_sha256'],'Source config changed')
    require(sha(archive/'shape.json')==protocol['shape_sha256']==m['shape_sha256'],'Shape changed')
    require(sha(archive/'config.json')==m['config_sha256'],'Resolved config differs')
    require(sha(archive/'latent-region-normalizer')==protocol['physical_executable_sha256']==BINARY_SHA
        and sha(archive/'source-bundle.json')==protocol['source_bundle_sha256']==BUNDLE_SHA,'Wrong reviewed executable')
    require(sha(archive/'source-review.json')==REVIEW_SHA and sha(archive/'python-review.json')==PYTHON_REVIEW_SHA,
        'Source review or reviewed observer closure changed')
    for name,digest in read(archive/'python-review.json')['source_sha256'].items():
        require(sha(inside(archive,name))==digest,'Frozen reviewed observer differs: '+name)
    require((archive/'source-bundle.json').read_bytes()in(archive/'latent-region-normalizer').read_bytes(),'Bundle not embedded in executable')
    for name,item in read(archive/'source-bundle.json')['files'].items():
        require(sha(inside(archive/'source',name))==item['sha256'],'Archived embedded Rust file differs')
    geometry=validate_region(region,cfg,read(archive/'shape.json'),m['shape_sha256'])
    require(geometry==read(archive/'geometry-preflight.json') and geometry['chart_anchor_index']==protocol['chart_anchor_index'],
        'Geometry preflight differs')
    seeds=allocation(protocol['samples_per_population'],protocol['populations'],protocol['seed_base'])
    require(protocol['seeds']==seeds and len(m['jobs'])==len(seeds),'Population design differs')
    require(protocol['total_unconditional_draws']==len(seeds)*protocol['samples_per_population'],'Total N differs')
    require(m['cloud_replicates']==protocol['cloud_replicates']==2 and m['lambda_ratio']==protocol['lambda_ratio']==cfg['poisson_lambda_ratio'],
        'Poisson allocation changed')
    require(m['physical_activity']==cfg['reservoir_density'],'Physical activity differs')
    require(m['workers']==protocol['maximum_physical_workers']==min(8,len(seeds)),'Worker allocation changed')
    for i,(job,seed)in enumerate(zip(m['jobs'],seeds)):
        require(job['id']==f'r{i:02d}'and job['seed']==seed and job['samples']==protocol['samples_per_population'],'Job allocation differs')
        require(job['directory']==str(out/'runs'/job['id'])and job['log']==str(out/'logs'/f"{job['id']}.log"),'Job output escaped campaign')
        require(job['command']==command(archive,job,m['lambda_ratio']),'Physical command differs')
    if protocol['reference_package']:
        path=archive/'reference-package';record=protocol['reference_package']
        require(sha(path/'plan.json')==record['plan_sha256']and sha(path/'freeze.json')==record['freeze_sha256'],'Reference package binding changed')
        validate_package(path,archive/'source-config.json',archive/'region.json')
    return protocol


def check(out):
    out=Path(out).resolve();protocol=validate(out)
    require(sha(__file__)==protocol['controller_sha256'],'Run the exact archived controller')
    require(not(out/'status.json').exists()and not(out/'assessment').exists(),'No retry/resume/audit overwrite')
    require(not any((out/'runs').iterdir())and not any((out/'logs').iterdir()),'Existing physical outputs or logs')
    require(not list(out.glob('r*-status.json')),'Existing population status')
    return protocol


def verify_output(out,manifest,job):
    directory=Path(job['directory']);summary=read(directory/'summary.json');pm=read(directory/'manifest.json')
    require(summary['complete']is True and summary['samples']==job['samples']and summary['manifest']==pm,'Incomplete population')
    region=read(Path(out)/'provenance/region.json')
    expected=dict(schema='uniform-latent-region-normalizer-v2',samples=job['samples'],seed=job['seed'],cloud_replicates=2,
        activity=manifest['physical_activity'],lambda_ratio=manifest['lambda_ratio'],region_sha256=manifest['region_sha256'],
        config_sha256=manifest['config_sha256'],shape_sha256=manifest['shape_sha256'],source_bundle_sha256=BUNDLE_SHA,
        executable_sha256=BINARY_SHA,physical_fixed_neighbors=read(Path(out)/'provenance/config.json')['fixed_poses'],
        chart_anchor=region['fixed_neighbor'])
    require(all(pm.get(k)==v for k,v in expected.items()),'Population law/input identity differs')
    for name,field in [('input-config.json','config_sha256'),('region.json','region_sha256'),('shape.json','shape_sha256'),('source-bundle.json','source_bundle_sha256')]:
        require(sha(directory/'provenance'/name)==pm[field],'Population input copy differs: '+name)
    sample_sha=sha(directory/'samples.jsonl');require(summary['samples_sha256']==sample_sha,'Saved row bytes differ')
    require(summary['estimates']['region']['draws']==summary['estimates']['hard_region']['draws']==job['samples'],'Unconditional denominator changed')
    return dict(samples_sha256=sample_sha,summary_sha256=sha(directory/'summary.json'),
        manifest_sha256=sha(directory/'manifest.json'),sampler_cpu_seconds=summary['sampler_cpu_seconds'])


def verify_assessment(out,manifest,analysis,jobs):
    require(analysis['region_sha256']==manifest['region_sha256']and 'importance_sampling'not in analysis,'Audit region or proposal law differs')
    require(analysis['estimate']['draws']==analysis['hard_region']['draws']==sum(j['samples']for j in jobs),'Audit dropped unconditional draws')
    require(analysis['physical_fixed_neighbors']==read(Path(out)/'provenance/config.json')['fixed_poses'],'Audit scaffold differs')
    populations=analysis['populations'];require(len(populations)==len(jobs)and {p['id']for p in populations}=={j['id']for j in jobs},'Audit omitted populations')
    for job in jobs:
        record=next(p for p in populations if p['id']==job['id'])
        require(record['seed']==job['seed']and record['estimate']['draws']==record['hard_region']['draws']==job['samples'],'Audit population allocation differs')
        require(record['samples_sha256']==job['output']['samples_sha256']==sha(Path(job['directory'])/'samples.jsonl'),'Audit row binding differs')


def run(out):
    out=Path(out).resolve();protocol=check(out);manifest=read(out/'manifest.json')
    state=dict(schema='latent-reference-status-v1',complete=False,phase='physical',started=time.time(),
        protocol_sha256=sha(out/'protocol.json'),jobs=[dict(j,status='pending')for j in manifest['jobs']],audit=None)
    with(out/'status.json').open('x')as stream:json.dump(state,stream,indent=2,allow_nan=False)
    def snapshot():write(out/'status.json',state)
    try:
        execute_batches(state['jobs'],snapshot)
        state['phase']='physical_validation';snapshot();validate(out)
        for job in state['jobs']:
            job['output']=verify_output(out,manifest,job)
            write(out/f"{job['id']}-status.json",dict(id=job['id'],returncode=job['returncode'],seed=job['seed'],samples=job['samples'],output=job['output']))
        state['phase']='audit';snapshot()
        argv=[sys.executable,'-B',str(out/'provenance/analyze_latent_region.py'),'--root',str(out)]
        state['audit']=dict(command=argv,started=time.time(),returncode=None);snapshot()
        with(out/'logs/audit.log').open('xb')as stream:
            result=subprocess.run(argv,stdout=stream,stderr=subprocess.STDOUT)
        state['audit'].update(returncode=result.returncode,finished=time.time());snapshot();result.check_returncode()
        verify_assessment(out,manifest,read(out/'assessment/analysis.json'),state['jobs'])
        state['audit']['analysis_sha256']=sha(out/'assessment/analysis.json')
        state.update(complete=True,phase='complete',finished=time.time());snapshot()
    except BaseException as error:
        state.update(complete=False,phase=state['phase']+'_failed',exception=repr(error),finished=time.time());snapshot();raise
    return state


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('freeze','preflight','run'))
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--config',type=Path);parser.add_argument('--region',type=Path)
    parser.add_argument('--reference-package',type=Path);parser.add_argument('--samples',type=int,default=16384)
    parser.add_argument('--replicates',type=int,default=4);parser.add_argument('--seed',type=int,default=128101010)
    args=parser.parse_args()
    if args.action=='freeze':
        if args.config is None or args.region is None:parser.error('freeze requires --config and --region')
        result=freeze(args.out,args.config,args.region,samples=args.samples,replicates=args.replicates,seed=args.seed,reference_package=args.reference_package)
    else:result=check(args.out)if args.action=='preflight'else run(args.out)
    print(json.dumps(dict(action=args.action,out=str(args.out.resolve()),complete=result.get('complete'),
        physical_launches=0 if args.action!='run'else len(result['jobs'])),indent=2))


if __name__=='__main__':main()
