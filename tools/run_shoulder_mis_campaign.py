#!/usr/bin/env python3
"""Freeze or run independent fixed-allocation guide/Cayley-cover MIS populations.

Each population contains exactly n_guide and n_cover unconditional draws. The
analysis uses their deterministic mixture density and compares component-only
estimates on these same fresh draws. No historical samples or fitted updates
enter the calculation. --execute-prepared launches an unchanged prepared plan.
"""
from __future__ import annotations

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time

from analyze_native_region_reference import GaussianGuide,validate_q_window
from prepare_cayley_rms_cover import derive_model

ROOT=Path(__file__).resolve().parents[1]
TOOLS=Path(__file__).resolve().parent
SCHEMA='shoulder-fixed-allocation-mis-v1'
PHYSICAL_KEYS=('fixed_poses','capture_center','capture_radius','depletant_radius','reservoir_density','metadata')
HELPERS=('shoulder_mis.py','analyze_shoulder_mis.py','analyze_native_region_reference.py',
    'analyze_latent_region.py','analyze_basin_normalizers.py','analyze_latent_region_shells.py',
    'analyze_shoulder_guides.py','prepare_smc_normalizer_atlas.py','prepare_cayley_rms_cover.py',
    'prepare_native_confirmation_atlas.py')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def require(condition,message):
    if not condition:raise ValueError(message)


def local_dependencies(paths):
    """Archive local imports, including imports nested in functions."""
    result={};pending=list(paths)
    while pending:
        path=Path(pending.pop()).resolve()
        if path.name in result:continue
        require(path.is_file(),f'Missing source dependency: {path}')
        result[path.name]=path
        tree=ast.parse(path.read_text(),filename=str(path))
        for node in ast.walk(tree):
            names=([a.name for a in node.names] if isinstance(node,ast.Import) else
                [node.module] if isinstance(node,ast.ImportFrom) and node.module else [])
            for name in names:
                dependency=path.parent/(name.split('.')[0]+'.py')
                if dependency.is_file() and dependency.name not in result:pending.append(dependency)
    return result


def verify_cover(region,cfg,shape_sha,window):
    require(region['physical_fixed_neighbors']==cfg['fixed_poses'],'Cover must retain the full physical neighbor list')
    require(region['fixed_neighbor'] in cfg['fixed_poses'],'Cover chart anchor is not a physical neighbor')
    for rkey,ckey in [('capture_center','capture_center'),('capture_radius','capture_radius'),
        ('activity','reservoir_density'),('depletant_radius','depletant_radius'),('physical_metric','metadata')]:
        require(region[rkey]==cfg[ckey],f'Cover/config physical mismatch: {rkey}')
    require(region['shape_sha256']==shape_sha,'Cover shape differs from configured shape')
    expected_window=dict(minimum=region['minimum_original_q'],maximum=region['maximum_original_q'],
        lower_inclusive=region.get('minimum_original_q_inclusive',True),
        upper_inclusive=region.get('maximum_original_q_inclusive',True))
    validate_q_window(expected_window)
    require(expected_window==window,'Guide and cover must use the same original-q window')
    require(region.get('minimum_mahalanobis_radius',0.)==0.,'Complete Cayley reference must have no inner hole')
    chart=region['gaussian_chart']
    derived,proof=derive_model(cfg['metadata'],region['fixed_neighbor'],shape_sha,window['maximum'],chart['angular_length'])
    require(region['mahalanobis_radius']==proof['mahalanobis_radius'],'Cover radius differs from geometric construction')
    for key in ('angular_length','shape_sha256','coordinate_convention','anchors','means','covariances','weights'):
        # Both persisted chart and fresh reconstruction use the same explicit
        # FP64 geometric formula. Exact equality avoids quietly shrinking it.
        require(chart[key]==derived[key],f'Cover chart differs from geometric construction: {key}')
    return proof


def prepare(args):
    root=args.out.resolve();require(not root.exists(),'Refuse an existing campaign; use a fresh output directory')
    require(min(args.populations,args.n_guide,args.n_cover,args.workers)>0,'Positive population, quota, and worker counts required')
    require(min(args.n_guide,args.n_cover)>=2,'Each fixed quota requires at least two draws for its variance estimate')
    window=validate_q_window(dict(minimum=args.q_min,maximum=args.q_max,
        lower_inclusive=args.q_lower_closed,upper_inclusive=args.q_upper_closed))
    require(math.isfinite(args.lambda_ratio) and args.lambda_ratio>0,'Positive finite cloud intensity ratio required')
    source_config=args.config.resolve();cfg=read(source_config)
    shape=Path(cfg['shape']);shape=(source_config.parent/shape).resolve() if not shape.is_absolute() else shape.resolve()
    shape_sha=sha(shape);model_path=args.model.resolve();model=read(model_path)
    region_path=args.region.resolve();region=read(region_path)
    require(0<=args.model_anchor_index<len(cfg['fixed_poses']),'Guide anchor outside physical neighbor list')
    description=dict(model_sha256=sha(model_path),weight=args.model_weight,uniform_probability=args.model_uniform_probability,
        anchor_index=args.model_anchor_index,anchor_pose=cfg['fixed_poses'][args.model_anchor_index],
        cube_lengths=[2*cfg['capture_radius']]*3,capture_center=cfg['capture_center'])
    GaussianGuide(description,model,cfg,shape_sha)
    proof=verify_cover(region,cfg,shape_sha,window)
    seeds=[seed+1009*i for i in range(args.populations) for seed in (args.guide_seed_base,args.cover_seed_base)]
    require(all(0<=seed<2**64 for seed in seeds) and len(set(seeds))==len(seeds),'Population/family seeds must be distinct unsigned 64-bit integers')
    binaries={'native-region-normalizer':args.native_binary.resolve(),'latent-region-normalizer':args.latent_binary.resolve()}
    for name,path in binaries.items():
        expected=args.expected_native_sha256 if name.startswith('native') else args.expected_latent_sha256
        if expected:require(sha(path)==expected,f'Unexpected {name} executable hash')
        help_text=subprocess.run([str(path),'--help'],check=True,capture_output=True,text=True).stdout
        required=('--q-min','--q-max','--q-lower-open','--q-upper-open','--model') if name.startswith('native') else ('--region','--cloud-replicates')
        require(all(flag in help_text for flag in required),f'{name} lacks the required interface')
    sources={'input-config.json':source_config,'shape.json':shape,'guide-model.json':model_path,'region.json':region_path,
        **binaries,'runner.py':Path(__file__).resolve()}
    helpers=local_dependencies([TOOLS/name for name in HELPERS if (TOOLS/name).is_file()])
    sources.update(helpers)
    source_hashes={name:sha(path) for name,path in sources.items()}
    archive=root/'provenance';archive.mkdir(parents=True)
    for name,path in sources.items():
        shutil.copy2(path,archive/name)
        require(sha(archive/name)==source_hashes[name],f'Source changed while freezing {name}')
    effective=read(archive/'input-config.json');effective['shape']=str(archive/'shape.json')
    write(archive/'config.json',effective)
    (root/'logs').mkdir();(root/'runs').mkdir()
    n=args.n_guide+args.n_cover
    guide=dict(model_weight=args.model_weight,uniform_probability=args.model_uniform_probability,anchor_index=args.model_anchor_index,
        cover_scales=[1.],cover_weights=[1.],description=description)
    allocation=dict(guide=args.n_guide,cover=args.n_cover,total=n,alpha_guide=args.n_guide/n,alpha_cover=args.n_cover/n)
    populations=[];jobs=[]
    for i in range(args.populations):
        population={'id':f'r{i:02d}'}
        for family,count,seed_base in [('guide',args.n_guide,args.guide_seed_base),('cover',args.n_cover,args.cover_seed_base)]:
            seed=seed_base+1009*i;output=root/'runs'/population['id']/family
            if family=='guide':
                command=[str(archive/'native-region-normalizer'),'--config',str(archive/'config.json'),'--out',str(output),
                    '--samples',str(count),'--seed',str(seed),'--lambda-ratio',str(args.lambda_ratio),'--cloud-replicates','2',
                    '--q-min',str(window['minimum']),'--q-max',str(window['maximum']),'--cover-scales','1',
                    '--model',str(archive/'guide-model.json'),'--model-weight',str(args.model_weight),
                    '--model-uniform-probability',str(args.model_uniform_probability),'--model-anchor-index',str(args.model_anchor_index)]
                if not window['lower_inclusive']:command.append('--q-lower-open')
                if not window['upper_inclusive']:command.append('--q-upper-open')
            else:
                command=[str(archive/'latent-region-normalizer'),'--config',str(archive/'config.json'),
                    '--region',str(archive/'region.json'),'--out',str(output),'--samples',str(count),'--seed',str(seed),
                    '--lambda-ratio',str(args.lambda_ratio),'--cloud-replicates','2']
            job=dict(population_id=population['id'],family=family,seed=seed,samples=count,output=str(output),command=command)
            population[family]={k:job[k] for k in ('seed','samples','output','command')};jobs.append(job)
        populations.append(population)
    manifest=dict(schema=SCHEMA,created_utc=datetime.now(timezone.utc).isoformat(),q_window=window,
        physical={k:effective[k] for k in PHYSICAL_KEYS},guide=guide,allocation=allocation,populations=populations,jobs=jobs,
        population_count=args.populations,total_unconditional_draws=args.populations*n,
        lambda_ratio=args.lambda_ratio,cloud_replicates=2,max_workers=args.workers,
        config_sha256=sha(archive/'config.json'),shape_sha256=shape_sha,model_sha256=sha(archive/'guide-model.json'),
        region_sha256=sha(archive/'region.json'),source_config_sha256=source_hashes['input-config.json'],
        archive_sha256={p.name:sha(p) for p in sorted(archive.iterdir())},
        source_inputs={name:dict(path=str(path),sha256=source_hashes[name]) for name,path in sources.items()},
        geometric_cover_reconstruction=proof,
        estimator='Each population: 1/(n_guide+n_cover) sum H Iwindow mean(W)/(alpha_guide*g_guide+alpha_cover*g_cover), including every zero. Fixed quotas; component-only controls use their own draws and original proposal density.',
        independence='Distinct family/population seeds; deterministic quotas declared before outcomes. No historical samples, fitting, outcome-dependent replacement or continuation.',
        scope='One fixed original-q contact window with every prescribed physical neighbor; no assembly or global equilibrium conclusion.')
    write(root/'manifest.json',manifest)
    (root/'manifest.sha256').write_text(sha(root/'manifest.json')+'\n')
    write(root/'runner-status.json',dict(prepared_only=True,running=False,complete=False,manifest_sha256=sha(root/'manifest.json'),jobs={}))
    return root,manifest


def verify_archive(root,manifest):
    require(sha(root/'manifest.json')==(root/'manifest.sha256').read_text().strip(),'Prepared manifest changed')
    require(manifest['schema']==SCHEMA,'Unsupported MIS campaign schema')
    for name,digest in manifest['archive_sha256'].items():require(sha(root/'provenance'/name)==digest,f'Frozen archive changed: {name}')


def execute(root):
    root=Path(root).resolve();manifest=read(root/'manifest.json');verify_archive(root,manifest)
    previous=read(root/'runner-status.json')
    require(previous.get('prepared_only') and not previous.get('running') and not previous.get('complete'),'Only a never-started prepared campaign may execute')
    require(all(not Path(job['output']).exists() for job in manifest['jobs']),'A population output already exists; never restart or replace it')
    state=dict(runner_pid=os.getpid(),prepared_only=False,running=True,complete=False,
        manifest_sha256=sha(root/'manifest.json'),jobs={});lock=threading.Lock()
    def persist():
        state['updated_utc']=datetime.now(timezone.utc).isoformat()
        write(root/'runner-status.tmp',state);(root/'runner-status.tmp').replace(root/'runner-status.json')
    persist()
    def run(job):
        verify_archive(root,manifest)
        key=f"{job['population_id']}-{job['family']}";started=time.monotonic()
        with (root/'logs'/f'{key}.log').open('x') as log:
            process=subprocess.Popen(job['command'],stdout=log,stderr=subprocess.STDOUT)
            with lock:state['jobs'][key]=dict(pid=process.pid,seed=job['seed'],samples=job['samples'],status='running');persist()
            while process.poll() is None:
                with lock:
                    progress=Path(job['output'])/'progress.json'
                    if progress.exists():
                        try:state['jobs'][key]['progress']=read(progress)
                        except json.JSONDecodeError:pass
                    persist()
                time.sleep(1)
        with lock:
            state['jobs'][key].update(status='complete' if process.returncode==0 else 'failed',exit_code=process.returncode,
                wall_seconds=time.monotonic()-started)
            summary=Path(job['output'])/'summary.json'
            if summary.exists():state['jobs'][key]['summary_sha256']=sha(summary)
            persist()
        return process.returncode
    try:
        with ThreadPoolExecutor(max_workers=manifest['max_workers']) as pool:codes=list(pool.map(run,manifest['jobs']))
        state.update(running=False,complete=True,success=all(code==0 for code in codes));persist()
    except Exception as error:
        state.update(running=False,complete=True,success=False,error=str(error));persist();raise
    require(state['success'],'A prescribed job failed; retain every output without replacement')
    print(json.dumps(dict(root=str(root),complete=True,jobs=len(codes),manifest_sha256=state['manifest_sha256'])),flush=True)


def parser():
    ap=argparse.ArgumentParser(description=__doc__)
    mode=ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--out',type=Path);mode.add_argument('--execute-prepared',type=Path,help='Execute only the unchanged frozen quotas, seeds and commands')
    ap.add_argument('--prepare-only',action='store_true')
    ap.add_argument('--config',type=Path,default=ROOT/'runs/ab-shoulder-guide-preparation-20260920/config.json')
    ap.add_argument('--model',type=Path,default=ROOT/'runs/ab-shoulder-guide-preparation-20260920/model-mixture.json')
    ap.add_argument('--region',type=Path,default=ROOT/'runs/ab-shoulder-cayley-cover-preparation-20260920/region-shoulder.json')
    ap.add_argument('--native-binary',type=Path,default=ROOT/'runs/ab-shoulder-mixture-4x16384-l64-20260920/provenance/native-region-normalizer')
    ap.add_argument('--latent-binary',type=Path,default=ROOT/'runs/ab-shoulder-cayley-reference-4x262144-l64-20260920/provenance/latent-region-normalizer')
    ap.add_argument('--expected-native-sha256');ap.add_argument('--expected-latent-sha256')
    ap.add_argument('--populations',type=int,default=4);ap.add_argument('--n-guide',type=int,default=4096)
    ap.add_argument('--n-cover',type=int,default=65536);ap.add_argument('--workers',type=int,default=8)
    ap.add_argument('--guide-seed-base',type=int,default=99761010);ap.add_argument('--cover-seed-base',type=int,default=99861010)
    ap.add_argument('--model-weight',type=float,default=.75);ap.add_argument('--model-uniform-probability',type=float,default=.05)
    ap.add_argument('--model-anchor-index',type=int,default=0);ap.add_argument('--lambda-ratio',type=float,default=64.)
    ap.add_argument('--q-min',type=float,default=1.);ap.add_argument('--q-max',type=float,default=2.)
    ap.add_argument('--q-lower-closed',action='store_true');ap.add_argument('--q-upper-closed',action='store_true')
    return ap


def main():
    args=parser().parse_args()
    if args.execute_prepared:
        require(not args.prepare_only,'--prepare-only cannot execute a prepared campaign')
        execute(args.execute_prepared);return
    root,manifest=prepare(args)
    if args.prepare_only:
        print(json.dumps(dict(prepared_only=True,root=str(root),populations=manifest['population_count'],
            allocation=manifest['allocation'],manifest_sha256=sha(root/'manifest.json'))),flush=True)
    else:execute(root)


if __name__=='__main__':main()
