#!/usr/bin/env python3
"""Prepare separated tetramers for matched fixed-K or variable-K RJ atlas runs.

Use --atlas LABEL=PATH twice with --reversible-jump for native-on/off controls,
and --reuse-starts to preserve earlier preparations and paired MC seeds exactly.
Every production atlas is explicitly labelled; the default contains native
examples and is NOT a no-native-prior experiment. No fit is trained here.
Run this supervisor under nohup for durable background execution.
"""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)
def seed(master,domain,index):
    return int.from_bytes(hashlib.sha256(f'{master}:{domain}:{index}'.encode()).digest()[:8],'little')

def prepare_poses(rng,shape,radius,rd,count=12):
    centers=np.asarray([a['center'] for a in shape['atoms']]);radii=np.asarray([a['radius'] for a in shape['atoms']])
    bound=float(np.max(np.linalg.norm(centers,axis=1)+radii)); exclusion_bound=bound+rd
    inner=radius-bound; positions=[];poses=[];attempts=0
    assert inner>0
    while len(poses)<count:
        attempts+=1
        if attempts>100000:raise RuntimeError('Could not prepare separated centers')
        direction=rng.normal(size=3);direction/=np.linalg.norm(direction)
        p=direction*inner*rng.random()**(1/3)
        if any(np.linalg.norm(p-other)<=2*exclusion_bound+1e-7 for other in positions):continue
        q=rng.normal(size=4);q/=np.linalg.norm(q)  # Uniform Haar, scalar first.
        positions.append(p);poses.append(dict(position=p.tolist(),orientation=q.tolist()))
    distances=[float(np.linalg.norm(p-q)) for i,p in enumerate(positions) for q in positions[i+1:]]
    certificate=dict(body_bound_A=bound,exclusion_bound_A=exclusion_bound,
        minimum_center_distance_A=min(distances),required_separation_A=2*exclusion_bound,
        minimum_body_bound_wall_clearance_A=min(radius-bound-float(np.linalg.norm(p)) for p in positions),
        exact_no_interbody_exclusion_overlap_by_bound=True,proposals=attempts,
        initial_registered_bonds=0,initial_nonspecific_exclusion_contacts=0,
        orientation_law='Independent normalized four-dimensional Gaussian: Haar SO(3)',
        preparation_law='Sequential uniform inner-ball placement conditioned on separation; not an equilibrium sample')
    return poses,certificate

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--binary',type=Path,default=ROOT/'target/release/tetramer-mc')
    parser.add_argument('--model',type=Path,default=ROOT/'examples/frozen-relative-mixture.json')
    parser.add_argument('--model-label',default='native-informed-frozen-atlas')
    parser.add_argument('--atlas',action='append',default=[],metavar='LABEL=PATH',
        help='Repeat for paired model variants; supersedes --model/--model-label')
    parser.add_argument('--reuse-starts',type=Path,help='Reuse exact preparation poses and MC seeds from an earlier campaign')
    parser.add_argument('--reversible-jump',action='store_true',help='One RJ+transport arm per atlas; labelled variable K')
    parser.add_argument('--rj-config',type=Path,help='Override the documented default RJ configuration')
    parser.add_argument('--sweeps',type=int,default=400)
    parser.add_argument('--sample-every',type=int,default=10)
    parser.add_argument('--replicates',type=int,default=4)
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--master-seed',type=int,default=2026092017)
    parser.add_argument('--auxiliary-json',type=Path)
    parser.add_argument('--write-examples',action='store_true')
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--no-gsd',action='store_true')
    args=parser.parse_args()
    if min(args.sweeps,args.sample_every,args.replicates,args.workers)<1:
        parser.error('Positive run sizes and worker count required')
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):parser.error('Output must be empty')
    for name in ('configs','logs','runs','provenance'):(out/name).mkdir()
    archive=out/'provenance'
    atlases=[]
    for item in args.atlas or [args.model_label+'='+str(args.model)]:
        if '=' not in item:parser.error('--atlas requires LABEL=PATH')
        label,path=item.split('=',1)
        if not label or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in label):
            parser.error('Atlas labels must contain only letters, numbers, - or _')
        if label in [a['label'] for a in atlases]:parser.error('Duplicate atlas label')
        atlases.append(dict(label=label,source=Path(path).resolve(),archive_name='model.json' if len(args.atlas)<=1 else f'model-{label}.json'))
    inputs={'tetramer-mc':args.binary.resolve(),
        'shape.json':ROOT/'examples/tetramer-shape.json','monomer-shape.json':ROOT/'examples/monomer-shape.json',
        'native-pair-motifs.json':ROOT/'examples/native-pair-motifs.json','launcher.py':Path(__file__).resolve()}
    inputs.update({a['archive_name']:a['source'] for a in atlases})
    for name,path in inputs.items():shutil.copy2(path,archive/name)
    binary=archive/'tetramer-mc'
    shape=json.loads((archive/'shape.json').read_text())
    for atlas in atlases:
        model_data=json.loads((archive/atlas['archive_name']).read_text())
        assert model_data['shape_sha256']==sha(archive/'shape.json')
        atlas['model']=str(archive/atlas['archive_name']);atlas['sha256']=sha(atlas['model']);atlas['source']=str(atlas['source'])
    source_bundle=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,check=True,capture_output=True,text=True).stdout.strip()
    auxiliary=json.loads(args.auxiliary_json.read_text()) if args.auxiliary_json else {}
    rj=(json.loads(args.rj_config.read_text()) if args.rj_config else dict(min_components=1,max_components=24,
        initial_components=8,poisson_mean=8.,attempts_per_sweep=4)) if args.reversible_jump else None
    reused=read_reused(args.reuse_starts,args.replicates) if args.reuse_starts else None
    if args.write_examples and (len(atlases)>1 or args.reversible_jump):parser.error('--write-examples supports only the original fixed-K pair')
    radius=354.50820786337056;rd=1.5;activity=.035
    jobs=[];preparations=[]
    for replicate in range(args.replicates):
        prepare_seed=seed(args.master_seed,'prepare',replicate);mc_seed=seed(args.master_seed,'mc',replicate)
        if reused:
            record=reused[replicate];poses=record['poses'];certificate=record['certificate']
            prepare_seed=record['preparation_seed'];mc_seed=record['mc_seed']
        else:poses,certificate=prepare_poses(np.random.default_rng(prepare_seed),shape,radius,rd)
        pose_hash=hashlib.sha256(json.dumps(poses,sort_keys=True).encode()).hexdigest()
        preparations.append(dict(replicate=replicate,preparation_seed=prepare_seed,mc_seed=mc_seed,
            initial_poses_sha256=pose_hash,certificate=certificate))
        for atlas,mode in [(atlas,mode) for atlas in atlases for mode in (('rj',) if args.reversible_jump else ('frozen','transport'))]:
            model=Path(atlas['model'])
            identifier=f"free-r{replicate:02d}-{atlas['label']}-{mode}" if len(atlases)>1 else f'free-r{replicate:02d}-{mode}'
            cfg=dict(shape=str(archive/'shape.json'),monomer_shape=str(archive/'monomer-shape.json'),
                initial_poses=copy.deepcopy(poses),box_lengths=[2*radius]*3,boundary=dict(kind='spherical',radius=radius),
                depletant_radius=rd,reservoir_density=activity,poisson_lambda_ratio=16.,
                endpoint_gate=dict(max_cells=2047,max_depth=14,min_width=.5),
                global_probability=.5,learned_uniform_weight=.1,local_translation_std_A=.2,
                local_small_angle_std_degrees=1.,gca_probability=1.,center_shift_probability=1.,
                fixed_body_indices=[],seed_labels=[],seed=mc_seed,sweeps=args.sweeps,sample_every=args.sample_every,
                metadata=dict(arm='dispersed-free',replicate=replicate,proposal_mode=mode,
                    model_label=atlas['label'],model_sha256=sha(model),native_pair_motifs=str(archive/'native-pair-motifs.json'),
                    initial_fragments=[[i] for i in range(12)],initial_free_tetramers=12,
                    preparation_seed=prepare_seed,initial_poses_sha256=pose_hash,
                    preparation_equilibrated=False,training_feedback=False,
                    initialization='Independent dispersed positions and Haar orientations; all exclusion bounds separated',
                    preparation_certificate=certificate))
            if mode in ('transport','rj'):cfg['auxiliary_transport']=copy.deepcopy(auxiliary)
            if mode=='rj':cfg['reversible_jump']=copy.deepcopy(rj)
            if reused:cfg['metadata']['reused_start_source']=record['source']
            cfg_path=out/'configs'/f'{identifier}.json';write(cfg_path,cfg)
            jobs.append(dict(id=identifier,replicate=replicate,mode=mode,config=str(cfg_path),
                config_sha256=sha(cfg_path),directory=str(out/'runs'/identifier),model=str(model),
                model_sha256=sha(model),model_label=atlas['label'],variant=atlas['label'] if len(atlases)>1 else mode))
            if args.write_examples and replicate==0:
                example=copy.deepcopy(cfg);example['shape']='tetramer-shape.json';example['monomer_shape']='monomer-shape.json'
                example['metadata']['native_pair_motifs']='native-pair-motifs.json'
                name='spherical-free.json' if mode=='frozen' else 'spherical-free-transport.json'
                target=ROOT/'examples'/name
                if target.exists():raise RuntimeError(f'Refusing to overwrite {target}')
                write(target,example)
    manifest=dict(schema=1,created_unix_time=time.time(),supervisor_pid=os.getpid(),jobs=jobs,preparations=preparations,
        sweeps=args.sweeps,sample_every=args.sample_every,workers=args.workers,master_seed=args.master_seed,
        binary_sha256=sha(binary),model_sha256=atlases[0]['sha256'] if len(atlases)==1 else None,
        model_label=atlases[0]['label'] if len(atlases)==1 else 'paired-atlas-comparison',atlases=atlases,git_head=source_bundle,
        reversible_jump=rj,reuse_starts=str(args.reuse_starts.resolve()) if args.reuse_starts else None,
        input_sha256={name:sha(archive/name) for name in inputs},
        scope='Independent separated preparations or exact reuses, paired MC streams. Model prior information is explicitly labelled. Optional labelled variable-K RJ with conditional-mean transport. No learning from campaign history; native templates are analysis inputs only beyond the labelled atlas.')
    write(out/'manifest.json',manifest)
    print(json.dumps(dict(prepared=True,pid=os.getpid(),out=str(out),jobs=len(jobs),binary_sha256=sha(binary))),flush=True)
    if args.prepare_only:return
    records=[];started=time.monotonic()
    def run(job):
        command=[str(binary),'run','--config',job['config'],'--model',job['model'],'--out',job['directory'],
            '--sweeps',str(args.sweeps),'--sample-every',str(args.sample_every)]
        if args.no_gsd:command.append('--no-gsd')
        begin=time.monotonic()
        with (out/'logs'/f'{job["id"]}.log').open('w') as stream:
            process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT)
            write(out/'logs'/f'{job["id"]}-process.json',dict(pid=process.pid,command=command,started_unix_time=time.time()))
            code=process.wait()
        summary_path=Path(job['directory'])/'summary.json'
        summary=json.loads(summary_path.read_text()) if summary_path.exists() else {}
        return dict(id=job['id'],returncode=code,wall_seconds=time.monotonic()-begin,
            sampler_cpu_seconds=summary.get('sampler_cpu_seconds'),complete=summary.get('complete',False))
    write(out/'status.json',dict(running=True,completed=0,total=len(jobs),supervisor_pid=os.getpid()))
    with ThreadPoolExecutor(max_workers=min(args.workers,len(jobs))) as pool:
        for future in as_completed([pool.submit(run,job) for job in jobs]):
            record=future.result();records.append(record);print(json.dumps(record),flush=True)
            write(out/'status.json',dict(running=True,completed=len(records),total=len(jobs),records=records,supervisor_pid=os.getpid()))
    for name,value in manifest['input_sha256'].items():assert sha(archive/name)==value
    failures=[r for r in records if r['returncode'] or not r['complete']]
    write(out/'summary.json',dict(complete=not failures,records=records,wall_seconds=time.monotonic()-started,
        total_sampler_cpu_seconds=sum(r['sampler_cpu_seconds'] or 0 for r in records)))
    write(out/'status.json',dict(running=False,complete=not failures,completed=len(records),total=len(jobs),records=records))
    if failures:raise SystemExit(f'{len(failures)} failed runs')

def read_reused(source,replicates):
    source=source.resolve();manifest=json.loads((source/'manifest.json').read_text());result={}
    for preparation in manifest['preparations']:
        index=preparation['replicate']
        if index>=replicates:continue
        job=next(j for j in manifest['jobs'] if j['replicate']==index)
        cfg=json.loads(Path(job['config']).read_text())
        assert cfg['boundary']['radius']==354.50820786337056 and cfg['depletant_radius']==1.5 and cfg['reservoir_density']==.035
        poses=cfg['initial_poses'];pose_hash=hashlib.sha256(json.dumps(poses,sort_keys=True).encode()).hexdigest()
        assert pose_hash==preparation['initial_poses_sha256']
        result[index]=dict(poses=poses,certificate=preparation['certificate'],preparation_seed=preparation['preparation_seed'],
            mc_seed=preparation['mc_seed'],source=dict(config=job['config'],config_sha256=sha(job['config']),
            manifest=str(source/'manifest.json'),manifest_sha256=sha(source/'manifest.json')))
    if set(result)!=set(range(replicates)):raise ValueError('Reused campaign lacks requested independent starts')
    return result

if __name__=='__main__':main()
