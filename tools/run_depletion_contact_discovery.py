#!/usr/bin/env python3
"""Freeze and run independent shape-only contact discovery populations.

Each population retains every start. Optimization is proposal design, not an
equilibrium calculation. The output can be combined by fit_depletion_contact_atlas.py.
"""
from __future__ import annotations
import argparse
import copy
import os
from pathlib import Path
import shutil
import subprocess
import time
from benchmark_depletion_contact_atlas import read, write, sha
from run_conditional_ray_campaign import execute_jobs


def run(args):
    out=Path(args.out).resolve()
    if out.exists(): raise ValueError('Fresh output directory required; no retries')
    if not 1 <= args.workers <= 4 or args.populations < 1:
        raise ValueError('Positive population count and 1..4 workers required')
    shape=Path(args.shape).resolve(); binary=Path(args.binary).resolve()
    out.mkdir(); archive=out/'inputs'; archive.mkdir()
    (out/'logs').mkdir(); (out/'populations').mkdir()
    for source,name in [(shape,'shape.json'),(binary,'depletion-contact-discovery'),(Path(__file__),'controller.py')]:
        shutil.copy2(source,archive/name)
    jobs=[]
    for index in range(args.populations):
        name=f'r{index:02d}'; directory=out/'populations'/name
        command=[str(archive/'depletion-contact-discovery'),'--shape',str(archive/'shape.json'),'--out',str(directory)]
        seed=args.seed+1009*index
        for key,value in dict(starts=args.starts,search_steps=args.search_steps,search_points=args.search_points,
            validation_points=args.validation_points,refine_steps=args.refine_steps,burn=args.burn,
            save_every=args.save_every,rd=args.rd,activity=args.activity,seed=seed).items():
            command.extend(['--'+key.replace('_','-'),str(value)])
        jobs.append(dict(id=name,seed=seed,directory=str(directory),log=str(out/'logs'/f'{name}.log'),command=command,status='pending'))
    plan=dict(schema='native-blind-discovery-campaign-v1',arguments=vars(args),jobs=jobs,
        input_sha256={str(p.relative_to(out)):sha(p) for p in sorted(archive.iterdir())},
        fitting=dict(initial_weight=.5,shrinkage=.25,covariance_floor=.01,translation_width=1.,angle_width_degrees=3.),
        benchmark=dict(streams_per_arm=2,sweeps=200,sample_every=100,
            arms=['matched-initial-only','half-initial-half-refined'],
            start='same previously supplied seed8/free256 configuration at 500 uM'),
        scope='All allocations fixed before sampling; no native data, validation-based selection or slot pruning. Pair search/refinement does not establish equilibrium weights or assembly stability.')
    write(out/'plan.json',plan);jobs=copy.deepcopy(jobs);started=time.time()
    def snapshot():
        write(out/'status.json',dict(complete=all(j['status']=='complete' for j in jobs),jobs=jobs,
            started=started,updated=time.time(),plan_sha256=sha(out/'plan.json')))
    env=dict(os.environ,**{k:'1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','RAYON_NUM_THREADS')})
    execute_jobs(jobs,snapshot,workers=args.workers,popen=lambda *a,**kw:subprocess.Popen(*a,**kw,env=env))
    for job in jobs:
        manifest=read(Path(job['directory'])/'manifest.json')
        if not manifest['complete'] or manifest['slots']!=args.starts: raise ValueError('Incomplete discovery')
        for name,digest in manifest['outputs_sha256'].items():
            if sha(Path(job['directory'])/name)!=digest: raise ValueError('Output hash mismatch')
    snapshot()
    print(out)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--shape',required=True);p.add_argument('--out',required=True)
    p.add_argument('--binary',default='target/release/depletion-contact-discovery')
    p.add_argument('--populations',type=int,default=4);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--starts',type=int,default=8);p.add_argument('--search-steps',type=int,default=128)
    p.add_argument('--search-points',type=int,default=2048);p.add_argument('--validation-points',type=int,default=65536)
    p.add_argument('--refine-steps',type=int,default=256);p.add_argument('--burn',type=int,default=64)
    p.add_argument('--save-every',type=int,default=4);p.add_argument('--rd',type=float,default=1.4)
    p.add_argument('--activity',type=float,default=.0275);p.add_argument('--seed',type=int,default=20260926301)
    run(p.parse_args())
