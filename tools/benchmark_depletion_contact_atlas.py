#!/usr/bin/env python3
"""Freeze and run a bounded comparison of initial versus discovered contact atlases.

All proposal files are immutable before sampling. The supplied physical start
may contain a native seed; this is a proposal-accessibility pilot, not a blind
initialization or equilibrium/assembly campaign. Existing execute_jobs supplies
failure draining and a bounded process pool. No native classifier enters runs.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from run_conditional_ray_campaign import execute_jobs


def read(path):
    return json.loads(Path(path).read_text())


def write(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(args):
    out=Path(args.out).resolve()
    if out.exists():
        raise ValueError('Fresh output directory required; no overwrite or retry')
    if not 1 <= args.workers <= 4 or args.streams < 1 or args.sweeps < 1 or args.sample_every < 1:
        raise ValueError('Require positive fixed allocations and 1..4 workers')
    source=Path(args.config).resolve(); cfg=read(source)
    estimated_bytes=2*args.streams*args.sweeps*len(cfg['initial_poses'])*8192+512*1024**2
    if shutil.disk_usage(out.parent).free < estimated_bytes:
        raise ValueError('Insufficient free space for the conservative move/trajectory budget; use a larger data volume')
    recovery=None
    if args.infrastructure_recovery_from:
        failed=Path(args.infrastructure_recovery_from).resolve()
        status=read(failed/'status.json')
        if not all(j['status']=='failed' for j in status['jobs']):
            raise ValueError('Recovery requires a completely failed, drained comparison')
        oldseeds={j['seed'] for j in status['jobs']}
        if any(args.seed+1009*i in oldseeds for i in range(args.streams)):
            raise ValueError('Use fresh independent streams for the new allocation')
        recovery=dict(previous_directory=str(failed),status_sha256=sha(failed/'status.json'),
            reason='Filesystem exhaustion; all old populations retained as failed. No partial population enters this comparison.')
    if cfg.get('contact_memory') or cfg.get('reversible_jump') or cfg.get('conditional_closure'):
        raise ValueError('Benchmark requires a frozen proposal, without online learning')
    shape=Path(cfg['shape'])
    if not shape.is_absolute(): shape=source.parent/shape
    shape=shape.resolve()
    models={k:Path(v).resolve() for k,v in [('initial',args.initial_model),('discovered',args.discovered_model)]}
    for path in models.values():
        model=read(path);base=model.get('base_model',model)
        if base.get('shape_sha256') != sha(shape): raise ValueError('Atlas/shape mismatch')
    out.mkdir(); archive=out/'inputs'; archive.mkdir();(out/'logs').mkdir();(out/'populations').mkdir()
    shutil.copy2(source,archive/'source-config.json');shutil.copy2(shape,archive/'shape.json')
    shutil.copy2(Path(__file__),archive/'controller.py')
    binary=Path(args.binary).resolve();shutil.copy2(binary,archive/'tetramer-mc')
    for name,path in models.items():shutil.copy2(path,archive/f'{name}.json')
    cfg['shape']=str(archive/'shape.json')
    for key in ['monomer_shape','native_pair_motifs']:
        if cfg.get(key):
            p=Path(cfg[key]);cfg[key]=str((p if p.is_absolute() else source.parent/p).resolve())
    jobs=[]
    for stream in range(args.streams):
        # Matching seeds are deliberate across arms; different kernels consume
        # different random variates, so no independent or paired-error claim.
        seed=args.seed+1009*stream
        for arm in models:
            name=f'{arm}-r{stream:02d}';c=copy.deepcopy(cfg)
            c.update(seed=seed,sweeps=args.sweeps,sample_every=args.sample_every)
            c.setdefault('metadata', {})['proposal_benchmark'] = dict(
                arm=arm, native_informed_proposal=False,
                frozen_model_sha256=sha(archive/f'{arm}.json'),
                discovery='shape-and-bath-only; supplied native seed is evaluation context',
                scope='Bounded proposal-accessibility pilot, not equilibrium assembly')
            path=archive/f'{name}-config.json';write(path,c)
            directory=out/'populations'/name
            command=[str(archive/'tetramer-mc'),'run','--config',str(path),'--model',str(archive/f'{arm}.json'),
                     '--out',str(directory),'--sweeps',str(args.sweeps),'--sample-every',str(args.sample_every)]
            jobs.append(dict(id=name,arm=arm,stream=stream,seed=seed,directory=str(directory),log=str(out/'logs'/f'{name}.log'),command=command,status='pending'))
    plan=dict(schema='native-blind-discovery-proposal-benchmark-v1',created=time.time(),jobs=jobs,
              workers=args.workers,streams=args.streams,sweeps=args.sweeps,
              source_config=str(source),initial_model=str(models['initial']),discovered_model=str(models['discovered']),
              infrastructure_recovery=recovery,
              storage_estimate_bytes=estimated_bytes,
              bath=dict(radius=cfg['depletant_radius'],activity=cfg['reservoir_density']),
              input_sha256={str(p.relative_to(out)):sha(p) for p in sorted(archive.iterdir())},
              scope='Frozen proposal comparison from the identical supplied physical start. Native labels excluded from proposal discovery/fitting, not necessarily from initial seed geometry. Keep every rejected state and every arm. No equilibrium, mixing-time, physical-kinetics, or assembly conclusion from this pilot.')
    write(out/'plan.json',plan)
    return out,plan


def run(args):
    out,plan=prepare(args)
    jobs=copy.deepcopy(plan['jobs'])
    env=dict(os.environ,**{k:'1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','RAYON_NUM_THREADS')})
    started=time.time()
    def snapshot():
        write(out/'status.json',dict(complete=all(j['status']=='complete' for j in jobs),jobs=jobs,started=started,updated=time.time(),plan_sha256=sha(out/'plan.json')))
    execute_jobs(jobs,snapshot,workers=args.workers,popen=lambda *a,**kw:subprocess.Popen(*a,**kw,env=env))
    for job in jobs:
        summary=read(Path(job['directory'])/'summary.json')
        if not summary['complete'] or summary['completed_sweeps']!=args.sweeps:raise ValueError('Incomplete benchmark run')
    snapshot()
    print(json.dumps(dict(complete=True,out=str(out),populations=len(jobs))))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--initial-model',required=True);p.add_argument('--discovered-model',required=True)
    p.add_argument('--binary',default='target/release/tetramer-mc');p.add_argument('--out',required=True)
    p.add_argument('--streams',type=int,default=2);p.add_argument('--sweeps',type=int,default=200)
    p.add_argument('--sample-every',type=int,default=100);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--seed',type=int,default=20260926101)
    p.add_argument('--infrastructure-recovery-from')
    run(p.parse_args())
