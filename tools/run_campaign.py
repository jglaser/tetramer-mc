#!/usr/bin/env python3
"""Launch matched Rust seeded/fluid arms using independent paired RNG seeds.

Replicates reuse the two supplied initial configurations; they are stochastic
repeats, not independently equilibrated preparations. Existing outputs are
never overwritten. Run this supervisor under nohup for a durable background job.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]

def write(path,value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--sweeps',type=int,default=400)
    parser.add_argument('--sample-every',type=int,default=10)
    parser.add_argument('--replicates',type=int,default=2)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--master-seed',type=int,default=2026091903)
    parser.add_argument('--binary',type=Path,default=ROOT/'target/release/tetramer-mc')
    parser.add_argument('--model',type=Path,default=ROOT/'examples/frozen-relative-mixture.json')
    parser.add_argument('--seeded',type=Path,default=ROOT/'examples/seeded.json')
    parser.add_argument('--fluid',type=Path,default=ROOT/'examples/seed-free.json')
    parser.add_argument('--no-gsd',action='store_true')
    args=parser.parse_args()
    if min(args.sweeps,args.sample_every,args.replicates,args.workers)<1:parser.error('Positive run sizes required')
    args.out=args.out.resolve();args.out.mkdir(parents=True,exist_ok=True)
    if any(args.out.iterdir()):parser.error('Campaign output must be empty')
    if not args.binary.is_file():parser.error('Build the release binary first')
    for name in ('configs','logs','runs'):(args.out/name).mkdir()
    model=args.model.resolve();model_hash=hashlib.sha256(model.read_bytes()).hexdigest()
    binary=args.binary.resolve();binary_hash=hashlib.sha256(binary.read_bytes()).hexdigest()
    jobs=[]
    for arm,base_path in [('seeded',args.seeded.resolve()),('seed-free',args.fluid.resolve())]:
        base=json.loads(base_path.read_text())
        for key in ('shape','monomer_shape'):
            if base.get(key):base[key]=str((base_path.parent/Path(base[key])).resolve())
        if base.get('metadata',{}).get('native_pair_motifs'):
            base['metadata']['native_pair_motifs']=str((base_path.parent/Path(base['metadata']['native_pair_motifs'])).resolve())
        for replicate in range(args.replicates):
            paired_seed=int.from_bytes(hashlib.sha256(f'{args.master_seed}:{arm}:{replicate}'.encode()).digest()[:8],'little')
            for method in ('local-uniform','learned'):
                name=f'{arm}-r{replicate:02d}-{method}'
                cfg=json.loads(json.dumps(base));cfg['seed']=paired_seed
                cfg.setdefault('metadata',{}).update(arm=arm,replicate=replicate,method=method,initialization='supplied fixed preparation; repeated with new RNG',training_feedback=False)
                cfg_path=args.out/'configs'/f'{name}.json';write(cfg_path,cfg)
                jobs.append(dict(id=name,config=str(cfg_path),directory=str(args.out/'runs'/name),method=method,arm=arm,replicate=replicate))
    write(args.out/'manifest.json',dict(jobs=jobs,model_sha256=model_hash,binary_sha256=binary_hash,sweeps=args.sweeps,sample_every=args.sample_every,scope='Matched RNG repeats of two supplied starts; frozen model; no online training'))
    def run(job):
        command=[str(binary),'run','--config',job['config'],'--out',job['directory'],'--method',job['method'],'--sweeps',str(args.sweeps),'--sample-every',str(args.sample_every)]
        if job['method']=='learned':command+=['--model',str(model)]
        if args.no_gsd:command+=['--no-gsd']
        begin=time.monotonic()
        with (args.out/'logs'/f'{job["id"]}.log').open('w') as stream:
            result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
        return dict(id=job['id'],returncode=result.returncode,wall_seconds=time.monotonic()-begin)
    records=[]
    with ThreadPoolExecutor(max_workers=min(args.workers,len(jobs))) as pool:
        for future in as_completed([pool.submit(run,job) for job in jobs]):
            record=future.result();records.append(record);print(json.dumps(record),flush=True)
            write(args.out/'progress.json',dict(completed=len(records),total=len(jobs),records=records))
    assert hashlib.sha256(model.read_bytes()).hexdigest()==model_hash,'Model changed during production'
    assert hashlib.sha256(binary.read_bytes()).hexdigest()==binary_hash,'Binary changed during campaign'
    failed=[r for r in records if r['returncode']]
    write(args.out/'summary.json',dict(complete=not failed,records=records))
    if failed:raise SystemExit(f'{len(failed)} jobs failed; inspect logs')

if __name__=='__main__':main()
